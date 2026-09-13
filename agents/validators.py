"""Post-parse response checks beyond what the Pydantic schema alone can
express -- passed as ``extra_validate`` to ``agents.base.call_structured``,
so a violation triggers the same reject-and-retry loop as a schema
parse failure (and counts toward the same logged retry rate, now broken
out by reason -- see agents/base.py's ``retry_reasons``).

Pydantic checks shape (types, closed relation vocabulary, confidence in
[0,1]); these checks the specific *contract* each agent has with the
pipeline (id-prefix per agent, uniqueness, the benign-hypothesis floor,
and -- new in this version -- that a hypothesis's prediction is actually
checkable against this pipeline's own data).
"""
from __future__ import annotations

import re
from typing import Iterable, Set

from agents.schema import ClaimsResponse, HypothesisResponse

#: A3/A4 must generate this many benign hypotheses minimum (the spec's
#: "at least two must be benign", enforced as a hard floor rather than
#: left to prompt-following alone -- an agent that can only imagine
#: malicious explanations is exactly the failure mode this floor exists
#: to catch, so it can't be silently satisfied by omission).
MIN_BENIGN_HYPOTHESES = 2
MIN_HYPOTHESES_TOTAL = 3

#: message prefix on a rejection caused specifically by the flow-level
#: prediction filter (as opposed to id-prefix/benign-floor violations) --
#: lets eval/agent_eval.py (and the ad-hoc analysis in this stop point)
#: count rejections attributable to this filter specifically from
#: agents/base.py's stored retry_reasons, without conflating it with
#: other validation failures.
FLOW_LEVEL_FILTER_PREFIX = "FLOW_LEVEL_FILTER:"

#: Concepts no agent in this pipeline can ever actually check --
#: everything here comes from application-layer/payload data, and this
#: pipeline only ever has flow-level CICFlowMeter statistics (the 77
#: features) plus the selector's own Tier-1/per-source trigger
#: measurements. A live stop-point-2 record showed A5 crediting a
#: hypothesis whose prediction ("packet payloads should contain standard
#: HTTP headers") could never be confirmed or denied by anything this
#: system actually observes -- rejecting these at the source is cheaper
#: and more reliable than hoping A5 notices during synthesis.
_NON_FLOW_LEVEL_CONCEPTS = [
    "payload", "payloads", "header", "headers", "user-agent", "user agent",
    "cookie", "cookies",
    "http request", "http requests", "http response", "http responses",
    "http header", "http headers", "get request", "post request",
    "request body", "response body", "status code", "status codes",
    "mime type", "content-type", "html", "json body", "file content",
    "file format", "packet capture", "uri", "uris", "url", "urls",
]

#: word-boundary matched, not naive substring search -- "uri" as a plain
#: substring check matches inside "during", "security" etc, which would
#: reject perfectly good flow-level predictions for the wrong reason.
#: Mirrors agents/contamination.py's same fix for the same class of bug.
_NON_FLOW_LEVEL_PATTERNS = [
    re.compile(rf"(?<![a-zA-Z0-9]){re.escape(term)}(?![a-zA-Z0-9])", re.IGNORECASE)
    for term in _NON_FLOW_LEVEL_CONCEPTS
]


def validate_claim_ids(agent: str, response: ClaimsResponse) -> None:
    prefix = f"{agent}_c"
    seen = set()
    for claim in response.claims:
        if not claim.claim_id.startswith(prefix):
            raise ValueError(f"claim_id {claim.claim_id!r} must start with {prefix!r}")
        if claim.claim_id in seen:
            raise ValueError(f"duplicate claim_id {claim.claim_id!r}")
        seen.add(claim.claim_id)


def validate_hypothesis_predictions_are_flow_level(
    response: HypothesisResponse, known_feature_names: Set[str]
) -> None:
    """Reject (triggering a retry) any hypothesis whose predicted_feature_
    profile names a feature outside the benign-reference vocabulary
    (``agents.grounding.reference_feature_names`` -- the Tier-1 features
    this project actually has Monday benign data for, since a profile
    entry for anything else could never be empirically tested), or whose
    free-text prediction mentions a non-flow-level concept even while
    nominally listing real feature names elsewhere. Both checks are
    needed: a model could satisfy one and dodge the other (list a real
    feature in predicted_feature_profile while still writing about HTTP
    headers in prose, or vice versa).
    """
    for hyp in response.hypotheses:
        unknown = [p.feature for p in hyp.predicted_feature_profile if p.feature not in known_feature_names]
        if unknown:
            raise ValueError(
                f"{FLOW_LEVEL_FILTER_PREFIX} {hyp.hypothesis_id}: predicted_feature_profile names "
                f"{unknown}, not in the benign-reference feature vocabulary (no empirical data to "
                "test this prediction against)"
            )
        hits = [
            term for term, pattern in zip(_NON_FLOW_LEVEL_CONCEPTS, _NON_FLOW_LEVEL_PATTERNS)
            if pattern.search(hyp.prediction)
        ]
        if hits:
            raise ValueError(
                f"{FLOW_LEVEL_FILTER_PREFIX} {hyp.hypothesis_id}: prediction references "
                f"non-flow-level concept(s) {hits} -- predictions must be checkable against "
                "flow-level features only"
            )


def validate_hypothesis_response(
    agent: str, response: HypothesisResponse, known_feature_names: Iterable[str] = ()
) -> None:
    validate_claim_ids(agent, response)
    h_prefix = f"{agent}_h"
    seen = set()
    for hyp in response.hypotheses:
        if not hyp.hypothesis_id.startswith(h_prefix):
            raise ValueError(f"hypothesis_id {hyp.hypothesis_id!r} must start with {h_prefix!r}")
        if hyp.hypothesis_id in seen:
            raise ValueError(f"duplicate hypothesis_id {hyp.hypothesis_id!r}")
        seen.add(hyp.hypothesis_id)

    if len(response.hypotheses) < MIN_HYPOTHESES_TOTAL:
        raise ValueError(
            f"expected at least {MIN_HYPOTHESES_TOTAL} hypotheses, got {len(response.hypotheses)}"
        )
    benign_count = sum(1 for h in response.hypotheses if h.benign)
    if benign_count < MIN_BENIGN_HYPOTHESES:
        raise ValueError(
            f"expected at least {MIN_BENIGN_HYPOTHESES} benign hypotheses, got {benign_count}"
        )

    known_feature_names = set(known_feature_names)
    if known_feature_names:
        validate_hypothesis_predictions_are_flow_level(response, known_feature_names)


def validate_a5_addresses_all_contradictions(response, required_claim_ids: Set[str]) -> None:
    """Fix 1: A5 must dispose of every claim any hypothesis's own
    generating agent flagged as contradicting it -- either rebutting it
    or lowering benign_plausibility on account of it, but it cannot
    simply not mention it. Enforced mechanically the only way code can:
    the claim_id must appear in A5's own cited_claim_ids. Whether A5's
    rationale actually *engages* with it (rebuts vs. accepts) still
    depends on the model's prose -- code can force acknowledgement, not
    the quality of the reasoning behind it. A live stop-point-2 audit
    found 7/20 records where A5 credited a benign hypothesis while never
    citing a claim that same hypothesis's own author had flagged as
    contradicting it (e.g. a source opening 4,614 flows, flagged by A4
    itself as contradicting its own "single download" hypothesis, never
    mentioned in A5's citations or rationale) -- this closes that gap
    structurally rather than depending on A5 choosing to be thorough.
    """
    missing = required_claim_ids - set(response.cited_claim_ids)
    if missing:
        raise ValueError(
            f"A5 must address every contradicting claim_id by citing it (rebut or lower "
            f"benign_plausibility accordingly) -- missing from cited_claim_ids: {sorted(missing)}"
        )


def validate_a5_credits_a_real_hypothesis(response, known_hypothesis_ids: Set[str]) -> None:
    """credited_hypothesis_id, when given, must name a hypothesis A5 was
    actually shown -- required so agents.grounding.
    apply_empirical_plausibility_cap can look up its real, already-computed
    empirical support rather than trusting an unchecked string. None is
    always valid (A5 crediting no hypothesis at all, e.g. when nothing
    survives)."""
    if response.credited_hypothesis_id is None:
        return
    if response.credited_hypothesis_id not in known_hypothesis_ids:
        raise ValueError(
            f"credited_hypothesis_id {response.credited_hypothesis_id!r} is not one of the "
            f"hypotheses shown: {sorted(known_hypothesis_ids)}"
        )
