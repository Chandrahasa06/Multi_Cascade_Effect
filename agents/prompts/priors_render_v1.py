"""Formatting for the priors run (a1..a5 *_priors_v1). Written fresh for this run.

Everything here formats either the flow's own properties or earlier agents' output. There
is no parameter through which a computed statistic could enter.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from agents.schema import Hypothesis, PriorsClaim as Claim

#: Feature names a hypothesis profile may not use: the flow-level text filter rejects the
#: word "header", so header-length features cannot appear in a prediction without a retry.
EXCLUDED_PROFILE_WORDS = ("Header",)


def flow_block(features: Dict[str, float], time_of_day: Optional[str]) -> str:
    port = features.get("Destination Port")
    lines = [
        f"Destination port: {int(port) if port is not None else 'not recorded'}",
        "Protocol: not recorded",
        f"Start time (local clock, minute resolution): {time_of_day or 'not recorded'}",
        "Measured values for this flow:",
    ]
    lines += [f"  {name} = {value:g}" for name, value in sorted(features.items())]
    return "\n".join(lines)


def profile_feature_names(features: Dict[str, float]) -> list:
    return sorted(n for n in features if not any(w in n for w in EXCLUDED_PROFILE_WORDS))


def claims_block(claims: Iterable[Claim]) -> str:
    out = []
    for c in claims:
        refs = ", ".join(f"{r.name} = {r.value:g}" for r in c.referenced_features)
        out.append(f"  {c.claim_id}: {c.statement}" + (f" [{refs}]" if refs else "") + f" (confidence {c.confidence:.2f})")
    return "\n".join(out) or "  (none)"


def hypotheses_block(hypotheses: Iterable[Hypothesis]) -> str:
    out = []
    for h in hypotheses:
        prof = "; ".join(
            f"{p.feature} in [{'' if p.expected_min is None else format(p.expected_min, 'g')}, "
            f"{'' if p.expected_max is None else format(p.expected_max, 'g')}]"
            for p in h.predicted_feature_profile
        )
        out.append(
            f"  {h.hypothesis_id} ({'benign' if h.benign else 'hostile'}, prior {h.prior_plausibility:.2f}): {h.description}\n"
            f"      predicts: {h.prediction}\n"
            f"      expected ranges: {prof}\n"
            f"      supported by: {', '.join(h.supporting_claim_ids) or 'none'}; "
            f"against: {', '.join(h.contradicting_claim_ids) or 'none'}"
        )
    return "\n".join(out) or "  (none)"


CLAIM_FORMAT = """\
Output format for claims. Each claim has:
- claim_id: the prefix given below, then a number (for example {prefix}1).
- statement: one sentence.
- referenced_features: the measured values the statement rests on, each as a name from the list of measured values and its value copied exactly as listed, for example {{"name": "Destination Port", "value": 80}}.
- confidence, from 0 to 1.
Also report, for your answer as a whole, three numbers from 0 to 1: confidence; \
evidence_support (how well your claims would hold if checked against the measured values); \
verification (how far you expect an independent reviewer of the same flow to agree with you)."""

HYPOTHESIS_FORMAT = """\
Output format for explanations. Give AT LEAST 3 and at most 5 (two is rejected). At least \
one must be benign and at least one hostile. Each has:
- hypothesis_id: the prefix given below, then a number.
- description: what activity this would be.
- benign: true or false.
- prior_plausibility, from 0 to 1.
- prediction: one checkable sentence about this flow's measured values if the explanation \
is right. Talk only about measured values: never about packet contents or anything the \
measurements do not show. Do not use the words "header" or "payload contents" in a \
prediction; they are rejected automatically.
- predicted_feature_profile: one or more expected ranges, each a feature name from this list \
with expected_min and/or expected_max: {names}
- supporting_claim_ids and contradicting_claim_ids: claim ids that support or count against it."""
