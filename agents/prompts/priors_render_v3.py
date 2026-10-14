"""Formatting for the priors-v3 run. Same flow block as v2 (protocol shown with its number; the
12 TCP flag columns not shown). Claims and explanations carry condition ids, and the output
formats require them."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from agents.prompts.priors_render_v2 import FLAG_COLUMNS, profile_feature_names, shown_features  # noqa: F401
from agents.prompts.priors_text_v3 import CONDITION_IDS
from agents.schema import V3Claim, V3Hypothesis

PROTOCOL_NUMBERS = {"TCP": 6, "UDP": 17}


def flow_block(features: Dict[str, float], protocol: Optional[str], time_of_day: Optional[str]) -> str:
    shown = shown_features(features)
    port = shown.get("Destination Port")
    proto = f"{protocol} (Protocol = {PROTOCOL_NUMBERS[protocol]})" if protocol in PROTOCOL_NUMBERS else "not recorded"
    lines = [
        f"Protocol: {proto}",
        f"Destination port: {int(port) if port is not None else 'not recorded'}",
        f"Start time (local clock, minute resolution): {time_of_day or 'not recorded'}",
        "Measured values for this flow (TCP flag indicators are not shown):",
    ]
    lines += [f"  {name} = {value:g}" for name, value in sorted(shown.items())]
    return "\n".join(lines)


def claims_block(claims: Iterable[V3Claim]) -> str:
    out = []
    for c in claims:
        refs = ", ".join(f"{r.name} = {r.value:g}" for r in c.referenced_features)
        conds = ", ".join(c.condition_ids) or "none"
        out.append(f"  {c.claim_id} [conditions: {conds}]: {c.statement}" + (f" ({refs})" if refs else "")
                   + f" (confidence {c.confidence:.2f})")
    return "\n".join(out) or "  (none)"


def hypotheses_block(hypotheses: Iterable[V3Hypothesis]) -> str:
    out = []
    for h in hypotheses:
        prof = "; ".join(
            f"{p.feature} in [{'' if p.expected_min is None else format(p.expected_min, 'g')}, "
            f"{'' if p.expected_max is None else format(p.expected_max, 'g')}]"
            for p in h.predicted_feature_profile
        )
        out.append(
            f"  {h.hypothesis_id} ({'benign' if h.benign else 'hostile'}, prior {h.prior_plausibility:.2f}) "
            f"[conditions: {', '.join(h.condition_ids) or 'none'}]: {h.description}\n"
            f"      predicts: {h.prediction}\n"
            f"      expected ranges: {prof}\n"
            f"      supported by: {', '.join(h.supporting_claim_ids) or 'none'}; "
            f"against: {', '.join(h.contradicting_claim_ids) or 'none'}"
        )
    return "\n".join(out) or "  (none)"


_IDS = ", ".join(CONDITION_IDS)

CLAIM_FORMAT = """\
Output format for claims. Each claim has:
- claim_id: the prefix given below, then a number.
- statement: one sentence.
- condition_ids: the condition ids (""" + _IDS + """) the claim is about.{require}
- referenced_features: the measured values the statement rests on, each as a name from the \
list of measured values and its value copied exactly as listed, for example \
{{"name": "Destination Port", "value": 80}}.
- confidence, from 0 to 1.
Also report, for your answer as a whole, three numbers from 0 to 1: confidence; \
evidence_support (how well your claims would hold if checked against the measured values); \
verification (how far you expect an independent reviewer of the same flow to agree with you)."""

REQUIRED = " Every claim must list at least one."
OPTIONAL = " List them where they apply; leave empty otherwise."

HYPOTHESIS_FORMAT = """\
Output format for explanations. Give EXACTLY THREE: explanation 1 benign, explanation 2 \
hostile, explanation 3 either benign or hostile (your next most plausible). Fewer than three \
is rejected. Each has:
- hypothesis_id: the prefix given below, then a number (1, 2, 3).
- description: what activity this would be.
- benign: true or false.
- condition_ids: the condition ids (""" + _IDS + """) this explanation accounts for. At least one.
- prior_plausibility, from 0 to 1.
- prediction: one checkable sentence about this flow's measured values if the explanation \
is right. Talk only about measured values. Do not use the words "header" or "payload \
contents" in a prediction; they are rejected automatically.
- predicted_feature_profile: one or more expected ranges, each a feature name from this list \
with expected_min and/or expected_max: {names}
- supporting_claim_ids and contradicting_claim_ids: claim ids that support or count against it."""

CHECKLIST_FORMAT = """\
Checklist (required; a verdict without it is rejected). For EVERY condition """ + _IDS + """, \
exactly one entry with:
- condition_id;
- holds: "yes", "no" or "not_applicable" (only for a condition scoped to the other protocol);
- observed: the measured value(s) you read to decide, each as {{"name": ..., "value": ...}} \
with the name exactly as listed and the value copied exactly. For "not_applicable", give \
{{"name": "Protocol", "value": 6}} for TCP or 17 for UDP."""
