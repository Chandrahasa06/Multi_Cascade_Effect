"""Formatting for the priors-v2 run. Differences from priors_render_v1:
  - the flow block shows the flow's own protocol (TCP or UDP), recovered from its own CSV row;
  - the 12 TCP flag columns are not shown (results/flag_column_diagnosis.md: none can be trusted);
  - the explanation format spells out the structure (exactly three: one benign, one hostile,
    one of either kind), because in v1 five records failed by giving only two.
Claims and explanation blocks are formatted exactly as in v1.
"""
from __future__ import annotations

from typing import Dict, Optional

from agents.prompts.priors_render_v1 import CLAIM_FORMAT, claims_block, hypotheses_block  # noqa: F401

#: Not shown to any agent in v2 (see module docstring).
FLAG_COLUMNS = frozenset({
    "SYN Flag Count", "ACK Flag Count", "FIN Flag Count", "PSH Flag Count", "RST Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
})
EXCLUDED_PROFILE_WORDS = ("Header",)


def shown_features(features: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in features.items() if k not in FLAG_COLUMNS}


def flow_block(features: Dict[str, float], protocol: Optional[str], time_of_day: Optional[str]) -> str:
    shown = shown_features(features)
    port = shown.get("Destination Port")
    lines = [
        f"Protocol: {protocol or 'not recorded'}",
        f"Destination port: {int(port) if port is not None else 'not recorded'}",
        f"Start time (local clock, minute resolution): {time_of_day or 'not recorded'}",
        "Measured values for this flow (TCP flag indicators are not shown):",
    ]
    lines += [f"  {name} = {value:g}" for name, value in sorted(shown.items())]
    return "\n".join(lines)


def profile_feature_names(features: Dict[str, float]) -> list:
    return sorted(n for n in shown_features(features) if not any(w in n for w in EXCLUDED_PROFILE_WORDS))


HYPOTHESIS_FORMAT = """\
Output format for explanations. Give EXACTLY THREE: explanation 1 benign, explanation 2 \
hostile, explanation 3 either benign or hostile (your next most plausible). Fewer than three \
is rejected. Each has:
- hypothesis_id: the prefix given below, then a number (1, 2, 3).
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
