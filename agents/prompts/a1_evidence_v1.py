"""A1 -- evidence interpretation. Prompt version 1.

Input: full feature set + the selector's structured trigger reasons.
Output: quantitative observations only, no conclusions about cause.
"""
from __future__ import annotations

from controlplane.record import EscalationRecord

from agents.prompts.render import FRAMING_PREAMBLE, render_features, render_trigger_reasons

PROMPT_VERSION = "a1_evidence_v1"

_INSTRUCTIONS = """\
You are reviewing one network flow's measured statistics. Your only job \
is to state, precisely and quantitatively, what is unusual about these \
measurements and by how much.

Do not speculate about who produced this traffic, why, or what software \
generated it. Do not draw any conclusion about cause. Just describe the \
numbers: which measurements are unusual, in which direction, and by \
roughly what multiple of a typical value. You may reference any feature \
in the full list below, not only the ones already flagged, if you \
notice something else unusual while reading it.

Emit your findings as a list of independent claims. Each claim must:
- have a claim_id of the form "a1_c1", "a1_c2", ... (sequential, unique)
- reference at least one specific feature by its exact name
- use exactly one of these relations per referenced feature: \
greater_than_typical, less_than_typical, equals, absent, present
- carry your own confidence (0 to 1) in that specific claim

Produce between 3 and 8 claims. Do not repeat the same observation in \
more than one claim.
"""


def build_prompt(record: EscalationRecord) -> str:
    return (
        FRAMING_PREAMBLE
        + "\n"
        + _INSTRUCTIONS
        + "\n\nFeatures already flagged as crossing a fitted threshold:\n"
        + render_trigger_reasons(record)
        + "\n\nFull feature set for this flow:\n"
        + render_features(record)
        + "\n"
    )
