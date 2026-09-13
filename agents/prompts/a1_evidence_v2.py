"""A1 -- evidence interpretation. Prompt version 2.

Changes from v1 (see STATUS.md's stop-point-2 findings):
- trigger-reason feature names are now shown with the trigger_ prefix
  (agents/prompts/render.py::render_trigger_reasons), and this prompt
  tells A1 to use that exact prefixed name when citing one, instead of
  colliding with the differently-scaled CICFlowMeter feature of the same
  bare name.
- adds CALIBRATION_NOTE so "typical-range threshold" has a stated scale
  (what fraction of ordinary traffic it represents) instead of being a
  bare, uncalibrated number.

Input: full feature set plus the selector's structured trigger reasons.
Output: quantitative observations only, no conclusions about cause.
"""
from __future__ import annotations

from controlplane.record import EscalationRecord

from agents.prompts.render import (
    CALIBRATION_NOTE,
    FRAMING_PREAMBLE,
    TRIGGER_FEATURE_PREFIX,
    render_features,
    render_trigger_reasons,
)

PROMPT_VERSION = "a1_evidence_v2"

_INSTRUCTIONS = f"""\
You are reviewing one network flow's measured statistics. Your only job \
is to state, precisely and quantitatively, what is unusual about these \
measurements and by how much.

Do not speculate about who produced this traffic, why, or what software \
generated it. Do not draw any conclusion about cause. Just describe the \
numbers: which measurements are unusual, in which direction, and by \
roughly what multiple of a typical value. You may reference any feature \
in the full list below, not only the ones already flagged, if you \
notice something else unusual while reading it.

The flagged measurements above the full feature list use the \
"{TRIGGER_FEATURE_PREFIX}" prefix (e.g. "{TRIGGER_FEATURE_PREFIX}flow_duration") \
and are on a DIFFERENT scale than the full feature list below -- when \
you cite one of those flagged measurements, use its exact prefixed name, \
not the bare name from the full feature list, even if they look similar.

Emit your findings as a list of independent claims. Each claim must:
- have a claim_id of the form "a1_c1", "a1_c2", ... (sequential, unique)
- reference at least one specific feature by its exact name (including \
the "{TRIGGER_FEATURE_PREFIX}" prefix if the feature came from the \
flagged-measurements list, not the full feature list)
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
        + CALIBRATION_NOTE
        + "\n"
        + _INSTRUCTIONS
        + "\n\nFeatures already flagged as crossing a fitted threshold:\n"
        + render_trigger_reasons(record)
        + "\n\nFull feature set for this flow:\n"
        + render_features(record)
        + "\n"
    )
