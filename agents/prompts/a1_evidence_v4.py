"""A1 -- evidence interpretation. Prompt version 4.

Changes from v3: adds empirical grounding against Monday's real benign
traffic (agents/grounding.py::render_empirical_grounding_block) -- for
every flagged measurement, what benign flows on this network actually
look like (median/p90/p99/p99.5/max, and how many were observed), plus
the nearest real benign flows in normalised feature space. This is the
core fix for agents inventing plausible-sounding benign stories the data
doesn't support: "this flow exceeds every benign flow observed" is a
fact about 566,864 real flows, not a threshold crossing dressed up as
one.

Input: full feature set, the selector's structured trigger reasons, and
empirical grounding against real benign traffic.
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

PROMPT_VERSION = "a1_evidence_v4"

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

Below the flagged measurements you'll also see, for each one, what real \
benign traffic on this network actually looks like (median, p90, p99, \
p99.5, and maximum observed across hundreds of thousands of real benign \
flows), and the nearest real benign flows found in normalised feature \
space. Use this: "1.5x the threshold" and "higher than every benign \
flow ever observed on this network" are very different claims, and you \
should say which one is actually true when you can.

The flagged measurements above the full feature list use the \
"{TRIGGER_FEATURE_PREFIX}" prefix (e.g. "{TRIGGER_FEATURE_PREFIX}flow_duration") \
and are on a DIFFERENT scale than the full feature list below -- when \
you cite one of those flagged measurements, use its exact prefixed name, \
not the bare name from the full feature list, even if they look similar. \
Each flagged measurement lists observed_value, threshold, and \
ratio_to_threshold on separate lines -- when you assert a value for one \
of these features, use its observed_value. ratio_to_threshold is a \
multiplier describing how extreme observed_value is, not a value in its \
own right; never assert it as if it were the measurement.

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


def build_prompt(record: EscalationRecord, empirical_grounding_block: str) -> str:
    return (
        FRAMING_PREAMBLE
        + "\n"
        + CALIBRATION_NOTE
        + "\n"
        + _INSTRUCTIONS
        + "\n\nFeatures already flagged as crossing a fitted threshold:\n"
        + render_trigger_reasons(record)
        + "\n\nEmpirical grounding against real benign traffic on this network:\n"
        + empirical_grounding_block
        + "\n\nFull feature set for this flow:\n"
        + render_features(record)
        + "\n"
    )
