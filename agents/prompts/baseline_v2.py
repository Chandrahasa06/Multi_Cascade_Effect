"""Single-LLM control. Prompt version 2.

Same input as the five-agent pipeline gets: the full feature set,
trigger reasons, AND the same empirical grounding (benign reference
distribution + nearest-neighbour context, controlplane/reference.py)
A1/A3/A4/A5 all see. v1 omitted the grounding block entirely -- which
meant a chain-vs-baseline comparison built on it couldn't tell "the
chain reasons better" apart from "the chain simply saw more data". Also
asks directly for a continuous benign_plausibility (same [0,1] scale as
A5Response), not a categorical verdict -- so its output is head-to-head
comparable to A5's own number, not only to a label derived from it.

Still no chain, no A4, no trust scoring, no hypotheses/claims apparatus:
one call, one pass, nothing to compare its own claims against.
"""
from __future__ import annotations

from controlplane.record import EscalationRecord

from agents.prompts.render import (
    CALIBRATION_NOTE,
    FRAMING_PREAMBLE,
    render_features,
    render_trigger_reasons,
)

PROMPT_VERSION = "baseline_v2"

_INSTRUCTIONS = """\
You are reviewing one network flow's measured statistics, in a single \
pass. Your question is the same one this project's full review process \
answers: given the benign traffic actually observed on this network, \
how plausible is it that this flow is ordinary?

1. Note what is quantitatively unusual about these measurements and by \
how much, using the benign reference numbers and nearest-neighbour \
grounding below -- not just your own intuition about what counts as \
normal.
2. Consider what the two communicating parties appear to be doing at a \
behavioural level.
3. Weigh candidate explanations, including ordinary/benign ones (a \
misconfigured client, a monitoring or health-check system, a scheduled \
backup job, a load-testing tool, a client- or server-side bug causing \
repeated retries) as well as less ordinary ones, if the evidence points \
that way.

Report benign_plausibility, from 0 to 1: 0 means nothing benign \
explains this flow; 1 means a benign explanation is fully confirmed by \
the benign reference data below, with nothing left unexplained. Do not \
round to a convenient category -- report your actual estimate, \
including values close to 0.5 if that is genuinely where the evidence \
leaves you.

Do not try to name what kind of activity this is or how severe it is; \
that is out of scope here.

Cite the specific feature names that most influenced your decision, and \
give your own confidence (0 to 1) in your estimate. In your rationale, \
explain in a sentence or two why a benign explanation does or doesn't \
hold up, referencing the benign reference / nearest-neighbour numbers \
where relevant.
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
