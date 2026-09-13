"""Single-LLM control. Prompt version 1.

Same input as the five-agent pipeline (full feature set + trigger
reasons), one call, the same three-way verdict -- no chain, no blind
replication, nothing to compare it against but itself.
"""
from __future__ import annotations

from controlplane.record import EscalationRecord

from agents.prompts.render import FRAMING_PREAMBLE, render_features, render_trigger_reasons

PROMPT_VERSION = "baseline_v1"

_INSTRUCTIONS = """\
You are reviewing one network flow's measured statistics. In a single \
pass, do the following:

1. Note what is quantitatively unusual about these measurements and by \
how much.
2. Consider what the two communicating parties appear to be doing at a \
behavioural level.
3. Weigh candidate explanations, including ordinary/benign ones (a \
misconfigured client, a monitoring or health-check system, a scheduled \
backup job, a load-testing tool, a client- or server-side bug causing \
repeated retries) as well as less ordinary ones, if the evidence points \
that way.
4. Decide among exactly three verdicts:
   - consistent_with_benign: the evidence is unremarkable, or fully \
consistent with ordinary use.
   - anomalous_but_explicable: the evidence is genuinely unusual, but a \
benign explanation survives scrutiny.
   - anomalous_and_unexplained: the evidence is unusual, and no benign \
explanation actually fits it.

Do not try to name what kind of activity this is or how severe it is; \
that is out of scope here. Base your decision only on whether the \
evidence is unremarkable, explicable, or unexplained.

Cite the specific feature names that most influenced your decision, and \
give your own confidence (0 to 1) in the verdict. In your rationale, \
explain in a sentence or two why a benign explanation does or doesn't \
hold up.
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
