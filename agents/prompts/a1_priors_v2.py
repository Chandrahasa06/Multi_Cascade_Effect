"""A1 for the priors-v2 run: observations about one flow against protocol-aware priors.
Sees only the flow's own properties (protocol, port, start time, measured values without the
TCP flag indicators) and the priors text v2."""
from __future__ import annotations

from agents.prompts.priors_render_v2 import CLAIM_FORMAT
from agents.prompts.priors_text_v2 import PRIORS_TEXT

PROMPT_VERSION = "a1_priors_v2"

_TASK = (
    "You are the first of several reviewers of one network flow. Your job is to observe, not to "
    "conclude. Start from the flow's protocol, then read its measured values and say which are "
    "ordinary and which are not, by the standards for that protocol below. Cite the numbered "
    "statement each observation relies on, for example [T4] or [D2]. Do not guess at intent yet.\n"
    "Give 4 to 8 claims."
)


def build_prompt(flow_text: str) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\n" + CLAIM_FORMAT.format(prefix="a1_c")
        + "\nUse the prefix a1_c.\n\nThe flow:\n" + flow_text + "\n"
    )
