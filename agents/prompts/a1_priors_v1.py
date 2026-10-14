"""A1 for the priors run: observations about one flow, read against written priors.
Written fresh; sees only the flow's own properties and the priors text."""
from __future__ import annotations

from agents.prompts.priors_render_v1 import CLAIM_FORMAT
from agents.prompts.priors_text_v1 import PRIORS_TEXT

PROMPT_VERSION = "a1_priors_v1"

_TASK = (
    "You are the first of several reviewers of one network flow. Your job is to observe, not to "
    "conclude. Read the flow's measured values and say which of them are ordinary and which are "
    "not, by the standards below. When an observation relies on one of the numbered statements, "
    "cite it in brackets, for example [S2] or [T3]. Do not guess at intent yet.\n"
    "Give 4 to 8 claims."
)


def build_prompt(flow_text: str) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\n" + CLAIM_FORMAT.format(prefix="a1_c")
        + "\nUse the prefix a1_c.\n\nThe flow:\n" + flow_text + "\n"
    )
