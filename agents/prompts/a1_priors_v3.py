"""A1 for the priors-v3 run: checks the conditions against the flow. Every claim must cite at
least one condition id. Sees only the flow's own properties and the condition list."""
from __future__ import annotations

from agents.prompts.priors_render_v3 import CLAIM_FORMAT, REQUIRED
from agents.prompts.priors_text_v3 import PRIORS_TEXT

PROMPT_VERSION = "a1_priors_v3"

_TASK = (
    "You are the first of several reviewers of one network flow. Check the conditions below "
    "against the flow's measured values. Report what you found as claims: each claim says which "
    "condition(s) it is about, whether they hold, and the values you read. Do not describe "
    "impressions; report checks. Do not guess at intent yet.\n"
    "Give 4 to 8 claims."
)


def build_prompt(flow_text: str) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\n" + CLAIM_FORMAT.format(require=REQUIRED)
        + "\nUse the claim prefix a1_c.\n\nThe flow:\n" + flow_text + "\n"
    )
