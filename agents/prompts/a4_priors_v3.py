"""A4 for the priors-v3 run: one independent reviewer, alone. Checks the conditions and proposes
explanations tied to them. Never sees the other reviewers' work."""
from __future__ import annotations

from agents.prompts.priors_render_v3 import CLAIM_FORMAT, HYPOTHESIS_FORMAT, OPTIONAL
from agents.prompts.priors_text_v3 import PRIORS_TEXT

PROMPT_VERSION = "a4_priors_v3"

_TASK = (
    "You are an independent reviewer of one network flow, working alone. Check the conditions below "
    "against the measured values, report what you found as claims tied to condition ids, and propose "
    "benign and hostile explanations, each saying which conditions it accounts for. A hostile "
    "explanation is not weaker just because the flow looks ordinary. Give 4 to 8 claims."
)


def build_prompt(flow_text: str, profile_names: list) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\n" + CLAIM_FORMAT.format(require=OPTIONAL)
        + "\nUse the claim prefix a4_c.\n\n" + HYPOTHESIS_FORMAT.format(names=", ".join(profile_names))
        + "\nUse the explanation prefix a4_h.\n\nThe flow:\n" + flow_text + "\n"
    )
