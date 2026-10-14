"""A4 for the priors run: one independent reviewer doing the whole analysis alone. It sees
only the flow and the priors, never the other reviewers' work."""
from __future__ import annotations

from agents.prompts.priors_render_v1 import CLAIM_FORMAT, HYPOTHESIS_FORMAT
from agents.prompts.priors_text_v1 import PRIORS_TEXT

PROMPT_VERSION = "a4_priors_v1"

_TASK = (
    "You are an independent reviewer of one network flow, working alone. Do three things: observe "
    "which measured values are ordinary or not by the standards below; describe how the "
    "connection behaved; and propose benign and hostile explanations with what each predicts. "
    "Cite the numbered statements you rely on. A hostile explanation is not weaker just because "
    "the flow looks ordinary. Give 4 to 8 claims."
)


def build_prompt(flow_text: str, profile_names: list) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\n" + CLAIM_FORMAT.format(prefix="a4_c")
        + "\nUse the prefix a4_c for claims.\n\n" + HYPOTHESIS_FORMAT.format(names=", ".join(profile_names))
        + "\nUse the prefix a4_h for explanations.\n\nThe flow:\n" + flow_text + "\n"
    )
