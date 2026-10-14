"""A2 for the priors-v3 run: the flow's behaviour, tied to conditions where they apply."""
from __future__ import annotations

from typing import Iterable

from agents.prompts.priors_render_v3 import CLAIM_FORMAT, OPTIONAL, claims_block
from agents.prompts.priors_text_v3 import PRIORS_TEXT
from agents.schema import V3Claim

PROMPT_VERSION = "a2_priors_v3"

_TASK = (
    "You are the second reviewer of one network flow. Describe how it behaved: whether the other "
    "side answered, which direction the data went and how much, how long it lasted and how the "
    "packets were spaced, and whether that fits the protocol and destination port. Tie each claim to "
    "the conditions below where one applies. You can see the first reviewer's checks; agree or "
    "disagree with them on the values.\n"
    "Give 3 to 6 claims."
)


def build_prompt(flow_text: str, a1_claims: Iterable[V3Claim]) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\n" + CLAIM_FORMAT.format(require=OPTIONAL)
        + "\nUse the claim prefix a2_c.\n\nThe flow:\n" + flow_text
        + "\n\nFirst reviewer's checks:\n" + claims_block(a1_claims) + "\n"
    )
