"""A2 for the priors-v2 run: the flow's behaviour. Sees the flow, the priors v2 and A1."""
from __future__ import annotations

from typing import Iterable

from agents.prompts.priors_render_v2 import CLAIM_FORMAT, claims_block
from agents.prompts.priors_text_v2 import PRIORS_TEXT
from agents.schema import PriorsClaim as Claim

PROMPT_VERSION = "a2_priors_v2"

_TASK = (
    "You are the second reviewer of one network flow. Describe how it behaved: whether the other "
    "side answered, which direction the data went and how much, how long it lasted and how the "
    "packets were spaced in time, and whether that fits the protocol and destination port. Cite the "
    "numbered statements you rely on. You can see the first reviewer's observations; agree or "
    "disagree with them on the evidence.\n"
    "Give 3 to 6 claims."
)


def build_prompt(flow_text: str, a1_claims: Iterable[Claim]) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\n" + CLAIM_FORMAT.format(prefix="a2_c")
        + "\nUse the prefix a2_c.\n\nThe flow:\n" + flow_text
        + "\n\nFirst reviewer's observations:\n" + claims_block(a1_claims) + "\n"
    )
