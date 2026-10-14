"""A3 for the priors run: candidate explanations. Sees the flow, the priors, A1 and A2."""
from __future__ import annotations

from typing import Iterable

from agents.prompts.priors_render_v1 import CLAIM_FORMAT, HYPOTHESIS_FORMAT, claims_block
from agents.prompts.priors_text_v1 import PRIORS_TEXT
from agents.schema import PriorsClaim as Claim

PROMPT_VERSION = "a3_priors_v1"

_TASK = (
    "You are the third reviewer of one network flow. Propose explanations for what this flow is, "
    "both benign and hostile, and say what each would predict about the measured values. Weigh "
    "each against the numbered statements and the earlier reviewers' claims. A hostile explanation "
    "is not weaker just because the flow looks ordinary. Also give 1 to 3 claims of your own."
)


def build_prompt(flow_text: str, a1_claims: Iterable[Claim], a2_claims: Iterable[Claim], profile_names: list) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\n" + CLAIM_FORMAT.format(prefix="a3_c")
        + "\nUse the prefix a3_c for claims.\n\n" + HYPOTHESIS_FORMAT.format(names=", ".join(profile_names))
        + "\nUse the prefix a3_h for explanations.\n\nThe flow:\n" + flow_text
        + "\n\nFirst reviewer's observations:\n" + claims_block(a1_claims)
        + "\n\nSecond reviewer's behaviour description:\n" + claims_block(a2_claims) + "\n"
    )
