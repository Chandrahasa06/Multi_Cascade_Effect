"""A3 for the priors-v3 run: explanations, each tied to the conditions it accounts for."""
from __future__ import annotations

from typing import Iterable

from agents.prompts.priors_render_v3 import CLAIM_FORMAT, HYPOTHESIS_FORMAT, OPTIONAL, claims_block
from agents.prompts.priors_text_v3 import PRIORS_TEXT
from agents.schema import V3Claim

PROMPT_VERSION = "a3_priors_v3"

_TASK = (
    "You are the third reviewer of one network flow. Propose benign and hostile explanations. Each "
    "must say which conditions it accounts for: a benign explanation must explain the conditions that "
    "hold; a hostile one must say which conditions point to it. A hostile explanation is not weaker "
    "just because the flow looks ordinary. Also give 1 to 3 claims of your own."
)


def build_prompt(flow_text: str, a1_claims: Iterable[V3Claim], a2_claims: Iterable[V3Claim], profile_names: list) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\n" + CLAIM_FORMAT.format(require=OPTIONAL)
        + "\nUse the claim prefix a3_c.\n\n" + HYPOTHESIS_FORMAT.format(names=", ".join(profile_names))
        + "\nUse the explanation prefix a3_h.\n\nThe flow:\n" + flow_text
        + "\n\nFirst reviewer's checks:\n" + claims_block(a1_claims)
        + "\n\nSecond reviewer's behaviour description:\n" + claims_block(a2_claims) + "\n"
    )
