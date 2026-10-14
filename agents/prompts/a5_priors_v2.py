"""A5 for the priors-v2 run: the verdict. Sees the flow, the priors v2, the three-reviewer
chain and the independent reviewer. Its number is the verdict; no cap is applied afterwards."""
from __future__ import annotations

from typing import Iterable

from agents.prompts.priors_render_v2 import claims_block, hypotheses_block
from agents.prompts.priors_text_v2 import PRIORS_TEXT
from agents.schema import Hypothesis, PriorsClaim as Claim

PROMPT_VERSION = "a5_priors_v2"

_TASK = (
    "You make the final call on one network flow: how plausible is it that this flow is benign? "
    "Base it on the flow's own measured values, the standards below for its protocol, and the "
    "reviewers' work. Two separate reviews follow: a chain of three reviewers, and one reviewer who "
    "worked alone without seeing the chain. Where they disagree, decide on the evidence.\n\n"
    "Report:\n"
    "- benign_plausibility, from 0 to 1. Low means this flow is not plausibly benign; high means it "
    "is clearly benign. Use the whole range; a value near the middle is right when the evidence is "
    "mixed.\n"
    "- credited_hypothesis_id: the explanation your number rests on most (from either review), or null.\n"
    "- cited_claim_ids: the claim ids that most influenced you (at least one).\n"
    "- rationale: name the specific measured values and the numbered statements that decided it. "
    "\"It looks ordinary\" is not a reason, and neither is the protocol alone.\n"
    "- confidence, evidence_support, verification: each from 0 to 1, as for the reviewers."
)


def build_prompt(flow_text: str, chain_claims: Iterable[Claim], chain_hypotheses: Iterable[Hypothesis],
                 a4_claims: Iterable[Claim], a4_hypotheses: Iterable[Hypothesis]) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\nThe flow:\n" + flow_text
        + "\n\nChain review, claims:\n" + claims_block(chain_claims)
        + "\n\nChain review, explanations:\n" + hypotheses_block(chain_hypotheses)
        + "\n\nIndependent review, claims:\n" + claims_block(a4_claims)
        + "\n\nIndependent review, explanations:\n" + hypotheses_block(a4_hypotheses) + "\n"
    )
