"""A5 for the priors-v3 run: the verdict, valid only with the full condition checklist. Its number
is the verdict; no cap is applied afterwards."""
from __future__ import annotations

from typing import Iterable

from agents.prompts.priors_render_v3 import CHECKLIST_FORMAT, claims_block, hypotheses_block
from agents.prompts.priors_text_v3 import PRIORS_TEXT
from agents.schema import V3Claim, V3Hypothesis

PROMPT_VERSION = "a5_priors_v3"

_TASK = (
    "You make the final call on one network flow: how plausible is it that this flow is benign? "
    "First fill in the checklist: check every condition below yourself against the measured values. "
    "Then decide, using the conditions that hold and the reviewers' work. Two separate reviews follow: "
    "a chain of three reviewers, and one reviewer who worked alone. Where they disagree, decide on "
    "the values.\n\n"
    "Report:\n"
    "- checklist: as specified below.\n"
    "- benign_plausibility, from 0 to 1. Low means this flow is not plausibly benign; high means it "
    "is clearly benign. Use the whole range; a value near the middle is right when the evidence is mixed.\n"
    "- credited_hypothesis_id: the explanation your number rests on most (from either review), or null.\n"
    "- cited_claim_ids: the claim ids that most influenced you (at least one).\n"
    "- rationale: name the conditions that decided it, by id, with the values. \"It looks ordinary\" "
    "is not a reason, and neither is the protocol alone.\n"
    "- confidence, evidence_support, verification: each from 0 to 1."
)


def build_prompt(flow_text: str, chain_claims: Iterable[V3Claim], chain_hypotheses: Iterable[V3Hypothesis],
                 a4_claims: Iterable[V3Claim], a4_hypotheses: Iterable[V3Hypothesis]) -> str:
    return (
        _TASK + "\n\n" + PRIORS_TEXT + "\n" + CHECKLIST_FORMAT.format() + "\n\nThe flow:\n" + flow_text
        + "\n\nChain review, claims:\n" + claims_block(chain_claims)
        + "\n\nChain review, explanations:\n" + hypotheses_block(chain_hypotheses)
        + "\n\nIndependent review, claims:\n" + claims_block(a4_claims)
        + "\n\nIndependent review, explanations:\n" + hypotheses_block(a4_hypotheses) + "\n"
    )
