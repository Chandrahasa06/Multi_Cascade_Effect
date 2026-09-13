"""A5 -- verdict. Prompt version 1.

Input: full feature set, the chain's combined claims+hypotheses
(A1+A2+A3), A4's independent claims+hypotheses, and the mechanical
agreement between the two sides (agents/verification.py's output --
computed in code, not by either reviewer).

Output: exactly one of three verdicts. Never asks about a zero-day --
that's a conclusion drawn from the data afterwards (step 3), not
something the model asserts here.
"""
from __future__ import annotations

from typing import Dict, Iterable

from controlplane.record import EscalationRecord

from agents.prompts.render import FRAMING_PREAMBLE, render_claims, render_features, render_hypotheses
from agents.schema import Claim, Hypothesis
from agents.verification import VerificationCounts, render_verification_summary

PROMPT_VERSION = "a5_verdict_v1"

_INSTRUCTIONS = """\
You are the final reviewer for one network flow. Below you'll see the \
combined output of a three-stage review chain (evidence, behaviour, \
candidate explanations) and a separate, fully independent review of the \
same flow performed without seeing the chain's output. You'll also see \
a mechanical comparison of which claims from the two sides corroborate, \
contradict, or simply don't overlap -- this comparison was computed by \
matching claims against each other and the raw feature data, not by \
either reviewer's own judgement.

Decide among exactly three verdicts:
- consistent_with_benign: the evidence is unremarkable, or fully \
consistent with ordinary use.
- anomalous_but_explicable: the evidence is genuinely unusual, but at \
least one benign explanation considered below survives scrutiny -- it \
is not contradicted by the evidence and remains plausible.
- anomalous_and_unexplained: the evidence is unusual, and every benign \
explanation considered below is either contradicted by the evidence or \
implausible given it. Use this only when no benign story actually \
fits, not merely because something is unusual.

Base your decision only on whether the evidence is unremarkable, \
explicable, or unexplained -- do not try to name what kind of activity \
this is or how severe it is; that is out of scope here.

Cite the claim_ids (from either side) that most influenced your \
decision, and give your own confidence (0 to 1) in the verdict. In your \
rationale, explain in a sentence or two why the surviving benign \
hypothesis does or doesn't hold up -- or why none does.
"""


def build_prompt(
    record: EscalationRecord,
    chain_claims: Iterable[Claim],
    chain_hypotheses: Iterable[Hypothesis],
    a4_claims: Iterable[Claim],
    a4_hypotheses: Iterable[Hypothesis],
    verification_by_agent: Dict[str, VerificationCounts],
) -> str:
    return (
        FRAMING_PREAMBLE
        + "\n"
        + _INSTRUCTIONS
        + "\n\nFull feature set for this flow:\n"
        + render_features(record)
        + "\n\nChain review -- combined claims (evidence + behaviour + hypothesis stages):\n"
        + render_claims(chain_claims)
        + "\n\nChain review -- candidate explanations:\n"
        + render_hypotheses(chain_hypotheses)
        + "\n\nIndependent review -- claims:\n"
        + render_claims(a4_claims)
        + "\n\nIndependent review -- candidate explanations:\n"
        + render_hypotheses(a4_hypotheses)
        + "\n\nMechanical agreement between chain and independent review "
        "(per chain stage: corroborated / contradicted / not addressed):\n"
        + render_verification_summary(verification_by_agent)
        + "\n"
    )
