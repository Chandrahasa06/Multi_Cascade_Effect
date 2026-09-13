"""A5 -- benign-plausibility scoring. Prompt version 2.

Changes from v1 (see STATUS.md's stop-point-2 findings): the categorical
three-way verdict is gone. A5 now reports benign_plausibility, a
continuous [0,1] estimate of how plausible the strongest benign
hypothesis is once checked against the independent evidence -- the
three-way label is derived from this score in code, at a swept
threshold (agents.schema.derive_verdict), not asserted by the model.
Also: A5 must now explicitly check each benign hypothesis's stated
prediction (a3_hypotheses_v2 / a4_replication_v2) against A4's
independent claims before crediting it -- a benign hypothesis nobody
tested is an escape hatch, not corroboration. Adds CALIBRATION_NOTE.

Input: full feature set, the chain's full output, A4's independent
output, and the mechanical agreement between them (agents/verification.py's
output -- computed in code, not by either reviewer).
"""
from __future__ import annotations

from typing import Dict, Iterable

from controlplane.record import EscalationRecord

from agents.prompts.render import CALIBRATION_NOTE, FRAMING_PREAMBLE, render_claims, render_features, render_hypotheses
from agents.schema import Claim, Hypothesis
from agents.verification import VerificationCounts, render_verification_summary

PROMPT_VERSION = "a5_verdict_v2"

_INSTRUCTIONS = """\
You are the final reviewer for one network flow. Below you'll see the \
combined output of a three-stage review chain (evidence, behaviour, \
candidate explanations) and a separate, fully independent review of the \
same flow performed without seeing the chain's output. You'll also see \
a mechanical comparison of which claims from the two sides corroborate, \
contradict, or simply don't overlap -- this comparison was computed by \
matching claims against each other and the raw feature data, not by \
either reviewer's own judgement.

Each candidate explanation below (from either side) states a \
prediction: a concrete, checkable claim about what the evidence would \
look like if that explanation were true. Before crediting any benign \
explanation, check whether its stated prediction actually holds against \
the INDEPENDENT review's claims (the side that did NOT generate that \
explanation) -- a benign explanation whose prediction is contradicted \
by the independent evidence, or that nobody's claims actually speak to \
either way, is weaker than one whose prediction is positively confirmed \
by evidence the explanation's own author never saw. Do not credit a \
benign explanation just because it sounds plausible in isolation.

Report a single number: benign_plausibility, from 0 to 1 -- your \
estimate of how plausible the STRONGEST benign explanation is, once you \
have checked its prediction against the independent evidence as \
described above. 0 means no benign explanation survives that check at \
all; 1 means a benign explanation is fully confirmed and nothing about \
the evidence is left unexplained. Values in between reflect a benign \
explanation that is plausible but not fully confirmed, or only \
partially consistent with the independent evidence. Do not round to a \
convenient category -- report your actual estimate, including values \
close to 0.5 if that's genuinely where the evidence leaves you.

Do not try to name what kind of activity this is or how severe it is; \
that is out of scope here. Your job is only to estimate how well the \
best available benign explanation survives contact with the \
independent evidence.

Cite the claim_ids (from either side) that most influenced your \
estimate, and give your own confidence (0 to 1) in the estimate itself \
(how sure you are of the benign_plausibility number, separate from what \
that number is). In your rationale, explain in a sentence or two which \
benign explanation you checked, what its prediction was, and whether \
the independent evidence confirmed, contradicted, or simply didn't \
address that prediction.
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
        + CALIBRATION_NOTE
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
