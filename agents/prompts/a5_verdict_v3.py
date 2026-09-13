"""A5 -- benign-plausibility scoring. Prompt version 3.

Changes from v2 (see STATUS.md's stop-point-2 findings): v2 asked A5 to
check a hypothesis's prediction against independent evidence, but left
it to A5's own thoroughness -- a live audit found 7/20 records where A5
credited a benign hypothesis while never citing a claim that same
hypothesis's own generating agent had flagged as contradicting it (e.g.
a source opening 4,614 flows, which A4 itself flagged as contradicting
its own "single download" hypothesis, went entirely unmentioned). This
version makes it structural: every claim_id any hypothesis flags as
contradicting it is listed explicitly and enforced in code
(agents/validators.py::validate_a5_addresses_all_contradictions) -- a
response missing one of them is invalid and gets retried, not merely
encouraged to do better.

Input: full feature set, the chain's full output, A4's independent
output, the mechanical agreement between them (agents/verification.py's
output), and the full set of claim_ids A5 must dispose of.
"""
from __future__ import annotations

from typing import Dict, Iterable

from controlplane.record import EscalationRecord

from agents.prompts.render import (
    CALIBRATION_NOTE,
    FRAMING_PREAMBLE,
    render_claims,
    render_features,
    render_hypotheses,
    render_required_claim_ids,
)
from agents.schema import Claim, Hypothesis
from agents.verification import VerificationCounts, render_verification_summary

PROMPT_VERSION = "a5_verdict_v3"

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
prediction and lists which claim_ids support it and which contradict \
it. Before crediting any benign explanation, check whether its stated \
prediction actually holds against the INDEPENDENT review's claims (the \
side that did NOT generate that explanation) -- a benign explanation \
whose prediction is contradicted by the independent evidence, or that \
nobody's claims actually speak to either way, is weaker than one whose \
prediction is positively confirmed by evidence the explanation's own \
author never saw. Do not credit a benign explanation just because it \
sounds plausible in isolation.

MANDATORY: every claim_id listed below under "claims you must address" \
was flagged by SOME hypothesis's own generating reviewer as \
contradicting that hypothesis. You must cite every single one of them \
in cited_claim_ids -- this is checked mechanically, and a response \
missing even one will be rejected and you will be asked to try again. \
Citing a claim_id there does not mean you have to agree it's fatal: for \
each one, either explain in your rationale why it doesn't actually \
undermine the hypothesis you're crediting, or let it lower your \
benign_plausibility estimate accordingly. What you may not do is simply \
not mention it.

Report a single number: benign_plausibility, from 0 to 1 -- your \
estimate of how plausible the STRONGEST benign explanation is, once you \
have checked its prediction against the independent evidence and \
addressed every contradicting claim as described above. 0 means no \
benign explanation survives that check at all; 1 means a benign \
explanation is fully confirmed and nothing about the evidence is left \
unexplained. Values in between reflect a benign explanation that is \
plausible but not fully confirmed, or only partially consistent with \
the independent evidence. Do not round to a convenient category -- \
report your actual estimate, including values close to 0.5 if that's \
genuinely where the evidence leaves you.

Do not try to name what kind of activity this is or how severe it is; \
that is out of scope here. Your job is only to estimate how well the \
best available benign explanation survives contact with the \
independent evidence -- including the evidence that was flagged \
against it.

Cite the claim_ids (from either side, including every claim_id listed \
under "claims you must address" above) that most influenced your \
estimate, and give your own confidence (0 to 1) in the estimate itself \
(how sure you are of the benign_plausibility number, separate from what \
that number is). In your rationale, explain in a sentence or two which \
benign explanation you checked, what its prediction was, whether the \
independent evidence confirmed or contradicted it, and how you \
addressed each contradicting claim you were required to cite.
"""


def build_prompt(
    record: EscalationRecord,
    chain_claims: Iterable[Claim],
    chain_hypotheses: Iterable[Hypothesis],
    a4_claims: Iterable[Claim],
    a4_hypotheses: Iterable[Hypothesis],
    verification_by_agent: Dict[str, VerificationCounts],
    required_claim_ids: Iterable[str],
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
        + "\n\nClaims you must address (cite every one of these in cited_claim_ids -- see MANDATORY above):\n"
        + render_required_claim_ids(required_claim_ids)
        + "\n"
    )
