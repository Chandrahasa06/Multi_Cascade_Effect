"""A5 -- benign-plausibility scoring. Prompt version 4.

Changes from v3 -- the question A5 is asked is reframed entirely. Not
"is this malicious" (never asked directly, but v1-v3 still implicitly
asked "does a plausible benign story exist"). Now: "given the benign
traffic actually observed on this network, how many benign flows
resemble this one, and does any surviving hypothesis have empirical
support?" Concretely:

- every hypothesis shown now carries its empirical support: how many
  real Monday benign flows match its predicted_feature_profile, and how
  many of those are within 2x of this flow's own observed values
  (agents/grounding.py, computed in code before this prompt is built --
  never something A5 has to take on faith).
- A5 must name credited_hypothesis_id -- the single hypothesis its
  estimate actually rests on (or none). Code (agents/grounding.py::
  apply_empirical_plausibility_cap) then looks up that hypothesis's
  REAL, already-computed empirical support and clamps benign_plausibility
  to at most 0.3 if it has zero matching benign flows -- regardless of
  what A5 wrote. This is stated here so the cap isn't a surprise, but it
  is enforced whether or not A5 accounts for it.

Input: full feature set, the chain's full output, A4's independent
output, the mechanical agreement between them, the empirical support for
every hypothesis on both sides, and the full set of claim_ids A5 must
dispose of.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

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

PROMPT_VERSION = "a5_verdict_v4"

_INSTRUCTIONS = """\
You are the final reviewer for one network flow. You are not being \
asked to judge this traffic's intent. Your question is: given the \
benign traffic actually observed on this network, how many benign \
flows resemble this one, and does any surviving hypothesis have \
empirical support?

Below you'll see the combined output of a three-stage review chain \
(evidence, behaviour, candidate explanations) and a separate, fully \
independent review of the same flow performed without seeing the \
chain's output. You'll also see a mechanical comparison of which claims \
from the two sides corroborate, contradict, or simply don't overlap.

Every candidate explanation below now carries its empirical support: \
how many real benign flows observed on this network actually match its \
predicted feature profile, and how many of those are close (within 2x) \
to this flow's own observed values. This was computed by querying real \
benign traffic, not estimated by either reviewer. A hypothesis marked \
EMPIRICALLY UNSUPPORTED matched zero real benign flows -- treat that as \
a fact about this network, not a technicality. A hypothesis with many \
matching flows but none close to what this flow actually shows is \
weaker than its raw match count suggests: it means flows like that \
exist, but not flows that look like THIS one.

Before crediting any benign explanation, check both its empirical \
support above and whether its stated prediction holds against the \
INDEPENDENT review's claims (the side that did NOT generate that \
explanation). A benign explanation with strong empirical support AND a \
confirmed prediction is real corroboration. A benign explanation that \
merely sounds plausible, with weak or zero empirical support, is not --  \
do not credit it just because no one has disproven it in words.

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

Report:
- credited_hypothesis_id: the single hypothesis_id (from either side) \
your benign_plausibility estimate actually rests on, or null if no \
hypothesis survives well enough to credit at all. NOTE: if the \
hypothesis you name here has zero empirically-matching benign flows, \
your benign_plausibility will be capped at 0.3 regardless of what you \
report -- this is enforced in code after you respond, not something you \
need to self-apply, but your own estimate should already reflect it: \
do not report a high plausibility for a hypothesis the data does not \
support.
- benign_plausibility, from 0 to 1 -- your estimate of how plausible the \
credited hypothesis is, once you have weighed its empirical support and \
checked its prediction against the independent evidence. 0 means \
nothing survives; 1 means a benign explanation is fully confirmed by \
both real benign traffic and independent evidence, with nothing left \
unexplained. Do not round to a convenient category -- report your \
actual estimate, including values close to 0.5 if that's genuinely \
where the evidence leaves you.

Do not try to name what kind of activity this is or how severe it is; \
that is out of scope here.

Cite the claim_ids (from either side, including every claim_id listed \
under "claims you must address" above) that most influenced your \
estimate, and give your own confidence (0 to 1) in the estimate itself. \
In your rationale, cite the empirical support numbers (match count, \
close-to-observed count) for the hypothesis you credited, explain \
whether its prediction was confirmed or contradicted by the independent \
evidence, and how you addressed each contradicting claim you were \
required to cite.
"""


def build_prompt(
    record: EscalationRecord,
    chain_claims: Iterable[Claim],
    chain_hypotheses: Iterable[Hypothesis],
    a4_claims: Iterable[Claim],
    a4_hypotheses: Iterable[Hypothesis],
    verification_by_agent: Dict[str, VerificationCounts],
    required_claim_ids: Iterable[str],
    empirical_support_lines: Optional[Dict[str, str]] = None,
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
        + "\n\nChain review -- candidate explanations (with empirical support):\n"
        + render_hypotheses(chain_hypotheses, empirical_support_lines)
        + "\n\nIndependent review -- claims:\n"
        + render_claims(a4_claims)
        + "\n\nIndependent review -- candidate explanations (with empirical support):\n"
        + render_hypotheses(a4_hypotheses, empirical_support_lines)
        + "\n\nMechanical agreement between chain and independent review "
        "(per chain stage: corroborated / contradicted / not addressed):\n"
        + render_verification_summary(verification_by_agent)
        + "\n\nClaims you must address (cite every one of these in cited_claim_ids -- see MANDATORY above):\n"
        + render_required_claim_ids(required_claim_ids)
        + "\n"
    )
