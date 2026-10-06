"""A5 -- benign-plausibility scoring, FIXED-SCALE diagnostic variant.

Identical to a5_verdict_v5 in every respect (framing, calibration note,
what to weigh, the empirical-support/contradiction-disposal/credited-
hypothesis machinery, the self-report triad) EXCEPT the single paragraph
telling A5 how to report benign_plausibility: instead of "any float in
[0,1], don't round to a convenient category", A5 must choose exactly one
of ten evenly-spaced, verbally-anchored points.

Built for results/plausibility_diagnostic.md's Step 3: the recorded v2/
batch-2 runs show benign_plausibility collapsing onto ~2 values out of 7
distinct ones observed across 36 records (63.9% of records land on
exactly 0.25 or 0.85). This variant asks whether that collapse is a
format artifact (the model reaches for round numbers when given a free
scale) or a genuine judgment (the model would still refuse the middle of
an explicitly-offered scale). Diagnostic only -- never imported by
agents/a5_verdict.py or agents/pipeline.py, and its output
(results/agent_run_a5_scale.jsonl) does not supersede the recorded v2/
batch-2 scores.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from controlplane.record import EscalationRecord

from agents.prompts.render import (
    CALIBRATION_NOTE,
    FRAMING_PREAMBLE,
    SELF_REPORT_INSTRUCTIONS_A5,
    render_claims,
    render_features,
    render_hypotheses,
    render_required_claim_ids,
)
from agents.schema import A5_SCALE_VALUES, Claim, Hypothesis
from agents.verification import VerificationCounts, render_verification_summary

PROMPT_VERSION = "a5_verdict_scale_v1"

_SCALE_ANCHORS = """\
    0.05 -- no benign explanation the data supports at all
    0.15 -- very little benign support; strong reasons to doubt it's benign
    0.25 -- weak benign support; more reasons to doubt than to credit it
    0.35 -- benign support and doubt are both present, tilted toward doubt
    0.45 -- genuinely unclear, tilted slightly toward NOT benign
    0.55 -- genuinely unclear, tilted slightly toward benign
    0.65 -- benign support and doubt are both present, tilted toward support
    0.75 -- fairly good benign support; some doubt remains
    0.85 -- strong benign support; only minor doubt remains
    0.95 -- a well-supported benign explanation; essentially no doubt left
"""

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
your benign_plausibility must be 0.25 or lower regardless of what you \
might otherwise be inclined to report -- this is enforced in code after \
you respond (snapped down to the nearest scale point at or below 0.25 \
if you report higher), but your own choice should already reflect it: \
do not pick a high point on the scale for a hypothesis the data does \
not support.
- benign_plausibility: choose EXACTLY ONE of the following ten points \
-- do not report any other number, and do not default to the extremes \
out of habit if the evidence genuinely lands you in the middle:
""" + _SCALE_ANCHORS + """
Weigh the evidence exactly as you would for a free-form estimate, then \
report whichever one of the ten anchors above is closest to your \
actual judgment. A number strictly between two anchors is not a valid \
answer -- pick the nearer one.

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
    instructions = _INSTRUCTIONS + SELF_REPORT_INSTRUCTIONS_A5
    return (
        FRAMING_PREAMBLE
        + "\n"
        + CALIBRATION_NOTE
        + "\n"
        + instructions
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


assert len(A5_SCALE_VALUES) == 10  # keep the rendered anchor list and the schema's allowed set in lockstep
