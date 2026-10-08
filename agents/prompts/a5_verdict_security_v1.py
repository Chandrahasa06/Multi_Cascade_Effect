"""A5 -- security judgment on flows statistics already call ordinary. Prompt version 1.

Used only by eval/run_grounded_only.py, on records whose benign-neighbourhood count is at
or above the ungrounded cutoff (the flows the count calls benign). Differs from
a5_verdict_v7 in three ways:
  - the question is a security judgment, not "is there a benign explanation";
  - A5 is shown NO count of any kind: no neighbourhood count, no "within 2x" closeness, no
    hypothesis profile match counts. The builder has no parameter through which one could
    reach it. Evidence is limited to features, destination port, time of day, the source's
    own earlier behaviour, and per-feature percentiles and direction;
  - no cap is applied afterwards (enforced in the runner), so the score is A5's own.

Same output (A5Response), same scale and direction: low benign_plausibility means the flow
is not plausibly benign. No threshold is stated. Older A5 prompts stay importable.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from controlplane.record import EscalationRecord

from agents.prompts.render import (
    SELF_REPORT_INSTRUCTIONS_A5,
    render_claims,
    render_features,
    render_hypotheses,
    render_required_claim_ids,
)
from agents.schema import Claim, Hypothesis
from agents.verification import VerificationCounts, render_verification_summary

PROMPT_VERSION = "a5_verdict_security_v1"

_INSTRUCTIONS = """\
You are the final security reviewer for one network flow.

Statistical checks have already found this flow unremarkable: it resembles \
ordinary traffic on this network. Given what you know about how networks are \
attacked, is there reason to think this flow is hostile anyway?

Consider:
- the destination port, and what normally runs there;
- the time of day;
- whether this source's behaviour fits its own recent history;
- whether this combination of features is one an attacker would plausibly \
produce while trying to look ordinary.

Resembling ordinary traffic is NOT by itself exculpatory. A capable attacker \
aims to produce flows that look ordinary. Do not conclude a flow is benign \
only because it looks typical; say what, specifically, makes it benign or \
hostile.

Below you will see the flow's features, the time it started, the source's \
behaviour in the preceding half hour, where each key feature sits in this \
network's ordinary traffic, and the work of two earlier reviews: a \
three-stage chain and a separate independent reviewer. Their candidate \
explanations are shown without any statistical support figures.

MANDATORY: every claim_id listed under "claims you must address" was \
flagged by some hypothesis's own reviewer as contradicting that hypothesis. \
Cite every one of them in cited_claim_ids. This is checked mechanically; a \
response missing one is rejected. Citing a claim does not mean agreeing with \
it: explain in your rationale how you weighed it.

Report:
- credited_hypothesis_id: the single hypothesis_id (from either review) your \
estimate rests on, or null if none.
- benign_plausibility, from 0 to 1: how plausible it is that this flow is \
benign, after the security judgment above. 0 means it is not plausibly \
benign; 1 means it is clearly benign. Report your actual estimate, \
including values near the middle if that is where the evidence leaves you.
- cited_claim_ids: the claim_ids that most influenced you, including every \
mandatory one.
- rationale: what you weighed. Name the specific evidence: the port, the \
time, the source history, particular feature values.

"""


def build_prompt(
    record: EscalationRecord,
    chain_claims: Iterable[Claim],
    chain_hypotheses: Iterable[Hypothesis],
    a4_claims: Iterable[Claim],
    a4_hypotheses: Iterable[Hypothesis],
    verification_by_agent: Dict[str, VerificationCounts],
    required_claim_ids: Iterable[str],
    context_block: str,
) -> str:
    """`context_block` is built by eval/run_grounded_only.render_context from features,
    time, source history, percentiles and direction only. No count parameter exists."""
    return (
        _INSTRUCTIONS
        + SELF_REPORT_INSTRUCTIONS_A5
        + "\n\nFull feature set for this flow:\n"
        + render_features(record)
        + "\n\nContext for this flow:\n"
        + context_block
        + "\n\nChain review -- combined claims (evidence + behaviour + hypothesis stages):\n"
        + render_claims(chain_claims)
        + "\n\nChain review -- candidate explanations:\n"
        + render_hypotheses(chain_hypotheses, None)
        + "\n\nIndependent review -- claims:\n"
        + render_claims(a4_claims)
        + "\n\nIndependent review -- candidate explanations:\n"
        + render_hypotheses(a4_hypotheses, None)
        + "\n\nMechanical agreement between chain and independent review "
        "(per chain stage: corroborated / contradicted / not addressed):\n"
        + render_verification_summary(verification_by_agent)
        + "\n\nClaims you must address (cite every one in cited_claim_ids):\n"
        + render_required_claim_ids(required_claim_ids)
        + "\n"
    )
