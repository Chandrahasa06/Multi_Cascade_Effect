"""Devil's-advocate agent (reframe arm B). Prompt version 1.

Its only job is to argue that the flow is malicious: which attack would produce
these values, what that attack would be trying to achieve, and what in the
evidence supports the reading. It does not weigh the benign side; A5 does that.

Input: full feature set, the multi-signal benign evidence block, the empirical
grounding block, the feature names its predicted profiles may use, and A1's claims.
Output: HypothesisResponse with attack hypotheses only (benign=False). Each
hypothesis carries a machine-checkable predicted_feature_profile, checked against
real benign traffic through the same grounding query the other agents use.

New module; nothing earlier references it, so there is no older version to keep.
"""
from __future__ import annotations

from typing import Iterable

from controlplane.record import EscalationRecord

from agents.prompts.render import (
    FRAMING_PREAMBLE,
    SELF_REPORT_INSTRUCTIONS,
    render_claims,
    render_features,
)
from agents.schema import Claim

PROMPT_VERSION = "da_advocate_v1"

_INSTRUCTIONS = """\
You are the ADVOCATE for the attack reading of one network flow. Your job is \
to argue, as strongly as the evidence allows, that this flow is malicious. \
Do not argue the benign side; a separate reviewer does that.

For each of 2 to 4 candidate attack behaviours:
- description: which attack behaviour would produce these values.
- prediction: one concrete, checkable statement about the flow's features if \
this attack is what happened (10 characters or more).
- predicted_feature_profile: the feature ranges that statement implies, using \
only the feature names listed below. Each entry needs expected_min and/or \
expected_max.
- prior_plausibility, from 0 to 1: how plausible this attack is on its own.
- benign: always false. You are generating attack hypotheses only.
- supporting_claim_ids: claim_ids from the first reviewer that support it.
- contradicting_claim_ids: leave empty. You are not weighing the benign side.

Also state, in claims, what the attacker is trying to achieve and what in the \
observed values supports that reading. Cite the feature values you rely on.

Attack reasoning means asking what the flow would look like if the \
attacker's aim were fulfilled, then checking whether the observed values \
match that. Do not stop at "this looks like ordinary traffic"; that is \
exactly what a competent attacker produces, so it is not evidence against \
the attack reading.

Predicted feature names must come from this list (the only names with a \
benign reference to check against):
{available_features}
"""


def build_prompt(
    record: EscalationRecord,
    a1_claims: Iterable[Claim],
    empirical_grounding_block: str,
    available_features: str,
    evidence_block: str,
) -> str:
    instructions = _INSTRUCTIONS.replace("{available_features}", available_features)
    instructions = instructions + "\n" + SELF_REPORT_INSTRUCTIONS
    return (
        FRAMING_PREAMBLE
        + "\n"
        + instructions
        + "\n\nEmpirical grounding against real benign traffic on this network:\n"
        + empirical_grounding_block
        + "\n\nMulti-signal benign evidence for this flow:\n"
        + evidence_block
        + "\n\nFull feature set for this flow:\n"
        + render_features(record)
        + "\n\nFirst reviewer's claims:\n"
        + render_claims(a1_claims)
        + "\n"
    )
