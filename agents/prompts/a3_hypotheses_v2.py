"""A3 -- hypothesis generation. Prompt version 2.

Changes from v1 (see STATUS.md's stop-point-2 findings):
- adds CALIBRATION_NOTE so hypothesis plausibility is weighed against a
  stated scale, not a bare ratio.
- every hypothesis must now state a falsifiable prediction: a concrete,
  checkable claim about the evidence that would hold if the hypothesis
  were true. Without this, a benign hypothesis can survive purely
  because nothing was ever asked of it -- an escape hatch, not
  corroboration. A5 checks these predictions against the independent
  (A4) evidence before trusting a hypothesis.

Input: full feature set plus A1's and A2's full claim lists.
Output: claims (optional, new observations only) + 3-5 hypotheses, each
with a prediction, at least 2 of which must be benign explanations.
"""
from __future__ import annotations

from typing import Iterable

from controlplane.record import EscalationRecord

from agents.prompts.render import CALIBRATION_NOTE, FRAMING_PREAMBLE, render_claims, render_features
from agents.schema import Claim

PROMPT_VERSION = "a3_hypotheses_v2"

_INSTRUCTIONS = """\
You are the third reviewer in a sequence looking at one network flow's \
measurements. Two prior reviewers already described what is \
quantitatively unusual (first reviewer) and what the communicating \
parties appear to be doing behaviourally (second reviewer), both shown \
below.

Your job is to generate between 3 and 5 candidate explanations for why \
this flow looks the way it does. At least two of your candidate \
explanations must be explanations where nothing is wrong at all -- for \
example: a misconfigured client retrying a request, a monitoring or \
health-check system polling on a schedule, a scheduled backup job, a \
load-testing tool exercising a service, or a client- or server-side bug \
causing repeated failed connection attempts. Do not limit yourself to \
that list -- think of whatever ordinary process could plausibly produce \
these measurements. You may also include explanations that would not \
be ordinary use, if the evidence points that way, but you must \
seriously consider the benign alternatives first, not as an \
afterthought to dismiss quickly.

For each candidate explanation, give:
- a hypothesis_id of the form "a3_h1", "a3_h2", ...
- a short description of the explanation
- whether it is an ordinary/benign explanation (true/false)
- a prior plausibility (0 to 1) for how likely this kind of explanation \
seems in general, before weighing the specific evidence below
- a PREDICTION: a concrete, checkable statement about what the evidence \
would look like if this explanation were actually true -- something \
that could be confirmed or contradicted by measurements, not a \
restatement of the explanation itself. For example, if you propose a \
monitoring/health-check system, a real prediction is "inter-arrival \
times between this source's flows should be regular, close to a fixed \
interval" -- not "this looks like monitoring traffic". A hypothesis \
with a vague or unfalsifiable prediction is not doing its job: someone \
else (a later reviewer, or a mechanical check) must be able to look at \
the evidence and tell whether your prediction held or was contradicted. \
Do not propose a hypothesis you cannot state a real prediction for.
- which claim_ids (from either reviewer above, or your own new claims) \
support it
- which claim_ids contradict it

You may also emit new claims of your own (claim_id "a3_c1", "a3_c2", \
...) if you notice something the prior reviewers missed, using the same \
schema as before: at least one referenced feature, relation from \
{greater_than_typical, less_than_typical, equals, absent, present}, and \
your own confidence. New claims are optional -- only add one if you \
have a genuinely new observation, not a restatement of a prior one.
"""


def build_prompt(
    record: EscalationRecord, a1_claims: Iterable[Claim], a2_claims: Iterable[Claim]
) -> str:
    return (
        FRAMING_PREAMBLE
        + "\n"
        + CALIBRATION_NOTE
        + "\n"
        + _INSTRUCTIONS
        + "\n\nFull feature set for this flow:\n"
        + render_features(record)
        + "\n\nFirst reviewer's claims:\n"
        + render_claims(a1_claims)
        + "\n\nSecond reviewer's claims:\n"
        + render_claims(a2_claims)
        + "\n"
    )
