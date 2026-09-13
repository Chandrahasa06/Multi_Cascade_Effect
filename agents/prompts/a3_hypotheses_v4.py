"""A3 -- hypothesis generation. Prompt version 4.

Changes from v3:
- now sees the trigger reasons and empirical grounding block directly
  (previously A3 only saw the full feature list and A1/A2's claims) --
  needed because building a predicted_feature_profile against real
  benign data requires knowing what that data says.
- predicted_feature_profile (a list of (feature, expected range) pairs,
  closed vocabulary) replaces the old bare-name referenced_features.
  agents/grounding.py queries Monday's real benign traffic against this
  profile before A5 ever sees the hypothesis, attaching how many real
  benign flows actually match it. A hypothesis whose profile matches
  zero real benign flows is empirically unsupported -- a fact A5 must
  account for, and one that mechanically caps how much plausibility that
  hypothesis can be credited with (agents/grounding.py::
  apply_empirical_plausibility_cap), regardless of how well the
  hypothesis reads in prose.

Input: full feature set, trigger reasons, empirical grounding against
real benign traffic, and A1's and A2's full claim lists.
Output: claims (optional, new observations only) + 3-5 hypotheses, each
with a falsifiable, empirically-checkable predicted_feature_profile, at
least 2 of which must be benign explanations.
"""
from __future__ import annotations

from typing import Iterable

from controlplane.record import EscalationRecord

from agents.prompts.render import (
    CALIBRATION_NOTE,
    FRAMING_PREAMBLE,
    render_claims,
    render_features,
    render_trigger_reasons,
)
from agents.schema import Claim

PROMPT_VERSION = "a3_hypotheses_v4"

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

Below the flagged measurements you'll see what real benign traffic on \
this network actually looks like (median, p90, p99, p99.5, and maximum \
observed across hundreds of thousands of real benign flows), and the \
nearest real benign flows found in normalised feature space. Weigh this \
BEFORE proposing a benign explanation, not after: a value that exceeds \
every benign flow ever observed on this network is a fact about this \
network, not a detail to explain around. An explanation that requires \
this flow to be an ordinary example of something, when nothing ordinary \
on this network has ever looked like this, is not a strong explanation \
even if it sounds plausible in isolation.

IMPORTANT -- what you can and cannot predict: this system only ever \
observes flow-level statistics (packet counts, byte counts, timing, TCP \
flags, port numbers -- the kind of numbers in the feature list below). \
It never sees packet payload content, HTTP headers, user-agent strings, \
URIs, cookies, or anything else at the application layer. A prediction \
about any of those cannot be checked by anything this system has, and \
will be rejected. Every prediction must be about a flow-level feature \
that is actually in the feature list or the flagged measurements.

For each candidate explanation, give:
- a hypothesis_id of the form "a3_h1", "a3_h2", ...
- a short description of the explanation
- whether it is an ordinary/benign explanation (true/false)
- a prior plausibility (0 to 1) for how likely this kind of explanation \
seems in general, before weighing the specific evidence below
- a PREDICTION: a concrete, checkable statement about what a FLOW-LEVEL \
FEATURE would look like if this explanation were actually true.
- predicted_feature_profile: the mechanically checkable form of your \
prediction -- a list of (feature, expected_min and/or expected_max) \
pairs, which will be queried against real benign traffic on this \
network before anyone reads your hypothesis. Use ONLY these feature \
names (anything else is rejected -- there is no benign reference data \
to check it against): {available_features}. For example, for a \
monitoring/health-check hypothesis predicting "flows per source should \
be moderate, not extreme", a real profile entry is {"feature": \
"trigger_flows_per_src", "expected_max": 500}. Set the range to what \
you actually believe, not artificially wide to guarantee matches -- \
this is checked against your own flow's observed values too, and a \
range so wide it's meaningless will show as such.
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
    record: EscalationRecord,
    a1_claims: Iterable[Claim],
    a2_claims: Iterable[Claim],
    empirical_grounding_block: str,
    available_features: str,
) -> str:
    instructions = _INSTRUCTIONS.replace("{available_features}", available_features)
    return (
        FRAMING_PREAMBLE
        + "\n"
        + CALIBRATION_NOTE
        + "\n"
        + instructions
        + "\n\nFeatures already flagged as crossing a fitted threshold:\n"
        + render_trigger_reasons(record)
        + "\n\nEmpirical grounding against real benign traffic on this network:\n"
        + empirical_grounding_block
        + "\n\nFull feature set for this flow:\n"
        + render_features(record)
        + "\n\nFirst reviewer's claims:\n"
        + render_claims(a1_claims)
        + "\n\nSecond reviewer's claims:\n"
        + render_claims(a2_claims)
        + "\n"
    )
