"""A4 -- blind replication. Prompt version 4.

Changes from v3:
- adds empirical grounding against Monday's real benign traffic (same as
  a1_evidence_v4).
- hypotheses now require predicted_feature_profile -- a list of (feature,
  expected range) pairs using ONLY the closed vocabulary this project has
  real benign data for -- instead of the old bare-name referenced_features.
  This is what agents/grounding.py queries against Monday's benign flows
  BEFORE A5 ever sees the hypothesis, attaching how many real benign
  flows actually match. Replaces "does this sound plausible" with "is
  this actually supported by real traffic".

Input: full feature set, trigger reasons, and empirical grounding against
real benign traffic -- no chain output, and this prompt must never reveal
that other reviewers or a chain exist.
Output: the same claims+hypotheses shape as A3, produced in one pass, so
verification.py can mechanically compare it to the chain's combined
output.
"""
from __future__ import annotations

from controlplane.record import EscalationRecord

from agents.prompts.render import (
    CALIBRATION_NOTE,
    FRAMING_PREAMBLE,
    TRIGGER_FEATURE_PREFIX,
    render_features,
    render_trigger_reasons,
)

PROMPT_VERSION = "a4_replication_v4"

_INSTRUCTIONS = f"""\
You are reviewing one network flow's measured statistics. Your job has \
three parts, all from this one flow's evidence:

1. State, precisely and quantitatively, what is unusual about these \
measurements and by how much.
2. Describe what the two communicating parties appear to be doing at a \
behavioural level: connection pattern, which side sends more data, \
whether exchanges complete normally, and how persistent or transient \
the communication looks.
3. Generate between 3 and 5 candidate explanations for why this flow \
looks the way it does. At least two must be explanations where nothing \
is wrong at all -- for example: a misconfigured client retrying a \
request, a monitoring or health-check system polling on a schedule, a \
scheduled backup job, a load-testing tool exercising a service, or a \
client- or server-side bug causing repeated failed connection \
attempts. Do not limit yourself to that list. You may include \
non-ordinary explanations too, if the evidence points that way, but \
seriously consider the benign alternatives first, not as an afterthought.

Do not speculate about who is doing this or what specific software \
produced it beyond what's stated above. Describe behaviour and \
evidence, not intent.

Below the flagged measurements you'll also see what real benign traffic \
on this network actually looks like (median, p90, p99, p99.5, and \
maximum observed across hundreds of thousands of real benign flows), and \
the nearest real benign flows found in normalised feature space. Before \
proposing a benign explanation, check whether anything resembling it \
actually exists in that reference data -- a value that exceeds every \
benign flow ever observed on this network is a fact you should weigh \
seriously, not explain away by assumption.

The flagged measurements above the full feature list use the \
"{TRIGGER_FEATURE_PREFIX}" prefix (e.g. "{TRIGGER_FEATURE_PREFIX}flow_duration") \
and are on a DIFFERENT scale than the full feature list below -- when \
you cite one of those flagged measurements, use its exact prefixed name, \
not the bare name from the full feature list, even if they look similar. \
Each flagged measurement lists observed_value, threshold, and \
ratio_to_threshold on separate lines -- when you assert a value, use \
observed_value. ratio_to_threshold is a multiplier, not a value in its \
own right; never assert it as if it were the measurement.

IMPORTANT -- what you can and cannot predict: this system only ever \
observes flow-level statistics (packet counts, byte counts, timing, TCP \
flags, port numbers -- the kind of numbers in the feature list below). \
It never sees packet payload content, HTTP headers, user-agent strings, \
URIs, cookies, or anything else at the application layer. A hypothesis \
prediction about any of those cannot be checked by anything this system \
has, and will be rejected.

Emit your findings as claims plus hypotheses:

Claims: claim_id of the form "a4_c1", "a4_c2", ... (sequential, \
unique), each referencing at least one specific feature by its exact \
name (including the "{TRIGGER_FEATURE_PREFIX}" prefix if the feature \
came from the flagged-measurements list), with relation from \
{{greater_than_typical, less_than_typical, equals, absent, present}}, \
and your own confidence (0 to 1). Produce between 5 and 12 claims \
covering both the quantitative observations from part 1 and the \
behavioural observations from part 2.

Hypotheses: hypothesis_id of the form "a4_h1", "a4_h2", ..., each with \
a description, whether it is benign (true/false), a prior plausibility \
(0 to 1), a PREDICTION about a flow-level feature, a \
predicted_feature_profile, and which of your own claim_ids support or \
contradict it. predicted_feature_profile is a list of (feature, \
expected_min and/or expected_max) pairs -- the mechanically checkable \
form of your prediction, which will be queried against real benign \
traffic on this network before anyone reads your hypothesis. Use ONLY \
these feature names in predicted_feature_profile (anything else is \
rejected -- there is no benign reference data to check it against): \
{{available_features}}. For example, for a monitoring/health-check \
hypothesis predicting "flows per source should be moderate, not \
extreme", a real profile entry is {{"feature": "{TRIGGER_FEATURE_PREFIX}flows_per_src", \
"expected_max": 500}}. The prediction text and the profile should agree \
-- the profile is what actually gets checked. Produce between 3 and 5 \
hypotheses, at least 2 of them benign.
"""


def build_prompt(record: EscalationRecord, empirical_grounding_block: str, available_features: str) -> str:
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
        + "\n"
    )
