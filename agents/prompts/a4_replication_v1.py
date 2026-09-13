"""A4 -- blind replication. Prompt version 1.

Input: full feature set + trigger reasons ONLY -- no chain output, and
this prompt must never reveal that other reviewers or a chain exist.
Output: the same claims+hypotheses shape as A3, produced in one pass,
so verification.py can mechanically compare it to the chain's combined
output.
"""
from __future__ import annotations

from controlplane.record import EscalationRecord

from agents.prompts.render import FRAMING_PREAMBLE, render_features, render_trigger_reasons

PROMPT_VERSION = "a4_replication_v1"

_INSTRUCTIONS = """\
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

Emit your findings as claims plus hypotheses:

Claims: claim_id of the form "a4_c1", "a4_c2", ... (sequential, \
unique), each referencing at least one specific feature by its exact \
name, with relation from {greater_than_typical, less_than_typical, \
equals, absent, present}, and your own confidence (0 to 1). Produce \
between 5 and 12 claims covering both the quantitative observations \
from part 1 and the behavioural observations from part 2.

Hypotheses: hypothesis_id of the form "a4_h1", "a4_h2", ..., each with \
a description, whether it is benign (true/false), a prior plausibility \
(0 to 1), and which of your own claim_ids support or contradict it. \
Produce between 3 and 5 hypotheses, at least 2 of them benign.
"""


def build_prompt(record: EscalationRecord) -> str:
    return (
        FRAMING_PREAMBLE
        + "\n"
        + _INSTRUCTIONS
        + "\n\nFeatures already flagged as crossing a fitted threshold:\n"
        + render_trigger_reasons(record)
        + "\n\nFull feature set for this flow:\n"
        + render_features(record)
        + "\n"
    )
