"""A2 -- behavioural characterisation. Prompt version 2.

Changes from v1: adds the self-reported trust triad (confidence,
evidence_support, verification -- agents/prompts/render.py::
SELF_REPORT_INSTRUCTIONS, agents/schema.py::ClaimsResponse) alongside
the existing claims. Nothing else about A2's task changes.

Input: full feature set + A1's full claim list.
Output: claims describing what the two parties appear to be doing --
may build on or explicitly dispute A1 -- plus this call's self-reported
trust triad.
"""
from __future__ import annotations

from typing import Iterable

from controlplane.record import EscalationRecord

from agents.prompts.render import FRAMING_PREAMBLE, SELF_REPORT_INSTRUCTIONS, render_claims, render_features
from agents.schema import Claim

PROMPT_VERSION = "a2_behaviour_v2"

_INSTRUCTIONS = f"""\
You are the second reviewer in a sequence looking at one network flow's \
measurements. A prior reviewer already described what is quantitatively \
unusual about this flow (shown below). Your job is to describe what the \
two communicating parties appear to be doing at a behavioural level: \
the connection pattern, which side sends more data, whether exchanges \
complete normally or get cut short, whether this looks like a single \
isolated exchange or part of a repeated pattern, and how persistent or \
transient the communication looks.

You may build on the prior reviewer's observations, or dispute them. If \
you think a specific prior claim is wrong or better explained \
differently, say so explicitly as one of your own claims -- reference \
the same feature(s) and state your own relation. This kind of \
disagreement is useful signal; don't soften it to seem agreeable.

Do not speculate about who is doing this, why, or what software \
produced it. Describe behaviour, not intent.

Emit your findings as a list of independent claims, same schema as \
before: claim_id of the form "a2_c1", "a2_c2", ... (sequential, \
unique), at least one referenced feature per claim, relation from \
{{greater_than_typical, less_than_typical, equals, absent, present}}, and \
your own confidence per claim.

Produce between 3 and 8 claims.

{SELF_REPORT_INSTRUCTIONS}
"""


def build_prompt(record: EscalationRecord, a1_claims: Iterable[Claim]) -> str:
    return (
        FRAMING_PREAMBLE
        + "\n"
        + _INSTRUCTIONS
        + "\n\nFull feature set for this flow:\n"
        + render_features(record)
        + "\n\nPrior reviewer's claims:\n"
        + render_claims(a1_claims)
        + "\n"
    )
