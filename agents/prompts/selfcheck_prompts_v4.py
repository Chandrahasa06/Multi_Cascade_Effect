"""Prompts for the self-check run, v4 (held-out A3): v3 with one change. A3 does not see three
of the flow's fields (agents/holdout.py) and must predict what they contain; every other prompt
is v3 verbatim, under v3's own version tags, so their cached outputs are reused unchanged.
A5b's text is v3's, but its version is bumped because its input (A3's checked hypotheses)
changes. v1-v3 stay importable.
"""
from __future__ import annotations

from typing import Sequence

from agents.prompts.selfcheck_prompts_v3 import (  # noqa: F401 -- unchanged in v4
    ABSENCE_RULE, CITE_RULE, COMPARISON_RULE, CONDITIONS_TEXT, CONDITIONS_WITH_MEANING, PROTOCOL_NUMBERS,
    VERSIONS as V3_VERSIONS, a1, a2, a4, a5a, a5b, absences_text, checklist_text, claims_text, flow_block,
    hypotheses_text,
)

VERSIONS = {**V3_VERSIONS, "a3": "a3_selfcheck_v4_holdout", "a5b": "a5b_selfcheck_v4_holdout"}


def holdout_rule(held: Sequence[str]) -> str:
    names = ", ".join(f'"{h}"' for h in held)
    return (
        "Hypotheses: 3 to 5, at least 2 benign. Each has an id (prefix given, then a number), a description, "
        "`benign` true/false, `condition_ids` it accounts for (may be empty), and `predictions`.\n"
        f"Three of this flow's fields are withheld from you: {names}. Their values exist; you are not shown "
        "them. Every hypothesis's predictions must say what these withheld fields must contain if the "
        "hypothesis is true: at least one prediction for EACH withheld field, and predictions ONLY about "
        "the withheld fields (a prediction about a field you can see tests nothing). Each prediction names a "
        "withheld field exactly as written above, a comparison, and a number.\n"
        "Code checks every prediction against the real withheld values before anyone reads it: a hypothesis "
        "with any false prediction is struck out and cannot be credited. One whose prediction for a withheld "
        "field cannot fail (a bound every possible value meets) earns no support."
    )


def a3(flow: str, a1_text: str, a2_text: str, held: Sequence[str] = ()) -> str:
    return (
        "You are reviewer A3. Propose explanations of what this flow is, each committing to predictions "
        "about fields you have not seen.\n\n"
        + CONDITIONS_TEXT + "\n\n" + holdout_rule(held) + " Prefix: a3_h. Optionally add claims: " + CITE_RULE
        + " Prefix: a3_c.\n\n" + a3_body(flow, a1_text, a2_text)
    )


def a3_body(flow: str, a1_text: str, a2_text: str) -> str:
    """The record-specific part of A3's prompt: the only part that may carry the flow's values."""
    return "The flow:\n" + flow + "\n\nA1's claims:\n" + a1_text + "\n\nA2's claims and absences:\n" + a2_text + "\n"
