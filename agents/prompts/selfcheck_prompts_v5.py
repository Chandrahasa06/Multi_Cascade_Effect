"""Prompts for the self-check run, v5 (held-out A3 and A4): v4 with one change. A4 is given the
row with the same three fields withheld as A3 (agents/holdout.py), and its hypotheses must
predict them, under the same rule as A3's. A4 still works alone: it sees no other reviewer's
output. A1, A2, A3 and A5a are v4 verbatim under v4's own version tags, so their cached outputs
are reused unchanged; A5b's text is v3's, versioned anew because its input changes.
"""
from __future__ import annotations

from typing import Sequence

from agents.prompts.selfcheck_prompts_v4 import (  # noqa: F401 -- unchanged in v5
    ABSENCE_RULE, CITE_RULE, COMPARISON_RULE, CONDITIONS_TEXT, CONDITIONS_WITH_MEANING, PROTOCOL_NUMBERS,
    VERSIONS as V4_VERSIONS, a1, a2, a3, a3_body, a5a, a5b, absences_text, checklist_text, claims_text, flow_block,
    holdout_rule, hypotheses_text,
)

VERSIONS = {**V4_VERSIONS, "a4": "a4_selfcheck_v5_holdout", "a5b": "a5b_selfcheck_v5_holdout"}


def a4(flow: str, held: Sequence[str] = ()) -> str:
    return (
        "You are reviewer A4, working alone: you see only this flow, never any other reviewer's work. Do "
        "three things, from the row only: report structural facts as claims, report what is absent, and "
        "propose explanations committing to predictions about fields you have not seen.\n\n"
        + CONDITIONS_TEXT + "\n\n" + CITE_RULE + " Prefix: a4_c. Give 4 to 8 claims.\n\n" + ABSENCE_RULE
        + "\n\n" + holdout_rule(held) + " Prefix: a4_h.\n\n" + a4_body(flow)
    )


def a4_body(flow: str) -> str:
    """The record-specific part of A4's prompt: the only part that may carry the flow's values."""
    return "The flow:\n" + flow + "\n"
