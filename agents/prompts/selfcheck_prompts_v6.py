"""Prompts for the self-check run, v6 (held-out A3 and A4, no benign quota): v5 with one sentence
changed in A3's and A4's hypothesis rule. "3 to 5, at least 2 benign" becomes "3 to 5", with
benign hypotheses proposed only where a falsifiable prediction about the withheld fields can be
stated. The new texts are built by replacing exactly that sentence in v4/v5's own prompts
(asserted), so nothing else can differ. A1, A2 and A5a are v5 verbatim under v5's tags; A5b's
text is unchanged, versioned anew because its input changes.
"""
from __future__ import annotations

from typing import Sequence

from agents.prompts import selfcheck_prompts_v5 as _v5
from agents.prompts.selfcheck_prompts_v5 import (  # noqa: F401 -- unchanged in v6
    ABSENCE_RULE, CITE_RULE, COMPARISON_RULE, CONDITIONS_TEXT, CONDITIONS_WITH_MEANING, PROTOCOL_NUMBERS,
    a1, a2, a3_body, a4_body, a5a, a5b, absences_text, checklist_text, claims_text, flow_block, hypotheses_text,
)

VERSIONS = {**_v5.VERSIONS, "a3": "a3_selfcheck_v6_noquota", "a4": "a4_selfcheck_v6_noquota",
            "a5b": "a5b_selfcheck_v6_noquota"}

#: the validator's minimum number of benign hypotheses per agent (v1-v5: 2)
MIN_BENIGN = 0

QUOTA_SENTENCE = "Hypotheses: 3 to 5, at least 2 benign. "
NO_QUOTA_SENTENCE = (
    "Hypotheses: 3 to 5. There is no required number of benign hypotheses: propose a benign hypothesis "
    "ONLY where you can state a falsifiable prediction about the withheld fields. A hypothesis that exists "
    "to fill a quota is not evidence. "
)


def holdout_rule(held: Sequence[str]) -> str:
    old = _v5.holdout_rule(held)
    assert old.count(QUOTA_SENTENCE) == 1
    return old.replace(QUOTA_SENTENCE, NO_QUOTA_SENTENCE)


def _swap(text: str, held: Sequence[str]) -> str:
    old = _v5.holdout_rule(held)
    assert text.count(old) == 1, "the v5 hypothesis rule must appear exactly once"
    return text.replace(old, holdout_rule(held))


def a3(flow: str, a1_text: str, a2_text: str, held: Sequence[str] = ()) -> str:
    return _swap(_v5.a3(flow, a1_text, a2_text, held), held)


def a4(flow: str, held: Sequence[str] = ()) -> str:
    return _swap(_v5.a4(flow, held), held)
