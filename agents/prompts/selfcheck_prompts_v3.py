"""Prompts for the self-check run, v3: v2 with A1 shown the complete comparison rule, generated
from the same constants the validator applies (agents/selfcheck_core.py), so the prompt and the
check cannot drift apart. Every other prompt's text is unchanged from v2; all six versions are
bumped so a run is on one version throughout. v1 and v2 stay importable.
"""
from __future__ import annotations

from agents.prompts.selfcheck_prompts_v2 import (  # noqa: F401 -- unchanged in v3
    ABSENCE_RULE, CITE_RULE, CONDITIONS_TEXT, CONDITIONS_WITH_MEANING, HYPOTHESIS_RULE, PROTOCOL_NUMBERS,
    a2, a3, a4, a5a, a5b, absences_text, checklist_text, claims_text, flow_block, hypotheses_text,
)
from agents.selfcheck_core import COMPARATIVES, JUDGE_WORDS, MAGNITUDE_WORDS

VERSIONS = {"a1": "a1_selfcheck_v3", "a2": "a2_selfcheck_v3", "a3": "a3_selfcheck_v3",
            "a4": "a4_selfcheck_v3", "a5a": "a5a_selfcheck_v3", "a5b": "a5b_selfcheck_v3"}

COMPARISON_RULE = (
    "Do not compare this flow with other traffic: there is no baseline here. Rejected automatically:\n"
    "- these words, anywhere: " + ", ".join(JUDGE_WORDS + MAGNITUDE_WORDS) + ";\n"
    "- " + ", ".join(COMPARATIVES) + ", unless the same sentence names two of the fields it cites (a "
    "comparison between two of this flow's own fields, for example \"Total Length of Bwd Packets is "
    "larger than Total Length of Fwd Packets\");\n"
    "- any comparison with something outside this flow, whatever the word: \"than most\", \"than usual\", "
    "\"than typical\", \"compared to other flows\", \"than other connections\", \"for a connection\" and "
    "the like."
)


def a1(flow: str) -> str:
    return (
        "You are reviewer A1. Report structural facts about this one flow, from its row only: its "
        "directionality, its duration against its packet count, how its bytes split between header and "
        "payload, and how regular its timing is.\n\n" + COMPARISON_RULE + "\n\n"
        + CONDITIONS_TEXT + "\n\n" + CITE_RULE + " Prefix: a1_c. Give 4 to 8 claims.\n\nThe flow:\n" + flow + "\n"
    )
