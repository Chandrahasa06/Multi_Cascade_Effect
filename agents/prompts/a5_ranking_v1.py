"""A5 -- RANK ORDERING diagnostic variant (results/plausibility_diagnostic.md
Step 4). Presents a batch of already-reviewed records' full evidence
(same chain claims, chain hypotheses, A4's independent claims and
hypotheses, empirical support, and mechanical corroboration summary A5
normally sees one record at a time) side by side, and asks for a
relative ranking from most to least likely benign -- deliberately no
numeric score at all, to test whether relative judgment across records
separates attacks from benign better than the same model's own absolute
benign_plausibility numbers do (agents/prompts/a5_verdict_v5.py).

Never used by the real pipeline. Only caller: eval/run_a5_rank.py.
"""
from __future__ import annotations

from typing import Dict, List

from agents.prompts.render import (
    CALIBRATION_NOTE,
    FRAMING_PREAMBLE,
    render_claims,
    render_hypotheses,
)
from agents.verification import VerificationCounts, render_verification_summary

PROMPT_VERSION = "a5_ranking_v1"

_INSTRUCTIONS = """\
You are the final reviewer for a BATCH of independently-selected network \
flows, shown together this time instead of one at a time. For each flow \
below you will see the same evidence a single-flow reviewer would see: \
the combined output of a three-stage review chain (evidence, behaviour, \
candidate explanations), a separate independent review of the same flow, \
empirical support for every candidate explanation (computed against real \
benign traffic on this network, not estimated), and a mechanical \
comparison of which claims from the two sides corroborate, contradict, \
or don't overlap.

Your task: rank these flows from MOST to LEAST likely to have an \
innocent (benign) explanation, using exactly the same considerations you \
would use to score any one of them alone -- empirical support for the \
credited hypothesis, whether its prediction is confirmed or contradicted \
by the independent review, and how well any contradicting claims are \
addressed. Compare the flows directly against EACH OTHER: a flow near \
the top of your ranking should have measurably stronger benign evidence \
than a flow near the bottom, not just an equally strong case argued more \
confidently.

Do NOT report a numeric score for any flow. Report only the ORDER. Do \
not group flows into tiers and call it a ranking -- if two flows are \
genuinely indistinguishable to you, say so in your rationale, but you \
must still place them in some definite order in the list (ties are not \
representable in a flat ranking).

Report:
- ranking: a list containing every record_id shown below, EXACTLY once \
each, ordered from the one you consider MOST likely benign (first) to \
the one you consider LEAST likely benign (last). Use the exact \
record_id strings shown in each flow's header.
- rationale: briefly justify the overall ordering -- which flows anchor \
the top and bottom of your ranking and why, and note any flows you found \
close to a neighbour in the ranking.

"""


def build_prompt(record_blocks: List[Dict[str, object]]) -> str:
    """``record_blocks``: one dict per record in this batch, each with
    keys record_id, chain_claims, chain_hypotheses, a4_claims,
    a4_hypotheses, verification_by_agent (Dict[str, VerificationCounts]),
    empirical_support_lines (Dict[str, str]) -- exactly the same pieces
    agents/a5_verdict.py assembles for one record, just not yet turned
    into a single-record prompt."""
    ids = [str(b["record_id"]) for b in record_blocks]
    sections = [
        FRAMING_PREAMBLE,
        "",
        CALIBRATION_NOTE,
        "",
        _INSTRUCTIONS,
        f"The {len(ids)} record_ids in this batch, in the order shown below: {', '.join(ids)}\n",
    ]
    for b in record_blocks:
        sections.append(f"\n{'=' * 20} RECORD {b['record_id']} {'=' * 20}\n")
        sections.append("Chain review -- combined claims (evidence + behaviour + hypothesis stages):")
        sections.append(render_claims(b["chain_claims"]))
        sections.append("\nChain review -- candidate explanations (with empirical support):")
        sections.append(render_hypotheses(b["chain_hypotheses"], b["empirical_support_lines"]))
        sections.append("\nIndependent review -- claims:")
        sections.append(render_claims(b["a4_claims"]))
        sections.append("\nIndependent review -- candidate explanations (with empirical support):")
        sections.append(render_hypotheses(b["a4_hypotheses"], b["empirical_support_lines"]))
        sections.append(
            "\nMechanical agreement between chain and independent review "
            "(per chain stage: corroborated / contradicted / not addressed):"
        )
        sections.append(render_verification_summary(b["verification_by_agent"]))
    return "\n".join(sections) + "\n"
