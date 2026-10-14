"""Prompts for the self-check run, v2: v1 with the JSON-shape prose removed. The response
shape is carried by the native strict JSON schema (agents/openrouter.py), so the prompts say
only what each field must contain; every content rule is unchanged. v1 stays importable.
Every agent sees only one flow's own row (or, for A5b, the checklist
and the agents' justifications). No population, baseline or neighbourhood quantity.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from agents.prompts.priors_render_v2 import FLAG_COLUMNS, shown_features
from agents.selfcheck_core import BANNED_WORDS, CONDITIONS, CONDITION_IDS

VERSIONS = {"a1": "a1_selfcheck_v2", "a2": "a2_selfcheck_v2", "a3": "a3_selfcheck_v2",
            "a4": "a4_selfcheck_v2", "a5a": "a5a_selfcheck_v2", "a5b": "a5b_selfcheck_v2"}
PROTOCOL_NUMBERS = {"TCP": 6, "UDP": 17}
_IDS = ", ".join(CONDITION_IDS)


def flow_block(features: Dict[str, float], protocol: Optional[str], time_of_day: Optional[str]) -> str:
    shown = shown_features(features)
    proto = f"{protocol} (Protocol = {PROTOCOL_NUMBERS[protocol]})" if protocol in PROTOCOL_NUMBERS else "not recorded"
    lines = [f"Protocol: {proto}", f"Start time (local clock): {time_of_day or 'not recorded'}",
             "Fields of this flow's row (TCP flag fields are not recorded and not shown):"]
    lines += [f"  {k} = {v:g}" for k, v in sorted(shown.items())]
    return "\n".join(lines)


CONDITIONS_TEXT = (
    "Conditions. Each is a yes/no test on this flow's own fields:\n"
    + "\n".join(f"[{cid}] ({scope}) {test}" for cid, scope, test, _m in CONDITIONS)
    + "\nA condition scoped TCP is not applicable to UDP flows and vice versa; \"any\" applies to every flow."
)
CONDITIONS_WITH_MEANING = (
    "Conditions. Each is a yes/no test on this flow's own fields:\n"
    + "\n".join(f"[{cid}] ({scope}) {test}\n     {meaning}" for cid, scope, test, meaning in CONDITIONS)
    + "\nA condition scoped TCP is not applicable to UDP flows and vice versa; \"any\" applies to every flow."
)

CITE_RULE = (
    "Every claim is one sentence that names at least one field (written exactly as listed, for example \"Flow Duration\") and its value from the row, and lists "
    "those fields in `cited`, each with its name exactly as listed and its value copied exactly. "
    "Optionally list condition ids the claim bears on in `condition_ids`. Give `confidence` from 0 to 1. "
    "Claim ids: the prefix given, then a number."
)

ABSENCE_RULE = (
    "Absences (required): exactly four entries, one each for no_payload_sent, no_reply, no_teardown, "
    "no_size_variation. For each, `holds` is \"yes\", \"no\" or \"cannot_tell\", and `cited` lists the "
    "fields and values you read (may be empty only for cannot_tell). What each means:\n"
    "- no_payload_sent: the client sent no application data (read Total Length of Fwd Packets).\n"
    "- no_reply: nothing came back in either reply direction (read Total Backward Packets).\n"
    "- no_teardown: the connection was never closed. Teardown is carried in TCP flags, which are not "
    "recorded here.\n"
    "- no_size_variation: every packet had the same size (read Packet Length Variance, and the packet "
    "counts: with fewer than 2 packets there is nothing to vary)."
)

HYPOTHESIS_RULE = (
    "Hypotheses: 3 to 5, at least 2 benign. Each has an id (prefix given, then a number), a description, "
    "`benign` true/false, `condition_ids` it accounts for (may be empty), and `predictions`: at least 2 "
    "statements about THIS flow's own fields that must hold if the hypothesis is true, each as "
    "each naming a field exactly as listed, a comparison, and a number. "
    "Code checks every prediction against the row before anyone reads it: a hypothesis with any false "
    "prediction is struck out; one whose predictions name fields not in the row, or that cannot fail, "
    "earns no support."
)


def a1(flow: str) -> str:
    return (
        "You are reviewer A1. Report structural facts about this one flow, from its row only: its "
        "directionality, its duration against its packet count, how its bytes split between header and "
        "payload, and how regular its timing is. Do not compare it with other traffic: there is no "
        "baseline here, so words like " + ", ".join(BANNED_WORDS[:12]) + " (and their synonyms) are "
        "unsupportable and are rejected automatically.\n\n"
        + CONDITIONS_TEXT + "\n\n" + CITE_RULE + " Prefix: a1_c. Give 4 to 8 claims.\n\nThe flow:\n" + flow + "\n"
    )


def a2(flow: str, a1_text: str) -> str:
    return (
        "You are reviewer A2. Describe how this flow behaved, from its row, and then what is ABSENT from "
        "it. Absence matters as much as presence.\n\n"
        + CONDITIONS_TEXT + "\n\n" + CITE_RULE + " Prefix: a2_c. Give 3 to 6 claims.\n\n" + ABSENCE_RULE
        + "\n\nThe flow:\n" + flow + "\n\nA1's claims:\n" + a1_text + "\n"
    )


def a3(flow: str, a1_text: str, a2_text: str) -> str:
    return (
        "You are reviewer A3. Propose explanations of what this flow is, each grounded in predictions "
        "about its own fields.\n\n"
        + CONDITIONS_TEXT + "\n\n" + HYPOTHESIS_RULE + " Prefix: a3_h. Optionally add claims: " + CITE_RULE
        + " Prefix: a3_c.\n\nThe flow:\n" + flow + "\n\nA1's claims:\n" + a1_text + "\n\nA2's claims and absences:\n" + a2_text + "\n"
    )


def a4(flow: str) -> str:
    return (
        "You are reviewer A4, working alone: you see only this flow, never any other reviewer's work. Do "
        "three things, from the row only: report structural facts as claims, report what is absent, and "
        "propose explanations grounded in predictions about the flow's own fields.\n\n"
        + CONDITIONS_TEXT + "\n\n" + CITE_RULE + " Prefix: a4_c. Give 4 to 8 claims.\n\n" + ABSENCE_RULE
        + "\n\n" + HYPOTHESIS_RULE + " Prefix: a4_h.\n\nThe flow:\n" + flow + "\n"
    )


def a5a(flow: str) -> str:
    return (
        "You are A5a. Fill in the checklist for this flow from its row. For EVERY condition " + _IDS
        + ", give exactly one entry: `condition_id`; `holds` \"yes\", \"no\" or \"not_applicable\" (only "
        "for a condition scoped to the other protocol); and `observed`: the field(s) and value(s) you read, "
        "each name and value copied exactly. For not_applicable cite the field Protocol with value 6 "
        "for TCP or 17 for UDP.\n\n"
        + CONDITIONS_WITH_MEANING + "\n\nThe flow:\n" + flow + "\n"
    )


def a5b(checklist_text: str, justifications_text: str, yes_ids: List[str]) -> str:
    yes = ", ".join(yes_ids) if yes_ids else "none"
    return (
        "You are A5b. You decide the verdict for one flow. You do not see the flow's row: you see the "
        "checklist another reviewer filled in from it, and the reviewers' justifications, sorted from most "
        "to least reliable, each labelled with how reliable its author proved to be on this flow.\n\n"
        + CONDITIONS_WITH_MEANING + "\n\n"
        "Conditions marked yes on this flow: " + yes + ".\n\n"
        "For every condition marked yes, give a disposition: `condition_id`; `refuted` true only if a "
        "benign explanation marked SUPPORTED accounts for it, citing that explanation's id in `refuted_by`; "
        "otherwise `refuted` false; and a one-sentence `reason`. Explanations marked CONTRADICTED or "
        "UNCHECKABLE cannot refute anything.\n\n"
        "Then the `verdict`, one of:\n"
        "- consistent_with_benign: no condition holds.\n"
        "- anomalous_but_explicable: conditions hold, and every one is refuted.\n"
        "- anomalous_and_unexplained: conditions hold, and at least one is not refuted.\n"
        "- insufficient_evidence: you cannot tell on the conditions that matter.\n"
        "Give a short `rationale` naming the conditions and explanations that decided it.\n\n"
        "Checklist:\n" + checklist_text + "\n\nJustifications:\n" + justifications_text + "\n"
    )


# ------------------------------------------------------------------ rendering helpers
def claims_text(claims) -> str:
    out = []
    for c in claims:
        cites = ", ".join(f"{v.name} = {v.value:g}" for v in c.cited)
        conds = f" [conditions: {', '.join(c.condition_ids)}]" if c.condition_ids else ""
        out.append(f"  {c.claim_id}: {c.statement} ({cites}){conds}")
    return "\n".join(out) or "  (none)"


def absences_text(absences) -> str:
    out = []
    for a in absences:
        cites = ", ".join(f"{v.name} = {v.value:g}" for v in a.cited) or "nothing cited"
        out.append(f"  {a.absence.value}: {a.holds.value} ({cites})")
    return "\n".join(out) or "  (none)"


def hypotheses_text(hyps, statuses: Dict[str, str]) -> str:
    out = []
    for h in hyps:
        st = statuses.get(h.hypothesis_id, "?")
        preds = "; ".join(f"{p.field} {p.op.value} {p.value:g}" for p in h.predictions)
        conds = ", ".join(h.condition_ids) or "none"
        mark = " (STRUCK OUT)" if st == "CONTRADICTED" else ""
        out.append(f"  {h.hypothesis_id} [{st}]{mark} ({'benign' if h.benign else 'hostile'}; accounts for: {conds}): "
                   f"{h.description}\n      predictions: {preds}")
    return "\n".join(out) or "  (none)"


def checklist_text(checklist) -> str:
    return "\n".join(
        f"  {r.condition_id}: {r.holds.value} ("
        + ", ".join(f"{v.name} = {v.value:g}" for v in r.observed) + ")" for r in checklist)
