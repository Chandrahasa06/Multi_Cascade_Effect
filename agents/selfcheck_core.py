"""Shared definitions for the self-check run (see results/selfcheck_preregistered.md).

Everything here is checked against one flow's own row. No population, baseline, reference
pool or neighbourhood quantity is used anywhere.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from agents.conditions_v3 import evaluate as evaluate_v3
from agents.prompts.priors_text_v3 import CONDITIONS as V3_CONDITIONS
from eval.conditions_v4 import CONDITIONS as V4_CONDITIONS

TOL = 1e-3

# ------------------------------------------------------------------ conditions (14)
_V4_KEEP = ("D2", "D4")
_V4 = {c.cid: c for c in V4_CONDITIONS if c.cid in _V4_KEEP}

#: (id, scope, test text, meaning)
CONDITIONS: Tuple[Tuple[str, str, str, str], ...] = tuple(
    (cid, scope, test, meaning) for cid, scope, test, meaning, _src in V3_CONDITIONS
) + tuple((cid, _V4[cid].scope, _V4[cid].expression, _V4[cid].justification) for cid in _V4_KEEP)
CONDITION_IDS = tuple(c[0] for c in CONDITIONS)
SCOPES = {c[0]: c[1] for c in CONDITIONS}

#: fields each condition's test needs (for checking an agent against its own citations)
CONDITION_FIELDS: Dict[str, Tuple[str, ...]] = {
    "C1": ("Init_Win_bytes_backward",), "C2": ("Total Backward Packets",),
    "C3": ("Total Length of Fwd Packets", "Total Length of Bwd Packets"), "C4": ("Init_Win_bytes_forward",),
    "C5": ("Flow IAT Mean",), "C6": ("Total Length of Fwd Packets", "Total Length of Bwd Packets"),
    "C7": ("Total Fwd Packets", "Total Backward Packets", "Packet Length Std"),
    "C8": ("Total Length of Fwd Packets", "Total Length of Bwd Packets"), "C9": ("Destination Port",),
    "C10": ("Destination Port", "Total Fwd Packets", "Total Backward Packets"),
    "C11": ("Flow Packets/s", "Average Packet Size"),
    "C12": ("Destination Port", "Total Length of Fwd Packets", "Total Length of Bwd Packets"),
    "D2": ("Fwd Packet Length Max", "Fwd Packet Length Min", "Total Fwd Packets"),
    "D4": ("act_data_pkt_fwd", "Total Fwd Packets"),
}


def condition_holds(values: Dict[str, float], protocol: Optional[str]) -> Dict[str, Optional[str]]:
    """'yes' / 'no' / 'not_applicable' per condition, from the given values. A condition whose
    fields are missing from `values` gives None (cannot be evaluated from these values)."""
    out: Dict[str, Optional[str]] = {}
    full = {k: float(v) for k, v in values.items()}
    for cid in CONDITION_IDS:
        scope = SCOPES[cid]
        if scope != "any" and scope != protocol:
            out[cid] = "not_applicable"
            continue
        if not all(f in full for f in CONDITION_FIELDS[cid]):
            out[cid] = None
            continue
        if cid in _V4:
            out[cid] = "yes" if _V4[cid].test(full) else "no"
        else:
            # conditions_v3.evaluate needs every v3 column; fill missing ones neutrally, then read only cid
            filler = {k: 1.0 for f in CONDITION_FIELDS.values() for k in f}
            filler.update(full)
            out[cid] = evaluate_v3(filler, protocol)[cid]
    return out


# ------------------------------------------------------------------ absences (A2, A4)
ABSENCE_FIELDS = {
    "no_payload_sent": ("Total Length of Fwd Packets",),
    "no_reply": ("Total Backward Packets",),
    "no_size_variation": ("Packet Length Variance", "Total Fwd Packets", "Total Backward Packets"),
    "no_teardown": (),  # not recorded: the TCP flag columns are hidden
}


def absence_truth(values: Dict[str, float]) -> Dict[str, Optional[str]]:
    """'yes'/'no' per absence from the given values; None when a needed field is missing.
    no_teardown is always None (not recorded). no_size_variation needs at least 2 packets."""
    out: Dict[str, Optional[str]] = {"no_teardown": None}
    v = values
    out["no_payload_sent"] = None if "Total Length of Fwd Packets" not in v else ("yes" if float(v["Total Length of Fwd Packets"]) == 0 else "no")
    out["no_reply"] = None if "Total Backward Packets" not in v else ("yes" if float(v["Total Backward Packets"]) == 0 else "no")
    if all(f in v for f in ABSENCE_FIELDS["no_size_variation"]):
        pk = float(v["Total Fwd Packets"]) + float(v["Total Backward Packets"])
        out["no_size_variation"] = None if pk < 2 else ("yes" if float(v["Packet Length Variance"]) == 0 else "no")
    else:
        out["no_size_variation"] = None
    return out


# ------------------------------------------------------------------ A1 banned words
BANNED_WORDS = ("high", "higher", "highest", "low", "lower", "lowest", "unusual", "unusually", "elevated",
                "abnormal", "abnormally", "typical", "typically", "atypical", "normal", "normally", "anomalous",
                "excessive", "excessively", "large", "larger", "small", "smaller", "significant", "significantly",
                "rare", "rarely", "ordinary", "extreme", "extremely")
_BANNED = re.compile(r"\b(" + "|".join(BANNED_WORDS) + r")\b", re.I)


def banned_words_in(text: str) -> List[str]:
    """The original word list, kept for reports on runs made under it. Claims are now checked
    with `comparison_violations`."""
    return sorted({m.group(0).lower() for m in _BANNED.finditer(text)})


# ------------------------------------------------------------------ A1 comparison rule (replaces the word list)
#: judge this flow against traffic outside it: always rejected
JUDGE_WORDS = ("unusual", "unusually", "abnormal", "abnormally", "typical", "typically", "atypical", "normal",
               "normally", "anomalous", "anomalously", "excessive", "excessively", "rare", "rarely", "ordinary",
               "extreme", "extremely", "significant", "significantly", "elevated")
#: a bare size judgement ("a small payload") has no second field to compare with: always rejected
MAGNITUDE_WORDS = ("high", "low", "large", "small", "highest", "lowest", "largest", "smallest")
#: allowed only when the same sentence names two fields it cites (a within-flow comparison)
COMPARATIVES = ("larger", "smaller", "higher", "lower", "longer", "shorter")
#: comparisons whose target is outside this flow: always rejected, whatever the word
EXTERNAL_PATTERNS = (
    r"\bthan\s+(?:(?:is|are|for|in|on|with)\s+)?(?:the\s+)?(?:most|many|usual|typical|normal|average|expected|"
    r"ordinary|common|standard|baseline|norm|others?|similar|comparable)\b",
    r"\bthan\s+(?:the\s+)?other\b",
    r"\b(?:compared|relative)\s+(?:to|with)\s+(?:the\s+)?(?:other|most|many|typical|normal|usual|average|similar)\b",
    r"\b(?:other|most|many|typical|normal|usual|average|similar|comparable)\s+"
    r"(?:flows?|connections?|traffic|sessions?|hosts?|clients?|servers?|requests?)\b",
    r"\bfor\s+(?:a|an|this\s+(?:kind|type)\s+of)\s+(?:\w+\s+)?(?:flow|connection|session)s?\b",
    r"\b(?:usual|usually|on\s+average|as\s+expected)\b",
)
_WORD_RE = {w: re.compile(rf"\b{w}\b", re.I) for w in JUDGE_WORDS + MAGNITUDE_WORDS + COMPARATIVES}
_EXTERNAL_RE = [re.compile(p, re.I) for p in EXTERNAL_PATTERNS]


def normalise_name(text: str) -> str:
    """Case, underscores and Fwd/Forward, Bwd/Backward do not change which field is named."""
    t = text.lower().replace("_", " ")
    t = re.sub(r"\bforward\b", "fwd", t)
    t = re.sub(r"\bbackward\b", "bwd", t)
    return " ".join(t.split())


def names_field(statement: str, field: str) -> bool:
    return normalise_name(field) in normalise_name(statement)


def comparison_violations(statement: str, cited_names: Sequence[str]) -> List[str]:
    """Every reason `statement` compares against something outside this flow, all at once."""
    found = [w for w in JUDGE_WORDS + MAGNITUDE_WORDS if _WORD_RE[w].search(statement)]
    named = {n for n in cited_names if names_field(statement, n)}
    if len(named) < 2:
        found += [w for w in COMPARATIVES if _WORD_RE[w].search(statement)]
    out = []
    if found:
        out.append(f"words that need a baseline: {found}"
                   + (" (larger/smaller/higher/lower/longer/shorter are allowed only between two cited fields "
                      "named in the same sentence)" if any(w in COMPARATIVES for w in found) else ""))
    ext = sorted({m.group(0).lower() for r in _EXTERNAL_RE for m in r.finditer(statement)})
    if ext:
        out.append(f"compares with traffic outside this flow: {ext}")
    return out


# ------------------------------------------------------------------ hypothesis status
SUPPORTED, CONTRADICTED, UNCHECKABLE = "SUPPORTED", "CONTRADICTED", "UNCHECKABLE"
_OPS = {"eq": lambda a, b: abs(a - b) <= max(abs(a), abs(b), 1e-9) * TOL,
        "lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b}


def is_vacuous(op: str, value: float) -> bool:
    return (op == "ge" and value <= 0) or (op == "gt" and value < 0)


def hypothesis_status(predictions: Sequence, row: Dict[str, float]) -> Tuple[str, List[dict]]:
    detail = []
    any_false = any_missing = False
    non_vacuous_true = 0
    for p in predictions:
        op = p.op.value if hasattr(p.op, "value") else str(p.op)
        if p.field not in row:
            detail.append({"field": p.field, "op": op, "value": p.value, "result": "field not in row"})
            any_missing = True
            continue
        ok = _OPS[op](float(row[p.field]), float(p.value))
        vac = is_vacuous(op, float(p.value))
        detail.append({"field": p.field, "op": op, "value": p.value, "actual": float(row[p.field]),
                       "result": "true" if ok else "false", "vacuous": vac})
        if not ok:
            any_false = True
        elif not vac:
            non_vacuous_true += 1
    if any_false:
        return CONTRADICTED, detail
    if any_missing or non_vacuous_true == 0:
        return UNCHECKABLE, detail
    return SUPPORTED, detail


# ------------------------------------------------------------------ trust (F, B, S, H)
def _match(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(a), abs(b), 1e-9) * TOL


def cited_pairs(claims=(), absences=(), checklist=()) -> List[Tuple[str, float]]:
    out = []
    for c in claims:
        out += [(v.name, float(v.value)) for v in c.cited]
    for a in absences:
        out += [(v.name, float(v.value)) for v in a.cited]
    for r in checklist:
        out += [(v.name, float(v.value)) for v in r.observed if v.name != "Protocol"]
    return out


def field_fidelity(pairs: Sequence[Tuple[str, float]], row: Dict[str, float]) -> Optional[float]:
    if not pairs:
        return None
    return sum(1 for n, v in pairs if n in row and _match(v, float(row[n]))) / len(pairs)


def blind_corroboration(pairs: Sequence[Tuple[str, float]], a4_pairs: Sequence[Tuple[str, float]]) -> Optional[float]:
    a4 = {}
    for n, v in a4_pairs:
        a4.setdefault(n, []).append(v)
    shared = [(n, v) for n, v in pairs if n in a4]
    if not shared:
        return None
    return sum(1 for n, v in shared if any(_match(v, w) for w in a4[n])) / len(shared)


def self_consistency(absences=(), checklist=(), protocol: Optional[str] = None) -> Optional[float]:
    """Assertions whose answer can be recomputed from the values the agent itself cited."""
    checks = []
    for a in absences:
        own = {v.name: float(v.value) for v in a.cited}
        truth = absence_truth(own).get(a.absence.value)
        if truth is not None and a.holds.value in ("yes", "no"):
            checks.append(truth == a.holds.value)
    for r in checklist:
        own = {v.name: float(v.value) for v in r.observed if v.name != "Protocol"}
        truth = condition_holds(own, protocol).get(r.condition_id)
        if truth in ("yes", "no") and r.holds.value in ("yes", "no"):
            checks.append(truth == r.holds.value)
    if not checks:
        return None
    return sum(checks) / len(checks)


def abstention_honesty(absences, row: Dict[str, float]) -> Optional[float]:
    opps = []
    for a in absences:
        kind = a.absence.value
        if kind == "no_teardown":
            opps.append(a.holds.value == "cannot_tell")
        elif kind == "no_size_variation":
            pk = float(row.get("Total Fwd Packets", 0)) + float(row.get("Total Backward Packets", 0))
            if pk < 2:
                opps.append(a.holds.value == "cannot_tell")
    if not opps:
        return None
    return sum(opps) / len(opps)


def trust_T(components: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [v for v in components.values() if v is not None]
    return sum(vals) / len(vals) if vals else None


HIGH, MEDIUM, UNRELIABLE, FLOOR = 0.75, 0.50, 0.25, 0.25


def band(T: Optional[float]) -> str:
    if T is None:
        return "unscored"
    if T >= HIGH:
        return "high"
    if T >= MEDIUM:
        return "medium"
    if T >= UNRELIABLE:
        return "unreliable"
    return "below floor"


def withheld(T: Optional[float]) -> bool:
    return T is not None and T < FLOOR


# ------------------------------------------------------------------ verdict rules
def allowed_verdicts(yes_ids: Iterable[str], dispositions) -> set:
    yes = set(yes_ids)
    allowed = {"insufficient_evidence"}
    if not yes:
        allowed.add("consistent_with_benign")
    elif all(d.refuted for d in dispositions if d.condition_id in yes):
        allowed.add("anomalous_but_explicable")
    else:
        allowed.add("anomalous_and_unexplained")
    return allowed


def checklist_alone_verdict(yes_ids: Iterable[str]) -> str:
    return "anomalous_and_unexplained" if set(yes_ids) else "consistent_with_benign"
