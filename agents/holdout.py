"""Held-out fields for A3: makes its hypothesis predictions falsifiable.

For each record, 3 of the 10 escalation features are withheld from A3, chosen by a seed so the
set is the same for every run. A3 sees the row without them (and without any field identical
to one of them), and without any A1/A2 claim or absence that names or reads them. Each A3
hypothesis must predict what the withheld fields contain; code checks the predictions against
the real values.

No label is read here: the choice depends only on the seed and the opaque record id.
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, Sequence, Tuple

from agents.escalation_grounding import ESCALATION_FEATURES
from agents.selfcheck_core import (ABSENCE_FIELDS, CONTRADICTED, SUPPORTED, UNCHECKABLE, _OPS, is_vacuous,
                                   normalise_name)

HOLDOUT_SEED = 20261016
N_HELD_OUT = 3

#: fields equal to an escalation feature on every one of the 20 records (checked before any
#: run): showing the twin would show the withheld value itself, so it is withheld too.
FIELD_TWINS = {
    "Subflow Fwd Packets": ("Total Fwd Packets",),
    "Total Backward Packets": ("Subflow Bwd Packets",),
}

#: hard ranges from protocol semantics, not from any population: a prediction that every value
#: in the range satisfies cannot fail. Init_Win_bytes is a 16-bit TCP field, -1 when absent.
FIELD_RANGE = {"Init_Win_bytes_forward": (-1.0, 65535.0), "Init_Win_bytes_backward": (-1.0, 65535.0)}


def held_out(record_id: str) -> Tuple[str, ...]:
    """The 3 withheld escalation features for this record, fixed by HOLDOUT_SEED."""
    rng = random.Random(f"{HOLDOUT_SEED}:{record_id}")
    return tuple(sorted(rng.sample(sorted(ESCALATION_FEATURES), N_HELD_OUT)))


def hidden_fields(record_id: str) -> Tuple[str, ...]:
    """Everything A3 must not see: the withheld features and their exact twins."""
    held = held_out(record_id)
    return tuple(sorted(set(held) | {t for h in held for t in FIELD_TWINS.get(h, ())}))


def redact_features(features: Dict[str, float], hidden: Iterable[str]) -> Dict[str, float]:
    hidden = set(hidden)
    return {k: v for k, v in features.items() if k not in hidden}


def mentions(text: str, field: str, visible: Iterable[str]) -> bool:
    """Whether `text` names `field`, not counting longer visible field names that merely contain
    it ("Fwd Packet Length Mean" does not name "Packet Length Mean")."""
    t = normalise_name(text)
    f = normalise_name(field)
    for v in sorted({normalise_name(x) for x in visible}, key=len, reverse=True):
        if f in v and v != f:
            t = t.replace(v, " ")
    return f in t


def leaked_fields(text: str, hidden: Iterable[str], visible: Iterable[str]) -> list:
    visible = list(visible)
    return [h for h in hidden if mentions(text, h, visible)]


def redact_claims(claims, hidden: Iterable[str], visible: Iterable[str]):
    """Drops every claim that cites or names a hidden field."""
    hidden, visible = tuple(hidden), tuple(visible)
    return [c for c in claims
            if not (set(v.name for v in c.cited) & set(hidden)) and not leaked_fields(c.statement, hidden, visible)]


def redact_absences(absences, hidden: Iterable[str]):
    """Drops every absence whose answer is read from a hidden field (its definition or its citations)."""
    hidden = set(hidden)
    keep = []
    for a in absences:
        kind = a.absence.value
        reads = set(ABSENCE_FIELDS.get(kind, ())) | {v.name for v in a.cited}
        if not reads & hidden:
            keep.append(a)
    return keep


def is_unbounded(field: str, op: str, value: float) -> bool:
    """True when the prediction cannot fail: the existing vacuity test, plus a hard field range."""
    if is_vacuous(op, value):
        return True
    lo, hi = FIELD_RANGE.get(field, (None, None))
    if hi is not None and ((op == "le" and value >= hi) or (op == "lt" and value > hi)):
        return True
    if lo is not None and ((op == "ge" and value <= lo) or (op == "gt" and value < lo)):
        return True
    return False


def holdout_status(predictions: Sequence, row: Dict[str, float], held: Sequence[str]) -> Tuple[str, list]:
    """CONTRADICTED if any prediction is false against the real value. Otherwise SUPPORTED only if
    every withheld field has at least one true prediction that could have failed; else UNCHECKABLE."""
    detail = []
    any_false = False
    bounded_true = set()
    for p in predictions:
        op = p.op.value if hasattr(p.op, "value") else str(p.op)
        if p.field not in row:
            detail.append({"field": p.field, "op": op, "value": p.value, "result": "field not in row"})
            continue
        ok = _OPS[op](float(row[p.field]), float(p.value))
        unb = is_unbounded(p.field, op, float(p.value))
        detail.append({"field": p.field, "op": op, "value": p.value, "actual": float(row[p.field]),
                       "result": "true" if ok else "false", "unbounded": unb})
        if not ok:
            any_false = True
        elif not unb:
            bounded_true.add(p.field)
    if any_false:
        return CONTRADICTED, detail
    if not set(held) <= bounded_true:
        return UNCHECKABLE, detail
    return SUPPORTED, detail


def validate_holdout_predictions(hyps, held: Sequence[str]) -> None:
    """Every A3 hypothesis predicts each withheld field, and only withheld fields. All problems
    reported at once."""
    held = set(held)
    errors = []
    for h in hyps:
        fields = {p.field for p in h.predictions}
        missing = sorted(held - fields)
        extra = sorted(fields - held)
        if missing:
            errors.append(f"{h.hypothesis_id}: give a prediction for each withheld field; missing {missing}")
        if extra:
            errors.append(f"{h.hypothesis_id}: predict only the withheld fields {sorted(held)}; "
                          f"{extra} are visible to you, so a prediction about them tests nothing")
    if errors:
        raise ValueError("; ".join(errors))
