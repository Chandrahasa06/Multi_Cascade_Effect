"""Compiles decision-tree root-to-leaf paths (from ``results/tree.json``,
see ``eval/parse_tree.py``) into standalone interval rules, and evaluates
them against a feature DataFrame.

A rule is a conjunction of per-feature intervals: taking a left branch at
some node contributes ``x_f <= t``, a right branch contributes
``x_f > t``. Repeated conditions on the same feature along one path are
merged into the single tightest interval implied by all of them. Where a
feature is integer-valued, a fractional threshold is converted to an
equivalent integer bound (``x > 4.5 and x <= 11.5`` -> ``5 <= x <= 11``)
-- this mirrors ``controller_rule_selection.pdf``'s own worked example
exactly.

**Which features are integer-valued is a judgement call, made explicit
here rather than inferred from dtype.** Of tree.txt's actual 10 features
(see eval/parse_tree.py's docstring for why this differs from the task
brief's original list), 7 are raw observed counts/byte-maxima that are
genuinely integer registers in a switch (Fwd Packet Length Max,
Init_Win_bytes_forward, Init_Win_bytes_backward, min_seg_size_forward,
act_data_pkt_fwd, Subflow Fwd Packets, Total Backward Packets) --
INTEGER_FEATURES below. The other 3 (Bwd Packets/s, Flow IAT Mean,
Packet Length Mean) are a rate and two means -- never integer-valued
even though their *inputs* are -- and additionally are exactly the
features that would need cross-multiplication against a switch register
in a real deployment (see DERIVED_FEATURES and this module's evaluation
note below).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from eval.parse_tree import LeafNode, SplitNode, iter_leaves, path_to_leaf

#: features tree.txt splits on that are genuine integer counters in a
#: switch register (see module docstring).
INTEGER_FEATURES = frozenset({
    "Fwd Packet Length Max",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "min_seg_size_forward",
    "act_data_pkt_fwd",
    "Subflow Fwd Packets",
    "Total Backward Packets",
})

#: rate/mean features that, in a real switch, would be evaluated by
#: cross-multiplying against the constant threshold rather than by
#: computing the rate/mean directly (see task brief: "b/dt <= t becomes
#: b <= t*dt"). This eval reads them straight from the CSV; rules that
#: constrain any of these are flagged so the report can call out which
#: ones would need the rearrangement in a real deployment.
DERIVED_FEATURES = frozenset({"Bwd Packets/s", "Flow IAT Mean", "Packet Length Mean"})

BENIGN_CLASS = "BENIGN"


@dataclass(frozen=True)
class Interval:
    """One feature's constraint within a rule. ``lo``/``hi`` are ``None``
    for an unconstrained side ("Any"). For an integer feature both bounds
    are inclusive; for a continuous feature ``lo`` is exclusive (strict
    ``>``) and ``hi`` is inclusive (``<=``), matching how the tree itself
    compares."""

    lo: Optional[float] = None
    hi: Optional[float] = None
    lo_inclusive: bool = False
    hi_inclusive: bool = True
    is_integer: bool = False

    def describe(self) -> str:
        if self.lo is None and self.hi is None:
            return "Any"
        if self.is_integer:
            if self.lo is not None and self.hi is not None:
                if self.lo == self.hi:
                    return f"= {self.lo:g}"
                return f"{self.lo:g}-{self.hi:g}"
            if self.lo is not None:
                return f">= {self.lo:g}"
            return f"<= {self.hi:g}"
        lo_op = ">=" if self.lo_inclusive else ">"
        hi_op = "<=" if self.hi_inclusive else "<"
        if self.lo is not None and self.hi is not None:
            return f"{lo_op} {self.lo:g} and {hi_op} {self.hi:g}"
        if self.lo is not None:
            return f"{lo_op} {self.lo:g}"
        return f"{hi_op} {self.hi:g}"

    def matches(self, values: np.ndarray) -> np.ndarray:
        mask = np.ones(len(values), dtype=bool)
        if self.lo is not None:
            mask &= (values >= self.lo) if self.lo_inclusive else (values > self.lo)
        if self.hi is not None:
            mask &= (values <= self.hi) if self.hi_inclusive else (values < self.hi)
        return mask


@dataclass(frozen=True)
class Rule:
    id: int
    leaf_id: int
    predicted_class: str
    intervals: Dict[str, Interval]
    raw_conditions: Tuple[Tuple[str, str, float], ...]  # (feature, op, threshold), root-to-leaf order

    @property
    def depends_on_derived(self) -> List[str]:
        return sorted(f for f in self.intervals if f in DERIVED_FEATURES)

    def matches(self, df: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(df), dtype=bool)
        for feat, interval in self.intervals.items():
            mask &= interval.matches(df[feat].to_numpy(dtype=float))
        return mask


def _merge_conditions(
    conditions: Sequence[Tuple[str, str, float]],
    integer_features: frozenset = INTEGER_FEATURES,
) -> Dict[str, Interval]:
    """Merges repeated per-feature conditions along one path into the
    tightest single interval per feature, applying the integer-bound
    conversion for features in `integer_features`."""
    lo: Dict[str, float] = {}
    hi: Dict[str, float] = {}
    for feature, op, threshold in conditions:
        is_int = feature in integer_features
        if op == "<=":
            bound = math.floor(threshold) if is_int else threshold
            hi[feature] = bound if feature not in hi else min(hi[feature], bound)
        else:  # ">"
            bound = math.floor(threshold) + 1 if is_int else threshold
            lo[feature] = bound if feature not in lo else max(lo[feature], bound)

    features = set(lo) | set(hi)
    intervals: Dict[str, Interval] = {}
    for feature in features:
        is_int = feature in integer_features
        intervals[feature] = Interval(
            lo=lo.get(feature), hi=hi.get(feature),
            lo_inclusive=is_int, hi_inclusive=True, is_integer=is_int,
        )
    return intervals


def compile_rule(root: SplitNode, leaf: LeafNode, rule_id: int) -> Rule:
    conditions = tuple(path_to_leaf(root, leaf.id))
    intervals = _merge_conditions(conditions)
    return Rule(id=rule_id, leaf_id=leaf.id, predicted_class=leaf.predicted_class,
                intervals=intervals, raw_conditions=conditions)


def compile_attack_rules(root: SplitNode, benign_class: str = BENIGN_CLASS) -> List[Rule]:
    """Every non-Benign leaf, in the order encountered by a pre-order
    traversal (matches eval/parse_tree.py's leaf ids) -- not hand-picked,
    per the brief."""
    leaves = [l for l in iter_leaves(root) if l.predicted_class != benign_class]
    return [compile_rule(root, leaf, i) for i, leaf in enumerate(leaves)]


def evaluate_rules(rules: Sequence[Rule], df: pd.DataFrame) -> np.ndarray:
    """Boolean matrix, shape (n_rules, n_rows): rule i's match mask."""
    return np.stack([r.matches(df) for r in rules]) if rules else np.zeros((0, len(df)), dtype=bool)


def evaluate_rules_union(rules: Sequence[Rule], df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (matched_any, n_rules_matched) -- the deduplicated union
    (a flow matching several rules is escalated once, not once per rule)
    plus, for reporting, how many distinct rules each flow matched."""
    if not rules:
        n = len(df)
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=int)
    matrix = evaluate_rules(rules, df)
    n_matched = matrix.sum(axis=0)
    return n_matched > 0, n_matched


def rule_predicted_class(rules: Sequence[Rule], df: pd.DataFrame, benign_class: str = BENIGN_CLASS) -> np.ndarray:
    """First-matching-rule's predicted class per flow (rules tried in id
    order), or `benign_class` if none match. Useful as a drop-in
    "predicted_class" array for callers built around a single-label
    Priority-1 classifier."""
    out = np.full(len(df), benign_class, dtype=object)
    matched = np.zeros(len(df), dtype=bool)
    for r in rules:
        m = r.matches(df) & ~matched
        out[m] = r.predicted_class
        matched |= m
    return out


def first_matching_rule(rules: Sequence[Rule], df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Like rule_predicted_class, but also returns WHICH rule id matched
    each flow (rules tried in id order, first match wins) -- for per-flow
    provenance reporting (e.g. an escalated-flow dump), where "the class
    it predicted" alone isn't enough to trace back to a specific rule.
    Unmatched flows get rule_id -1 and predicted_class ''."""
    n = len(df)
    rule_id = np.full(n, -1, dtype=int)
    predicted_class = np.full(n, "", dtype=object)
    matched = np.zeros(n, dtype=bool)
    for r in rules:
        m = r.matches(df) & ~matched
        rule_id[m] = r.id
        predicted_class[m] = r.predicted_class
        matched |= m
    return rule_id, predicted_class
