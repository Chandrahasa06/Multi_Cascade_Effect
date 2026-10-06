"""Budgeted rule selection for Priority 1 (the spec's S* formulation).

    S* = argmax  |A ∩ (∪_{r∈S} M_r)|   subject to   |∪_{r∈S} M_r| / N <= budget

where M_r is rule r's match mask over the N selection-population rows and A is
the attack rows. Greedy selection repeatedly takes the rule that adds the most
*uncovered* attack flows per unit of *new* union budget (a flow matched by
several rules is counted once, so the union, not the per-rule sum, is what the
budget charges).

Guarantee: ratio-greedy alone has no approximation bound under a knapsack
constraint. Taking the better of (ratio-greedy, best single feasible rule) does
give the classic (1 - 1/e) bound for budgeted max-coverage (Khuller, Moss &
Naor 1999). The result reports which of the two won, so the bound is only
claimed for that combined procedure, and it is an approximation, not the exact
optimum.

Rule matrices follow dataplane.dt_rules.evaluate_rules: shape
(n_rules, n_rows), one boolean row per rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dataplane.dt_rules import Rule, evaluate_rules


@dataclass(frozen=True)
class SelectionStep:
    rule_index: int  # position in the rule matrix (not Rule.id)
    gain_attacks: int  # newly covered attack rows from this step
    cost: int  # newly added union rows from this step
    union_size: int  # cumulative union size after this step
    attack_coverage: int  # cumulative covered attack rows after this step


@dataclass(frozen=True)
class SelectionResult:
    selected: Tuple[int, ...]  # rule-matrix positions, in selection order
    union_size: int
    attack_coverage: int
    strategy: str  # "greedy" or "best_single"
    greedy_selected: Tuple[int, ...]
    greedy_attack_coverage: int
    best_single: Optional[int]
    best_single_attack_coverage: int
    budget_rows: int
    trace: Tuple[SelectionStep, ...] = field(default_factory=tuple)


def rule_matrix(rules: Sequence[Rule], df) -> np.ndarray:
    return evaluate_rules(rules, df).astype(bool, copy=False)


def budget_rows(n_rows: int, budget_fraction: float) -> int:
    """Union-size cap in rows: floor(budget_fraction * N). Floored (not
    ceiled) so the selected union never exceeds the stated fraction."""
    return int(np.floor(budget_fraction * n_rows))


def _union_size(matrix: np.ndarray, picked: Sequence[int]) -> int:
    if not picked:
        return 0
    return int(matrix[list(picked)].any(axis=0).sum())


def _attack_coverage(matrix: np.ndarray, picked: Sequence[int], is_attack: np.ndarray) -> int:
    if not picked:
        return 0
    return int((matrix[list(picked)].any(axis=0) & is_attack).sum())


def greedy_ratio(matrix: np.ndarray, is_attack: np.ndarray, budget_rows_: int):
    """Ratio greedy: at each step take the feasible rule maximising
    (newly covered attack rows) / (newly added union rows). Ties go to the
    larger gain, then to the lower rule position, so the result is
    deterministic. Returns (selected positions, trace)."""
    n_rules, n_rows = matrix.shape
    covered = np.zeros(n_rows, dtype=bool)
    chosen: List[int] = []
    trace: List[SelectionStep] = []
    union = 0
    attacks_covered = 0
    remaining = set(range(n_rules))
    while True:
        best = None  # (ratio, gain, -position, position, cost)
        for r in sorted(remaining):
            new = matrix[r] & ~covered
            cost = int(new.sum())
            gain = int((new & is_attack).sum())
            if gain == 0 or union + cost > budget_rows_:
                continue
            key = (gain / cost, gain, -r)
            if best is None or key > best[0]:
                best = (key, r, cost, gain)
        if best is None:
            break
        _, r, cost, gain = best
        covered |= matrix[r]
        union += cost
        attacks_covered += gain
        chosen.append(r)
        remaining.discard(r)
        trace.append(SelectionStep(r, gain, cost, union, attacks_covered))
    return chosen, trace


def best_single(matrix: np.ndarray, is_attack: np.ndarray, budget_rows_: int) -> Optional[int]:
    """The single rule with the most attack coverage that fits the budget on
    its own. Ties go to the lower position."""
    best = None
    for r in range(matrix.shape[0]):
        size = int(matrix[r].sum())
        if size > budget_rows_:
            continue
        cov = int((matrix[r] & is_attack).sum())
        if best is None or cov > best[0]:
            best = (cov, r)
    return None if best is None else best[1]


def select_rules(matrix: np.ndarray, is_attack: np.ndarray, budget_fraction: float) -> SelectionResult:
    """Selection population is the rows of `matrix`/`is_attack`; the budget is
    `budget_fraction` of that population's size."""
    matrix = np.asarray(matrix, dtype=bool)
    is_attack = np.asarray(is_attack, dtype=bool)
    if matrix.shape[1] != is_attack.shape[0]:
        raise ValueError("matrix columns must align with is_attack rows")
    b = budget_rows(matrix.shape[1], budget_fraction)
    greedy_pos, trace = greedy_ratio(matrix, is_attack, b)
    greedy_cov = _attack_coverage(matrix, greedy_pos, is_attack)
    single = best_single(matrix, is_attack, b)
    single_cov = _attack_coverage(matrix, [single], is_attack) if single is not None else 0
    if single is not None and single_cov > greedy_cov:
        picked, strategy = [single], "best_single"
    else:
        picked, strategy = greedy_pos, "greedy"
    return SelectionResult(
        selected=tuple(picked),
        union_size=_union_size(matrix, picked),
        attack_coverage=_attack_coverage(matrix, picked, is_attack),
        strategy=strategy,
        greedy_selected=tuple(greedy_pos),
        greedy_attack_coverage=greedy_cov,
        best_single=single,
        best_single_attack_coverage=single_cov,
        budget_rows=b,
        trace=tuple(trace),
    )


def evaluate_selection(matrix: np.ndarray, is_attack: np.ndarray, picked: Sequence[int]) -> dict:
    """Union size, attack/benign match counts and precision for a fixed
    selection, on any population (e.g. a held-out one)."""
    matrix = np.asarray(matrix, dtype=bool)
    is_attack = np.asarray(is_attack, dtype=bool)
    if not picked:
        union = np.zeros(matrix.shape[1], dtype=bool)
    else:
        union = matrix[list(picked)].any(axis=0)
    n_att = int((union & is_attack).sum())
    n_ben = int((union & ~is_attack).sum())
    return {
        "n_rows": int(matrix.shape[1]),
        "union_size": n_att + n_ben,
        "union_rate": (n_att + n_ben) / matrix.shape[1] if matrix.shape[1] else None,
        "attack_matches": n_att,
        "benign_matches": n_ben,
        "precision": (n_att / (n_att + n_ben)) if (n_att + n_ben) else None,
        "attack_total": int(is_attack.sum()),
        "attack_recall": (n_att / int(is_attack.sum())) if is_attack.sum() else None,
    }
