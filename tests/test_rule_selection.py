"""Tests for eval/rule_selection.py (budgeted greedy P1 selection) and for the
v2 Priority 2 bin change (K stays benign-only, the wired constants are used)."""
import numpy as np
import pandas as pd
import pytest

from dataplane.escalation_policy import PRIORITY2_FEATURES, build_signature_table
from eval.rule_selection import (
    best_single,
    budget_rows,
    evaluate_selection,
    greedy_ratio,
    select_rules,
)


def _matrix(rows):
    return np.array(rows, dtype=bool)


def test_greedy_respects_budget():
    # 3 rules over 10 rows; budget 4 rows
    m = _matrix([
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ])
    is_att = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
    res = select_rules(m, is_att, budget_fraction=0.4)
    assert res.budget_rows == 4
    assert res.union_size <= 4


def test_union_is_deduplicated_when_rules_overlap():
    # two identical rules: the second adds zero new rows, so it must not be
    # selected and must not be charged against the budget
    m = _matrix([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    is_att = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    res = select_rules(m, is_att, budget_fraction=1.0)
    assert len(res.selected) == 1
    assert res.union_size == 2
    assert res.attack_coverage == 2


def test_union_counts_flows_matched_by_several_rules_once():
    m = _matrix([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
    ])
    is_att = np.array([1, 1, 1, 1, 0], dtype=bool)
    ev = evaluate_selection(m, is_att, [0, 1])
    assert ev["union_size"] == 4  # rows 0..3, row 1 and 2 counted once each
    assert ev["attack_matches"] == 4
    assert ev["benign_matches"] == 0


def test_coverage_is_monotone_along_the_trace():
    rng = np.random.default_rng(0)
    m = rng.random((12, 200)) < 0.15
    is_att = rng.random(200) < 0.3
    b = budget_rows(200, 0.5)
    chosen, trace = greedy_ratio(m, is_att, b)
    cov = [s.attack_coverage for s in trace]
    uni = [s.union_size for s in trace]
    assert cov == sorted(cov)
    assert uni == sorted(uni)
    assert all(u <= b for u in uni)


def test_selection_is_deterministic():
    rng = np.random.default_rng(7)
    m = rng.random((20, 300)) < 0.1
    is_att = rng.random(300) < 0.2
    a = select_rules(m, is_att, 0.2)
    b = select_rules(m, is_att, 0.2)
    assert a.selected == b.selected
    assert a.trace == b.trace


def test_best_single_beats_greedy_when_ratio_greedy_is_fooled():
    # Rule 0 has the best ratio (1 attack / 1 row = 1.0) and, once taken,
    # leaves only 9 rows of budget, so the bigger rule 1 (9 attacks + 1 benign
    # over 10 rows, ratio 0.9) no longer fits. Greedy covers 1 attack; the best
    # single rule (rule 1) covers 9.
    n = 20
    r0 = np.zeros(n, dtype=bool)
    r0[0] = True
    r1 = np.zeros(n, dtype=bool)
    r1[1:11] = True
    m = np.stack([r0, r1])
    is_att = np.zeros(n, dtype=bool)
    is_att[0] = True
    is_att[1:10] = True  # row 10 is benign
    res = select_rules(m, is_att, budget_fraction=0.5)  # budget 10 rows
    assert res.greedy_attack_coverage == 1
    assert res.best_single == 1
    assert res.best_single_attack_coverage == 9
    assert res.attack_coverage == 9
    assert res.strategy == "best_single"


def test_no_rule_chosen_when_none_fits_budget():
    m = _matrix([[1, 1, 1, 1, 1, 1]])
    is_att = np.ones(6, dtype=bool)
    res = select_rules(m, is_att, budget_fraction=0.1)  # budget 0 rows
    assert res.selected == ()
    assert res.attack_coverage == 0


def test_budget_rows_floors_not_ceils():
    assert budget_rows(1000, 0.001) == 1
    assert budget_rows(1999, 0.001) == 1


def _benign_fit(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.integers(0, 20, n).astype(float) for f in PRIORITY2_FEATURES})
    df["Label"] = "Benign"
    return df


def test_p2_table_is_benign_only_at_chosen_bin_count():
    fit = _benign_fit()
    attacked = fit.copy()
    attacked.loc[0, "Label"] = "DoS Hulk"
    with pytest.raises(AssertionError):
        build_signature_table(attacked, n_bins=20, floor=5)


def test_p2_common_signatures_meet_floor_at_chosen_bins():
    from dataplane.escalation_policy import signatures_for  # noqa: F401  (import check)
    from eval.generalization_experiments import compute_bin_edges, signatures_for as sigs_for
    from collections import Counter

    fit = _benign_fit(n=2000, seed=3)
    table = build_signature_table(fit, n_bins=20, floor=5)
    edges = compute_bin_edges(fit, list(PRIORITY2_FEATURES), 20)
    counts = Counter(sigs_for(fit, list(PRIORITY2_FEATURES), edges))
    assert table.common_signatures == frozenset(s for s, c in counts.items() if c >= 5)
    assert all(counts[s] >= 5 for s in table.common_signatures)


def test_production_call_sites_use_chosen_bins():
    # n_bins is now decided by results/p2_final.md Part 1 (6). This test checks that the
    # call sites read the constant, so the decision is in one place.
    import eval.escalation_eval as ee
    assert ee.P2_N_BINS == 6
    assert ee.P2_FLOOR == 5
