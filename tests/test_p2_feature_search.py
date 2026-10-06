"""Tests for eval/p2_feature_search.py: encoder equivalence with production
signatures, deterministic search, and that selection never reads rows outside
the fit split (holdout, sample, and pool rows cannot change the chosen path)."""
import numpy as np
import pandas as pd

from dataplane.escalation_policy import build_signature_table, priority2_escalate
from eval.generalization_experiments import build_K, compute_bin_edges, signatures_for
from eval.p2_feature_search import (
    FitTable,
    P2Evaluator,
    backward_eliminate,
    build_fit_table,
    escalate,
    fast_codes,
    greedy_forward,
    signature_space,
)

FEATS = ["a", "b", "c", "d"]


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.gamma(2.0, 3.0, n) for f in FEATS})
    df.loc[rng.random(n) < 0.05, "b"] = np.nan  # NaN sentinel must be exercised
    df["Label"] = np.where(rng.random(n) < 0.2, "DoS Hulk", "BENIGN")
    return df


def _evaluator(df, fit_rows=300, hold_rows=(300, 450), seed=1):
    n = len(df)
    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[:fit_rows] = True
    hold_mask = np.zeros(n, dtype=bool)
    hold_mask[hold_rows[0]:hold_rows[1]] = True
    rng = np.random.default_rng(seed)
    sample_index = rng.choice(n, size=200, replace=False)
    p1_pool = np.zeros(n, dtype=bool)
    p1_sample = p1_pool[sample_index]
    return P2Evaluator(df, fit_mask, hold_mask, sample_index, p1_pool, p1_sample)


def test_fast_codes_match_production_signatures_including_nan():
    df = _synthetic()
    benign = df[df["Label"] == "BENIGN"]
    edges = compute_bin_edges(benign, FEATS, 8)
    sigs = signatures_for(df, FEATS, edges)
    codes = fast_codes({f: df[f].to_numpy(float) for f in FEATS}, FEATS, edges)
    # injective encoding: equal tuples <=> equal codes
    tuple_to_code = {}
    for s, c in zip(sigs, codes):
        assert tuple_to_code.setdefault(s, c) == c
    assert len(set(tuple_to_code.values())) == len(tuple_to_code)


def test_fast_escalation_matches_production_p2_on_five_features():
    from dataplane.escalation_policy import PRIORITY2_FEATURES
    rng = np.random.default_rng(5)
    n = 3000
    df = pd.DataFrame({f: rng.integers(0, 40, n).astype(float) for f in PRIORITY2_FEATURES})
    df.loc[rng.random(n) < 0.03, PRIORITY2_FEATURES[2]] = np.nan
    df["Label"] = "Benign"
    prod = build_signature_table(df, n_bins=20, floor=5)
    ft = build_fit_table({f: df[f].to_numpy(float) for f in PRIORITY2_FEATURES}, PRIORITY2_FEATURES, 20, 5)
    assert len(ft.common) == len(prod.common_signatures)
    p = priority2_escalate(df, prod)
    q = escalate({f: df[f].to_numpy(float) for f in PRIORITY2_FEATURES}, ft)
    assert (p == q).all()


def test_signature_space_is_product_of_effective_bins():
    edges = {"a": np.array([1.0, 2.0]), "b": np.array([0.5])}
    assert signature_space(["a", "b"], edges) == 3 * 2


def test_k_is_benign_fit_only_and_floor_respected():
    df = _synthetic(n=800, seed=3)
    fit = df.iloc[:400]
    fb = {f: fit[fit["Label"] == "BENIGN"][f].to_numpy(float) for f in FEATS}
    t = build_fit_table(fb, FEATS, 10, 5)
    edges = compute_bin_edges(fit[fit["Label"] == "BENIGN"], FEATS, 10)
    counts = pd.Series(fast_codes(fb, FEATS, edges)).value_counts()
    assert set(t.common.tolist()) == set(counts[counts >= 5].index.tolist())


def test_fit_objective_ignores_non_fit_rows():
    df = _synthetic()
    ev1 = _evaluator(df)
    before = ev1.fit_objective(FEATS, 8, 3)
    df2 = df.copy()
    rest = np.arange(len(df2)) >= 300
    for f in FEATS:
        df2.loc[rest, f] = df2.loc[rest, f] * 7 + 1.0
    df2.loc[rest, "Label"] = "DoS Hulk"
    ev2 = _evaluator(df2)
    after = ev2.fit_objective(FEATS, 8, 3)
    assert before["fit_expected_attacks_admitted"] == after["fit_expected_attacks_admitted"]
    assert before["fit_flagged"] == after["fit_flagged"]


def test_search_path_never_reads_holdout_or_sample_or_pool_outside_fit():
    df = _synthetic(n=900, seed=11)
    ev_a = _evaluator(df, fit_rows=300, hold_rows=(300, 600))
    path_a, _ = greedy_forward(ev_a, FEATS, 8, 3, 3)
    df_b = df.copy()
    rest = np.arange(len(df_b)) >= 300
    rng = np.random.default_rng(99)
    for f in FEATS:
        df_b.loc[rest, f] = rng.gamma(1.0, 50.0, rest.sum())
    df_b.loc[rest, "Label"] = np.where(rng.random(rest.sum()) < 0.5, "PortScan", "BENIGN")
    ev_b = _evaluator(df_b, fit_rows=300, hold_rows=(300, 600))
    path_b, _ = greedy_forward(ev_b, FEATS, 8, 3, 3)
    assert path_a == path_b


def test_greedy_is_deterministic():
    df = _synthetic(n=700, seed=2)
    ev = _evaluator(df)
    p1, s1 = greedy_forward(ev, FEATS, 8, 3, 3)
    p2, s2 = greedy_forward(ev, FEATS, 8, 3, 3)
    assert p1 == p2
    assert s1 == s2


def test_backward_elimination_shrinks_and_stays_nested():
    df = _synthetic(n=700, seed=4)
    ev = _evaluator(df)
    path, _ = backward_eliminate(ev, FEATS, 8, 3, 2)
    sizes = sorted(path)
    assert sizes == [2, 3, 4]
    for s in sizes[:-1]:
        assert set(path[s]) < set(path[s + 1])
