"""Tests for the out-of-period evaluation and the change-1 no-op claim."""
import numpy as np
import pandas as pd

from dataplane.escalation_policy import PRIORITY2_FEATURES, build_signature_table, priority2_escalate
from eval.p2_feature_search import build_fit_table, escalate, signature_space
from eval.p2_out_of_period import (
    ORIGINAL_FIVE,
    chronological_cap,
    configurations,
    evaluate_unit,
    out_of_period_mask,
)


def test_change1_removed_syn_from_production_constant():
    assert "syn_without_synack_count" not in PRIORITY2_FEATURES
    assert "syn_without_synack_count" in ORIGINAL_FIVE
    assert tuple(PRIORITY2_FEATURES) == tuple(f for f in ORIGINAL_FIVE if f != "syn_without_synack_count")


def test_change1_is_a_noop_for_K_escalation_and_bins():
    """A count feature that is zero for most benign rows has quantile edges
    collapsing to [0]; every value then lands in one bin, so dropping it
    cannot change which signatures are common or which flows escalate."""
    rng = np.random.default_rng(0)
    n = 3000
    syn = np.where(rng.random(n) < 0.98, 0.0, rng.integers(1, 5, n).astype(float))
    other = rng.integers(0, 30, n).astype(float)
    benign = pd.DataFrame({"syn_without_synack_count": syn, "flows_per_src": other})
    with_syn = ["flows_per_src", "syn_without_synack_count"]
    without = ["flows_per_src"]
    vals = {f: benign[f].to_numpy(float) for f in with_syn}
    t_with = build_fit_table(vals, with_syn, 20, 5)
    t_without = build_fit_table({"flows_per_src": vals["flows_per_src"]}, without, 20, 5)
    assert t_with.edges["syn_without_synack_count"].tolist() == [0.0]
    assert len(t_with.common) == len(t_without.common)
    # every row lands in the same syn bin, so escalation is identical
    assert (escalate(vals, t_with) == escalate({"flows_per_src": vals["flows_per_src"]}, t_without)).all()
    # the space shrinks by exactly the syn bin count
    assert signature_space(with_syn, t_with.edges) == 2 * signature_space(without, t_without.edges)


def test_change1_noop_matches_production_table_on_synthetic_rows():
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame({f: rng.integers(0, 25, n).astype(float) for f in ORIGINAL_FIVE})
    df["syn_without_synack_count"] = np.where(rng.random(n) < 0.9, 0.0, 1.0)
    df["Label"] = "Benign"
    five = build_signature_table(df, n_bins=20, floor=5, features=list(ORIGINAL_FIVE))
    four = build_signature_table(df, n_bins=20, floor=5, features=list(configurations()["C1_four_minus_syn"]))
    assert len(five.common_signatures) == len(four.common_signatures)
    assert (priority2_escalate(df, five) == priority2_escalate(df, four)).all()


def test_out_of_period_mask_excludes_fit_days():
    wd = np.array([0, 0, 1, 1, 2, 3, 4, 4])
    m = out_of_period_mask(wd)
    assert m.tolist() == [False, False, False, False, True, True, True, True]
    assert not m[wd <= 1].any()


def test_configurations_are_the_pre_registered_set():
    c = configurations()
    assert list(c) == ["R_five_production", "C1_four_minus_syn", "C2_minus_pkt_len_range", "C3_minus_bwd_pkt_len_mean"]
    assert "pkt_len_range" not in c["C2_minus_pkt_len_range"]
    assert "bwd_pkt_len_mean" not in c["C3_minus_bwd_pkt_len_mean"]
    assert "pkt_len_range" in c["C3_minus_bwd_pkt_len_mean"]
    assert "bwd_pkt_len_mean" in c["C2_minus_pkt_len_range"]


def test_chronological_cap_admits_earliest_flagged_and_reports_fill():
    rows = np.arange(10)
    t = np.array([5, 1, 9, 2, 7, 3, 8, 0, 6, 4], dtype=float)
    row_idx = np.arange(10)
    eligible = np.ones(10, dtype=bool)
    flagged = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
    is_attack = np.array([0, 1, 0, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    out = chronological_cap(rows, t, row_idx, eligible, flagged, is_attack)
    # 7 flagged rows, far below the 2,400 cap: all are admitted and fill is below 1
    assert out["flagged_total"] == int(flagged.sum())
    assert out["admitted"] == out["flagged_total"]
    assert out["fill_rate"] == out["admitted"] / 2400
    assert out["fill_rate"] < 1.0


def test_filled_cap_uses_full_2400_only_when_enough_flagged():
    n = 5000
    rows = np.arange(n)
    rng = np.random.default_rng(2)
    t = rng.random(n)
    row_idx = np.arange(n)
    eligible = np.ones(n, dtype=bool)
    flagged = np.ones(n, dtype=bool)
    is_attack = rng.random(n) < 0.1
    out = chronological_cap(rows, t, row_idx, eligible, flagged, is_attack)
    assert out["admitted"] == 2400
    assert out["fill_rate"] == 1.0


def test_evaluate_unit_counts_benign_and_classes_on_the_unit_only():
    labels = np.array(["BENIGN", "BENIGN", "Bot", "Bot", "BENIGN", "PortScan"], dtype=object)
    is_attack = labels != "BENIGN"
    flagged = np.array([1, 0, 1, 0, 1, 1], dtype=bool)
    rows = np.array([0, 1, 2, 3])  # the unit: excludes rows 4 and 5
    t = np.arange(6, dtype=float)
    out = evaluate_unit(rows, flagged, labels, is_attack, t, np.arange(6), np.ones(6, dtype=bool))
    assert out["benign_n"] == 2 and out["benign_escalated"] == 1
    assert out["recall__Bot__n"] == 2 and out["recall__Bot__caught"] == 1
    assert "recall__PortScan__n" not in out  # PortScan row is outside the unit
