"""Tests for agents/grounding_v2.py: transform monotone and point-mass safe, k-NN distance
always defined, calibration built from benign rows only, no label reaches the signal."""
import numpy as np
import pandas as pd
import pytest

from agents import grounding_v2 as g2
from agents.escalation_grounding import ESCALATION_FEATURES


def _benign_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    data = {f: rng.lognormal(mean=0.0, sigma=1.0, size=n) for f in ESCALATION_FEATURES}
    # point mass: a third of the rows are exactly zero on one feature
    data["Init_Win_bytes_forward"][: n // 3] = 0.0
    data["Label"] = "BENIGN"
    return pd.DataFrame(data)


def test_midrank_is_monotone_non_decreasing():
    col = np.sort(np.random.default_rng(1).normal(size=500))
    xs = np.linspace(col.min() - 1, col.max() + 1, 2000)
    p = g2.midrank_percentiles(col, xs)
    assert np.all(np.diff(p) >= 0)
    assert p.min() >= 0 and p.max() <= 1


def test_midrank_handles_point_mass_at_its_middle():
    col = np.sort(np.array([0.0] * 40 + [1.0] * 10 + [2.0] * 50))
    mass_zero = g2.midrank_percentiles(col, np.array([0.0]))[0]
    assert abs(mass_zero - 20 / 100) < 1e-12  # midpoint of the zero mass, not its top (0.4)
    assert g2.midrank_percentiles(col, np.array([1.0]))[0] > mass_zero


def test_knn_distance_is_always_defined_leave_one_out():
    ref = g2.build_reference_v2(_benign_frame(), label="test")
    for k in (10, 50, 100):
        d = g2.knn_distance_loo(ref, ref.pct[:20], k)
        assert np.isfinite(d).all()
        assert (d >= 0).all()


def test_knn_distance_refuses_k_at_or_above_reference_size():
    ref = g2.build_reference_v2(_benign_frame(n=200), label="small")
    with pytest.raises(ValueError):
        g2.knn_distance_loo(ref, ref.pct[:1], ref.n)


def test_leave_one_out_removes_the_flows_own_copy():
    ref = g2.build_reference_v2(_benign_frame(), label="test")
    own = ref.pct[5:6]
    # the flow is in the reference, so its own copy is at distance 0; LOO must not return 0 for k>=1
    d = g2.knn_distance_loo(ref, own, 10)[0]
    assert d > 0


def test_calibration_is_built_from_benign_rows_only():
    frame = _benign_frame()
    frame.loc[0, "Label"] = "DoS Hulk"
    with pytest.raises(ValueError, match="benign"):
        g2.build_reference_v2(frame, label="bad")


def test_calibration_rows_index_the_reference():
    ref = g2.build_reference_v2(_benign_frame(n=300), label="test")
    assert ref.calibration_rows.max() < ref.n
    assert len(set(ref.calibration_rows.tolist())) == len(ref.calibration_rows)


def test_score_ignores_a_label_key():
    ref = g2.build_reference_v2(_benign_frame(), label="test")
    feats = {f: 1.0 for f in ESCALATION_FEATURES}
    plain = g2.score_flow(ref, feats)
    labelled = g2.score_flow(ref, dict(feats, Label="DoS Hulk", is_attack=True))
    assert plain["isolation_percentile"] == labelled["isolation_percentile"]
    assert plain["radius_count"] == labelled["radius_count"]


def test_isolation_percentile_is_bounded():
    ref = g2.build_reference_v2(_benign_frame(), label="test")
    for v in (0.0, 1.0, 1e9):
        s = g2.score_flow(ref, {f: v for f in ESCALATION_FEATURES})
        assert 0.0 <= s["isolation_percentile"] <= 100.0


def test_per_feature_contributions_cover_every_feature():
    ref = g2.build_reference_v2(_benign_frame(), label="test")
    s = g2.score_flow(ref, {f: 2.0 for f in ESCALATION_FEATURES})
    assert {r["feature"] for r in s["per_feature"]} == set(ESCALATION_FEATURES)
