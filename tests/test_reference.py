import numpy as np
import pandas as pd
import pytest

from controlplane.reference import (
    FeatureReference,
    find_nearest_benign_flows,
    reference_from_dataframe,
    render_nearest_neighbour_line,
    render_reference_line,
)


def _synthetic_benign_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "label": ["BENIGN"] * n,
        "first_ts": range(n),
        "last_ts": range(n),
        "closure_reason": ["idle_timeout"] * n,
        "flows_per_src": rng.uniform(0, 100, n),
        # some undefined values, like real bwd_pkt_len_mean for
        # no-backward-packet flows
        "bwd_pkt_len_mean": [None if i % 10 == 0 else float(i) for i in range(n)],
    })


def test_reference_from_dataframe_computes_percentiles_and_counts():
    df = _synthetic_benign_df()
    ref = reference_from_dataframe(df)
    assert set(ref) == {"flows_per_src", "bwd_pkt_len_mean"}
    fps = ref["flows_per_src"]
    assert fps.n == 1000
    assert 0 <= fps.p50 <= 100
    assert fps.p50 <= fps.p90 <= fps.p99 <= fps.p995 <= fps.max


def test_reference_from_dataframe_excludes_undefined_values_from_n():
    df = _synthetic_benign_df()
    ref = reference_from_dataframe(df)
    # 100 of 1000 rows have bwd_pkt_len_mean = None (every 10th)
    assert ref["bwd_pkt_len_mean"].n == 900


def test_reference_from_dataframe_skips_low_confidence_and_metadata_columns():
    df = _synthetic_benign_df()
    df["flows_per_src__low_confidence"] = False
    ref = reference_from_dataframe(df)
    assert "flows_per_src__low_confidence" not in ref
    assert "label" not in ref
    assert "first_ts" not in ref


def test_reference_from_dataframe_skips_feature_with_no_defined_values():
    df = _synthetic_benign_df(n=10)
    df["always_undefined"] = [None] * 10
    ref = reference_from_dataframe(df)
    assert "always_undefined" not in ref


# ---------- render_reference_line ----------

def test_render_reference_line_flags_exceeding_the_max():
    ref = FeatureReference(feature="flows_per_src", p50=226, p90=719, p99=1314, p995=1551, max=1993, n=566864)
    line = render_reference_line("trigger_flows_per_src", 4614.0, ref)
    assert "exceeds every benign flow observed" in line
    assert "n=566864" in line


def test_render_reference_line_flags_exceeding_p995_but_not_max():
    ref = FeatureReference(feature="syn_ratio", p50=0, p90=0.1, p99=0.4, p995=0.45, max=1.0, n=566864)
    line = render_reference_line("trigger_syn_ratio", 0.5, ref)
    assert "99.5th percentile" in line
    assert "exceeds every benign flow" not in line


def test_render_reference_line_no_verdict_when_unremarkable():
    ref = FeatureReference(feature="syn_ratio", p50=0, p90=0.1, p99=0.4, p995=0.45, max=1.0, n=566864)
    line = render_reference_line("trigger_syn_ratio", 0.05, ref)
    assert "->" not in line


def test_render_reference_line_handles_missing_reference():
    line = render_reference_line("trigger_unknown_feature", 5.0, None)
    assert "no benign reference available" in line


# ---------- nearest-neighbour grounding ----------

def test_find_nearest_benign_flows_finds_k_and_computes_distance():
    df = _synthetic_benign_df(n=200)
    ref = reference_from_dataframe(df)
    result = find_nearest_benign_flows({"flows_per_src": 50.0}, df, ref, k=10)
    assert result.k_found == 10
    assert result.nearest_distance is not None
    assert result.nearest_distance >= 0
    assert result.features_used == ["flows_per_src"]


def test_find_nearest_benign_flows_reports_no_result_when_no_overlap():
    df = _synthetic_benign_df(n=50)
    ref = reference_from_dataframe(df)
    result = find_nearest_benign_flows({"totally_unknown_feature": 1.0}, df, ref, k=10)
    assert result.k_found == 0
    assert result.nearest_distance is None


def test_find_nearest_benign_flows_does_not_explode_on_a_sparse_feature():
    # regression: a feature that's 0 for the overwhelming majority of
    # benign flows (p50 == p90 == 0) must not blow up the distance when
    # the escalated flow's own observed value is large and nonzero --
    # confirmed live: this produced distances in the hundreds of billions
    # for real stop-point-2 PortScan records before the (max - p50) fix.
    n = 1000
    sparse = [0.0] * (n - 5) + [1.0, 1.0, 1.0, 1.0, 14.0]  # p50=p90=0, max=14
    df = pd.DataFrame({
        "label": ["BENIGN"] * n,
        "first_ts": range(n), "last_ts": range(n), "closure_reason": ["x"] * n,
        "flows_per_src": [float(i % 100) for i in range(n)],
        "syn_without_synack_count": sparse,
    })
    ref = reference_from_dataframe(df)
    assert ref["syn_without_synack_count"].p50 == 0
    assert ref["syn_without_synack_count"].p90 == 0

    result = find_nearest_benign_flows(
        {"flows_per_src": 50.0, "syn_without_synack_count": 205.0}, df, ref, k=10
    )
    assert result.nearest_distance < 1000  # sane magnitude, not billions


def test_find_nearest_benign_flows_identifies_most_differing_feature():
    # construct a benign population tightly clustered on two features,
    # then query a point far away on one of them.
    df = pd.DataFrame({
        "label": ["BENIGN"] * 100,
        "first_ts": range(100), "last_ts": range(100), "closure_reason": ["x"] * 100,
        "flows_per_src": [10.0] * 100,
        "syn_ratio": [0.1] * 100,
    })
    ref = reference_from_dataframe(df)
    # give syn_ratio a nonzero scale so distance is computable (p90-p50);
    # tight cluster means scale is ~0 -- add tiny spread
    df["syn_ratio"] = [0.1 + i * 1e-6 for i in range(100)]
    ref = reference_from_dataframe(df)
    result = find_nearest_benign_flows({"flows_per_src": 500.0, "syn_ratio": 0.1}, df, ref, k=5)
    assert result.top_differing_features[0][0] == "flows_per_src"


def test_render_nearest_neighbour_line_handles_no_result():
    from controlplane.reference import NearestNeighbourResult

    result = NearestNeighbourResult(
        features_used=[], k_requested=10, k_found=0,
        nearest_distance=None, mean_k_distance=None, top_differing_features=[],
    )
    line = render_nearest_neighbour_line(result)
    assert "no benign reference available" in line
