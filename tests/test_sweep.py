import numpy as np
import pandas as pd

from dataplane.fitting import BENIGN_LABEL
from dataplane.selector import EscalationRule, FeatureThreshold
from eval.sweep import (
    all_feature_names,
    compute_crossings,
    escalate_from_crossings,
    nearest_point,
    per_class_recall,
    run_sweep,
    split_monday_chronologically,
    tier1_feature_names,
    trigger_frequency,
)


def make_feature_frame(n, label=BENIGN_LABEL, feature_a=None, feature_b=None, first_ts=None):
    return pd.DataFrame(
        {
            "label": [label] * n,
            "mixed_label": [False] * n,
            "residual_ambiguous": [False] * n,
            "n_source_rows": [1] * n,
            "first_ts": first_ts if first_ts is not None else list(range(n)),
            "last_ts": list(range(n)),
            "closure_reason": ["fin"] * n,
            "feature_a": feature_a if feature_a is not None else list(range(n)),
            "feature_a__low_confidence": [False] * n,
            "feature_b": feature_b if feature_b is not None else [0.0] * n,
            "feature_b__low_confidence": [False] * n,
            "flows_per_src": [1] * n,
            "flows_per_src__low_confidence": [False] * n,
            "distinct_dst_ports_per_src": [1] * n,
            "distinct_dst_ports_per_src__low_confidence": [False] * n,
            "distinct_dst_ips_per_src": [1] * n,
            "distinct_dst_ips_per_src__low_confidence": [False] * n,
            "syn_without_synack_count": [0] * n,
            "syn_without_synack_count__low_confidence": [False] * n,
        }
    )


class TestSplitMonday:
    def test_splits_by_time_not_row_position(self):
        df = make_feature_frame(10, first_ts=[9, 8, 7, 6, 5, 4, 3, 2, 1, 0])  # reverse order
        fit_half, holdout_half = split_monday_chronologically(df)
        assert len(fit_half) == 5
        assert len(holdout_half) == 5
        assert fit_half["first_ts"].max() < holdout_half["first_ts"].min()

    def test_odd_length_splits_reasonably(self):
        df = make_feature_frame(7)
        fit_half, holdout_half = split_monday_chronologically(df)
        assert len(fit_half) + len(holdout_half) == 7
        assert len(fit_half) == 3


class TestFeatureNameDiscovery:
    def test_tier1_excludes_src_features(self):
        df = make_feature_frame(5)
        names = tier1_feature_names(df)
        assert "feature_a" in names
        assert "flows_per_src" not in names

    def test_all_includes_src_features(self):
        df = make_feature_frame(5)
        names = all_feature_names(df)
        assert "flows_per_src" in names
        assert "feature_a" in names

    def test_excludes_metadata_and_low_confidence_columns(self):
        df = make_feature_frame(5)
        names = all_feature_names(df)
        assert "label" not in names
        assert "first_ts" not in names
        assert "feature_a__low_confidence" not in names


class TestComputeCrossings:
    def test_high_threshold(self):
        df = make_feature_frame(4, feature_a=[1, 100, 5, 200])
        crossings = compute_crossings(df, {"feature_a": FeatureThreshold(high=50)})
        assert list(crossings["feature_a"]) == [False, True, False, True]

    def test_low_threshold(self):
        df = make_feature_frame(4, feature_a=[1, 100, 5, 200])
        crossings = compute_crossings(df, {"feature_a": FeatureThreshold(low=3)})
        assert list(crossings["feature_a"]) == [True, False, False, False]

    def test_nan_never_crosses(self):
        df = make_feature_frame(3, feature_a=[np.nan, 100, np.nan])
        crossings = compute_crossings(df, {"feature_a": FeatureThreshold(high=1, low=1)})
        assert list(crossings["feature_a"]) == [False, True, False]

    def test_missing_column_skipped(self):
        df = make_feature_frame(3)
        crossings = compute_crossings(df, {"nonexistent": FeatureThreshold(high=1)})
        assert "nonexistent" not in crossings.columns


class TestEscalateFromCrossings:
    def test_any_rule(self):
        crossings = pd.DataFrame({"a": [True, False], "b": [False, False]})
        result = escalate_from_crossings(crossings, EscalationRule.ANY, k=1)
        assert list(result) == [True, False]

    def test_k_of_n(self):
        crossings = pd.DataFrame({"a": [True, True], "b": [True, False], "c": [False, False]})
        result = escalate_from_crossings(crossings, EscalationRule.K_OF_N, k=2)
        assert list(result) == [True, False]

    def test_no_thresholds_never_escalates(self):
        crossings = pd.DataFrame(index=range(3))
        result = escalate_from_crossings(crossings, EscalationRule.ANY, k=1)
        assert list(result) == [False, False, False]


class TestPerClassRecall:
    def test_excludes_benign_and_computes_recall(self):
        df = pd.DataFrame({"label": ["BENIGN", "BENIGN", "DoS", "DoS", "PortScan"]})
        escalated = pd.Series([True, False, True, False, True])
        result = per_class_recall(df, escalated).set_index("label")
        assert result.loc["DoS", "escalated_count"] == 1
        assert result.loc["DoS", "total_count"] == 2
        assert result.loc["DoS", "recall"] == 0.5
        assert result.loc["PortScan", "recall"] == 1.0
        assert "BENIGN" not in result.index


class TestNearestPoint:
    def test_finds_closest_by_escalation_rate(self):
        from eval.sweep import SweepPoint, SweepResult

        result = SweepResult(
            points=[
                SweepPoint(90.0, 0.20, 0.20, pd.DataFrame(), 0.0, 0.0, 1),
                SweepPoint(95.0, 0.05, 0.05, pd.DataFrame(), 0.0, 0.0, 1),
                SweepPoint(99.0, 0.01, 0.01, pd.DataFrame(), 0.0, 0.0, 1),
            ]
        )
        point = nearest_point(result, 0.06)
        assert point.percentile == 95.0


class TestTriggerFrequency:
    def test_crossing_rate_and_trigger_share(self):
        df = make_feature_frame(4)
        crossings = pd.DataFrame(
            {"a": [True, True, False, False], "b": [True, False, False, False]}
        )
        escalated = pd.Series([True, True, False, False])
        freq = trigger_frequency(df, crossings, escalated).set_index("feature")
        assert freq.loc["a", "crossing_rate"] == 0.5
        assert freq.loc["a", "trigger_share"] == 1.0  # a crossed on both escalated flows
        assert freq.loc["b", "trigger_share"] == 0.5  # b crossed on only one of the two


class TestRunSweepEndToEnd:
    def test_higher_percentile_gives_lower_or_equal_escalation_rate(self):
        rng = np.random.default_rng(0)
        n = 500
        fit_half = make_feature_frame(n, feature_a=rng.normal(0, 1, n).tolist())
        holdout_half = make_feature_frame(n, feature_a=rng.normal(0, 1, n).tolist())
        week_df = make_feature_frame(n, feature_a=rng.normal(0, 1, n).tolist())

        result, _ = run_sweep(
            fit_half, holdout_half, week_df, percentiles=[90.0, 95.0, 99.0, 99.9]
        )
        rates = [p.overall_escalation_rate for p in result.points]
        assert rates == sorted(rates, reverse=True)

    def test_per_class_recall_present_per_point(self):
        n = 50
        fit_half = make_feature_frame(n)
        holdout_half = make_feature_frame(n)
        week_df = pd.concat(
            [make_feature_frame(20, label=BENIGN_LABEL), make_feature_frame(10, label="DoS")],
            ignore_index=True,
        )
        result, _ = run_sweep(fit_half, holdout_half, week_df, percentiles=[99.0])
        assert "DoS" in result.points[0].per_class_recall["label"].values
