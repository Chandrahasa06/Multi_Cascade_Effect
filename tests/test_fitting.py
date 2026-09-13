import numpy as np
import pandas as pd
import pytest

from dataplane.fitting import (
    BENIGN_LABEL,
    FeatureKind,
    feature_kind,
    fit_thresholds,
    fit_thresholds_from_frame,
    load_thresholds,
    save_thresholds,
)
from dataplane.flow_state import FlowState, Packet
from dataplane.src_table import SrcTable


def make_packet(**overrides):
    defaults = dict(
        src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1234, dst_port=80,
        protocol=6, timestamp_us=0, length=100,
    )
    defaults.update(overrides)
    return Packet(**defaults)


def two_way_flow(fwd_len=100, bwd_len=200, gap_us=1000, src_ip="10.0.0.1", resolution=1):
    state = FlowState.start(
        make_packet(src_ip=src_ip, timestamp_us=0, length=fwd_len, timestamp_resolution_us=resolution)
    )
    reply = make_packet(
        src_ip="10.0.0.2", dst_ip=src_ip, src_port=80, dst_port=1234,
        timestamp_us=gap_us, length=bwd_len, timestamp_resolution_us=resolution,
    )
    state.update(reply)
    return state


def single_packet_flow(src_ip="10.0.0.1"):
    return FlowState.start(make_packet(src_ip=src_ip, timestamp_us=0))


class TestBenignOnlyAssertion:
    def test_rejects_any_non_benign_label(self):
        flows = [two_way_flow(), two_way_flow()]
        labels = [BENIGN_LABEL, "DoS Hulk"]
        with pytest.raises(AssertionError):
            fit_thresholds(flows, labels)

    def test_assertion_fires_before_length_mismatch_is_checked(self):
        # one attack label AND a length mismatch: must fail on the label
        # check (AssertionError), not the length check (ValueError) —
        # proves the safety check really does run first.
        flows = [two_way_flow()]
        labels = [BENIGN_LABEL, "PortScan"]
        with pytest.raises(AssertionError):
            fit_thresholds(flows, labels)

    def test_accepts_all_benign_labels(self):
        flows = [two_way_flow() for _ in range(5)]
        labels = [BENIGN_LABEL] * 5
        result = fit_thresholds(flows, labels)
        assert result.config.thresholds  # fit something

    def test_mismatched_lengths_raises_value_error_when_labels_are_fine(self):
        flows = [two_way_flow(), two_way_flow()]
        labels = [BENIGN_LABEL]
        with pytest.raises(ValueError):
            fit_thresholds(flows, labels)


class TestUndefinedValueExclusion:
    def test_none_values_excluded_not_treated_as_zero(self):
        # 8 multi-packet flows (iat_mean defined) + 2 single-packet flows
        # (iat_mean undefined) — the undefined ones must not drag the
        # percentile toward zero.
        flows = [two_way_flow(gap_us=1000) for _ in range(8)] + [
            single_packet_flow() for _ in range(2)
        ]
        labels = [BENIGN_LABEL] * 10
        result = fit_thresholds(flows, labels, percentile=50)
        report = result.reports["flow_iat_mean"]
        assert report.total_flows == 10
        assert report.excluded_undefined == 2
        assert report.used == 8
        # median of eight identical 1000us gaps is exactly 1000, not
        # dragged toward 0 by the two undefined flows.
        assert result.config.thresholds["flow_iat_mean"].high == 1000.0

    def test_feature_undefined_for_all_flows_is_omitted_not_zeroed(self):
        flows = [single_packet_flow() for _ in range(5)]
        labels = [BENIGN_LABEL] * 5
        result = fit_thresholds(flows, labels)
        assert "flow_iat_mean" not in result.config.thresholds
        assert result.reports["flow_iat_mean"].used == 0
        assert result.reports["flow_iat_mean"].excluded_undefined == 5

    def test_report_is_a_finding_not_silently_dropped(self):
        flows = [single_packet_flow() for _ in range(9)] + [two_way_flow()]
        labels = [BENIGN_LABEL] * 10
        result = fit_thresholds(flows, labels)
        report = result.reports["flow_iat_mean"]
        assert report.excluded_fraction == 0.9  # 90% of benign flows can't inform this feature


class TestLowConfidenceExclusion:
    def test_low_confidence_excluded_by_default(self):
        precise = [two_way_flow(gap_us=1000, resolution=1) for _ in range(8)]
        coarse = [two_way_flow(gap_us=999_999_999, resolution=60_000_000) for _ in range(2)]
        result = fit_thresholds(precise + coarse, [BENIGN_LABEL] * 10, percentile=50)
        report = result.reports["flow_duration"]
        assert report.excluded_low_confidence == 2
        assert report.used == 8
        assert result.config.thresholds["flow_duration"].high == 1000.0

    def test_low_confidence_included_when_disabled(self):
        precise = [two_way_flow(gap_us=1000, resolution=1) for _ in range(2)]
        coarse = [two_way_flow(gap_us=5000, resolution=60_000_000) for _ in range(2)]
        result = fit_thresholds(
            precise + coarse, [BENIGN_LABEL] * 4, exclude_low_confidence=False, percentile=50
        )
        assert result.reports["flow_duration"].excluded_low_confidence == 0
        assert result.reports["flow_duration"].used == 4


class TestTwoSidedFeatures:
    def test_two_sided_feature_gets_low_threshold(self):
        flows = [two_way_flow(fwd_len=v, bwd_len=200) for v in range(50, 150)]
        result = fit_thresholds(flows, [BENIGN_LABEL] * len(flows))
        threshold = result.config.thresholds["fwd_pkt_len_mean"]
        assert threshold.low is not None
        assert threshold.high is not None
        assert threshold.low < threshold.high

    def test_single_sided_feature_has_no_low_threshold(self):
        flows = [two_way_flow(gap_us=g) for g in range(1000, 1100)]
        result = fit_thresholds(flows, [BENIGN_LABEL] * len(flows))
        assert result.config.thresholds["flow_bytes_per_sec"].low is None


class TestSrcFeatures:
    def test_src_features_fitted_when_src_table_given(self):
        src_table = SrcTable()
        flows = []
        for i in range(5):
            src_ip = f"10.0.0.{i}"
            for port in range(i + 1):
                src_table.note_flow_start(src_ip, "10.0.0.9", port, 0)
            flows.append(two_way_flow(src_ip=src_ip))
        result = fit_thresholds(flows, [BENIGN_LABEL] * 5, src_table=src_table, now_us_by_flow=[0] * 5)
        assert "distinct_dst_ports_per_src" in result.config.thresholds
        assert "flows_per_src" in result.config.thresholds

    def test_no_src_features_without_src_table(self):
        flows = [two_way_flow() for _ in range(3)]
        result = fit_thresholds(flows, [BENIGN_LABEL] * 3)
        assert "distinct_dst_ports_per_src" not in result.config.thresholds


class TestEmptyInput:
    def test_no_flows_produces_no_thresholds(self):
        result = fit_thresholds([], [])
        assert result.config.thresholds == {}
        assert result.reports == {}


class TestConfigHash:
    def test_same_thresholds_same_hash(self):
        flows = [two_way_flow() for _ in range(5)]
        labels = [BENIGN_LABEL] * 5
        a = fit_thresholds(flows, labels)
        b = fit_thresholds(flows, labels)
        assert a.config.config_hash == b.config.config_hash

    def test_different_thresholds_different_hash(self):
        flows_a = [two_way_flow(fwd_len=100) for _ in range(5)]
        flows_b = [two_way_flow(fwd_len=999) for _ in range(5)]
        a = fit_thresholds(flows_a, [BENIGN_LABEL] * 5)
        b = fit_thresholds(flows_b, [BENIGN_LABEL] * 5)
        assert a.config.config_hash != b.config.config_hash


class TestPercentileCorrectness:
    def test_matches_numpy_reference_directly(self):
        flows = [two_way_flow(fwd_len=v, bwd_len=1000) for v in range(1, 201)]
        result = fit_thresholds(flows, [BENIGN_LABEL] * 200, percentile=99.5)
        expected = float(np.percentile(list(range(1, 201)), 99.5))
        assert result.config.thresholds["fwd_pkt_len_mean"].high == expected


class TestFitThresholdsFromFrame:
    def make_frame(self, n=10, undefined=0, low_conf=0):
        values = list(range(1, n + 1))
        df = pd.DataFrame(
            {
                "label": [BENIGN_LABEL] * n,
                "flow_bytes_per_sec": values,
                "flow_bytes_per_sec__low_confidence": [False] * n,
            }
        )
        for i in range(undefined):
            df.loc[i, "flow_bytes_per_sec"] = np.nan
        for i in range(undefined, undefined + low_conf):
            df.loc[i, "flow_bytes_per_sec__low_confidence"] = True
        return df

    def test_asserts_on_non_benign_label(self):
        df = self.make_frame()
        df.loc[0, "label"] = "DoS Hulk"
        with pytest.raises(AssertionError):
            fit_thresholds_from_frame(df)

    def test_matches_flowstate_based_fitting_on_equivalent_data(self):
        df = self.make_frame(n=200)
        result = fit_thresholds_from_frame(df, percentile=99.5)
        expected = float(np.percentile(list(range(1, 201)), 99.5))
        assert result.config.thresholds["flow_bytes_per_sec"].high == expected

    def test_excludes_undefined_values(self):
        df = self.make_frame(n=10, undefined=3)
        result = fit_thresholds_from_frame(df, percentile=50)
        report = result.reports["flow_bytes_per_sec"]
        assert report.excluded_undefined == 3
        assert report.used == 7

    def test_excludes_low_confidence_by_default(self):
        df = self.make_frame(n=10, low_conf=4)
        result = fit_thresholds_from_frame(df, percentile=50)
        assert result.reports["flow_bytes_per_sec"].excluded_low_confidence == 4

    def test_feature_names_restricts_fitting(self):
        df = self.make_frame()
        df["other_feature"] = range(len(df))
        result = fit_thresholds_from_frame(df, feature_names=["flow_bytes_per_sec"])
        assert "flow_bytes_per_sec" in result.config.thresholds
        assert "other_feature" not in result.config.thresholds
        assert "other_feature" not in result.reports

    def test_missing_low_confidence_column_treated_as_all_confident(self):
        df = pd.DataFrame({"label": [BENIGN_LABEL] * 5, "some_feature": [1, 2, 3, 4, 5]})
        result = fit_thresholds_from_frame(df, feature_names=["some_feature"], percentile=50)
        assert result.reports["some_feature"].excluded_low_confidence == 0
        assert result.reports["some_feature"].used == 5


class TestFeatureKindClassification:
    def test_known_bounded_features(self):
        assert feature_kind("no_response_flag") == FeatureKind.BOOLEAN
        assert feature_kind("syn_ratio") == FeatureKind.BOUNDED_RATIO
        assert feature_kind("rst_ratio") == FeatureKind.BOUNDED_RATIO

    def test_unknown_feature_defaults_continuous(self):
        assert feature_kind("flow_duration") == FeatureKind.CONTINUOUS
        assert feature_kind("some_new_feature") == FeatureKind.CONTINUOUS


class TestBoundedFeatureFitting:
    def make_frame(self, n, feature_values, name="no_response_flag"):
        return pd.DataFrame(
            {
                "label": [BENIGN_LABEL] * n,
                name: feature_values,
                f"{name}__low_confidence": [False] * n,
            }
        )

    def test_boolean_rare_true_is_already_fittable_without_saturating(self):
        # 1 in 500 (0.2%) benign flows has no_response_flag=True: rarer
        # than the p99.5 budget (0.5%), so the percentile itself lands at
        # 0.0 (the majority value), not at the 1.0 bound — ordinary `>`
        # fitting already separates the rare True from the common False
        # without needing any special-casing.
        values = [0.0] * 499 + [1.0]
        df = self.make_frame(len(values), values)
        result = fit_thresholds_from_frame(df, feature_names=["no_response_flag"], percentile=99.5)
        threshold = result.config.thresholds["no_response_flag"]
        assert threshold.high == 0.0  # fires on value=1.0 via plain `> 0.0`
        report = result.reports["no_response_flag"]
        assert report.kind == "boolean"
        assert report.saturated is False

    def test_boolean_common_true_is_unfittable_not_uncrossable(self):
        # old bug: p99.5 of a mostly-True boolean is 1.0, and `value > 1.0`
        # can never be true — silently dead forever. New behavior: report
        # it as saturated and omit the threshold rather than fit a dead one.
        values = [1.0] * 20 + [0.0] * 80  # 20% True: far above any p99.x budget
        df = self.make_frame(len(values), values)
        result = fit_thresholds_from_frame(df, feature_names=["no_response_flag"], percentile=99.5)
        assert "no_response_flag" not in result.config.thresholds
        report = result.reports["no_response_flag"]
        assert report.saturated is True
        assert report.used == 100  # unfittable due to saturation, not lack of data

    def test_boolean_all_false_still_fits_a_useful_threshold(self):
        # a feature that's constant in benign data isn't the saturation
        # pathology — high=0.0 correctly fires on any nonzero (attack)
        # value, same as ordinary percentile fitting on a constant column.
        values = [0.0] * 100
        df = self.make_frame(len(values), values)
        result = fit_thresholds_from_frame(df, feature_names=["no_response_flag"], percentile=99.5)
        assert result.config.thresholds["no_response_flag"].high == 0.0
        assert result.reports["no_response_flag"].saturated is False

    def test_bounded_ratio_saturating_at_one_is_unfittable(self):
        values = [0.1] * 80 + [1.0] * 20  # 20% at the bound: far above any p99.x budget
        df = self.make_frame(len(values), values, name="syn_ratio")
        result = fit_thresholds_from_frame(df, feature_names=["syn_ratio"], percentile=99.5)
        assert "syn_ratio" not in result.config.thresholds
        report = result.reports["syn_ratio"]
        assert report.kind == "bounded_ratio"
        assert report.saturated is True

    def test_bounded_ratio_interior_percentile_behaves_like_continuous(self):
        # no saturation: p99.5 lands well inside (0, 1), ordinary fitting.
        values = list(np.linspace(0.0, 0.5, 200))
        df = self.make_frame(len(values), values, name="syn_ratio")
        result = fit_thresholds_from_frame(df, feature_names=["syn_ratio"], percentile=50.0)
        report = result.reports["syn_ratio"]
        assert report.saturated is False
        expected = float(np.percentile(values, 50.0))
        assert result.config.thresholds["syn_ratio"].high == pytest.approx(expected)

    def test_continuous_feature_never_flagged_saturated(self):
        df = pd.DataFrame(
            {
                "label": [BENIGN_LABEL] * 5,
                "flow_duration": [1.0, 2.0, 3.0, 4.0, 5.0],
                "flow_duration__low_confidence": [False] * 5,
            }
        )
        result = fit_thresholds_from_frame(df, feature_names=["flow_duration"], percentile=80.0)
        assert result.reports["flow_duration"].kind == "continuous"
        assert result.reports["flow_duration"].saturated is False


class TestJsonRoundTrip:
    def test_save_and_load_thresholds(self, tmp_path):
        flows = [two_way_flow() for _ in range(5)]
        result = fit_thresholds(flows, [BENIGN_LABEL] * 5)
        path = tmp_path / "thresholds.json"
        save_thresholds(result.config, path)
        restored = load_thresholds(path)
        assert restored == result.config
