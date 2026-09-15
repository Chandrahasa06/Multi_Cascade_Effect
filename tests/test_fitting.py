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
        # boolean's value space is closed to {0, 1} -- a rarity fit built
        # from training data could never flag anything a boolean couldn't
        # already take, so it's always excluded on saturation, never
        # rarity-fit (see FeatureKind.BOOLEAN's docstring).
        assert report.action == "excluded"

    def test_boolean_all_false_still_fits_a_useful_threshold(self):
        # a feature that's constant in benign data isn't the saturation
        # pathology — high=0.0 correctly fires on any nonzero (attack)
        # value, same as ordinary percentile fitting on a constant column.
        values = [0.0] * 100
        df = self.make_frame(len(values), values)
        result = fit_thresholds_from_frame(df, feature_names=["no_response_flag"], percentile=99.5)
        assert result.config.thresholds["no_response_flag"].high == 0.0
        assert result.reports["no_response_flag"].saturated is False

    def test_bounded_ratio_saturating_at_one_gets_rarity_fit_not_excluded(self):
        # 20% at the bound: far above any p99.x budget, so ordinary `>`
        # fitting is unreachable (same as before). Unlike a boolean,
        # syn_ratio isn't closed to exactly {0.1, 1.0} in principle — a
        # third value is possible in held-out/attack data — so with only
        # 2 distinct training values it's eligible for a rarity fit
        # rather than being dropped outright.
        values = [0.1] * 80 + [1.0] * 20
        df = self.make_frame(len(values), values, name="syn_ratio")
        result = fit_thresholds_from_frame(df, feature_names=["syn_ratio"], percentile=99.5)
        report = result.reports["syn_ratio"]
        assert report.kind == "bounded_ratio"
        assert report.saturated is True
        assert report.action == "rarity_fit"
        threshold = result.config.thresholds["syn_ratio"]
        assert threshold.common_values == frozenset({0.1, 1.0})
        assert report.rarity_coverage == pytest.approx(1.0)  # both values clear budget: degenerate but reported, not hidden

    def test_bounded_ratio_interior_percentile_behaves_like_continuous(self):
        # no saturation: p99.5 lands well inside (0, 1), ordinary fitting.
        values = list(np.linspace(0.0, 0.5, 200))
        df = self.make_frame(len(values), values, name="syn_ratio")
        result = fit_thresholds_from_frame(df, feature_names=["syn_ratio"], percentile=50.0)
        report = result.reports["syn_ratio"]
        assert report.saturated is False
        expected = float(np.percentile(values, 50.0))
        assert result.config.thresholds["syn_ratio"].high == pytest.approx(expected)

    def test_continuous_feature_with_no_tied_extreme_is_not_saturated(self):
        # continuous features go through the SAME saturation check as
        # bounded ones now (see FeatureKind's docstring) -- this isn't
        # "continuous is exempt," it's "this particular data has no tied
        # mass at its own extreme to saturate on" (5 evenly-spread
        # distinct values, p80 lands at 4.2, not at the max of 5.0).
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
        assert result.reports["flow_duration"].action == "kept"


class TestSaturationRegressions:
    """One test per real saturation instance hit in this project, per the
    generic fix's own spec: percentile fitting silently breaking whenever
    a feature has enough tied mass at its own extreme, regardless of
    whether that extreme is a hard theoretical bound (boolean/ratio) or
    just where a particular day's/adapter's data happens to pile up
    (an unbounded count feature)."""

    def make_frame(self, n, feature_values, name):
        return pd.DataFrame(
            {
                "label": [BENIGN_LABEL] * n,
                name: feature_values,
                f"{name}__low_confidence": [False] * n,
            }
        )

    def test_case_1_no_response_flag_boolean_excluded(self):
        # a pure-SYN port scan or unanswered-SYN probe pins this at its
        # bound (True) often enough to saturate -- percentile fitting
        # alone would have produced an uncrossable `> 1.0`.
        values = [1.0] * 15 + [0.0] * 85  # 15% True, over the p99.5 budget
        df = self.make_frame(len(values), values, "no_response_flag")
        result = fit_thresholds_from_frame(df, feature_names=["no_response_flag"], percentile=99.5)
        report = result.reports["no_response_flag"]
        assert report.saturated is True
        assert report.action == "excluded"
        assert "no_response_flag" not in result.config.thresholds

    def test_case_2_rst_ratio_bounded_gets_rarity_fit(self):
        # bounded [0, 1], benign mass piling at the bound the same way
        # syn_ratio's does -- unlike a boolean, a third value is possible
        # in principle, so with few distinct training values it's
        # rarity-fit rather than dropped.
        values = [0.2] * 70 + [0.6] * 20 + [1.0] * 10  # 10% at the bound
        df = self.make_frame(len(values), values, "rst_ratio")
        result = fit_thresholds_from_frame(df, feature_names=["rst_ratio"], percentile=99.5)
        report = result.reports["rst_ratio"]
        assert report.kind == "bounded_ratio"
        assert report.saturated is True
        assert report.action == "rarity_fit"
        assert result.config.thresholds["rst_ratio"].common_values is not None

    def test_case_3_syn_ratio_saturates_above_p99_5(self):
        # matches the real production shape: syn_ratio stays fittable at
        # p99.5 (mass at 1.0 is under budget there) but saturates once
        # the percentile tightens past it (same fixed mass, smaller
        # budget) -- the SAME feature, same data, crossing from "kept" to
        # "saturated" purely because the operating point tightened.
        values = [0.3] * 991 + [1.0] * 9  # 0.9% at the bound
        df = self.make_frame(len(values), values, "syn_ratio")

        loose = fit_thresholds_from_frame(df, feature_names=["syn_ratio"], percentile=99.0)  # budget 1.0% > 0.9%
        assert loose.reports["syn_ratio"].saturated is False
        assert loose.config.thresholds["syn_ratio"].high == pytest.approx(0.3)

        tight = fit_thresholds_from_frame(df, feature_names=["syn_ratio"], percentile=99.5)  # budget 0.5% < 0.9%
        assert tight.reports["syn_ratio"].saturated is True
        assert tight.reports["syn_ratio"].action == "rarity_fit"

    def test_case_4_flows_per_src_unbounded_count_tied_at_adapter_maximum(self):
        # the CSV-adapter regression this generic fix exists for:
        # flows_per_src has NO theoretical bound (unlike the three cases
        # above, it was never in FEATURE_KINDS) but the CSV adapter's
        # lack of true concurrency let a handful of "busy" simulated
        # sources tie 14.2% of a real day's fit-half at a single maximum
        # -- reproduced here with the same shape (some spread of normal
        # values, ~14% piled at one tied max, single-sided/high-only like
        # the real feature). Old code: kind=CONTINUOUS meant NO saturation
        # check ever ran, so this threshold was silently kept, flat, and
        # permanently unreachable. New code must catch it.
        normal = [float(v) for v in range(1, 201)] * 43  # 200 distinct values, spread, ~8,600 rows
        tied_max = [999999.0] * 1420  # 14.2% of the total, all identical, all above `normal`
        values = normal + tied_max
        df = self.make_frame(len(values), values, "flows_per_src")

        for pct in (90.0, 95.0, 99.0, 99.5):  # budgets 10%/5%/1%/0.5%, ALL under the 14.2% tied mass
            result = fit_thresholds_from_frame(df, feature_names=["flows_per_src"], percentile=pct)
            report = result.reports["flows_per_src"]
            assert report.kind == "continuous"  # never hand-listed in FEATURE_KINDS -- this is the point
            assert report.feature_max == 999999.0
            assert report.tied_mass_high == pytest.approx(1420 / len(values))
            assert report.saturated is True, f"p{pct} should be saturated (budget < 14.2% tied mass)"
            # the old bug: a plain `high=999999.0` threshold would have
            # been kept here, silently, at every one of these percentiles.
            assert report.action != "kept"

        # eligible for rarity (201 distinct of ~10,020 used -> well under
        # both RARITY_MAX_DISTINCT_ABS and the 5% ratio cap) -- not just
        # excluded, since real held-out data (a different day/adapter)
        # legitimately can and does take values never seen in this
        # fit-half, which rarity-fit can still catch.
        result = fit_thresholds_from_frame(df, feature_names=["flows_per_src"], percentile=99.5)
        report = result.reports["flows_per_src"]
        assert report.action == "rarity_fit"
        threshold = result.config.thresholds["flows_per_src"]
        # a value never seen during fitting (e.g. a real attack's own
        # flows_per_src) is exactly what rarity-fit exists to catch.
        assert 5_000_000.0 not in threshold.common_values


class TestJsonRoundTrip:
    def test_save_and_load_thresholds(self, tmp_path):
        flows = [two_way_flow() for _ in range(5)]
        result = fit_thresholds(flows, [BENIGN_LABEL] * 5)
        path = tmp_path / "thresholds.json"
        save_thresholds(result.config, path)
        restored = load_thresholds(path)
        assert restored == result.config
