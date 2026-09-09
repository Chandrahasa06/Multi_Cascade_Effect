import csv

import pytest

from dataplane.flow_state import FlowState
from eval.labels import (
    RowGroundTruth,
    join_eval_flows,
    join_fidelity_flows,
    load_row_ground_truth,
    resolve_eval_label,
    resolve_fidelity_label,
)


def make_flow(row_ids, first_ts=0):
    return FlowState(
        key=("10.0.0.1", "10.0.0.2", 1, 2, 6),
        fwd_ip="10.0.0.1",
        fwd_port=1,
        protocol=6,
        first_ts=first_ts,
        source_row_ids=set(row_ids),
    )


ROW_TRUTH = [
    RowGroundTruth(label="BENIGN", timestamp_us=0),
    RowGroundTruth(label="BENIGN", timestamp_us=1_000_000),
    RowGroundTruth(label="DoS Hulk", timestamp_us=2_000_000),
    RowGroundTruth(label="DoS slowloris", timestamp_us=2_500_000),  # 0.5s from row 2
    RowGroundTruth(label="PortScan", timestamp_us=50_000_000),  # far from everything
]


class TestEvalResolution:
    def test_singleton_row_id_resolves_directly(self):
        flow = make_flow([0])
        assert resolve_eval_label(flow, ROW_TRUTH) == "BENIGN"

    def test_asserts_on_more_than_one_row_id(self):
        flow = make_flow([0, 1])
        with pytest.raises(AssertionError):
            resolve_eval_label(flow, ROW_TRUTH)

    def test_asserts_on_zero_row_ids(self):
        flow = make_flow([])
        with pytest.raises(AssertionError):
            resolve_eval_label(flow, ROW_TRUTH)


class TestFidelityResolutionSingleLabel:
    def test_single_row_not_mixed(self):
        flow = make_flow([2])
        resolution = resolve_fidelity_label(flow, ROW_TRUTH)
        assert resolution.label == "DoS Hulk"
        assert not resolution.mixed
        assert not resolution.ambiguous

    def test_merged_but_agreeing_rows_not_mixed(self):
        flow = make_flow([0, 1])  # both BENIGN
        resolution = resolve_fidelity_label(flow, ROW_TRUTH)
        assert resolution.label == "BENIGN"
        assert not resolution.mixed


class TestFidelityResolutionMixedLabels:
    def test_benign_attack_mix_classified_correctly(self):
        flow = make_flow([0, 2], first_ts=0)  # BENIGN @0, DoS Hulk @2e6
        resolution = resolve_fidelity_label(flow, ROW_TRUTH)
        assert resolution.mixed
        assert resolution.mixing_kind == "benign_attack"
        assert resolution.label == "BENIGN"  # row 0 (ts=0) nearer to first_ts=0

    def test_attack_attack_mix_classified_correctly(self):
        flow = make_flow([2, 3], first_ts=2_000_000)  # DoS Hulk @2e6, DoS slowloris @2.5e6
        resolution = resolve_fidelity_label(flow, ROW_TRUTH)
        assert resolution.mixed
        assert resolution.mixing_kind == "attack_attack"

    def test_nearest_timestamp_wins(self):
        flow = make_flow([2, 4], first_ts=1_900_000)  # nearer to row 2 (2e6) than row 4 (50e6)
        resolution = resolve_fidelity_label(flow, ROW_TRUTH)
        assert resolution.label == "DoS Hulk"

    def test_close_rows_flagged_residually_ambiguous(self):
        # rows 2 and 3 are 0.5s apart — below the 1s ambiguity threshold
        flow = make_flow([2, 3], first_ts=2_000_000)
        resolution = resolve_fidelity_label(flow, ROW_TRUTH)
        assert resolution.ambiguous

    def test_well_separated_rows_not_ambiguous(self):
        # row 2 (2e6) is nearest; row 4 (50e6) is 48s away — well clear
        flow = make_flow([2, 4], first_ts=1_900_000)
        resolution = resolve_fidelity_label(flow, ROW_TRUTH)
        assert not resolution.ambiguous


class TestJoinEvalFlows:
    def test_all_flows_get_labels(self):
        flows = [make_flow([0]), make_flow([2]), make_flow([4])]
        labels, report = join_eval_flows(flows, ROW_TRUTH)
        assert labels == ["BENIGN", "DoS Hulk", "PortScan"]
        assert report.join_rate == 1.0
        assert report.total_flows == 3


class TestJoinFidelityFlows:
    def test_report_counts_add_up(self):
        flows = [
            make_flow([0]),  # single row
            make_flow([0, 1]),  # merged, same label
            make_flow([0, 2], first_ts=0),  # merged, benign+attack mix, not ambiguous
            make_flow([2, 3], first_ts=2_000_000),  # merged, attack+attack mix, ambiguous
        ]
        labels, report = join_fidelity_flows(flows, ROW_TRUTH)
        assert len(labels) == 4
        assert report.total_flows == 4
        assert report.single_row_flows == 1
        assert report.merged_flows == 3
        assert report.merged_same_label_flows == 1
        assert report.merged_mixed_label_flows == 2
        assert report.benign_attack_mixed == 1
        assert report.attack_attack_mixed == 1
        assert report.residual_ambiguous == 1
        assert report.join_rate == 1.0 - (1 / 4)

    def test_join_rate_is_1_when_nothing_is_ambiguous(self):
        flows = [make_flow([0]), make_flow([0, 1])]
        _, report = join_fidelity_flows(flows, ROW_TRUTH)
        assert report.join_rate == 1.0

    def test_empty_flows_gives_join_rate_1(self):
        _, report = join_fidelity_flows([], ROW_TRUTH)
        assert report.join_rate == 1.0
        assert report.total_flows == 0


class TestLoadRowGroundTruth:
    def test_row_order_matches_adapter_indexing(self, tmp_path):
        # simulate the leading-space column-name quirk on Label too
        path = tmp_path / "day.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Timestamp", " Label"])
            writer.writeheader()
            writer.writerow({"Timestamp": "03/07/2017 08:55:58", " Label": "BENIGN"})
            writer.writerow({"Timestamp": "03/07/2017 08:56:22", " Label": "DoS Hulk"})

        truth = load_row_ground_truth(path)
        assert truth[0].label == "BENIGN"
        assert truth[1].label == "DoS Hulk"
        assert truth[1].timestamp_us > truth[0].timestamp_us
