from pathlib import Path

import pytest

from adapters.pcap_adapter import iter_packets_from_pcap
from controlplane.extractor import (
    anonymize_features,
    assert_no_leaked_identity,
    collect_flow_packets,
    extract_record,
    make_flow_id,
    _select_representative_row,
)
from controlplane.record import EscalationRecord, PacketWindowSummary
from dataplane.flow_state import FlowState, canonical_key
from dataplane.flow_table import FlowTable, KeyMode
from dataplane.selector import EscalationRule, FeatureThreshold, SelectorConfig, TriggerReason

FIXTURE = Path(__file__).parent / "fixtures" / "monday_60s_slice.pcap"


def _raw_cicflowmeter_row(**overrides):
    """A representative raw output row shaped like what
    run_cicflowmeter_python returns (string values, since it comes back
    through a CSV round trip) -- see cicflowmeter's own column list."""
    row = {
        "src_ip": "192.168.10.25",
        "dst_ip": "54.192.37.123",
        "src_port": "51820",
        "dst_port": "443",
        "protocol": "6",
        "timestamp": "2017-07-03 15:59:12",
        "flow_duration": "1.234",
        "tot_fwd_pkts": "10",
        "tot_bwd_pkts": "7",
        "flow_byts_s": "1024.5",
    }
    row.update(overrides)
    return row


class TestAnonymizeFeatures:
    def test_strips_source_and_destination_ip(self):
        out = anonymize_features(_raw_cicflowmeter_row())
        assert "src_ip" not in out
        assert "dst_ip" not in out

    def test_strips_source_port(self):
        out = anonymize_features(_raw_cicflowmeter_row())
        assert "src_port" not in out

    def test_strips_protocol_and_absolute_timestamp(self):
        # allow-list: only dst_port survives from the identity columns
        out = anonymize_features(_raw_cicflowmeter_row())
        assert "timestamp" not in out
        assert "protocol" not in out

    def test_keeps_destination_port(self):
        out = anonymize_features(_raw_cicflowmeter_row(dst_port="443"))
        assert out["dst_port"] == 443.0

    def test_keeps_numeric_feature_columns(self):
        out = anonymize_features(_raw_cicflowmeter_row())
        assert out["flow_duration"] == 1.234
        assert out["tot_fwd_pkts"] == 10.0
        assert out["flow_byts_s"] == 1024.5

    def test_no_leaked_identity_survives_round_trip(self):
        out = anonymize_features(_raw_cicflowmeter_row())
        assert_no_leaked_identity(out)  # must not raise


class TestAssertNoLeakedIdentityCatchesRealLeaks:
    """The checker itself must actually catch what it claims to --
    tested against deliberately-unanonymised input so a bug in the
    regexes doesn't silently make every other test meaningless."""

    def test_catches_ipv4_shaped_string(self):
        with pytest.raises(AssertionError):
            assert_no_leaked_identity({"some_field": "192.168.10.25"})

    def test_catches_absolute_timestamp_string(self):
        with pytest.raises(AssertionError):
            assert_no_leaked_identity({"some_field": "2017-07-03 15:59:12"})

    def test_recurses_into_nested_structures(self):
        with pytest.raises(AssertionError):
            assert_no_leaked_identity({"outer": {"inner": ["10.0.0.1"]}})

    def test_passes_on_clean_numeric_data(self):
        assert_no_leaked_identity({"flow_duration": 1.5, "tot_fwd_pkts": 10})  # no raise

    def test_passes_on_non_ip_shaped_string(self):
        # a version string or hex token must not false-positive
        assert_no_leaked_identity({"flow_id": "528d06d9f23d4b888a8effa8d1f6dbb8"})  # no raise


class TestMakeFlowId:
    def test_returns_hex_string(self):
        flow_id = make_flow_id()
        assert isinstance(flow_id, str)
        int(flow_id, 16)  # raises if not valid hex

    def test_not_ip_shaped(self):
        assert_no_leaked_identity({"flow_id": make_flow_id()})

    def test_successive_calls_differ(self):
        assert make_flow_id() != make_flow_id()


class TestSelectRepresentativeRow:
    def test_empty_returns_none(self):
        assert _select_representative_row([]) is None

    def test_single_row_returned_as_is(self):
        row = _raw_cicflowmeter_row()
        assert _select_representative_row([row]) is row

    def test_multiple_rows_picks_most_packets(self):
        small = _raw_cicflowmeter_row(tot_fwd_pkts="2", tot_bwd_pkts="1")
        large = _raw_cicflowmeter_row(tot_fwd_pkts="50", tot_bwd_pkts="40")
        assert _select_representative_row([small, large]) is large


class TestEscalationRecordSerialization:
    def test_to_dict_carries_no_leaked_identity(self):
        record = EscalationRecord(
            flow_id=make_flow_id(),
            trigger_reasons=[
                TriggerReason(
                    feature="flow_iat_regularity",
                    observed_value=0.99,
                    threshold=0.95,
                    direction="high",
                    ratio=1.04,
                )
            ],
            features=anonymize_features(_raw_cicflowmeter_row()),
            packet_window=PacketWindowSummary(packet_count=17, byte_count=4096, duration_us=1_234_567),
            selector_config_hash="abc123",
        )
        assert_no_leaked_identity(record.to_dict())

    def test_to_dict_has_no_label_field(self):
        # ground truth lives in the eval harness, never in the record --
        # a "label"/"ground_truth" key here would be a real regression.
        record = EscalationRecord(
            flow_id=make_flow_id(),
            trigger_reasons=[],
            features={},
            packet_window=PacketWindowSummary(packet_count=1, byte_count=1, duration_us=0),
            selector_config_hash="x",
        )
        d = record.to_dict()
        assert "label" not in d
        assert "ground_truth" not in d


class TestCollectFlowPacketsAgainstRealFixture:
    """Integration test against the real 60s test slice -- confirms the
    identity+time-window matching actually finds a flow's own packets
    in the source pcap, not just that the code runs."""

    def test_collected_packet_count_matches_flow_table_count(self):
        table = FlowTable(key_mode=KeyMode.FIDELITY)
        closed = []
        for p in iter_packets_from_pcap(FIXTURE):
            result = table.process(p)
            closed.extend(e.state for e in result.expired)
        closed.extend(e.state for e in table.flush())

        target = max(closed, key=lambda f: f.total_pkt_count)
        collected = collect_flow_packets(FIXTURE, [target])
        packets = collected[(target.key, target.first_ts)]
        assert len(packets) == target.total_pkt_count

    def test_disjoint_flows_sharing_a_key_get_disjoint_packets(self):
        # regression guard for the identity-vs-time-window lesson from
        # eval/labels_pcap.py: two FlowState instances with the same key
        # (a 5-tuple reused later) must each only claim their own packets.
        table = FlowTable(key_mode=KeyMode.FIDELITY)
        closed = []
        for p in iter_packets_from_pcap(FIXTURE):
            result = table.process(p)
            closed.extend(e.state for e in result.expired)
        closed.extend(e.state for e in table.flush())

        by_key = {}
        for f in closed:
            by_key.setdefault(f.key, []).append(f)
        reused = [flows for flows in by_key.values() if len(flows) > 1]
        if not reused:
            pytest.skip("no reused 5-tuple in this fixture slice to test against")

        instances = reused[0]
        collected = collect_flow_packets(FIXTURE, instances)
        for f in instances:
            packets = collected[(f.key, f.first_ts)]
            assert len(packets) == f.total_pkt_count
            for ts_us, _raw in packets:
                assert f.first_ts <= ts_us <= f.last_ts


class TestExtractRecordEndToEnd:
    def test_produces_anonymised_record_with_real_cicflowmeter(self):
        table = FlowTable(key_mode=KeyMode.FIDELITY)
        closed = []
        for p in iter_packets_from_pcap(FIXTURE):
            result = table.process(p)
            closed.extend(e.state for e in result.expired)
        closed.extend(e.state for e in table.flush())

        target = max(closed, key=lambda f: f.total_pkt_count)
        packets = collect_flow_packets(FIXTURE, [target])[(target.key, target.first_ts)]

        config = SelectorConfig(
            thresholds={"flow_duration": FeatureThreshold(high=0.0)},
            rule=EscalationRule.K_OF_N,
            k=1,
            config_hash="test-hash",
        )
        record = extract_record(target, trigger_reasons=[], packets=packets, selector_config=config)

        assert record is not None
        assert record.packet_window.packet_count == len(packets)
        assert len(record.features) > 0
        assert "dst_port" in record.features
        assert_no_leaked_identity(record.to_dict())
