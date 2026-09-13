import pytest

from dataplane.flow_state import Packet
from dataplane.flow_table import DEFAULT_CAPACITY, UNBOUNDED_CAPACITY, FlowTable, KeyMode


def make_packet(**overrides):
    defaults = dict(
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=1234,
        dst_port=80,
        protocol=6,
        timestamp_us=0,
        length=100,
    )
    defaults.update(overrides)
    return Packet(**defaults)


class TestBasicFlowCreationAndReuse:
    def test_first_packet_creates_new_flow(self):
        table = FlowTable()
        result = table.process(make_packet())
        assert result.is_new_flow
        assert len(table) == 1
        assert table.total_flows_created == 1

    def test_second_packet_same_flow_reuses_state(self):
        table = FlowTable()
        table.process(make_packet(timestamp_us=0))
        result = table.process(make_packet(timestamp_us=100))
        assert not result.is_new_flow
        assert len(table) == 1
        assert result.state.fwd_pkt_count == 2

    def test_bidirectional_packets_join_same_flow(self):
        table = FlowTable()
        table.process(make_packet(timestamp_us=0))
        reply = make_packet(
            src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=80, dst_port=1234,
            timestamp_us=10,
        )
        result = table.process(reply)
        assert not result.is_new_flow
        assert len(table) == 1
        assert result.state.bwd_pkt_count == 1

    def test_different_5_tuples_are_different_flows(self):
        table = FlowTable()
        table.process(make_packet(dst_port=80))
        table.process(make_packet(dst_port=443))
        assert len(table) == 2


class TestTermination:
    def test_rst_removes_flow_from_table(self):
        table = FlowTable()
        table.process(make_packet(timestamp_us=0))
        result = table.process(make_packet(timestamp_us=10, rst=True))
        assert len(table) == 0
        assert any(e.reason == "rst" for e in result.expired)

    def test_fin_handshake_removes_flow(self):
        table = FlowTable()
        table.process(make_packet(timestamp_us=0))
        table.process(make_packet(timestamp_us=10, fin=True))
        reply_fin = make_packet(
            src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=80, dst_port=1234,
            timestamp_us=20, fin=True,
        )
        result = table.process(reply_fin)
        assert len(table) == 0
        assert any(e.reason == "fin" for e in result.expired)

    def test_flush_closes_remaining_flows(self):
        table = FlowTable()
        table.process(make_packet())
        expired = table.flush()
        assert len(expired) == 1
        assert len(table) == 0


class TestIdleTimeout:
    def test_flow_expires_after_idle_gap(self):
        table = FlowTable(idle_timeout_us=1000)
        table.process(make_packet(timestamp_us=0))
        # a later, unrelated packet advances the clock past the idle timeout
        other = make_packet(dst_port=443, timestamp_us=5000)
        result = table.process(other)
        assert table.idle_timeout_count == 1
        assert any(e.reason == "idle_timeout" for e in result.expired)
        # the idle flow is gone, only the new one remains
        assert len(table) == 1

    def test_flow_survives_within_idle_window(self):
        table = FlowTable(idle_timeout_us=10_000)
        table.process(make_packet(timestamp_us=0))
        result = table.process(make_packet(timestamp_us=5000))
        assert result.is_new_flow is False
        assert table.idle_timeout_count == 0


class TestActiveTimeout:
    def test_long_lived_flow_force_closed_and_restarted(self):
        table = FlowTable(active_timeout_us=1000, idle_timeout_us=10**9)
        table.process(make_packet(timestamp_us=0))
        result = table.process(make_packet(timestamp_us=2000))
        assert table.active_timeout_count == 1
        assert any(e.reason == "active_timeout" for e in result.expired)
        # a fresh flow record was started under the same key
        assert result.is_new_flow
        assert result.state.first_ts == 2000


class TestCapacityEviction:
    def test_evicts_least_recently_used_when_full(self):
        table = FlowTable(capacity=2)
        table.process(make_packet(dst_port=1))
        table.process(make_packet(dst_port=2))
        # touch flow 1 so flow 2 becomes the least-recently-used
        table.process(make_packet(dst_port=1, timestamp_us=1))
        result = table.process(make_packet(dst_port=3, timestamp_us=2))
        assert table.eviction_count == 1
        assert any(e.reason == "evicted" for e in result.expired)
        assert len(table) == 2

    def test_capacity_never_exceeded(self):
        table = FlowTable(capacity=4)
        for port in range(20):
            table.process(make_packet(dst_port=port, timestamp_us=port))
        assert len(table) <= 4

    def test_rejects_nonpositive_capacity(self):
        with pytest.raises(ValueError):
            FlowTable(capacity=0)


class TestKeyMode:
    def test_default_is_fidelity(self):
        table = FlowTable()
        assert table.key_mode == KeyMode.FIDELITY

    def test_fidelity_mode_merges_same_5_tuple_regardless_of_row_id(self):
        table = FlowTable(key_mode=KeyMode.FIDELITY)
        table.process(make_packet(timestamp_us=0, source_row_id=1))
        result = table.process(make_packet(timestamp_us=10, source_row_id=2))
        assert not result.is_new_flow
        assert len(table) == 1
        assert result.state.source_row_ids == {1, 2}

    def test_eval_mode_keeps_rows_separate_even_with_same_5_tuple(self):
        table = FlowTable(key_mode=KeyMode.EVAL)
        table.process(make_packet(timestamp_us=0, source_row_id=1))
        result = table.process(make_packet(timestamp_us=10, source_row_id=2))
        assert result.is_new_flow
        assert len(table) == 2

    def test_eval_mode_still_joins_bidirectional_packets_of_the_same_row(self):
        table = FlowTable(key_mode=KeyMode.EVAL)
        table.process(make_packet(timestamp_us=0, source_row_id=1))
        reply = make_packet(
            src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=80, dst_port=1234,
            timestamp_us=10, source_row_id=1,
        )
        result = table.process(reply)
        assert not result.is_new_flow
        assert len(table) == 1

    def test_eval_mode_flows_have_singleton_row_id_set(self):
        table = FlowTable(key_mode=KeyMode.EVAL)
        table.process(make_packet(timestamp_us=0, source_row_id=1))
        result = table.process(make_packet(dst_port=443, timestamp_us=0, source_row_id=2))
        assert result.state.source_row_ids == {2}

    def test_eval_mode_ignores_idle_timeout_within_one_row(self):
        # a real Monday row: 2 packets, ~64s apart, well over the 15s
        # idle timeout — but it's one CICFlowMeter-recorded flow, and
        # EVAL mode must not split it on that basis alone.
        table = FlowTable(key_mode=KeyMode.EVAL, idle_timeout_us=15_000_000)
        table.process(make_packet(timestamp_us=0, source_row_id=1))
        result = table.process(make_packet(timestamp_us=64_000_000, source_row_id=1))
        assert not result.is_new_flow
        assert len(table) == 1
        assert table.idle_timeout_count == 0

    def test_eval_mode_ignores_active_timeout_within_one_row(self):
        table = FlowTable(key_mode=KeyMode.EVAL, active_timeout_us=1000)
        table.process(make_packet(timestamp_us=0, source_row_id=1))
        result = table.process(make_packet(timestamp_us=2000, source_row_id=1))
        assert not result.is_new_flow
        assert table.active_timeout_count == 0

    def test_eval_mode_still_evicts_under_capacity_pressure(self):
        table = FlowTable(key_mode=KeyMode.EVAL, capacity=2)
        table.process(make_packet(dst_port=1, timestamp_us=0, source_row_id=1))
        table.process(make_packet(dst_port=2, timestamp_us=0, source_row_id=2))
        result = table.process(make_packet(dst_port=3, timestamp_us=0, source_row_id=3))
        assert table.eviction_count == 1
        assert len(table) == 2

    def test_fidelity_default_capacity_is_65536(self):
        table = FlowTable(key_mode=KeyMode.FIDELITY)
        assert table.capacity == DEFAULT_CAPACITY

    def test_eval_default_capacity_is_unbounded(self):
        table = FlowTable(key_mode=KeyMode.EVAL)
        assert table.capacity == UNBOUNDED_CAPACITY

    def test_eval_capacity_still_overridable_for_testing_eviction(self):
        table = FlowTable(key_mode=KeyMode.EVAL, capacity=2)
        table.process(make_packet(dst_port=1, timestamp_us=0, source_row_id=1))
        table.process(make_packet(dst_port=2, timestamp_us=0, source_row_id=2))
        table.process(make_packet(dst_port=3, timestamp_us=0, source_row_id=3))
        assert table.eviction_count == 1

    def test_eval_unbounded_capacity_never_evicts_many_flows(self):
        table = FlowTable(key_mode=KeyMode.EVAL)
        for i in range(5000):
            table.process(make_packet(dst_port=i, timestamp_us=0, source_row_id=i))
        assert table.eviction_count == 0
        assert len(table) == 5000

    def test_fidelity_mode_still_applies_idle_timeout(self):
        table = FlowTable(key_mode=KeyMode.FIDELITY, idle_timeout_us=1000)
        table.process(make_packet(timestamp_us=0, source_row_id=1))
        other = make_packet(dst_port=443, timestamp_us=5000, source_row_id=2)
        result = table.process(other)
        assert table.idle_timeout_count == 1
        assert any(e.reason == "idle_timeout" for e in result.expired)


class TestThroughputCounters:
    def test_total_packets_seen_counts_every_call(self):
        table = FlowTable()
        for i in range(5):
            table.process(make_packet(timestamp_us=i))
        assert table.total_packets_seen == 5
