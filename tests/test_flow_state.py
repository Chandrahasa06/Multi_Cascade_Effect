from dataplane.flow_state import FlowState, Packet, UNSET, canonical_key


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


class TestCanonicalKey:
    def test_bidirectional_packets_map_to_same_key(self):
        fwd = canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 6)
        bwd = canonical_key("10.0.0.2", "10.0.0.1", 80, 1234, 6)
        assert fwd == bwd

    def test_different_protocol_gives_different_key(self):
        tcp = canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 6)
        udp = canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 17)
        assert tcp != udp

    def test_different_flow_gives_different_key(self):
        a = canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 6)
        b = canonical_key("10.0.0.1", "10.0.0.3", 1234, 80, 6)
        assert a != b

    def test_row_discriminator_distinguishes_same_5_tuple(self):
        a = canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 6, row_discriminator=1)
        b = canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 6, row_discriminator=2)
        assert a != b

    def test_row_discriminator_still_bidirectional(self):
        fwd = canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 6, row_discriminator=5)
        bwd = canonical_key("10.0.0.2", "10.0.0.1", 80, 1234, 6, row_discriminator=5)
        assert fwd == bwd

    def test_no_discriminator_matches_default(self):
        assert canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 6, row_discriminator=None) == \
            canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 6)


class TestFlowStateStart:
    def test_start_records_first_packet_as_forward(self):
        pkt = make_packet(timestamp_us=100, length=64)
        state = FlowState.start(pkt)
        assert state.fwd_ip == "10.0.0.1"
        assert state.fwd_port == 1234
        assert state.fwd_pkt_count == 1
        assert state.bwd_pkt_count == 0
        assert state.fwd_byte_sum == 64
        assert state.first_ts == 100
        assert state.last_ts == 100


class TestFlowStateUpdate:
    def test_reverse_direction_packet_counts_as_backward(self):
        state = FlowState.start(make_packet(timestamp_us=0, length=50))
        reply = make_packet(
            src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=80, dst_port=1234,
            timestamp_us=10, length=200,
        )
        state.update(reply)
        assert state.fwd_pkt_count == 1
        assert state.bwd_pkt_count == 1
        assert state.bwd_byte_sum == 200

    def test_byte_and_packet_counts_accumulate(self):
        state = FlowState.start(make_packet(timestamp_us=0, length=100))
        state.update(make_packet(timestamp_us=1000, length=200))
        state.update(make_packet(timestamp_us=2000, length=50))
        assert state.fwd_pkt_count == 3
        assert state.fwd_byte_sum == 350

    def test_pkt_len_min_max_track_extremes(self):
        state = FlowState.start(make_packet(timestamp_us=0, length=100))
        state.update(make_packet(timestamp_us=1000, length=40))
        state.update(make_packet(timestamp_us=2000, length=900))
        assert state.fwd_pkt_len_min == 40
        assert state.fwd_pkt_len_max == 900

    def test_iat_tracks_min_max_sum(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        state.update(make_packet(timestamp_us=1000))  # iat 1000
        state.update(make_packet(timestamp_us=1500))  # iat 500
        state.update(make_packet(timestamp_us=5000))  # iat 3500
        assert state.iat_sum == 1000 + 500 + 3500
        assert state.iat_min == 500
        assert state.iat_max == 3500

    def test_first_packet_has_no_iat(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        assert state.iat_sum == 0
        assert state.iat_min == UNSET
        assert state.iat_max == 0

    def test_out_of_order_timestamp_clamps_iat_to_zero(self):
        state = FlowState.start(make_packet(timestamp_us=1000))
        state.update(make_packet(timestamp_us=900))  # earlier than prev
        assert state.iat_min == 0
        assert state.iat_sum == 0

    def test_init_win_bytes_captured_once_per_direction(self):
        state = FlowState.start(make_packet(timestamp_us=0, win_size=65535))
        state.update(make_packet(timestamp_us=100, win_size=1000))
        assert state.init_win_bytes_fwd == 65535  # not overwritten by 2nd fwd pkt

        reply = make_packet(
            src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=80, dst_port=1234,
            timestamp_us=50, win_size=29200,
        )
        state.update(reply)
        assert state.init_win_bytes_bwd == 29200

    def test_flag_counts(self):
        state = FlowState.start(make_packet(timestamp_us=0, syn=True))
        state.update(make_packet(timestamp_us=10, ack=True, psh=True))
        state.update(make_packet(timestamp_us=20, urg=True))
        assert state.syn_count == 1
        assert state.ack_count == 1
        assert state.psh_count == 1
        assert state.urg_count == 1


class TestTimestampResolution:
    def test_default_resolution_is_exact(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        assert state.timestamp_resolution_us == 1

    def test_flow_inherits_packet_resolution(self):
        state = FlowState.start(make_packet(timestamp_us=0, timestamp_resolution_us=60_000_000))
        assert state.timestamp_resolution_us == 60_000_000

    def test_flow_keeps_coarsest_resolution_seen(self):
        state = FlowState.start(make_packet(timestamp_us=0, timestamp_resolution_us=1_000_000))
        state.update(make_packet(timestamp_us=10, timestamp_resolution_us=60_000_000))
        assert state.timestamp_resolution_us == 60_000_000
        state.update(make_packet(timestamp_us=20, timestamp_resolution_us=1))
        assert state.timestamp_resolution_us == 60_000_000  # never gets more confident


class TestSourceRowIds:
    def test_default_is_empty(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        assert state.source_row_ids == set()

    def test_records_row_id_from_packets(self):
        state = FlowState.start(make_packet(timestamp_us=0, source_row_id=42))
        assert state.source_row_ids == {42}

    def test_accumulates_distinct_row_ids_across_updates(self):
        state = FlowState.start(make_packet(timestamp_us=0, source_row_id=1))
        state.update(make_packet(timestamp_us=10, source_row_id=2))
        state.update(make_packet(timestamp_us=20, source_row_id=1))  # repeat, no dup
        assert state.source_row_ids == {1, 2}

    def test_packets_without_row_id_do_not_pollute_set(self):
        state = FlowState.start(make_packet(timestamp_us=0, source_row_id=7))
        state.update(make_packet(timestamp_us=10))  # no row id (e.g. real capture)
        assert state.source_row_ids == {7}


class TestTermination:
    def test_rst_terminates_immediately(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        state.update(make_packet(timestamp_us=10, rst=True))
        assert state.terminated
        assert state.termination_reason == "rst"

    def test_single_sided_fin_does_not_terminate(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        state.update(make_packet(timestamp_us=10, fin=True))
        assert not state.terminated

    def test_fin_from_both_sides_terminates(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        state.update(make_packet(timestamp_us=10, fin=True))
        reply_fin = make_packet(
            src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=80, dst_port=1234,
            timestamp_us=20, fin=True,
        )
        state.update(reply_fin)
        assert state.terminated
        assert state.termination_reason == "fin"

    def test_total_pkt_and_byte_count_properties(self):
        state = FlowState.start(make_packet(timestamp_us=0, length=100))
        reply = make_packet(
            src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=80, dst_port=1234,
            timestamp_us=10, length=200,
        )
        state.update(reply)
        assert state.total_pkt_count == 2
        assert state.total_byte_sum == 300
