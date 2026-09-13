from dataplane.src_table import SrcTable

US = 1_000_000  # 1 second in microseconds


class TestFlowsPerSrc:
    def test_counts_flows_within_window(self):
        table = SrcTable(window_us=60 * US, num_buckets=6)
        table.note_flow_start("10.0.0.1", "10.0.0.2", 80, 0)
        table.note_flow_start("10.0.0.1", "10.0.0.3", 443, 5 * US)
        assert table.flows_per_src("10.0.0.1", now_us=10 * US) == 2

    def test_flows_age_out_of_window(self):
        table = SrcTable(window_us=60 * US, num_buckets=6)
        table.note_flow_start("10.0.0.1", "10.0.0.2", 80, 0)
        # far beyond the 60s window
        assert table.flows_per_src("10.0.0.1", now_us=120 * US) == 0

    def test_partial_window_expiry_by_bucket(self):
        table = SrcTable(window_us=60 * US, num_buckets=6)  # 10s buckets
        table.note_flow_start("10.0.0.1", "10.0.0.2", 80, 0)
        table.note_flow_start("10.0.0.1", "10.0.0.2", 81, 55 * US)
        # at t=65s, the bucket holding the t=0 event (bucket 0) has aged out
        # (65 - 0 = 65s > 60s window), but the t=55s event's bucket has not.
        assert table.flows_per_src("10.0.0.1", now_us=65 * US) == 1

    def test_unknown_source_returns_zero(self):
        table = SrcTable()
        assert table.flows_per_src("192.0.2.1", now_us=0) == 0

    def test_independent_sources_do_not_interfere(self):
        table = SrcTable(window_us=60 * US)
        table.note_flow_start("10.0.0.1", "10.0.0.9", 80, 0)
        table.note_flow_start("10.0.0.2", "10.0.0.9", 80, 0)
        assert table.flows_per_src("10.0.0.1", now_us=0) == 1
        assert table.flows_per_src("10.0.0.2", now_us=0) == 1


class TestDistinctCounts:
    def test_distinct_ports_approx_correct(self):
        table = SrcTable(window_us=60 * US, hll_precision=10)
        for port in range(50):
            table.note_flow_start("10.0.0.1", "10.0.0.2", port, 0)
        estimate = table.distinct_dst_ports_per_src("10.0.0.1", now_us=0)
        tolerance = 5 * table.sketch_standard_error * 50
        assert abs(estimate - 50) < tolerance

    def test_distinct_ips_approx_correct(self):
        table = SrcTable(window_us=60 * US, hll_precision=10)
        for i in range(50):
            table.note_flow_start("10.0.0.1", f"10.0.1.{i}", 80, 0)
        estimate = table.distinct_dst_ips_per_src("10.0.0.1", now_us=0)
        tolerance = 5 * table.sketch_standard_error * 50
        assert abs(estimate - 50) < tolerance

    def test_repeated_destination_does_not_inflate_distinct_count(self):
        table = SrcTable(window_us=60 * US)
        for _ in range(20):
            table.note_flow_start("10.0.0.1", "10.0.0.2", 80, 0)
        assert table.distinct_dst_ports_per_src("10.0.0.1", now_us=0) < 3

    def test_distinct_counts_respect_window_expiry(self):
        table = SrcTable(window_us=60 * US)
        table.note_flow_start("10.0.0.1", "10.0.0.2", 80, 0)
        assert table.distinct_dst_ports_per_src("10.0.0.1", now_us=120 * US) == 0

    def test_sketch_standard_error_reported(self):
        table = SrcTable(hll_precision=6)
        assert abs(table.sketch_standard_error - (1.04 / 8)) < 1e-9


class TestSynWithoutSynAck:
    def test_tcp_syn_with_no_response_counts(self):
        table = SrcTable(window_us=60 * US)
        table.note_flow_closed(
            "10.0.0.1", 0, protocol=6, syn_count=1, bwd_pkt_count=0
        )
        assert table.syn_without_synack_count("10.0.0.1", now_us=0) == 1

    def test_tcp_syn_with_response_does_not_count(self):
        table = SrcTable(window_us=60 * US)
        table.note_flow_closed(
            "10.0.0.1", 0, protocol=6, syn_count=1, bwd_pkt_count=3
        )
        assert table.syn_without_synack_count("10.0.0.1", now_us=0) == 0

    def test_udp_flows_never_count(self):
        table = SrcTable(window_us=60 * US)
        table.note_flow_closed(
            "10.0.0.1", 0, protocol=17, syn_count=0, bwd_pkt_count=0
        )
        assert table.syn_without_synack_count("10.0.0.1", now_us=0) == 0

    def test_no_syn_at_all_does_not_count(self):
        table = SrcTable(window_us=60 * US)
        table.note_flow_closed(
            "10.0.0.1", 0, protocol=6, syn_count=0, bwd_pkt_count=0
        )
        assert table.syn_without_synack_count("10.0.0.1", now_us=0) == 0

    def test_accumulates_across_multiple_scans(self):
        table = SrcTable(window_us=60 * US)
        for t in range(0, 5):
            table.note_flow_closed(
                "10.0.0.1", t * US, protocol=6, syn_count=1, bwd_pkt_count=0
            )
        assert table.syn_without_synack_count("10.0.0.1", now_us=5 * US) == 5


class TestConstructorValidation:
    def test_rejects_nonpositive_window(self):
        try:
            SrcTable(window_us=0)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_rejects_nonpositive_buckets(self):
        try:
            SrcTable(num_buckets=0)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_rejects_nonpositive_capacity(self):
        try:
            SrcTable(capacity=0)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_default_capacity_is_8192(self):
        assert SrcTable().capacity == 8192


class TestCapacityEviction:
    def test_capacity_never_exceeded(self):
        table = SrcTable(capacity=4)
        for i in range(20):
            table.note_flow_start(f"10.0.0.{i}", "10.0.1.1", 80, 0)
        assert len(table._sources) <= 4

    def test_evicts_least_recently_touched_source(self):
        table = SrcTable(capacity=2)
        table.note_flow_start("10.0.0.1", "10.0.0.9", 80, 0)
        table.note_flow_start("10.0.0.2", "10.0.0.9", 80, 0)
        table.note_flow_start("10.0.0.1", "10.0.0.9", 81, 1)  # refresh source 1
        table.note_flow_start("10.0.0.3", "10.0.0.9", 80, 2)  # forces eviction
        assert table.eviction_count == 1
        assert table.flows_per_src("10.0.0.2", now_us=2) == 0  # evicted
        assert table.flows_per_src("10.0.0.1", now_us=2) == 2  # survived
        assert table.flows_per_src("10.0.0.3", now_us=2) == 1

    def test_note_flow_closed_also_refreshes_lru(self):
        table = SrcTable(capacity=2)
        table.note_flow_start("10.0.0.1", "10.0.0.9", 80, 0)
        table.note_flow_start("10.0.0.2", "10.0.0.9", 80, 0)
        table.note_flow_closed("10.0.0.1", 1, protocol=6, syn_count=1, bwd_pkt_count=0)
        table.note_flow_start("10.0.0.3", "10.0.0.9", 80, 2)
        assert table.flows_per_src("10.0.0.1", now_us=2) == 1  # survived
        assert table.flows_per_src("10.0.0.2", now_us=2) == 0  # evicted


class TestMemoryCostReporting:
    def test_estimated_bytes_per_source_matches_formula(self):
        table = SrcTable(hll_precision=8, num_buckets=6)
        # 2 sketches x 256 registers + 4 counter bytes, x6 buckets
        assert table.estimated_bytes_per_source == (2 * 256 + 4) * 6

    def test_estimated_total_bytes_scales_with_capacity(self):
        table = SrcTable(hll_precision=8, num_buckets=6, capacity=1000)
        assert table.estimated_total_bytes == table.estimated_bytes_per_source * 1000
