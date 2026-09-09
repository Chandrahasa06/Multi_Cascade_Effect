from dataplane.flow_state import FlowState, Packet
from dataplane.selector import (
    EscalationRule,
    FeatureThreshold,
    Selector,
    SelectorConfig,
    compute_tier1_features,
    should_run_selector,
)
from dataplane.src_table import SrcTable


def make_packet(**overrides):
    defaults = dict(
        src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1234, dst_port=80,
        protocol=6, timestamp_us=0, length=100,
    )
    defaults.update(overrides)
    return Packet(**defaults)


def two_way_flow(fwd_len=100, bwd_len=200, gap_us=1000, resolution=1):
    state = FlowState.start(make_packet(timestamp_us=0, length=fwd_len, timestamp_resolution_us=resolution))
    reply = make_packet(
        src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=80, dst_port=1234,
        timestamp_us=gap_us, length=bwd_len, timestamp_resolution_us=resolution,
    )
    state.update(reply)
    return state


class TestComputeTier1Features:
    def test_flow_duration(self):
        state = two_way_flow(gap_us=5000)
        assert compute_tier1_features(state)["flow_duration"].value == 5000

    def test_rates_defined_when_duration_positive(self):
        state = two_way_flow(fwd_len=100, bwd_len=200, gap_us=1000)
        features = compute_tier1_features(state)
        # (100+200) bytes / 1000us * 1e6 = 300,000 bytes/sec
        assert features["flow_bytes_per_sec"].value == 300_000
        assert features["flow_pkts_per_sec"].value == 2_000

    def test_rates_undefined_at_zero_duration(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        state.update(make_packet(timestamp_us=0))  # same instant: zero duration
        features = compute_tier1_features(state)
        assert features["flow_bytes_per_sec"].value is None
        assert features["flow_pkts_per_sec"].value is None

    def test_pkt_len_mean(self):
        state = two_way_flow(fwd_len=100, bwd_len=200)
        features = compute_tier1_features(state)
        assert features["fwd_pkt_len_mean"].value == 100
        assert features["bwd_pkt_len_mean"].value == 200

    def test_pkt_len_range_combines_both_directions(self):
        state = two_way_flow(fwd_len=40, bwd_len=900)
        assert compute_tier1_features(state)["pkt_len_range"].value == 900 - 40

    def test_pkt_len_range_undefined_only_impossible_since_fwd_always_present(self):
        # a flow always has >= 1 fwd packet, so pkt_len_range is always defined
        state = FlowState.start(make_packet(timestamp_us=0, length=64))
        assert compute_tier1_features(state)["pkt_len_range"].value == 0.0

    def test_down_up_pkt_ratio(self):
        state = two_way_flow()
        assert compute_tier1_features(state)["down_up_pkt_ratio"].value == 1.0

    def test_bwd_fwd_byte_ratio(self):
        state = two_way_flow(fwd_len=100, bwd_len=300)
        assert compute_tier1_features(state)["bwd_fwd_byte_ratio"].value == 3.0

    def test_flow_iat_mean_undefined_for_single_packet(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        assert compute_tier1_features(state)["flow_iat_mean"].value is None

    def test_flow_iat_mean_defined_for_multi_packet(self):
        state = two_way_flow(gap_us=4000)
        assert compute_tier1_features(state)["flow_iat_mean"].value == 4000

    def test_flow_iat_min_unset_for_single_packet(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        assert compute_tier1_features(state)["flow_iat_min"].value is None

    def test_flow_iat_max_defined_even_for_single_packet(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        assert compute_tier1_features(state)["flow_iat_max"].value == 0.0

    def test_no_response_flag(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        assert compute_tier1_features(state)["no_response_flag"].value == 1.0
        state2 = two_way_flow()
        assert compute_tier1_features(state2)["no_response_flag"].value == 0.0

    def test_syn_rst_ratio(self):
        state = FlowState.start(make_packet(timestamp_us=0, syn=True))
        state.update(make_packet(timestamp_us=10, rst=True))
        features = compute_tier1_features(state)
        assert features["syn_ratio"].value == 0.5
        assert features["rst_ratio"].value == 0.5

    def test_init_win_bytes_passthrough(self):
        state = FlowState.start(make_packet(timestamp_us=0, win_size=65535))
        assert compute_tier1_features(state)["init_win_bytes_fwd"].value == 65535
        assert compute_tier1_features(state)["init_win_bytes_bwd"].value is None


class TestLowConfidenceFlagging:
    def test_high_resolution_flow_not_flagged(self):
        state = two_way_flow(resolution=1)
        features = compute_tier1_features(state)
        assert features["flow_duration"].low_confidence is False
        assert features["flow_iat_mean"].low_confidence is False

    def test_second_resolution_not_flagged(self):
        state = two_way_flow(resolution=1_000_000)
        features = compute_tier1_features(state)
        assert features["flow_duration"].low_confidence is False

    def test_minute_resolution_flags_timing_features_only(self):
        state = two_way_flow(resolution=60_000_000)
        features = compute_tier1_features(state)
        assert features["flow_duration"].low_confidence is True
        assert features["flow_bytes_per_sec"].low_confidence is True
        assert features["flow_iat_mean"].low_confidence is True
        # non-timing features are unaffected
        assert features["syn_ratio"].low_confidence is False
        assert features["fwd_pkt_len_mean"].low_confidence is False


class TestShouldRunSelector:
    def test_fires_every_n_packets(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        for _ in range(6):
            state.update(make_packet(timestamp_us=1))
        assert state.total_pkt_count == 7
        assert not should_run_selector(state, check_every=8)
        state.update(make_packet(timestamp_us=1))
        assert state.total_pkt_count == 8
        assert should_run_selector(state, check_every=8)

    def test_always_fires_on_termination(self):
        state = FlowState.start(make_packet(timestamp_us=0))
        state.update(make_packet(timestamp_us=1, rst=True))
        assert state.total_pkt_count == 2  # not a multiple of 8
        assert should_run_selector(state, check_every=8)


class TestSelectorEscalation:
    def test_any_rule_escalates_on_single_crossing(self):
        config = SelectorConfig(
            thresholds={"flow_bytes_per_sec": FeatureThreshold(high=1000)},
            rule=EscalationRule.ANY,
        )
        state = two_way_flow(fwd_len=100, bwd_len=200, gap_us=1)  # huge rate
        decision = Selector(config).evaluate(state)
        assert decision.escalate
        assert len(decision.trigger_reasons) == 1
        assert decision.trigger_reasons[0].feature == "flow_bytes_per_sec"

    def test_no_escalation_when_within_threshold(self):
        config = SelectorConfig(
            thresholds={"flow_bytes_per_sec": FeatureThreshold(high=10**12)},
        )
        state = two_way_flow(gap_us=1000)
        decision = Selector(config).evaluate(state)
        assert not decision.escalate
        assert decision.trigger_reasons == []

    def test_k_of_n_requires_k_distinct_features(self):
        config = SelectorConfig(
            thresholds={
                "flow_bytes_per_sec": FeatureThreshold(high=1),
                "flow_pkts_per_sec": FeatureThreshold(high=1),
                "syn_ratio": FeatureThreshold(high=10),  # never crosses
            },
            rule=EscalationRule.K_OF_N,
            k=2,
        )
        state = two_way_flow(gap_us=1)
        decision = Selector(config).evaluate(state)
        assert decision.escalate  # both rate features cross

    def test_k_of_n_does_not_escalate_below_k(self):
        config = SelectorConfig(
            thresholds={
                "flow_bytes_per_sec": FeatureThreshold(high=1),
                "syn_ratio": FeatureThreshold(high=10),
            },
            rule=EscalationRule.K_OF_N,
            k=2,
        )
        state = two_way_flow(gap_us=1)
        decision = Selector(config).evaluate(state)
        assert not decision.escalate  # only 1 of 2 required features crossed

    def test_low_side_threshold_crossing(self):
        config = SelectorConfig(
            thresholds={"fwd_pkt_len_mean": FeatureThreshold(low=50)},
        )
        state = two_way_flow(fwd_len=10, bwd_len=200)
        decision = Selector(config).evaluate(state)
        assert decision.escalate
        assert decision.trigger_reasons[0].direction == "low"

    def test_ratio_for_high_crossing(self):
        config = SelectorConfig(thresholds={"flow_bytes_per_sec": FeatureThreshold(high=1000)})
        state = two_way_flow(fwd_len=100, bwd_len=200, gap_us=1000)  # 300,000 bytes/sec
        decision = Selector(config).evaluate(state)
        assert decision.trigger_reasons[0].ratio == 300.0

    def test_ratio_for_low_crossing(self):
        config = SelectorConfig(thresholds={"fwd_pkt_len_mean": FeatureThreshold(low=200)})
        state = two_way_flow(fwd_len=50, bwd_len=200)
        decision = Selector(config).evaluate(state)
        assert decision.trigger_reasons[0].ratio == 4.0  # 200 / 50

    def test_zero_threshold_high_crossing_reports_infinite_ratio(self):
        config = SelectorConfig(thresholds={"syn_ratio": FeatureThreshold(high=0)})
        state = FlowState.start(make_packet(timestamp_us=0, syn=True))
        decision = Selector(config).evaluate(state)
        assert decision.trigger_reasons[0].ratio == float("inf")

    def test_undefined_feature_never_triggers(self):
        config = SelectorConfig(thresholds={"flow_iat_mean": FeatureThreshold(high=0)})
        state = FlowState.start(make_packet(timestamp_us=0))  # single packet: iat_mean undefined
        decision = Selector(config).evaluate(state)
        assert not decision.escalate

    def test_trigger_reason_carries_low_confidence_flag(self):
        config = SelectorConfig(thresholds={"flow_duration": FeatureThreshold(high=0)})
        state = two_way_flow(resolution=60_000_000)
        decision = Selector(config).evaluate(state)
        assert decision.trigger_reasons[0].low_confidence is True

    def test_feature_with_no_configured_threshold_is_ignored(self):
        config = SelectorConfig(thresholds={})
        state = two_way_flow(gap_us=1)
        decision = Selector(config).evaluate(state)
        assert not decision.escalate
        assert decision.trigger_reasons == []

    def test_src_features_included_when_src_table_given(self):
        src_table = SrcTable()
        src_table.note_flow_start("10.0.0.1", "10.0.0.9", 1, 0)
        src_table.note_flow_start("10.0.0.1", "10.0.0.9", 2, 0)
        src_table.note_flow_start("10.0.0.1", "10.0.0.9", 3, 0)
        config = SelectorConfig(thresholds={"distinct_dst_ports_per_src": FeatureThreshold(high=2)})
        state = two_way_flow()
        decision = Selector(config).evaluate(state, src_table=src_table, now_us=0)
        assert decision.escalate
        assert "flows_per_src" in decision.features


class TestConfigSerialization:
    def test_round_trips_through_dict(self):
        config = SelectorConfig(
            thresholds={"flow_duration": FeatureThreshold(high=100, low=1)},
            rule=EscalationRule.K_OF_N,
            k=3,
            config_hash="abc123",
        )
        restored = SelectorConfig.from_dict(config.to_dict())
        assert restored == config
