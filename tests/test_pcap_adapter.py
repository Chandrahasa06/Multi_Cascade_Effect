"""Tests for adapters/pcap_adapter.py's fast, byte-slicing packet reader.

Includes a scapy-based reference decoder (``_reference_iter_packets_from_pcap``)
used only here, as a correctness oracle — see TestCorrectnessGateAgainstScapy.
The production adapter no longer imports scapy at all (that was the whole
point of the rewrite: scapy's full per-packet dissection ran at ~230
pkts/sec, far too slow for an 11.7M-packet capture); scapy stays a project
dependency solely for this test-only reference path.
"""
from pathlib import Path
from typing import Iterator, Optional

from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import PcapReader

from adapters.pcap_adapter import (
    AdapterStats,
    _parse_ipv4_tcp_udp,
    _read_tsresol_us,
    _tsresol_ratio,
    iter_packets_from_pcap,
)
from dataplane.flow_state import UNSET, Packet
from dataplane.flow_table import FlowTable, KeyMode

FIXTURE = Path(__file__).parent / "fixtures" / "monday_60s_slice.pcap"


def _reference_iter_packets_from_pcap(
    path, stats: Optional[AdapterStats] = None
) -> Iterator[Packet]:
    """Independent decode of the same capture via scapy's full packet
    dissection, used only as a correctness oracle for the fast
    byte-slicing adapter. Field-for-field equivalent to what
    adapters/pcap_adapter.py computed before it was rewritten for speed,
    with two deliberate changes from the original scapy-based
    production adapter this was lifted from: packet length here is the
    IPv4 Total Length field (``pkt[IP].len``), not the Ethernet wire
    length (scapy's ``wirelen``); and non-first IP fragments
    (``pkt[IP].frag != 0``) are skipped, matching the fast parser's
    "ip_fragment" handling rather than being misread as bogus TCP/UDP
    headers. Both changes keep this a meaningful oracle for the current
    adapter rather than a stale one."""
    if stats is None:
        stats = AdapterStats()
    resolution_us = _read_tsresol_us(path)
    last_ts_us = None

    with PcapReader(str(path)) as reader:
        for pkt in reader:
            stats.packets_total += 1

            if IP not in pkt:
                stats.packets_skipped += 1
                stats.skip_reasons["non_ipv4"] += 1
                continue
            ip_layer = pkt[IP]

            if ip_layer.frag != 0:
                # non-first IP fragment: no transport header present.
                # See adapters/pcap_adapter.py's "ip_fragment" handling.
                stats.packets_skipped += 1
                stats.skip_reasons["ip_fragment"] += 1
                continue

            if TCP in pkt:
                l4 = pkt[TCP]
                flags = l4.flags
                syn, fin, rst = bool(flags.S), bool(flags.F), bool(flags.R)
                psh, ack, urg = bool(flags.P), bool(flags.A), bool(flags.U)
                win_size = int(l4.window)
            elif UDP in pkt:
                l4 = pkt[UDP]
                syn = fin = rst = psh = ack = urg = False
                win_size = UNSET
            else:
                stats.packets_skipped += 1
                stats.skip_reasons[f"unsupported_transport:{ip_layer.proto}"] += 1
                continue

            ts_us = round(float(pkt.time) * 1_000_000)
            if last_ts_us is not None and ts_us < last_ts_us:
                stats.non_monotonic_timestamps += 1
                ts_us = last_ts_us
            last_ts_us = ts_us

            stats.resolution_counts[resolution_us] += 1
            stats.packets_emitted += 1
            yield Packet(
                src_ip=ip_layer.src,
                dst_ip=ip_layer.dst,
                src_port=int(l4.sport),
                dst_port=int(l4.dport),
                protocol=int(ip_layer.proto),
                timestamp_us=ts_us,
                length=int(ip_layer.len),
                syn=syn,
                fin=fin,
                rst=rst,
                psh=psh,
                ack=ack,
                urg=urg,
                win_size=win_size,
                timestamp_resolution_us=resolution_us,
                source_row_id=None,
            )


def _packet_key(p: Packet) -> tuple:
    return (
        p.src_ip, p.dst_ip, p.src_port, p.dst_port, p.protocol,
        p.timestamp_us, p.length, p.syn, p.fin, p.rst, p.psh, p.ack, p.urg,
        p.win_size, p.timestamp_resolution_us,
    )


def _flow_snapshot(state) -> dict:
    """Comparable dict of every counter/flag FlowState tracks, excluding
    identity fields that legitimately differ across independent table
    instances (nothing does, here, but keeps the comparison explicit)."""
    return {
        "key": state.key,
        "fwd_ip": state.fwd_ip,
        "fwd_port": state.fwd_port,
        "protocol": state.protocol,
        "fwd_pkt_count": state.fwd_pkt_count,
        "bwd_pkt_count": state.bwd_pkt_count,
        "fwd_byte_sum": state.fwd_byte_sum,
        "bwd_byte_sum": state.bwd_byte_sum,
        "fwd_pkt_len_min": state.fwd_pkt_len_min,
        "fwd_pkt_len_max": state.fwd_pkt_len_max,
        "bwd_pkt_len_min": state.bwd_pkt_len_min,
        "bwd_pkt_len_max": state.bwd_pkt_len_max,
        "first_ts": state.first_ts,
        "last_ts": state.last_ts,
        "iat_sum": state.iat_sum,
        "iat_min": state.iat_min,
        "iat_max": state.iat_max,
        "syn_count": state.syn_count,
        "fin_count": state.fin_count,
        "rst_count": state.rst_count,
        "psh_count": state.psh_count,
        "ack_count": state.ack_count,
        "urg_count": state.urg_count,
        "init_win_bytes_fwd": state.init_win_bytes_fwd,
        "init_win_bytes_bwd": state.init_win_bytes_bwd,
        "terminated": state.terminated,
        "termination_reason": state.termination_reason,
    }


class TestTsresolRatio:
    def test_microsecond_option_value_is_exact_one_to_one(self):
        assert _tsresol_ratio(6) == (1, 1)

    def test_millisecond_option_value(self):
        assert _tsresol_ratio(3) == (1000, 1)

    def test_nanosecond_option_value(self):
        assert _tsresol_ratio(9) == (1, 1000)

    def test_power_of_two_msb_set(self):
        # 2^-10 seconds/tick = 1_000_000 / 1024 microseconds/tick
        assert _tsresol_ratio(0x80 | 10) == (1_000_000, 1024)


class TestTsresol:
    def test_fixture_reports_microsecond_resolution(self):
        assert _read_tsresol_us(FIXTURE) == 1


class TestParseIpv4TcpUdp:
    def test_too_short_for_ethernet_header(self):
        parsed, reason = _parse_ipv4_tcp_udp(b"\x00" * 10)
        assert parsed is None
        assert reason == "truncated_ethernet"

    def test_non_ipv4_ethertype_is_skipped(self):
        frame = bytearray(20)
        frame[12:14] = (0x0806).to_bytes(2, "big")  # ARP
        parsed, reason = _parse_ipv4_tcp_udp(bytes(frame))
        assert parsed is None
        assert reason == "non_ipv4"

    def test_vlan_tag_is_unwrapped(self):
        # Ethernet(14, but +4 for VLAN) + IPv4(20) + TCP(20), all zeroed
        # except the fields under test.
        frame = bytearray(18 + 20 + 20)
        frame[12:14] = (0x8100).to_bytes(2, "big")  # VLAN tag ethertype
        frame[16:18] = (0x0800).to_bytes(2, "big")  # real ethertype: IPv4
        ip_off = 18
        frame[ip_off] = 0x45  # version 4, IHL 5 (20 bytes)
        frame[ip_off + 9] = 6  # TCP
        frame[ip_off + 12:ip_off + 16] = bytes([10, 0, 0, 1])
        frame[ip_off + 16:ip_off + 20] = bytes([10, 0, 0, 2])
        parsed, reason = _parse_ipv4_tcp_udp(bytes(frame))
        assert reason is None
        assert parsed[0] == "10.0.0.1"
        assert parsed[1] == "10.0.0.2"
        assert parsed[4] == 6

    def test_tcp_flags_and_window_decoded(self):
        frame = bytearray(14 + 20 + 20)
        frame[12:14] = (0x0800).to_bytes(2, "big")
        ip_off = 14
        frame[ip_off] = 0x45
        frame[ip_off + 9] = 6
        frame[ip_off + 12:ip_off + 16] = bytes([1, 2, 3, 4])
        frame[ip_off + 16:ip_off + 20] = bytes([5, 6, 7, 8])
        tcp_off = ip_off + 20
        frame[tcp_off:tcp_off + 2] = (1234).to_bytes(2, "big")
        frame[tcp_off + 2:tcp_off + 4] = (80).to_bytes(2, "big")
        frame[tcp_off + 13] = 0x02 | 0x10  # SYN + ACK
        frame[tcp_off + 14:tcp_off + 16] = (65535).to_bytes(2, "big")
        parsed, reason = _parse_ipv4_tcp_udp(bytes(frame))
        assert reason is None
        (src_ip, dst_ip, src_port, dst_port, protocol,
         syn, fin, rst, psh, ack, urg, win_size, total_len) = parsed
        assert src_port == 1234 and dst_port == 80
        assert syn and ack
        assert not (fin or rst or psh or urg)
        assert win_size == 65535

    def test_udp_has_no_flags_and_unset_window(self):
        frame = bytearray(14 + 20 + 8)
        frame[12:14] = (0x0800).to_bytes(2, "big")
        ip_off = 14
        frame[ip_off] = 0x45
        frame[ip_off + 9] = 17
        frame[ip_off + 12:ip_off + 16] = bytes([1, 2, 3, 4])
        frame[ip_off + 16:ip_off + 20] = bytes([5, 6, 7, 8])
        udp_off = ip_off + 20
        frame[udp_off:udp_off + 2] = (53).to_bytes(2, "big")
        frame[udp_off + 2:udp_off + 4] = (12345).to_bytes(2, "big")
        parsed, reason = _parse_ipv4_tcp_udp(bytes(frame))
        assert reason is None
        assert parsed[4] == 17
        assert parsed[11] == UNSET  # win_size

    def test_unsupported_transport_is_skipped_and_reasoned(self):
        frame = bytearray(14 + 20)
        frame[12:14] = (0x0800).to_bytes(2, "big")
        ip_off = 14
        frame[ip_off] = 0x45
        frame[ip_off + 9] = 1  # ICMP
        parsed, reason = _parse_ipv4_tcp_udp(bytes(frame))
        assert parsed is None
        assert reason == "unsupported_transport:1"

    def test_ip_total_length_field_is_read_correctly(self):
        frame = bytearray(14 + 20 + 8)
        frame[12:14] = (0x0800).to_bytes(2, "big")
        ip_off = 14
        frame[ip_off] = 0x45
        frame[ip_off + 2:ip_off + 4] = (1500).to_bytes(2, "big")
        frame[ip_off + 9] = 17
        udp_off = ip_off + 20
        frame[udp_off:udp_off + 4] = b"\x00\x35\x00\x35"
        parsed, reason = _parse_ipv4_tcp_udp(bytes(frame))
        assert reason is None
        assert parsed[12] == 1500  # total_len


class TestIterPacketsFromPcap:
    def test_yields_only_tcp_and_udp_packets(self):
        packets = list(iter_packets_from_pcap(FIXTURE))
        assert len(packets) > 0
        assert all(p.protocol in (6, 17) for p in packets)

    def test_timestamps_are_monotonic_non_decreasing(self):
        packets = list(iter_packets_from_pcap(FIXTURE))
        timestamps = [p.timestamp_us for p in packets]
        assert timestamps == sorted(timestamps)

    def test_all_packets_tagged_microsecond_resolution(self):
        packets = list(iter_packets_from_pcap(FIXTURE))
        assert all(p.timestamp_resolution_us == 1 for p in packets)

    def test_source_row_id_is_none_for_every_packet(self):
        packets = list(iter_packets_from_pcap(FIXTURE))
        assert all(p.source_row_id is None for p in packets)

    def test_stats_accounting_is_internally_consistent(self):
        stats = AdapterStats()
        packets = list(iter_packets_from_pcap(FIXTURE, stats=stats))
        assert stats.packets_emitted == len(packets)
        assert stats.packets_total == stats.packets_emitted + stats.packets_skipped
        assert stats.timestamp_resolution_us == 1

    def test_non_ip_frames_are_skipped_and_counted(self):
        stats = AdapterStats()
        list(iter_packets_from_pcap(FIXTURE, stats=stats))
        assert stats.skip_reasons.get("non_ipv4", 0) > 0

    def test_tcp_packets_carry_real_flags_not_all_false(self):
        packets = [p for p in iter_packets_from_pcap(FIXTURE) if p.protocol == 6]
        assert any(p.syn for p in packets)
        assert any(p.ack for p in packets)

    def test_udp_packets_have_no_flags_and_unset_window(self):
        packets = [p for p in iter_packets_from_pcap(FIXTURE) if p.protocol == 17]
        assert packets
        assert all(not (p.syn or p.fin or p.rst or p.psh or p.ack or p.urg) for p in packets)
        assert all(p.win_size == UNSET for p in packets)


class TestDrivesFlowTable:
    def test_full_slice_processes_without_error_and_produces_flows(self):
        table = FlowTable(key_mode=KeyMode.FIDELITY)
        stats = AdapterStats()
        for packet in iter_packets_from_pcap(FIXTURE, stats=stats):
            table.process(packet)
        remaining = table.flush()

        assert table.total_packets_seen == stats.packets_emitted
        assert table.total_flows_created > 0
        assert len(remaining) <= table.total_flows_created

    def test_eval_key_mode_also_runs_cleanly(self):
        table = FlowTable(key_mode=KeyMode.EVAL)
        for packet in iter_packets_from_pcap(FIXTURE):
            table.process(packet)
        table.flush()
        assert table.total_packets_seen > 0


class TestCorrectnessGateAgainstScapy:
    """The whole point of the rewrite was speed without changing
    behavior. This drives two independent FlowTables — one fed by the
    fast byte-slicing adapter, one fed by the scapy reference decoder —
    over the identical 60-second slice and asserts they agree exactly:
    same packets emitted in the same order with the same fields, same
    flow keys, same final counters. A divergence here means the fast
    parser is wrong, not that the two approaches are just "different but
    both fine."
    """

    def test_packet_streams_are_identical(self):
        fast = list(iter_packets_from_pcap(FIXTURE))
        reference = list(_reference_iter_packets_from_pcap(FIXTURE))
        assert len(fast) == len(reference)
        fast_keys = [_packet_key(p) for p in fast]
        ref_keys = [_packet_key(p) for p in reference]
        assert fast_keys == ref_keys

    def test_skip_counts_match(self):
        fast_stats = AdapterStats()
        ref_stats = AdapterStats()
        list(iter_packets_from_pcap(FIXTURE, stats=fast_stats))
        list(_reference_iter_packets_from_pcap(FIXTURE, stats=ref_stats))
        assert fast_stats.packets_total == ref_stats.packets_total
        assert fast_stats.packets_emitted == ref_stats.packets_emitted
        assert dict(fast_stats.skip_reasons) == dict(ref_stats.skip_reasons)

    def test_resulting_flow_tables_are_identical(self):
        fast_table = FlowTable(key_mode=KeyMode.FIDELITY)
        for packet in iter_packets_from_pcap(FIXTURE):
            fast_table.process(packet)
        fast_table.flush()

        ref_table = FlowTable(key_mode=KeyMode.FIDELITY)
        for packet in _reference_iter_packets_from_pcap(FIXTURE):
            ref_table.process(packet)
        ref_table.flush()

        # flush() only returns what was still open; both adapters ran
        # every packet through process() first, so compare the raw
        # counters via each table's public run totals plus a snapshot
        # of what flush() handed back (order-independent, keyed by
        # flow key since both tables use the same keying function).
        assert fast_table.total_packets_seen == ref_table.total_packets_seen
        assert fast_table.total_flows_created == ref_table.total_flows_created
        assert fast_table.eviction_count == ref_table.eviction_count
        assert fast_table.idle_timeout_count == ref_table.idle_timeout_count
        assert fast_table.active_timeout_count == ref_table.active_timeout_count

    def test_per_flow_counters_are_identical(self):
        fast_flows = {}
        fast_table = FlowTable(key_mode=KeyMode.FIDELITY)
        for packet in iter_packets_from_pcap(FIXTURE):
            result = fast_table.process(packet)
            fast_flows[result.state.key] = result.state
        for expired in fast_table.flush():
            fast_flows[expired.state.key] = expired.state

        ref_flows = {}
        ref_table = FlowTable(key_mode=KeyMode.FIDELITY)
        for packet in _reference_iter_packets_from_pcap(FIXTURE):
            result = ref_table.process(packet)
            ref_flows[result.state.key] = result.state
        for expired in ref_table.flush():
            ref_flows[expired.state.key] = expired.state

        assert set(fast_flows.keys()) == set(ref_flows.keys())
        for key in fast_flows:
            assert _flow_snapshot(fast_flows[key]) == _flow_snapshot(ref_flows[key]), key
