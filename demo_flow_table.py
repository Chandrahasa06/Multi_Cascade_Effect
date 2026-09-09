"""Manual smoke-test for FlowTable/FlowState on a synthetic packet stream.

No PCAP is available yet, so this stands in for "run it on a slice of
a real capture": a hand-built TCP handshake + a small HTTP-like
request/response + FIN close, plus a second flow, run through the same
table to show keying, counters and eviction bookkeeping together.
"""
from dataplane.flow_state import Packet
from dataplane.flow_table import FlowTable

CLIENT = ("10.0.0.5", 51000)
SERVER = ("93.184.216.34", 80)


def pkt(src, dst, ts, length, **flags):
    return Packet(
        src_ip=src[0], dst_ip=dst[0], src_port=src[1], dst_port=dst[1],
        protocol=6, timestamp_us=ts, length=length, **flags,
    )


def main():
    table = FlowTable(capacity=65536)

    stream = [
        pkt(CLIENT, SERVER, 0, 60, syn=True, win_size=65535),
        pkt(SERVER, CLIENT, 1_000, 60, syn=True, ack=True, win_size=29200),
        pkt(CLIENT, SERVER, 2_000, 52, ack=True),
        pkt(CLIENT, SERVER, 5_000, 500, psh=True, ack=True),
        pkt(SERVER, CLIENT, 15_000, 1200, psh=True, ack=True),
        pkt(CLIENT, SERVER, 16_000, 52, ack=True),
        pkt(CLIENT, SERVER, 20_000, 52, fin=True, ack=True),
        pkt(SERVER, CLIENT, 21_000, 52, fin=True, ack=True),
    ]

    for packet in stream:
        result = table.process(packet)
        for expired in result.expired:
            print(f"  [expired] key={expired.state.key} reason={expired.reason}")

    print(f"packets processed : {table.total_packets_seen}")
    print(f"flows created     : {table.total_flows_created}")
    print(f"flows still open  : {len(table)}")
    print(f"evictions         : {table.eviction_count}")
    print(f"idle timeouts     : {table.idle_timeout_count}")
    print(f"active timeouts   : {table.active_timeout_count}")

    # second, independent flow from a different source to show keying
    other_stream = [
        pkt(("10.0.0.9", 40000), ("93.184.216.34", 443), 0, 60, syn=True),
        pkt(("93.184.216.34", 443), ("10.0.0.9", 40000), 500, 60, syn=True, ack=True),
    ]
    for packet in other_stream:
        table.process(packet)

    expired = table.flush()
    print(f"\nafter flush, {len(expired)} flow(s) force-closed:")
    for e in expired:
        s = e.state
        duration_us = s.last_ts - s.first_ts
        print(
            f"  key={s.key} reason={e.reason} "
            f"fwd_pkts={s.fwd_pkt_count} bwd_pkts={s.bwd_pkt_count} "
            f"fwd_bytes={s.fwd_byte_sum} bwd_bytes={s.bwd_byte_sum} "
            f"duration_us={duration_us} "
            f"iat_sum={s.iat_sum} iat_min={s.iat_min} iat_max={s.iat_max} "
            f"init_win_fwd={s.init_win_bytes_fwd} init_win_bwd={s.init_win_bytes_bwd} "
            f"syn={s.syn_count} fin={s.fin_count} ack={s.ack_count}"
        )


if __name__ == "__main__":
    main()
