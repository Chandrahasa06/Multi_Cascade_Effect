"""Adapts a real packet capture (pcapng) into the abstract ``Packet``
stream that :class:`dataplane.flow_table.FlowTable` consumes.

This is the peer of ``adapters/csv_flow_adapter.py`` referenced in that
module's docstring: FlowTable.process() takes one Packet at a time and
knows nothing about where packets come from, so this adapter drives the
exact same table with no changes there. Where the CSV adapter had to
*synthesize* a plausible packet sequence from CICFlowMeter's aggregated
flow records, this adapter has the real thing — every Packet here is a
direct, lossless reading of one observed frame (real inter-arrival
timing, real per-packet lengths, real per-packet flag placement) rather
than an approximation built to match summary statistics.

PARSING STRATEGY: direct byte-slicing, not a full packet-dissection
library. An earlier version of this module used
:class:`scapy.utils.PcapReader`, which is correct but processes ~230
packets/sec on this hardware because it fully dissects every layer of
every packet (builds an Ethernet/IP/TCP object graph, validates fields,
etc.) even though the flow table only needs ~40 bytes of header out of
each frame. At that rate, Monday's 11.7M-packet capture is a ~14 hour
run. This module instead extends the same sequential pcapng
block-parser used to verify the raw captures (packet count, time range,
timestamp resolution) — that parser handled both ~10GB files end to end
in under a minute doing header-only reads — to also byte-slice the
handful of fields a 5-tuple flow table actually consumes:

  - timestamp: straight from the Enhanced Packet Block header (two
    32-bit words + the interface's ``if_tsresol``), no packet-body
    parsing needed at all.
  - EtherType at offset 12-13 (or 16-17 if an 0x8100 VLAN tag pushed
    everything forward by 4 bytes); anything but 0x0800 (IPv4) is
    skipped and counted — see "Only IPv4 TCP/UDP" below.
  - IPv4 header length from the low nibble of the first header byte
    (IHL, in 32-bit words) — needed to find where the transport header
    starts, since IPv4 options make this variable.
  - protocol at IP header byte 9 (6=TCP, 17=UDP, else skipped+counted).
  - source/destination IP from IP header bytes 12-19.
  - source/destination port from the transport header's first 4 bytes
    (same layout for TCP and UDP).
  - for TCP: flags at transport header byte 13, window at bytes 14-15.
  - IP Total Length (header byte 2-3) — see "Packet length" below.

No :class:`Packet` field needs anything past that, so nothing past that
is read. scapy remains a project dependency and is used as a
correctness reference in ``tests/test_pcap_adapter.py`` (its
byte-for-byte-independent decode of the same capture is what the
correctness gate compares this fast parser against), but is no longer
imported here.

PACKET LENGTH is the IPv4 Total Length field (payload + IP header,
excluding the Ethernet frame around it), not the on-the-wire Ethernet
frame size. This matches CICFlowMeter's own packet-length convention
(what ``adapters/csv_flow_adapter.py``'s reconstructed lengths are
ultimately trying to approximate), and is what the byte-offset spec
this module was built against calls out explicitly. Earlier revisions
of this module used the Ethernet frame's on-the-wire length instead
(scapy's ``wirelen``, equivalently the pcapng EPB's Original Packet
Length field) — a different number by exactly the size of the Ethernet
header (14 bytes, 18 with one VLAN tag) on every packet. Anything
comparing flow-level byte counters against a run captured before this
change will see that fixed per-packet offset compounded across every
counter.

Only IPv4 TCP and UDP packets are turned into flow-table events. Real
captures carry protocols a 5-tuple flow table has no defined behavior
for:

  - Non-IPv4 frames (ARP, IPv6, ...) have no notion of "port," which the
    table's canonical key requires.
  - IPv4 protocols other than TCP/UDP (ICMP, etc.) likewise have no
    ports; inventing a port encoding for them (e.g. NetFlow's
    type/code-as-port convention) is a real design choice with real
    consequences for the eventual feature set, not a detail to default
    silently — deferred rather than guessed here.
  - A single 802.1Q VLAN tag (EtherType 0x8100) is unwrapped by
    skipping its 4 bytes and re-reading EtherType past it. Double
    tagging (QinQ) and other EtherType-preceding encapsulations are not
    handled and fall through to "non_ipv4" — not expected on this
    dataset's single-interface testbed capture, uncounted separately
    from genuine ARP/IPv6 traffic if present.
  - Non-first IP fragments (fragment offset != 0 in the IP header) carry
    no transport header — the real TCP/UDP header only exists once, in
    the first fragment; later fragments are a raw continuation of the
    datagram's payload. Skipped as ``"ip_fragment"`` rather than parsed
    as a bogus port pair. Found by the correctness gate against the
    scapy reference in ``tests/test_pcap_adapter.py`` (2 packets in the
    60s test slice) — worth knowing this dataset does carry some
    fragmented traffic, however rare.

Every skipped packet is counted and reasoned in ``AdapterStats``, the
same pattern the CSV adapter uses for unusable rows.

This module does not use :class:`FlowTable.KeyMode.EVAL` — that mode's
discriminator is a CSV row index, and a real capture has no such row to
key on (see ``FlowTable.KeyMode`` and the README's "Known Limitations"
section). Every Packet's ``source_row_id`` is left ``None`` accordingly;
a caller driving a table from this adapter gets FIDELITY-mode behavior
(or whatever KeyMode it explicitly asks for), never EVAL.
"""
from __future__ import annotations

import socket
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

#: bump whenever packet-to-Packet translation logic changes (field
#: mapping, skip rules, timestamp handling, length semantics, ...) —
#: cached feature extractions key on this the same way they key on
#: adapters.csv_flow_adapter.ADAPTER_VERSION. See eval/simulate.py.
#: Bumped from "1": packet length changed from Ethernet wire length to
#: IPv4 Total Length (see module docstring), and the reader itself
#: changed from scapy to direct byte-slicing.
ADAPTER_VERSION = "2"

from dataplane.flow_state import UNSET, Packet

# pcapng block types.
_SHB = 0x0A0D0D0A
_IDB = 0x00000001
_EPB = 0x00000006

_READ_CHUNK = 32 * 1024 * 1024
_COMPACT_THRESHOLD = 64 * 1024 * 1024
#: bytes of packet payload copied out of the read buffer per packet.
#: Worst case needed: 18 (VLAN-tagged Ethernet) + 60 (max IPv4 header
#: with options) + 16 (TCP header through the window field) = 94; give
#: it headroom.
_HEADER_PEEK_BYTES = 128


def _tsresol_ratio(opt_value: int) -> Tuple[int, int]:
    """Decode a pcapng ``if_tsresol`` option byte into an exact
    (microseconds-per-tick numerator, denominator) pair, so
    tick-to-microsecond conversion is done in integer arithmetic with no
    floating-point rounding risk — important since this value is
    multiplied against tick counts as large as ~1.5e15 for a real
    capture's epoch timestamp. MSB=0 means resol=10^-opt_value seconds
    per tick (the common case: opt_value=6 is exactly 1 tick = 1
    microsecond, the confirmed resolution of both CICIDS2017 captures —
    see STATUS.md); MSB=1 means resol=2^-(opt_value & 0x7f) seconds per
    tick (rare, power-of-two resolutions)."""
    if opt_value & 0x80:
        exp = opt_value & 0x7F
        return 1_000_000, 2 ** exp
    if opt_value <= 6:
        return 10 ** (6 - opt_value), 1
    return 1, 10 ** (opt_value - 6)


#: block_type, block_total_len, interface_id, ts_high, ts_low, cap_len —
#: the 8-byte block header plus an Enhanced Packet Block's 5 fixed
#: 32-bit fields, minus the trailing (unused here) orig_len, in one
#: struct read instead of two. EPB is by far the dominant block type in
#: any real capture (millions of packets vs. one SHB and a handful of
#: IDBs), so this is the layout the hot path is shaped around; SHB/IDB
#: blocks fall back to a slower, separate parse below.
_EPB_PREFIX = struct.Struct("<IIIIII")
_EPB_PREFIX_LEN = 24  # bytes covered by _EPB_PREFIX — less than the 28
#: bytes to the start of packet data (the skipped orig_len field makes
#: up the difference); data_start is computed independently, not by
#: extending this struct to cover it.


def _parse_idb_ratio(buf: bytearray, pos: int, block_total_len: int) -> Tuple[int, int]:
    """Extract the ``if_tsresol`` option (or the spec default) from one
    Interface Description Block already known to span
    ``buf[pos:pos+block_total_len]``."""
    body_len = block_total_len - 12
    opt_off = pos + 16  # block header(8) + linktype/reserved/snaplen(8)
    opt_end = pos + 8 + body_len
    while opt_off + 4 <= opt_end:
        opt_code, opt_len = struct.unpack_from("<HH", buf, opt_off)
        if opt_code == 0 and opt_len == 0:
            break
        if opt_code == 9 and opt_len >= 1:
            return _tsresol_ratio(buf[opt_off + 4])
        opt_off += 4 + (opt_len + 3) // 4 * 4
    return 1, 1  # spec default: 10^-6 s/tick = 1us/tick


def _iter_pcapng_packet_records(
    path: Union[str, Path], peek_bytes: int = _HEADER_PEEK_BYTES
) -> Iterator[Tuple[int, int, bytes]]:
    """Yields ``(timestamp_us, resolution_us, header_bytes)`` for every
    Enhanced Packet Block in a pcapng file, in file order.

    ``peek_bytes`` defaults to ``_HEADER_PEEK_BYTES`` (just enough for
    this module's own header-only parsing) but callers that need real
    packet content — e.g. controlplane/extractor.py re-extracting full
    packets for CICFlowMeter, which reads ``len(packet)`` for its
    byte-count features and so cannot work from a truncated peek — can
    pass a larger value (up to 65535, the maximum possible IP packet
    size) to get the whole captured packet back instead.

    Sequential, buffered, single pass — the same block-walking approach
    used to verify the raw captures (packet count / time range /
    resolution) before any adapter code existed, extended here to also
    hand back the first ``peek_bytes`` of each packet's captured
    data instead of skipping over it. No per-packet disk seeks; the
    32MB read chunks make this bound by sequential disk throughput, not
    per-packet syscall count.

    The hot path (an Enhanced Packet Block, i.e. almost every block in a
    real capture) speculatively ensures and unpacks the block header and
    EPB's fixed fields together in one buffered-read check and one
    ``struct.unpack_from`` call, instead of one of each for the header
    and a second round for the EPB body — profiling showed this pair
    doubling up as the largest single cost after the flow table itself.
    SHB/IDB blocks (a handful per file, not millions) fall back to a
    plainer two-step parse.
    """
    f = open(path, "rb", buffering=1024 * 1024 * 4)
    buf = bytearray()
    pos = 0

    def ensure(n: int) -> bool:
        while len(buf) - pos < n:
            chunk = f.read(_READ_CHUNK)
            if not chunk:
                return False
            buf.extend(chunk)
        return True

    def compact() -> None:
        nonlocal pos
        if pos > _COMPACT_THRESHOLD:
            del buf[:pos]
            pos = 0

    ratio_by_if: Dict[int, Tuple[int, int]] = {}
    unpack_prefix = _EPB_PREFIX.unpack_from

    try:
        while True:
            if not ensure(_EPB_PREFIX_LEN):
                # not enough left for a full speculative EPB-shaped read;
                # could be a legitimately small trailing block, or true
                # EOF — fall back to the minimal 8-byte header check.
                if not ensure(8):
                    break
                block_type, block_total_len = struct.unpack_from("<II", buf, pos)
                if block_total_len < 12:
                    raise ValueError(
                        f"corrupt pcapng block (declared total_len={block_total_len})"
                    )
                if not ensure(block_total_len):
                    break
                if block_type == _SHB:
                    ratio_by_if = {}
                elif block_type == _IDB:
                    ratio_by_if[len(ratio_by_if)] = _parse_idb_ratio(buf, pos, block_total_len)
                pos += block_total_len
                compact()
                continue

            block_type, block_total_len, if_id, ts_high, ts_low, cap_len = unpack_prefix(
                buf, pos
            )
            if block_total_len < 12:
                raise ValueError(
                    f"corrupt pcapng block (declared total_len={block_total_len})"
                )

            if block_type != _EPB:
                # SHB or IDB, occasionally something else — slow path.
                if not ensure(block_total_len):
                    break
                if block_type == _SHB:
                    ratio_by_if = {}
                elif block_type == _IDB:
                    ratio_by_if[len(ratio_by_if)] = _parse_idb_ratio(buf, pos, block_total_len)
                pos += block_total_len
                compact()
                continue

            if not ensure(block_total_len):
                break  # truncated final block; stop rather than raise

            mult, div = ratio_by_if.get(if_id, (1, 1))
            ts_us = ((ts_high << 32) | ts_low) * mult // div
            # microseconds per tick, floored at 1 for resolutions finer
            # than a microsecond (a conservative approximation — see
            # _tsresol_ratio docstring; not exercised by either real
            # capture, both confirmed exactly 1 tick = 1 microsecond).
            resolution_us = mult // div if mult >= div else 1
            data_start = pos + 28  # block header(8) + 5 x uint32 fixed fields
            peek = cap_len if cap_len < peek_bytes else peek_bytes
            header_bytes = bytes(buf[data_start: data_start + peek])
            yield ts_us, resolution_us, header_bytes

            pos += block_total_len
            compact()
    finally:
        f.close()


def _read_tsresol_us(path: Union[str, Path]) -> int:
    """The capture's timestamp-tick resolution in microseconds, read
    from the first Enhanced Packet Block's interface. Kept as a small
    standalone helper (rather than requiring a caller to pull one item
    out of :func:`_iter_pcapng_packet_records` itself) since it's used
    on its own as a quick diagnostic — see tests."""
    gen = _iter_pcapng_packet_records(path)
    try:
        first = next(gen, None)
    finally:
        gen.close()
    return first[1] if first else 1


def _parse_ipv4_tcp_udp(
    data: bytes,
) -> Tuple[Optional[tuple], Optional[str]]:
    """Byte-sliced parse of one Ethernet frame's IPv4/TCP or IPv4/UDP
    headers. Returns ``(fields, None)`` on success — where ``fields`` is
    ``(src_ip, dst_ip, src_port, dst_port, protocol, syn, fin, rst, psh,
    ack, urg, win_size, total_len)`` — or ``(None, skip_reason)``.

    Deliberately reads only the fixed-offset header bytes a 5-tuple flow
    table needs (see module docstring for the exact offsets and why);
    never touches payload bytes, never allocates a parsed-object graph
    per packet."""
    n = len(data)
    if n < 14:
        return None, "truncated_ethernet"

    eth_offset = 14
    ethertype = (data[12] << 8) | data[13]
    if ethertype == 0x8100:
        if n < 18:
            return None, "truncated_vlan"
        eth_offset = 18
        ethertype = (data[16] << 8) | data[17]

    if ethertype != 0x0800:
        return None, "non_ipv4"

    if n < eth_offset + 20:
        return None, "truncated_ip"
    ip_off = eth_offset
    version_ihl = data[ip_off]
    if version_ihl >> 4 != 4:
        return None, "non_ipv4"  # EtherType claimed IPv4 but version nibble disagrees
    ihl = (version_ihl & 0x0F) * 4
    if ihl < 20 or n < ip_off + ihl:
        return None, "truncated_ip"

    total_len = (data[ip_off + 2] << 8) | data[ip_off + 3]
    protocol = data[ip_off + 9]
    src_ip = socket.inet_ntoa(bytes(data[ip_off + 12: ip_off + 16]))
    dst_ip = socket.inet_ntoa(bytes(data[ip_off + 16: ip_off + 20]))

    # Non-first IP fragments (fragment offset != 0) carry no transport
    # header at all -- their payload is a raw continuation of the
    # datagram, not a fresh TCP/UDP header. Reading src/dst port out of
    # that would be reading garbage. Only the first fragment (offset 0,
    # whether or not More-Fragments is set) has real header bytes here.
    frag_field = (data[ip_off + 6] << 8) | data[ip_off + 7]
    fragment_offset = frag_field & 0x1FFF
    if fragment_offset != 0:
        return None, "ip_fragment"

    l4_off = ip_off + ihl
    if protocol == 6:  # TCP
        if n < l4_off + 16:
            return None, "truncated_tcp"
        src_port = (data[l4_off] << 8) | data[l4_off + 1]
        dst_port = (data[l4_off + 2] << 8) | data[l4_off + 3]
        flags_byte = data[l4_off + 13]
        fin = bool(flags_byte & 0x01)
        syn = bool(flags_byte & 0x02)
        rst = bool(flags_byte & 0x04)
        psh = bool(flags_byte & 0x08)
        ack = bool(flags_byte & 0x10)
        urg = bool(flags_byte & 0x20)
        win_size = (data[l4_off + 14] << 8) | data[l4_off + 15]
    elif protocol == 17:  # UDP
        if n < l4_off + 4:
            return None, "truncated_udp"
        src_port = (data[l4_off] << 8) | data[l4_off + 1]
        dst_port = (data[l4_off + 2] << 8) | data[l4_off + 3]
        syn = fin = rst = psh = ack = urg = False
        win_size = UNSET
    else:
        return None, f"unsupported_transport:{protocol}"

    return (
        src_ip, dst_ip, src_port, dst_port, protocol,
        syn, fin, rst, psh, ack, urg, win_size, total_len,
    ), None


@dataclass
class AdapterStats:
    packets_total: int = 0
    packets_skipped: int = 0
    packets_emitted: int = 0
    skip_reasons: Counter = field(default_factory=Counter)
    #: count of packets whose capture timestamp went backwards relative
    #: to the previous packet fed to this adapter's output stream; each
    #: is clamped to the previous packet's timestamp rather than passed
    #: through, since Packet.timestamp_us must be non-decreasing for
    #: FlowTable. True packet captures are overwhelmingly monotonic
    #: already (single interface, kernel/NIC capture ordering) — this
    #: guards against the rare exception rather than expecting it.
    non_monotonic_timestamps: int = 0
    #: mirrors adapters.csv_flow_adapter.AdapterStats.resolution_counts;
    #: a pcapng capture normally has one uniform resolution for its one
    #: interface, so this is almost always a single-key Counter, but kept
    #: as a Counter (not a scalar) for the same reason the CSV adapter
    #: keeps one: a mixed-resolution file would be worth knowing about,
    #: not silently averaged away.
    resolution_counts: Counter = field(default_factory=Counter)

    @property
    def timestamp_resolution_us(self) -> Optional[int]:
        if len(self.resolution_counts) == 1:
            return next(iter(self.resolution_counts))
        return None

    def as_dict(self) -> Dict[str, object]:
        return {
            "packets_total": self.packets_total,
            "packets_skipped": self.packets_skipped,
            "packets_emitted": self.packets_emitted,
            "skip_reasons": dict(self.skip_reasons),
            "non_monotonic_timestamps": self.non_monotonic_timestamps,
            "resolution_counts": dict(self.resolution_counts),
        }


def iter_packets_from_pcap(
    path: Union[str, Path], stats: Optional[AdapterStats] = None
) -> Iterator[Packet]:
    """Yield one :class:`~dataplane.flow_state.Packet` per IPv4 TCP/UDP
    frame in a capture, in capture order — a direct reading, not a
    synthesis (see module docstring). Every packet's ``source_row_id``
    is ``None``; there is no CSV row here to trace back to."""
    if stats is None:
        stats = AdapterStats()
    last_ts_us: Optional[int] = None

    for ts_us, resolution_us, header_bytes in _iter_pcapng_packet_records(path):
        stats.packets_total += 1
        parsed, skip_reason = _parse_ipv4_tcp_udp(header_bytes)
        if parsed is None:
            stats.packets_skipped += 1
            stats.skip_reasons[skip_reason] += 1
            continue
        (
            src_ip, dst_ip, src_port, dst_port, protocol,
            syn, fin, rst, psh, ack, urg, win_size, total_len,
        ) = parsed

        if last_ts_us is not None and ts_us < last_ts_us:
            stats.non_monotonic_timestamps += 1
            ts_us = last_ts_us
        last_ts_us = ts_us

        stats.resolution_counts[resolution_us] += 1
        stats.packets_emitted += 1
        yield Packet(
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            timestamp_us=ts_us,
            length=total_len,
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
