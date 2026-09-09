"""Per-flow counter state with O(1) packet updates.

Every field here must map onto a P4 register: integers only, no
variance/std-dev/percentiles, no loops over packet history. Derived
quantities (rates, means, ratios) belong in the selector, computed at
check time from these counters, never stored here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import FrozenSet, Optional

#: sentinel for "no packet observed yet in this direction/field" — real
#: packet lengths, window sizes and IATs are always >= 0, so -1 is safe.
UNSET = -1


class Direction(IntEnum):
    FWD = 0
    BWD = 1


@dataclass(frozen=True, slots=True)
class Packet:
    """A single packet event as seen by the data plane.

    ``timestamp_us`` must be monotonically non-decreasing across the
    stream fed to one :class:`~dataplane.flow_table.FlowTable`.
    """

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # IANA protocol number, e.g. 6=TCP, 17=UDP
    timestamp_us: int
    length: int  # total packet length in bytes
    syn: bool = False
    fin: bool = False
    rst: bool = False
    psh: bool = False
    ack: bool = False
    urg: bool = False
    win_size: int = UNSET  # TCP window size; UNSET if not applicable
    #: size, in microseconds, of one tick of the clock that produced
    #: ``timestamp_us`` — 1 for a real capture (PCAP-derived), larger for
    #: a source that only reconstructs approximate timing (e.g. the CSV
    #: adapter's second/minute-resolution flow-start column). Flows
    #: inherit the coarsest value seen from any contributing packet, and
    #: the selector uses it to flag timing-derived features low-confidence.
    timestamp_resolution_us: int = 1
    #: opaque identifier for whatever produced this packet when that's
    #: not a real capture — e.g. the CSV adapter's row index. None for a
    #: real capture, where there is no "source row" to trace back to.
    #: Carried through to FlowState.source_row_ids regardless of keying
    #: mode, so even a fidelity-mode flow that merged several rows can be
    #: traced back to exactly which ones.
    source_row_id: Optional[int] = None


def canonical_key(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    protocol: int,
    row_discriminator: Optional[int] = None,
) -> tuple:
    """5-tuple key identical for A->B and B->A packets of the same flow.

    ``row_discriminator``, when given, is appended to the key so that
    packets from different sources (e.g. different CSV rows) never
    collide even if they share a 5-tuple — this is what
    :data:`~dataplane.flow_table.KeyMode.EVAL` uses to keep one CSV row
    mapped to exactly one simulated flow. It has no effect on the A->B
    vs B->A symmetry since both directions of one row share the same
    discriminator.
    """
    endpoint_a = (src_ip, src_port)
    endpoint_b = (dst_ip, dst_port)
    if endpoint_a <= endpoint_b:
        lo, hi = endpoint_a, endpoint_b
    else:
        lo, hi = endpoint_b, endpoint_a
    key = (lo[0], hi[0], lo[1], hi[1], protocol)
    if row_discriminator is not None:
        key = key + (row_discriminator,)
    return key


@dataclass(slots=True)
class FlowState:
    """Fixed-width per-flow register state, keyed on the canonical 5-tuple.

    ``fwd_ip``/``fwd_port`` record the source of the first packet seen for
    this flow; that endpoint defines "forward" for the lifetime of the
    flow regardless of which side sends next.
    """

    key: tuple
    fwd_ip: str
    fwd_port: int
    protocol: int

    fwd_pkt_count: int = 0
    bwd_pkt_count: int = 0
    fwd_byte_sum: int = 0
    bwd_byte_sum: int = 0
    fwd_pkt_len_min: int = UNSET
    fwd_pkt_len_max: int = 0
    bwd_pkt_len_min: int = UNSET
    bwd_pkt_len_max: int = 0

    first_ts: int = 0
    last_ts: int = 0
    prev_pkt_ts: int = UNSET

    iat_sum: int = 0
    iat_min: int = UNSET
    iat_max: int = 0

    syn_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    ack_count: int = 0
    urg_count: int = 0

    init_win_bytes_fwd: int = UNSET
    init_win_bytes_bwd: int = UNSET

    #: coarsest timestamp_resolution_us seen from any packet in this flow
    timestamp_resolution_us: int = 1

    #: every distinct Packet.source_row_id that contributed to this flow.
    #: A singleton set is the common case; more than one entry means
    #: multiple source rows were merged into this one simulated flow
    #: (only possible in fidelity-mode keying — see FlowTable.KeyMode).
    source_row_ids: set = field(default_factory=set)

    terminated: bool = False
    termination_reason: Optional[str] = None

    # internal: per-direction FIN tracking to detect a full FIN handshake
    _fin_seen_fwd: bool = False
    _fin_seen_bwd: bool = False

    @classmethod
    def start(cls, packet: Packet, key: Optional[tuple] = None) -> "FlowState":
        """``key`` is normally supplied by FlowTable, which owns the
        keying policy (fidelity vs eval mode); it defaults to the plain
        5-tuple for callers (mostly tests) that construct a FlowState
        directly without going through a table."""
        if key is None:
            key = canonical_key(
                packet.src_ip, packet.dst_ip, packet.src_port, packet.dst_port, packet.protocol
            )
        state = cls(
            key=key,
            fwd_ip=packet.src_ip,
            fwd_port=packet.src_port,
            protocol=packet.protocol,
        )
        state.update(packet)
        return state

    def direction_of(self, packet: Packet) -> Direction:
        if packet.src_ip == self.fwd_ip and packet.src_port == self.fwd_port:
            return Direction.FWD
        return Direction.BWD

    @property
    def total_pkt_count(self) -> int:
        return self.fwd_pkt_count + self.bwd_pkt_count

    @property
    def total_byte_sum(self) -> int:
        return self.fwd_byte_sum + self.bwd_byte_sum

    def update(self, packet: Packet) -> None:
        """O(1) counter update. No loops, no floats, no history scan."""
        direction = self.direction_of(packet)

        if packet.timestamp_resolution_us > self.timestamp_resolution_us:
            self.timestamp_resolution_us = packet.timestamp_resolution_us
        if packet.source_row_id is not None:
            self.source_row_ids.add(packet.source_row_id)

        if self.total_pkt_count == 0:
            self.first_ts = packet.timestamp_us
        self.last_ts = packet.timestamp_us

        if self.prev_pkt_ts != UNSET:
            iat = packet.timestamp_us - self.prev_pkt_ts
            if iat < 0:
                iat = 0
            self.iat_sum += iat
            if self.iat_min == UNSET or iat < self.iat_min:
                self.iat_min = iat
            if iat > self.iat_max:
                self.iat_max = iat
        self.prev_pkt_ts = packet.timestamp_us

        if direction == Direction.FWD:
            self.fwd_pkt_count += 1
            self.fwd_byte_sum += packet.length
            if self.fwd_pkt_len_min == UNSET or packet.length < self.fwd_pkt_len_min:
                self.fwd_pkt_len_min = packet.length
            if packet.length > self.fwd_pkt_len_max:
                self.fwd_pkt_len_max = packet.length
            if self.fwd_pkt_count == 1 and packet.win_size != UNSET:
                self.init_win_bytes_fwd = packet.win_size
        else:
            self.bwd_pkt_count += 1
            self.bwd_byte_sum += packet.length
            if self.bwd_pkt_len_min == UNSET or packet.length < self.bwd_pkt_len_min:
                self.bwd_pkt_len_min = packet.length
            if packet.length > self.bwd_pkt_len_max:
                self.bwd_pkt_len_max = packet.length
            if self.bwd_pkt_count == 1 and packet.win_size != UNSET:
                self.init_win_bytes_bwd = packet.win_size

        if packet.syn:
            self.syn_count += 1
        if packet.psh:
            self.psh_count += 1
        if packet.ack:
            self.ack_count += 1
        if packet.urg:
            self.urg_count += 1

        if packet.rst:
            self.rst_count += 1
            self.terminated = True
            self.termination_reason = "rst"
        if packet.fin:
            self.fin_count += 1
            if direction == Direction.FWD:
                self._fin_seen_fwd = True
            else:
                self._fin_seen_bwd = True
            if self._fin_seen_fwd and self._fin_seen_bwd and not self.terminated:
                self.terminated = True
                self.termination_reason = "fin"
