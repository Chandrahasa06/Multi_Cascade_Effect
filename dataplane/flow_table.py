"""Fixed-capacity, 5-tuple-keyed flow table with O(1) amortized eviction
and timeout handling.

A real switch has a finite register array (default 65,536 entries here,
matching a modest P4 flow-table size). The table is kept as an
``OrderedDict`` whose iteration order equals *last-touched* order (every
update moves its flow to the back). That lets both idle-timeout expiry
and capacity eviction pop from the front in O(1) amortized time, without
ever scanning the whole table on a per-packet basis.

NOTE ON HARDWARE FIDELITY: the LRU-by-last-touched ``OrderedDict`` is a
*simulation convenience*, not something a P4 switch can do. Real register
arrays are flat, hash-indexed memory with no notion of access recency —
there is no linked list threading entries in touch order. On hardware,
capacity pressure is resolved by **hash collision**: a new flow's index
is computed straight from its 5-tuple, and if that slot is occupied by a
different flow, the incumbent is evicted (or the new flow is dropped),
regardless of which one was touched more recently. This simulator's LRU
policy is a reasonable stand-in for measuring "how often does the table
run out of room," but the *identity* of which flow gets evicted, and the
resulting eviction rate, would differ under real hash-collision eviction.
Worth calling out explicitly in the writeup.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from dataplane.flow_state import FlowState, Packet, canonical_key

DEFAULT_CAPACITY = 65_536
# EVAL mode's default when `capacity` isn't given explicitly. Capacity
# pressure is a question about the data plane's own flow definition —
# that's what FIDELITY mode is for. EVAL mode's job is a clean label per
# flow, so it shouldn't lose flows to eviction by default; `capacity` is
# still a constructor argument in EVAL mode for deliberately exercising
# the eviction path (e.g. in tests).
UNBOUNDED_CAPACITY = sys.maxsize
DEFAULT_ACTIVE_TIMEOUT_US = 120_000_000  # 120s, matches CICFlowMeter default
DEFAULT_IDLE_TIMEOUT_US = 15_000_000  # 15s, matches CICFlowMeter default


class KeyMode(str, Enum):
    """How FlowTable turns a packet into a table key.

    FIDELITY: plain 5-tuple. Packets from different sources (e.g.
    different CSV rows) that happen to share a 5-tuple can merge into one
    simulated flow if the earlier one hasn't timed out yet — this is what
    exercises the table's real merge/timeout/eviction behavior, and is
    the right mode for reporting on that behavior itself.

    EVAL: 5-tuple + Packet.source_row_id. Guarantees one source row maps
    to exactly one simulated flow, so labels stay 1:1 with flows. Use
    this for anything that reports numbers derived from ground-truth
    labels (recall, escalation rate, etc.) — the CSV adapter's
    merge-across-rows behavior is a data-plane-fidelity finding worth
    reporting on its own, not something that should silently contaminate
    a label-based result. Idle/active timeout enforcement is also
    disabled in this mode (see FlowTable.process) — keying alone isn't
    quite enough for an exact 1:1 guarantee, because a row with a long
    real duration but few packets can produce a synthetic inter-packet
    gap over the idle timeout even though it's one continuous flow;
    timeouts would then split it regardless of the key. Capacity
    eviction is unaffected and still applies — and becomes the *only*
    way a flow leaves the table short of a FIN/RST, since idle/active
    timeout no longer does that housekeeping. Confirmed on the real
    Monday file: EVAL mode gives an exact 529,918-rows-to-529,918-flows
    mapping (0 idle timeouts, 0 active timeouts, as designed), but
    capacity eviction then fires 464,310 times at the default 65,536
    capacity — most rows never see a natural close and are evicted
    instead. EVAL mode's eviction count is therefore not a realistic
    data-plane resource-pressure number either; it's an artifact of
    trading timeout-based cleanup for the label-accuracy guarantee.
    """

    FIDELITY = "fidelity"
    EVAL = "eval"


@dataclass(frozen=True, slots=True)
class ExpiredFlow:
    """A flow removed from the table, with the reason it left."""

    state: FlowState
    reason: str  # "fin", "rst", "idle_timeout", "active_timeout", "evicted"


@dataclass(frozen=True, slots=True)
class ProcessResult:
    state: FlowState
    is_new_flow: bool
    expired: List[ExpiredFlow] = field(default_factory=list)


class FlowTable:
    """Simulates a fixed-capacity per-flow register array."""

    def __init__(
        self,
        capacity: Optional[int] = None,
        active_timeout_us: int = DEFAULT_ACTIVE_TIMEOUT_US,
        idle_timeout_us: int = DEFAULT_IDLE_TIMEOUT_US,
        key_mode: KeyMode = KeyMode.FIDELITY,
    ) -> None:
        if capacity is None:
            capacity = DEFAULT_CAPACITY if key_mode == KeyMode.FIDELITY else UNBOUNDED_CAPACITY
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.active_timeout_us = active_timeout_us
        self.idle_timeout_us = idle_timeout_us
        self.key_mode = key_mode

        self._table: "OrderedDict[tuple, FlowState]" = OrderedDict()

        self.total_packets_seen = 0
        self.total_flows_created = 0
        self.eviction_count = 0
        self.idle_timeout_count = 0
        self.active_timeout_count = 0

    def __len__(self) -> int:
        return len(self._table)

    def _compute_key(self, packet: Packet) -> tuple:
        row_discriminator = packet.source_row_id if self.key_mode == KeyMode.EVAL else None
        return canonical_key(
            packet.src_ip,
            packet.dst_ip,
            packet.src_port,
            packet.dst_port,
            packet.protocol,
            row_discriminator=row_discriminator,
        )

    def process(self, packet: Packet) -> ProcessResult:
        """Feed one packet into the table. O(1) amortized.

        Idle/active timeout enforcement is skipped entirely in EVAL mode
        (see KeyMode). EVAL mode's whole purpose is guaranteeing one
        source row maps to exactly one flow; without this, a row with a
        long real duration but few packets — where this adapter's
        even-spaced packet synthesis produces a synthetic inter-packet
        gap over 15s even though it's one CICFlowMeter-recorded flow —
        could self-idle-timeout mid-row (observed on real data: 2 flows
        leaked past a naive 1:1 mapping on the Monday file this way
        before this guard was added). Capacity eviction still applies in
        EVAL mode; it just can never target a row's own still-in-progress
        entry, since this adapter never interleaves another row's packets
        into the middle of one row's own packet block.
        """
        self.total_packets_seen += 1
        now = packet.timestamp_us
        expired = self._expire_idle(now) if self.key_mode == KeyMode.FIDELITY else []

        key = self._compute_key(packet)
        existing = self._table.get(key)

        if (
            self.key_mode == KeyMode.FIDELITY
            and existing is not None
            and (now - existing.first_ts) > self.active_timeout_us
        ):
            existing.terminated = True
            existing.termination_reason = "active_timeout"
            del self._table[key]
            self.active_timeout_count += 1
            expired.append(ExpiredFlow(existing, "active_timeout"))
            existing = None

        is_new = existing is None
        if is_new:
            if len(self._table) >= self.capacity:
                evicted = self._evict_lru()
                if evicted is not None:
                    self.eviction_count += 1
                    expired.append(ExpiredFlow(evicted, "evicted"))
            state = FlowState.start(packet, key=key)
            self._table[key] = state
            self.total_flows_created += 1
        else:
            state = existing
            state.update(packet)
            self._table.move_to_end(key)

        if state.terminated:
            del self._table[key]
            expired.append(ExpiredFlow(state, state.termination_reason or "unknown"))

        return ProcessResult(state=state, is_new_flow=is_new, expired=expired)

    def flush(self) -> List[ExpiredFlow]:
        """Force-close every remaining flow (e.g. end of capture)."""
        expired = []
        for key in list(self._table.keys()):
            state = self._table.pop(key)
            state.terminated = True
            state.termination_reason = state.termination_reason or "flush"
            expired.append(ExpiredFlow(state, state.termination_reason))
        return expired

    def active_flows(self) -> List[FlowState]:
        return list(self._table.values())

    def _expire_idle(self, now: int) -> List[ExpiredFlow]:
        expired: List[ExpiredFlow] = []
        while self._table:
            oldest_key = next(iter(self._table))
            oldest = self._table[oldest_key]
            if now - oldest.last_ts > self.idle_timeout_us:
                del self._table[oldest_key]
                oldest.terminated = True
                oldest.termination_reason = "idle_timeout"
                self.idle_timeout_count += 1
                expired.append(ExpiredFlow(oldest, "idle_timeout"))
            else:
                break
        return expired

    def _evict_lru(self) -> Optional[FlowState]:
        if not self._table:
            return None
        _, victim = self._table.popitem(last=False)
        victim.terminated = True
        victim.termination_reason = "evicted"
        return victim
