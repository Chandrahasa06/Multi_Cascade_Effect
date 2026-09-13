"""Per-source-IP sliding-window counters.

A single flow looks normal in isolation; a source hammering many
destinations or ports does not. These features feed the selector
alongside the per-flow ones and are fitted the same benign-only way.

Sliding-window implementation note: an exact 60s sliding window would
need to expire individual (source, event) records as they age out —
either a per-event timestamp scan (O(n) in events seen) or an exact
per-flow removal (needs a priority queue per source). Neither is
something a P4 register array can do. Instead this uses **time-bucketed
counters**: each source gets a fixed, small ring of buckets (default 6
buckets x 10s = 60s window); an event increments the *current* bucket,
and reading a feature merges/sums the still-live buckets. Advancing to
a new bucket, and evicting buckets that have aged out of the window, is
O(num_buckets) — a small constant, not O(events) — so the whole thing
is O(1) per event. This is close to what hardware would actually do
with a small counter array per source.

Distinct destination ports/IPs use a :class:`~dataplane.hyperloglog.HyperLogLog`
sketch per bucket (exact sets are not P4-implementable at line rate);
sketches merge across buckets in O(num_buckets) at read time.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from dataplane.hyperloglog import HyperLogLog

DEFAULT_WINDOW_US = 60_000_000
DEFAULT_NUM_BUCKETS = 6
# Independent of the flow table's capacity (see the memory-cost note
# below) — a source is a host, not a flow, and there are far fewer
# concurrent hosts than concurrent flows in practice. LRU-evicted the
# same way the flow table is (see flow_table.py's hardware-fidelity
# note: real hardware would evict on hash collision, not touch-recency).
DEFAULT_SRC_CAPACITY = 8_192
# 256 registers/sketch, ~6.5% standard error (1.04/sqrt(256)). precision=6
# (64 registers, ~13%) was the initial choice but that's too coarse for
# what this sketch actually needs to resolve: telling a benign host
# touching ~3 ports apart from a scanner touching ~30 is exactly a
# small-cardinality distinction, where 13% relative noise is a large
# fraction of the gap.
#
# Memory cost per tracked source, one register byte per HLL slot:
#   2 sketches (port, IP) x 256 registers x 1 byte  = 512 bytes/bucket
#   + ~4 bytes of small counters (flow_count, syn_without_synack_count)
#   = ~516 bytes/bucket x num_buckets=6                = ~3,096 bytes/source
# (see SrcTable.estimated_bytes_per_source). At the flow table's full
# 65,536-flow capacity, one source per flow would be ~203MB — more than
# a single P4 pipeline stage's on-chip SRAM budget (typically tens of
# MB) — which is exactly why DEFAULT_SRC_CAPACITY is its own, smaller,
# independently-configurable number rather than inherited from the flow
# table: at the 8,192-source default this table costs ~25.4MB instead.
DEFAULT_HLL_PRECISION = 8

TCP_PROTOCOL = 6


@dataclass
class _Bucket:
    bucket_id: int
    flow_count: int = 0
    syn_without_synack_count: int = 0
    port_hll: HyperLogLog = field(default_factory=lambda: HyperLogLog(DEFAULT_HLL_PRECISION))
    ip_hll: HyperLogLog = field(default_factory=lambda: HyperLogLog(DEFAULT_HLL_PRECISION))


class SrcTable:
    """Per-source sliding-window register array."""

    def __init__(
        self,
        window_us: int = DEFAULT_WINDOW_US,
        num_buckets: int = DEFAULT_NUM_BUCKETS,
        hll_precision: int = DEFAULT_HLL_PRECISION,
        capacity: int = DEFAULT_SRC_CAPACITY,
    ) -> None:
        if window_us <= 0 or num_buckets <= 0:
            raise ValueError("window_us and num_buckets must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.window_us = window_us
        self.num_buckets = num_buckets
        self.bucket_width_us = max(1, window_us // num_buckets)
        self.hll_precision = hll_precision
        self.capacity = capacity
        self._sources: "OrderedDict[str, Deque[_Bucket]]" = OrderedDict()
        self.eviction_count = 0

    @property
    def sketch_standard_error(self) -> float:
        """Relative standard error of the HLL sketches backing the
        distinct-port/distinct-IP counts, e.g. 0.065 for 6.5% at the
        default precision=8 (256 registers)."""
        return HyperLogLog(self.hll_precision).standard_error

    @property
    def estimated_bytes_per_source(self) -> int:
        """Register-array cost per tracked source: 2 HLL sketches +
        small integer counters, times num_buckets."""
        sketch_bytes = 2 * (1 << self.hll_precision)
        counter_bytes = 4  # flow_count + syn_without_synack_count
        return (sketch_bytes + counter_bytes) * self.num_buckets

    @property
    def estimated_total_bytes(self) -> int:
        """Worst-case register-array cost at full capacity."""
        return self.estimated_bytes_per_source * self.capacity

    def _new_bucket(self, bucket_id: int) -> _Bucket:
        return _Bucket(
            bucket_id=bucket_id,
            port_hll=HyperLogLog(self.hll_precision),
            ip_hll=HyperLogLog(self.hll_precision),
        )

    def _evict_stale(self, entry: Deque[_Bucket], now_us: int) -> None:
        if not entry:
            return
        now_bucket_id = now_us // self.bucket_width_us
        while entry and now_bucket_id - entry[0].bucket_id >= self.num_buckets:
            entry.popleft()

    def _touch(self, src_ip: str, timestamp_us: int) -> _Bucket:
        """Get the bucket for this event, creating/rotating as needed. O(1)
        amortized: at most `num_buckets` buckets ever exist per source.
        Also refreshes src_ip's LRU position and evicts the
        least-recently-touched source if this creates a new one over
        capacity — the same touch-recency convenience flow_table.py uses
        (see its hardware-fidelity note: real hardware evicts on hash
        collision, not touch recency)."""
        bucket_id = timestamp_us // self.bucket_width_us
        entry = self._sources.get(src_ip)
        if entry is None:
            if len(self._sources) >= self.capacity:
                self._sources.popitem(last=False)
                self.eviction_count += 1
            entry = deque(maxlen=self.num_buckets)
            self._sources[src_ip] = entry
        else:
            self._sources.move_to_end(src_ip)

        if not entry:
            entry.append(self._new_bucket(bucket_id))
            return entry[-1]

        if bucket_id > entry[-1].bucket_id:
            if bucket_id - entry[-1].bucket_id >= self.num_buckets:
                entry.clear()  # entire window has aged out since the last touch
            entry.append(self._new_bucket(bucket_id))
            self._evict_stale(entry, timestamp_us)
            return entry[-1]

        if bucket_id == entry[-1].bucket_id:
            return entry[-1]

        # event slightly out of order relative to the latest bucket: fold
        # into the matching bucket if it's still in the window, else into
        # the oldest bucket we still have (best-effort, not exact).
        for bucket in entry:
            if bucket.bucket_id == bucket_id:
                return bucket
        return entry[0]

    def note_flow_start(self, src_ip: str, dst_ip: str, dst_port: int, timestamp_us: int) -> None:
        bucket = self._touch(src_ip, timestamp_us)
        bucket.flow_count += 1
        bucket.port_hll.add(dst_port)
        bucket.ip_hll.add(dst_ip)

    def note_flow_closed(
        self,
        src_ip: str,
        timestamp_us: int,
        *,
        protocol: int,
        syn_count: int,
        bwd_pkt_count: int,
    ) -> None:
        """Record a closed flow's outcome. A flow counts as an unanswered
        SYN — the classic half-open scan / brute-force-attempt signature —
        when it is TCP, sent at least one SYN, and never received a single
        packet back. Flow-level state doesn't split flag counts by
        direction (see FlowState), so "no SYN-ACK" is approximated as "no
        response packets at all"; this is the same signal as the Tier-1
        `no_response_flag` feature, rolled up per source instead of per flow.
        """
        if protocol == TCP_PROTOCOL and syn_count >= 1 and bwd_pkt_count == 0:
            bucket = self._touch(src_ip, timestamp_us)
            bucket.syn_without_synack_count += 1

    def flows_per_src(self, src_ip: str, now_us: int) -> int:
        entry = self._sources.get(src_ip)
        if not entry:
            return 0
        self._evict_stale(entry, now_us)
        return sum(b.flow_count for b in entry)

    def syn_without_synack_count(self, src_ip: str, now_us: int) -> int:
        entry = self._sources.get(src_ip)
        if not entry:
            return 0
        self._evict_stale(entry, now_us)
        return sum(b.syn_without_synack_count for b in entry)

    def distinct_dst_ports_per_src(self, src_ip: str, now_us: int) -> float:
        return self._merged_cardinality(src_ip, now_us, "port_hll")

    def distinct_dst_ips_per_src(self, src_ip: str, now_us: int) -> float:
        return self._merged_cardinality(src_ip, now_us, "ip_hll")

    def _merged_cardinality(self, src_ip: str, now_us: int, attr: str) -> float:
        entry = self._sources.get(src_ip)
        if not entry:
            return 0.0
        self._evict_stale(entry, now_us)
        if not entry:
            return 0.0
        merged: Optional[HyperLogLog] = None
        for bucket in entry:
            sketch = getattr(bucket, attr)
            merged = sketch if merged is None else merged.merge(sketch)
        return merged.cardinality() if merged is not None else 0.0
