"""Adapts CICIDS2017 TrafficLabelling flow-record CSVs into the abstract
``Packet`` stream that :class:`dataplane.flow_table.FlowTable` consumes.

FlowTable.process() takes one Packet at a time and knows nothing about
where packets come from — a PCAP reader can plug into the exact same
table with no changes there. This module is the CSV-shaped source; a
PCAP-shaped source (``adapters/pcap_flow_adapter.py``, to follow once
captures are available) is a peer of this file, not a replacement.

THIS ADAPTER IS NECESSARILY LOSSY. The CSVs are CICFlowMeter's finished,
already-aggregated flow records — packet count, byte sum, min/max
lengths, IAT summary stats, flag *counts* — not the packets themselves.
There is no way to recover the true per-packet sequence from that
summary. Two choices were considered:

  1. Direct flow-state injection: construct a FlowState directly from
     the row's aggregate numbers, bypassing FlowState.update() entirely.
  2. Synthesize a plausible packet sequence per row and feed it through
     FlowTable.process() exactly like any other packet source.

This module takes approach (2). Direct injection would validate nothing
about FlowTable/FlowState beyond dict assignment — the keying, the O(1)
counter-update arithmetic, and the eviction/timeout machinery being
exercised here are the whole point of this checkpoint, and only
approach (2) drives them. The cost is that the synthesized per-packet
detail (exact arrival times within a flow, which packet carries which
TCP flag, byte-for-byte length sequence) is invented to match the row's
*aggregate* statistics, not observed. Concretely:

  - Packet lengths are reconstructed to hit the row's reported min, max,
    and total byte sum exactly; only the interior packets (whatever's
    left after min/max are placed) are spread evenly, which is not
    necessarily how the real capture distributed them.
  - Inter-packet timestamps are spread evenly across the row's
    ``Flow Duration``, not drawn from the real IAT distribution the row
    also reports (Fwd/Bwd/Flow IAT mean/min/max are known but not used
    for this checkpoint's timing).
  - TCP flags are placed by count only (e.g. "2 SYNs" go on the first
    two packets), since the CSV has no per-packet flag record.
  - Forward/backward interleaving is a proportional round-robin on the
    known packet counts, not the real send order.

  One consequence called out explicitly: the selector's "check every N
  packets" cadence (default N=8) cannot be honestly validated against
  this adapter. It will run at synthetic-derived packet indices that
  have no relationship to when a real switch would have seen 8 real
  packets go by. This adapter is good for exercising keying, counter
  arithmetic, timeout/eviction bookkeeping, and CSV-robustness (bad
  rows, zero-duration flows, Infinity/NaN) on real IPs/ports/timing —
  it is not a substitute for validating early-escalation behaviour,
  which needs a real PCAP.

  A second cost, separate from per-flow fidelity: rows are fed to the
  table in file order, and each row's own synthesized packets are fed
  to completion before the next row starts. CICIDS2017 rows are very
  close to chronological (confirmed by inspection), so this preserves
  realistic *ordering*, but it does not reproduce realistic
  *concurrency* — two flows whose real durations overlapped in time are
  not interleaved in the packet stream. That understates how many flows
  are simultaneously open in the table at once, which understates
  capacity-eviction pressure at the default 65,536-entry size. The
  eviction *mechanism* is still exercised faithfully (see the
  artificially small capacity run in the report script), but the
  eviction *rate* measured this way is not a faithful estimate of the
  rate under real traffic and should not be quoted as one.

  A third, related cost surfaced empirically on the real Monday file, and
  it turned out to matter more than it first looked: CICIDS2017 rows are
  not all distinct 5-tuples — the same (src ip, dst ip, src port, dst
  port, protocol) recurs across many rows (NetBIOS/NTP background
  chatter, and on attack days, DoS tools reusing a narrow ephemeral port
  range against the same victim), because CICFlowMeter itself split one
  repeating conversation, or a run of separate attack tools, into
  several flow records. When two such rows land close enough in time
  that this simulator's idle/active timeout hasn't fired yet by the time
  the second row's synthetic packets start, plain-5-tuple keying merges
  them into one continuous simulated flow instead of the separate
  episodes CICFlowMeter recorded. On Monday (all-benign) this merged
  529,918 rows into 409,476 simulated flows with no label consequence —
  but checked against an attack day (Wednesday), 5.4% of merge-groups,
  covering 32.7% of all rows, mixed multiple labels under one 5-tuple
  (BENIGN merged with an attack, or several different attack types
  merged together). Fed through a naive label join, that's not a rounding
  error — it's the join silently attributing one label to traffic that
  was several different things.

  This is a property of the adapter's keying choice, not of the dataset:
  CICFlowMeter had already told us these were separate flows (they're
  separate rows), and folding them back together was our doing.
  FlowTable.KeyMode is the fix — see :class:`~dataplane.flow_table.KeyMode`:

    - ``KeyMode.FIDELITY`` (5-tuple only) is what you want when the
      *question* is "how does the data plane's flow definition behave on
      real traffic" — merging, timeouts, eviction. It's a legitimate
      finding about the data plane, reported as such.
    - ``KeyMode.EVAL`` (5-tuple + the row's index) is what you want for
      anything that reports numbers derived from ground-truth labels:
      it guarantees one CSV row produces exactly one simulated flow, so
      "this flow's label" is never ambiguous. This is the default for
      any script here that produces numbers from labeled data.

  Every synthesized Packet carries ``source_row_id`` regardless of mode
  (see Packet.source_row_id / FlowState.source_row_ids), so even a
  fidelity-mode flow that merged several rows can be traced back to
  exactly which ones — eval mode sidesteps the ambiguity, it doesn't
  erase the visibility into it.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Union

#: bump whenever packet-synthesis logic changes (lengths, timing, flag
#: placement, ...) — cached feature extractions key on this so a code
#: change invalidates the cache instead of silently serving stale
#: features computed under the old synthesis rules. See eval/simulate.py.
ADAPTER_VERSION = "1"

import pandas as pd

from dataplane.flow_state import UNSET, Direction, Packet

REQUIRED_FIELDS = (
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Total Fwd Packets",
    "Total Backward Packets",
)

# CICIDS2017 uses at least two Timestamp formats across days: Monday has
# second resolution ("%d/%m/%Y %H:%M:%S"); Tuesday-Friday drop seconds
# ("%d/%m/%Y %H:%M"). Both are attempted; unparsable timestamps skip the row.
# Each format is paired with the clock resolution it implies, in
# microseconds — this rides along on every Packet synthesized from a row
# parsed with that format (see Packet.timestamp_resolution_us) so
# downstream timing-derived features can be flagged low-confidence on
# coarse-resolution days instead of silently treated as exact.
TIMESTAMP_FORMATS = (
    ("%d/%m/%Y %H:%M:%S", 1_000_000),
    ("%d/%m/%Y %H:%M", 60_000_000),
)

_EPOCH = datetime(1970, 1, 1)


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a TrafficLabelling CSV, stripping the leading-space column
    names. Not every file in the original release is valid UTF-8 — the
    WebAttacks file's Label column uses an en-dash ("Web Attack - Brute
    Force") encoded as Windows-1252, which raises UnicodeDecodeError
    under the default UTF-8 read. Fall back to cp1252, which covers
    every byte value and is the most likely original encoding for a
    dataset released via Windows tooling, rather than silently mojibake
    -ing it with errors="replace"."""
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="cp1252")
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_timestamp(raw: str) -> Optional[tuple]:
    """Returns ``(timestamp_us, resolution_us)``, or None if unparsable."""
    text = str(raw).strip()
    for fmt, resolution_us in TIMESTAMP_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        delta = dt - _EPOCH
        ts_us = delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
        return ts_us, resolution_us
    return None


def parse_timestamp_us(raw: str) -> Optional[int]:
    """Timestamp only, discarding resolution. Prefer :func:`parse_timestamp`
    when the caller needs to track confidence."""
    result = parse_timestamp(raw)
    return result[0] if result else None


def _num(row: dict, col: str, default: float = 0.0) -> float:
    """Coerce a field to a finite float, tolerating missing/NaN/Infinity."""
    val = row.get(col, default)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return default
    if math.isnan(val) or math.isinf(val):
        return default
    return val


def _interleave_directions(n_fwd: int, n_bwd: int) -> List[Direction]:
    """Proportional round-robin over the two known packet counts."""
    seq: List[Direction] = []
    f = b = 0
    while f < n_fwd or b < n_bwd:
        f_ratio = f / n_fwd if n_fwd else float("inf")
        b_ratio = b / n_bwd if n_bwd else float("inf")
        if f_ratio <= b_ratio:
            seq.append(Direction.FWD)
            f += 1
        else:
            seq.append(Direction.BWD)
            b += 1
    return seq


def _synth_lengths(n: int, total_bytes: int, len_min: int, len_max: int) -> List[int]:
    """Per-packet lengths that hit the reported total exactly and touch
    the reported min/max at the endpoints; interior packets split the
    remainder evenly (an approximation — see module docstring)."""
    if n <= 0:
        return []
    total_bytes = max(0, total_bytes)
    len_min = max(0, len_min)
    len_max = max(0, len_max)
    if n == 1:
        return [total_bytes]
    if n == 2:
        return [len_min, max(0, total_bytes - len_min)]

    remaining = total_bytes - len_min - len_max
    middle_n = n - 2
    if remaining < 0:
        # reported min/max don't reconcile with the reported total on
        # this row (seen in real data) — fall back to an even split
        # rather than fabricating negative lengths.
        base, extra = divmod(total_bytes, n)
        return [base + (1 if i < extra else 0) for i in range(n)]
    base, extra = divmod(remaining, middle_n)
    middles = [base + (1 if i < extra else 0) for i in range(middle_n)]
    return [len_min] + middles + [len_max]


def _flag_positions(n: int, k: int, bias: str) -> set:
    """Choose k of n packet indices to carry a flag, per a placement bias."""
    k = max(0, min(k, n))
    if k == 0 or n == 0:
        return set()
    if bias == "start":
        return set(range(k))
    if bias == "end":
        return set(range(n - k, n))
    if k >= n:
        return set(range(n))
    step = n / k
    return {int(i * step) for i in range(k)}


class SynthResult(NamedTuple):
    packets: Optional[List[Packet]]
    skip_reason: Optional[str]
    duration_clamped: bool
    zero_duration: bool
    timestamp_resolution_us: int = 1


def synthesize_packets(row: dict, row_id: int = 0) -> SynthResult:
    """Turn one CICFlowMeter flow-record row into a synthetic packet
    sequence. Returns a skip_reason instead of packets for rows that
    can't be honestly turned into a flow (missing identity fields,
    unparsable timestamp, zero total packets).

    ``row_id`` is stamped onto every packet as ``Packet.source_row_id``
    regardless of the FlowTable KeyMode a caller will use — it's what
    lets a merged fidelity-mode flow be traced back to its source rows,
    and what EVAL-mode keying folds into the table key to keep one row
    mapped to exactly one flow (see the module docstring)."""
    for field_name in REQUIRED_FIELDS:
        val = row.get(field_name)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return SynthResult(None, f"missing_field:{field_name}", False, False)

    try:
        src_ip = str(row["Source IP"])
        dst_ip = str(row["Destination IP"])
        src_port = int(_num(row, "Source Port", -1))
        dst_port = int(_num(row, "Destination Port", -1))
        protocol = int(_num(row, "Protocol", -1))
    except (TypeError, ValueError):
        return SynthResult(None, "unparsable_identity", False, False)
    if src_port < 0 or dst_port < 0 or protocol < 0:
        return SynthResult(None, "unparsable_identity", False, False)

    parsed_ts = parse_timestamp(row["Timestamp"])
    if parsed_ts is None:
        return SynthResult(None, "unparsable_timestamp", False, False)
    ts_us, resolution_us = parsed_ts

    n_fwd = int(_num(row, "Total Fwd Packets", 0))
    n_bwd = int(_num(row, "Total Backward Packets", 0))
    n = n_fwd + n_bwd
    if n_fwd <= 0 or n <= 0:
        return SynthResult(None, "no_forward_packets", False, False)

    raw_duration = int(_num(row, "Flow Duration", 0))
    duration_clamped = raw_duration < 0
    duration_us = max(0, raw_duration)
    zero_duration = duration_us == 0

    if n == 1:
        timestamps = [ts_us]
    else:
        timestamps = [ts_us + round(i * duration_us / (n - 1)) for i in range(n)]

    directions = _interleave_directions(n_fwd, n_bwd)

    fwd_lengths = _synth_lengths(
        n_fwd,
        int(_num(row, "Total Length of Fwd Packets", 0)),
        int(_num(row, "Fwd Packet Length Min", 0)),
        int(_num(row, "Fwd Packet Length Max", 0)),
    )
    bwd_lengths = _synth_lengths(
        n_bwd,
        int(_num(row, "Total Length of Bwd Packets", 0)),
        int(_num(row, "Bwd Packet Length Min", 0)),
        int(_num(row, "Bwd Packet Length Max", 0)),
    )

    syn_positions = _flag_positions(n, int(_num(row, "SYN Flag Count", 0)), "start")
    fin_positions = _flag_positions(n, int(_num(row, "FIN Flag Count", 0)), "end")
    rst_positions = _flag_positions(n, int(_num(row, "RST Flag Count", 0)), "end")
    ack_positions = _flag_positions(n, int(_num(row, "ACK Flag Count", 0)), "spread")
    psh_count = int(_num(row, "Fwd PSH Flags", 0)) + int(_num(row, "Bwd PSH Flags", 0))
    urg_count = int(_num(row, "Fwd URG Flags", 0)) + int(_num(row, "Bwd URG Flags", 0))
    psh_positions = _flag_positions(n, psh_count, "spread")
    urg_positions = _flag_positions(n, urg_count, "spread")

    init_win_fwd = int(_num(row, "Init_Win_bytes_forward", UNSET))
    init_win_bwd = int(_num(row, "Init_Win_bytes_backward", UNSET))

    packets: List[Packet] = []
    fi = bi = 0
    fwd_seen = bwd_seen = False
    for i, direction in enumerate(directions):
        if direction == Direction.FWD:
            length = fwd_lengths[fi]
            fi += 1
            pkt_src, pkt_dst, pkt_sport, pkt_dport = src_ip, dst_ip, src_port, dst_port
            win = init_win_fwd if not fwd_seen else UNSET
            fwd_seen = True
        else:
            length = bwd_lengths[bi]
            bi += 1
            pkt_src, pkt_dst, pkt_sport, pkt_dport = dst_ip, src_ip, dst_port, src_port
            win = init_win_bwd if not bwd_seen else UNSET
            bwd_seen = True

        packets.append(
            Packet(
                src_ip=pkt_src,
                dst_ip=pkt_dst,
                src_port=pkt_sport,
                dst_port=pkt_dport,
                protocol=protocol,
                timestamp_us=timestamps[i],
                length=max(0, length),
                syn=i in syn_positions,
                fin=i in fin_positions,
                rst=i in rst_positions,
                psh=i in psh_positions,
                ack=i in ack_positions,
                urg=i in urg_positions,
                win_size=win,
                timestamp_resolution_us=resolution_us,
                source_row_id=row_id,
            )
        )
    return SynthResult(packets, None, duration_clamped, zero_duration, resolution_us)


@dataclass
class AdapterStats:
    rows_total: int = 0
    rows_skipped: int = 0
    rows_duration_clamped: int = 0
    rows_zero_duration: int = 0
    packets_synthesized: int = 0
    skip_reasons: Counter = field(default_factory=Counter)
    #: rows_total broken down by the Timestamp resolution that parsed
    #: them (in microseconds per tick) — e.g. {1_000_000: N} for an
    #: all-second-resolution day, mixed keys if a file isn't uniform.
    resolution_counts: Counter = field(default_factory=Counter)

    @property
    def timestamp_resolution_us(self) -> Optional[int]:
        """The single resolution used across every row, or None if the
        file mixed formats (which would itself be worth investigating)."""
        if len(self.resolution_counts) == 1:
            return next(iter(self.resolution_counts))
        return None

    def as_dict(self) -> Dict[str, object]:
        return {
            "rows_total": self.rows_total,
            "rows_skipped": self.rows_skipped,
            "rows_duration_clamped": self.rows_duration_clamped,
            "rows_zero_duration": self.rows_zero_duration,
            "packets_synthesized": self.packets_synthesized,
            "skip_reasons": dict(self.skip_reasons),
            "resolution_counts": dict(self.resolution_counts),
        }


def iter_packets_from_csv(
    path: Union[str, Path], stats: Optional[AdapterStats] = None
) -> Iterator[Packet]:
    """Yield synthesized packets for every row of a TrafficLabelling CSV,
    in file order (see module docstring for what that costs). Every
    packet carries ``source_row_id`` set to that row's position in the
    file (0-based) — pass ``FlowTable(key_mode=KeyMode.EVAL)`` to use it
    for 1-row-to-1-flow keying, or leave the table at its default
    FIDELITY keying to study/report on cross-row merging instead."""
    if stats is None:
        stats = AdapterStats()
    df = load_csv(path)
    for row_id, row in enumerate(df.to_dict("records")):
        stats.rows_total += 1
        result = synthesize_packets(row, row_id)
        if result.skip_reason is not None:
            stats.rows_skipped += 1
            stats.skip_reasons[result.skip_reason] += 1
            continue
        stats.resolution_counts[result.timestamp_resolution_us] += 1
        if result.duration_clamped:
            stats.rows_duration_clamped += 1
        if result.zero_duration:
            stats.rows_zero_duration += 1
        stats.packets_synthesized += len(result.packets)
        yield from result.packets
