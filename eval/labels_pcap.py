"""Join real-packet PCAP-derived flows (adapters/pcap_adapter.py) back
to CICIDS2017 ground truth for days that have a corresponding CSV.

Necessarily different from eval/labels.py's join: a PCAP flow carries no
``source_row_id`` at all (there is no CSV row to trace back to — see
pcap_adapter.py's module docstring), so this can't use EVAL/FIDELITY
row-index keying. Two-stage best-effort join per flow instead:

  1. Exact 5-tuple match against the session's own CSV rows.
     ``FlowState.key`` IS the canonical 5-tuple (bidirectional-
     canonicalized via ``dataplane.flow_state.canonical_key`` — the same
     function a CSV row's Source/Destination IP/Port/Protocol columns
     are run through here), so no direction-fixing is needed to compare
     the two directly. When more than one CSV row shares the flow's
     exact 5-tuple, resolve by nearest Timestamp among just those
     candidates (same spirit as eval/labels.py's FIDELITY resolution).
  2. If no CSV row shares the flow's 5-tuple at all (a real flow
     CICFlowMeter recorded, split, or merged differently than our own
     flow table did — see README's "Known Limitations"), fall back to
     nearest-Timestamp against every row in that session.

Friday's single continuous capture spans three CICIDS2017 CSV sessions
(Bot in the morning, PortScan and DDoS in the afternoon). Session
membership is decided purely by each CSV's own observed Timestamp range
(min/max of its own Timestamp column) — a flow whose first_ts doesn't
fall inside any of the three ranges is excluded from the labeled set
and counted, not guessed at.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from adapters.csv_flow_adapter import load_csv, parse_timestamp_us
from dataplane.flow_state import FlowState, canonical_key

BENIGN_LABEL = "BENIGN"


def _num(row: dict, col: str, default: int = -1) -> int:
    try:
        val = float(row.get(col, default))
    except (TypeError, ValueError):
        return default
    if math.isnan(val) or math.isinf(val):
        return default
    return int(val)


@dataclass(frozen=True, slots=True)
class SessionIndex:
    """One CSV session's rows, indexed for both join strategies."""

    name: str
    csv_path: str
    by_key: Dict[tuple, List[int]]  # canonical 5-tuple -> row indices sharing it
    timestamps_us: List[Optional[int]]  # parallel to `labels`, None if unparsable
    labels: List[str]
    sorted_ts: List[Tuple[int, int]]  # (timestamp_us, row_idx), sorted, for bisect
    min_ts: int
    max_ts: int


#: microseconds in 12 hours, for the PM-hour correction below.
_TWELVE_HOURS_US = 12 * 3_600 * 1_000_000


def load_session_index(name: str, csv_path: str, pm_hour_correction: bool = False) -> SessionIndex:
    """``pm_hour_correction``: a real, evidenced quirk in two of
    CICIDS2017's Friday CSVs (Afternoon-PortScan, Afternoon-DDos) —
    their Timestamp column's afternoon hours were written in 12-hour
    format with no AM/PM marker and no +12h applied (e.g. "1:00" for
    1:00 PM, not 1:00 AM), so parsing them as 24-hour clock time (as
    ``adapters.csv_flow_adapter.parse_timestamp`` correctly does for
    every other CICIDS2017 file) reads 12 hours early. Confirmed by
    cross-referencing against the PCAP's own real embedded UTC
    timestamps: uncorrected, both sessions' windows fall entirely
    *before* the Friday capture even starts (~8-11 hours earlier, which
    is impossible); with a flat +12h shift both land inside the
    capture's real range, with the corrected PortScan window picking up
    almost exactly where Bot's ends and the corrected DDoS window
    ending within a minute of the capture's own last packet. This never
    surfaced on the CSV-only path because it never compared a CSV's
    Timestamp column against an external absolute clock — only against
    itself, within one file, where a uniform 12-hour offset changes
    nothing about row order."""
    df = load_csv(csv_path)
    by_key: Dict[tuple, List[int]] = {}
    timestamps: List[Optional[int]] = []
    labels: List[str] = []

    for i, row in enumerate(df.to_dict("records")):
        labels.append(str(row.get("Label", "")).strip())
        ts = parse_timestamp_us(str(row.get("Timestamp", "")))
        if ts is not None and pm_hour_correction:
            ts += _TWELVE_HOURS_US
        timestamps.append(ts)
        try:
            key = canonical_key(
                str(row["Source IP"]),
                str(row["Destination IP"]),
                _num(row, "Source Port"),
                _num(row, "Destination Port"),
                _num(row, "Protocol"),
            )
        except (KeyError, TypeError, ValueError):
            continue
        by_key.setdefault(key, []).append(i)

    sorted_ts = sorted((ts, i) for i, ts in enumerate(timestamps) if ts is not None)
    if not sorted_ts:
        raise ValueError(f"{csv_path}: no parsable Timestamp values")

    return SessionIndex(
        name=name,
        csv_path=str(csv_path),
        by_key=by_key,
        timestamps_us=timestamps,
        labels=labels,
        sorted_ts=sorted_ts,
        min_ts=sorted_ts[0][0],
        max_ts=sorted_ts[-1][0],
    )


def _nearest_by_timestamp(sorted_ts: List[Tuple[int, int]], target: int) -> int:
    """Row index in `sorted_ts` whose timestamp is closest to `target`."""
    keys = [t for t, _ in sorted_ts]
    pos = bisect.bisect_left(keys, target)
    candidates = []
    if pos < len(sorted_ts):
        candidates.append(sorted_ts[pos])
    if pos > 0:
        candidates.append(sorted_ts[pos - 1])
    best_ts, best_idx = min(candidates, key=lambda e: abs(e[0] - target))
    return best_idx


@dataclass
class SessionJoinReport:
    name: str
    flows_in_window: int = 0
    matched_5tuple: int = 0
    matched_nearest_timestamp: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "flows_in_window": self.flows_in_window,
            "matched_5tuple": self.matched_5tuple,
            "matched_nearest_timestamp": self.matched_nearest_timestamp,
        }


def join_flow_to_session(flow: FlowState, index: SessionIndex) -> Tuple[str, str]:
    """Returns (label, match_kind); match_kind is "5tuple" or
    "nearest_timestamp"."""
    candidates = index.by_key.get(flow.key)
    if candidates:
        if len(candidates) == 1:
            return index.labels[candidates[0]], "5tuple"
        dated = [
            (idx, index.timestamps_us[idx])
            for idx in candidates
            if index.timestamps_us[idx] is not None
        ]
        if dated:
            dated.sort(key=lambda e: abs(e[1] - flow.first_ts))
            return index.labels[dated[0][0]], "5tuple"
        return index.labels[candidates[0]], "5tuple"

    nearest_idx = _nearest_by_timestamp(index.sorted_ts, flow.first_ts)
    return index.labels[nearest_idx], "nearest_timestamp"


@dataclass
class FridaySessionJoinResult:
    labels: List[Optional[str]]  # None for flows outside every session window
    session_of_flow: List[Optional[str]]
    match_kind_of_flow: List[Optional[str]]
    reports: Dict[str, SessionJoinReport] = field(default_factory=dict)
    n_outside_all_windows: int = 0
    n_multi_window: int = 0
    session_windows: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def to_meta_dict(self) -> dict:
        return {
            "session_reports": {k: v.to_dict() for k, v in self.reports.items()},
            "n_outside_all_windows": self.n_outside_all_windows,
            "n_multi_window": self.n_multi_window,
            "session_windows": {k: list(v) for k, v in self.session_windows.items()},
        }


def join_friday_sessions(
    flows: Sequence[FlowState], sessions: Sequence[SessionIndex]
) -> FridaySessionJoinResult:
    """Assign every Friday PCAP flow to a session by IDENTITY first,
    TIME second.

    A flow's 5-tuple is checked against every session's exact-key index
    before anything time-based is considered. This matters because
    FlowTable runs continuously across the whole day (FIDELITY mode, no
    per-session reset): a 5-tuple used during, say, Bot's official
    morning window can legitimately reopen as a brand-new FlowState
    later the same day (idle/active timeout or RST/FIN closed the
    earlier instance; the same ephemeral port then got reused), giving
    that *later* instance a first_ts outside Bot's CSV-observed
    Timestamp range even though its identity still matches a Bot-labeled
    row exactly. Checked directly on this dataset: naively filtering
    candidates by first_ts-in-window *before* attempting a 5-tuple match
    found essentially none of the 1,228 distinct Bot 5-tuples (2 of
    52,514 morning-window flows) even though **all 1,228** of them exist
    somewhere in the full day's flow set — the flows were real and
    correctly keyed, just not the ones sitting in the pre-filtered
    window. Identity-first fixes this: a flow whose key exactly matches
    a session's CSV rows belongs to that session regardless of when our
    own table happened to (re)open it.

    Only when a flow's key matches no session's rows at all does this
    fall back to first_ts window membership (a flow inside more than
    one overlapping window, unexpected for sequential sessions, picks
    whichever window's center its first_ts is closest to) plus
    nearest-Timestamp resolution within it.
    """
    windows = {s.name: (s.min_ts, s.max_ts) for s in sessions}
    reports = {s.name: SessionJoinReport(name=s.name) for s in sessions}

    labels: List[Optional[str]] = []
    session_of_flow: List[Optional[str]] = []
    match_kind_of_flow: List[Optional[str]] = []
    n_outside = 0
    n_multi = 0

    for flow in flows:
        identity_matches = [s for s in sessions if flow.key in s.by_key]

        if identity_matches:
            if len(identity_matches) > 1:
                n_multi += 1
                session = min(
                    identity_matches,
                    key=lambda s: abs(flow.first_ts - (s.min_ts + s.max_ts) // 2),
                )
            else:
                session = identity_matches[0]
            label, kind = join_flow_to_session(flow, session)
            labels.append(label)
            session_of_flow.append(session.name)
            match_kind_of_flow.append(kind)
            reports[session.name].flows_in_window += 1
            reports[session.name].matched_5tuple += 1
            continue

        window_matches = [s for s in sessions if s.min_ts <= flow.first_ts <= s.max_ts]
        if not window_matches:
            labels.append(None)
            session_of_flow.append(None)
            match_kind_of_flow.append(None)
            n_outside += 1
            continue
        if len(window_matches) > 1:
            n_multi += 1
            window_matches = [
                min(
                    window_matches,
                    key=lambda s: abs(flow.first_ts - (s.min_ts + s.max_ts) // 2),
                )
            ]

        session = window_matches[0]
        label, kind = join_flow_to_session(flow, session)
        labels.append(label)
        session_of_flow.append(session.name)
        match_kind_of_flow.append(kind)
        reports[session.name].flows_in_window += 1
        reports[session.name].matched_nearest_timestamp += 1

    return FridaySessionJoinResult(
        labels=labels,
        session_of_flow=session_of_flow,
        match_kind_of_flow=match_kind_of_flow,
        reports=reports,
        n_outside_all_windows=n_outside,
        n_multi_window=n_multi,
        session_windows=windows,
    )
