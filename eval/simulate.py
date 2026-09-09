"""Run the flow-table + selector-feature simulation once per day and
cache the result, so eval/sweep.py sweeps many thresholds against
pre-computed features instead of re-running the simulator per point.

Cache key: (day, key_mode, ADAPTER_VERSION, FEATURE_SCHEMA_VERSION). A
change to either version constant invalidates the cache automatically —
see adapters/csv_flow_adapter.py and dataplane/selector.py.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from adapters.csv_flow_adapter import ADAPTER_VERSION, AdapterStats, iter_packets_from_csv
from dataplane.flow_state import FlowState
from dataplane.flow_table import FlowTable, KeyMode
from dataplane.selector import FEATURE_SCHEMA_VERSION, compute_src_features, compute_tier1_features
from dataplane.src_table import SrcTable
from eval.labels import (
    JoinReport,
    build_fidelity_join_report,
    join_eval_flows,
    load_row_ground_truth,
    resolve_fidelity_labels,
)

DEFAULT_CACHE_DIR = Path("results/cache")

#: CICIDS2017 TrafficLabelling files. "day" here means "file" — Thursday
#: and Friday each split into more than one CSV.
DAY_FILES: Dict[str, str] = {
    "monday": "data/csv/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv",
    "tuesday": "data/csv/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
    "wednesday": "data/csv/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv",
    "thursday_morning_webattacks": "data/csv/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "thursday_afternoon_infiltration": "data/csv/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "friday_morning": "data/csv/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "friday_afternoon_ddos": "data/csv/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "friday_afternoon_portscan": "data/csv/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
}


@dataclass
class DayExtractionMeta:
    day: str
    csv_path: str
    key_mode: str
    adapter_version: str
    feature_schema_version: str
    rows_total: int
    rows_skipped: int
    skip_reasons: Dict[str, int]
    flows_total: int
    flow_table_eviction_count: int
    flow_table_idle_timeout_count: int
    flow_table_active_timeout_count: int
    src_table_eviction_count: int
    join_report: Dict[str, object]
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_paths(day: str, key_mode: KeyMode, cache_dir: Path) -> Tuple[Path, Path]:
    base = f"{day}__{key_mode.value}__adapter{ADAPTER_VERSION}__features{FEATURE_SCHEMA_VERSION}"
    return cache_dir / f"{base}.parquet", cache_dir / f"{base}.meta.json"


def _close_flow(state: FlowState, src_table: SrcTable) -> None:
    src_table.note_flow_closed(
        state.fwd_ip,
        state.last_ts,
        protocol=state.protocol,
        syn_count=state.syn_count,
        bwd_pkt_count=state.bwd_pkt_count,
    )


def _run_simulation(
    csv_path: Union[str, Path], key_mode: KeyMode, capacity: Optional[int]
) -> Tuple[List[FlowState], List[str], AdapterStats, FlowTable, SrcTable]:
    table = FlowTable(key_mode=key_mode, capacity=capacity)
    src_table = SrcTable()
    stats = AdapterStats()
    closed_flows: List[FlowState] = []
    closure_reasons: List[str] = []

    for packet in iter_packets_from_csv(csv_path, stats=stats):
        result = table.process(packet)
        if result.is_new_flow:
            src_table.note_flow_start(
                packet.src_ip, packet.dst_ip, packet.dst_port, packet.timestamp_us
            )
        for expired in result.expired:
            _close_flow(expired.state, src_table)
            closed_flows.append(expired.state)
            closure_reasons.append(expired.reason)

    for expired in table.flush():
        _close_flow(expired.state, src_table)
        closed_flows.append(expired.state)
        closure_reasons.append(expired.reason)

    return closed_flows, closure_reasons, stats, table, src_table


def extract_day_features(
    csv_path: Union[str, Path],
    day: str,
    key_mode: KeyMode,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    capacity: Optional[int] = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Run (or load a cached run of) one day's simulation, returning a
    flat per-flow feature table plus a metadata dict (join rate, eviction
    counts, etc.). Every row is one closed flow, labeled at whatever
    point it left the table (FIN, RST, idle/active timeout, capacity
    eviction, or end-of-file flush) — see eval.labels for how the label
    was resolved for that key_mode.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_path, meta_path = _cache_paths(day, key_mode, cache_dir)

    if not force and features_path.exists() and meta_path.exists():
        df = pd.read_parquet(features_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return df, meta

    start = time.perf_counter()
    row_truth = load_row_ground_truth(csv_path)
    closed_flows, closure_reasons, stats, table, src_table = _run_simulation(
        csv_path, key_mode, capacity
    )

    if key_mode == KeyMode.EVAL:
        labels, join_report = join_eval_flows(closed_flows, row_truth)
        mixed_flags = [False] * len(closed_flows)
        ambiguous_flags = [False] * len(closed_flows)
    else:
        resolutions = resolve_fidelity_labels(closed_flows, row_truth)
        labels = [r.label for r in resolutions]
        mixed_flags = [r.mixed for r in resolutions]
        ambiguous_flags = [r.ambiguous for r in resolutions]
        join_report = build_fidelity_join_report(closed_flows, resolutions)

    records = []
    for flow, label, mixed, ambiguous, reason in zip(
        closed_flows, labels, mixed_flags, ambiguous_flags, closure_reasons
    ):
        row: Dict[str, object] = {
            "label": label,
            "mixed_label": mixed,
            "residual_ambiguous": ambiguous,
            "n_source_rows": len(flow.source_row_ids),
            # first_ts anchors this flow in real time, needed for a true
            # *chronological* split (e.g. eval/sweep.py's Monday
            # fit/held-out halves) — the order flows appear in this table
            # is closure order, not start order, since a flow with no
            # FIN/RST sits open until end-of-file flush regardless of
            # when it actually started.
            "first_ts": flow.first_ts,
            "last_ts": flow.last_ts,
            "closure_reason": reason,
        }
        features = dict(compute_tier1_features(flow))
        features.update(compute_src_features(src_table, flow.fwd_ip, flow.last_ts))
        for name, observation in features.items():
            row[name] = observation.value
            row[f"{name}__low_confidence"] = observation.low_confidence
        records.append(row)

    df = pd.DataFrame.from_records(records)
    df.to_parquet(features_path)

    elapsed = time.perf_counter() - start
    meta = DayExtractionMeta(
        day=day,
        csv_path=str(csv_path),
        key_mode=key_mode.value,
        adapter_version=ADAPTER_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        rows_total=stats.rows_total,
        rows_skipped=stats.rows_skipped,
        skip_reasons=dict(stats.skip_reasons),
        flows_total=len(closed_flows),
        flow_table_eviction_count=table.eviction_count,
        flow_table_idle_timeout_count=table.idle_timeout_count,
        flow_table_active_timeout_count=table.active_timeout_count,
        src_table_eviction_count=src_table.eviction_count,
        join_report=join_report.to_dict(),
        elapsed_seconds=elapsed,
    ).to_dict()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return df, meta
