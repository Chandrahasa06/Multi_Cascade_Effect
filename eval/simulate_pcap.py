"""Run the flow-table + selector-feature simulation on a real pcapng
capture (adapters/pcap_adapter.py) and cache the result — the PCAP-path
peer of eval/simulate.py.

Only FIDELITY-mode keying makes sense here (see FlowTable.KeyMode and
adapters/pcap_adapter.py's module docstring: a real packet has no CSV
row to key EVAL mode's discriminator on), so this module doesn't take a
key_mode argument at all.

Monday is benign-only by construction — CICIDS2017's own design, and
this project's existing CSV-path treats it the same way (see
eval/sweep.py: "Fitting is Monday-only... benign-only") — so it needs
no CSV join at all; every closed flow is simply labeled BENIGN. Friday
needs eval.labels_pcap's session-based join (real packet flows carry no
source_row_id to join on directly) since it's one continuous capture
spanning three separate CICIDS2017 CSV sessions.

Cache key: (day, "pcap", ADAPTER_VERSION, FEATURE_SCHEMA_VERSION).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from adapters.pcap_adapter import ADAPTER_VERSION, AdapterStats, iter_packets_from_pcap
from dataplane.flow_state import FlowState
from dataplane.flow_table import FlowTable, KeyMode
from dataplane.selector import (
    FEATURE_SCHEMA_VERSION,
    FeatureObservation,
    compute_src_features,
    compute_tier1_features,
)
from dataplane.src_table import SrcTable

DEFAULT_CACHE_DIR = Path("results/cache")

PCAP_PATHS: Dict[str, str] = {
    "monday": "data/pcap/Monday-WorkingHours.pcap",
    "friday": "data/pcap/Friday-WorkingHours.pcap",
}


@dataclass
class PcapDayExtractionMeta:
    day: str
    pcap_path: str
    adapter_version: str
    feature_schema_version: str
    packets_total: int
    packets_emitted: int
    packets_skipped: int
    skip_reasons: Dict[str, int]
    non_monotonic_timestamps: int
    flows_total: int
    flow_table_eviction_count: int
    flow_table_idle_timeout_count: int
    flow_table_active_timeout_count: int
    closure_reason_counts: Dict[str, int]
    peak_concurrent_flows: int
    flow_table_capacity: int
    src_table_eviction_count: int
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_paths(day: str, cache_dir: Path) -> Tuple[Path, Path]:
    base = f"{day}_pcap__fidelity__adapter{ADAPTER_VERSION}__features{FEATURE_SCHEMA_VERSION}"
    return cache_dir / f"{base}.parquet", cache_dir / f"{base}.meta.json"


def _finalize_flow(
    state: FlowState,
    reason: str,
    src_table: SrcTable,
    closed_flows: List[FlowState],
    closure_reasons: List[str],
    feature_rows: List[Dict[str, FeatureObservation]],
) -> None:
    """Same snapshot-at-closure discipline as eval/simulate.py's
    _finalize_flow — see that function's docstring for why per-source
    features must be read at this exact instant, not in a later pass."""
    src_table.note_flow_closed(
        state.fwd_ip,
        state.last_ts,
        protocol=state.protocol,
        syn_count=state.syn_count,
        bwd_pkt_count=state.bwd_pkt_count,
    )
    features: Dict[str, FeatureObservation] = dict(compute_tier1_features(state))
    features.update(compute_src_features(src_table, state.fwd_ip, state.last_ts))

    closed_flows.append(state)
    closure_reasons.append(reason)
    feature_rows.append(features)


def _run_simulation(
    pcap_path: Union[str, Path]
) -> Tuple[
    List[FlowState], List[str], List[Dict[str, FeatureObservation]], AdapterStats,
    FlowTable, SrcTable, int,
]:
    table = FlowTable(key_mode=KeyMode.FIDELITY)  # default capacity 65,536
    src_table = SrcTable()
    stats = AdapterStats()
    closed_flows: List[FlowState] = []
    closure_reasons: List[str] = []
    feature_rows: List[Dict[str, FeatureObservation]] = []
    peak_concurrent = 0

    for packet in iter_packets_from_pcap(pcap_path, stats=stats):
        result = table.process(packet)
        size = len(table)
        if size > peak_concurrent:
            peak_concurrent = size
        if result.is_new_flow:
            src_table.note_flow_start(
                packet.src_ip, packet.dst_ip, packet.dst_port, packet.timestamp_us
            )
        for expired in result.expired:
            _finalize_flow(
                expired.state, expired.reason, src_table, closed_flows, closure_reasons, feature_rows
            )

    for expired in table.flush():
        _finalize_flow(
            expired.state, expired.reason, src_table, closed_flows, closure_reasons, feature_rows
        )

    return closed_flows, closure_reasons, feature_rows, stats, table, src_table, peak_concurrent


def extract_day_features_pcap(
    day: str,
    pcap_path: Union[str, Path],
    label_fn,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Run (or load a cached run of) one day's PCAP simulation.

    ``label_fn(closed_flows) -> (labels, extra_meta)`` assigns a label
    per closed flow; a ``None`` label excludes that flow from the
    cached frame entirely (used for Friday flows outside every session
    window — see eval/labels_pcap.py). ``extra_meta`` is merged into the
    saved metadata dict verbatim (e.g. Friday's session-join report).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_path, meta_path = _cache_paths(day, cache_dir)

    if not force and features_path.exists() and meta_path.exists():
        df = pd.read_parquet(features_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return df, meta

    start = time.perf_counter()
    closed_flows, closure_reasons, feature_rows, stats, table, src_table, peak = _run_simulation(
        pcap_path
    )
    labels, label_meta = label_fn(closed_flows)

    records = []
    for flow, label, reason, features in zip(closed_flows, labels, closure_reasons, feature_rows):
        if label is None:
            continue
        row: Dict[str, object] = {
            "label": label,
            "first_ts": flow.first_ts,
            "last_ts": flow.last_ts,
            "closure_reason": reason,
        }
        for name, observation in features.items():
            row[name] = observation.value
            row[f"{name}__low_confidence"] = observation.low_confidence
        records.append(row)

    df = pd.DataFrame.from_records(records)
    df.to_parquet(features_path)

    elapsed = time.perf_counter() - start
    closure_counter = Counter(closure_reasons)
    meta = PcapDayExtractionMeta(
        day=day,
        pcap_path=str(pcap_path),
        adapter_version=ADAPTER_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        packets_total=stats.packets_total,
        packets_emitted=stats.packets_emitted,
        packets_skipped=stats.packets_skipped,
        skip_reasons=dict(stats.skip_reasons),
        non_monotonic_timestamps=stats.non_monotonic_timestamps,
        flows_total=len(closed_flows),
        flow_table_eviction_count=table.eviction_count,
        flow_table_idle_timeout_count=table.idle_timeout_count,
        flow_table_active_timeout_count=table.active_timeout_count,
        closure_reason_counts=dict(closure_counter),
        peak_concurrent_flows=peak,
        flow_table_capacity=table.capacity,
        src_table_eviction_count=src_table.eviction_count,
        elapsed_seconds=elapsed,
    ).to_dict()
    meta.update(label_meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)

    return df, meta


def label_all_benign(closed_flows: List[FlowState]) -> Tuple[List[str], dict]:
    """Monday: every flow is BENIGN by construction, no CSV join."""
    return ["BENIGN"] * len(closed_flows), {}
