"""Regenerate results/escalation_records_sample.jsonl (+ its INTERNAL
ground-truth manifest) at a given per-class stratification.

Same method as the original 200-record sample (STATUS.md: "50 each of
Bot/PortScan/DDoS/BENIGN ... Selector config fitted on Monday PCAP
fit-half, k_of_n(k=2), exclude_low_confidence=False"), generalized to an
arbitrary per-class count and re-implemented from scratch: the original
generator was a throwaway script, never committed (same pattern as the
pcapng verification script mentioned in STATUS.md's PCAP section).

Steps:
  1. Fit selector thresholds on Monday PCAP's chronological fit-half,
     sweep percentiles, and pick the one whose held-out-benign escalation
     rate is closest to 5% -- "the same selector operating point" as the
     original sample, reproduced by construction rather than hardcoded,
     and checked against the original's config_hash (9581bdb7e3859338)
     as a sanity gate.
  2. Re-run the Friday PCAP simulation fresh (selector evaluated inline,
     per flow, at closure -- not from the cached feature-only parquet,
     which doesn't retain the 5-tuple key packet re-collection needs),
     keeping only escalated flows to bound memory.
  3. Join the escalated flows to their Friday CSV session labels
     (eval/labels_pcap.py), same as the PCAP sweep.
  4. Stratified-sample the requested per-class counts (fixed seed, no
     replacement) from the escalated+labeled pool.
  5. Re-collect each sampled flow's raw packets from the Friday PCAP and
     run them through CICFlowMeter (controlplane/extractor.py) to build
     anonymised EscalationRecords.
  6. Write the records file (grouped by class, in the class order given
     on the command line -- callers needing a specific run order for the
     agent pipeline should pass --class-order to match) and the
     INTERNAL manifest (flow_id -> true label).

Run: python -m eval.generate_escalation_sample
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from adapters.pcap_adapter import ADAPTER_VERSION, AdapterStats, iter_packets_from_pcap
from controlplane.extractor import collect_flow_packets, extract_record
from controlplane.record import EscalationRecord
from dataplane.flow_state import FlowState
from dataplane.flow_table import FlowTable, KeyMode
from dataplane.selector import (
    EscalationRule,
    FeatureObservation,
    Selector,
    SelectorConfig,
    TriggerReason,
    compute_src_features,
    compute_tier1_features,
)
from dataplane.src_table import SrcTable
from eval.labels_pcap import join_friday_sessions, load_session_index
from eval.run_pcap_sweep import (
    FRIDAY_SESSION_CSVS,
    FRIDAY_SESSION_PM_CORRECTION,
    load_monday_pcap,
)
from eval.sweep import (
    PERCENTILE_SWEEP,
    nearest_point,
    run_sweep,
    split_monday_chronologically,
)

FRIDAY_PCAP_PATH = "data/pcap/Friday-WorkingHours.pcap"

#: the original 200-record sample's fitted config_hash (STATUS.md) --
#: sanity check, not a hard requirement (a code change to feature
#: computation since then would legitimately move it).
EXPECTED_CONFIG_HASH = "9581bdb7e3859338"

DEFAULT_COUNTS = {"BENIGN": 70, "DDoS": 50, "PortScan": 50, "Bot": 30}
DEFAULT_CLASS_ORDER = ["DDoS", "BENIGN", "PortScan", "Bot"]

RECORDS_OUT = Path("results/escalation_records_sample.jsonl")
MANIFEST_OUT = Path("results/escalation_records_sample_manifest.INTERNAL.jsonl")

SAMPLE_SEED = 20260911  # fixed, for a reproducible stratified draw


def fit_operating_point(target_benign_rate: float = 0.05) -> SelectorConfig:
    """Same fitting discipline as eval/run_pcap_sweep.run_pcap_sweep:
    Monday PCAP fit-half fits thresholds at each swept percentile,
    Monday PCAP holdout-half measures the out-of-sample benign escalation
    rate. Returns the config whose measured rate is closest to
    ``target_benign_rate``."""
    monday_df, _meta = load_monday_pcap()
    monday_fit, monday_holdout = split_monday_chronologically(monday_df)

    # week_df only needs to exist for run_sweep's per-class recall table
    # (unused here) -- reuse monday_holdout so this doesn't force a
    # fresh Friday simulation just to fit thresholds.
    result, fitting_results = run_sweep(
        monday_fit,
        monday_holdout,
        monday_holdout,
        percentiles=PERCENTILE_SWEEP,
        rule=EscalationRule.K_OF_N,
        k=2,
        exclude_low_confidence=False,
        ablation_name="generate_escalation_sample",
    )
    point = nearest_point(result, target_benign_rate, by="benign_escalation_rate")
    config = fitting_results[point.percentile].config
    print(
        f"fitted operating point: percentile={point.percentile:.4f}, "
        f"measured held-out benign_escalation_rate={point.benign_escalation_rate:.4%}, "
        f"config_hash={config.config_hash}"
    )
    if config.config_hash != EXPECTED_CONFIG_HASH:
        print(
            f"[note] config_hash differs from the original sample's recorded "
            f"{EXPECTED_CONFIG_HASH} -- expected if feature computation changed "
            "since then, otherwise worth double-checking."
        )
    return config


def simulate_and_escalate(
    pcap_path: str, selector: Selector
) -> Tuple[List[FlowState], Dict[int, List[TriggerReason]]]:
    """One pass over the Friday PCAP: flow-table simulation with the
    selector evaluated inline at each flow's closure (same
    snapshot-at-closure discipline as eval/simulate_pcap.py's
    _finalize_flow). Returns only the escalated FlowState objects (to
    bound memory -- Friday has ~678k flows total, a small fraction
    escalate) plus their trigger reasons, keyed by id(flow) since
    FlowState isn't hashable/frozen-dataclass-eq by content."""
    table = FlowTable(key_mode=KeyMode.FIDELITY)
    src_table = SrcTable()
    stats = AdapterStats()
    escalated_flows: List[FlowState] = []
    trigger_reasons_by_id: Dict[int, List[TriggerReason]] = {}

    def _finalize(state: FlowState) -> None:
        src_table.note_flow_closed(
            state.fwd_ip, state.last_ts, protocol=state.protocol,
            syn_count=state.syn_count, bwd_pkt_count=state.bwd_pkt_count,
        )
        features: Dict[str, FeatureObservation] = dict(compute_tier1_features(state))
        features.update(compute_src_features(src_table, state.fwd_ip, state.last_ts))
        decision = selector.evaluate_features(features)
        if decision.escalate:
            escalated_flows.append(state)
            trigger_reasons_by_id[id(state)] = decision.trigger_reasons

    start = time.perf_counter()
    n_packets = 0
    for packet in iter_packets_from_pcap(pcap_path, stats=stats):
        result = table.process(packet)
        if result.is_new_flow:
            src_table.note_flow_start(packet.src_ip, packet.dst_ip, packet.dst_port, packet.timestamp_us)
        for expired in result.expired:
            _finalize(expired.state)
        n_packets += 1
        if n_packets % 2_000_000 == 0:
            print(f"  ... {n_packets:,} packets, {len(escalated_flows):,} escalated so far "
                  f"({time.perf_counter() - start:.0f}s elapsed)")

    for expired in table.flush():
        _finalize(expired.state)

    elapsed = time.perf_counter() - start
    print(
        f"Friday simulation done: {stats.packets_total:,} packets, {len(escalated_flows):,} "
        f"escalated flows, {elapsed:.1f}s"
    )
    return escalated_flows, trigger_reasons_by_id


def label_escalated_flows(escalated_flows: List[FlowState]) -> Dict[int, str]:
    """Friday session join, restricted to the escalated subset -- the
    join logic (eval/labels_pcap.join_friday_sessions) only needs the
    flows being labeled plus each session's own CSV index, not the full
    day's flow population."""
    sessions = [
        load_session_index(name, path, pm_hour_correction=FRIDAY_SESSION_PM_CORRECTION[name])
        for name, path in FRIDAY_SESSION_CSVS.items()
    ]
    result = join_friday_sessions(escalated_flows, sessions)
    labels_by_id: Dict[int, str] = {}
    for flow, label in zip(escalated_flows, result.labels):
        if label is not None:
            labels_by_id[id(flow)] = label
    print(
        f"session join: {len(labels_by_id):,}/{len(escalated_flows):,} escalated flows labeled "
        f"({result.n_outside_all_windows} outside every session window)"
    )
    return labels_by_id


def stratified_draw(
    pool_by_class: Dict[str, List[FlowState]], counts: Dict[str, int], seed: int
) -> Dict[str, List[FlowState]]:
    rng = random.Random(seed)
    out: Dict[str, List[FlowState]] = {}
    for label, n in counts.items():
        pool = pool_by_class.get(label, [])
        print(f"  {label}: pool={len(pool):,}, requesting {n}")
        if len(pool) < n:
            print(f"  [warn] {label} pool ({len(pool)}) is smaller than requested ({n}) -- taking all of it")
        # sort first for a deterministic draw regardless of dict/set
        # iteration order upstream, then sample without replacement.
        ordered = sorted(pool, key=lambda f: (f.first_ts, f.key))
        out[label] = rng.sample(ordered, min(n, len(ordered)))
    return out


def build_records(
    pcap_path: str,
    sampled_by_class: Dict[str, List[FlowState]],
    trigger_reasons_by_id: Dict[int, List[TriggerReason]],
    selector_config: SelectorConfig,
    class_order: Sequence[str],
) -> Tuple[List[Tuple[EscalationRecord, str]], List[str]]:
    all_targets: List[FlowState] = []
    for label in class_order:
        all_targets.extend(sampled_by_class.get(label, []))

    print(f"re-collecting packets for {len(all_targets)} sampled flows from {pcap_path} ...")
    start = time.perf_counter()
    packets_by_token = collect_flow_packets(pcap_path, all_targets)
    print(f"  packet re-collection: {time.perf_counter() - start:.1f}s")

    out: List[Tuple[EscalationRecord, str]] = []
    dropped: List[str] = []
    for label in class_order:
        for flow in sampled_by_class.get(label, []):
            token = (flow.key, flow.first_ts)
            packets = packets_by_token.get(token, [])
            trigger_reasons = trigger_reasons_by_id[id(flow)]
            record = extract_record(flow, trigger_reasons, packets, selector_config)
            if record is None:
                dropped.append(f"{label}:{flow.key}:{flow.first_ts}")
                continue
            out.append((record, label))
    return out, dropped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benign", type=int, default=DEFAULT_COUNTS["BENIGN"])
    parser.add_argument("--ddos", type=int, default=DEFAULT_COUNTS["DDoS"])
    parser.add_argument("--portscan", type=int, default=DEFAULT_COUNTS["PortScan"])
    parser.add_argument("--bot", type=int, default=DEFAULT_COUNTS["Bot"])
    parser.add_argument("--class-order", default=",".join(DEFAULT_CLASS_ORDER))
    parser.add_argument("--target-benign-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--out", default=str(RECORDS_OUT))
    parser.add_argument("--manifest-out", default=str(MANIFEST_OUT))
    args = parser.parse_args(argv)

    counts = {"BENIGN": args.benign, "DDoS": args.ddos, "PortScan": args.portscan, "Bot": args.bot}
    class_order = [c.strip() for c in args.class_order.split(",")]
    assert set(class_order) == set(counts), f"--class-order must list exactly {sorted(counts)}"

    print(f"adapter_version={ADAPTER_VERSION}")
    print("step 1/5: fitting selector thresholds on Monday PCAP fit-half ...")
    config = fit_operating_point(args.target_benign_rate)
    selector = Selector(config)

    print("step 2/5: simulating Friday PCAP with the selector evaluated inline ...")
    escalated_flows, trigger_reasons_by_id = simulate_and_escalate(FRIDAY_PCAP_PATH, selector)

    print("step 3/5: joining escalated flows to Friday session labels ...")
    labels_by_id = label_escalated_flows(escalated_flows)

    pool_by_class: Dict[str, List[FlowState]] = defaultdict(list)
    for flow in escalated_flows:
        label = labels_by_id.get(id(flow))
        if label in counts:
            pool_by_class[label].append(flow)
        # labels outside the four classes this sample cares about (e.g.
        # a Web-Attack-shaped label leaking into a session window) are
        # silently excluded from the pool -- they were never part of
        # the target stratification.

    print("step 4/5: stratified sampling ...")
    sampled_by_class = stratified_draw(pool_by_class, counts, args.seed)

    print("step 5/5: re-collecting packets + running CICFlowMeter extraction ...")
    records, dropped = build_records(
        FRIDAY_PCAP_PATH, sampled_by_class, trigger_reasons_by_id, config, class_order
    )
    if dropped:
        print(f"[warn] {len(dropped)} sampled flows produced no usable CICFlowMeter row, dropped: {dropped}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f_records, \
         open(args.manifest_out, "w", encoding="utf-8") as f_manifest:
        for record, label in records:
            f_records.write(json.dumps(record.to_dict()) + "\n")
            f_manifest.write(json.dumps({"flow_id": record.flow_id, "label": label}) + "\n")

    counts_written = {}
    for _, label in records:
        counts_written[label] = counts_written.get(label, 0) + 1
    print(f"wrote {len(records)} records to {args.out} ({counts_written}), manifest to {args.manifest_out}")
    print(f"selector_config_hash={config.config_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
