"""Checkpoint script: run FlowTable over a full real day of CICIDS2017
TrafficLabelling CSV data via the CSV adapter, and report what happened.

Runs both FlowTable.KeyMode values and labels every number with which one
produced it — see adapters/csv_flow_adapter.py and dataplane/flow_table.py
(KeyMode) for what the difference is and why it matters: FIDELITY (plain
5-tuple) is a legitimate finding about the data plane's own merge/timeout
behavior; EVAL (5-tuple + source row) is what anything reporting
label-derived numbers must use, since FIDELITY can merge rows that carry
different ground-truth labels.

Usage: python run_csv_stress_test.py [path-to-csv]
Defaults to the Monday (benign-only) file, since that's also fitting.py's
eventual input and has the cleanest (second-resolution) timestamps.
"""
from __future__ import annotations

import sys
import time

from adapters.csv_flow_adapter import AdapterStats, iter_packets_from_csv
from dataplane.flow_table import DEFAULT_CAPACITY, FlowTable, KeyMode

DEFAULT_PATH = "data/csv/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv"


def run(path: str, key_mode: KeyMode, capacity=None) -> None:
    table = FlowTable(capacity=capacity, key_mode=key_mode)
    stats = AdapterStats()

    start = time.perf_counter()
    for packet in iter_packets_from_csv(path, stats=stats):
        table.process(packet)
    remaining = table.flush()
    elapsed = time.perf_counter() - start

    merged_flows = sum(1 for f in table.active_flows() if len(f.source_row_ids) > 1)

    print(f"=== key_mode={key_mode.value} capacity={table.capacity} ===")
    print(f"rows total              : {stats.rows_total}")
    print(f"rows skipped            : {stats.rows_skipped}")
    if stats.skip_reasons:
        for reason, count in stats.skip_reasons.most_common():
            print(f"    {reason:<28}: {count}")
    print(f"rows negative-duration clamped : {stats.rows_duration_clamped}")
    print(f"rows zero-duration      : {stats.rows_zero_duration}")
    print(f"packets synthesized     : {stats.packets_synthesized}")
    print(f"packets processed       : {table.total_packets_seen}")
    print(f"flows created           : {table.total_flows_created}")
    print(f"flows force-closed at flush (still open at EOF): {len(remaining)}")
    print(f"capacity evictions      : {table.eviction_count}")
    print(f"idle timeouts           : {table.idle_timeout_count}")
    print(f"active timeouts         : {table.active_timeout_count}")
    print(f"wall clock              : {elapsed:.1f}s")
    if elapsed > 0:
        print(f"throughput              : {table.total_packets_seen / elapsed:,.0f} pkts/sec, "
              f"{table.total_flows_created / elapsed:,.0f} flows/sec")
    print()


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    print(f"file: {path}\n")

    # EVAL first: this is the mode any label-derived numbers should use.
    # Unbounded capacity by default (see KeyMode.EVAL) — capacity pressure
    # is a FIDELITY-mode question.
    run(path, key_mode=KeyMode.EVAL)

    # FIDELITY: a finding about the data plane's own flow definition, not
    # a numbers-producing run — rows can merge across a shared 5-tuple.
    run(path, capacity=DEFAULT_CAPACITY, key_mode=KeyMode.FIDELITY)

    # artificially tiny capacity to force and confirm the eviction path
    # actually fires on real data (default capacity rarely fills up when
    # rows are fed sequentially rather than with realistic concurrency —
    # see the adapter's module docstring on why concurrency isn't faithful).
    run(path, capacity=64, key_mode=KeyMode.FIDELITY)


if __name__ == "__main__":
    main()
