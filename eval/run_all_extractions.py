"""Run (or load from cache) feature extraction for every CICIDS2017 day,
in both key modes: EVAL (feeds the sweep) and FIDELITY (characterises
the data plane's own merge behavior / join-rate cost). One-time cost;
eval/sweep.py reads the cached parquet files afterward.
"""
from __future__ import annotations

import sys
import time

from dataplane.flow_table import KeyMode
from eval.simulate import DAY_FILES, extract_day_features


def main() -> None:
    for day, path in DAY_FILES.items():
        for key_mode in (KeyMode.EVAL, KeyMode.FIDELITY):
            start = time.perf_counter()
            df, meta = extract_day_features(path, day, key_mode)
            elapsed = time.perf_counter() - start
            cached = elapsed < 1.0
            print(
                f"{day:35s} {key_mode.value:9s} flows={meta['flows_total']:>7} "
                f"join_rate={meta['join_report']['join_rate']:.4f} "
                f"evictions(flow/src)={meta['flow_table_eviction_count']}/{meta['src_table_eviction_count']} "
                f"{'[cached]' if cached else f'[{elapsed:.0f}s]'}"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
