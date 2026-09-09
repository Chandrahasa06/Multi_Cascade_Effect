"""End-to-end: load cached per-day features (running extraction if the
cache is cold), fit on Monday's chronological first half, sweep, run the
three ablations, and write everything to results/.
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from dataplane.flow_table import KeyMode
from dataplane.selector import EscalationRule
from eval.report import per_class_ablation_delta, write_results
from eval.simulate import DAY_FILES, extract_day_features
from eval.sweep import (
    all_feature_names,
    compute_crossings,
    escalate_from_crossings,
    nearest_point,
    run_sweep,
    split_monday_chronologically,
    tier1_feature_names,
    trigger_frequency,
)

TARGET_TRIGGER_FREQ_RATE = 0.05


def load_all(key_mode: KeyMode):
    frames, metas = {}, {}
    for day, path in DAY_FILES.items():
        t0 = time.perf_counter()
        df, meta = extract_day_features(path, day, key_mode)
        cached = (time.perf_counter() - t0) < 1.0
        print(
            f"  {day:35s} flows={meta['flows_total']:>7} "
            f"join_rate={meta['join_report']['join_rate']:.4f} {'[cached]' if cached else '[ran]'}"
        )
        sys.stdout.flush()
        frames[day] = df
        metas[day] = meta
    return frames, metas


def build_join_rate_table(eval_metas: dict, fidelity_metas: dict) -> pd.DataFrame:
    rows = []
    for day in DAY_FILES:
        rows.append(
            {"day": day, "key_mode": "eval", "join_rate": eval_metas[day]["join_report"]["join_rate"]}
        )
        rows.append(
            {
                "day": day,
                "key_mode": "fidelity",
                "join_rate": fidelity_metas[day]["join_report"]["join_rate"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print("=== EVAL-mode extraction (feeds the sweep) ===")
    eval_frames, eval_metas = load_all(KeyMode.EVAL)

    print("\n=== FIDELITY-mode extraction (join-rate characterisation only) ===")
    _, fidelity_metas = load_all(KeyMode.FIDELITY)

    monday = eval_frames["monday"]
    fit_half, holdout_half = split_monday_chronologically(monday)
    print(f"\nMonday split: {len(fit_half)} fit / {len(holdout_half)} held out (chronological)")

    other_days = pd.concat(
        [eval_frames[d] for d in DAY_FILES if d != "monday"], ignore_index=True
    )
    week_df = pd.concat([holdout_half, other_days], ignore_index=True)
    print(f"operational week (Monday holdout + Tue-Fri): {len(week_df)} flows")

    total_flows = sum(m["flows_total"] for m in eval_metas.values())
    total_src_evictions = sum(m["src_table_eviction_count"] for m in eval_metas.values())
    src_eviction_rate = total_src_evictions / total_flows if total_flows else 0.0
    print(f"source-table eviction rate across the week: {src_eviction_rate:.4%}")
    print("flow-table eviction rate: 0.0000% (EVAL mode is unbounded by design)")

    all_features = all_feature_names(monday)
    flow_features = tier1_feature_names(monday)

    print("\n=== baseline sweep ===")
    baseline_result, baseline_fits = run_sweep(
        fit_half,
        holdout_half,
        week_df,
        feature_names=all_features,
        rule=EscalationRule.ANY,
        exclude_low_confidence=True,
        src_table_eviction_rate=src_eviction_rate,
        ablation_name="baseline",
    )

    print("=== ablation: per-flow features only ===")
    flow_only_result, _ = run_sweep(
        fit_half,
        holdout_half,
        week_df,
        feature_names=flow_features,
        rule=EscalationRule.ANY,
        exclude_low_confidence=True,
        src_table_eviction_rate=src_eviction_rate,
        ablation_name="per_flow_only",
    )

    print("=== ablation: k_of_n k=2 ===")
    k2_result, _ = run_sweep(
        fit_half,
        holdout_half,
        week_df,
        feature_names=all_features,
        rule=EscalationRule.K_OF_N,
        k=2,
        exclude_low_confidence=True,
        src_table_eviction_rate=src_eviction_rate,
        ablation_name="k_of_n_k2",
    )

    print("=== ablation: k_of_n k=3 ===")
    k3_result, _ = run_sweep(
        fit_half,
        holdout_half,
        week_df,
        feature_names=all_features,
        rule=EscalationRule.K_OF_N,
        k=3,
        exclude_low_confidence=True,
        src_table_eviction_rate=src_eviction_rate,
        ablation_name="k_of_n_k3",
    )

    print("=== ablation: low-confidence features included ===")
    low_conf_result, _ = run_sweep(
        fit_half,
        holdout_half,
        week_df,
        feature_names=all_features,
        rule=EscalationRule.ANY,
        exclude_low_confidence=False,
        src_table_eviction_rate=src_eviction_rate,
        ablation_name="low_confidence_included",
    )

    ablations = {
        "baseline": baseline_result,
        "per_flow_only": flow_only_result,
        "k_of_n_k2": k2_result,
        "k_of_n_k3": k3_result,
        "low_confidence_included": low_conf_result,
    }

    point5 = nearest_point(baseline_result, TARGET_TRIGGER_FREQ_RATE)
    fit_at_5 = baseline_fits[point5.percentile]
    crossings = compute_crossings(week_df, fit_at_5.config.thresholds)
    escalated = escalate_from_crossings(crossings, EscalationRule.ANY, 1)
    freq = trigger_frequency(week_df, crossings, escalated)

    join_rates = build_join_rate_table(eval_metas, fidelity_metas)
    flagged = join_rates[join_rates["join_rate"] < 0.95]
    if len(flagged):
        print("\n!!! join rate below 95% for:")
        print(flagged.to_string(index=False))

    delta = per_class_ablation_delta(baseline_result, flow_only_result)

    write_results(baseline_result, ablations, freq, join_rates, feature_ablation_delta=delta)
    print("\nresults written to results/")


if __name__ == "__main__":
    main()
