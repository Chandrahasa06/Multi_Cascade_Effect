"""Two free (zero-API-call) experiments probing WHY Monday-fitted
thresholds don't transfer to Friday (eval/percentile_k_sweep.py found
the Friday/Monday benign-FPR ratio exploding from 7.7x to literal
infinity as the percentile tightens), and whether combination-based
signatures generalize better than independent per-feature thresholds.

=== Experiment 1: pooled Monday+Friday-benign fitting ===

Refits thresholds on Monday benign + a held-out-safe half of Friday's
own BENIGN-labelled flows (never attack-labelled Friday flows — the
zero-day property is asserted, not assumed, exactly the way
dataplane/fitting.py already asserts it for the single-day case).
Evaluates at the SAME percentile x k grid as eval/percentile_k_sweep.py,
against held-out halves of BOTH days' benign traffic, and reports
whether the Monday-only fit's benign-FPR gap closes when re-measured
against the identical held-out populations (the earlier 7.7x-to-infinity
numbers used the FULL Friday benign population for the Monday-only fit,
which this script does NOT reuse directly -- re-derived here against the
same held-out Friday-benign half used for the pooled fit, so the
before/after comparison is apples-to-apples; expect small numeric drift
from the previous message for that reason, not a bug).

=== Experiment 2: signature table (combination-based) ===

Bins the five load-bearing features (flows_per_src,
distinct_dst_ports_per_src, syn_without_synack_count, bwd_pkt_len_mean,
pkt_len_range) into quantile-derived ranges (edges fixed once, from the
pooled fit population, so the two K variants below differ ONLY in which
signatures they've seen -- not in how values are binned, keeping this a
controlled comparison). A flow's "signature" is its 5-tuple of bin
indices (NaN gets its own sentinel bin per feature, since undefined
values are themselves informative). Builds K = the set of signatures
seen at least `floor` times in a training population, two ways (Monday
fit-half only vs Monday+Friday-benign fit-half), and escalates any flow
whose signature isn't in K.

Both experiments read only already-cached PCAP feature parquets --
no simulation, no API calls.

Run: python -m eval.generalization_experiments
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from dataplane.fitting import BENIGN_LABEL, fit_thresholds_from_frame
from dataplane.selector import EscalationRule
from eval.run_pcap_sweep import load_friday_pcap, load_monday_pcap
from eval.sweep import (
    compute_crossings,
    escalate_from_crossings,
    split_monday_chronologically,  # generic chronological halver, not Monday-specific
)

PERCENTILES = (99.5, 99.9, 99.95, 99.99, 99.995, 99.999)
KS = (2, 3, 4, 5)
ATTACK_LABELS = ("PortScan", "DDoS", "Bot")
LOAD_BEARING_FEATURES = (
    "flows_per_src", "distinct_dst_ports_per_src", "syn_without_synack_count",
    "bwd_pkt_len_mean", "pkt_len_range",
)
N_BINS = 6
FLOORS = (1, 5, 20)

OUT_DIR = Path("results")


# ------------------------------------------------------------------ data ---

def load_data():
    monday_df, _ = load_monday_pcap()
    friday_df, _ = load_friday_pcap()

    monday_fit, monday_holdout = split_monday_chronologically(monday_df)

    friday_benign = friday_df[friday_df["label"] == BENIGN_LABEL].reset_index(drop=True)
    friday_attacks = friday_df[friday_df["label"] != BENIGN_LABEL].reset_index(drop=True)
    friday_benign_fit, friday_benign_holdout = split_monday_chronologically(friday_benign)

    pooled_fit = pd.concat([monday_fit, friday_benign_fit], ignore_index=True)
    assert (pooled_fit["label"] == BENIGN_LABEL).all(), (
        "pooled fit set contains a non-BENIGN label -- the zero-day property "
        "(thresholds/signatures fit only on benign traffic) must hold"
    )

    friday_eval = pd.concat([friday_benign_holdout, friday_attacks], ignore_index=True)
    combined_holdout = pd.concat([monday_holdout, friday_eval], ignore_index=True)

    return {
        "monday_df": monday_df, "friday_df": friday_df,
        "monday_fit": monday_fit, "monday_holdout": monday_holdout,
        "friday_benign": friday_benign, "friday_attacks": friday_attacks,
        "friday_benign_fit": friday_benign_fit, "friday_benign_holdout": friday_benign_holdout,
        "pooled_fit": pooled_fit, "friday_eval": friday_eval, "combined_holdout": combined_holdout,
    }


def counts_by_label(df: pd.DataFrame, escalated: pd.Series, label_col: str = "label") -> dict:
    total = df[label_col].value_counts().to_dict()
    esc = df.loc[escalated, label_col].value_counts().to_dict()
    return {"total": {k: int(v) for k, v in total.items()}, "escalated": {k: int(esc.get(k, 0)) for k in total}}


def grid_row(config, k, monday_holdout, friday_eval, combined_holdout) -> dict:
    mon_crossings = compute_crossings(monday_holdout, config.thresholds)
    fri_crossings = compute_crossings(friday_eval, config.thresholds)
    comb_crossings = compute_crossings(combined_holdout, config.thresholds)

    mon_esc = escalate_from_crossings(mon_crossings, EscalationRule.K_OF_N, k)
    fri_esc = escalate_from_crossings(fri_crossings, EscalationRule.K_OF_N, k)
    comb_esc = escalate_from_crossings(comb_crossings, EscalationRule.K_OF_N, k)

    mon_counts = counts_by_label(monday_holdout, mon_esc)
    fri_counts = counts_by_label(friday_eval, fri_esc)
    comb_counts = counts_by_label(combined_holdout, comb_esc)

    mon_fpr = mon_counts["escalated"].get("BENIGN", 0) / mon_counts["total"].get("BENIGN", 1)
    fri_benign_total = fri_counts["total"].get("BENIGN", 0)
    fri_fpr = (fri_counts["escalated"].get("BENIGN", 0) / fri_benign_total) if fri_benign_total else None
    gap_ratio = (fri_fpr / mon_fpr) if (fri_fpr is not None and mon_fpr > 0) else (float("inf") if fri_fpr else None)

    recall = {
        l: (fri_counts["escalated"].get(l, 0) / fri_counts["total"][l]) if fri_counts["total"].get(l) else None
        for l in ATTACK_LABELS
    }
    attacks_escalated = sum(fri_counts["escalated"].get(l, 0) for l in ATTACK_LABELS)
    attacks_total = sum(fri_counts["total"].get(l, 0) for l in ATTACK_LABELS)
    pooled_recall = (attacks_escalated / attacks_total) if attacks_total else None
    fri_precision = (attacks_escalated / int(fri_esc.sum())) if int(fri_esc.sum()) else None

    return {
        "monday_fpr": mon_fpr, "friday_fpr": fri_fpr, "gap_ratio": gap_ratio,
        "friday_benign_n": fri_benign_total,
        "monday_escalation_rate": mon_fpr,
        "friday_escalation_rate": (int(fri_esc.sum()) / len(friday_eval)) if len(friday_eval) else None,
        "combined_escalation_rate": (int(comb_esc.sum()) / len(combined_holdout)) if len(combined_holdout) else None,
        "recall_portscan": recall.get("PortScan"), "recall_ddos": recall.get("DDoS"), "recall_bot": recall.get("Bot"),
        "pooled_attack_recall": pooled_recall, "friday_precision": fri_precision,
    }


# ------------------------------------------------------------- experiment 1

def run_experiment1(data: dict) -> List[dict]:
    rows = []
    for pct in PERCENTILES:
        monday_only_fit = fit_thresholds_from_frame(
            data["monday_fit"], percentile=pct, rule=EscalationRule.K_OF_N, k=2, exclude_low_confidence=False,
        )
        pooled_fit_result = fit_thresholds_from_frame(
            data["pooled_fit"], percentile=pct, rule=EscalationRule.K_OF_N, k=2, exclude_low_confidence=False,
        )
        for k in KS:
            for variant, fit in (("monday_only", monday_only_fit), ("pooled", pooled_fit_result)):
                r = grid_row(fit.config, k, data["monday_holdout"], data["friday_eval"], data["combined_holdout"])
                r.update({"variant": variant, "percentile": pct, "k": k, "config_hash": fit.config.config_hash,
                          "n_thresholds": len(fit.config.thresholds)})
                rows.append(r)
    return rows


def feature_distribution_comparison(data: dict) -> List[dict]:
    monday_benign = data["monday_df"]  # Monday is benign-only by construction
    friday_benign = data["friday_df"][data["friday_df"]["label"] == BENIGN_LABEL]
    out = []
    for feat in LOAD_BEARING_FEATURES:
        m = monday_benign[feat].dropna()
        f = friday_benign[feat].dropna()
        out.append({
            "feature": feat,
            "monday_median": float(m.median()), "friday_median": float(f.median()),
            "monday_p99": float(np.percentile(m, 99)), "friday_p99": float(np.percentile(f, 99)),
            "monday_p99_5": float(np.percentile(m, 99.5)), "friday_p99_5": float(np.percentile(f, 99.5)),
            "monday_max": float(m.max()), "friday_max": float(f.max()),
        })
    return out


# ------------------------------------------------------------- experiment 2

def compute_bin_edges(df: pd.DataFrame, features: Sequence[str], n_bins: int) -> Dict[str, np.ndarray]:
    """Interior quantile edges per feature (n_bins-1 cut points), from
    `df`. duplicates dropped (heavy point-mass features, e.g. most-benign
    syn_without_synack_count==0, legitimately collapse to fewer effective
    bins near that mass) -- fixed once and reused for BOTH K variants so
    only "which signatures were seen" varies, not "how values are binned"."""
    edges = {}
    for feat in features:
        vals = df[feat].dropna().to_numpy(dtype=float)
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        cuts = np.unique(np.quantile(vals, qs))
        edges[feat] = cuts
    return edges


def signatures_for(df: pd.DataFrame, features: Sequence[str], edges: Dict[str, np.ndarray]) -> List[tuple]:
    cols = []
    for feat in features:
        vals = df[feat].to_numpy(dtype=float)
        idx = np.digitize(vals, edges[feat]).astype(object)
        nan_mask = np.isnan(vals)
        idx[nan_mask] = -1  # NaN/undefined is its own sentinel bin, not silently coerced
        cols.append(idx)
    return list(zip(*cols))


def build_K(signatures: List[tuple], floor: int) -> set:
    counts = Counter(signatures)
    return {sig for sig, c in counts.items() if c >= floor}


def evaluate_K(df: pd.DataFrame, signatures: List[tuple], K: set) -> pd.Series:
    escalated = pd.Series([sig not in K for sig in signatures], index=df.index)
    return escalated


def summarize_K_run(df: pd.DataFrame, escalated: pd.Series) -> dict:
    counts = counts_by_label(df, escalated)
    fpr = counts["escalated"].get("BENIGN", 0) / counts["total"].get("BENIGN", 1) if counts["total"].get("BENIGN") else None
    recall = {
        l: (counts["escalated"].get(l, 0) / counts["total"][l]) if counts["total"].get(l) else None
        for l in ATTACK_LABELS
    }
    attacks_escalated = sum(counts["escalated"].get(l, 0) for l in ATTACK_LABELS)
    attacks_total = sum(counts["total"].get(l, 0) for l in ATTACK_LABELS)
    pooled_recall = (attacks_escalated / attacks_total) if attacks_total else None
    total = len(df)
    esc_rate = int(escalated.sum()) / total if total else None
    return {"escalation_rate": esc_rate, "benign_fpr": fpr, "recall": recall, "pooled_attack_recall": pooled_recall,
            "n": total, "n_escalated": int(escalated.sum())}


def run_experiment2(data: dict) -> dict:
    edges = compute_bin_edges(data["pooled_fit"], LOAD_BEARING_FEATURES, N_BINS)

    sig_monday_fit = signatures_for(data["monday_fit"], LOAD_BEARING_FEATURES, edges)
    sig_pooled_fit = signatures_for(data["pooled_fit"], LOAD_BEARING_FEATURES, edges)

    sig_monday_holdout = signatures_for(data["monday_holdout"], LOAD_BEARING_FEATURES, edges)
    sig_friday_eval = signatures_for(data["friday_eval"], LOAD_BEARING_FEATURES, edges)
    sig_combined_holdout = signatures_for(data["combined_holdout"], LOAD_BEARING_FEATURES, edges)

    out = {"bin_edges": {k: v.tolist() for k, v in edges.items()}, "floors": {}}
    for floor in FLOORS:
        K_monday = build_K(sig_monday_fit, floor)
        K_pooled = build_K(sig_pooled_fit, floor)

        row = {}
        for variant, K, train_sig in (("monday_only", K_monday, sig_monday_fit), ("pooled", K_pooled, sig_pooled_fit)):
            esc_mon = evaluate_K(data["monday_holdout"], sig_monday_holdout, K)
            esc_fri = evaluate_K(data["friday_eval"], sig_friday_eval, K)
            esc_comb = evaluate_K(data["combined_holdout"], sig_combined_holdout, K)
            row[variant] = {
                "n_signatures_in_K": len(K),
                "n_distinct_signatures_seen_in_training": len(set(train_sig)),
                "monday_holdout": summarize_K_run(data["monday_holdout"], esc_mon),
                "friday_eval": summarize_K_run(data["friday_eval"], esc_fri),
                "combined_holdout": summarize_K_run(data["combined_holdout"], esc_comb),
            }
        out["floors"][floor] = row
    return out


# --------------------------------------------------------------------- main

def print_experiment1(rows: List[dict], dist: List[dict]):
    print("=" * 150)
    print("EXPERIMENT 1: Monday-only fit vs Monday+Friday-benign pooled fit")
    print("=" * 150)
    hdr = (f"{'pct':>8} {'k':>2} {'variant':>12} | {'mon_fpr':>9} {'fri_fpr':>9} {'gap_ratio':>10} | "
           f"{'comb_esc':>9} | {'PortScan':>9} {'DDoS':>7} {'Bot':>7} {'pooled':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def p(v):
            return f"{v:.3%}" if isinstance(v, float) and not np.isinf(v) else ("  inf" if v == float("inf") else "  n/a")
        gap = "inf" if r["gap_ratio"] == float("inf") else (f"{r['gap_ratio']:.1f}x" if r["gap_ratio"] is not None else "n/a")
        print(f"{r['percentile']:>8} {r['k']:>2} {r['variant']:>12} | {p(r['monday_fpr']):>9} {p(r['friday_fpr']):>9} "
              f"{gap:>10} | {p(r['combined_escalation_rate']):>9} | "
              f"{p(r['recall_portscan']):>9} {p(r['recall_ddos']):>7} {p(r['recall_bot']):>7} {p(r['pooled_attack_recall']):>7}")
    print()
    print("Feature distribution: Monday benign vs Friday benign (full populations)")
    print(f"{'feature':35s} {'mon_median':>11} {'fri_median':>11} {'mon_p99':>10} {'fri_p99':>10} "
          f"{'mon_p99.5':>10} {'fri_p99.5':>10} {'mon_max':>12} {'fri_max':>12}")
    for d in dist:
        print(f"{d['feature']:35s} {d['monday_median']:>11.4g} {d['friday_median']:>11.4g} "
              f"{d['monday_p99']:>10.4g} {d['friday_p99']:>10.4g} "
              f"{d['monday_p99_5']:>10.4g} {d['friday_p99_5']:>10.4g} "
              f"{d['monday_max']:>12.4g} {d['friday_max']:>12.4g}")
    print()


def print_experiment2(result: dict):
    print("=" * 150)
    print("EXPERIMENT 2: signature table K (combination-based) vs percentile thresholds")
    print("=" * 150)
    print(f"bin edges (from pooled Monday+Friday-benign fit population, {N_BINS} bins, fixed for both K variants):")
    for feat, edges in result["bin_edges"].items():
        print(f"  {feat:35s} {['%.4g' % e for e in edges]}")
    print()
    for floor, row in result["floors"].items():
        print(f"--- floor={floor} (signature must appear >= {floor}x in training to enter K) ---")
        for variant in ("monday_only", "pooled"):
            v = row[variant]
            print(f"  K_{variant}: {v['n_signatures_in_K']} signatures in K "
                  f"(of {v['n_distinct_signatures_seen_in_training']} distinct seen in training)")
            for pop in ("monday_holdout", "friday_eval", "combined_holdout"):
                s = v[pop]
                rec = s["recall"]
                rec_str = ", ".join(f"{l}={rec[l]:.1%}" if rec[l] is not None else f"{l}=n/a" for l in ATTACK_LABELS)
                fpr_str = f"{s['benign_fpr']:.3%}" if s["benign_fpr"] is not None else "n/a"
                print(f"    {pop:18s} escalation={s['escalation_rate']:.3%}  benign_fpr={fpr_str}  "
                      f"{rec_str}  pooled_recall={s['pooled_attack_recall']:.1%}" if s['pooled_attack_recall'] is not None
                      else f"    {pop:18s} escalation={s['escalation_rate']:.3%}  benign_fpr={fpr_str}  {rec_str}")
        print()


def main() -> int:
    print("loading cached PCAP feature parquets ...")
    data = load_data()
    print(f"Monday: total={len(data['monday_df']):,}  fit={len(data['monday_fit']):,}  holdout={len(data['monday_holdout']):,}")
    print(f"Friday: total={len(data['friday_df']):,}  benign={len(data['friday_benign']):,} "
          f"(fit={len(data['friday_benign_fit']):,}, holdout={len(data['friday_benign_holdout']):,})  "
          f"attacks={len(data['friday_attacks']):,}")
    print(f"pooled fit set: {len(data['pooled_fit']):,} (Monday fit {len(data['monday_fit']):,} "
          f"+ Friday-benign fit {len(data['friday_benign_fit']):,}) -- all BENIGN, asserted above")
    print()

    exp1_rows = run_experiment1(data)
    dist = feature_distribution_comparison(data)
    print_experiment1(exp1_rows, dist)

    exp2_result = run_experiment2(data)
    print_experiment2(exp2_result)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "generalization_exp1.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp1_rows[0].keys()))
        writer.writeheader()
        for r in exp1_rows:
            writer.writerow({k: (v if v != float("inf") else "inf") for k, v in r.items()})
    with open(OUT_DIR / "generalization_feature_dist.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dist[0].keys()))
        writer.writeheader()
        writer.writerows(dist)
    with open(OUT_DIR / "generalization_exp2.json", "w", encoding="utf-8") as f:
        json.dump(exp2_result, f, indent=2, default=str)
    print("wrote results/generalization_exp1.csv, results/generalization_feature_dist.csv, results/generalization_exp2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
