"""Does normalising the load-bearing count features fix the Monday->Friday
transfer failure that both generalization experiments (see
eval/generalization_experiments.py, STATUS.md's "Two follow-up
experiments") diagnosed as a tail problem, not a baseline-selection
artifact?

Both experiments failed specifically because the five load-bearing
features (flows_per_src, distinct_dst_ports_per_src,
syn_without_synack_count, bwd_pkt_len_mean, pkt_len_range) are absolute
counts/magnitudes: heavy-tailed and unbounded, so a percentile fitted on
one day's tail can't bound another day's tail when benign traffic *volume*
differs between days (Friday's legitimately busier hosts open more flows,
touch more ports, than anything Monday samples -- median behavior is
identical, only the tail shifts).

This adds three normalised, provably-bounded replacements for the three
per-source counts (dataplane/selector.py's compute_src_features), each a
ratio of an existing counter to flows_per_src:

    port_diversity_ratio = distinct_dst_ports_per_src / flows_per_src
    unanswered_syn_ratio = syn_without_synack_count / flows_per_src
    dst_concentration    = distinct_dst_ips_per_src / flows_per_src

Each numerator is a sub-count of flows_per_src (can't touch more distinct
ports/ips than flows opened, can't have more unanswered SYNs than flows),
so all three are bounded in [0, 1] by construction -- registered as
FeatureKind.BOUNDED_RATIO in dataplane/fitting.py, same saturation-aware
fitting already used for syn_ratio/rst_ratio.

bwd_pkt_len_mean and pkt_len_range are NOT re-derived here: they are
already per-flow magnitudes (not per-source aggregates), and normalising
them would need a different denominator (they're not sub-counts of
anything) -- out of scope for this experiment, called out explicitly in
the report below rather than silently left out.

The pre-existing per-flow ratio `down_up_pkt_ratio` (bwd_pkt_count /
fwd_pkt_count) already covers the "per-flow equivalent" case: it is
normalised (scale-invariant to a flow's packet count, unlike a raw
fwd/bwd packet count would be) but NOT hard-bounded at 1 the way the three
per-source ratios above are (a flow can legitimately have far more
backward than forward packets), so it stays FeatureKind.CONTINUOUS and is
reported on separately, not folded into the "bounded ratio" claim.

=== Part 1: distribution comparison (report first, before any sweep) ===

Monday vs Friday benign distribution (median, p99, p99.5, max, and the
Friday/Monday ratio at p99.5) for the three new ratio features, compared
directly against the same statistics recomputed for the five original
count/magnitude features (matching eval/generalization_experiments.py's
LOAD_BEARING_FEATURES table, with the previously-missing Friday/Monday
p99.5 ratio column added so the two tables line up one-for-one).

=== Part 2: percentile x k sweep with ratios substituted for counts ===

Re-runs eval/percentile_k_sweep.py's exact grid (6 percentiles x 4 k
values) and fitting discipline (Monday PCAP chronological fit/holdout,
exclude_low_confidence=False), but fits on a feature set with the three
raw per-source counts (flows_per_src, distinct_dst_ports_per_src,
syn_without_synack_count) replaced by their ratio equivalents.
distinct_dst_ips_per_src / dst_concentration is included in the same
substitution even though it isn't one of the five original load-bearing
features, since it's the same per-source-count pathology and the natural
per-source feature set to fit on is "whichever form of each per-source
quantity we're using," not a mix of counts and ratios for different
counters. Reports escalation rate, per-class recall, and the
Monday->Friday FPR gap, in the same schema as percentile_k_sweep.csv for
direct row-by-row comparison.

Zero-day property unchanged from every other script in this family: all
three new features are unsupervised transforms of already-existing
counters, fitted on benign-only traffic (dataplane/fitting.py's assertion
enforces this identically regardless of which columns are fit).

Requires the PCAP feature cache to have been regenerated under
FEATURE_SCHEMA_VERSION="5" (dataplane/selector.py) -- the three new ratio
columns don't exist in older cached parquets and load_monday_pcap /
load_friday_pcap will re-run extraction automatically on a cache miss.

Pure re-analysis / one feature-extraction pass -- no API calls.

Run: python -m eval.ratio_feature_experiments
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from dataplane.fitting import BENIGN_LABEL, fit_thresholds_from_frame
from dataplane.selector import EscalationRule
from eval.run_pcap_sweep import load_friday_pcap, load_monday_pcap
from eval.sweep import (
    compute_crossings,
    escalate_from_crossings,
    split_monday_chronologically,
)

PERCENTILES = (99.5, 99.9, 99.95, 99.99, 99.995, 99.999)
KS = (2, 3, 4, 5)
ATTACK_LABELS = ("PortScan", "DDoS", "Bot")

#: the five features eval/generalization_experiments.py's
#: feature_distribution_comparison() already profiled.
COUNT_FEATURES = (
    "flows_per_src",
    "distinct_dst_ports_per_src",
    "syn_without_synack_count",
    "bwd_pkt_len_mean",
    "pkt_len_range",
)

#: their normalised replacements (per-source only -- bwd_pkt_len_mean and
#: pkt_len_range are per-flow magnitudes, not per-source counts, and have
#: no ratio analog here; see module docstring).
RATIO_FEATURES = (
    "port_diversity_ratio",
    "unanswered_syn_ratio",
    "dst_concentration",
)

#: the three raw per-source counts being replaced by ratios in the sweep.
#: distinct_dst_ips_per_src is included alongside the two
#: originally-named load-bearing counts (flows_per_src,
#: syn_without_synack_count already covered; distinct_dst_ports_per_src
#: too) so every per-source feature is consistently in ratio form, not a
#: mix -- see module docstring.
COUNTS_REPLACED_BY_RATIOS = (
    "flows_per_src",
    "distinct_dst_ports_per_src",
    "distinct_dst_ips_per_src",
    "syn_without_synack_count",
)

OUT_DIR = Path("results")
OUT_DIST_CSV = OUT_DIR / "ratio_feature_distribution.csv"
OUT_SWEEP_CSV = OUT_DIR / "percentile_k_sweep_ratios.csv"


# ------------------------------------------------------------------ data ---

def load_data() -> dict:
    monday_df, _ = load_monday_pcap()
    friday_df, _ = load_friday_pcap()
    monday_fit, monday_holdout = split_monday_chronologically(monday_df)
    combined_holdout = pd.concat([monday_holdout, friday_df], ignore_index=True)
    return {
        "monday_df": monday_df,
        "friday_df": friday_df,
        "monday_fit": monday_fit,
        "monday_holdout": monday_holdout,
        "combined_holdout": combined_holdout,
    }


# ------------------------------------------------------- part 1: distribution

def distribution_row(feat: str, monday_benign: pd.DataFrame, friday_benign: pd.DataFrame) -> dict:
    m = monday_benign[feat].dropna()
    f = friday_benign[feat].dropna()
    m_p99_5 = float(np.percentile(m, 99.5))
    f_p99_5 = float(np.percentile(f, 99.5))
    ratio_p99_5 = (f_p99_5 / m_p99_5) if m_p99_5 != 0 else (float("inf") if f_p99_5 != 0 else None)
    return {
        "feature": feat,
        "monday_median": float(m.median()),
        "friday_median": float(f.median()),
        "monday_p99": float(np.percentile(m, 99)),
        "friday_p99": float(np.percentile(f, 99)),
        "monday_p99_5": m_p99_5,
        "friday_p99_5": f_p99_5,
        "friday_monday_ratio_p99_5": ratio_p99_5,
        "monday_max": float(m.max()),
        "friday_max": float(f.max()),
    }


def distribution_comparison(data: dict) -> List[dict]:
    """Count features (recomputed with the ratio column added, matching
    generalization_experiments.py's numbers otherwise) followed by the
    new ratio features -- one table, directly comparable row-by-row."""
    monday_benign = data["monday_df"]  # Monday is benign-only by construction
    friday_benign = data["friday_df"][data["friday_df"]["label"] == BENIGN_LABEL]
    rows = [distribution_row(f, monday_benign, friday_benign) for f in COUNT_FEATURES]
    rows += [distribution_row(f, monday_benign, friday_benign) for f in RATIO_FEATURES]
    return rows


def print_distribution(rows: List[dict]) -> None:
    print("=" * 130)
    print("Monday vs Friday BENIGN distribution: counts (top) vs their ratio replacements (bottom)")
    print("=" * 130)
    hdr = (f"{'feature':32s} {'mon_med':>10} {'fri_med':>10} {'mon_p99':>10} {'fri_p99':>10} "
           f"{'mon_p99.5':>10} {'fri_p99.5':>10} {'fri/mon@p99.5':>14} {'mon_max':>12} {'fri_max':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ratio = r["friday_monday_ratio_p99_5"]
        ratio_s = f"{ratio:.2f}x" if ratio is not None and ratio != float("inf") else ("inf" if ratio == float("inf") else "n/a")
        print(f"{r['feature']:32s} {r['monday_median']:>10.4g} {r['friday_median']:>10.4g} "
              f"{r['monday_p99']:>10.4g} {r['friday_p99']:>10.4g} "
              f"{r['monday_p99_5']:>10.4g} {r['friday_p99_5']:>10.4g} {ratio_s:>14} "
              f"{r['monday_max']:>12.4g} {r['friday_max']:>12.4g}")
        if r["feature"] == COUNT_FEATURES[-1]:
            print("-" * len(hdr))
    print()


# ------------------------------------------------------------- part 2: sweep

def counts_by_label(df: pd.DataFrame, escalated: pd.Series, label_col: str = "label") -> dict:
    total = df[label_col].value_counts().to_dict()
    esc = df.loc[escalated, label_col].value_counts().to_dict()
    return {"total": {k: int(v) for k, v in total.items()}, "escalated": {k: int(esc.get(k, 0)) for k in total}}


def ratio_feature_names(df: pd.DataFrame) -> List[str]:
    """All fittable columns, with the raw per-source counts dropped in
    favor of their ratio replacements -- everything else (Tier-1
    per-flow features, including down_up_pkt_ratio and bwd_pkt_len_mean/
    pkt_len_range, which have no per-source-count analog) is unchanged
    from the default feature set."""
    from dataplane.fitting import NON_FEATURE_COLUMNS

    return sorted(
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS
        and not c.endswith("__low_confidence")
        and c not in COUNTS_REPLACED_BY_RATIOS
    )


def run_sweep(data: dict) -> List[dict]:
    monday_fit = data["monday_fit"]
    monday_holdout = data["monday_holdout"]
    friday_df = data["friday_df"]
    combined_holdout = data["combined_holdout"]
    feature_names = ratio_feature_names(monday_fit)

    rows: List[dict] = []
    for pct in PERCENTILES:
        fit = fit_thresholds_from_frame(
            monday_fit,
            feature_names=feature_names,
            percentile=pct,
            rule=EscalationRule.K_OF_N,
            k=2,
            exclude_low_confidence=False,
        )
        config = fit.config
        mon_crossings = compute_crossings(monday_holdout, config.thresholds)
        fri_crossings = compute_crossings(friday_df, config.thresholds)
        comb_crossings = compute_crossings(combined_holdout, config.thresholds)

        for k in KS:
            mon_esc = escalate_from_crossings(mon_crossings, EscalationRule.K_OF_N, k)
            fri_esc = escalate_from_crossings(fri_crossings, EscalationRule.K_OF_N, k)
            comb_esc = escalate_from_crossings(comb_crossings, EscalationRule.K_OF_N, k)

            mon_counts = counts_by_label(monday_holdout, mon_esc)
            fri_counts = counts_by_label(friday_df, fri_esc)
            comb_counts = counts_by_label(combined_holdout, comb_esc)

            mon_fpr = mon_counts["escalated"].get("BENIGN", 0) / mon_counts["total"].get("BENIGN", 1)
            fri_fpr = fri_counts["escalated"].get("BENIGN", 0) / fri_counts["total"].get("BENIGN", 1)
            fri_esc_rate = int(fri_esc.sum()) / len(friday_df) if len(friday_df) else None
            comb_esc_rate = int(comb_esc.sum()) / len(combined_holdout) if len(combined_holdout) else None

            recall = {
                l: (fri_counts["escalated"].get(l, 0) / fri_counts["total"][l]) if fri_counts["total"].get(l) else None
                for l in ATTACK_LABELS
            }
            attacks_escalated = sum(fri_counts["escalated"].get(l, 0) for l in ATTACK_LABELS)
            attacks_total = sum(fri_counts["total"].get(l, 0) for l in ATTACK_LABELS)
            pooled_recall = (attacks_escalated / attacks_total) if attacks_total else None

            fri_precision = (attacks_escalated / int(fri_esc.sum())) if int(fri_esc.sum()) else None
            comb_attacks_escalated = sum(comb_counts["escalated"].get(l, 0) for l in ATTACK_LABELS)
            comb_precision = (comb_attacks_escalated / int(comb_esc.sum())) if int(comb_esc.sum()) else None

            rows.append({
                "percentile": pct, "k": k, "config_hash": config.config_hash,
                "monday_fpr": mon_fpr, "friday_escalation_rate": fri_esc_rate, "friday_fpr": fri_fpr,
                "fpr_gap": fri_fpr - mon_fpr, "combined_escalation_rate": comb_esc_rate,
                "recall_portscan": recall.get("PortScan"), "recall_ddos": recall.get("DDoS"),
                "recall_bot": recall.get("Bot"), "pooled_attack_recall": pooled_recall,
                "friday_precision": fri_precision, "combined_precision": comb_precision,
                "friday_escalated_n": int(fri_esc.sum()), "combined_escalated_n": int(comb_esc.sum()),
                "combined_n": len(combined_holdout), "n_thresholds_fit": len(config.thresholds),
            })
    return rows


def print_sweep(rows: List[dict], feature_names: List[str]) -> None:
    print("=" * 150)
    print(f"percentile x k grid -- ratio features substituted for counts ({', '.join(COUNTS_REPLACED_BY_RATIOS)} "
          f"-> {', '.join(RATIO_FEATURES)})")
    print(f"fitted feature set ({len(feature_names)}): {', '.join(feature_names)}")
    print("=" * 150)
    hdr = (f"{'pct':>8} {'k':>2} | {'mon_fpr':>8} {'fri_esc':>8} {'fri_fpr':>8} {'gap':>8} "
           f"{'comb_esc':>9} | {'PortScan':>9} {'DDoS':>7} {'Bot':>7} {'pooled':>7} | "
           f"{'fri_prec':>8} {'comb_prec':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def pct_or_dash(v):
            return f"{v:.2%}" if v is not None else "  n/a"
        print(f"{r['percentile']:>8} {r['k']:>2} | {pct_or_dash(r['monday_fpr']):>8} "
              f"{pct_or_dash(r['friday_escalation_rate']):>8} {pct_or_dash(r['friday_fpr']):>8} "
              f"{r['fpr_gap']*100:>+7.2f}% {pct_or_dash(r['combined_escalation_rate']):>9} | "
              f"{pct_or_dash(r['recall_portscan']):>9} {pct_or_dash(r['recall_ddos']):>7} "
              f"{pct_or_dash(r['recall_bot']):>7} {pct_or_dash(r['pooled_attack_recall']):>7} | "
              f"{pct_or_dash(r['friday_precision']):>8} {pct_or_dash(r['combined_precision']):>9}")
    print()


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (v if v != float("inf") else "inf") for k, v in r.items()})
    print(f"wrote {path}")


def main() -> int:
    print("loading cached PCAP feature parquets (schema version 5 -- forces fresh extraction if missing) ...")
    data = load_data()
    print(f"Monday total={len(data['monday_df'])}  fit_half={len(data['monday_fit'])}  "
          f"holdout_half={len(data['monday_holdout'])}")
    print(f"Friday total={len(data['friday_df'])}  by label: {data['friday_df']['label'].value_counts().to_dict()}")
    for f in RATIO_FEATURES:
        assert f in data["monday_df"].columns, (
            f"{f} missing from cached frame -- re-run with a clean cache "
            "(dataplane.selector.FEATURE_SCHEMA_VERSION must be 5+)"
        )
    print()

    dist_rows = distribution_comparison(data)
    print_distribution(dist_rows)
    write_csv(dist_rows, OUT_DIST_CSV)
    print()

    feature_names = ratio_feature_names(data["monday_fit"])
    sweep_rows = run_sweep(data)
    print_sweep(sweep_rows, feature_names)
    write_csv(sweep_rows, OUT_SWEEP_CSV)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
