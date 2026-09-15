"""Clean re-run of the cross-day test with the adapter confound removed:
fit both the 5-feature and 18-feature count-AND-ratio configs on Monday
**CSV** (adapter1) benign, chronologically split -- the same adapter as
Tuesday/Wednesday/Thursday -- instead of reusing Monday PCAP-fitted
thresholds. Only the fitting source changes; rule, percentile, and
feature sets are unchanged from results/final_selector_report.md /
results/seven_feature_selector_report.md.

`eval/cross_day_followups.py` found the previous cross-day test
confounded: Monday CSV vs Monday PCAP, same day, same PCAP-fitted
thresholds, showed a 100-600x gap in per-source COUNT feature crossing
rates. This script removes that confound by fitting on Monday CSV
directly, so any remaining Tue-Thu weakness is attributable to real
generalization limits, not an adapter mismatch.

The original PCAP-fitted, PCAP-tested Monday/Friday results
(`results/final_selector_report.md`, `results/seven_feature_selector_report.md`)
are recomputed here unchanged (same code path, same cached PCAP parquets)
and reported in a separate section for reference -- they never crossed
the adapter boundary and aren't affected by this refit.

No API calls, no re-simulation -- reads already-cached feature parquets.

Run: python -m eval.csv_refit_cross_day
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from dataplane.fitting import BENIGN_LABEL, fit_thresholds_from_frame
from dataplane.selector import EscalationRule
from eval.final_selector_report import fit_final as fit_18_pcap
from eval.five_feature_cross_day import (
    FRIDAY_CLASSES,
    THURSDAY_CLASSES,
    TUESDAY_CLASSES,
    WEDNESDAY_CLASSES,
    load_csv_day,
    normalize_thursday_labels,
)
from eval.reduced_selector_report import (
    ATTACK_LABELS,
    COUNT_FEATURES,
    RATIO_FEATURES,
    load_data as load_pcap_data,
)
from eval.run_pcap_sweep import load_friday_pcap
from eval.seven_feature_selector import COUNT_5, RATIO_5, fit_subset as fit_5_pcap
from eval.sweep import compute_crossings, split_monday_chronologically

OUT_MD = Path("results/csv_refit_cross_day_report.md")
PERCENTILE = 99.5
TINY_CLASS_FLOOR = 50


def fit_subset_generic(fit_df: pd.DataFrame, count_feats, ratio_feats, percentile: float = PERCENTILE):
    features = list(count_feats) + list(ratio_feats)
    return fit_thresholds_from_frame(
        fit_df, feature_names=features, percentile=percentile,
        rule=EscalationRule.K_OF_N, k=2, exclude_low_confidence=False,
    )


def escalate_generic(df: pd.DataFrame, fit, count_feats, ratio_feats) -> pd.Series:
    count_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in count_feats}
    ratio_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in ratio_feats}
    count_x = compute_crossings(df, count_thresholds)
    ratio_x = compute_crossings(df, ratio_thresholds)
    count_any = count_x.any(axis=1) if count_x.shape[1] else pd.Series(False, index=df.index)
    ratio_any = ratio_x.any(axis=1) if ratio_x.shape[1] else pd.Series(False, index=df.index)
    return count_any & ratio_any


def day_report(df: pd.DataFrame, classes: List[str], escalated: pd.Series) -> dict:
    is_benign = df["label"] == BENIGN_LABEL
    n_total = len(df)
    n_benign = int(is_benign.sum())
    n_escalated = int(escalated.sum())
    n_escalated_benign = int((escalated & is_benign).sum())
    per_class = {}
    for cls in classes:
        mask = df["label"] == cls
        n = int(mask.sum())
        n_esc = int((escalated & mask).sum())
        per_class[cls] = {
            "n": n, "escalated": n_esc, "missed": n - n_esc,
            "recall": (n_esc / n) if n else None,
            "unreliable": 0 < n < TINY_CLASS_FLOOR,
        }
    return {
        "n_total": n_total, "n_benign": n_benign, "n_escalated": n_escalated,
        "n_escalated_benign": n_escalated_benign,
        "escalation_rate": n_escalated / n_total if n_total else None,
        "benign_fpr": n_escalated_benign / n_benign if n_benign else None,
        "per_class": per_class,
    }


def render_day(name: str, config_label: str, r: dict) -> List[str]:
    lines = [f"### {name} -- {config_label}\n"]
    lines.append(f"Total flows: {r['n_total']:,}  |  Escalated: **{r['n_escalated']:,}** "
                 f"({r['escalation_rate']:.3%})  |  Benign FPR: **{r['benign_fpr']:.3%}**\n")
    lines.append("| class | n | escalated | recall | missed |")
    lines.append("|---|---|---|---|---|")
    for cls, c in r["per_class"].items():
        if c["n"] == 0:
            continue
        recall_str = f"{c['recall']:.1%}" + (" (unreliable, n<50)" if c["unreliable"] else "")
        lines.append(f"| {cls} | {c['n']:,} | {c['escalated']:,} | {recall_str} | {c['missed']:,} |")
    lines.append("")
    return lines


def main() -> int:
    print("loading Monday CSV, splitting chronologically, fitting both configs ...")
    monday_csv = load_csv_day("monday")
    monday_csv_fit, monday_csv_holdout = split_monday_chronologically(monday_csv)
    assert (monday_csv_fit["label"] == BENIGN_LABEL).all(), "Monday CSV fit half must be benign-only"
    print(f"Monday CSV fit={len(monday_csv_fit):,}  holdout={len(monday_csv_holdout):,}")

    # The CSV-path cache is schema4, which predates the 4->5 bump that added
    # port_diversity_ratio/unanswered_syn_ratio/dst_concentration -- those
    # three RATIO columns don't exist in this cache. Rather than trigger a
    # fresh multi-day CSV-adapter re-simulation just to regenerate schema5
    # caches, use the intersection and report exactly what that leaves,
    # instead of silently dropping columns or crashing.
    count_18_csv = [f for f in COUNT_FEATURES if f in monday_csv_fit.columns]
    ratio_18_csv = [f for f in RATIO_FEATURES if f in monday_csv_fit.columns]
    missing_ratio_18 = [f for f in RATIO_FEATURES if f not in monday_csv_fit.columns]
    if missing_ratio_18:
        print(f"NOTE: CSV cache is schema4 (predates the 4->5 bump); {missing_ratio_18} not "
              f"present, dropped from the '18-feature' config for this CSV-fit run only "
              f"({len(count_18_csv)} COUNT + {len(ratio_18_csv)} RATIO = "
              f"{len(count_18_csv)+len(ratio_18_csv)} features).")

    fit5_csv = fit_subset_generic(monday_csv_fit, COUNT_5, RATIO_5, PERCENTILE)
    fit18_csv = fit_subset_generic(monday_csv_fit, count_18_csv, ratio_18_csv, PERCENTILE)
    print(f"5-feature (CSV-fit): {len(fit5_csv.config.thresholds)}/5 active")
    print(f"18-feature (CSV-fit): {len(fit18_csv.config.thresholds)}/{len(count_18_csv)+len(ratio_18_csv)} active")
    print()

    # holdout FPR, both configs
    esc5_holdout = escalate_generic(monday_csv_holdout, fit5_csv, COUNT_5, RATIO_5)
    esc18_holdout = escalate_generic(monday_csv_holdout, fit18_csv, count_18_csv, ratio_18_csv)
    fpr5_holdout = esc5_holdout.mean()
    fpr18_holdout = esc18_holdout.mean()
    print(f"Monday CSV holdout FPR -- 5-feature: {fpr5_holdout:.3%}   18-feature: {fpr18_holdout:.3%}")
    print()

    print("loading Tuesday/Wednesday/Thursday (CSV, cached) ...")
    tuesday_df = load_csv_day("tuesday")
    wednesday_df = load_csv_day("wednesday")
    thursday_df = normalize_thursday_labels(load_csv_day("thursday_morning_webattacks", "thursday_afternoon_infiltration"))

    days = [
        ("Tuesday", tuesday_df, TUESDAY_CLASSES),
        ("Wednesday", wednesday_df, WEDNESDAY_CLASSES),
        ("Thursday", thursday_df, THURSDAY_CLASSES),
    ]

    lines: List[str] = []
    lines.append("# Clean cross-day test: Monday-CSV-fitted configs on Tue/Wed/Thu\n")
    lines.append(
        "Both configs refit on Monday **CSV** (adapter1) benign, chronologically split, "
        f"p{PERCENTILE} -- the same adapter as Tuesday-Thursday, removing the CSV/PCAP "
        "confound `eval/cross_day_followups.py` found in the previous version of this test "
        "(a 100-600x same-day crossing-rate gap on the per-source COUNT features when PCAP-"
        "fitted thresholds were applied to CSV data). No API calls, no re-simulation.\n\n"
        f"Monday CSV fit-half n={len(monday_csv_fit):,}, holdout n={len(monday_csv_holdout):,} "
        "(zero-day property asserted: fit half is 100% BENIGN).\n"
    )
    lines.append(f"**Monday CSV holdout FPR -- 5-feature: {fpr5_holdout:.3%}   "
                 f"18-feature: {fpr18_holdout:.3%}** (where the operating point actually "
                 f"landed on the same adapter/day used for fitting).\n")
    lines.append(f"5-feature active thresholds: {len(fit5_csv.config.thresholds)}/5. "
                 f"18-feature active thresholds: {len(fit18_csv.config.thresholds)}/"
                 f"{len(count_18_csv)+len(ratio_18_csv)} "
                 f"({'all 18 available' if not missing_ratio_18 else f'schema4 cache is missing {missing_ratio_18}, so this run uses {len(count_18_csv)+len(ratio_18_csv)} of the full 18-feature set'}).\n")

    lines.append("## Tuesday / Wednesday / Thursday, Monday-CSV-fitted configs\n")
    summary_rows = []
    for name, df, classes in days:
        esc5 = escalate_generic(df, fit5_csv, COUNT_5, RATIO_5)
        esc18 = escalate_generic(df, fit18_csv, count_18_csv, ratio_18_csv)
        r5 = day_report(df, classes, esc5)
        r18 = day_report(df, classes, esc18)
        lines.extend(render_day(name, "5-feature (Monday-CSV-fitted)", r5))
        lines.extend(render_day(name, "18-feature (Monday-CSV-fitted)", r18))
        print(f"{name}: 5-feat esc={r5['escalation_rate']:.3%} fpr={r5['benign_fpr']:.3%}  |  "
              f"18-feat esc={r18['escalation_rate']:.3%} fpr={r18['benign_fpr']:.3%}")
        for cls in classes:
            c5, c18 = r5["per_class"][cls], r18["per_class"][cls]
            if c5["n"]:
                print(f"    {cls}: n={c5['n']:,}  5-feat={c5['recall']:.1%}  18-feat={c18['recall']:.1%}")
                summary_rows.append((name, cls, c5["n"], c5["recall"], c18["recall"]))

    lines.append("## Summary: per-class recall, both configs, Monday-CSV-fitted\n")
    lines.append("| day | class | n | 5-feature recall | 18-feature recall |")
    lines.append("|---|---|---|---|---|")
    for name, cls, n, rec5, rec18 in summary_rows:
        lines.append(f"| {name} | {cls} | {n:,} | {rec5:.1%} | {rec18:.1%} |")
    lines.append("")

    # ---- unchanged PCAP-fitted, PCAP-tested section, for reference ----
    print()
    print("recomputing (unchanged) PCAP-fitted / PCAP-tested Monday+Friday numbers for reference ...")
    monday_pcap_fit, monday_pcap_holdout, friday_pcap_df, _ = load_pcap_data()
    fit5_pcap_cfg = fit_5_pcap(monday_pcap_fit, COUNT_5, RATIO_5, PERCENTILE)
    fit18_pcap_cfg = fit_18_pcap(monday_pcap_fit, PERCENTILE)

    lines.append("## Reference (unchanged): PCAP-fitted, PCAP-tested Monday + Friday\n")
    lines.append(
        "Never crossed the adapter boundary -- included for side-by-side reference only, "
        "recomputed from the same cached PCAP parquets as `results/final_selector_report.md` "
        "/ `results/seven_feature_selector_report.md`, not re-derived.\n"
    )
    for name, df, classes in (("Monday PCAP holdout", monday_pcap_holdout, []),
                               ("Friday PCAP", friday_pcap_df, FRIDAY_CLASSES)):
        esc5 = escalate_generic(df, fit5_pcap_cfg, COUNT_5, RATIO_5)
        esc18 = escalate_generic(df, fit18_pcap_cfg, COUNT_FEATURES, RATIO_FEATURES)
        r5 = day_report(df, classes, esc5)
        r18 = day_report(df, classes, esc18)
        lines.extend(render_day(name, "5-feature (PCAP-fitted)", r5))
        lines.extend(render_day(name, "18-feature (PCAP-fitted)", r18))
        print(f"{name}: 5-feat esc={r5['escalation_rate']:.3%} fpr={r5['benign_fpr']:.3%}  |  "
              f"18-feat esc={r18['escalation_rate']:.3%} fpr={r18['benign_fpr']:.3%}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
