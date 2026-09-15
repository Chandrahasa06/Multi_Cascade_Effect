"""Two checks before results/csv_refit_cross_day_report.md's "narrowness
is real" conclusion gets recorded as final.

1. Is DoS Hulk's 0.0% (n=231,073) a feature that fires-and-misses, or a
   feature that structurally can't fire at the CSV-fitted threshold?
   Reports each of the 5 features' crossing rate on Hulk flows
   specifically, and the fitted threshold value itself, Monday-CSV vs
   Monday-PCAP side by side -- if CSV-fitted thresholds are far higher,
   fitting saw inflated CSV-adapter benign counts and set the bar above
   anything an attack (or the CSV adapter itself) ever reaches.

2. Monday CSV holdout FPR landed at 0.080% when targeting p99.5 (~0.5%
   expected) -- 6x tighter than intended, worth knowing why before
   trusting the operating point. Sweeps p98/p99/p99.5 on the CSV path
   (same fit discipline: Monday CSV benign, chronological split) and
   reports benign FPR + per-class recall at each, to separate "recall
   stays zero at any plausible looseness" from "there's a real point
   where it turns on."

No API calls, no re-simulation -- reads already-cached feature parquets.

Run: python -m eval.csv_refit_diagnostics
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from dataplane.fitting import BENIGN_LABEL
from eval.csv_refit_cross_day import escalate_generic, fit_subset_generic
from eval.five_feature_cross_day import (
    THURSDAY_CLASSES,
    TUESDAY_CLASSES,
    WEDNESDAY_CLASSES,
    load_csv_day,
    normalize_thursday_labels,
)
from eval.reduced_selector_report import load_data as load_pcap_data
from eval.seven_feature_selector import COUNT_5, RATIO_5, fit_subset as fit_5_pcap
from eval.sweep import compute_crossings, split_monday_chronologically

OUT_MD = Path("results/csv_refit_diagnostics_report.md")
FEATURES_5 = list(COUNT_5) + list(RATIO_5)
SWEEP_PERCENTILES = (98.0, 99.0, 99.5)
TINY_CLASS_FLOOR = 50


def threshold_str(fit, feat: str) -> str:
    t = fit.config.thresholds.get(feat)
    if t is None:
        return "not fit / saturated"
    parts = []
    if t.high is not None:
        parts.append(f"high>{t.high:.6g}")
    if t.low is not None:
        parts.append(f"low<{t.low:.6g}")
    return ", ".join(parts) if parts else "(no bound)"


def main() -> int:
    print("=== Check 1: DoS Hulk per-feature crossing rate + CSV vs PCAP thresholds ===")
    monday_csv = load_csv_day("monday")
    monday_csv_fit, monday_csv_holdout = split_monday_chronologically(monday_csv)
    assert (monday_csv_fit["label"] == BENIGN_LABEL).all()

    monday_pcap_fit, monday_pcap_holdout, friday_pcap_df, _ = load_pcap_data()

    fit5_csv = fit_subset_generic(monday_csv_fit, COUNT_5, RATIO_5, 99.5)
    fit5_pcap = fit_5_pcap(monday_pcap_fit, COUNT_5, RATIO_5, 99.5)

    wednesday_df = load_csv_day("wednesday")
    hulk = wednesday_df[wednesday_df["label"] == "DoS Hulk"]
    print(f"DoS Hulk n={len(hulk):,}")

    hulk_crossings = compute_crossings(hulk, fit5_csv.config.thresholds)
    hulk_rates = hulk_crossings.mean(axis=0).to_dict() if hulk_crossings.shape[1] else {}

    lines: List[str] = []
    lines.append("# CSV refit diagnostics: threshold inflation + percentile sweep\n")
    lines.append("## Check 1: DoS Hulk crossing rate per feature, and CSV vs PCAP fitted thresholds\n")
    lines.append(f"DoS Hulk n={len(hulk):,} (Wednesday CSV).\n")
    lines.append("| feature | Hulk crossing rate | CSV-fitted threshold | PCAP-fitted threshold |")
    lines.append("|---|---|---|---|")
    for feat in FEATURES_5:
        rate = hulk_rates.get(feat, 0.0)
        csv_t = threshold_str(fit5_csv, feat)
        pcap_t = threshold_str(fit5_pcap, feat)
        lines.append(f"| `{feat}` | {rate:.4%} | {csv_t} | {pcap_t} |")
        print(f"  {feat}: Hulk crossing={rate:.4%}  CSV_threshold=[{csv_t}]  PCAP_threshold=[{pcap_t}]")

    # also report Hulk's own raw feature medians/p99 for direct comparison against the threshold
    lines.append("\nHulk's own observed feature distribution (median / p99 / max), for direct comparison against the thresholds above:\n")
    lines.append("| feature | Hulk median | Hulk p99 | Hulk max |")
    lines.append("|---|---|---|---|")
    for feat in FEATURES_5:
        vals = hulk[feat].dropna()
        if len(vals):
            lines.append(f"| `{feat}` | {vals.median():.4g} | {vals.quantile(0.99):.4g} | {vals.max():.4g} |")
            print(f"    Hulk {feat}: median={vals.median():.4g} p99={vals.quantile(0.99):.4g} max={vals.max():.4g}")
        else:
            lines.append(f"| `{feat}` | undefined for all Hulk flows | -- | -- |")
    lines.append("")

    print()
    print("=== Check 2: percentile sweep on CSV path, p98/p99/p99.5 ===")
    tuesday_df = load_csv_day("tuesday")
    thursday_df = normalize_thursday_labels(load_csv_day("thursday_morning_webattacks", "thursday_afternoon_infiltration"))

    days = [
        ("Tuesday", tuesday_df, TUESDAY_CLASSES),
        ("Wednesday", wednesday_df, WEDNESDAY_CLASSES),
        ("Thursday", thursday_df, THURSDAY_CLASSES),
    ]

    lines.append("## Check 2: percentile sweep, Monday-CSV-fitted 5-feature config\n")
    lines.append("Same fit discipline throughout: Monday CSV benign, chronological split, refit at each percentile.\n")

    for pct in SWEEP_PERCENTILES:
        fit_pct = fit_subset_generic(monday_csv_fit, COUNT_5, RATIO_5, pct)
        esc_holdout = escalate_generic(monday_csv_holdout, fit_pct, COUNT_5, RATIO_5)
        holdout_fpr = esc_holdout.mean()
        print(f"\np{pct}: Monday CSV holdout FPR = {holdout_fpr:.4%}  "
              f"(active thresholds: {len(fit_pct.config.thresholds)}/5)")

        lines.append(f"### p{pct}\n")
        lines.append(f"Monday CSV holdout FPR: **{holdout_fpr:.4%}**  "
                     f"(active thresholds: {len(fit_pct.config.thresholds)}/5)\n")
        lines.append("| day | class | n | recall | benign FPR (that day) |")
        lines.append("|---|---|---|---|---|")
        for name, df, classes in days:
            esc = escalate_generic(df, fit_pct, COUNT_5, RATIO_5)
            is_benign = df["label"] == BENIGN_LABEL
            day_fpr = (esc & is_benign).sum() / is_benign.sum() if is_benign.sum() else float("nan")
            print(f"  {name} benign FPR: {day_fpr:.4%}")
            for cls in classes:
                mask = df["label"] == cls
                n = int(mask.sum())
                if n == 0:
                    continue
                n_esc = int((esc & mask).sum())
                recall = n_esc / n
                unreliable = " (unreliable, n<50)" if n < TINY_CLASS_FLOOR else ""
                lines.append(f"| {name} | {cls} | {n:,} | {recall:.1%}{unreliable} | {day_fpr:.4%} |")
                print(f"    {cls}: n={n:,} recall={recall:.1%}{unreliable}")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
