"""Two checks before trusting the cross-day overfit conclusion in
results/five_feature_cross_day_report.md:

1. Does the 18-feature config ALSO fail on Tuesday-Thursday? The 5/7-
   feature reduction was never itself cross-day tested at 18 features
   either -- so a Tue-Thu failure can't yet be attributed to reduction
   specifically versus the count-AND-ratio-at-p99.5 approach in general.
   If 18 also fails: reduction is innocent, the finding becomes
   "count-AND-ratio at p99.5 is PortScan/DDoS-specific," not a reduction
   artifact. If 18 succeeds where 5 doesn't: reduction caused it, revert.

2. Is Tue-Thu's failure actually adapter mismatch, not generalization?
   syn_ratio and syn_without_synack_count both need per-packet SYN/flag
   observation, which the CSV adapter RECONSTRUCTS from CICFlowMeter's
   aggregated flag counts rather than observing directly from real
   packets (see adapters/csv_flow_adapter.py's module docstring on lossy
   packet synthesis). Several classes reading exactly 0.0% recall is
   consistent with "this feature structurally cannot fire on CSV-adapter
   data" as much as with "it fires and misses." Checked directly: Monday
   CSV vs Monday PCAP per-feature crossing rate (same fitted thresholds,
   same day, only the adapter differs), and each feature's BENIGN-only
   crossing rate on every CSV day.

No API calls, no re-simulation -- reads already-cached feature parquets.

Run: python -m eval.cross_day_followups
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from eval.final_selector_report import fit_final as fit_18
from eval.five_feature_cross_day import (
    FRIDAY_CLASSES,
    THURSDAY_CLASSES,
    TUESDAY_CLASSES,
    WEDNESDAY_CLASSES,
    load_csv_day,
    normalize_thursday_labels,
)
from eval.reduced_selector_report import (
    COUNT_FEATURES,
    RATIO_FEATURES,
    load_data as load_pcap_data,
)
from eval.run_pcap_sweep import load_friday_pcap
from eval.seven_feature_selector import COUNT_5, RATIO_5, escalate_subset, fit_subset
from eval.sweep import compute_crossings

OUT_MD = Path("results/cross_day_followups_report.md")
PERCENTILE = 99.5
TINY_CLASS_FLOOR = 50


def escalate_18(df: pd.DataFrame, fit) -> pd.Series:
    count_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in COUNT_FEATURES}
    ratio_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in RATIO_FEATURES}
    count_x = compute_crossings(df, count_thresholds)
    ratio_x = compute_crossings(df, ratio_thresholds)
    count_any = count_x.any(axis=1) if count_x.shape[1] else pd.Series(False, index=df.index)
    ratio_any = ratio_x.any(axis=1) if ratio_x.shape[1] else pd.Series(False, index=df.index)
    return count_any & ratio_any


def day_recall_table(df: pd.DataFrame, classes: List[str], escalated: pd.Series) -> Dict[str, dict]:
    out = {}
    for cls in classes:
        mask = df["label"] == cls
        n = int(mask.sum())
        n_esc = int((escalated & mask).sum())
        out[cls] = {"n": n, "escalated": n_esc, "recall": (n_esc / n) if n else None,
                    "unreliable": 0 < n < TINY_CLASS_FLOOR}
    return out


def benign_fpr(df: pd.DataFrame, escalated: pd.Series) -> float:
    is_benign = df["label"] == "BENIGN"
    n_benign = int(is_benign.sum())
    return (int((escalated & is_benign).sum()) / n_benign) if n_benign else float("nan")


def per_feature_benign_crossing_rates(df: pd.DataFrame, fit, features: List[str]) -> Dict[str, float]:
    is_benign = df["label"] == "BENIGN"
    thresholds = {k: v for k, v in fit.config.thresholds.items() if k in features}
    x = compute_crossings(df.loc[is_benign], thresholds)
    if x.shape[1] == 0:
        return {f: float("nan") for f in features}
    rates = x.mean(axis=0).to_dict()
    return {f: rates.get(f, float("nan")) for f in features}


def main() -> int:
    print("fitting 18-feature and 5-feature configs on Monday PCAP benign (unchanged) ...")
    monday_pcap_fit, monday_pcap_holdout, friday_pcap_df, _ = load_pcap_data()
    fit18 = fit_18(monday_pcap_fit, PERCENTILE)
    fit5 = fit_subset(monday_pcap_fit, COUNT_5, RATIO_5, PERCENTILE)
    print(f"18-feature: {len(fit18.config.thresholds)} active. 5-feature: {len(fit5.config.thresholds)} active.")
    print()

    print("loading Monday CSV (adapter1) + Tue/Wed/Thu (CSV) + Friday (PCAP, cached) ...")
    monday_csv_df = load_csv_day("monday")
    tuesday_df = load_csv_day("tuesday")
    wednesday_df = load_csv_day("wednesday")
    thursday_df = normalize_thursday_labels(load_csv_day("thursday_morning_webattacks", "thursday_afternoon_infiltration"))

    days = [
        ("Tuesday", tuesday_df, TUESDAY_CLASSES),
        ("Wednesday", wednesday_df, WEDNESDAY_CLASSES),
        ("Thursday", thursday_df, THURSDAY_CLASSES),
        ("Friday", friday_pcap_df, FRIDAY_CLASSES),
    ]

    lines: List[str] = []
    lines.append("# Cross-day follow-ups: is it reduction, or is it adapter mismatch?\n")
    lines.append(
        "Two checks against `results/five_feature_cross_day_report.md`'s overfit "
        "conclusion, both zero-API-call re-analysis of already-cached feature "
        "parquets. (1) whether the unreduced 18-feature config also fails on "
        "Tuesday-Thursday (isolating reduction from the count-AND-ratio approach "
        "itself), and (2) whether the 5 features can even fire on CSV-adapter "
        "benign traffic in the first place, checked on Monday (same day, only "
        "the adapter differs) and on every CSV day's own BENIGN population.\n"
    )

    # ---- Check 1: 18-feature vs 5-feature cross-day ----
    lines.append("## Check 1: does the 18-feature config also fail on Tue-Thu?\n")
    lines.append("| day | class | n | 5-feature recall | 18-feature recall |")
    lines.append("|---|---|---|---|---|")
    for name, df, classes in days:
        esc5, _, _ = escalate_subset(df, fit5, COUNT_5, RATIO_5)
        esc18 = escalate_18(df, fit18)
        r5 = day_recall_table(df, classes, esc5)
        r18 = day_recall_table(df, classes, esc18)
        for cls in classes:
            n = r5[cls]["n"]
            if n == 0:
                continue
            rec5 = f"{r5[cls]['recall']:.1%}" + (" (unreliable)" if r5[cls]["unreliable"] else "")
            rec18 = f"{r18[cls]['recall']:.1%}" + (" (unreliable)" if r18[cls]["unreliable"] else "")
            lines.append(f"| {name} | {cls} | {n:,} | {rec5} | {rec18} |")
        fpr5 = benign_fpr(df, esc5)
        fpr18 = benign_fpr(df, esc18)
        lines.append(f"| {name} | *(BENIGN false-alarm rate)* | -- | {fpr5:.3%} | {fpr18:.3%} |")
        print(f"{name}: 5-feat benign_fpr={fpr5:.3%}  18-feat benign_fpr={fpr18:.3%}")
        for cls in classes:
            if r5[cls]["n"]:
                print(f"    {cls}: n={r5[cls]['n']:,}  5-feat={r5[cls]['recall']:.1%}  18-feat={r18[cls]['recall']:.1%}")
    lines.append("")

    # ---- Check 2: Monday CSV vs Monday PCAP, same thresholds ----
    lines.append("## Check 2: Monday CSV vs Monday PCAP -- same fitted thresholds, same day\n")
    lines.append(
        "Both populations are 100% BENIGN (Monday, either adapter). Same fitted "
        "thresholds (from Monday PCAP fit-half) applied to both. A feature that "
        "crosses ~0% on CSV but a comparable, non-zero rate on PCAP -- on the SAME "
        "day -- can't be a generalization miss; it's the CSV adapter not producing "
        "the signal the feature needs at all.\n"
    )
    features5 = list(COUNT_5) + list(RATIO_5)
    mon_csv_rates = per_feature_benign_crossing_rates(monday_csv_df, fit5, features5)
    mon_pcap_rates = per_feature_benign_crossing_rates(monday_pcap_holdout, fit5, features5)
    lines.append("| feature | Monday CSV benign crossing rate | Monday PCAP holdout benign crossing rate |")
    lines.append("|---|---|---|")
    for f in features5:
        lines.append(f"| `{f}` | {mon_csv_rates[f]:.4%} | {mon_pcap_rates[f]:.4%} |")
        print(f"Monday {f}: CSV={mon_csv_rates[f]:.4%}  PCAP={mon_pcap_rates[f]:.4%}")
    lines.append("")

    # ---- Check 2b: per CSV day, BENIGN-only crossing rate per feature ----
    lines.append("## Check 2b: per-feature crossing rate on BENIGN flows, each CSV day\n")
    lines.append("| feature | Tuesday BENIGN | Wednesday BENIGN | Thursday BENIGN |")
    lines.append("|---|---|---|---|")
    tue_rates = per_feature_benign_crossing_rates(tuesday_df, fit5, features5)
    wed_rates = per_feature_benign_crossing_rates(wednesday_df, fit5, features5)
    thu_rates = per_feature_benign_crossing_rates(thursday_df, fit5, features5)
    for f in features5:
        lines.append(f"| `{f}` | {tue_rates[f]:.4%} | {wed_rates[f]:.4%} | {thu_rates[f]:.4%} |")
        print(f"{f}: Tue={tue_rates[f]:.4%}  Wed={wed_rates[f]:.4%}  Thu={thu_rates[f]:.4%}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
