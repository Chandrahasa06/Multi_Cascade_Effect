"""Part 3: does syn_without_synack_count + bwd_fwd_byte_ratio alone carry
DoS Hulk (and whatever else has a real, non-saturated signal on these two
features)? Both loosen correctly by p98 (results/csv_refit_diagnostics_report.md)
and neither has flows_per_src's saturation problem; flows_per_src/
distinct_dst_ports_per_src were ruled out for Hulk specifically as a
genuine feature-shape limit (Hulk's own values, 58 flows/src and 12.29
ports/src, sit below even PCAP-scale thresholds) -- not a fitting bug,
so dropping them here isn't circular with Part 1/2's fix.

COUNT = syn_without_synack_count, RATIO = bwd_fwd_byte_ratio. Same rule
(>=1 COUNT AND >=1 RATIO), same fitting discipline (Monday CSV benign,
chronological split, saturation-aware), swept across p98/p99/p99.5 for
direct comparability with results/saturation_fixed_cross_day_report.md.

No API calls, no re-simulation.

Run: python -m eval.two_feature_hulk_test
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from dataplane.fitting import BENIGN_LABEL
from eval.five_feature_cross_day import (
    THURSDAY_CLASSES,
    TUESDAY_CLASSES,
    WEDNESDAY_CLASSES,
    load_csv_day,
    normalize_thursday_labels,
)
from eval.saturation_fixed_cross_day import day_report, escalate, fit_subset, firing_rates, render_day, render_saturation_report
from eval.sweep import split_monday_chronologically

OUT_MD = Path("results/two_feature_hulk_report.md")
PERCENTILES = (98.0, 99.0, 99.5)
COUNT_2 = ("syn_without_synack_count",)
RATIO_2 = ("bwd_fwd_byte_ratio",)


def main() -> int:
    print("loading Monday CSV, splitting chronologically ...")
    monday_csv = load_csv_day("monday")
    monday_fit, monday_holdout = split_monday_chronologically(monday_csv)
    assert (monday_fit["label"] == BENIGN_LABEL).all()

    tuesday_df = load_csv_day("tuesday")
    wednesday_df = load_csv_day("wednesday")
    thursday_df = normalize_thursday_labels(load_csv_day("thursday_morning_webattacks", "thursday_afternoon_infiltration"))
    days = [
        ("Tuesday", tuesday_df, TUESDAY_CLASSES),
        ("Wednesday", wednesday_df, WEDNESDAY_CLASSES),
        ("Thursday", thursday_df, THURSDAY_CLASSES),
    ]

    lines: List[str] = []
    lines.append("# 2-feature variant: syn_without_synack_count (COUNT) + bwd_fwd_byte_ratio (RATIO)\n")
    lines.append(
        "Tests whether these two features alone -- both confirmed loosening correctly with "
        "percentile, neither saturated on `flows_per_src`'s pattern -- carry DoS Hulk (and "
        "anything else) where the 5/18-feature configs don't. Same rule, same fitting "
        "discipline (Monday CSV benign, chronological split), same percentile sweep as "
        "`results/saturation_fixed_cross_day_report.md` for direct comparison.\n"
    )

    summary_rows = []
    for pct in PERCENTILES:
        lines.append(f"## p{pct}\n")
        fit = fit_subset(monday_fit, COUNT_2, RATIO_2, pct)
        lines.append("### saturation report\n")
        lines.extend(render_saturation_report(fit, list(COUNT_2) + list(RATIO_2)))
        lines.append("")

        esc_holdout, _, _ = escalate(monday_holdout, fit, COUNT_2, RATIO_2)
        holdout_fpr = esc_holdout.mean()
        print(f"p{pct}: Monday CSV holdout FPR = {holdout_fpr:.4%}")
        lines.append(f"Monday CSV holdout FPR: **{holdout_fpr:.4%}**\n")

        for day_name, day_df, classes in days:
            esc, count_x, ratio_x = escalate(day_df, fit, COUNT_2, RATIO_2)
            r = day_report(day_df, classes, esc)
            lines.extend(render_day(day_name, f"2-feature, p{pct}", r))
            print(f"  {day_name}: esc={r['escalation_rate']:.3%} fpr={r['benign_fpr']:.3%}")
            for cls, c in r["per_class"].items():
                if c["n"]:
                    print(f"      {cls}: n={c['n']:,} recall={c['recall']:.1%}")
                    summary_rows.append((pct, day_name, cls, c["n"], c["recall"]))
            combined_x = pd.concat([count_x, ratio_x], axis=1)
            fri = firing_rates(day_df, combined_x)
            lines.append(f"Features firing ({day_name}, p{pct}):\n")
            lines.append("| feature | crossing rate |")
            lines.append("|---|---|")
            for feat, rate in sorted(fri.items(), key=lambda kv: -kv[1]):
                lines.append(f"| `{feat}` | {rate:.4%} |")
            lines.append("")

    lines.append("## Summary\n")
    lines.append("| pct | day | class | n | recall |")
    lines.append("|---|---|---|---|---|")
    for pct, day_name, cls, n, recall in summary_rows:
        lines.append(f"| p{pct} | {day_name} | {cls} | {n:,} | {recall:.1%} |")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
