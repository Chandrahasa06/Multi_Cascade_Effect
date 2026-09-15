"""Final selector config: the working subset of eval/reduced_selector_report.py's
three proposed changes, with the window-rarity OR-path dropped entirely
(it produced 55,208 of 56,822 Monday false alarms there -- unusable as
specified, needs value-bucketing before it's revisited, not attempted
here).

FINAL CONFIG:
  - drop distinct_dst_ips_per_src, flow_bytes_per_sec (dead, validated)
  - drop init_win_bytes_fwd/bwd and the window rarity path entirely
  - remaining features split into COUNT / RATIO groups (same split as
    eval/reduced_selector_report.py, reused directly -- not redefined)
  - rule: escalate iff (>=1 COUNT crosses) AND (>=1 RATIO crosses)
  - single operating point: p99.5 -- syn_ratio (the one RATIO feature
    that tracks scan/flood behaviour) saturates above this point, per
    the diagnosis in results/reduced_selector_report.md

Monday-only fit, chronologically split (first half fits, second half
measures); zero-day property asserted. No API calls -- pure re-analysis
of cached PCAP feature parquets.

Run: python -m eval.final_selector_report
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from dataplane.fitting import fit_thresholds_from_frame
from dataplane.selector import EscalationRule
from eval.reduced_selector_report import (
    ALL_LABELS,
    ATTACK_LABELS,
    COUNT_FEATURES,
    RATIO_FEATURES,
    escalation_breakdown,
    fit_old,
    firing_rates,
    load_data,
    metrics_from_breakdown,
    render_population_table,
)
from eval.sweep import compute_crossings, escalate_from_crossings

PERCENTILE = 99.5
OUT_MD = Path("results/final_selector_report.md")


def fit_final(monday_fit: pd.DataFrame, percentile: float = PERCENTILE):
    features = list(COUNT_FEATURES) + list(RATIO_FEATURES)
    return fit_thresholds_from_frame(
        monday_fit, feature_names=features, percentile=percentile,
        rule=EscalationRule.K_OF_N, k=2, exclude_low_confidence=False,
    )


def escalate_final(df: pd.DataFrame, fit) -> tuple:
    count_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in COUNT_FEATURES}
    ratio_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in RATIO_FEATURES}
    count_crossings = compute_crossings(df, count_thresholds)
    ratio_crossings = compute_crossings(df, ratio_thresholds)
    count_any = count_crossings.any(axis=1) if count_crossings.shape[1] else pd.Series(False, index=df.index)
    ratio_any = ratio_crossings.any(axis=1) if ratio_crossings.shape[1] else pd.Series(False, index=df.index)
    escalated = count_any & ratio_any
    return escalated, count_crossings, ratio_crossings


def main() -> int:
    print("loading cached PCAP feature parquets ...")
    monday_fit, monday_holdout, friday_df, combined = load_data()
    print(f"Monday fit={len(monday_fit):,}  Monday holdout={len(monday_holdout):,}")
    print(f"Friday total={len(friday_df):,}  by label: {friday_df['label'].value_counts().to_dict()}")
    print(f"combined={len(combined):,}")
    print()

    old_fit = fit_old(monday_fit, PERCENTILE)
    final_fit = fit_final(monday_fit, PERCENTILE)

    syn_ratio_active = "syn_ratio" in final_fit.config.thresholds
    print(f"syn_ratio active (not saturated) at p{PERCENTILE}: {syn_ratio_active}")
    assert syn_ratio_active, (
        "syn_ratio is not active at the chosen operating point -- this config's whole "
        "rationale (the one ratio feature that tracks scan/flood behaviour) doesn't hold"
    )

    lines: List[str] = []
    lines.append("# Final selector report\n")
    lines.append(
        f"Ships the validated subset of `results/reduced_selector_report.md`'s three "
        f"proposed changes at a single operating point (**p{PERCENTILE}**), with the "
        f"window-rarity OR-path dropped entirely -- it produced 55,208 of 56,822 Monday "
        f"false alarms there and is not usable as specified (needs value-bucketing, not "
        f"attempted here). Monday-only fit, chronologically split (first half fits, second "
        f"half held out); zero-day property asserted (fit half is 100% BENIGN, checked by "
        f"`fit_thresholds_from_frame` itself). No API calls -- pure re-analysis of cached "
        f"PCAP feature parquets.\n\n"
        f"Monday fit-half n={len(monday_fit):,}, Monday holdout n={len(monday_holdout):,}, "
        f"Friday n={len(friday_df):,} ({friday_df['label'].value_counts().to_dict()}), "
        f"combined n={len(combined):,}.\n"
    )
    lines.append(
        f"**`syn_ratio` confirmed active (not saturated) at p{PERCENTILE}** -- this rule's "
        f"whole rationale depends on it; it saturates and drops out above this operating "
        f"point (see `results/reduced_selector_report.md`'s diagnosis section), which is "
        f"why p{PERCENTILE} is the only operating point used here.\n"
    )

    lines.append("## Final feature set\n")
    lines.append("| feature | group | Friday crossing rate | Monday holdout crossing rate |")
    lines.append("|---|---|---|---|")
    final_count_x_fri = compute_crossings(friday_df, {k: v for k, v in final_fit.config.thresholds.items() if k in COUNT_FEATURES})
    final_ratio_x_fri = compute_crossings(friday_df, {k: v for k, v in final_fit.config.thresholds.items() if k in RATIO_FEATURES})
    final_count_x_mon = compute_crossings(monday_holdout, {k: v for k, v in final_fit.config.thresholds.items() if k in COUNT_FEATURES})
    final_ratio_x_mon = compute_crossings(monday_holdout, {k: v for k, v in final_fit.config.thresholds.items() if k in RATIO_FEATURES})
    fri_rates = {**firing_rates(friday_df, final_count_x_fri), **firing_rates(friday_df, final_ratio_x_fri)}
    mon_rates = {**firing_rates(monday_holdout, final_count_x_mon), **firing_rates(monday_holdout, final_ratio_x_mon)}
    all_active = sorted(final_fit.config.thresholds.keys(), key=lambda f: -fri_rates.get(f, 0))
    dropped_missing = [f for f in (list(COUNT_FEATURES) + list(RATIO_FEATURES)) if f not in final_fit.config.thresholds]
    for feat in all_active:
        group = "COUNT" if feat in COUNT_FEATURES else "RATIO"
        lines.append(f"| `{feat}` | {group} | {fri_rates.get(feat, 0):.4%} | {mon_rates.get(feat, 0):.4%} |")
    lines.append("")
    lines.append(f"Dropped outright: `distinct_dst_ips_per_src`, `flow_bytes_per_sec`, "
                 f"`init_win_bytes_fwd`, `init_win_bytes_bwd` (window rarity path removed).")
    if dropped_missing:
        lines.append(f"\nFit but saturated/unusable at p{PERCENTILE} (excluded from the active set, "
                     f"not manually dropped): {', '.join(f'`{f}`' for f in dropped_missing)}.")
    lines.append("")

    pop_defs = [
        ("Monday holdout (benign-only -- every escalation here is a false alarm)", monday_holdout),
        ("Friday", friday_df),
        ("Combined (Monday holdout + Friday)", combined),
    ]

    lines.append(f"## Escalation report at p{PERCENTILE}\n")
    for pop_name, pop_df in pop_defs:
        old_crossings = compute_crossings(pop_df, old_fit.config.thresholds)
        old_escalated = escalate_from_crossings(old_crossings, EscalationRule.K_OF_N, 2)
        final_escalated, _, _ = escalate_final(pop_df, final_fit)

        old_b = escalation_breakdown(pop_df, old_escalated)
        final_b = escalation_breakdown(pop_df, final_escalated)

        if pop_name.startswith("Monday"):
            lines.append(f"Monday holdout false alarms -- old (20-feature, k=2): "
                         f"**{old_b['n_escalated']:,}** of {old_b['n_total']:,} "
                         f"({old_b['n_escalated']/old_b['n_total']:.3%}); "
                         f"final (count AND ratio): **{final_b['n_escalated']:,}** "
                         f"({final_b['n_escalated']/final_b['n_total']:.3%}).\n")

        lines.extend(render_population_table(pop_name, {
            "old (20-feature, k=2)": old_b,
            "final (count AND ratio)": final_b,
        }))

        m_final = metrics_from_breakdown(final_b)
        m_old = metrics_from_breakdown(old_b)
        if pop_name != "Monday holdout (benign-only -- every escalation here is a false alarm)":
            lines.append(f"Attacks NOT escalated ({pop_name}, final config), by class: " +
                         ", ".join(f"{l}={m_final['missed'][l]:,} of {final_b['by_label_total'][l]:,}" for l in ATTACK_LABELS) + "\n")
            lines.append(f"Attacks NOT escalated ({pop_name}, old config), by class: " +
                         ", ".join(f"{l}={m_old['missed'][l]:,} of {old_b['by_label_total'][l]:,}" for l in ATTACK_LABELS) + "\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
