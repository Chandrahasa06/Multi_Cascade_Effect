"""Does the 18-feature count-AND-ratio selector (results/final_selector_report.md)
need its long tail? 11 of its 18 features fire on under 2% of Friday
flows. Tests a 7-feature subset (the ones above ~10% Friday crossing
rate) and a 5-feature variant (dropping the two COUNT features that are
closest to that 10% line) against both the 18-feature config and the
original 20-feature k_of_n(2) baseline, same rule (>=1 COUNT crosses
AND >=1 RATIO crosses), same fit (Monday benign only, chronological
split, p99.5).

Also decomposes, for the 2-RATIO-feature configs (7 and 5), which of
`syn_ratio`/`bwd_fwd_byte_ratio` is actually satisfying the ratio side
on each population/class -- with only two RATIO features the AND rule
has no slack left if either one underperforms.

No API calls -- pure re-analysis of cached PCAP feature parquets.

Run: python -m eval.seven_feature_selector
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from dataplane.fitting import fit_thresholds_from_frame
from dataplane.selector import EscalationRule
from eval.final_selector_report import fit_final as fit_18
from eval.reduced_selector_report import (
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
OUT_MD = Path("results/seven_feature_selector_report.md")

COUNT_7 = ("flows_per_src", "distinct_dst_ports_per_src", "syn_without_synack_count",
           "bwd_pkt_len_mean", "pkt_len_range")
RATIO_7 = ("syn_ratio", "bwd_fwd_byte_ratio")

COUNT_5 = ("flows_per_src", "distinct_dst_ports_per_src", "syn_without_synack_count")
RATIO_5 = RATIO_7


def fit_subset(monday_fit: pd.DataFrame, count_feats: Sequence[str], ratio_feats: Sequence[str], percentile: float = PERCENTILE):
    features = list(count_feats) + list(ratio_feats)
    return fit_thresholds_from_frame(
        monday_fit, feature_names=features, percentile=percentile,
        rule=EscalationRule.K_OF_N, k=2, exclude_low_confidence=False,
    )


def escalate_subset(df: pd.DataFrame, fit, count_feats: Sequence[str], ratio_feats: Sequence[str]) -> tuple:
    count_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in count_feats}
    ratio_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in ratio_feats}
    count_crossings = compute_crossings(df, count_thresholds)
    ratio_crossings = compute_crossings(df, ratio_thresholds)
    count_any = count_crossings.any(axis=1) if count_crossings.shape[1] else pd.Series(False, index=df.index)
    ratio_any = ratio_crossings.any(axis=1) if ratio_crossings.shape[1] else pd.Series(False, index=df.index)
    escalated = count_any & ratio_any
    return escalated, count_crossings, ratio_crossings


def ratio_contribution(df: pd.DataFrame, ratio_crossings: pd.DataFrame, ratio_feats: Sequence[str], escalated: pd.Series) -> dict:
    """Among ESCALATED flows, which ratio feature(s) fired: only feat A,
    only feat B, or both -- answers 'which one is actually carrying the
    ratio side.'"""
    esc_ratio = ratio_crossings.loc[escalated]
    if len(ratio_feats) != 2 or esc_ratio.shape[1] < 2:
        return {}
    a, b = ratio_feats[0], ratio_feats[1]
    a_only = int((esc_ratio[a] & ~esc_ratio[b]).sum())
    b_only = int((esc_ratio[b] & ~esc_ratio[a]).sum())
    both = int((esc_ratio[a] & esc_ratio[b]).sum())
    total = int(escalated.sum())
    return {a: a_only, b: b_only, "both": both, "total_escalated": total}


def main() -> int:
    print("loading cached PCAP feature parquets ...")
    monday_fit, monday_holdout, friday_df, combined = load_data()
    print(f"Monday fit={len(monday_fit):,}  Monday holdout={len(monday_holdout):,}")
    print(f"Friday total={len(friday_df):,}")
    print()

    old_fit = fit_old(monday_fit, PERCENTILE)
    fit18 = fit_18(monday_fit, PERCENTILE)
    fit7 = fit_subset(monday_fit, COUNT_7, RATIO_7, PERCENTILE)
    fit5 = fit_subset(monday_fit, COUNT_5, RATIO_5, PERCENTILE)

    for name, feats, fit in (("7-feature", list(COUNT_7) + list(RATIO_7), fit7), ("5-feature", list(COUNT_5) + list(RATIO_5), fit5)):
        missing = [f for f in feats if f not in fit.config.thresholds]
        if missing:
            print(f"WARNING: {name} config -- these did not fit / saturated at p{PERCENTILE}: {missing}")
        else:
            print(f"{name} config: all {len(feats)} features active at p{PERCENTILE}")
    print()

    configs = [
        ("old (20-feature, k=2)", None, None, old_fit, "old"),
        ("18-feature (count AND ratio)", COUNT_FEATURES, RATIO_FEATURES, fit18, "subset"),
        ("7-feature (count AND ratio)", COUNT_7, RATIO_7, fit7, "subset"),
        ("5-feature (count AND ratio)", COUNT_5, RATIO_5, fit5, "subset"),
    ]

    pop_defs = [
        ("Monday holdout (benign-only -- every escalation here is a false alarm)", monday_holdout),
        ("Friday", friday_df),
        ("Combined (Monday holdout + Friday)", combined),
    ]

    lines: List[str] = []
    lines.append("# 7-feature (and 5-feature) selector report\n")
    lines.append(
        f"Tests whether the 18-feature count-AND-ratio config (`results/final_selector_report.md`) "
        f"needs its long tail -- 11 of its 18 features fire on under 2% of Friday flows. Same rule "
        f"(>=1 COUNT crosses AND >=1 RATIO crosses), same fit (Monday benign only, chronological "
        f"split, p{PERCENTILE}). No API calls -- pure re-analysis of cached PCAP feature parquets.\n\n"
        f"Monday fit-half n={len(monday_fit):,}, Monday holdout n={len(monday_holdout):,}, "
        f"Friday n={len(friday_df):,} ({friday_df['label'].value_counts().to_dict()}), "
        f"combined n={len(combined):,}.\n\n"
        f"**7-feature set**: COUNT = {', '.join('`%s`' % f for f in COUNT_7)}; "
        f"RATIO = {', '.join('`%s`' % f for f in RATIO_7)}.\n\n"
        f"**5-feature set**: COUNT = {', '.join('`%s`' % f for f in COUNT_5)} "
        f"(drops `bwd_pkt_len_mean`, `pkt_len_range`); RATIO unchanged.\n"
    )

    escalated_cache: Dict[str, Dict[str, pd.Series]] = {}
    crossings_cache: Dict[str, Dict[str, tuple]] = {}

    for pop_name, pop_df in pop_defs:
        breakdowns = {}
        escalated_cache[pop_name] = {}
        crossings_cache[pop_name] = {}
        for label, count_feats, ratio_feats, fit, kind in configs:
            if kind == "old":
                crossings = compute_crossings(pop_df, fit.config.thresholds)
                escalated = escalate_from_crossings(crossings, EscalationRule.K_OF_N, 2)
                crossings_cache[pop_name][label] = (None, None)
            else:
                escalated, count_x, ratio_x = escalate_subset(pop_df, fit, count_feats, ratio_feats)
                crossings_cache[pop_name][label] = (count_x, ratio_x)
            escalated_cache[pop_name][label] = escalated
            breakdowns[label] = escalation_breakdown(pop_df, escalated)

        if pop_name.startswith("Monday"):
            lines.append("Monday holdout false alarms:\n")
            for label, _, _, _, _ in configs:
                b = breakdowns[label]
                lines.append(f"- {label}: **{b['n_escalated']:,}** of {b['n_total']:,} ({b['n_escalated']/b['n_total']:.3%})")
            lines.append("")

        lines.extend(render_population_table(pop_name, breakdowns))

        if pop_name != "Monday holdout (benign-only -- every escalation here is a false alarm)":
            for label, _, _, _, _ in configs:
                b = breakdowns[label]
                m = metrics_from_breakdown(b)
                lines.append(f"Attacks NOT escalated ({pop_name}, {label}): " +
                             ", ".join(f"{l}={m['missed'][l]:,} of {b['by_label_total'][l]:,}" for l in ATTACK_LABELS))
            lines.append("")

    # ---- ratio-feature contribution for the 2-RATIO configs ----
    lines.append("## Which RATIO feature is carrying the ratio side (7- and 5-feature configs)\n")
    lines.append(
        "With only `syn_ratio` + `bwd_fwd_byte_ratio` as RATIO features, the AND rule has no "
        "slack if either underperforms. Decomposed among ESCALATED flows only: how many were "
        "escalated with `syn_ratio` firing alone, `bwd_fwd_byte_ratio` alone, or both.\n"
    )
    for label, count_feats, ratio_feats, fit, kind in configs:
        if kind != "subset" or len(ratio_feats) != 2:
            continue
        lines.append(f"### {label}\n")
        lines.append("| population / class | syn_ratio only | bwd_fwd_byte_ratio only | both | total escalated |")
        lines.append("|---|---|---|---|---|")
        for pop_name, pop_df in pop_defs:
            _, ratio_x = crossings_cache[pop_name][label]
            escalated = escalated_cache[pop_name][label]
            contrib = ratio_contribution(pop_df, ratio_x, ratio_feats, escalated)
            if contrib:
                lines.append(f"| {pop_name} (all) | {contrib['syn_ratio']:,} | {contrib['bwd_fwd_byte_ratio']:,} | "
                             f"{contrib['both']:,} | {contrib['total_escalated']:,} |")
            if pop_name == "Friday":
                for cls in ATTACK_LABELS:
                    mask = pop_df["label"] == cls
                    cls_escalated = escalated & mask
                    contrib_cls = ratio_contribution(pop_df.loc[mask], ratio_x.loc[mask], ratio_feats, escalated.loc[mask])
                    if contrib_cls:
                        lines.append(f"| Friday {cls} | {contrib_cls['syn_ratio']:,} | {contrib_cls['bwd_fwd_byte_ratio']:,} | "
                                     f"{contrib_cls['both']:,} | {contrib_cls['total_escalated']:,} |")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
