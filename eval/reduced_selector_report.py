"""Three changes to the selector, evaluated against the old (current)
config on the same held-out data, at three operating points. Monday-only
fit, chronologically split (first half fits, second half measures);
zero-day property asserted, not assumed. No API calls -- pure re-analysis
of cached PCAP feature parquets.

=== Change 1: drop dead features ===
`distinct_dst_ips_per_src` and `flow_bytes_per_sec` are dropped outright
(as currently fitted, confirmed below to rarely-to-never cross).

=== Change 2: window-size features, refit by rarity ===
`init_win_bytes_fwd`/`init_win_bytes_bwd` are re-added but NOT fit by
ordinary percentile: their Monday-benign distribution (reported below)
shows heavy point-mass at the TCP window field's own hard bounds --
16-bit-field 0 and 65535 -- each carrying multiple percent of benign
mass, orders of magnitude more than any percentile budget in this
project's grid. This is the discrete-distribution twin of the
bounded-ratio bug fixed earlier for syn_ratio/no_response_flag: the
percentile computation lands exactly on 0 or 65535, and nothing can
ever cross a `>`/`<` comparison against a feature's own hard bound.
Instead, fit a rarity set: the values that cover at least
(100-percentile)% of Monday-fit-half benign mass become "known"; escalate
if a held-out flow's exact window value (fwd OR bwd) isn't in that set.
This ties the rarity floor to the SAME percentile knob driving every
other feature, at no extra free parameter. Window escalation is treated
as an independent OR-path into the final rule (see change 3's rationale
for why it isn't gated behind the count+ratio AND) -- documented, not
assumed.

=== Change 3: count-AND-ratio escalation rule ===
The remaining 20 features split into COUNT (magnitude/volume: flow
sizes, durations, distinct-port/flow counts -- FeatureKind.CONTINUOUS by
construction) and RATIO (proportions: FeatureKind.BOUNDED_RATIO/BOOLEAN
from dataplane/fitting.py's own classification, plus the two unbounded-
but-still-literally-a-ratio features down_up_pkt_ratio/
bwd_fwd_byte_ratio). New rule: escalate if (>=1 COUNT feature crosses
AND >=1 RATIO feature crosses) OR a window-size rarity flag fires.
Window size is kept OUTSIDE the count/ratio AND on purpose: it's a
third, structurally different signal (a discrete protocol fingerprint,
not a magnitude or a proportion), and gating a rare/unseen window value
behind also needing a ratio anomaly would suppress exactly the case
it's meant to catch on its own.

Old baseline (for direct comparison): the CURRENT, unmodified selector
config -- all 24 available features, ordinary per-feature-kind fitting,
k_of_n(k=2) -- at the same three operating points.

Run: python -m eval.reduced_selector_report
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from dataplane.fitting import BENIGN_LABEL, NON_FEATURE_COLUMNS, fit_thresholds_from_frame
from dataplane.selector import EscalationRule
from eval.run_pcap_sweep import load_friday_pcap, load_monday_pcap
from eval.sweep import compute_crossings, escalate_from_crossings, split_monday_chronologically

OPERATING_PERCENTILES = (99.5, 99.9, 99.99)
ATTACK_LABELS = ("PortScan", "DDoS", "Bot")
ALL_LABELS = ("BENIGN", "PortScan", "DDoS", "Bot")

DEAD_FEATURES = ("distinct_dst_ips_per_src", "flow_bytes_per_sec")
WINDOW_FEATURES = ("init_win_bytes_fwd", "init_win_bytes_bwd")

RATIO_FEATURES = (
    "syn_ratio", "rst_ratio", "no_response_flag", "flow_iat_regularity",
    "port_diversity_ratio", "unanswered_syn_ratio", "dst_concentration",
    "down_up_pkt_ratio", "bwd_fwd_byte_ratio",
)
COUNT_FEATURES = (
    "flow_duration", "flow_pkts_per_sec", "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "pkt_len_range", "flow_iat_mean", "flow_iat_max", "flow_iat_min",
    "flows_per_src", "distinct_dst_ports_per_src", "syn_without_synack_count",
)

OUT_MD = Path("results/reduced_selector_report.md")


# ------------------------------------------------------------------ data ---

def load_data():
    monday_df, _ = load_monday_pcap()
    friday_df, _ = load_friday_pcap()
    monday_fit, monday_holdout = split_monday_chronologically(monday_df)
    assert (monday_fit["label"] == BENIGN_LABEL).all(), "fit half must be benign-only -- zero-day property"
    combined = pd.concat([monday_holdout, friday_df], ignore_index=True)
    return monday_fit, monday_holdout, friday_df, combined


def window_distribution_report(monday_fit: pd.DataFrame, friday_df: pd.DataFrame) -> str:
    friday_benign = friday_df[friday_df["label"] == BENIGN_LABEL]
    lines = ["## Window-size feature diagnosis (before deciding how to fit)\n"]
    lines.append(
        "Percentile fitting on `init_win_bytes_fwd`/`init_win_bytes_bwd` was structurally "
        "dead for the same reason `syn_ratio`/`no_response_flag` were: the feature has hard "
        "bounds (a 16-bit TCP window field, 0 to 65535) and a huge share of benign mass sits "
        "exactly AT those bounds -- so a percentile-fit `>`/`<` comparison lands on the bound "
        "itself and can never be crossed, by protocol construction, not just by chance.\n"
    )
    lines.append("| feature | day | distinct values | defined | % at value 0 | % at value 65535 | top value | top value share |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for feat in WINDOW_FEATURES:
        for name, df in (("Monday benign (fit half)", monday_fit), ("Friday benign", friday_benign)):
            vals = df[feat].dropna()
            n = len(df)
            vc = vals.value_counts()
            top_val, top_cnt = (vc.index[0], vc.iloc[0]) if len(vc) else (None, 0)
            frac0 = (vals == 0).sum() / len(vals) if len(vals) else float("nan")
            frac_max = (vals == 65535).sum() / len(vals) if len(vals) else float("nan")
            lines.append(
                f"| `{feat}` | {name} | {vals.nunique():,} | {len(vals)/n:.1%} "
                f"| {frac0:.2%} | {frac_max:.2%} | {top_val:.0f} | {top_cnt/len(vals):.2%} |"
            )
    lines.append("")
    lines.append(
        "Both bounds carry several percent of benign mass on both days, far above any "
        "percentile budget in this project's grid (down to 0.001% at p99.999) -- saturated "
        "at every operating point tested, on both features, both days. Distinct-value counts "
        "and bound shares are similar Monday-to-Friday (order of magnitude, not the 5-20x "
        "blowup seen in the volume features) -- consistent with window size reflecting a "
        "fixed population of OS/TCP-stack implementations rather than traffic volume, so it "
        "should transfer across days better than the heavy-tailed count features do.\n"
    )
    return "\n".join(lines)


# ------------------------------------------------------------- fitting -----

def fit_old(monday_fit: pd.DataFrame, percentile: float):
    return fit_thresholds_from_frame(
        monday_fit, percentile=percentile, rule=EscalationRule.K_OF_N, k=2, exclude_low_confidence=False,
    )


def fit_rarity(values: pd.Series, floor_fraction: float) -> frozenset:
    vals = values.dropna()
    if len(vals) == 0:
        return frozenset()
    vc = vals.value_counts(normalize=True)
    return frozenset(vc[vc >= floor_fraction].index)


def rarity_crosses(values: pd.Series, common: frozenset) -> pd.Series:
    defined = values.notna()
    return defined & (~values.isin(common))


def fit_new(monday_fit: pd.DataFrame, percentile: float):
    count_ratio_features = list(COUNT_FEATURES) + list(RATIO_FEATURES)
    fit = fit_thresholds_from_frame(
        monday_fit, feature_names=count_ratio_features, percentile=percentile,
        rule=EscalationRule.K_OF_N, k=2, exclude_low_confidence=False,
    )
    floor = (100.0 - percentile) / 100.0
    window_common = {feat: fit_rarity(monday_fit[feat], floor) for feat in WINDOW_FEATURES}
    return fit, window_common, floor


def escalate_new(df: pd.DataFrame, fit, window_common: dict) -> tuple:
    """Returns (escalated, count_crossings_df, ratio_crossings_df, window_crossings_df)."""
    count_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in COUNT_FEATURES}
    ratio_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in RATIO_FEATURES}
    count_crossings = compute_crossings(df, count_thresholds)
    ratio_crossings = compute_crossings(df, ratio_thresholds)

    window_crossings = pd.DataFrame(
        {feat: rarity_crosses(df[feat], window_common[feat]) for feat in WINDOW_FEATURES}, index=df.index
    )

    count_any = count_crossings.any(axis=1) if count_crossings.shape[1] else pd.Series(False, index=df.index)
    ratio_any = ratio_crossings.any(axis=1) if ratio_crossings.shape[1] else pd.Series(False, index=df.index)
    window_any = window_crossings.any(axis=1)

    escalated = (count_any & ratio_any) | window_any
    return escalated, count_crossings, ratio_crossings, window_crossings


# ------------------------------------------------------------ reporting ----

def escalation_breakdown(df: pd.DataFrame, escalated: pd.Series, label_col: str = "label") -> dict:
    total = df[label_col].value_counts().to_dict()
    esc = df.loc[escalated, label_col].value_counts().to_dict()
    return {
        "n_total": len(df),
        "n_escalated": int(escalated.sum()),
        "by_label_total": {l: int(total.get(l, 0)) for l in ALL_LABELS},
        "by_label_escalated": {l: int(esc.get(l, 0)) for l in ALL_LABELS},
    }


def metrics_from_breakdown(b: dict) -> dict:
    attacks_escalated = sum(b["by_label_escalated"].get(l, 0) for l in ATTACK_LABELS)
    attacks_total = sum(b["by_label_total"].get(l, 0) for l in ATTACK_LABELS)
    precision = (attacks_escalated / b["n_escalated"]) if b["n_escalated"] else None
    escalation_rate = (b["n_escalated"] / b["n_total"]) if b["n_total"] else None
    recall = {
        l: (b["by_label_escalated"].get(l, 0) / b["by_label_total"][l]) if b["by_label_total"].get(l) else None
        for l in ATTACK_LABELS
    }
    missed = {l: b["by_label_total"].get(l, 0) - b["by_label_escalated"].get(l, 0) for l in ATTACK_LABELS}
    return {"escalation_rate": escalation_rate, "precision": precision, "recall": recall, "missed": missed}


def render_population_table(name: str, breakdowns: Dict[str, dict]) -> List[str]:
    """breakdowns: {config_label: breakdown_dict} for one population at one percentile."""
    lines = [f"**{name}**\n"]
    lines.append("| config | total flows | escalated | BENIGN esc. | PortScan esc. | DDoS esc. | Bot esc. | "
                  "escalation rate | precision | PortScan recall | DDoS recall | Bot recall |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg_label, b in breakdowns.items():
        m = metrics_from_breakdown(b)
        def r(l):
            v = m["recall"].get(l)
            return f"{v:.1%}" if v is not None else "n/a"
        def p():
            return f"{m['precision']:.1%}" if m["precision"] is not None else "n/a"
        lines.append(
            f"| {cfg_label} | {b['n_total']:,} | {b['n_escalated']:,} | "
            f"{b['by_label_escalated'].get('BENIGN', 0):,} | {b['by_label_escalated'].get('PortScan', 0):,} | "
            f"{b['by_label_escalated'].get('DDoS', 0):,} | {b['by_label_escalated'].get('Bot', 0):,} | "
            f"{m['escalation_rate']:.3%} | {p()} | {r('PortScan')} | {r('DDoS')} | {r('Bot')} |"
        )
    lines.append("")
    return lines


def firing_rates(df: pd.DataFrame, crossings: pd.DataFrame) -> Dict[str, float]:
    if len(df) == 0 or crossings.shape[1] == 0:
        return {}
    return {k: float(v) for k, v in crossings.mean(axis=0).items()}


def main() -> int:
    print("loading cached PCAP feature parquets ...")
    monday_fit, monday_holdout, friday_df, combined = load_data()
    print(f"Monday fit={len(monday_fit):,}  Monday holdout={len(monday_holdout):,}")
    print(f"Friday total={len(friday_df):,}  by label: {friday_df['label'].value_counts().to_dict()}")
    print(f"combined (Monday holdout + Friday)={len(combined):,}")
    print()

    report_lines: List[str] = []
    report_lines.append("# Reduced selector report\n")
    report_lines.append(
        "Three changes evaluated against the current (\"old\") selector config, at three "
        "operating points (p99.5 / p99.9 / p99.99), on Monday-only fitting (chronologically "
        "split: first half fits, second half held out) and measured against Monday holdout, "
        "Friday (real PortScan/DDoS/Bot labels), and both combined. Zero API calls -- pure "
        "re-analysis of already-cached PCAP feature parquets. "
        f"Monday fit-half n={len(monday_fit):,}, Monday holdout n={len(monday_holdout):,}, "
        f"Friday n={len(friday_df):,} ({friday_df['label'].value_counts().to_dict()}), "
        f"combined n={len(combined):,}.\n"
    )
    report_lines.append(
        "**Old config**: all 24 available features (current per-feature-kind fitting), "
        "`k_of_n(k=2)`.\n\n"
        "**New config**: drops `distinct_dst_ips_per_src` + `flow_bytes_per_sec`; "
        "`init_win_bytes_fwd`/`init_win_bytes_bwd` refit by rarity (escalate on a window value "
        "covering less than (100-percentile)% of Monday-fit-half benign mass) as an independent "
        "OR-path; the remaining 20 features split into "
        f"{len(COUNT_FEATURES)} COUNT ({', '.join('`%s`' % f for f in COUNT_FEATURES)}) and "
        f"{len(RATIO_FEATURES)} RATIO ({', '.join('`%s`' % f for f in RATIO_FEATURES)}) groups; "
        "escalate if (>=1 COUNT crosses AND >=1 RATIO crosses) OR a window rarity flag fires.\n"
    )

    monday_fit_local, monday_holdout_local, friday_local, combined_local = monday_fit, monday_holdout, friday_df, combined
    report_lines.append(window_distribution_report(monday_fit_local, friday_local))

    all_firing: Dict[float, dict] = {}

    for pct in OPERATING_PERCENTILES:
        report_lines.append(f"## Operating point: p{pct}\n")

        old_fit = fit_old(monday_fit_local, pct)
        new_fit, window_common, floor = fit_new(monday_fit_local, pct)
        report_lines.append(
            f"Old: {len(old_fit.config.thresholds)} thresholds fit. New: "
            f"{len(new_fit.config.thresholds)} count/ratio thresholds fit + window rarity floor "
            f"{floor:.4%} (common-value-set sizes: "
            f"{', '.join(f'{f}={len(window_common[f])}' for f in WINDOW_FEATURES)}).\n"
        )

        pop_defs = [("Monday holdout (benign-only -- every escalation here is a false alarm)", monday_holdout_local),
                    ("Friday", friday_local), ("Combined (Monday holdout + Friday)", combined_local)]

        firing_this_pct = {}
        for pop_name, pop_df in pop_defs:
            old_crossings = compute_crossings(pop_df, old_fit.config.thresholds)
            old_escalated = escalate_from_crossings(old_crossings, EscalationRule.K_OF_N, 2)
            new_escalated, new_count_x, new_ratio_x, new_window_x = escalate_new(pop_df, new_fit, window_common)

            old_b = escalation_breakdown(pop_df, old_escalated)
            new_b = escalation_breakdown(pop_df, new_escalated)

            if pop_name.startswith("Monday"):
                report_lines.append(f"Monday holdout false alarms -- old: **{old_b['n_escalated']:,}**"
                                     f" of {old_b['n_total']:,} ({old_b['n_escalated']/old_b['n_total']:.3%}); "
                                     f"new: **{new_b['n_escalated']:,}** ({new_b['n_escalated']/new_b['n_total']:.3%}).\n")

            report_lines.extend(render_population_table(pop_name, {"old (20-feature, k=2)": old_b, "new (reduced)": new_b}))

            if pop_name == "Friday":
                b = escalation_breakdown(pop_df, new_escalated)
                m = metrics_from_breakdown(b)
                report_lines.append("Friday attacks NOT escalated (new config), by class: " +
                                     ", ".join(f"{l}={m['missed'][l]:,} of {b['by_label_total'][l]:,}" for l in ATTACK_LABELS) + "\n")
                b_old = escalation_breakdown(pop_df, old_escalated)
                m_old = metrics_from_breakdown(b_old)
                report_lines.append("Friday attacks NOT escalated (old config), by class: " +
                                     ", ".join(f"{l}={m_old['missed'][l]:,} of {b_old['by_label_total'][l]:,}" for l in ATTACK_LABELS) + "\n")

                firing_this_pct["friday_old"] = firing_rates(pop_df, old_crossings)
                combined_new_crossings = pd.concat([new_count_x, new_ratio_x, new_window_x], axis=1)
                firing_this_pct["friday_new"] = firing_rates(pop_df, combined_new_crossings)
            if pop_name.startswith("Monday"):
                firing_this_pct["monday_old"] = firing_rates(pop_df, old_crossings)
                combined_new_crossings = pd.concat([new_count_x, new_ratio_x, new_window_x], axis=1)
                firing_this_pct["monday_new"] = firing_rates(pop_df, combined_new_crossings)

        all_firing[pct] = firing_this_pct

    # ---- features firing section ----
    report_lines.append("## Features actually firing (crossing rate), new (reduced) config\n")
    for pct in OPERATING_PERCENTILES:
        report_lines.append(f"### p{pct}\n")
        report_lines.append("| feature | group | Friday crossing rate | Monday holdout crossing rate |")
        report_lines.append("|---|---|---|---|")
        fri = all_firing[pct].get("friday_new", {})
        mon = all_firing[pct].get("monday_new", {})
        all_feats = sorted(set(fri) | set(mon), key=lambda f: -fri.get(f, 0))
        for feat in all_feats:
            group = "COUNT" if feat in COUNT_FEATURES else ("RATIO" if feat in RATIO_FEATURES else "WINDOW")
            report_lines.append(f"| `{feat}` | {group} | {fri.get(feat, 0):.4%} | {mon.get(feat, 0):.4%} |")
        report_lines.append("")

    report_lines.append("## Features actually firing (crossing rate), old (20-feature) config\n")
    for pct in OPERATING_PERCENTILES:
        report_lines.append(f"### p{pct}\n")
        report_lines.append("| feature | Friday crossing rate | Monday holdout crossing rate |")
        report_lines.append("|---|---|---|")
        fri = all_firing[pct].get("friday_old", {})
        mon = all_firing[pct].get("monday_old", {})
        all_feats = sorted(set(fri) | set(mon), key=lambda f: -fri.get(f, 0))
        for feat in all_feats:
            report_lines.append(f"| `{feat}` | {fri.get(feat, 0):.4%} | {mon.get(feat, 0):.4%} |")
        report_lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
