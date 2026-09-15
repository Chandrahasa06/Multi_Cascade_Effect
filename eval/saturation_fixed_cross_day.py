"""Part 2 of the saturation-fix task: re-run the cross-day test now that
`flows_per_src`'s percentile-saturation bug (dataplane/fitting.py) is
fixed generically. The previous cross-day test measured "5 features
don't generalize to other days" with the single highest-firing feature
(`flows_per_src`) structurally frozen on the CSV path — invalid. This
repeats it with the fix in place, and at three operating points (the
earlier p99.5-only sweep was meaningless for `flows_per_src` since its
threshold never moved across any tested percentile).

Both configs (5-feature and 18-feature-as-available, both count-AND-
ratio) are fit on Monday CSV benign only, chronologically split — same
adapter as Tuesday-Thursday, so both sides of the comparison come from
the same adapter, same discipline as
results/csv_refit_cross_day_report.md.

No API calls, no re-simulation -- reads already-cached feature parquets.

Run: python -m eval.saturation_fixed_cross_day
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from dataplane.fitting import BENIGN_LABEL, fit_thresholds_from_frame
from dataplane.selector import EscalationRule
from eval.five_feature_cross_day import (
    THURSDAY_CLASSES,
    TUESDAY_CLASSES,
    WEDNESDAY_CLASSES,
    load_csv_day,
    normalize_thursday_labels,
)
from eval.reduced_selector_report import COUNT_FEATURES, RATIO_FEATURES
from eval.seven_feature_selector import COUNT_5, RATIO_5
from eval.sweep import compute_crossings, split_monday_chronologically

OUT_MD = Path("results/saturation_fixed_cross_day_report.md")
PERCENTILES = (98.0, 99.0, 99.5)
TINY_CLASS_FLOOR = 50


def fit_subset(fit_df: pd.DataFrame, count_feats: Sequence[str], ratio_feats: Sequence[str], percentile: float):
    features = list(count_feats) + list(ratio_feats)
    return fit_thresholds_from_frame(
        fit_df, feature_names=features, percentile=percentile,
        rule=EscalationRule.K_OF_N, k=2, exclude_low_confidence=False,
    )


def escalate(df: pd.DataFrame, fit, count_feats: Sequence[str], ratio_feats: Sequence[str]) -> tuple:
    count_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in count_feats}
    ratio_thresholds = {k: v for k, v in fit.config.thresholds.items() if k in ratio_feats}
    count_x = compute_crossings(df, count_thresholds)
    ratio_x = compute_crossings(df, ratio_thresholds)
    count_any = count_x.any(axis=1) if count_x.shape[1] else pd.Series(False, index=df.index)
    ratio_any = ratio_x.any(axis=1) if ratio_x.shape[1] else pd.Series(False, index=df.index)
    escalated = count_any & ratio_any
    return escalated, count_x, ratio_x


def day_report(df: pd.DataFrame, classes: Sequence[str], escalated: pd.Series) -> dict:
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


def firing_rates(df: pd.DataFrame, crossings: pd.DataFrame) -> Dict[str, float]:
    if len(df) == 0 or crossings.shape[1] == 0:
        return {}
    return {k: float(v) for k, v in crossings.mean(axis=0).items()}


def render_saturation_report(fit, feature_names: Sequence[str]) -> List[str]:
    lines = ["| feature | action | threshold | feature max/min | tied mass | n_distinct |", "|---|---|---|---|---|---|"]
    for feat in feature_names:
        r = fit.reports.get(feat)
        if r is None:
            continue
        if r.action == "rarity_fit":
            t = fit.config.thresholds.get(feat)
            n_common = len(t.common_values) if t and t.common_values else 0
            threshold_str = f"rarity set ({n_common} common values)"
        elif feat in fit.config.thresholds:
            t = fit.config.thresholds[feat]
            parts = []
            if t.high is not None:
                parts.append(f"high>{t.high:.6g}")
            if t.low is not None:
                parts.append(f"low<{t.low:.6g}")
            threshold_str = ", ".join(parts)
        else:
            threshold_str = "--"
        extreme_str = f"max={r.feature_max:.6g}" if r.feature_max is not None else ""
        if r.feature_min is not None:
            extreme_str += f" min={r.feature_min:.6g}"
        tied_str = f"{r.tied_mass_high:.4%}" if r.tied_mass_high is not None else "--"
        lines.append(f"| `{feat}` | {r.action} | {threshold_str} | {extreme_str} | {tied_str} | {r.n_distinct} |")
    return lines


def render_day(name: str, config_label: str, r: dict) -> List[str]:
    lines = [f"#### {name} -- {config_label}\n"]
    lines.append(f"Total: {r['n_total']:,}  |  Escalated: **{r['n_escalated']:,}** "
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
    print("loading Monday CSV, splitting chronologically ...")
    monday_csv = load_csv_day("monday")
    monday_fit, monday_holdout = split_monday_chronologically(monday_csv)
    assert (monday_fit["label"] == BENIGN_LABEL).all()
    print(f"Monday CSV fit={len(monday_fit):,}  holdout={len(monday_holdout):,}")

    count_18 = [f for f in COUNT_FEATURES if f in monday_fit.columns]
    ratio_18 = [f for f in RATIO_FEATURES if f in monday_fit.columns]
    missing_18 = [f for f in RATIO_FEATURES if f not in monday_fit.columns]
    print(f"18-feature config uses {len(count_18)} COUNT + {len(ratio_18)} RATIO "
          f"(missing from schema4 cache: {missing_18})")

    print("loading Tuesday/Wednesday/Thursday (CSV, cached) ...")
    tuesday_df = load_csv_day("tuesday")
    wednesday_df = load_csv_day("wednesday")
    thursday_df = normalize_thursday_labels(load_csv_day("thursday_morning_webattacks", "thursday_afternoon_infiltration"))

    days = [
        ("Tuesday", tuesday_df, TUESDAY_CLASSES),
        ("Wednesday", wednesday_df, WEDNESDAY_CLASSES),
        ("Thursday", thursday_df, THURSDAY_CLASSES),
    ]

    configs = [
        ("5-feature", COUNT_5, RATIO_5),
        ("18-feature", count_18, ratio_18),
    ]

    lines: List[str] = []
    lines.append("# Saturation-fixed cross-day test\n")
    lines.append(
        "Re-run of the cross-day comparison with `dataplane/fitting.py`'s generic "
        "saturation detection in place (`flows_per_src`'s CSV threshold was previously "
        "frozen at its own tied maximum across p90-p100, silently kept as an ordinary "
        "threshold -- the single highest-firing feature disabled for the whole prior test). "
        "Both configs fit on Monday CSV benign, chronological split, at p98/p99/p99.5. "
        "No API calls, no re-simulation.\n\n"
        f"Monday CSV fit-half n={len(monday_fit):,}, holdout n={len(monday_holdout):,} "
        "(zero-day property asserted).\n"
    )

    all_summary_rows = []

    for pct in PERCENTILES:
        lines.append(f"## p{pct}\n")
        fits = {}
        for cfg_name, count_feats, ratio_feats in configs:
            fit = fit_subset(monday_fit, count_feats, ratio_feats, pct)
            fits[cfg_name] = fit
            n_active = len(fit.config.thresholds)
            n_total = len(count_feats) + len(ratio_feats)
            print(f"p{pct} {cfg_name}: {n_active}/{n_total} active thresholds")

            lines.append(f"### {cfg_name} saturation report (p{pct})\n")
            lines.extend(render_saturation_report(fit, list(count_feats) + list(ratio_feats)))
            lines.append("")

            esc_holdout, _, _ = escalate(monday_holdout, fit, count_feats, ratio_feats)
            holdout_fpr = esc_holdout.mean()
            print(f"  Monday CSV holdout FPR: {holdout_fpr:.4%}")
            lines.append(f"Monday CSV holdout FPR ({cfg_name}, p{pct}): **{holdout_fpr:.4%}**\n")

            for day_name, day_df, classes in days:
                esc, count_x, ratio_x = escalate(day_df, fit, count_feats, ratio_feats)
                r = day_report(day_df, classes, esc)
                lines.extend(render_day(day_name, f"{cfg_name}, p{pct}", r))
                print(f"  {day_name} ({cfg_name}): esc={r['escalation_rate']:.3%} fpr={r['benign_fpr']:.3%}")
                for cls, c in r["per_class"].items():
                    if c["n"]:
                        print(f"      {cls}: n={c['n']:,} recall={c['recall']:.1%}")
                        all_summary_rows.append((pct, cfg_name, day_name, cls, c["n"], c["recall"]))

                combined_x = pd.concat([count_x, ratio_x], axis=1)
                fri = firing_rates(day_df, combined_x)
                lines.append(f"Features firing ({day_name}, {cfg_name}, p{pct}):\n")
                lines.append("| feature | crossing rate |")
                lines.append("|---|---|")
                for feat, rate in sorted(fri.items(), key=lambda kv: -kv[1]):
                    lines.append(f"| `{feat}` | {rate:.4%} |")
                lines.append("")

    lines.append("## Summary: per-class recall across all percentiles/configs\n")
    lines.append("| pct | config | day | class | n | recall |")
    lines.append("|---|---|---|---|---|---|")
    for pct, cfg_name, day_name, cls, n, recall in all_summary_rows:
        lines.append(f"| p{pct} | {cfg_name} | {day_name} | {cls} | {n:,} | {recall:.1%} |")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
