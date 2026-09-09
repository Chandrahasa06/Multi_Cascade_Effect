"""Render sweep results to results/: CSV tables, a markdown summary, and
the recall-vs-escalation-rate curve as a plot.

Chart form: small multiples (one panel per attack class) rather than one
shared-axis line chart with a dozen-plus overlapping colors — CICIDS2017
has more attack classes than a categorical palette can keep
distinguishable at a glance, and a per-class panel is also just the
better comparison form here. Each panel is a single series, so it needs
no legend (the title names it, per the dataviz skill) — color is used
only to flag statistical reliability (blue = enough flows to trust the
curve, muted gray = a small-sample class, see SMALL_SAMPLE_THRESHOLD),
not to distinguish the ~14 class identities from each other.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eval.sweep import SweepResult, nearest_point

DEFAULT_RESULTS_DIR = Path("results")

#: classes this small swing tens of percentage points per flow — report
#: raw counts and flag them, never just a percentage.
SMALL_SAMPLE_THRESHOLD = 50

#: dataviz skill's validated categorical slot 1 (blue) / muted secondary ink
RELIABLE_COLOR = "#2a78d6"
UNRELIABLE_COLOR = "#9a9990"


def per_class_recall_table(
    result: SweepResult,
    target_rates: Sequence[float] = (0.01, 0.05, 0.10),
    by: str = "overall_escalation_rate",
) -> pd.DataFrame:
    """Per-class recall at specific escalation-rate operating points
    (see nearest_point for what `by` selects), with raw counts and a
    small_sample flag alongside every row."""
    frames = []
    for rate in target_rates:
        point = nearest_point(result, rate, by=by)
        table = point.per_class_recall.copy()
        table["target_escalation_rate"] = rate
        table["actual_escalation_rate"] = point.overall_escalation_rate
        table["percentile"] = point.percentile
        table["small_sample"] = table["total_count"] < SMALL_SAMPLE_THRESHOLD
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def build_recall_curve_frame(result: SweepResult) -> pd.DataFrame:
    rows = []
    for point in result.points:
        for _, row in point.per_class_recall.iterrows():
            rows.append(
                {
                    "percentile": point.percentile,
                    "overall_escalation_rate": point.overall_escalation_rate,
                    "label": row["label"],
                    "recall": row["recall"],
                    "escalated_count": row["escalated_count"],
                    "total_count": row["total_count"],
                }
            )
    return pd.DataFrame(rows)


def plot_recall_curve(curve_frame: pd.DataFrame, path) -> None:
    labels = sorted(curve_frame["label"].unique())
    n = len(labels)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False, sharey=True
    )

    for i, label in enumerate(labels):
        ax = axes[i // ncols][i % ncols]
        group = curve_frame[curve_frame["label"] == label].sort_values("overall_escalation_rate")
        max_n = int(group["total_count"].max())
        color = RELIABLE_COLOR if max_n >= SMALL_SAMPLE_THRESHOLD else UNRELIABLE_COLOR

        ax.plot(
            group["overall_escalation_rate"] * 100,
            group["recall"] * 100,
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax.set_xscale("log")
        ax.set_ylim(-5, 105)
        suffix = " (unreliable, n<50)" if max_n < SMALL_SAMPLE_THRESHOLD else ""
        ax.set_title(f"{label} (n={max_n}){suffix}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    # blank any unused grid cells
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.supxlabel("Overall escalation rate (%, log scale)", fontsize=9)
    fig.supylabel("Recall (%)", fontsize=9)
    fig.suptitle("Recall vs escalation rate, per attack class", fontsize=11)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ablation_comparison_table(
    ablations: Dict[str, SweepResult], target_rate: float = 0.05
) -> pd.DataFrame:
    """One row per ablation at a fixed operating point (default 5%
    overall escalation rate), so ablations are compared like-for-like."""
    rows = []
    for name, result in ablations.items():
        point = nearest_point(result, target_rate)
        overall = point.per_class_recall
        total_escalated = int(overall["escalated_count"].sum())
        total_attacks = int(overall["total_count"].sum())
        rows.append(
            {
                "ablation": name,
                "percentile": point.percentile,
                "actual_escalation_rate": point.overall_escalation_rate,
                "benign_escalation_rate": point.benign_escalation_rate,
                "overall_attack_recall": total_escalated / total_attacks if total_attacks else 0.0,
                "n_thresholds_fit": point.n_thresholds_fit,
            }
        )
    return pd.DataFrame(rows)


def per_class_ablation_delta(
    baseline: SweepResult,
    ablation: SweepResult,
    target_rate: float = 0.05,
    by: str = "benign_escalation_rate",
) -> pd.DataFrame:
    """Per-class recall for two sweeps at the same operating point, with
    the delta — the tool for "quantify the per-source-feature effect on
    Patator/Bot/Infiltration" rather than eyeballing two tables. Anchored
    on benign rate by default, not overall rate — see nearest_point's
    docstring for why the overall rate barely moves across this dataset's
    percentile range and would put both sweeps at nearly the same
    (uninformatively tight) point."""
    base_point = nearest_point(baseline, target_rate, by=by)
    other_point = nearest_point(ablation, target_rate, by=by)
    merged = base_point.per_class_recall.merge(
        other_point.per_class_recall, on="label", suffixes=("_baseline", "_ablation"), how="outer"
    ).fillna(0.0)
    merged["recall_delta"] = merged["recall_baseline"] - merged["recall_ablation"]
    return merged.sort_values("recall_delta", ascending=False)


def write_results(
    sweep_result: SweepResult,
    ablations: Dict[str, SweepResult],
    trigger_freq: pd.DataFrame,
    join_rates: pd.DataFrame,
    results_dir=DEFAULT_RESULTS_DIR,
    feature_ablation_delta: pd.DataFrame = None,
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    curve_frame = build_recall_curve_frame(sweep_result)
    curve_frame.to_csv(results_dir / "recall_curve.csv", index=False)
    plot_recall_curve(curve_frame, results_dir / "recall_curve.png")

    class_table = per_class_recall_table(sweep_result)
    class_table.to_csv(results_dir / "per_class_recall.csv", index=False)

    # supplementary: same table anchored on the *benign* escalation rate
    # rather than the overall one — see nearest_point's docstring for why
    # they diverge sharply on this dataset (attack traffic is ~22% of the
    # week's total flow volume, so "overall" rarely reaches single digits).
    class_table_by_benign = per_class_recall_table(sweep_result, by="benign_escalation_rate")
    class_table_by_benign.to_csv(results_dir / "per_class_recall_by_benign_rate.csv", index=False)

    sweep_result.as_frame().to_csv(results_dir / "sweep_summary.csv", index=False)

    ablation_table = ablation_comparison_table(ablations)
    ablation_table.to_csv(results_dir / "ablation_comparison.csv", index=False)

    trigger_freq.to_csv(results_dir / "trigger_frequency.csv", index=False)
    join_rates.to_csv(results_dir / "join_rates.csv", index=False)
    if feature_ablation_delta is not None:
        feature_ablation_delta.to_csv(results_dir / "per_source_feature_effect.csv", index=False)

    _write_markdown_summary(
        results_dir / "summary.md",
        class_table,
        class_table_by_benign,
        ablation_table,
        trigger_freq,
        join_rates,
        feature_ablation_delta,
    )


def _write_markdown_summary(
    path,
    class_table: pd.DataFrame,
    class_table_by_benign: pd.DataFrame,
    ablation_table: pd.DataFrame,
    trigger_freq: pd.DataFrame,
    join_rates: pd.DataFrame,
    feature_ablation_delta: pd.DataFrame = None,
) -> None:
    lines: List[str] = ["# Selector sweep results", ""]

    lines.append("## Per-class recall at fixed *overall* escalation rates")
    lines.append("")
    lines.append(
        "CICIDS2017's attack traffic is ~22% of the operational week's total "
        "flow volume (unrealistic vs. a real network) — DoS Hulk and DDoS alone "
        "are large enough classes that even modest recall on them keeps the "
        "*overall* escalation rate above 11% at every percentile this sweep "
        "tried (90.0-99.99), so the 1%/5%/10% targets below all land on the "
        "same (tightest) point. See the benign-anchored table further down for "
        "the operating points that actually vary."
    )
    lines.append("")
    for rate in sorted(class_table["target_escalation_rate"].unique()):
        subset = class_table[class_table["target_escalation_rate"] == rate]
        actual = subset["actual_escalation_rate"].iloc[0]
        lines.append(f"### Target {rate:.0%} (actual {actual:.2%})")
        lines.append("")
        lines.append("| class | recall | escalated | total | reliable |")
        lines.append("|---|---|---|---|---|")
        for _, row in subset.sort_values("recall", ascending=False).iterrows():
            reliability = "no (n<50)" if row["small_sample"] else "yes"
            lines.append(
                f"| {row['label']} | {row['recall']:.1%} | {int(row['escalated_count'])} "
                f"| {int(row['total_count'])} | {reliability} |"
            )
        lines.append("")

    lines.append("## Per-class recall at fixed *benign* escalation (false-positive) rates")
    lines.append("")
    lines.append(
        "The more operationally meaningful cost axis: an attack flow escalating "
        "is the point, not a cost; a benign flow escalating is the actual false-"
        "positive burden on the control plane, and this rate spans a wide, "
        "controllable range across the percentile sweep (unlike the overall rate)."
    )
    lines.append("")
    for rate in sorted(class_table_by_benign["target_escalation_rate"].unique()):
        subset = class_table_by_benign[class_table_by_benign["target_escalation_rate"] == rate]
        actual = subset["actual_escalation_rate"].iloc[0]
        lines.append(f"### Target benign rate {rate:.0%} (actual overall {actual:.2%})")
        lines.append("")
        lines.append("| class | recall | escalated | total | reliable |")
        lines.append("|---|---|---|---|---|")
        for _, row in subset.sort_values("recall", ascending=False).iterrows():
            reliability = "no (n<50)" if row["small_sample"] else "yes"
            lines.append(
                f"| {row['label']} | {row['recall']:.1%} | {int(row['escalated_count'])} "
                f"| {int(row['total_count'])} | {reliability} |"
            )
        lines.append("")

    lines.append("## Ablation comparison (at ~5% overall escalation rate)")
    lines.append("")
    lines.append("| ablation | percentile | escalation rate | benign escalation | overall attack recall | # thresholds |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in ablation_table.iterrows():
        lines.append(
            f"| {row['ablation']} | {row['percentile']:.3f} | {row['actual_escalation_rate']:.2%} "
            f"| {row['benign_escalation_rate']:.2%} | {row['overall_attack_recall']:.1%} "
            f"| {int(row['n_thresholds_fit'])} |"
        )
    lines.append("")

    lines.append("## Per-feature trigger frequency")
    lines.append("")
    lines.append("| feature | crossing rate (all flows) | trigger share (of escalations) |")
    lines.append("|---|---|---|")
    for _, row in trigger_freq.iterrows():
        lines.append(f"| {row['feature']} | {row['crossing_rate']:.2%} | {row['trigger_share']:.2%} |")
    dead = trigger_freq[trigger_freq["crossing_rate"] == 0.0]["feature"].tolist()
    saturated = trigger_freq[trigger_freq["crossing_rate"] > 0.5]["feature"].tolist()
    lines.append("")
    if dead:
        lines.append(f"**Dead features (never cross):** {', '.join(dead)}")
        lines.append("")
    if saturated:
        lines.append(f"**Saturated features (cross on >50% of flows — threshold too loose):** {', '.join(saturated)}")
        lines.append("")

    if feature_ablation_delta is not None:
        lines.append("## Per-source-feature effect (baseline vs per-flow-only), per class")
        lines.append("")
        lines.append("| class | recall (all features) | recall (flow-only) | delta |")
        lines.append("|---|---|---|---|")
        for _, row in feature_ablation_delta.iterrows():
            lines.append(
                f"| {row['label']} | {row['recall_baseline']:.1%} | {row['recall_ablation']:.1%} "
                f"| {row['recall_delta']:+.1%} |"
            )
        lines.append("")

    lines.append("## Join rate per day")
    lines.append("")
    lines.append("| day | key_mode | join rate | flagged (<95%) |")
    lines.append("|---|---|---|---|")
    for _, row in join_rates.iterrows():
        flag = "YES" if row["join_rate"] < 0.95 else ""
        lines.append(f"| {row['day']} | {row['key_mode']} | {row['join_rate']:.2%} | {flag} |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
