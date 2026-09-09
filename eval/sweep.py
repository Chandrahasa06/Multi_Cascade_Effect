"""Sweep the selector's fitting percentile and measure recall vs
escalation-rate, using cached per-flow features (eval/simulate.py) so no
re-simulation happens per sweep point — only fit_thresholds_from_frame
(pure numpy over already-computed arrays) and vectorized threshold
comparison run per point.

Fitting is Monday-only, and only on Monday's chronological first half —
the second half is held out to measure the benign escalation rate
out-of-sample. Fitting on the same data you measure on gives you back
the percentile by construction (fit at p99.5, measure 0.5%, exactly,
always) — that's circular, not a result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dataplane.fitting import (
    BENIGN_LABEL,
    NON_FEATURE_COLUMNS,
    FittingResult,
    fit_thresholds_from_frame,
)
from dataplane.selector import EscalationRule, FeatureThreshold, SelectorConfig
from dataplane.src_table import SrcTable

#: 20 points from p90.0 to p99.99, geometrically spaced in (100-p) so
#: the sweep is denser where the interesting recall/escalation tradeoff
#: actually lives (the high-percentile end), not evenly wasted on the
#: coarse low end.
PERCENTILE_SWEEP: List[float] = sorted(100.0 - np.geomspace(10.0, 0.01, 20))

SRC_FEATURES = (
    "flows_per_src",
    "distinct_dst_ports_per_src",
    "distinct_dst_ips_per_src",
    "syn_without_synack_count",
)


def split_monday_chronologically(monday_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by first_ts and split at the midpoint by row count. The order
    rows happen to sit in the cached table is *closure* order, not start
    order (see eval/simulate.py) — this must sort, not just slice."""
    ordered = monday_df.sort_values("first_ts", kind="stable")
    mid = len(ordered) // 2
    return ordered.iloc[:mid], ordered.iloc[mid:]


def tier1_feature_names(df: pd.DataFrame) -> List[str]:
    return sorted(
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS
        and not c.endswith("__low_confidence")
        and c not in SRC_FEATURES
    )


def all_feature_names(df: pd.DataFrame) -> List[str]:
    return sorted(
        c for c in df.columns if c not in NON_FEATURE_COLUMNS and not c.endswith("__low_confidence")
    )


def compute_crossings(df: pd.DataFrame, thresholds: Dict[str, FeatureThreshold]) -> pd.DataFrame:
    """One boolean column per thresholded feature: did this flow cross it?
    NaN (undefined) never crosses — matches Selector.evaluate()'s
    `if observation.value is None: continue`."""
    crossings = {}
    for name, threshold in thresholds.items():
        if name not in df.columns:
            continue
        values = df[name]
        crossed = pd.Series(False, index=df.index)
        if threshold.high is not None:
            crossed = crossed | (values > threshold.high)
        if threshold.low is not None:
            crossed = crossed | (values < threshold.low)
        crossings[name] = crossed.fillna(False)
    return pd.DataFrame(crossings, index=df.index)


def escalate_from_crossings(crossings: pd.DataFrame, rule: EscalationRule, k: int) -> pd.Series:
    if crossings.shape[1] == 0:
        return pd.Series(False, index=crossings.index)
    if rule == EscalationRule.ANY:
        return crossings.any(axis=1)
    return crossings.sum(axis=1) >= k


def per_class_recall(df: pd.DataFrame, escalated: pd.Series) -> pd.DataFrame:
    """Recall per attack class (BENIGN excluded), with raw counts."""
    attack_mask = df["label"] != BENIGN_LABEL
    grouped = (
        pd.DataFrame({"label": df.loc[attack_mask, "label"], "escalated": escalated[attack_mask]})
        .groupby("label")["escalated"]
        .agg(escalated_count="sum", total_count="count")
    )
    grouped["recall"] = grouped["escalated_count"] / grouped["total_count"]
    return grouped.reset_index()


@dataclass
class SweepPoint:
    percentile: float
    overall_escalation_rate: float
    benign_escalation_rate: float
    per_class_recall: pd.DataFrame
    flow_table_eviction_rate: float
    src_table_eviction_rate: float
    n_thresholds_fit: int


@dataclass
class SweepResult:
    points: List[SweepPoint] = field(default_factory=list)
    ablation_name: str = "baseline"

    def as_frame(self) -> pd.DataFrame:
        rows = []
        for point in self.points:
            rows.append(
                {
                    "percentile": point.percentile,
                    "overall_escalation_rate": point.overall_escalation_rate,
                    "benign_escalation_rate": point.benign_escalation_rate,
                    "flow_table_eviction_rate": point.flow_table_eviction_rate,
                    "src_table_eviction_rate": point.src_table_eviction_rate,
                    "n_thresholds_fit": point.n_thresholds_fit,
                }
            )
        return pd.DataFrame(rows)


def run_sweep(
    monday_fit_half: pd.DataFrame,
    monday_holdout_half: pd.DataFrame,
    week_df: pd.DataFrame,
    *,
    percentiles: Sequence[float] = PERCENTILE_SWEEP,
    feature_names: Optional[Sequence[str]] = None,
    rule: EscalationRule = EscalationRule.ANY,
    k: int = 1,
    exclude_low_confidence: bool = True,
    flow_table_eviction_rate: float = 0.0,
    src_table_eviction_rate: float = 0.0,
    ablation_name: str = "baseline",
) -> Tuple[SweepResult, Dict[float, FittingResult]]:
    """One full percentile sweep: fit on monday_fit_half at each
    percentile, measure benign escalation on monday_holdout_half and
    overall/per-class numbers on week_df (which should be the
    operational week: Monday's holdout half + the rest of the week, so
    nothing is measured on data it was fit on).
    """
    result = SweepResult(ablation_name=ablation_name)
    fitting_results: Dict[float, FittingResult] = {}

    for percentile in percentiles:
        fit = fit_thresholds_from_frame(
            monday_fit_half,
            feature_names=feature_names,
            percentile=percentile,
            rule=rule,
            k=k,
            exclude_low_confidence=exclude_low_confidence,
        )
        fitting_results[percentile] = fit
        thresholds = fit.config.thresholds

        holdout_crossings = compute_crossings(monday_holdout_half, thresholds)
        benign_escalated = escalate_from_crossings(holdout_crossings, rule, k)

        week_crossings = compute_crossings(week_df, thresholds)
        week_escalated = escalate_from_crossings(week_crossings, rule, k)

        result.points.append(
            SweepPoint(
                percentile=percentile,
                overall_escalation_rate=float(week_escalated.mean()) if len(week_df) else 0.0,
                benign_escalation_rate=float(benign_escalated.mean()) if len(monday_holdout_half) else 0.0,
                per_class_recall=per_class_recall(week_df, week_escalated),
                flow_table_eviction_rate=flow_table_eviction_rate,
                src_table_eviction_rate=src_table_eviction_rate,
                n_thresholds_fit=len(thresholds),
            )
        )

    return result, fitting_results


def nearest_point(
    result: SweepResult, target_escalation_rate: float, by: str = "overall_escalation_rate"
) -> SweepPoint:
    """Closest sweep point to a target rate. `by` selects which rate:
    "overall_escalation_rate" (cost axis across the whole week, as
    literally specified) or "benign_escalation_rate" (false-positive
    rate on held-out Monday). On CICIDS2017 these behave very
    differently: attack traffic is ~22% of the operational week's total
    flow volume, so "overall" rarely drops into single digits even at
    the tightest percentile fit — see eval/run_all.py's printed note.
    "benign_escalation_rate" is the metric that actually varies smoothly
    across the percentile sweep and is the more operationally meaningful
    cost axis (an attack flow escalating is the point, not a cost)."""
    return min(result.points, key=lambda p: abs(getattr(p, by) - target_escalation_rate))


def trigger_frequency(df: pd.DataFrame, crossings: pd.DataFrame, escalated: pd.Series) -> pd.DataFrame:
    """Per-feature diagnostics at one operating point:
    - crossing_rate: fraction of ALL flows this feature independently crossed
      (near 100% => threshold too loose; near 0% => dead weight)
    - trigger_share: of flows that escalated at all, the fraction this
      feature contributed to ("how often is it the one that fires")
    """
    n_total = len(df)
    n_escalated = int(escalated.sum())
    rows = []
    for name in crossings.columns:
        crossed = crossings[name]
        rows.append(
            {
                "feature": name,
                "crossing_rate": float(crossed.sum()) / n_total if n_total else 0.0,
                "trigger_share": (
                    float((crossed & escalated).sum()) / n_escalated if n_escalated else 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("trigger_share", ascending=False).reset_index(drop=True)
