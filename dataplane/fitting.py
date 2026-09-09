"""Benign-only threshold fitting.

Zero-day safety hinges on this module: thresholds must never be
influenced by attack traffic, or the selector degrades into a signature
matcher for known attacks and the whole zero-day claim collapses. That
is enforced here as an assertion, not a docstring promise or a comment
above the loop — see the first line of :func:`fit_thresholds` and
:func:`fit_thresholds_from_frame`. Attack labels exist for evaluation
only, in a separate module, never here.

Two entry points share one percentile-fitting core (:func:`_fit_feature`):

- :func:`fit_thresholds` takes live FlowState objects (and, optionally, a
  SrcTable) and computes features on the fly via selector.py. Good for
  small/ad-hoc fitting and what the unit tests exercise directly.
- :func:`fit_thresholds_from_frame` takes a DataFrame of *already
  computed* features (see eval/simulate.py's cache). This is what
  eval/sweep.py uses — fitting is re-run at every point of a percentile
  sweep, and re-deriving features from FlowState each time would mean
  re-running the whole flow-table simulation per sweep point, which is
  exactly the cost the caching layer exists to avoid.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from dataplane.flow_state import FlowState
from dataplane.selector import (
    EscalationRule,
    FeatureObservation,
    FeatureThreshold,
    SelectorConfig,
    compute_src_features,
    compute_tier1_features,
)
from dataplane.src_table import SrcTable

BENIGN_LABEL = "BENIGN"
DEFAULT_PERCENTILE = 99.5

#: columns in an eval/simulate.py feature frame that are metadata, not a
#: fittable feature (and not a "<feature>__low_confidence" companion).
NON_FEATURE_COLUMNS = frozenset(
    {"label", "mixed_label", "residual_ambiguous", "n_source_rows", "first_ts", "last_ts", "closure_reason"}
)

#: features that can be anomalous when either unusually high or unusually
#: low (e.g. tiny mean packet length is a scan/probe signature just as
#: much as a huge one is a data-exfiltration signature). Everything else
#: only gets a high-side threshold — "too low" isn't a meaningful anomaly
#: for, say, flow_bytes_per_sec.
TWO_SIDED_FEATURES = frozenset(
    {
        "fwd_pkt_len_mean",
        "bwd_pkt_len_mean",
        "pkt_len_range",
        "flow_iat_mean",
        "flow_iat_min",
        "init_win_bytes_fwd",
        "init_win_bytes_bwd",
    }
)


@dataclass(frozen=True, slots=True)
class FeatureFittingReport:
    """Per-feature fitting diagnostics. A feature that's undefined (or
    excluded as low-confidence) for most benign flows is a finding about
    that feature's usability, not a detail to bury in a log line."""

    total_flows: int
    excluded_undefined: int
    excluded_low_confidence: int
    used: int

    @property
    def excluded_fraction(self) -> float:
        if self.total_flows == 0:
            return 0.0
        return (self.excluded_undefined + self.excluded_low_confidence) / self.total_flows

    def to_dict(self) -> dict:
        return {
            "total_flows": self.total_flows,
            "excluded_undefined": self.excluded_undefined,
            "excluded_low_confidence": self.excluded_low_confidence,
            "used": self.used,
            "excluded_fraction": self.excluded_fraction,
        }


@dataclass(frozen=True, slots=True)
class FittingResult:
    config: SelectorConfig
    reports: Dict[str, FeatureFittingReport] = field(default_factory=dict)


def _compute_config_hash(thresholds: Dict[str, FeatureThreshold]) -> str:
    payload = json.dumps(
        {name: t.to_dict() for name, t in sorted(thresholds.items())}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _fit_feature(
    values: np.ndarray,
    low_confidence: np.ndarray,
    *,
    percentile: float,
    two_sided: bool,
    exclude_low_confidence: bool,
) -> Tuple[Optional[FeatureThreshold], FeatureFittingReport]:
    """Shared percentile-fitting core. `values` uses NaN for "undefined
    for this flow" (never 0); `low_confidence` is a same-length bool mask.
    Both exclusions are counted, never silently folded into the fit."""
    total = len(values)
    undefined_mask = np.isnan(values)
    excluded_undefined = int(undefined_mask.sum())

    if exclude_low_confidence:
        low_conf_mask = (~undefined_mask) & low_confidence
    else:
        low_conf_mask = np.zeros(total, dtype=bool)
    excluded_low_confidence = int(low_conf_mask.sum())

    used_values = values[(~undefined_mask) & (~low_conf_mask)]

    report = FeatureFittingReport(
        total_flows=total,
        excluded_undefined=excluded_undefined,
        excluded_low_confidence=excluded_low_confidence,
        used=len(used_values),
    )

    if len(used_values) == 0:
        return None, report

    high = float(np.percentile(used_values, percentile))
    low = float(np.percentile(used_values, 100.0 - percentile)) if two_sided else None
    return FeatureThreshold(high=high, low=low), report


def fit_thresholds(
    flows: Sequence[FlowState],
    labels: Sequence[str],
    *,
    src_table: Optional[SrcTable] = None,
    now_us_by_flow: Optional[Sequence[int]] = None,
    percentile: float = DEFAULT_PERCENTILE,
    two_sided_features: Set[str] = TWO_SIDED_FEATURES,
    rule: EscalationRule = EscalationRule.ANY,
    k: int = 1,
    exclude_low_confidence: bool = True,
) -> FittingResult:
    """Fit benign-only percentile thresholds for every Tier-1 feature
    (and every per-source feature, if `src_table` is given).

    `labels` must all be "BENIGN" — checked immediately, before anything
    else runs, not as an afterthought. Values of None (a feature that's
    undefined for a given flow — e.g. a rate over zero duration) are
    excluded from that feature's percentile computation entirely, never
    treated as 0; so are low-confidence observations (coarse-resolution
    timing — see selector.py), unless `exclude_low_confidence=False`.
    Both exclusions are counted and reported per feature.
    """
    assert all(label == BENIGN_LABEL for label in labels), (
        "fit_thresholds received a non-benign label — thresholds must be "
        "fitted on benign traffic only; attack labels are for evaluation, "
        "never for fitting."
    )

    if len(flows) != len(labels):
        raise ValueError("flows and labels must have the same length")
    if now_us_by_flow is not None and len(now_us_by_flow) != len(flows):
        raise ValueError("now_us_by_flow must have the same length as flows")

    per_flow_features: List[Dict[str, FeatureObservation]] = []
    for i, flow in enumerate(flows):
        features: Dict[str, FeatureObservation] = dict(compute_tier1_features(flow))
        if src_table is not None:
            now_us = now_us_by_flow[i] if now_us_by_flow is not None else flow.last_ts
            features.update(compute_src_features(src_table, flow.fwd_ip, now_us))
        per_flow_features.append(features)

    feature_names: Set[str] = set()
    for features in per_flow_features:
        feature_names.update(features.keys())

    thresholds: Dict[str, FeatureThreshold] = {}
    reports: Dict[str, FeatureFittingReport] = {}

    for feature_name in sorted(feature_names):
        values = np.full(len(per_flow_features), np.nan)
        low_conf = np.zeros(len(per_flow_features), dtype=bool)
        for i, features in enumerate(per_flow_features):
            observation = features.get(feature_name)
            if observation is None or observation.value is None:
                continue
            values[i] = observation.value
            low_conf[i] = observation.low_confidence

        threshold, report = _fit_feature(
            values,
            low_conf,
            percentile=percentile,
            two_sided=feature_name in two_sided_features,
            exclude_low_confidence=exclude_low_confidence,
        )
        reports[feature_name] = report
        if threshold is not None:
            thresholds[feature_name] = threshold

    config = SelectorConfig(
        thresholds=thresholds,
        rule=rule,
        k=k,
        config_hash=_compute_config_hash(thresholds),
    )
    return FittingResult(config=config, reports=reports)


def fit_thresholds_from_frame(
    df: pd.DataFrame,
    *,
    feature_names: Optional[Sequence[str]] = None,
    percentile: float = DEFAULT_PERCENTILE,
    two_sided_features: Set[str] = TWO_SIDED_FEATURES,
    rule: EscalationRule = EscalationRule.ANY,
    k: int = 1,
    exclude_low_confidence: bool = True,
) -> FittingResult:
    """Same fitting core as :func:`fit_thresholds`, but sourced from a
    cached per-flow feature DataFrame (see eval/simulate.py) instead of
    live FlowState objects — the fast path a percentile sweep needs.

    `feature_names` restricts fitting to a subset of columns (e.g. the
    per-flow-features-only ablation, which excludes the four per-source
    columns entirely rather than just leaving them unthresholded).
    """
    assert (df["label"] == BENIGN_LABEL).all(), (
        "fit_thresholds_from_frame received a non-benign label — thresholds "
        "must be fitted on benign traffic only; attack labels are for "
        "evaluation, never for fitting."
    )

    if feature_names is None:
        feature_names = sorted(
            c
            for c in df.columns
            if c not in NON_FEATURE_COLUMNS and not c.endswith("__low_confidence")
        )

    thresholds: Dict[str, FeatureThreshold] = {}
    reports: Dict[str, FeatureFittingReport] = {}

    for feature_name in sorted(feature_names):
        values = df[feature_name].to_numpy(dtype=float)
        low_conf_col = f"{feature_name}__low_confidence"
        if low_conf_col in df.columns:
            low_conf = df[low_conf_col].to_numpy(dtype=bool)
        else:
            low_conf = np.zeros(len(df), dtype=bool)

        threshold, report = _fit_feature(
            values,
            low_conf,
            percentile=percentile,
            two_sided=feature_name in two_sided_features,
            exclude_low_confidence=exclude_low_confidence,
        )
        reports[feature_name] = report
        if threshold is not None:
            thresholds[feature_name] = threshold

    config = SelectorConfig(
        thresholds=thresholds,
        rule=rule,
        k=k,
        config_hash=_compute_config_hash(thresholds),
    )
    return FittingResult(config=config, reports=reports)


def save_thresholds(config: SelectorConfig, path: Union[str, Path]) -> None:
    """Persist fitted thresholds to JSON so runs are reproducible."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)


def load_thresholds(path: Union[str, Path]) -> SelectorConfig:
    with open(path, "r", encoding="utf-8") as f:
        return SelectorConfig.from_dict(json.load(f))


def save_fitting_report(reports: Dict[str, FeatureFittingReport], path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({name: r.to_dict() for name, r in reports.items()}, f, indent=2, sort_keys=True)
