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
from enum import Enum
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


class FeatureKind(str, Enum):
    """Plain percentile fitting silently breaks for anything with a hard
    value bound: if benign mass sits exactly at the bound (a boolean's
    only two values, or a ratio saturating at 0/1 for short flows), the
    percentile itself lands on that bound, and a plain `>`/`<` comparison
    can then never cross it — the threshold is fit, looks reasonable, and
    is structurally uncrossable forever. syn_ratio/rst_ratio/
    no_response_flag are exactly this: a pure-SYN port scan or an
    unanswered-SYN probe pins them at their bound, which is the whole
    signal — and percentile fitting was quietly discarding it.

    CONTINUOUS: ordinary percentile fitting (the default for anything not
        listed in FEATURE_KINDS below).
    BOUNDED_RATIO: value lives in [0, 1] and can genuinely saturate at
        either end (syn_ratio, rst_ratio). Fit with an explicit
        saturation check (see _fit_bounded).
    BOOLEAN: value is exactly {0, 1} (no_response_flag). Same fix as
        BOUNDED_RATIO — a boolean is just a ratio with no interior.
    """

    CONTINUOUS = "continuous"
    BOUNDED_RATIO = "bounded_ratio"
    BOOLEAN = "boolean"


FEATURE_KINDS: Dict[str, FeatureKind] = {
    "no_response_flag": FeatureKind.BOOLEAN,
    "syn_ratio": FeatureKind.BOUNDED_RATIO,
    "rst_ratio": FeatureKind.BOUNDED_RATIO,
}


def feature_kind(name: str) -> FeatureKind:
    return FEATURE_KINDS.get(name, FeatureKind.CONTINUOUS)


@dataclass(frozen=True, slots=True)
class FeatureFittingReport:
    """Per-feature fitting diagnostics. A feature that's undefined (or
    excluded as low-confidence) for most benign flows is a finding about
    that feature's usability, not a detail to bury in a log line — and so
    is a feature whose fitted threshold turned out to be unfittable
    because benign mass saturates its hard bound (`saturated=True`) at
    this percentile. `used`/`excluded_*` and `saturated` are different
    failures: the first is about whether there was data to fit on at
    all; the second is about whether a bound made that data unusable
    regardless of how much of it there was."""

    total_flows: int
    excluded_undefined: int
    excluded_low_confidence: int
    used: int
    kind: str = FeatureKind.CONTINUOUS.value
    saturated: bool = False

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
            "kind": self.kind,
            "saturated": self.saturated,
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


def _fit_unbounded(used_values: np.ndarray, percentile: float, two_sided: bool) -> FeatureThreshold:
    high = float(np.percentile(used_values, percentile))
    low = float(np.percentile(used_values, 100.0 - percentile)) if two_sided else None
    return FeatureThreshold(high=high, low=low)


def _fit_bounded_side(used_values: np.ndarray, percentile: float, bound: float) -> Tuple[Optional[float], bool]:
    """One side (high or low) of a [0, 1]-bounded feature. Returns
    (threshold_value_or_None, saturated).

    Ordinary percentile fitting is used unless it lands exactly on the
    theoretical bound — the only case a plain `>`/`<` comparison can
    never satisfy (nothing can exceed 1.0 or go below 0.0). When that
    happens there is provably no threshold at this percentile that both
    fires and respects the false-positive budget: for linear-
    interpolation percentiles, the computed value can only equal a hard
    bound exactly when at least the top (100-percentile)% of the data
    already sits at that bound — i.e. the very existence of saturation
    means benign mass there already meets or exceeds this percentile's
    own budget. So: saturated always means unfittable at this
    percentile, not "maybe, check the rate" — omit the feature and
    report why, rather than fit a threshold nothing can ever cross.
    """
    raw = float(np.percentile(used_values, percentile))
    if np.isclose(raw, bound):
        return None, True
    return raw, False


def _fit_bounded(
    used_values: np.ndarray, percentile: float, two_sided: bool
) -> Tuple[Optional[FeatureThreshold], bool]:
    """Fit a feature hard-bounded to [0, 1] (a boolean is just a bound
    with no interior — see _fit_bounded_side)."""
    high, saturated_high = _fit_bounded_side(used_values, percentile, 1.0)

    low, saturated_low = None, False
    if two_sided:
        low, saturated_low = _fit_bounded_side(used_values, 100.0 - percentile, 0.0)

    saturated = saturated_high or saturated_low
    if high is None and low is None:
        return None, saturated
    return FeatureThreshold(high=high, low=low), saturated


def _fit_feature(
    values: np.ndarray,
    low_confidence: np.ndarray,
    *,
    percentile: float,
    two_sided: bool,
    exclude_low_confidence: bool,
    kind: FeatureKind = FeatureKind.CONTINUOUS,
) -> Tuple[Optional[FeatureThreshold], FeatureFittingReport]:
    """Shared fitting core. `values` uses NaN for "undefined for this
    flow" (never 0); `low_confidence` is a same-length bool mask. Both
    exclusions are counted, never silently folded into the fit. `kind`
    selects ordinary percentile fitting (CONTINUOUS) or the
    saturation-aware rarity fit (BOUNDED_RATIO/BOOLEAN) — see
    FeatureKind and _fit_bounded."""
    total = len(values)
    undefined_mask = np.isnan(values)
    excluded_undefined = int(undefined_mask.sum())

    if exclude_low_confidence:
        low_conf_mask = (~undefined_mask) & low_confidence
    else:
        low_conf_mask = np.zeros(total, dtype=bool)
    excluded_low_confidence = int(low_conf_mask.sum())

    used_values = values[(~undefined_mask) & (~low_conf_mask)]

    saturated = False
    threshold: Optional[FeatureThreshold] = None
    if len(used_values) > 0:
        if kind in (FeatureKind.BOUNDED_RATIO, FeatureKind.BOOLEAN):
            threshold, saturated = _fit_bounded(used_values, percentile, two_sided)
        else:
            threshold = _fit_unbounded(used_values, percentile, two_sided)

    report = FeatureFittingReport(
        total_flows=total,
        excluded_undefined=excluded_undefined,
        excluded_low_confidence=excluded_low_confidence,
        used=len(used_values),
        kind=kind.value,
        saturated=saturated,
    )
    return threshold, report


def fit_thresholds(
    flows: Sequence[FlowState],
    labels: Sequence[str],
    *,
    src_table: Optional[SrcTable] = None,
    now_us_by_flow: Optional[Sequence[int]] = None,
    percentile: float = DEFAULT_PERCENTILE,
    two_sided_features: Set[str] = TWO_SIDED_FEATURES,
    rule: EscalationRule = EscalationRule.K_OF_N,
    k: int = 2,
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
            kind=feature_kind(feature_name),
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
    rule: EscalationRule = EscalationRule.K_OF_N,
    k: int = 2,
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
            kind=feature_kind(feature_name),
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
