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
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

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
    """`kind` is still recorded per feature (diagnostic metadata, and one
    real fitting decision — see BOOLEAN below) but as of the generic
    saturation fix it no longer selects *how* a feature is fit: every
    feature goes through the same saturation-aware core
    (:func:`_fit_side`), continuous or not. Percentile fitting silently
    breaks whenever fit-half mass piles up at a feature's own extreme
    value: the percentile lands exactly there, and a plain `>`/`<`
    comparison can then never cross it — the threshold is fit, looks
    reasonable, and is structurally uncrossable forever. This has now
    been hit three separate ways in this project: a boolean's only
    "true" value (`no_response_flag`), a [0,1] ratio saturating at its
    bound (`syn_ratio`/`rst_ratio`), and — the one the old bounded-only
    check missed entirely — an *unbounded* count feature
    (`flows_per_src` on the CSV adapter) whose per-source accounting
    piles 14.2% of a day's benign mass onto a single tied maximum. One
    mechanism now covers all three; `kind` only still matters for
    deciding what to do once saturation IS detected (see
    RARITY_MAX_DISTINCT_* and _fit_feature below).

    CONTINUOUS: no special handling beyond the generic saturation check
        (the default for anything not listed in FEATURE_KINDS below).
    BOUNDED_RATIO: value lives in [0, 1] and can genuinely saturate at
        either end (syn_ratio, rst_ratio). No different fitting path
        from CONTINUOUS any more — kept as a label for reporting/tests.
    BOOLEAN: value is exactly {0, 1} (no_response_flag). The one place
        `kind` still changes behavior: a saturated boolean is always
        EXCLUDED, never rarity-fit — its value space is closed to two
        states, so a "common value set" built from training data can
        never flag anything a boolean couldn't already take, making
        rarity-fit provably useless for this kind specifically.
    """

    CONTINUOUS = "continuous"
    BOUNDED_RATIO = "bounded_ratio"
    BOOLEAN = "boolean"


FEATURE_KINDS: Dict[str, FeatureKind] = {
    "no_response_flag": FeatureKind.BOOLEAN,
    "syn_ratio": FeatureKind.BOUNDED_RATIO,
    "rst_ratio": FeatureKind.BOUNDED_RATIO,
    # bounded in (0, 1], saturating at 1.0 for a perfectly regular
    # beacon (every IAT identical) — see selector._iat_regularity.
    "flow_iat_regularity": FeatureKind.BOUNDED_RATIO,
    # per-source ratios: numerator is a sub-count of the denominator
    # (distinct ports/ips touched, or SYNs-without-response, can't exceed
    # flows_per_src) so each is bounded in [0, 1] by construction, same
    # saturation risk as syn_ratio/rst_ratio above — see
    # selector.compute_src_features.
    "port_diversity_ratio": FeatureKind.BOUNDED_RATIO,
    "unanswered_syn_ratio": FeatureKind.BOUNDED_RATIO,
    "dst_concentration": FeatureKind.BOUNDED_RATIO,
}


def feature_kind(name: str) -> FeatureKind:
    return FEATURE_KINDS.get(name, FeatureKind.CONTINUOUS)


#: A saturated feature is eligible for a rarity fit (value-membership,
#: not magnitude) instead of outright exclusion when it has few enough
#: distinct states for membership to be meaningful. Both caps must hold
#: — the absolute one alone is fooled by a huge fit-half, the ratio one
#: alone by a tiny one. Calibrated against this project's own discrete-
#: but-not-boolean features: `flows_per_src` on the CSV adapter (191
#: distinct of 264,959 informative rows, ~0.07%) and
#: `init_win_bytes_fwd`/`bwd` (on the order of 4,000-5,600 distinct of
#: 170-330K informative rows, ~1-3%) — vs. a genuinely continuous
#: feature (e.g. `flow_iat_mean` under real microsecond timing), whose
#: informative values are close to unique.
RARITY_MAX_DISTINCT_ABS = 10_000
RARITY_MAX_DISTINCT_RATIO = 0.05


@dataclass(frozen=True, slots=True)
class FeatureFittingReport:
    """Per-feature fitting diagnostics. A feature that's undefined (or
    excluded as low-confidence) for most benign flows is a finding about
    that feature's usability, not a detail to bury in a log line — and so
    is a feature whose fitted threshold turned out to be unfittable
    because benign mass saturates its own extreme (`saturated=True`) at
    this percentile. `used`/`excluded_*` and `saturated` are different
    failures: the first is about whether there was data to fit on at
    all; the second is about whether saturation made that data unusable
    regardless of how much of it there was.

    `action` is the outcome once saturation is (or isn't) detected —
    "kept" (ordinary threshold, the common case), "excluded" (dropped
    for this operating point), or "rarity_fit" (value-membership
    instead of magnitude) — always present, so a saturated feature's
    fate is visible without cross-referencing `saturated` and whether
    the feature shows up in `SelectorConfig.thresholds` separately.
    `feature_max`/`feature_min`/`tied_mass_high`/`tied_mass_low` are the
    generic saturation diagnostics themselves — the exact numbers behind
    the `action`, not just the boolean. `rarity_coverage`, when
    `action == "rarity_fit"`, is the fraction of the fit-half's own used
    values that landed in the common set — a coverage near 1.0 is a
    visible warning that this particular rarity fit has little room left
    to flag anything within the training distribution (it can still flag
    genuinely novel values), reported rather than silently upgraded to
    "excluded" so the report itself carries that judgment call, not the
    code."""

    total_flows: int
    excluded_undefined: int
    excluded_low_confidence: int
    used: int
    kind: str = FeatureKind.CONTINUOUS.value
    saturated: bool = False
    action: str = "kept"
    threshold_high: Optional[float] = None
    threshold_low: Optional[float] = None
    feature_max: Optional[float] = None
    feature_min: Optional[float] = None
    tied_mass_high: Optional[float] = None
    tied_mass_low: Optional[float] = None
    n_distinct: Optional[int] = None
    rarity_coverage: Optional[float] = None

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
            "action": self.action,
            "threshold_high": self.threshold_high,
            "threshold_low": self.threshold_low,
            "feature_max": self.feature_max,
            "feature_min": self.feature_min,
            "tied_mass_high": self.tied_mass_high,
            "tied_mass_low": self.tied_mass_low,
            "n_distinct": self.n_distinct,
            "rarity_coverage": self.rarity_coverage,
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


@dataclass(frozen=True, slots=True)
class _SideFit:
    """One side (high or low) of a saturation-aware percentile fit."""

    value: float
    extreme: float
    tied_mass_at_extreme: float
    saturated: bool


def _fit_side(used_values: np.ndarray, np_percentile: float, budget: float, direction: str) -> _SideFit:
    """Fit one side of a threshold and check whether it's reachable.
    `np_percentile` is what gets handed to `np.percentile` (== the
    caller's percentile for the high side, `100 - percentile` for the
    low side); `budget` is always `(100 - percentile) / 100` for BOTH
    sides — the same intended false-positive rate either direction,
    independent of which `np_percentile` value produces it.

    Saturated means the threshold is provably unreachable by a
    `>`/`<` comparison: either it lands exactly on this feature's own
    observed extreme (nothing in the fit-half exceeds a true maximum /
    goes below a true minimum), or the mass tied AT that extreme already
    exceeds the percentile's own false-positive budget — the general
    form of the old bounded-ratio check (which only looked for the
    *theoretical* 0/1 bound): a percentile can only land exactly on the
    extreme once enough mass sits there, so checking the tied mass
    directly catches near-misses (e.g. floating-point interpolation)
    that an exact `np.isclose(raw, extreme)` alone could paper over.

    `n_distinct > 1` guards a genuinely constant feature (every used
    value identical, e.g. a boolean that never once fired True in
    training): 100% of mass necessarily sits "at the extreme" then, but
    that's not the saturation pathology — a plain `>`/`<` at that
    constant value still correctly separates it from anything different
    that attack data might present. The pathology needs OTHER values
    genuinely present too, with the tied mass crowding out the tail
    anyway.

    The tied-mass condition additionally requires at least 2 flows
    actually AT the extreme (genuine duplication), not just a high
    *fraction* — with a small fit-half, the single largest of many
    otherwise-unique values trivially represents a large share of the
    sample (10 rows, all distinct: the max is "10% of the data" by
    construction, with nothing tied there at all) and must not read as
    saturation. Real saturation needs repeated identical observations
    piling up, which is what actually blocks a `>`/`<` comparison from
    separating anything — a unique top value never does.
    """
    raw = float(np.percentile(used_values, np_percentile))
    extreme = float(used_values.max()) if direction == "high" else float(used_values.min())
    n_distinct = int(np.unique(used_values).size)
    at_extreme = np.isclose(used_values, extreme)
    count_at_extreme = int(at_extreme.sum())
    tied_mass_at_extreme = float(count_at_extreme / len(used_values))
    saturated = n_distinct > 1 and (
        bool(np.isclose(raw, extreme)) or (count_at_extreme >= 2 and tied_mass_at_extreme > budget)
    )
    return _SideFit(value=raw, extreme=extreme, tied_mass_at_extreme=tied_mass_at_extreme, saturated=saturated)


def _fit_rarity(used_values: np.ndarray, budget: float) -> Tuple[FrozenSet[float], float]:
    """Value-membership fit: values individually covering at least
    `budget` (== the percentile's own false-positive rate) of
    `used_values` are "common"; anything else — including values never
    seen during fitting at all — is rare enough to escalate on. Returns
    (common_values, coverage), coverage being the fraction of
    `used_values` whose own value landed in the common set: reported
    directly (see FeatureFittingReport) rather than silently overridden
    when it's high, so a near-useless rarity fit (little room left to
    flag anything within the training distribution) is visible in the
    report instead of hidden by a code-side judgment call."""
    uniq, counts = np.unique(used_values, return_counts=True)
    freqs = counts / len(used_values)
    common_mask = freqs >= budget
    common = frozenset(float(v) for v in uniq[common_mask])
    coverage = float(counts[common_mask].sum() / len(used_values))
    return common, coverage


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
    exclusions are counted, never silently folded into the fit.

    Every feature goes through the same saturation-aware fit
    (:func:`_fit_side`) regardless of `kind` — see FeatureKind's
    docstring for why the old kind-based branch (bounded vs. unbounded
    fitting) was itself the bug (it only protected features hand-listed
    in FEATURE_KINDS, missing e.g. `flows_per_src` entirely). `kind`
    still decides one thing: whether a saturated feature is eligible for
    a rarity fit at all (BOOLEAN never is — see FeatureKind) — cardinality
    (RARITY_MAX_DISTINCT_*) decides it for everything else.
    """
    total = len(values)
    undefined_mask = np.isnan(values)
    excluded_undefined = int(undefined_mask.sum())

    if exclude_low_confidence:
        low_conf_mask = (~undefined_mask) & low_confidence
    else:
        low_conf_mask = np.zeros(total, dtype=bool)
    excluded_low_confidence = int(low_conf_mask.sum())

    used_values = values[(~undefined_mask) & (~low_conf_mask)]

    threshold: Optional[FeatureThreshold] = None
    saturated = False
    action = "kept"
    n_distinct: Optional[int] = None
    rarity_coverage: Optional[float] = None
    high_fit: Optional[_SideFit] = None
    low_fit: Optional[_SideFit] = None

    if len(used_values) > 0:
        n_distinct = int(np.unique(used_values).size)
        budget = (100.0 - percentile) / 100.0
        high_fit = _fit_side(used_values, percentile, budget, "high")
        low_fit = _fit_side(used_values, 100.0 - percentile, budget, "low") if two_sided else None
        saturated = high_fit.saturated or (low_fit.saturated if low_fit is not None else False)

        if not saturated:
            threshold = FeatureThreshold(high=high_fit.value, low=(low_fit.value if low_fit else None))
        else:
            eligible_for_rarity = (
                kind != FeatureKind.BOOLEAN
                and n_distinct <= RARITY_MAX_DISTINCT_ABS
                and n_distinct <= RARITY_MAX_DISTINCT_RATIO * len(used_values)
            )
            if eligible_for_rarity:
                common, rarity_coverage = _fit_rarity(used_values, budget)
                threshold = FeatureThreshold(common_values=common)
                action = "rarity_fit"
            else:
                threshold = None
                action = "excluded"

    report = FeatureFittingReport(
        total_flows=total,
        excluded_undefined=excluded_undefined,
        excluded_low_confidence=excluded_low_confidence,
        used=len(used_values),
        kind=kind.value,
        saturated=saturated,
        action=action,
        threshold_high=(high_fit.value if high_fit else None),
        threshold_low=(low_fit.value if low_fit else None),
        feature_max=(high_fit.extreme if high_fit else None),
        feature_min=(low_fit.extreme if low_fit else None),
        tied_mass_high=(high_fit.tied_mass_at_extreme if high_fit else None),
        tied_mass_low=(low_fit.tied_mass_at_extreme if low_fit else None),
        n_distinct=n_distinct,
        rarity_coverage=rarity_coverage,
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
