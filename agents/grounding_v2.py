"""Grounding v2: a self-calibrated isolation signal over benign traffic.

Replaces the band count in agents/escalation_grounding.py for the new arms. The old
path is untouched and stays reachable for reproducibility.

Parameters, fixed before any score was seen (do not change after running):
  features        the ten escalation features (CSV columns), ESCALATION_FEATURES
  transform       midrank ECDF of each feature within the reference window. A point
                  mass (many identical benign values) maps to its middle rank, so
                  zero-heavy features get a band, not an exact-match trap
  distance        Euclidean distance in percentile space, ten dimensions
  k               primary k = 50. Sensitivity only: 10 and 100 (not used for decisions)
  calibration     10,000 benign rows drawn from the reference window, seed 0
  isolation score share (x100) of calibration flows whose own k-NN distance is strictly
                  below this flow's distance, i.e. "more isolated than N% of normal"
  radius          median of the calibration 10-NN distances (set from benign, not a
                  hardcoded multiplier). The radius count is the number of reference flows
                  within that radius.
  leave-one-out   exactly one zero-distance reference row is removed for every query,
                  when one exists. This is the copy of the flow itself when the flow is
                  benign; it is the same rule for every flow (features only, no label).

Benign-only: the reference is built from BENIGN rows and asserted before anything else
happens, as in dataplane/fitting.py. No attack row reaches any signal in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from agents.escalation_grounding import ESCALATION_FEATURES

BENIGN = "BENIGN"
PRIMARY_K = 50
SENSITIVITY_KS = (10, 100)
CALIBRATION_N = 10_000
CALIBRATION_SEED = 0
RADIUS_K = 10
ZERO_TOL = 1e-12


def assert_benign_only(labels: pd.Series) -> None:
    bad = int((labels != BENIGN).sum())
    if bad:
        raise ValueError(f"{bad} non-benign rows reached the grounding v2 reference; reference must be benign-only")


def midrank_percentiles(sorted_col: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Midrank ECDF of x within sorted_col, in [0, 1]. Monotone non-decreasing in x.
    A value held by a point mass gets the middle of its mass, not its top or bottom."""
    n = sorted_col.shape[0]
    left = np.searchsorted(sorted_col, x, side="left")
    right = np.searchsorted(sorted_col, x, side="right")
    return (left + right) / (2.0 * n)


@dataclass
class ReferenceV2:
    """Benign reference window for the v2 signal. Build with `build_reference_v2`."""

    label: str  # which window this is, e.g. "all-days", "monday", "monday+tuesday"
    sorted_cols: List[np.ndarray]
    raw: np.ndarray  # (N, 10) benign raw values, ESCALATION_FEATURES order
    pct: np.ndarray  # (N, 10) percentile matrix, float64
    tree: KDTree
    calibration: Dict[int, np.ndarray] = field(default_factory=dict)  # k -> sorted calibration distances
    radius: float = 0.0
    calibration_rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def n(self) -> int:
        return int(self.raw.shape[0])


def build_reference_v2(benign: pd.DataFrame, label: str) -> ReferenceV2:
    """`benign` must hold the reference window's rows, with BENIGN labels only."""
    assert_benign_only(benign["Label"])
    if len(benign) == 0:
        raise ValueError(f"empty reference window {label!r}")
    raw = benign[list(ESCALATION_FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("non-finite values in the reference window")
    sorted_cols = [np.sort(raw[:, j]) for j in range(raw.shape[1])]
    pct = np.column_stack([midrank_percentiles(sorted_cols[j], raw[:, j]) for j in range(raw.shape[1])])
    tree = KDTree(pct)
    ref = ReferenceV2(label=label, sorted_cols=sorted_cols, raw=raw, pct=pct, tree=tree)

    rng = np.random.default_rng(CALIBRATION_SEED)
    m = min(CALIBRATION_N, ref.n)
    ref.calibration_rows = rng.choice(ref.n, size=m, replace=False)
    for k in (RADIUS_K, PRIMARY_K) + SENSITIVITY_KS:
        d = knn_distance_loo(ref, pct[ref.calibration_rows], k)
        ref.calibration[k] = np.sort(d)
    ref.radius = float(np.median(ref.calibration[RADIUS_K]))
    return ref


def query_percentiles(ref: ReferenceV2, features: Dict[str, float]) -> np.ndarray:
    x = np.array([float(features[f]) for f in ESCALATION_FEATURES], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite query features")
    return np.array([midrank_percentiles(ref.sorted_cols[j], np.array([x[j]]))[0] for j in range(len(x))])


def knn_distance_loo(ref: ReferenceV2, q_pct: np.ndarray, k: int) -> np.ndarray:
    """Distance to the k-th nearest reference flow, leave-one-out. Always defined: the
    reference has more than k rows, so the k-th neighbour exists. One zero-distance row
    (the query's own copy, if present) is removed before counting."""
    if k >= ref.n:
        raise ValueError(f"k={k} must be smaller than the reference size {ref.n}")
    q = np.atleast_2d(q_pct)
    dist, _ = ref.tree.query(q, k=min(k + 2, ref.n))
    out = np.empty(q.shape[0])
    for i in range(q.shape[0]):
        d = dist[i]
        zero = np.flatnonzero(d <= ZERO_TOL)
        if zero.size:
            d = np.delete(d, zero[0])  # leave-one-out: drop exactly one identical copy
        out[i] = d[k - 1]
    return out


def isolation_percentile(ref: ReferenceV2, distance: float, k: int = PRIMARY_K) -> float:
    """Share (x100) of benign calibration flows strictly less isolated than this flow."""
    cal = ref.calibration[k]
    return 100.0 * float(np.searchsorted(cal, distance, side="left")) / float(cal.shape[0])


def radius_count(ref: ReferenceV2, q_pct: np.ndarray) -> int:
    """Reference flows within the benign-derived radius, leave-one-out."""
    q = np.atleast_2d(q_pct)
    n_in = int(ref.tree.query_radius(q, r=ref.radius, count_only=True)[0])
    # leave-one-out: remove one zero-distance copy when present
    d0, _ = ref.tree.query(q, k=1)
    if d0[0][0] <= ZERO_TOL and n_in > 0:
        n_in -= 1
    return n_in


def per_feature_isolation(ref: ReferenceV2, q_pct: np.ndarray, k: int = PRIMARY_K) -> List[dict]:
    """For each feature: this flow's percentile, the median percentile of its k nearest
    benign neighbours, and the signed gap. The gap is the feature's contribution to
    isolation. Reported per record so isolation is not hidden in one scalar."""
    q = np.atleast_2d(q_pct)
    _, idx = ref.tree.query(q, k=min(k + 2, ref.n))
    idx = idx[0]
    d_all = ref.tree.query(q, k=min(k + 2, ref.n))[0][0]
    keep = [i for i in idx]
    zero = [j for j, dd in enumerate(d_all) if dd <= ZERO_TOL]
    if zero:
        keep = [i for j, i in enumerate(idx) if j != zero[0]]
    keep = keep[:k]
    nb_pct = ref.pct[keep]
    med = np.median(nb_pct, axis=0)
    out = []
    for j, f in enumerate(ESCALATION_FEATURES):
        out.append({"feature": f, "flow_pct": float(q_pct[j]), "neighbour_median_pct": float(med[j]),
                    "gap": float(q_pct[j] - med[j])})
    out.sort(key=lambda r: -abs(r["gap"]))
    return out


def score_flow(ref: ReferenceV2, features: Dict[str, float]) -> dict:
    """The full v2 signal for one flow. Features only; no label is read."""
    q = query_percentiles(ref, features)
    dist = {k: float(knn_distance_loo(ref, q, k)[0]) for k in (RADIUS_K, PRIMARY_K) + SENSITIVITY_KS}
    return {
        "reference": ref.label,
        "reference_size": ref.n,
        "knn_distance": {str(k): v for k, v in dist.items()},
        "isolation_percentile": isolation_percentile(ref, dist[PRIMARY_K], PRIMARY_K),
        "isolation_percentile_sensitivity": {str(k): isolation_percentile(ref, dist[k], k) for k in SENSITIVITY_KS},
        "radius": ref.radius,
        "radius_count": radius_count(ref, q),
        "per_feature": per_feature_isolation(ref, q, PRIMARY_K),
    }
