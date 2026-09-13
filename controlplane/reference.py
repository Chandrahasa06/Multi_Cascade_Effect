"""Benign reference distribution: what does normal actually look like on
this network, in numbers an agent can be handed directly.

Built from Monday's benign-only PCAP simulation (already-cached, full
population -- ``results/cache/monday_pcap__fidelity__adapter2__features4
.parquet``, 566,864 real flows, no new extraction needed) -- see
``dataplane/selector.py``'s Tier-1/per-source feature computation for
what these columns are.

SCOPE, DELIBERATE: this covers the selector's Tier-1/per-source feature
space (the same ~21 features shown to agents as ``trigger_``-prefixed
measurements -- see ``agents/prompts/render.py``), not the full 77-feature
CICFlowMeter space (``EscalationRecord.features``). Building a reference
distribution over the CICFlowMeter space would require running
``controlplane/extractor.py``'s full per-flow packet-based extraction
against Monday's ~566,864 benign flows -- at the per-flow cost observed
extracting the 200-record escalation sample (STATUS.md: ~71s packet
recollection + CICFlowMeter time for 200 flows), that's multiple days of
compute, not attempted here. If CICFlowMeter-space grounding is wanted
later, a large random sample (a few thousand flows) is the tractable
path, not the full population.

Also deliberately asymmetric: only the high side is covered
(p50/p90/p99/p99.5/max), not low-tail percentiles -- several real
trigger reasons are "low" direction (e.g. an unusually short
flow_iat_mean), and those get median/p90 for context but no equally
rigorous "below every benign flow observed" statement. Flagged here
rather than silently implied to be symmetric.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: the already-cached, full-population Monday benign feature parquet --
#: see this module's docstring for provenance.
DEFAULT_REFERENCE_PARQUET = Path("results/cache/monday_pcap__fidelity__adapter2__features4.parquet")
#: precomputed percentiles, cached once (percentile computation over
#: 566,864 rows x 21 columns is cheap, but re-reading+recomputing on
#: every pipeline run is still wasted work across hundreds of calls).
REFERENCE_CACHE_PATH = Path("results/cache/benign_reference_distribution.json")

_NON_FEATURE_COLUMNS = frozenset({"label", "first_ts", "last_ts", "closure_reason"})


@dataclass(frozen=True)
class FeatureReference:
    """One feature's benign reference stats -- p50/p90/p99/p99.5/max plus
    how many benign flows actually had a defined value for it (some
    Tier-1 features are undefined for flows below a minimum packet count,
    e.g. bwd_pkt_len_mean when there were no backward packets at all;
    ``n`` is the count of flows the percentiles were computed over, not
    the total flow count)."""

    feature: str
    p50: float
    p90: float
    p99: float
    p995: float
    max: float
    n: int

    def to_dict(self) -> dict:
        return {
            "feature": self.feature, "p50": self.p50, "p90": self.p90,
            "p99": self.p99, "p995": self.p995, "max": self.max, "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureReference":
        return cls(
            feature=d["feature"], p50=d["p50"], p90=d["p90"], p99=d["p99"],
            p995=d["p995"], max=d["max"], n=d["n"],
        )


def reference_from_dataframe(df: pd.DataFrame) -> Dict[str, FeatureReference]:
    """Pure computation, no disk I/O -- split out from
    ``build_reference_distribution`` so it's directly unit-testable
    against a small synthetic DataFrame instead of requiring the real
    566,864-row parquet on disk."""
    feature_cols = [c for c in df.columns if not c.endswith("__low_confidence") and c not in _NON_FEATURE_COLUMNS]
    out: Dict[str, FeatureReference] = {}
    for col in feature_cols:
        values = df[col].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        out[col] = FeatureReference(
            feature=col,
            p50=float(np.percentile(values, 50)),
            p90=float(np.percentile(values, 90)),
            p99=float(np.percentile(values, 99)),
            p995=float(np.percentile(values, 99.5)),
            max=float(values.max()),
            n=int(values.size),
        )
    return out


def build_reference_distribution(
    parquet_path: Path = DEFAULT_REFERENCE_PARQUET,
) -> Dict[str, FeatureReference]:
    return reference_from_dataframe(load_benign_dataframe(parquet_path))


@lru_cache(maxsize=4)
def load_benign_dataframe(parquet_path: Path = DEFAULT_REFERENCE_PARQUET) -> pd.DataFrame:
    """Cached in-process (lru_cache, keyed on path) -- a 566,864-row
    parquet has no business being re-read from disk on every one of the
    ~100+ calls a single pipeline run makes into this module."""
    df = pd.read_parquet(parquet_path)
    assert (df["label"] == "BENIGN").all(), (
        "reference distribution must be built from benign-only data -- "
        "found a non-BENIGN label in the source parquet"
    )
    return df


def save_reference_distribution(
    ref: Dict[str, FeatureReference], path: Path = REFERENCE_CACHE_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.to_dict() for k, v in ref.items()}))


def load_reference_distribution(path: Path = REFERENCE_CACHE_PATH) -> Dict[str, FeatureReference]:
    data = json.loads(path.read_text())
    return {k: FeatureReference.from_dict(v) for k, v in data.items()}


@lru_cache(maxsize=4)
def get_reference_distribution(
    cache_path: Path = REFERENCE_CACHE_PATH, parquet_path: Path = DEFAULT_REFERENCE_PARQUET
) -> Dict[str, FeatureReference]:
    """Load the cached reference distribution, building and caching it on
    first use (disk cache), and memoized in-process too (lru_cache) so
    the ~100+ calls a single pipeline run makes don't each re-read and
    re-parse the JSON. This is the entry point everything else in the
    agent pipeline should call -- never rebuild from the raw parquet per
    call."""
    if cache_path.exists():
        return load_reference_distribution(cache_path)
    ref = build_reference_distribution(parquet_path)
    save_reference_distribution(ref, cache_path)
    return ref


def render_reference_line(feature: str, observed_value: float, ref: Optional[FeatureReference]) -> str:
    """One line of empirical grounding for a single feature, e.g.:

    flows_per_src: this flow 4614 | benign median 226, p90 700, p99 1583,
        p99.5 1794, benign max 1993, n=566864 -> this flow exceeds every
        benign flow observed

    The verdict clause (last part) is the load-bearing fact: "exceeds the
    benign maximum across N observed flows" is empirical, not a threshold
    crossing dressed up as one.
    """
    if ref is None:
        return f"{feature}: this flow {observed_value:g} | no benign reference available for this feature"
    if observed_value > ref.max:
        verdict = f" -> this flow exceeds every benign flow observed (n={ref.n})"
    elif observed_value > ref.p995:
        verdict = f" -> this flow exceeds the 99.5th percentile of benign traffic (n={ref.n})"
    elif observed_value > ref.p99:
        verdict = f" -> this flow exceeds the 99th percentile of benign traffic (n={ref.n})"
    else:
        verdict = ""
    return (
        f"{feature}: this flow {observed_value:g} | benign median {ref.p50:g}, p90 {ref.p90:g}, "
        f"p99 {ref.p99:g}, p99.5 {ref.p995:g}, benign max {ref.max:g}, n={ref.n}{verdict}"
    )


# ---------- nearest-neighbour grounding ----------


@dataclass(frozen=True)
class NearestNeighbourResult:
    """Nearest-benign-flow grounding for one escalated flow.

    Computed only over the Tier-1 features this specific flow actually
    has an observed value for (its own trigger_reasons) -- the
    EscalationRecord never carries the full ~21-feature Tier-1 vector,
    only the features that crossed a threshold, so a true full-dimension
    nearest-neighbour search isn't possible from this data. Distance is
    normalised per feature by (p90 - p50) from the reference distribution
    (a robust scale estimate that doesn't require computing extra
    percentiles beyond what's already specified), so features are
    comparable despite very different raw units/ranges.
    """

    features_used: List[str]
    k_requested: int
    k_found: int
    nearest_distance: Optional[float]
    mean_k_distance: Optional[float]
    top_differing_features: List[Tuple[str, float]]  # (feature, ratio_to_nearest), sorted most-different first

    def to_dict(self) -> dict:
        return {
            "features_used": self.features_used,
            "k_requested": self.k_requested,
            "k_found": self.k_found,
            "nearest_distance": self.nearest_distance,
            "mean_k_distance": self.mean_k_distance,
            "top_differing_features": [[f, r] for f, r in self.top_differing_features],
        }


def find_nearest_benign_flows(
    observed: Dict[str, float],
    benign_df: pd.DataFrame,
    reference: Dict[str, FeatureReference],
    k: int = 10,
) -> NearestNeighbourResult:
    """``observed``: bare (unprefixed) Tier-1 feature name -> this flow's
    value, for whichever features are actually known (its trigger
    reasons). Only features present in both ``observed``, ``benign_df``,
    and ``reference`` are used."""
    features = [f for f in observed if f in benign_df.columns and f in reference]
    if not features:
        return NearestNeighbourResult(
            features_used=[], k_requested=k, k_found=0,
            nearest_distance=None, mean_k_distance=None, top_differing_features=[],
        )

    sub = benign_df[features].dropna()
    if sub.empty:
        return NearestNeighbourResult(
            features_used=features, k_requested=k, k_found=0,
            nearest_distance=None, mean_k_distance=None, top_differing_features=[],
        )

    # Scale by (max - p50), not (p90 - p50): several Tier-1 features are
    # heavily right-skewed/sparse (e.g. syn_without_synack_count has
    # p50=p90=0 -- zero for the vast majority of benign flows), so p90-p50
    # collapses to the 1e-9 floor while a real observed value of, say, 205
    # produces a normalised term of ~2e11 that swamps every other feature
    # in the Euclidean sum and both picks the wrong "nearest" flow and
    # corrupts top_differing_features. (max - p50) can't collapse the same
    # way unless the feature is a literal constant across all benign
    # flows, and even then stays anchored to a real observed range rather
    # than an arbitrary near-median slice. Confirmed live: this produced
    # distances in the hundreds of billions for several stop-point-2
    # PortScan records before the fix.
    scale = {f: max(reference[f].max - reference[f].p50, 1e-9) for f in features}
    sq_dist = np.zeros(len(sub))
    for f in features:
        sq_dist += ((sub[f].to_numpy(dtype=float) - observed[f]) / scale[f]) ** 2
    dist = np.sqrt(sq_dist)

    k_found = min(k, len(dist))
    nearest_order = np.argsort(dist)[:k_found]
    nearest_distances = dist[nearest_order]

    nearest_row = sub.iloc[nearest_order[0]]
    ratios: List[Tuple[str, float]] = []
    for f in features:
        benign_val = float(nearest_row[f])
        obs_val = observed[f]
        if benign_val == 0 and obs_val == 0:
            ratio = 1.0
        elif benign_val == 0 or obs_val == 0:
            ratio = float("inf")
        else:
            ratio = max(obs_val / benign_val, benign_val / obs_val)
        ratios.append((f, ratio))
    ratios.sort(key=lambda t: t[1], reverse=True)

    return NearestNeighbourResult(
        features_used=features,
        k_requested=k,
        k_found=k_found,
        nearest_distance=float(nearest_distances[0]),
        mean_k_distance=float(nearest_distances.mean()),
        top_differing_features=ratios,
    )


def render_nearest_neighbour_line(result: NearestNeighbourResult) -> str:
    if result.k_found == 0:
        return "  (no benign reference available for any feature this flow has an observed value for)"
    diffs = ", ".join(
        f"{ratio:.1f}x on {feature}" if ratio != float("inf") else f"undefined ratio (benign value is 0) on {feature}"
        for feature, ratio in result.top_differing_features[:3]
    )
    return (
        f"  nearest of {result.k_found} benign flows found (of {result.k_requested} requested) is at "
        f"normalised distance {result.nearest_distance:.2f} (mean over the {result.k_found} nearest: "
        f"{result.mean_k_distance:.2f}); the closest benign flow differs most by {diffs}. "
        f"(computed over: {', '.join(result.features_used)} -- only features this flow has an observed "
        "value for; a full-dimension comparison isn't possible from this data)"
    )
