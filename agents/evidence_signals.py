"""Four benign-only evidence signals, given to the agents beside the existing
neighbourhood count (which is kept as a fifth input).

Benign-only is the constraint that makes this deployable: the reference is built
from BENIGN rows and nothing else. Calibration is "watch the network run normally";
it needs no labels and no attack examples. The one known assumption is that a
network already compromised during the baseline window will learn the attack as
normal. The rule-B refit experiment showed that failure directly.

Signals (all computed from benign rows, for one query flow):
  1. per-feature percentile of the flow's value within benign traffic
  2. nearest benign neighbours, with their values and distances
  3. source history: the same source IP's benign flows in earlier 5-minute buckets
  4. directional deviation: above/below the benign median, and extreme tails

Leakage rule (leave-one-out). A benign flow is itself a row in the benign reference,
so an unfiltered count or neighbour search includes the flow itself. Attacks are never
in the reference, so that copy would reveal the label. Every signal here removes exactly
one identical reference row (on the ten escalation features) when one exists, which is
the leave-one-out a deployed controller would face for a new flow. Other identical rows
are kept: they are genuine benign twins. The same rule is used for the agents and for
the logistic control. Percentiles include the query's own copy when present; at 1 row in
2,273,097 that effect is negligible and is not corrected.

Source history is computed from benign rows over earlier time only, never the whole
window. dataplane/src_table.py was not used: it is streaming state, and replaying
the benign pool through it in time order is not equivalent to a per-bucket query.
The pooled CSV cache has no byte-count column, so history reports flows, distinct
destination ports and packets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from agents.escalation_grounding import ESCALATION_FEATURES

BENIGN = "BENIGN"  # same string as eval.escalation_data.BENIGN_LABEL; kept local so this module imports without the CSV pool

NEAREST_K = 5
NEIGHBOUR_QUERY_K = 60  # extra candidates so identical rows can be dropped and K kept
SOURCE_BUCKET_US = 5 * 60 * 1_000_000  # 5-minute buckets (first_ts is microseconds)
SOURCE_BUCKETS = 6  # 30 minutes of history, strictly before the flow's own time
TAIL_LOW_PCT = 1.0
TAIL_HIGH_PCT = 99.0
IDENTICAL_TOL = 1e-12

# Escalation features as the agents see them, in a fixed order.
FEATURES = tuple(ESCALATION_FEATURES)


FIXED_POOL_COLUMNS = ("Label", "first_ts", "Source IP", "Total Fwd Packets", "Total Backward Packets", "Destination Port")


def load_pool_slim(extra_columns=()) -> pd.DataFrame:
    """Reads only the columns this module needs from the pooled CSV cache. The full
    cache is 36 columns across 2.8M rows, which does not fit alongside other work on
    this machine. `extra_columns` adds the blind flows' feature names (for matching)."""
    import pyarrow.parquet as pq

    from eval.escalation_data import CACHE_PATH

    names = set(pq.read_schema(CACHE_PATH).names)
    want = set(FEATURES) | set(FIXED_POOL_COLUMNS) | set(extra_columns)
    return pd.read_parquet(CACHE_PATH, columns=sorted(want & names))


def assert_benign_only(frame: pd.DataFrame) -> None:
    """Mirrors dataplane/fitting.py: the first thing that happens to the reference
    is a check that no attack row reached it. Raises rather than filtering silently."""
    labels = frame["Label"]
    bad = int((labels != BENIGN).sum())
    if bad:
        raise ValueError(f"{bad} non-benign rows reached the evidence reference; evidence must be benign-only")


def _transform(x: np.ndarray) -> np.ndarray:
    # sign-preserving log, because several features are negative or zero
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class BenignReference:
    raw: np.ndarray  # (N, 10) benign values, FEATURES order
    sorted_cols: List[np.ndarray]  # per-feature sorted values, for percentiles
    medians: np.ndarray  # (10,)
    scale: np.ndarray  # (10,) std of the transformed values, for distances
    nn: NearestNeighbors
    src_times: Dict[str, np.ndarray] = field(default_factory=dict)  # Source IP -> sorted first_ts
    src_packets: Dict[str, np.ndarray] = field(default_factory=dict)
    src_ports: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.raw.shape[0])


def build_reference(frame: pd.DataFrame) -> BenignReference:
    """Build from benign rows only. The caller passes the pool; this function
    filters to BENIGN, then asserts the filtered frame is benign-only."""
    benign = frame.loc[frame["Label"] == BENIGN]
    assert_benign_only(benign)
    if len(benign) == 0:
        raise ValueError("no benign rows in the evidence reference")
    raw = benign[list(FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("non-finite values in the benign reference")
    sorted_cols = [np.sort(raw[:, j]) for j in range(raw.shape[1])]
    medians = np.median(raw, axis=0)
    trans = _transform(raw)
    scale = trans.std(axis=0)
    scale[scale == 0] = 1.0
    nn = NearestNeighbors(algorithm="kd_tree")
    nn.fit(trans / scale)

    ref = BenignReference(raw=raw, sorted_cols=sorted_cols, medians=medians, scale=scale, nn=nn)
    # per-source index over benign rows, sorted by time, for source history
    src = benign[["Source IP", "first_ts", "Total Fwd Packets", "Total Backward Packets", "Destination Port"]]
    for ip, g in src.sort_values("first_ts").groupby("Source IP", sort=False):
        ref.src_times[str(ip)] = g["first_ts"].to_numpy(dtype=np.int64)
        ref.src_packets[str(ip)] = (g["Total Fwd Packets"] + g["Total Backward Packets"]).to_numpy(dtype=np.float64)
        ref.src_ports[str(ip)] = g["Destination Port"].to_numpy(dtype=np.int64)
    return ref


# ----------------------------------------------------------------- query
@dataclass
class EvidenceBundle:
    values: Dict[str, float]
    percentiles: Dict[str, float]
    direction: Dict[str, str]
    neighbours: List[dict]  # value rows, each with distance
    count_all: int  # same band rule as escalation_grounding (includes identical rows)
    identical: int  # benign rows identical to the flow on all ten features
    count_excl: int  # count_all minus identical
    history: Optional[List[dict]]  # None when the source is unknown
    history_note: str

    def as_signals(self) -> dict:
        return {"percentiles": self.percentiles, "direction": self.direction, "neighbours": self.neighbours,
                "count_all": self.count_all, "count_excl": self.count_excl, "identical": self.identical,
                "history": self.history, "history_note": self.history_note}


def _vector(features: Dict[str, float]) -> np.ndarray:
    # the query takes the flow's features only; a Label key in the dict is ignored
    return np.array([float(features[f]) for f in FEATURES], dtype=np.float64)


def percentiles(ref: BenignReference, x: np.ndarray) -> np.ndarray:
    return np.array([100.0 * np.searchsorted(ref.sorted_cols[j], x[j], side="right") / ref.n
                     for j in range(len(FEATURES))])


def count_band(ref: BenignReference, x: np.ndarray) -> Tuple[int, int]:
    """Same rule as agents/escalation_grounding.compute_feature_neighbourhood:
    within [0.5x, 2x] on every feature, or == 0 when the flow's value is 0."""
    mask = np.ones(ref.n, dtype=bool)
    for j, v in enumerate(x):
        col = ref.raw[:, j]
        if v == 0:
            mask &= col == 0
        else:
            lo, hi = sorted([v * 0.5, v * 2.0])
            mask &= (col >= lo) & (col <= hi)
    count_all = int(mask.sum())
    identical = int(np.all(np.abs(ref.raw - x) <= IDENTICAL_TOL, axis=1).sum())
    return count_all, identical


def nearest(ref: BenignReference, x: np.ndarray) -> List[dict]:
    q = (_transform(x[None, :]) / ref.scale)
    dist, ind = ref.nn.kneighbors(q, n_neighbors=min(NEIGHBOUR_QUERY_K, ref.n))
    out = []
    dropped_self = False
    for d, i in zip(dist[0], ind[0]):
        if d <= IDENTICAL_TOL and not dropped_self:
            dropped_self = True  # leave-one-out: one identical copy (the flow itself) is removed
            continue
        out.append({"distance": float(d), "values": {f: float(ref.raw[i, j]) for j, f in enumerate(FEATURES)}})
        if len(out) == NEAREST_K:
            break
    return out


def source_history(ref: BenignReference, source_ip: str, t0_us: int) -> List[dict]:
    """Benign flows from this source in each of the SOURCE_BUCKETS 5-minute buckets
    strictly before t0. Bucket 1 is the most recent one. Nothing at or after t0 is read."""
    times = ref.src_times.get(source_ip)
    out = []
    for k in range(1, SOURCE_BUCKETS + 1):
        lo = t0_us - k * SOURCE_BUCKET_US
        hi = t0_us - (k - 1) * SOURCE_BUCKET_US
        if times is None:
            out.append({"bucket": k, "flows": 0, "distinct_dst_ports": 0, "packets": 0.0})
            continue
        i0 = int(np.searchsorted(times, lo, side="left"))
        i1 = int(np.searchsorted(times, hi, side="left"))
        out.append({
            "bucket": k,
            "flows": i1 - i0,
            "distinct_dst_ports": int(len(np.unique(ref.src_ports[source_ip][i0:i1]))) if i1 > i0 else 0,
            "packets": float(ref.src_packets[source_ip][i0:i1].sum()) if i1 > i0 else 0.0,
        })
    return out


def locate_source(pool: pd.DataFrame, features: Dict[str, float], match_cols: List[str]) -> Tuple[Optional[Tuple[str, int]], str]:
    """Recovers a blind flow's own source IP and time by matching its feature
    vector against the pool. Reads only Source IP and first_ts of matching rows.
    Label is never read. Returns ((ip, t), 'unique') or (None, 'ambiguous'|'none')."""
    x = np.array([float(features[c]) for c in match_cols])
    m = np.all(pool[match_cols].to_numpy(dtype=np.float64) == x, axis=1)
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        return None, "none"
    pairs = set(zip(pool["Source IP"].to_numpy()[idx].astype(str), pool["first_ts"].to_numpy()[idx].astype(np.int64)))
    if len(pairs) != 1:
        return None, "ambiguous"
    ip, t = next(iter(pairs))
    return (str(ip), int(t)), "unique"


def bundle(ref: BenignReference, features: Dict[str, float], source: Optional[Tuple[str, int]]) -> EvidenceBundle:
    x = _vector(features)
    pct = percentiles(ref, x)
    direction = {}
    for j, f in enumerate(FEATURES):
        if pct[j] >= TAIL_HIGH_PCT:
            d = "extreme high (top 1% of benign)"
        elif pct[j] <= TAIL_LOW_PCT:
            d = "extreme low (bottom 1% of benign)"
        elif x[j] > ref.medians[j]:
            d = "above benign median"
        elif x[j] < ref.medians[j]:
            d = "below benign median"
        else:
            d = "at benign median"
        direction[f] = d
    count_all, identical = count_band(ref, x)
    if source is None:
        history, note = None, "source IP or time unavailable for this flow"
    else:
        history, note = source_history(ref, source[0], source[1]), "benign flows from this source IP, earlier buckets only"
    return EvidenceBundle(
        values={f: float(x[j]) for j, f in enumerate(FEATURES)},
        percentiles={f: float(pct[j]) for j, f in enumerate(FEATURES)},
        direction=direction,
        neighbours=nearest(ref, x),
        count_all=count_all, identical=identical, count_excl=count_all - min(identical, 1),
        history=history, history_note=note,
    )


def render_evidence_block(b: EvidenceBundle) -> str:
    """Text the agents receive. Gives the signals; gives no rule for combining
    them and no threshold."""
    L = []
    L.append("Benign-only evidence for this flow. Every figure comes from real benign traffic on this network. "
             "No rule for combining these signals is given.\n")
    L.append("1. Percentile of each feature within benign traffic (0 = lowest benign value, 100 = highest):")
    for f in FEATURES:
        L.append(f"   - {f}: percentile {b.percentiles[f]:.2f} (flow value {b.values[f]:g})")
    L.append("\n2. Direction against benign:")
    for f in FEATURES:
        L.append(f"   - {f}: {b.direction[f]}")
    L.append(f"\n3. Nearest benign flows (distance in standardised feature units; the flow's own reference copy is removed if present). "
             f"This flow's values first, then each neighbour's:")
    L.append("   this flow: " + ", ".join(f"{f}={b.values[f]:g}" for f in FEATURES))
    for i, nb in enumerate(b.neighbours, 1):
        L.append(f"   neighbour {i} (distance {nb['distance']:.3f}): "
                 + ", ".join(f"{f}={nb['values'][f]:g}" for f in FEATURES))
    if not b.neighbours:
        L.append("   (no non-identical benign neighbours found)")
    L.append(f"\n4. Neighbourhood count within [0.5x, 2x] on all ten features: {b.count_all} benign flows; "
             f"{b.count_excl} after leave-one-out (" + (f"{b.identical} identical benign flow(s) present, one removed)." if b.identical else "no identical benign flow present, none removed).") )
    L.append(f"\n5. Source history ({b.history_note}), 5-minute buckets before this flow, most recent first:")
    if b.history is None:
        L.append("   unavailable")
    else:
        for h in b.history:
            start, end = -5 * h["bucket"], -5 * (h["bucket"] - 1)
            L.append(f"   {start} to {end} min: {h['flows']} flows, {h['distinct_dst_ports']} distinct destination ports, "
                     f"{h['packets']:g} packets")
    return "\n".join(L) + "\n"
