"""Reference-window grounding for the deployment-realistic arms.

Parts 1 to 2 of the brief, fixed before any arm ran:

  - one population: the CSV pool, restricted to a reference window of days. Every number
    A5 reads (profile matches, closeness, neighbourhood count) is computed from the SAME
    benign window, and the population size R is stated once, in both halves of each sentence.
  - the real count is shown. Nothing is replaced by 0 when a flow is ungrounded. The
    ungrounded flag (count < MIN_NEIGHBOURHOOD_SIZE) still drives the cap, on the score.
  - the flow's own benign copy is removed (leave-one-out) from counts and matches.
  - profile features are mapped to CSV columns. Each is reported as direct, derived,
    not loaded in the cached pool, or per-source. A prediction that cannot be checked is
    said so in the text A5 reads; it is never silently dropped.

Benign-only: the window is built from BENIGN rows and asserted first, as in fitting.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agents.escalation_grounding import (
    BAND_HIGH_MULT,
    BAND_LOW_MULT,
    ESCALATION_FEATURES,
    MIN_NEIGHBOURHOOD_SIZE,
)
from agents.grounding import HypothesisSupport

BENIGN = "BENIGN"
TIER1_PREFIX = "trigger_"

# ------------------------------------------------------------------ profile mapping
# name -> (kind, detail). kinds:
#   direct       a CSV column in the cached pool, used as-is
#   derived      computed from cached CSV columns (definition given)
#   not_loaded   maps to CSV columns that the cached pool does not carry (not loaded here)
#   per_source   needs a per-source window (dataplane/src_table.py); not computable per flow
PROFILE_MAP: Dict[str, Tuple[str, object]] = {
    "flow_duration": ("direct", "Flow Duration"),
    "flow_iat_mean": ("direct", "Flow IAT Mean"),
    "flow_bytes_per_sec": ("direct", "Flow Bytes/s"),
    "flow_pkts_per_sec": ("direct", "Flow Packets/s"),
    "init_win_bytes_bwd": ("direct", "Init_Win_bytes_backward"),
    "init_win_bytes_fwd": ("direct", "Init_Win_bytes_forward"),
    "bwd_pkt_len_mean": ("direct", "Bwd Packet Length Mean"),
    "pkt_len_range": ("derived", "Max Packet Length - Min Packet Length"),
    "syn_ratio": ("derived", "SYN Flag Count / (Total Fwd Packets + Total Backward Packets)"),
    "no_response_flag": ("derived", "1 if Total Backward Packets == 0 else 0"),
    "fwd_pkt_len_mean": ("not_loaded", "Fwd Packet Length Mean"),
    "flow_iat_max": ("not_loaded", "Flow IAT Max"),
    "flow_iat_min": ("not_loaded", "Flow IAT Min"),
    "down_up_pkt_ratio": ("not_loaded", "Down/Up Ratio"),
    "bwd_fwd_byte_ratio": ("not_loaded", "Total Length of Bwd Packets / Total Length of Fwd Packets"),
    "rst_ratio": ("not_loaded", "RST Flag Count / packets"),
    "flow_iat_regularity": ("not_loaded", "Flow IAT Std / Flow IAT Mean"),
    "flows_per_src": ("per_source", "flows from this source over the selector's window"),
    "distinct_dst_ips_per_src": ("per_source", "distinct destinations from this source over the window"),
    "distinct_dst_ports_per_src": ("per_source", "distinct destination ports from this source over the window"),
    "syn_without_synack_count": ("per_source", "SYNs from this source without a SYN-ACK over the window"),
}

#: Columns the window needs from the pool (beyond the ten escalation features).
EXTRA_POOL_COLUMNS = (
    "Flow Duration", "Flow IAT Mean", "Flow Bytes/s", "Flow Packets/s",
    "Init_Win_bytes_backward", "Init_Win_bytes_forward", "Bwd Packet Length Mean",
    "Max Packet Length", "Min Packet Length", "SYN Flag Count",
    "Total Fwd Packets", "Total Backward Packets", "weekday_idx", "day",
)


def _bare(feature: str) -> str:
    return feature[len(TIER1_PREFIX):] if feature.startswith(TIER1_PREFIX) else feature


def profile_status(hyp_features: Iterable[str]) -> Tuple[bool, List[str]]:
    """(evaluable, unevaluable_feature_names). A profile is evaluable only if every
    feature it names maps to a direct or derived CSV quantity."""
    bad = []
    for f in hyp_features:
        kind = PROFILE_MAP.get(_bare(f), ("unknown", None))[0]
        if kind not in ("direct", "derived"):
            bad.append(f)
    return (not bad), bad


def window_values(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Mapped quantities for every window row, as numpy arrays."""
    out = {}
    for name, (kind, detail) in PROFILE_MAP.items():
        if kind == "direct":
            out[name] = df[detail].to_numpy(dtype=np.float64)
        elif name == "pkt_len_range":
            out[name] = df["Max Packet Length"].to_numpy(dtype=np.float64) - df["Min Packet Length"].to_numpy(dtype=np.float64)
        elif name == "syn_ratio":
            pk = df["Total Fwd Packets"].to_numpy(dtype=np.float64) + df["Total Backward Packets"].to_numpy(dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[name] = np.where(pk > 0, df["SYN Flag Count"].to_numpy(dtype=np.float64) / pk, np.nan)
        elif name == "no_response_flag":
            out[name] = (df["Total Backward Packets"].to_numpy(dtype=np.float64) == 0).astype(np.float64)
    return out


def flow_values(features: Dict[str, float]) -> Dict[str, float]:
    """The same mapped quantities for one flow, from its CICFlowMeter feature dict."""
    f = {k: float(v) for k, v in features.items() if isinstance(v, (int, float, np.floating, np.integer))}
    out = {}
    for name, (kind, detail) in PROFILE_MAP.items():
        if kind == "direct" and detail in f:
            out[name] = f[detail]
        elif name == "pkt_len_range" and "Max Packet Length" in f and "Min Packet Length" in f:
            out[name] = f["Max Packet Length"] - f["Min Packet Length"]
        elif name == "syn_ratio" and "SYN Flag Count" in f:
            pk = f.get("Total Fwd Packets", 0.0) + f.get("Total Backward Packets", 0.0)
            out[name] = f["SYN Flag Count"] / pk if pk > 0 else float("nan")
        elif name == "no_response_flag" and "Total Backward Packets" in f:
            out[name] = 1.0 if f["Total Backward Packets"] == 0 else 0.0
    return out


# ------------------------------------------------------------------ reference window
@dataclass
class WindowRef:
    label: str
    weekdays: Tuple[int, ...]
    esc: np.ndarray  # (N, 10) benign escalation values, ESCALATION_FEATURES order
    mapped: Dict[str, np.ndarray]
    frame: pd.DataFrame  # benign window rows (for the evidence module)

    @property
    def size(self) -> int:
        return int(self.esc.shape[0])


def build_window(pool: pd.DataFrame, weekdays: Sequence[int], window_name: str) -> WindowRef:
    """Benign rows from the named weekdays only (0 Monday, 1 Tuesday, ...). Asserted."""
    sub = pool.loc[pool["weekday_idx"].isin(list(weekdays))]
    benign = sub.loc[sub["Label"] == BENIGN]
    # the window is benign-only by construction: no non-benign row may enter the reference
    if (benign["Label"] != BENIGN).any():
        raise ValueError("non-benign row in the reference window")
    if len(benign) == 0:
        raise ValueError(f"empty reference window {window_name!r}")
    esc = benign[list(ESCALATION_FEATURES)].to_numpy(dtype=np.float64)
    return WindowRef(label=window_name, weekdays=tuple(int(w) for w in weekdays), esc=esc,
                     mapped=window_values(benign), frame=benign.reset_index(drop=True))


def assert_outside(window: WindowRef, eval_weekdays: Iterable[int]) -> None:
    """Every evaluated record must come from outside the reference window."""
    overlap = set(window.weekdays) & set(int(w) for w in eval_weekdays)
    if overlap:
        raise AssertionError(f"evaluated records from reference days {sorted(overlap)} in window {window.label}")


# ------------------------------------------------------------------ the count, real and LOO
def neighbourhood(window: WindowRef, features: Dict[str, float]) -> dict:
    """Band count on the ten escalation features, shown in full.
    count_all: every window row inside the band on all ten features (as before).
    identical: window rows identical to the flow on all ten features (its own copy, if benign).
    count_loo: count_all with exactly one identical copy removed (leave-one-out).
    ungrounded: count_loo < MIN_NEIGHBOURHOOD_SIZE. The cap uses this flag; nothing else."""
    x = np.array([float(features[f]) for f in ESCALATION_FEATURES], dtype=np.float64)
    mask = np.ones(window.size, dtype=bool)
    for j, v in enumerate(x):
        col = window.esc[:, j]
        if v == 0:
            mask &= col == 0
        else:
            lo, hi = sorted([v * BAND_LOW_MULT, v * BAND_HIGH_MULT])
            mask &= (col >= lo) & (col <= hi)
    identical_mask = np.all(window.esc == x, axis=1)
    identical = int(identical_mask.sum())
    count_all = int(mask.sum())
    count_loo = count_all - (1 if identical > 0 else 0)
    r_loo = window.size - (1 if identical > 0 else 0)
    return {"count_all": count_all, "identical": identical, "count_loo": count_loo,
            "population_R": r_loo, "ungrounded": count_loo < MIN_NEIGHBOURHOOD_SIZE}


# ------------------------------------------------------------------ profile evaluation
def evaluate_profile(window: WindowRef, flow_feats: Dict[str, float], flow_vals: Dict[str, float],
                     hyp_features: Sequence[str], predicted: Dict[str, Tuple[Optional[float], Optional[float]]],
                     self_identical: np.ndarray) -> dict:
    """Matches of one hypothesis's predicted ranges over the window, leave-one-out, and
    the closeness of those matches to the flow's own values on the mapped features.
    Returns status 'unevaluable' with the feature names if any feature cannot be mapped."""
    ok, bad = profile_status(hyp_features)
    if not ok:
        return {"status": "unevaluable", "unevaluable": bad}
    n = window.size
    mask = np.ones(n, dtype=bool)
    checked: List[str] = []
    for f in hyp_features:
        b = _bare(f)
        lo, hi = predicted.get(f, (None, None))
        col = window.mapped[b]
        m = np.isfinite(col)
        if lo is not None:
            m &= col >= lo
        if hi is not None:
            m &= col <= hi
        mask &= m
        checked.append(f)
    # close: among matches, within [0.5x, 2x] on each mapped feature, using the flow's own value
    close = mask.copy()
    for f in checked:
        b = _bare(f)
        v = flow_vals.get(b)
        if v is None or not np.isfinite(v):
            continue
        col = window.mapped[b]
        if v == 0:
            close &= col == 0
        else:
            lo2, hi2 = sorted([v * BAND_LOW_MULT, v * BAND_HIGH_MULT])
            close &= (col >= lo2) & (col <= hi2)
    n_match = int(mask.sum())
    n_close = int(close.sum())
    if np.any(self_identical & mask):  # leave-one-out: the flow's own copy is removed from both
        n_match -= 1
        n_close -= int(np.any(self_identical & close))
    return {"status": "evaluated", "n_match": n_match, "n_close": n_close, "checked": checked}


def sentence_profile(R: int, ev: dict, hyp_id: str) -> str:
    if ev["status"] == "unevaluable":
        return (f"{hyp_id}: this prediction cannot be checked against this reference. It names "
                f"{', '.join(ev['unevaluable'])}, which this reference cannot evaluate per flow. Its support is unknown here.")
    names = ", ".join(ev["checked"]) if ev["checked"] else "no mapped feature"
    return (f"{hyp_id}: of the {R} benign flows in the reference window, {ev['n_match']} match this predicted profile, "
            f"and {ev['n_close']} of those are within 2x of this flow's own values on {names}.")


def sentence_neighbourhood(nb: dict) -> str:
    return (f"Of the {nb['population_R']} benign flows in the reference window, {nb['count_loo']} are within "
            f"[0.5x, 2x] of this flow's values on all ten escalation features"
            + (f" (the flow's own copy is removed; {nb['identical']} identical benign flow(s) were present, one removed)"
               if nb["identical"] else "")
            + f". Count shown in full: {nb['count_loo']}.")


def hypothesis_support(window: WindowRef, flow_feats: Dict[str, float], hyps, nb: dict) -> Tuple[Dict[str, HypothesisSupport], Dict[str, str]]:
    """HypothesisSupport objects (for the zero-support cap) and the text A5 reads, per hypothesis.
    The population R is the same as in the neighbourhood sentence."""
    flow_vals = flow_values(flow_feats)
    x = np.array([float(flow_feats[f]) for f in ESCALATION_FEATURES], dtype=np.float64)
    self_identical = np.all(window.esc == x, axis=1)
    supports, lines = {}, {}
    R = nb["population_R"]
    for h in hyps:
        hf = [p.feature for p in h.predicted_feature_profile]
        predicted = {p.feature: (p.expected_min, p.expected_max) for p in h.predicted_feature_profile}
        ev = evaluate_profile(window, flow_feats, flow_vals, hf, predicted, self_identical)
        if ev["status"] == "evaluated":
            supports[h.hypothesis_id] = HypothesisSupport(
                hypothesis_id=h.hypothesis_id, matching_profile_count=ev["n_match"],
                close_to_observed_count=ev["n_close"], checked_features=ev["checked"],
                benign_population_size=R)
        else:
            # not evaluable: matching count is undefined, so the zero-support cap cannot fire on it
            supports[h.hypothesis_id] = HypothesisSupport(
                hypothesis_id=h.hypothesis_id, matching_profile_count=-1,
                close_to_observed_count=None, checked_features=[], benign_population_size=R)
        lines[h.hypothesis_id] = sentence_profile(R, ev, h.hypothesis_id)
    return supports, lines


def describe_mapping(features: Iterable[str]) -> List[dict]:
    """Report of how each profile feature is mapped (for the write-up)."""
    out = []
    for f in features:
        b = _bare(f)
        kind, detail = PROFILE_MAP.get(b, ("unknown", None))
        out.append({"feature": f, "kind": kind, "csv": detail if kind != "per_source" else None})
    return out
