"""Three-priority controller escalation policy from
``controller_rule_selection.pdf``, with the SpliDT model
(``dataplane/splidt_inference.py``) as Priority 1.

    ESCALATE(x) = RULE_MATCH(x) [Priority 1, SpliDT]
                  v UNCOMMON(x)  [Priority 2, benign-only signature table]
                  v SAMPLE(x)    [Priority 3, deterministic hash sampling]

Processing order per the PDF's "Data-plane processing order": for each
flow, in arrival order, check Priority 1 first, then 2, then 3 — the
first one that fires is THE reason (mutually exclusive by construction,
matching the PDF's per-flow reported bit: one digest per flow, never
more). Only escalate if the meter still has budget; a flow that matched
a priority but arrived after its budget filled is forwarded normally,
un-reported.

=== Priority 1 — SpliDT, terminal only ===

Only the terminal-leaf variant is implemented. The "early" variant from
the original (packet-data) task doesn't apply here: every subtree sees
the SAME whole-flow CSV feature vector (see splidt_inference.py's
docstring — there is no sliding window on this path), so there is no
meaningfully earlier decision to make. Escalates when the model's final
leaf predicts anything other than "Benign".

**Priority 1 is supervised inference. It carries no zero-day property —
the model was trained on labeled attack classes and can only ever name
one of its own 10 training classes.**

=== Priority 2 — uncommon signature (benign-only K) ===

Reuses eval/generalization_experiments.py's actual binning/signature
functions (``compute_bin_edges``, ``signatures_for``, ``build_K`` —
imported, not reimplemented or modified) but feeds them a DIFFERENT
source than that module's own Experiment 2: this project's simulated
EVAL-mode parquet cache has no Flow-ID/row join key back to the raw CSV
rows Priority 1 uses (checked: eval/simulate.py's per-flow records carry
only `n_source_rows`/`first_ts`/`last_ts`, and row order there is
CLOSURE order, not file order — see its own comment — so there's no
honest 1:1 alignment back to this module's row-indexed flows without a
fragile heuristic join). Recomputing the same 5 signature features
directly from the raw CSV keeps Priority 1/2/3 operating on ONE
consistent, row-aligned population.

The recomputation is a **simplified, clearly-flagged stand-in** for
dataplane/src_table.py's real per-source state (rotating 10s buckets,
HyperLogLog cardinality): `flows_per_src` / `distinct_dst_ports_per_src`
/ `syn_without_synack_count` here are plain per-(day, Source IP)
aggregates over that whole day, not a live rolling window. `bwd_pkt_len_mean`
and `pkt_len_range` are exact (directly available per-flow CSV columns,
no simplification needed there).

K is asserted benign-only at construction — the ONLY component in this
policy carrying a zero-day claim; Priority 1 is supervised, Priority 3
guarantees nothing (see below).

=== Priority 3 — deterministic sampling ===

``SAMPLE(x) = 1[H(flowID) mod 10000 < tau]``, a STABLE hash (Python's
built-in ``hash()`` is salted per-process by default — PYTHONHASHSEED —
and would silently disagree across runs/processes; MD5 is used here
instead, truncated to an integer). Decided once, from the flow's own
identifier, not per packet — deterministic across any number of
re-evaluations or separate processes.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from eval.generalization_experiments import build_K, compute_bin_edges, signatures_for

PRIORITY2_FEATURES: Tuple[str, ...] = (
    "flows_per_src",
    "distinct_dst_ports_per_src",
    "syn_without_synack_count",
    "bwd_pkt_len_mean",
    "pkt_len_range",
)

BENIGN_CLASS = "Benign"


# --------------------------------------------------------------------- #
# Priority 1 — SpliDT (terminal leaf)
# --------------------------------------------------------------------- #

def priority1_escalate(predicted_class: np.ndarray) -> np.ndarray:
    """Escalate iff the model's terminal leaf predicts non-Benign."""
    return np.asarray(predicted_class) != BENIGN_CLASS


# --------------------------------------------------------------------- #
# Priority 2 — uncommon signature, benign-only K
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class SignatureTable:
    bin_edges: Dict[str, np.ndarray]
    common_signatures: FrozenSet[tuple]
    features: Tuple[str, ...]
    floor: int
    fit_n: int


def compute_priority2_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the 5 Experiment-2 signature columns to `df`, per-(day,
    Source IP) aggregates for the 3 per-source ones (see module
    docstring: a simplified stand-in for the project's real rolling
    per-source state) and direct per-flow values for the other 2."""
    out = df.copy()
    group_keys = ["day", "_p2_source_ip"]
    grouped = out.groupby(group_keys, sort=False)
    out["flows_per_src"] = grouped["_p2_source_ip"].transform("size").astype(float)
    out["distinct_dst_ports_per_src"] = grouped["Dst Port"].transform("nunique").astype(float)
    out["syn_without_synack_count"] = grouped["_p2_unanswered_syn"].transform("sum").astype(float)
    out["bwd_pkt_len_mean"] = out["_p2_bwd_pkt_len_mean"]
    out["pkt_len_range"] = out["_p2_pkt_len_range"]
    return out


def build_signature_table(
    benign_fit_df: pd.DataFrame,
    *,
    n_bins: int = 6,
    floor: int = 5,
    features: Sequence[str] = PRIORITY2_FEATURES,
    label_col: str = "Label",
) -> SignatureTable:
    """K built from BENIGN flows only — asserted here as the literal
    first check, exactly as dataplane/fitting.py does for its own
    threshold fitting. This is the only component in the whole policy
    carrying a zero-day claim."""
    assert (benign_fit_df[label_col] == BENIGN_CLASS).all(), (
        "build_signature_table received a non-benign label -- Priority 2's K "
        "table must be fit on benign traffic only, or its zero-day property "
        "collapses into a signature matcher for known attacks."
    )
    edges = compute_bin_edges(benign_fit_df, list(features), n_bins)
    sigs = signatures_for(benign_fit_df, list(features), edges)
    common = build_K(sigs, floor)
    return SignatureTable(
        bin_edges=edges, common_signatures=frozenset(common),
        features=tuple(features), floor=floor, fit_n=len(benign_fit_df),
    )


def priority2_escalate(df: pd.DataFrame, table: SignatureTable) -> np.ndarray:
    sigs = signatures_for(df, list(table.features), table.bin_edges)
    return np.array([s not in table.common_signatures for s in sigs], dtype=bool)


def check_bin_tie_saturation(benign_fit_df: pd.DataFrame, table: SignatureTable) -> pd.DataFrame:
    """For every feature and every interior bin edge, the tied-mass check
    that has silently disabled features four times in this project
    (dataplane/fitting.py's saturation fix; see STATUS.md) — generalized
    to a signature table's discrete bins: if the mass sitting exactly AT
    a bin edge's value exceeds that edge's own bin's share of the
    intended false-positive budget, values there could saturate one bin
    the same way a percentile threshold can land on a bound."""
    rows = []
    for feat, edges in table.bin_edges.items():
        vals = benign_fit_df[feat].dropna().to_numpy(dtype=float)
        n = len(vals)
        for edge in edges:
            tied = int(np.isclose(vals, edge).sum())
            rows.append({
                "feature": feat, "edge": float(edge), "n": n,
                "tied_count": tied, "tied_mass": tied / n if n else None,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Priority 3 — deterministic hash sampling
# --------------------------------------------------------------------- #

def stable_hash(flow_id: str) -> int:
    """Stable across processes and runs (unlike Python's salted built-in
    hash()) — MD5 is used purely as a fast, well-distributed digest, not
    for any security property."""
    digest = hashlib.md5(str(flow_id).encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def priority3_sample(flow_ids: Sequence[str], tau: int, mod: int = 10_000) -> np.ndarray:
    """SAMPLE(x) = 1[H(flowID) mod 10000 < tau]. Decided once per flow
    identity, never per packet."""
    return np.array([(stable_hash(fid) % mod) < tau for fid in flow_ids], dtype=bool)


# --------------------------------------------------------------------- #
# Policy composition: priority order, per-flow reported bit, meters
# --------------------------------------------------------------------- #

@dataclass
class Meter:
    """A hard capacity on how many escalations may be admitted. `capacity`
    is fixed at construction (from a fraction of total flow count);
    `used` only ever increases."""
    capacity: int
    used: int = 0

    def try_admit(self) -> bool:
        if self.used < self.capacity:
            self.used += 1
            return True
        return False

    @property
    def remaining(self) -> int:
        return max(0, self.capacity - self.used)


@dataclass
class PolicyResult:
    escalated: np.ndarray  # bool, admitted (meter-permitted) escalations only
    priority: np.ndarray  # int, 0 = not escalated, else 1/2/3
    matched_priority1: np.ndarray  # bool, WOULD have matched P1 (meter-independent)
    matched_priority2: np.ndarray
    matched_priority3: np.ndarray
    global_capacity: int
    priority2_reserved_capacity: int
    global_used: int
    priority_used: Dict[int, int] = field(default_factory=dict)


def run_policy(
    df: pd.DataFrame,
    predicted_class: np.ndarray,
    signature_table: SignatureTable,
    *,
    tau: int,
    global_budget_fraction: float,
    priority2_reserved_fraction: float = 0.0,
    order_col: str = "first_ts",
) -> PolicyResult:
    """Simulates the data-plane processing order over `df`, sorted by
    `order_col` (chronological arrival — a batch stand-in for a live
    stream), with a hard global meter and a reserved Priority-2
    sub-budget (so Priority 1 alone — which can dominate escalation
    volume, see the report's honesty notes on the binary-flag mismatch —
    can't starve Priority 2's visibility into novel signatures)."""
    n = len(df)
    order = np.argsort(df[order_col].to_numpy(), kind="stable")

    m1 = priority1_escalate(predicted_class)
    m2 = priority2_escalate(df, signature_table)
    m3 = priority3_sample(df["Flow ID"].tolist(), tau)

    global_capacity = math.ceil(global_budget_fraction * n)
    p2_reserved_capacity = math.ceil(priority2_reserved_fraction * n)
    p2_reserved_capacity = min(p2_reserved_capacity, global_capacity)
    p1_capacity = global_capacity - p2_reserved_capacity

    global_meter = Meter(capacity=global_capacity)
    p1_meter = Meter(capacity=p1_capacity)
    p2_meter = Meter(capacity=p2_reserved_capacity)

    escalated = np.zeros(n, dtype=bool)
    priority = np.zeros(n, dtype=int)

    for idx in order:
        if global_meter.remaining <= 0:
            break
        if m1[idx]:
            if p1_meter.try_admit() and global_meter.try_admit():
                escalated[idx] = True
                priority[idx] = 1
            continue
        if m2[idx]:
            # Priority 2 draws first from its own reservation, then from
            # any global headroom left once Priority 1's own sub-budget
            # isn't being used (a flow that only matches Priority 2 never
            # touches p1_meter at all).
            if p2_meter.try_admit():
                if global_meter.try_admit():
                    escalated[idx] = True
                    priority[idx] = 2
                else:
                    p2_meter.used -= 1  # global meter is the hard limit; refund
            elif global_meter.remaining > 0 and global_meter.try_admit():
                escalated[idx] = True
                priority[idx] = 2
            continue
        if m3[idx]:
            if global_meter.try_admit():
                escalated[idx] = True
                priority[idx] = 3
            continue

    return PolicyResult(
        escalated=escalated, priority=priority,
        matched_priority1=m1, matched_priority2=m2, matched_priority3=m3,
        global_capacity=global_capacity, priority2_reserved_capacity=p2_reserved_capacity,
        global_used=global_meter.used,
        priority_used={1: p1_meter.used, 2: p2_meter.used, 3: int((priority == 3).sum())},
    )


@dataclass(frozen=True)
class EscalationDigest:
    """One per admitted escalation — the fields the PDF specifies a real
    digest must carry."""
    flow_id: str
    priority: int
    reason: str
    matched_subtree: Optional[int]
    matched_leaf: Optional[int]
    feature_bins: Optional[Tuple]
    timestamp: Optional[float]


def build_digests(
    df: pd.DataFrame,
    result: PolicyResult,
    final_subtree: Optional[np.ndarray] = None,
    final_leaf: Optional[np.ndarray] = None,
) -> List[EscalationDigest]:
    digests = []
    flow_ids = df["Flow ID"].tolist()
    timestamps = df["first_ts"].tolist() if "first_ts" in df.columns else [None] * len(df)
    for i in np.nonzero(result.escalated)[0]:
        p = result.priority[i]
        reason = {1: "splidt_non_benign", 2: "uncommon_signature", 3: "sampled"}[p]
        digests.append(EscalationDigest(
            flow_id=flow_ids[i],
            priority=int(p),
            reason=reason,
            matched_subtree=int(final_subtree[i]) if (p == 1 and final_subtree is not None) else None,
            matched_leaf=int(final_leaf[i]) if (p == 1 and final_leaf is not None) else None,
            feature_bins=None,
            timestamp=timestamps[i],
        ))
    return digests
