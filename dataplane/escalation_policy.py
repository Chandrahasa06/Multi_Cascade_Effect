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
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from eval.generalization_experiments import build_K, compute_bin_edges, signatures_for

#: syn_without_synack_count was removed: its quantile edges collapse to [0], so
#: every value lands in one bin and it cannot alter any signature (verified
#: across the full pool, see results/p2_out_of_period.md). Removing it changes
#: the signature space (174,080 -> 87,040) and nothing else.
PRIORITY2_FEATURES: Tuple[str, ...] = (
    "flows_per_src",
    "distinct_dst_ports_per_src",
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


# --------------------------------------------------------------------- #
# Token-bucket meters (additive). `Meter` above is unchanged: the per-day
# results depend on it and must stay reproducible. These meters refill with
# wall-clock time from flow timestamps, not with flow count.
# --------------------------------------------------------------------- #

MICROS_PER_SECOND = 1_000_000


def assert_microseconds(ts_us: float) -> None:
    """first_ts in this project is microseconds since the epoch (2017-07-03 is
    1,499,072,158,000,000). A seconds value would make every refill ~10^6 times
    too small, silently, so refuse anything outside the plausible range."""
    if not (1e14 <= float(ts_us) <= 2e15):
        raise ValueError(f"timestamp {ts_us!r} is not microseconds since the epoch")


@dataclass
class TokenBucket:
    """Continuous-refill token bucket. Tokens accrue at `rate_per_second` with
    wall-clock time between calls, capped at `capacity` (the burst). Starts
    full, as a real meter does. Time must not go backwards."""
    rate_per_second: float
    capacity: float
    tokens: float = -1.0
    last_ts_us: Optional[float] = None

    def __post_init__(self):
        if not self.rate_per_second > 0:
            raise ValueError("rate_per_second must be positive")
        if self.capacity < 1:
            raise ValueError("capacity below one token can never admit anything")
        if self.tokens < 0:
            self.tokens = float(self.capacity)

    def advance(self, ts_us: float) -> float:
        assert_microseconds(ts_us)
        if self.last_ts_us is None:
            self.last_ts_us = float(ts_us)
        if ts_us < self.last_ts_us:
            raise ValueError("token bucket time went backwards")
        dt = (float(ts_us) - self.last_ts_us) / MICROS_PER_SECOND
        self.tokens = min(self.capacity, self.tokens + self.rate_per_second * dt)
        self.last_ts_us = float(ts_us)
        return self.tokens

    def has_token(self, ts_us: float) -> bool:
        return self.advance(ts_us) >= 1.0

    def take(self, ts_us: float, n: float = 1.0) -> bool:
        if self.advance(ts_us) >= n:
            self.tokens -= n
            return True
        return False


# Fraction of each priority's burst that is reserved: a bucket can lend only the
# tokens above this floor, so a lender always keeps it. Chosen as one half, stated
# as a choice, not tuned.
RESERVED_FLOOR_FRACTION = 0.5


@dataclass
class PriorityTokenBuckets:
    """Per-priority buckets with a global bucket checked alongside. The priority
    rates and bursts are fractions of the global rate and burst (default 10/70/20).

    Admission needs a token from the global bucket and from one priority bucket.
    A priority first uses its own bucket. If its own bucket is empty and `borrow`
    is on, it may take one token from another priority's SURPLUS: tokens above
    that priority's reserved floor. A lender is never drawn below its floor, so a
    burst on one priority cannot starve another of its reserved capacity. With
    `borrow=False` the buckets are partitioned, as before."""
    global_bucket: TokenBucket
    buckets: Dict[int, TokenBucket]
    borrow: bool = True
    floors: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_budget(cls, global_rate_per_second: float, global_capacity: float,
                    fractions: Tuple[float, float, float] = (0.10, 0.70, 0.20),
                    borrow: bool = True) -> "PriorityTokenBuckets":
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("priority fractions must sum to 1")
        buckets = {}
        for p, f in zip((1, 2, 3), fractions):
            buckets[p] = TokenBucket(rate_per_second=f * global_rate_per_second,
                                     capacity=max(1.0, f * global_capacity))
        floors = {p: RESERVED_FLOOR_FRACTION * b.capacity for p, b in buckets.items()}
        return cls(global_bucket=TokenBucket(global_rate_per_second, global_capacity),
                   buckets=buckets, borrow=borrow, floors=floors)

    def _best_lender(self, priority: int) -> Optional[int]:
        """The other priority with the largest surplus above its floor, if any
        holds at least one spare token. Ties go to the lower priority number."""
        best, best_surplus = None, 0.0
        for q in sorted(self.buckets):
            if q == priority:
                continue
            surplus = self.buckets[q].tokens - self.floors.get(q, 0.0)
            if surplus >= 1.0 and (best is None or surplus > best_surplus):
                best, best_surplus = q, surplus
        return best

    def admit(self, ts_us: float, priority: int) -> bool:
        if priority not in self.buckets:
            raise ValueError(f"unknown priority {priority!r}")
        g = self.global_bucket.advance(ts_us)
        for b in self.buckets.values():
            b.advance(ts_us)
        if g < 1.0:
            return False
        own = self.buckets[priority]
        if own.tokens >= 1.0:
            own.tokens -= 1.0
            self.global_bucket.tokens -= 1.0
            return True
        if not self.borrow:
            return False
        lender = self._best_lender(priority)
        if lender is None:
            return False
        self.buckets[lender].tokens -= 1.0
        self.global_bucket.tokens -= 1.0
        return True


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


# --------------------------------------------------------------------- #
# Three-reserved-meter composition -- an ADDITIVE variant of run_policy()
# above, for eval/escalation_eval.py's decision-tree-rule-based Priority 1
# (dataplane/dt_rules.py, compiled from tree.txt) rather than SpliDT
# classification. run_policy()/priority1_escalate() above are untouched;
# this is a parallel path with its own signature, not a replacement.
#
# Motivation: run_policy() reserves a sub-budget for Priority 2 only, so a
# burst of Priority 1 matches can still starve Priority 3 entirely (it
# only ever sees whatever the global meter has left after P1 and P2).
# The escalation report this function feeds needs "a burst of rule
# matches cannot starve P2 AND P3" (both), so all three priorities get
# their own reserved sub-budget here, sized off the PDF's own illustrative
# 10%/70%/20% split (`controller_rule_selection.pdf`'s Table).
# --------------------------------------------------------------------- #

@dataclass
class ThreePriorityResult:
    escalated: np.ndarray
    priority: np.ndarray
    matched_priority1: np.ndarray
    matched_priority2: np.ndarray
    matched_priority3: np.ndarray
    global_capacity: int
    priority_capacity: Dict[int, int]
    priority_used: Dict[int, int]
    global_used: int


def run_policy_reserved_thirds(
    df: pd.DataFrame,
    priority1_matched: np.ndarray,
    signature_table: SignatureTable,
    *,
    tau: int,
    global_budget_fraction: float,
    priority_fractions: Tuple[float, float, float] = (0.10, 0.70, 0.20),
    order_col: str = "first_ts",
    flow_id_col: str = "flow_uid",
    meter: str = "token",
    burst_seconds: float = 86400.0,
    time_col: str = "first_ts",
) -> ThreePriorityResult:
    """Priority order P1 -> P2 -> P3, per the PDF's "Data-plane processing
    order". Each priority draws from its OWN reserved sub-budget (default
    split 10/70/20 of the global budget) as well as the shared hard
    global meter, so a burst on any one priority cannot starve either of
    the other two. `priority1_matched` is a precomputed boolean match
    array (e.g. from dataplane.dt_rules.evaluate_rules_union), not a
    predicted-class array -- keeps this function classifier-agnostic,
    unlike priority1_escalate()/run_policy() above which assume a
    "Benign"-vs-other predicted-class array.

    meter="token" (default): token-bucket meters refilled with wall-clock time
    from `time_col`, served in time order; `order_col` is ignored because a
    token meter is only meaningful in time order. meter="per_day": the original
    Meter-based composition below, honouring `order_col` (kept so the earlier
    shuffled-sample results stay reproducible)."""
    if meter == "token":
        return run_policy_token(df, priority1_matched, signature_table, tau=tau,
                                global_budget_fraction=global_budget_fraction,
                                priority_fractions=priority_fractions,
                                burst_seconds=burst_seconds, time_col=time_col,
                                flow_id_col=flow_id_col)
    if meter != "per_day":
        raise ValueError(f"unknown meter {meter!r}")
    n = len(df)
    order = np.argsort(df[order_col].to_numpy(), kind="stable")

    m1 = np.asarray(priority1_matched, dtype=bool)
    m2 = priority2_escalate(df, signature_table)
    m3 = priority3_sample(df[flow_id_col].tolist(), tau)

    global_capacity = math.ceil(global_budget_fraction * n)
    f1, f2, f3 = priority_fractions
    # `priority_fractions` are read as MINIMUM GUARANTEES for P2 and P3,
    # each a fraction of the global budget (the PDF's Table: "Priority 2:
    # 0.70%" out of a 1.00% total) -- f1 is accepted for API symmetry with
    # the PDF's own 3-row table but unused directly: Priority 1 is
    # HIGHEST priority and gets whatever the global budget has left once
    # P2 and P3's guarantees are carved out, not a fixed slice of its own.
    # This generalizes run_policy()'s existing two-tier pattern above
    # (p1_capacity = global_capacity - p2_reserved_capacity) to three
    # tiers, so a P1 burst cannot starve EITHER P2 or P3 (run_policy()
    # only protects P2). A P2 burst still cannot starve P3, symmetrically.
    reserved2 = min(math.ceil(f2 * global_capacity), global_capacity)
    reserved3 = min(math.ceil(f3 * global_capacity), global_capacity - reserved2)
    cap1 = max(0, global_capacity - reserved2 - reserved3)
    cap2 = global_capacity - reserved3  # P2 may also draw on any of P1's unused headroom
    cap3 = reserved3
    caps = {1: cap1, 2: cap2, 3: cap3}

    global_meter = Meter(capacity=global_capacity)
    meters = {p: Meter(capacity=c) for p, c in caps.items()}

    escalated = np.zeros(n, dtype=bool)
    priority = np.zeros(n, dtype=int)

    # NOTE: unlike run_policy() above, there is deliberately no separate
    # "overflow into leftover global headroom" branch here. cap2/cap3
    # already bake in every bit of headroom each priority is entitled to
    # (cap2 = global_capacity - reserved3, cap3 = reserved3), so a
    # fallback that additionally checks bare global_meter.remaining > 0
    # once a priority's own meter is exhausted would let it keep
    # admitting past its own ceiling by drawing on capacity that's
    # supposed to be protected for a LOWER priority -- caught by
    # `test_p1_burst_cannot_starve_p2_or_p3` failing until this was
    # removed: P2's own meter (capacity=cap2) filled first, and the
    # overflow branch then let it keep consuming the remaining global
    # budget that was meant to be P3's reserved floor, before any
    # P3-only-eligible flow was ever reached.
    for idx in order:
        if global_meter.remaining <= 0:
            break
        if m1[idx]:
            if meters[1].try_admit() and global_meter.try_admit():
                escalated[idx] = True
                priority[idx] = 1
            continue
        if m2[idx]:
            if meters[2].try_admit():
                if global_meter.try_admit():
                    escalated[idx] = True
                    priority[idx] = 2
                else:
                    meters[2].used -= 1
            continue
        if m3[idx]:
            if meters[3].try_admit():
                if global_meter.try_admit():
                    escalated[idx] = True
                    priority[idx] = 3
                else:
                    meters[3].used -= 1
            continue

    return ThreePriorityResult(
        escalated=escalated, priority=priority,
        matched_priority1=m1, matched_priority2=m2, matched_priority3=m3,
        global_capacity=global_capacity, priority_capacity=caps,
        priority_used={p: int((priority == p).sum()) for p in (1, 2, 3)},
        global_used=global_meter.used,
    )


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


# ---------------------------------------------------------------------------
# Token-bucket policy (the default meter) and rule-A refit.
# ---------------------------------------------------------------------------

DEFAULT_BURST_SECONDS = 86400.0
MICROS_PER_CLOCK_HOUR = 3_600_000_000


def _time_us(df: pd.DataFrame, time_col: str) -> np.ndarray:
    t = df[time_col].to_numpy(dtype=float)
    if len(t) == 0:
        raise ValueError("empty frame")
    assert_microseconds(float(np.nanmin(t)))
    assert_microseconds(float(np.nanmax(t)))
    return t


def _token_meters(n_rows: int, t: np.ndarray, global_budget_fraction: float,
                  priority_fractions: Tuple[float, float, float], burst_seconds: float):
    """Rate: global_budget_fraction of the frame's own wall-clock arrival rate.
    Burst: `burst_seconds` of banked global budget, at least one token."""
    span_s = (float(t.max()) - float(t.min())) / MICROS_PER_SECOND
    if span_s <= 0:
        raise ValueError("token meter needs a positive time span")
    rate = global_budget_fraction * n_rows / span_s
    burst_tokens = max(1.0, float(math.ceil(burst_seconds * rate)))
    pb = PriorityTokenBuckets.from_budget(rate, burst_tokens, priority_fractions)
    return pb, rate, burst_tokens, span_s


def _admit_in_time_order(t, m1, m2, m3, pb: PriorityTokenBuckets, rows: np.ndarray,
                         escalated: np.ndarray, priority: np.ndarray) -> None:
    """Precedence as in run_policy_reserved_thirds: a P1-matched flow is only
    offered to P1; a K-flagged flow only to P2; P3 sees the rest."""
    for idx in rows:
        if m1[idx]:
            if pb.admit(float(t[idx]), 1):
                escalated[idx] = True
                priority[idx] = 1
            continue
        if m2[idx]:
            if pb.admit(float(t[idx]), 2):
                escalated[idx] = True
                priority[idx] = 2
            continue
        if m3[idx]:
            if pb.admit(float(t[idx]), 3):
                escalated[idx] = True
                priority[idx] = 3


def _result(escalated, priority, m1, m2, m3, pb, rate, span_s, n) -> "ThreePriorityResult":
    return ThreePriorityResult(
        escalated=escalated, priority=priority,
        matched_priority1=m1, matched_priority2=m2, matched_priority3=m3,
        global_capacity=int(math.ceil(rate * span_s)),
        priority_capacity={p: int(b.capacity) for p, b in pb.buckets.items()},
        priority_used={p: int((priority == p).sum()) for p in (1, 2, 3)},
        global_used=int(escalated.sum()),
    )


def run_policy_token(df: pd.DataFrame, priority1_matched: np.ndarray, signature_table: SignatureTable, *,
                     tau: int, global_budget_fraction: float,
                     priority_fractions: Tuple[float, float, float] = (0.10, 0.70, 0.20),
                     burst_seconds: float = DEFAULT_BURST_SECONDS,
                     time_col: str = "first_ts", flow_id_col: str = "flow_uid") -> ThreePriorityResult:
    """Static-K policy under per-priority token buckets, served in time order.
    `priority_capacity` reports each bucket's burst in tokens; `global_capacity`
    reports the budget over the frame's span (rate times span)."""
    t = _time_us(df, time_col)
    n = len(df)
    m1 = np.asarray(priority1_matched, dtype=bool)
    m2 = priority2_escalate(df, signature_table)
    m3 = priority3_sample(df[flow_id_col].tolist(), tau)
    pb, rate, _, span_s = _token_meters(n, t, global_budget_fraction, priority_fractions, burst_seconds)
    escalated = np.zeros(n, dtype=bool)
    priority = np.zeros(n, dtype=int)
    _admit_in_time_order(t, m1, m2, m3, pb, np.argsort(t, kind="stable"), escalated, priority)
    return _result(escalated, priority, m1, m2, m3, pb, rate, span_s, n)


class OnlineSignatureTable:
    """Priority 2's K, updatable from observations.

    The only way counts change is `update(df, benign_judged)`. It takes the
    observed features and a boolean verdict. It has NO label parameter, so an
    update cannot read a label by construction. Bin edges are fixed from the
    initial benign fit and never refit; only counts move."""

    def __init__(self, bin_edges: Dict[str, np.ndarray], features: Tuple[str, ...], floor: int,
                 counts: Counter, fit_n: int):
        self.bin_edges = bin_edges
        self.features = tuple(features)
        self.floor = int(floor)
        self.counts = Counter(counts)
        self.fit_n = int(fit_n)
        self._common = {sig for sig, c in self.counts.items() if c >= self.floor}

    @classmethod
    def from_benign_fit(cls, benign_fit_df: pd.DataFrame, *, n_bins: int, floor: int = 5,
                        features: Sequence[str] = PRIORITY2_FEATURES,
                        label_col: str = "Label") -> "OnlineSignatureTable":
        """Same benign-only fit as build_signature_table. The assertion is the
        same one that guards the static K: a non-benign row is a bug."""
        assert (benign_fit_df[label_col] == BENIGN_CLASS).all(), (
            "OnlineSignatureTable fit received a non-benign label; K must be "
            "fit on benign traffic only.")
        edges = compute_bin_edges(benign_fit_df, list(features), n_bins)
        sigs = signatures_for(benign_fit_df, list(features), edges)
        return cls(edges, tuple(features), floor, Counter(sigs), len(benign_fit_df))

    @property
    def common(self) -> FrozenSet[tuple]:
        return frozenset(self._common)

    def escalate(self, df: pd.DataFrame) -> np.ndarray:
        """True where a flow's signature is not in K (uncommon)."""
        sigs = signatures_for(df, list(self.features), self.bin_edges)
        return np.array([s not in self._common for s in sigs], dtype=bool)

    def update(self, df: pd.DataFrame, benign_judged: np.ndarray) -> int:
        """Folds judged-benign rows into the counts. Returns how many were added."""
        judged = np.asarray(benign_judged, dtype=bool)
        if len(judged) != len(df):
            raise ValueError("verdict length must match the frame")
        sigs = signatures_for(df[judged], list(self.features), self.bin_edges)
        for s in sigs:
            c = self.counts[s] + 1
            self.counts[s] = c
            if c == self.floor:
                self._common.add(s)
        return len(sigs)

    def as_signature_table(self) -> SignatureTable:
        return SignatureTable(bin_edges=self.bin_edges, common_signatures=frozenset(self._common),
                              features=self.features, floor=self.floor, fit_n=self.fit_n)


def run_policy_token_refit(df: pd.DataFrame, priority1_matched: np.ndarray, online_table: OnlineSignatureTable,
                           verdict, *, tau: int, global_budget_fraction: float,
                           priority_fractions: Tuple[float, float, float] = (0.10, 0.70, 0.20),
                           burst_seconds: float = DEFAULT_BURST_SECONDS,
                           time_col: str = "first_ts", flow_id_col: str = "flow_uid"):
    """Rule A, hourly, unbounded K. Each clock hour is classified with K as it
    stands at that hour's start. Afterwards K is updated from the hour's
    P2-admitted flows that `verdict` judged benign. Only P2-admitted flows
    can update K.

    `verdict` is a dataplane.controller_verdict.ControllerVerdict. Its
    judgement is the only place a label can enter, and with the stand-in it is
    an UPPER BOUND (see that module).

    Returns (ThreePriorityResult, the updated table, [(hour, K size), ...])."""
    t = _time_us(df, time_col)
    n = len(df)
    m1 = np.asarray(priority1_matched, dtype=bool)
    m3 = priority3_sample(df[flow_id_col].tolist(), tau)
    pb, rate, _, span_s = _token_meters(n, t, global_budget_fraction, priority_fractions, burst_seconds)
    m2 = np.zeros(n, dtype=bool)
    escalated = np.zeros(n, dtype=bool)
    priority = np.zeros(n, dtype=int)
    k_sizes = []
    order = np.argsort(t, kind="stable")
    hours = np.floor(t[order] / MICROS_PER_CLOCK_HOUR).astype(np.int64)
    bounds = np.nonzero(np.r_[True, hours[1:] != hours[:-1]])[0]
    ends = np.r_[bounds[1:], len(order)]
    for s, e in zip(bounds, ends):
        rows = order[s:e]
        m2[rows] = online_table.escalate(df.iloc[rows])
        k_sizes.append((int(hours[s]), len(online_table._common)))
        _admit_in_time_order(t, m1, m2, m3, pb, rows, escalated, priority)
        p2_rows = rows[priority[rows] == 2]
        if len(p2_rows):
            judged = verdict.judge(df.iloc[p2_rows])
            online_table.update(df.iloc[p2_rows], judged)
    result = _result(escalated, priority, m1, m2, m3, pb, rate, span_s, n)
    return result, online_table, k_sizes


# ---------------------------------------------------------------------------
# Unusualness scores for ranking P2's flagged flows. Neither reads a label: they
# take signatures, the benign fit counts, and the known signatures in K.
# ---------------------------------------------------------------------------

def benign_signature_counts(benign_fit_df: pd.DataFrame, features: Sequence[str],
                            bin_edges: Dict[str, np.ndarray]) -> Counter:
    """How often each signature occurs in the benign fit population. The same
    counts build_K thresholds at the floor; here they are kept for scoring."""
    return Counter(signatures_for(benign_fit_df, list(features), bin_edges))


def frequency_unusualness(sigs: Sequence[tuple], counts: Counter) -> np.ndarray:
    """1 / (1 + benign count). Never-seen signatures score 1.0; a signature seen
    four times (just under the floor of five) scores 0.2. Higher is more unusual.
    Flagged flows have benign count below the floor, so the score has only five
    levels (counts 0 to 4); ties are broken by arrival time, which is the
    honest limit of this score."""
    return np.array([1.0 / (1.0 + counts.get(s, 0)) for s in sigs], dtype=float)


def nearest_known_distance(sigs: Sequence[tuple], common: FrozenSet[tuple]) -> np.ndarray:
    """Minimum number of features whose bin index differs from any signature in
    K. A flow one bin away from a common signature scores 1; far from everything
    scores the feature count. Higher is more unusual. With four features there
    are only four levels (1 to 4)."""
    if not sigs:
        return np.zeros(0, dtype=float)
    width = len(sigs[0])
    if not common:
        return np.full(len(sigs), float(width))
    known = np.array(sorted(common), dtype=np.int64)
    cache: Dict[tuple, float] = {}
    out = np.empty(len(sigs), dtype=float)
    for i, sig in enumerate(sigs):
        if sig not in cache:
            cache[sig] = float(np.min(np.sum(known != np.array(sig, dtype=np.int64), axis=1)))
        out[i] = cache[sig]
    return out


# ---------------------------------------------------------------------------
# Ranking buffer. A window of arrivals is held, scored, and admitted in score
# order under the same token meters. Window 0 is first-come-first-served, with
# no buffer and no delay.
# ---------------------------------------------------------------------------

SCORERS = ("fcfs", "frequency", "distance")


def run_policy_token_buffered(df: pd.DataFrame, priority1_matched: np.ndarray, signature_table: SignatureTable, *,
                              tau: int, global_budget_fraction: float,
                              priority_fractions: Tuple[float, float, float] = (0.10, 0.70, 0.20),
                              burst_seconds: float = DEFAULT_BURST_SECONDS,
                              window_us: float = 0.0, scorer: str = "fcfs",
                              benign_counts: Optional[Counter] = None,
                              borrow: bool = True,
                              time_col: str = "first_ts", flow_id_col: str = "flow_uid"):
    """Token-metered policy with an optional ranking buffer.

    window_us == 0 (or scorer == "fcfs" with window 0): first-come-first-served,
    each flow decided at its own arrival, latency zero.

    window_us > 0: arrivals are grouped into windows [k*W, (k+1)*W). At each
    window's close the candidates are decided against the meters, all at the
    close time. P1 first (arrival order), then P2 ordered by score (scorer
    "fcfs" keeps arrival order, the buffered control), then P3 (arrival order).
    A flow's latency is its close time minus its arrival.

    Returns (ThreePriorityResult, latency_us per row (NaN if not admitted),
    buffer dict with per-window candidate counts)."""
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r}")
    if scorer == "frequency" and benign_counts is None:
        raise ValueError("frequency scorer needs the benign signature counts")
    t = _time_us(df, time_col)
    n = len(df)
    m1 = np.asarray(priority1_matched, dtype=bool)
    m2 = priority2_escalate(df, signature_table)
    m3 = priority3_sample(df[flow_id_col].tolist(), tau)
    pb, rate, _, span_s = _token_meters(n, t, global_budget_fraction, priority_fractions, burst_seconds)
    pb.borrow = borrow
    escalated = np.zeros(n, dtype=bool)
    priority = np.zeros(n, dtype=int)
    latency = np.full(n, np.nan)
    order = np.argsort(t, kind="stable")
    stats = {"windows": 0, "candidates_per_window": [], "p2_per_window": [], "p2_admitted_per_window": []}

    if window_us <= 0:
        for idx in order:
            if not (m1[idx] or m2[idx] or m3[idx]):
                continue
            before = escalated[idx]
            _admit_in_time_order(t, m1, m2, m3, pb, np.array([idx]), escalated, priority)
            if escalated[idx] and not before:
                latency[idx] = 0.0
        return _result(escalated, priority, m1, m2, m3, pb, rate, span_s, n), latency, stats

    sigs = None
    score = np.zeros(n, dtype=float)
    if scorer != "fcfs":
        sigs = signatures_for(df, list(signature_table.features), signature_table.bin_edges)
        p2_idx = np.nonzero(m2 & ~m1)[0]
        p2_sigs = [sigs[i] for i in p2_idx]
        if scorer == "frequency":
            score[p2_idx] = frequency_unusualness(p2_sigs, benign_counts)
        else:
            score[p2_idx] = nearest_known_distance(p2_sigs, signature_table.common_signatures)

    windows = np.floor(t[order] / window_us).astype(np.int64)
    bounds = np.nonzero(np.r_[True, windows[1:] != windows[:-1]])[0]
    ends = np.r_[bounds[1:], len(order)]
    for s, e in zip(bounds, ends):
        rows = order[s:e]
        close = float((windows[s] + 1) * window_us)
        p1_rows = rows[m1[rows]]
        p2_rows = rows[m2[rows] & ~m1[rows]]
        p3_rows = rows[m3[rows] & ~m1[rows] & ~m2[rows]]
        p2_rows = np.array(sorted(p2_rows, key=lambda i: (-score[i], t[i], i)), dtype=np.int64)
        stats["windows"] += 1
        stats["candidates_per_window"].append(int(len(p1_rows) + len(p2_rows) + len(p3_rows)))
        stats["p2_per_window"].append(int(len(p2_rows)))
        before_p2 = int((priority[rows] == 2).sum())
        for group, pr in ((p1_rows, 1), (p2_rows, 2), (p3_rows, 3)):
            for idx in group:
                if pb.admit(close, pr):
                    escalated[idx] = True
                    priority[idx] = pr
                    latency[idx] = close - float(t[idx])
        stats["p2_admitted_per_window"].append(int((priority[rows] == 2).sum()) - before_p2)
    return _result(escalated, priority, m1, m2, m3, pb, rate, span_s, n), latency, stats
