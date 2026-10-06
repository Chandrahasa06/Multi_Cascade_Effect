"""Periodic refit of Priority 2's signature table K, walked in time order and
tested against the out-of-period drift.

Protocol (from the task):
  1. Fit K on Monday benign rows (edges and counts).
  2. Walk Tuesday, Wednesday, Friday's windows in order. Each window is
     classified with K as it stands at the window's start, then K is updated
     from that window's stream. Evaluation always precedes the update.
  Three walks: no refit (baseline), rule A, rule B.

Update rules
  A  escalated-and-cleared: only flows P2 admitted (escalated), judged benign.
     The judgement is a STAND-IN for a perfect controller: the true label of
     admitted flows. That makes A an upper bound on what escalated-only
     feedback can deliver. This is the only place a label is read by an update,
     and it is confined to `_standin_perfect_controller_verdict`.
  B  everything the switch forwarded: flows not admitted by P2 are treated as
     benign and added to K. Rule B's update never receives labels. It is
     structurally label-free, not just by convention.

Granularity: 'day' applies updates at the end of each day; 'hour' applies them
at the end of each clock hour. Forgetting: 'unbounded' keeps all counts;
'window_<N>h' keeps only counts from the last N hours (a signature is in K
while it has at least FLOOR sightings in that window).

Budget: 2,400 slots per evaluation day, admitted in time order, as in
eval/p2_out_of_period.py. Unused slots do not carry over.

Scope bounds that apply to every number here:
  - P2 per-source features are whole-day batch aggregates, not streaming state.
  - P1-matched flows are excluded from P2 eligibility and from both update
    streams (P2-only experiment, P1 held out).
  - Bin edges are fixed from Monday benign and never refit. Only counts move.

Run: python -m eval.p2_refit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from eval.escalation_data import BENIGN_LABEL
from eval.generalization_experiments import compute_bin_edges

N_BINS = 20
FLOOR = 5
SLOTS_PER_DAY = 2400
EVAL_WEEKDAYS = (1, 2, 3, 4)  # Tuesday .. Friday
FIT_WEEKDAY = 0  # Monday

RULES = ("no_refit", "A", "B")
GRANULARITIES = ("day", "hour")
POLICIES = ("unbounded", "window_6h", "window_24h")

REPORT_MD = Path("results/p2_refit.md")
RESULTS_CSV = Path("results/p2_refit.csv")
KSIZE_CSV = Path("results/p2_refit_ksize_timeline.csv")
PROTECTED = ("Bot", "Infiltration", "Heartbleed", "Web Attack – Brute Force", "Web Attack – XSS",
             "Web Attack – Sql Injection")


# ---------------------------------------------------------------------------
# Label-free signature encoding (same mixed radix as eval.p2_feature_search).
# ---------------------------------------------------------------------------

def encode(values: Dict[str, np.ndarray], features: Sequence[str], edges: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int]:
    """int64 code per row and the code-space size R."""
    n = len(next(iter(values.values())))
    code = np.zeros(n, dtype=np.int64)
    R = 1
    for f in features:
        v = values[f]
        idx = np.digitize(v, edges[f]).astype(np.int64) + 1
        idx[np.isnan(v)] = 0
        radix = len(edges[f]) + 2
        code = code * radix + idx
        R *= radix
    return code, R


def fit_edges(monday_benign: Dict[str, np.ndarray], features: Sequence[str], n_bins: int = N_BINS) -> Dict[str, np.ndarray]:
    edges = {}
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    for f in features:
        vals = monday_benign[f]
        vals = vals[~np.isnan(vals)]
        edges[f] = np.unique(np.quantile(vals, qs))
    return edges


# ---------------------------------------------------------------------------
# Count state: dense per-signature counts, optionally windowed by hour.
# ---------------------------------------------------------------------------

class CountState:
    def __init__(self, R: int, policy: str):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        self.R = R
        self.policy = policy
        self.window_h = None if policy == "unbounded" else int(policy.split("_")[1].rstrip("h"))
        self.total = np.zeros(R, dtype=np.int64)
        self.buckets: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def add(self, codes: np.ndarray, hours: np.ndarray) -> None:
        """Adds one sighting per element. Hours are used only for windowed policies."""
        if len(codes) == 0:
            return
        if self.window_h is None:
            u, c = np.unique(codes, return_counts=True)
            np.add.at(self.total, u, c)
            return
        for h in np.unique(hours):
            m = hours == h
            u, c = np.unique(codes[m], return_counts=True)
            np.add.at(self.total, u, c)
            if h in self.buckets:
                pu, pc = self.buckets[int(h)]
                uu = np.union1d(pu, u)
                cc = np.zeros(len(uu), dtype=np.int64)
                cc[np.searchsorted(uu, pu)] += pc
                cc[np.searchsorted(uu, u)] += c
                self.buckets[int(h)] = (uu, cc)
            else:
                self.buckets[int(h)] = (u, c)

    def expire(self, now_hour: int) -> None:
        """Drops sightings older than the window ending at now_hour."""
        if self.window_h is None:
            return
        cutoff = now_hour - self.window_h
        for h in [h for h in self.buckets if h <= cutoff]:
            u, c = self.buckets.pop(h)
            np.add.at(self.total, u, -c)

    def common_mask(self) -> np.ndarray:
        return self.total >= FLOOR

    def size(self) -> int:
        return int((self.total >= FLOOR).sum())


# ---------------------------------------------------------------------------
# The rule-A stand-in. This is the ONLY function that reads labels inside an
# update path, and it accepts only admitted rows.
# ---------------------------------------------------------------------------

def _standin_perfect_controller_verdict(admitted_is_attack: np.ndarray) -> np.ndarray:
    """Stand-in for a perfect controller's verdict on escalated flows:
    True = judged benign. Used by rule A only, and documented as an upper bound.
    Accepts a boolean array of is_attack for admitted rows only."""
    return ~admitted_is_attack


# ---------------------------------------------------------------------------
# The walk.
# ---------------------------------------------------------------------------

@dataclass
class WalkInputs:
    codes: np.ndarray  # int64, per pool row
    weekday: np.ndarray  # int, per pool row
    hour: np.ndarray  # int global clock hour, per pool row
    first_ts: np.ndarray
    row_idx: np.ndarray
    eligible: np.ndarray  # bool: not P1-matched
    is_attack: np.ndarray  # bool, for METRICS and the rule-A stand-in only
    fit_codes: np.ndarray  # Monday benign
    fit_hours: np.ndarray
    R: int


@dataclass
class WalkResult:
    flagged: np.ndarray  # bool per pool row: K flagged it at its window start
    admitted: np.ndarray  # bool per pool row: admitted under the day budget
    in_update: np.ndarray  # bool per pool row: contributed to an update
    ksize_log: List[dict] = field(default_factory=list)  # K size at each window start
    day_end_ksize: Dict[int, int] = field(default_factory=dict)


def _windows(inp: WalkInputs, mask: np.ndarray, granularity: str):
    """Chronological windows over masked rows: yields (day, hour, rows) with rows
    sorted by (first_ts, row_idx)."""
    rows = np.nonzero(mask)[0]
    order = rows[np.lexsort((inp.row_idx[rows], inp.first_ts[rows]))]
    key = [inp.weekday[order]] + ([inp.hour[order]] if granularity == "hour" else [])
    keys = np.stack(key, axis=1)
    if granularity == "day":
        bounds = np.nonzero(np.r_[True, keys[1:, 0] != keys[:-1, 0]])[0]
    else:
        bounds = np.nonzero(np.r_[True, (keys[1:] != keys[:-1]).any(axis=1)])[0]
    ends = np.r_[bounds[1:], len(order)]
    for s, e in zip(bounds, ends):
        rows_w = order[s:e]
        yield int(inp.weekday[rows_w[0]]), int(inp.hour[rows_w[0]]), rows_w


def run_walk(inp: WalkInputs, rule: str, granularity: str, policy: str, budget: str = "day",
             token_bucket=None, slots_per_day: int = SLOTS_PER_DAY) -> WalkResult:
    """budget='day': 2,400 slots per day, admitted in time order (primary).
    budget='hour': 2,400/24 = 100 slots per clock hour, unused slots lost
    (sensitivity; requires granularity='hour')."""
    if budget not in ("day", "hour", "token"):
        raise ValueError(f"unknown budget {budget!r}")
    if budget == "token" and token_bucket is None:
        raise ValueError("token budget needs a token_bucket")
    if budget == "hour" and granularity != "hour":
        raise ValueError("hour budget needs hour granularity")
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    if rule != "no_refit" and granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}")
    n = len(inp.codes)
    res = WalkResult(flagged=np.zeros(n, dtype=bool), admitted=np.zeros(n, dtype=bool),
                     in_update=np.zeros(n, dtype=bool))
    state = CountState(inp.R, policy)
    state.add(inp.fit_codes, inp.fit_hours)

    eval_mask = np.isin(inp.weekday, EVAL_WEEKDAYS)
    pending_codes: List[np.ndarray] = []
    pending_hours: List[np.ndarray] = []
    last_hour: Optional[int] = None
    current_day = None
    remaining = 0
    gran = granularity if rule != "no_refit" else ("hour" if budget == "hour" else "day")
    for day, hour, rows_w in _windows(inp, eval_mask, gran):
        if day != current_day:
            # apply any deferred daily updates from the previous day first
            _flush(state, pending_codes, pending_hours, last_hour)
            if current_day is not None:
                res.day_end_ksize[current_day] = state.size()
            current_day = day
            if budget == "day":
                remaining = slots_per_day
        if budget == "hour":
            remaining = SLOTS_PER_DAY // 24
        state.expire(hour)
        last_hour = hour
        mask_k = state.common_mask()
        res.ksize_log.append({"weekday": day, "hour": hour, "K_size": int(mask_k.sum())})
        fl = ~mask_k[inp.codes[rows_w]]
        res.flagged[rows_w] = fl
        cand = rows_w[fl & inp.eligible[rows_w]]  # chronological already
        if budget == "token":
            # time-ordered token bucket: admit each candidate that finds a token
            take = np.array([r for r in cand if token_bucket.take(float(inp.first_ts[r]))], dtype=np.int64)
        else:
            take = cand[:remaining]
            remaining -= len(take)
        res.admitted[take] = True

        if rule == "no_refit":
            continue
        if rule == "A":
            # escalated-and-cleared: only admitted rows, judged benign (stand-in)
            verdict = _standin_perfect_controller_verdict(inp.is_attack[take])
            upd = take[verdict]
        else:  # rule B: label-free. Forwarded = eligible and not admitted.
            upd = rows_w[inp.eligible[rows_w] & ~res.admitted[rows_w]]
        res.in_update[upd] = True
        if granularity == "hour":
            state.add(inp.codes[upd], inp.hour[upd])
            state.expire(hour)  # keep the window tight at end of hour
        else:
            pending_codes.append(inp.codes[upd])
            pending_hours.append(inp.hour[upd])
    _flush(state, pending_codes, pending_hours, last_hour)
    if current_day is not None:
        res.day_end_ksize[current_day] = state.size()
    return res


def _flush(state: CountState, pc: List[np.ndarray], ph: List[np.ndarray], now_hour: Optional[int]) -> None:
    """Applies a deferred (daily) update stream at the end of a day, then expires
    the window ending at now_hour (the last window hour of that day)."""
    if pc:
        codes = np.concatenate(pc)
        hours = np.concatenate(ph)
        if len(codes):
            state.add(codes, hours)
    pc.clear()
    ph.clear()
    if now_hour is not None:
        state.expire(now_hour)


# ---------------------------------------------------------------------------
# Metrics. Evaluation uses the flags recorded BEFORE each window's update.
# ---------------------------------------------------------------------------

def day_metrics(inp: WalkInputs, res: WalkResult, rule: str, granularity: str, policy: str, day: int,
                labels: np.ndarray, nominal_slots: float = SLOTS_PER_DAY) -> dict:
    rows = np.nonzero((inp.weekday == day))[0]
    benign = rows[~inp.is_attack[rows]]
    att = rows[inp.is_attack[rows]]
    out = {
        "rule": rule, "granularity": granularity, "policy": policy, "weekday": day,
        "benign_n": int(len(benign)), "benign_flagged": int(res.flagged[benign].sum()),
        "benign_escalation_rate": float(res.flagged[benign].mean()) if len(benign) else None,
        "eligible_n": int(inp.eligible[rows].sum()),
        "flagged_total": int((res.flagged[rows] & inp.eligible[rows]).sum()),
        "admitted": int(res.admitted[rows].sum()),
        "fill_rate": float(res.admitted[rows].sum()) / nominal_slots,
        "nominal_slots": float(nominal_slots),
        "attacks_admitted": int((res.admitted[rows] & inp.is_attack[rows]).sum()),
        "update_rows": int(res.in_update[rows].sum()),
        "update_attack_rows": int((res.in_update[rows] & inp.is_attack[rows]).sum()),
    }
    out["attacks_per_100_slots"] = (100.0 * out["attacks_admitted"] / out["admitted"]) if out["admitted"] else None
    classes = sorted(pd.unique(labels[att])) if len(att) else []
    for c in classes:
        m = att[labels[att] == c]
        out[f"recall__{c}__n"] = int(len(m))
        out[f"recall__{c}__caught"] = int(res.flagged[m].sum())
    return out


def ksize_timeline(res: WalkResult, rule: str, granularity: str, policy: str) -> pd.DataFrame:
    df = pd.DataFrame(res.ksize_log)
    df.insert(0, "policy", policy)
    df.insert(0, "granularity", granularity)
    df.insert(0, "rule", rule)
    return df


# ---------------------------------------------------------------------------
# Inputs and runner.
# ---------------------------------------------------------------------------

MICROS_PER_HOUR = 3_600_000_000


def clock_hour(first_ts_us: np.ndarray) -> np.ndarray:
    """Absolute clock hour from first_ts. first_ts is microseconds since the
    epoch in this pool (2017-07-03 11:55:58 UTC = 1499072158000000). Asserted,
    because a seconds/microseconds mix-up silently turns hours into ms slices."""
    v = np.asarray(first_ts_us, dtype=float)
    finite = v[np.isfinite(v)]
    if len(finite) == 0 or finite.min() < 1e14 or finite.max() > 2e15:
        raise ValueError("first_ts is not in microseconds since the epoch; refusing to bucket hours")
    return np.floor(v / MICROS_PER_HOUR).astype(np.int64)


def build_inputs(ev, pool: pd.DataFrame, features: Sequence[str], n_bins: int = N_BINS) -> Tuple[WalkInputs, np.ndarray]:
    """Inputs for the walk, built from the P2Evaluator's arrays. Edges come from
    Monday benign only. Returns (inputs, labels) where labels is used for the
    class-level metrics and the rule-A stand-in, never by rule B's update."""
    wd = pool["weekday_idx"].to_numpy()
    first_ts = pool["first_ts"].to_numpy(dtype=float)
    hour = clock_hour(first_ts)
    labels = ev.labels
    monday = wd == FIT_WEEKDAY
    monday_benign = monday & ~ev.is_attack
    values = {f: ev.features_all[f] for f in features}
    edges = fit_edges({f: values[f][monday_benign] for f in features}, features, n_bins)
    codes, R = encode(values, features, edges)
    inp = WalkInputs(
        codes=codes, weekday=wd, hour=hour, first_ts=first_ts,
        row_idx=pool["row_idx"].to_numpy(), eligible=~ev.p1_pool, is_attack=ev.is_attack,
        fit_codes=codes[monday_benign], fit_hours=hour[monday_benign], R=R,
    )
    return inp, labels


def all_runs() -> List[Tuple[str, str, str]]:
    runs = [("no_refit", "none", "none")]
    for rule in ("A", "B"):
        for g in GRANULARITIES:
            for p in POLICIES:
                runs.append((rule, g, p))
    return runs


def main() -> int:
    from eval.p2_feature_search import make_context
    from eval.escalation_data import BENIGN_LABEL as _B  # noqa: F401
    from dataplane.escalation_policy import PRIORITY2_FEATURES

    features = list(PRIORITY2_FEATURES)  # the four features after change 1
    ev, pool, _ = make_context()
    inp, labels = build_inputs(ev, pool, features)
    rows, ks = [], []
    plan = [(r, g, p, "day") for r, g, p in all_runs()]
    plan += [(r, "hour", p, "hour") for r, p in [("no_refit", "none")]]
    plan += [(r, "hour", p, "hour") for r in ("A", "B") for p in POLICIES]
    seen = set()
    for rule, gran, policy, budget in plan:
        key = (rule, gran, policy, budget)
        if key in seen or (rule == "no_refit" and budget == "hour" and policy != "none"):
            continue
        seen.add(key)
        pol = policy if rule != "no_refit" else "none"
        res = run_walk(inp, rule, gran, pol if pol != "none" else "unbounded", budget=budget)
        for day in EVAL_WEEKDAYS:
            m = day_metrics(inp, res, rule, gran, pol, day, labels)
            m["K_size_day_end"] = res.day_end_ksize.get(day)
            m["budget"] = budget
            rows.append(m)
        ks.append(ksize_timeline(res, rule, gran, pol).assign(budget=budget))
        print(f"done {rule} {gran} {pol} budget={budget}", flush=True)
    out = pd.DataFrame(rows)
    out.insert(0, "budget", out.pop("budget"))
    out.to_csv(RESULTS_CSV, index=False)
    pd.concat(ks, ignore_index=True).to_csv(KSIZE_CSV, index=False)
    print(f"wrote {RESULTS_CSV} ({len(out)} rows) and {KSIZE_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
