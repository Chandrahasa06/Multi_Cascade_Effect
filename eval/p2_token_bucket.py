"""Priority meters: token bucket versus the per-day allowance, chronological.

Replaces the per-day allowance (first-come-first-served, exhausted about
8 minutes into each day) with a token bucket that refills with wall-clock time
from the flow timestamps. Everything runs in time order; nothing is shuffled.

Full-policy simulation: P1 rule matches, P2 (benign-only K, production four
features, n_bins 20, floor 5), P3 (hash sample, tau 20 of 10,000). Precedence
matches eval.escalation_eval: a P1-matched flow is only offered to P1; a
K-flagged flow is only offered to P2; P3 sees only the rest.

Meter models:
  'day'    the current per-day meter: 1% of that day's flows, reset each day,
           split 10/70/20 with the existing `Meter` class (unchanged).
  'token'  PriorityTokenBuckets at the design rate (1% of the wall-clock arrival
           rate), with a burst of S seconds of banked global budget.

Run: python -m eval.p2_token_bucket
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from dataplane.escalation_policy import (
    MICROS_PER_SECOND,
    Meter,
    PriorityTokenBuckets,
    assert_microseconds,
    priority3_sample,
)
from dataplane.escalation_policy import TokenBucket
from eval.escalation_data import BENIGN_LABEL

GLOBAL_FRACTION = 0.01
PRIORITY_FRACTIONS = (0.10, 0.70, 0.20)
TAU = 20
BURST_SECONDS = (1, 10, 60, 600, 3600, 86400)
MICROS_PER_MINUTE = 60 * MICROS_PER_SECOND
MICROS_PER_HOUR = 60 * MICROS_PER_MINUTE
FRONT_LOAD_MINUTES = 60
WED_FRI = (2, 3, 4)
EVAL_WEEKDAYS = (1, 2, 3, 4)
RESULTS_CSV = Path("results/p2_token_bucket.csv")
SUMMARY_JSON = Path("results/p2_token_bucket_summary.json")


# ---------------------------------------------------------------------------
# Budgets.
# ---------------------------------------------------------------------------

def wallclock_arrival_rate(first_ts_us: np.ndarray) -> float:
    """Flows per second over the capture's wall-clock span, nights included."""
    lo, hi = float(first_ts_us.min()), float(first_ts_us.max())
    assert_microseconds(lo)
    assert_microseconds(hi)
    return len(first_ts_us) / ((hi - lo) / MICROS_PER_SECOND)


def burst_capacity(seconds: float, rate_per_second: float) -> float:
    """Capacity in tokens: `seconds` of banked global budget, at least one token."""
    return max(1.0, float(math.ceil(seconds * rate_per_second)))


# ---------------------------------------------------------------------------
# Candidate priorities. K and P3 are fixed; only the meter varies.
# ---------------------------------------------------------------------------

def p3_ids(weekday: np.ndarray, row_idx: np.ndarray) -> np.ndarray:
    return np.array([f"{d}::{r}" for d, r in zip(weekday, row_idx)], dtype=object)


def candidate_priorities(ev, features: Sequence[str], n_bins: int, floor: int,
                         ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (prio per row: 0 none, 1 P1, 2 P2, 3 P3) and the K-flag per row."""
    from eval.p2_feature_search import build_fit_table, escalate
    table = build_fit_table(ev.fit_benign, list(features), n_bins, floor)
    flag = escalate({f: ev.features_all[f] for f in features}, table)
    m3 = np.asarray(priority3_sample(list(ids), TAU), dtype=bool)
    prio = np.where(ev.p1_pool, 1, np.where(flag, 2, np.where(m3, 3, 0))).astype(np.int8)
    return prio, flag


# ---------------------------------------------------------------------------
# Chronological simulations.
# ---------------------------------------------------------------------------

def simulate_day_meter(t: np.ndarray, wd: np.ndarray, cands: np.ndarray, prio: np.ndarray,
                       day_counts: Dict[int, int]) -> dict:
    """Current per-day meter. Each day's capacity is 1% of that day's flows,
    split with the existing reserved-thirds caps, served in time order, reset
    each day."""
    n = len(t)
    admitted = np.zeros(n, dtype=bool)
    adm_prio = np.zeros(n, dtype=np.int8)
    exhausted_us: Dict[int, float] = {}
    exhausted2_us: Dict[int, float] = {}
    cur = None
    g = None
    meters: Dict[int, Meter] = {}
    for r in cands:
        d = int(wd[r])
        if d != cur:
            cur = d
            G = int(math.ceil(GLOBAL_FRACTION * day_counts[d]))
            f1, f2, f3 = PRIORITY_FRACTIONS
            reserved2 = min(math.ceil(f2 * G), G)
            reserved3 = min(math.ceil(f3 * G), G - reserved2)
            caps = {1: G - reserved2 - reserved3, 2: G - reserved3, 3: reserved3}
            g = Meter(capacity=G)
            meters = {p: Meter(capacity=c) for p, c in caps.items()}
            exhausted_us[d] = None
            exhausted2_us[d] = None
        p = int(prio[r])
        if g.remaining > 0 and meters[p].remaining > 0:
            g.try_admit()
            meters[p].try_admit()
            admitted[r] = True
            adm_prio[r] = p
        if g.remaining == 0 and exhausted_us.get(d) is None:
            exhausted_us[d] = float(t[r])
        # P2 is blocked by its own cap or by the global cap, whichever binds
        if (g.remaining == 0 or meters[2].remaining == 0) and exhausted2_us.get(d) is None:
            exhausted2_us[d] = float(t[r])
    return {"admitted": admitted, "adm_prio": adm_prio, "cand_prio": prio, "exhausted_us": exhausted_us,
            "exhausted2_us": exhausted2_us, "empty_us": None, "empty2_us": None}


def simulate_token_meter(t: np.ndarray, wd: np.ndarray, cands: np.ndarray, prio: np.ndarray,
                         rate: float, capacity: float) -> dict:
    """Token-bucket meter at `rate` tokens/s globally, with per-priority
    buckets at 10/70/20 of rate and capacity. Served in time order.

    Empty time is accumulated per day: between two candidate events, the global
    bucket is empty for min(gap, time to refill to one token) when it was below
    one token at the earlier event."""
    n = len(t)
    admitted = np.zeros(n, dtype=bool)
    adm_prio = np.zeros(n, dtype=np.int8)
    pb = PriorityTokenBuckets.from_budget(rate, capacity, PRIORITY_FRACTIONS)
    empty_us: Dict[int, float] = defaultdict(float)
    empty2_us: Dict[int, float] = defaultdict(float)
    rate2 = PRIORITY_FRACTIONS[1] * rate
    prev_t = None
    prev_tok = None
    prev_tok2 = None
    prev_day = None
    for r in cands:
        ts = int(t[r])
        d = int(wd[r])
        if prev_t is not None:
            gap = ts - prev_t
            if prev_tok < 1.0:
                empty_us[prev_day] += min(gap, (1.0 - prev_tok) / rate * MICROS_PER_SECOND)
            if prev_tok2 < 1.0:
                empty2_us[prev_day] += min(gap, (1.0 - prev_tok2) / rate2 * MICROS_PER_SECOND)
        p = int(prio[r])
        if pb.admit(ts, p):
            admitted[r] = True
            adm_prio[r] = p
        prev_t = ts
        # admit() advances only the global bucket and the served priority's
        # bucket, so advance P2's bucket explicitly before reading it.
        prev_tok = pb.global_bucket.advance(ts)
        prev_tok2 = pb.buckets[2].advance(ts)
        prev_day = d
    return {"admitted": admitted, "adm_prio": adm_prio, "cand_prio": prio, "exhausted_us": None,
            "exhausted2_us": None, "empty_us": dict(empty_us), "empty2_us": dict(empty2_us)}


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

def _day_spans(t: np.ndarray, wd: np.ndarray) -> Dict[int, Tuple[float, float]]:
    out = {}
    for d in np.unique(wd):
        m = wd == d
        out[int(d)] = (float(t[m].min()), float(t[m].max()))
    return out


def _empty_fraction(sim: dict, spans: Dict[int, Tuple[float, float]], d: int, which: str = "global") -> float:
    """Fraction of day d's active span with no capacity. 'global' is the global
    bucket (or the day meter's global cap); 'p2' is P2's own bucket or cap."""
    lo, hi = spans[d]
    span = hi - lo
    if span <= 0:
        return float("nan")
    if which == "global" and sim["empty_us"] is not None:
        return sim["empty_us"].get(d, 0.0) / span
    if which == "p2" and sim["empty2_us"] is not None:
        return sim["empty2_us"].get(d, 0.0) / span
    key = "exhausted_us" if which == "global" else "exhausted2_us"
    ex = sim[key].get(d)
    return 0.0 if ex is None else (hi - ex) / span


def day_metrics(sim: dict, t, wd, labels, is_attack, src, spans, rule_name: str, day_label) -> dict:
    """One row for a day (int weekday) or for the pooled Wed-Fri set (a tuple)."""
    days = [day_label] if isinstance(day_label, int) else list(day_label)
    rows = np.isin(wd, days)
    adm = sim["admitted"] & rows
    prio = sim["adm_prio"]
    att = rows & is_attack
    ben = rows & ~is_attack
    out = {
        "meter": rule_name,
        "weekday": ",".join(str(d) for d in days),
        "rows": int(rows.sum()),
        "admitted": int(adm.sum()),
        "admitted_P1": int((adm & (prio == 1)).sum()),
        "admitted_P2": int((adm & (prio == 2)).sum()),
        "admitted_P3": int((adm & (prio == 3)).sum()),
        "attacks_admitted": int((adm & is_attack).sum()),
        "attacks_admitted_P1": int((adm & is_attack & (prio == 1)).sum()),
        "attacks_admitted_P2": int((adm & is_attack & (prio == 2)).sum()),
        "attacks_admitted_P3": int((adm & is_attack & (prio == 3)).sum()),
        "benign_n": int(ben.sum()),
        "benign_admitted": int((adm & ~is_attack).sum()),
        "benign_admit_rate": float((adm & ~is_attack).sum() / ben.sum()) if ben.sum() else None,
        "attack_n": int(att.sum()),
        "precision": float((adm & is_attack).sum() / adm.sum()) if adm.sum() else None,
    }
    for which, key in (("global", "empty_fraction_global"), ("p2", "empty_fraction_P2")):
        if isinstance(day_label, int):
            out[key] = _empty_fraction(sim, spans, day_label, which)
        else:
            num = sum(_empty_fraction(sim, spans, d, which) * (spans[d][1] - spans[d][0]) for d in days)
            den = sum(spans[d][1] - spans[d][0] for d in days)
            out[key] = num / den if den else None
    # time of admitted attacks relative to each day's first flow
    offs = []
    for d in days:
        lo = spans[d][0]
        m = adm & is_attack & (wd == d)
        offs.append((t[m] - lo) / MICROS_PER_MINUTE)
    offs = np.concatenate(offs) if offs else np.array([])
    # demand-weighted: share of P2 demand (candidate flows) refused by the meter.
    # Time-based empty fraction understates blocking when demand arrives in bursts.
    p2_demand = int(((sim["cand_prio"] == 2) & rows).sum())
    p2_admit = int((adm & (prio == 2)).sum())
    out["P2_demand_events"] = p2_demand
    out["P2_demand_refused_share"] = (1.0 - p2_admit / p2_demand) if p2_demand else None
    # active hours (clock hours from each day's first flow) with >= 1 P2 admission
    hours_total = 0
    hours_with_p2 = 0
    for d in days:
        lo, hi = spans[d]
        n_hours = int(math.ceil((hi - lo) / MICROS_PER_HOUR))
        hours_total += n_hours
        m = adm & (prio == 2) & (wd == d)
        hrs = np.unique(((t[m] - lo) // MICROS_PER_HOUR).astype(np.int64))
        hours_with_p2 += int(min(len(hrs), n_hours))
    out["active_hours_n"] = hours_total
    out["active_hours_with_P2_admission"] = hours_with_p2
    out["fraction_active_hours_with_P2_admission"] = (hours_with_p2 / hours_total) if hours_total else None
    out["attacks_admitted_first_attack_offset_min"] = float(offs.min()) if len(offs) else None
    out["attacks_admitted_median_offset_min"] = float(np.median(offs)) if len(offs) else None
    out["attacks_admitted_share_after_60min"] = (float((offs > FRONT_LOAD_MINUTES).mean()) if len(offs) else None)
    for c in sorted(pd.unique(labels[rows & is_attack])) if (rows & is_attack).any() else []:
        m = rows & (labels == c)
        out[f"class__{c}__n"] = int(m.sum())
        out[f"class__{c}__admitted"] = int((adm & (labels == c)).sum())
    return out


def incident_metrics(sim: dict, t, wd, labels, is_attack, src, days=WED_FRI) -> dict:
    """Per-incident coverage over the given days (pooled), alongside per-flow."""
    rows = np.isin(wd, list(days))
    adm = sim["admitted"] & rows
    att = rows & is_attack
    out = {}
    atk = pd.DataFrame({"src": src[att], "label": labels[att], "t": t[att]})
    adm_df = pd.DataFrame({"src": src[adm], "label": labels[adm], "t": t[adm]})
    ip_att = set(atk["src"]) if len(atk) else set()
    ip_esc = set(adm_df["src"]) if len(adm_df) else set()
    out["attack_source_ips_n"] = len(ip_att)
    out["attack_source_ips_escalated"] = len(ip_att & ip_esc)
    out["attack_source_ip_coverage"] = (len(ip_att & ip_esc) / len(ip_att)) if ip_att else None
    pairs_att = set(zip(atk["src"], atk["label"])) if len(atk) else set()
    pairs_esc = set(zip(adm_df["src"], adm_df["label"])) if len(adm_df) else set()
    out["src_class_pairs_n"] = len(pairs_att)
    out["src_class_pairs_escalated"] = len(pairs_att & pairs_esc)
    out["src_class_pair_coverage"] = (len(pairs_att & pairs_esc) / len(pairs_att)) if pairs_att else None
    out["attack_flows_n"] = int(att.sum())
    out["attack_flows_escalated"] = int((adm & is_attack).sum())
    out["attack_flow_coverage"] = (int((adm & is_attack).sum()) / int(att.sum())) if att.sum() else None
    lat = []
    never = []
    for c in sorted(atk["label"].unique()) if len(atk) else []:
        first_attack = atk.loc[atk["label"] == c, "t"].min()
        a = adm_df.loc[adm_df["label"] == c, "t"]
        if len(a):
            m = (float(a.min()) - float(first_attack)) / MICROS_PER_MINUTE
            lat.append(m)
            out[f"latency_min__{c}"] = m
        else:
            never.append(c)
            out[f"latency_min__{c}"] = None
    out["classes_never_escalated"] = ";".join(never)
    out["median_class_latency_min"] = float(np.median(lat)) if lat else None
    return out


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------

def run_all(ev, pool: pd.DataFrame) -> Tuple[List[dict], dict]:
    t = pool["first_ts"].to_numpy(dtype=float).astype(np.int64)
    wd = pool["weekday_idx"].to_numpy()
    row_idx = pool["row_idx"].to_numpy()
    labels = ev.labels
    is_attack = ev.is_attack
    src = pool["Source IP"].astype(str).to_numpy()
    spans = _day_spans(t, wd)
    day_counts = {int(d): int((wd == d).sum()) for d in np.unique(wd)}
    order = np.lexsort((row_idx, t))

    from dataplane.escalation_policy import PRIORITY2_FEATURES

    ids = p3_ids(wd, row_idx)
    prio, _ = candidate_priorities(ev, PRIORITY2_FEATURES, 20, 5, ids)
    cands = order[prio[order] > 0]

    wallclock = wallclock_arrival_rate(t)
    rate = GLOBAL_FRACTION * wallclock
    summary = {"wallclock_arrival_rate_per_s": wallclock, "global_rate_per_s": rate,
               "global_rate_per_day": rate * 86400, "candidate_events": int(len(cands))}

    rows: List[dict] = []
    sims = {}
    base = simulate_day_meter(t, wd, cands, prio, day_counts)
    sims["day_meter"] = base
    for d in EVAL_WEEKDAYS:
        rows.append(day_metrics(base, t, wd, labels, is_attack, src, spans, "day_meter", d))
    rows.append(day_metrics(base, t, wd, labels, is_attack, src, spans, "day_meter", WED_FRI))
    summary["day_meter_incidents"] = incident_metrics(base, t, wd, labels, is_attack, src)

    per_burst = {}
    for S in BURST_SECONDS:
        cap = burst_capacity(S, rate)
        sim = simulate_token_meter(t, wd, cands, prio, rate, cap)
        name = f"token_S{S}s"
        sims[name] = sim
        for d in EVAL_WEEKDAYS:
            rows.append(day_metrics(sim, t, wd, labels, is_attack, src, spans, name, d))
        rows.append(day_metrics(sim, t, wd, labels, is_attack, src, spans, name, WED_FRI))
        inc = incident_metrics(sim, t, wd, labels, is_attack, src)
        m = rows[-1]
        per_burst[S] = {"capacity_tokens": cap, "share_after_60min_WedFri": m["attacks_admitted_share_after_60min"],
                        "attacks_admitted_WedFri": m["attacks_admitted"],
                        "empty_fraction_global_WedFri": m["empty_fraction_global"],
                        "empty_fraction_P2_WedFri": m["empty_fraction_P2"],
                        "incidents": inc}
    summary["bursts"] = {str(k): v for k, v in per_burst.items()}
    summary["rate_per_s"] = rate
    summary["_sims"] = sims
    return rows, summary


def choose_burst(per_burst: dict) -> Tuple[int, str]:
    """Pre-registered rule: largest S whose Wed-Fri admitted attacks are at least
    half more than FRONT_LOAD_MINUTES after each day's first flow. A burst with
    no admitted attacks has no evidence and does not pass."""
    passing = []
    for S, v in per_burst.items():
        share = v["share_after_60min_WedFri"]
        if share is not None and share >= 0.5:
            passing.append(S)
    if not passing:
        return None, "no burst avoids front-loading (rule not met)"
    return max(passing), "largest burst meeting the rule"


def p2_comparisons(ev, pool: pd.DataFrame, rate: float, chosen_s: int, tag: str) -> List[dict]:
    """Part 3: the P2-only comparisons under the chosen token meter, against the
    fixed per-day allowance (2,400, the earlier reference) and a fixed allowance
    matched to the token meter's average daily total."""
    from eval.p2_refit import build_inputs, day_metrics as refit_day_metrics, run_walk
    from dataplane.escalation_policy import PRIORITY2_FEATURES
    four = list(PRIORITY2_FEATURES)
    five = four + ["syn_without_synack_count"]
    p2_rate = PRIORITY_FRACTIONS[1] * rate
    p2_cap = burst_capacity(chosen_s, p2_rate)
    matched_slots = int(round(p2_rate * 86400))
    configs = [
        ("nb6_4f", 6, four, "no_refit", None),
        ("nb20_4f", 20, four, "no_refit", None),
        ("nb20_5f", 20, five, "no_refit", None),
        ("nb20_4f_ruleA_hour", 20, four, "A", "hour"),
    ]
    meters = [
        ("token_chosen", "token"),
        ("fixed_2400", "fixed2400"),
        ("fixed_matched", "fixedmatched"),
    ]
    out = []
    for name, nb, feats, rule, gran in configs:
        inp, labels = build_inputs(ev, pool, feats, n_bins=nb)
        for mname, kind in meters:
            if kind == "token":
                res = run_walk(inp, rule, gran or "day", "unbounded", budget="token",
                               token_bucket=TokenBucket(p2_rate, p2_cap))
                nominal = p2_rate * 86400
            elif kind == "fixed2400":
                res = run_walk(inp, rule, gran or "day", "unbounded", budget="day", slots_per_day=2400)
                nominal = 2400
            else:
                res = run_walk(inp, rule, gran or "day", "unbounded", budget="day", slots_per_day=matched_slots)
                nominal = matched_slots
            for day in (1, 2, 3, 4):
                m = refit_day_metrics(inp, res, rule, gran or "day", "unbounded", day, labels, nominal_slots=nominal)
                m.update({"section": "p2_comparison", "config": name, "meter": mname, "tag": tag,
                          "burst_seconds": chosen_s if kind == "token" else None})
                out.append(m)
    return out


def main() -> int:
    import json
    from eval.p2_feature_search import make_context
    ev, pool, _ = make_context()
    rows, summary = run_all(ev, pool)
    summary.pop("_sims", None)
    per_burst = {int(k): v for k, v in summary["bursts"].items()}
    chosen, why = choose_burst(per_burst)
    summary["chosen_burst_seconds"] = chosen
    summary["choice_rule"] = why
    comp_s = chosen if chosen is not None else 1
    summary["comparison_burst_seconds"] = comp_s
    comp = p2_comparisons(ev, pool, summary["rate_per_s"], comp_s, "rule_choice")
    if comp_s != 60:
        comp += p2_comparisons(ev, pool, summary["rate_per_s"], 60, "sensitivity_60s")
    out = pd.DataFrame(rows)
    out.insert(0, "section", "meter_sweep")
    out = pd.concat([out, pd.DataFrame(comp)], ignore_index=True)
    out.to_csv(RESULTS_CSV, index=False)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {RESULTS_CSV} ({len(out)} rows); chosen burst: {chosen} ({why})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
