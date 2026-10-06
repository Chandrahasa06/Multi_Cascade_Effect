"""Ranking escalation candidates instead of first-come-first-served.

Four parts, see results/p2_ranking.md:
  1. borrowing between priorities: admitted versus nominal, before and after
  2. unusualness scores: frequency and distance; the score-separation check
     between flagged attack and flagged benign flows
  3. the ranking buffer: windows 0, 0.1, 1, 10, 60 seconds, with latency and
     buffer occupancy next to every gain
  4. the key comparison, per window: first-come-first-served, frequency rank,
     distance rank, and the buffered-FCFS control

Population: the 300k production sample (SAMPLE_RANDOM_STATE 42), sorted by time
and never shuffled. Static K at n_bins 6 (P2_N_BINS), floor 5, the four
production features. Scores and the buffer use the benign fit counts; no score
reads a label. Labels are used only to measure outcomes.

Run: python -m eval.p2_ranking
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataplane.dt_rules import evaluate_rules_union
from dataplane.escalation_policy import (
    PRIORITY2_FEATURES,
    benign_signature_counts,
    build_signature_table,
    compute_priority2_source_features,
    run_policy_token_buffered,
)
from eval.escalation_data import BENIGN_LABEL, load_pool
from eval.escalation_eval import (
    DEFAULT_TAU,
    P2_FLOOR,
    P2_N_BINS,
    PRIORITY_FRACTIONS,
    SAMPLE_BUDGET,
    _relabel_for_build_signature_table,
    build_p2_fit_pool,
    compile_p1_rules,
    draw_random_sample,
)
from eval.p2_metrics import class_recall, incident_coverage, precision

WED_FRI = (2, 3, 4)
WINDOWS_S = (0.0, 0.1, 1.0, 10.0, 60.0)
SCORERS = ("fcfs", "frequency", "distance")
OUT_CSV = Path("results/p2_ranking.csv")
OUT_JSON = Path("results/p2_ranking.json")
MICROS = 1_000_000


def _auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """P(score(attack) > score(benign)) with ties counted half."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    neg_sorted = np.sort(neg)
    lo = np.searchsorted(neg_sorted, pos, side="left")
    hi = np.searchsorted(neg_sorted, pos, side="right")
    return float(((lo + (hi - lo) / 2.0)).mean() / len(neg))


def _quartiles(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0, "p25": None, "median": None, "p75": None, "mean": None}
    return {"n": int(len(x)), "p25": float(np.percentile(x, 25)), "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)), "mean": float(x.mean())}


def run_one(sample_sorted, p1s, table, *, window_s: float, scorer: str, counts, borrow: bool):
    res, lat, stats = run_policy_token_buffered(
        sample_sorted, p1s, table, tau=DEFAULT_TAU, global_budget_fraction=SAMPLE_BUDGET,
        priority_fractions=PRIORITY_FRACTIONS, burst_seconds=86400.0,
        window_us=window_s * MICROS, scorer=scorer, benign_counts=counts, borrow=borrow)
    return res, lat, stats


def metrics_for(res, lat, stats, sample_sorted) -> dict:
    adm = res.escalated
    prec = precision(adm, sample_sorted)
    inc = incident_coverage(adm, sample_sorted)
    rec = class_recall(adm, sample_sorted)
    lab = sample_sorted["Label"].to_numpy()
    wd = sample_sorted["weekday_idx"].to_numpy()
    wf = np.isin(wd, WED_FRI)
    ps = wf & (lab == "PortScan")
    adm_wf_att = int((adm & wf & (lab != BENIGN_LABEL)).sum())
    lat_adm = lat[adm & wf]
    lat_p2 = lat[(res.priority == 2) & wf]
    cand = np.array(stats["candidates_per_window"], dtype=float)
    p2c = np.array(stats["p2_per_window"], dtype=float)
    p2a = np.array(stats["p2_admitted_per_window"], dtype=float)
    choice = float(((p2c > p2a) & (p2c > 0)).sum()) / max(1.0, float((p2c > 0).sum()))
    return {
        "attacks_admitted_wedfri": adm_wf_att,
        "attacks_admitted_wedfri_denominator": int((wf & (lab != BENIGN_LABEL)).sum()),
        "pairs_covered": inc["attack_pairs_covered"],
        "pairs_n": inc["attack_pairs_n"],
        "flow_coverage": inc["flow_coverage"],
        "precision": prec["precision"],
        "admitted_wedfri": prec["admitted_n"],
        "benign_admitted_wedfri": prec["admitted_n"] - prec["attacks_admitted"],
        "portscan_admitted_wedfri": int((adm & ps).sum()),
        "portscan_denominator": int(ps.sum()),
        "admitted_by_priority": {p: int(((res.priority == p) & wf).sum()) for p in (1, 2, 3)},
        "latency_s_median": float(np.nanmedian(lat_adm) / MICROS) if np.isfinite(lat_adm).any() else None,
        "latency_s_p99": float(np.nanpercentile(lat_adm, 99) / MICROS) if np.isfinite(lat_adm).any() else None,
        "latency_s_p2_median": float(np.nanmedian(lat_p2) / MICROS) if np.isfinite(lat_p2).any() else None,
        "latency_s_p2_p99": float(np.nanpercentile(lat_p2, 99) / MICROS) if np.isfinite(lat_p2).any() else None,
        "buffer_candidates_mean": float(cand.mean()) if len(cand) else 0.0,
        "buffer_candidates_p99": float(np.percentile(cand, 99)) if len(cand) else 0.0,
        "buffer_candidates_max": float(cand.max()) if len(cand) else 0.0,
        "buffer_windows": int(stats["windows"]),
        "share_p2_windows_with_a_choice": choice,
        "recall_wedfri": {c: v["recall"] for c, v in rec.items()},
        "admitted_by_class_wedfri": {c: v["admitted"] for c, v in rec.items()},
        "n_by_class_wedfri": {c: v["n"] for c, v in rec.items()},
        "incident": {k: v for k, v in inc.items() if k != "latency_min_by_class"},
    }


def main() -> int:
    pool = compute_priority2_source_features(load_pool())
    rules = compile_p1_rules()
    fit = build_p2_fit_pool(pool)
    sample = draw_random_sample(pool)
    sample_sorted = sample.sort_values("first_ts", kind="stable").reset_index(drop=True)
    p1s = evaluate_rules_union(rules, sample_sorted)[0]
    table = build_signature_table(_relabel_for_build_signature_table(fit), n_bins=P2_N_BINS, floor=P2_FLOOR,
                                  features=PRIORITY2_FEATURES)
    counts = benign_signature_counts(fit, PRIORITY2_FEATURES, table.bin_edges)

    out: dict = {"population": {"sample_rows": int(len(sample_sorted)),
                                "wedfri_rows": int(np.isin(sample_sorted["weekday_idx"], WED_FRI).sum()),
                                "wedfri_attack_rows": int((np.isin(sample_sorted["weekday_idx"], WED_FRI)
                                                           & (sample_sorted["Label"] != BENIGN_LABEL)).sum()),
                                "portscan_wedfri_rows": int((np.isin(sample_sorted["weekday_idx"], WED_FRI)
                                                             & (sample_sorted["Label"] == "PortScan")).sum()),
                                "nominal_budget_total": int(np.ceil(SAMPLE_BUDGET * len(sample_sorted))),
                                "nominal_by_priority": {p: int(np.ceil(f * np.ceil(SAMPLE_BUDGET * len(sample_sorted))))
                                                        for p, f in zip((1, 2, 3), PRIORITY_FRACTIONS)},
                                "n_bins": P2_N_BINS, "floor": P2_FLOOR, "K_size": len(table.common_signatures)}}

    # ---- Part 1: borrowing, before and after, FCFS
    part1 = {}
    for borrow in (False, True):
        res, lat, st = run_one(sample_sorted, p1s, table, window_s=0.0, scorer="fcfs", counts=counts, borrow=borrow)
        lab1 = sample_sorted["Label"].to_numpy()
        wd1 = sample_sorted["weekday_idx"].to_numpy()
        wf1 = np.isin(wd1, WED_FRI)
        att1 = lab1 != BENIGN_LABEL
        part1["borrow" if borrow else "partitioned"] = {
            "admitted_total": int(res.escalated.sum()),
            "by_priority": {p: int((res.priority == p).sum()) for p in (1, 2, 3)},
            "attacks_by_priority_week": {p: int(((res.priority == p) & att1).sum()) for p in (1, 2, 3)},
            "attacks_by_priority_wedfri": {p: int(((res.priority == p) & att1 & wf1).sum()) for p in (1, 2, 3)},
            "precision_by_priority_wedfri": {
                p: (int(((res.priority == p) & att1 & wf1).sum()) / int(((res.priority == p) & wf1).sum()))
                if int(((res.priority == p) & wf1).sum()) else None for p in (1, 2, 3)},
            "priority_capacity": res.priority_capacity,
            "nominal_total": out["population"]["nominal_budget_total"],
            "max_possible_with_initial_burst": out["population"]["nominal_budget_total"] + int(sum(res.priority_capacity.values())),
        }
    out["part1_borrowing"] = part1
    print("part 1 done", flush=True)

    # ---- Part 2: score separation on flagged candidates, Wed-Fri, by true label
    wd = sample_sorted["weekday_idx"].to_numpy()
    lab = sample_sorted["Label"].to_numpy()
    from dataplane.escalation_policy import priority2_escalate, signatures_for, frequency_unusualness, nearest_known_distance
    m2 = priority2_escalate(sample_sorted, table) & ~p1s
    wf = np.isin(wd, WED_FRI)
    flagged = np.nonzero(m2 & wf)[0]
    sigs_all = signatures_for(sample_sorted, list(table.features), table.bin_edges)
    sigs = [sigs_all[i] for i in flagged]
    freq = frequency_unusualness(sigs, counts)
    dist = nearest_known_distance(sigs, table.common_signatures)
    is_att = lab[flagged] != BENIGN_LABEL
    sep = {}
    for name, sc in (("frequency", freq), ("distance", dist)):
        sep[name] = {"attack": _quartiles(sc[is_att]), "benign": _quartiles(sc[~is_att]),
                     "auc_attack_vs_benign": _auc(sc[is_att], sc[~is_att])}
    sep["flagged_wedfri_n"] = int(len(flagged))
    sep["flagged_attack_n"] = int(is_att.sum())
    sep["flagged_benign_n"] = int((~is_att).sum())
    # Robustness: the same separation over the whole pool, Wednesday to Friday,
    # where the attack side has far more flagged flows than the sample.
    pool_p1 = evaluate_rules_union(rules, pool)[0]
    pwf = np.isin(pool["weekday_idx"].to_numpy(), WED_FRI)
    pool_flag = priority2_escalate(pool, table) & ~pool_p1 & pwf
    pool_rows = np.nonzero(pool_flag)[0]
    pool_sigs_all = signatures_for(pool, list(table.features), table.bin_edges)
    pool_sigs = [pool_sigs_all[i] for i in pool_rows]
    pool_is_att = pool["Label"].to_numpy()[pool_rows] != BENIGN_LABEL
    pf = frequency_unusualness(pool_sigs, counts)
    pd_ = nearest_known_distance(pool_sigs, table.common_signatures)
    psep = {"flagged_wedfri_n": int(len(pool_rows)), "flagged_attack_n": int(pool_is_att.sum()),
            "flagged_benign_n": int((~pool_is_att).sum())}
    for name, sc in (("frequency", pf), ("distance", pd_)):
        psep[name] = {"attack": _quartiles(sc[pool_is_att]), "benign": _quartiles(sc[~pool_is_att]),
                      "auc_attack_vs_benign": _auc(sc[pool_is_att], sc[~pool_is_att])}
    sep["pool_wedfri"] = psep
    out["part2_separation"] = sep
    print("part 2 done", flush=True)

    # ---- Parts 3 and 4: windows x scorers, borrowing on
    grid = {}
    for w in WINDOWS_S:
        for scorer in SCORERS:
            if w == 0.0 and scorer != "fcfs":
                # window 0 gives a single flow at a time: nothing to rank. Reported as such.
                grid[f"w{w}_{scorer}"] = {"identical_to_fcfs_by_construction": True}
                continue
            res, lat, st = run_one(sample_sorted, p1s, table, window_s=w, scorer=scorer, counts=counts, borrow=True)
            grid[f"w{w}_{scorer}"] = {"window_s": w, "scorer": scorer, **metrics_for(res, lat, st, sample_sorted)}
            print(f"window {w}s {scorer}: attacks {grid[f'w{w}_{scorer}'].get('attacks_admitted_wedfri')}", flush=True)
    out["grid"] = grid

    rows = []
    for key, v in grid.items():
        if v.get("identical_to_fcfs_by_construction"):
            continue
        r = {k: val for k, val in v.items() if not isinstance(val, (dict, list))}
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_CSV} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
