"""Part 1 of results/p2_final.md: n_bins 6, 10, 20 under the token meter,
chronological, Wednesday to Friday. The decision is computed here by the rule
written in results/p2_final.md before this ran; it is not a judgement call.

Run: python -m eval.p2_nbins
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dataplane.dt_rules import evaluate_rules_union
from dataplane.escalation_policy import (
    build_signature_table,
    compute_priority2_source_features,
    priority2_escalate,
    run_policy_token,
)
from eval.escalation_data import BENIGN_LABEL, load_pool
from eval.escalation_eval import (
    _relabel_for_build_signature_table,
    build_p2_fit_pool,
    compile_p1_rules,
    draw_random_sample,
)
from eval.p2_metrics import incident_coverage

NBINS = (6, 10, 20)
FLOOR = 5
TAU = 20
GLOBAL_FRACTION = 0.01
BURST_SECONDS = 86400.0
WED_FRI = (2, 3, 4)
PROTECTED = ("Bot", "Infiltration", "Heartbleed", "Web Attack – Brute Force",
             "Web Attack – XSS", "Web Attack – Sql Injection")
OUT_CSV = Path("results/p2_nbins.csv")
OUT_JSON = Path("results/p2_nbins.json")


def decide(by_nb: dict) -> dict:
    """The pre-registered rule, applied to the measured numbers."""
    a6, a20 = by_nb[6], by_nb[20]
    m1 = a20["pool_recall_PortScan"] - a6["pool_recall_PortScan"] >= 0.50
    m2 = a20["sample_admitted_recall_PortScan"] - a6["sample_admitted_recall_PortScan"] >= 0.005
    regressions = []
    for c in PROTECTED:
        n = a6["pool_n"][c]
        if n == 0:
            continue
        k6, k20 = a6["pool_caught"][c], a20["pool_caught"][c]
        if n >= 50:
            p6 = k6 / n
            se = math.sqrt(p6 * (1 - p6) / n)
            if k20 / n < p6 - 2 * se:
                regressions.append(c)
        elif k20 <= k6 - 3:
            regressions.append(c)
    protected_ok = not regressions
    keep20 = bool(m1 and m2 and protected_ok)
    return {
        "M1_pool_PortScan_recall_gain_pp": 100 * (a20["pool_recall_PortScan"] - a6["pool_recall_PortScan"]),
        "M1_holds": bool(m1),
        "M2_sample_admitted_PortScan_recall_gain_pp": 100 * (a20["sample_admitted_recall_PortScan"] - a6["sample_admitted_recall_PortScan"]),
        "M2_holds": bool(m2),
        "protected_regressions": regressions,
        "protected_do_not_regress": bool(protected_ok),
        "decision_n_bins": 20 if keep20 else 6,
    }


def measure(nb: int, pool: pd.DataFrame, rules, fit_benign: pd.DataFrame, sample: pd.DataFrame,
            p1_pool: np.ndarray, p1_sample: np.ndarray) -> dict:
    table = build_signature_table(_relabel_for_build_signature_table(fit_benign), n_bins=nb, floor=FLOOR)
    # chronological: the sample sorted by time, never shuffled
    sample_sorted = sample.sort_values("first_ts", kind="stable").reset_index(drop=True)
    p1s = evaluate_rules_union(rules, sample_sorted)[0]
    res = run_policy_token(sample_sorted, p1s, table, tau=TAU, global_budget_fraction=GLOBAL_FRACTION,
                           burst_seconds=BURST_SECONDS)
    wd_s = sample_sorted["weekday_idx"].to_numpy()
    lab_s = sample_sorted["Label"].to_numpy()
    att_s = lab_s != BENIGN_LABEL
    wf_s = np.isin(wd_s, WED_FRI)
    out = {"n_bins": nb, "K_size": len(table.common_signatures)}
    # --- sample, Wednesday to Friday, admitted under the token meter
    out["sample_wedfri_n"] = int(wf_s.sum())
    out["sample_attacks_admitted_wedfri"] = int((res.escalated & att_s & wf_s).sum())
    out["sample_attacks_admitted_week"] = int((res.escalated & att_s).sum())
    out["sample_admitted_wedfri"] = int((res.escalated & wf_s).sum())
    out["sample_benign_admitted_wedfri"] = int((res.escalated & ~att_s & wf_s).sum())
    for c in sorted(pd.unique(lab_s[att_s & wf_s])):
        m = wf_s & (lab_s == c)
        out[f"sample_n__{c}"] = int(m.sum())
        out[f"sample_admitted__{c}"] = int((res.escalated & m).sum())
    n_ps = int((wf_s & (lab_s == "PortScan")).sum())
    out["sample_admitted_recall_PortScan"] = (
        int((res.escalated & wf_s & (lab_s == "PortScan")).sum()) / n_ps) if n_ps else None
    out["sample_PortScan_n_wedfri"] = n_ps
    # --- pool, Wednesday to Friday, K flags regardless of P1 (the original sweep's measure)
    pw = pool[pool["weekday_idx"].isin(WED_FRI)]
    flag_pool = priority2_escalate(pw, table)
    lab_p = pw["Label"].to_numpy()
    att_p = lab_p != BENIGN_LABEL
    out["pool_wedfri_n"] = int(len(pw))
    out["pool_benign_wedfri_n"] = int((~att_p).sum())
    out["pool_benign_flagged_wedfri"] = int(flag_pool[~att_p].sum())
    out["pool_benign_flag_rate_wedfri"] = float(flag_pool[~att_p].mean())
    pool_n, pool_caught = {}, {}
    for c in sorted(pd.unique(lab_p[att_p])):
        m = lab_p == c
        pool_n[c] = int(m.sum())
        pool_caught[c] = int(flag_pool[m].sum())
    out["pool_n"] = pool_n
    out["pool_caught"] = pool_caught
    out["pool_recall_PortScan"] = pool_caught.get("PortScan", 0) / pool_n["PortScan"] if pool_n.get("PortScan") else None
    # --- per-incident (source, class) coverage on the sample, Wed-Fri
    out["incident"] = incident_coverage(res.escalated, sample_sorted)
    return out


def main() -> int:
    pool = compute_priority2_source_features(load_pool())
    rules = compile_p1_rules()
    fit_benign = build_p2_fit_pool(pool)
    sample = draw_random_sample(pool)
    p1_pool = evaluate_rules_union(rules, pool)[0]
    p1_sample = evaluate_rules_union(rules, sample)[0]
    by_nb = {}
    for nb in NBINS:
        by_nb[nb] = measure(nb, pool, rules, fit_benign, sample, p1_pool, p1_sample)
        print(f"n_bins {nb}: sample attacks Wed-Fri {by_nb[nb]['sample_attacks_admitted_wedfri']}", flush=True)
    decision = decide(by_nb)
    out = {"measured": {str(k): v for k, v in by_nb.items()}, "decision": decision}
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    rows = []
    for nb in NBINS:
        r = {k: v for k, v in by_nb[nb].items() if not isinstance(v, (dict, list))}
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
