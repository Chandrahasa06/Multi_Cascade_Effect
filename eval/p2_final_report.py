"""Builds the v1 / v2 / v3 comparison for results/p2_final.md.

Every version is read from its CSV and mapped back onto the same 300k sample
(SAMPLE_RANDOM_STATE 42) by flow_uid, so denominators are shared. Order does not
affect any metric here (they are counts and sets), so v1 and v2 (shuffled) and
v3 (chronological) are directly comparable on admitted sets.

Writes results/p2_final_tables.md and results/p2_final_metrics.json.

Run: python -m eval.p2_final_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.escalation_eval import draw_random_sample
from eval.escalation_data import load_pool
from dataplane.escalation_policy import compute_priority2_source_features
from eval.p2_metrics import WED_FRI, class_recall, incident_coverage, precision

VERSIONS = {
    "v1": Path("results/escalated_flows.csv"),
    "v2": Path("results/escalated_flows_v2.csv"),
    "v3": Path("results/escalated_flows_v3.csv"),
}
TABLES = Path("results/p2_final_tables.md")
METRICS = Path("results/p2_final_metrics.json")


def admitted_mask(sample: pd.DataFrame, csv: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    ids = set(csv["flow_id"].astype(str))
    mask = sample["flow_uid"].astype(str).isin(ids).to_numpy()
    prio = csv.set_index(csv["flow_id"].astype(str))["admitted_by"]
    prio_by_sample = sample["flow_uid"].astype(str).map(prio).fillna("")
    return mask, prio_by_sample


def per_priority(sample, mask, prio, days=WED_FRI) -> dict:
    wd = sample["weekday_idx"].to_numpy()
    lab = sample["Label"].to_numpy()
    in_days = np.isin(wd, list(days))
    att = lab != "BENIGN"
    out = {}
    for p in ("P1", "P2", "P3"):
        m = mask & in_days & (prio.to_numpy() == p)
        out[p] = {"admitted": int(m.sum()), "attacks": int((m & att).sum())}
    return out


def build() -> dict:
    pool = compute_priority2_source_features(load_pool())
    sample = draw_random_sample(pool)
    sample = sample.reset_index(drop=True)
    wd = sample["weekday_idx"].to_numpy()
    wf = np.isin(wd, list(WED_FRI))
    results = {"sample_rows": int(len(sample)), "sample_wedfri_rows": int(wf.sum()),
               "sample_wedfri_attack_rows": int((wf & (sample["Label"].to_numpy() != "BENIGN")).sum())}
    # v3 without refit: the same chronological sample, n_bins 6, static K, token meter.
    # The difference from v3 is the refit's effect in isolation.
    from dataplane.dt_rules import evaluate_rules_union
    from dataplane.escalation_policy import build_signature_table, run_policy_token
    from eval.escalation_eval import (P2_FLOOR, P2_N_BINS, SAMPLE_BUDGET, DEFAULT_TAU,
                                      PRIORITY_FRACTIONS, _relabel_for_build_signature_table,
                                      build_p2_fit_pool, compile_p1_rules)
    rules = compile_p1_rules()
    fit = build_p2_fit_pool(pool)
    table = build_signature_table(_relabel_for_build_signature_table(fit), n_bins=P2_N_BINS, floor=P2_FLOOR)
    order = sample.sort_values("first_ts", kind="stable").index.to_numpy()
    s_sorted = sample.loc[order].reset_index(drop=True)
    p1s = evaluate_rules_union(rules, s_sorted)[0]
    static = run_policy_token(s_sorted, p1s, table, tau=DEFAULT_TAU, global_budget_fraction=SAMPLE_BUDGET,
                              priority_fractions=PRIORITY_FRACTIONS)
    static_mask = np.zeros(len(sample), dtype=bool)
    static_mask[order] = static.escalated
    static_prio = np.empty(len(sample), dtype=object)
    static_prio[order] = np.array([f"P{p}" if p else "" for p in static.priority], dtype=object)
    results["v3_static"] = {
        "csv_rows": int(static_mask.sum()),
        "priority_wedfri": per_priority(sample, static_mask, pd.Series(static_prio)),
        "precision_wedfri": precision(static_mask, sample),
        "precision_week": precision(static_mask, sample, days=(0, 1, 2, 3, 4)),
        "class_recall_wedfri": class_recall(static_mask, sample),
        "incident_wedfri": incident_coverage(static_mask, sample),
    }
    for ver, path in VERSIONS.items():
        csv = pd.read_csv(path)
        mask, prio = admitted_mask(sample, csv)
        results[ver] = {
            "csv_rows": int(len(csv)),
            "priority_wedfri": per_priority(sample, mask, prio),
            "precision_wedfri": precision(mask, sample),
            "precision_week": precision(mask, sample, days=(0, 1, 2, 3, 4)),
            "class_recall_wedfri": class_recall(mask, sample),
            "incident_wedfri": incident_coverage(mask, sample),
        }
    return results


def tables(res: dict) -> str:
    lines = []
    wf = res["sample_wedfri_rows"]
    lines.append(f"Denominators: sample Wednesday to Friday rows = {wf:,}; attack rows in that set = "
                 f"{res['sample_wedfri_attack_rows']:,}.\n")
    lines.append("### Admitted per priority and attacks per priority (Wed–Fri)\n")
    lines.append("| version | P1 admitted / attacks | P2 admitted / attacks | P3 admitted / attacks | total admitted | attacks | precision (Wed–Fri) |")
    lines.append("|---|---|---|---|---|---|---|")
    for ver in ("v1", "v2", "v3", "v3_static"):
        p = res[ver]["priority_wedfri"]
        pr = res[ver]["precision_wedfri"]
        pr_s = f"{pr['precision'] * 100:.2f}% ({pr['attacks_admitted']:,} / {pr['admitted_n']:,})" if pr["admitted_n"] else "n/a"
        lines.append(f"| {ver} | {p['P1']['admitted']:,} / {p['P1']['attacks']:,} | {p['P2']['admitted']:,} / {p['P2']['attacks']:,} | "
                     f"{p['P3']['admitted']:,} / {p['P3']['attacks']:,} | {pr['admitted_n']:,} | {pr['attacks_admitted']:,} | {pr_s} |")
    lines.append("\n### Incident coverage, the headline (Wed–Fri)\n")
    lines.append("| version | (source, class) pairs covered | per-flow attacks admitted | median minutes to first escalation (classes escalated) |")
    lines.append("|---|---|---|---|")
    for ver in ("v1", "v2", "v3", "v3_static"):
        inc = res[ver]["incident_wedfri"]
        med = f"{inc['median_first_escalation_min']:.1f} min ({inc['classes_escalated_n']} classes)" if inc["median_first_escalation_min"] is not None else "n/a"
        lines.append(f"| {ver} | {inc['attack_pairs_covered']} / {inc['attack_pairs_n']} "
                     f"({(inc['pair_coverage'] or 0) * 100:.1f}%) | {inc['attack_flows_admitted']} / {inc['attack_flows_n']:,} "
                     f"({(inc['flow_coverage'] or 0) * 100:.3f}%) | {med} |")
    lines.append("\n### Per-class recall, admitted (Wed–Fri), with n\n")
    classes = sorted(res["v1"]["class_recall_wedfri"].keys())
    lines.append("| class | n (sample) | v1 admitted | v2 admitted | v3 admitted | v3 static admitted | v1 recall | v2 recall | v3 recall | v3 static recall |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in classes:
        n = res["v1"]["class_recall_wedfri"][c]["n"]
        row = [c, f"{n:,}"]
        row += [f"{res[v]['class_recall_wedfri'][c]['admitted']:,}" for v in ("v1", "v2", "v3", "v3_static")]
        row += [f"{res[v]['class_recall_wedfri'][c]['recall'] * 100:.3f}%" for v in ("v1", "v2", "v3", "v3_static")]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    res = build()
    TABLES.write_text(tables(res), encoding="utf-8")
    METRICS.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"wrote {TABLES} and {METRICS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
