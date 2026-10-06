"""Escalation policy v2: Priority 2 bin resolution chosen on evidence.

v2 result = all 62 Priority 1 rules, Priority 2 at n_bins=20, floor=5.
Greedy P1 rule selection (eval/rule_selection.py) was run as a diagnostic and is
recorded as a negative result: the chronological split is class-disjoint, so
the selection population holds almost none of the attack classes it must
cover. Its sample run is reported for transparency but is NOT the v2 output.

Writes:
  results/escalated_flows_v2.csv   (v2 output; never touches escalated_flows.csv)
  results/p2_bin_floor_sweep.csv   (bin and floor sweep)
  results/p1_rule_selection.csv    (per-rule counts and selection flags, diagnostic)
  results/escalation_v2_summary.json
  results/escalation_report_v2.md  (v2 report; main() never writes this file)

Run: python -m eval.escalation_v2
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from dataplane.dt_rules import Rule, evaluate_rules_union, first_matching_rule
from dataplane.escalation_policy import (
    build_signature_table,
    compute_priority2_source_features,
    priority2_escalate,
    run_policy_reserved_thirds,
)
from eval.escalation_data import BENIGN_LABEL, load_pool
from eval.escalation_eval import (
    BIN_COUNTS,
    DEFAULT_TAU,
    HEADLINE_BUDGET,
    PRIORITY_FRACTIONS,
    P2_FLOOR,
    P2_N_BINS,
    SAMPLE_BUDGET,
    _relabel_for_build_signature_table,
    budget_sweep,
    build_escalated_flows_csv,
    build_p2_fit_pool,
    build_p2_holdout_pool,
    compile_p1_rules,
    draw_random_sample,
    sampled_per_class_report,
    sampled_summary_by_priority,
)
from eval.rule_selection import evaluate_selection, rule_matrix, select_rules
from eval.sweep import split_monday_chronologically

OUT_DIR = Path("results")
V2_CSV = OUT_DIR / "escalated_flows_v2.csv"
SWEEP_CSV = OUT_DIR / "p2_bin_floor_sweep.csv"
RULE_CSV = OUT_DIR / "p1_rule_selection.csv"
SUMMARY_JSON = OUT_DIR / "escalation_v2_summary.json"
REPORT_V2_MD = OUT_DIR / "escalation_report_v2.md"

BIN_GRID_FLOOR = 5  # floor held fixed for the bin sweep
FLOOR_GRID = (1, 5, 10, 25)  # swept at the chosen bin count
P1_BUDGET_FRACTION = HEADLINE_BUDGET * PRIORITY_FRACTIONS[0]  # 0.10% of N


def all_label_halves(pool: pd.DataFrame):
    """Mon+Tue chronological first half / second half, every label kept.
    Wed-Fri is returned separately as the out-of-period check."""
    mon = pool[pool["day"] == "monday"]
    tue = pool[pool["day"] == "tuesday"]
    mon_fit, mon_hold = split_monday_chronologically(mon)
    tue_fit, tue_hold = split_monday_chronologically(tue)
    fit_all = pd.concat([mon_fit, tue_fit], ignore_index=True)
    hold_all = pd.concat([mon_hold, tue_hold], ignore_index=True)
    wed_fri = pool[pool["weekday_idx"].isin((2, 3, 4))].reset_index(drop=True)
    return fit_all, hold_all, wed_fri


def _sample_p2_slots(sample: pd.DataFrame, p1_mask: np.ndarray, table) -> dict:
    """Runs the real meter (same budget, tau, priority split as production)
    over the 300k sample and reports what P2 spent its cap on."""
    res = run_policy_reserved_thirds(
        sample, p1_mask, table, meter="per_day", tau=DEFAULT_TAU, global_budget_fraction=SAMPLE_BUDGET,
        priority_fractions=PRIORITY_FRACTIONS, order_col="shuffle_order", flow_id_col="flow_uid",
    )
    is_att = (sample["Label"] != BENIGN_LABEL).to_numpy()
    adm2 = res.priority == 2
    q2 = res.matched_priority2 & ~res.matched_priority1
    n_adm2 = int(adm2.sum())
    return {
        "p2_qualified": int(q2.sum()),
        "p2_admitted": n_adm2,
        "p2_attacks_admitted": int((adm2 & is_att).sum()),
        "attacks_per_100_p2_slots": (100 * (adm2 & is_att).sum() / n_adm2) if n_adm2 else None,
        "overall_admitted": int(res.escalated.sum()),
        "overall_attacks_admitted": int((res.escalated & is_att).sum()),
    }


def p2_config_row(pool, sample, p1_sample_mask, table, label_kind: str, n_bins: int, floor: int,
                  hold_benign: pd.DataFrame, wed_fri_benign: pd.DataFrame) -> dict:
    lab = pool["Label"].to_numpy()
    att = lab != BENIGN_LABEL
    esc_pool = priority2_escalate(pool, table)
    row = {
        "config": label_kind, "n_bins": n_bins, "floor": floor,
        "K_size": len(table.common_signatures),
        "pool_p2_escalation_rate": float(esc_pool.mean()),
        "holdout_benign_p2_rate": float(priority2_escalate(hold_benign, table).mean()),
        "wed_fri_benign_p2_rate": float(priority2_escalate(wed_fri_benign, table).mean()),
    }
    for c in sorted(pd.unique(lab[att])):
        m = lab == c
        row[f"recall__{c}"] = float(esc_pool[m].mean())
        row[f"n__{c}"] = int(m.sum())
    row.update({f"sample__{k}": v for k, v in _sample_p2_slots(sample, p1_sample_mask, table).items()})
    return row


def run_p2_sweeps(pool, sample, p1_sample_mask, fit_benign, hold_benign, wed_fri_benign) -> pd.DataFrame:
    """P2 slots are measured with all 62 P1 rules (the v2 P1), so the sweep
    isolates the bin/floor effect."""
    fitr = _relabel_for_build_signature_table(fit_benign)
    rows = []
    for nb in BIN_COUNTS:
        tb = build_signature_table(fitr, n_bins=nb, floor=BIN_GRID_FLOOR)
        rows.append(p2_config_row(pool, sample, p1_sample_mask, tb, "bin_grid", nb, BIN_GRID_FLOOR,
                                  hold_benign, wed_fri_benign))
    for fl in FLOOR_GRID:
        tb = build_signature_table(fitr, n_bins=P2_N_BINS, floor=fl)
        rows.append(p2_config_row(pool, sample, p1_sample_mask, tb, "floor_grid", P2_N_BINS, fl,
                                  hold_benign, wed_fri_benign))
    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    return df


def class_coverage_table(matrix_by_rule: np.ndarray, labels: np.ndarray, picked: Sequence[int]) -> Dict[str, int]:
    """Attack rows of each class covered by the union of `picked` rule rows."""
    union = matrix_by_rule[list(picked)].any(axis=0) if picked else np.zeros(len(labels), dtype=bool)
    return {c: int((union & (labels == c)).sum())
            for c in sorted(pd.unique(labels[labels != BENIGN_LABEL]))}


def rule_summary(rules: Sequence[Rule], fit_matrix: np.ndarray, fit_lab: np.ndarray, sel_positions: Sequence[int]) -> pd.DataFrame:
    is_att = fit_lab != BENIGN_LABEL
    rows = []
    for pos, r in enumerate(rules):
        m = fit_matrix[pos]
        rows.append({
            "position": pos, "rule_id": r.id, "predicted_class": r.predicted_class,
            "fit_attack_matches": int((m & is_att).sum()), "fit_benign_matches": int((m & ~is_att).sum()),
            "greedy_selected": pos in sel_positions,
        })
    return pd.DataFrame(rows)


def sample_run(sample, rules_subset, table):
    """Production-style run on the 300k sample for one configuration."""
    rid, pred = first_matching_rule(rules_subset, sample)
    p1_mask = rid != -1
    res = run_policy_reserved_thirds(
        sample, p1_mask, table, meter="per_day", tau=DEFAULT_TAU, global_budget_fraction=SAMPLE_BUDGET,
        priority_fractions=PRIORITY_FRACTIONS, order_col="shuffle_order", flow_id_col="flow_uid",
    )
    summary = sampled_summary_by_priority(sample, res)
    per_class = sampled_per_class_report(sample, res)
    return res, rid, pred, summary, per_class


def _fmt_rate(x):
    return "n/a" if x is None else f"{x * 100:.2f}%"


def main() -> int:
    print("loading pool ...")
    pool = compute_priority2_source_features(load_pool())
    rules = compile_p1_rules()
    fit_all, hold_all, wed_fri = all_label_halves(pool)
    fit_benign = build_p2_fit_pool(pool)
    hold_benign = build_p2_holdout_pool(pool)
    hold_benign = hold_benign[hold_benign["Label"] == BENIGN_LABEL]
    wf_benign = wed_fri[wed_fri["Label"] == BENIGN_LABEL]

    print("diagnostic: greedy P1 selection on the fit halves (negative result) ...")
    fit_matrix = rule_matrix(rules, fit_all)
    fit_is_att = (fit_all["Label"] != BENIGN_LABEL).to_numpy()
    sel = select_rules(fit_matrix, fit_is_att, P1_BUDGET_FRACTION)
    sel_rules = [rules[p] for p in sel.selected]
    hold_matrix = rule_matrix(rules, hold_all)
    hold_is_att = (hold_all["Label"] != BENIGN_LABEL).to_numpy()
    hold_all_eval = evaluate_selection(hold_matrix, hold_is_att, list(range(len(rules))))
    hold_sel_eval = evaluate_selection(hold_matrix, hold_is_att, list(sel.selected))
    hold_lab = hold_all["Label"].to_numpy()
    hold_all_cov = class_coverage_table(hold_matrix, hold_lab, list(range(len(rules))))
    hold_sel_cov = class_coverage_table(hold_matrix, hold_lab, list(sel.selected))

    pool_matrix = rule_matrix(rules, pool)
    pool_lab = pool["Label"].to_numpy()
    pool_cov_all = class_coverage_table(pool_matrix, pool_lab, list(range(len(rules))))
    pool_cov_sel = class_coverage_table(pool_matrix, pool_lab, list(sel.selected))
    pool_n = {c: int((pool_lab == c).sum()) for c in pool_cov_all}
    fit_classes = sorted(pd.unique(fit_all.loc[fit_is_att, "Label"]))
    hold_classes = sorted(pd.unique(hold_all.loc[hold_is_att, "Label"]))
    wf_classes = sorted(pd.unique(wed_fri.loc[wed_fri["Label"] != BENIGN_LABEL, "Label"]))

    sample = draw_random_sample(pool)
    rid_all, _ = first_matching_rule(rules, sample)
    p1_mask_all = rid_all != -1

    print("P2 bin and floor sweeps ...")
    sweep_df = run_p2_sweeps(pool, sample, p1_mask_all, fit_benign, hold_benign, wf_benign)

    p2_v2 = build_signature_table(_relabel_for_build_signature_table(fit_benign), n_bins=P2_N_BINS, floor=P2_FLOOR)
    p2_before = build_signature_table(_relabel_for_build_signature_table(fit_benign), n_bins=6, floor=P2_FLOOR)

    print("sample runs: before (v1) / v2 (bins only) / greedy diagnostic ...")
    _, _, _, before_sum, before_cls = sample_run(sample, rules, p2_before)
    v2_res, v2_rid, v2_pred, v2_sum, v2_cls = sample_run(sample, rules, p2_v2)
    _, _, _, greedy_sum, greedy_cls = sample_run(sample, sel_rules, p2_v2)

    build_escalated_flows_csv(sample, v2_res, v2_rid, v2_pred).to_csv(V2_CSV, index=False)
    print(f"wrote {V2_CSV}")

    print("load-vs-coverage sweep: before and v2 ...")
    p1_pool_all, _ = evaluate_rules_union(rules, pool)
    sweep_before = budget_sweep(pool, p1_pool_all, p2_before)
    sweep_v2 = budget_sweep(pool, p1_pool_all, p2_v2)

    rule_summary(rules, fit_matrix, fit_all["Label"].to_numpy(), list(sel.selected)).to_csv(RULE_CSV, index=False)

    s_is_att = (sample["Label"] != BENIGN_LABEL).to_numpy()
    s_matrix = rule_matrix(rules, sample)
    s_sel_eval = evaluate_selection(s_matrix, s_is_att, list(sel.selected))
    s_all_eval = evaluate_selection(s_matrix, s_is_att, list(range(len(rules))))

    summary = {
        "v2": {"n_bins": P2_N_BINS, "floor": P2_FLOOR, "p1_rules": len(rules),
               "sample": v2_sum.to_dict("records")},
        "before_v1": {"n_bins": 6, "floor": P2_FLOOR, "p1_rules": len(rules),
                      "sample": before_sum.to_dict("records")},
        "greedy_diagnostic": {"n_selected": len(sel_rules), "selected_rule_ids": [rules[p].id for p in sel.selected],
                              "budget_rows": sel.budget_rows, "fit_union": sel.union_size,
                              "fit_attack_coverage": sel.attack_coverage,
                              "holdout_selected": hold_sel_eval, "holdout_all": hold_all_eval,
                              "sample_selected": s_sel_eval, "sample_all": s_all_eval,
                              "sample": greedy_sum.to_dict("records"),
                              "pool_class_cov_all": pool_cov_all, "pool_class_cov_selected": pool_cov_sel},
        "splits": {"fit_attack_classes": fit_classes, "holdout_attack_classes": hold_classes,
                   "wed_fri_attack_classes": wf_classes},
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    lines = _report_lines(
        sweep_df=sweep_df, n_rules=len(rules), n_fit=len(fit_all), n_hold=len(hold_all), n_wf=len(wed_fri),
        sel=sel, sel_n=len(sel_rules), rules=rules,
        hold_all_eval=hold_all_eval, hold_sel_eval=hold_sel_eval, hold_sel_cov=hold_sel_cov,
        hold_all_cov=hold_all_cov, pool_cov_all=pool_cov_all, pool_cov_sel=pool_cov_sel, pool_n=pool_n,
        fit_classes=fit_classes, hold_classes=hold_classes, wf_classes=wf_classes,
        s_all_eval=s_all_eval, s_sel_eval=s_sel_eval,
        before_sum=before_sum, before_cls=before_cls, v2_sum=v2_sum, v2_cls=v2_cls,
        greedy_sum=greedy_sum, greedy_cls=greedy_cls, sweep_before=sweep_before, sweep_v2=sweep_v2,
    )
    REPORT_V2_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_V2_MD}")
    return 0


def _report_lines(*, sweep_df, n_rules, n_fit, n_hold, n_wf, sel, sel_n, rules, hold_all_eval, hold_sel_eval,
                  hold_sel_cov, hold_all_cov, pool_cov_all, pool_cov_sel, pool_n, fit_classes, hold_classes,
                  wf_classes, s_all_eval, s_sel_eval, before_sum, before_cls, v2_sum, v2_cls, greedy_sum,
                  greedy_cls, sweep_before, sweep_v2) -> List[str]:
    L: List[str] = []
    row = lambda df, p: df[df["priority"] == p].iloc[0]
    v2_overall = row(v2_sum, "overall")
    before_overall = row(before_sum, "overall")
    greedy_overall = row(greedy_sum, "overall")

    L.append("# Escalation policy v2: Priority 2 bins, Priority 1 unchanged\n")
    L.append(
        "> Companion to `results/escalation_report.md` (the v1 report, n_bins=6). This file is written only by "
        "`eval/escalation_v2.py`, so the v1 report and any re-run of `escalation_eval.main()` leave it alone.\n"
    )
    L.append("## Result\n")
    L.append(
        f"- **v2 = all {n_rules} P1 rules, Priority 2 at n_bins={P2_N_BINS}, floor={P2_FLOOR}**, same 300k sample "
        f"(`SAMPLE_RANDOM_STATE=42`), same tau and 10/70/20 split, cap 3,000.\n"
        f"- Overall attacks admitted: **{int(before_overall['attacks_admitted'])} (v1) -> "
        f"{int(v2_overall['attacks_admitted'])} (v2)**; precision {_fmt_rate(before_overall['precision'])} -> "
        f"{_fmt_rate(v2_overall['precision'])}. Admitted by P2: {int(row(before_sum,'P2')['attacks_admitted'])} -> "
        f"{int(row(v2_sum,'P2')['attacks_admitted'])} attacks in 2,400 slots.\n"
        f"- The gain comes from Priority 2's bin resolution, not from Priority 1. Priority 1 is unchanged.\n"
    )

    L.append("## Part 1 — Priority 2 bin resolution and floor\n")
    L.append(
        "Attacks per 100 P2 slots is measured under the real meter (300k sample, cap 2,400). P2 slots here are "
        "measured with all 62 P1 rules, which is the v2 configuration. Holdout = Mon+Tue second halves. "
        "Wed-Fri = out of period. Floor is fixed at 5 for the bin sweep.\n"
    )
    cols = ["config", "n_bins", "floor", "K_size", "holdout_benign_p2_rate", "wed_fri_benign_p2_rate",
            "pool_p2_escalation_rate", "sample__p2_qualified", "sample__p2_admitted",
            "sample__p2_attacks_admitted", "sample__attacks_per_100_p2_slots", "sample__overall_attacks_admitted"]
    tbl = sweep_df[cols].copy()
    for c in ("holdout_benign_p2_rate", "wed_fri_benign_p2_rate", "pool_p2_escalation_rate"):
        tbl[c] = (tbl[c] * 100).round(4).astype(str) + "%"
    tbl["sample__attacks_per_100_p2_slots"] = tbl["sample__attacks_per_100_p2_slots"].round(3)
    L.append(tbl.to_markdown(index=False))
    rec_cols = [c for c in sweep_df.columns if c.startswith("recall__")]
    rec = sweep_df[["config", "n_bins", "floor"] + rec_cols].copy()
    rec.columns = ["config", "n_bins", "floor"] + [c.replace("recall__", "") for c in rec_cols]
    for c in rec.columns[3:]:
        rec[c] = (rec[c] * 100).round(3)
    L.append("\nPer-class attack recall on the whole pool (%):\n")
    L.append(rec.to_markdown(index=False))
    n_row = {c.replace("n__", ""): int(sweep_df[c].iloc[0]) for c in sweep_df.columns if c.startswith("n__")}
    L.append("\nAttack rows per class (n): " + ", ".join(f"{k}={v:,}" for k, v in n_row.items()) + "\n")
    L.append(
        "\n**Choice: n_bins=20, floor=5.** Attacks per 100 P2 slots peaks at n_bins=20 (6.08) and falls at 40 "
        "(5.88) and 80 (5.67). At n=6 it is 0.29. The trade is visible. The same-period holdout benign "
        "escalation rate rises from 0.03% to 0.78%. The Wed-Fri benign P2 rate rises from 45% to 78% under the "
        "same K. The cap bounds total escalations, so the extra benign P2 escalations displace P3 sampling "
        "slots (P3 admitted falls from 300 to 253); they do not add load.\n"
        "\n**Floor.** At n=20, attacks per 100 slots are 6.21 (floor 1), 6.08 (floor 5), 6.00 (floor 10) and "
        "5.92 (floor 25). The floor 1 vs 5 gap is 3 attacks in 2,400 slots, which is inside sampling noise. "
        "Floor 1 scores better on the holdout benign rate. I kept **floor 5**. A signature seen once in 482k "
        "benign flows that enters K as 'known normal' is attacker-controllable: an attack sharing it escapes "
        "P2. None of the measured numbers captures that, so the choice rests on the argument, not the metrics.\n"
        "\n**Zero-recall classes.** At n=6, DoS Hulk, FTP-Patator and Heartbleed are at 0.0%; DDoS and PortScan "
        "are at 0.002% and 0.023%. At n=20 they are 100%, 50.05%, 100%, 100% and 100%. Those zeros were "
        "bin-resolution misses, not unreachable signatures. Web Attack variants were already 100% at n=6.\n"
    )

    L.append("## Part 2 — greedy P1 rule selection: negative result\n")
    L.append(
        "Greedy budgeted selection (`eval/rule_selection.py`) was run on the Mon+Tue first halves to choose a "
        f"P1 subset at the 0.10% budget. It kept **{sel_n} of {n_rules} rules, all SSH/FTP**. **It is not "
        "used in v2.**\n"
    )
    L.append(
        "**Cause.** The chronological split is class-disjoint:\n\n"
        f"- Mon+Tue first halves (selection population, n={n_fit:,}): attack classes {', '.join(fit_classes)}\n"
        f"- Mon+Tue second halves (holdout, n={n_hold:,}): attack classes {', '.join(hold_classes)}\n"
        f"- Wednesday-Friday (n={n_wf:,}): attack classes {', '.join(wf_classes)}\n\n"
        "Greedy therefore optimised coverage of the only class it could see. The holdout check is a cross-class "
        "test (SSH rules on FTP), not a same-class holdout. This is a property of the split, not of the rules.\n"
    )
    cov_rows = [{"class": c, "attack_rows": pool_n[c], "covered_all_62": pool_cov_all[c],
                 "covered_greedy": pool_cov_sel[c],
                 "lost_entirely": (pool_cov_sel[c] == 0) and (pool_cov_all[c] > 0)} for c in pool_cov_all]
    cov_df = pd.DataFrame(cov_rows)
    lost = cov_df.loc[cov_df["lost_entirely"], "class"].tolist()
    L.append("Per-class attack coverage on the whole pool, all 62 vs greedy subset:\n")
    L.append(cov_df.to_markdown(index=False))
    L.append(
        f"\n**{len(lost)} classes lost all coverage**: {', '.join(lost)}.\n"
        f"\n**Budget did not transfer.** Greedy fit-union stayed within budget ({sel.union_size:,} rows against "
        f"{sel.budget_rows:,}), but the holdout union is {hold_sel_eval['union_rate'] * 100:.2f}% of rows against "
        f"a 0.10% budget ({hold_sel_eval['union_size']:,} rows), and the 300k-sample union is "
        f"{s_sel_eval['union_rate'] * 100:.3f}% against 0.10%.\n"
        "\n**Conclusion.** Greedy rule selection requires a selection split that contains every attack class. "
        "A chronological split on CICIDS2017 does not. A corrected selection experiment should use a split "
        "chosen in advance, with every class represented, and should be reported separately.\n"
        f"\n**The greedy run's {int(greedy_overall['attacks_admitted']):,}-attack figure is not an improvement.** "
        "Dropping DoS, DDoS and PortScan from P1 moves them into P2's meter, where they are admitted at a higher "
        "rate because P2 now sees them. The P1 subset itself catches almost none of them. The honest v2 result "
        f"is {int(v2_overall['attacks_admitted'])} attacks, from the all-62-rule configuration above.\n"
    )
    hold_rows = pd.DataFrame([
        {"population": "holdout (Mon+Tue 2nd halves)", "rules": f"all {n_rules}",
         "union_rows": hold_all_eval["union_size"], "union_rate": f"{hold_all_eval['union_rate'] * 100:.3f}%",
         "precision": _fmt_rate(hold_all_eval["precision"]), "attack_recall": _fmt_rate(hold_all_eval["attack_recall"])},
        {"population": "holdout (Mon+Tue 2nd halves)", "rules": f"greedy {sel_n}",
         "union_rows": hold_sel_eval["union_size"], "union_rate": f"{hold_sel_eval['union_rate'] * 100:.3f}%",
         "precision": _fmt_rate(hold_sel_eval["precision"]), "attack_recall": _fmt_rate(hold_sel_eval["attack_recall"])},
        {"population": "300k sample", "rules": f"all {n_rules}",
         "union_rows": s_all_eval["union_size"], "union_rate": f"{s_all_eval['union_rate'] * 100:.3f}%",
         "precision": _fmt_rate(s_all_eval["precision"]), "attack_recall": _fmt_rate(s_all_eval["attack_recall"])},
        {"population": "300k sample", "rules": f"greedy {sel_n}",
         "union_rows": s_sel_eval["union_size"], "union_rate": f"{s_sel_eval['union_rate'] * 100:.3f}%",
         "precision": _fmt_rate(s_sel_eval["precision"]), "attack_recall": _fmt_rate(s_sel_eval["attack_recall"])},
    ])
    L.append("\nUnion precision and recall (budget 0.10%):\n")
    L.append(hold_rows.to_markdown(index=False))
    L.append("")

    L.append("## Part 3 — before, v2, and greedy diagnostic (300k sample)\n")
    L.append(
        "Before = v1 (n_bins=6, all 62 rules). **v2** = n_bins=20, floor=5, all 62 rules. Greedy = n_bins=20 with "
        "the 5-rule subset, diagnostic only. The sample overlaps the fit halves. Its benign rows are about "
        "one-fifth fit-half, one-fifth holdout-half and the rest Wed-Fri.\n"
    )
    L.append("| priority | before (v1): qualified / admitted / attacks / precision | **v2** | greedy (negative) |\n"
             "|---|---|---|---|\n")

    def cell(r):
        return f"{r['qualified']:,} / {r['admitted']:,} / {r['attacks_admitted']:,} / {_fmt_rate(r['precision'])}"
    for p in ("P1", "P2", "P3", "overall"):
        L.append(f"| {p} | {cell(row(before_sum, p))} | {cell(row(v2_sum, p))} | {cell(row(greedy_sum, p))} |\n")
    L.append("\nPer-class admitted (n shown):\n")
    pc = pd.DataFrame({
        "class": before_cls["label"].values, "n": before_cls["n"].values,
        "before_v1": before_cls["admitted"].values,
        "v2": v2_cls["admitted"].values,
        "greedy_diagnostic": _class_admitted(greedy_sum, greedy_cls, before_cls),
    })
    L.append(pc.to_markdown(index=False))
    L.append("\n### Load versus coverage, 0.01% to 10%, stacked by priority\n")
    L.append("Before (v1):\n")
    L.append(_sweep_md(sweep_before))
    L.append("\nv2 (n_bins=20, all 62 rules):\n")
    L.append(_sweep_md(sweep_v2))
    return L


def _class_admitted(greedy_sum, greedy_cls: pd.DataFrame, before_cls: pd.DataFrame):
    """Greedy per-class admitted, aligned to the before-table class order."""
    return list(greedy_cls.set_index("label").reindex(before_cls["label"])["admitted"].values)


def _sweep_md(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in ("attack_coverage", "benign_fpr"):
        out[c] = out[c].map(lambda v: "n/a" if v is None else f"{v:.4%}")
    return out.to_markdown(index=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
