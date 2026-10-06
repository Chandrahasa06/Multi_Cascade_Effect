"""Scoring for the v3 re-run (multi-signal evidence) on the same 45 records.

Compares:
  before: agent_run_45_v2.jsonl re-scored with the corrected cap (0.29), no API calls
  after:  agent_run_45_v3.jsonl (evidence block to A1, A3, A5; corrected cap)
  control: logistic regression on the same signals (eval/logistic_baseline.py)
  reference: the count-only rule (old count, and leave-one-out count)

Writes results/agent_scoring_45_v3.md and results/agent_scoring_45_v3.json.
Counts primary, Wilson 95% intervals on every rate. n = 45 is small.
This is a re-run after a change, not an independent measurement.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from agents.schema import VerdictLabel, derive_verdict
from eval.count_only_baseline import corrected_agent_bp

V2_RUN = Path("results/agent_run_45_v2.jsonl")
V3_RUN = Path("results/agent_run_45_v3.jsonl")
V3_TALLY = Path("results/agent_run_45_v3_tally.json")
LOGISTIC = Path("results/logistic_baseline_45.json")
KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]
OUT_MD = Path("results/agent_scoring_45_v3.md")
OUT_JSON = Path("results/agent_scoring_45_v3.json")
THRESHOLD = 0.30
GROUNDED_ATTACKS_PREVIOUSLY_CLEARED = ["2c27e528", "5b397707", "70448e2e", "b32f2676", "d07e7212", "d67a88f4"]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    pct = f"{100 * k / n:.1f}%" if n else "n/a"
    return f"{k}/{n} = {pct} (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def load_jsonl(path: Path) -> Dict[str, dict]:
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["record_id"]] = r
    return out


def load_keys() -> Dict[str, dict]:
    out = {}
    for p in KEYS:
        for row in csv.DictReader(open(p, encoding="utf-8")):
            out[row["record_id"]] = {"true_label": row["true_label"], "is_attack": row["is_attack"] == "True"}
    return out


def confusion(pred: Dict[str, bool], truth: Dict[str, bool], ids: List[str]) -> dict:
    tp = sum(1 for i in ids if truth[i] and pred[i])
    fn = sum(1 for i in ids if truth[i] and not pred[i])
    fp = sum(1 for i in ids if not truth[i] and pred[i])
    tn = sum(1 for i in ids if not truth[i] and not pred[i])
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def main() -> int:
    v2 = load_jsonl(V2_RUN)
    v3 = load_jsonl(V3_RUN)
    keys = load_keys()
    ids = sorted(v3)
    # 44 of 45 when a record failed schema validation in the v3 run; reported below, not dropped
    assert len(ids) in (44, 45) and set(ids) <= set(v2) and set(ids) <= set(keys), (len(ids), len(v2))
    failed_v3 = sorted(set(v2) - set(ids))
    truth = {i: keys[i]["is_attack"] for i in ids}
    n_att = sum(truth.values())
    n_ben = len(ids) - n_att
    tally = json.loads(V3_TALLY.read_text(encoding="utf-8")) if V3_TALLY.exists() else {}
    logi = json.loads(LOGISTIC.read_text(encoding="utf-8"))

    rec_v2_recorded = {i: float(v2[i]["a5"]["benign_plausibility"]) for i in ids}
    rec_v2_corr = {i: corrected_agent_bp(v2[i], 0.29, 0.29)[0] for i in ids}
    rec_v3 = {i: float(v3[i]["a5"]["benign_plausibility"]) for i in ids}
    flag = lambda bp: bp < THRESHOLD  # same rule as derive_verdict's ANOMALOUS_AND_UNEXPLAINED
    assert all((derive_verdict(rec_v3[i]) is VerdictLabel.ANOMALOUS_AND_UNEXPLAINED) == flag(rec_v3[i]) for i in ids)

    pred_v2_rec = {i: flag(rec_v2_recorded[i]) for i in ids}
    pred_v2_corr = {i: flag(rec_v2_corr[i]) for i in ids}
    pred_v3 = {i: flag(rec_v3[i]) for i in ids}
    p_lr = {i: logi["scores"][i]["p_real"] for i in ids}
    pred_lr = {i: p_lr[i] < THRESHOLD for i in ids}
    cnt_old = {i: int(v3[i]["evidence_signals"]["count_all"]) for i in ids}
    cnt_loo = {i: int(v3[i]["evidence_signals"]["count_excl"]) for i in ids}
    pred_count_old = {i: cnt_old[i] < 30 for i in ids}
    pred_count_loo = {i: cnt_loo[i] < 30 for i in ids}

    conf = {
        "v2 recorded (cap 0.30)": confusion(pred_v2_rec, truth, ids),
        "v2 corrected (cap 0.29)": confusion(pred_v2_corr, truth, ids),
        "v3 (evidence, cap 0.29)": confusion(pred_v3, truth, ids),
        "logistic control": confusion(pred_lr, truth, ids),
        "count-only (old count)": confusion(pred_count_old, truth, ids),
        "count-only (leave-one-out count)": confusion(pred_count_loo, truth, ids),
    }

    def spread(values: Dict[str, float]) -> dict:
        c = Counter(round(v, 4) for v in values.values())
        mid = sum(1 for v in values.values() if 0.35 <= v <= 0.65)
        return {"distinct": len(c), "middle_0.35_0.65": mid, "counts": dict(sorted(c.items()))}

    flips = [i for i in ids if pred_v2_corr[i] != pred_v3[i]]
    six = {}
    for pref in GROUNDED_ATTACKS_PREVIOUSLY_CLEARED:
        rid = next(i for i in ids if i.startswith(pref))
        six[rid] = {"count_old": cnt_old[rid], "v2_corr": rec_v2_corr[rid], "v3": rec_v3[rid],
                    "changed_call": pred_v2_corr[rid] != pred_v3[rid], "v3_rationale": v3[rid]["a5"]["rationale"]}

    disagree_lr = [i for i in ids if pred_lr[i] != pred_v3[i]]
    disagree_count = [i for i in ids if pred_count_loo[i] != pred_v3[i]]
    status = Counter(v3[i].get("evidence_source_status", "unknown") for i in ids)

    result = {
        "n": len(ids), "failed_v3": failed_v3, "n_attacks": n_att, "n_benign": n_ben, "confusion": conf,
        "spread_v2_corrected": spread(rec_v2_corr), "spread_v3": spread(rec_v3),
        "flips_v2c_to_v3": [(i, rec_v2_corr[i], rec_v3[i], truth[i]) for i in flips],
        "six_grounded": six, "disagree_logistic_vs_v3": disagree_lr,
        "disagree_countloo_vs_v3": disagree_count,
        "tally": tally, "source_status": dict(status), "logistic_validation_accuracy": logi["validation_accuracy_real_vs_synthetic"],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    write_md(ids, truth, n_att, n_ben, v2, v3, keys, rec_v2_recorded, rec_v2_corr, rec_v3, p_lr, pred_lr,
             cnt_old, cnt_loo, pred_v2_corr, pred_v3, conf, flips, six, disagree_lr, disagree_count,
             tally, logi, status, spread)
    print(f"wrote {OUT_MD} and {OUT_JSON}")
    return 0


def write_md(ids, truth, n_att, n_ben, v2, v3, keys, rec_v2_recorded, rec_v2_corr, rec_v3, p_lr, pred_lr,
             cnt_old, cnt_loo, pred_v2_corr, pred_v3, conf, flips, six, disagree_lr, disagree_count,
             tally, logi, status, spread) -> None:
    L: List[str] = []
    L.append("# Agent pipeline with multi-signal evidence (v3), on the same 45 records\n")
    L.append("**A re-run after a change, not an independent measurement.** Same 45 records, blind inputs, model, temperature and 0.30 threshold. The change: A1, A3 and A5 receive four benign-only signals (percentiles, nearest benign flows, source history, directional deviation) beside the existing count. The cap is 0.29 (corrected, see below).\n")
    L.append(f"n = {len(ids)} ({n_att} attacks, {n_ben} benign). Counts primary, Wilson 95% intervals on every rate.\n")
    for r in sorted(set(v2) - set(ids)):
        bp_old = corrected_agent_bp(v2[r], 0.29, 0.29)[0]
        L.append(f"**Not scored in v3: {r[:10]} ({keys[r]['true_label']}) failed schema validation after retries in A3** (the flow-level filter and the feature-vocabulary check rejected every attempt). Its v2 corrected score was {bp_old:.2f}. Scores below are on the paired subset.\n")

    L.append("## 0. Calls made\n")
    L.append(f"- API calls this run: {tally.get('api_calls_made')} (including schema retries), cache hits: {tally.get('cache_hits')}, schema retries: {tally.get('schema_retries')}, failed records: {len(tally.get('failed_records', {}))}, stopped on quota: {tally.get('stopped_on_quota')}.")
    L.append(f"- Source IP and time recovered for flows: {dict(status)}. Where the match was ambiguous, source history is unavailable and the note says so.\n")

    L.append("## 1. Confusion matrices (counts)\n")
    L.append("| method | TP (attack caught) | FN | FP (benign flagged) | TN |")
    L.append("|---|---|---|---|---|")
    for name, c in conf.items():
        L.append(f"| {name} | {c['tp']} | {c['fn']} | {c['fp']} | {c['tn']} |")
    L.append("")
    L.append("Attack recall and benign specificity:\n")
    L.append("| method | attacks caught | benign cleared |")
    L.append("|---|---|---|")
    for name, c in conf.items():
        L.append(f"| {name} | {rate(c['tp'], n_att)} | {rate(c['tn'], n_ben)} |")
    L.append("")
    L.append("The count-only leave-one-out row removes one identical benign copy per flow, as the agents' evidence does. The old count does not. Benign flows always find themselves in the reference, so the old count is one higher for every benign flow. On these 44 records the difference changes nothing: attacks caught and benign flagged are identical under both counts at every cutoff in the sweep (0, 1, 5, 10, 30, 100, 500). The smallest benign count is 2,069, far above any cutoff tested, so the self-copy never moves a benign flow across a cutoff. An earlier version of this note said it inflated the low-cutoff sweep; that was not borne out.\n")

    L.append("## 2. Does the output spread?\n")
    L.append("| score set | distinct values | scores in 0.35-0.65 | value counts |")
    L.append("|---|---|---|---|")
    sp_before = spread(rec_v2_corr)
    sp_after = spread(rec_v3)
    L.append(f"| v2 corrected | {sp_before['distinct']} | {sp_before['middle_0.35_0.65']} of {len(ids)} | {sp_before['counts']} |")
    L.append(f"| v3 | {sp_after['distinct']} | {sp_after['middle_0.35_0.65']} of {len(ids)} | {sp_after['counts']} |\n")

    L.append("## 3. Sorted plausibility distributions\n")
    L.append("| group | v2 corrected | v3 |")
    L.append("|---|---|---|")
    for lab, is_att in [("attacks", True), ("benign", False)]:
        sel = [i for i in ids if truth[i] == is_att]
        L.append(f"| {lab} | {sorted(round(rec_v2_corr[i], 2) for i in sel)} | {sorted(round(rec_v3[i], 2) for i in sel)} |")
    L.append("")

    L.append("## 4. Per-record: old and new score\n")
    L.append("| record | true label | count (old) | leave-one-out count | v2 recorded | v2 corrected | v3 | logistic p | v2 verdict -> v3 verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i in ids:
        v2v = "attack" if pred_v2_corr[i] else "benign"
        v3v = "attack" if pred_v3[i] else "benign"
        L.append(f"| {i[:10]} | {keys[i]['true_label']} | {cnt_old[i]} | {cnt_loo[i]} | {rec_v2_recorded[i]:.2f} | {rec_v2_corr[i]:.2f} | {rec_v3[i]:.2f} | {p_lr[i]:.2f} | {v2v} -> {v3v} |")
    L.append("")

    L.append("## 5. Verdict flips (v2 corrected -> v3), with A5's reasoning\n")
    if not flips:
        L.append("No record changed its verdict.\n")
    for i in flips:
        L.append(f"### {i[:10]}  true label: {keys[i]['true_label']}\n")
        L.append(f"- score: {rec_v2_corr[i]:.2f} -> {rec_v3[i]:.2f}; count {cnt_old[i]} (leave-one-out {cnt_loo[i]})")
        L.append(f"- A5 reasoning (v3): {v3[i]['a5']['rationale']}\n")

    L.append("## 6. The six grounded attacks the agents cleared before\n")
    L.append("| record | count | v2 corrected | v3 | call changed | v3 A5 reasoning (truncated to 400 characters) |")
    L.append("|---|---|---|---|---|---|")
    for i, s in six.items():
        L.append(f"| {i[:10]} | {s['count_old']} | {s['v2_corr']:.2f} | {s['v3']:.2f} | {'yes' if s['changed_call'] else 'no'} | {s['v3_rationale'][:400]} |")
    L.append("")

    L.append("## 7. Control: logistic regression on the same signals\n")
    L.append(f"Fitted without attack data: real benign flows against synthetic flows that break the joint distribution. Validation accuracy on that task: {logi['validation_accuracy_real_vs_synthetic']:.3f}. This is a benign-only density-ratio classifier, not an attack classifier. Scored against the same 0.30 rule (p(real) below 0.30 means attack).\n")
    L.append("Full disagreements with the v3 agents, printed in full:\n")
    if not disagree_lr:
        L.append("None.\n")
    for i in disagree_lr:
        who = "logistic attack, agents benign" if pred_lr[i] else "agents attack, logistic benign"
        L.append(f"### {i[:10]}  true label: {keys[i]['true_label']} ({who})\n")
        L.append(f"- logistic p(real) {p_lr[i]:.3f}; agents score {rec_v3[i]:.2f}; count {cnt_old[i]} (leave-one-out {cnt_loo[i]})")
        L.append(f"- A5 reasoning: {v3[i]['a5']['rationale']}\n")

    L.append("## 8. Leave-one-out count vs agents\n")
    L.append(f"Disagreements between the leave-one-out count rule and v3: {len(disagree_count)} of {len(ids)}. Printed in full below.\n")
    for i in disagree_count:
        who = "count attack, agents benign" if pred_count_loo[i] else "agents attack, count benign"
        L.append(f"- {i[:10]} ({keys[i]['true_label']}, {who}): count {cnt_loo[i]}, agents {rec_v3[i]:.2f}")
    L.append("")

    L.append("## 9. Notes and open items\n")
    L.append("- Cap mismatch, not fixed here: the A5 prompts (a5_verdict_v5 and v6) still state the cap as 0.3. The code clamps at 0.29. A5 can emit 0.30, which the cap then clamps to 0.29.")
    L.append("- The count-only baseline's sweep at low cutoffs used the old count, which includes the flow's own benign copy. See section 1.")
    L.append("- Source IP and time are recovered by matching a blind flow's features against the pool (see `agents/evidence_signals.py`). That match reads the flow's own metadata row, including attack rows, but never reads its label. Attack rows are not used as evidence. This is the one place the evidence path touches the pool beyond the benign reference, and it is noted here rather than hidden. A deployed controller would have the source IP and time directly.")
    L.append("- The control's training sample excludes the 45 evaluation flows by feature identity.")
    L.append("- Limits: n = 45. One record moves a rate by 2.2 points.\n")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
