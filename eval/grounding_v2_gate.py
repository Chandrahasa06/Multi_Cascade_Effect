"""Part 0 gate: does the v2 isolation score separate attacks from benign better than the
current neighbourhood count, on the 44 scored records? No API calls.

Pre-registered rule (fixed before running): the gate PASSES iff the AUC of the primary
isolation score (k=50, self-calibrated, all-days benign reference) is strictly greater
than the AUC of the current pipeline count (the count the v3 run used, as stored).
Higher isolation -> more attack-like; lower count -> more attack-like.

Caveat stated in the report: the all-days reference is the same population the test
flows were drawn from (in-sample for their days). This gate tests whether the signal
carries information at all. The deployment-realistic test is the Monday and Monday+Tuesday
arms, which run only if this gate passes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from agents import grounding_v2 as g2
from eval.escalation_data import BENIGN_LABEL
from agents import evidence_signals  # for the slim pool loader only
from eval.run_blind_pipeline_45_v2 import load_45

V3_RUN = Path("results/agent_run_45_v3.jsonl")
KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]
OUT_MD = Path("results/grounding_v2_validation.md")
OUT_JSON = Path("results/grounding_v2_validation.json")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    pool = evidence_signals.load_pool_slim(extra_columns=())
    benign = pool.loc[pool["Label"] == BENIGN_LABEL]
    ref = g2.build_reference_v2(benign, label="all-days")
    print(f"reference {ref.n} benign rows; radius {ref.radius:.6f}; calibration {len(ref.calibration[g2.PRIMARY_K])}")

    blinds = {b["record_id"]: b for b in load_45()}
    v3 = {json.loads(l)["record_id"]: json.loads(l) for l in open(V3_RUN, encoding="utf-8") if l.strip()}
    truth = {}
    for f in KEYS:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            truth[r["record_id"]] = r["is_attack"] == "True"
    ids = sorted(v3)
    assert len(ids) == 44

    rows = []
    for rid in ids:
        s = g2.score_flow(ref, blinds[rid]["features"])
        rows.append({
            "record_id": rid, "is_attack": truth[rid],
            "isolation": s["isolation_percentile"],
            "iso_k10": s["isolation_percentile_sensitivity"]["10"],
            "iso_k100": s["isolation_percentile_sensitivity"]["100"],
            "radius_count": s["radius_count"],
            "current_count": int(v3[rid]["escalation_neighbourhood"]["neighbourhood_size"]),
            "current_count_loo": int(v3[rid]["evidence_signals"]["count_excl"]),
            "per_feature": s["per_feature"][:3],
        })

    y = np.array([r["is_attack"] for r in rows], dtype=int)
    auc_iso = roc_auc_score(y, [r["isolation"] for r in rows])
    auc_iso10 = roc_auc_score(y, [r["iso_k10"] for r in rows])
    auc_iso100 = roc_auc_score(y, [r["iso_k100"] for r in rows])
    auc_rad = roc_auc_score(y, [-r["radius_count"] for r in rows])
    auc_cur = roc_auc_score(y, [-r["current_count"] for r in rows])
    auc_cur_loo = roc_auc_score(y, [-r["current_count_loo"] for r in rows])
    passed = auc_iso > auc_cur
    summary = {
        "n": len(rows), "attacks": int(y.sum()), "benign": int(len(y) - y.sum()),
        "auc_primary_k50": auc_iso, "auc_k10": auc_iso10, "auc_k100": auc_iso100,
        "auc_radius_count": auc_rad, "auc_current_count": auc_cur, "auc_current_count_loo": auc_cur_loo,
        "gate_passes": bool(passed), "reference_size": ref.n, "radius": ref.radius,
    }
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str), encoding="utf-8")

    L = []
    L.append("# Grounding v2, Part 0 gate\n")
    L.append("No API calls. Reference: all-days benign, 2,273,097 rows (the same population as the current count). Rule fixed before running: **pass iff AUC(primary isolation score, k=50) > AUC(current pipeline count).**\n")
    L.append("Caveat: this reference is in-sample for the test flows' own days. The gate asks whether the signal carries information at all. Deployment realism is tested by the arms, which run only if this gate passes.\n")
    L.append("## Result\n")
    L.append(f"- AUC, primary isolation score (k=50): **{auc_iso:.3f}**")
    L.append(f"- AUC, current pipeline count (as stored in the v3 run): {auc_cur:.3f}")
    L.append(f"- AUC, current count with leave-one-out applied: {auc_cur_loo:.3f}")
    L.append(f"- AUC, sensitivity k=10 / k=100 (not used for the decision): {auc_iso10:.3f} / {auc_iso100:.3f}")
    L.append(f"- AUC, radius count (benign-derived radius {ref.radius:.6f}): {auc_rad:.3f}")
    L.append(f"- **Gate: {'PASSES' if passed else 'FAILS'}**. Stop and report if it fails; no agent run is justified.\n")
    L.append(f"n = {len(rows)}: {summary['attacks']} attacks, {summary['benign']} benign. One record moves AUC by up to about 1/(29*15).\n")
    L.append("## Distribution by true label (primary isolation score, higher = more isolated)\n")
    for lab in (True, False):
        vals = sorted(round(r["isolation"], 2) for r in rows if r["is_attack"] == lab)
        L.append(f"- {'attacks' if lab else 'benign'} ({len(vals)}): {vals}")
    L.append("")
    L.append("## Current count distribution by true label (for comparison)\n")
    for lab in (True, False):
        vals = sorted(r["current_count"] for r in rows if r["is_attack"] == lab)
        L.append(f"- {'attacks' if lab else 'benign'} ({len(vals)}): {vals}")
    L.append("")
    L.append("## Per record\n")
    L.append("| record | truth | isolation (k=50) | radius count | current count | leave-one-out count | top features by gap |")
    L.append("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["is_attack"], r["isolation"])):
        top = "; ".join(f"{p['feature']} {p['gap']:+.2f}" for p in r["per_feature"])
        L.append(f"| {r['record_id'][:10]} | {'attack' if r['is_attack'] else 'benign'} | {r['isolation']:.1f} | {r['radius_count']:,} | {r['current_count']:,} | {r['current_count_loo']:,} | {top} |")
    L.append("")
    L.append("Per-feature gap: this flow's percentile minus the median percentile of its 50 nearest benign flows, on each feature. Positive means this flow sits above its neighbourhood on that feature.\n")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"AUC iso50={auc_iso:.3f} current={auc_cur:.3f} -> gate {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
