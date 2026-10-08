"""Scoring for the reference arms (A, M, MT) against the count-only control on the same
window and the same records. Writes results/agent_reference_arms.md and a JSON.

Counts primary. Wilson 95% intervals on every rate. Arms are smaller than 44; say so.
Labels are read here, in scoring, never in any signal.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]
ARMS = {"a": "all-days", "m": "monday", "mt": "monday+tuesday"}
V3 = Path("results/agent_run_45_v3.jsonl")
OUT_MD = Path("results/agent_reference_arms.md")
OUT_JSON = Path("results/agent_reference_arms.json")
SIX = ["2c27e528", "5b397707", "70448e2e", "b32f2676", "d07e7212", "d67a88f4"]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> str:
    if n == 0:
        return f"{k}/{n} (no records)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def load_jsonl(p: Path) -> Dict[str, dict]:
    if not p.exists():
        return {}
    return {json.loads(l)["record_id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def load_keys() -> Dict[str, Tuple[str, bool]]:
    out = {}
    for p in KEYS:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            out[r["record_id"]] = (r["true_label"], r["is_attack"] == "True")
    return out


def tally_score(rows: List[dict], truth: Dict[str, Tuple[str, bool]], pred_key: str):
    att = [r for r in rows if truth[r["record_id"]][1]]
    ben = [r for r in rows if not truth[r["record_id"]][1]]
    caught = sum(1 for r in att if r[pred_key] == "attack")
    cleared = sum(1 for r in ben if r[pred_key] == "benign")
    return caught, len(att), cleared, len(ben)


def main() -> int:
    truth = load_keys()
    arms = {a: load_jsonl(Path(f"results/agent_run_arm_{a}.jsonl")) for a in ARMS}
    tallies = {a: json.loads(Path(f"results/agent_run_arm_{a}_tally.json").read_text(encoding="utf-8"))
               if Path(f"results/agent_run_arm_{a}_tally.json").exists() else {} for a in ARMS}
    v3 = load_jsonl(V3)
    common = None
    for a in ARMS:
        ids = set(arms[a])
        common = ids if common is None else common & ids
    common = sorted(common)

    L: List[str] = []
    L.append("# Reference arms: all-days, Monday, Monday+Tuesday\n")
    L.append("**Setup.** A5 is the only model call per record. Upstream A1 to A4 are the recorded v2 outputs; their grounding block is the Monday Tier-1 reference for every arm, so the upstream is identical across arms. A5 uses the corrected prompt (`a5_verdict_v7`, cap 0.29), the real leave-one-out count, and profile sentences drawn from the arm's reference window. The evidence block is rebuilt from the same window.\n")
    L.append("**Honesty.** Counts primary; Wilson 95% intervals on every rate. Arms are smaller than 44. Monday is one day of reference tested up to four days later, so Arm M is a lower bound, not a deployment estimate. All-days grounding was not deployment-realistic: it includes the same days as the attacks it is tested on.\n")

    L.append("## 1. Qualifying records per arm\n")
    L.append("| arm | window | evaluated | dropped | attacks | benign |")
    L.append("|---|---|---|---|---|---|")
    for a, label in ARMS.items():
        rows = list(arms[a].values())
        na = sum(truth[r["record_id"]][1] for r in rows)
        nb = len(rows) - na
        t = tallies[a]
        L.append(f"| {a.upper()} | {label} | {len(rows)} of {t.get('records_total', 45)} | {len(t.get('dropped', []))} | {na} | {nb} |")
    L.append("")
    for a in ARMS:
        drops = tallies[a].get("dropped", [])
        if drops:
            L.append(f"Dropped from arm {a.upper()}:\n")
            for rid, why in drops:
                lab = truth.get(rid, ("?", False))
                L.append(f"- {rid[:10]} ({lab[0]}): {why}")
            L.append("")
    L.append(f"**Common subset** present in all three arms: {len(common)} records "
             f"({sum(truth[r][1] for r in common)} attacks, {len(common) - sum(truth[r][1] for r in common)} benign).\n")

    L.append("## 2. Agents and count-only control, per arm\n")
    L.append("| arm | attacks caught, agents | attacks caught, count-only | benign cleared, agents | benign cleared, count-only | disagreements |")
    L.append("|---|---|---|---|---|---|")
    per_arm = {}
    for a in ARMS:
        rows = list(arms[a].values())
        c1, na, cl1, nb = tally_score(rows, truth, "verdict")
        c2, _, cl2, _ = tally_score(rows, truth, "count_only_verdict")
        dis = [r for r in rows if r["verdict"] != r["count_only_verdict"]]
        per_arm[a] = {"n": len(rows), "agents": [c1, na, cl1, nb], "count_only": [c2, na, cl2, nb], "disagree": len(dis)}
        L.append(f"| {a.upper()} | {rate(c1, na)} | {rate(c2, na)} | {rate(cl1, nb)} | {rate(cl2, nb)} | {len(dis)} |")
    L.append("")

    L.append("## 3. Common subset, identical flows in all three arms\n")
    L.append("| method | attacks caught | benign cleared |")
    L.append("|---|---|---|")
    for a in ARMS:
        rows = [arms[a][r] for r in common]
        c1, na, cl1, nb = tally_score(rows, truth, "verdict")
        c2, _, cl2, _ = tally_score(rows, truth, "count_only_verdict")
        L.append(f"| arm {a.upper()} agents | {rate(c1, na)} | {rate(cl1, nb)} |")
        L.append(f"| arm {a.upper()} count-only | {rate(c2, na)} | {rate(cl2, nb)} |")
    L.append("")

    L.append("## 4. Disagreements, printed in full\n")
    for a in ARMS:
        rows = [r for r in arms[a].values() if r["verdict"] != r["count_only_verdict"]]
        L.append(f"### Arm {a.upper()} ({len(rows)})\n")
        if not rows:
            L.append("None.\n")
        for r in sorted(rows, key=lambda r: r["record_id"]):
            lab = truth[r["record_id"]]
            L.append(f"- **{r['record_id'][:10]}**, true label {lab[0]} ({'attack' if lab[1] else 'benign'}); "
                     f"count (leave-one-out) {r['count_loo']} of R={r['R']}; "
                     f"agents: {r['benign_plausibility']:.2f} -> **{r['verdict']}**; count-only: **{r['count_only_verdict']}**.")
            L.append(f"  A5: {r['rationale']}\n")

    L.append("## 5. Six previously-missed grounded attacks (old counts 30 to 117)\n")
    L.append("| record | old count | v3 score | arm A | arm M | arm MT |")
    L.append("|---|---|---|---|---|---|")
    for pref in SIX:
        rid = next((r for r in list(arms["a"]) if r.startswith(pref)), None)
        if rid is None:
            continue
        v3s = f"{float(v3[rid]['a5']['benign_plausibility']):.2f}" if rid in v3 else "n/a"
        cells = []
        for a in ARMS:
            r = arms[a].get(rid)
            cells.append("not evaluated" if r is None else f"{r['benign_plausibility']:.2f} ({r['verdict']}, count {r['count_loo']})")
        old = v3[rid]["escalation_neighbourhood"]["neighbourhood_size"] if rid in v3 else "n/a"
        L.append(f"| {rid[:10]} | {old} | {v3s} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")

    L.append("## 6. Does the real count change anything? (Arm A against the recorded all-days result)\n")
    flips, changed = [], 0
    for rid, r in arms["a"].items():
        if rid not in v3:
            continue
        old_bp = float(v3[rid]["a5"]["benign_plausibility"])
        if abs(old_bp - r["benign_plausibility"]) > 1e-9:
            changed += 1
        old_v = "attack" if old_bp < THRESHOLD else "benign"
        if old_v != r["verdict"]:
            flips.append((rid, old_bp, r["benign_plausibility"], r))
    L.append(f"Scores that changed between the recorded v3 run and arm A: {changed} of {len([r for r in arms['a'] if r in v3])} records in both. Verdict flips: {len(flips)}.\n")
    for rid, ob, nb_, r in flips:
        lab = truth[rid]
        L.append(f"- {rid[:10]} (true {lab[0]}): {ob:.2f} -> {nb_:.2f}. A5: {r['rationale']}\n")

    L.append("## 7. Score spread per arm\n")
    L.append("| arm | distinct benign-plausibility values | scores in 0.35-0.65 | value counts |")
    L.append("|---|---|---|---|")
    for a in ARMS:
        vals = [round(float(r["benign_plausibility"]), 2) for r in arms[a].values()]
        mid = sum(1 for v in vals if 0.35 <= v <= 0.65)
        L.append(f"| {a.upper()} | {len(set(vals))} | {mid} of {len(vals)} | {dict(sorted(Counter(vals).items()))} |")
    L.append("")

    L.append("## 8. Specificity trade\n")
    for a in ARMS:
        n_ben = per_arm[a]["agents"][3]
        L.append(f"- Arm {a.upper()}: benign cleared by agents {rate(per_arm[a]['agents'][2], n_ben)}. Recall {rate(per_arm[a]['agents'][0], per_arm[a]['agents'][1])}.")
    L.append("")

    L.append("## Limits\n")
    L.append("- Small arms. One record moves a rate by a few points. Wilson intervals are wide.")
    L.append("- The mapping leaves seven profile features unevaluable (not loaded in the cached pool) and four per-source features unevaluable per flow. Predictions that name them are reported as unevaluable, with no support count.")
    L.append("- The zero-support cap cannot fire on an unevaluable hypothesis. The ungrounded cap still fires on the count.")
    L.append("- Source IP and time for the evidence history come from matching the flow's features against the pool (see the v3 report). Five records have ambiguous source matches and no history.")
    L.append("- A1 to A4 are the v2 outputs. The v3 evidence block is not reused in the arms; the arms rebuild it per window.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"per_arm": per_arm, "common": common, "flips_vs_v3_arm_a": [f[0] for f in flips]}, indent=2), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    return 0


THRESHOLD = 0.30

if __name__ == "__main__":
    raise SystemExit(main())
