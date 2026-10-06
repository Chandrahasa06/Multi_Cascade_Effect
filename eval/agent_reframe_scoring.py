"""Scoring for the three reframe arms, against the v3 agents, count-only, and logistic controls.

Writes results/agent_reframe.md and results/agent_reframe.json.

Counts primary. Wilson 95% intervals on every rate. n = 44 upstream records. One record
(1b21af7f) has no v3 upstream output and is unscored in every arm. Arm B (devil) has
7 records where the advocate failed validation; those are unscored in that arm, and the
controls are also reported on exactly those 37 for a like-for-like comparison.

Three arms on the same records is three chances to beat the baseline by luck. The
report says so.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agents.verdict_rules import two_sided_state

V3_RUN = Path("results/agent_run_45_v3.jsonl")
REFRAME = Path("results/agent_run_45_reframe.jsonl")
LOGISTIC = Path("results/logistic_baseline_45.json")
KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]
OUT_MD = Path("results/agent_reframe.md")
OUT_JSON = Path("results/agent_reframe.json")
THRESHOLD = 0.30
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
        return f"{k}/{n} (n/a)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_keys() -> Dict[str, bool]:
    out = {}
    for p in KEYS:
        for row in csv.DictReader(open(p, encoding="utf-8")):
            out[row["record_id"]] = row["is_attack"] == "True"
    return out


def score_set(pred: Dict[str, bool], truth: Dict[str, bool], ids: List[str]) -> Tuple[int, int, int, int]:
    """(attacks caught, attacks, benign cleared, benign) over ids."""
    att = [i for i in ids if truth[i]]
    ben = [i for i in ids if not truth[i]]
    caught = sum(1 for i in att if pred[i])
    cleared = sum(1 for i in ben if not pred[i])
    return caught, len(att), cleared, len(ben)


def spread(vals: List[float]) -> Tuple[int, int]:
    distinct = len({round(v, 4) for v in vals})
    middle = sum(1 for v in vals if 0.35 <= v <= 0.65)
    return distinct, middle


def main() -> int:
    v3 = {r["record_id"]: r for r in load_jsonl(V3_RUN)}
    truth = load_keys()
    reframe = load_jsonl(REFRAME)
    arms = {a: {r["record_id"]: r for r in reframe if r["arm"] == a} for a in ("adversarial", "devil", "twosided")}
    logi = json.loads(LOGISTIC.read_text(encoding="utf-8"))["scores"]

    all_ids = sorted(v3)
    assert len(all_ids) == 44
    devil_ids = sorted(arms["devil"])
    assert set(devil_ids) <= set(all_ids)

    bp_v3 = {i: float(v3[i]["a5"]["benign_plausibility"]) for i in all_ids}
    pred_v3 = {i: bp_v3[i] < THRESHOLD for i in all_ids}
    count = {i: int(v3[i]["evidence_signals"]["count_all"]) for i in all_ids}
    pred_count = {i: count[i] < 30 for i in all_ids}
    pred_logi = {i: logi[i]["p_real"] < THRESHOLD for i in all_ids}

    def pred_of(arm: str) -> Dict[str, bool]:
        out = {}
        for i in all_ids:
            r = arms[arm].get(i)
            if r is not None:
                out[i] = r["verdict"] == "attack"
        return out

    def bp_of(arm: str) -> Dict[str, float]:
        return {i: float(arms[arm][i]["benign_plausibility"]) for i in arms[arm]}

    pred_adv, pred_dev, pred_two = pred_of("adversarial"), pred_of("devil"), pred_of("twosided")
    ids_adv = sorted(arms["adversarial"])
    ids_two = sorted(arms["twosided"])
    ids_dev = devil_ids

    result = {"n_upstream": len(all_ids), "arms": {}}

    rows = []

    def add_row(name, ids, pred, bp: Optional[Dict[str, float]], note=""):
        caught, na, cleared, nb = score_set(pred, truth, ids)
        d, m = spread([bp[i] for i in ids]) if bp else (None, None)
        rows.append((name, len(ids), caught, na, cleared, nb, d, m, note))
        result["arms"][name] = {"n": len(ids), "caught": caught, "n_att": na, "cleared": cleared, "n_ben": nb,
                                "distinct": d, "middle": m}

    add_row("v3 agents (single score, evidence block)", all_ids, pred_v3, bp_v3)
    add_row("count-only (leave-one-out not applied; old count < 30)", all_ids, pred_count, None)
    add_row("logistic control", all_ids, pred_logi, None)
    add_row("Arm A: adversarial framing", ids_adv, pred_adv, bp_of("adversarial"))
    add_row("Arm B: devil's advocate (37 scored)", ids_dev, pred_dev, bp_of("devil"),
            note="7 unscored: advocate validation failures")
    add_row("Arm C: two-sided scoring", ids_two, pred_two, bp_of("twosided"))

    # like-for-like on the 37 devil records
    like = {
        "v3 agents": (score_set(pred_v3, truth, ids_dev), spread([bp_v3[i] for i in ids_dev])),
        "count-only": (score_set(pred_count, truth, ids_dev), None),
        "logistic": (score_set(pred_logi, truth, ids_dev), None),
    }
    result["like_for_like_devil_ids"] = {k: list(v[0]) for k, v in like.items()}

    # six grounded attacks
    six_rows = []
    for pref in SIX:
        rid = next(i for i in all_ids if i.startswith(pref))
        base_bp = bp_v3[rid]
        base_call = "attack" if pred_v3[rid] else "benign"
        cells = []
        for arm_name, arm in [("A", "adversarial"), ("B", "devil"), ("C", "twosided")]:
            r = arms[arm].get(rid)
            if r is None:
                cells.append(f"{arm_name}: unscored")
            else:
                changed = (r["verdict"] == "attack") != pred_v3[rid]
                extra = f" (ap {r['attack_plausibility']:.2f})" if arm == "twosided" else ""
                cells.append(f"{arm_name}: {float(r['benign_plausibility']):.2f}{extra}, call {'changed' if changed else 'unchanged'}")
        six_rows.append((rid, count[rid], base_bp, base_call, cells))
    result["six"] = [(r[0], r[1], r[2], r[3]) for r in six_rows]

    # flips vs v3
    flips = {}
    for name, arm, pred in [("A", "adversarial", pred_adv), ("B", "devil", pred_dev), ("C", "twosided", pred_two)]:
        out = []
        for i in arms[arm]:
            if pred[i] != pred_v3[i]:
                out.append((i, bp_v3[i], float(arms[arm][i]["benign_plausibility"]), truth[i], arms[arm][i]["rationale"]))
        flips[name] = out
    result["flips"] = {k: [(a, b, c, d) for a, b, c, d, _ in v] for k, v in flips.items()}

    # benign cost: specificity per arm already in rows. Also recall-minus-benign-cost trade.
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    L: List[str] = []
    L.append("# Verdict reframings, tested head to head\n")
    L.append("**Three arms on the same records, against the v3 agents and two controls.** Each arm changes only the verdict stage. Upstream A1–A4 and the evidence block are the stored v3 outputs. **Three arms on the same records is three chances to beat the baseline by luck.** Any arm that looks best is reported with that caveat, not as the result.\n")
    L.append("- Arm A (adversarial): A5 asked whether a capable attacker would plausibly produce this flow. Resembling ordinary traffic is stated as not exculpatory. Single score, same 0.30 threshold.")
    L.append("- Arm B (devil's advocate): a separate agent argues only the attack reading, with flow-level predictions checked by the same filter and grounding path. A5 then weighs both sides. Single score, same 0.30 threshold.")
    L.append("- Arm C (two-sided): A5 reports benign and attack plausibility separately. Verdict rule fixed in code before the run: attack iff attack_plausibility > benign_plausibility (strict, ties benign). The 0.30 threshold is not used.\n")
    L.append(f"n = {len(all_ids)} scored upstream records (29 attacks, 15 benign for the full set). 1b21af7f has no v3 upstream and is unscored in every arm. Arm B has 37 scored records. Counts primary; Wilson 95% intervals on every rate.\n")
    L.append("**Calls this run:** adversarial 44, devil 111 (advocate and A5), two-sided 44; total 199 tallied, plus the failed advocate attempts counted by the quota. Schema failures: 7 advocate records in arm B, listed in section 9.\n")

    L.append("## Headline table\n")
    L.append("| method | records | attacks caught | benign cleared | distinct score values | any in 0.35–0.65 |")
    L.append("|---|---|---|---|---|---|")
    for name, n, caught, na, cleared, nb, d, m, note in rows:
        dstr = "n/a (binary)" if d is None else str(d)
        mstr = "n/a" if m is None else f"{m} of {n}"
        label = name + (f" — {note}" if note else "")
        L.append(f"| {label} | {n} | {rate(caught, na)} | {rate(cleared, nb)} | {dstr} | {mstr} |")
    L.append("")

    L.append("## Like-for-like on the 37 records arm B scored\n")
    L.append("| method | attacks caught | benign cleared |")
    L.append("|---|---|---|")
    for k, ((caught, na, cleared, nb), _) in like.items():
        L.append(f"| {k} | {rate(caught, na)} | {rate(cleared, nb)} |")
    L.append("")

    L.append("## 1. The six grounded attacks\n")
    L.append("| record | count | v3 score | v3 call | arm A | arm B | arm C |")
    L.append("|---|---|---|---|---|---|---|")
    for rid, cnt, bp, call, cells in six_rows:
        L.append(f"| {rid[:10]} | {cnt} | {bp:.2f} | {call} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")

    L.append("## 2. Verdict flips against v3, with A5's reasoning\n")
    for name, title in [("A", "Arm A"), ("B", "Arm B"), ("C", "Arm C")]:
        L.append(f"### {title}\n")
        if not flips[name]:
            L.append("No flips.\n")
        for i, b0, b1, is_att, rat in flips[name]:
            L.append(f"- {i[:10]} (truth: {'attack' if is_att else 'benign'}): v3 {b0:.2f} -> {b1:.2f}. A5: {rat}\n")

    L.append("## 3. Did any arm beat count-only (14/29)?\n")
    best = max(rows, key=lambda r: r[2] - (r[5] - r[4]))
    beats = [r for r in rows if r[2] > 14 and r[4] == 15 and "count-only" not in r[0]]
    if beats:
        L.append("Arms with more attacks caught than count-only and full benign specificity, on their own record set:\n")
        for r in beats:
            L.append(f"- {r[0]}: {r[2]} of {r[3]} caught on {r[1]} records.")
        L.append("\nRemember: several arms on one sample, and the count-only comparison is on 44 records (or 37 for arm B). The comparison is not paired on identical records for arm B.\n")
    else:
        L.append("**No arm beat count-only (14/29) with full benign specificity.** Each arm either matched it or lost ground. Four reframings have now failed to add value over a threshold on this sample. That is the finding, stated plainly.\n")

    L.append("## 4. Benign cost\n")
    L.append("Specificity is reported in the headline table for every arm. Any arm that gains recall while losing benign is a trade, not a gain. Benign cleared is out of 15 on the full set and out of the arm's own benign count where the arm is partial.\n")

    L.append("## 5. Score distribution per arm\n")
    L.append("Distinct values and how many scores fall in 0.35–0.65. Across every run before this one, 0 of 44 have landed there, including on a ten-point anchored scale.\n")
    L.append("| arm | scores | distinct values | in 0.35–0.65 |")
    L.append("|---|---|---|---|")
    for name, n, caught, na, cleared, nb, d, m, note in rows:
        if d is not None:
            L.append(f"| {name} | {n} | {d} | {m} of {n} |")
    L.append("")
    L.append("Full value counts:\n")
    for arm in ("adversarial", "devil", "twosided"):
        vals = [round(float(r["benign_plausibility"]), 2) for r in arms[arm].values()]
        from collections import Counter
        L.append(f"- {arm}: {dict(sorted(Counter(vals).items()))}")
    attack_vals = [round(float(r["attack_plausibility"]), 2) for r in arms["twosided"].values()]
    L.append(f"- twosided attack_plausibility: {dict(sorted(Counter(attack_vals).items()))}\n")
    L.append("Arm C's two scores are not forced to sum to 1. The state counts show how often the pair says both or neither:\n")
    # recomputed from the two scores (the stored label came from a since-corrected mapping)
    states = Counter(two_sided_state(float(r["benign_plausibility"]), float(r["attack_plausibility"])) for r in arms["twosided"].values())
    for s, c in sorted(states.items()):
        L.append(f"- {s}: {c} of {len(arms['twosided'])}")
    L.append("")

    L.append("## 6. Calls and failures\n")
    L.append("Tally: `results/agent_run_45_reframe_tally.json`. Advocate failures are the only failures; A5 never failed validation in any arm.\n")

    L.append("## 9. Advocate failures (arm B), unscored\n")
    for f in json.loads(Path("results/agent_run_45_reframe_tally.json").read_text(encoding="utf-8"))["failed"]:
        L.append(f"- {f['record_id'][:10]}: {f['error'][:200]}")
    L.append("")

    L.append("## Limits and open items\n")
    L.append("- n = 44 (29 attacks). One record moves a rate by 2.3 points.")
    L.append("- Arm B's 7 failures are a prompt gap: the advocate prompt does not specify claim_id prefixes, and the validator requires `da_c`. Not fixed after seeing results, per the rule. Those records are unscored in arm B, not counted as benign.")
    L.append("- Three arms, one sample: chance can produce an apparent winner. Read section 3 with that in mind.")
    L.append("- Cap mismatch, carried forward: the A5 prompts (v5, v6, and the three reframe prompts) still state the cap as 0.3. The code clamps at 0.29.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
