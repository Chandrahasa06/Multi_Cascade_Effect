"""Writes results/agent_grounded_only.md from results/agent_grounded_only.jsonl. No model calls.
Labels are read here, for scoring only."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

RUN = Path("results/agent_grounded_only.jsonl")
TALLY = Path("results/agent_grounded_only_tally.json")
OUT = Path("results/agent_grounded_only.md")
KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]

#: evidence types, matched case-insensitively in each rationale (fixed before reading them)
EVIDENCE_TYPES = {
    "destination port / service": r"\bport\b|\bhttp\b|\bhttps\b|\bssh\b|\bdns\b|\bftp\b|\b(80|443|22|53|21|8080)\b",
    "time of day": r"time of day|\bhour\b|\bo'clock\b|\b\d{1,2}:\d{2}\b|business hours|working hours|morning|afternoon|night",
    "source history": r"history|preceding|prior (flows|activity|minutes)|previous (flows|minutes)|in the (last|preceding|prior) \d+|bucket|this source",
    "percentile / distribution": r"percentile|median",
    "attacker mimicry considered": r"attacker|adversar|mimic|blend|look(s|ing)? (ordinary|normal)|evad|camouflag",
    "'ordinary' or 'typical' as a reason": r"\bordinary\b|\btypical\b|\bnormal\b|\bcommon\b|\bstandard\b|\broutine\b",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k} of {n} (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def main() -> int:
    truth = {}
    for p in KEYS:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            truth[r["record_id"]] = (r["true_label"], r["is_attack"] == "True")
    rows = [json.loads(l) for l in open(RUN, encoding="utf-8") if l.strip()]
    tally = json.loads(TALLY.read_text(encoding="utf-8"))
    att = [r for r in rows if truth[r["record_id"]][1]]
    ben = [r for r in rows if not truth[r["record_id"]][1]]
    caught = sum(r["verdict"] == "attack" for r in att)
    flagged = sum(r["verdict"] == "attack" for r in ben)
    scores = [r["benign_plausibility"] for r in rows]
    mid = [r for r in rows if 0.35 <= r["benign_plausibility"] <= 0.65]

    L = []
    L.append("# Eighth attempt: A5 as a security judge, only where the count says benign\n")
    L.append("**This is the eighth attempt, after seven that failed to beat the count, run on the subset where the count scores zero by construction.** It is a last clear test, not a fresh start. n is small (22 records, 9 attacks), and the intervals below are wide enough that only a large effect could show.\n")
    L.append("## Setup (fixed before any call)\n")
    L.append("- Records: every Arm M record with leave-one-out count of 30 or more (the flows the count calls benign). Monday benign reference; records from Tuesday onward.")
    L.append("- A1-A4: recorded v2 outputs. Only A5 was called, with `a5_verdict_security_v1`: \"statistical checks found this flow unremarkable; given how networks are attacked, is there reason to think it is hostile anyway?\" The prompt states that resembling ordinary traffic is not by itself exculpatory. No threshold or target verdict is given.")
    L.append("- **A5 saw no count.** No neighbourhood count, no \"within 2x\" closeness, no hypothesis profile counts. The context builder has no count parameter; every prompt was checked for count phrasing, and for the record's count value appearing outside feature, claim and history numbers. No prompt failed. No count value collided with any number in its prompt.")
    L.append("- **No cap was applied.** Scores are A5's own.")
    L.append("- A5 saw: features, destination port, time of day, the source's flows in the previous 30 minutes, and per-feature percentiles and direction against Monday benign traffic.")
    L.append("- Time of day: CICIDS2017 CSVs record afternoon times in 12-hour form (no recorded hour 13-17 exists; afternoon-only files sit at 01-05). Hours below 08 were shifted by 12. Applied to every record identically.")
    L.append("- Source history: all flows from the same source, read without labels. A Monday-only benign index would have shown every Tuesday-onward source as silent; a benign filter on the evaluation day would have read labels and hidden a source's own earlier flows. One record (3da627f9) had no unique source match; A5 was told its history and time were unavailable.")
    L.append("- Not shown: source IP, date, weekday (a model that memorised the published CICIDS2017 attack schedule could score from them), protocol (not recorded in the blind features or the pool).")
    L.append(f"- Calls: {tally['calls']} (schema retries {tally['schema_retries']}, failures {len(tally['failed'])}).\n")

    L.append("## 1 and 2. The trade\n")
    L.append("| | result | count-only baseline on these records |")
    L.append("|---|---|---|")
    L.append(f"| attacks caught, of the {len(att)} the count misses | **{rate(caught, len(att))}** | 0 of {len(att)} by construction |")
    L.append(f"| benign flagged (the cost) | **{rate(flagged, len(ben))}** | 0 of {len(ben)} |")
    L.append("")
    if caught == 0:
        L.append(f"**A5 caught none of the {len(att)} attacks the count misses.** With no count to anchor on and a prompt that asks for a security judgment, it called every record benign. It flagged no benign record either, so this is not a trade: it is no detection. The 95% upper bound on recall here is about {100 * wilson(0, len(att))[1]:.0f}%, so the run rules out only a large effect.\n")
    L.append("## 3. Per record\n")
    for r in sorted(rows, key=lambda r: (not truth[r["record_id"]][1], r["count_loo"])):
        lab = truth[r["record_id"]]
        L.append(f"### {r['record_id'][:10]} — true label {lab[0]} ({'attack' if lab[1] else 'benign'})\n")
        L.append(f"- leave-one-out count (not shown to A5): {r['count_loo']}; destination port {int(r['destination_port'])}; time {r['time_of_day'] or 'unavailable'}")
        hist = r["history"]
        if hist:
            L.append(f"- source flows in previous 30 min: {sum(h['flows'] for h in hist)}")
        L.append(f"- Arm M score (with count and cap): {r['arm_m_score']:.2f}. **This run: {r['benign_plausibility']:.2f} -> {r['verdict']}**")
        L.append(f"- A5's reasoning: {r['rationale']}\n")

    L.append("## 4. Does the output spread?\n")
    L.append(f"Distinct values: {len(set(scores))}. Counts: {dict(sorted(Counter(round(s, 2) for s in scores).items()))}.")
    L.append(f"Scores in 0.35-0.65: {len(mid)} of {len(rows)}" + (f" ({', '.join(r['record_id'][:10] + ' ' + truth[r['record_id']][0] + ' at ' + format(r['benign_plausibility'], '.2f') for r in mid)})." if mid else "."))
    ss = sorted(scores)
    L.append(f"Lowest score: {ss[0]:.2f}. Second lowest: {ss[1]:.2f}.\n")

    L.append("## 5. What A5 cited\n")
    L.append("Each rationale was scanned for evidence types using keyword patterns fixed before reading the rationales (`eval/grounded_only_report.py`). Keyword matching is coarse: a hit means the topic is mentioned, not that it was weighed well.\n")
    L.append("| evidence type | rationales mentioning it, of 22 | attacks | benign |")
    L.append("|---|---|---|---|")
    for name, pat in EVIDENCE_TYPES.items():
        rx = re.compile(pat, re.I)
        a = sum(bool(rx.search(r["rationale"])) for r in att)
        b = sum(bool(rx.search(r["rationale"])) for r in ben)
        L.append(f"| {name} | {a + b} | {a} of {len(att)} | {b} of {len(ben)} |")
    L.append("")
    schedule = [r for r in rows if re.search(r"cicids|2017|dataset", r["rationale"], re.I)]
    L.append(f"Rationales naming the dataset or its year (a sign of memorised schedule use): {len(schedule)}.\n")

    L.append("## Limits\n")
    L.append(f"- n = {len(rows)}: {len(att)} attacks and {len(ben)} benign. Each attack moves recall by {100 / len(att):.0f} points.")
    mix = Counter(f"{truth[r['record_id']][0]} (port {int(r['destination_port'])})" for r in att)
    L.append("- The attacks here, by label and destination port: " + ", ".join(f"{k} x{v}" for k, v in sorted(mix.items())) + ".")
    L.append("- Time of day depends on the 12-hour correction described above.")
    L.append("- Source history includes the source's attack flows when it was an attacker. That is observable traffic in deployment, not a label.")
    L.append("- Model: the same single model as every earlier run.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: caught {caught}/{len(att)}, benign flagged {flagged}/{len(ben)}, middle {len(mid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
