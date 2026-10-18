"""Writes results/agent_priors_v2.md from results/agent_priors_v2.jsonl. No model calls.
Labels are read here, for scoring only."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from agents.prompts.priors_text_v2 import PRIOR_STATEMENTS, PRIORS_TEXT
from eval.run_blind_pipeline_45_v2 import load_45

RUN = Path("results/agent_priors_v2.jsonl")
TALLY = Path("results/agent_priors_v2_tally.json")
PROMPTS = Path("results/agent_priors_v2_prompts")
OUT = Path("results/agent_priors_v2.md")
V1 = Path("results/agent_priors.jsonl")
V1_TALLY = Path("results/agent_priors_tally.json")
ARM_M = Path("results/agent_run_arm_m.jsonl")
PROTOCOLS = Path("results/record_protocols.json")
KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]

ORDINARY = re.compile(r"\bordinary\b|\btypical\b|\bnormal\b|\bcommon\b|\bstandard\b|\broutine\b", re.I)
CITE = re.compile(r"\b([UTDP][1-8])\b")
FLAGS = re.compile(r"\bflag|\bSYN\b|\bACK\b|\bFIN\b|\bRST\b|\bURG\b|\bPSH\b|handshake", re.I)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k, n):
    if n == 0:
        return "0 of 0"
    lo, hi = wilson(k, n)
    return f"{k} of {n} (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def load(path):
    return {json.loads(l)["record_id"]: json.loads(l) for l in open(path, encoding="utf-8") if l.strip()} if path.exists() else {}


def main() -> int:
    truth = {}
    for p in KEYS:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            truth[r["record_id"]] = (r["true_label"], r["is_attack"] == "True")
    v2, v1, m = load(RUN), load(V1), load(ARM_M)
    proto = {k: v["protocol"] for k, v in json.loads(PROTOCOLS.read_text(encoding="utf-8")).items()}
    tally = json.loads(TALLY.read_text(encoding="utf-8")) if TALLY.exists() else {}
    v1_failed = [f["record_id"] for f in json.loads(V1_TALLY.read_text(encoding="utf-8"))["failed"]] if V1_TALLY.exists() else []
    blinds = {b["record_id"]: b for b in load_45()}

    def tally_on(ids, verdict_of):
        a = [i for i in ids if truth[i][1]]
        b = [i for i in ids if not truth[i][1]]
        return sum(verdict_of(i) == "attack" for i in a), len(a), sum(verdict_of(i) == "attack" for i in b), len(b)

    L = []
    L.append("# Tenth attempt: protocol-aware written priors\n")
    L.append("**The tenth attempt, and the fourth with no computed reference.** Same 45 records, same blind protocol, same 0.30 threshold, no cap, no reference data, all five agents rebuilt (`*_priors_v2`).\n")
    L.append("## The bound from Part 1\n")
    L.append("`results/flag_column_diagnosis.md` found that **no TCP flag column in this data can be trusted**: every one is a 0/1 indicator, `SYN Flag Count` is identical to `Fwd PSH Flags` on all 1.83M TCP rows, `URG Flag Count` is set on 20.7% of benign TCP flows, and Monday's real packets show SYNs in 39.9% of TCP flows against 11.1% in the CSV. The information was lost at extraction; recomputation from packets could cover at most 20 of the 45 records. So the flag columns were removed from what the agents see, and the priors make no handshake argument. Whatever this run finds, it cannot use TCP connection state.\n")
    L.append("## Setup (fixed before the run)\n")
    L.append("- Agents see: the flow's protocol (TCP or UDP, from its own CSV row; 36 TCP, 9 UDP), destination port, start time (time of day), and measured values without the 12 flag columns.")
    L.append("- Explanation format asks for exactly three (one benign, one hostile, one either), to address v1's five failures.")
    L.append("- One smoke-test record was run, then its cache entries were deleted, so every call in the run was fresh.\n")

    L.append("## 1. Confusion matrix against the controls, on the same records\n")
    ids = sorted(v2)
    ca, na, fb, nb = tally_on(ids, lambda i: v2[i]["verdict"])
    L.append(f"Scored: {len(ids)} of 45 ({na} attacks, {nb} benign). TP {ca}, FN {na - ca}, FP {fb}, TN {nb - fb}.\n")
    L.append("| method | records | attacks caught | benign flagged |")
    L.append("|---|---|---|---|")
    L.append(f"| **priors v2 (this run)** | {len(ids)} | **{rate(ca, na)}** | **{rate(fb, nb)}** |")
    both_m = [i for i in ids if i in m]
    c2, n2, f2, b2 = tally_on(both_m, lambda i: v2[i]["verdict"])
    cm, _, fm, _ = tally_on(both_m, lambda i: m[i]["count_only_verdict"])
    L.append(f"| priors v2, records also in Arm M | {len(both_m)} | {rate(c2, n2)} | {rate(f2, b2)} |")
    L.append(f"| count-only, Monday reference, same records | {len(both_m)} | {rate(cm, n2)} | {rate(fm, b2)} |")
    both_1 = [i for i in ids if i in v1]
    c3, n3, f3, b3 = tally_on(both_1, lambda i: v2[i]["verdict"])
    c4, _, f4, _ = tally_on(both_1, lambda i: v1[i]["verdict"])
    L.append(f"| priors v2, records also scored in v1 | {len(both_1)} | {rate(c3, n3)} | {rate(f3, b3)} |")
    L.append(f"| priors v1, same records | {len(both_1)} | {rate(c4, n3)} | {rate(f4, b3)} |")
    L.append("| attempt 8 (count >= 30 flows only, different set) | 22 | 0 of 9 | 0 of 13 |")
    L.append("")

    L.append("## 2. Does it still sort by protocol?\n")
    L.append("| protocol | attacks flagged | benign flagged |")
    L.append("|---|---|---|")
    for pr in ("TCP", "UDP"):
        sub = [i for i in ids if proto.get(i) == pr]
        a, na_, b, nb_ = tally_on(sub, lambda i: v2[i]["verdict"])
        L.append(f"| {pr} | {rate(a, na_)} | {rate(b, nb_)} |")
    L.append("")
    tcp = [i for i in ids if proto.get(i) == "TCP"]
    ta, tna, tb, tnb = tally_on(tcp, lambda i: v2[i]["verdict"])
    L.append("## 3. Within TCP only (the real test)\n")
    L.append(f"Attacks caught {rate(ta, tna)}; benign flagged {rate(tb, tnb)}. With {tnb} benign TCP flows, the benign rate is very uncertain.\n")
    tcp1 = [i for i in tcp if i in v1]
    if tcp1:
        a1_, n1_, b1_, m1_ = tally_on(tcp1, lambda i: v1[i]["verdict"])
        a2_, _, b2_, _ = tally_on(tcp1, lambda i: v2[i]["verdict"])
        L.append(f"On TCP records scored in both runs ({len(tcp1)}): v1 caught {a1_} of {n1_} and flagged {b1_} of {m1_} benign; v2 caught {a2_} of {n1_} and flagged {b2_} of {m1_}.\n")
    tcpm = [i for i in tcp if i in m]
    if tcpm:
        a_, n_, b_, mb_ = tally_on(tcpm, lambda i: v2[i]["verdict"])
        ac, _, bc, _ = tally_on(tcpm, lambda i: m[i]["count_only_verdict"])
        L.append(f"On TCP records also in Arm M ({len(tcpm)}): v2 caught {a_} of {n_} and flagged {b_} of {mb_}; the Monday count caught {ac} of {n_} and flagged {bc} of {mb_}.\n")

    from sklearn.metrics import roc_auc_score

    def auc(rows_ids, runs):
        y = [truth[i][1] for i in rows_ids]
        return roc_auc_score(y, [-runs[i]["benign_plausibility"] for i in rows_ids]) if 0 < sum(y) < len(y) else float("nan")

    L.append("### Ranking, threshold-free (diagnostic only; the 0.30 threshold is unchanged)\n")
    L.append(f"AUC of A5's score (lower = more attack-like): v2 all records {auc(ids, v2):.3f}, v2 TCP only {auc(tcp, v2):.3f}; "
             f"v1 all scored {auc(sorted(v1), v1):.3f}, v1 TCP only {auc([i for i in v1 if proto.get(i) == 'TCP'], v1):.3f}.")
    L.append("Within TCP, v2 ranks attacks above benign flows little better than chance (6 benign TCP flows, so very uncertain). Its overall AUC comes from the 9 UDP flows, all benign, all scored high. Protocol still carries the separation, through the ranking rather than the threshold.\n")

    L.append("## 4. What the rationales cite\n")
    calls = [v2[i] for i in ids if v2[i]["verdict"] == "attack"]
    flag_v2 = sum(bool(FLAGS.search(r["a5"]["rationale"])) for r in calls)
    L.append(f"- Attack calls whose rationale mentions flags, SYN/ACK/FIN/RST/URG/PSH or a handshake: {flag_v2} of {len(calls)}. (v1: 17 of 22 rested on missing flags.)")
    cited = Counter(s for r in calls for s in set(CITE.findall(r["a5"]["rationale"])))
    L.append(f"- Numbered statements cited in attack calls: {dict(sorted(cited.items()))}.")
    allc = Counter(s for i in ids for s in set(CITE.findall(v2[i]["a5"]["rationale"])))
    L.append(f"- Numbered statements cited across all rationales: {dict(sorted(allc.items()))}.")
    cite_any = sum(bool(CITE.search(v2[i]["a5"]["rationale"])) for i in ids)
    ordn = [i for i in ids if ORDINARY.search(v2[i]["a5"]["rationale"])]
    L.append(f"- Rationales citing at least one numbered statement: {cite_any} of {len(ids)}. Using \"ordinary\"-type words: {len(ordn)} of {len(ids)}, of which {sum(1 for i in ordn if not CITE.search(v2[i]['a5']['rationale']))} cite no statement.\n")

    L.append("## 5. Score spread\n")
    sc = [v2[i]["benign_plausibility"] for i in ids]
    L.append(f"Distinct values: {len(set(round(s, 2) for s in sc))}. In 0.35-0.65: {sum(0.35 <= s <= 0.65 for s in sc)} of {len(sc)}. Exactly 0.30: {sum(abs(s - 0.30) < 1e-9 for s in sc)}. (v1: 15 distinct, 8 of 40 in 0.35-0.65.)")
    L.append(f"Value counts: {dict(sorted(Counter(round(s, 2) for s in sc).items()))}.")
    L.append(f"- attacks: {sorted(round(v2[i]['benign_plausibility'], 2) for i in ids if truth[i][1])}")
    L.append(f"- benign: {sorted(round(v2[i]['benign_plausibility'], 2) for i in ids if not truth[i][1])}\n")

    L.append("## 6. The five records v1 could not score\n")
    for i in v1_failed:
        r = v2.get(i)
        L.append(f"- {i[:10]} ({truth[i][0]}): " + ("**scored in v2**, " + f"{r['benign_plausibility']:.2f} -> {r['verdict']}" if r else "still not scored in v2"))
    v2_failed = tally.get("failed", [])
    L.append(f"\nRecords not scored in v2: {len(v2_failed)}.")
    for f in v2_failed:
        L.append(f"- {f['record_id'][:10]} ({truth[f['record_id']][0]}), stage {f['agent']}: {'; '.join(x[:150] for x in f['reasons'])}")
    L.append("")

    L.append("## 7. Per record\n")
    L.append("| record | true label | protocol | port | score | verdict | v1 score |")
    L.append("|---|---|---|---|---|---|---|")
    for i in sorted(blinds, key=lambda i: (not truth[i][1], proto.get(i), i)):
        r = v2.get(i)
        v1s = f"{v1[i]['benign_plausibility']:.2f}" if i in v1 else "not scored"
        port = int(blinds[i]["features"]["Destination Port"])
        if r is None:
            L.append(f"| {i[:10]} | {truth[i][0]} | {proto.get(i)} | {port} | not scored | | {v1s} |")
        else:
            L.append(f"| {i[:10]} | {truth[i][0]} | {proto.get(i)} | {port} | {r['benign_plausibility']:.2f} | {r['verdict']} | {v1s} |")
    L.append("\nA5's full reasoning:\n")
    for i in ids:
        L.append(f"- **{i[:10]}** ({truth[i][0]}, {proto.get(i)}, {v2[i]['benign_plausibility']:.2f}): {v2[i]['a5']['rationale']}\n")

    L.append("## 8. Calls\n")
    L.append(f"{tally.get('calls')} calls including {tally.get('schema_retries')} retries; {tally.get('cache_hits')} cache hits; stopped on quota: {tally.get('stopped_on_quota')}. Plus 5 calls for the one-record smoke test, whose cache entries were deleted before the run.\n")

    L.append("## The prior text v2, verbatim, with sources\n")
    L.append("| id | source | statement |")
    L.append("|---|---|---|")
    for sid, src, text in PRIOR_STATEMENTS:
        L.append(f"| {sid} | {src} | {text} |")
    L.append("\nAs it appears in every prompt:\n\n```text\n" + PRIORS_TEXT + "```\n")
    L.append("Statement T3 (zero initial window) carries over v1's T4, which was written while the v1 brief named a zero-window DoS Hulk flow; its independence from this sample is weaker than the others'. Statement U4 states a property of this run's measurements (flags not shown), from Part 1.\n")

    L.append("## One full prompt per agent\n")
    rid = (PROMPTS / "record_id.txt").read_text(encoding="utf-8").strip() if (PROMPTS / "record_id.txt").exists() else "?"
    L.append(f"All five are for record `{rid[:10]}`, as sent.\n")
    for a in ("a1", "a2", "a3", "a4", "a5"):
        p = PROMPTS / f"{a}.txt"
        if p.exists():
            L.append(f"### {a.upper()}\n\n```text\n{p.read_text(encoding='utf-8')}\n```\n")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: caught {ca}/{na}, benign flagged {fb}/{nb}, TCP {ta}/{tna} & {tb}/{tnb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
