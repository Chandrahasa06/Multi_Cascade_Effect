"""Writes results/agent_priors.md from results/agent_priors.jsonl. No model calls.
Labels are read here, for scoring only."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from agents.prompts.priors_text_v1 import PRIOR_STATEMENTS, PRIORS_TEXT
from eval.run_blind_pipeline_45_v2 import load_45

RUN = Path("results/agent_priors.jsonl")
TALLY = Path("results/agent_priors_tally.json")
PROMPTS = Path("results/agent_priors_prompts")
OUT = Path("results/agent_priors.md")
KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]
ARM_A = Path("results/agent_run_arm_a.jsonl")
ARM_M = Path("results/agent_run_arm_m.jsonl")
THRESHOLD = 0.30

#: same coarse pattern attempt 8 used for falling back on "ordinary"
ORDINARY = re.compile(r"\bordinary\b|\btypical\b|\bnormal\b|\bcommon\b|\bstandard\b|\broutine\b", re.I)
PRIOR_CITE = re.compile(r"\[?\b(U[1-3]|S[1-6]|T[1-4]|X[1-6])\b\]?")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h / d), min(1.0, c + h / d))


def rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k} of {n} (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)" if n else f"{k} of 0"


def main() -> int:
    truth = {}
    for p in KEYS:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            truth[r["record_id"]] = (r["true_label"], r["is_attack"] == "True")
    rows = [json.loads(l) for l in open(RUN, encoding="utf-8") if l.strip()]
    tally = json.loads(TALLY.read_text(encoding="utf-8")) if TALLY.exists() else {}
    blinds = {b["record_id"]: b for b in load_45()}
    all_ids = sorted(blinds)
    att_all = [i for i in all_ids if truth[i][1]]
    scored = {r["record_id"]: r for r in rows}
    att = [r for r in rows if truth[r["record_id"]][1]]
    ben = [r for r in rows if not truth[r["record_id"]][1]]
    caught = sum(r["verdict"] == "attack" for r in att)
    flagged = sum(r["verdict"] == "attack" for r in ben)

    L = []
    L.append("# Ninth attempt: prompt-encoded priors, agents see only the flow\n")
    L.append("**The ninth attempt to make the agents add to detection, and the third with no computed reference.** The two before it with no reference (the first blind run with grounding broken, and attempt 8) called nearly everything benign. This run tests whether a written description of ordinary traffic can stand in for a measured baseline.\n")

    L.append("## Setup (fixed before the run)\n")
    L.append("- Five new prompt versions (`a1_priors_v1` ... `a5_priors_v1`), written fresh. No recorded A1-A4 output was reused; every stage ran under the new versions. Older prompts stay importable.")
    L.append("- Every agent saw only the flow's own properties: its measured values, destination port, and its own start time (time of day only; afternoon hours recorded in 12-hour form were shifted by 12). Protocol is not in the data and was shown as \"not recorded\". Date, weekday and source IP were not shown.")
    L.append("- No neighbourhood count, profile count, percentile, nearest-neighbour list, comparison against benign, calibration note, `*_typical` relation, source history, or other dataset statistic appears in any prompt. Each prompt was checked before sending: the whole prompt for count and calibration phrasing, and the builder's own text also for statistic words and for this record's stored neighbourhood counts (number pattern with the trailing-full-stop fix). No prompt failed.")
    L.append("- No cap. A5's number is the verdict: attack if below 0.30.")
    L.append("- Claims in this run use a schema with no relation field (`agents/schema.PriorsClaim`), so the forbidden `greater_than_typical` relation cannot occur.\n")
    L.append("**Changes made after smoke tests and before any score was seen**, all to output-format wording, none to the priors: the claim format was spelled out with an example after A1 omitted a required value three times; the claim schema was replaced by the relation-free one after A1 kept using `greater_than_typical`; the explanation format was told \"at least 3\" and to avoid the word \"header\" after A3 failed validation. Smoke-test cache entries were deleted so all 45 records were generated with the final prompts.\n")

    L.append("## 1. Confusion matrix and controls\n")
    L.append(f"Records: 45 ({len(att_all)} attacks, {45 - len(att_all)} benign). Scored: {len(rows)} ({len(att)} attacks, {len(ben)} benign). Not scored (a stage failed validation after retries): {45 - len(rows)}.\n")
    L.append("| | attacks caught | benign flagged |")
    L.append("|---|---|---|")
    L.append(f"| **priors run (this)** | **{rate(caught, len(att))}** | **{rate(flagged, len(ben))}** |")
    if ARM_A.exists():
        a = {json.loads(l)["record_id"]: json.loads(l) for l in open(ARM_A, encoding="utf-8") if l.strip()}
        ids = [i for i in scored if i in a]
        ca = sum(1 for i in ids if truth[i][1] and a[i]["count_only_verdict"] == "attack")
        fa = sum(1 for i in ids if not truth[i][1] and a[i]["count_only_verdict"] == "attack")
        na = sum(1 for i in ids if truth[i][1])
        L.append(f"| count-only rule, all-days reference, same scored records | {rate(ca, na)} | {rate(fa, len(ids) - na)} |")
    L.append("| count-only rule, Monday reference (Arm M, 42 records) | 20 of 29 | 0 of 13 |")
    L.append("| attempt 8 (security framing, no count; count >= 30 flows only) | 0 of 9 | 0 of 13 |")
    L.append("")
    L.append("Confusion matrix (counts): " + f"TP {caught}, FN {len(att) - caught}, FP {flagged}, TN {len(ben) - flagged}.\n")
    if ARM_M.exists():
        m = {json.loads(l)["record_id"]: json.loads(l) for l in open(ARM_M, encoding="utf-8") if l.strip()}
        ids = [i for i in scored if i in m]
        a_ids = [i for i in ids if truth[i][1]]
        b_ids = [i for i in ids if not truth[i][1]]
        L.append(f"On the {len(ids)} records scored in both this run and Arm M (Monday reference): this run caught "
                 f"{sum(scored[i]['verdict'] == 'attack' for i in a_ids)} of {len(a_ids)} attacks and flagged "
                 f"{sum(scored[i]['verdict'] == 'attack' for i in b_ids)} of {len(b_ids)} benign; the Monday count-only rule caught "
                 f"{sum(m[i]['count_only_verdict'] == 'attack' for i in a_ids)} of {len(a_ids)} and flagged "
                 f"{sum(m[i]['count_only_verdict'] == 'attack' for i in b_ids)} of {len(b_ids)}.\n")

    # ---- what actually drove the calls: computed, not asserted
    L.append("## What drove the calls: a feature-extraction quirk, not the priors\n")
    tcp = [r for r in rows if float(blinds[r["record_id"]]["features"]["Init_Win_bytes_forward"]) != -1]
    notcp = [r for r in rows if float(blinds[r["record_id"]]["features"]["Init_Win_bytes_forward"]) == -1]

    def split(rs):
        a = [r for r in rs if truth[r["record_id"]][1]]
        b = [r for r in rs if not truth[r["record_id"]][1]]
        return sum(r["verdict"] == "attack" for r in a), len(a), sum(r["verdict"] == "attack" for r in b), len(b)

    ta, tna, tb, tnb = split(tcp)
    ua, una, ub, unb = split(notcp)
    L.append("| flows | attacks flagged | benign flagged |")
    L.append("|---|---|---|")
    L.append(f"| with a TCP window (Init_Win_bytes_forward not -1) | {rate(ta, tna)} | {rate(tb, tnb)} |")
    L.append(f"| no TCP window (-1; DNS and similar) | {rate(ua, una)} | {rate(ub, unb)} |")
    L.append("")
    syn0 = sum(float(blinds[r["record_id"]]["features"]["SYN Flag Count"]) == 0 for r in rows)
    ack0 = sum(float(blinds[r["record_id"]]["features"]["ACK Flag Count"]) == 0 for r in rows)
    ackcite = sum(bool(re.search(r"ACK Flag Count|ACK flag|zero ACK|absence of ACK|lack of [^.]*ACK", r["a5"]["rationale"], re.I))
                  for r in rows if r["verdict"] == "attack")
    n_flag = sum(r["verdict"] == "attack" for r in rows)
    L.append(f"- In the scored records, \"SYN Flag Count\" is 0 on {syn0} of {len(rows)} flows and \"ACK Flag Count\" is 0 on {ack0} of {len(rows)}, benign and attack alike. Every TCP connection begins with a SYN, so these columns do not record what prior T1 assumes they do.")
    try:
        import pandas as pd
        from eval.escalation_data import CACHE_PATH
        pool = pd.read_parquet(CACHE_PATH, columns=["Label", "SYN Flag Count", "ACK Flag Count", "Init_Win_bytes_forward"])
        bt = pool[(pool["Label"] == "BENIGN") & (pool["Init_Win_bytes_forward"] != -1)]
        L.append(f"- Across all {len(bt):,} benign flows with a TCP window in the pool, {100 * (bt['SYN Flag Count'] == 0).mean():.1f}% have SYN Flag Count 0 and {100 * (bt['ACK Flag Count'] == 0).mean():.1f}% have ACK Flag Count 0. (Diagnostic only; computed for this report, never shown to any agent.)")
    except Exception as exc:  # the pool is optional for the report
        L.append(f"- Pool-wide flag shares not computed: {exc}")
    L.append(f"- Of the {n_flag} flows A5 called attacks, {ackcite} rationales cite missing ACK (or SYN) flags as evidence against TCP normality.")
    L.append(f"- All {una + unb} flows without a TCP window are benign here, and A5 cleared them. All {tna} scored attacks have a TCP window.")
    L.append("")
    L.append("**Reading.** Within TCP flows, the run flags attacks and benign flows at similar rates. Its apparent advantage over the count comes from clearing the non-TCP benign flows while treating most TCP flows as anomalous, because the flag-count columns read as \"no handshake\". That is a quirk of how these columns were recorded, read through a correct general prior, not evidence that written priors discriminate. With 6 benign TCP flows, the benign rate is very uncertain.\n")

    L.append("## 2. Did it use the priors?\n")
    cite = [r for r in rows if PRIOR_CITE.search(r["a5"]["rationale"])]
    ordin = [r for r in rows if ORDINARY.search(r["a5"]["rationale"])]
    only_ord = [r for r in ordin if r not in cite]
    L.append(f"- A5 rationales citing at least one numbered prior statement: {len(cite)} of {len(rows)}.")
    L.append(f"- A5 rationales using \"ordinary\", \"typical\", \"normal\", \"common\", \"standard\" or \"routine\" (attempt 8's pattern): {len(ordin)} of {len(rows)}; of these, {len(only_ord)} cite no numbered statement. In attempt 8 the figure was 21 of 22.")
    ids_cited = Counter(m for r in rows for m in set(PRIOR_CITE.findall(r["a5"]["rationale"])))
    L.append(f"- Statements cited by A5, with the number of rationales citing each: {dict(sorted(ids_cited.items()))}.\n")

    L.append("## 3. Score spread\n")
    sc = [r["benign_plausibility"] for r in rows]
    mid = [r for r in rows if 0.35 <= r["benign_plausibility"] <= 0.65]
    L.append(f"Distinct values: {len(set(round(s, 2) for s in sc))}. Value counts: {dict(sorted(Counter(round(s, 2) for s in sc).items()))}.")
    L.append(f"Scores in 0.35-0.65: {len(mid)} of {len(rows)}. Exactly 0.30 (benign under the strict rule): {sum(1 for s in sc if abs(s - 0.30) < 1e-9)}.\n")
    L.append("By true label:\n")
    L.append(f"- attacks: {sorted(round(r['benign_plausibility'], 2) for r in att)}")
    L.append(f"- benign: {sorted(round(r['benign_plausibility'], 2) for r in ben)}\n")

    L.append("## 4. The four Bot flows and the DoS Hulk flow with Init_Win_bytes_forward = 0\n")
    focus = [i for i in all_ids if truth[i][0] == "Bot"]
    focus += [i for i in all_ids if truth[i][0] == "DoS Hulk" and float(blinds[i]["features"].get("Init_Win_bytes_forward", -9)) == 0]
    for i in focus:
        r = scored.get(i)
        f = blinds[i]["features"]
        head = (f"**{i[:10]}** ({truth[i][0]}): port {int(f['Destination Port'])}, Init_Win_bytes_forward {f.get('Init_Win_bytes_forward'):g}, "
                f"fwd/bwd packets {f.get('Total Fwd Packets'):g}/{f.get('Total Backward Packets'):g}, duration {f.get('Flow Duration'):g} us")
        if r is None:
            L.append(f"- {head}. Not scored (a stage failed).")
        else:
            L.append(f"- {head}. **Score {r['benign_plausibility']:.2f} -> {r['verdict']}.** A5: {r['a5']['rationale']}")
    L.append("")
    L.append("Statement T4 (zero initial window is not ordinary) was written while the task brief named this DoS Hulk flow's zero window as a property a prior could catch. It is general TCP knowledge, but its independence from this sample is weaker than the other statements'.\n")

    L.append("## 5. Per record\n")
    L.append("| record | true label | port | time | score | verdict |")
    L.append("|---|---|---|---|---|---|")
    for i in all_ids:
        r = scored.get(i)
        if r is None:
            L.append(f"| {i[:10]} | {truth[i][0]} | {int(blinds[i]['features']['Destination Port'])} | | not scored | |")
        else:
            L.append(f"| {i[:10]} | {truth[i][0]} | {int(r['destination_port'])} | {r['time_of_day'] or 'n/a'} | {r['benign_plausibility']:.2f} | {r['verdict']} |")
    L.append("\nA5's full reasoning, per record:\n")
    for i in all_ids:
        r = scored.get(i)
        if r is not None:
            L.append(f"- **{i[:10]}** ({truth[i][0]}, {r['benign_plausibility']:.2f}): {r['a5']['rationale']}\n")

    L.append("## 6. Calls\n")
    L.append(f"From the run tally: {tally.get('calls')} calls including retries ({tally.get('schema_retries')} retries), {tally.get('cache_hits')} cache hits, {len(tally.get('failed', []))} records not scored, stopped on quota: {tally.get('stopped_on_quota')}. Failed records:\n")
    for f in tally.get("failed", []):
        L.append(f"- {f['record_id'][:10]} ({truth[f['record_id']][0]}), stage {f['agent']}: {'; '.join(x[:160] for x in f['reasons'])}")
    L.append("")

    L.append("## The prior text, verbatim, with sources\n")
    L.append("| id | source | statement |")
    L.append("|---|---|---|")
    for sid, src, text in PRIOR_STATEMENTS:
        L.append(f"| {sid} | {src} | {text} |")
    L.append("\nAs it appears in every prompt:\n")
    L.append("```text\n" + PRIORS_TEXT + "```\n")
    L.append("The numbers are typical ranges from general knowledge. None was taken from CICIDS2017 or from this project's results; none was checked against the data.\n")

    L.append("## One full prompt per agent\n")
    rid = (PROMPTS / "record_id.txt").read_text(encoding="utf-8").strip() if (PROMPTS / "record_id.txt").exists() else "?"
    L.append(f"All five are for record `{rid[:10]}`, as sent.\n")
    for a in ("a1", "a2", "a3", "a4", "a5"):
        p = PROMPTS / f"{a}.txt"
        if p.exists():
            L.append(f"### {a.upper()}\n\n```text\n{p.read_text(encoding='utf-8')}\n```\n")

    L.append("## Limits\n")
    L.append(f"- n is small: {len(att)} attacks and {len(ben)} benign scored. Each attack moves recall by about {100 / max(len(att), 1):.0f} points.")
    L.append("- Same single model as every earlier run.")
    L.append("- Time of day depends on the 12-hour correction.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: caught {caught}/{len(att)}, benign flagged {flagged}/{len(ben)}, scored {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
