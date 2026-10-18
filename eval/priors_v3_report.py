"""Writes results/agent_priors_v3.md from results/agent_priors_v3.jsonl. No model calls.
Labels are read here, for scoring only."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from sklearn.metrics import roc_auc_score

from agents.prompts.priors_text_v3 import CONDITION_IDS, CONDITIONS, PRIORS_TEXT
from eval.run_blind_pipeline_45_v2 import load_45

RUN = Path("results/agent_priors_v3.jsonl")
TALLY = Path("results/agent_priors_v3_tally.json")
PROMPTS = Path("results/agent_priors_v3_prompts")
OUT = Path("results/agent_priors_v3.md")
V1, V2 = Path("results/agent_priors.jsonl"), Path("results/agent_priors_v2.jsonl")
ARM_M = Path("results/agent_run_arm_m.jsonl")
PROTOCOLS = Path("results/record_protocols.json")
KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]

ORDINARY = re.compile(r"\bordinary\b|\btypical\b|\bnormal\b|\bcommon\b|\bstandard\b|\broutine\b", re.I)
COND = re.compile(r"\bC(1[0-2]|[1-9])\b")


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


def load(p):
    return {json.loads(l)["record_id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()} if p.exists() else {}


def main() -> int:
    truth = {}
    for p in KEYS:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            truth[r["record_id"]] = (r["true_label"], r["is_attack"] == "True")
    v3, v2, v1, m = load(RUN), load(V2), load(V1), load(ARM_M)
    proto = {k: v["protocol"] for k, v in json.loads(PROTOCOLS.read_text(encoding="utf-8")).items()}
    tally = json.loads(TALLY.read_text(encoding="utf-8")) if TALLY.exists() else {}
    blinds = {b["record_id"]: b for b in load_45()}
    ids = sorted(v3)
    tcp = [i for i in ids if proto.get(i) == "TCP"]

    def t(ids_, verdict):
        a = [i for i in ids_ if truth[i][1]]
        b = [i for i in ids_ if not truth[i][1]]
        return sum(verdict(i) == "attack" for i in a), len(a), sum(verdict(i) == "attack" for i in b), len(b)

    def auc(ids_, runs):
        y = [truth[i][1] for i in ids_]
        return roc_auc_score(y, [-runs[i]["benign_plausibility"] for i in ids_]) if 0 < sum(y) < len(y) else float("nan")

    L = []
    L.append("# Eleventh attempt: priors as checkable conditions, with forced citation\n")
    L.append("**The eleventh attempt, and the fifth with no computed reference.** Same 45 records, blind protocol, 0.30 threshold; no cap, no reference data, no TCP flag column (all twelve untrustworthy, `results/flag_column_diagnosis.md`). All five agents rebuilt (`*_priors_v3`).\n")
    L.append("New in v3: the priors are 12 yes/no conditions on the flow's own values (below, verbatim). A5's verdict is rejected unless it carries a checklist entry for every condition, with the value it read. A1's claims and A3/A4's explanations must cite condition ids.\n")
    L.append(f"Scored: {len(ids)} of 45. Calls: {tally.get('calls')} including {tally.get('schema_retries')} retries, {tally.get('cache_hits')} cache hits; stopped on quota: {tally.get('stopped_on_quota')}; not scored after retries: {len(tally.get('failed', []))}. Plus 8 calls for a one-record smoke test whose cache entries were deleted before the run.\n")

    L.append("## 1. Confusion matrix and controls (overall and within TCP)\n")
    ca, na, fb, nb = t(ids, lambda i: v3[i]["verdict"])
    L.append(f"v3, all scored: TP {ca}, FN {na - ca}, FP {fb}, TN {nb - fb}.\n")
    L.append("| method | records | attacks caught | benign flagged | within TCP: attacks | within TCP: benign |")
    L.append("|---|---|---|---|---|---|")

    def row(name, ids_, verdict):
        a, na_, b, nb_ = t(ids_, verdict)
        tc = [i for i in ids_ if proto.get(i) == "TCP"]
        ta, tna, tb, tnb = t(tc, verdict)
        L.append(f"| {name} | {len(ids_)} | {rate(a, na_)} | {rate(b, nb_)} | {rate(ta, tna)} | {rate(tb, tnb)} |")

    row("**priors v3 (this run)**", ids, lambda i: v3[i]["verdict"])
    both_m = [i for i in ids if i in m]
    row("priors v3, records in Arm M", both_m, lambda i: v3[i]["verdict"])
    row("count-only, Monday reference, same records", both_m, lambda i: m[i]["count_only_verdict"])
    both_1 = [i for i in ids if i in v1]
    row("priors v1, records scored in both", both_1, lambda i: v1[i]["verdict"])
    row("priors v3, same records", both_1, lambda i: v3[i]["verdict"])
    both_2 = [i for i in ids if i in v2]
    row("priors v2, records scored in both", both_2, lambda i: v2[i]["verdict"])
    row("priors v3, same records ", both_2, lambda i: v3[i]["verdict"])
    L.append("")

    L.append("## 2. Protocol split, and the within-TCP ranking\n")
    L.append("| protocol | attacks flagged | benign flagged |")
    L.append("|---|---|---|")
    for pr in ("TCP", "UDP"):
        a, na_, b, nb_ = t([i for i in ids if proto.get(i) == pr], lambda i: v3[i]["verdict"])
        L.append(f"| {pr} | {rate(a, na_)} | {rate(b, nb_)} |")
    L.append("")
    L.append(f"AUC of A5's score (diagnostic, threshold-free): v3 all {auc(ids, v3):.3f}, **v3 within TCP {auc(tcp, v3):.3f}**; "
             f"v2 all {auc(sorted(v2), v2):.3f}, v2 within TCP {auc([i for i in v2 if proto.get(i) == 'TCP'], v2):.3f}. "
             f"The within-TCP figure is the real one: every UDP flow here is benign, so overall AUC rewards clearing UDP.\n")

    L.append("## 3. Which conditions fired, and on what\n")
    L.append("Counts from A5's checklists (\"yes\" answers), split by true label, beside the code-side truth for the same flows. A condition that holds at the same rate on attacks and benign flows carries no information.\n")
    L.append("| condition | A5 yes on attacks | A5 yes on benign | true yes on attacks | true yes on benign | A5 agrees with code | reading |")
    L.append("|---|---|---|---|---|---|---|")
    att = [i for i in ids if truth[i][1]]
    ben = [i for i in ids if not truth[i][1]]
    agree_total = 0
    rows_total = 0
    for cid in CONDITION_IDS:
        def a5_yes(i):
            return any(c["condition_id"] == cid and c["holds"] == "yes" for c in v3[i]["a5"]["checklist"])

        def code_yes(i):
            return v3[i]["conditions_code"][cid] == "yes"

        ya, yb = sum(a5_yes(i) for i in att), sum(a5_yes(i) for i in ben)
        ca_, cb_ = sum(code_yes(i) for i in att), sum(code_yes(i) for i in ben)
        agree = sum(next(c["holds"] for c in v3[i]["a5"]["checklist"] if c["condition_id"] == cid) == v3[i]["conditions_code"][cid] for i in ids)
        agree_total += agree
        rows_total += len(ids)
        ra, rb = (ca_ / len(att) if att else 0), (cb_ / len(ben) if ben else 0)
        if ca_ + cb_ == 0:
            reading = "never holds here"
        elif abs(ra - rb) < 0.1:
            reading = "no information (similar on both)"
        elif ra > rb:
            reading = "more often on attacks"
        else:
            reading = "more often on benign"
        L.append(f"| {cid} | {ya} of {len(att)} | {yb} of {len(ben)} | {ca_} of {len(att)} | {cb_} of {len(ben)} | {agree} of {len(ids)} | {reading} |")
    L.append(f"\nA5's checklist answers agree with the code-side evaluation on {agree_total} of {rows_total} entries.\n")

    L.append("## 4. Did forcing citation change the reasoning?\n")
    cite = [i for i in ids if COND.search(v3[i]["a5"]["rationale"])]
    ordn = [i for i in ids if ORDINARY.search(v3[i]["a5"]["rationale"])]
    both = [i for i in ordn if i in cite]
    L.append(f"- Rationales naming at least one condition id: {len(cite)} of {len(ids)}.")
    L.append(f"- Rationales using \"ordinary\"-type words: {len(ordn)} of {len(ids)}; of these, {len(both)} also name a condition and {len(ordn) - len(both)} name none.")
    L.append("- v2 for comparison: 17 of 45 cited a numbered statement; 35 of 45 used \"ordinary\"-type words, 21 of them citing nothing.")
    cited = Counter(c for i in ids for c in set("C" + x for x in COND.findall(v3[i]["a5"]["rationale"])))
    L.append(f"- Condition ids named in rationales: {dict(sorted(cited.items(), key=lambda kv: int(kv[0][1:])))}.")
    fallback = [i for i in ids if ORDINARY.search(v3[i]["a5"]["rationale"]) and v3[i]["verdict"] == "benign"
                and any(c["holds"] == "yes" for c in v3[i]["a5"]["checklist"])]
    L.append(f"- Records where the checklist shows at least one condition holding, the verdict is benign, and the rationale still uses \"ordinary\"-type words: {len(fallback)} of {len(ids)}.\n")

    L.append("## 5. Score spread\n")
    sc = [v3[i]["benign_plausibility"] for i in ids]
    L.append(f"Distinct values: {len(set(round(s, 2) for s in sc))}. In 0.35-0.65: {sum(0.35 <= s <= 0.65 for s in sc)} of {len(sc)}. Exactly 0.30: {sum(abs(s - 0.30) < 1e-9 for s in sc)}. (v2: 14 distinct, 19 of 45.)")
    L.append(f"- attacks: {sorted(round(v3[i]['benign_plausibility'], 2) for i in att)}")
    L.append(f"- benign: {sorted(round(v3[i]['benign_plausibility'], 2) for i in ben)}\n")

    L.append("## 6. Per record\n")
    for i in sorted(ids, key=lambda i: (not truth[i][1], proto.get(i), i)):
        r = v3[i]
        held = [c["condition_id"] for c in r["a5"]["checklist"] if c["holds"] == "yes"]
        code_held = [k for k, v in r["conditions_code"].items() if v == "yes"]
        L.append(f"### {i[:10]}, {truth[i][0]} ({proto.get(i)}, port {int(blinds[i]['features']['Destination Port'])})\n")
        L.append(f"- score {r['benign_plausibility']:.2f} -> **{r['verdict']}**")
        L.append(f"- conditions A5 marked yes: {', '.join(held) or 'none'}; true by code: {', '.join(code_held) or 'none'}")
        L.append("- checklist: " + "; ".join(
            f"{c['condition_id']} {c['holds']} ({', '.join(f'{o['name']}={o['value']:g}' for o in c['observed'])})" for c in r["a5"]["checklist"]))
        L.append(f"- A5: {r['a5']['rationale']}\n")
    L.append("### Records not scored\n")
    failed = tally.get("failed", [])
    for f in failed:
        L.append(f"- failed validation after retries: {f['record_id'][:10]} ({truth[f['record_id']][0]}), stage {f['agent']}: {'; '.join(x[:150] for x in f['reasons'])}")
    reasons = Counter(re.sub(r"[a-z0-9]+_[ch]\d+", "<id>", r.split("\n")[0])[:100] for f in failed for r in f["reasons"])
    L.append("\nRetry reasons across the failed records: " + "; ".join(f"{v} x \"{k}\"" for k, v in reasons.most_common()))
    L.append("\nThe most common reason is this run's own rule that every explanation must cite a condition. A benign explanation for a flow on which no condition holds has nothing to cite, so A3 and A4 kept leaving it empty. That is a flaw in v3's validator, recorded here, not fixed after seeing results.")
    attempted = set(ids) | {f["record_id"] for f in failed}
    skipped = [i for i in sorted(blinds) if i not in attempted]
    if skipped:
        L.append(f"\nNot attempted because the daily quota ran out (stopped_on_quota = {tally.get('stopped_on_quota')}): "
                 + ", ".join(f"{i[:10]} ({truth[i][0]})" for i in skipped) + ".")
    L.append("")

    L.append("## The conditions, verbatim, with sources\n")
    L.append("| id | scope | test | meaning | source |")
    L.append("|---|---|---|---|---|")
    for cid, scope, test, meaning, src in CONDITIONS:
        L.append(f"| {cid} | {scope} | {test} | {meaning} | {src} |")
    L.append("\nAs it appears in every prompt:\n\n```text\n" + PRIORS_TEXT + "```\n")
    L.append("Thresholds are round general-knowledge values. The author had seen some of these records' values in earlier reports when writing them, so their independence from this sample is not complete. C4 carries over the zero-window statement written while an earlier brief named a zero-window DoS Hulk flow.\n")

    L.append("## One full prompt per agent\n")
    rid = (PROMPTS / "record_id.txt").read_text(encoding="utf-8").strip() if (PROMPTS / "record_id.txt").exists() else "?"
    L.append(f"All five are for record `{rid[:10]}`, as sent.\n")
    for a in ("a1", "a2", "a3", "a4", "a5"):
        p = PROMPTS / f"{a}.txt"
        if p.exists():
            L.append(f"### {a.upper()}\n\n```text\n{p.read_text(encoding='utf-8')}\n```\n")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    ta, tna, tb, tnb = t(tcp, lambda i: v3[i]["verdict"])
    print(f"wrote {OUT}: caught {ca}/{na}, benign flagged {fb}/{nb}; TCP {ta}/{tna}, {tb}/{tnb}; AUC TCP {auc(tcp, v3):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
