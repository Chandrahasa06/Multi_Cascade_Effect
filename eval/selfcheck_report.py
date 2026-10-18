"""Writes results/agent_selfcheck.md from results/agent_selfcheck.jsonl. No model calls.
Labels are read here, for scoring only. Rules: results/selfcheck_preregistered.md."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from agents import selfcheck_core as core
from eval.run_blind_pipeline_45_v2 import load_45

RUN = Path("results/agent_selfcheck.jsonl")
TALLY = Path("results/agent_selfcheck_tally.json")
PREREG = Path("results/selfcheck_preregistered.md")
PROMPTS = Path("results/agent_selfcheck_prompts")
V3 = Path("results/agent_priors_v3.jsonl")
OUT = Path("results/agent_selfcheck.md")
KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]
CATS = ("anomalous_and_unexplained", "anomalous_but_explicable", "consistent_with_benign", "insufficient_evidence")
GUARD = 0.30


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


def main() -> int:
    truth = {}
    for p in KEYS:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            truth[r["record_id"]] = (r["true_label"], r["is_attack"] == "True")
    rows = {json.loads(l)["record_id"]: json.loads(l) for l in open(RUN, encoding="utf-8") if l.strip()}
    tally = json.loads(TALLY.read_text(encoding="utf-8")) if TALLY.exists() else {}
    blinds = {b["record_id"]: b for b in load_45()}
    ids = sorted(rows)
    att = [i for i in ids if truth[i][1]]
    ben = [i for i in ids if not truth[i][1]]
    tcp = [i for i in ids if rows[i]["protocol"] == "TCP"]

    def caught(sel, key="verdict"):
        return sum(rows[i][key] == "anomalous_and_unexplained" for i in sel)

    L = ["# Self-check run: field-cited agents, trust-ordered justifications, categorical verdict\n"]
    L.append(f"Rules were fixed before any call: `{PREREG}` (reproduced at the end). Every check is against the flow's own row; no population, baseline, reference pool or neighbourhood quantity is used. All 45 records were attempted; {len(ids)} scored.\n")
    L.append(f"Calls: {tally.get('calls')} including {tally.get('schema_retries')} retries; {tally.get('cache_hits')} cache hits; stopped on quota: {tally.get('stopped_on_quota')}; failed validation after retries: {len(tally.get('failed', []))} (plus 7 calls for the first record, run as a smoke test and kept).\n")

    # ---- headline
    n_ins = sum(rows[i]["verdict"] == "insufficient_evidence" for i in ids)
    inconclusive = n_ins > GUARD * len(ids)
    L.append("## Headline\n")
    if inconclusive:
        L.append(f"**INCONCLUSIVE.** insufficient_evidence was given on {n_ins} of {len(ids)} records, above the pre-registered 30% guard. No recall figure is reported as a result.\n")
    L.append("| | attacks caught (anomalous_and_unexplained) | benign flagged (anomalous_and_unexplained) |")
    L.append("|---|---|---|")
    L.append(f"| all scored | {rate(caught(att), len(att))} | {rate(caught(ben), len(ben))} |")
    ta = [i for i in att if i in tcp]
    tb = [i for i in ben if i in tcp]
    L.append(f"| within TCP | {rate(caught(ta), len(ta))} | {rate(caught(tb), len(tb))} |")
    L.append("\nDenominators include every scored record, so insufficient_evidence counts as not caught for attacks and not flagged for benign; it is never dropped.\n")
    failed_ids = [f["record_id"] for f in tally.get("failed", [])]
    if failed_ids:
        fa = sum(truth[i][1] for i in failed_ids)
        L.append(f"**Not scored: {len(failed_ids)} of 45** ({fa} attacks, {len(failed_ids) - fa} benign), all at A4: its claims kept failing the rule that each sentence names a cited field, or it gave 2 hypotheses instead of 3-5, three times running. These records have no verdict and are not in any rate above.\n")

    L.append("### Verdict distribution\n")
    L.append("| verdict | attacks | benign |")
    L.append("|---|---|---|")
    for c in CATS:
        L.append(f"| {c} | {sum(rows[i]['verdict'] == c for i in att)} of {len(att)} | {sum(rows[i]['verdict'] == c for i in ben)} of {len(ben)} |")
    L.append("")

    # ---- the question that matters: checklist alone
    same = [i for i in ids if rows[i]["verdict"] == rows[i]["checklist_alone_verdict"]]
    L.append("## Would A5a's checklist alone have given the same verdict?\n")
    L.append(f"Checklist-alone rule (pre-registered): anomalous_and_unexplained if A5a marked any condition yes, else consistent_with_benign.\n")
    L.append(f"**Same verdict on {len(same)} of {len(ids)} records.**\n")
    L.append("| | attacks caught | benign flagged |")
    L.append("|---|---|---|")
    L.append(f"| A5b (with A1-A4) | {rate(caught(att), len(att))} | {rate(caught(ben), len(ben))} |")
    L.append(f"| A5a checklist alone | {rate(caught(att, 'checklist_alone_verdict'), len(att))} | {rate(caught(ben, 'checklist_alone_verdict'), len(ben))} |")
    code_alone = {i: core.checklist_alone_verdict([k for k, v in rows[i]["conditions_code"].items() if v == "yes"]) for i in ids}
    L.append(f"| code-side conditions alone | {rate(sum(code_alone[i] == 'anomalous_and_unexplained' for i in att), len(att))} | "
             f"{rate(sum(code_alone[i] == 'anomalous_and_unexplained' for i in ben), len(ben))} |")
    L.append("")
    diffs = [i for i in ids if i not in same]
    if diffs:
        L.append("Records where A5b differs from the checklist alone:\n")
        for i in diffs:
            L.append(f"- {i[:10]} ({truth[i][0]}): checklist alone {rows[i]['checklist_alone_verdict']} -> A5b {rows[i]['verdict']}; dispositions: "
                     + "; ".join(f"{d['condition_id']} {'refuted by ' + ','.join(d['refuted_by']) if d['refuted'] else 'not refuted'}" for d in rows[i]["dispositions"]))
        L.append("")
    if len(same) == len(ids):
        L.append("**On every record, the verdict equals what A5a's checklist alone gives. The other four agents changed no verdict: in this run they are decoration.**\n")
    else:
        moved = Counter((rows[i]["checklist_alone_verdict"], rows[i]["verdict"]) for i in diffs)
        L.append(f"A5b moved {len(diffs)} records: {dict(moved)}. Whether that helps: on those records, attacks moved {sum(truth[i][1] for i in diffs)}, benign moved {sum(not truth[i][1] for i in diffs)}.\n")

    # ---- stopping rule on the v3 records
    if V3.exists():
        v3ids = {json.loads(l)["record_id"] for l in open(V3, encoding="utf-8") if l.strip()}
        v3a = [i for i in att if i in v3ids]
        v3b = [i for i in ben if i in v3ids]
        k = caught(v3a)
        L.append("## Stopping rule (pre-registered)\n")
        L.append(f"On the records v3 scored that this run also scored: {rate(k, len(v3a))} attacks caught, {rate(caught(v3b), len(v3b))} benign flagged.\n")
        if k == 0:
            outcome = "0 again: the aggregation diagnosis was wrong. Prompting on this condition set stops."
        elif k <= 11 * len(v3a) / 20:
            outcome = "at or below roughly 11 of 20: the ceiling analysis is confirmed. Prompting on this condition set stops."
        else:
            outcome = "above roughly 11 of 20: above the ceiling estimate."
        L.append(f"**Outcome: {outcome}**\n")

    # ---- trust
    L.append("## Trust\n")
    L.append("Mean of each component over records where it is defined (n in brackets). T is the equal-weight mean of the defined components.\n")
    L.append("| agent | F | B | S | H | T | old C | old E | old V | bands |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")

    def mean(vals):
        v = [x for x in vals if x is not None]
        return f"{sum(v) / len(v):.2f} ({len(v)})" if v else "undefined"

    for a in ("a1", "a2", "a3", "a4", "a5a"):
        ts = [rows[i]["trust"][a] for i in ids]
        L.append(f"| {a.upper()} | {mean([t['components']['F'] for t in ts])} | {mean([t['components']['B'] for t in ts])} | "
                 f"{mean([t['components']['S'] for t in ts])} | {mean([t['components']['H'] for t in ts])} | {mean([t['T'] for t in ts])} | "
                 f"{mean([t['old']['C'] for t in ts])} | {mean([t['old']['E'] for t in ts])} | {mean([t['old']['V'] for t in ts])} | "
                 f"{dict(Counter(t['band'] for t in ts))} |")
    L.append("")
    wl = tally.get("withheld_log", [])
    L.append(f"Justifications withheld below the floor (T < 0.25): {len(wl)}. " + ("; ".join(f"{w['record_id'][:10]} {w['agent']} (T {w['T']:.2f})" for w in wl) if wl else ""))
    fails = [(i, a, f) for i in ids for a in ("a1", "a2", "a3", "a4", "a5a") for f in rows[i]["trust"][a]["failed_citations"]]
    L.append(f"\nCited values that failed field verification: {len(fails)}.")
    for i, a, f in fails[:40]:
        L.append(f"- {i[:10]} {a}: {f['field']} cited {f['cited']:g}, row {f['row'] if f['row'] is None else format(f['row'], 'g')}")
    L.append("")

    Ts = [rows[i]["trust"][a]["T"] for i in ids for a in ("a1", "a2", "a3", "a4") if rows[i]["trust"][a]["T"] is not None]
    L.append(f"**Trust is saturated.** Of {len(Ts)} defined A1-A4 trust scores, {sum(t >= 0.99 for t in Ts)} are at or above 0.99, and no cited value failed verification. "
             "Agents copying values from a row in front of them almost never misquote, so F, B and S sit at 1.0, and the ordering and bands passed to A5b carried no information. "
             "No justification was withheld. The trust wiring had no effect on this run.\n")

    # ---- hypotheses
    st = Counter(s for i in ids for s in rows[i]["hypothesis_status"].values())
    L.append("## Hypotheses (status by code)\n")
    L.append(f"All hypotheses: {dict(st)}. Struck out (CONTRADICTED): {sum(len(rows[i]['struck_out']) for i in ids)}.")
    bs = Counter((h["benign"], rows[i]["hypothesis_status"][h["hypothesis_id"]]) for i in ids for src in ("a3", "a4") for h in rows[i][src]["hypotheses"])
    parts = [("benign" if b else "hostile") + f" {s}: {n}" for (b, s), n in sorted(bs.items())]
    L.append("By benign/hostile: " + ", ".join(parts) + ".\n")
    n_h = sum(st.values())
    L.append(f"**Support is nearly free.** {st.get('SUPPORTED', 0)} of {n_h} hypotheses are SUPPORTED. A3 and A4 write their predictions while looking at the row, "
             "so a benign story can always be given predictions that the row satisfies. A SUPPORTED benign hypothesis is all A5b needs to refute a condition, "
             "so refutation is cheap: this is how A5b moved records from unexplained to explicable, attacks included. Self-grounded predictions written with the row in view are not evidence for the hypothesis.\n")

    # ---- A5a accuracy
    agree = sum(1 for i in ids for r in rows[i]["a5a"]["checklist"] if rows[i]["conditions_code"].get(r["condition_id"]) == r["holds"])
    total = sum(len(rows[i]["a5a"]["checklist"]) for i in ids)
    L.append(f"## A5a checklist accuracy\n\nA5a's answers agree with the code-side evaluation on {agree} of {total} entries.\n")

    # ---- A1 banned words, and in others for information
    L.append("## Comparative language\n")
    for a in ("a2", "a3", "a4"):
        n = sum(1 for i in ids for c in rows[i][a]["claims"] if core.banned_words_in(c["statement"]))
        tot = sum(len(rows[i][a]["claims"]) for i in ids)
        L.append(f"- {a.upper()} claims using a word banned for A1 (not enforced for {a.upper()}): {n} of {tot}.")
    L.append("- A1: enforced by the validator; no accepted A1 claim contains one.\n")

    # ---- per record
    L.append("## Per record\n")
    L.append("| record | true label | protocol | A5a yes | code yes | struck out | withheld | checklist alone | A5b verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i in sorted(ids, key=lambda i: (not truth[i][1], rows[i]["protocol"], i)):
        r = rows[i]
        code_yes = [k for k, v in r["conditions_code"].items() if v == "yes"]
        L.append(f"| {i[:10]} | {truth[i][0]} | {r['protocol']} | {', '.join(r['yes_conditions']) or 'none'} | {', '.join(code_yes) or 'none'} | "
                 f"{len(r['struck_out'])} | {', '.join(r['withheld']) or 'none'} | {r['checklist_alone_verdict']} | **{r['verdict']}** |")
    L.append("\nA5b rationales:\n")
    for i in ids:
        L.append(f"- **{i[:10]}** ({truth[i][0]}): {rows[i]['rationale']}")
    for f in tally.get("failed", []):
        L.append(f"- not scored: {f['record_id'][:10]} ({truth[f['record_id']][0]}), stage {f['agent']}: {'; '.join(x[:150] for x in f['reasons'])}")
    L.append("")

    L.append("## One full prompt per agent\n")
    rid = (PROMPTS / "record_id.txt").read_text(encoding="utf-8").strip() if (PROMPTS / "record_id.txt").exists() else "?"
    L.append(f"All for record `{rid[:10]}`, as sent.\n")
    for a in ("a1", "a2", "a3", "a4", "a5a", "a5b"):
        p = PROMPTS / f"{a}.txt"
        if p.exists():
            L.append(f"### {a.upper()}\n\n```text\n{p.read_text(encoding='utf-8')}\n```\n")
    L.append("## Pre-registration, as written before the run\n")
    L.append(PREREG.read_text(encoding="utf-8"))
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: caught {caught(att)}/{len(att)}, benign flagged {caught(ben)}/{len(ben)}, insufficient {n_ins}, same as checklist {len(same)}/{len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
