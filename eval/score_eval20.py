"""Scores the eval20 runs against the key, by the rule fixed in results/eval20_prereg.md.
This is the only eval20 code that opens the label key; no decision path imports it.

Verdict -> call (results/selfcheck_preregistered.md, carried into eval20_prereg.md):
  attack  = anomalous_and_unexplained
  benign  = consistent_with_benign or anomalous_but_explicable
  neither = insufficient_evidence, or a record lost to validation (no verdict)
J = caught/10 + cleared/10 - 1. Winner: highest J; ties -> fewer records lost to validation,
then lower cost. Raw recall is not the criterion.

Run: python -m eval.score_eval20 [--models slug ...]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

from agents import openrouter
from eval.run_eval20 import LEDGER, run_dir, run_name
from eval.run_selfcheck import failures_path

KEY = Path("results/eval20_records.json")
REPORT = Path("results/eval20_report.md")
VERDICTS = ("anomalous_and_unexplained", "anomalous_but_explicable", "consistent_with_benign", "insufficient_evidence")
ATTACK_CALL = {"anomalous_and_unexplained"}
BENIGN_CALL = {"consistent_with_benign", "anomalous_but_explicable"}
AGENTS = ("a1", "a2", "a3", "a4", "a5a", "a5b")
N_PER_GROUP = 10
#: facts about runs made before per-call retry waits and per-invocation concurrency were
#: recorded, taken from those runs' console output. Haiku's run printed no HTTP retry of any
#: kind, so its retry wait is a known zero; DeepSeek's first invocation ran at 8.
PRE_RECORDING = {
    "anthropic/claude-haiku-5.5": {"concurrency": [8], "retry_wait_known_zero": True},
    "deepseek/deepseek-v4.1-flash": {"first_invocation_concurrency": 8},
}


def _jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def hypothesis_counts(recs) -> dict:
    """SUPPORTED / CONTRADICTED / UNCHECKABLE per hypothesis source (a3_h..., a4_h...)."""
    out = {}
    for r in recs:
        for hid, st in r["hypothesis_status"].items():
            src = hid[:2]
            out.setdefault(src, {"SUPPORTED": 0, "CONTRADICTED": 0, "UNCHECKABLE": 0})
            out[src][st] += 1
    return out


def score_model(model: str, key: Dict[str, dict], spend: Dict[str, float], variant: str = "") -> dict:
    d = run_dir(model, variant)
    recs = {r["record_id"]: r for r in _jsonl(d / "records.jsonl")}
    fails = _jsonl(failures_path(d / "records.jsonl"))
    tally = json.loads((d / "tally.json").read_text(encoding="utf-8")) if (d / "tally.json").exists() else {}
    att = [k for k, v in key.items() if v["is_attack"]]
    ben = [k for k, v in key.items() if not v["is_attack"]]
    v = {k: recs[k]["verdict"] if k in recs else None for k in key}
    caught = sum(v[k] in ATTACK_CALL for k in att)
    flagged = sum(v[k] in ATTACK_CALL for k in ben)
    cleared = sum(v[k] in BENIGN_CALL for k in ben)
    lost = {a: sum(f["agent"] == a for f in fails) for a in AGENTS}
    not_run = [k for k in key if k not in recs and k not in {f["record_id"] for f in fails}]
    dist = {x: sum(r["verdict"] == x for r in recs.values()) for x in VERDICTS}
    pre = PRE_RECORDING.get(model, {})
    calls = [m for r in recs.values() for m in r["stage_meta"].values()
             if not m.get("cached") and m.get("elapsed_s") is not None]
    call_lat = [m["elapsed_s"] for m in calls]
    if pre.get("retry_wait_known_zero"):
        excl = list(call_lat)
    else:
        excl = [m["elapsed_s"] - m["retry_wait_s"] for m in calls if m.get("retry_wait_s") is not None]
    rec_lat = [r["elapsed_s"] for r in recs.values() if r.get("elapsed_s") is not None]
    http_retries = sum(m.get("http_retries") or 0 for m in calls)
    retry_wait = sum(m.get("retry_wait_s") or 0.0 for m in calls)
    conc = pre.get("concurrency") or [
        (inv["concurrency"] if inv["concurrency"] is not None else pre.get("first_invocation_concurrency"))
        for inv in tally.get("invocations", [])]
    insuff = dist["insufficient_evidence"]
    per_class = {}
    for k in att:
        c = key[k]["true_label"]
        per_class.setdefault(c, [0, 0])
        per_class[c][1] += 1
        per_class[c][0] += v[k] in ATTACK_CALL
    return {
        "model": model, "scored": len(recs), "caught": caught, "flagged": flagged, "cleared": cleared,
        "J": caught / N_PER_GROUP + cleared / N_PER_GROUP - 1,
        "lost": lost, "lost_total": sum(lost.values()), "not_run": not_run, "verdicts": dist,
        "insufficient_share": insuff / len(key), "inconclusive": insuff / len(key) > 0.30,
        "cost_usd": spend.get(run_name(model, variant), 0.0),
        "hypotheses": hypothesis_counts(recs.values()), "variant": variant,
        "median_call_s": statistics.median(call_lat) if call_lat else None,
        "median_call_excl_retry_s": statistics.median(excl) if excl else None,
        "calls_with_known_retry_wait": len(excl), "calls_timed": len(call_lat),
        "http_retries_recorded": http_retries, "retry_wait_s_recorded": retry_wait,
        "concurrency": conc,
        "median_record_s": statistics.median(rec_lat) if rec_lat else None,
        "stopped_at": tally.get("stopped_at"), "per_class": per_class,
    }


def winner(rows: List[dict]) -> dict:
    return sorted(rows, key=lambda r: (-round(r["J"], 9), r["lost_total"], r["cost_usd"]))[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--variant", default="")
    args = ap.parse_args()
    global REPORT
    if args.variant:
        REPORT = Path(f"results/eval20_{args.variant}_report.md")
    key = {r["record_id"]: r for r in json.loads(KEY.read_text(encoding="utf-8"))["records"]}
    _, spend = openrouter.ledger_totals(LEDGER)
    models = args.models or openrouter.load_config()["sequence"]
    rows = [score_model(m, key, spend, args.variant) for m in models
            if (run_dir(m, args.variant) / "tally.json").exists()]
    L = ["# eval20 results", "", "Rule: `results/eval20_prereg.md`. Caught = anomalous_and_unexplained on an attack; "
         "cleared = consistent_with_benign or anomalous_but_explicable on a benign flow; insufficient_evidence and "
         "lost records are neither. J = caught/10 + cleared/10 - 1.", "",
         "| model | caught /10 | benign flagged /10 | benign cleared /10 | J | lost (A1/A2/A3/A4/A5a/A5b) | "
         "unexplained / explicable / consistent / insufficient | cost | concurrency | median call s, raw | "
         "median call s, excl. retry waits | median record s, raw |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.1f}"
    for r in rows:
        lost = "/".join(str(r["lost"][a]) for a in AGENTS)
        vd = " / ".join(str(r["verdicts"][x]) for x in VERDICTS)
        conc = " then ".join(str(c) for c in r["concurrency"]) or "-"
        excl = f(r["median_call_excl_retry_s"])
        if r["calls_with_known_retry_wait"] < r["calls_timed"]:
            excl += f" (n={r['calls_with_known_retry_wait']} of {r['calls_timed']})"
        L.append(f"| `{r['model']}` | {r['caught']} | {r['flagged']} | {r['cleared']} | {r['J']:+.2f} | {lost} | {vd} | "
                 f"${r['cost_usd']:.4f} | {conc} | {f(r['median_call_s'])} | {excl} | {f(r['median_record_s'])} |")
    L.append("")
    for r in rows:
        notes = []
        if r["inconclusive"]:
            notes.append(f"insufficient_evidence on {r['insufficient_share']:.0%} of records (>30%): inconclusive")
        if r["not_run"]:
            notes.append(f"{len(r['not_run'])} record(s) not run")
        if r["stopped_at"]:
            notes.append(f"stopped at {r['stopped_at']['record_id'][:10]}: {r['stopped_at']['reason'][:120]}")
        if r["http_retries_recorded"]:
            notes.append(f"{r['http_retries_recorded']} HTTP retries recorded, {r['retry_wait_s_recorded']:.0f}s of retry waits")
        hc = "; ".join(f"{src.upper()} hypotheses {h['SUPPORTED']} supported / {h['CONTRADICTED']} struck out / "
                       f"{h['UNCHECKABLE']} uncheckable" for src, h in sorted(r["hypotheses"].items()))
        notes.append(hc)
        pc = ", ".join(f"{c} {k}/{n}" for c, (k, n) in sorted(r["per_class"].items()))
        L.append(f"- `{r['model']}` attacks by class: {pc}" + ("; " + "; ".join(notes) if notes else ""))
    if rows:
        w = winner(rows)
        L += ["", f"**Winner by the pre-registered rule: `{w['model']}`** (J {w['J']:+.2f}, "
                  f"{w['lost_total']} lost, ${w['cost_usd']:.4f})."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(json.dumps(rows, indent=1, default=str), file=open(REPORT.with_suffix(".json"), "w", encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
