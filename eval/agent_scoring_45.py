"""Before and after on the same 45 records (batches 1-3): the re-run after fixes 1
and 2. Writes results/agent_scoring_45_v2.md and results/agent_scoring_45_v2.json.

Definitions are the ones the earlier reports use, so the numbers are comparable:
  caught   = benign_plausibility < 0.30 (VERDICT_THRESHOLD), predicted attack
  ungrounded = empty escalation neighbourhood (neighbourhood size 0)

Counts are primary. Wilson 95% intervals are given on every rate. n = 45 is small.
This is a re-run after a fix, not an independent measurement: say so wherever
the result is quoted.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

THRESHOLD = 0.30
BEFORE_SOURCES = [
    ("batch1", Path("results/agent_run_20_v2.jsonl"), Path("results/agent_key_20.csv")),
    ("batch2", Path("results/agent_run_20b.jsonl"), Path("results/agent_key_20b.csv")),
    ("batch3", Path("results/agent_run_10c.jsonl"), Path("results/agent_key_10c.csv")),
]
AFTER_PATH = Path("results/agent_run_45_v2.jsonl")
TALLY_PATH = Path("results/agent_run_45_v2_tally.json")
OUT_MD = Path("results/agent_scoring_45_v2.md")
OUT_JSON = Path("results/agent_scoring_45_v2.json")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return f"{k}/0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_before() -> Dict[str, dict]:
    """The 45 records as originally scored, with their true labels."""
    out = {}
    for batch, run_path, key_path in BEFORE_SOURCES:
        key = pd.read_csv(key_path).set_index("record_id")
        for r in load_jsonl(run_path):
            k = key.loc[r["record_id"]]
            out[r["record_id"]] = {"batch": batch, "run": r, "is_attack": bool(k["is_attack"]),
                                   "true_label": k["true_label"], "admitted_by": k["admitted_by"]}
    return out


def load_after(before: Dict[str, dict]) -> Dict[str, dict]:
    out = {}
    for r in load_jsonl(AFTER_PATH):
        b = before[r["record_id"]]
        out[r["record_id"]] = {"batch": b["batch"], "run": r, "is_attack": b["is_attack"],
                               "true_label": b["true_label"], "admitted_by": b["admitted_by"]}
    return out


def per_record(side: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for rid, d in side.items():
        r = d["run"]
        bp = float(r["a5"]["benign_plausibility"])
        rows.append({
            "record_id": rid, "batch": d["batch"], "true_label": d["true_label"], "is_attack": d["is_attack"],
            "bp": bp, "caught": bp < THRESHOLD, "at_threshold": bp == THRESHOLD,
            # the code's own flag: neighbourhood_size < MIN_NEIGHBOURHOOD_SIZE (30)
            "ungrounded": bool(r["escalation_neighbourhood"]["ungrounded"]),
            "schema_retries": sum(int(m.get("schema_retries", 0)) for m in r["call_metadata"].values()),
            "neighbourhood_size": r["escalation_neighbourhood"]["neighbourhood_size"],
            "A2_C": r["trust_scores"]["a2"]["C"], "A2_E": r["trust_scores"]["a2"]["E"],
            "A2_V": r["trust_scores"]["a2"]["V"], "A2_T": r["trust_scores"]["a2"]["T"],
            "chain_vs_independent": r.get("chain_vs_independent"),
            "rationale": r["a5"]["rationale"],
        })
    return pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)


def confusion(df: pd.DataFrame) -> dict:
    tp = int((df.is_attack & df.caught).sum())
    fn = int((df.is_attack & ~df.caught).sum())
    fp = int((~df.is_attack & df.caught).sum())
    tn = int((~df.is_attack & ~df.caught).sum())
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def sorted_bp(df: pd.DataFrame, is_attack: bool) -> List[float]:
    return sorted(round(x, 3) for x in df.loc[df.is_attack == is_attack, "bp"].tolist())


def main() -> int:
    before = load_before()
    after = load_after(before)
    assert len(before) == 45, len(before)
    tally = json.loads(TALLY_PATH.read_text(encoding="utf-8")) if TALLY_PATH.exists() else {}
    dfb = per_record({k: before[k] for k in after})  # same 45 ids on both sides
    dfa = per_record(after)
    if len(dfa) == 0:
        raise SystemExit("no after-records yet")

    merged = dfb.merge(dfa, on="record_id", suffixes=("_old", "_new"), how="inner")
    flips = merged[merged.caught_old != merged.caught_new]
    result = {
        "n_before": len(dfb), "n_after": len(dfa), "n_paired": len(merged),
        "confusion_before": confusion(dfb), "confusion_after": confusion(dfa),
        "bp_before_attack": sorted_bp(dfb, True), "bp_before_benign": sorted_bp(dfb, False),
        "bp_after_attack": sorted_bp(dfa, True), "bp_after_benign": sorted_bp(dfa, False),
        "at_threshold_before": int(dfb.at_threshold.sum()), "at_threshold_after": int(dfa.at_threshold.sum()),
        "flips": flips[["record_id", "batch_old", "true_label_old", "bp_old", "bp_new", "caught_old", "caught_new"]].to_dict("records"),
        "tally": tally,
        "a2_trust_before": {c: dfb[f"A2_{c}"].describe().to_dict() for c in "CEVT"},
        "a2_trust_after": {c: dfa[f"A2_{c}"].describe().to_dict() for c in "CEVT"},
        "chain_vs_independent_before": dfb.chain_vs_independent.describe().to_dict(),
        "chain_vs_independent_after": dfa.chain_vs_independent.describe().to_dict(),
        "caught_attacks_ungrounded_after": [int(x) for x in dfa.loc[dfa.is_attack & dfa.caught, "ungrounded"]],
        "missed_attacks_ungrounded_after": [int(x) for x in dfa.loc[dfa.is_attack & ~dfa.caught, "ungrounded"]],
        "caught_attacks_ungrounded_before": [int(x) for x in dfb.loc[dfb.is_attack & dfb.caught, "ungrounded"]],
        "missed_attacks_ungrounded_before": [int(x) for x in dfb.loc[dfb.is_attack & ~dfb.caught, "ungrounded"]],
    }
    paired = merged[["record_id", "A2_V_old", "A2_V_new", "A2_C_old", "A2_C_new", "A2_E_old", "A2_E_new", "A2_T_old", "A2_T_new"]]
    result["a2_paired_mean_diff"] = {c: float((paired[f"A2_{c}_new"] - paired[f"A2_{c}_old"]).mean()) for c in "CEVT"}
    result["a2_paired_counts"] = {c: {"up": int((paired[f"A2_{c}_new"] > paired[f"A2_{c}_old"] + 1e-9).sum()),
                                     "down": int((paired[f"A2_{c}_new"] < paired[f"A2_{c}_old"] - 1e-9).sum()),
                                     "same": int((abs(paired[f"A2_{c}_new"] - paired[f"A2_{c}_old"]) <= 1e-9).sum())}
                                  for c in "CEVT"}

    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    write_md(result, dfb, dfa, merged, flips)
    print(f"wrote {OUT_MD} and {OUT_JSON}")
    return 0


def write_md(r: dict, dfb: pd.DataFrame, dfa: pd.DataFrame, merged: pd.DataFrame, flips: pd.DataFrame) -> None:
    L: List[str] = []
    L.append("# Agent pipeline, before and after fixes 1 and 2, on the same 45 records\n")
    L.append("**This is a re-run after a fix, not an independent measurement.** The same 45 records (batches 1, 2 and 3, the ones scored originally), the same blind inputs, the same 0.30 threshold and the same model. The earlier results stand as recorded.\n")
    L.append("Fix 1: the payload filter accepts claims about the absence or size of payload and rejects claims about its contents. Fix 2: A2 (`a2_behaviour_v3`) receives the empirical grounding block the other chain agents receive. A2, A3 and A5 were called fresh and did not use the cache; A1 and A4 used their recorded responses.\n")
    L.append(f"n before = {r['n_before']}, n after = {r['n_after']}, paired = {r['n_paired']}. Every rate is a count with a Wilson 95% interval.\n")
    L.append("## 1. Confusion matrix (counts)\n")
    cb, ca = r["confusion_before"], r["confusion_after"]
    L.append("| | before: caught | before: not caught | after: caught | after: not caught |")
    L.append("|---|---|---|---|---|")
    L.append(f"| attack (n={int(dfb.is_attack.sum())} before, n={int(dfa.is_attack.sum())} after) | {cb['tp']} | {cb['fn']} | {ca['tp']} | {ca['fn']} |")
    L.append(f"| benign (n={int((~dfb.is_attack).sum())} before, n={int((~dfa.is_attack).sum())} after) | {cb['fp']} | {cb['tn']} | {ca['fp']} | {ca['tn']} |\n")
    L.append(f"Recall (attacks caught): before {fmt_rate(cb['tp'], cb['tp'] + cb['fn'])}; after {fmt_rate(ca['tp'], ca['tp'] + ca['fn'])}.")
    L.append(f"Specificity (benign not caught): before {fmt_rate(cb['tn'], cb['tn'] + cb['fp'])}; after {fmt_rate(ca['tn'], ca['tn'] + ca['fp'])}.\n")
    L.append(f"Records scoring exactly 0.30 (reported as such, not rounded): before {r['at_threshold_before']}; after {r['at_threshold_after']}.\n")
    L.append("## 2. Sorted plausibility distributions\n")
    L.append("| group | before | after |")
    L.append("|---|---|---|")
    L.append(f"| attacks | {r['bp_before_attack']} | {r['bp_after_attack']} |")
    L.append(f"| benign | {r['bp_before_benign']} | {r['bp_after_benign']} |\n")
    L.append("## 3. Verdict flips (records that crossed the threshold)\n")
    if len(flips) == 0:
        L.append("No record changed its caught / not-caught verdict.\n")
    else:
        L.append("| record | batch | true label | old bp | new bp | old verdict | new verdict |")
        L.append("|---|---|---|---|---|---|---|")
        for _, f in flips.iterrows():
            L.append(f"| {f.record_id[:10]} | {f.batch_old} | {f.true_label_old} | {f.bp_old:.2f} | {f.bp_new:.2f} | "
                     f"{'caught' if f.caught_old else 'not caught'} | {'caught' if f.caught_new else 'not caught'} |")
        L.append("")
    L.append("## 4. Per-record table (old and new score side by side)\n")
    L.append("| record | batch | true label | old bp | new bp | old ungrounded | new ungrounded | A2 V old -> new |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, m in merged.iterrows():
        L.append(f"| {m.record_id[:10]} | {m.batch_old} | {m.true_label_old} | {m.bp_old:.2f} | {m.bp_new:.2f} | "
                 f"{m.ungrounded_old} | {m.ungrounded_new} | {m.A2_V_old:.2f} -> {m.A2_V_new:.2f} |")
    L.append("")
    L.append("## 5. Schema failures\n")
    t = r.get("tally", {})
    failed_after = t.get("failed_records", {})
    L.append("Denominators differ on purpose. 'Selected' is what batches 1-3 were given. 'Scored' is what they produced, and it is the set the re-run covers. The records that failed in the original run produced no output, so the re-run did not include them and the after-fix rate cannot be measured on them.\n")
    L.append("| batch | selected | failed before fix 1 | scored before (= re-run set) | failed in re-run | schema retries before (scored set) | schema retries in re-run |")
    L.append("|---|---|---|---|---|---|---|")
    for batch, selected, failed_before in [("batch1", 20, 0), ("batch2", 20, 4), ("batch3", 10, 1)]:
        scored = int((dfb.batch == batch).sum())
        n_failed_after = sum(1 for v in failed_after.values() if v.get("batch") == batch)
        ret_b = int(dfb.loc[dfb.batch == batch, "schema_retries"].sum())
        ret_a = int(dfa.loc[dfa.batch == batch, "schema_retries"].sum())
        L.append(f"| {batch} | {selected} | {failed_before} of {selected} | {scored} | {n_failed_after} of {scored} | {ret_b} | {ret_a} |")
    L.append(f"| all | 50 | 5 of 50 | 45 | {len(failed_after)} of 45 | {int(dfb.schema_retries.sum())} | {int(dfa.schema_retries.sum())} |\n")
    L.append("Retries are counts of re-prompts within calls that eventually succeeded. A failure is a call that exhausted its retries. Fix 1 changes which predictions are accepted, so it changes the retry count, not what a failure is.\n")
    for rid, v in failed_after.items():
        L.append(f"- `{rid}` ({v.get('batch')}, {v.get('agent')}): " + " | ".join(v.get("retry_reasons", []))[:300])
    L.append("")
    L.append("## 6. A2 trust scores, before and after fix 2 (C, E, V, T)\n")
    L.append("| component | before median | after median | paired up | paired down | paired same | mean paired diff |")
    L.append("|---|---|---|---|---|---|---|")
    for c in "CEVT":
        b = r["a2_trust_before"][c]; a = r["a2_trust_after"][c]; cnt = r["a2_paired_counts"][c]
        L.append(f"| {c} | {b['50%']:.3f} | {a['50%']:.3f} | {cnt['up']} | {cnt['down']} | {cnt['same']} | {r['a2_paired_mean_diff'][c]:+.3f} |")
    L.append("")
    L.append("## 7. chain_vs_independent\n")
    cb2 = r["chain_vs_independent_before"]; ca2 = r["chain_vs_independent_after"]
    L.append(f"Before: mean {cb2.get('mean', float('nan')):.3f} over n={int(cb2.get('count', 0))}. After: mean {ca2.get('mean', float('nan')):.3f} over n={int(ca2.get('count', 0))}.\n")
    L.append("## 8. Grounding correlation (caught implies ungrounded)\n")
    cab, cmb = r["caught_attacks_ungrounded_before"], r["missed_attacks_ungrounded_before"]
    caa, cma = r["caught_attacks_ungrounded_after"], r["missed_attacks_ungrounded_after"]
    L.append("'Ungrounded' is the code's own flag: escalation neighbourhood smaller than MIN_NEIGHBOURHOOD_SIZE = 30. The ungrounded cap limits benign_plausibility to 0.30, so an ungrounded flow sits at or below the threshold.\n")
    L.append(f"- Caught attacks that are ungrounded (forward claim: caught implies ungrounded): before {fmt_rate(sum(cab), len(cab))}; after {fmt_rate(sum(caa), len(caa))}.")
    L.append(f"- Ungrounded misses (ungrounded attacks not caught, i.e. at exactly 0.30): before {sum(cmb)} of {len(cmb)} missed attacks; after {sum(cma)} of {len(cma)} missed attacks.")
    L.append(f"- Forward claim holds in the re-run: {'yes' if sum(caa) == len(caa) else 'no'} (every caught attack ungrounded: {sum(caa)} of {len(caa)}).\n")
    L.append("## Limits\n")
    L.append("- n = 45. One record flipping changes a rate by 2.2 points. Counts are primary.")
    L.append("- The fixes change what A2 and A3 can say, so downstream A5 scores move with them. This is measured, not separated.")
    L.append("- The re-run uses the same model and temperature. A stochastic model gives a different draw on each call. Two runs of unfixed code would not match exactly either. That noise is not measured here.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
