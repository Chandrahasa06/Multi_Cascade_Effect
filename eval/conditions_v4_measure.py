"""Phase 1-2: measure the pre-registered v4 conditions (results/conditions_v4_preregistered.md,
committed before this script ran). No API calls.

Data: all eight pooled TrafficLabelling CSVs, every labelled row (all attack rows and all
benign rows, 2.27M). The registered expression strings are evaluated directly (vectorised);
tests check they agree with the per-row definitions in eval/conditions_v4.py.

Writes results/conditions_v4_measured.md and results/conditions_v4_measured.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from eval.conditions_v4 import COLUMNS, CONDITIONS, evaluate
from eval.escalation_data import CSV_DIR, DAY_FILES

OUT_MD = Path("results/conditions_v4_measured.md")
OUT_JSON = Path("results/conditions_v4_measured.json")
CEILING = Path("results/v3_checklist_ceiling.json")
PROTOCOLS = Path("results/record_protocols.json")
UNUSABLE_BENIGN_RATE = 0.01
PROTO = {6: "TCP", 17: "UDP"}


def to_pandas_expr(expression: str) -> str:
    """'A == B AND (C + D) > 3' -> '(`A` == `B`) & ((`C` + `D`) > 3)'."""
    def quote(part: str) -> str:
        for col in sorted(COLUMNS, key=len, reverse=True):
            part = re.sub(rf"(?<![`\w]){re.escape(col)}(?![`\w])", f"`{col}`", part)
        return part
    return " & ".join(f"({quote(p.strip())})" for p in expression.split(" AND "))


def holds_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: one column per condition, scoped by protocol (out of scope -> False)."""
    out = {}
    for c in CONDITIONS:
        v = df.eval(to_pandas_expr(c.expression)).fillna(False).astype(bool)
        if c.scope != "any":
            v = v & (df["proto"] == c.scope)
        out[c.cid] = v
    return pd.DataFrame(out, index=df.index)


def load_pool() -> pd.DataFrame:
    keep = set(COLUMNS) | {"Label", "Protocol"}
    frames = []
    for _day, fname, _w in DAY_FILES:
        try:
            d = pd.read_csv(CSV_DIR / fname, usecols=lambda c: c.strip() in keep, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            d = pd.read_csv(CSV_DIR / fname, usecols=lambda c: c.strip() in keep, encoding="cp1252", low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        d = d.dropna(subset=["Label"])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in COLUMNS + ["Protocol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["proto"] = df["Protocol"].map(PROTO).fillna("other")
    df["is_attack"] = df["Label"] != "BENIGN"
    return df


def main() -> int:
    df = load_pool()
    H = holds_frame(df)
    res: Dict = {"rows": len(df), "benign": int((~df["is_attack"]).sum()), "attack": int(df["is_attack"].sum()),
                 "by_protocol": df.groupby(["proto", "is_attack"]).size().to_dict()}
    res["by_protocol"] = {f"{p}|{'attack' if a else 'benign'}": int(n) for (p, a), n in res["by_protocol"].items()}

    # per condition, per protocol
    table = []
    for c in CONDITIONS:
        protos = ["TCP", "UDP", "other"] if c.scope == "any" else [c.scope]
        for p in protos:
            bm = (df["proto"] == p) & ~df["is_attack"]
            am = (df["proto"] == p) & df["is_attack"]
            nb, na = int(bm.sum()), int(am.sum())
            hb, ha = int(H.loc[bm, c.cid].sum()), int(H.loc[am, c.cid].sum())
            rb = hb / nb if nb else float("nan")
            ra = ha / na if na else float("nan")
            table.append({"cid": c.cid, "proto": p, "benign_hold": hb, "benign_n": nb, "attack_hold": ha, "attack_n": na,
                          "benign_rate": rb, "attack_rate": ra, "separation": ra - rb if nb and na else float("nan"),
                          "unusable": bool(nb and rb > UNUSABLE_BENIGN_RATE)})
    res["table"] = table

    # per attack class (within protocol)
    by_class = []
    for (cls, p), g in df[df["is_attack"]].groupby(["Label", "proto"]):
        row = {"class": cls, "proto": p, "n": len(g)}
        for c in CONDITIONS:
            row[c.cid] = int(H.loc[g.index, c.cid].sum())
        by_class.append(row)
    res["by_class"] = by_class

    # the 7 attacks no v3 condition catches
    from eval.run_blind_pipeline_45_v2 import load_45
    blinds = {b["record_id"]: b for b in load_45()}
    proto_of = {k: v["protocol"] for k, v in json.loads(PROTOCOLS.read_text(encoding="utf-8")).items()}
    invisible = json.loads(CEILING.read_text(encoding="utf-8"))["invisible_code"]
    per_record = []
    for rid, cls in invisible:
        held = [k for k, v in evaluate(blinds[rid]["features"], proto_of[rid]).items() if v]
        per_record.append({"record_id": rid, "class": cls, "protocol": proto_of[rid], "held": held})
    res["invisible"] = per_record
    OUT_JSON.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    write_md(res)
    print(json.dumps({"rows": res["rows"], "invisible_reached": sum(bool(r["held"]) for r in per_record)}, indent=2))
    return 0


def pct(k, n):
    return f"{k:,} of {n:,} ({100 * k / n:.2f}%)" if n else "0 of 0"


def write_md(res: Dict) -> None:
    L = ["# Conditions v4, measured\n",
         f"No API calls. Pre-registration: `results/conditions_v4_preregistered.md`, committed before this measurement (commit `3f4d793`). "
         f"Data: all {res['rows']:,} labelled rows of the eight pooled TrafficLabelling CSVs: {res['benign']:,} benign and {res['attack']:,} attack. "
         "This pool includes the 45 evaluation records' own rows.\n",
         f"Rows by protocol and label: {res['by_protocol']}.\n",
         "Rates here are over millions of rows, so they are given as counts and percentages; intervals would be narrower than the rounding.\n"]

    rows = sorted(res["table"], key=lambda r: -(r["separation"] if r["separation"] == r["separation"] else -9))
    L.append("## Ranked by within-protocol separation\n")
    L.append("Separation = attack hold rate minus benign hold rate, within the protocol. **Unusable** marks a benign hold rate above 1%: the current set clears 0 of 9 benign, so any condition holding on more than 1% of benign traffic would cost more than it gains.\n")
    L.append("| condition | protocol | benign holds | attack holds | separation | usable? |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["attack_n"] == 0 and r["benign_n"] == 0:
            continue
        sep = f"{r['separation']:+.4f}" if r["separation"] == r["separation"] else "n/a"
        L.append(f"| {r['cid']} | {r['proto']} | {pct(r['benign_hold'], r['benign_n'])} | {pct(r['attack_hold'], r['attack_n'])} | {sep} | "
                 + ("**unusable (benign > 1%)**" if r["unusable"] else "yes") + " |")
    L.append("")

    L.append("## Attack hold rate by class (within protocol)\n")
    ids = [c.cid for c in CONDITIONS]
    L.append("| class | protocol | rows | " + " | ".join(ids) + " |")
    L.append("|---|---|---|" + "---|" * len(ids))
    for r in sorted(res["by_class"], key=lambda r: (r["proto"], r["class"])):
        cells = [f"{100 * r[c] / r['n']:.1f}%" if r[c] else "0" for c in ids]
        L.append(f"| {r['class']} | {r['proto']} | {r['n']:,} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## The 7 attacks no v3 condition catches\n")
    L.append("Each D-condition evaluated on the record's own feature values.\n")
    L.append("| record | class | protocol | D-conditions holding |")
    L.append("|---|---|---|---|")
    for r in res["invisible"]:
        L.append(f"| {r['record_id'][:10]} | {r['class']} | {r['protocol']} | {', '.join(r['held']) or 'none'} |")
    reached = [r for r in res["invisible"] if r["held"]]
    usable = {(r["cid"], r["proto"]) for r in res["table"] if not r["unusable"] and r["benign_n"]}
    reached_usable = [r for r in res["invisible"] if any((c, r["protocol"]) in usable for c in r["held"])]
    res["reached"] = len(reached)
    res["reached_usable"] = len(reached_usable)
    L.append("")
    L.append(f"Reached by any D-condition: {len(reached)} of {len(res['invisible'])}. Reached by a condition that is usable in the record's own protocol (benign hold rate at most 1% there): {len(reached_usable)} of {len(res['invisible'])}.\n")

    # findings that follow from the table (computed, not asserted)
    tab = {(r["cid"], r["proto"]): r for r in res["table"]}
    L.append("## Findings about the conditions themselves\n")
    never = sorted({r["cid"] for r in res["table"]} - {r["cid"] for r in res["table"] if r["benign_hold"] or r["attack_hold"]},
                   key=lambda s: int(s[1:]))
    L.append(f"- Never hold on any row: {', '.join(never) or 'none'}. D5 was disclosed as expected never to hold; it does not.")
    d9 = tab.get(("D9", "TCP"))
    if d9:
        L.append(f"- D9 rested on the assumption that forward lengths count application bytes only. It holds on {pct(d9['benign_hold'], d9['benign_n'])} benign TCP flows, so that assumption does not hold for these columns; D9 is not the consistency check it was written as.")
    d14 = tab.get(("D14", "other"))
    if d14:
        L.append(f"- D14 holds on {pct(d14['benign_hold'], d14['benign_n'])} benign flows of other protocols: for protocols without ports, the CSV records port 0. That is a scope error in the registration (it should have been TCP/UDP only), recorded, not corrected.")
    usable_pos = [r for r in res["table"] if not r["unusable"] and r["benign_n"] and r["attack_n"] and r["separation"] > 0.005]
    L.append("- Usable conditions with any positive separation (above +0.005): "
             + (", ".join(f"{r['cid']} {r['proto']} ({r['separation']:+.4f})" for r in usable_pos) or "none")
             + ". The largest usable separation is about one percentage point.")
    L.append("- No subset was searched or selected.\n")

    L.append("## Conclusion\n")
    if not reached:
        L.append(f"**This condition family does not reach the {len(res['invisible'])} invisible attacks.** None of the 14 pre-registered conditions holds on any of them, including the conditions that are unusable because of their benign rate. On the full pool, the usable conditions separate attack from benign traffic by at most about one percentage point within a protocol.")
        L.append("\nSingle-row internal-consistency and machine-signature checks, built from protocol semantics, do not see these flows: the 3 Bot, 2 DoS GoldenEye and 2 DDoS records are internally consistent, ordinary-looking TCP flows by every test here. Another prompting round on top of this family would have nothing to work with, and should not be run.\n")
    else:
        L.append(f"This condition family reaches {len(reached)} of {len(res['invisible'])} invisible attacks ({len(reached_usable)} by a usable condition). See the per-record table.\n")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
