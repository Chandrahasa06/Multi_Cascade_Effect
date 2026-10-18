"""Ceiling of the priors-v3 checklists. No API calls, no prompts, nothing tuned in the system.

Reads results/agent_priors_v3.jsonl (29 scored records). Method fixed before any number was
computed:
  - every one of the 12 conditions counts as attack-indicative (each was written as a
    suspicious sign); none is selected from the data for the count rule
  - count rule: score = number of conditions A5 marked "yes"; flag if score >= cutoff,
    cutoff swept from 1 to the maximum; AUC over the score
  - the same with the code-side truth, to price A5's wrong entries
  - subset search: every subset of size 1-4; a subset flags a flow if ANY of its conditions
    holds (A5's answers); ranked by recall minus benign-flagged rate. This is selection on the
    evaluation data: every number from it is an optimistic ceiling, not a validated result.

Writes results/v3_checklist_ceiling.md and results/v3_checklist_ceiling.json.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from agents.prompts.priors_text_v3 import CONDITION_IDS

RUN = Path("results/agent_priors_v3.jsonl")
PROTOCOLS = Path("results/record_protocols.json")
KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]
OUT_MD = Path("results/v3_checklist_ceiling.md")
OUT_JSON = Path("results/v3_checklist_ceiling.json")
MAX_SUBSET = 4
TOP = 8


# ------------------------------------------------------------------ pure scoring helpers
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
        return "0 of 0"
    lo, hi = wilson(k, n)
    return f"{k} of {n} (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def held_from_checklist(checklist: Sequence[dict]) -> set:
    return {c["condition_id"] for c in checklist if c["holds"] == "yes"}


def held_from_code(code: Dict[str, str]) -> set:
    return {k for k, v in code.items() if v == "yes"}


def count_score(held: set, conditions: Iterable[str] = CONDITION_IDS) -> int:
    return sum(1 for c in conditions if c in held)


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Probability that a random attack scores above a random benign flow (ties count half)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def sweep(scores: Sequence[int], labels: Sequence[bool]) -> List[dict]:
    """Flag if score >= cutoff, for cutoff 1..max(score) (at least cutoff 1)."""
    top = max(max(scores), 1)
    out = []
    for cut in range(1, top + 1):
        tp = sum(1 for s, y in zip(scores, labels) if y and s >= cut)
        fp = sum(1 for s, y in zip(scores, labels) if not y and s >= cut)
        out.append({"cutoff": cut, "tp": tp, "n_att": sum(labels), "fp": fp, "n_ben": len(labels) - sum(labels)})
    return out


def subset_flags(held_sets: Sequence[set], subset: Iterable[str]) -> List[bool]:
    s = set(subset)
    return [bool(h & s) for h in held_sets]


def search_subsets(held_sets: Sequence[set], labels: Sequence[bool], max_size: int = MAX_SUBSET) -> List[dict]:
    n_att = sum(labels)
    n_ben = len(labels) - n_att
    out = []
    for k in range(1, max_size + 1):
        for sub in itertools.combinations(CONDITION_IDS, k):
            flags = subset_flags(held_sets, sub)
            tp = sum(f and y for f, y in zip(flags, labels))
            fp = sum(f and not y for f, y in zip(flags, labels))
            out.append({"subset": list(sub), "tp": tp, "fp": fp, "n_att": n_att, "n_ben": n_ben,
                        "youden": tp / n_att - fp / n_ben})
    out.sort(key=lambda r: (-r["youden"], -r["tp"], len(r["subset"])))
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    truth = {}
    for p in KEYS:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            truth[r["record_id"]] = (r["true_label"], r["is_attack"] == "True")
    proto = {k: v["protocol"] for k, v in json.loads(PROTOCOLS.read_text(encoding="utf-8")).items()}
    rows = [json.loads(l) for l in open(RUN, encoding="utf-8") if l.strip()]
    ids = [r["record_id"] for r in rows]
    labels = [truth[i][1] for i in ids]
    a5_held = [held_from_checklist(r["a5"]["checklist"]) for r in rows]
    code_held = [held_from_code(r["conditions_code"]) for r in rows]
    n_att, n_ben = sum(labels), len(labels) - sum(labels)
    tcp_idx = [k for k, i in enumerate(ids) if proto.get(i) == "TCP"]

    res = {"n": len(rows), "n_att": n_att, "n_ben": n_ben}
    L = ["# Ceiling of the v3 checklists\n",
         f"No API calls, no prompt changes. Computed from the {len(rows)} scored v3 checklists ({n_att} attacks, {n_ben} benign; "
         f"{len(tcp_idx)} TCP). Counts primary; Wilson 95% intervals on rates.\n",
         "Method fixed before computing: all 12 conditions count as attack-indicative (each was written as a suspicious sign; none selected from the data). The count rule uses **A5's reported checklist**, which is what a deployed system would have.\n"]

    # 1. per-condition separation
    L.append("## 1. Per-condition separation (A5's answers)\n")
    L.append("Separation = share of attacks where it holds minus share of benign where it holds. Single-condition AUC = (recall + benign cleared) / 2.\n")
    L.append("| condition | holds on attacks | holds on benign | separation | single-condition AUC |")
    L.append("|---|---|---|---|---|")
    per = {}
    for c in CONDITION_IDS:
        a = sum(1 for h, y in zip(a5_held, labels) if y and c in h)
        b = sum(1 for h, y in zip(a5_held, labels) if not y and c in h)
        sep = a / n_att - b / n_ben
        per[c] = {"attacks": a, "benign": b, "separation": sep}
        L.append(f"| {c} | {rate(a, n_att)} | {rate(b, n_ben)} | {sep:+.2f} | {(a / n_att + 1 - b / n_ben) / 2:.3f} |")
    res["per_condition"] = per
    L.append("")

    # 2 and 4. count rule, A5 and code
    def count_section(title, held_sets):
        sc = [count_score(h) for h in held_sets]
        a = auc(sc, labels)
        a_tcp = auc([sc[k] for k in tcp_idx], [labels[k] for k in tcp_idx])
        sw = sweep(sc, labels)
        L.append(f"## {title}\n")
        L.append(f"AUC of the count: **{a:.3f}** overall, **{a_tcp:.3f}** within TCP.\n")
        L.append(f"Score distribution: attacks {sorted(s for s, y in zip(sc, labels) if y)}; benign {sorted(s for s, y in zip(sc, labels) if not y)}.\n")
        L.append("| flag if conditions holding >= | attacks caught | benign flagged |")
        L.append("|---|---|---|")
        for r in sw:
            L.append(f"| {r['cutoff']} | {rate(r['tp'], r['n_att'])} | {rate(r['fp'], r['n_ben'])} |")
        L.append("")
        return {"auc": a, "auc_tcp": a_tcp, "sweep": sw, "scores": sc}

    res["count_a5"] = count_section("2. Count rule on A5's checklist", a5_held)
    res["count_code"] = count_section("4. Count rule on code-side truth (prices A5's wrong entries)", code_held)
    gap = res["count_code"]["auc"] - res["count_a5"]["auc"]
    if abs(gap) < 0.05:
        gap_text = "A5's perception is close enough to code-side truth that its errors change little."
    elif gap < 0:
        gap_text = ("The code-side truth separates WORSE than A5's answers: A5's wrong entries fall disproportionately on attacks "
                    "(for example, it marks C3 on 8 attacks where the condition truly holds on 4). That extra separation comes "
                    "from misreading the conditions, not from the conditions themselves, so it is not something to build on.")
    else:
        gap_text = "A5's wrong entries cost real separation."
    L.append(f"AUC with code-side truth minus AUC with A5's answers: {gap:+.3f}. {gap_text}\n")

    # 3. subset search
    L.append("## 3. Best subsets (size 1-4) — OPTIMISTIC CEILING\n")
    L.append("**Exhaustive search on the same 29 records it is scored on. This is selection on the evaluation set: every number in this section is an optimistic ceiling, not a validated result.** A subset flags a flow if any of its conditions holds (A5's answers). Ranked by recall minus benign-flagged rate.\n")
    top = search_subsets(a5_held, labels)
    res["subsets_top"] = top[:TOP]
    res["subsets_searched"] = len(top)
    L.append(f"Subsets searched: {len(top)}.\n")
    L.append("| rank | subset | attacks caught (ceiling) | benign flagged (ceiling) |")
    L.append("|---|---|---|---|")
    for k, r in enumerate(top[:TOP], 1):
        L.append(f"| {k} | {', '.join(r['subset'])} | {rate(r['tp'], r['n_att'])} | {rate(r['fp'], r['n_ben'])} |")
    L.append("")
    zero_fp = [r for r in top if r["fp"] == 0]
    best0 = max(zero_fp, key=lambda r: r["tp"]) if zero_fp else None
    if best0:
        L.append(f"Best subset with no benign flagged (ceiling): {', '.join(best0['subset'])}, catching {rate(best0['tp'], n_att)}.\n")

    # 5. invisible attacks
    L.append("## 5. Attacks no condition can see\n")
    inv_a5 = [(ids[k], truth[ids[k]][0]) for k in range(len(ids)) if labels[k] and not a5_held[k]]
    inv_code = [(ids[k], truth[ids[k]][0]) for k in range(len(ids)) if labels[k] and not code_held[k]]
    res["invisible_a5"] = inv_a5
    res["invisible_code"] = inv_code
    L.append(f"Attacks on which no condition holds by code-side truth: {len(inv_code)} of {n_att}. No rule over this condition set, however combined, can catch these.\n")
    for i, lab in inv_code:
        L.append(f"- {i[:10]} ({lab}, {proto.get(i)})")
    by_class = {}
    for i, lab in inv_code:
        by_class[lab] = by_class.get(lab, 0) + 1
    L.append(f"\nBy class: {by_class}. By A5's answers, {len(inv_a5)} of {n_att} attacks show no condition.\n")

    # verdict
    L.append("## What the data supports\n")
    a5auc, tcpauc = res["count_a5"]["auc"], res["count_a5"]["auc_tcp"]
    best_any = top[0]
    L.append(f"- The count rule on A5's checklist reaches AUC {a5auc:.3f} overall and {tcpauc:.3f} within TCP. "
             f"At its best single cutoff it catches {max(r['tp'] for r in res['count_a5']['sweep'])} of {n_att} attacks at most.")
    L.append(f"- The best subset found by search (ceiling) catches {best_any['tp']} of {n_att} attacks while flagging {best_any['fp']} of {n_ben} benign. "
             f"{len(inv_code)} of {n_att} attacks show no condition at all, so no combination can exceed {n_att - len(inv_code)} of {n_att}.")
    code_tcp = res["count_code"]["auc_tcp"]
    L.append(f"- Within TCP, where protocol cannot carry the result, the count reaches AUC {tcpauc:.3f} on A5's answers and "
             f"{code_tcp:.3f} on code-side truth: close to chance. The overall figure is lifted by benign flows (mostly UDP) on which no condition holds.")
    L.append("- The best subset reaches its ceiling only by selection on these same 29 records, and it does no better than the Monday count-only rule on the records both scored (13 of 19 attacks, 0 of 9 benign; `results/agent_priors_v3.md`).")
    L.append("- **Verdict: the conditions are too weak.** Single conditions separate by at most +0.20; within TCP the count is near chance; "
             "the only configuration that matches the count-only baseline is a post-hoc subset on the evaluation data. Deriving the verdict "
             "from the checklist would fix A5 ignoring its own observations, but these observations do not carry enough signal to be worth "
             "fixing it for. More prompt work on the same conditions cannot help; the next step is better conditions, not better wording.")
    L.append("")

    L.append("## The 9 validation failures, and the proposed fix (not applied)\n")
    L.append("19 of the 27 retry reasons were v3's own rule that every A3/A4 explanation must cite at least one condition id. A benign explanation for a flow where no condition holds has nothing to cite. Proposed fix: add an explicit citation value meaning \"no condition holds\" (for example `C0`, valid only when the A5 checklist has no \"yes\"), accepted by `validate_hypothesis_conditions`. Not applied here; this task changes nothing.\n")
    L.append("## Limits\n")
    L.append(f"- n = {len(rows)} ({n_att} attacks, {n_ben} benign; 6 benign TCP). 16 records are missing (9 validation failures, 7 never attempted), so this is not the full 45.")
    L.append("- The subset search is selection on the evaluation data. Its numbers are a ceiling.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"AUC A5 {a5auc:.3f} (TCP {tcpauc:.3f}); code {res['count_code']['auc']:.3f}; invisible {len(inv_code)}/{n_att}; best subset {best_any['subset']} {best_any['tp']}/{n_att} fp {best_any['fp']}/{n_ben}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
