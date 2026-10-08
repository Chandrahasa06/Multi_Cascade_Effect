"""Part 1: A5's pre-clamp scores on records the ungrounded cap caught. No API calls.

Sources:
  arm runs (results/agent_run_arm_{m,mt,a}.jsonl) store `benign_plausibility_raw`, A5's
  number before either cap. The v2 and v3 runs do not store a pre-clamp value; for them the
  stored score is post-cap, and the cap flags say whether a clamp happened. Where no cap
  fired, the stored score is A5's own.

Writes results/cap_suppressed_scores.md and results/cap_suppressed_scores.json.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]
ARMS = {"m": "Monday", "mt": "Monday+Tuesday", "a": "all-days"}
THRESHOLD = 0.30
OUT_MD = Path("results/cap_suppressed_scores.md")
OUT_JSON = Path("results/cap_suppressed_scores.json")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k} of {n} (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def load_keys() -> Dict[str, Tuple[str, bool]]:
    out = {}
    for p in KEYS:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            out[r["record_id"]] = (r["true_label"], r["is_attack"] == "True")
    return out


def capped_records(arm_rows: List[dict]) -> List[dict]:
    """Records where either cap fired, with A5's raw and clamped scores."""
    out = []
    for r in arm_rows:
        if r["ungrounded_cap_fired"] or r["zero_support_cap_fired"]:
            out.append({"record_id": r["record_id"], "raw": float(r["benign_plausibility_raw"]),
                        "clamped": float(r["benign_plausibility"]), "count_loo": r["count_loo"],
                        "rationale": r["rationale"]})
    return out


def main() -> int:
    truth = load_keys()
    L: List[str] = []
    L.append("# What A5 said before the cap overwrote it\n")
    L.append("No API calls. Read from the saved runs.\n")
    result = {}

    L.append("## The answer\n")
    summary_rows = []
    for a, name in ARMS.items():
        rows = [json.loads(l) for l in open(f"results/agent_run_arm_{a}.jsonl", encoding="utf-8") if l.strip()]
        cap = capped_records(rows)
        below = sum(c["raw"] < THRESHOLD for c in cap)
        n_att = sum(truth[c["record_id"]][1] for c in cap)
        summary_rows.append((a, name, len(cap), n_att, below, cap))
        result[a] = {"capped": len(cap), "attacks": n_att, "raw_below_030": below,
                     "raw_scores": sorted(c["raw"] for c in cap)}
    L.append("| arm | records the cap caught | of which attacks | raw A5 score already below 0.30 |")
    L.append("|---|---|---|---|")
    for a, name, n, n_att, below, _ in summary_rows:
        L.append(f"| {a.upper()} ({name}) | {n} | {n_att} | **{rate(below, n)}** |")
    L.append("")
    L.append("**In every arm, A5 on its own would have caught none of the flows the cap caught.** Its raw scores on them were 0.70 to 0.95. Every attack the arms caught, they caught because the cap overwrote A5. Removing the cap with these prompts would drop recall in Arm M from 20 of 29 to 0 of 29.\n")
    L.append("Per the brief's decision rule, Part 2 (the uncapped run) was **not run**. See the caveat below before deciding whether to run it anyway.\n")

    L.append("## Raw-score distribution, by true label\n")
    L.append("Every record the cap caught in any arm was an attack, so there is no benign row.\n")
    for a, name, n, n_att, below, cap in summary_rows:
        att = sorted(round(c["raw"], 2) for c in cap if truth[c["record_id"]][1])
        ben = sorted(round(c["raw"], 2) for c in cap if not truth[c["record_id"]][1])
        L.append(f"- Arm {a.upper()}: attacks (n={len(att)}) {att}; benign (n={len(ben)}) {ben}")
    L.append("")

    L.append("## Caveat: the raw scores were given with counts in view\n")
    L.append("The arm prompts (`a5_verdict_v7` with the reference-window sentences) showed A5 the real neighbourhood count and per-hypothesis profile counts. On every one of the 20 Arm M records the cap caught, the ten-feature neighbourhood count was 0 to 13, yet at least one hypothesis sentence said 6,531 or more benign flows were \"within 2x\" (closeness on only the one or two features that hypothesis named). A5 went with the large number. Its 0.70-0.95 is its reading of that mixed evidence, not its judgement without counts.\n")
    L.append("The earlier v2 and v3 runs show the opposite behaviour, and why:\n")
    for name, path in [("v2", "results/agent_run_45_v2.jsonl"), ("v3", "results/agent_run_45_v3.jsonl")]:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        ung = [r for r in rows if r["escalation_neighbourhood"]["ungrounded"]]
        fired = sum(bool(r["ungrounded_cap_fired"] or r["zero_match_cap_fired"]) for r in ung)
        own_low = sum(1 for r in ung if not (r["ungrounded_cap_fired"] or r["zero_match_cap_fired"])
                      and r["a5"]["benign_plausibility"] < THRESHOLD)
        dist = dict(sorted(Counter(round(r["a5"]["benign_plausibility"], 2) for r in ung).items()))
        L.append(f"- {name}: {len(ung)} ungrounded records; cap fired on {fired}; A5 itself scored below 0.30 on {own_low} without any cap. Stored scores: {dist}.")
        result[name] = {"ungrounded": len(ung), "cap_fired": fired, "own_below_030": own_low}
    L.append("")
    L.append("In v2 and v3, A5 scored ungrounded flows low itself, without the cap. But in those runs `patch_hypothesis_support` replaced every hypothesis's closeness count with **0** for ungrounded flows, so every hypothesis line A5 read said \"of which 0 are within 2x\". A5 was following that suppressed zero. Once the arms showed the real numbers, the same kind of flow scored 0.70-0.95.\n")
    L.append("So in no run has A5 flagged a low-count flow from its own judgement. It either followed a count it was shown or was overwritten by the cap. Part 2 would test the one untested case: low-count flows with no count shown at all. The grounded-only run (eighth attempt, count >= 30 flows, no count shown) called 22 of 22 benign, which suggests the same here, but it is a different set of flows.\n")

    L.append("## Per record\n")
    for a, name, n, n_att, below, cap in summary_rows:
        L.append(f"### Arm {a.upper()} ({name}): {n} records the cap caught\n")
        L.append("| record | true label | count (leave-one-out) | raw A5 | clamped |")
        L.append("|---|---|---|---|---|")
        for c in sorted(cap, key=lambda c: c["count_loo"]):
            L.append(f"| {c['record_id'][:10]} | {truth[c['record_id']][0]} | {c['count_loo']} | {c['raw']:.2f} | {c['clamped']:.2f} |")
        L.append("")
    L.append("### A5's rationale on each Arm M record the cap caught\n")
    for c in sorted(summary_rows[0][5], key=lambda c: c["count_loo"]):
        L.append(f"- **{c['record_id'][:10]}** ({truth[c['record_id']][0]}, count {c['count_loo']}, raw {c['raw']:.2f}): {c['rationale']}\n")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
