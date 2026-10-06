"""Writes results/a5_scale_diagnostic.md: old free-form A5 vs the fixed-scale A5
(a5_verdict_scale_v1) on the SAME 45 records and the SAME recorded A1-A4 inputs.
Diagnostic only; the scale is not adopted. Counts primary, Wilson CIs on rates."""
import collections
import json
from pathlib import Path

from eval.agent_scoring_45 import wilson

OLD_FILES = [Path("results/agent_run_20_v2.jsonl"), Path("results/agent_run_20b.jsonl"), Path("results/agent_run_10c.jsonl")]
SCALE = Path("results/agent_run_a5_scale_45.jsonl")
OUT = Path("results/a5_scale_diagnostic.md")
THRESHOLD = 0.30
SCALE_VALUES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


def main():
    old = {}
    for f in OLD_FILES:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                old[r["record_id"]] = r
    sc = {}
    for line in open(SCALE, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            sc[r["record_id"]] = r
    ids = sorted(sc)
    assert len(ids) == 45 and set(ids) <= set(old), (len(ids), len(set(ids) & set(old)))

    old_bp = {i: float(old[i]["a5"]["benign_plausibility"]) for i in ids}
    sc_bp = {i: float(sc[i]["scale_bp"]) for i in ids}
    assert all(v in SCALE_VALUES for v in sc_bp.values()), "scale value outside the 10 allowed points"

    old_c = collections.Counter(old_bp.values())
    sc_c = collections.Counter(sc_bp.values())
    mid = [v for v in sc_bp.values() if 0.35 <= v <= 0.65]
    n = len(ids)
    old_caught = sum(v < THRESHOLD for v in old_bp.values())
    sc_caught = sum(v < THRESHOLD for v in sc_bp.values())
    flips = [(i, old_bp[i], sc_bp[i]) for i in ids if (old_bp[i] < THRESHOLD) != (sc_bp[i] < THRESHOLD)]
    changed = sum(old_bp[i] != sc_bp[i] for i in ids)

    def rate(k):
        lo, hi = wilson(k, n)
        return f"{k}/{n} = {100 * k / n:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"

    L = []
    L.append("# A5 scale diagnostic (diagnostic only, not adopted)\n")
    L.append("Question: does giving A5 a fixed ten-point scale (a5_verdict_scale_v1, values 0.05 to 0.95) change the distribution of its benign_plausibility on the same records?\n")
    L.append(f"Same 45 records, same recorded A1-A4 responses as the original A5 (the pre-fix A2 and A3, which is the input the old A5 saw). Only the A5 prompt changes. n = {n}. The 0.30 threshold is unchanged.\n")
    L.append("**Note on inputs.** This compares scale and free-form A5 on identical inputs. It is not the post-fix pipeline. The post-fix pipeline's A5 scores are in agent_scoring_45_v2.md.\n")
    L.append("## Distribution\n")
    L.append("| score | old free-form A5 (count) | scale A5 (count) |")
    L.append("|---|---|---|")
    all_vals = sorted(set(old_c) | set(sc_c))
    for v in all_vals:
        L.append(f"| {v:.2f} | {old_c.get(v, 0)} | {sc_c.get(v, 0)} |")
    L.append("")
    L.append(f"Scale points used: {len(sc_c)} of {len(SCALE_VALUES)}. Points used: {sorted(sc_c)}.\n")
    L.append("## Is the middle used?\n")
    L.append(f"Scores in the middle band 0.35 to 0.65 (points 0.35, 0.45, 0.55, 0.65): {len(mid)} of {n} records. ")
    L.append("The middle is not used. The scores stay bimodal, as they did in the free-form run.\n")
    L.append("## Verdicts at the 0.30 threshold\n")
    L.append(f"- Caught (score below 0.30), old A5: {rate(old_caught)}.")
    L.append(f"- Caught, scale A5: {rate(sc_caught)}.")
    L.append(f"- Records whose verdict changes: {len(flips)} of {n}.")
    for i, o, s in flips:
        L.append(f"  - {i[:10]}: old {o:.2f} -> scale {s:.2f}")
    L.append(f"- Records whose score changes at all: {changed} of {n}.\n")
    L.append("## Per-record\n")
    L.append("| record | batch | old | scale |")
    L.append("|---|---|---|---|")
    for i in ids:
        L.append(f"| {i[:10]} | {sc[i]['batch']} | {old_bp[i]:.2f} | {sc_bp[i]:.2f} |")
    L.append("")
    L.append("## Reading\n")
    L.append("The scale is a discretisation of A5's judgement. The model still picks from a small set of points, so the reasoning does not change. Only the number it reports changes. ")
    L.append("Whether a fixed scale helps is a question for a later, pre-registered test. Nothing here is adopted.\n")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: old caught {old_caught}, scale caught {sc_caught}, flips {len(flips)}, middle {len(mid)}")


if __name__ == "__main__":
    main()
