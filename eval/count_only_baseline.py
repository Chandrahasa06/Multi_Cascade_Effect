"""Count-only baseline against the agent pipeline, on the same 45 records.

No agents and no API calls. The baseline reads only each flow's features, computes
the benign neighbourhood count with the same agents/escalation_grounding.py path
the pipeline uses, and flags the flow if the count is below the ungrounded cutoff.
Labels are read only afterwards, for scoring.

The agent side is re-scored from recorded outputs (results/agent_run_45_v2.jsonl),
with the corrected cap (0.29) re-applied in code to A5's recorded benign_plausibility.
That is what the corrected pipeline would produce on those outputs. No model calls.

Writes results/count_baseline.md and results/count_baseline.json.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

from agents.escalation_grounding import (
    MIN_NEIGHBOURHOOD_SIZE,
    FeatureNeighbourhood,
    apply_ungrounded_neighbourhood_cap,
    compute_feature_neighbourhood,
    observed_escalation_values,
)
from agents.grounding import apply_empirical_plausibility_cap
from agents.schema import A5Response, VerdictLabel, derive_verdict
from eval.run_blind_pipeline_45_v2 import load_45

AGENT_RUN = Path("results/agent_run_45_v2.jsonl")
KEYS = [Path("results/agent_key_20.csv"), Path("results/agent_key_20b.csv"), Path("results/agent_key_10c.csv")]
OUT_MD = Path("results/count_baseline.md")
OUT_JSON = Path("results/count_baseline.json")
SWEEP = [0, 1, 5, 10, 30, 100, 500]
CUTOFF = MIN_NEIGHBOURHOOD_SIZE  # 30: the cutoff the pipeline already uses


# --------------------------------------------------------------- baseline
def count_only_neighbourhood(features: Dict[str, float]) -> FeatureNeighbourhood:
    """The baseline's only input is the flow's features. No label, no key."""
    return compute_feature_neighbourhood(observed_escalation_values(features))


def count_only_flag(count: int, cutoff: int = CUTOFF) -> bool:
    """True = predicted attack. Same rule the pipeline's ungrounded flag uses."""
    return count < cutoff


# --------------------------------------------------------------- agents, re-scored
def corrected_agent_bp(rec: dict, cap_zero_support: float, cap_ungrounded: float) -> Tuple[float, bool, bool]:
    """Re-applies the two caps in code to the recorded A5 score. Returns
    (corrected bp, zero-support clamp fired, ungrounded clamp fired)."""
    a5 = A5Response(**rec["a5"])
    supports = {
        h: SimpleNamespace(matching_profile_count=s["matching_profile_count"])
        for h, s in rec["hypothesis_support"].items()
    }
    a5, zero_fired = apply_empirical_plausibility_cap(a5, supports, cap=cap_zero_support)
    nb = FeatureNeighbourhood(**rec["escalation_neighbourhood"])
    a5, ung_fired = apply_ungrounded_neighbourhood_cap(a5, nb, cap=cap_ungrounded)
    return float(a5.benign_plausibility), zero_fired, ung_fired


def agent_caught(bp: float) -> bool:
    return derive_verdict(bp) is VerdictLabel.ANOMALOUS_AND_UNEXPLAINED


# --------------------------------------------------------------- scoring
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
    pct = f"{100 * k / n:.1f}%" if n else "n/a"
    return f"{k}/{n} = {pct} (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def load_keys() -> Dict[str, dict]:
    out = {}
    for p in KEYS:
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["record_id"]] = {"true_label": row["true_label"], "is_attack": row["is_attack"] == "True"}
    return out


def confusion(pred: Dict[str, bool], truth: Dict[str, bool]) -> dict:
    tp = sum(1 for i in truth if truth[i] and pred[i])
    fn = sum(1 for i in truth if truth[i] and not pred[i])
    fp = sum(1 for i in truth if not truth[i] and pred[i])
    tn = sum(1 for i in truth if not truth[i] and not pred[i])
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


# --------------------------------------------------------------- main
def main() -> int:
    records = load_45()
    keys = load_keys()
    agent_rows = {}
    with open(AGENT_RUN, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                agent_rows[r["record_id"]] = r
    ids = sorted(b["record_id"] for b in records)
    # the key files list all 50 selected records; the 5 that failed in the original
    # run have no output, so only the 45 scored records are scored here
    assert len(ids) == 45 and set(ids) == set(agent_rows) and set(ids) <= set(keys)
    truth = {i: keys[i]["is_attack"] for i in ids}

    # baseline: features in, count out, flag. Labels not touched here.
    base_count = {}
    for b in records:
        base_count[b["record_id"]] = count_only_neighbourhood(b["features"]).neighbourhood_size
    base_flag = {i: count_only_flag(base_count[i]) for i in ids}

    # sanity: the baseline count must equal the count the pipeline recorded
    recorded_count = {i: agent_rows[i]["escalation_neighbourhood"]["neighbourhood_size"] for i in ids}
    count_mismatch = [i for i in ids if base_count[i] != recorded_count[i]]

    # agents, old cap (0.3 as recorded) and corrected cap (0.29), re-derived from recorded A5 scores
    old_bp = {i: float(agent_rows[i]["a5"]["benign_plausibility"]) for i in ids}
    old_flag = {i: agent_caught(old_bp[i]) for i in ids}
    rescored = {i: corrected_agent_bp(agent_rows[i], 0.29, 0.29) for i in ids}
    # reproduction check: at the old 0.3 caps the re-score must equal the recorded bp
    repro = {i: corrected_agent_bp(agent_rows[i], 0.30, 0.30)[0] for i in ids}
    repro_mismatch = [i for i in ids if abs(repro[i] - old_bp[i]) > 1e-9]
    new_bp = {i: rescored[i][0] for i in ids}
    new_flag = {i: agent_caught(new_bp[i]) for i in ids}
    changed = [i for i in ids if new_bp[i] != old_bp[i]]

    conf = {
        "baseline": confusion(base_flag, truth),
        "agents_old_cap_0.30": confusion(old_flag, truth),
        "agents_corrected_cap_0.29": confusion(new_flag, truth),
    }
    sweep = []
    for cut in SWEEP:
        pred = {i: count_only_flag(base_count[i], cut) for i in ids}
        c = confusion(pred, truth)
        sweep.append({"cutoff": cut, **c})

    disagree = [i for i in ids if base_flag[i] != new_flag[i]]

    OUT_JSON.write_text(json.dumps({
        "n": len(ids), "cutoff": CUTOFF, "confusion": conf, "sweep": sweep,
        "count_mismatch_vs_recorded": count_mismatch, "repro_mismatch_at_0.30": repro_mismatch,
        "records_whose_bp_changes_under_0.29": changed, "disagreements": disagree,
    }, indent=2), encoding="utf-8")
    write_md(ids, records, keys, agent_rows, base_count, base_flag, old_bp, new_bp, old_flag, new_flag,
             rescored, conf, sweep, disagree, changed, count_mismatch, repro_mismatch)
    print(f"wrote {OUT_MD}; disagreements {len(disagree)}; bp changed {len(changed)}; "
          f"count mismatches {len(count_mismatch)}; repro mismatches {len(repro_mismatch)}")
    return 0


def write_md(ids, records, keys, agent_rows, base_count, base_flag, old_bp, new_bp, old_flag, new_flag,
             rescored, conf, sweep, disagree, changed, count_mismatch, repro_mismatch):
    n = len(ids)
    n_att = sum(keys[i]["is_attack"] for i in ids)
    n_ben = n - n_att
    L: List[str] = []
    L.append("# Count-only baseline vs the agent pipeline, on the same 45 records\n")
    L.append("No API calls were made. The agent side is re-scored from recorded outputs in `results/agent_run_45_v2.jsonl`, with the corrected cap (0.29) re-applied to A5's recorded `benign_plausibility`. The baseline reads features only and uses the same `agents/escalation_grounding.py` neighbourhood path the pipeline uses.\n")
    L.append("**What the comparison is.** The agents receive the same grounding code, so this compares 'count plus agents' against 'count alone', not two different detectors. The agents' ungrounded verdict is also a function of the count: the cap forces `benign_plausibility` to at most the cap whenever the neighbourhood is ungrounded. So on ungrounded flows the agents can only agree with the count. The agents can differ only on grounded flows, where the cap does not apply.\n")
    L.append(f"n = {n} ({n_att} attacks, {n_ben} benign). The cutoff is the pipeline's own, {CUTOFF}. It is not tuned.\n")

    L.append("## 1. Side by side\n")
    L.append("| | agents (corrected cap 0.29) | agents (recorded, cap 0.30) | count-only baseline |")
    L.append("|---|---|---|---|")
    ca, co, cb = conf["agents_corrected_cap_0.29"], conf["agents_old_cap_0.30"], conf["baseline"]
    L.append(f"| attacks caught of {n_att} | {rate(ca['tp'], n_att)} | {rate(co['tp'], n_att)} | {rate(cb['tp'], n_att)} |")
    L.append(f"| benign cleared of {n_ben} | {rate(ca['tn'], n_ben)} | {rate(co['tn'], n_ben)} | {rate(cb['tn'], n_ben)} |")
    L.append(f"| benign wrongly flagged | {ca['fp']} of {n_ben} | {co['fp']} of {n_ben} | {cb['fp']} of {n_ben} |")
    L.append(f"| records where agents and baseline disagree | {len(disagree)} of {n} | | |\n")
    L.append("Confusion matrices (counts):\n")
    L.append("| | caught (predicted attack) | not caught (predicted benign) |")
    L.append("|---|---|---|")
    for name, c in [("agents, corrected cap 0.29: attack", ca), ("count-only baseline: attack", cb)]:
        L.append(f"| {name} | TP {c['tp']} | FN {c['fn']} |")
        L.append(f"| {name.replace('attack', 'benign')} | FP {c['fp']} | TN {c['tn']} |")
    L.append("")
    L.append("Reference figures from the earlier reports: 13/30 caught before the fixes, 9/30 after the fixes (both at the 0.30 cap, before the cap change).\n")

    L.append("## 2. Per-class recall (count n)\n")
    L.append("| true label | n | agents (corrected) caught | baseline caught |")
    L.append("|---|---|---|---|")
    labels = sorted({keys[i]["true_label"] for i in ids}, key=lambda x: (x == "BENIGN", x))
    for lab in labels:
        members = [i for i in ids if keys[i]["true_label"] == lab]
        a = sum(new_flag[i] for i in members)
        b = sum(base_flag[i] for i in members)
        L.append(f"| {lab} | {len(members)} | {a} of {len(members)} | {b} of {len(members)} |")
    L.append("")

    L.append("## 3. Disagreements\n")
    if not disagree:
        L.append("The agents and the baseline agree on every one of the 45 records. No disagreement record exists to inspect, so the question of whether the agents add judgment is not answered by this sample.\n")
    else:
        L.append("For each disagreement: neighbourhood count, A5's corrected score, A5's reasoning, and the true label.\n")
        for i in disagree:
            r = agent_rows[i]
            L.append(f"### {i[:10]}  true label: {keys[i]['true_label']}\n")
            L.append(f"- neighbourhood count: {base_count[i]} (cutoff {CUTOFF}); baseline says {'attack' if base_flag[i] else 'benign'}")
            L.append(f"- A5 score: recorded {old_bp[i]:.2f}, corrected {new_bp[i]:.2f}; agents say {'attack' if new_flag[i] else 'benign'}")
            L.append(f"- A5 reasoning: {r['a5']['rationale']}\n")
    L.append("")

    L.append("## 4. Cutoff sweep (baseline)\n")
    L.append("Baseline flags a flow as attack when its count is below the cutoff. Recall and specificity are reported for each cutoff. The cutoff is not tuned to the agents; this table is the whole sweep.\n")
    L.append("| cutoff | attacks caught | benign cleared |")
    L.append("|---|---|---|")
    for s in sweep:
        mark = " (pipeline cutoff)" if s["cutoff"] == CUTOFF else ""
        L.append(f"| {s['cutoff']}{mark} | {rate(s['tp'], n_att)} | {rate(s['tn'], n_ben)} |")
    L.append("")
    L.append(f"Compare to the agents (corrected): {ca['tp']} of {n_att} attacks caught, {ca['fp']} of {n_ben} benign wrongly flagged.\n")
    L.append("**Cutoffs that beat the agents outright on this sample.** Any cutoff at or above 100 catches more attacks than the agents, and none flags a benign record. The sweep was run after the 45-record results were known, so it is post-hoc. The pipeline cutoff stays 30. Nothing here is adopted.\n")
    for cut in SWEEP:
        if cut == CUTOFF:
            continue
        flags = {i: count_only_flag(base_count[i], cut) for i in ids}
        diff = [i for i in ids if flags[i] != new_flag[i]]
        if not diff:
            continue
        L.append(f"### Disagreements at cutoff {cut}: {len(diff)} of {n} records\n")
        for i in diff:
            r = agent_rows[i]
            who = "baseline attack, agents benign" if flags[i] else "agents attack, baseline benign"
            L.append(f"- {i[:10]} ({keys[i]['true_label']}): count {base_count[i]}; A5 corrected {new_bp[i]:.2f}; {who}. A5: {r['a5']['rationale'][:400]}")
        L.append("")

    L.append("## 5. Cap change: what moved\n")
    L.append(f"Records whose benign_plausibility changes under the 0.29 cap: {len(changed)} of {n}.")
    for i in changed:
        L.append(f"- {i[:10]} ({keys[i]['true_label']}): {old_bp[i]:.2f} -> {new_bp[i]:.2f}, neighbourhood {base_count[i]}")
    L.append("")
    L.append("The earlier pipeline had no clamp firing on these records. The 0.30 values were A5's own output, because the prompt tells A5 the cap is 0.3. A clamp only fires when a value is strictly above the cap, so a 0.30 output passed through unchanged. Under 0.29 it is clamped. Applying the corrected cap is therefore a real change to A5's recorded outputs.\n")

    L.append("## 6. Checks\n")
    L.append(f"- Baseline count equals the count the pipeline recorded: {n - len(count_mismatch)} of {n} records (mismatches: {len(count_mismatch)}).")
    L.append(f"- Re-score at the old 0.30 caps reproduces the recorded benign_plausibility exactly: {n - len(repro_mismatch)} of {n} (mismatches: {len(repro_mismatch)}). This shows the re-scoring path is faithful.\n")

    L.append("## 7. Conclusion\n")
    L.append(conclusion(ca, cb, n_att, n_ben, disagree, sweep))
    L.append("\n## Limits\n")
    L.append(f"- n = {n}. One record moves a rate by {100 / n:.1f} points.")
    L.append("- The agents' verdicts on ungrounded flows are fixed by the cap, so agreement on those is by construction, not evidence the agents match the count's judgment. The discriminating test is on grounded flows, and this sample has no grounded attack that the agents caught.")
    L.append("- The baseline is one rule on one feature space. A different baseline could differ.\n")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def conclusion(ca, cb, n_att, n_ben, disagree, sweep) -> str:
    beats = [s for s in sweep if s["cutoff"] != CUTOFF and s["tp"] > ca["tp"] and s["fp"] == 0]
    if not disagree:
        text = ("**At the pipeline's own cutoff (30), the baseline matches the agents.** On these 45 records the count-only rule "
                "gets the same verdict on every record. That is a negative result about the agents at this cutoff: the pipeline "
                "reproduces what the neighbourhood count already says. The design gives the agents nothing to add on ungrounded "
                "flows, and no grounded attack was caught, so the sample cannot show they add judgment on grounded ones.")
        if beats:
            cuts = ", ".join(str(s["cutoff"]) for s in beats)
            text += (" **But the sweep shows the baseline beats the agents at cutoffs "
                     f"{cuts}** (more attacks caught, no benign flagged). That result is post-hoc, from the same 45 records, "
                     "and is not adopted. A cutoff needs its own held-out check before anyone should trust it.")
        return text + " It is a sample of 45, and it is not an independent test."
    if ca["tp"] > cb["tp"] or ca["tn"] > cb["tn"]:
        return ("**Agents beat the baseline on these records.** Read the disagreement list above for what the agents weighed.")
    if ca["tp"] < cb["tp"] or ca["tn"] < cb["tn"]:
        return ("**Baseline beats the agents on these records.** The agents are degrading a usable signal on the disagreement records. Read the list above.")
    return ("**Agents and baseline differ but neither dominates.** Read the disagreement list.")


if __name__ == "__main__":
    raise SystemExit(main())
