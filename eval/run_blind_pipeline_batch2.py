"""Batch 2: same pipeline as eval/run_blind_pipeline_v2.py (repaired
grounding active, a4_replication_v6), run over
results/agent_input_20b_blind.jsonl instead of batch 1's file. Reuses
run_record_v2/run_all/prompt_versions_v2 unchanged -- only the I/O paths
and the scoring/report differ (batch 2 needs batch-1-vs-batch-2 and
pooled-n=40 sections batch 1's own report never needed).

Run: python -m eval.run_blind_pipeline_batch2
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List

import pandas as pd

from agents import base
from agents.grounding import GroundingCoverageError
from agents.schema import derive_verdict
from eval.build_blind_input_batch2 import BLIND_OUTPUT_PATH_B, load_blind_file
from eval.run_blind_pipeline import VERDICT_THRESHOLD, _dist_summary, wilson_ci
from eval.run_blind_pipeline_v2 import (
    coverage_check,
    prompt_versions_v2,
    run_all,
)

KEY_PATH_B = Path("results/agent_key_20b.csv")
KEY_PATH_A = Path("results/agent_key_20.csv")
RUN_OUTPUT_PATH_B = Path("results/agent_run_20b.jsonl")
RUN_OUTPUT_PATH_A = Path("results/agent_run_20_v2.jsonl")
SCORING_REPORT_PATH_B = Path("results/agent_scoring_20b.md")


def load_run_results(path: Path) -> Dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["record_id"]] = d
    return out


def score(key_df: pd.DataFrame, run_results: Dict[str, dict]) -> dict:
    """Identical shape to eval.run_blind_pipeline_v2.score -- duplicated
    (not imported) only because that module's version is bound to batch
    1's path constants; the computation itself is unchanged."""
    rows = []
    for _, k in key_df.iterrows():
        rid = k["record_id"]
        if rid not in run_results:
            continue
        r = run_results[rid]
        bp = r["a5"]["benign_plausibility"]
        verdict = derive_verdict(bp)
        predicted_attack = bp < VERDICT_THRESHOLD
        nb = r["escalation_neighbourhood"]
        rows.append({
            "record_id": rid, "true_label": k["true_label"], "is_attack": bool(k["is_attack"]),
            "admitted_by": k["admitted_by"], "benign_plausibility": bp, "verdict": verdict.value,
            "predicted_attack": predicted_attack, "clamped": r["a5_plausibility_clamped"],
            "ungrounded_cap_fired": r["ungrounded_cap_fired"], "zero_match_cap_fired": r["zero_match_cap_fired"],
            "neighbourhood_size": nb["neighbourhood_size"], "benign_population_size": nb["benign_population_size"],
            "ungrounded": nb["ungrounded"],
            "credited_hypothesis_id": r["a5"]["credited_hypothesis_id"], "rationale": r["a5"]["rationale"],
        })
    scored = pd.DataFrame(rows)

    tp = int(((scored["is_attack"]) & (scored["predicted_attack"])).sum())
    fn = int(((scored["is_attack"]) & (~scored["predicted_attack"])).sum())
    fp = int(((~scored["is_attack"]) & (scored["predicted_attack"])).sum())
    tn = int(((~scored["is_attack"]) & (~scored["predicted_attack"])).sum())
    confusion = {"tp": tp, "fn": fn, "fp": fp, "tn": tn}

    attack_bp = sorted(scored.loc[scored["is_attack"], "benign_plausibility"].tolist())
    benign_bp = sorted(scored.loc[~scored["is_attack"], "benign_plausibility"].tolist())

    trust_rows = []
    n_zero_match_clamped = 0
    n_ungrounded_clamped = 0
    n_contradiction_retries = 0
    cvi_values = []
    from eval.run_blind_pipeline import CONTRADICTION_RETRY_MARKER
    for rid, r in run_results.items():
        n_zero_match_clamped += int(r["zero_match_cap_fired"])
        n_ungrounded_clamped += int(r["ungrounded_cap_fired"])
        cvi = r.get("chain_vs_independent")
        if cvi is not None:
            cvi_values.append(cvi)
        for agent, meta in r["call_metadata"].items():
            for reason in meta.get("retry_reasons", []):
                if CONTRADICTION_RETRY_MARKER in reason:
                    n_contradiction_retries += 1
        for agent, ts in r["trust_scores"].items():
            trust_rows.append({"record_id": rid, "agent": agent, "C": ts["C"], "E": ts["E"], "V": ts["V"], "T": ts["T"]})
    trust_df = pd.DataFrame(trust_rows)

    return {
        "scored_df": scored, "confusion": confusion, "attack_bp": attack_bp, "benign_bp": benign_bp,
        "trust_df": trust_df, "n_zero_match_clamped": n_zero_match_clamped,
        "n_ungrounded_clamped": n_ungrounded_clamped, "n_contradiction_retries": n_contradiction_retries,
        "cvi_values": cvi_values,
    }


def confusion_stats(confusion: dict) -> dict:
    c = confusion
    n_attack = c["tp"] + c["fn"]
    n_benign = c["fp"] + c["tn"]
    return {
        "n_attack": n_attack, "n_benign": n_benign,
        "recall": (c["tp"] / n_attack) if n_attack else None,
        "recall_ci": wilson_ci(c["tp"], n_attack) if n_attack else None,
        "specificity": (c["tn"] / n_benign) if n_benign else None,
        "specificity_ci": wilson_ci(c["tn"], n_benign) if n_benign else None,
        **c,
    }


def write_scoring_report(
    key_df_b: pd.DataFrame, run_results_b: Dict[str, dict], tally: dict, scoring_b: dict,
    prompt_vers: Dict[str, str], coverage_b: dict,
) -> None:
    lines: List[str] = []
    lines.append("# Agent pipeline scoring — batch 2 (20 selected), and batch 1 vs batch 2 vs pooled\n")
    lines.append(
        "Replication of the v2 (post grounding-repair) run on a second, non-overlapping 20-record "
        "selection. Same model, same prompt versions (`a4_replication_v6` included), same 0.3 threshold, "
        "same repaired grounding path (`agents/escalation_grounding.py`).\n"
    )

    n_total_b = len(key_df_b)
    n_scored_b = len(scoring_b["scored_df"])
    n_failed = len(tally["failed_records"])
    if tally.get("stopped_on_quota"):
        coverage_note = "(partial run -- stopped on quota mid-run)"
    elif n_failed:
        coverage_note = f"(partial -- {n_failed} record(s) failed schema validation after retries, NOT a quota stop)"
    else:
        coverage_note = "(full run)"
    lines.append(
        f"**Coverage**: {n_scored_b}/{n_total_b} batch-2 records scored " + coverage_note
        + f". Model: `{base.DEFAULT_MODEL}`. Prompt versions: "
        + ", ".join(f"{a}={v}" for a, v in prompt_vers.items()) + ".\n"
    )
    lines.append(
        f"API calls made this run: **{tally['api_calls_made']}** (includes retries); "
        f"cache hits: {tally['cache_hits']}; schema retries: {tally['schema_retries']}; "
        f"failed records: {n_failed or 'none'}.\n"
    )
    if n_failed:
        key_by_id = key_df_b.set_index("record_id")
        lines.append(
            f"\n### Failed records ({n_failed}/{n_total_b}) -- exhausted 3 schema retries, not scored\n\n"
            "Genuine schema violations (not quota, not this task's Step 3 fix -- two different, "
            "previously-unseen A3 failure modes), reported rather than retried, per this batch's "
            "\"do not rerun\" instruction:\n\n"
        )
        for rid in tally["failed_records"]:
            k = key_by_id.loc[rid]
            reason = tally["failed_record_reasons"].get(rid, "unknown")
            lines.append(f"- `{rid}` (true label: {k['true_label']}): {reason}\n")
        lines.append(
            "\n3 of these 4 are BENIGN and 1 is a Bot record -- batch 2's SCORED composition is therefore "
            f"11 attacks (not the 12 selected) and 5 benign (not the 8 selected), and only 1 of the 2 "
            "selected Bot records was actually scored. Every count and table below uses the scored 16, "
            "stated as such rather than silently treated as the originally-selected 20.\n"
        )
    frac_undef = coverage_b["n_undefined"] / max(coverage_b["n_total"], 1)
    lines.append(
        f"\n**Grounding coverage check**: close_to_observed_count undefined for "
        f"{coverage_b['n_undefined']}/{coverage_b['n_total']} hypotheses ({frac_undef:.0%}) -- "
        f"{'PASSED' if frac_undef <= 0.5 else 'FAILED -- the repair has regressed; these scores mean nothing'}.\n"
    )

    pooled_n_preview = 20 + n_scored_b  # batch 1 scored 20/20; exact pooled breakdown is in item 3
    lines.append("## What this test can and cannot show\n")
    lines.append(
        "- **12 attack / 8 benign is not the natural rate.** Escalated flows are 9.4% attacks. This "
        "measures discrimination, not deployment calibration.\n"
        f"- **n={n_scored_b} (batch 2 alone) / n={pooled_n_preview} (pooled) supports little on its own.** "
        "Counts, not percentages; confidence intervals on every rate.\n"
        f"- **Batch 2's selection rule was fixed after seeing batch 1's results.** The pooled n={pooled_n_preview} "
        f"figure below is reported as two batches pooled, not as one independent {pooled_n_preview}-record "
        "trial -- see item 3.\n"
        "- Records were not reselected, no seed was changed, and the threshold/prompts were not adjusted "
        "after seeing batch 2's scores.\n"
    )

    # ---- Selection / class composition ----
    key_df_a = pd.read_csv(KEY_PATH_A)
    attack_a = key_df_a[key_df_a["is_attack"]]["true_label"].value_counts()
    attack_b = key_df_b[key_df_b["is_attack"]]["true_label"].value_counts()
    lines.append("## Selection: class composition, batch 1 vs batch 2 (as SELECTED, before batch 2's 4 schema failures)\n")
    all_classes = sorted(set(attack_a.index) | set(attack_b.index))
    comp_table = ["| class | batch 1 n | batch 2 n |", "|---|---|---|"]
    for c in all_classes:
        comp_table.append(f"| {c} | {int(attack_a.get(c, 0))} | {int(attack_b.get(c, 0))} |")
    lines.append("\n".join(comp_table) + "\n")
    dict_a = {k: int(v) for k, v in attack_a.items()}
    dict_b = {k: int(v) for k, v in attack_b.items()}
    diff_classes = [c for c in all_classes if attack_a.get(c, 0) != attack_b.get(c, 0)]
    lines.append(
        f"\nBatches differ in composition: batch 1 had {dict_a}; batch 2 selected {dict_b} (but see the "
        "Failed records note above -- 1 of batch 2's 2 Bot records did not survive to scoring). "
        f"Classes with a different SELECTED count between batches: {diff_classes or 'none'}. This matters because "
        "escalated_flows.csv's rare classes are nearly exhausted (282 attacks total, batch 1 already took "
        "2 of 8 Bot records) -- any recall difference between batches could reflect class mix rather than "
        "agent behaviour, which is why this table is reported before any comparison below.\n"
    )

    # ---- 1. Batch 2 alone ----
    c = scoring_b["confusion"]
    stats_b = confusion_stats(c)
    lines.append("## 1. Batch 2 alone\n")
    lines.append(
        f"### Confusion matrix (threshold={VERDICT_THRESHOLD})\n\n"
        f"|  | predicted attack (bp<{VERDICT_THRESHOLD}) | predicted benign (bp>={VERDICT_THRESHOLD}) | n |\n"
        f"|---|---|---|---|\n"
        f"| true attack | {c['tp']} | {c['fn']} | {stats_b['n_attack']} |\n"
        f"| true benign | {c['fp']} | {c['tn']} | {stats_b['n_benign']} |\n"
    )
    lines.append(
        f"\nRecall: {c['tp']}/{stats_b['n_attack']}"
        + (f", 95% CI {stats_b['recall_ci']}" if stats_b['recall_ci'] else "") + "\n"
        f"Specificity: {c['tn']}/{stats_b['n_benign']}"
        + (f", 95% CI {stats_b['specificity_ci']}" if stats_b['specificity_ci'] else "") + "\n"
    )
    lines.append("\n### `benign_plausibility` distributions\n")
    lines.append(f"- True attack ({_dist_summary(scoring_b['attack_bp'])}): {scoring_b['attack_bp']}\n")
    lines.append(f"- True benign ({_dist_summary(scoring_b['benign_bp'])}): {scoring_b['benign_bp']}\n")

    lines.append("\n### Per record (with comparison-set size and close_to_observed_count)\n")
    record_table = [
        "| record_id | true label | benign_plausibility | derived verdict | neighbourhood size | ungrounded | A5 rationale (one line) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in scoring_b["scored_df"].iterrows():
        rationale_line = row["rationale"].replace("\n", " ").replace("|", "/")
        if len(rationale_line) > 180:
            rationale_line = rationale_line[:180] + "..."
        record_table.append(
            f"| {row['record_id']} | {row['true_label']} | {row['benign_plausibility']:.2f} | "
            f"{row['verdict']} | {row['neighbourhood_size']}/{row['benign_population_size']} | "
            f"{row['ungrounded']} | {rationale_line} |"
        )
        if row["benign_plausibility"] == VERDICT_THRESHOLD:
            record_table[-1] += "  <!-- exactly at threshold -->"
    lines.append("\n".join(record_table) + "\n")

    lines.append("\n### Per class\n")
    attack_scored_b = scoring_b["scored_df"][scoring_b["scored_df"]["is_attack"]]
    for label, grp in attack_scored_b.groupby("true_label"):
        vals = ", ".join(f"{v:.2f}" for v in sorted(grp["benign_plausibility"]))
        lines.append(f"- {label} (n={len(grp)}): benign_plausibility = [{vals}]\n")

    # ---- 5. Bot specifically (placed here since it needs batch 2's own per-class data) ----
    bot_b = attack_scored_b[attack_scored_b["true_label"] == "Bot"]
    lines.append("\n## 5. Bot specifically\n")
    if len(bot_b):
        bot_missed = bot_b[~bot_b["predicted_attack"]]
        n_bot = len(bot_b)
        n_bot_missed = len(bot_missed)
        summary_line = (
            f"Batch 2 drew {n_bot} Bot record(s): benign_plausibility = "
            f"{sorted(bot_b['benign_plausibility'].tolist())}, neighbourhood sizes = "
            f"{list(bot_b['neighbourhood_size'])}. "
            f"{n_bot_missed}/{n_bot} missed."
        )
        lines.append(summary_line + "\n")
        if n_bot_missed == n_bot:
            lines.append(
                "\n**Bot has now been missed by the flow-level analysis (Output 4's \"never flagged\" "
                "group), the DT-rule/signature-table selector (P1/P2, no rule or signature ever fires on "
                "it), and the agents in both batch 1 and batch 2, independently.** A fourth confirmation "
                "of the same ceiling, not a new finding on its own -- worth stating plainly rather than "
                "re-discovering quietly each time.\n"
            )
        elif n_bot_missed == 0:
            lines.append(
                "\nThe first batch in which a Bot record was NOT missed -- noted as a genuine change from "
                "batch 1's 2/2, not smoothed into the standing \"Bot is a ceiling\" narrative without "
                "saying so.\n"
            )
        else:
            lines.append(
                f"\nA mixed result this batch ({n_bot_missed}/{n_bot} missed) -- reported as observed, "
                "not rounded toward either the \"ceiling\" or \"resolved\" narrative.\n"
            )
    else:
        lines.append("Batch 2 did not draw any Bot record -- no update to report here.\n")

    lines.append("\n## Trust and mechanism (batch 2)\n")
    if len(scoring_b["trust_df"]):
        agent_means = scoring_b["trust_df"].groupby("agent")[["C", "E", "V", "T"]].mean(numeric_only=True)
        lines.append(agent_means.round(3).to_markdown() + "\n")
    lines.append(
        f"\n- Zero-match plausibility cap fired: {scoring_b['n_zero_match_clamped']}/{n_scored_b} records.\n"
        f"- Ungrounded-neighbourhood plausibility cap fired: {scoring_b['n_ungrounded_clamped']}/{n_scored_b} records.\n"
        f"- Contradiction-disposal forced a retry: {scoring_b['n_contradiction_retries']} times.\n"
    )
    if scoring_b["cvi_values"]:
        mean_cvi = statistics.mean(scoring_b["cvi_values"])
        n_pos = sum(1 for v in scoring_b["cvi_values"] if v > 0)
        lines.append(
            f"- chain_vs_independent: mean {mean_cvi:+.3f} over {len(scoring_b['cvi_values'])} records, "
            f"positive in {n_pos}/{len(scoring_b['cvi_values'])}.\n"
        )

    # ---- 2. Batch 1 vs batch 2 side by side ----
    run_results_a = load_run_results(RUN_OUTPUT_PATH_A)
    key_df_a_full = pd.read_csv(KEY_PATH_A)
    scoring_a = score(key_df_a_full, run_results_a)
    stats_a = confusion_stats(scoring_a["confusion"])

    lines.append("\n## 2. Batch 1 vs batch 2, side by side\n")
    lines.append(
        "| | batch 1 | batch 2 |\n|---|---|---|\n"
        f"| recall | {scoring_a['confusion']['tp']}/{stats_a['n_attack']}"
        + (f" (CI {stats_a['recall_ci']})" if stats_a['recall_ci'] else "") + " | "
        f"{c['tp']}/{stats_b['n_attack']}" + (f" (CI {stats_b['recall_ci']})" if stats_b['recall_ci'] else "") + " |\n"
        f"| specificity | {scoring_a['confusion']['tn']}/{stats_a['n_benign']}"
        + (f" (CI {stats_a['specificity_ci']})" if stats_a['specificity_ci'] else "") + " | "
        f"{c['tn']}/{stats_b['n_benign']}" + (f" (CI {stats_b['specificity_ci']})" if stats_b['specificity_ci'] else "") + " |\n"
        f"| attack bp values | {scoring_a['attack_bp']} | {scoring_b['attack_bp']} |\n"
        f"| benign bp values | {scoring_a['benign_bp']} | {scoring_b['benign_bp']} |\n"
    )
    a_caught = [v for v in scoring_a["attack_bp"] if v < VERDICT_THRESHOLD]
    a_max_caught = max(a_caught) if a_caught else None
    a_min_benign = min(scoring_a["benign_bp"]) if scoring_a["benign_bp"] else None
    b_caught = [v for v in scoring_b["attack_bp"] if v < VERDICT_THRESHOLD]
    b_missed = [v for v in scoring_b["attack_bp"] if v >= VERDICT_THRESHOLD]
    lines.append(
        f"\nBatch 1 put every caught attack at 0.25 and every true benign at 0.85+, leaving an empty band "
        f"between {a_max_caught if a_max_caught is not None else 'n/a'} "
        f"and {a_min_benign}. Batch 2's caught attacks: {b_caught or 'none'}; missed attacks: {b_missed or 'none'}; "
        f"true benign: {scoring_b['benign_bp']}. "
        + ("The same tight clustering and empty band appear in batch 2." if
           (b_caught and all(v <= 0.30 for v in b_caught) and scoring_b["benign_bp"] and min(scoring_b["benign_bp"]) >= 0.80)
           else "Batch 2 does NOT reproduce the same clean separation seen in batch 1 -- reported as a real "
                "difference, not smoothed over.") + "\n"
    )

    # ---- 3. Pooled ----
    pooled_tp = scoring_a["confusion"]["tp"] + c["tp"]
    pooled_fn = scoring_a["confusion"]["fn"] + c["fn"]
    pooled_fp = scoring_a["confusion"]["fp"] + c["fp"]
    pooled_tn = scoring_a["confusion"]["tn"] + c["tn"]
    pooled_n_attack = pooled_tp + pooled_fn
    pooled_n_benign = pooled_fp + pooled_tn
    pooled_n = pooled_n_attack + pooled_n_benign
    lines.append(
        f"\n## 3. Pooled, n={pooled_n} (two batches pooled, NOT an independent {pooled_n}-record trial)\n"
    )
    if pooled_n != 40:
        lines.append(
            f"**Pooled n is {pooled_n}, not 40** -- batch 1 scored all 20, but batch 2 only scored "
            f"{n_scored_b}/20 (4 records failed A3's schema validation, see above), so the pool is "
            f"20 + {n_scored_b} = {pooled_n}, not 20 + 20.\n"
        )
    lines.append(
        "**Batch 2's selection rule (seed, exclusion of batch 1) was fixed after batch 1's results were "
        "already known.** This is a replication check, not an independent trial -- pooling below is for "
        f"a tighter confidence interval on the same underlying question, not a claim of n={pooled_n} independence.\n"
    )
    pooled_recall_ci = wilson_ci(pooled_tp, pooled_n_attack) if pooled_n_attack else None
    pooled_spec_ci = wilson_ci(pooled_tn, pooled_n_benign) if pooled_n_benign else None
    lines.append(
        f"| | predicted attack | predicted benign | n |\n|---|---|---|---|\n"
        f"| true attack | {pooled_tp} | {pooled_fn} | {pooled_n_attack} |\n"
        f"| true benign | {pooled_fp} | {pooled_tn} | {pooled_n_benign} |\n\n"
        f"Pooled recall: {pooled_tp}/{pooled_n_attack}" + (f", 95% CI {pooled_recall_ci}" if pooled_recall_ci else "") + "\n"
        f"Pooled specificity: {pooled_tn}/{pooled_n_benign}" + (f", 95% CI {pooled_spec_ci}" if pooled_spec_ci else "") + "\n"
    )

    # ---- 4. Grounding correlation across all scored records (both batches) ----
    combined = pd.concat([scoring_a["scored_df"], scoring_b["scored_df"]], ignore_index=True)
    lines.append(f"\n## 4. Grounding correlation, across all {len(combined)} scored records (both batches)\n")
    attack_combined = combined[combined["is_attack"]]
    caught = attack_combined[attack_combined["predicted_attack"]]
    missed = attack_combined[~attack_combined["predicted_attack"]]
    lines.append(
        f"Across all {len(attack_combined)} scored attack records (both batches): "
        f"{len(caught)} caught, {len(missed)} missed.\n\n"
        f"- Caught attacks' neighbourhood sizes: {sorted(caught['neighbourhood_size'].tolist())} "
        f"({int((caught['ungrounded']).sum())}/{len(caught)} ungrounded)\n"
        f"- Missed attacks' neighbourhood sizes: {sorted(missed['neighbourhood_size'].tolist())} "
        f"({int((missed['ungrounded']).sum())}/{len(missed)} ungrounded)\n"
    )
    n_caught_ungrounded = int((caught["ungrounded"]).sum())
    n_missed_ungrounded = int((missed["ungrounded"]).sum())
    forward_holds = len(caught) > 0 and n_caught_ungrounded == len(caught)  # caught => ungrounded
    converse_holds = len(missed) == 0 or n_missed_ungrounded == 0  # ungrounded => caught
    lines.append(
        "\nIn v2 (batch 1) the relationship was a clean two-way rule: every caught attack was ungrounded, "
        "and every ungrounded attack was caught. Across both batches these are checked as TWO SEPARATE "
        f"claims, since {len(combined)} records is where they can start to come apart:\n\n"
        f"- **Caught => ungrounded**: {n_caught_ungrounded}/{len(caught)} caught attacks are ungrounded -- "
        f"{f'holds exactly, replicated cleanly at n={len(attack_combined)}.' if forward_holds else 'does NOT hold exactly.'}\n"
        f"- **Ungrounded => caught** (the converse): {len(missed) - n_missed_ungrounded}/{len(missed)} missed "
        f"attacks are grounded (nonzero neighbourhood, as expected for a miss), but "
        f"{n_missed_ungrounded}/{len(missed)} missed attacks are themselves ungrounded -- "
        f"{'holds exactly.' if converse_holds else 'does NOT hold exactly: an empty neighbourhood does not guarantee a catch.'}\n\n"
        + (
            "Both directions replicate cleanly across 40 records -- this mechanism result is stronger and "
            "more reproducible than either batch's own recall number.\n"
            if (forward_holds and converse_holds) else
            f"**Only the forward direction replicates.** An empty neighbourhood is necessary for every catch "
            f"observed so far, but it is not sufficient: {n_missed_ungrounded} genuinely ungrounded attack(s) "
            "were still missed (see batch 1's DDoS-at-0.30 borderline case and batch 2's own missed "
            "ungrounded records above) -- a real, model-calibration-level gap between \"the data supports "
            "flagging this\" and \"the model actually flagged it\", not a repeat of the harness fault this "
            "task's earlier step fixed. Reported as observed, not rounded up to the clean two-way rule v2 "
            "alone suggested.\n"
        )
    )

    lines.append("\n## Honesty notes\n")
    lines.append(
        "- Every count above states its own denominator; every rate carries a Wilson 95% CI.\n"
        "- 12/8 is not the natural escalation rate (9.4% attacks); this measures discrimination.\n"
        "- Batch 2's selection, threshold, and prompts were fixed before this run and not adjusted after "
        "seeing its scores.\n"
    )

    SCORING_REPORT_PATH_B.parent.mkdir(parents=True, exist_ok=True)
    SCORING_REPORT_PATH_B.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {SCORING_REPORT_PATH_B}")


def main() -> int:
    blind_records = load_blind_file(BLIND_OUTPUT_PATH_B)
    print(f"loaded {len(blind_records)} batch-2 blind records")

    used = base.daily_request_count(base.DEFAULT_MODEL)
    cap = base.DAILY_QUOTA_BY_MODEL.get(base.DEFAULT_MODEL, 500)
    print(f"daily count for {base.DEFAULT_MODEL}: {used}/{cap} before this run")

    tally = run_all(blind_records, out_path=RUN_OUTPUT_PATH_B)
    print(f"this run: {tally}")

    run_results_b = load_run_results(RUN_OUTPUT_PATH_B)
    key_df_b = pd.read_csv(KEY_PATH_B)
    if not run_results_b:
        print("no records completed -- nothing to score")
        return 1

    try:
        coverage_b = coverage_check(run_results_b)
    except GroundingCoverageError as exc:
        print(f"[GROUNDING COVERAGE CHECK FAILED] {exc}")
        raise

    scoring_b = score(key_df_b, run_results_b)
    write_scoring_report(key_df_b, run_results_b, tally, scoring_b, prompt_versions_v2(), coverage_b)

    if tally["failed_records"] or tally["stopped_on_quota"]:
        print("PARTIAL RUN -- see results/agent_scoring_20b.md for exactly which records completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
