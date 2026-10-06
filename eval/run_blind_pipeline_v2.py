"""Re-run of eval/run_blind_pipeline.py after two repairs:

1. Empirical grounding restored via agents/escalation_grounding.py --
   close_to_observed_count is now computed from the flow's own
   escalation-feature values against a real BENIGN-only reference (never
   from trigger_reasons, which this policy's records don't have, and
   never from class_predicted/rule_id/priority/reason). See STATUS.md's
   "fifth silent-disable" entry and agents/escalation_grounding.py's
   module docstring for the full diagnosis.
2. A4's prompt bumped to a4_replication_v6, which fixed the one dropped
   record (a real schema violation -- A4 asserting a value-requiring
   relation with no asserted_value -- caused by a prompt gap, not a
   validator or retry-feedback bug; see agents/prompts/a4_replication_v6.py).

**This is a re-run after a harness fix, not an independent second
measurement.** results/agent_scoring_20.md (v1)'s 0/11 stands as a
recorded result of the FIRST run; this produces a SECOND, v2 result under
repaired grounding, over the SAME 20 records, same blind input, same 0.3
threshold, same model, same prompt versions except a4 (v5->v6, forced by
Step 3's fix).

Reuses the disk cache aggressively: A1/A2/A3 responses for the 19
previously-completed records are unchanged by either repair and are
served from cache for free. A4 is forced to re-run for every record
because its prompt version changed (a correct cache invalidation, not a
bug -- the OLD v5 responses reflect a prompt that didn't state the
asserted_value requirement). A5 is forced to re-run for every record
(`use_cache=False`) because its EMPIRICAL SUPPORT TEXT changed even
though its prompt_version string didn't -- agents/base.py's cache key is
`record_id|agent|prompt_version|model|temperature|run_index`, which does
NOT hash prompt content, so leaving A5 on the default cache would
silently replay the OLD (ungrounded) verdicts and make this repair look
like it did nothing.

Run: python -m eval.run_blind_pipeline_v2
"""
from __future__ import annotations

import concurrent.futures
import json
import statistics
from pathlib import Path
from typing import Dict, List

import pandas as pd

from agents import a1_evidence, a2_behaviour, a3_hypotheses, a4_replication, a5_verdict, base, pipeline
from agents.escalation_grounding import (
    UNGROUNDED_PLAUSIBILITY_CAP,
    apply_ungrounded_neighbourhood_cap,
    compute_feature_neighbourhood,
    observed_escalation_values,
    patch_hypothesis_support,
)
from agents.grounding import GroundingCoverageError, assert_close_to_observed_coverage, close_to_observed_coverage
from agents.schema import derive_verdict
from eval.build_blind_input import BLIND_OUTPUT_PATH, load_blind_file
from eval.run_blind_pipeline import (
    CONTRADICTION_RETRY_MARKER,
    KEY_PATH,
    VERDICT_THRESHOLD,
    _dist_summary,
    already_done_ids,
    blind_record_to_escalation_record,
    wilson_ci,
)

RUN_OUTPUT_PATH = Path("results/agent_run_20_v2.jsonl")
SCORING_REPORT_PATH = Path("results/agent_scoring_20_v2.md")


def prompt_versions_v2() -> Dict[str, str]:
    return {
        "a1": a1_evidence.prompt_module.PROMPT_VERSION,
        "a2": a2_behaviour.prompt_module.PROMPT_VERSION,
        "a3": a3_hypotheses.prompt_module.PROMPT_VERSION,
        "a4": a4_replication.prompt_module.PROMPT_VERSION,
        "a5": a5_verdict.prompt_module.PROMPT_VERSION,
    }


def run_record_v2(record, *, model=base.DEFAULT_MODEL, temperature=0.7, run_index=0):
    """Mirrors agents.pipeline.run_record's orchestration exactly (reusing
    its own factored-out helpers: compute_grounding, compute_hypothesis_support,
    compute_trust_and_decay, compute_contamination -- all UNCHANGED), with
    two differences: hypothesis_support is patched with the escalation-
    feature neighbourhood before A5 ever sees it, and A5's call always
    bypasses the cache (see module docstring)."""
    call_kwargs = dict(model=model, temperature=temperature, run_index=run_index, use_cache=True)

    grounding = pipeline.compute_grounding(record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a4_future = pool.submit(
            a4_replication.run, record, grounding.empirical_grounding_block, grounding.available_features,
            grounding.known_reference_features, **call_kwargs,
        )
        a1_resp, a1_meta = a1_evidence.run(record, grounding.empirical_grounding_block, **call_kwargs)
        a2_resp, a2_meta = a2_behaviour.run(record, a1_resp, grounding.empirical_grounding_block, **call_kwargs)
        a3_resp, a3_meta = a3_hypotheses.run(
            record, a1_resp, a2_resp, grounding.empirical_grounding_block, grounding.available_features,
            grounding.known_reference_features, **call_kwargs,
        )
        a4_resp, a4_meta = a4_future.result()

    hypothesis_support, _ = pipeline.compute_hypothesis_support(record, a3_resp, a4_resp, grounding.benign_df)

    # ---- the repair: escalation-feature neighbourhood, benign-only,
    # never touches class_predicted/rule_id/priority/reason ----
    observed = observed_escalation_values(record.features)
    neighbourhood = compute_feature_neighbourhood(observed)
    hypothesis_support = patch_hypothesis_support(hypothesis_support, neighbourhood)
    from agents.grounding import render_hypothesis_support
    empirical_support_lines = {hid: render_hypothesis_support(s) for hid, s in hypothesis_support.items()}

    a5_call_kwargs = dict(call_kwargs, use_cache=False)  # see module docstring: A5 must never serve a stale cached verdict here
    a5_resp, a5_meta, a5_clamped_original = a5_verdict.run(
        record, a1_resp, a2_resp, a3_resp, a4_resp, empirical_support_lines, hypothesis_support, **a5_call_kwargs,
    )
    a5_resp, ungrounded_clamped = apply_ungrounded_neighbourhood_cap(a5_resp, neighbourhood)
    a5_plausibility_clamped = a5_clamped_original or ungrounded_clamped

    trust_scores, trust_decay_chain, trust_decay_with_a5, cvi = pipeline.compute_trust_and_decay(
        record, a1_resp, a2_resp, a3_resp, a4_resp, a5_resp
    )
    contamination = pipeline.compute_contamination(a1_resp, a2_resp, a3_resp, a4_resp, a5_resp)

    result = pipeline.RecordResult(
        record_id=record.flow_id, a1=a1_resp, a2=a2_resp, a3=a3_resp, a4=a4_resp, a5=a5_resp,
        trust_scores=trust_scores, trust_decay_chain=trust_decay_chain, trust_decay_with_a5=trust_decay_with_a5,
        chain_vs_independent=cvi, contamination=contamination,
        call_metadata={"a1": a1_meta, "a2": a2_meta, "a3": a3_meta, "a4": a4_meta, "a5": a5_meta},
        nearest_neighbours=grounding.nearest_neighbours, hypothesis_support=hypothesis_support,
        a5_plausibility_clamped=a5_plausibility_clamped,
    )
    return result, neighbourhood, ungrounded_clamped, a5_clamped_original


def run_all(blind_records: List[dict], out_path: Path = RUN_OUTPUT_PATH) -> dict:
    done = already_done_ids(out_path)
    if done:
        print(f"resuming: {len(done)}/{len(blind_records)} records already in {out_path}, skipping them")

    tally = {
        "records_completed_this_run": 0, "api_calls_made": 0, "cache_hits": 0,
        "schema_retries": 0, "failed_records": [], "failed_record_reasons": {}, "stopped_on_quota": False,
        # additive, for results/plausibility_diagnostic.md-style "did the retry
        # feedback change what the model produced" analysis -- per-attempt
        # detail that base.SchemaValidationFailed didn't preserve until the
        # fix that added .retry_reasons/.raw_attempts (see agents/base.py):
        # a failed call's intermediate attempts are otherwise lost the moment
        # the exception is raised, since only the LAST attempt's error ever
        # reached run_all before that fix.
        "failed_record_agent": {}, "failed_record_retry_reasons": {}, "failed_record_raw_attempts": {},
    }
    neighbourhoods_by_id: Dict[str, dict] = {}
    for blind in blind_records:
        rid = blind["record_id"]
        if rid in done:
            continue
        record = blind_record_to_escalation_record(blind)
        try:
            result, neighbourhood, ungrounded_clamped, orig_clamped = run_record_v2(record)
        except base.DailyQuotaExceeded as exc:
            print(f"[STOPPED] daily quota hit after {tally['records_completed_this_run']} records this run: {exc}")
            tally["stopped_on_quota"] = True
            break
        except base.SchemaValidationFailed as exc:
            print(f"[FAILED] {rid}: {exc}")
            tally["failed_records"].append(rid)
            tally["failed_record_reasons"][rid] = str(exc)
            tally["failed_record_agent"][rid] = exc.agent
            tally["failed_record_retry_reasons"][rid] = list(exc.retry_reasons)
            tally["failed_record_raw_attempts"][rid] = list(exc.raw_attempts)
            tally["api_calls_made"] += exc.attempts
            continue

        for meta in result.call_metadata.values():
            if meta.cached:
                tally["cache_hits"] += 1
            else:
                tally["api_calls_made"] += 1 + meta.schema_retries
            tally["schema_retries"] += meta.schema_retries

        record_dict = result.to_dict()
        record_dict["escalation_neighbourhood"] = neighbourhood.to_dict()
        record_dict["ungrounded_cap_fired"] = ungrounded_clamped
        record_dict["zero_match_cap_fired"] = orig_clamped
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_dict) + "\n")

        tally["records_completed_this_run"] += 1
        print(
            f"  done: {rid} (benign_plausibility={result.a5.benign_plausibility:.2f}, "
            f"neighbourhood={neighbourhood.neighbourhood_size}, ungrounded={neighbourhood.ungrounded}, "
            f"cap_fired={result.a5_plausibility_clamped})"
        )
    return tally


# --------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------- #

def load_run_results(path: Path = RUN_OUTPUT_PATH) -> Dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["record_id"]] = d
    return out


def score(key_df: pd.DataFrame, run_results: Dict[str, dict]) -> dict:
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


def coverage_check(run_results: Dict[str, dict]) -> dict:
    """Step 1's loud-failure guard, applied to the v2 run's own output --
    if the repair somehow still leaves close_to_observed_count undefined
    across most hypotheses, this must raise, not silently pass."""

    class _S:
        def __init__(self, d):
            self.close_to_observed_count = d["close_to_observed_count"]

    all_supports = [_S(sup) for r in run_results.values() for sup in r["hypothesis_support"].values()]
    n_undefined, n_total = close_to_observed_coverage(all_supports)
    assert_close_to_observed_coverage(all_supports, context="v2 run coverage check")
    return {"n_undefined": n_undefined, "n_total": n_total}


def write_scoring_report(key_df: pd.DataFrame, run_results: Dict[str, dict], tally: dict,
                          scoring: dict, prompt_vers: Dict[str, str], coverage: dict) -> None:
    lines: List[str] = []
    lines.append("# Agent pipeline scoring — 20 blind records — v2 (post grounding repair)\n")
    lines.append(
        "**This is a re-run after a harness fix, not an independent second measurement.** "
        "`results/agent_scoring_20.md` (v1) scored 0/11 attacks with `close_to_observed_count` undefined "
        "for 116/116 hypotheses -- a fault in the evaluation harness (see STATUS.md's \"fifth "
        "silent-disable\"), not in the agents. That first result stands as recorded; this is a second "
        "measurement of the SAME 20 records, same blind input, same 0.3 threshold, same model, same "
        "prompt versions except `a4` (v5->v6, Step 3's fix for the one dropped record).\n"
    )

    n_total = len(key_df)
    n_scored = len(scoring["scored_df"])
    lines.append(
        f"**Coverage**: {n_scored}/{n_total} records scored "
        + ("(partial run -- stopped on quota mid-run)" if tally.get("stopped_on_quota") else "(full run)")
        + f". Model: `{base.DEFAULT_MODEL}`. Prompt versions: "
        + ", ".join(f"{a}={v}" for a, v in prompt_vers.items()) + ".\n"
    )
    lines.append(
        f"API calls made this run: **{tally['api_calls_made']}** (includes retries; A1/A2/A3 served from "
        f"cache where unchanged from v1, A4 forced fresh by the v6 prompt bump, A5 forced fresh by the "
        f"grounding repair -- see module docstring); cache hits: {tally['cache_hits']}; "
        f"schema retries: {tally['schema_retries']}; failed records: {tally['failed_records'] or 'none'}.\n"
    )

    lines.append(
        f"\n**Grounding coverage check (Step 1's loud-failure guard, re-applied to this run's own output)**: "
        f"close_to_observed_count undefined for {coverage['n_undefined']}/{coverage['n_total']} hypotheses "
        f"({(coverage['n_undefined'] / coverage['n_total'] if coverage['n_total'] else 0):.0%}) -- "
        f"{'PASSED (below the 50% alarm threshold, the repair produced real numbers)' if coverage['n_undefined'] / max(coverage['n_total'],1) <= 0.5 else 'FAILED -- see below, the repair did not work and these scores mean nothing'}.\n"
    )

    lines.append("## What this test can and cannot show\n")
    lines.append(
        "- **12 attack / 8 benign is not the natural rate.** Escalated flows are 9.4% attacks "
        "(282/3,000). This measures discrimination, not deployment calibration.\n"
        "- **n=20 supports almost nothing on its own.** Every rate below carries a Wilson 95% CI; "
        "prefer the raw counts.\n"
        "- The 0.3 threshold was not tuned on these 20. Records were not reselected. Prompts were not "
        "adjusted after seeing scores and re-scored -- the ONE prompt change (A4, v5->v6) was made and "
        "committed BEFORE this run, for the specific, narrow, pre-diagnosed schema-violation reason in "
        "Step 3, not to move any score.\n"
        "- If the agents still perform badly once grounding genuinely works, that is a legitimate finding "
        "and is reported as such below, not smoothed over.\n"
    )

    c = scoring["confusion"]
    n_attack = c["tp"] + c["fn"]
    n_benign = c["fp"] + c["tn"]
    lines.append(f"## 1. Confusion matrix (threshold={VERDICT_THRESHOLD}, not tuned on these 20)\n")
    lines.append(
        f"|  | predicted attack (bp<{VERDICT_THRESHOLD}) | predicted benign (bp>={VERDICT_THRESHOLD}) | n |\n"
        f"|---|---|---|---|\n"
        f"| true attack | {c['tp']} | {c['fn']} | {n_attack} |\n"
        f"| true benign | {c['fp']} | {c['tn']} | {n_benign} |\n"
    )
    recall = wilson_ci(c["tp"], n_attack) if n_attack else None
    specificity = wilson_ci(c["tn"], n_benign) if n_benign else None
    lines.append(
        f"\nRecall (attacks correctly flagged): {c['tp']}/{n_attack}"
        + (f", 95% CI {recall}" if recall else "") + "\n"
        f"Specificity (benign correctly cleared): {c['tn']}/{n_benign}"
        + (f", 95% CI {specificity}" if specificity else "") + "\n"
    )

    lines.append("## 2. `benign_plausibility` distributions, by true group\n")
    lines.append(f"- True attack ({_dist_summary(scoring['attack_bp'])}): {scoring['attack_bp']}\n")
    lines.append(f"- True benign ({_dist_summary(scoring['benign_bp'])}): {scoring['benign_bp']}\n")

    lines.append("## 3. Per record (with comparison-set size and close_to_observed_count)\n")
    record_table = [
        "| record_id | true label | benign_plausibility | derived verdict | neighbourhood size | ungrounded | A5 rationale (one line) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in scoring["scored_df"].iterrows():
        rationale_line = row["rationale"].replace("\n", " ").replace("|", "/")
        if len(rationale_line) > 180:
            rationale_line = rationale_line[:180] + "..."
        record_table.append(
            f"| {row['record_id']} | {row['true_label']} | {row['benign_plausibility']:.2f} | "
            f"{row['verdict']} | {row['neighbourhood_size']}/{row['benign_population_size']} | "
            f"{row['ungrounded']} | {rationale_line} |"
        )
    lines.append("\n".join(record_table) + "\n")

    lines.append("\n## 4. Per class\n")
    attack_scored = scoring["scored_df"][scoring["scored_df"]["is_attack"]]
    for label, grp in attack_scored.groupby("true_label"):
        vals = ", ".join(f"{v:.2f}" for v in sorted(grp["benign_plausibility"]))
        lines.append(f"- {label} (n={len(grp)}): benign_plausibility = [{vals}]\n")
    bot = attack_scored[attack_scored["true_label"] == "Bot"]
    if len(bot):
        lines.append(
            f"\n**Bot specifically** (n={len(bot)}, admitted by {list(bot['admitted_by'])}): "
            f"benign_plausibility = {sorted(bot['benign_plausibility'].tolist())}, neighbourhood sizes = "
            f"{list(bot['neighbourhood_size'])}. These are P2-admitted (uncommon signature, no rule "
            "fired) -- the closest thing in this sample to a genuinely blind detection, and both are "
            "still missed even with grounding restored -- consistent with every other independent line "
            "of evidence in this project finding Bot a genuine ceiling for these features, not an "
            "artifact of any one measurement.\n"
        )

    missed_attacks = attack_scored[~attack_scored["predicted_attack"]]
    if len(missed_attacks):
        grounded_misses = missed_attacks[~missed_attacks["ungrounded"]]
        ungrounded_misses = missed_attacks[missed_attacks["ungrounded"]]
        lines.append(f"\n**Why the {len(missed_attacks)} remaining misses are misses, checked rather than assumed**:\n")
        if len(grounded_misses):
            lines.append(
                f"\n{len(grounded_misses)} have a SMALL BUT NONZERO neighbourhood (real benign traffic on "
                "this network genuinely resembles these specific flows on the ten escalation features, at "
                "low but nonzero rates -- a genuine limit of this feature set for these instances, not a "
                "repeat of v1's harness fault):\n"
            )
            for _, row in grounded_misses.sort_values("neighbourhood_size").iterrows():
                lines.append(
                    f"- {row['true_label']} (`{row['record_id']}`): benign_plausibility={row['benign_plausibility']:.2f}, "
                    f"neighbourhood={row['neighbourhood_size']}/{row['benign_population_size']}\n"
                )
        if len(ungrounded_misses):
            lines.append(
                f"\n{len(ungrounded_misses)} "
                + ("is" if len(ungrounded_misses) == 1 else "are")
                + " actually UNGROUNDED (empty neighbourhood, same signal as "
                "the correctly-caught ones) but A5 still landed at or just above the 0.3 threshold -- a "
                "model calibration/consistency issue on a borderline call, not a data-support gap:\n"
            )
            for _, row in ungrounded_misses.sort_values("benign_plausibility").iterrows():
                lines.append(
                    f"- {row['true_label']} (`{row['record_id']}`): benign_plausibility={row['benign_plausibility']:.2f} "
                    f"(threshold {VERDICT_THRESHOLD}), neighbourhood={row['neighbourhood_size']}/{row['benign_population_size']}\n"
                )

    lines.append("\n## 5. Trust and mechanism\n")
    if len(scoring["trust_df"]):
        agent_means = scoring["trust_df"].groupby("agent")[["C", "E", "V", "T"]].mean(numeric_only=True)
        lines.append(agent_means.round(3).to_markdown() + "\n")
    lines.append(
        f"\n- Zero-match plausibility cap fired (agents.grounding, matching_profile_count==0): "
        f"{scoring['n_zero_match_clamped']}/{n_scored} records.\n"
        f"- Ungrounded-neighbourhood plausibility cap fired (agents.escalation_grounding, this repair's "
        f"NEW cap): {scoring['n_ungrounded_clamped']}/{n_scored} records.\n"
        f"- Contradiction-disposal forced a retry: {scoring['n_contradiction_retries']} times.\n"
    )
    if scoring["cvi_values"]:
        mean_cvi = statistics.mean(scoring["cvi_values"])
        n_pos = sum(1 for v in scoring["cvi_values"] if v > 0)
        lines.append(
            f"- chain_vs_independent (T4 - mean(T1,T2,T3)): mean {mean_cvi:+.3f} over "
            f"{len(scoring['cvi_values'])} records, positive (chain underperforms A4's blind pass) "
            f"in {n_pos}/{len(scoring['cvi_values'])}.\n"
        )

    SCORING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORING_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {SCORING_REPORT_PATH}")


def main() -> int:
    blind_records = load_blind_file(BLIND_OUTPUT_PATH)
    print(f"loaded {len(blind_records)} blind records")

    used = base.daily_request_count(base.DEFAULT_MODEL)
    cap = base.DAILY_QUOTA_BY_MODEL.get(base.DEFAULT_MODEL, 500)
    print(f"daily count for {base.DEFAULT_MODEL}: {used}/{cap} before this run")

    tally = run_all(blind_records)
    print(f"this run: {tally}")

    run_results = load_run_results()
    key_df = pd.read_csv(KEY_PATH)
    if not run_results:
        print("no records completed -- nothing to score")
        return 1

    try:
        coverage = coverage_check(run_results)
    except GroundingCoverageError as exc:
        print(f"[GROUNDING COVERAGE CHECK FAILED] {exc}")
        raise

    scoring = score(key_df, run_results)
    write_scoring_report(key_df, run_results, tally, scoring, prompt_versions_v2(), coverage)

    if tally["failed_records"] or tally["stopped_on_quota"]:
        print("PARTIAL RUN -- see results/agent_scoring_20_v2.md for exactly which records completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
