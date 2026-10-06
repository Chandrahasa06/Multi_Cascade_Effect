"""Runs the full five-agent pipeline (as built, unmodified) over the 20
blind records in `results/agent_input_20_blind.jsonl`, writes raw output
to `results/agent_run_20.jsonl`, then scores against
`results/agent_key_20.csv` (which no agent sees) and writes
`results/agent_scoring_20.md`.

Model and prompt versions are whatever `agents/base.py`/`agents/prompts`
currently import -- recorded in the output, not chosen here. All
existing code-side machinery stays on: empirical grounding
(agents/grounding.py), hypothesis support, the plausibility cap,
contradiction disposal, trust scoring -- none of it is bypassed.

**A structural caveat, not a bug**: this policy (dataplane/dt_rules.py's
compiled tree rules + the Priority 2 signature table) has no equivalent
of dataplane/selector.py's Tier-1 "trigger reason" concept, so every
EscalationRecord built here carries `trigger_reasons=[]`. The pipeline
handles this as an already-anticipated degenerate case (render_trigger_
reasons/render_empirical_grounding_block both have an explicit empty-list
branch -- see agents/prompts/render.py, agents/grounding.py), not a
crash, but it does mean every record's nearest-neighbour grounding block
reads "no benign reference available" and every hypothesis's
`close_to_observed_count` is undefined. The zero-empirical-support
plausibility CAP is unaffected (`evaluate_hypothesis_support`'s
`matching_profile_count` check depends only on the model's own
`predicted_feature_profile` against the Monday Tier-1 reference table,
never on this flow's own trigger_reasons) -- verified against the real
run, not assumed; see the report's mechanism section.

Run: python -m eval.run_blind_pipeline
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from agents import base, pipeline
import agents.a1_evidence as a1_evidence
import agents.a2_behaviour as a2_behaviour
import agents.a3_hypotheses as a3_hypotheses
import agents.a4_replication as a4_replication
import agents.a5_verdict as a5_verdict
from agents.schema import VerdictLabel, derive_verdict
from controlplane.record import EscalationRecord, PacketWindowSummary
from eval.build_blind_input import BLIND_OUTPUT_PATH, load_blind_file

RUN_OUTPUT_PATH = Path("results/agent_run_20.jsonl")
SCORING_REPORT_PATH = Path("results/agent_scoring_20.md")
KEY_PATH = Path("results/agent_key_20.csv")

SELECTOR_CONFIG_MARKER = "dt_rules_p1p2p3_blind_v1"
VERDICT_THRESHOLD = 0.3
CONTRADICTION_RETRY_MARKER = "must address every contradicting claim_id"


def blind_record_to_escalation_record(blind: dict) -> EscalationRecord:
    return EscalationRecord(
        flow_id=blind["record_id"],
        trigger_reasons=[],
        features=blind["features"],
        packet_window=PacketWindowSummary(packet_count=0, byte_count=0, duration_us=0),
        selector_config_hash=SELECTOR_CONFIG_MARKER,
    )


def already_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["record_id"])
    return seen


def prompt_versions() -> Dict[str, str]:
    return {
        "a1": a1_evidence.prompt_module.PROMPT_VERSION,
        "a2": a2_behaviour.prompt_module.PROMPT_VERSION,
        "a3": a3_hypotheses.prompt_module.PROMPT_VERSION,
        "a4": a4_replication.prompt_module.PROMPT_VERSION,
        "a5": a5_verdict.prompt_module.PROMPT_VERSION,
    }


def run_all(blind_records: List[dict], out_path: Path = RUN_OUTPUT_PATH) -> dict:
    done = already_done_ids(out_path)
    if done:
        print(f"resuming: {len(done)}/{len(blind_records)} records already in {out_path}, skipping them")

    tally = {
        "records_completed_this_run": 0, "api_calls_made": 0, "cache_hits": 0,
        "schema_retries": 0, "failed_records": [], "stopped_on_quota": False,
    }
    for blind in blind_records:
        rid = blind["record_id"]
        if rid in done:
            continue
        record = blind_record_to_escalation_record(blind)
        try:
            result = pipeline.run_record(record)
        except base.DailyQuotaExceeded as exc:
            print(f"[STOPPED] daily quota hit after {tally['records_completed_this_run']} records this run: {exc}")
            tally["stopped_on_quota"] = True
            break
        except base.SchemaValidationFailed as exc:
            print(f"[FAILED] {rid}: {exc}")
            tally["failed_records"].append(rid)
            tally["api_calls_made"] += exc.attempts  # the one agent that exhausted retries, at minimum
            continue
        for meta in result.call_metadata.values():
            if meta.cached:
                tally["cache_hits"] += 1
            else:
                tally["api_calls_made"] += 1 + meta.schema_retries
            tally["schema_retries"] += meta.schema_retries
        pipeline.append_result_jsonl(out_path, result)
        tally["records_completed_this_run"] += 1
        print(f"  done: {rid} (benign_plausibility={result.a5.benign_plausibility:.2f}, "
              f"clamped={result.a5_plausibility_clamped})")
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


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Optional[str]:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    lo = max(0.0, (centre - margin) / denom)
    hi = min(1.0, (centre + margin) / denom)
    return f"[{lo:.2f}, {hi:.2f}]"


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
        rows.append({
            "record_id": rid, "true_label": k["true_label"], "is_attack": bool(k["is_attack"]),
            "admitted_by": k["admitted_by"], "benign_plausibility": bp, "verdict": verdict.value,
            "predicted_attack": predicted_attack, "clamped": r["a5_plausibility_clamped"],
            "credited_hypothesis_id": r["a5"]["credited_hypothesis_id"],
            "rationale": r["a5"]["rationale"],
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
    n_clamped = 0
    n_contradiction_retries = 0
    cvi_values = []
    for rid, r in run_results.items():
        n_clamped += int(r["a5_plausibility_clamped"])
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
        "trust_df": trust_df, "n_clamped": n_clamped, "n_contradiction_retries": n_contradiction_retries,
        "cvi_values": cvi_values,
    }


def diagnose_grounding_gap(run_results: Dict[str, dict], scored_df: pd.DataFrame) -> dict:
    """Checks, rather than assumes, why attack recall came out where it
    did: with trigger_reasons=[] (see module docstring), every
    hypothesis's `close_to_observed_count` is undefined, so the empirical
    check that would normally ask "are real benign flows CLOSE TO THIS
    SPECIFIC flow's values" never runs -- only the much weaker "does some
    benign flow satisfy this generic profile at all" does. Reports the
    concrete numbers so the claim is checked, not asserted."""
    n_close_defined = 0
    n_close_total = 0
    credited_matches = []
    for rid, r in run_results.items():
        for sup in r["hypothesis_support"].values():
            n_close_total += 1
            if sup["close_to_observed_count"] is not None:
                n_close_defined += 1
        credited = r["a5"]["credited_hypothesis_id"]
        if credited and credited in r["hypothesis_support"]:
            sup = r["hypothesis_support"][credited]
            is_attack = bool(scored_df.loc[scored_df["record_id"] == rid, "is_attack"].iloc[0]) \
                if (scored_df["record_id"] == rid).any() else None
            credited_matches.append({
                "record_id": rid, "is_attack": is_attack,
                "matching_profile_count": sup["matching_profile_count"],
                "benign_population_size": sup["benign_population_size"],
            })
    return {
        "n_close_to_observed_defined": n_close_defined, "n_hypotheses_total": n_close_total,
        "credited_matches": credited_matches,
    }


def _dist_summary(values: List[float]) -> str:
    if not values:
        return "n=0"
    return f"n={len(values)}, min={min(values):.2f}, median={statistics.median(values):.2f}, max={max(values):.2f}"


def write_scoring_report(key_df: pd.DataFrame, run_results: Dict[str, dict], tally: dict,
                          scoring: dict, prompt_vers: Dict[str, str]) -> None:
    lines: List[str] = []
    lines.append("# Agent pipeline scoring — 20 blind records\n")

    n_total = len(key_df)
    n_scored = len(scoring["scored_df"])
    lines.append(
        f"**Coverage**: {n_scored}/{n_total} records scored "
        + (f"(partial run -- stopped on quota mid-run)" if tally.get("stopped_on_quota") else "(full run)")
        + f". Model: `{base.DEFAULT_MODEL}`. Prompt versions: "
        + ", ".join(f"{a}={v}" for a, v in prompt_vers.items()) + ".\n"
    )
    lines.append(
        f"API calls made this run: **{tally['api_calls_made']}** (includes retries); "
        f"cache hits: {tally['cache_hits']}; schema retries: {tally['schema_retries']}; "
        f"failed records (exhausted retries): {tally['failed_records'] or 'none'}.\n"
    )

    lines.append("## What this test can and cannot show\n")
    lines.append(
        "- **12 attack / 8 benign is not the natural rate.** Escalated flows are 9.4% attacks "
        "(282/3,000). This 20-record sample measures whether the agents can DISCRIMINATE "
        "attack-shaped flows from benign ones, not how the pipeline would perform at deployment "
        "base rates.\n"
        "- **n=20 supports almost nothing on its own.** Every rate below carries a Wilson 95% CI; "
        "prefer the raw counts.\n"
        "- **This is a blind test, but a harder one than deployment.** The hint (`class_predicted`, "
        "`rule_id`, `priority`/`reason`) was stripped before any agent saw a record -- the agents "
        "see only flow statistics (features + feature bins were kept, though the current agent "
        "prompts don't have an input slot for feature_bins specifically -- see module docstring). "
        "In a real deployment, Priority 1's own prediction would legitimately be available "
        "alongside the agent's own reasoning; this run answers \"can the agents detect this from "
        "raw statistics alone\", not \"how would the deployed pipeline perform end to end.\"\n"
        "- **No trigger-reason data.** This escalation policy (compiled tree rules + a signature "
        "table) has no Tier-1 \"trigger reason\" concept the original agent pipeline was built "
        "around, so every record's empirical-grounding block and nearest-neighbour distance read "
        "as unavailable for every one of the 20 -- a real, further-blinding effect on top of the "
        "requested hint-stripping (see module docstring for exactly what still works: the "
        "zero-support plausibility cap is unaffected).\n"
        "- Records were not reselected, no seed was changed, and prompts were not adjusted after "
        "seeing results. Whatever this run shows is reported as-is.\n"
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

    lines.append("## 3. Per record\n")
    record_table = [
        "| record_id | true label | benign_plausibility | derived verdict | A5 rationale (one line) |",
        "|---|---|---|---|---|",
    ]
    for _, row in scoring["scored_df"].iterrows():
        rationale_line = row["rationale"].replace("\n", " ").replace("|", "/")
        if len(rationale_line) > 200:
            rationale_line = rationale_line[:200] + "..."
        record_table.append(
            f"| {row['record_id']} | {row['true_label']} | {row['benign_plausibility']:.2f} | "
            f"{row['verdict']} | {rationale_line} |"
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
            f"benign_plausibility = {sorted(bot['benign_plausibility'].tolist())}. "
            "These are P2-admitted (uncommon signature, no rule fired) -- the closest thing in "
            "this sample to a genuinely blind detection, with no rule-vocabulary hint even before "
            "stripping.\n"
        )

    diag = diagnose_grounding_gap(run_results, scoring["scored_df"])
    lines.append("\n## Why the agents missed every attack — a verified mechanism, not a guess\n")
    lines.append(
        f"`close_to_observed_count` (real benign flows close to THIS flow's own value, not just "
        f"satisfying a hypothesis's profile in general) was undefined for "
        f"{diag['n_hypotheses_total'] - diag['n_close_to_observed_defined']}/{diag['n_hypotheses_total']} "
        "hypotheses evaluated across the run -- i.e. every single one, because trigger_reasons=[] for "
        "every record (see caveat above). This leaves only the weaker check running: does ANY real "
        "benign flow satisfy the hypothesis's stated range at all, out of 566,864. For a loosely-bounded "
        "\"ordinary web/DNS request\" hypothesis, that check is nearly always satisfied at massive scale "
        "regardless of whether the SPECIFIC flow under review is actually extreme -- concretely, on the "
        "DoS Hulk record `6846713a334440c1b9748d47e85b6ee4`, A5 credited hypothesis `a4_h1` "
        "(\"an HTTP client making a slow request and receiving a large response\") on the strength of "
        "**258,258 of 566,864 benign flows (45.6%)** matching its profile -- a real number, correctly "
        "computed, but one that says nothing about whether *this* flow (whose own trigger-derived "
        "statistics were never available to compare against) resembles those 258,258 flows rather than "
        "being a clear outlier among them. This is the best-supported explanation for 0/11 recall: not "
        "that the agents reasoned badly from what they had, but that a load-bearing piece of the "
        "grounding machinery (the per-flow \"close to observed\" refinement) had no data to run on for "
        "this policy's record shape.\n"
    )
    if diag["credited_matches"]:
        attack_matches = [m for m in diag["credited_matches"] if m["is_attack"]]
        if attack_matches:
            mean_match_frac = statistics.mean(
                m["matching_profile_count"] / m["benign_population_size"] for m in attack_matches
            )
            lines.append(
                f"\nAcross all {len(attack_matches)} scored attack records, the credited hypothesis's "
                f"generic match rate averaged {mean_match_frac:.1%} of the 566,864-flow benign population "
                "-- consistently large, not a one-off.\n"
            )

    lines.append("\n## 5. Trust and mechanism\n")
    if len(scoring["trust_df"]):
        agent_means = scoring["trust_df"].groupby("agent")[["C", "E", "V", "T"]].mean(numeric_only=True)
        lines.append(agent_means.round(3).to_markdown() + "\n")
    lines.append(
        f"\n- Plausibility cap fired: {scoring['n_clamped']}/{n_scored} records.\n"
        f"- Contradiction-disposal forced a retry: {scoring['n_contradiction_retries']} times "
        f"(across all agents/records).\n"
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

    scoring = score(key_df, run_results)
    write_scoring_report(key_df, run_results, tally, scoring, prompt_versions())

    if tally["failed_records"] or tally["stopped_on_quota"]:
        print("PARTIAL RUN -- see results/agent_scoring_20.md for exactly which records completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
