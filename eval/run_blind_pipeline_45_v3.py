"""Re-run of the full pipeline on the SAME 45 records, with the multi-signal benign
evidence block given to A1, A3 and A5 (new prompts a1/a3/a5 v6). The corrected cap
(0.29, agents/escalation_grounding.py) is in effect.

This is a RE-RUN after a change, not an independent measurement. Same records, same
blind inputs, same 0.30 threshold, same model and temperature.

Cache policy:
  - A4 uses the cache. Its prompt and grounding block are unchanged, so its recorded
    responses are still what would be produced.
  - A1, A2, A3, A5 do not read or write the cache. Their inputs changed (the evidence
    block and, for A2/A3/A5, the upstream outputs). The cache key has no upstream or
    evidence fields, so a cached entry could be stale.

Evidence is built from benign rows only (agents/evidence_signals.py). The source IP and
time of each blind flow are recovered by matching its features against the pool
(evidence_signals.locate_source). Label is never read there.

Output: results/agent_run_45_v3.jsonl (append, resumable). Failures and quota stops are
recorded, not hidden.

Run: python -m eval.run_blind_pipeline_45_v3
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from agents import a1_evidence, a2_behaviour, a3_hypotheses, a4_replication, a5_verdict, base, evidence_signals, pipeline
from agents.escalation_grounding import (
    apply_ungrounded_neighbourhood_cap,
    compute_feature_neighbourhood,
    observed_escalation_values,
    patch_hypothesis_support,
)
from agents.grounding import render_hypothesis_support
from eval.escalation_data import CACHE_PATH
from eval.run_blind_pipeline import blind_record_to_escalation_record
from eval.run_blind_pipeline_45_v2 import load_45, load_api_key_from_dotenv, already_done

RUN_OUTPUT_V3 = Path("results/agent_run_45_v3.jsonl")
TALLY_PATH = Path("results/agent_run_45_v3_tally.json")


def build_evidence_index(records: List[dict]) -> Dict[str, dict]:
    """record_id -> {'block': rendered evidence text, 'signals': dict, 'source_status': str}."""
    pool = evidence_signals.load_pool_slim(extra_columns=records[0]["features"].keys())
    ref = evidence_signals.build_reference(pool)
    match_cols = [c for c in records[0]["features"] if c in pool.columns]
    out = {}
    for blind in records:
        source, status = evidence_signals.locate_source(pool, blind["features"], match_cols)
        b = evidence_signals.bundle(ref, blind["features"], source)
        out[blind["record_id"]] = {
            "block": evidence_signals.render_evidence_block(b),
            "signals": b.as_signals(),
            "source_status": status,
        }
    return out


def run_record_v3(record, evidence_block: str, *, model: str = base.DEFAULT_MODEL, temperature: float = 0.7,
                  run_index: int = 0):
    cache_on = dict(model=model, temperature=temperature, run_index=run_index, use_cache=True)
    cache_off = dict(cache_on, use_cache=False)
    grounding = pipeline.compute_grounding(record)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a4_future = pool.submit(
            a4_replication.run, record, grounding.empirical_grounding_block, grounding.available_features,
            grounding.known_reference_features, **cache_on,
        )
        a1_resp, a1_meta = a1_evidence.run(record, grounding.empirical_grounding_block,
                                           evidence_block=evidence_block, **cache_off)
        a2_resp, a2_meta = a2_behaviour.run(record, a1_resp, grounding.empirical_grounding_block, **cache_off)
        a3_resp, a3_meta = a3_hypotheses.run(
            record, a1_resp, a2_resp, grounding.empirical_grounding_block, grounding.available_features,
            grounding.known_reference_features, evidence_block=evidence_block, **cache_off,
        )
        a4_resp, a4_meta = a4_future.result()

    hypothesis_support, _ = pipeline.compute_hypothesis_support(record, a3_resp, a4_resp, grounding.benign_df)
    observed = observed_escalation_values(record.features)
    neighbourhood = compute_feature_neighbourhood(observed)
    hypothesis_support = patch_hypothesis_support(hypothesis_support, neighbourhood)
    empirical_support_lines = {hid: render_hypothesis_support(s) for hid, s in hypothesis_support.items()}

    a5_resp, a5_meta, a5_clamped_original = a5_verdict.run(
        record, a1_resp, a2_resp, a3_resp, a4_resp, empirical_support_lines, hypothesis_support,
        evidence_block=evidence_block, **cache_off,
    )
    a5_resp, ungrounded_clamped = apply_ungrounded_neighbourhood_cap(a5_resp, neighbourhood)
    a5_plausibility_clamped = a5_clamped_original or ungrounded_clamped

    trust_scores, trust_decay_chain, trust_decay_with_a5, cvi = pipeline.compute_trust_and_decay(
        record, a1_resp, a2_resp, a3_resp, a4_resp, a5_resp,
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


def run_all_v3() -> dict:
    records = load_45()
    evidence = build_evidence_index(records)
    done = already_done(RUN_OUTPUT_V3)
    tally: Dict = {
        "records_total": len(records), "records_completed_this_run": 0, "already_done_before_run": len(done),
        "api_calls_made": 0, "cache_hits": 0, "schema_retries": 0, "failed_records": {},
        "stopped_on_quota": False, "stopped_at": None,
        "source_status": {},
    }
    for blind in records:
        tally["source_status"][evidence[blind["record_id"]]["source_status"]] = (
            tally["source_status"].get(evidence[blind["record_id"]]["source_status"], 0) + 1)
    for blind in records:
        rid = blind["record_id"]
        if rid in done:
            continue
        record = blind_record_to_escalation_record(blind)
        ev = evidence[rid]
        try:
            result, neighbourhood, ungrounded_clamped, orig_clamped = run_record_v3(record, ev["block"])
        except base.DailyQuotaExceeded as exc:
            print(f"[STOPPED] daily quota: {exc}", flush=True)
            tally["stopped_on_quota"] = True
            tally["stopped_at"] = rid
            break
        except base.SchemaValidationFailed as exc:
            print(f"[FAILED] {rid} ({blind['_batch']}): {exc}", flush=True)
            tally["failed_records"][rid] = {"batch": blind["_batch"], "agent": exc.agent,
                                            "retry_reasons": list(exc.retry_reasons)}
            tally["api_calls_made"] += exc.attempts
            continue
        for meta in result.call_metadata.values():
            if meta.cached:
                tally["cache_hits"] += 1
            else:
                tally["api_calls_made"] += 1 + meta.schema_retries
            tally["schema_retries"] += meta.schema_retries
        rec = result.to_dict()
        rec["escalation_neighbourhood"] = neighbourhood.to_dict()
        rec["ungrounded_cap_fired"] = ungrounded_clamped
        rec["zero_match_cap_fired"] = orig_clamped
        rec["batch"] = blind["_batch"]
        rec["evidence_signals"] = ev["signals"]
        rec["evidence_source_status"] = ev["source_status"]
        RUN_OUTPUT_V3.parent.mkdir(parents=True, exist_ok=True)
        with open(RUN_OUTPUT_V3, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        tally["records_completed_this_run"] += 1
        print(f"  done {rid} ({blind['_batch']}): bp={result.a5.benign_plausibility:.2f}", flush=True)
    TALLY_PATH.write_text(json.dumps(tally, indent=2, default=str), encoding="utf-8")
    return tally


def main() -> int:
    if not load_api_key_from_dotenv():
        raise SystemExit("GEMINI_API_KEY not available (not in environment, not in .env)")
    tally = run_all_v3()
    print(json.dumps({k: v for k, v in tally.items() if k != "failed_records"}, indent=2))
    print(f"failed records: {len(tally['failed_records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
