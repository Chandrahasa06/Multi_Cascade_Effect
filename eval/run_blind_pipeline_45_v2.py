"""Re-run of the full pipeline on the SAME 45 records scored by batches 1, 2 and 3,
after two fixes:
  - fix 1: the payload filter accepts absence-or-size claims (agents/validators.py)
  - fix 2: A2 receives the empirical grounding block (agents/prompts/a2_behaviour_v3.py)

This is a RE-RUN after a fix, not an independent measurement. Same records, same
blind inputs, same 0.30 threshold, same model, same temperature.

Cache policy, chosen to avoid silently serving stale results:
  - A1 and A4 use the cache. Their inputs did not change, and their prompts did not
    change, so their recorded responses are still exactly what would be produced.
  - A2, A3 and A5 do NOT read or write the cache. A2's output changed, so A3 and A5
    (which read A2) would otherwise serve results computed from the old A2. The cache
    key has no upstream fields, so it cannot detect this. Writing nothing also keeps
    the historical entries intact.

Output: results/agent_run_45_v2.jsonl (append, resumable). Failures and quota stops
are recorded and reported, not hidden.

Run: python -m eval.run_blind_pipeline_45_v2
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Dict, List

from agents import a1_evidence, a2_behaviour, a3_hypotheses, a4_replication, a5_verdict, base, pipeline
from agents.escalation_grounding import (
    apply_ungrounded_neighbourhood_cap,
    compute_feature_neighbourhood,
    observed_escalation_values,
    patch_hypothesis_support,
)
from agents.grounding import render_hypothesis_support
from eval.run_blind_pipeline import blind_record_to_escalation_record

RUN_OUTPUT_45 = Path("results/agent_run_45_v2.jsonl")
TALLY_PATH = Path("results/agent_run_45_v2_tally.json")
BATCHES = [
    ("batch1", Path("results/agent_input_20_blind.jsonl"), Path("results/agent_run_20_v2.jsonl")),
    ("batch2", Path("results/agent_input_20b_blind.jsonl"), Path("results/agent_run_20b.jsonl")),
    ("batch3", Path("results/agent_input_10c_blind.jsonl"), Path("results/agent_run_10c.jsonl")),
]


def load_api_key_from_dotenv(path: Path = Path(".env")) -> bool:
    """Exports GEMINI_API_KEY from the project's .env into this process only, if it
    is not already set. The value is never printed or written anywhere. The project
    keeps the key in that gitignored file; base._get_client reads the environment."""
    import os
    if os.environ.get("GEMINI_API_KEY") or not path.exists():
        return bool(os.environ.get("GEMINI_API_KEY"))
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                os.environ["GEMINI_API_KEY"] = value
                return True
    return False


def scored_ids(path: Path) -> List[str]:
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["record_id"])
    return ids


def load_45() -> List[dict]:
    """The 45 records: the ones batches 1-3 actually scored (20 + 16 + 9), in batch
    order, with their blind inputs. Asserted, so the set cannot drift."""
    out = []
    for batch, blind_path, run_path in BATCHES:
        ids = set(scored_ids(run_path))
        with open(blind_path, encoding="utf-8") as f:
            blind = [json.loads(line) for line in f if line.strip()]
        picked = [b for b in blind if b["record_id"] in ids]
        assert len(picked) == len(ids), f"{batch}: {len(ids) - len(picked)} scored ids not in blind input"
        for b in picked:
            b["_batch"] = batch
        out.extend(picked)
    assert len(out) == 45, f"expected 45 records, got {len(out)}"
    return out


def run_record_45(record, *, model: str = base.DEFAULT_MODEL, temperature: float = 0.7, run_index: int = 0):
    cache_on = dict(model=model, temperature=temperature, run_index=run_index, use_cache=True)
    cache_off = dict(cache_on, use_cache=False)  # A2, A3, A5: see module docstring
    grounding = pipeline.compute_grounding(record)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a4_future = pool.submit(
            a4_replication.run, record, grounding.empirical_grounding_block, grounding.available_features,
            grounding.known_reference_features, **cache_on,
        )
        a1_resp, a1_meta = a1_evidence.run(record, grounding.empirical_grounding_block, **cache_on)
        a2_resp, a2_meta = a2_behaviour.run(record, a1_resp, grounding.empirical_grounding_block, **cache_off)
        a3_resp, a3_meta = a3_hypotheses.run(
            record, a1_resp, a2_resp, grounding.empirical_grounding_block, grounding.available_features,
            grounding.known_reference_features, **cache_off,
        )
        a4_resp, a4_meta = a4_future.result()

    hypothesis_support, _ = pipeline.compute_hypothesis_support(record, a3_resp, a4_resp, grounding.benign_df)
    observed = observed_escalation_values(record.features)
    neighbourhood = compute_feature_neighbourhood(observed)
    hypothesis_support = patch_hypothesis_support(hypothesis_support, neighbourhood)
    empirical_support_lines = {hid: render_hypothesis_support(s) for hid, s in hypothesis_support.items()}

    a5_resp, a5_meta, a5_clamped_original = a5_verdict.run(
        record, a1_resp, a2_resp, a3_resp, a4_resp, empirical_support_lines, hypothesis_support, **cache_off,
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


def already_done(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["record_id"] for line in f if line.strip()}


def run_all_45() -> dict:
    records = load_45()
    done = already_done(RUN_OUTPUT_45)
    tally: Dict = {
        "records_total": len(records), "records_completed_this_run": 0, "already_done_before_run": len(done),
        "api_calls_made": 0, "cache_hits": 0, "schema_retries": 0, "failed_records": {},
        "stopped_on_quota": False, "stopped_at": None,
    }
    for blind in records:
        rid = blind["record_id"]
        if rid in done:
            continue
        record = blind_record_to_escalation_record(blind)
        try:
            result, neighbourhood, ungrounded_clamped, orig_clamped = run_record_45(record)
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
        RUN_OUTPUT_45.parent.mkdir(parents=True, exist_ok=True)
        with open(RUN_OUTPUT_45, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        tally["records_completed_this_run"] += 1
        print(f"  done {rid} ({blind['_batch']}): bp={result.a5.benign_plausibility:.2f}", flush=True)
    TALLY_PATH.write_text(json.dumps(tally, indent=2, default=str), encoding="utf-8")
    return tally


def main() -> int:
    if not load_api_key_from_dotenv():
        raise SystemExit("GEMINI_API_KEY not available (not in environment, not in .env)")
    tally = run_all_45()
    print(json.dumps({k: v for k, v in tally.items() if k != "failed_records"}, indent=2))
    print(f"failed records: {len(tally['failed_records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
