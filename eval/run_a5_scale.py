"""Step 3 of results/plausibility_diagnostic.md: re-run A5 ONLY, on all 36
records scored by the v2/batch-2 runs, using the fixed-scale prompt
(agents/prompts/a5_verdict_scale_v1.py / agents/a5_verdict_scale.py)
instead of the free-form one. A1-A4 are NOT re-called -- their already-
recorded responses (results/agent_run_20_v2.jsonl,
results/agent_run_20b.jsonl) are reused verbatim, reconstructed into the
same Pydantic objects the real pipeline produced. hypothesis_support and
escalation_neighbourhood are likewise reused verbatim (already patched
by agents/escalation_grounding.py in the original run) -- nothing about
the evidence or the grounding changes, only A5's own output format.

This is a DIAGNOSTIC re-elicitation, not a replacement measurement: its
output (results/agent_run_a5_scale.jsonl) does not supersede
results/agent_scoring_20_v2.md or results/agent_scoring_20b.md, and nei-
ther the 0.3 verdict threshold nor any prompt used in those two recorded
runs is touched by this script.

Run: python -m eval.run_a5_scale
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from agents import a5_verdict_scale, base
from agents.escalation_grounding import apply_ungrounded_neighbourhood_cap
from agents.grounding import HypothesisSupport
from agents.schema import Claim, ClaimsResponse, Hypothesis, HypothesisResponse
from agents.a5_verdict_scale import SCALE_CAP
from controlplane.record import EscalationRecord, PacketWindowSummary

SOURCE_RUNS = [
    ("results/agent_run_20_v2.jsonl", "results/agent_input_20_blind.jsonl", "batch1"),
    ("results/agent_run_20b.jsonl", "results/agent_input_20b_blind.jsonl", "batch2"),
    ("results/agent_run_10c.jsonl", "results/agent_input_10c_blind.jsonl", "batch3"),
]
#: The 45-record diagnostic (batches 1-3). The earlier 36-record run stays in
#: results/agent_run_a5_scale.jsonl; its inputs are identical, so its entries are
#: cache hits here and cost no quota.
OUT_PATH = Path("results/agent_run_a5_scale_45.jsonl")


def _load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _blind_to_record(blind: dict) -> EscalationRecord:
    return EscalationRecord(
        flow_id=blind["record_id"],
        trigger_reasons=[],
        features=blind["features"],
        packet_window=PacketWindowSummary(packet_count=0, byte_count=0, duration_us=0),
        selector_config_hash="dt_rules_p1p2p3_blind_v1",
    )


def _already_done(path: Path) -> set:
    if not path.exists():
        return set()
    return {d["record_id"] for d in _load_jsonl(str(path))}


def load_all_source_records() -> List[dict]:
    """One entry per already-scored record: {record_id, batch, record
    (EscalationRecord), a1, a2, a3, a4 (Pydantic objects), hypothesis_support
    (HypothesisSupport dict), neighbourhood (dict), original_bp (float,
    the ALREADY-RECORDED free-form v2/batch-2 score, for the report's
    old-vs-new comparison -- never used as input to the new prompt)."""
    out = []
    for run_path, blind_path, batch in SOURCE_RUNS:
        run_rows = {d["record_id"]: d for d in _load_jsonl(run_path)}
        blind_rows = {d["record_id"]: d for d in _load_jsonl(blind_path)}
        for rid, d in run_rows.items():
            blind = blind_rows[rid]
            supports = {
                hid: HypothesisSupport(
                    hypothesis_id=s["hypothesis_id"],
                    matching_profile_count=s["matching_profile_count"],
                    close_to_observed_count=s["close_to_observed_count"],
                    checked_features=s["checked_features"],
                    benign_population_size=s["benign_population_size"],
                )
                for hid, s in d["hypothesis_support"].items()
            }
            out.append({
                "record_id": rid,
                "batch": batch,
                "record": _blind_to_record(blind),
                "a1": ClaimsResponse.model_validate(d["a1"]),
                "a2": ClaimsResponse.model_validate(d["a2"]),
                "a3": HypothesisResponse.model_validate(d["a3"]),
                "a4": HypothesisResponse.model_validate(d["a4"]),
                "hypothesis_support": supports,
                "neighbourhood_dict": d["escalation_neighbourhood"],
                "original_bp": d["a5"]["benign_plausibility"],
            })
    return out


def run_all() -> dict:
    from agents.grounding import render_hypothesis_support
    from agents.escalation_grounding import FeatureNeighbourhood

    entries = load_all_source_records()
    done = _already_done(OUT_PATH)
    if done:
        print(f"resuming: {len(done)}/{len(entries)} already in {OUT_PATH}, skipping them")

    tally = {"completed_this_run": 0, "api_calls_made": 0, "cache_hits": 0,
             "schema_retries": 0, "failed_records": [], "failed_record_reasons": {},
             "stopped_on_quota": False}

    for e in entries:
        rid = e["record_id"]
        if rid in done:
            continue
        empirical_support_lines = {
            hid: render_hypothesis_support(s) for hid, s in e["hypothesis_support"].items()
        }
        try:
            response, meta, floor_fired = a5_verdict_scale.run(
                e["record"], e["a1"], e["a2"], e["a3"], e["a4"],
                empirical_support_lines, e["hypothesis_support"], use_cache=True,
            )
        except base.DailyQuotaExceeded as exc:
            print(f"[STOPPED] daily quota hit after {tally['completed_this_run']} this run: {exc}")
            tally["stopped_on_quota"] = True
            break
        except base.SchemaValidationFailed as exc:
            print(f"[FAILED] {rid}: {exc}")
            tally["failed_records"].append(rid)
            tally["failed_record_reasons"][rid] = str(exc)
            tally["api_calls_made"] += exc.attempts
            continue

        neighbourhood = FeatureNeighbourhood(**e["neighbourhood_dict"])
        response, ungrounded_cap_fired = apply_ungrounded_neighbourhood_cap(
            response, neighbourhood, cap=SCALE_CAP
        )

        if meta.cached:
            tally["cache_hits"] += 1
        else:
            tally["api_calls_made"] += 1 + meta.schema_retries
        tally["schema_retries"] += meta.schema_retries

        record_dict = {
            "record_id": rid,
            "batch": e["batch"],
            "original_bp": e["original_bp"],
            "scale_bp": response.benign_plausibility,
            "confidence": response.confidence,
            "evidence_support": response.evidence_support,
            "verification": response.verification,
            "credited_hypothesis_id": response.credited_hypothesis_id,
            "cited_claim_ids": response.cited_claim_ids,
            "rationale": response.rationale,
            "floor_fired": floor_fired,
            "ungrounded_cap_fired": ungrounded_cap_fired,
            "neighbourhood_size": e["neighbourhood_dict"]["neighbourhood_size"],
            "ungrounded": e["neighbourhood_dict"]["ungrounded"],
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_dict) + "\n")
        tally["completed_this_run"] += 1
        print(f"  done: {rid} (original_bp={e['original_bp']:.2f} -> scale_bp={response.benign_plausibility:.2f})")

    return tally


def main() -> int:
    used = base.daily_request_count(base.DEFAULT_MODEL)
    cap = base.DAILY_QUOTA_BY_MODEL.get(base.DEFAULT_MODEL, 500)
    print(f"daily count for {base.DEFAULT_MODEL}: {used}/{cap} before this run")
    tally = run_all()
    print(f"this run: {tally}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
