"""Reframe experiment: three verdict-stage arms on the same records, reusing the v3
upstream outputs. Only the verdict stage (and, for arm B, the advocate) is new.

Arms, per record (cache OFF for every new call; the new prompts never have cache entries):
  adversarial  A5 with the adversarial question                 1 call
  devil        advocate (DA) + A5 with both sides shown         2 calls
  twosided     A5 reporting benign and attack plausibility       1 call

Upstream A1-A4 and the evidence block are rebuilt from results/agent_run_45_v3.jsonl
with no API calls: the stored outputs are the same ones the v3 run used.

Verdict rules, fixed before any arm was run:
  adversarial, devil: attack iff benign_plausibility < 0.30 (unchanged threshold)
  twosided:           attack iff attack_plausibility > benign_plausibility
                      (agents/verdict_rules.py)
Both benign-side scores have the empirical and ungrounded caps applied, as in v3.

Record 1b21af7f failed A3 in v3 and has no upstream output. It is reported as unscored,
not substituted.

Output: results/agent_run_45_reframe.jsonl (one line per record and arm; resumable).
Tally:  results/agent_run_45_reframe_tally.json.
Run:    python -m eval.run_reframe_45
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from agents import da_advocate, evidence_signals as es
from agents import pipeline
from agents.a5_reframe import ARMS, run_arm
from agents.escalation_grounding import (
    FeatureNeighbourhood,
    apply_ungrounded_neighbourhood_cap,
    patch_hypothesis_support,
)
from agents.grounding import HypothesisSupport, evaluate_all, observed_tier1_values, render_hypothesis_support
from agents.schema import ClaimsResponse, HypothesisResponse
from agents.verdict_rules import ARM_C_RULE, derive_two_sided_verdict, two_sided_state
from eval.run_blind_pipeline import blind_record_to_escalation_record
from eval.run_blind_pipeline_45_v2 import load_api_key_from_dotenv, load_45
from agents import base

V3_RUN = Path("results/agent_run_45_v3.jsonl")
OUT = Path("results/agent_run_45_reframe.jsonl")
TALLY = Path("results/agent_run_45_reframe_tally.json")
THRESHOLD = 0.30


def reconstruct(rec: dict, blind: dict) -> dict:
    """Upstream state for one record, rebuilt from the stored v3 output. No API calls."""
    a1 = ClaimsResponse(**rec["a1"])
    a2 = ClaimsResponse(**rec["a2"])
    a3 = HypothesisResponse(**rec["a3"])
    a4 = HypothesisResponse(**rec["a4"])
    supports = {h: HypothesisSupport(**d) for h, d in rec["hypothesis_support"].items()}
    lines = {h: render_hypothesis_support(s) for h, s in supports.items()}
    nb = FeatureNeighbourhood(**rec["escalation_neighbourhood"])
    s = rec["evidence_signals"]
    bundle = es.EvidenceBundle(
        values={f: float(blind["features"][f]) for f in es.FEATURES},
        percentiles=s["percentiles"], direction=s["direction"], neighbours=s["neighbours"],
        count_all=s["count_all"], identical=s["identical"], count_excl=s["count_excl"],
        history=s["history"], history_note=s["history_note"],
    )
    return {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "supports": supports, "lines": lines, "nb": nb,
            "block": es.render_evidence_block(bundle)}


def _calls(meta) -> int:
    return 0 if meta.cached else 1 + meta.schema_retries


def done_keys() -> set:
    if not OUT.exists():
        return set()
    return {(json.loads(l)["record_id"], json.loads(l)["arm"]) for l in open(OUT, encoding="utf-8") if l.strip()}


def finish(resp, arm: str, neighbourhood, record_id: str, batch: str, meta, da_info, clamped: bool, ung_fired: bool) -> dict:
    bp = float(resp.benign_plausibility)
    out = {
        "record_id": record_id, "batch": batch, "arm": arm,
        "benign_plausibility": bp,
        "verdict": ("attack" if bp < THRESHOLD else "benign") if arm != "twosided" else None,
        "credited_hypothesis_id": resp.credited_hypothesis_id,
        "cited_claim_ids": list(resp.cited_claim_ids),
        "rationale": resp.rationale,
        "zero_match_cap_fired": clamped, "ungrounded_cap_fired": ung_fired,
        "calls": _calls(meta),
    }
    if arm == "twosided":
        ap = float(resp.attack_plausibility)
        out["attack_plausibility"] = ap
        out["verdict"] = derive_two_sided_verdict(bp, ap)
        out["state"] = two_sided_state(bp, ap)
    if da_info is not None:
        out["advocate_hypotheses"] = da_info
    return out


def run_record(blind: dict, rec: dict, g, done: set, tally: dict) -> List[dict]:
    rid = blind["record_id"]
    record = blind_record_to_escalation_record(blind)
    up = reconstruct(rec, blind)
    written = []

    def emit(line: dict):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
        written.append(line)
        tally["calls"][line["arm"]] = tally["calls"].get(line["arm"], 0) + line["calls"]
        tally["completed"][line["arm"]] = tally["completed"].get(line["arm"], 0) + 1

    def fail(arm: str, exc: Exception):
        tally["failed"].append({"record_id": rid, "arm": arm, "error": str(exc)[:400]})

    common = dict(
        a1=up["a1"], a2=up["a2"], a3=up["a3"], a4=up["a4"], supports=up["supports"], lines=up["lines"],
        block=up["block"], nb=up["nb"],
    )

    if (rid, "adversarial") not in done:
        try:
            resp, meta, clamped = run_arm("adversarial", record, common["a1"], common["a2"], common["a3"], common["a4"],
                                          common["lines"], common["supports"], common["block"], use_cache=False)
            resp, ung = apply_ungrounded_neighbourhood_cap(resp, common["nb"])
            emit(finish(resp, "adversarial", common["nb"], rid, blind["_batch"], meta, None, clamped, ung))
        except base.DailyQuotaExceeded:
            raise
        except base.SchemaValidationFailed as exc:
            fail("adversarial", exc)

    if (rid, "devil") not in done:
        try:
            da_resp, da_meta = da_advocate.run(
                record, common["a1"], g.empirical_grounding_block, g.available_features,
                g.known_reference_features, common["block"], use_cache=False,
            )
            da_sup = evaluate_all(list(da_resp.hypotheses), observed_tier1_values(record, prefixed=True), g.benign_df)
            da_sup = patch_hypothesis_support(da_sup, common["nb"])
            da_lines = {h: render_hypothesis_support(s) for h, s in da_sup.items()}
            da_info = [{"id": h.hypothesis_id, "description": h.description, "prediction": h.prediction,
                        "profile": [p.model_dump() for p in h.predicted_feature_profile]} for h in da_resp.hypotheses]
            resp, meta, clamped = run_arm("devil", record, common["a1"], common["a2"], common["a3"], common["a4"],
                                          common["lines"], common["supports"], common["block"],
                                          da_hypotheses=da_resp.hypotheses, da_support_lines=da_lines, use_cache=False)
            resp, ung = apply_ungrounded_neighbourhood_cap(resp, common["nb"])
            line = finish(resp, "devil", common["nb"], rid, blind["_batch"], meta, da_info, clamped, ung)
            line["calls"] += _calls(da_meta)
            emit(line)
        except base.DailyQuotaExceeded:
            raise
        except base.SchemaValidationFailed as exc:
            fail("devil", exc)

    if (rid, "twosided") not in done:
        try:
            resp, meta, clamped = run_arm("twosided", record, common["a1"], common["a2"], common["a3"], common["a4"],
                                          common["lines"], common["supports"], common["block"], use_cache=False)
            resp, ung = apply_ungrounded_neighbourhood_cap(resp, common["nb"])
            emit(finish(resp, "twosided", common["nb"], rid, blind["_batch"], meta, None, clamped, ung))
        except base.DailyQuotaExceeded:
            raise
        except base.SchemaValidationFailed as exc:
            fail("twosided", exc)
    return written


def main() -> int:
    if not load_api_key_from_dotenv():
        raise SystemExit("GEMINI_API_KEY not available")
    v3 = {}
    for line in open(V3_RUN, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            v3[r["record_id"]] = r
    blinds = {b["record_id"]: b for b in load_45()}
    ids = [rid for rid in blinds if rid in v3]
    unscored = [rid for rid in blinds if rid not in v3]
    done = done_keys()
    tally = {"arms": list(ARMS), "rule_arm_c": ARM_C_RULE, "records_scored_upstream": len(ids),
             "unscored_no_v3_upstream": unscored, "calls": {}, "completed": {}, "failed": [],
             "stopped_on_quota": False}
    stopped = False
    try:
        for rid in ids:
            blind = blinds[rid]
            g = pipeline.compute_grounding(blind_record_to_escalation_record(blind))
            run_record(blind, v3[rid], g, done, tally)
            print(f"  done {rid} ({blind['_batch']})", flush=True)
    except base.DailyQuotaExceeded as exc:
        print(f"[STOPPED] daily quota: {exc}", flush=True)
        tally["stopped_on_quota"] = True
        stopped = True
    TALLY.write_text(json.dumps(tally, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: tally[k] for k in ("calls", "completed", "stopped_on_quota")}, indent=2))
    print(f"failed: {len(tally['failed'])}")
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
