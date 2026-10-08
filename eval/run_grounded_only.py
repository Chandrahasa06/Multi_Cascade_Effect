"""Eighth attempt: A5 as a security judge, only on flows the count calls benign.

Records: every Arm M record (results/agent_run_arm_m.jsonl) with leave-one-out count >= 30
(MIN_NEIGHBOURHOOD_SIZE). Reference for percentiles and direction: Monday BENIGN, as in
Arm M. A1-A4 are the recorded v2 outputs. Only A5 is called (a5_verdict_security_v1).

Fixed before any call (not changed after seeing scores):
  - no cap of any kind is applied; the score is A5's own, and the runner asserts no clamp
  - A5 sees no count: not the neighbourhood count, no "within 2x" closeness, no hypothesis
    profile counts. build_context never receives a count; the prompt is checked for count
    phrasing, and for the record's count value outside feature/claim/history numbers
  - time of day: CICIDS2017 CSVs record afternoon times in 12-hour form without AM/PM (no
    hour 13-17 exists in any file; afternoon-only files sit at 01-05). Hours below 08 get
    +12 h. The capture ran in working hours. Applied to every record identically
  - source history: all flows from the same source in the 30 minutes before the flow
    (6 x 5-minute buckets), read from the pool with no label. A Monday-only benign index
    would show every Tuesday-onward source as silent; a benign filter on the evaluation
    day would read labels and hide a source's own earlier flows
  - not shown: source IP, weekday, date (a model that memorised the published CICIDS2017
    schedule could score from them); protocol (not recorded in the blind features or pool)
  - verdict: attack iff benign_plausibility < 0.30, unchanged

Run: python -m eval.run_grounded_only
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents import base, evidence_signals as es
from agents import reference_arms as ra
from agents.a5_verdict import _validate_cites_something
from agents.escalation_grounding import MIN_NEIGHBOURHOOD_SIZE
from agents.prompts import a5_verdict_security_v1 as prompt_module
from agents.schema import A5Response, ClaimsResponse, HypothesisResponse
from agents.validators import validate_a5_addresses_all_contradictions, validate_a5_credits_a_real_hypothesis
from agents.verification import counts_by_agent, match_claims
from eval.run_blind_pipeline import blind_record_to_escalation_record
from eval.run_blind_pipeline_45_v2 import load_45, load_api_key_from_dotenv

ARM_M = Path("results/agent_run_arm_m.jsonl")
V2_RUN = Path("results/agent_run_45_v2.jsonl")
OUT = Path("results/agent_grounded_only.jsonl")
TALLY = Path("results/agent_grounded_only_tally.json")
THRESHOLD = 0.30
DRY_PROMPT = "C:/Users/chand/AppData/Local/Temp/claude/c--BTP-Multi-Cascade-Effect/11e205e0-629c-450f-8eb6-a10c87c2d022/scratchpad/grounded_only_prompt.txt"
HOUR_US = 3_600_000_000
BUCKET_US = 5 * 60 * 1_000_000
N_BUCKETS = 6
AFTERNOON_HOUR_FIX = 8  # recorded hour below this -> +12 h

#: phrasing that would carry a count into the prompt (checked case-insensitively)
FORBIDDEN_PHRASES = (
    "within 2x", "within [0.5x", "match this predicted profile", "benign flow(s) match",
    "benign flows match", "neighbourhood", "neighborhood", "empirically unsupported",
    "close_to_observed", "matching_profile", "benign flows in the reference window",
    "cicids", "nearest benign",
)


# ------------------------------------------------------------------ record selection
def select_records(arm_m_rows: List[dict]) -> List[dict]:
    """Arm M records the count calls benign: leave-one-out count >= the cutoff."""
    return [r for r in arm_m_rows if r["count_loo"] >= MIN_NEIGHBOURHOOD_SIZE]


# ------------------------------------------------------------------ time and history
def corrected_ts(ts_us: int) -> int:
    hour = datetime.fromtimestamp(ts_us / 1e6, tz=timezone.utc).hour
    return ts_us + 12 * HOUR_US if hour < AFTERNOON_HOUR_FIX else ts_us


def time_of_day(ts_us: int) -> str:
    return datetime.fromtimestamp(corrected_ts(ts_us) / 1e6, tz=timezone.utc).strftime("%H:%M")


class SourceIndex:
    """All pool flows per source IP, sorted by corrected time. Reads no label column."""

    def __init__(self, pool: pd.DataFrame):
        cols = ["Source IP", "first_ts", "Total Fwd Packets", "Total Backward Packets", "Destination Port"]
        df = pool[cols].copy()
        df["t"] = np.array([corrected_ts(int(t)) for t in df["first_ts"].to_numpy()], dtype=np.int64)
        df = df.sort_values("t")
        self.idx = {}
        for ip, g in df.groupby("Source IP", sort=False):
            self.idx[str(ip)] = (
                g["t"].to_numpy(dtype=np.int64),
                (g["Total Fwd Packets"] + g["Total Backward Packets"]).to_numpy(dtype=np.float64),
                g["Destination Port"].to_numpy(dtype=np.int64),
            )

    def history(self, ip: str, t0: int) -> List[dict]:
        """Buckets strictly before t0 (corrected time). Bucket 1 is the most recent."""
        t0 = corrected_ts(t0)
        times, pk, ports = self.idx.get(ip, (np.empty(0, np.int64), np.empty(0), np.empty(0, np.int64)))
        out = []
        for k in range(1, N_BUCKETS + 1):
            lo, hi = t0 - k * BUCKET_US, t0 - (k - 1) * BUCKET_US
            i0, i1 = int(np.searchsorted(times, lo, "left")), int(np.searchsorted(times, hi, "left"))
            out.append({"bucket": k, "flows": i1 - i0,
                        "distinct_dst_ports": int(len(np.unique(ports[i0:i1]))) if i1 > i0 else 0,
                        "top_dst_ports": [int(p) for p, _ in sorted(
                            zip(*np.unique(ports[i0:i1], return_counts=True)), key=lambda x: -x[1])[:3]] if i1 > i0 else [],
                        "packets": float(pk[i0:i1].sum()) if i1 > i0 else 0.0})
        return out


# ------------------------------------------------------------------ the context A5 reads
def build_context(features: Dict[str, float], tod: Optional[str], history: Optional[List[dict]],
                  percentiles: Dict[str, float], direction: Dict[str, str]) -> str:
    """Everything A5 reads besides features and A1-A4. No count parameter exists."""
    port = features.get("Destination Port")
    L = [f"Destination port: {int(port) if port is not None else 'unknown'}.",
         "Protocol: not recorded for this flow.",
         f"Time of day the flow started: {tod if tod else 'unavailable'} (minute resolution)."]
    L.append("\nThis source's behaviour in the 30 minutes before this flow (all its flows, 5-minute buckets, most recent first):")
    if history is None:
        L.append("  unavailable (the source of this flow could not be identified uniquely)")
    else:
        for h in history:
            start, end = -5 * h["bucket"], -5 * (h["bucket"] - 1)
            top = ", ".join(str(p) for p in h["top_dst_ports"]) or "none"
            L.append(f"  {start} to {end} min: {h['flows']} flows, {h['distinct_dst_ports']} distinct destination ports "
                     f"(most frequent: {top}), {h['packets']:g} packets")
    L.append("\nWhere this flow's key features sit in this network's ordinary traffic "
             "(percentile 0 = lowest ordinary value, 100 = highest), and the direction:")
    for f in es.FEATURES:
        L.append(f"  {f}: percentile {percentiles[f]:.2f}, {direction[f]}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ the count-absence check
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)(?!\.\d)")  # a trailing full stop is allowed


def check_no_count(prompt: str, count: int, allowed_numbers: set) -> List[str]:
    """Raises if count phrasing appears, or if the record's count appears as a number
    outside the set of numbers legitimately present (feature values, A1-A4 text, history).
    Returns where the count value collided with a legitimate number (for the report)."""
    low = prompt.lower()
    hits = [p for p in FORBIDDEN_PHRASES if p in low]
    if hits:
        raise AssertionError(f"count phrasing in prompt: {hits}")
    nums = {float(n) for n in _NUM.findall(prompt)}
    if float(count) in nums and float(count) not in allowed_numbers:
        raise AssertionError(f"count {count} appears in the prompt outside features, claims and history")
    return ["value collides with a feature, claim or history number"] if float(count) in nums else []


def numbers_in(text: str) -> set:
    return {float(n) for n in _NUM.findall(text)}


# ------------------------------------------------------------------ main
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build and check every prompt; no model calls")
    dry = ap.parse_args().dry_run
    if not dry and not load_api_key_from_dotenv():
        raise SystemExit("GEMINI_API_KEY not available")
    arm_m = [json.loads(l) for l in open(ARM_M, encoding="utf-8") if l.strip()]
    selected = select_records(arm_m)
    assert all(r["count_loo"] >= MIN_NEIGHBOURHOOD_SIZE for r in selected)
    v2 = {json.loads(l)["record_id"]: json.loads(l) for l in open(V2_RUN, encoding="utf-8") if l.strip()}
    blinds = {b["record_id"]: b for b in load_45()}
    keys = list(next(iter(blinds.values()))["features"].keys())
    pool = es.load_pool_slim(extra_columns=list(ra.EXTRA_POOL_COLUMNS) + keys)
    match_cols = [c for c in keys if c in pool.columns]
    window = ra.build_window(pool, (0,), "monday")
    for r in selected:
        ra.assert_outside(window, r["weekdays"])
    es_cols = list(dict.fromkeys(list(es.FEATURES) + ["Source IP", "first_ts", "Total Fwd Packets",
                                                       "Total Backward Packets", "Destination Port", "Label"]))
    ref = es.build_reference(window.frame[es_cols])
    src_index = SourceIndex(pool)

    if OUT.exists() and not dry:
        OUT.unlink()
    tally = {"selected": len(selected), "calls": 0, "schema_retries": 0, "failed": [], "stopped_on_quota": False,
             "source_unavailable": []}
    for r in selected:
        rid = r["record_id"]
        b = blinds[rid]
        rec_v2 = v2[rid]
        a1, a2 = ClaimsResponse(**rec_v2["a1"]), ClaimsResponse(**rec_v2["a2"])
        a3, a4 = HypothesisResponse(**rec_v2["a3"]), HypothesisResponse(**rec_v2["a4"])
        source, status = es.locate_source(pool, b["features"], match_cols)
        if source is None:
            tally["source_unavailable"].append(rid)
        bundle = es.bundle(ref, b["features"], None)  # percentiles and direction only are used below
        history = src_index.history(source[0], source[1]) if source else None
        tod = time_of_day(source[1]) if source else None
        context = build_context(b["features"], tod, history, bundle.percentiles, bundle.direction)

        record = blind_record_to_escalation_record(b)
        chain_claims = [*a1.claims, *a2.claims, *a3.claims]
        chain_hyps = list(a3.hypotheses)
        all_hyps = [*chain_hyps, *a4.hypotheses]
        verification = counts_by_agent(match_claims(chain_claims, a4.claims))
        required = set()
        for h in all_hyps:
            required.update(h.contradicting_claim_ids)
        prompt = prompt_module.build_prompt(record, chain_claims, chain_hyps, a4.claims, a4.hypotheses,
                                            verification, required, context)
        upstream_text = prompt[prompt.index("Full feature set for this flow:"):]
        allowed = numbers_in(upstream_text)  # features, history, percentiles, A1-A4 text
        collisions = check_no_count(prompt, r["count_loo"], allowed)

        if dry:
            tally.setdefault("dry", []).append({"record_id": rid, "chars": len(prompt), "collisions": collisions, "tod": tod})
            if len(tally["dry"]) == 1:
                Path(DRY_PROMPT).write_text(prompt, encoding="utf-8")
            continue
        known_h = {h.hypothesis_id for h in all_hyps}
        known_c = {c.claim_id for c in chain_claims} | {c.claim_id for c in a4.claims}

        def validate(resp) -> None:
            _validate_cites_something(resp, known_c)
            validate_a5_addresses_all_contradictions(resp, required)
            validate_a5_credits_a_real_hypothesis(resp, known_h)

        try:
            resp, meta = base.call_structured(
                record_id=rid, agent="a5", prompt_version=prompt_module.PROMPT_VERSION, prompt=prompt,
                response_schema=A5Response, model=base.DEFAULT_MODEL, temperature=0.7, run_index=0,
                use_cache=False, extra_validate=validate,
            )
        except base.DailyQuotaExceeded as exc:
            tally["stopped_on_quota"] = True
            print(f"[STOPPED] {exc}", flush=True)
            break
        except base.SchemaValidationFailed as exc:
            tally["failed"].append({"record_id": rid, "error": str(exc)[:300]})
            continue
        tally["calls"] += 0 if meta.cached else 1 + meta.schema_retries
        tally["schema_retries"] += meta.schema_retries
        bp = float(resp.benign_plausibility)
        # no cap is applied; the stored score is exactly what A5 returned
        assert bp == float(resp.benign_plausibility)
        line = {
            "record_id": rid, "batch": b["_batch"], "weekdays": r["weekdays"], "count_loo": r["count_loo"],
            "arm_m_score": r["benign_plausibility"], "benign_plausibility": bp, "cap_applied": False,
            "verdict": "attack" if bp < THRESHOLD else "benign",
            "credited_hypothesis_id": resp.credited_hypothesis_id, "cited_claim_ids": list(resp.cited_claim_ids),
            "rationale": resp.rationale, "time_of_day": tod, "source_status": status,
            "destination_port": b["features"].get("Destination Port"), "history": history,
            "count_value_collisions": collisions, "prompt_chars": len(prompt),
        }
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
        print(f"  {rid[:10]} bp={bp:.2f}", flush=True)
    if dry:
        print(json.dumps(tally, indent=2, default=str))
        return 0
    TALLY.write_text(json.dumps(tally, indent=2, default=str), encoding="utf-8")
    print(json.dumps(tally, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
