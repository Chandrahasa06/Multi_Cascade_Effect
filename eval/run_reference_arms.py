"""Deployment-realistic arms. Only A5 is called: A1-A4 are the recorded v2 outputs, whose
grounding block is the Monday Tier-1 reference for every arm, so the upstream is identical
across arms. A5 gets the corrected prompt (a5_verdict_v7), the real count, and the
reference window's profile sentences.

Arms (fixed before running):
  A   reference = all benign days (current setup), Parts 1-3 applied. All 45 records.
  M   reference = Monday benign. Evaluate records from Tuesday onward only.
  MT  reference = Monday + Tuesday benign. Evaluate records from Wednesday onward only.

Every evaluated record's day is recovered from its matched pool row. A record whose day
is ambiguous is dropped and listed with the reason. The window assertion (agents/
reference_arms.assert_outside) fails the run if an evaluated record came from a reference day.

Run one arm at a time: python -m eval.run_reference_arms --arm M
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from agents import base, evidence_signals as es
from agents import reference_arms as ra
from agents.a5_verdict import _validate_cites_something
from agents.escalation_grounding import FeatureNeighbourhood, apply_ungrounded_neighbourhood_cap
from agents.grounding import apply_empirical_plausibility_cap
from agents.prompts import a5_verdict_v7
from agents.schema import A5Response, ClaimsResponse, HypothesisResponse
from agents.validators import validate_a5_addresses_all_contradictions, validate_a5_credits_a_real_hypothesis
from agents.verification import counts_by_agent, match_claims
from eval.run_blind_pipeline import blind_record_to_escalation_record
from eval.run_blind_pipeline_45_v2 import load_45, load_api_key_from_dotenv

V2_RUN = Path("results/agent_run_45_v2.jsonl")
THRESHOLD = 0.30
ARMS = {
    "a": {"window": (0, 1, 2, 3, 4), "label": "all-days", "eval_min_weekday": 0},
    "m": {"window": (0,), "label": "monday", "eval_min_weekday": 1},
    "mt": {"window": (0, 1), "label": "monday+tuesday", "eval_min_weekday": 2},
}
# columns the evidence module reads from a window frame
ES_COLUMNS = list(dict.fromkeys(list(ra.ESCALATION_FEATURES) + ["Source IP", "first_ts", "Total Fwd Packets",
                                                                  "Total Backward Packets", "Destination Port", "Label"]))


def locate(pool: pd.DataFrame, features: Dict[str, float], match_cols: List[str]):
    """(source, weekday_set, status). Source is (ip, t) when unique; weekdays are the
    weekdays of every matching row, so an ambiguous day is visible."""
    x = [float(features[c]) for c in match_cols]
    import numpy as np
    m = np.all(pool[match_cols].to_numpy(dtype=np.float64) == np.array(x), axis=1)
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        return None, set(), "no match"
    weekdays = set(int(w) for w in pool["weekday_idx"].to_numpy()[idx])
    src, status = es.locate_source(pool, features, match_cols)
    return src, weekdays, status


def run_a5(record, a1, a2, a3, a4, lines: Dict[str, str], supports, evidence_block: str, *,
           model: str = base.DEFAULT_MODEL, temperature: float = 0.7, run_index: int = 0):
    chain_claims = [*a1.claims, *a2.claims, *a3.claims]
    chain_hyps = list(a3.hypotheses)
    all_hyps = [*chain_hyps, *a4.hypotheses]
    verification = counts_by_agent(match_claims(chain_claims, a4.claims))
    required = set()
    for h in all_hyps:
        required.update(h.contradicting_claim_ids)
    known_h = {h.hypothesis_id for h in all_hyps}
    known_c = {c.claim_id for c in chain_claims} | {c.claim_id for c in a4.claims}
    prompt = a5_verdict_v7.build_prompt(
        record, chain_claims, chain_hyps, a4.claims, a4.hypotheses, verification, required, lines, evidence_block,
    )

    def validate(r) -> None:
        _validate_cites_something(r, known_c)
        validate_a5_addresses_all_contradictions(r, required)
        validate_a5_credits_a_real_hypothesis(r, known_h)

    resp, meta = base.call_structured(
        record_id=record.flow_id, agent="a5", prompt_version=a5_verdict_v7.PROMPT_VERSION, prompt=prompt,
        response_schema=A5Response, model=model, temperature=temperature, run_index=run_index,
        use_cache=False, extra_validate=validate,
    )
    return resp, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = ap.parse_args()
    spec = ARMS[args.arm]
    if not load_api_key_from_dotenv():
        raise SystemExit("GEMINI_API_KEY not available")

    v2 = {json.loads(l)["record_id"]: json.loads(l) for l in open(V2_RUN, encoding="utf-8") if l.strip()}
    blinds = {b["record_id"]: b for b in load_45()}
    assert len(v2) == 45 and set(v2) == set(blinds)

    match_cols = None
    pool = es.load_pool_slim(extra_columns=list(ra.EXTRA_POOL_COLUMNS) + list(next(iter(blinds.values()))["features"].keys()))
    match_cols = [c for c in next(iter(blinds.values()))["features"] if c in pool.columns]

    # day of every record, from its matched pool rows
    day_info = {}
    for rid, b in blinds.items():
        src, weekdays, status = locate(pool, b["features"], match_cols)
        day_info[rid] = {"source": src, "weekdays": sorted(weekdays), "status": status}

    # eligibility for this arm
    eligible, dropped = [], []
    for rid in sorted(blinds):
        wd = day_info[rid]["weekdays"]
        if spec["eval_min_weekday"] == 0:
            eligible.append(rid)
            continue
        if not wd:
            dropped.append((rid, "no pool match for the day"))
        elif len(wd) > 1:
            dropped.append((rid, f"ambiguous day: matching rows span weekdays {wd}"))
        elif wd[0] < spec["eval_min_weekday"]:
            dropped.append((rid, f"inside the reference window (weekday {wd[0]})"))
        else:
            eligible.append(rid)

    window = ra.build_window(pool, spec["window"], spec["label"])
    # the assertion: no evaluated record comes from a reference day
    for rid in eligible:
        if spec["eval_min_weekday"] > 0:
            ra.assert_outside(window, day_info[rid]["weekdays"])

    ref = es.build_reference(window.frame[ES_COLUMNS])
    out_path = Path(f"results/agent_run_arm_{args.arm}.jsonl")
    if out_path.exists():
        out_path.unlink()
    tally = {"arm": args.arm, "window_days": list(spec["window"]), "window_rows": window.size,
             "records_total": len(blinds), "eligible": len(eligible), "dropped": dropped,
             "calls": 0, "failed": [], "stopped_on_quota": False}

    for rid in eligible:
        b = blinds[rid]
        record = blind_record_to_escalation_record(b)
        rec = v2[rid]  # a1..a4 from v2 (Monday grounding only); supports and count recomputed here
        up = {"a1": ClaimsResponse(**rec["a1"]), "a2": ClaimsResponse(**rec["a2"]),
              "a3": HypothesisResponse(**rec["a3"]), "a4": HypothesisResponse(**rec["a4"])}
        nb = ra.neighbourhood(window, b["features"])
        supports, lines = ra.hypothesis_support(
            window, b["features"], list(up["a3"].hypotheses) + list(up["a4"].hypotheses), nb,
        )
        nb_sentence = ra.sentence_neighbourhood(nb)
        bundle = es.bundle(ref, b["features"], day_info[rid]["source"])
        evidence_text = nb_sentence + "\n" + es.render_evidence_block(bundle)
        try:
            resp, meta = run_a5(record, up["a1"], up["a2"], up["a3"], up["a4"], lines, supports, evidence_text)
        except base.DailyQuotaExceeded as exc:
            print(f"[STOPPED] {exc}", flush=True)
            tally["stopped_on_quota"] = True
            break
        except base.SchemaValidationFailed as exc:
            tally["failed"].append({"record_id": rid, "error": str(exc)[:300]})
            continue
        tally["calls"] += 0 if meta.cached else 1 + meta.schema_retries

        bp_raw = float(resp.benign_plausibility)
        resp, zero_fired = apply_empirical_plausibility_cap(resp, supports)
        nb_obj = FeatureNeighbourhood(
            features_used=list(ra.ESCALATION_FEATURES), band_description="reference-window band count (leave-one-out)",
            neighbourhood_size=nb["count_loo"], benign_population_size=nb["population_R"],
            ungrounded=nb["ungrounded"], min_required=ra.MIN_NEIGHBOURHOOD_SIZE,
        )
        resp, ung_fired = apply_ungrounded_neighbourhood_cap(resp, nb_obj)
        bp = float(resp.benign_plausibility)
        line = {
            "record_id": rid, "batch": b["_batch"], "arm": args.arm,
            "weekdays": day_info[rid]["weekdays"], "window": window.label, "R": nb["population_R"],
            "count_all": nb["count_all"], "identical": nb["identical"], "count_loo": nb["count_loo"],
            "benign_plausibility_raw": bp_raw, "benign_plausibility": bp,
            "verdict": "attack" if bp < THRESHOLD else "benign",
            "count_only_verdict": "attack" if nb["count_loo"] < ra.MIN_NEIGHBOURHOOD_SIZE else "benign",
            "credited_hypothesis_id": resp.credited_hypothesis_id,
            "cited_claim_ids": list(resp.cited_claim_ids), "rationale": resp.rationale,
            "zero_support_cap_fired": bool(zero_fired), "ungrounded_cap_fired": bool(ung_fired),
            "profile_sentences": lines, "neighbourhood_sentence": nb_sentence,
            "profile_mapping": ra.describe_mapping([p.feature for h in list(up["a3"].hypotheses) + list(up["a4"].hypotheses) for p in h.predicted_feature_profile]),
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
        print(f"  {rid[:10]} bp={bp:.2f} count_loo={nb['count_loo']}", flush=True)

    Path(f"results/agent_run_arm_{args.arm}_tally.json").write_text(json.dumps(tally, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: tally[k] for k in ("window_rows", "eligible", "calls", "stopped_on_quota")}, indent=2))
    print("dropped:", tally["dropped"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
