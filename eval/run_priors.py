"""Ninth attempt: prompt-encoded priors. All five agents are new (*_priors_v1); every agent
sees only the flow's own properties (features, destination port, start time) and a written
description of ordinary traffic (agents/prompts/priors_text_v1.py). No grounding, no
reference data, no recorded A1-A4 output, no cap.

Fixed before any call:
  - records: all 45 blind records (results/agent_input_*_blind.jsonl); labels are not read
  - start time: the flow's own timestamp, found by matching its features to its pool row
    (Source IP and time only; no label). Afternoon hours recorded in 12-hour form are shifted
    by 12 (eval/run_grounded_only.time_of_day). Date, weekday and source IP are not shown
  - protocol: not recorded in the blind features, shown as "not recorded"
  - verdict: attack iff A5's benign_plausibility < 0.30; no clamp of any kind
  - cache on, for resumption only: the prompt versions are new, so first calls are fresh

Checks on every prompt before it is sent (eval.run_grounded_only's phrase list and its
number pattern, with the trailing-full-stop fix):
  - the whole prompt must contain none of ALWAYS_FORBIDDEN
  - the builder's own text (the prompt rebuilt with empty upstream blocks) must also contain
    none of BUILDER_FORBIDDEN, and none of this record's stored neighbourhood counts unless
    that number is one of the flow's own values or appears in the fixed prior text

Run: python -m eval.run_priors [--dry-run]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Dict, List, Optional

from agents import base, evidence_signals as es
from agents import priors_pipeline as pp
from agents import reference_arms as ra
from agents.prompts import a1_priors_v1, a2_priors_v1, a3_priors_v1, a4_priors_v1, a5_priors_v1
from agents.prompts.priors_render_v1 import flow_block, profile_feature_names
from agents.prompts.priors_text_v1 import PRIORS_TEXT
from agents.schema import ClaimsResponse, HypothesisResponse
from eval.run_blind_pipeline_45_v2 import load_45, load_api_key_from_dotenv
from eval.run_grounded_only import FORBIDDEN_PHRASES, _NUM, time_of_day

OUT = Path("results/agent_priors.jsonl")
TALLY = Path("results/agent_priors_tally.json")
PROMPT_DIR = Path("results/agent_priors_prompts")
THRESHOLD = 0.30
COUNT_SOURCES = (Path("results/agent_run_arm_a.jsonl"), Path("results/agent_run_arm_m.jsonl"))
PROMPT_MODULES = (a1_priors_v1, a2_priors_v1, a3_priors_v1, a4_priors_v1, a5_priors_v1)

ALWAYS_FORBIDDEN = tuple(FORBIDDEN_PHRASES) + (
    "greater_than_typical", "less_than_typical", "calibration context", "typical-range threshold",
    "fitted threshold", "ratio to threshold", "trigger_", "benign reference",
)
#: words that are fine coming from a model's own reasoning but must never be supplied by us
BUILDER_FORBIDDEN = ("percentile", "median", "nearest", "history", "previous 30 minutes",
                     "reference window", "within [", "of which")


def numbers_in(text: str) -> set:
    return {float(n) for n in _NUM.findall(text)}


def known_counts() -> Dict[str, set]:
    """This record's stored neighbourhood counts (all-days and Monday), for the number check
    only. They never enter a prompt."""
    out: Dict[str, set] = {}
    for p in COUNT_SOURCES:
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["record_id"], set()).update({float(r["count_all"]), float(r["count_loo"])})
    return out


def check_prompt(full_prompt: str, builder_prompt: str, counts: set, flow_values: set) -> None:
    low = full_prompt.lower()
    hits = [p for p in ALWAYS_FORBIDDEN if p in low]
    if hits:
        raise AssertionError(f"forbidden content in prompt: {hits}")
    blow = builder_prompt.lower()
    hits = [p for p in BUILDER_FORBIDDEN if p in blow]
    if hits:
        raise AssertionError(f"forbidden content in the builder's own text: {hits}")
    allowed = flow_values | TEMPLATE_NUMBERS
    leaked = (numbers_in(builder_prompt) & counts) - allowed
    if leaked:
        raise AssertionError(f"stored neighbourhood count(s) {sorted(leaked)} appear in the builder's text")


def _template_numbers() -> set:
    """Numbers in the fixed, record-independent text: every prompt built with an empty flow.
    A stored count that equals one of these cannot be told apart from the template; such
    collisions are reported by the run, not treated as leaks."""
    empty = flow_block({}, None)
    return set().union(*(numbers_in(p) for p in builder_prompts(empty, []).values()))


def builder_prompts(flow_text: str, profile_names: list) -> Dict[str, str]:
    """Each prompt rebuilt with empty upstream blocks: the text this code supplies."""
    return {
        "a1": a1_priors_v1.build_prompt(flow_text),
        "a2": a2_priors_v1.build_prompt(flow_text, []),
        "a3": a3_priors_v1.build_prompt(flow_text, [], [], profile_names),
        "a4": a4_priors_v1.build_prompt(flow_text, profile_names),
        "a5": a5_priors_v1.build_prompt(flow_text, [], [], [], []),
    }


TEMPLATE_NUMBERS = _template_numbers()


def meta_dict(meta) -> dict:
    return {"cached": bool(meta.cached), "schema_retries": int(meta.schema_retries),
            "prompt_version": getattr(meta, "prompt_version", None)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build and check the A1 and A4 prompts only; no calls")
    dry = ap.parse_args().dry_run
    if not dry and not load_api_key_from_dotenv():
        raise SystemExit("GEMINI_API_KEY not available")

    blinds = load_45()
    import os
    if os.environ.get("PRIORS_LIMIT"):  # smoke-test limit; unset for the real run
        blinds = blinds[: int(os.environ["PRIORS_LIMIT"])]
    keys = list(blinds[0]["features"].keys())
    pool = es.load_pool_slim(extra_columns=list(ra.EXTRA_POOL_COLUMNS) + keys)
    match_cols = [c for c in keys if c in pool.columns]
    counts = known_counts()
    done = set()
    if OUT.exists():
        done = {json.loads(l)["record_id"] for l in open(OUT, encoding="utf-8") if l.strip()}
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    tally = {"records": len(blinds), "already_done": len(done), "calls": 0, "cache_hits": 0,
             "schema_retries": 0, "failed": [], "stopped_on_quota": False}
    example_saved = (PROMPT_DIR / "a5.txt").exists()

    for b in blinds:
        rid = b["record_id"]
        if rid in done:
            continue
        feats = b["features"]
        src, _status = es.locate_source(pool, feats, match_cols)
        tod = time_of_day(src[1]) if src else None
        flow_text = flow_block(feats, tod)
        names = set(feats)
        profile = profile_feature_names(feats)
        fvals = {float(v) for v in feats.values()} | (numbers_in(tod) if tod else set())
        collisions = sorted(rcounts_all & TEMPLATE_NUMBERS) if (rcounts_all := counts.get(rid, set())) else []
        rcounts = counts.get(rid, set())
        for agent, bp in builder_prompts(flow_text, profile).items():
            check_prompt(bp, bp, rcounts, fvals)
        if dry:
            continue
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool_x:
                fut4 = pool_x.submit(pp.run_a4, rid, flow_text, names, set(profile))
                (a1, m1), p1 = pp.run_a1(rid, flow_text, names)
                check_prompt(p1, builder_prompts(flow_text, profile)["a1"], rcounts, fvals)
                (a2, m2), p2 = pp.run_a2(rid, flow_text, names, a1)
                check_prompt(p2, builder_prompts(flow_text, profile)["a2"], rcounts, fvals)
                (a3, m3), p3 = pp.run_a3(rid, flow_text, names, set(profile), a1, a2)
                check_prompt(p3, builder_prompts(flow_text, profile)["a3"], rcounts, fvals)
                (a4, m4), p4 = fut4.result()
                check_prompt(p4, builder_prompts(flow_text, profile)["a4"], rcounts, fvals)
            (a5, m5), p5 = pp.run_a5(rid, flow_text, a1, a2, a3, a4)
            check_prompt(p5, builder_prompts(flow_text, profile)["a5"], rcounts, fvals)
        except base.DailyQuotaExceeded as exc:
            tally["stopped_on_quota"] = True
            print(f"[STOPPED] {exc}", flush=True)
            break
        except base.SchemaValidationFailed as exc:
            tally["failed"].append({"record_id": rid, "agent": exc.agent, "reasons": list(exc.retry_reasons)[:3]})
            tally["calls"] += exc.attempts
            print(f"[FAILED] {rid[:10]} {exc.agent}", flush=True)
            continue

        metas = {"a1": m1, "a2": m2, "a3": m3, "a4": m4, "a5": m5}
        for m in metas.values():
            if m.cached:
                tally["cache_hits"] += 1
            else:
                tally["calls"] += 1 + m.schema_retries
            tally["schema_retries"] += m.schema_retries
        bp = float(a5.benign_plausibility)
        line = {
            "record_id": rid, "batch": b["_batch"], "time_of_day": tod,
            "destination_port": feats.get("Destination Port"),
            "benign_plausibility": bp, "cap_applied": False,
            "verdict": "attack" if bp < THRESHOLD else "benign",
            "a1": a1.model_dump(), "a2": a2.model_dump(), "a3": a3.model_dump(),
            "a4": a4.model_dump(), "a5": a5.model_dump(),
            "stage_meta": {k: meta_dict(m) for k, m in metas.items()},
            "count_template_collisions": collisions,
            "prompt_versions": {f"a{i+1}": m.PROMPT_VERSION for i, m in enumerate(PROMPT_MODULES)},
        }
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
        if not example_saved:
            for name, text in zip(("a1", "a2", "a3", "a4", "a5"), (p1, p2, p3, p4, p5)):
                (PROMPT_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
            (PROMPT_DIR / "record_id.txt").write_text(rid, encoding="utf-8")
            example_saved = True
        print(f"  {rid[:10]} bp={bp:.2f}", flush=True)

    if dry:
        print(f"dry run: builder text of all five prompts passed for {len(blinds)} records")
        return 0
    TALLY.write_text(json.dumps(tally, indent=2, default=str), encoding="utf-8")
    print(json.dumps(tally, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
