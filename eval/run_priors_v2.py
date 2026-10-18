"""Tenth attempt: protocol-aware written priors (priors_text_v2), all five agents new
(*_priors_v2). Same blind protocol as eval/run_priors.py, with three differences fixed before
any call:
  - agents see the flow's own protocol (TCP/UDP), recovered from its own CSV row
    (results/record_protocols.json, built by eval/record_protocols.py; Label never read);
  - the 12 TCP flag columns are not shown (results/flag_column_diagnosis.md);
  - the explanation format asks for exactly three explanations.
No grounding, no reference data, no recorded agent output, no cap. Verdict: attack iff
benign_plausibility < 0.30.

Run: python -m eval.run_priors_v2 [--dry-run]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Dict

from agents import base, evidence_signals as es
from agents import priors_pipeline as pp
from agents import reference_arms as ra
from agents.prompts import a1_priors_v2, a2_priors_v2, a3_priors_v2, a4_priors_v2, a5_priors_v2
from agents.prompts.priors_render_v2 import FLAG_COLUMNS, flow_block, profile_feature_names, shown_features
from eval.run_blind_pipeline_45_v2 import load_45, load_api_key_from_dotenv
from eval.run_grounded_only import time_of_day
from eval.run_priors import ALWAYS_FORBIDDEN, BUILDER_FORBIDDEN, known_counts, meta_dict, numbers_in

OUT = Path("results/agent_priors_v2.jsonl")
TALLY = Path("results/agent_priors_v2_tally.json")
PROMPT_DIR = Path("results/agent_priors_v2_prompts")
PROTOCOLS = Path("results/record_protocols.json")
THRESHOLD = 0.30
MODULES = (a1_priors_v2, a2_priors_v2, a3_priors_v2, a4_priors_v2, a5_priors_v2)
BUILDER_FORBIDDEN_V2 = tuple(BUILDER_FORBIDDEN) + tuple(c.lower() for c in FLAG_COLUMNS) + ("flag count", "handshake completed")


def builder_prompts(flow_text: str, profile_names: list) -> Dict[str, str]:
    return {
        "a1": a1_priors_v2.build_prompt(flow_text),
        "a2": a2_priors_v2.build_prompt(flow_text, []),
        "a3": a3_priors_v2.build_prompt(flow_text, [], [], profile_names),
        "a4": a4_priors_v2.build_prompt(flow_text, profile_names),
        "a5": a5_priors_v2.build_prompt(flow_text, [], [], [], []),
    }


TEMPLATE_NUMBERS = set().union(*(numbers_in(p) for p in builder_prompts(flow_block({}, None, None), []).values()))


def check_prompt(full_prompt: str, builder_prompt: str, counts: set, flow_values: set) -> None:
    low = full_prompt.lower()
    hits = [p for p in ALWAYS_FORBIDDEN if p in low]
    if hits:
        raise AssertionError(f"forbidden content in prompt: {hits}")
    blow = builder_prompt.lower()
    hits = [p for p in BUILDER_FORBIDDEN_V2 if p in blow]
    if hits:
        raise AssertionError(f"forbidden content in the builder's own text: {hits}")
    leaked = (numbers_in(builder_prompt) & counts) - (flow_values | TEMPLATE_NUMBERS)
    if leaked:
        raise AssertionError(f"stored neighbourhood count(s) {sorted(leaked)} appear in the builder's text")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run
    if not dry and not load_api_key_from_dotenv():
        raise SystemExit("GEMINI_API_KEY not available")
    blinds = load_45()
    if os.environ.get("PRIORS_LIMIT"):
        blinds = blinds[: int(os.environ["PRIORS_LIMIT"])]
    protocols = json.loads(PROTOCOLS.read_text(encoding="utf-8"))
    keys = list(blinds[0]["features"].keys())
    pool = es.load_pool_slim(extra_columns=list(ra.EXTRA_POOL_COLUMNS) + keys)
    match_cols = [c for c in keys if c in pool.columns]
    counts = known_counts()
    done = {json.loads(l)["record_id"] for l in open(OUT, encoding="utf-8") if l.strip()} if OUT.exists() else set()
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    tally = {"records": len(blinds), "already_done": len(done), "calls": 0, "cache_hits": 0,
             "schema_retries": 0, "failed": [], "stopped_on_quota": False}
    example_saved = (PROMPT_DIR / "a5.txt").exists()

    for b in blinds:
        rid = b["record_id"]
        if rid in done:
            continue
        feats = b["features"]
        proto = protocols[rid]["protocol"]
        src, _status = es.locate_source(pool, feats, match_cols)
        tod = time_of_day(src[1]) if src else None
        flow_text = flow_block(feats, proto, tod)
        shown = shown_features(feats)
        names = set(shown)
        profile = profile_feature_names(feats)
        fvals = {float(v) for v in shown.values()} | (numbers_in(tod) if tod else set())
        rcounts = counts.get(rid, set())
        builders = builder_prompts(flow_text, profile)
        for text in builders.values():
            check_prompt(text, text, rcounts, fvals)
        if dry:
            continue
        kw = dict(prompts="v2")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as px:
                fut4 = px.submit(pp.run_a4, rid, flow_text, names, set(profile), **kw)
                (a1, m1), p1 = pp.run_a1(rid, flow_text, names, **kw)
                check_prompt(p1, builders["a1"], rcounts, fvals)
                (a2, m2), p2 = pp.run_a2(rid, flow_text, names, a1, **kw)
                check_prompt(p2, builders["a2"], rcounts, fvals)
                (a3, m3), p3 = pp.run_a3(rid, flow_text, names, set(profile), a1, a2, **kw)
                check_prompt(p3, builders["a3"], rcounts, fvals)
                (a4, m4), p4 = fut4.result()
                check_prompt(p4, builders["a4"], rcounts, fvals)
            (a5, m5), p5 = pp.run_a5(rid, flow_text, a1, a2, a3, a4, **kw)
            check_prompt(p5, builders["a5"], rcounts, fvals)
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
            "record_id": rid, "batch": b["_batch"], "protocol": proto, "time_of_day": tod,
            "destination_port": feats.get("Destination Port"),
            "benign_plausibility": bp, "cap_applied": False,
            "verdict": "attack" if bp < THRESHOLD else "benign",
            "a1": a1.model_dump(), "a2": a2.model_dump(), "a3": a3.model_dump(),
            "a4": a4.model_dump(), "a5": a5.model_dump(),
            "stage_meta": {k: meta_dict(m) for k, m in metas.items()},
            "prompt_versions": {f"a{i+1}": m.PROMPT_VERSION for i, m in enumerate(MODULES)},
        }
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
        if not example_saved:
            for name, text in zip(("a1", "a2", "a3", "a4", "a5"), (p1, p2, p3, p4, p5)):
                (PROMPT_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
            (PROMPT_DIR / "record_id.txt").write_text(rid, encoding="utf-8")
            example_saved = True
        print(f"  {rid[:10]} {proto} bp={bp:.2f}", flush=True)

    if dry:
        print(f"dry run: builder text of all five prompts passed for {len(blinds)} records")
        return 0
    TALLY.write_text(json.dumps(tally, indent=2, default=str), encoding="utf-8")
    print(json.dumps(tally, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
