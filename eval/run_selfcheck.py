"""Self-check run. Rules fixed in results/selfcheck_preregistered.md before any call.

A1-A4 outputs are cached per record in results/selfcheck_upstream/<record_id>.json on first
run, so later A5 iterations replay the chain without paying for it again.

Run: python -m eval.run_selfcheck [--dry-run]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from agents import base, evidence_signals as es, openrouter
from agents import holdout as ho
from agents import reference_arms as ra
from agents import selfcheck_core as core
from agents import selfcheck_pipeline as sp
from agents.prompts import selfcheck_prompts_v1 as P
from agents.prompts.priors_render_v2 import FLAG_COLUMNS, shown_features
from agents.schema import SCBehaviourResponse, SCBlindResponse, SCClaimsResponse, SCHypothesisResponse
from eval.run_blind_pipeline_45_v2 import load_45, load_api_key_from_dotenv
from eval.run_grounded_only import time_of_day
from eval.run_priors import ALWAYS_FORBIDDEN, BUILDER_FORBIDDEN, known_counts, numbers_in

OUT = Path("results/agent_selfcheck.jsonl")
TALLY = Path("results/agent_selfcheck_tally.json")
UPSTREAM = Path("results/selfcheck_upstream")
PROMPT_DIR = Path("results/agent_selfcheck_prompts")
PROTOCOLS = Path("results/record_protocols.json")
COUNTER = Path("results/agent_cache/_daily_request_count.json")
DAILY_CAP = 500
CALLS_PER_UPSTREAM = 4  # A1-A4
CALLS_PER_VERDICT = 2   # A5a, A5b
RETRY_ALLOWANCE = 0.25
BUILDER_FORBIDDEN_SC = tuple(BUILDER_FORBIDDEN) + tuple(c.lower() for c in FLAG_COLUMNS) + ("neighbour", "population", "baseline pool")


# ------------------------------------------------------------------ prompt checks
def template_prompts(flow: str, prompts=P) -> Dict[str, str]:
    return {"a1": prompts.a1(flow), "a2": prompts.a2(flow, ""), "a3": prompts.a3(flow, "", ""), "a4": prompts.a4(flow),
            "a5a": prompts.a5a(flow), "a5b": prompts.a5b("", "", [])}


def template_numbers(prompts=P) -> set:
    return set().union(*(numbers_in(t) for t in template_prompts(prompts.flow_block({}, None, None), prompts).values()))


TEMPLATE_NUMBERS = template_numbers(P)


def check_prompt(full: str, builder: str, counts: set, flow_values: set, tmpl_numbers: Optional[set] = None) -> None:
    low = full.lower()
    hits = [p for p in ALWAYS_FORBIDDEN if p in low]
    if hits:
        raise AssertionError(f"forbidden content in prompt: {hits}")
    blow = builder.lower()
    hits = [p for p in BUILDER_FORBIDDEN_SC if p in blow]
    if hits:
        raise AssertionError(f"forbidden content in the builder's own text: {hits}")
    leaked = (numbers_in(builder) & counts) - (flow_values | (TEMPLATE_NUMBERS if tmpl_numbers is None else tmpl_numbers))
    if leaked:
        raise AssertionError(f"stored neighbourhood count(s) {sorted(leaked)} appear in the builder's text")


# ------------------------------------------------------------------ trust
def agent_trust(name: str, out, row: Dict[str, float], a4_pairs, protocol: Optional[str], checklist=()) -> dict:
    claims = getattr(out, "claims", []) if out is not None else []
    absences = getattr(out, "absences", []) if out is not None else []
    pairs = core.cited_pairs(claims, absences, checklist)
    comp = {
        "F": core.field_fidelity(pairs, row),
        "B": None if name in ("a4",) else core.blind_corroboration(pairs, a4_pairs),
        "S": core.self_consistency(absences, checklist, protocol),
        "H": core.abstention_honesty(absences, row) if absences else None,
    }
    T = core.trust_T(comp)
    failed = [{"field": n, "cited": v, "row": row.get(n)} for n, v in pairs if not (n in row and core._match(v, float(row[n])))]
    # old C/E/V, adapted to this schema, for comparison only
    old_C = sum(c.confidence for c in claims) / len(claims) if claims else None
    old_E = (sum(1 for c in claims if all(v.name in row and core._match(float(v.value), float(row[v.name])) for v in c.cited)) / len(claims)
             if claims else None)
    a4_set = {(n, round(v, 9)) for n, v in a4_pairs}
    old_V = (None if name == "a4" or not claims else
             sum(1 for c in claims if any((v.name, round(float(v.value), 9)) in a4_set for v in c.cited)) / len(claims))
    return {"components": comp, "T": T, "band": core.band(T), "withheld": core.withheld(T),
            "failed_citations": failed, "old": {"C": old_C, "E": old_E, "V": old_V}}


def stage_meta(m) -> dict:
    return {"cached": bool(m.cached), "schema_retries": int(m.schema_retries), "retry_reasons": list(m.retry_reasons),
            "provider": m.provider, "served_model": m.served_model, "generation_ids": list(m.generation_ids),
            "cost_usd": m.cost_usd, "elapsed_s": m.elapsed_s, "retry_wait_s": m.retry_wait_s,
            "http_retries": m.http_retries, "rejected_attempts": list(m.rejected_attempts)}


def failures_path(out: Path) -> Path:
    return out.with_name(out.stem + "_failed.jsonl")


def check_upstream_versions(up: dict, prompts, path: Path) -> None:
    """A1-A4 replayed from disk must have been produced by the prompt version this run uses."""
    stored = up.get("prompt_versions")
    expected = {k: prompts.VERSIONS[k] for k in ("a1", "a2", "a3", "a4")}
    if stored is None and prompts is P:
        return  # files written by the original run, before versions were stored
    if stored != expected:
        raise RuntimeError(f"{path}: upstream written under {stored}, this run uses {expected}; "
                           "delete it or use a separate run directory")


# ------------------------------------------------------------------ justification assembly for A5b
def justifications(order: List[str], texts: Dict[str, str], trust: Dict[str, dict]) -> str:
    blocks = []
    for name in order:
        blocks.append(f"[{name.upper()} -- reliability on this flow: {trust[name]['band']}]\n{texts[name]}")
    return "\n\n".join(blocks) or "(none shown)"


def order_by_trust(names: List[str], trust: Dict[str, dict]) -> List[str]:
    scored = sorted([n for n in names if trust[n]["T"] is not None], key=lambda n: -trust[n]["T"])
    unscored = [n for n in names if trust[n]["T"] is None]
    return scored + unscored


# ------------------------------------------------------------------ quota
def remaining_quota() -> int:
    if not COUNTER.exists():
        return DAILY_CAP
    d = json.loads(COUNTER.read_text(encoding="utf-8"))
    if d.get("date") != dt.date.today().isoformat():
        return DAILY_CAP
    return DAILY_CAP - int(d.get("counts", {}).get(base.DEFAULT_MODEL, 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run
    blinds = load_45()
    if os.environ.get("SELFCHECK_LIMIT"):
        blinds = blinds[: int(os.environ["SELFCHECK_LIMIT"])]
    UPSTREAM.mkdir(parents=True, exist_ok=True)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["record_id"] for l in open(OUT, encoding="utf-8") if l.strip()} if OUT.exists() else set()
    todo = [b for b in blinds if b["record_id"] not in done]
    need_up = sum(1 for b in todo if not (UPSTREAM / f"{b['record_id']}.json").exists())
    estimate = int((need_up * CALLS_PER_UPSTREAM + len(todo) * CALLS_PER_VERDICT) * (1 + RETRY_ALLOWANCE))
    left = remaining_quota()
    print(f"records to run: {len(todo)} (upstream needed for {need_up}); estimated calls incl. {int(RETRY_ALLOWANCE * 100)}% retry allowance: {estimate}; remaining quota today: {left}")
    if not dry and estimate > left:
        print("STOP: estimated calls exceed remaining quota; nothing run.")
        return 2

    protocols = json.loads(PROTOCOLS.read_text(encoding="utf-8"))
    keys = list(blinds[0]["features"].keys())
    pool = es.load_pool_slim(extra_columns=list(ra.EXTRA_POOL_COLUMNS) + keys)
    match_cols = [c for c in keys if c in pool.columns]
    counts = known_counts()
    if not dry and not load_api_key_from_dotenv():
        raise SystemExit("GEMINI_API_KEY not available")
    return run(blinds, out=OUT, tally_path=TALLY, upstream=UPSTREAM, prompt_dir=PROMPT_DIR, protocols=protocols,
               pool=pool, match_cols=match_cols, counts=counts, dry=dry)


def run(blinds: List[dict], *, out: Path, tally_path: Path, upstream: Path, prompt_dir: Path, protocols: dict,
        pool, match_cols: List[str], counts: Dict[str, set], dry: bool = False, prompts=P,
        workers: int = 1, holdout: bool = False, holdout_a4: bool = False) -> int:
    """The record loop. `sp` must already be configured for the model and prompt module.
    Returns 0, or 3 if the run was stopped (quota, provider error or a cost ceiling); where it
    stopped is recorded in the tally."""
    upstream.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["record_id"] for l in open(out, encoding="utf-8") if l.strip()} if out.exists() else set()
    todo = [b for b in blinds if b["record_id"] not in done]
    tmpl_numbers = template_numbers(prompts)
    tally = {"records": len(blinds), "already_done": len(done), "model": sp.MODEL, "calls": 0, "cache_hits": 0,
             "schema_retries": 0, "cost_usd": 0.0, "failed": [], "stopped_on_quota": False, "withheld_log": []}
    example_saved = (prompt_dir / "a5b.txt").exists()

    stop = threading.Event()

    def one(b: dict):
        """One record, start to finish. Touches no shared state: everything it learns is
        returned, and the caller (one thread) does all counting and writing."""
        rid = b["record_id"]
        if stop.is_set():
            return ("skipped", rid)
        t0 = time.monotonic()
        feats = b["features"]
        proto = protocols[rid]["protocol"]
        src, _ = es.locate_source(pool, feats, match_cols)
        tod = time_of_day(src[1]) if src else None
        flow = prompts.flow_block(feats, proto, tod)
        row = {k: float(v) for k, v in shown_features(feats).items()}
        fvals = set(row.values()) | (numbers_in(tod) if tod else set())
        rcounts = counts.get(rid, set())
        builders = template_prompts(flow, prompts)
        for t in builders.values():
            check_prompt(t, t, rcounts, fvals, tmpl_numbers)
        held = hidden = ()
        assert holdout or not holdout_a4, "A4 hold-out uses A3's held-out set"
        if holdout:
            held, hidden = ho.held_out(rid), ho.hidden_fields(rid)
            feats_a3 = ho.redact_features(shown_features(feats), hidden)
            assert not set(hidden) & set(feats_a3), "withheld fields must not be in A3's row"
            flow_a3 = prompts.flow_block(feats_a3, proto, tod)
        if dry:
            return ("dry", rid)
        metas = {}
        prompts_used = {}
        try:
            up_path = upstream / f"{rid}.json"
            if up_path.exists():
                up = json.loads(up_path.read_text(encoding="utf-8"))
                check_upstream_versions(up, prompts, up_path)
                a1 = SCClaimsResponse(**up["a1"]); a2 = SCBehaviourResponse(**up["a2"])
                a3 = SCHypothesisResponse(**up["a3"]); a4 = SCBlindResponse(**up["a4"])
                upstream_meta = up.get("meta", {})
                upstream_replayed = True
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as px:
                    # A4 is submitted before A1 runs and never sees chain output, in both modes
                    if holdout_a4:
                        f4 = px.submit(sp.run_a4_holdout, rid, flow_a3, held, hidden, list(feats_a3))
                    else:
                        f4 = px.submit(sp.run_a4, rid, flow)
                    (a1, metas["a1"]), prompts_used["a1"] = sp.run_a1(rid, flow)
                    (a2, metas["a2"]), prompts_used["a2"] = sp.run_a2(rid, flow, a1)
                    if holdout:
                        (a3, metas["a3"]), prompts_used["a3"] = sp.run_a3_holdout(
                            rid, flow_a3, a1, a2, held, hidden, list(feats_a3))
                    else:
                        (a3, metas["a3"]), prompts_used["a3"] = sp.run_a3(rid, flow, a1, a2)
                    (a4, metas["a4"]), prompts_used["a4"] = f4.result()
                for k in ("a1", "a2", "a3", "a4"):
                    check_prompt(prompts_used[k], builders[k], rcounts, fvals, tmpl_numbers)
                upstream_meta = {k: stage_meta(metas[k]) for k in ("a1", "a2", "a3", "a4")}
                up_path.write_text(json.dumps({**{k: v.model_dump(mode="json") for k, v in
                                                  {"a1": a1, "a2": a2, "a3": a3, "a4": a4}.items()},
                                               "model": sp.MODEL, "meta": upstream_meta,
                                               "prompt_versions": {k: prompts.VERSIONS[k] for k in ("a1", "a2", "a3", "a4")}}),
                                    encoding="utf-8")
                upstream_replayed = False

            (a5a, metas["a5a"]), prompts_used["a5a"] = sp.run_a5a(rid, flow, set(row))
            check_prompt(prompts_used["a5a"], builders["a5a"], rcounts, fvals, tmpl_numbers)

            # hypothesis status, by code, before any agent reads it
            status = {}
            status_detail = {}
            for h in [*a3.hypotheses, *a4.hypotheses]:
                if (holdout and h.hypothesis_id.startswith("a3_")) or (holdout_a4 and h.hypothesis_id.startswith("a4_")):
                    # the agent predicted fields it never saw
                    st, det = ho.holdout_status(h.predictions, row, held)
                else:
                    st, det = core.hypothesis_status(h.predictions, row)
                status[h.hypothesis_id] = st
                status_detail[h.hypothesis_id] = det

            # trust
            a4_pairs = core.cited_pairs(a4.claims, a4.absences)
            trust = {
                "a1": agent_trust("a1", a1, row, a4_pairs, proto),
                "a2": agent_trust("a2", a2, row, a4_pairs, proto),
                "a3": agent_trust("a3", a3, row, a4_pairs, proto),
                "a4": agent_trust("a4", a4, row, a4_pairs, proto),
                "a5a": agent_trust("a5a", None, row, a4_pairs, proto, checklist=a5a.checklist),
            }
            texts = {
                "a1": prompts.claims_text(a1.claims),
                "a2": prompts.claims_text(a2.claims) + "\n  absences:\n" + prompts.absences_text(a2.absences),
                "a3": prompts.claims_text(a3.claims) + "\n  explanations:\n" + prompts.hypotheses_text(a3.hypotheses, status),
                "a4": prompts.claims_text(a4.claims) + "\n  absences:\n" + prompts.absences_text(a4.absences)
                      + "\n  explanations:\n" + prompts.hypotheses_text(a4.hypotheses, status),
            }
            shown = [n for n in ("a1", "a2", "a3", "a4") if not trust[n]["withheld"]]
            withheld_log = [{"record_id": rid, "agent": n, "T": trust[n]["T"]}
                            for n in ("a1", "a2", "a3", "a4") if trust[n]["withheld"]]
            order = order_by_trust(shown, trust)
            creditable = {h.hypothesis_id for src_name, hyps in (("a3", a3.hypotheses), ("a4", a4.hypotheses))
                          if src_name in shown for h in hyps if h.benign and status[h.hypothesis_id] == core.SUPPORTED}
            yes_ids = [r.condition_id for r in a5a.checklist if r.holds.value == "yes"]
            cl_text = f"[A5a -- reliability on this flow: {trust['a5a']['band']}]\n" + prompts.checklist_text(a5a.checklist)
            (a5b, metas["a5b"]), prompts_used["a5b"] = sp.run_a5b(rid, cl_text, justifications(order, texts, trust), yes_ids, creditable)
            check_prompt(prompts_used["a5b"], builders["a5b"], rcounts, fvals, tmpl_numbers)
            assert "Fields of this flow's row" not in prompts_used["a5b"], "A5b must never see the raw row"
        except base.DailyQuotaExceeded as exc:
            stop.set()
            return ("quota", rid, str(exc))
        except (openrouter.OpenRouterError, openrouter.OpenRouterConfigError) as exc:
            stop.set()
            return ("stopped", rid, f"{type(exc).__name__}: {exc}")
        except base.SchemaValidationFailed as exc:
            failure = {"record_id": rid, "model": sp.MODEL, "agent": exc.agent,
                       "prompt_version": prompts.VERSIONS.get(exc.agent), "reasons": list(exc.retry_reasons),
                       "calls": getattr(exc, "calls", []), "rejected_attempts": getattr(exc, "rejected_attempts", []),
                       "elapsed_s": time.monotonic() - t0}
            return ("failed", failure, exc.attempts, metas)

        line = {
            "record_id": rid, "batch": b.get("_batch"), "model": sp.MODEL, "protocol": proto, "time_of_day": tod,
            "verdict": a5b.verdict.value,
            "checklist_alone_verdict": core.checklist_alone_verdict(yes_ids),
            "yes_conditions": yes_ids,
            "conditions_code": core.condition_holds(row, proto),
            "dispositions": [d.model_dump(mode="json") for d in a5b.dispositions],
            "rationale": a5b.rationale,
            "hypothesis_status": status, "hypothesis_detail": status_detail,
            "struck_out": [h for h, s in status.items() if s == core.CONTRADICTED],
            "creditable": sorted(creditable),
            "trust": trust, "shown_order": order,
            "withheld": [n for n in ("a1", "a2", "a3", "a4") if trust[n]["withheld"]],
            "upstream_replayed": upstream_replayed,
            "a1": a1.model_dump(mode="json"), "a2": a2.model_dump(mode="json"), "a3": a3.model_dump(mode="json"),
            "a4": a4.model_dump(mode="json"), "a5a": a5a.model_dump(mode="json"), "a5b": a5b.model_dump(mode="json"),
            "stage_meta": {**upstream_meta, **{k: stage_meta(m) for k, m in metas.items()}},
            "prompt_versions": dict(prompts.VERSIONS),
            "elapsed_s": time.monotonic() - t0,
            "held_out": list(held), "hidden_from_a3": list(hidden),
        }
        return ("ok", line, metas, prompts_used, withheld_log)

    def count(metas) -> None:
        for m in metas.values():
            if m.cached:
                tally["cache_hits"] += 1
            else:
                tally["calls"] += 1 + m.schema_retries
                tally["cost_usd"] += m.cost_usd or 0.0
            tally["schema_retries"] += m.schema_retries

    not_run = []
    for res in _results(one, todo, workers):
        kind = res[0]
        if kind in ("dry",):
            continue
        if kind == "skipped":
            not_run.append(res[1])
        elif kind == "quota":
            tally["stopped_on_quota"] = True
            tally.setdefault("stopped_at", {"record_id": res[1], "reason": res[2]})
            print(f"[STOPPED] {res[1][:10]}: {res[2]}", flush=True)
        elif kind == "stopped":
            tally.setdefault("stopped_on_error", f"{res[1]}: {res[2]}")
            tally.setdefault("stopped_at", {"record_id": res[1], "reason": res[2]})
            print(f"[STOPPED] {res[1][:10]}: {res[2]}", flush=True)
        elif kind == "failed":
            failure, attempts, metas = res[1], res[2], res[3]
            tally["failed"].append(failure)
            with open(failures_path(out), "a", encoding="utf-8") as f:
                f.write(json.dumps(failure, default=str) + "\n")
            count(metas)
            tally["calls"] += attempts
            tally["cost_usd"] += sum(c.get("cost_usd") or 0.0 for c in failure["calls"])
            print(f"[FAILED] {failure['record_id'][:10]} {failure['agent']}: {failure['reasons'][:3]}", flush=True)
        else:
            _, line, metas, prompts_used, withheld_log = res
            count(metas)
            tally["withheld_log"] += withheld_log
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=str) + "\n")
            if not example_saved and len(prompts_used) == 6:
                for k, t in prompts_used.items():
                    (prompt_dir / f"{k}.txt").write_text(t, encoding="utf-8")
                (prompt_dir / "record_id.txt").write_text(line["record_id"], encoding="utf-8")
                example_saved = True
            print(f"  {line['record_id'][:10]} {line['protocol']} verdict={line['verdict']} "
                  f"checklist-alone={line['checklist_alone_verdict']}", flush=True)
    if not_run:
        tally["not_run_after_stop"] = not_run

    if dry:
        print(f"dry run: every prompt template passed for {len(todo)} records")
        return 0
    if sp.MODEL and openrouter.is_configured(sp.MODEL):
        tally["cost_guard"] = {k: v for k, v in openrouter.cost_guard_state().items() if k != "ledger"}
    # a resumed run keeps the history of every invocation, so e.g. a change of concurrency shows
    prior = json.loads(tally_path.read_text(encoding="utf-8")) if tally_path.exists() else {}
    tally["invocations"] = prior.get("invocations", [{"concurrency": None, "note": "first invocation, not recorded",
                                                      "calls": prior.get("calls"), "cost_usd": prior.get("cost_usd")}]
                                     if prior else []) + [
        {"t": time.time(), "concurrency": workers, "already_done": len(done), "calls": tally["calls"],
         "cost_usd": tally["cost_usd"], "stopped_at": tally.get("stopped_at")}]
    tally_path.write_text(json.dumps(tally, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in tally.items() if k not in ("withheld_log", "failed")}, indent=2, default=str))
    return 3 if ("stopped_at" in tally) else 0


def _results(one, todo, workers: int):
    """Yields each record's result in completion order. With one worker it is a plain loop,
    exactly the original behaviour; with more, records run concurrently."""
    if workers <= 1:
        for b in todo:
            yield one(b)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, b) for b in todo]
        for f in concurrent.futures.as_completed(futures):
            yield f.result()


if __name__ == "__main__":
    raise SystemExit(main())
