"""eval20 run for one OpenRouter model. Rules fixed in results/eval20_prereg.md before any run.

The model must be named in config/openrouter.json; it is verified live (slug, pinned provider,
structured-output support) before any call. Input is the label-free
results/eval20_input_blind.jsonl only. Everything a run writes is namespaced by model, the A1-A4
upstream cache included, so one model's chain is never replayed for another.

Run: python -m eval.run_eval20 --model <slug from config> [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents import evidence_signals as es
from agents import openrouter
from agents import reference_arms as ra
from agents import selfcheck_pipeline as sp
from agents.prompts import selfcheck_prompts_v3 as PROMPTS
from agents.prompts import selfcheck_prompts_v4 as PROMPTS_HOLDOUT
from eval import run_selfcheck as rs
from eval.build_blind_input import load_blind_file, verify_blind_file
from eval.record_protocols import protocols_for
from eval.run_priors import known_counts

BLIND = Path("results/eval20_input_blind.jsonl")
PROTOCOLS = Path("results/eval20_protocols.json")
RUNS = Path("results/eval20_runs")
LEDGER = RUNS / "_spend_ledger.jsonl"


#: variant runs: name -> (prompt module, holdout mode, per-run cost stop in USD)
VARIANTS = {"holdout3": (PROMPTS_HOLDOUT, True, 0.50)}


def run_dir(model: str, variant: str = "") -> Path:
    return RUNS / (model.replace("/", "__").replace(":", "_") + (f"__{variant}" if variant else ""))


def run_name(model: str, variant: str = "") -> str:
    return f"{model}#{variant}" if variant else model


def load_protocols(blinds) -> dict:
    if not PROTOCOLS.exists():
        PROTOCOLS.write_text(json.dumps(protocols_for(blinds), indent=2), encoding="utf-8")
    protocols = json.loads(PROTOCOLS.read_text(encoding="utf-8"))
    missing = [b["record_id"] for b in blinds if not protocols.get(b["record_id"], {}).get("protocol")]
    assert not missing, f"protocol not recovered for {missing}"
    return protocols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="")
    args = ap.parse_args()
    if not openrouter.is_configured(args.model):
        raise SystemExit(f"{args.model!r} is not in {openrouter.CONFIG_PATH}")
    pinned = openrouter.verify(args.model)
    print(f"verified: {pinned['model']} via {pinned['provider_tag']} ({pinned['endpoint_name']})")

    blinds = load_blind_file(BLIND)
    verify_blind_file(blinds)
    if args.limit:
        blinds = blinds[: args.limit]
    protocols = load_protocols(load_blind_file(BLIND))
    keys = list(blinds[0]["features"].keys())
    pool = es.load_pool_slim(extra_columns=list(ra.EXTRA_POOL_COLUMNS) + keys)
    match_cols = [c for c in keys if c in pool.columns]

    prompts, holdout, variant_ceiling = VARIANTS[args.variant] if args.variant else (PROMPTS, False, None)
    sp.configure(args.model, prompts)
    d = run_dir(args.model, args.variant)
    d.mkdir(parents=True, exist_ok=True)
    cfg = openrouter.load_config()
    m = openrouter.model_config(args.model)
    if "cost_ceiling_usd" not in m or "global_cost_ceiling_usd" not in cfg:
        raise SystemExit(f"{args.model}: no cost ceiling in {openrouter.CONFIG_PATH}; refusing to run unguarded")
    concurrency = m.get("max_concurrent", cfg.get("max_concurrent", 1))
    ceiling = variant_ceiling if variant_ceiling is not None else m["cost_ceiling_usd"]
    openrouter.set_cost_guard(args.model, ceiling, cfg["global_cost_ceiling_usd"], LEDGER, concurrency,
                              run=run_name(args.model, args.variant))
    g = openrouter.cost_guard_state()
    print(f"cost guard: run ${g['run_spent']:.4f} of ${g['run_ceiling']:.2f}, "
          f"global ${g['global_spent']:.4f} of ${g['global_ceiling']:.2f}; concurrency {concurrency}")
    return rs.run(blinds, out=d / "records.jsonl", tally_path=d / "tally.json", upstream=d / "upstream",
                  prompt_dir=d / "prompts", protocols=protocols, pool=pool, match_cols=match_cols,
                  counts=known_counts(), dry=args.dry_run, prompts=prompts, workers=concurrency, holdout=holdout)


if __name__ == "__main__":
    raise SystemExit(main())
