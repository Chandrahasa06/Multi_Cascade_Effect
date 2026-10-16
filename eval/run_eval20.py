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
from agents.prompts import selfcheck_prompts_v5 as PROMPTS_HOLDOUT_A4
from agents.prompts import selfcheck_prompts_v6 as PROMPTS_NO_QUOTA
from eval import run_selfcheck as rs
from eval.build_blind_input import load_blind_file, verify_blind_file
from eval.record_protocols import protocols_for
from eval.run_priors import known_counts

BLIND = Path("results/eval20_input_blind.jsonl")
PROTOCOLS = Path("results/eval20_protocols.json")
RUNS = Path("results/eval20_runs")
LEDGER = RUNS / "_spend_ledger.jsonl"


#: variant -> (prompts, A3 hold-out, A4 hold-out, guard in USD). Guards trip at $0.45 so calls already
#: in flight (at most 8, ~$0.005 each for Haiku) cannot carry spend past the $0.50 limit.
VARIANTS = {"holdout3": (PROMPTS_HOLDOUT, True, False, 0.45),
            "holdout3_a4": (PROMPTS_HOLDOUT_A4, True, True, 0.45),
            "holdout3_noquota": (PROMPTS_NO_QUOTA, True, True, 0.45)}


#: dataset -> (blind input, protocols file, runs dir, guard in USD or None for the variant's own).
#: v3eval is a different policy run from eval20 and is never merged with it. Its guard trips at
#: $1.95 so calls in flight cannot carry spend past its $2.00 limit.
DATASETS = {
    "eval20": (BLIND, PROTOCOLS, RUNS, None),
    "v3eval": (Path("results/v3eval_input_blind.jsonl"), Path("results/v3eval_protocols.json"),
               Path("results/v3eval_runs"), 1.95),
}
#: the key's reported balance lags recent spend; require this much beyond the run's own ceiling
KEY_MARGIN_USD = 0.25


def run_dir(model: str, variant: str = "", dataset: str = "eval20") -> Path:
    return DATASETS[dataset][2] / (model.replace("/", "__").replace(":", "_") + (f"__{variant}" if variant else ""))


def run_name(model: str, variant: str = "", dataset: str = "eval20") -> str:
    name = f"{model}#{variant}" if variant else model
    return name if dataset == "eval20" else f"{dataset}:{name}"


def load_protocols(blinds, path: Path = PROTOCOLS) -> dict:
    if not path.exists():
        path.write_text(json.dumps(protocols_for(blinds), indent=2), encoding="utf-8")
    protocols = json.loads(path.read_text(encoding="utf-8"))
    missing = [b["record_id"] for b in blinds if not protocols.get(b["record_id"], {}).get("protocol")]
    assert not missing, f"protocol not recovered for {missing}"
    return protocols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="eval20")
    args = ap.parse_args()
    blind_path, protocols_path, _, dataset_ceiling = DATASETS[args.dataset]
    if not openrouter.is_configured(args.model):
        raise SystemExit(f"{args.model!r} is not in {openrouter.CONFIG_PATH}")
    pinned = openrouter.verify(args.model)
    print(f"verified: {pinned['model']} via {pinned['provider_tag']} ({pinned['endpoint_name']})")

    blinds = load_blind_file(blind_path)
    verify_blind_file(blinds)
    if args.limit:
        blinds = blinds[: args.limit]
    protocols = load_protocols(load_blind_file(blind_path), protocols_path)
    keys = list(blinds[0]["features"].keys())
    pool = es.load_pool_slim(extra_columns=list(ra.EXTRA_POOL_COLUMNS) + keys)
    match_cols = [c for c in keys if c in pool.columns]

    prompts, holdout, holdout_a4, variant_ceiling = (VARIANTS[args.variant] if args.variant
                                                     else (PROMPTS, False, False, None))
    sp.configure(args.model, prompts)
    d = run_dir(args.model, args.variant, args.dataset)
    d.mkdir(parents=True, exist_ok=True)
    cfg = openrouter.load_config()
    m = openrouter.model_config(args.model)
    if "cost_ceiling_usd" not in m or "global_cost_ceiling_usd" not in cfg:
        raise SystemExit(f"{args.model}: no cost ceiling in {openrouter.CONFIG_PATH}; refusing to run unguarded")
    concurrency = m.get("max_concurrent", cfg.get("max_concurrent", 1))
    ceiling = (dataset_ceiling if dataset_ceiling is not None
               else variant_ceiling if variant_ceiling is not None else m["cost_ceiling_usd"])
    openrouter.set_cost_guard(args.model, ceiling, cfg["global_cost_ceiling_usd"], LEDGER, concurrency,
                              run=run_name(args.model, args.variant, args.dataset))
    g = openrouter.cost_guard_state()
    if not args.dry_run:
        left = openrouter.key_remaining()
        need = g["run_ceiling"] - g["run_spent"] + KEY_MARGIN_USD
        if left is not None and left < need:
            print(f"STOP: the OpenRouter key has ${left:.2f} left; this run may spend up to "
                  f"${g['run_ceiling'] - g['run_spent']:.2f} (+${KEY_MARGIN_USD:.2f} margin for the key's reporting lag). "
                  "Raise the key's cap first. Nothing was run.")
            return 2
    print(f"cost guard: run ${g['run_spent']:.4f} of ${g['run_ceiling']:.2f}, "
          f"global ${g['global_spent']:.4f} of ${g['global_ceiling']:.2f}; concurrency {concurrency}")
    return rs.run(blinds, out=d / "records.jsonl", tally_path=d / "tally.json", upstream=d / "upstream",
                  prompt_dir=d / "prompts", protocols=protocols, pool=pool, match_cols=match_cols,
                  counts=known_counts(), dry=args.dry_run, prompts=prompts, workers=concurrency, holdout=holdout,
                  holdout_a4=holdout_a4)


if __name__ == "__main__":
    raise SystemExit(main())
