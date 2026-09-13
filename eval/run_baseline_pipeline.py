"""Cost-controlled runner for the single-LLM baseline control
(agents/baseline.py) -- the "did you need five agents" comparison.

Same record file, same cache/resume/daily-quota machinery as
eval/run_agent_pipeline.py, but one call per record instead of five.

Example::

    python -m eval.run_baseline_pipeline \\
        --records results/escalation_records_sample.jsonl \\
        --out results/baseline_200.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from agents import base, baseline, pipeline
from agents.contamination import scan_agent_response
from agents.schema import BaselineResponse, derive_verdict
from controlplane.record import EscalationRecord


def append_result_jsonl(path: str, record: EscalationRecord, response: BaselineResponse, meta: base.CallMetadata) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    contamination = scan_agent_response(response)
    row = {
        "record_id": record.flow_id,
        "response": response.model_dump(mode="json"),
        "derived_verdict": derive_verdict(response.benign_plausibility).value,
        "contamination": contamination.to_dict(),
        "call_metadata": {
            "cached": meta.cached,
            "schema_retries": meta.schema_retries,
            "retry_reasons": meta.retry_reasons,
            "elapsed_s": meta.elapsed_s,
            "input_tokens": meta.input_tokens,
            "output_tokens": meta.output_tokens,
        },
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def already_written_record_ids(out_path: str) -> set:
    """Same resumability fix as eval/run_agent_pipeline.py's function of
    the same name -- see its docstring."""
    if not Path(out_path).exists():
        return set()
    seen = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["record_id"])
    return seen


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", default="results/escalation_records_sample.jsonl")
    parser.add_argument("--n", type=int, default=None, help="total records to run (first N, in file order)")
    parser.add_argument("--out", default="results/baseline_200.jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    records = pipeline.load_records(args.records)
    if args.n is not None:
        records = records[: args.n]

    already_done = already_written_record_ids(args.out)
    if already_done:
        before = len(records)
        records = [r for r in records if r.flow_id not in already_done]
        print(f"resuming: {before - len(records)}/{before} records already in {args.out}, skipping them")

    call_kwargs = dict(temperature=args.temperature, run_index=args.run_index, use_cache=not args.no_cache)
    if args.model:
        call_kwargs["model"] = args.model
    model_name = call_kwargs.get("model", base.DEFAULT_MODEL)

    cap = base.DAILY_QUOTA_BY_MODEL.get(model_name, base._UNKNOWN_MODEL_DAILY_QUOTA)
    used = base.daily_request_count(model_name)
    remaining = max(0, cap - used)
    print(
        f"running {len(records)} records against model={model_name} "
        f"(daily count for this model: {used}/{cap}, ~{remaining} calls remaining "
        f"today -- enough for ~{remaining} records if nothing is cached yet, 1 call/record)"
    )

    failures: List[str] = []
    n_done = 0
    for record in records:
        try:
            response, meta = baseline.run(record, **call_kwargs)
        except base.SchemaValidationFailed as exc:
            print(f"[FAILED] {record.flow_id}: {exc}")
            failures.append(record.flow_id)
            continue
        except base.DailyQuotaExceeded as exc:
            print(f"[STOPPED] daily quota hit after {n_done}/{len(records)} records: {exc}")
            print(
                "Recoverable -- rerun this same command (today or another day); "
                "the cache skips every record already done."
            )
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[STOPPED] unexpected error after {n_done}/{len(records)} records: "
                  f"{type(exc).__name__}: {exc}")
            print("Recoverable -- rerun this same command; the cache skips every record already done.")
            break
        append_result_jsonl(args.out, record, response, meta)
        n_done += 1

    print(f"done. {n_done}/{len(records)} records written this run. daily count for {model_name} "
          f"now: {base.daily_request_count(model_name)}/{cap}")
    if failures:
        print(f"{len(failures)}/{len(records)} records failed schema validation after retries: {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
