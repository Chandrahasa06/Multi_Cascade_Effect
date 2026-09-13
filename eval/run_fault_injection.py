"""Cost-controlled runner for the trust-propagation fault-injection
study (eval/fault_injection.py) -- the poster's Fig 1-4 data source.

Per record: one CLEAN run (5 calls, under the current a1-a5 prompt
versions -- see STATUS.md on why this can't reuse the old v4 cache: the
self-reported trust triad didn't exist in that schema), then one rerun
per fault condition (agents/fault_injection.py::RERUN_AGENTS -- 4+3+2+1
= 10 calls). 15 calls/record x 20 records = 300 calls total for this
phase (not the ~200 originally estimated for the fault reruns alone --
that estimate didn't count the clean baseline's own 100 calls, which
this run can't skip: Fig 1's "Clean" line needs the SAME self-reported
fields as every fault condition, and the old v4 cache predates that
schema).

Uses the SAME 20 flow_ids as the original v4 run (5/class) for
comparability, pulled from the backed-up v1 (50-each) escalation sample
-- see results/escalation_records_sample.v1_50each.jsonl.

Output: one combined JSONL, each line {"condition": ..., **RecordResult
dict}, plus a resumability check so re-running the same command only
pays for (record, condition) pairs not already written.

Example::

    python -m eval.run_fault_injection --out results/fault_injection_20.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

from agents import base, pipeline
from controlplane.record import EscalationRecord
from eval import fault_injection as fi

DEFAULT_V4_RESULTS = "results/agent_pipeline_20_v4.jsonl"
DEFAULT_SOURCE_SAMPLE = "results/escalation_records_sample.v1_50each.jsonl"
DEFAULT_OUT = "results/fault_injection_20.jsonl"

CONDITIONS_IN_ORDER: Tuple[str, ...] = (base.CLEAN_FAULT_CONDITION, *fi.ALL_CONDITIONS)


def load_v4_flow_ids(path: str = DEFAULT_V4_RESULTS) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["record_id"])
    return ids


def load_records_by_flow_id(flow_ids: List[str], path: str = DEFAULT_SOURCE_SAMPLE) -> List[EscalationRecord]:
    wanted = set(flow_ids)
    by_id = {r.flow_id: r for r in pipeline.load_records(path) if r.flow_id in wanted}
    missing = wanted - set(by_id)
    if missing:
        raise ValueError(f"{len(missing)} v4 flow_ids not found in {path}: {sorted(missing)[:5]}...")
    return [by_id[fid] for fid in flow_ids]  # preserve the v4 file's own record order


def already_done_pairs(out_path: str) -> Set[Tuple[str, str]]:
    """(record_id, condition) pairs already written -- same resumability
    fix as eval/run_agent_pipeline.py::already_written_record_ids, just
    keyed on the pair since this file holds 5 conditions per record."""
    if not Path(out_path).exists():
        return set()
    done = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                done.add((row["record_id"], row["condition"]))
    return done


def append_condition_result(out_path: str, condition: str, result: pipeline.RecordResult) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    row = {"condition": condition, **result.to_dict()}
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v4-results", default=DEFAULT_V4_RESULTS)
    parser.add_argument("--source-sample", default=DEFAULT_SOURCE_SAMPLE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    flow_ids = load_v4_flow_ids(args.v4_results)
    records = load_records_by_flow_id(flow_ids, args.source_sample)
    done = already_done_pairs(args.out)

    call_kwargs = dict(temperature=args.temperature, run_index=args.run_index, use_cache=not args.no_cache)
    if args.model:
        call_kwargs["model"] = args.model
    model_name = call_kwargs.get("model", base.DEFAULT_MODEL)

    cap = base.DAILY_QUOTA_BY_MODEL.get(model_name, base._UNKNOWN_MODEL_DAILY_QUOTA)
    used = base.daily_request_count(model_name)
    remaining = max(0, cap - used)
    max_affordable_records = remaining // 15  # 1 clean (5) + 4 fault reruns (10) per record, worst case uncached
    print(
        f"{len(records)} records x 5 conditions each; daily count for {model_name}: {used}/{cap}, "
        f"~{remaining} calls remaining today -- enough for ~{max_affordable_records} records if nothing is cached"
    )

    n_done = 0
    for record in records:
        clean_key = (record.flow_id, base.CLEAN_FAULT_CONDITION)
        try:
            if clean_key in done:
                # Still need the clean RecordResult object in memory to
                # drive the fault reruns below -- re-derive it from cache
                # (free: every clean call was already made, so this is
                # all cache hits) rather than re-reading it back out of
                # --out's own JSON (which would need reconstructing
                # every nested schema object by hand).
                clean = pipeline.run_record(record, **call_kwargs)
            else:
                clean = pipeline.run_record(record, **call_kwargs)
                append_condition_result(args.out, base.CLEAN_FAULT_CONDITION, clean)
                n_done += 1
                print(f"[done] {record.flow_id} clean")

            for condition in fi.ALL_CONDITIONS:
                key = (record.flow_id, condition)
                if key in done:
                    continue
                result = fi.run_fault_condition(clean, record, condition, **call_kwargs)
                append_condition_result(args.out, condition, result)
                n_done += 1
                print(f"[done] {record.flow_id} {condition}")

        except base.SchemaValidationFailed as exc:
            print(f"[FAILED] {record.flow_id}: {exc}")
            continue
        except base.DailyQuotaExceeded as exc:
            print(f"[STOPPED] daily quota hit after {n_done} (record, condition) pairs this run: {exc}")
            print("Recoverable -- rerun this same command; already-written pairs are skipped.")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[STOPPED] unexpected error after {n_done} (record, condition) pairs this run: "
                  f"{type(exc).__name__}: {exc}")
            print("Recoverable -- rerun this same command; already-written pairs are skipped.")
            break

    print(f"done. {n_done} (record, condition) pairs written this run. "
          f"daily count for {model_name} now: {base.daily_request_count(model_name)}/{cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
