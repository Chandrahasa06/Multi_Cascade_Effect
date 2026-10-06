"""Batch 3's blind-input build -- identical stripping/verification logic
to eval/build_blind_input.py (strip_to_blind, verify_blind_file, reused
unchanged), applied to results/agent_input_10c.jsonl instead of batches
1/2's files.

Run: python -m eval.build_blind_input_batch3
"""
from __future__ import annotations

import json
from pathlib import Path

from eval.build_blind_input import strip_to_blind, verify_blind_file

AGENT_INPUT_PATH_C = Path("results/agent_input_10c.jsonl")
BLIND_OUTPUT_PATH_C = Path("results/agent_input_10c_blind.jsonl")


def build_blind_records(input_path: Path = AGENT_INPUT_PATH_C):
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return [strip_to_blind(r) for r in records]


def write_blind_file(blind_records, out_path: Path = BLIND_OUTPUT_PATH_C) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in blind_records:
            f.write(json.dumps(r) + "\n")


def load_blind_file(path: Path = BLIND_OUTPUT_PATH_C):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    blind_records = build_blind_records()
    write_blind_file(blind_records)
    print(f"wrote {BLIND_OUTPUT_PATH_C} ({len(blind_records)} records)")

    reloaded = load_blind_file()
    verify_blind_file(reloaded)
    print(f"VERIFICATION GATE PASSED: {len(reloaded)} records, identical key set, "
          "no forbidden field survives, escalation_reason uniform.")

    print("\none full record, for inspection:")
    print(json.dumps(reloaded[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
