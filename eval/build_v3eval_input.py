"""Builds the label-free agent input for v3eval with the identical steps eval20 used
(eval.build_eval20_input.build: features, bins, digest, ground-truth check, strip, verify, and the
no-label/priority/reason/rule-class assertion), reading the digest from the v3 policy run.

Run: python -m eval.build_v3eval_input
"""
from __future__ import annotations

from pathlib import Path

from eval.build_eval20_input import build

KEY = Path("results/v3eval_records.json")
V3_CSV = Path("results/escalated_flows_v3.csv")
INPUT = Path("results/v3eval_input.jsonl")
BLIND = Path("results/v3eval_input_blind.jsonl")

if __name__ == "__main__":
    raise SystemExit(build(KEY, V3_CSV, INPUT, BLIND))
