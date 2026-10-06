"""Strips every escalation-mechanism hint from `results/agent_input_20.jsonl`
before any agent sees a record, producing `results/agent_input_20_blind.jsonl`.

Removed/replaced, per the brief:
- `class_predicted` -- the tree's own guess (present on 10/20 records,
  10/12 attacks and 0/8 benign -- an agent reading it would parrot P1
  rather than reason from the features).
- `rule_id` -- identifies which attack-leaf rule fired; the same leak by
  another name.
- `priority` and `reason` -- collapsed into ONE uniform string
  (`UNIFORM_ESCALATION_REASON`) on every record, since "P1 rule match"
  vs "P3 sampled" is itself informative (all 12 attacks came from P1 or
  P2; the only P3 record is benign).

Kept: the full CICFlowMeter feature dict, `feature_bins`, `record_id`.

This module is intentionally import-safe with no API/network dependency,
so `tests/test_build_blind_input.py` can exercise `strip_to_blind`/
`verify_blind_file` without ever touching agents/base.py's API client.

Run: python -m eval.build_blind_input
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

AGENT_INPUT_PATH = Path("results/agent_input_20.jsonl")
BLIND_OUTPUT_PATH = Path("results/agent_input_20_blind.jsonl")

UNIFORM_ESCALATION_REASON = "escalated for controller analysis"

#: fields that must never appear anywhere in a blind record. Maps to the
#: brief's "five stripped fields" as: class_predicted, rule_id, reason,
#: priority (the four named explicitly) plus true_label/is_attack (the
#: ground-truth pair already guaranteed absent upstream by
#: eval/select_agent_sample.py -- reconfirmed here defensively rather
#: than trusted blindly, since this file is a second, independent build
#: step over that one's output).
FORBIDDEN_FIELDS = frozenset({
    "class_predicted", "rule_id", "reason", "priority", "true_label", "is_attack",
})

REQUIRED_TOP_LEVEL_KEYS = frozenset({"record_id", "features", "feature_bins", "escalation_reason"})


def strip_to_blind(record: dict) -> dict:
    esc = record.get("escalation", {})
    return {
        "record_id": record["record_id"],
        "features": record["features"],
        "feature_bins": dict(esc.get("feature_bins", {})),
        "escalation_reason": UNIFORM_ESCALATION_REASON,
    }


def _walk_keys(obj, path: str, out: List[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k in FORBIDDEN_FIELDS:
                out.append(f"{path}.{k}")
            _walk_keys(v, f"{path}.{k}", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk_keys(v, f"{path}[{i}]", out)


def verify_blind_file(records: List[dict]) -> None:
    """The verification gate: must pass before any API call is made.

    Checks, in order:
    1. every record has the exact same top-level key set;
    2. that key set is exactly REQUIRED_TOP_LEVEL_KEYS -- nothing extra,
       nothing missing;
    3. `features` key sets are identical across all records (an agent
       could otherwise key on "this record has an extra/missing feature"
       as a structural tell between attack and benign);
    4. `feature_bins` key sets are identical across all records, same
       reasoning;
    5. `escalation_reason` is the exact same literal string on every
       record -- not just non-null, IDENTICAL, so it carries zero bits;
    6. none of FORBIDDEN_FIELDS appears anywhere, recursively, in any
       record.
    """
    if not records:
        raise AssertionError("no records to verify")

    key_sets = [frozenset(r.keys()) for r in records]
    if len(set(key_sets)) != 1:
        raise AssertionError(f"records do not share an identical top-level key set: {set(key_sets)}")
    if key_sets[0] != REQUIRED_TOP_LEVEL_KEYS:
        raise AssertionError(f"top-level keys {key_sets[0]} != required {REQUIRED_TOP_LEVEL_KEYS}")

    feature_key_sets = {frozenset(r["features"].keys()) for r in records}
    if len(feature_key_sets) != 1:
        raise AssertionError("'features' key sets differ between records -- a structural tell")

    bin_key_sets = {frozenset(r["feature_bins"].keys()) for r in records}
    if len(bin_key_sets) != 1:
        raise AssertionError("'feature_bins' key sets differ between records -- a structural tell")

    reasons = {r["escalation_reason"] for r in records}
    if reasons != {UNIFORM_ESCALATION_REASON}:
        raise AssertionError(f"escalation_reason is not uniform across records: {reasons}")

    for i, r in enumerate(records):
        hits: List[str] = []
        _walk_keys(r, f"record[{i}]", hits)
        if hits:
            raise AssertionError(f"forbidden field(s) survived: {hits}")


def build_blind_records(input_path: Path = AGENT_INPUT_PATH) -> List[dict]:
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return [strip_to_blind(r) for r in records]


def write_blind_file(blind_records: List[dict], out_path: Path = BLIND_OUTPUT_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in blind_records:
            f.write(json.dumps(r) + "\n")


def load_blind_file(path: Path = BLIND_OUTPUT_PATH) -> List[dict]:
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
    print(f"wrote {BLIND_OUTPUT_PATH} ({len(blind_records)} records)")

    # Verification gate: reload from disk (an independent pass over what's
    # actually written, not the in-memory objects) and verify before
    # anything downstream is allowed to spend API quota.
    reloaded = load_blind_file()
    verify_blind_file(reloaded)
    print(f"VERIFICATION GATE PASSED: {len(reloaded)} records, identical key set, "
          "no forbidden field survives, escalation_reason uniform.")

    print("\none full record, for inspection:")
    print(json.dumps(reloaded[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
