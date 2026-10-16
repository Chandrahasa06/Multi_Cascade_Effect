"""Builds the agent input for the eval20 sample through the identical steps the earlier
batches used: eval.select_agent_sample's feature extraction, feature bins, escalation
digest and ground-truth check (-> results/eval20_input.jsonl), then
eval.build_blind_input's strip_to_blind + verify_blind_file
(-> results/eval20_input_blind.jsonl).

The key file (results/eval20_records.json) is read here only for record_id -> flow_id;
nothing else from it reaches either output. The blind file is additionally checked to
carry no label, no priority, no reason and no rule class prediction, by key and by value.

Run: python -m eval.build_eval20_input
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eval.build_blind_input import (
    UNIFORM_ESCALATION_REASON, load_blind_file, strip_to_blind, verify_blind_file, write_blind_file,
)
from eval.select_agent_sample import (
    ESCALATED_FLOWS_CSV, assert_no_ground_truth, build_escalation_digest, compute_feature_bins,
    fit_signature_table_for_bins, full_cicflowmeter_features, write_and_verify_jsonl,
)

KEY = Path("results/eval20_records.json")
INPUT = Path("results/eval20_input.jsonl")
BLIND = Path("results/eval20_input_blind.jsonl")

BANNED_KEYS = frozenset({"true_label", "is_attack", "label", "priority", "reason", "admitted_by",
                         "class_predicted", "rule_id", "matched_rule_id", "escalation"})


def assert_stripped(records: list, labels: set) -> None:
    def walk(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert k.lower() not in BANNED_KEYS, f"banned key {path}.{k}"
                walk(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            assert obj not in labels, f"label value at {path}: {obj!r}"
    for i, r in enumerate(records):
        walk(r, f"record[{i}]")
        strings = [v for k, v in r.items() if isinstance(v, str) and k != "record_id"]
        assert strings == [UNIFORM_ESCALATION_REASON], f"record[{i}] carries extra strings: {strings}"
        assert all(not isinstance(v, str) for v in r["features"].values())
        assert all(isinstance(v, int) for v in r["feature_bins"].values())


def main() -> int:
    return build(KEY, ESCALATED_FLOWS_CSV, INPUT, BLIND)


def build(key_path: Path, flows_csv: Path, input_path: Path, blind_path: Path) -> int:
    """The identical build/strip/verify for any key file and the escalated-flows run it came from."""
    key = json.loads(key_path.read_text(encoding="utf-8"))["records"]
    flows = pd.read_csv(flows_csv).set_index("flow_id")
    flow_ids = [k["flow_id"] for k in key]

    pool, table = fit_signature_table_for_bins()
    bins = compute_feature_bins(flow_ids, pool, table)
    records = []
    for k in key:
        rec = {
            "record_id": k["record_id"],
            "features": full_cicflowmeter_features(k["flow_id"]),
            "escalation": build_escalation_digest(flows.loc[k["flow_id"]], bins[k["flow_id"]]),
        }
        assert_no_ground_truth(rec)
        records.append(rec)
    write_and_verify_jsonl(records, input_path)
    print(f"wrote {input_path} ({len(records)} records)")

    write_blind_file([strip_to_blind(r) for r in records], blind_path)
    reloaded = load_blind_file(blind_path)
    verify_blind_file(reloaded)
    all_labels = set(flows["true_label"].unique()) | set(flows["class_predicted"].dropna().unique())
    assert_stripped(reloaded, all_labels)
    assert [r["record_id"] for r in reloaded] == [k["record_id"] for k in key]
    print(f"wrote {blind_path} ({len(reloaded)} records); verify_blind_file + no label/priority/reason/"
          "rule-class-prediction checks PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
