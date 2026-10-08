"""Part 1: raw (pre-clamp) A5 scores are read correctly from the saved arm runs."""
import json
from pathlib import Path

import pytest

from eval.cap_suppressed_scores import capped_records


def test_capped_records_returns_raw_and_clamped_for_capped_rows_only():
    rows = [
        {"record_id": "a", "ungrounded_cap_fired": True, "zero_support_cap_fired": False,
         "benign_plausibility_raw": 0.85, "benign_plausibility": 0.29, "count_loo": 0, "rationale": "r"},
        {"record_id": "b", "ungrounded_cap_fired": False, "zero_support_cap_fired": False,
         "benign_plausibility_raw": 0.9, "benign_plausibility": 0.9, "count_loo": 500, "rationale": "r"},
    ]
    out = capped_records(rows)
    assert [o["record_id"] for o in out] == ["a"]
    assert out[0]["raw"] == 0.85 and out[0]["clamped"] == 0.29


@pytest.mark.parametrize("arm", ["m", "mt", "a"])
def test_saved_runs_raw_scores_are_consistent_with_the_cap(arm):
    p = Path(f"results/agent_run_arm_{arm}.jsonl")
    if not p.exists():
        pytest.skip("arm output not present")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    for r in rows:
        if r["ungrounded_cap_fired"]:
            # the clamp fired, so the stored score is the cap and the raw score was above it
            assert r["benign_plausibility"] == pytest.approx(0.29)
            assert r["benign_plausibility_raw"] > 0.29
            assert r["count_loo"] < 30
        elif not r["zero_support_cap_fired"]:
            # no clamp: stored and raw scores are the same number
            assert r["benign_plausibility"] == pytest.approx(r["benign_plausibility_raw"])
