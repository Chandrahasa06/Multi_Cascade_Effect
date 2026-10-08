"""Tests for the eighth attempt (eval/run_grounded_only.py, a5_verdict_security_v1):
the count appears nowhere in the prompt, no cap is applied, only count >= 30 records are
selected, no label reaches the prompt, and time and history use earlier data only."""
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents import evidence_signals as es
from agents.escalation_grounding import MIN_NEIGHBOURHOOD_SIZE
from agents.prompts import a5_verdict_security_v1 as prompt_module
from eval import run_grounded_only as rg

HOUR = rg.HOUR_US


def _ts(h, m=0):
    return int(datetime(2017, 7, 5, h, m, tzinfo=timezone.utc).timestamp() * 1e6)


# ---------- record selection ----------

def test_selects_only_records_at_or_above_the_cutoff():
    rows = [{"record_id": "a", "count_loo": 29}, {"record_id": "b", "count_loo": 30}, {"record_id": "c", "count_loo": 500}]
    assert [r["record_id"] for r in rg.select_records(rows)] == ["b", "c"]
    assert MIN_NEIGHBOURHOOD_SIZE == 30


def test_selection_ignores_any_label_field():
    rows = [{"record_id": "a", "count_loo": 40, "true_label": "DoS Hulk"}, {"record_id": "b", "count_loo": 10, "is_attack": False}]
    assert [r["record_id"] for r in rg.select_records(rows)] == ["a"]


# ---------- no count in the prompt ----------

def test_context_builder_has_no_count_parameter():
    names = list(inspect.signature(rg.build_context).parameters)
    assert names == ["features", "tod", "history", "percentiles", "direction"]
    assert not any("count" in n or "neighbour" in n for n in names)


def test_prompt_builder_has_no_count_or_support_parameter():
    names = list(inspect.signature(prompt_module.build_prompt).parameters)
    assert not any(k in n for n in names for k in ("count", "support", "neighbour", "evidence_block", "label"))


def _context():
    pct = {f: 50.0 for f in es.FEATURES}
    direc = {f: "at benign median" for f in es.FEATURES}
    hist = [{"bucket": k, "flows": 3, "distinct_dst_ports": 1, "top_dst_ports": [80], "packets": 9.0} for k in range(1, 7)]
    return rg.build_context({"Destination Port": 80.0}, "10:15", hist, pct, direc)


def test_context_contains_no_count_phrasing():
    low = _context().lower()
    assert not [p for p in rg.FORBIDDEN_PHRASES if p in low]


def test_check_no_count_rejects_count_phrasing():
    with pytest.raises(AssertionError):
        rg.check_no_count("Of the 529918 benign flows in the reference window, 31 are close", 31, set())
    with pytest.raises(AssertionError):
        rg.check_no_count("... of which 31 are within 2x of this flow", 31, set())


def test_check_no_count_rejects_the_count_value_outside_legitimate_numbers():
    with pytest.raises(AssertionError):
        rg.check_no_count("Something about 4321 here.", 4321, allowed_numbers={80.0})


def test_check_no_count_reports_a_legitimate_collision_without_failing():
    out = rg.check_no_count("Destination port: 80.", 80, allowed_numbers={80.0})
    assert out  # reported, not hidden


def test_instructions_carry_no_count_and_no_threshold():
    text = prompt_module._INSTRUCTIONS.lower()
    assert not [p for p in rg.FORBIDDEN_PHRASES if p in text]
    assert "0.3" not in text and "0.29" not in text and "threshold" not in text
    assert "not by itself exculpatory" in text


# ---------- no cap ----------

def test_runner_applies_no_cap():
    src = inspect.getsource(rg)
    assert "apply_ungrounded_neighbourhood_cap" not in src
    assert "apply_empirical_plausibility_cap" not in src
    assert '"cap_applied": False' in src


def test_recorded_run_has_no_clamp_if_present():
    p = Path("results/agent_grounded_only.jsonl")
    if not p.exists():
        pytest.skip("run output not present")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    assert all(r["cap_applied"] is False for r in rows)
    assert all(r["count_loo"] >= MIN_NEIGHBOURHOOD_SIZE for r in rows)
    assert all(r["benign_plausibility"] != 0.29 for r in rows)  # the cap value never appears


# ---------- no label reaches the prompt ----------

def test_context_ignores_a_label_in_the_features():
    pct = {f: 50.0 for f in es.FEATURES}
    direc = {f: "at benign median" for f in es.FEATURES}
    a = rg.build_context({"Destination Port": 80.0}, "10:15", None, pct, direc)
    b = rg.build_context({"Destination Port": 80.0, "Label": "DoS Hulk", "is_attack": True}, "10:15", None, pct, direc)
    assert a == b
    assert "dos hulk" not in b.lower()


def test_source_index_reads_no_label_column():
    pool = pd.DataFrame({"Source IP": ["1.1.1.1"] * 3, "first_ts": [_ts(9), _ts(9, 5), _ts(9, 10)],
                         "Total Fwd Packets": [1, 1, 1], "Total Backward Packets": [1, 1, 1],
                         "Destination Port": [80, 80, 443]})
    idx = rg.SourceIndex(pool)  # no Label column at all: must still work
    assert sum(h["flows"] for h in idx.history("1.1.1.1", _ts(9, 12))) == 3


# ---------- time of day and history ----------

def test_afternoon_hours_recorded_in_12_hour_form_are_corrected():
    assert rg.time_of_day(_ts(3, 14)) == "15:14"
    assert rg.time_of_day(_ts(9, 30)) == "09:30"
    assert rg.time_of_day(_ts(12, 5)) == "12:05"


def test_history_uses_only_flows_strictly_before_the_flow():
    pool = pd.DataFrame({"Source IP": ["9.9.9.9"] * 4,
                         "first_ts": [_ts(10, 0), _ts(10, 20), _ts(10, 30), _ts(10, 35)],
                         "Total Fwd Packets": [1] * 4, "Total Backward Packets": [1] * 4,
                         "Destination Port": [80] * 4})
    hist = rg.SourceIndex(pool).history("9.9.9.9", _ts(10, 30))
    # 10:00 and 10:20 are earlier; 10:30 (same time) and 10:35 (later) are excluded
    assert sum(h["flows"] for h in hist) == 2
    assert hist[1]["flows"] == 1  # 10:20 sits in the 5-10 minute bucket


def test_old_prompts_still_importable():
    from agents.prompts import a5_verdict_v6, a5_verdict_v7
    assert a5_verdict_v6.PROMPT_VERSION == "a5_verdict_v6"
    assert a5_verdict_v7.PROMPT_VERSION == "a5_verdict_v7"
    assert prompt_module.PROMPT_VERSION == "a5_verdict_security_v1"
