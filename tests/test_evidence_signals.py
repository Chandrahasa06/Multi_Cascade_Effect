"""Tests for agents/evidence_signals.py: the four benign-only signals.

Properties pinned here: no signal reads a label; the reference is benign-only; source
history uses only earlier time; leave-one-out removes exactly one identical copy;
the output is deterministic; prompt versions are routed so old cache keys still hold."""
import inspect

import numpy as np
import pandas as pd
import pytest

from agents import a1_evidence, evidence_signals as es

BUCKET = es.SOURCE_BUCKET_US


def _frame(rows):
    """Synthetic benign-reference frame. Each row: escalation feature values plus
    Source IP, first_ts, packet counts, dst port, Label."""
    out = []
    for i, r in enumerate(rows):
        base = {f: float(r.get("base", 1.0)) for f in es.FEATURES}
        base.update({f: r[f] for f in r.get("set", {})})
        base.update({
            "Source IP": r.get("src", "10.0.0.1"),
            "first_ts": r["t"],
            "Total Fwd Packets": r.get("fwd", 1),
            # 'Total Backward Packets' is also an escalation feature; it keeps the base value here
            "Destination Port": r.get("port", 80),
            "Label": r.get("label", "BENIGN"),
        })
        out.append(base)
    return pd.DataFrame(out)


def _features(value=1.0):
    return {f: float(value) for f in es.FEATURES}


# ---------- benign-only ----------

def test_reference_refuses_attack_rows():
    frame = _frame([{"t": 1}, {"t": 2, "label": "DoS Hulk"}])
    with pytest.raises(ValueError, match="benign"):
        es.assert_benign_only(frame)


def test_build_reference_keeps_only_benign_rows_and_asserts():
    frame = _frame([{"t": i} for i in range(10)] + [{"t": 99, "label": "PortScan"}])
    ref = es.build_reference(frame)
    assert ref.n == 10  # the attack row was filtered out before the assertion


def test_query_ignores_a_label_key_in_the_features():
    ref = es.build_reference(_frame([{"t": i, "base": 1.0 + i} for i in range(40)]))
    plain = es.bundle(ref, _features(3.0), None)
    with_label = es.bundle(ref, dict(_features(3.0), Label="DoS Hulk"), None)
    assert plain.count_all == with_label.count_all
    assert plain.percentiles == with_label.percentiles
    assert plain.neighbours == with_label.neighbours


def test_locate_source_never_reads_the_label_column():
    pool = pd.DataFrame({"f1": [1.0, 2.0], "Source IP": ["1.1.1.1", "2.2.2.2"], "first_ts": [100, 200]})
    features = {"f1": 2.0}
    src, status = es.locate_source(pool, features, ["f1"])  # no Label column: must still work
    assert status == "unique" and src == ("2.2.2.2", 200)


def test_locate_source_reports_ambiguity_instead_of_guessing():
    pool = pd.DataFrame({"f1": [1.0, 1.0], "Source IP": ["1.1.1.1", "2.2.2.2"], "first_ts": [100, 200]})
    src, status = es.locate_source(pool, {"f1": 1.0}, ["f1"])
    assert src is None and status == "ambiguous"


# ---------- source history: earlier time only ----------

def test_source_history_reads_only_earlier_buckets():
    t0 = 10 * BUCKET
    rows = [
        {"t": t0 - 1, "src": "9.9.9.9"},            # bucket 1 (most recent earlier bucket)
        {"t": t0 - BUCKET - 1, "src": "9.9.9.9"},   # bucket 2
        {"t": t0, "src": "9.9.9.9"},                # same instant: excluded
        {"t": t0 + 1, "src": "9.9.9.9"},            # later: excluded
        {"t": t0 - 5, "src": "8.8.8.8"},            # another source: excluded
    ]
    ref = es.build_reference(_frame(rows + [{"t": 1, "src": "7.7.7.7"}]))
    hist = es.source_history(ref, "9.9.9.9", t0)
    assert hist[0]["flows"] == 1 and hist[1]["flows"] == 1
    assert sum(h["flows"] for h in hist) == 2  # the t0 and t0+1 rows are not counted


def test_source_history_unknown_source_is_empty_not_error():
    ref = es.build_reference(_frame([{"t": 1}]))
    hist = es.source_history(ref, "1.2.3.4", 10 * BUCKET)
    assert all(h["flows"] == 0 for h in hist) and len(hist) == es.SOURCE_BUCKETS


# ---------- leave-one-out ----------

def test_leave_one_out_removes_exactly_one_identical_copy():
    # three benign rows identical on the escalation features; the query is one of them
    rows = [{"t": i, "base": 5.0} for i in range(3)] + [{"t": 100 + i, "base": 50.0 + i} for i in range(30)]
    ref = es.build_reference(_frame(rows))
    q = _features(5.0)
    b = es.bundle(ref, q, None)
    assert b.identical == 3
    assert b.count_excl == b.count_all - 1  # one copy removed, two genuine twins kept
    assert b.neighbours and b.neighbours[0]["distance"] <= es.IDENTICAL_TOL  # a twin remains visible


def test_leave_one_out_does_not_change_attack_free_query():
    rows = [{"t": i, "base": 1.0 + i} for i in range(40)]
    ref = es.build_reference(_frame(rows))
    b = es.bundle(ref, _features(1e6), None)  # matches no reference row
    assert b.identical == 0 and b.count_excl == b.count_all


def test_count_matches_pipeline_neighbourhood_rule():
    rows = [{"t": i, "base": 1.0 + 0.1 * i} for i in range(60)]
    ref = es.build_reference(_frame(rows))
    x = np.full(len(es.FEATURES), 1.5)
    count_all, _ = es.count_band(ref, x)
    expected = int(np.all((ref.raw >= 0.75) & (ref.raw <= 3.0), axis=1).sum())
    assert count_all == expected


# ---------- percentiles and direction ----------

def test_percentiles_are_bounded_and_monotone():
    rows = [{"t": i, "base": float(i)} for i in range(1, 101)]
    ref = es.build_reference(_frame(rows))
    low = es.percentiles(ref, np.full(len(es.FEATURES), -1.0))
    high = es.percentiles(ref, np.full(len(es.FEATURES), 1e9))
    assert (low == 0).all() and (high == 100).all()
    mid = es.percentiles(ref, np.full(len(es.FEATURES), 50.0))
    assert ((0 <= mid) & (mid <= 100)).all()


def test_direction_labels_above_and_below_median():
    rows = [{"t": i, "base": float(i)} for i in range(1, 101)]
    ref = es.build_reference(_frame(rows))
    b_hi = es.bundle(ref, _features(1e9), None)
    assert all(v == "extreme high (top 1% of benign)" for v in b_hi.direction.values())


# ---------- determinism and rendering ----------

def test_bundle_is_deterministic():
    rows = [{"t": i, "base": 1.0 + 0.05 * i} for i in range(80)]
    ref = es.build_reference(_frame(rows))
    a = es.render_evidence_block(es.bundle(ref, _features(2.0), ("9.9.9.9", 5 * BUCKET)))
    b = es.render_evidence_block(es.bundle(ref, _features(2.0), ("9.9.9.9", 5 * BUCKET)))
    assert a == b


def test_render_states_no_combination_rule():
    ref = es.build_reference(_frame([{"t": i} for i in range(40)]))
    text = es.render_evidence_block(es.bundle(ref, _features(1.0), None))
    assert "No rule for combining these signals is given" in text
    assert "threshold" not in text.lower()


# ---------- prompt-version routing keeps old cache keys valid ----------

def _capture_prompt_version(monkeypatch):
    seen = {}

    def fake_call(**kwargs):
        seen["prompt_version"] = kwargs["prompt_version"]
        seen["prompt"] = kwargs["prompt"]
        return ("resp", "meta")

    monkeypatch.setattr(a1_evidence.base, "call_structured", fake_call)
    return seen


def _real_record():
    import json
    from pathlib import Path

    from eval.run_blind_pipeline import blind_record_to_escalation_record

    blind = json.loads(Path("results/agent_input_20_blind.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return blind_record_to_escalation_record(blind)


def test_a1_without_block_keeps_v5_prompt_version(monkeypatch):
    seen = _capture_prompt_version(monkeypatch)
    a1_evidence.run(_real_record(), "grounding", use_cache=False)
    assert seen["prompt_version"] == "a1_evidence_v5"


def test_a1_with_block_uses_v6_and_contains_the_block(monkeypatch):
    seen = _capture_prompt_version(monkeypatch)
    a1_evidence.run(_real_record(), "grounding", use_cache=False, evidence_block="EVIDENCE-MARKER")
    assert seen["prompt_version"] == "a1_evidence_v6"
    assert "EVIDENCE-MARKER" in seen["prompt"]


def test_old_prompt_modules_still_importable():
    from agents.prompts import a1_evidence_v5, a3_hypotheses_v5, a5_verdict_v5
    assert a1_evidence_v5.PROMPT_VERSION == "a1_evidence_v5"
    assert a3_hypotheses_v5.PROMPT_VERSION == "a3_hypotheses_v5"
    assert a5_verdict_v5.PROMPT_VERSION == "a5_verdict_v5"


def test_v6_prompts_take_evidence_and_v5_signatures_are_unchanged():
    from agents.prompts import a1_evidence_v5, a1_evidence_v6, a3_hypotheses_v5, a3_hypotheses_v6, a5_verdict_v5, a5_verdict_v6
    assert "evidence_block" not in inspect.signature(a1_evidence_v5.build_prompt).parameters
    assert "evidence_block" in inspect.signature(a1_evidence_v6.build_prompt).parameters
    assert "evidence_block" not in inspect.signature(a3_hypotheses_v5.build_prompt).parameters
    assert "evidence_block" in inspect.signature(a3_hypotheses_v6.build_prompt).parameters
    assert "evidence_block" not in inspect.signature(a5_verdict_v5.build_prompt).parameters
    assert "evidence_block" in inspect.signature(a5_verdict_v6.build_prompt).parameters

