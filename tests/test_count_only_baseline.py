"""Count-only baseline: reads no label, uses the same grounding path as the
pipeline, and is deterministic. Also pins the corrected re-score path that the
baseline report uses for the agent side."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval.count_only_baseline as cob
from agents.escalation_grounding import (
    MIN_NEIGHBOURHOOD_SIZE,
    FeatureNeighbourhood,
    compute_feature_neighbourhood,
    observed_escalation_values,
)
from agents.schema import VerdictLabel, derive_verdict

BLIND = Path("results/agent_input_20_blind.jsonl")


def _first_blind_features():
    with open(BLIND, encoding="utf-8") as f:
        return json.loads(f.readline())["features"]


def test_baseline_takes_features_only_no_label_parameter():
    params = list(inspect.signature(cob.count_only_neighbourhood).parameters)
    assert params == ["features"]


def test_baseline_ignores_a_label_smuggled_into_features():
    feats = _first_blind_features()
    with_label = dict(feats, Label="DoS Hulk", is_attack=True)
    a = cob.count_only_neighbourhood(feats).neighbourhood_size
    b = cob.count_only_neighbourhood(with_label).neighbourhood_size
    assert a == b


def test_baseline_uses_the_pipeline_grounding_path(monkeypatch):
    calls = []
    real = cob.compute_feature_neighbourhood

    def spy(observed, *a, **k):
        calls.append(observed)
        return real(observed, *a, **k)

    monkeypatch.setattr(cob, "compute_feature_neighbourhood", spy)
    feats = _first_blind_features()
    cob.count_only_neighbourhood(feats)
    assert len(calls) == 1
    assert calls[0] == observed_escalation_values(feats)


def test_baseline_count_equals_pipeline_neighbourhood_count():
    feats = _first_blind_features()
    expected = compute_feature_neighbourhood(observed_escalation_values(feats)).neighbourhood_size
    assert cob.count_only_neighbourhood(feats).neighbourhood_size == expected


def test_baseline_is_deterministic():
    feats = _first_blind_features()
    counts = {cob.count_only_neighbourhood(feats).neighbourhood_size for _ in range(3)}
    assert len(counts) == 1


def test_flag_is_strictly_below_the_cutoff():
    assert cob.CUTOFF == MIN_NEIGHBOURHOOD_SIZE == 30
    assert cob.count_only_flag(29) is True
    assert cob.count_only_flag(30) is False
    assert cob.count_only_flag(0, cutoff=0) is False


def _recorded(bp, neighbourhood_size, credited_count):
    return {
        "a5": {
            "benign_plausibility": bp, "confidence": 0.8, "evidence_support": 0.8, "verification": 0.8,
            "credited_hypothesis_id": "h1", "cited_claim_ids": ["a1_c1"], "rationale": "x",
        },
        "hypothesis_support": {"h1": {"matching_profile_count": credited_count}},
        "escalation_neighbourhood": {
            "features_used": ["Flow Duration"], "band_description": "t",
            "neighbourhood_size": neighbourhood_size, "benign_population_size": 1000,
            "ungrounded": neighbourhood_size < 30, "min_required": 30,
        },
    }


def test_rescore_at_old_cap_reproduces_recorded_score():
    rec = _recorded(bp=0.30, neighbourhood_size=5, credited_count=100)
    bp, _, _ = cob.corrected_agent_bp(rec, 0.30, 0.30)
    assert bp == pytest.approx(0.30)


def test_rescore_at_corrected_cap_clamps_the_boundary_value_under_threshold():
    rec = _recorded(bp=0.30, neighbourhood_size=5, credited_count=100)
    bp, zero_fired, ung_fired = cob.corrected_agent_bp(rec, 0.29, 0.29)
    assert ung_fired is True and zero_fired is False
    assert bp == pytest.approx(0.29)
    assert derive_verdict(bp) is VerdictLabel.ANOMALOUS_AND_UNEXPLAINED


def test_rescore_leaves_grounded_scores_alone():
    rec = _recorded(bp=0.85, neighbourhood_size=200, credited_count=100)
    bp, _, ung_fired = cob.corrected_agent_bp(rec, 0.29, 0.29)
    assert ung_fired is False
    assert bp == pytest.approx(0.85)


def test_rescore_zero_support_clamp_uses_matching_count_zero():
    rec = _recorded(bp=0.9, neighbourhood_size=200, credited_count=0)
    bp, zero_fired, _ = cob.corrected_agent_bp(rec, 0.29, 0.29)
    assert zero_fired is True
    assert bp == pytest.approx(0.29)
