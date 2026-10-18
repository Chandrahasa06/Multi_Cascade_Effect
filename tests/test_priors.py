"""Tests for the priors run (ninth attempt): no computed statistic in any prompt, no cap,
no label reaches a prompt, A1-A4 are freshly generated under new versions, and the
run-specific validators behave."""
import inspect
import json
from pathlib import Path

import pytest

from agents import priors_pipeline as pp
from agents.prompts import a1_priors_v1, a2_priors_v1, a3_priors_v1, a4_priors_v1, a5_priors_v1
from agents.prompts.priors_render_v1 import flow_block, profile_feature_names
from agents.prompts.priors_text_v1 import PRIOR_IDS, PRIOR_STATEMENTS, PRIORS_TEXT
from agents.schema import Hypothesis, PredictedRange, PriorsClaim, PriorsClaimsResponse, PriorsFeatureValue, PriorsHypothesisResponse
from eval import run_priors as rp
from eval.run_blind_pipeline_45_v2 import load_45

MODULES = (a1_priors_v1, a2_priors_v1, a3_priors_v1, a4_priors_v1, a5_priors_v1)


@pytest.fixture(scope="module")
def record():
    return load_45()[0]


# ---------- no computed statistic in any prompt ----------

def test_prior_text_contains_no_forbidden_phrase_or_dataset_name():
    low = PRIORS_TEXT.lower()
    assert not [p for p in rp.ALWAYS_FORBIDDEN + rp.BUILDER_FORBIDDEN if p in low]
    assert "cicids" not in low and "2017" not in low


def test_every_prior_statement_has_a_source():
    assert len(PRIOR_IDS) == len(set(PRIOR_IDS))
    assert all(src.strip() for _sid, src, _t in PRIOR_STATEMENTS)


def test_builder_text_of_all_five_prompts_passes_for_a_real_record(record):
    feats = record["features"]
    flow = flow_block(feats, "10:15")
    for agent, text in rp.builder_prompts(flow, profile_feature_names(feats)).items():
        rp.check_prompt(text, text, set(), {float(v) for v in feats.values()})


def test_check_rejects_count_phrasing_and_a_leaked_count():
    with pytest.raises(AssertionError):
        rp.check_prompt("of which 31 are within 2x of this flow", "x", set(), set())
    with pytest.raises(AssertionError):
        rp.check_prompt("ok", "Neighbours found: 98765.", {98765.0}, set())
    with pytest.raises(AssertionError):
        rp.check_prompt("ok", "This flow is at the 99th percentile.", set(), set())


def test_check_handles_numbers_followed_by_a_full_stop():
    with pytest.raises(AssertionError):
        rp.check_prompt("ok", "The count was 98765.", {98765.0}, set())


def test_flow_block_holds_only_the_flows_own_properties(record):
    text = flow_block(record["features"], "10:15").lower()
    assert not [p for p in rp.ALWAYS_FORBIDDEN + rp.BUILDER_FORBIDDEN if p in text]
    assert "protocol: not recorded" in text


def test_prompt_builders_take_no_statistic_or_label():
    bad = ("count", "support", "neighbour", "percentile", "history", "label", "grounding", "evidence")
    for m in MODULES:
        names = list(inspect.signature(m.build_prompt).parameters)
        assert not [n for n in names for b in bad if b in n.lower()], (m.__name__, names)


# ---------- no cap ----------

def test_runner_applies_no_cap():
    src = inspect.getsource(rp) + inspect.getsource(pp)
    assert "apply_ungrounded_neighbourhood_cap" not in src
    assert "apply_empirical_plausibility_cap" not in src


def test_recorded_run_has_no_clamp_if_present():
    p = Path("results/agent_priors.jsonl")
    if not p.exists():
        pytest.skip("run output not present")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    for r in rows:
        assert r["cap_applied"] is False
        assert r["benign_plausibility"] == r["a5"]["benign_plausibility"]


# ---------- no label reaches a prompt ----------

def test_runner_never_reads_the_key_files():
    src = inspect.getsource(rp)
    assert "agent_key" not in src and "true_label" not in src and "is_attack" not in src


def test_blind_records_carry_no_label(record):
    assert "Label" not in record["features"]
    assert not any("label" in k.lower() for k in record)


# ---------- A1-A4 are freshly generated ----------

def test_runner_never_reads_recorded_agent_outputs():
    src = inspect.getsource(rp)
    for old in ("agent_run_45_v2", "agent_run_45_v3", "agent_run_20", "agent_run_10c", "agent_run_45_reframe"):
        assert old not in src


def test_all_five_prompt_versions_are_new():
    assert [m.PROMPT_VERSION for m in MODULES] == [f"a{i}_priors_v1" for i in range(1, 6)]


def test_recorded_run_used_the_new_versions_if_present():
    p = Path("results/agent_priors.jsonl")
    if not p.exists():
        pytest.skip("run output not present")
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            assert r["prompt_versions"] == {f"a{i}": f"a{i}_priors_v1" for i in range(1, 6)}


def test_old_prompt_versions_still_importable():
    from agents.prompts import a1_evidence_v5, a3_hypotheses_v5, a4_replication_v6, a5_verdict_v6, a5_verdict_v7
    assert a5_verdict_v7.PROMPT_VERSION == "a5_verdict_v7"


# ---------- run-specific validators ----------

def _claim(cid, name="Flow Duration"):
    return PriorsClaim(claim_id=cid, statement="s", referenced_features=[PriorsFeatureValue(name=name, value=1.0)], confidence=0.5)


def test_claim_schema_has_no_relation_field():
    assert "relation" not in PriorsFeatureValue.model_fields


def test_claims_must_use_the_flows_own_feature_names():
    pp.validate_claims([_claim("a1_c1")], "a1_c", {"Flow Duration"})
    with pytest.raises(ValueError):
        pp.validate_claims([_claim("a1_c1", "made_up")], "a1_c", {"Flow Duration"})
    with pytest.raises(ValueError):
        pp.validate_claims([_claim("x1")], "a1_c", {"Flow Duration"})


def _hyp(hid, benign):
    return Hypothesis(hypothesis_id=hid, description="d", benign=benign, prior_plausibility=0.5,
                      prediction="flow duration is long here", contradicting_claim_ids=[],
                      predicted_feature_profile=[PredictedRange(feature="Flow Duration", expected_min=1.0)])


def test_explanations_need_both_kinds_and_three_to_five():
    ok = PriorsHypothesisResponse(claims=[], hypotheses=[_hyp("a3_h1", True), _hyp("a3_h2", False), _hyp("a3_h3", True)],
                                  confidence=0.5, evidence_support=0.5, verification=0.5)
    pp.validate_explanations(ok, "a3_h", {"Flow Duration"})
    two = ok.model_copy(update={"hypotheses": ok.hypotheses[:2]})
    with pytest.raises(ValueError):
        pp.validate_explanations(two, "a3_h", {"Flow Duration"})
    all_benign = ok.model_copy(update={"hypotheses": [_hyp(f"a3_h{i}", True) for i in range(3)]})
    with pytest.raises(ValueError):
        pp.validate_explanations(all_benign, "a3_h", {"Flow Duration"})
