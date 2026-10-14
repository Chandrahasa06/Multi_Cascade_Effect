"""Tests for the priors-v3 run (eleventh attempt): the checklist is required and validated, no
computed statistic in any prompt, no flag column referenced in the conditions, no cap, no
label reaches any prompt, and the code-side evaluator matches the written conditions."""
import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents import priors_pipeline_v3 as pp3
from agents.conditions_v3 import evaluate
from agents.prompts import a1_priors_v3, a2_priors_v3, a3_priors_v3, a4_priors_v3, a5_priors_v3
from agents.prompts import a1_priors_v1, a1_priors_v2
from agents.prompts.priors_render_v3 import FLAG_COLUMNS, flow_block, profile_feature_names, shown_features
from agents.prompts.priors_text_v3 import CONDITION_IDS, CONDITIONS, PRIORS_TEXT
from agents.schema import ConditionCheck, V3A5Response, V3Claim
from eval import run_priors_v3 as rp3
from eval.run_blind_pipeline_45_v2 import load_45

MODULES = (a1_priors_v3, a2_priors_v3, a3_priors_v3, a4_priors_v3, a5_priors_v3)
NAMES = {"Init_Win_bytes_backward", "Total Backward Packets", "Destination Port"}


@pytest.fixture(scope="module")
def record():
    return load_45()[0]


def _row(cid, holds="no", obs=(("Destination Port", 80.0),)):
    return {"condition_id": cid, "holds": holds, "observed": [{"name": n, "value": v} for n, v in obs]}


def _a5(checklist):
    return V3A5Response(checklist=checklist, benign_plausibility=0.5, confidence=0.5, evidence_support=0.5,
                        verification=0.5, cited_claim_ids=["a1_c1"], rationale="r")


# ---------- the checklist is required and validated ----------

def test_a5_response_without_checklist_is_rejected_by_the_schema():
    with pytest.raises(ValidationError):
        V3A5Response(benign_plausibility=0.5, confidence=0.5, evidence_support=0.5, verification=0.5,
                     cited_claim_ids=["a1_c1"], rationale="r")


def test_checklist_row_without_a_value_is_rejected_by_the_schema():
    with pytest.raises(ValidationError):
        ConditionCheck(condition_id="C1", holds="no", observed=[])


def test_complete_checklist_passes():
    r = _a5([_row(c) for c in CONDITION_IDS])
    pp3.validate_checklist(r.checklist, NAMES)


def test_missing_condition_is_rejected():
    r = _a5([_row(c) for c in CONDITION_IDS if c != "C7"])
    with pytest.raises(ValueError, match="missing"):
        pp3.validate_checklist(r.checklist, NAMES)


def test_duplicate_and_unknown_conditions_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        pp3.validate_checklist(_a5([_row(c) for c in CONDITION_IDS] + [_row("C1")]).checklist, NAMES)
    with pytest.raises(ValueError, match="unknown"):
        pp3.validate_checklist(_a5([_row(c) for c in CONDITION_IDS] + [_row("C99")]).checklist, NAMES)


def test_value_name_must_be_a_measured_value_or_protocol():
    rows = [_row(c) for c in CONDITION_IDS]
    rows[0] = _row("C1", obs=(("made up", 1.0),))
    with pytest.raises(ValueError, match="not names"):
        pp3.validate_checklist(_a5(rows).checklist, NAMES)
    rows[0] = _row("C1", "not_applicable", obs=(("Protocol", 17.0),))
    pp3.validate_checklist(_a5(rows).checklist, NAMES)


def test_a1_claims_must_cite_a_condition():
    c = V3Claim(claim_id="a1_c1", statement="s", condition_ids=[], confidence=0.5)
    with pytest.raises(ValueError):
        pp3.validate_condition_refs([c], require=True)
    pp3.validate_condition_refs([c], require=False)
    with pytest.raises(ValueError):
        pp3.validate_condition_refs([c.model_copy(update={"condition_ids": ["C42"]})], require=False)


# ---------- conditions ----------

def test_no_condition_references_a_flag_column():
    for _cid, _scope, test, meaning, _src in CONDITIONS:
        for col in FLAG_COLUMNS:
            assert col not in test and col not in meaning


def test_conditions_are_sourced_and_ids_are_stable():
    assert CONDITION_IDS == tuple(f"C{i}" for i in range(1, 13))
    assert all(src.strip() for *_, src in CONDITIONS)
    assert "not by itself exculpatory" in PRIORS_TEXT
    assert "cicids" not in PRIORS_TEXT.lower() and "2017" not in PRIORS_TEXT


def test_code_evaluator_scopes_by_protocol(record):
    tcp = evaluate(record["features"], "TCP")
    udp = evaluate(record["features"], "UDP")
    assert set(tcp) == set(CONDITION_IDS)
    assert tcp["C12"] == "not_applicable" and udp["C1"] == "not_applicable" and udp["C4"] == "not_applicable"


# ---------- no statistic, no label, no cap ----------

def test_builder_text_of_all_five_prompts_passes_for_a_real_record(record):
    feats = record["features"]
    flow = flow_block(feats, "TCP", "10:15")
    for text in rp3.builder_prompts(flow, profile_feature_names(feats)).values():
        rp3.check_prompt(text, text, set(), {float(v) for v in shown_features(feats).values()})


def test_check_rejects_count_phrasing_flag_columns_and_leaked_counts():
    with pytest.raises(AssertionError):
        rp3.check_prompt("of which 31 are within 2x", "ok", set(), set())
    with pytest.raises(AssertionError):
        rp3.check_prompt("ok", "ACK Flag Count = 0", set(), set())
    with pytest.raises(AssertionError):
        rp3.check_prompt("ok", "The count was 98765.", {98765.0}, set())


def test_prompt_builders_take_no_statistic_or_label():
    bad = ("count", "support", "neighbour", "percentile", "history", "label", "grounding", "evidence")
    for m in MODULES:
        names = list(inspect.signature(m.build_prompt).parameters)
        assert not [n for n in names for b in bad if b in n.lower()], (m.__name__, names)


def test_runner_has_no_cap_reads_no_label_or_old_output():
    src = inspect.getsource(rp3) + inspect.getsource(pp3)
    assert "apply_ungrounded_neighbourhood_cap" not in src and "apply_empirical_plausibility_cap" not in src
    assert "agent_key" not in src and "true_label" not in src and "is_attack" not in src
    assert "eval20_records" not in src
    for old in ("agent_run_45_v2", "agent_run_45_v3", "agent_priors.jsonl", "agent_priors_v2.jsonl"):
        assert old not in src


def test_versions_are_new_and_older_ones_importable():
    assert [m.PROMPT_VERSION for m in MODULES] == [f"a{i}_priors_v3" for i in range(1, 6)]
    assert a1_priors_v1.PROMPT_VERSION == "a1_priors_v1" and a1_priors_v2.PROMPT_VERSION == "a1_priors_v2"


def test_recorded_run_if_present():
    p = Path("results/agent_priors_v3.jsonl")
    if not p.exists():
        pytest.skip("run output not present")
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            assert r["cap_applied"] is False
            assert r["benign_plausibility"] == r["a5"]["benign_plausibility"]
            assert sorted(c["condition_id"] for c in r["a5"]["checklist"]) == sorted(CONDITION_IDS)
            assert all(c["observed"] for c in r["a5"]["checklist"])
            assert not any(m["cached"] for m in r["stage_meta"].values())
