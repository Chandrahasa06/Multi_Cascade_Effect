"""Tests for the reframe arms: no arm reads a label; arm C's derivation rule is fixed
in code; advocate predictions go through the same flow-level filter; old prompt
versions stay importable."""
import inspect

import pytest

from agents import da_advocate, verdict_rules
from agents.a5_reframe import run_arm
from agents.schema import HypothesisResponse, Hypothesis, PredictedRange, TwoSidedA5Response
from agents.validators import FLOW_LEVEL_FILTER_PREFIX
from agents.verdict_rules import ARM_C_RULE, derive_two_sided_verdict, two_sided_state


def _hyp(hid="da_h1", benign=False, prediction="flow duration should be unusually long",
         features=("trigger_flow_duration",), contradicting=()):
    return Hypothesis(
        hypothesis_id=hid, description="d", benign=benign, prior_plausibility=0.5, prediction=prediction,
        predicted_feature_profile=[PredictedRange(feature=f, expected_max=1000.0) for f in features],
        contradicting_claim_ids=list(contradicting),
    )


def _resp(*hyps):
    return HypothesisResponse(claims=[], hypotheses=list(hyps), confidence=0.5, evidence_support=0.5, verification=0.5)


KNOWN = {"trigger_flow_duration", "trigger_flows_per_src"}


# ---------- no label is read by any arm ----------

@pytest.mark.parametrize("fn", [run_arm, da_advocate.run])
def test_arm_entry_points_take_no_label(fn):
    names = list(inspect.signature(fn).parameters)
    assert not any("label" in n.lower() or "attack" in n.lower() for n in names), names


def test_prompt_builders_take_no_label():
    from agents.prompts import a5_verdict_adversarial_v1, a5_verdict_devil_v1, a5_verdict_twosided_v1, da_advocate_v1
    for mod in (a5_verdict_adversarial_v1, a5_verdict_devil_v1, a5_verdict_twosided_v1, da_advocate_v1):
        names = list(inspect.signature(mod.build_prompt).parameters)
        assert not any("label" in n.lower() or "is_attack" in n.lower() for n in names), (mod.__name__, names)


def test_reconstruction_takes_no_label():
    from eval import run_reframe_45
    names = list(inspect.signature(run_reframe_45.reconstruct).parameters)
    assert names == ["rec", "blind"]


# ---------- arm C rule fixed in code ----------

def test_arm_c_rule_is_stated_exactly():
    assert ARM_C_RULE == "attack iff attack_plausibility > benign_plausibility (strict); ties are benign"


@pytest.mark.parametrize("bp,ap,expected", [
    (0.30, 0.30, "benign"),   # tie is benign
    (0.20, 0.60, "attack"),
    (0.60, 0.20, "benign"),
    (0.95, 0.96, "attack"),   # strict, not a margin
    (0.00, 0.00, "benign"),
])
def test_arm_c_derivation_cases(bp, ap, expected):
    assert derive_two_sided_verdict(bp, ap) == expected


def test_arm_c_does_not_use_the_030_threshold():
    # 0.29 vs 0.31: the single-score rule would call the first attack and the second benign,
    # but arm C ignores the threshold entirely and compares the pair
    assert derive_two_sided_verdict(0.29, 0.10) == "benign"
    assert derive_two_sided_verdict(0.31, 0.40) == "attack"


def test_two_sided_state_is_diagnostic_only():
    assert two_sided_state(0.1, 0.1) == "neither plausible"
    assert two_sided_state(0.9, 0.9) == "both plausible"
    assert verdict_rules.DIAGNOSTIC_LOW == 0.30


def test_two_sided_schema_requires_attack_plausibility():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TwoSidedA5Response(benign_plausibility=0.5, confidence=0.5, evidence_support=0.5, verification=0.5,
                           cited_claim_ids=["a1_c1"], rationale="x")


# ---------- advocate: same flow-level filter, attack-only contract ----------

def test_advocate_accepts_clean_attack_hypotheses():
    resp = _resp(_hyp("da_h1"), _hyp("da_h2", prediction="flows per source should be very high",
                                     features=("trigger_flows_per_src",)))
    da_advocate.validate_advocate_response(resp, KNOWN)  # must not raise


def test_advocate_rejects_a_benign_hypothesis():
    resp = _resp(_hyp("da_h1"), _hyp("da_h2", benign=True))
    with pytest.raises(ValueError, match="benign=false"):
        da_advocate.validate_advocate_response(resp, KNOWN)


def test_advocate_rejects_wrong_id_prefix():
    resp = _resp(_hyp("a3_h1"), _hyp("da_h2"))
    with pytest.raises(ValueError, match="da_h"):
        da_advocate.validate_advocate_response(resp, KNOWN)


def test_advocate_predictions_go_through_the_flow_level_filter():
    resp = _resp(_hyp("da_h1", prediction="payloads should contain a login string"), _hyp("da_h2"))
    with pytest.raises(ValueError, match=FLOW_LEVEL_FILTER_PREFIX):
        da_advocate.validate_advocate_response(resp, KNOWN)


def test_advocate_predictions_must_name_known_reference_features():
    resp = _resp(_hyp("da_h1", features=("trigger_not_a_real_feature",)), _hyp("da_h2"))
    with pytest.raises(ValueError, match=FLOW_LEVEL_FILTER_PREFIX):
        da_advocate.validate_advocate_response(resp, KNOWN)


def test_advocate_rejects_contradicting_claims():
    resp = _resp(_hyp("da_h1", contradicting=["a1_c1"]), _hyp("da_h2"))
    with pytest.raises(ValueError, match="contradicting"):
        da_advocate.validate_advocate_response(resp, KNOWN)


def test_run_reframe_applies_the_grounding_path_to_advocate_hypotheses():
    # the advocate's hypotheses are scored by the same evaluate_all + patch used for the chain
    from eval import run_reframe_45
    src = inspect.getsource(run_reframe_45.run_record)
    assert "evaluate_all(list(da_resp.hypotheses)" in src
    assert "patch_hypothesis_support(da_sup, common[\"nb\"])" in src


# ---------- prompts: content and stability ----------

def test_adversarial_prompt_states_non_exculpatory_rule_and_question():
    from agents.prompts import a5_verdict_adversarial_v1 as m
    assert "would a capable attacker plausibly produce this flow" in m._INSTRUCTIONS or "would a capable attacker plausibly produce this flow" in m.__dict__.get("_INSTRUCTIONS", "")
    text = inspect.getsource(m)
    assert "NOT by itself exculpatory" in text


def test_older_prompt_versions_still_importable():
    from agents.prompts import a1_evidence_v5, a3_hypotheses_v5, a5_verdict_v6, a5_verdict_v5
    assert a5_verdict_v6.PROMPT_VERSION == "a5_verdict_v6"
    assert a5_verdict_v5.PROMPT_VERSION == "a5_verdict_v5"
    assert a1_evidence_v5.PROMPT_VERSION == "a1_evidence_v5"
    assert a3_hypotheses_v5.PROMPT_VERSION == "a3_hypotheses_v5"


def test_new_prompt_versions_have_distinct_identifiers():
    from agents.prompts import a5_verdict_adversarial_v1, a5_verdict_devil_v1, a5_verdict_twosided_v1, a5_verdict_v6, da_advocate_v1
    ids = {m.PROMPT_VERSION for m in (a5_verdict_adversarial_v1, a5_verdict_devil_v1, a5_verdict_twosided_v1, a5_verdict_v6, da_advocate_v1)}
    assert len(ids) == 5
