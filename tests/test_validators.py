import pytest

from agents.schema import A5Response, Hypothesis, HypothesisResponse, PredictedRange
from agents.validators import (
    FLOW_LEVEL_FILTER_PREFIX,
    validate_a5_addresses_all_contradictions,
    validate_a5_credits_a_real_hypothesis,
    validate_hypothesis_predictions_are_flow_level,
    validate_hypothesis_response,
)

KNOWN_FEATURES = {"trigger_flow_duration", "trigger_flows_per_src", "trigger_syn_ratio"}


def _hresp(**kw):
    """HypothesisResponse with the self-reported trust triad defaulted --
    these tests are about validators.py's mechanical checks, not about
    the trust fields, so a fixed default keeps every call site short."""
    kw.setdefault("confidence", 0.7)
    kw.setdefault("evidence_support", 0.7)
    kw.setdefault("verification", 0.7)
    return HypothesisResponse(**kw)


def _a5resp(**kw):
    kw.setdefault("confidence", 0.7)
    kw.setdefault("evidence_support", 0.7)
    kw.setdefault("verification", 0.7)
    return A5Response(**kw)


def _hyp(hid="a3_h1", benign=True, prediction="inter-arrival times should be regular",
         profile_features=("trigger_flow_duration",), contradicting=()):
    profile = [PredictedRange(feature=f, expected_max=1000.0) for f in profile_features]
    return Hypothesis(
        hypothesis_id=hid, description="d", benign=benign, prior_plausibility=0.5,
        prediction=prediction, predicted_feature_profile=profile,
        contradicting_claim_ids=list(contradicting),
    )


# ---------- flow-level prediction filter ----------

def test_accepts_a_clean_flow_level_prediction():
    resp = _hresp(claims=[], hypotheses=[_hyp()])
    validate_hypothesis_predictions_are_flow_level(resp, KNOWN_FEATURES)  # must not raise


def test_rejects_feature_not_in_known_vocabulary():
    resp = _hresp(claims=[], hypotheses=[_hyp(profile_features=["totally_made_up"])])
    with pytest.raises(ValueError, match=FLOW_LEVEL_FILTER_PREFIX):
        validate_hypothesis_predictions_are_flow_level(resp, KNOWN_FEATURES)


def test_rejects_payload_content_prediction():
    resp = _hresp(
        claims=[],
        hypotheses=[_hyp(prediction="the packet payload should contain a health-check endpoint")],
    )
    with pytest.raises(ValueError, match=FLOW_LEVEL_FILTER_PREFIX):
        validate_hypothesis_predictions_are_flow_level(resp, KNOWN_FEATURES)


def test_rejects_http_headers_prediction():
    resp = _hresp(
        claims=[], hypotheses=[_hyp(prediction="the response should contain valid HTTP headers")],
    )
    with pytest.raises(ValueError):
        validate_hypothesis_predictions_are_flow_level(resp, KNOWN_FEATURES)


def test_rejects_user_agent_prediction():
    resp = _hresp(
        claims=[], hypotheses=[_hyp(prediction="the user-agent string should match a known crawler")],
    )
    with pytest.raises(ValueError):
        validate_hypothesis_predictions_are_flow_level(resp, KNOWN_FEATURES)


def test_rejects_uri_prediction():
    resp = _hresp(
        claims=[], hypotheses=[_hyp(prediction="the URI requested should match a known health-check path")],
    )
    with pytest.raises(ValueError):
        validate_hypothesis_predictions_are_flow_level(resp, KNOWN_FEATURES)


def test_does_not_false_positive_on_words_containing_uri_as_a_substring():
    # regression: naive substring search on "uri" matches inside "during"
    # and "security" -- these are perfectly good flow-level predictions
    # and must not be rejected.
    resp = _hresp(
        claims=[],
        hypotheses=[_hyp(prediction="during the flow, timing should stay regular for security reasons")],
    )
    validate_hypothesis_predictions_are_flow_level(resp, KNOWN_FEATURES)  # must not raise


def test_plural_forms_are_also_caught():
    resp = _hresp(
        claims=[], hypotheses=[_hyp(prediction="the response headers should be consistent across flows")],
    )
    with pytest.raises(ValueError):
        validate_hypothesis_predictions_are_flow_level(resp, KNOWN_FEATURES)


def test_validate_hypothesis_response_runs_flow_level_check_when_known_features_given():
    resp = _hresp(
        claims=[],
        hypotheses=[
            _hyp("a3_h1", prediction="the payload should reveal the request type"),
            _hyp("a3_h2", benign=True),
            _hyp("a3_h3", benign=False),
        ],
    )
    with pytest.raises(ValueError):
        validate_hypothesis_response("a3", resp, KNOWN_FEATURES)


def test_validate_hypothesis_response_skips_flow_level_check_when_no_known_features_given():
    # backward-compatible default: an empty/omitted known_feature_names
    # means "don't run this check" -- used only where a caller genuinely
    # has no record context (not the case for a3/a4 in the real pipeline,
    # which always pass a real set).
    resp = _hresp(
        claims=[],
        hypotheses=[
            _hyp("a3_h1", prediction="the payload should reveal the request type"),
            _hyp("a3_h2", benign=True),
            _hyp("a3_h3", benign=False),
        ],
    )
    validate_hypothesis_response("a3", resp)  # must not raise -- no known_feature_names given


# ---------- A5 contradiction disposal (fix 1) ----------

def test_a5_passes_when_all_contradicting_claims_are_cited():
    resp = _a5resp(
        benign_plausibility=0.6, confidence=0.7,
        cited_claim_ids=["a1_c1", "a4_c4"], rationale="r",
    )
    validate_a5_addresses_all_contradictions(resp, {"a4_c4"})  # must not raise


def test_a5_fails_when_a_contradicting_claim_is_not_cited():
    resp = _a5resp(
        benign_plausibility=0.75, confidence=0.85,
        cited_claim_ids=["a1_c1", "a3_h1"], rationale="r",
    )
    with pytest.raises(ValueError, match="a4_c4"):
        validate_a5_addresses_all_contradictions(resp, {"a4_c4"})


def test_a5_passes_trivially_when_nothing_is_required():
    resp = _a5resp(
        benign_plausibility=0.5, confidence=0.5, cited_claim_ids=["a1_c1"], rationale="r",
    )
    validate_a5_addresses_all_contradictions(resp, set())  # must not raise


def test_a5_must_address_the_union_across_multiple_hypotheses():
    resp = _a5resp(
        benign_plausibility=0.5, confidence=0.5, cited_claim_ids=["a1_c1", "a2_c2"], rationale="r",
    )
    with pytest.raises(ValueError, match="a4_c9"):
        validate_a5_addresses_all_contradictions(resp, {"a2_c2", "a4_c9"})


# ---------- A5 credited_hypothesis_id (fix 4) ----------

def test_a5_credited_hypothesis_none_is_always_valid():
    resp = _a5resp(
        benign_plausibility=0.1, confidence=0.5, cited_claim_ids=["a1_c1"], rationale="r",
        credited_hypothesis_id=None,
    )
    validate_a5_credits_a_real_hypothesis(resp, {"a3_h1", "a4_h1"})  # must not raise


def test_a5_credited_hypothesis_must_be_one_actually_shown():
    resp = _a5resp(
        benign_plausibility=0.6, confidence=0.5, cited_claim_ids=["a1_c1"], rationale="r",
        credited_hypothesis_id="a3_h9",
    )
    with pytest.raises(ValueError, match="a3_h9"):
        validate_a5_credits_a_real_hypothesis(resp, {"a3_h1", "a4_h1"})


def test_a5_credited_hypothesis_passes_when_real():
    resp = _a5resp(
        benign_plausibility=0.6, confidence=0.5, cited_claim_ids=["a1_c1"], rationale="r",
        credited_hypothesis_id="a4_h1",
    )
    validate_a5_credits_a_real_hypothesis(resp, {"a3_h1", "a4_h1"})  # must not raise
