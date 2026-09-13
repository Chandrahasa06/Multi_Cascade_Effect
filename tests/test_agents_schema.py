import pytest
from pydantic import ValidationError

from agents.schema import (
    A5Response,
    Claim,
    FeatureReference,
    Hypothesis,
    HypothesisResponse,
    PredictedRange,
    Relation,
    VerdictLabel,
    derive_verdict,
)


def test_feature_reference_requires_value_for_value_relations():
    with pytest.raises(ValidationError):
        FeatureReference(name="syn_ratio", relation=Relation.GREATER_THAN_TYPICAL)


def test_feature_reference_allows_missing_value_for_presence_relations():
    ref = FeatureReference(name="fwd_psh_flags", relation=Relation.PRESENT)
    assert ref.asserted_value is None


def test_feature_reference_accepts_value_when_given():
    ref = FeatureReference(
        name="syn_ratio", asserted_value=0.5, relation=Relation.GREATER_THAN_TYPICAL
    )
    assert ref.asserted_value == 0.5


def test_claim_confidence_bounds():
    ref = FeatureReference(name="x", asserted_value=1.0, relation=Relation.EQUALS)
    with pytest.raises(ValidationError):
        Claim(claim_id="a1_c1", statement="s", referenced_features=[ref], confidence=1.5)
    with pytest.raises(ValidationError):
        Claim(claim_id="a1_c1", statement="s", referenced_features=[ref], confidence=-0.1)
    Claim(claim_id="a1_c1", statement="s", referenced_features=[ref], confidence=0.5)


_PROFILE = [PredictedRange(feature="trigger_dst_port", expected_min=1.0, expected_max=65535.0)]


def test_predicted_range_requires_at_least_one_bound():
    with pytest.raises(ValidationError):
        PredictedRange(feature="trigger_flow_duration")


def test_predicted_range_rejects_min_greater_than_max():
    with pytest.raises(ValidationError):
        PredictedRange(feature="trigger_flow_duration", expected_min=10.0, expected_max=1.0)


def test_predicted_range_accepts_one_sided_bounds():
    assert PredictedRange(feature="trigger_flows_per_src", expected_max=500.0).expected_min is None
    assert PredictedRange(feature="trigger_flows_per_src", expected_min=1.0).expected_max is None


def test_hypothesis_requires_a_prediction():
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="a3_h1", description="retry loop", benign=True, prior_plausibility=0.6,
            predicted_feature_profile=_PROFILE,
        )


def test_hypothesis_rejects_a_too_short_placeholder_prediction():
    # min_length blocks a non-answer like "n/a" from satisfying the
    # falsifiable-prediction requirement.
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="a3_h1", description="retry loop", benign=True,
            prior_plausibility=0.6, prediction="n/a", predicted_feature_profile=_PROFILE,
        )


def test_hypothesis_requires_at_least_one_predicted_feature():
    with pytest.raises(ValidationError):
        Hypothesis(
            hypothesis_id="a3_h1", description="retry loop", benign=True, prior_plausibility=0.6,
            prediction="the same destination port should recur across this source's other flows",
            predicted_feature_profile=[],
        )


def test_hypothesis_accepts_a_real_prediction():
    h = Hypothesis(
        hypothesis_id="a3_h1", description="retry loop", benign=True, prior_plausibility=0.6,
        prediction="if this is a retry loop, the same destination port should recur across "
                   "this source's other flows in short succession",
        predicted_feature_profile=_PROFILE,
    )
    assert h.prediction


def test_hypothesis_response_parses_from_json():
    payload = {
        "claims": [
            {
                "claim_id": "a3_c1",
                "statement": "unusual burst",
                "referenced_features": [
                    {"name": "flow_iat_std", "asserted_value": 0.01, "relation": "greater_than_typical"}
                ],
                "confidence": 0.7,
            }
        ],
        "hypotheses": [
            {
                "hypothesis_id": "a3_h1",
                "description": "retry loop",
                "benign": True,
                "prior_plausibility": 0.6,
                "prediction": "inter-arrival times should cluster near a fixed retry interval",
                "predicted_feature_profile": [
                    {"feature": "trigger_flow_iat_mean", "expected_max": 1000.0}
                ],
                "supporting_claim_ids": ["a1_c1"],
                "contradicting_claim_ids": [],
            },
            {
                "hypothesis_id": "a3_h2",
                "description": "health check",
                "benign": True,
                "prior_plausibility": 0.4,
                "prediction": "the destination port should be a well-known service port",
                "predicted_feature_profile": [
                    {"feature": "trigger_dst_port", "expected_min": 1.0, "expected_max": 1024.0}
                ],
                "supporting_claim_ids": [],
                "contradicting_claim_ids": ["a2_c2"],
            },
        ],
        "confidence": 0.7,
        "evidence_support": 0.7,
        "verification": 0.7,
    }
    resp = HypothesisResponse.model_validate(payload)
    assert len(resp.claims) == 1
    assert len(resp.hypotheses) == 2
    assert resp.hypotheses[0].benign is True
    assert resp.hypotheses[0].prediction


def test_a5response_rejects_out_of_range_benign_plausibility():
    with pytest.raises(ValidationError):
        A5Response(
            benign_plausibility=1.2, confidence=0.9, evidence_support=0.9, verification=0.9,
            cited_claim_ids=["a1_c1"], rationale="r",
        )


def test_a5response_accepts_valid_payload():
    resp = A5Response(
        benign_plausibility=0.42, confidence=0.8, evidence_support=0.8, verification=0.8,
        cited_claim_ids=["a1_c1"], rationale="r",
    )
    assert resp.benign_plausibility == 0.42


def test_a5response_credited_hypothesis_id_defaults_to_none():
    resp = A5Response(
        benign_plausibility=0.42, confidence=0.8, evidence_support=0.8, verification=0.8,
        cited_claim_ids=["a1_c1"], rationale="r",
    )
    assert resp.credited_hypothesis_id is None


def test_a5response_credited_hypothesis_id_can_be_set():
    resp = A5Response(
        benign_plausibility=0.42, confidence=0.8, evidence_support=0.8, verification=0.8,
        cited_claim_ids=["a1_c1"], rationale="r",
        credited_hypothesis_id="a3_h1",
    )
    assert resp.credited_hypothesis_id == "a3_h1"


# ---------- derive_verdict: code-side, swept, not model-asserted ----------

def test_derive_verdict_low_plausibility_is_unexplained():
    assert derive_verdict(0.1) == VerdictLabel.ANOMALOUS_AND_UNEXPLAINED


def test_derive_verdict_high_plausibility_is_consistent_with_benign():
    assert derive_verdict(0.9) == VerdictLabel.CONSISTENT_WITH_BENIGN


def test_derive_verdict_middle_is_explicable():
    assert derive_verdict(0.5) == VerdictLabel.ANOMALOUS_BUT_EXPLICABLE


def test_derive_verdict_thresholds_are_overridable_for_sweeping():
    # same score, different threshold choice -> different label: this is
    # the whole point of moving the verdict into code (sweepable).
    assert derive_verdict(0.6, low_threshold=0.2, high_threshold=0.5) == VerdictLabel.CONSISTENT_WITH_BENIGN
    assert derive_verdict(0.6, low_threshold=0.2, high_threshold=0.9) == VerdictLabel.ANOMALOUS_BUT_EXPLICABLE
