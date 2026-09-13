import pytest

from agents.schema import Claim, FeatureReference, Hypothesis, PredictedRange, Relation
from agents.trust import (
    DEFAULT_WEIGHTS,
    check_claim,
    chain_vs_independent,
    compute_trust_decay,
    compute_trust_score,
    confidence_score,
    evidence_score,
)
from agents.verification import VerificationCounts


FEATURES = {"syn_ratio": 0.5, "flow_duration": 1.2, "dst_port": 8080.0}


def _claim(claim_id, feature, value, relation=Relation.EQUALS, confidence=0.8):
    ref = FeatureReference(name=feature, asserted_value=value, relation=relation)
    return Claim(claim_id=claim_id, statement="s", referenced_features=[ref], confidence=confidence)


def test_check_claim_flags_hallucinated_feature():
    claim = _claim("a1_c1", "totally_made_up_feature", 1.0)
    check = check_claim(claim, FEATURES)
    assert check.hallucinated_features == ["totally_made_up_feature"]
    assert not check.survives


def test_check_claim_flags_factual_error():
    claim = _claim("a1_c1", "syn_ratio", 0.9)  # actual is 0.5
    check = check_claim(claim, FEATURES)
    assert check.factual_errors == [("syn_ratio", 0.9, 0.5)]
    assert not check.survives


def test_check_claim_survives_when_value_matches_within_tolerance():
    claim = _claim("a1_c1", "flow_duration", 1.2000001)
    check = check_claim(claim, FEATURES)
    assert check.survives


def test_check_claim_skips_value_check_for_present_relation():
    ref = FeatureReference(name="syn_ratio", relation=Relation.PRESENT)
    claim = Claim(claim_id="a1_c1", statement="s", referenced_features=[ref], confidence=0.5)
    check = check_claim(claim, FEATURES)
    assert check.survives  # no asserted_value given, nothing to contradict


# ---------- trigger-reason-name fallback (regression: stop-point-2 finding) ----------
#
# The selector's trigger-reason feature names and the CICFlowMeter-derived
# `features` dict can share an identical name while disagreeing in scale --
# confirmed live on a real stop-point-2 record: flow_duration's trigger
# reason value was exactly 1,000,000x the `features` value (microseconds
# vs seconds). A claim quoting the trigger reason's own number verbatim is
# accurate and must not be flagged as hallucinated/factually wrong just
# because `features` disagrees under the same name.

def test_check_claim_survives_via_trigger_reason_value_not_in_features():
    # "flow_bytes_per_sec" is a trigger-reason-only name, never a key in
    # the CICFlowMeter features dict (which uses "flow_byts_s" instead).
    claim = _claim("a1_c1", "flow_bytes_per_sec", 4705882.35)
    check = check_claim(claim, FEATURES, trigger_reason_values={"flow_bytes_per_sec": 4705882.35})
    assert check.survives


def test_check_claim_survives_when_matching_trigger_reason_despite_features_disagreeing():
    # same name in both sources, wildly different scale (the real bug):
    # features says 115.993 (seconds), the trigger reason says
    # 115993217.0 (microseconds) for the same underlying flow_duration.
    claim = _claim("a1_c1", "flow_duration", 115993217.0)
    check = check_claim(
        claim,
        {"flow_duration": 115.99321699142456},
        trigger_reason_values={"flow_duration": 115993217.0},
    )
    assert check.survives, "claim matching the trigger reason's own value must not be a factual error"


def test_check_claim_without_trigger_reason_values_still_flags_the_mismatch():
    # same inputs as above, but without the trigger_reason_values fallback
    # -- documents the exact false-positive this fix resolves.
    claim = _claim("a1_c1", "flow_duration", 115993217.0)
    check = check_claim(claim, {"flow_duration": 115.99321699142456})
    assert not check.survives
    assert check.factual_errors


def test_check_claim_flags_hallucination_only_when_absent_from_both_sources():
    claim = _claim("a1_c1", "not_a_real_feature_anywhere", 1.0)
    check = check_claim(claim, FEATURES, trigger_reason_values={"flow_duration": 1.2})
    assert check.hallucinated_features == ["not_a_real_feature_anywhere"]


def test_check_claim_reports_features_value_as_actual_when_both_sources_disagree_and_neither_matches():
    claim = _claim("a1_c1", "flow_duration", 999.0)  # matches neither source
    check = check_claim(
        claim,
        {"flow_duration": 115.99321699142456},
        trigger_reason_values={"flow_duration": 115993217.0},
    )
    assert check.factual_errors == [("flow_duration", 999.0, 115.99321699142456)]


def test_evidence_score_is_fraction_surviving():
    claims = [
        _claim("a1_c1", "syn_ratio", 0.5),  # survives
        _claim("a1_c2", "syn_ratio", 0.9),  # factual error
        _claim("a1_c3", "made_up", 1.0),  # hallucination
    ]
    assert evidence_score(claims, FEATURES) == pytest.approx(1 / 3)


def test_evidence_score_undefined_when_no_claims():
    assert evidence_score([], FEATURES) is None


def test_confidence_score_means_claim_confidence():
    claims = [_claim("a1_c1", "syn_ratio", 0.5, confidence=0.6), _claim("a1_c2", "flow_duration", 1.2, confidence=0.8)]
    assert confidence_score(claims) == pytest.approx(0.7)


def test_confidence_score_falls_back_to_hypothesis_plausibility_when_no_claims():
    hyps = [
        Hypothesis(hypothesis_id="a3_h1", description="d", benign=True, prior_plausibility=0.4,
                   prediction="if true, some measurable feature should behave a specific way",
                   predicted_feature_profile=[PredictedRange(feature="trigger_flow_duration", expected_max=100.0)]),
        Hypothesis(hypothesis_id="a3_h2", description="d", benign=True, prior_plausibility=0.6,
                   prediction="if true, some other measurable feature should behave a specific way",
                   predicted_feature_profile=[PredictedRange(feature="trigger_flow_iat_mean", expected_max=100.0)]),
    ]
    assert confidence_score([], hyps) == pytest.approx(0.5)


def test_confidence_score_undefined_when_nothing_at_all():
    assert confidence_score([], []) is None


def test_trust_score_weighted_sum():
    claims = [_claim("a1_c1", "syn_ratio", 0.5, confidence=0.9)]  # C=0.9, E=1.0
    counts = VerificationCounts(corroborated=1, contradicted=0, uncorroborated=0)  # V=1.0
    score = compute_trust_score(
        agent="a1", record_id="r1", claims=claims, features=FEATURES, verification_counts=counts
    )
    assert score.C == pytest.approx(0.9)
    assert score.E == pytest.approx(1.0)
    assert score.V == pytest.approx(1.0)
    w_c, w_e, w_v = DEFAULT_WEIGHTS
    assert score.T == pytest.approx(w_c * 0.9 + w_e * 1.0 + w_v * 1.0)
    assert score.CTG == pytest.approx(0.9 - score.T)
    assert not score.degenerate


def test_trust_score_renormalises_when_v_undefined():
    claims = [_claim("a1_c1", "syn_ratio", 0.5, confidence=0.9)]  # C=0.9, E=1.0
    counts = VerificationCounts()  # nothing checkable -> V undefined
    score = compute_trust_score(
        agent="a1", record_id="r1", claims=claims, features=FEATURES, verification_counts=counts
    )
    assert score.V is None
    w_c, w_e, _ = DEFAULT_WEIGHTS
    expected = (w_c * 0.9 + w_e * 1.0) / (w_c + w_e)
    assert score.T == pytest.approx(expected)


def test_trust_decay_excludes_negative_deltas_via_max_zero():
    class Fake:
        def __init__(self, agent, t):
            self.agent = agent
            self.T = t

    scores = [Fake("a1", 0.9), Fake("a2", 0.95), Fake("a3", 0.6)]
    decay = compute_trust_decay(scores)
    assert decay.drops == [0.0, pytest.approx(0.35)]
    assert decay.TD == pytest.approx(0.35)
    assert decay.TD_normalised == pytest.approx(0.175)
    assert decay.max_single_drop == pytest.approx(0.35)
    assert decay.max_single_drop_stage == "a2->a3"


def test_trust_decay_none_when_fewer_than_two_scores():
    class Fake:
        agent = "a1"
        T = 0.9

    assert compute_trust_decay([Fake()]) is None


def test_trust_decay_none_when_any_score_undefined():
    class Fake:
        def __init__(self, agent, t):
            self.agent = agent
            self.T = t

    assert compute_trust_decay([Fake("a1", 0.9), Fake("a2", None)]) is None


def test_chain_vs_independent_sign():
    class Fake:
        def __init__(self, agent, t):
            self.agent = agent
            self.T = t

    chain = [Fake("a1", 0.5), Fake("a2", 0.5), Fake("a3", 0.5)]
    a4 = Fake("a4", 0.8)
    assert chain_vs_independent(chain, a4) == pytest.approx(0.3)
