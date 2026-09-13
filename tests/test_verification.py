from agents.schema import Claim, FeatureReference, Relation
from agents.verification import MatchOutcome, counts_by_agent, match_claims, render_verification_summary


def _claim(claim_id, feature, relation, value=None):
    ref = FeatureReference(name=feature, asserted_value=value, relation=relation)
    return Claim(claim_id=claim_id, statement="s", referenced_features=[ref], confidence=0.5)


def test_same_feature_same_direction_corroborated():
    chain = [_claim("a1_c1", "syn_ratio", Relation.GREATER_THAN_TYPICAL, 0.5)]
    a4 = [_claim("a4_c1", "syn_ratio", Relation.GREATER_THAN_TYPICAL, 0.5)]
    results = match_claims(chain, a4)
    assert results[0].outcome == MatchOutcome.CORROBORATED


def test_same_feature_opposite_direction_contradicted():
    chain = [_claim("a1_c1", "syn_ratio", Relation.GREATER_THAN_TYPICAL, 0.5)]
    a4 = [_claim("a4_c1", "syn_ratio", Relation.LESS_THAN_TYPICAL, 0.1)]
    results = match_claims(chain, a4)
    assert results[0].outcome == MatchOutcome.CONTRADICTED


def test_present_absent_are_opposites():
    chain = [_claim("a1_c1", "fwd_psh_flags", Relation.PRESENT)]
    a4 = [_claim("a4_c1", "fwd_psh_flags", Relation.ABSENT)]
    results = match_claims(chain, a4)
    assert results[0].outcome == MatchOutcome.CONTRADICTED


def test_equals_conflicts_with_directional_claim():
    chain = [_claim("a1_c1", "dst_port", Relation.EQUALS, 8080.0)]
    a4 = [_claim("a4_c1", "dst_port", Relation.GREATER_THAN_TYPICAL, 9000.0)]
    results = match_claims(chain, a4)
    assert results[0].outcome == MatchOutcome.CONTRADICTED


def test_unmentioned_feature_is_uncorroborated_not_contradicted():
    chain = [_claim("a1_c1", "syn_ratio", Relation.GREATER_THAN_TYPICAL, 0.5)]
    a4 = [_claim("a4_c1", "bwd_pkt_len_mean", Relation.LESS_THAN_TYPICAL, 12.0)]
    results = match_claims(chain, a4)
    assert results[0].outcome == MatchOutcome.UNCORROBORATED


def test_unrelated_relation_pair_is_uncorroborated_not_a_clash():
    # PRESENT vs GREATER_THAN_TYPICAL on the same feature answers different
    # questions -- neither confirms nor clashes mechanically.
    chain = [_claim("a1_c1", "fwd_psh_flags", Relation.PRESENT)]
    a4 = [_claim("a4_c1", "fwd_psh_flags", Relation.GREATER_THAN_TYPICAL, 3.0)]
    results = match_claims(chain, a4)
    assert results[0].outcome == MatchOutcome.UNCORROBORATED


def test_counts_by_agent_groups_by_claim_id_prefix():
    chain = [
        _claim("a1_c1", "syn_ratio", Relation.GREATER_THAN_TYPICAL, 0.5),
        _claim("a2_c1", "down_up_ratio", Relation.GREATER_THAN_TYPICAL, 2.0),
    ]
    a4 = [
        _claim("a4_c1", "syn_ratio", Relation.GREATER_THAN_TYPICAL, 0.5),  # corroborates a1
        _claim("a4_c2", "down_up_ratio", Relation.LESS_THAN_TYPICAL, 0.1),  # contradicts a2
    ]
    results = match_claims(chain, a4)
    by_agent = counts_by_agent(results)
    assert by_agent["a1"].corroborated == 1
    assert by_agent["a1"].V == 1.0
    assert by_agent["a2"].contradicted == 1
    assert by_agent["a2"].V == 0.0


def test_render_verification_summary_handles_empty():
    assert "no checkable claims" in render_verification_summary({})


def test_a_corroboration_beats_a_contradiction_from_a_different_claim():
    # if one A4 claim corroborates and another contradicts the same
    # feature, corroboration should win (best_outcome logic prefers it).
    chain = [_claim("a1_c1", "flow_iat_std", Relation.GREATER_THAN_TYPICAL, 0.02)]
    a4 = [
        _claim("a4_c1", "flow_iat_std", Relation.LESS_THAN_TYPICAL, 0.001),
        _claim("a4_c2", "flow_iat_std", Relation.GREATER_THAN_TYPICAL, 0.02),
    ]
    results = match_claims(chain, a4)
    assert results[0].outcome == MatchOutcome.CORROBORATED
