import pandas as pd
import pytest

from agents.grounding import (
    ZERO_SUPPORT_PLAUSIBILITY_CAP,
    apply_empirical_plausibility_cap,
    evaluate_hypothesis_support,
    reference_feature_names,
    render_hypothesis_support,
)
from agents.prompts.render import TRIGGER_FEATURE_PREFIX
from agents.schema import A5Response, Hypothesis, PredictedRange
from controlplane.reference import reference_from_dataframe


def _benign_df():
    return pd.DataFrame({
        "label": ["BENIGN"] * 100,
        "first_ts": range(100), "last_ts": range(100), "closure_reason": ["x"] * 100,
        "flows_per_src": [float(i) for i in range(100)],  # 0..99
        "syn_ratio": [0.0] * 100,
    })


def _hyp(hid, profile, contradicting=()):
    return Hypothesis(
        hypothesis_id=hid, description="d", benign=True, prior_plausibility=0.5,
        prediction="if true, some feature should behave a specific way",
        predicted_feature_profile=profile, contradicting_claim_ids=list(contradicting),
    )


def test_reference_feature_names_are_prefixed():
    df = _benign_df()
    ref = reference_from_dataframe(df)
    names = reference_feature_names(ref)
    assert names == {f"{TRIGGER_FEATURE_PREFIX}flows_per_src", f"{TRIGGER_FEATURE_PREFIX}syn_ratio"}


def test_evaluate_hypothesis_support_counts_matching_and_close_flows():
    df = _benign_df()
    hyp = _hyp("a3_h1", [PredictedRange(feature=f"{TRIGGER_FEATURE_PREFIX}flows_per_src", expected_min=0, expected_max=50)])
    # this flow's own observed value is 40 -- flows within [20, 80] (2x
    # band) among the matching [0,50] flows should count as "close".
    support = evaluate_hypothesis_support(hyp, {f"{TRIGGER_FEATURE_PREFIX}flows_per_src": 40.0}, df)
    assert support.matching_profile_count == 51  # 0..50 inclusive
    assert support.checked_features == [f"{TRIGGER_FEATURE_PREFIX}flows_per_src"]
    assert support.close_to_observed_count == 31  # 20..50 inclusive, within [0,50] match


def test_evaluate_hypothesis_support_zero_matches_for_impossible_profile():
    df = _benign_df()
    hyp = _hyp("a3_h1", [PredictedRange(feature=f"{TRIGGER_FEATURE_PREFIX}syn_ratio", expected_min=5.0, expected_max=10.0)])
    support = evaluate_hypothesis_support(hyp, {f"{TRIGGER_FEATURE_PREFIX}syn_ratio": 0.5}, df)
    assert support.matching_profile_count == 0
    # an observed value WAS available to check against (checked_features
    # is non-empty) -- there just were zero matching flows to check it
    # against, so 0 is the correct answer, not None (None means "no
    # observed value at all", a different case -- see the next test).
    assert support.close_to_observed_count == 0
    assert support.checked_features == [f"{TRIGGER_FEATURE_PREFIX}syn_ratio"]


def test_evaluate_hypothesis_support_close_count_none_when_no_observed_value():
    df = _benign_df()
    hyp = _hyp("a3_h1", [PredictedRange(feature=f"{TRIGGER_FEATURE_PREFIX}flows_per_src", expected_max=99)])
    support = evaluate_hypothesis_support(hyp, {}, df)  # no observed values at all
    assert support.matching_profile_count > 0
    assert support.close_to_observed_count is None
    assert support.checked_features == []


def test_evaluate_hypothesis_support_ignores_unknown_features_in_profile():
    df = _benign_df()
    hyp = _hyp("a3_h1", [PredictedRange(feature=f"{TRIGGER_FEATURE_PREFIX}not_a_real_column", expected_max=10)])
    support = evaluate_hypothesis_support(hyp, {}, df)
    assert support.matching_profile_count == 0
    assert support.benign_population_size == 100


def test_render_hypothesis_support_labels_zero_as_unsupported():
    from agents.grounding import HypothesisSupport

    support = HypothesisSupport("a3_h1", 0, None, [], 100)
    line = render_hypothesis_support(support)
    assert "EMPIRICALLY UNSUPPORTED" in line


def test_render_hypothesis_support_reports_close_count():
    from agents.grounding import HypothesisSupport

    support = HypothesisSupport("a3_h1", 51, 31, [f"{TRIGGER_FEATURE_PREFIX}flows_per_src"], 100)
    line = render_hypothesis_support(support)
    assert "51 benign flow" in line
    assert "31 are within 2x" in line


# ---------- code-enforced plausibility cap (fix 4) ----------

def test_cap_applied_when_credited_hypothesis_has_zero_support():
    from agents.grounding import HypothesisSupport

    resp = A5Response(
        benign_plausibility=0.9, confidence=0.9, evidence_support=0.9, verification=0.9, cited_claim_ids=["a1_c1"], rationale="r",
        credited_hypothesis_id="a3_h1",
    )
    supports = {"a3_h1": HypothesisSupport("a3_h1", 0, None, [], 566864)}
    clamped, was_clamped = apply_empirical_plausibility_cap(resp, supports)
    assert was_clamped is True
    assert clamped.benign_plausibility == ZERO_SUPPORT_PLAUSIBILITY_CAP


def test_cap_not_applied_when_credited_hypothesis_has_support():
    from agents.grounding import HypothesisSupport

    resp = A5Response(
        benign_plausibility=0.9, confidence=0.9, evidence_support=0.9, verification=0.9, cited_claim_ids=["a1_c1"], rationale="r",
        credited_hypothesis_id="a3_h1",
    )
    supports = {"a3_h1": HypothesisSupport("a3_h1", 500, 10, ["x"], 566864)}
    clamped, was_clamped = apply_empirical_plausibility_cap(resp, supports)
    assert was_clamped is False
    assert clamped.benign_plausibility == pytest.approx(0.9)


def test_cap_not_applied_when_already_below_cap():
    from agents.grounding import HypothesisSupport

    resp = A5Response(
        benign_plausibility=0.2, confidence=0.9, evidence_support=0.9, verification=0.9, cited_claim_ids=["a1_c1"], rationale="r",
        credited_hypothesis_id="a3_h1",
    )
    supports = {"a3_h1": HypothesisSupport("a3_h1", 0, None, [], 566864)}
    clamped, was_clamped = apply_empirical_plausibility_cap(resp, supports)
    assert was_clamped is False
    assert clamped.benign_plausibility == pytest.approx(0.2)


def test_cap_not_applied_when_no_hypothesis_credited():
    from agents.grounding import HypothesisSupport

    resp = A5Response(
        benign_plausibility=0.9, confidence=0.9, evidence_support=0.9, verification=0.9, cited_claim_ids=["a1_c1"], rationale="r",
        credited_hypothesis_id=None,
    )
    supports = {"a3_h1": HypothesisSupport("a3_h1", 0, None, [], 566864)}
    clamped, was_clamped = apply_empirical_plausibility_cap(resp, supports)
    assert was_clamped is False
    assert clamped.benign_plausibility == pytest.approx(0.9)


def test_cap_not_applied_when_credited_hypothesis_id_unknown_to_supports_map():
    resp = A5Response(
        benign_plausibility=0.9, confidence=0.9, evidence_support=0.9, verification=0.9, cited_claim_ids=["a1_c1"], rationale="r",
        credited_hypothesis_id="a3_h_missing",
    )
    clamped, was_clamped = apply_empirical_plausibility_cap(resp, {})
    assert was_clamped is False
    assert clamped.benign_plausibility == pytest.approx(0.9)
