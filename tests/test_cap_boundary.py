"""The cap/threshold collision: a clamp to exactly 0.30 lands on the verdict
boundary (derive_verdict reads bp < 0.30 as ANOMALOUS_AND_UNEXPLAINED), so the
clamped record was read as explicable. The cap is set to 0.29; the threshold is
unchanged. These tests pin both halves of that."""
import pytest

from agents.escalation_grounding import (
    ESCALATION_FEATURES,
    UNGROUNDED_PLAUSIBILITY_CAP,
    FeatureNeighbourhood,
    apply_ungrounded_neighbourhood_cap,
)
from agents.grounding import (
    ZERO_SUPPORT_PLAUSIBILITY_CAP,
    HypothesisSupport,
    apply_empirical_plausibility_cap,
)
from agents.schema import (
    DEFAULT_LOW_PLAUSIBILITY_THRESHOLD,
    A5Response,
    VerdictLabel,
    derive_verdict,
)


def _response(bp):
    return A5Response(
        benign_plausibility=bp, confidence=0.8, evidence_support=0.8, verification=0.8,
        credited_hypothesis_id="h1", cited_claim_ids=["a1_c1"], rationale="x",
    )


def _ungrounded():
    return FeatureNeighbourhood(
        features_used=list(ESCALATION_FEATURES), band_description="test",
        neighbourhood_size=0, benign_population_size=1000, ungrounded=True, min_required=30,
    )


def test_both_caps_sit_strictly_below_the_verdict_threshold():
    assert DEFAULT_LOW_PLAUSIBILITY_THRESHOLD == 0.3
    assert UNGROUNDED_PLAUSIBILITY_CAP < DEFAULT_LOW_PLAUSIBILITY_THRESHOLD
    assert ZERO_SUPPORT_PLAUSIBILITY_CAP < DEFAULT_LOW_PLAUSIBILITY_THRESHOLD


def test_ungrounded_clamped_record_derives_anomalous_and_unexplained():
    clamped, fired = apply_ungrounded_neighbourhood_cap(_response(0.9), _ungrounded())
    assert fired is True
    assert clamped.benign_plausibility == UNGROUNDED_PLAUSIBILITY_CAP
    assert derive_verdict(clamped.benign_plausibility) is VerdictLabel.ANOMALOUS_AND_UNEXPLAINED


def test_zero_support_clamped_record_derives_anomalous_and_unexplained():
    support = {"h1": HypothesisSupport(
        hypothesis_id="h1", matching_profile_count=0, benign_population_size=1000,
        close_to_observed_count=None, checked_features=[],
    )}
    clamped, fired = apply_empirical_plausibility_cap(_response(0.9), support)
    assert fired is True
    assert clamped.benign_plausibility == ZERO_SUPPORT_PLAUSIBILITY_CAP
    assert derive_verdict(clamped.benign_plausibility) is VerdictLabel.ANOMALOUS_AND_UNEXPLAINED


@pytest.mark.parametrize("bp", [0.30, 0.31])
def test_model_output_at_or_just_above_threshold_is_clamped_under_the_boundary(bp):
    # A model that copies the stated cap ("at most 0.3") emits exactly 0.30.
    # The clamp only fires strictly above the cap, so 0.30 must still be clamped
    # to 0.29 here. Under the old 0.3 cap it passed through unchanged.
    clamped, fired = apply_ungrounded_neighbourhood_cap(_response(bp), _ungrounded())
    assert fired is True
    assert clamped.benign_plausibility == UNGROUNDED_PLAUSIBILITY_CAP
    assert derive_verdict(clamped.benign_plausibility) is VerdictLabel.ANOMALOUS_AND_UNEXPLAINED
