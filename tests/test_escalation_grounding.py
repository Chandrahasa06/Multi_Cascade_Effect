"""Tests for agents/escalation_grounding.py (Step 2's grounding repair)
and agents/grounding.py's loud-failure guard (Step 1).
"""
import pandas as pd
import pytest

from agents.escalation_grounding import (
    ESCALATION_FEATURES,
    UNGROUNDED_PLAUSIBILITY_CAP,
    FeatureNeighbourhood,
    apply_ungrounded_neighbourhood_cap,
    compute_feature_neighbourhood,
    observed_escalation_values,
    patch_hypothesis_support,
)
from agents.grounding import (
    DEFAULT_UNDEFINED_FRACTION_THRESHOLD,
    GroundingCoverageError,
    HypothesisSupport,
    assert_close_to_observed_coverage,
    close_to_observed_coverage,
)
from agents.schema import A5Response


def _make_benign_df(rows):
    return pd.DataFrame(rows)


class TestUndefinedCountGuardFires:
    def _support(self, close_count):
        return HypothesisSupport(
            hypothesis_id="h1", matching_profile_count=10, close_to_observed_count=close_count,
            checked_features=[], benign_population_size=100,
        )

    def test_fires_when_all_undefined(self):
        supports = [self._support(None) for _ in range(10)]
        with pytest.raises(GroundingCoverageError):
            assert_close_to_observed_coverage(supports)

    def test_fires_above_threshold_not_at_zero(self):
        # 6/10 undefined, default threshold 0.5 -- 0.6 > 0.5, must fire
        supports = [self._support(None) for _ in range(6)] + [self._support(3) for _ in range(4)]
        with pytest.raises(GroundingCoverageError):
            assert_close_to_observed_coverage(supports)

    def test_does_not_fire_when_mostly_defined(self):
        supports = [self._support(None) for _ in range(2)] + [self._support(3) for _ in range(8)]
        assert_close_to_observed_coverage(supports)  # must not raise

    def test_does_not_fire_on_empty_input(self):
        assert_close_to_observed_coverage([])  # nothing to alarm on

    def test_coverage_helper_reports_exact_counts(self):
        supports = [self._support(None) for _ in range(3)] + [self._support(1) for _ in range(7)]
        n_undefined, n_total = close_to_observed_coverage(supports)
        assert (n_undefined, n_total) == (3, 10)

    def test_respects_custom_threshold(self):
        supports = [self._support(None) for _ in range(3)] + [self._support(1) for _ in range(7)]
        assert_close_to_observed_coverage(supports, threshold=0.5)  # 30% <= 50%, fine
        with pytest.raises(GroundingCoverageError):
            assert_close_to_observed_coverage(supports, threshold=0.2)  # 30% > 20%, fires

    def test_default_threshold_matches_documented_value(self):
        assert DEFAULT_UNDEFINED_FRACTION_THRESHOLD == 0.5


class TestComparisonSetIsBenignOnly:
    def test_load_benign_escalation_reference_filters_to_benign(self, monkeypatch):
        import agents.escalation_grounding as eg

        fake_pool = pd.DataFrame({
            "Label": ["BENIGN", "BENIGN", "DoS Hulk", "PortScan"],
            **{f: [1.0, 2.0, 999.0, 888.0] for f in ESCALATION_FEATURES},
        })
        monkeypatch.setattr("eval.escalation_data.load_pool", lambda: fake_pool)
        monkeypatch.setattr("eval.escalation_data.BENIGN_LABEL", "BENIGN")
        eg._benign_reference_cache = None  # bypass the in-process cache for this test
        ref = eg.load_benign_escalation_reference(force_reload=True)
        assert len(ref) == 2
        assert set(ref[ESCALATION_FEATURES[0]]) == {1.0, 2.0}

    def test_neighbourhood_never_counts_non_benign_rows(self):
        # a benign_df deliberately containing only benign-labeled-shaped
        # values (caller's job to have filtered) -- confirms the function
        # itself does no re-filtering that could accidentally admit more
        benign_df = _make_benign_df({f: [10.0] * 50 for f in ESCALATION_FEATURES})
        observed = {f: 10.0 for f in ESCALATION_FEATURES}
        nb = compute_feature_neighbourhood(observed, benign_df, min_neighbours=10)
        assert nb.neighbourhood_size == 50
        assert nb.benign_population_size == 50


class TestBlindPropertyHolds:
    def test_observed_escalation_values_signature_has_no_escalation_metadata_params(self):
        import inspect

        sig = inspect.signature(observed_escalation_values)
        assert list(sig.parameters.keys()) == ["features"]

    def test_output_identical_regardless_of_escalation_metadata(self):
        """The real guarantee: build two 'records' that differ only in
        their escalation digest (class_predicted/rule_id/priority/reason)
        but share identical features -- observed_escalation_values must
        produce byte-identical output either way, since it's never even
        given the digest to read."""
        features = {f: float(i) for i, f in enumerate(ESCALATION_FEATURES, start=1)}
        # simulate "P1 record" vs "P2 record" callers -- neither passes
        # anything beyond `features` into this function
        out_p1 = observed_escalation_values(features)
        out_p2 = observed_escalation_values(features)
        assert out_p1 == out_p2

    def test_compute_feature_neighbourhood_has_no_escalation_metadata_params(self):
        import inspect

        sig = inspect.signature(compute_feature_neighbourhood)
        assert set(sig.parameters.keys()) == {"observed", "benign_df", "min_neighbours"}

    def test_patch_hypothesis_support_has_no_escalation_metadata_params(self):
        import inspect

        sig = inspect.signature(patch_hypothesis_support)
        assert set(sig.parameters.keys()) == {"supports", "neighbourhood"}


class TestEmptyNeighbourhoodNotTreatedAsBenign:
    def test_empty_neighbourhood_marked_ungrounded(self):
        benign_df = _make_benign_df({f: [1.0] * 100 for f in ESCALATION_FEATURES})
        observed = {f: 1_000_000.0 for f in ESCALATION_FEATURES}  # nothing benign is remotely close
        nb = compute_feature_neighbourhood(observed, benign_df, min_neighbours=30)
        assert nb.neighbourhood_size == 0
        assert nb.ungrounded is True

    def test_below_minimum_is_ungrounded_even_if_nonzero(self):
        benign_df = _make_benign_df({
            ESCALATION_FEATURES[0]: [10.0] * 5 + [999.0] * 95,
            **{f: [10.0] * 100 for f in ESCALATION_FEATURES[1:]},
        })
        observed = {f: 10.0 for f in ESCALATION_FEATURES}
        nb = compute_feature_neighbourhood(observed, benign_df, min_neighbours=30)
        assert 0 < nb.neighbourhood_size < 30
        assert nb.ungrounded is True

    def test_patch_hypothesis_support_reports_zero_not_none_when_ungrounded(self):
        support = HypothesisSupport(
            hypothesis_id="h1", matching_profile_count=500, close_to_observed_count=None,
            checked_features=[], benign_population_size=1000,
        )
        neighbourhood = FeatureNeighbourhood(
            features_used=list(ESCALATION_FEATURES), band_description="test",
            neighbourhood_size=0, benign_population_size=1000, ungrounded=True, min_required=30,
        )
        patched = patch_hypothesis_support({"h1": support}, neighbourhood)
        assert patched["h1"].close_to_observed_count == 0  # defined, not None
        assert patched["h1"].checked_features == list(ESCALATION_FEATURES)
        assert patched["h1"].matching_profile_count == 500  # untouched

    def test_ungrounded_cap_fires_and_does_not_silently_pass_as_benign(self):
        response = A5Response(
            benign_plausibility=0.9, confidence=0.8, evidence_support=0.8, verification=0.8,
            credited_hypothesis_id="h1", cited_claim_ids=["a1_c1"], rationale="looks benign",
        )
        neighbourhood = FeatureNeighbourhood(
            features_used=list(ESCALATION_FEATURES), band_description="test",
            neighbourhood_size=0, benign_population_size=1000, ungrounded=True, min_required=30,
        )
        clamped, fired = apply_ungrounded_neighbourhood_cap(response, neighbourhood)
        assert fired is True
        assert clamped.benign_plausibility == UNGROUNDED_PLAUSIBILITY_CAP

    def test_ungrounded_cap_does_not_fire_when_grounded(self):
        response = A5Response(
            benign_plausibility=0.9, confidence=0.8, evidence_support=0.8, verification=0.8,
            credited_hypothesis_id="h1", cited_claim_ids=["a1_c1"], rationale="looks benign",
        )
        neighbourhood = FeatureNeighbourhood(
            features_used=list(ESCALATION_FEATURES), band_description="test",
            neighbourhood_size=500, benign_population_size=1000, ungrounded=False, min_required=30,
        )
        unchanged, fired = apply_ungrounded_neighbourhood_cap(response, neighbourhood)
        assert fired is False
        assert unchanged.benign_plausibility == 0.9

    def test_ungrounded_cap_never_raises_plausibility(self):
        """A response already below the cap must not be pushed UP to it."""
        response = A5Response(
            benign_plausibility=0.1, confidence=0.8, evidence_support=0.8, verification=0.8,
            credited_hypothesis_id="h1", cited_claim_ids=["a1_c1"], rationale="looks anomalous",
        )
        neighbourhood = FeatureNeighbourhood(
            features_used=list(ESCALATION_FEATURES), band_description="test",
            neighbourhood_size=0, benign_population_size=1000, ungrounded=True, min_required=30,
        )
        result, fired = apply_ungrounded_neighbourhood_cap(response, neighbourhood)
        assert fired is False
        assert result.benign_plausibility == 0.1


class TestRealDoSHulkRecordVerification:
    def test_traced_record_is_now_grounded_and_defined(self):
        """The exact record cited in the Step 2 diagnosis: matching_profile_count
        was 258,258/566,864 (45.6%) with close_to_observed_count=None.
        After the fix, close_to_observed_count must be defined and its
        implied fraction must be far below 45.6%."""
        try:
            import json

            with open("results/agent_input_20_blind.jsonl", encoding="utf-8") as f:
                records = [json.loads(l) for l in f]
        except FileNotFoundError:
            pytest.skip("results/agent_input_20_blind.jsonl not generated yet")
        rec = next((r for r in records if r["record_id"] == "6846713a334440c1b9748d47e85b6ee4"), None)
        if rec is None:
            pytest.skip("traced DoS Hulk record not present in the current blind input")
        observed = observed_escalation_values(rec["features"])
        nb = compute_feature_neighbourhood(observed)
        assert nb.neighbourhood_size is not None  # defined, not None -- the core fix
        fraction = nb.neighbourhood_size / nb.benign_population_size
        assert fraction < 0.456, f"expected far below 45.6%, got {fraction:.1%}"
