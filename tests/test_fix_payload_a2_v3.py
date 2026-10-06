"""Tests for the re-run fixes: (1) the flow-level filter accepts claims about the
absence or size of payload and still rejects claims about its contents;
(2) A2 receives the empirical grounding block (a2_behaviour_v3); (3) older prompt
versions stay importable so cache keys for historical results stay valid."""
import inspect

import pytest

from agents.prompts import a2_behaviour_v2, a2_behaviour_v3
from agents.schema import Hypothesis, HypothesisResponse, PredictedRange
from agents.validators import (
    FLOW_LEVEL_FILTER_PREFIX,
    payload_claim_is_flow_observable,
    unobservable_payload_terms,
    validate_hypothesis_predictions_are_flow_level,
)

KNOWN_FEATURES = {"trigger_flow_duration", "trigger_flows_per_src", "trigger_syn_ratio"}


def _resp(prediction):
    hyp = Hypothesis(
        hypothesis_id="a3_h1", description="d", benign=True, prior_plausibility=0.5,
        prediction=prediction,
        predicted_feature_profile=[PredictedRange(feature="trigger_flow_duration", expected_max=1000.0)],
        contradicting_claim_ids=[],
    )
    extra = Hypothesis(
        hypothesis_id="a3_h2", description="d", benign=False, prior_plausibility=0.5,
        prediction="flow count per source should be high",
        predicted_feature_profile=[PredictedRange(feature="trigger_flows_per_src", expected_max=1000.0)],
        contradicting_claim_ids=[],
    )
    return HypothesisResponse(
        claims=[], hypotheses=[hyp, extra], confidence=0.7, evidence_support=0.7, verification=0.7,
    )


# ---------- fix 1: absence or size of payload is accepted ----------

@pytest.mark.parametrize("prediction", [
    "The flow should carry no payload, so the forward byte count is zero.",
    "Zero payload is expected in this flow.",
    "The flow is expected to have a zero data payload length.",
    "Total payload size is 0 bytes.",
    "Such flows show the lack of any payload.",
    "The flow shows an absence of payload bytes.",
])
def test_filter_accepts_absence_or_size_of_payload(prediction):
    validate_hypothesis_predictions_are_flow_level(_resp(prediction), KNOWN_FEATURES)  # must not raise


# ---------- fix 1: payload contents are still rejected ----------

@pytest.mark.parametrize("prediction", [
    "The payloads should contain standard HTTP headers.",
    "Zero payload bytes but the payload contains a login string.",
    "Payload data shows a login form in the body.",
    "The payload contents are a command string.",
])
def test_filter_rejects_payload_contents(prediction):
    with pytest.raises(ValueError, match=FLOW_LEVEL_FILTER_PREFIX):
        validate_hypothesis_predictions_are_flow_level(_resp(prediction), KNOWN_FEATURES)


def test_absence_marker_does_not_rescue_a_content_claim_in_the_same_sentence():
    assert payload_claim_is_flow_observable("zero payload but contains a login string") is False


def test_payload_claim_in_a_second_sentence_is_still_checked():
    # The first sentence is an observable size claim; the second is a content claim.
    text = "Total payload size is 0 bytes. The payload contains a login string."
    assert unobservable_payload_terms(text) == ["payload"]


def test_non_payload_banned_terms_still_rejected_even_with_absence_wording():
    with pytest.raises(ValueError, match=FLOW_LEVEL_FILTER_PREFIX):
        validate_hypothesis_predictions_are_flow_level(
            _resp("No HTTP headers are expected in these flows."), KNOWN_FEATURES
        )


# ---------- fix 2: A2 v3 receives the grounding block ----------

def test_a2_v3_build_prompt_takes_grounding_block():
    # Signature only. Whether the block appears in the rendered prompt is covered by
    # the scored re-run; rendering needs a full EscalationRecord fixture.
    sig = inspect.signature(a2_behaviour_v3.build_prompt)
    assert list(sig.parameters) == ["record", "a1_claims", "empirical_grounding_block"]


def test_a2_v3_prompt_version_string():
    assert a2_behaviour_v3.PROMPT_VERSION == "a2_behaviour_v3"


def test_a2_run_requires_grounding_block_positionally():
    from agents import a2_behaviour
    sig = inspect.signature(a2_behaviour.run)
    params = list(sig.parameters.values())
    assert params[2].name == "empirical_grounding_block"
    assert params[2].default is inspect.Parameter.empty


def test_a2_v3_instructions_unchanged_from_v2():
    assert a2_behaviour_v3._INSTRUCTIONS == a2_behaviour_v2._INSTRUCTIONS
    assert a2_behaviour_v3.FRAMING_PREAMBLE == a2_behaviour_v2.FRAMING_PREAMBLE


# ---------- stability: older prompt versions stay importable ----------

def test_a2_v2_still_importable_with_its_version_string():
    assert a2_behaviour_v2.PROMPT_VERSION == "a2_behaviour_v2"
    # v2 keeps its original signature, so historical cache keys still resolve.
    assert list(inspect.signature(a2_behaviour_v2.build_prompt).parameters) == ["record", "a1_claims"]


def test_a5_scale_prompt_importable():
    from agents.prompts import a5_verdict_scale_v1
    assert a5_verdict_scale_v1.PROMPT_VERSION == "a5_verdict_scale_v1"
