import pytest

from controlplane.record import EscalationRecord, PacketWindowSummary
from dataplane.selector import TriggerReason

from agents import base, pipeline
from agents.prompts.render import TRIGGER_FEATURE_PREFIX
from agents.schema import (
    A5Response,
    Claim,
    ClaimsResponse,
    FeatureReference,
    Hypothesis,
    HypothesisResponse,
    PredictedRange,
    Relation,
)
from agents.verification import VerificationCounts
from eval import fault_injection as fi

FEATURES = {"syn_ratio": 0.5, "flow_duration": 1.2, "dst_port": 8080.0}

_PROFILE = [PredictedRange(feature=f"{TRIGGER_FEATURE_PREFIX}syn_ratio", expected_max=1.0)]


def _make_record(flow_id="test-flow-1") -> EscalationRecord:
    return EscalationRecord(
        flow_id=flow_id,
        trigger_reasons=[
            TriggerReason(feature="syn_ratio", observed_value=0.5, threshold=0.33,
                           direction="high", ratio=1.5, low_confidence=False)
        ],
        features=dict(FEATURES),
        packet_window=PacketWindowSummary(packet_count=10, byte_count=1000, duration_us=5000),
        selector_config_hash="deadbeef",
    )


def _claim(claim_id, relation=Relation.GREATER_THAN_TYPICAL, feature="syn_ratio", asserted_value=0.5):
    ref = FeatureReference(name=feature, asserted_value=asserted_value, relation=relation)
    return Claim(claim_id=claim_id, statement=f"{feature} claim", referenced_features=[ref], confidence=0.7)


def _claims_response(n=4, prefix="a1"):
    return ClaimsResponse(
        claims=[_claim(f"{prefix}_c{i+1}") for i in range(n)],
        confidence=0.8, evidence_support=0.8, verification=0.8,
    )


def _hypothesis_response(prefix="a3"):
    hyps = [
        Hypothesis(hypothesis_id=f"{prefix}_h1", description="d", benign=True, prior_plausibility=0.5,
                   prediction="the syn ratio should stay bounded", predicted_feature_profile=_PROFILE),
        Hypothesis(hypothesis_id=f"{prefix}_h2", description="d", benign=True, prior_plausibility=0.4,
                   prediction="the syn ratio should stay near typical", predicted_feature_profile=_PROFILE),
        Hypothesis(hypothesis_id=f"{prefix}_h3", description="d", benign=False, prior_plausibility=0.3,
                   prediction="the syn ratio should be elevated", predicted_feature_profile=_PROFILE),
    ]
    return HypothesisResponse(
        claims=[_claim(f"{prefix}_c1")], hypotheses=hyps,
        confidence=0.7, evidence_support=0.7, verification=0.7,
    )


# ---------- corrupt_missing_evidence ----------

def test_missing_evidence_drops_half_the_claims():
    a1 = _claims_response(n=4)
    corrupted = fi.corrupt_missing_evidence(a1, "flow-1")
    assert len(corrupted.claims) == 2


def test_missing_evidence_odd_count_rounds_drop_down():
    a1 = _claims_response(n=5)
    corrupted = fi.corrupt_missing_evidence(a1, "flow-1")
    assert len(corrupted.claims) == 3  # drop floor(5/2)=2, keep 3


def test_missing_evidence_kept_claims_are_a_subset():
    a1 = _claims_response(n=6)
    corrupted = fi.corrupt_missing_evidence(a1, "flow-1")
    kept_ids = {c.claim_id for c in corrupted.claims}
    original_ids = {c.claim_id for c in a1.claims}
    assert kept_ids <= original_ids
    assert len(kept_ids) == 3


def test_missing_evidence_is_reproducible_for_same_record_id():
    a1 = _claims_response(n=6)
    first = fi.corrupt_missing_evidence(a1, "flow-1")
    second = fi.corrupt_missing_evidence(a1, "flow-1")
    assert [c.claim_id for c in first.claims] == [c.claim_id for c in second.claims]


def test_missing_evidence_differs_across_record_ids_in_general():
    # not a hard guarantee for every possible pair, but true for these
    # two given the fixed module seed strings -- regression check that
    # record_id actually enters the seed, not a constant.
    a1 = _claims_response(n=6)
    a = fi.corrupt_missing_evidence(a1, "flow-1")
    b = fi.corrupt_missing_evidence(a1, "flow-2")
    assert [c.claim_id for c in a.claims] != [c.claim_id for c in b.claims]


def test_missing_evidence_response_is_otherwise_unchanged():
    a1 = _claims_response(n=4)
    corrupted = fi.corrupt_missing_evidence(a1, "flow-1")
    assert corrupted.confidence == a1.confidence
    assert corrupted.evidence_support == a1.evidence_support
    assert corrupted.verification == a1.verification


def test_missing_evidence_handles_zero_claims():
    a1 = ClaimsResponse(claims=[], confidence=0.5, evidence_support=0.5, verification=0.5)
    corrupted = fi.corrupt_missing_evidence(a1, "flow-1")
    assert corrupted.claims == []


# ---------- corrupt_incorrect_behavior ----------

def test_incorrect_behavior_flips_greater_and_less():
    a2 = ClaimsResponse(
        claims=[
            _claim("a2_c1", relation=Relation.GREATER_THAN_TYPICAL),
            _claim("a2_c2", relation=Relation.LESS_THAN_TYPICAL),
        ],
        confidence=0.7, evidence_support=0.7, verification=0.7,
    )
    corrupted = fi.corrupt_incorrect_behavior(a2)
    assert corrupted.claims[0].referenced_features[0].relation == Relation.LESS_THAN_TYPICAL
    assert corrupted.claims[1].referenced_features[0].relation == Relation.GREATER_THAN_TYPICAL


def test_incorrect_behavior_leaves_equals_absent_present_untouched():
    a2 = ClaimsResponse(
        claims=[
            _claim("a2_c1", relation=Relation.EQUALS),
            _claim("a2_c2", relation=Relation.ABSENT, asserted_value=None),
            _claim("a2_c3", relation=Relation.PRESENT, asserted_value=None),
        ],
        confidence=0.7, evidence_support=0.7, verification=0.7,
    )
    corrupted = fi.corrupt_incorrect_behavior(a2)
    relations = [c.referenced_features[0].relation for c in corrupted.claims]
    assert relations == [Relation.EQUALS, Relation.ABSENT, Relation.PRESENT]


def test_incorrect_behavior_leaves_statement_and_confidence_untouched():
    a2 = ClaimsResponse(claims=[_claim("a2_c1")], confidence=0.65, evidence_support=0.6, verification=0.55)
    corrupted = fi.corrupt_incorrect_behavior(a2)
    assert corrupted.claims[0].statement == a2.claims[0].statement
    assert corrupted.claims[0].confidence == a2.claims[0].confidence
    assert corrupted.confidence == a2.confidence
    assert corrupted.evidence_support == a2.evidence_support
    assert corrupted.verification == a2.verification


def test_incorrect_behavior_is_deterministic_no_rng():
    a2 = ClaimsResponse(claims=[_claim("a2_c1")], confidence=0.7, evidence_support=0.7, verification=0.7)
    first = fi.corrupt_incorrect_behavior(a2)
    second = fi.corrupt_incorrect_behavior(a2)
    assert first.claims[0].referenced_features[0].relation == second.claims[0].referenced_features[0].relation


# ---------- corrupt_hallucinated_hypothesis ----------

def test_hallucinated_hypothesis_appends_exactly_one():
    a3 = _hypothesis_response()
    corrupted = fi.corrupt_hallucinated_hypothesis(a3)
    assert len(corrupted.hypotheses) == len(a3.hypotheses) + 1


def test_hallucinated_hypothesis_original_hypotheses_untouched():
    a3 = _hypothesis_response()
    corrupted = fi.corrupt_hallucinated_hypothesis(a3)
    assert corrupted.hypotheses[: len(a3.hypotheses)] == a3.hypotheses


def test_hallucinated_hypothesis_has_high_prior_plausibility():
    a3 = _hypothesis_response()
    corrupted = fi.corrupt_hallucinated_hypothesis(a3)
    fabricated = corrupted.hypotheses[-1]
    assert fabricated.prior_plausibility == 0.9


def test_hallucinated_hypothesis_references_a_nonexistent_feature():
    a3 = _hypothesis_response()
    corrupted = fi.corrupt_hallucinated_hypothesis(a3)
    fabricated = corrupted.hypotheses[-1]
    real_features = {p.feature for h in a3.hypotheses for p in h.predicted_feature_profile}
    fabricated_features = {p.feature for p in fabricated.predicted_feature_profile}
    assert fabricated_features.isdisjoint(real_features)
    assert fi.HALLUCINATED_FEATURE in fabricated_features
    # and it really isn't a real Tier-1/reference feature name:
    assert "process_creation" in fi.HALLUCINATED_FEATURE  # host-telemetry-shaped, not flow-shaped


def test_hallucinated_hypothesis_id_does_not_collide():
    a3 = _hypothesis_response()
    corrupted = fi.corrupt_hallucinated_hypothesis(a3)
    ids = [h.hypothesis_id for h in corrupted.hypotheses]
    assert len(ids) == len(set(ids))


# ---------- corrupt_faulty_verification ----------

def test_faulty_verification_swaps_corroborated_and_contradicted():
    counts = {"a1": VerificationCounts(corroborated=5, contradicted=2, uncorroborated=3)}
    corrupted = fi.corrupt_faulty_verification(counts)
    assert corrupted["a1"].corroborated == 2
    assert corrupted["a1"].contradicted == 5
    assert corrupted["a1"].uncorroborated == 3


def test_faulty_verification_preserves_agent_keys():
    counts = {
        "a1": VerificationCounts(corroborated=1, contradicted=0, uncorroborated=0),
        "a2": VerificationCounts(corroborated=0, contradicted=1, uncorroborated=2),
    }
    corrupted = fi.corrupt_faulty_verification(counts)
    assert set(corrupted) == {"a1", "a2"}
    assert corrupted["a2"].corroborated == 1
    assert corrupted["a2"].contradicted == 0


def test_faulty_verification_is_its_own_inverse():
    counts = {"a1": VerificationCounts(corroborated=5, contradicted=2, uncorroborated=3)}
    twice = fi.corrupt_faulty_verification(fi.corrupt_faulty_verification(counts))
    assert twice["a1"].corroborated == 5
    assert twice["a1"].contradicted == 2


# ---------- RERUN_AGENTS table matches the module docstring's spec ----------

def test_rerun_agents_table_matches_spec():
    assert fi.RERUN_AGENTS[fi.MISSING_EVIDENCE] == ("a2", "a3", "a4", "a5")
    assert fi.RERUN_AGENTS[fi.INCORRECT_BEHAVIOR] == ("a3", "a4", "a5")
    assert fi.RERUN_AGENTS[fi.HALLUCINATED_HYPOTHESIS] == ("a4", "a5")
    assert fi.RERUN_AGENTS[fi.FAULTY_VERIFICATION] == ("a5",)
    assert set(fi.RERUN_AGENTS) == set(fi.ALL_CONDITIONS)


def test_run_fault_condition_rejects_unknown_condition():
    with pytest.raises(ValueError, match="unknown fault condition"):
        fi.run_fault_condition(None, _make_record(), "not_a_real_condition")


# ---------- orchestration wiring (mocked calls, no real API access) ----------


def _canned_clean_responses():
    a1 = _claims_response(n=4, prefix="a1")
    a2 = _claims_response(n=3, prefix="a2")
    a3 = _hypothesis_response(prefix="a3")
    a4 = _hypothesis_response(prefix="a4")
    a5 = A5Response(
        benign_plausibility=0.6, confidence=0.6, evidence_support=0.6, verification=0.6,
        credited_hypothesis_id="a3_h1", cited_claim_ids=["a1_c1"], rationale="r",
    )
    return {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5}


def _fake_call_structured_factory(responses, calls):
    def fake_call_structured(*, record_id, agent, prompt_version, prompt, response_schema, model,
                              temperature, run_index, use_cache=True, extra_validate=None,
                              fault_condition=base.CLEAN_FAULT_CONDITION):
        calls.append((agent, fault_condition))
        resp = responses[agent]
        if extra_validate is not None:
            extra_validate(resp)
        return resp, base.CallMetadata(cached=False, schema_retries=0, elapsed_s=0.01,
                                        input_tokens=10, output_tokens=5)

    return fake_call_structured


@pytest.fixture
def clean_result(monkeypatch):
    responses = _canned_clean_responses()
    calls = []
    monkeypatch.setattr(base, "call_structured", _fake_call_structured_factory(responses, calls))
    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)
    calls.clear()  # only care about calls made during fault injection from here
    return record, result, responses, calls


def test_missing_evidence_reruns_exactly_a2_a3_a4_a5(monkeypatch, clean_result):
    record, clean, responses, calls = clean_result
    monkeypatch.setattr(base, "call_structured", _fake_call_structured_factory(responses, calls))

    result = fi.run_fault_condition(clean, record, fi.MISSING_EVIDENCE, use_cache=False)

    called_agents = {agent for agent, _ in calls}
    assert called_agents == {"a2", "a3", "a4", "a5"}  # a1 NOT re-called
    assert all(fc == fi.MISSING_EVIDENCE for _, fc in calls)
    # a1 in the result is the corrupted (shorter) version, not clean's
    assert len(result.a1.claims) < len(clean.a1.claims)
    assert result.a1.claims != clean.a1.claims or len(result.a1.claims) != len(clean.a1.claims)


def test_incorrect_behavior_reruns_exactly_a3_a4_a5_and_reuses_a1(monkeypatch, clean_result):
    record, clean, responses, calls = clean_result
    monkeypatch.setattr(base, "call_structured", _fake_call_structured_factory(responses, calls))

    result = fi.run_fault_condition(clean, record, fi.INCORRECT_BEHAVIOR, use_cache=False)

    called_agents = {agent for agent, _ in calls}
    assert called_agents == {"a3", "a4", "a5"}
    assert all(fc == fi.INCORRECT_BEHAVIOR for _, fc in calls)
    assert result.a1 is clean.a1  # untouched, reused directly


def test_hallucinated_hypothesis_reruns_exactly_a4_a5_and_reuses_a1_a2(monkeypatch, clean_result):
    record, clean, responses, calls = clean_result
    monkeypatch.setattr(base, "call_structured", _fake_call_structured_factory(responses, calls))

    result = fi.run_fault_condition(clean, record, fi.HALLUCINATED_HYPOTHESIS, use_cache=False)

    called_agents = {agent for agent, _ in calls}
    assert called_agents == {"a4", "a5"}
    assert all(fc == fi.HALLUCINATED_HYPOTHESIS for _, fc in calls)
    assert result.a1 is clean.a1
    assert result.a2 is clean.a2
    assert len(result.a3.hypotheses) == len(clean.a3.hypotheses) + 1


def test_faulty_verification_reruns_only_a5_and_reuses_a1_a2_a3_a4(monkeypatch, clean_result):
    record, clean, responses, calls = clean_result
    monkeypatch.setattr(base, "call_structured", _fake_call_structured_factory(responses, calls))

    result = fi.run_fault_condition(clean, record, fi.FAULTY_VERIFICATION, use_cache=False)

    called_agents = {agent for agent, _ in calls}
    assert called_agents == {"a5"}
    assert all(fc == fi.FAULTY_VERIFICATION for _, fc in calls)
    assert result.a1 is clean.a1
    assert result.a2 is clean.a2
    assert result.a3 is clean.a3
    assert result.a4 is clean.a4
