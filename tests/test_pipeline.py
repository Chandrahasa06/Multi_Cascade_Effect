import dataclasses
import re

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

FEATURES = {
    "syn_ratio": 0.5,
    "flow_duration": 1.2,
    "dst_port": 8080.0,
    "down_up_ratio": 1.5,
    "fwd_pkt_len_mean": 3400.0,
}

# Real Tier-1 feature names (from controlplane/reference.py's benign
# reference distribution) used in predicted_feature_profile entries below
# -- that field's vocabulary is validated against the real reference set,
# not the record's own (CICFlowMeter-namespaced) `features` dict, so
# fixtures must use names this project actually has benign data for.
_WIDE_SYN_RATIO_PROFILE = [PredictedRange(feature=f"{TRIGGER_FEATURE_PREFIX}syn_ratio", expected_max=1.0)]
_WIDE_DURATION_PROFILE = [PredictedRange(feature=f"{TRIGGER_FEATURE_PREFIX}flow_duration", expected_max=1e9)]
# deliberately impossible: syn_ratio is bounded [0,1], so nothing can ever
# match this -- used to exercise the zero-empirical-support cap (fix 4).
_IMPOSSIBLE_PROFILE = [PredictedRange(feature=f"{TRIGGER_FEATURE_PREFIX}syn_ratio", expected_min=100.0, expected_max=200.0)]


def _make_record(flow_id="test-flow-1") -> EscalationRecord:
    return EscalationRecord(
        flow_id=flow_id,
        trigger_reasons=[
            TriggerReason(
                feature="syn_ratio", observed_value=0.5, threshold=0.33,
                direction="high", ratio=1.5, low_confidence=False,
            )
        ],
        features=dict(FEATURES),
        packet_window=PacketWindowSummary(packet_count=10, byte_count=1000, duration_us=5000),
        selector_config_hash="deadbeef",
    )


def _canned_responses():
    a1 = ClaimsResponse(
        claims=[
            Claim(
                claim_id="a1_c1",
                statement="syn ratio is elevated",
                referenced_features=[
                    FeatureReference(name="syn_ratio", asserted_value=0.5, relation=Relation.GREATER_THAN_TYPICAL)
                ],
                confidence=0.8,
            )
        ],
        confidence=0.8, evidence_support=0.8, verification=0.8,
    )
    a2 = ClaimsResponse(
        claims=[
            Claim(
                claim_id="a2_c1",
                statement="duration matches a short exchange",
                referenced_features=[
                    FeatureReference(name="flow_duration", asserted_value=1.2, relation=Relation.EQUALS)
                ],
                confidence=0.7,
            )
        ],
        confidence=0.7, evidence_support=0.7, verification=0.7,
    )
    a3 = HypothesisResponse(
        claims=[],
        hypotheses=[
            Hypothesis(hypothesis_id="a3_h1", description="retry loop", benign=True, prior_plausibility=0.5,
                       prediction="the syn ratio should stay within the normal bounded range",
                       predicted_feature_profile=_WIDE_SYN_RATIO_PROFILE),
            Hypothesis(hypothesis_id="a3_h2", description="health check", benign=True, prior_plausibility=0.4,
                       prediction="the flow duration should be unremarkable",
                       predicted_feature_profile=_WIDE_DURATION_PROFILE),
            Hypothesis(hypothesis_id="a3_h3", description="scripted probing", benign=False, prior_plausibility=0.2,
                       prediction="the syn ratio should be elevated",
                       predicted_feature_profile=_WIDE_SYN_RATIO_PROFILE),
        ],
        confidence=0.7, evidence_support=0.7, verification=0.7,
    )
    a4 = HypothesisResponse(
        claims=[
            Claim(
                claim_id="a4_c1",
                statement="syn ratio also looks elevated independently",
                referenced_features=[
                    FeatureReference(name="syn_ratio", asserted_value=0.5, relation=Relation.GREATER_THAN_TYPICAL)
                ],
                confidence=0.75,
            ),
            Claim(
                claim_id="a4_c2",
                statement="destination port is a common alternate http port",
                referenced_features=[
                    FeatureReference(name="dst_port", asserted_value=8080.0, relation=Relation.EQUALS)
                ],
                confidence=0.6,
            ),
        ],
        hypotheses=[
            Hypothesis(hypothesis_id="a4_h1", description="retry loop", benign=True, prior_plausibility=0.5,
                       prediction="the syn ratio should stay within the normal bounded range",
                       predicted_feature_profile=_WIDE_SYN_RATIO_PROFILE),
            Hypothesis(hypothesis_id="a4_h2", description="load test", benign=True, prior_plausibility=0.3,
                       prediction="the flow duration should be unremarkable",
                       predicted_feature_profile=_WIDE_DURATION_PROFILE),
            Hypothesis(hypothesis_id="a4_h3", description="scripted probing", benign=False, prior_plausibility=0.3,
                       prediction="the syn ratio should be elevated",
                       predicted_feature_profile=_WIDE_SYN_RATIO_PROFILE),
        ],
        confidence=0.7, evidence_support=0.7, verification=0.7,
    )
    a5 = A5Response(
        benign_plausibility=0.65,
        confidence=0.65,
        evidence_support=0.65,
        verification=0.65,
        credited_hypothesis_id="a3_h1",
        cited_claim_ids=["a1_c1", "a3_h1"],
        rationale="Elevated syn ratio is unusual but a retry-loop explanation survives.",
    )
    return {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5}


def _fake_call_structured_factory(responses, captured_prompts=None):
    def fake_call_structured(*, record_id, agent, prompt_version, prompt, response_schema, model,
                              temperature, run_index, use_cache=True, extra_validate=None,
                              fault_condition=base.CLEAN_FAULT_CONDITION):
        if captured_prompts is not None:
            captured_prompts.append((agent, prompt))
        resp = responses[agent]
        if extra_validate is not None:
            extra_validate(resp)  # exercises the real validators against canned data
        return resp, base.CallMetadata(cached=False, schema_retries=0, elapsed_s=0.01,
                                        input_tokens=100, output_tokens=50)

    return fake_call_structured


@pytest.fixture
def canned(monkeypatch):
    responses = _canned_responses()
    captured_prompts = []
    monkeypatch.setattr(base, "call_structured", _fake_call_structured_factory(responses, captured_prompts))
    return responses, captured_prompts


def test_record_from_dict_round_trips(tmp_path):
    record = _make_record()
    path = tmp_path / "records.jsonl"
    path.write_text(__import__("json").dumps(record.to_dict()) + "\n")
    loaded = pipeline.load_records(path)
    assert len(loaded) == 1
    assert loaded[0].flow_id == record.flow_id
    assert loaded[0].features == record.features


def test_run_record_wires_chain_and_a4_correctly(canned):
    responses, captured_prompts = canned
    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)

    assert result.a1 is responses["a1"]
    assert result.a5.benign_plausibility == pytest.approx(0.65)

    agents_called = {agent for agent, _ in captured_prompts}
    assert agents_called == {"a1", "a2", "a3", "a4", "a5"}


def test_a4_prompt_never_receives_chain_output(canned):
    """Permanent regression guard (not a one-off check): A4's prompt must
    never carry any trace of A1/A2/A3's output for the same record --
    not their claim text, not their hypothesis text, not their claim_ids
    (whose mere presence would reveal that other agents exist even
    without repeating their content). Runs against the full canned
    chain/hypothesis fixture every time this suite runs.
    """
    responses, captured_prompts = canned
    record = _make_record()
    pipeline.run_record(record, use_cache=False)

    a4_prompts = [p for agent, p in captured_prompts if agent == "a4"]
    assert len(a4_prompts) == 1
    a4_prompt = a4_prompts[0]

    # A4's prompt is a pure function of the record (+ grounding data that
    # is itself a pure function of the record) -- rebuilding it directly,
    # with no chain in scope at all, must reproduce the exact string the
    # pipeline actually sent, proving nothing else could have been mixed
    # in via pipeline wiring.
    from agents.prompts import a4_replication_v5
    from agents.grounding import compute_nearest_neighbours, render_available_profile_features, render_empirical_grounding_block
    from controlplane.reference import get_reference_distribution, load_benign_dataframe

    reference = get_reference_distribution()
    benign_df = load_benign_dataframe()
    nn = compute_nearest_neighbours(record, benign_df, reference)
    block = render_empirical_grounding_block(record, reference, nn)
    available = render_available_profile_features(reference)
    assert a4_prompt == a4_replication_v5.build_prompt(record, block, available)

    # every claim statement and hypothesis description from A1/A2/A3
    # (canned above) must be absent verbatim -- not just the two
    # substrings a prior, narrower version of this test happened to check.
    for agent_key in ("a1", "a2", "a3"):
        resp = responses[agent_key]
        for claim in resp.claims:
            assert claim.statement not in a4_prompt, (
                f"{agent_key} claim statement leaked into A4's prompt: {claim.statement!r}"
            )
        for hyp in getattr(resp, "hypotheses", []):
            assert hyp.description not in a4_prompt, (
                f"{agent_key} hypothesis description leaked into A4's prompt: {hyp.description!r}"
            )

    # claim_id / hypothesis_id prefixes from the chain must never appear
    # either -- their presence alone would reveal that other agents exist,
    # even without repeating any of their actual content.
    for leaking_prefix in ("a1_c", "a2_c", "a3_c", "a3_h"):
        assert leaking_prefix not in a4_prompt, (
            f"chain claim/hypothesis id prefix {leaking_prefix!r} leaked into A4's prompt"
        )

    # and the reverse framing check: no language implying a chain or other
    # reviewers exists at all.
    lowered = a4_prompt.lower()
    for forbidden in ("reviewer", "prior stage", "previous stage", "chain", "other agent"):
        assert forbidden not in lowered, f"A4 prompt implies pipeline awareness: {forbidden!r}"


def test_a5_wiring_rejects_a_response_missing_a_flagged_contradiction(monkeypatch):
    """Integration test for fix 1 (the 7/20 pattern): a3's hypothesis
    flags a2_c1 as contradicting it; A5's canned response cites a1_c1
    but never a2_c1. agents.a5_verdict.run must compute a2_c1 into
    required_claim_ids and pass it through to the real validator, so
    this must be rejected even though nothing else about the canned data
    is wrong.
    """
    responses = _canned_responses()
    # mutate a3's first hypothesis to flag a claim as contradicting it
    a3_hyps = list(responses["a3"].hypotheses)
    a3_hyps[0] = a3_hyps[0].model_copy(update={"contradicting_claim_ids": ["a2_c1"]})
    responses["a3"] = responses["a3"].model_copy(update={"hypotheses": a3_hyps})
    # a5's canned response does NOT cite a2_c1
    assert "a2_c1" not in responses["a5"].cited_claim_ids

    monkeypatch.setattr(base, "call_structured", _fake_call_structured_factory(responses))

    record = _make_record()
    with pytest.raises(ValueError, match="a2_c1"):
        pipeline.run_record(record, use_cache=False)


def test_a5_credited_hypothesis_id_must_be_real(monkeypatch):
    responses = _canned_responses()
    responses["a5"] = responses["a5"].model_copy(update={"credited_hypothesis_id": "a3_h9_does_not_exist"})
    monkeypatch.setattr(base, "call_structured", _fake_call_structured_factory(responses))

    record = _make_record()
    with pytest.raises(ValueError, match="a3_h9_does_not_exist"):
        pipeline.run_record(record, use_cache=False)


def test_empirical_plausibility_cap_applied_for_zero_support_hypothesis(monkeypatch):
    """Fix 4: a hypothesis with zero empirically matching benign flows
    cannot support a plausibility above the cap -- enforced in code
    regardless of what the model reported. syn_ratio is bounded [0,1], so
    a predicted profile requiring it to be in [100, 200] can never match
    any real benign flow.
    """
    responses = _canned_responses()
    a3_hyps = list(responses["a3"].hypotheses)
    a3_hyps[0] = a3_hyps[0].model_copy(update={"predicted_feature_profile": _IMPOSSIBLE_PROFILE})
    responses["a3"] = responses["a3"].model_copy(update={"hypotheses": a3_hyps})
    responses["a5"] = responses["a5"].model_copy(
        update={"credited_hypothesis_id": "a3_h1", "benign_plausibility": 0.9}
    )
    monkeypatch.setattr(base, "call_structured", _fake_call_structured_factory(responses))

    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)

    assert result.a5_plausibility_clamped is True
    assert result.a5.benign_plausibility <= 0.3
    assert result.hypothesis_support["a3_h1"].matching_profile_count == 0


def test_empirical_plausibility_not_clamped_when_support_exists(canned):
    # the default canned fixture's credited hypothesis (a3_h1) predicts
    # syn_ratio <= 1.0, which virtually every benign flow satisfies.
    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)
    assert result.a5_plausibility_clamped is False
    assert result.a5.benign_plausibility == pytest.approx(0.65)
    assert result.hypothesis_support["a3_h1"].matching_profile_count > 0


def test_nearest_neighbours_computed_for_every_record(canned):
    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)
    assert result.nearest_neighbours.k_found > 0
    assert "syn_ratio" in result.nearest_neighbours.features_used


def test_hypothesis_support_computed_for_all_chain_and_a4_hypotheses(canned):
    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)
    expected_ids = {"a3_h1", "a3_h2", "a3_h3", "a4_h1", "a4_h2", "a4_h3"}
    assert set(result.hypothesis_support) == expected_ids


def test_trust_scores_computed_for_all_five_agents(canned):
    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)
    assert set(result.trust_scores) == {"a1", "a2", "a3", "a4", "a5"}
    # a1's claim (syn_ratio, greater_than_typical) is corroborated by a4_c1
    assert result.trust_scores["a1"].corroborated == 1
    assert result.trust_scores["a1"].V == pytest.approx(1.0)
    assert result.trust_scores["a1"].E == pytest.approx(1.0)  # asserted_value matches the record


def test_trigger_only_feature_name_survives_evidence_check_via_prefix(monkeypatch):
    """Regression test for the stop-point-2 finding: a claim about a
    trigger-reason-only feature (never present in the CICFlowMeter
    `features` dict at all -- unlike "syn_ratio" in the other canned
    fixture, which happens to exist in both) must resolve via the
    TRIGGER_FEATURE_PREFIX-prefixed lookup, not get flagged as a
    hallucination just because the bare name isn't a features dict key.
    """
    record = EscalationRecord(
        flow_id="trigger-only-flow",
        trigger_reasons=[
            TriggerReason(
                feature="flow_bytes_per_sec", observed_value=999.0, threshold=500.0,
                direction="high", ratio=2.0, low_confidence=False,
            )
        ],
        features={"flow_byts_s": 999000.0},  # deliberately NOT "flow_bytes_per_sec"
        packet_window=PacketWindowSummary(packet_count=1, byte_count=1, duration_us=1),
        selector_config_hash="deadbeef",
    )

    profile = [PredictedRange(feature=f"{TRIGGER_FEATURE_PREFIX}flow_bytes_per_sec", expected_min=0.0, expected_max=1e12)]

    a1 = ClaimsResponse(
        claims=[
            Claim(
                claim_id="a1_c1",
                statement="the trigger-reported byte rate is elevated",
                referenced_features=[
                    FeatureReference(
                        name=f"{TRIGGER_FEATURE_PREFIX}flow_bytes_per_sec",
                        asserted_value=999.0,
                        relation=Relation.GREATER_THAN_TYPICAL,
                    )
                ],
                confidence=0.9,
            )
        ],
        confidence=0.9, evidence_support=0.9, verification=0.9,
    )
    a2 = ClaimsResponse(claims=[], confidence=0.5, evidence_support=0.5, verification=0.5)

    def make_hyps(prefix):
        return [
            Hypothesis(hypothesis_id=f"{prefix}_h1", description="d", benign=True, prior_plausibility=0.5,
                       prediction="some measurable feature should behave a specific way",
                       predicted_feature_profile=profile),
            Hypothesis(hypothesis_id=f"{prefix}_h2", description="d", benign=True, prior_plausibility=0.5,
                       prediction="some other measurable feature should behave a specific way",
                       predicted_feature_profile=profile),
            Hypothesis(hypothesis_id=f"{prefix}_h3", description="d", benign=False, prior_plausibility=0.2,
                       prediction="yet another measurable feature should behave a specific way",
                       predicted_feature_profile=profile),
        ]

    a3 = HypothesisResponse(claims=[], hypotheses=make_hyps("a3"), confidence=0.7, evidence_support=0.7, verification=0.7)
    a4 = HypothesisResponse(claims=[], hypotheses=make_hyps("a4"), confidence=0.7, evidence_support=0.7, verification=0.7)
    a5 = A5Response(
        benign_plausibility=0.5, confidence=0.5, evidence_support=0.5, verification=0.5,
        cited_claim_ids=["a1_c1"], rationale="r",
    )

    responses = {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5}
    monkeypatch.setattr(base, "call_structured", _fake_call_structured_factory(responses))

    result = pipeline.run_record(record, use_cache=False)
    assert result.trust_scores["a1"].E == pytest.approx(1.0), (
        "trigger-only feature name should resolve via the prefix, not be flagged as hallucinated"
    )


def test_trust_decay_never_includes_a4(canned):
    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)
    if result.trust_decay_chain is not None:
        assert "a4" not in result.trust_decay_chain.stage_labels[0]
        assert all("a4" not in label for label in result.trust_decay_chain.stage_labels)
    # exactly two gaps over [a1, a2, a3]
    if result.trust_decay_chain is not None:
        assert len(result.trust_decay_chain.drops) == 2


def test_chain_vs_independent_is_computed(canned):
    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)
    assert result.chain_vs_independent is not None


def test_contamination_report_present_per_agent(canned):
    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)
    assert set(result.contamination) == {"a1", "a2", "a3", "a4", "a5"}
    for report in result.contamination.values():
        assert report.is_clean  # canned responses above are clean prose


def test_result_serializes_to_json_safely(canned):
    import json

    record = _make_record()
    result = pipeline.run_record(record, use_cache=False)
    # round-trips through json.dumps without raising
    json.dumps(result.to_dict())


# ---------- ground truth must never reach a prompt ----------

_LABEL_WORDS = ["BENIGN", "Bot", "PortScan", "DDoS", "Infiltration", "Heartbleed", "Patator"]


def test_escalation_record_has_no_label_field():
    field_names = {f.name for f in dataclasses.fields(EscalationRecord)}
    assert not any("label" in name.lower() for name in field_names)


def test_no_ground_truth_label_reaches_any_prompt(canned):
    _, captured_prompts = canned
    record = _make_record()
    pipeline.run_record(record, use_cache=False)

    for agent, prompt in captured_prompts:
        for word in _LABEL_WORDS:
            assert not re.search(rf"\b{re.escape(word)}\b", prompt), (
                f"ground-truth-shaped word {word!r} leaked into {agent}'s prompt"
            )
        assert "label" not in prompt.lower()
