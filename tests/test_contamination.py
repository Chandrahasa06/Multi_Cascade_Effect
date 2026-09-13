import importlib
import pkgutil

from agents.contamination import scan_agent_response, scan_prompt, scan_text, summarize, ContaminationReport
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


def _claim(text, claim_id="a1_c1"):
    ref = FeatureReference(name="flow_duration", asserted_value=1.0, relation=Relation.EQUALS)
    return Claim(claim_id=claim_id, statement=text, referenced_features=[ref], confidence=0.5)


def test_scan_text_catches_attack_tool_names():
    report = scan_text("This looks like an Nmap SYN scan of the target.")
    assert "nmap" in report.lexicon_hits


def test_scan_text_catches_framing_words():
    report = scan_text("This flow is clearly malicious and part of an attack.")
    assert "malicious" in report.lexicon_hits
    assert "attack" in report.lexicon_hits


def test_scan_text_word_boundaries_avoid_false_positives():
    # "hydra" is banned but "hydrated"/"hydraulic" should not trip it.
    report = scan_text("The system stayed well hydrated and hydraulic pressure held.")
    assert "hydra" not in report.lexicon_hits


def test_scan_text_clean_prose_has_no_hits():
    report = scan_text(
        "The source sent more packets than it received and the exchange never completed."
    )
    assert report.is_clean


def test_scan_text_catches_ip_mac_and_timestamp():
    report = scan_text("Traffic from 192.168.10.5 to AA:BB:CC:DD:EE:FF at 2017-07-07 08:12:00.")
    assert report.ip_hits == ["192.168.10.5"]
    assert report.mac_hits == ["AA:BB:CC:DD:EE:FF"]
    assert report.timestamp_hits
    assert not report.is_clean


def test_scan_agent_response_covers_claims_hypotheses_and_rationale():
    claims_resp = ClaimsResponse(
        claims=[_claim("nmap-style behavior observed")],
        confidence=0.5, evidence_support=0.5, verification=0.5,
    )
    report = scan_agent_response(claims_resp)
    assert "nmap" in report.lexicon_hits

    hyp = Hypothesis(
        hypothesis_id="a3_h1",
        description="this is a botnet beaconing pattern",
        benign=False,
        prior_plausibility=0.3,
        prediction="inter-arrival times should be tightly periodic",
        predicted_feature_profile=[PredictedRange(feature="trigger_flow_iat_mean", expected_max=100.0)],
    )
    hyp_resp = HypothesisResponse(
        claims=[], hypotheses=[hyp], confidence=0.5, evidence_support=0.5, verification=0.5,
    )
    report2 = scan_agent_response(hyp_resp)
    assert "botnet" in report2.lexicon_hits

    a5 = A5Response(
        benign_plausibility=0.1,
        confidence=0.8,
        evidence_support=0.8,
        verification=0.8,
        cited_claim_ids=["a1_c1"],
        rationale="This is a classic DDoS attack signature.",
    )
    report3 = scan_agent_response(a5)
    assert "ddos" in report3.lexicon_hits
    assert "attack" in report3.lexicon_hits


def test_scan_agent_response_covers_hypothesis_prediction_field():
    hyp = Hypothesis(
        hypothesis_id="a3_h1",
        description="a benign monitoring poll",
        benign=True,
        prior_plausibility=0.5,
        prediction="if this were an nmap-style scan, ports touched should be sequential",
        predicted_feature_profile=[
            PredictedRange(feature="trigger_distinct_dst_ports_per_src", expected_max=100.0)
        ],
    )
    hyp_resp = HypothesisResponse(
        claims=[], hypotheses=[hyp], confidence=0.5, evidence_support=0.5, verification=0.5,
    )
    report = scan_agent_response(hyp_resp)
    assert "nmap" in report.lexicon_hits


def test_summarize_computes_per_agent_contamination_rate():
    clean = ContaminationReport()
    dirty = ContaminationReport(lexicon_hits=["nmap"])
    summary = summarize({"a1": [clean, clean, dirty]})["a1"]
    assert summary.per_agent_responses == 3
    assert summary.per_agent_contaminated == 1
    assert abs(summary.contamination_rate - (1 / 3)) < 1e-9
    assert summary.hit_counts["nmap"] == 1


def test_prompt_templates_never_contain_banned_terms():
    # every prompt version ever shipped, not just the currently-wired
    # ones -- old versions stay importable (agents/prompts/__init__.py's
    # own docstring: a cache-key history reason), so a leak in a
    # superseded version should still fail this test, not go unchecked
    # just because nothing currently points at it.
    modules = [
        "a1_evidence_v1",
        "a1_evidence_v2",
        "a1_evidence_v3",
        "a1_evidence_v4",
        "a1_evidence_v5",
        "a2_behaviour_v1",
        "a2_behaviour_v2",
        "a3_hypotheses_v1",
        "a3_hypotheses_v2",
        "a3_hypotheses_v3",
        "a3_hypotheses_v4",
        "a3_hypotheses_v5",
        "a4_replication_v1",
        "a4_replication_v2",
        "a4_replication_v3",
        "a4_replication_v4",
        "a4_replication_v5",
        "a5_verdict_v1",
        "a5_verdict_v2",
        "a5_verdict_v3",
        "a5_verdict_v4",
        "a5_verdict_v5",
        "baseline_v1",
        "baseline_v2",
    ]
    for name in modules:
        mod = importlib.import_module(f"agents.prompts.{name}")
        # every prompt module exposes _INSTRUCTIONS as its free-text body
        text = getattr(mod, "_INSTRUCTIONS")
        report = scan_prompt(text)
        assert report.is_clean, f"{name} leaks banned terms: {report.to_dict()}"


def test_calibration_note_never_contains_banned_terms():
    from agents.prompts.render import CALIBRATION_NOTE

    report = scan_prompt(CALIBRATION_NOTE)
    assert report.is_clean, f"CALIBRATION_NOTE leaks banned terms: {report.to_dict()}"


def test_self_report_instructions_never_contain_banned_terms():
    from agents.prompts.render import SELF_REPORT_INSTRUCTIONS, SELF_REPORT_INSTRUCTIONS_A5

    for name, text in (
        ("SELF_REPORT_INSTRUCTIONS", SELF_REPORT_INSTRUCTIONS),
        ("SELF_REPORT_INSTRUCTIONS_A5", SELF_REPORT_INSTRUCTIONS_A5),
    ):
        report = scan_prompt(text)
        assert report.is_clean, f"{name} leaks banned terms: {report.to_dict()}"


def test_a4_prompt_never_mentions_other_reviewers():
    from agents.prompts import (
        a4_replication_v1,
        a4_replication_v2,
        a4_replication_v3,
        a4_replication_v4,
        a4_replication_v5,
    )

    for mod in (
        a4_replication_v1, a4_replication_v2, a4_replication_v3, a4_replication_v4, a4_replication_v5,
    ):
        text = mod._INSTRUCTIONS.lower()
        for forbidden in ("reviewer", "prior stage", "chain", "other agent", "a1", "a2", "a3", "a5"):
            assert forbidden not in text, f"{mod.PROMPT_VERSION} leaks pipeline awareness: {forbidden!r}"
