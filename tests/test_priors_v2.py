"""Tests for the priors-v2 run (tenth attempt): no computed statistic and no flag column in any
prompt, no cap, no label reaches a prompt, protocol comes from the flow's own row, and the
v2 prior text makes no handshake or flag argument."""
import inspect
import json
from pathlib import Path

import pytest

from agents.prompts import a1_priors_v1, a1_priors_v2, a2_priors_v2, a3_priors_v2, a4_priors_v2, a5_priors_v2
from agents.prompts import priors_text_v1, priors_text_v2
from agents.prompts.priors_render_v2 import FLAG_COLUMNS, flow_block, profile_feature_names, shown_features
from eval import run_priors_v2 as rp2
from eval.run_blind_pipeline_45_v2 import load_45

MODULES = (a1_priors_v2, a2_priors_v2, a3_priors_v2, a4_priors_v2, a5_priors_v2)


@pytest.fixture(scope="module")
def record():
    return load_45()[0]


def test_v2_prior_text_has_no_flag_or_handshake_argument_except_the_rule():
    import re
    text = priors_text_v2.PRIORS_TEXT.lower()
    assert not re.findall(r"\b(syn|ack|fin|urg|psh|rst|syn-ack)\b", text)
    assert "these measurements do not record the tcp handshake" in text
    assert "cicids" not in text and "2017" not in text
    assert "not by itself exculpatory" in text


def test_v2_prior_text_has_tcp_udp_and_any_protocol_sections():
    text = priors_text_v2.PRIORS_TEXT
    assert "TCP flows:" in text and "UDP flows:" in text and "Any protocol:" in text
    assert all(src.strip() for _sid, src, _t in priors_text_v2.PRIOR_STATEMENTS)


def test_flow_block_hides_every_flag_column(record):
    text = flow_block(record["features"], "TCP", "10:15")
    for col in FLAG_COLUMNS:
        assert col not in text
    assert set(shown_features(record["features"])).isdisjoint(FLAG_COLUMNS)
    assert set(profile_feature_names(record["features"])).isdisjoint(FLAG_COLUMNS)


def test_builder_text_of_all_five_prompts_passes_for_a_real_record(record):
    feats = record["features"]
    flow = flow_block(feats, "TCP", "10:15")
    for text in rp2.builder_prompts(flow, profile_feature_names(feats)).values():
        rp2.check_prompt(text, text, set(), {float(v) for v in shown_features(feats).values()})


def test_check_rejects_a_flag_column_and_count_phrasing():
    with pytest.raises(AssertionError):
        rp2.check_prompt("ok", "SYN Flag Count = 0", set(), set())
    with pytest.raises(AssertionError):
        rp2.check_prompt("of which 31 are within 2x", "ok", set(), set())
    with pytest.raises(AssertionError):
        rp2.check_prompt("ok", "The count was 98765.", {98765.0}, set())


def test_prompt_builders_take_no_statistic_or_label():
    bad = ("count", "support", "neighbour", "percentile", "history", "label", "grounding", "evidence")
    for m in MODULES:
        names = list(inspect.signature(m.build_prompt).parameters)
        assert not [n for n in names for b in bad if b in n.lower()], (m.__name__, names)


def test_runner_applies_no_cap_and_reads_no_label_or_old_output():
    src = inspect.getsource(rp2)
    assert "apply_ungrounded_neighbourhood_cap" not in src and "apply_empirical_plausibility_cap" not in src
    assert "agent_key" not in src and "true_label" not in src and "is_attack" not in src
    for old in ("agent_run_45_v2", "agent_run_45_v3", "agent_priors.jsonl"):
        assert old not in src


def test_protocol_lookup_reads_no_label():
    from eval import record_protocols
    src = inspect.getsource(record_protocols)
    assert '"Label"' not in src and "'Label'" not in src


def test_versions_are_new_and_v1_still_importable():
    assert [m.PROMPT_VERSION for m in MODULES] == [f"a{i}_priors_v2" for i in range(1, 6)]
    assert a1_priors_v1.PROMPT_VERSION == "a1_priors_v1"
    assert priors_text_v1.PRIORS_VERSION == "priors_text_v1"


def test_recorded_run_if_present():
    p = Path("results/agent_priors_v2.jsonl")
    if not p.exists():
        pytest.skip("run output not present")
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            assert r["cap_applied"] is False
            assert r["benign_plausibility"] == r["a5"]["benign_plausibility"]
            assert r["prompt_versions"] == {f"a{i}": f"a{i}_priors_v2" for i in range(1, 6)}
            assert not any(m["cached"] for m in r["stage_meta"].values())
