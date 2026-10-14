import json

import pytest

from agents import base, openrouter
from agents import selfcheck_core as core
from agents import selfcheck_pipeline as sp
from agents.prompts import selfcheck_prompts_v1 as v1, selfcheck_prompts_v3 as v3
from agents.schema import CitedValue, SCClaim, SCClaimsResponse
from eval import run_selfcheck as rs

BWD, FWD = ("Total Length of Bwd Packets", 11632.0), ("Total Length of Fwd Packets", 1362.0)
DUR = ("Flow Duration", 8527728.0)


def claim(statement, *cited, cid="a1_c1"):
    return SCClaim(claim_id=cid, statement=statement, cited=[CitedValue(name=n, value=v) for n, v in cited],
                   confidence=0.9)


def verdict(statement, *cited):
    try:
        sp.validate_claims([claim(statement, *cited)], "a1_c", ban=True)
        return "ok"
    except ValueError as exc:
        return str(exc)


# ---------------------------------------------------------------- comparison rule
def test_the_known_leak_is_now_rejected():
    assert "outside this flow" in verdict("Flow Duration is far longer than most connections.", DUR)


@pytest.mark.parametrize("word", core.COMPARATIVES)
def test_comparative_between_two_cited_fields_is_allowed(word):
    assert verdict(f"Total Length of Bwd Packets is {word} than Total Length of Fwd Packets.", BWD, FWD) == "ok"


@pytest.mark.parametrize("word", core.COMPARATIVES)
def test_comparative_with_one_named_field_is_rejected(word):
    assert word in verdict(f"Total Length of Bwd Packets is {word} than the request.", BWD, FWD)
    assert word in verdict(f"Flow Duration is {word}.", DUR)


def test_second_field_must_be_named_not_just_cited():
    assert "larger" in verdict("Total Length of Bwd Packets is larger.", BWD, FWD)


@pytest.mark.parametrize("word", core.JUDGE_WORDS + core.MAGNITUDE_WORDS)
def test_judgement_words_are_always_rejected_even_between_two_fields(word):
    assert word in verdict(f"Total Length of Bwd Packets is {word} next to Total Length of Fwd Packets.", BWD, FWD)


@pytest.mark.parametrize("phrase", [
    "than most", "than usual", "than typical", "compared to other flows", "than other connections",
    "than the average", "relative to other traffic", "for a TCP connection", "than is normal",
])
def test_external_targets_are_rejected_whatever_the_word(phrase):
    out = verdict(f"Total Length of Bwd Packets is larger than Total Length of Fwd Packets, {phrase}.", BWD, FWD)
    assert "outside this flow" in out


def test_comparisons_inside_the_flow_without_banned_words_still_pass():
    assert verdict("Total Length of Bwd Packets exceeds Total Length of Fwd Packets.", BWD, FWD) == "ok"
    assert verdict("Flow Duration is 8527728 microseconds.", DUR) == "ok"


def test_every_violation_is_reported_at_once():
    claims = [claim("Flow Duration is unusually large, longer than most.", DUR),
              claim("The flow sent replies.", FWD, cid="a1_c2")]
    with pytest.raises(ValueError) as exc:
        sp.validate_claims(claims, "a1_c", ban=True)
    msg = str(exc.value)
    for part in ("unusually", "large", "longer", "than most", "a1_c2: the sentence must name"):
        assert part in msg, part


# ---------------------------------------------------------------- field-name rule
@pytest.mark.parametrize("statement,field", [
    ("Total Fwd Packets is 12.", "Total Fwd Packets"),
    ("Total Forward Packets is 12.", "Total Fwd Packets"),
    ("Total Bwd Packets is 6.", "Total Backward Packets"),
    ("Init Win bytes forward is 0.", "Init_Win_bytes_forward"),
    ("init_win_bytes_fwd is 0.", "Init_Win_bytes_forward"),
    ("`total fwd packets` = 12.", "Total Fwd Packets"),
])
def test_field_name_variants_are_accepted(statement, field):
    sp.validate_claims([claim(statement, (field, 1.0))], "a1_c")


@pytest.mark.parametrize("statement", ["The flow sent 12 forward packets.", "Twelve packets went out."])
def test_paraphrase_that_names_no_field_is_rejected(statement):
    with pytest.raises(ValueError, match="must name"):
        sp.validate_claims([claim(statement, ("Total Fwd Packets", 12.0))], "a1_c")


# ---------------------------------------------------------------- prompt
def test_a1_v3_shows_the_complete_rule_and_v1_did_not():
    text = v3.a1(v3.flow_block({}, None, None))
    for w in core.JUDGE_WORDS + core.MAGNITUDE_WORDS + core.COMPARATIVES:
        assert w in text, w
    assert "large" not in v1.a1(v1.flow_block({}, None, None))
    assert set(v3.VERSIONS.values()).isdisjoint(v1.VERSIONS.values())


def test_v3_prompts_pass_the_runner_statistic_checks():
    flow = v3.flow_block({"Flow Duration": 1000.0, "Destination Port": 80.0}, "TCP", "10:15")
    nums = rs.template_numbers(v3)
    for t in rs.template_prompts(flow, v3).values():
        rs.check_prompt(t, t, set(), {1000.0, 80.0, 10.0, 15.0}, nums)


# ---------------------------------------------------------------- persistence of rejected attempts
SLUG = json.loads(openrouter.CONFIG_PATH.read_text(encoding="utf-8"))["shakedown_model"]


def _fake(texts):
    it = iter(texts)

    def gen(model, prompt, schema, temperature):
        return next(it), {"generation_id": "g", "provider": "Anthropic", "model": model, "cost_usd": 0.01,
                          "input_tokens": 1, "output_tokens": 1, "finish_reason": "stop"}
    return gen


def test_rejected_attempt_text_is_saved_with_message_and_cost(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)
    bad = json.dumps({"claims": [claim("Flow Duration is huge, larger than most.", DUR).model_dump()]})
    good = json.dumps({"claims": []})
    monkeypatch.setattr(openrouter, "generate", _fake([bad, good]))
    _, meta = base.call_structured(record_id="r", agent="a1", prompt_version="v", prompt="p",
                                   response_schema=SCClaimsResponse, model=SLUG,
                                   temperature=openrouter.temperature_for(SLUG),
                                   extra_validate=lambda r: sp.validate_claims(r.claims, "a1_c", ban=True))
    rej = meta.rejected_attempts
    assert len(rej) == 1 and rej[0]["text"] == bad and "than most" in rej[0]["reason"]
    assert rej[0]["call"]["cost_usd"] == 0.01
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["rejected_attempts"] == rej


def test_failed_call_carries_every_rejected_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(openrouter, "generate", _fake(["x"] * base.MAX_SCHEMA_RETRIES))
    with pytest.raises(base.SchemaValidationFailed) as exc:
        base.call_structured(record_id="r", agent="a1", prompt_version="v", prompt="p",
                             response_schema=SCClaimsResponse, model=SLUG,
                             temperature=openrouter.temperature_for(SLUG))
    assert [a["text"] for a in exc.value.rejected_attempts] == ["x"] * base.MAX_SCHEMA_RETRIES
    assert base.MAX_SCHEMA_RETRIES == 3


def test_upstream_from_another_prompt_version_is_refused(tmp_path):
    up = {"prompt_versions": {k: v1.VERSIONS[k] for k in ("a1", "a2", "a3", "a4")}}
    with pytest.raises(RuntimeError, match="upstream written under"):
        rs.check_upstream_versions(up, v3, tmp_path / "x.json")
    with pytest.raises(RuntimeError):
        rs.check_upstream_versions({}, v3, tmp_path / "x.json")
    rs.check_upstream_versions({}, v1, tmp_path / "x.json")
