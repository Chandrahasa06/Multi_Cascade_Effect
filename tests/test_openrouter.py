import inspect
import json

import pytest

from agents import base, openrouter
from agents.prompts import selfcheck_prompts_v1 as v1, selfcheck_prompts_v2 as v2
from agents.schema import SCBlindResponse, SCClaimsResponse, SCVerdictResponse

CFG = json.loads(openrouter.CONFIG_PATH.read_text(encoding="utf-8"))
SLUG = CFG["shakedown_model"]


def _write_cfg(tmp_path, cfg):
    p = tmp_path / "openrouter.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    openrouter.load_config.cache_clear()
    return p


@pytest.fixture(autouse=True)
def _reset_config():
    openrouter.load_config.cache_clear()
    yield
    openrouter.load_config.cache_clear()


# ---------------------------------------------------------------- config
@pytest.mark.parametrize("where", ["models", "sequence", "shakedown_model"])
def test_free_router_is_rejected_loudly(tmp_path, where):
    cfg = {"api_base": "x", "shakedown_model": "a/b", "sequence": [], "models": {"a/b": {"provider": "p"}}}
    if where == "models":
        cfg["models"]["openrouter/free"] = {"provider": "p"}
    elif where == "sequence":
        cfg["sequence"] = ["openrouter/free"]
    else:
        cfg["shakedown_model"] = "openrouter/free"
    with pytest.raises(openrouter.OpenRouterConfigError, match="free router"):
        openrouter.load_config(_write_cfg(tmp_path, cfg))


def test_every_configured_model_pins_a_provider(tmp_path):
    cfg = {"api_base": "x", "shakedown_model": "a/b", "sequence": [], "models": {"a/b": {}}}
    with pytest.raises(openrouter.OpenRouterConfigError, match="no pinned provider"):
        openrouter.load_config(_write_cfg(tmp_path, cfg))


def test_no_model_slug_is_hardcoded_in_code():
    from eval import run_eval20

    src = "".join(inspect.getsource(m) for m in (openrouter, base, run_eval20))
    for slug in set(CFG["models"]) | set(CFG["sequence"]):
        assert slug not in src, slug


def test_gemini_default_is_not_routed_to_openrouter():
    assert not openrouter.is_configured(base.DEFAULT_MODEL)
    assert openrouter.is_configured(SLUG)


# ---------------------------------------------------------------- schema
@pytest.mark.parametrize("cls", [SCBlindResponse, SCClaimsResponse, SCVerdictResponse])
def test_strict_schema_is_closed_and_portable(cls):
    schema = openrouter.strict_schema(cls)

    def walk(node):
        if isinstance(node, dict):
            for k in openrouter._DROP_KEYS:
                assert k not in node
            assert node.get("minItems", 0) <= 1
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert sorted(node["required"]) == sorted(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)


# ---------------------------------------------------------------- request
def _fake_body(content="{}", provider="Anthropic"):
    return {"id": "gen-1", "provider": provider, "model": SLUG,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001}}


def test_request_pins_provider_without_fallback(monkeypatch):
    sent = {}
    monkeypatch.setattr(openrouter, "verify", lambda m: {"provider_name": "Anthropic"})
    monkeypatch.setattr(openrouter, "_post", lambda payload: (sent.update(payload) or _fake_body(), []))
    _, meta = openrouter.generate(SLUG, "p", SCClaimsResponse, openrouter.temperature_for(SLUG))
    prov = CFG["models"][SLUG]["provider"]
    assert sent["provider"] == {"order": [prov], "only": [prov], "allow_fallbacks": False, "require_parameters": True}
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["model"] == SLUG
    assert isinstance(meta.pop("latency_s"), float)
    assert meta.pop("http_retries") == [] and meta.pop("retry_wait_s") == 0
    assert isinstance(meta.pop("latency_excl_retry_s"), float)
    assert meta == {"generation_id": "gen-1", "provider": "Anthropic", "model": SLUG, "requested_model": SLUG,
                    "pinned_provider": prov, "cost_usd": 0.0001, "input_tokens": 10, "output_tokens": 5,
                    "finish_reason": "stop"}


def test_served_by_another_provider_fails_loudly(monkeypatch):
    monkeypatch.setattr(openrouter, "verify", lambda m: {"provider_name": "Anthropic"})
    monkeypatch.setattr(openrouter, "_post", lambda payload: (_fake_body(provider="Somebody Else"), []))
    with pytest.raises(openrouter.OpenRouterError, match="pinned"):
        openrouter.generate(SLUG, "p", SCClaimsResponse, openrouter.temperature_for(SLUG))


def test_temperature_must_match_config(monkeypatch):
    with pytest.raises(openrouter.OpenRouterConfigError, match="temperature"):
        openrouter.generate(SLUG, "p", SCClaimsResponse, 0.123)


def test_call_structured_saves_provider_model_cost_and_keeps_validators(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)
    good = json.dumps({"claims": []})
    replies = iter([("not json", 1), (good, 2)])

    def fake_generate(model, prompt, schema, temperature):
        text, n = next(replies)
        return text, {"generation_id": f"gen-{n}", "provider": "Anthropic", "model": model, "requested_model": model,
                      "pinned_provider": "anthropic", "cost_usd": 0.001 * n, "input_tokens": 1, "output_tokens": 1,
                      "finish_reason": "stop"}

    monkeypatch.setattr(openrouter, "generate", fake_generate)
    checked = []
    out, meta = base.call_structured(record_id="r", agent="a1", prompt_version="v", prompt="p",
                                     response_schema=SCClaimsResponse, model=SLUG,
                                     temperature=openrouter.temperature_for(SLUG),
                                     extra_validate=lambda r: checked.append(r))
    assert checked and meta.schema_retries == 1
    assert meta.provider == "Anthropic" and meta.served_model == SLUG
    assert meta.generation_ids == ["gen-1", "gen-2"] and meta.cost_usd == pytest.approx(0.003)
    rec = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert rec["provider"] == "Anthropic" and rec["generation_ids"] == ["gen-1", "gen-2"]
    assert rec["cost_usd"] == pytest.approx(0.003) and len(rec["calls"]) == 2
    _, cached = base.call_structured(record_id="r", agent="a1", prompt_version="v", prompt="p",
                                     response_schema=SCClaimsResponse, model=SLUG,
                                     temperature=openrouter.temperature_for(SLUG))
    assert cached.cached and cached.cost_usd == pytest.approx(0.003) and cached.provider == "Anthropic"


def test_validation_failure_keeps_per_attempt_spend(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(openrouter, "generate", lambda *a: ("bad", {"generation_id": "g", "cost_usd": 0.01,
                                                                     "provider": "Anthropic", "model": SLUG,
                                                                     "input_tokens": 1, "output_tokens": 1}))
    with pytest.raises(base.SchemaValidationFailed) as exc:
        base.call_structured(record_id="r", agent="a4", prompt_version="v", prompt="p",
                             response_schema=SCClaimsResponse, model=SLUG,
                             temperature=openrouter.temperature_for(SLUG))
    assert len(exc.value.calls) == base.MAX_SCHEMA_RETRIES


# ---------------------------------------------------------------- prompts and isolation
def test_v2_prompts_drop_only_the_json_shape_prose():
    flow = v1.flow_block({"Flow Duration": 1.0}, "TCP", "10:00")
    for p in (v2.a1(flow), v2.a2(flow, ""), v2.a3(flow, "", ""), v2.a4(flow), v2.a5a(flow)):
        assert '{"' not in p
    assert v2.a5b("c", "j", ["C1"]) == v1.a5b("c", "j", ["C1"])
    assert set(v2.VERSIONS.values()).isdisjoint(v1.VERSIONS.values())
    for rule in ("3 to 5, at least 2 benign", "exactly four entries", "names at least one field"):
        assert rule in v2.a4(flow)


def test_eval20_runner_reads_no_label():
    from eval import run_eval20

    src = inspect.getsource(run_eval20) + inspect.getsource(openrouter)
    for word in ("eval20_records", "agent_key", "true_label", "is_attack", "eval20_input.jsonl"):
        assert word not in src, word


def test_eval20_upstream_cache_is_per_model():
    from eval import run_eval20

    a, b = run_eval20.run_dir("x/one"), run_eval20.run_dir("x/two")
    assert a != b and a.parent == b.parent


def test_disclosed_quantization_is_pinned_too(monkeypatch):
    sent = {}
    monkeypatch.setattr(openrouter, "verify", lambda m: {"provider_name": "Anthropic", "quantization": "fp8"})
    monkeypatch.setattr(openrouter, "_post", lambda payload: (sent.update(payload) or _fake_body(), []))
    openrouter.generate(SLUG, "p", SCClaimsResponse, openrouter.temperature_for(SLUG))
    assert sent["provider"]["quantizations"] == ["fp8"]
    monkeypatch.setattr(openrouter, "verify", lambda m: {"provider_name": "Anthropic", "quantization": "unknown"})
    sent.clear()
    openrouter.generate(SLUG, "p", SCClaimsResponse, openrouter.temperature_for(SLUG))
    assert "quantizations" not in sent["provider"]


def test_sequence_is_five_models_none_of_them_a_router_or_variant():
    seq = CFG["sequence"]
    assert len(seq) == len(set(seq)) and all(s in CFG["models"] for s in seq)
    assert not any(s.startswith("~") or ":" in s or s.startswith("openrouter/") for s in seq)
