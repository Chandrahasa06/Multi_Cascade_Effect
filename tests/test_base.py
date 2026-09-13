import json

import pytest
from pydantic import BaseModel

from agents import base


class _Resp(BaseModel):
    value: int


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(base, "DAILY_COUNTER_PATH", tmp_path / "_daily_request_count.json")
    monkeypatch.setattr(base, "_get_client", lambda: object())
    # `_bucket` is a process-wide singleton (deliberately, for real rate
    # limiting) whose `_next_slot` bookkeeping advances even when sleep is
    # faked -- without resetting it here, an earlier test's faked sleep
    # leaves a "debt" that a later test (with real time.sleep) pays for as
    # an actual multi-second wait. Always patch sleep to a no-op and reset
    # the bucket before every test in this file, not just the ones that
    # exercise it directly.
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    monkeypatch.setattr(base.random, "uniform", lambda a, b: 0)
    monkeypatch.setattr(base._bucket, "_next_slot", 0.0)
    yield


def test_cache_key_changes_with_every_input():
    base_key = base._cache_key("r1", "a1", "v1", "m1", 0.7, 0)
    assert base_key != base._cache_key("r2", "a1", "v1", "m1", 0.7, 0)
    assert base_key != base._cache_key("r1", "a2", "v1", "m1", 0.7, 0)
    assert base_key != base._cache_key("r1", "a1", "v2", "m1", 0.7, 0)
    assert base_key != base._cache_key("r1", "a1", "v1", "m2", 0.7, 0)
    assert base_key != base._cache_key("r1", "a1", "v1", "m1", 0.9, 0)
    assert base_key != base._cache_key("r1", "a1", "v1", "m1", 0.7, 1)
    assert base_key == base._cache_key("r1", "a1", "v1", "m1", 0.7, 0)


def test_cache_key_default_fault_condition_is_unchanged_from_before_fault_injection():
    # backward compatibility is the entire point: every cached response
    # written before fault_condition existed was implicitly "clean", and
    # must still resolve to the exact same key, or every already-paid-for
    # entry under results/agent_cache/ becomes unreachable.
    assert (
        base._cache_key("r1", "a1", "v1", "m1", 0.7, 0)
        == base._cache_key("r1", "a1", "v1", "m1", 0.7, 0, fault_condition="clean")
        == base._cache_key("r1", "a1", "v1", "m1", 0.7, 0, fault_condition=base.CLEAN_FAULT_CONDITION)
    )


def test_cache_key_distinguishes_fault_conditions_from_clean_and_each_other():
    clean = base._cache_key("r1", "a1", "v1", "m1", 0.7, 0)
    missing_evidence = base._cache_key("r1", "a1", "v1", "m1", 0.7, 0, fault_condition="missing_evidence")
    incorrect_behavior = base._cache_key("r1", "a1", "v1", "m1", 0.7, 0, fault_condition="incorrect_behavior")
    assert len({clean, missing_evidence, incorrect_behavior}) == 3


def test_call_structured_clean_and_faulted_runs_never_collide(monkeypatch):
    # end-to-end version of the key-uniqueness property above: a clean
    # call and a fault-injected call for the SAME (record_id, agent,
    # prompt_version, model, temperature, run_index) must be cached and
    # served independently, never one silently overwriting or being
    # returned for the other.
    responses = iter(['{"value": 1}', '{"value": 2}'])

    def fake_raw_generate(client, model, prompt, response_schema, temperature):
        return next(responses), 1, 1

    monkeypatch.setattr(base, "_raw_generate", fake_raw_generate)

    kwargs = dict(
        record_id="r1", agent="a1", prompt_version="v1", prompt="p",
        response_schema=_Resp, model="m", temperature=0.7, run_index=0,
    )
    clean_resp, clean_meta = base.call_structured(**kwargs)
    faulted_resp, faulted_meta = base.call_structured(**kwargs, fault_condition="missing_evidence")

    assert clean_resp.value == 1
    assert faulted_resp.value == 2
    assert clean_meta.cached is False and faulted_meta.cached is False  # neither served the other's cache entry

    # re-requesting each again now hits its own cache entry, not the other's.
    clean_again, clean_again_meta = base.call_structured(**kwargs)
    faulted_again, faulted_again_meta = base.call_structured(**kwargs, fault_condition="missing_evidence")
    assert clean_again.value == 1 and clean_again_meta.cached is True
    assert faulted_again.value == 2 and faulted_again_meta.cached is True


def test_call_structured_hits_cache_on_second_call(monkeypatch):
    calls = {"n": 0}

    def fake_raw_generate(client, model, prompt, response_schema, temperature):
        calls["n"] += 1
        return '{"value": 42}', 10, 5

    monkeypatch.setattr(base, "_raw_generate", fake_raw_generate)

    kwargs = dict(
        record_id="r1", agent="a1", prompt_version="v1", prompt="p",
        response_schema=_Resp, model="m", temperature=0.7, run_index=0,
    )
    resp1, meta1 = base.call_structured(**kwargs)
    resp2, meta2 = base.call_structured(**kwargs)

    assert calls["n"] == 1  # second call served from cache
    assert resp1.value == resp2.value == 42
    assert meta1.cached is False
    assert meta2.cached is True


def test_call_structured_retries_on_invalid_json_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def fake_raw_generate(client, model, prompt, response_schema, temperature):
        attempts["n"] += 1
        if attempts["n"] < 2:
            return "not valid json", 1, 1
        return '{"value": 7}', 1, 1

    monkeypatch.setattr(base, "_raw_generate", fake_raw_generate)

    resp, meta = base.call_structured(
        record_id="r1", agent="a1", prompt_version="v1", prompt="p",
        response_schema=_Resp, model="m", temperature=0.7, run_index=0,
    )
    assert resp.value == 7
    assert meta.schema_retries == 1
    assert attempts["n"] == 2


def test_call_structured_raises_after_max_schema_retries(monkeypatch):
    def fake_raw_generate(client, model, prompt, response_schema, temperature):
        return "still not json", 1, 1

    monkeypatch.setattr(base, "_raw_generate", fake_raw_generate)

    with pytest.raises(base.SchemaValidationFailed):
        base.call_structured(
            record_id="r1", agent="a1", prompt_version="v1", prompt="p",
            response_schema=_Resp, model="m", temperature=0.7, run_index=0,
        )


def test_call_structured_extra_validate_triggers_retry(monkeypatch):
    calls = {"n": 0}

    def fake_raw_generate(client, model, prompt, response_schema, temperature):
        calls["n"] += 1
        return '{"value": 42}', 1, 1

    monkeypatch.setattr(base, "_raw_generate", fake_raw_generate)

    def reject_always(resp):
        raise ValueError("nope")

    with pytest.raises(base.SchemaValidationFailed):
        base.call_structured(
            record_id="r1", agent="a1", prompt_version="v1", prompt="p",
            response_schema=_Resp, model="m", temperature=0.7, run_index=0,
            extra_validate=reject_always,
        )
    assert calls["n"] == base.MAX_SCHEMA_RETRIES


def test_daily_quota_warns_and_refuses(monkeypatch, tmp_path, capsys):
    # per-model cap, matching the real observed shape (quotaId is
    # per-model, not project-wide) -- see DAILY_QUOTA_BY_MODEL's docstring.
    monkeypatch.setitem(base.DAILY_QUOTA_BY_MODEL, "test-model", 5)
    monkeypatch.setattr(base, "DAILY_WARN_HEADROOM", 3)
    monkeypatch.setattr(base, "DAILY_REFUSE_HEADROOM", 1)

    for _ in range(2):
        base._bump_daily_count("test-model")
    base._check_daily_quota("test-model")  # count=2, warn at cap-3=2
    out = capsys.readouterr().out
    assert "warn" in out.lower()

    for _ in range(2):
        base._bump_daily_count("test-model")  # count=4, refuse at cap-1=4
    with pytest.raises(base.DailyQuotaExceeded):
        base._check_daily_quota("test-model")


def test_daily_count_resets_on_new_day(monkeypatch):
    monkeypatch.setattr(base, "_today", lambda: "2020-01-01")
    base._bump_daily_count("m1")
    base._bump_daily_count("m1")
    assert base.daily_request_count("m1") == 2

    monkeypatch.setattr(base, "_today", lambda: "2020-01-02")
    assert base.daily_request_count("m1") == 0


def test_daily_count_is_tracked_per_model_not_pooled():
    # the real quota is per-model (quotaDimensions.model in the 429 body)
    # -- calls against one model must not consume another's budget.
    base._bump_daily_count("model-a")
    base._bump_daily_count("model-a")
    base._bump_daily_count("model-b")
    assert base.daily_request_count("model-a") == 2
    assert base.daily_request_count("model-b") == 1
    assert base.daily_request_count() == 3  # unfiltered = total across models


def test_raw_generate_fails_fast_on_daily_quota_error_without_retrying(monkeypatch):
    # a per-day 429's ~60s retryDelay can't help -- the quota resets at
    # the next UTC day, not 60 seconds later. Retrying it would just
    # waste the whole retry budget for a guaranteed second failure.
    monkeypatch.setattr(
        base.time, "sleep",
        lambda s: (_ for _ in ()).throw(AssertionError("must not sleep/retry on a daily-quota error")),
    )
    calls = {"n": 0}

    def daily_exhausted(model, contents, config):
        calls["n"] += 1
        raise RuntimeError(
            "429 RESOURCE_EXHAUSTED ... quotaId: "
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier ... Please retry in 58s."
        )

    fake_client = type(
        "C", (), {"models": type("M", (), {"generate_content": staticmethod(daily_exhausted)})()}
    )()

    with pytest.raises(base.DailyQuotaExceeded):
        base._raw_generate(fake_client, "m", "p", _Resp, 0.7)
    assert calls["n"] == 1  # no retry attempted


def test_rate_limit_error_triggers_backoff_then_succeeds(monkeypatch):
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    monkeypatch.setattr(base.random, "uniform", lambda a, b: 0)

    attempts = {"n": 0}

    def flaky(model, contents, config):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        class R:
            text = '{"value": 1}'
            usage_metadata = None

        return R()

    fake_client = type("C", (), {"models": type("M", (), {"generate_content": staticmethod(flaky)})()})()

    text, in_tok, out_tok = base._raw_generate(fake_client, "m", "p", _Resp, 0.7)
    assert text == '{"value": 1}'
    assert attempts["n"] == 2


def test_network_error_triggers_backoff_then_succeeds(monkeypatch):
    # regression: a live stop-point-2 rerun died with httpx.ConnectError
    # ("[Errno 11001] getaddrinfo failed" -- a transient DNS blip) because
    # only API-level error text was treated as retryable; a connection
    # failure crashed the whole batch instead of retrying like a 503 would.
    import httpx

    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    monkeypatch.setattr(base.random, "uniform", lambda a, b: 0)

    attempts = {"n": 0}

    def flaky(model, contents, config):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise httpx.ConnectError("[Errno 11001] getaddrinfo failed")

        class R:
            text = '{"value": 1}'
            usage_metadata = None

        return R()

    fake_client = type("C", (), {"models": type("M", (), {"generate_content": staticmethod(flaky)})()})()

    text, in_tok, out_tok = base._raw_generate(fake_client, "m", "p", _Resp, 0.7)
    assert text == '{"value": 1}'
    assert attempts["n"] == 2


def test_non_rate_limit_error_raises_immediately(monkeypatch):
    def always_fail(model, contents, config):
        raise RuntimeError("400 INVALID_ARGUMENT: malformed request payload")

    fake_client = type(
        "C", (), {"models": type("M", (), {"generate_content": staticmethod(always_fail)})()}
    )()

    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
        base._raw_generate(fake_client, "m", "p", _Resp, 0.7)
