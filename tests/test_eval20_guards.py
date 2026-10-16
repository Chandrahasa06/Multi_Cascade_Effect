import inspect
import json

import pytest

from agents import openrouter
from agents.schema import SCClaimsResponse
from eval import run_selfcheck as rs
from eval import score_eval20 as se

CFG = json.loads(openrouter.CONFIG_PATH.read_text(encoding="utf-8"))
SLUG = CFG["shakedown_model"]


@pytest.fixture(autouse=True)
def _clear_guard():
    yield
    openrouter._guard.clear()


def _body(cost):
    return {"id": "g", "provider": "Anthropic", "model": SLUG,
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "cost": cost}}


def _pin(monkeypatch, cost):
    monkeypatch.setattr(openrouter, "verify", lambda m: {"provider_name": "Anthropic", "quantization": None,
                                                          "pricing": {"prompt": "0.000001", "completion": "0.000002"}})
    sent = []
    monkeypatch.setattr(openrouter, "_post", lambda payload: sent.append(1) or _body(cost))
    return sent


def test_every_model_in_the_sequence_has_a_ceiling_and_the_values_are_as_set():
    want = {"anthropic/claude-haiku-5.5": 0.50, "deepseek/deepseek-v4.1-flash": 0.75, "z-ai/glm-5.2": 2.50,
            "google/gemini-3.5-flash-lite": 2.50, "moonshotai/kimi-k2.6": 4.50}
    assert {s: CFG["models"][s]["cost_ceiling_usd"] for s in CFG["sequence"]} == want
    assert CFG["global_cost_ceiling_usd"] == 10.0 and CFG["max_concurrent"] == 8
    assert all(CFG["models"][s]["temperature"] is None for s in CFG["sequence"])


def test_run_ceiling_stops_before_the_next_call_and_ledger_records_spend(monkeypatch, tmp_path):
    sent = _pin(monkeypatch, 0.30)
    ledger = tmp_path / "l.jsonl"
    openrouter.set_cost_guard(SLUG, 0.50, 10.0, ledger, 8)
    t = openrouter.temperature_for(SLUG)
    openrouter.generate(SLUG, "p", SCClaimsResponse, t)
    openrouter.generate(SLUG, "p", SCClaimsResponse, t)          # 0.30 < 0.50 at check time
    with pytest.raises(openrouter.CostCeilingExceeded, match="per-run ceiling"):
        openrouter.generate(SLUG, "p", SCClaimsResponse, t)      # 0.60 >= 0.50: refused before sending
    assert len(sent) == 2
    total, by_model = openrouter.ledger_totals(ledger)
    assert total == pytest.approx(0.60) and by_model[SLUG] == pytest.approx(0.60)


def test_global_ceiling_counts_earlier_runs_from_the_ledger(monkeypatch, tmp_path):
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(json.dumps({"model": "other/model", "cost_usd": 9.99, "generation_id": "x"}) + "\n")
    sent = _pin(monkeypatch, 0.02)
    openrouter.set_cost_guard(SLUG, 0.50, 10.0, ledger, 8)
    t = openrouter.temperature_for(SLUG)
    openrouter.generate(SLUG, "p", SCClaimsResponse, t)
    with pytest.raises(openrouter.CostCeilingExceeded, match="global ceiling"):
        openrouter.generate(SLUG, "p", SCClaimsResponse, t)
    assert len(sent) == 1


def test_unpriced_call_is_charged_from_tokens_not_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(openrouter, "_cost_from_generation", lambda g: None)
    _pin(monkeypatch, None)
    ledger = tmp_path / "l.jsonl"
    openrouter.set_cost_guard(SLUG, 0.50, 10.0, ledger, 8)
    openrouter.generate(SLUG, "p", SCClaimsResponse, openrouter.temperature_for(SLUG))
    e = json.loads(ledger.read_text().splitlines()[0])
    assert e["estimated"] is True and e["cost_usd"] == pytest.approx(1000 * 1e-6 + 1000 * 2e-6)


def test_concurrent_results_cover_every_record_once():
    out = list(rs._results(lambda b: ("ok", b), list(range(20)), workers=8))
    assert sorted(x[1] for x in out) == list(range(20))
    assert [x[1] for x in rs._results(lambda b: ("ok", b), [3, 1, 2], workers=1)] == [3, 1, 2]


def test_winner_rule_is_J_then_fewer_lost_then_lower_cost():
    rows = [{"model": "a", "J": 0.5, "lost_total": 2, "cost_usd": 0.1},
            {"model": "b", "J": 0.5, "lost_total": 1, "cost_usd": 0.9},
            {"model": "c", "J": 0.4, "lost_total": 0, "cost_usd": 0.0},
            {"model": "d", "J": 0.5, "lost_total": 1, "cost_usd": 0.5}]
    assert se.winner(rows)["model"] == "d"


def test_verdict_mapping_is_the_preregistered_one():
    assert se.ATTACK_CALL == {"anomalous_and_unexplained"}
    assert se.BENIGN_CALL == {"consistent_with_benign", "anomalous_but_explicable"}
    assert "insufficient_evidence" not in se.ATTACK_CALL | se.BENIGN_CALL


def test_scorer_is_not_reachable_from_the_decision_path():
    from agents import selfcheck_pipeline as sp
    from eval import run_eval20
    src = "".join(inspect.getsource(m) for m in (run_eval20, rs, sp, openrouter))
    assert "score_eval20" not in src and "eval20_records" not in src
