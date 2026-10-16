import inspect
import json

import pytest

from agents import holdout as ho
from agents import selfcheck_core as core
from agents import selfcheck_pipeline as sp
from agents.escalation_grounding import ESCALATION_FEATURES
from agents.prompts import selfcheck_prompts_v3 as v3, selfcheck_prompts_v4 as v4
from agents.schema import (Absence, CitedValue, FieldPrediction, SCBehaviourResponse, SCClaim, SCClaimsResponse,
                           SCHypothesis)

BLIND = [json.loads(l) for l in open("results/eval20_input_blind.jsonl", encoding="utf-8")] \
    if __import__("pathlib").Path("results/eval20_input_blind.jsonl").exists() else []


def claim(cid, statement, *cited):
    return SCClaim(claim_id=cid, statement=statement, cited=[CitedValue(name=n, value=v) for n, v in cited],
                   confidence=0.9)


def hyp(hid, *preds):
    return SCHypothesis(hypothesis_id=hid, description="d", benign=True,
                        predictions=[FieldPrediction(field=f, op=o, value=v) for f, o, v in preds])


# ---------------------------------------------------------------- the held-out set
def test_held_out_is_three_escalation_features_fixed_by_the_seed():
    a = ho.held_out("abc")
    assert len(a) == 3 and set(a) <= set(ESCALATION_FEATURES)
    assert ho.held_out("abc") == a and ho.HOLDOUT_SEED == 20261016
    assert len({ho.held_out(f"r{i}") for i in range(50)}) > 10   # varies by record


def test_exact_twins_are_withheld_with_their_field():
    rid = next(f"r{i}" for i in range(1000) if "Total Backward Packets" in ho.held_out(f"r{i}"))
    assert "Subflow Bwd Packets" in ho.hidden_fields(rid)


@pytest.mark.skipif(not BLIND, reason="needs results/eval20_input_blind.jsonl")
def test_twins_really_are_identical_on_every_eval20_record():
    for f, twins in ho.FIELD_TWINS.items():
        for t in twins:
            assert all(float(b["features"][f]) == float(b["features"][t]) for b in BLIND)


# ---------------------------------------------------------------- nothing withheld reaches A3
def test_mentions_does_not_confuse_a_longer_visible_field():
    vis = ["Fwd Packet Length Mean", "Bwd Packet Length Mean"]
    assert not ho.mentions("Fwd Packet Length Mean is 3.", "Packet Length Mean", vis)
    assert ho.mentions("Packet Length Mean is 3.", "Packet Length Mean", vis)
    assert ho.mentions("init win bytes backward = 0", "Init_Win_bytes_backward", [])


def test_claims_and_absences_that_touch_a_withheld_field_are_dropped():
    hidden = ("Total Backward Packets", "Subflow Bwd Packets")
    cl = [claim("a1_c1", "Total Backward Packets is 0.", ("Total Backward Packets", 0)),
          claim("a1_c2", "Flow Duration is 5.", ("Flow Duration", 5)),
          claim("a1_c3", "Flow Duration is 5 and subflow bwd packets 0.", ("Flow Duration", 5))]
    assert [c.claim_id for c in ho.redact_claims(cl, hidden, ["Flow Duration"])] == ["a1_c2"]
    ab = [Absence(absence=k, holds="yes", cited=[]) for k in ("no_reply", "no_payload_sent", "no_teardown",
                                                                "no_size_variation")]
    kept = {a.absence.value for a in ho.redact_absences(ab, hidden)}
    assert kept == {"no_payload_sent", "no_teardown"}   # no_reply and no_size_variation read Total Backward Packets


def test_run_a3_holdout_refuses_a_prompt_that_carries_a_withheld_field(monkeypatch):
    monkeypatch.setattr(sp, "P", v4)
    monkeypatch.setattr(sp, "_call", lambda *a, **k: ("called", a[2]))
    held = ("Bwd Packets/s", "Flow IAT Mean", "min_seg_size_forward")
    a1 = SCClaimsResponse(claims=[claim("a1_c1", "Flow Duration is 5.", ("Flow Duration", 5))])
    a2 = SCBehaviourResponse(claims=[], absences=[])
    leaky_flow = v4.flow_block({"Flow Duration": 5.0, "Flow IAT Mean": 2.0}, "TCP", None)
    with pytest.raises(AssertionError, match="would reach A3"):
        sp.run_a3_holdout("r", leaky_flow, a1, a2, held, held, ["Flow Duration"])
    clean = v4.flow_block({"Flow Duration": 5.0}, "TCP", None)
    (_, prompt), _ = sp.run_a3_holdout("r", clean, a1, a2, held, held, ["Flow Duration"])
    body = prompt[prompt.index("The flow:"):]
    assert ho.leaked_fields(body, held, ["Flow Duration"]) == []
    for h in held:
        assert h in prompt[:prompt.index("The flow:")]           # named only in the instruction


# ---------------------------------------------------------------- predictions and status
HELD = ("Flow IAT Mean", "Init_Win_bytes_backward", "min_seg_size_forward")
ROW = {"Flow IAT Mean": 100.0, "Init_Win_bytes_backward": 229.0, "min_seg_size_forward": 20.0, "Flow Duration": 5.0}


def test_every_hypothesis_must_predict_each_withheld_field_and_only_those():
    good = hyp("a3_h1", ("Flow IAT Mean", "lt", 1e3), ("Init_Win_bytes_backward", "gt", 0),
               ("min_seg_size_forward", "eq", 20))
    ho.validate_holdout_predictions([good], HELD)
    bad = hyp("a3_h2", ("Flow IAT Mean", "lt", 1e3), ("Flow Duration", "gt", 1))
    with pytest.raises(ValueError) as e:
        ho.validate_holdout_predictions([bad], HELD)
    assert "missing" in str(e.value) and "visible to you" in str(e.value)


def test_status_rules():
    sup = hyp("a3_h1", ("Flow IAT Mean", "lt", 1e3), ("Init_Win_bytes_backward", "gt", 0),
              ("min_seg_size_forward", "eq", 20))
    assert ho.holdout_status(sup.predictions, ROW, HELD)[0] == core.SUPPORTED
    con = hyp("a3_h2", ("Flow IAT Mean", "gt", 1e3), ("Init_Win_bytes_backward", "gt", 0),
              ("min_seg_size_forward", "eq", 20))
    assert ho.holdout_status(con.predictions, ROW, HELD)[0] == core.CONTRADICTED
    # Init_Win_bytes <= 65535 and min_seg >= 0 cannot fail: those fields earn no support
    vac = hyp("a3_h3", ("Flow IAT Mean", "lt", 1e3), ("Init_Win_bytes_backward", "le", 65535),
              ("min_seg_size_forward", "ge", 0))
    assert ho.holdout_status(vac.predictions, ROW, HELD)[0] == core.UNCHECKABLE


# ---------------------------------------------------------------- everything else unchanged
def test_only_a3_and_a5b_change_version_and_the_rest_is_v3_verbatim():
    changed = {k for k in v3.VERSIONS if v3.VERSIONS[k] != v4.VERSIONS[k]}
    assert changed == {"a3", "a5b"}
    flow = v3.flow_block({"Flow Duration": 1.0}, "TCP", "10:00")
    assert v4.a1(flow) == v3.a1(flow) and v4.a2(flow, "x") == v3.a2(flow, "x") and v4.a4(flow) == v3.a4(flow)
    assert v4.a5a(flow) == v3.a5a(flow) and v4.a5b("c", "j", ["C1"]) == v3.a5b("c", "j", ["C1"])


def test_holdout_choice_reads_no_label():
    src = inspect.getsource(ho)
    for word in ("eval20_records", "true_label", "is_attack", "agent_key"):
        assert word not in src
