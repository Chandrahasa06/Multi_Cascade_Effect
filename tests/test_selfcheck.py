"""Self-check run: hypothesis status by code, trust components, bands and floor, validators,
verdict consistency, checklist-alone rule, A5b never sees the raw row, no label anywhere."""
import inspect

import pytest

from agents import selfcheck_core as core
from agents import selfcheck_pipeline as sp
from agents.prompts import selfcheck_prompts_v1 as P
from agents.schema import (Absence, CitedValue, ConditionCheck, ConditionDisposition, FieldPrediction, SCClaim,
                           SCHypothesis, SCVerdictResponse)
from eval import run_selfcheck as rs

ROW = {"Total Fwd Packets": 3.0, "Total Backward Packets": 0.0, "Total Length of Fwd Packets": 0.0,
       "Flow Duration": 1000.0, "Packet Length Variance": 0.0, "Init_Win_bytes_backward": -1.0}


def _pred(field, op, value):
    return FieldPrediction(field=field, op=op, value=value)


# ---------- hypothesis status ----------

def test_status_supported_contradicted_uncheckable():
    assert core.hypothesis_status([_pred("Total Fwd Packets", "eq", 3), _pred("Total Backward Packets", "eq", 0)], ROW)[0] == core.SUPPORTED
    assert core.hypothesis_status([_pred("Total Fwd Packets", "eq", 3), _pred("Total Backward Packets", "gt", 5)], ROW)[0] == core.CONTRADICTED
    assert core.hypothesis_status([_pred("Total Fwd Packets", "eq", 3), _pred("No Such Field", "gt", 1)], ROW)[0] == core.UNCHECKABLE


def test_vacuous_predictions_buy_no_support():
    assert core.hypothesis_status([_pred("Total Fwd Packets", "ge", 0), _pred("Flow Duration", "gt", -1)], ROW)[0] == core.UNCHECKABLE


def test_a_false_prediction_beats_a_missing_field():
    assert core.hypothesis_status([_pred("Total Fwd Packets", "eq", 9), _pred("No Such Field", "gt", 1)], ROW)[0] == core.CONTRADICTED


# ---------- trust components ----------

def test_field_fidelity_counts_misquotes_and_unknown_fields():
    pairs = [("Total Fwd Packets", 3.0), ("Total Fwd Packets", 4.0), ("Nope", 1.0)]
    assert core.field_fidelity(pairs, ROW) == pytest.approx(1 / 3)
    assert core.field_fidelity([], ROW) is None


def test_blind_corroboration_matches_on_field_names():
    assert core.blind_corroboration([("Total Fwd Packets", 3.0)], [("Total Fwd Packets", 3.0)]) == 1.0
    assert core.blind_corroboration([("Total Fwd Packets", 3.0)], [("Total Fwd Packets", 5.0)]) == 0.0
    assert core.blind_corroboration([("Total Fwd Packets", 3.0)], [("Flow Duration", 1.0)]) is None


def _absence(kind, holds, cited=()):
    return Absence(absence=kind, holds=holds, cited=[CitedValue(name=n, value=v) for n, v in cited])


def test_self_consistency_against_own_cited_values():
    good = _absence("no_reply", "yes", [("Total Backward Packets", 0.0)])
    bad = _absence("no_reply", "no", [("Total Backward Packets", 0.0)])
    assert core.self_consistency([good]) == 1.0
    assert core.self_consistency([bad]) == 0.0
    assert core.self_consistency([_absence("no_reply", "yes")]) is None  # nothing cited: not checkable


def test_abstention_honesty_on_teardown_and_single_packet():
    ok = [_absence("no_teardown", "cannot_tell")]
    asserted = [_absence("no_teardown", "yes")]
    assert core.abstention_honesty(ok, ROW) == 1.0
    assert core.abstention_honesty(asserted, ROW) == 0.0
    one_pkt = {"Total Fwd Packets": 1.0, "Total Backward Packets": 0.0}
    assert core.abstention_honesty([_absence("no_size_variation", "yes", [("Packet Length Variance", 0.0)])], one_pkt) == 0.0


def test_trust_renormalises_over_defined_components_and_bands():
    assert core.trust_T({"F": 1.0, "B": None, "S": 0.5, "H": None}) == pytest.approx(0.75)
    assert core.trust_T({"F": None, "B": None, "S": None, "H": None}) is None
    assert core.band(0.9) == "high" and core.band(0.6) == "medium" and core.band(0.3) == "unreliable"
    assert core.withheld(0.2) and not core.withheld(0.25) and not core.withheld(None)


def test_justifications_sorted_by_trust_with_bands_and_no_numbers():
    trust = {"a1": {"T": 0.6, "band": "medium"}, "a2": {"T": 0.9, "band": "high"}, "a3": {"T": None, "band": "unscored"}}
    order = rs.order_by_trust(["a1", "a2", "a3"], trust)
    assert order == ["a2", "a1", "a3"]
    text = rs.justifications(order, {"a1": "x", "a2": "y", "a3": "z"}, trust)
    assert text.index("A2") < text.index("A1") < text.index("A3")
    assert "0.6" not in text and "0.9" not in text


# ---------- validators ----------

def _claim(cid, statement, cited=(("Flow Duration", 1000.0),)):
    return SCClaim(claim_id=cid, statement=statement, cited=[CitedValue(name=n, value=v) for n, v in cited], confidence=0.5)


def test_a1_bans_comparative_words_and_needs_the_field_in_the_sentence():
    sp.validate_claims([_claim("a1_c1", "Flow Duration is 1000 microseconds.")], "a1_c", ban=True)
    with pytest.raises(ValueError, match=r"need a baseline: \['unusually', 'high'\]"):
        sp.validate_claims([_claim("a1_c1", "Flow Duration is unusually high at 1000.")], "a1_c", ban=True)
    with pytest.raises(ValueError, match="name at least one"):
        sp.validate_claims([_claim("a1_c1", "The connection lasted a second.")], "a1_c", ban=True)
    sp.validate_claims([_claim("a1_c1", "flow duration is 1000.")], "a1_c", ban=True)  # case-insensitive


def test_absences_must_be_exactly_the_four():
    four = [_absence(k, "cannot_tell") for k in sp.ABSENCE_KINDS]
    sp.validate_absences(four)
    with pytest.raises(ValueError):
        sp.validate_absences(four[:3])
    with pytest.raises(ValueError, match="cite"):
        sp.validate_absences(four[:3] + [_absence("no_size_variation", "yes")])


def test_hypotheses_need_two_benign():
    def h(i, b):
        return SCHypothesis(hypothesis_id=f"a3_h{i}", description="d", benign=b,
                            predictions=[_pred("Flow Duration", "eq", 1000), _pred("Total Fwd Packets", "eq", 3)])
    sp.validate_hypotheses([h(1, True), h(2, True), h(3, False)], "a3_h")
    with pytest.raises(ValueError, match="benign"):
        sp.validate_hypotheses([h(1, True), h(2, False), h(3, False)], "a3_h")


def _row(cid, holds="no"):
    return ConditionCheck(condition_id=cid, holds=holds, observed=[{"name": "Flow Duration", "value": 1.0}])


def test_checklist_requires_all_fourteen():
    sp.validate_checklist([_row(c) for c in core.CONDITION_IDS], {"Flow Duration"})
    with pytest.raises(ValueError, match="missing"):
        sp.validate_checklist([_row(c) for c in core.CONDITION_IDS if c != "D4"], {"Flow Duration"})
    assert len(core.CONDITION_IDS) == 14 and "C12" in core.CONDITION_IDS


def _verdict(verdict, disps):
    return SCVerdictResponse(verdict=verdict, rationale="r",
                             dispositions=[ConditionDisposition(condition_id=c, refuted=r, refuted_by=by, reason="x") for c, r, by in disps])


def test_verdict_consistency_rules():
    sp.validate_verdict(_verdict("consistent_with_benign", []), [], set())
    sp.validate_verdict(_verdict("anomalous_and_unexplained", [("C5", False, [])]), ["C5"], set())
    sp.validate_verdict(_verdict("anomalous_but_explicable", [("C5", True, ["a3_h1"])]), ["C5"], {"a3_h1"})
    sp.validate_verdict(_verdict("insufficient_evidence", [("C5", False, [])]), ["C5"], set())
    with pytest.raises(ValueError, match="inconsistent"):
        sp.validate_verdict(_verdict("consistent_with_benign", [("C5", False, [])]), ["C5"], set())
    with pytest.raises(ValueError, match="cannot refute"):
        sp.validate_verdict(_verdict("anomalous_but_explicable", [("C5", True, ["a3_h2"])]), ["C5"], {"a3_h1"})
    with pytest.raises(ValueError, match="exactly one disposition"):
        sp.validate_verdict(_verdict("anomalous_and_unexplained", []), ["C5"], set())


def test_checklist_alone_rule():
    assert core.checklist_alone_verdict([]) == "consistent_with_benign"
    assert core.checklist_alone_verdict(["C1"]) == "anomalous_and_unexplained"


# ---------- A5b never sees the raw row; no population; no label ----------

def test_a5b_prompt_has_no_row_block():
    text = P.a5b("  C1: no (Init_Win_bytes_backward = 235)", "[A1 -- reliability on this flow: high]\n  a1_c1: x", ["C5"])
    assert "Fields of this flow's row" not in text


def test_templates_pass_the_statistic_checks():
    flow = P.flow_block({"Flow Duration": 1000.0, "Destination Port": 80.0}, "TCP", "10:15")
    for t in rs.template_prompts(flow).values():
        rs.check_prompt(t, t, set(), {1000.0, 80.0, 10.0, 15.0})


def test_no_label_or_population_reaches_the_code():
    src = inspect.getsource(rs) + inspect.getsource(sp) + inspect.getsource(core) + inspect.getsource(P)
    assert "agent_key" not in src and "true_label" not in src and "is_attack" not in src
    assert "eval20_records" not in src
    # the word "neighbourhood" appears only in the runner's own leak-guard message; check the population code paths
    for word in ("benign_df", "load_benign", "compute_feature_neighbourhood", "escalation_grounding", "reference_window"):
        assert word not in src, word
