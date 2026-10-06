"""Required deliverable test: the blind agent-input file must contain
none of the five stripped fields (class_predicted, rule_id, reason,
priority, and -- reconfirmed defensively -- true_label/is_attack), and
must satisfy the full verification gate before any API call is made.
"""
import json

import pytest

from eval.build_blind_input import (
    FORBIDDEN_FIELDS,
    REQUIRED_TOP_LEVEL_KEYS,
    UNIFORM_ESCALATION_REASON,
    build_blind_records,
    strip_to_blind,
    verify_blind_file,
)


def _make_p1_record(rid="r1", predicted="PortScan", rule_id=9):
    return {
        "record_id": rid,
        "features": {"Flow Duration": 1.0, "Destination Port": 80.0},
        "escalation": {
            "priority": 1, "reason": "rule_match", "matched_rule_id": rule_id,
            "class_predicted": predicted, "feature_bins": {"flows_per_src": 2, "pkt_len_range": 1},
        },
    }


def _make_p2_record(rid="r2"):
    return {
        "record_id": rid,
        "features": {"Flow Duration": 2.0, "Destination Port": 443.0},
        "escalation": {
            "priority": 2, "reason": "uncommon_signature", "matched_rule_id": None,
            "class_predicted": None, "feature_bins": {"flows_per_src": 3, "pkt_len_range": 0},
        },
    }


class TestStripToBlind:
    def test_removes_class_predicted_and_rule_id(self):
        blind = strip_to_blind(_make_p1_record())
        assert "class_predicted" not in json.dumps(blind)
        assert "rule_id" not in json.dumps(blind)

    def test_replaces_priority_and_reason_with_uniform_string(self):
        p1 = strip_to_blind(_make_p1_record())
        p2 = strip_to_blind(_make_p2_record())
        assert p1["escalation_reason"] == UNIFORM_ESCALATION_REASON
        assert p2["escalation_reason"] == UNIFORM_ESCALATION_REASON
        assert "priority" not in json.dumps(p1) and "priority" not in json.dumps(p2)

    def test_keeps_features_bins_and_record_id(self):
        blind = strip_to_blind(_make_p1_record(rid="xyz"))
        assert blind["record_id"] == "xyz"
        assert blind["features"] == {"Flow Duration": 1.0, "Destination Port": 80.0}
        assert blind["feature_bins"] == {"flows_per_src": 2, "pkt_len_range": 1}

    def test_top_level_keys_are_identical_for_p1_and_p2_p3_records(self):
        """The real structural-leak risk: a P2/P3 record's original
        `escalation` dict has class_predicted=None (present but null) vs
        a P1 record's real string -- if the blind builder just deleted
        None values instead of unconditionally rebuilding the dict, a P1
        record could end up with more top-level keys than a P2 one."""
        p1 = strip_to_blind(_make_p1_record())
        p2 = strip_to_blind(_make_p2_record())
        assert set(p1.keys()) == set(p2.keys()) == REQUIRED_TOP_LEVEL_KEYS


class TestVerificationGate:
    def test_passes_on_a_clean_blind_set(self):
        blind = [strip_to_blind(_make_p1_record("a")), strip_to_blind(_make_p2_record("b"))]
        verify_blind_file(blind)  # must not raise

    def test_fails_on_empty_input(self):
        with pytest.raises(AssertionError):
            verify_blind_file([])

    def test_fails_if_a_forbidden_field_survives(self):
        blind = [strip_to_blind(_make_p1_record("a")), strip_to_blind(_make_p2_record("b"))]
        blind[0]["escalation_reason_debug"] = {"rule_id": 9}  # simulated leak
        with pytest.raises(AssertionError):
            verify_blind_file(blind)

    @pytest.mark.parametrize("field", sorted(FORBIDDEN_FIELDS))
    def test_fails_for_each_individually_forbidden_field(self, field):
        blind = [strip_to_blind(_make_p1_record("a")), strip_to_blind(_make_p2_record("b"))]
        blind[1][field] = "leaked"
        with pytest.raises(AssertionError):
            verify_blind_file(blind)

    def test_fails_if_escalation_reason_not_uniform(self):
        blind = [strip_to_blind(_make_p1_record("a")), strip_to_blind(_make_p2_record("b"))]
        blind[1]["escalation_reason"] = "a different string"
        with pytest.raises(AssertionError):
            verify_blind_file(blind)

    def test_fails_if_feature_key_sets_differ(self):
        blind = [strip_to_blind(_make_p1_record("a")), strip_to_blind(_make_p2_record("b"))]
        blind[1]["features"]["Extra Feature Only On Benign"] = 1.0
        with pytest.raises(AssertionError):
            verify_blind_file(blind)

    def test_fails_if_top_level_key_sets_differ(self):
        blind = [strip_to_blind(_make_p1_record("a")), strip_to_blind(_make_p2_record("b"))]
        blind[1]["extra_top_level_key"] = "leak"
        with pytest.raises(AssertionError):
            verify_blind_file(blind)


class TestRealBlindFile:
    def test_real_generated_file_passes_the_gate(self):
        try:
            records = build_blind_records()
        except FileNotFoundError:
            pytest.skip("results/agent_input_20.jsonl not generated yet")
        verify_blind_file(records)
        raw_text = "\n".join(json.dumps(r) for r in records)
        for field in ("class_predicted", "rule_id"):
            assert field not in raw_text
        # bare "priority"/"reason"/"true_label"/"is_attack" keys, not the
        # substring (which legitimately appears inside "escalation_reason")
        for r in records:
            assert "priority" not in r and "reason" not in r
            assert "true_label" not in r and "is_attack" not in r
