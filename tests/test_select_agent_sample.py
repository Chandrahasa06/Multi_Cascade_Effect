"""Tests for eval/select_agent_sample.py -- the ground-truth-isolation
guard (the required deliverable test: the agent-input file must never
carry a ground-truth field) plus the class-quota selection algorithm.
"""
import json

import numpy as np
import pandas as pd
import pytest

from eval.select_agent_sample import (
    _ALLOWED_KEYS_WITH_FORBIDDEN_SUBSTRING,
    assert_no_ground_truth,
    select_attack_flows,
    select_benign_flows,
)


class TestGroundTruthIsolation:
    def test_class_predicted_is_explicitly_allowed(self):
        assert "class_predicted" in _ALLOWED_KEYS_WITH_FORBIDDEN_SUBSTRING

    def test_passes_on_a_clean_record(self):
        rec = {
            "record_id": "abc123",
            "features": {"Flow Duration": 1.0, "Destination Port": 80.0},
            "escalation": {
                "priority": 1, "reason": "rule_match", "matched_rule_id": 9,
                "class_predicted": "PortScan", "feature_bins": {"flows_per_src": 2},
            },
        }
        assert_no_ground_truth(rec)  # must not raise

    @pytest.mark.parametrize("bad_key", ["true_label", "True_Label", "is_attack", "Is_Attack", "ground_truth", "attack_label"])
    def test_rejects_ground_truth_shaped_keys_anywhere_in_the_record(self, bad_key):
        rec = {
            "record_id": "abc123",
            "features": {"Flow Duration": 1.0},
            "escalation": {"priority": 2, bad_key: "whatever"},
        }
        with pytest.raises(AssertionError):
            assert_no_ground_truth(rec)

    def test_rejects_nested_inside_a_list(self):
        rec = {"record_id": "x", "features": [{"is_attack": True}]}
        with pytest.raises(AssertionError):
            assert_no_ground_truth(rec)

    def test_real_agent_input_file_carries_no_ground_truth(self, tmp_path):
        """The actual deliverable check: load results/agent_input_20.jsonl
        (if it has been generated) and verify every record independently."""
        path = "results/agent_input_20.jsonl"
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            pytest.skip("results/agent_input_20.jsonl not generated yet")
        assert len(lines) > 0
        for line in lines:
            rec = json.loads(line)
            assert_no_ground_truth(rec)
            assert "true_label" not in json.dumps(rec)  # belt-and-braces substring check on the raw JSON text
            assert "is_attack" not in json.dumps(rec)


class TestAttackClassSelection:
    def _make_attacks(self, counts: dict) -> pd.DataFrame:
        rows = []
        i = 0
        for label, n in counts.items():
            for _ in range(n):
                rows.append({"flow_id": f"flow-{i}", "true_label": label, "is_attack": True,
                             "admitted_by": "P1", "rule_id": 0.0, "class_predicted": label})
                i += 1
        return pd.DataFrame(rows)

    def test_spreads_across_many_small_classes_before_filling_from_one_big_one(self):
        counts = {"A": 50, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1, "G": 1}
        attacks = self._make_attacks(counts)
        rng = np.random.default_rng(0)
        sel = select_attack_flows(attacks, rng, n=12)
        assert len(sel) == 12
        class_counts = sel["true_label"].value_counts()
        # every small (n=1) class should appear -- "at most 2, alphabetical
        # first pass" gives each of them their one shot before A can crowd
        # them out entirely (A is capped at 2 in that first pass, same as
        # everyone else; only the TOP-UP phase afterward draws more from A,
        # since it's the only class with anything left over)
        for label in ("B", "C", "D", "E", "F", "G"):
            assert class_counts.get(label, 0) == 1
        assert class_counts["A"] == 12 - 6  # 6 slots went to the small classes, A tops up the rest

    def test_never_exceeds_two_per_class_unless_topping_up(self):
        counts = {"A": 20, "B": 20, "C": 20}
        attacks = self._make_attacks(counts)
        rng = np.random.default_rng(1)
        sel = select_attack_flows(attacks, rng, n=6)  # exactly 2*3 -- no top-up needed
        class_counts = sel["true_label"].value_counts()
        assert (class_counts <= 2).all()
        assert class_counts.sum() == 6

    def test_tops_up_from_largest_when_classes_run_out(self):
        counts = {"A": 20, "B": 1}  # only 2 classes, at most 2 each = 3 < 12 needed
        attacks = self._make_attacks(counts)
        rng = np.random.default_rng(2)
        sel = select_attack_flows(attacks, rng, n=12)
        assert len(sel) == 12
        class_counts = sel["true_label"].value_counts()
        assert class_counts["B"] == 1  # B only ever had 1 to give
        assert class_counts["A"] == 11  # everything else tops up from the only other class

    def test_sample_without_replacement_no_duplicate_flow_ids(self):
        counts = {"A": 5, "B": 5, "C": 5}
        attacks = self._make_attacks(counts)
        rng = np.random.default_rng(3)
        sel = select_attack_flows(attacks, rng, n=12)
        assert sel["flow_id"].is_unique

    def test_deterministic_given_same_random_state(self):
        counts = {"A": 10, "B": 10, "C": 10}
        attacks = self._make_attacks(counts)
        sel1 = select_attack_flows(attacks, np.random.default_rng(42), n=12)
        sel2 = select_attack_flows(attacks, np.random.default_rng(42), n=12)
        assert list(sel1["flow_id"]) == list(sel2["flow_id"])

    def test_raises_if_not_enough_attacks_available(self):
        attacks = self._make_attacks({"A": 3})
        rng = np.random.default_rng(0)
        with pytest.raises(AssertionError):
            select_attack_flows(attacks, rng, n=12)


class TestBenignSelection:
    def test_uniform_sample_without_replacement(self):
        benign = pd.DataFrame({"flow_id": [f"b-{i}" for i in range(2718)]})
        rng = np.random.default_rng(5)
        sel = select_benign_flows(benign, rng, n=8)
        assert len(sel) == 8
        assert sel["flow_id"].is_unique

    def test_deterministic_given_same_random_state(self):
        benign = pd.DataFrame({"flow_id": [f"b-{i}" for i in range(100)]})
        sel1 = select_benign_flows(benign, np.random.default_rng(7), n=8)
        sel2 = select_benign_flows(benign, np.random.default_rng(7), n=8)
        assert list(sel1["flow_id"]) == list(sel2["flow_id"])
