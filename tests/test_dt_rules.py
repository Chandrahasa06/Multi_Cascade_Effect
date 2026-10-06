import numpy as np
import pandas as pd
import pytest

from dataplane.dt_rules import (
    DERIVED_FEATURES,
    INTEGER_FEATURES,
    Interval,
    Rule,
    _merge_conditions,
    compile_attack_rules,
    evaluate_rules,
    evaluate_rules_union,
    first_matching_rule,
    rule_predicted_class,
)
from eval.parse_tree import parse_tree_text

TINY_TREE = """\
|--- feat_a <= 5.00
|   |--- feat_b <= 1.50
|   |   |--- class: BENIGN
|   |--- feat_b >  1.50
|   |   |--- class: DoS Hulk
|--- feat_a >  5.00
|   |--- Fwd Packet Length Max <= 100.00
|   |   |--- class: PortScan
|   |--- Fwd Packet Length Max >  100.00
|   |   |--- class: PortScan
"""


def test_integer_bound_conversion_matches_pdf_example():
    """PSH > 4.5 and PSH <= 11.5 becomes 5 <= PSH <= 11 -- the exact
    worked example from controller_rule_selection.pdf, using one of our
    own integer features in place of PSH."""
    conditions = [("act_data_pkt_fwd", ">", 4.5), ("act_data_pkt_fwd", "<=", 11.5)]
    intervals = _merge_conditions(conditions, integer_features=frozenset({"act_data_pkt_fwd"}))
    iv = intervals["act_data_pkt_fwd"]
    assert iv.lo == 5
    assert iv.hi == 11
    assert iv.is_integer is True


def test_integer_bound_conversion_on_exact_integer_threshold():
    """x <= 10500.00 (already integer) and x > 10500.00 elsewhere convert
    to <=10500 and >=10501 respectively -- not off-by-one."""
    le = _merge_conditions([("Fwd Packet Length Max", "<=", 10500.0)])
    assert le["Fwd Packet Length Max"].hi == 10500
    gt = _merge_conditions([("Fwd Packet Length Max", ">", 10500.0)])
    assert gt["Fwd Packet Length Max"].lo == 10501


def test_continuous_feature_keeps_fractional_threshold():
    intervals = _merge_conditions([("Flow IAT Mean", "<=", 5088373.5)])
    assert intervals["Flow IAT Mean"].hi == 5088373.5
    assert intervals["Flow IAT Mean"].is_integer is False


def test_interval_merging_repeated_feature_on_one_path():
    """Two '>' conditions on the same feature along one path merge into
    the tightest (max) lower bound; two '<=' merge into the tightest
    (min) upper bound."""
    conditions = [
        ("Init_Win_bytes_backward", ">", 226.5),
        ("Flow IAT Mean", "<=", 100.0),
        ("Init_Win_bytes_backward", ">", 238.5),  # tighter than the first
        ("Flow IAT Mean", "<=", 50.0),  # tighter than the first
    ]
    intervals = _merge_conditions(conditions)
    # Init_Win_bytes_backward is an integer feature: > 238.5 -> >= 239
    assert intervals["Init_Win_bytes_backward"].lo == 239
    assert intervals["Flow IAT Mean"].hi == 50.0


def test_unconstrained_feature_describes_as_any():
    iv = Interval()
    assert iv.describe() == "Any"


def test_compile_attack_rules_excludes_benign_and_covers_all_attack_leaves():
    root = parse_tree_text(TINY_TREE)
    rules = compile_attack_rules(root)
    assert len(rules) == 3  # DoS Hulk + 2 PortScan leaves
    assert {r.predicted_class for r in rules} == {"DoS Hulk", "PortScan"}


def test_rule_matches_vectorized_against_dataframe():
    root = parse_tree_text(TINY_TREE)
    rules = compile_attack_rules(root)
    hulk_rule = next(r for r in rules if r.predicted_class == "DoS Hulk")
    df = pd.DataFrame({
        "feat_a": [1.0, 1.0, 10.0],
        "feat_b": [0.5, 2.0, 2.0],
        "Fwd Packet Length Max": [0.0, 0.0, 0.0],
    })
    matched = hulk_rule.matches(df)
    assert list(matched) == [False, True, False]


def test_union_deduplication_counts_each_flow_once():
    """A flow matching multiple rules is counted once in the union, with
    n_matched reporting the real multiplicity for honesty. Sibling leaves
    of ONE tree can never overlap (decision-tree leaves partition the
    space by construction), so genuine multi-rule overlap is built here
    directly from two synthetic rules constraining independent features
    -- exercising the dedup logic itself, not tree-leaf geometry."""
    rule_a = Rule(id=0, leaf_id=0, predicted_class="DoS Hulk",
                  intervals={"feat_a": Interval(lo=5.0, lo_inclusive=False)}, raw_conditions=())
    rule_b = Rule(id=1, leaf_id=1, predicted_class="PortScan",
                  intervals={"feat_b": Interval(lo=1.0, lo_inclusive=False)}, raw_conditions=())
    df = pd.DataFrame({
        "feat_a": [10.0, 10.0, 1.0],
        "feat_b": [10.0, 0.0, 0.0],
    })
    matched_any, n_matched = evaluate_rules_union([rule_a, rule_b], df)
    assert list(matched_any) == [True, True, False]
    assert list(n_matched) == [2, 1, 0]  # row 0 matches both rules but is counted once in matched_any


def test_evaluate_rules_union_empty_rules():
    df = pd.DataFrame({"x": [1, 2, 3]})
    matched_any, n_matched = evaluate_rules_union([], df)
    assert list(matched_any) == [False, False, False]
    assert list(n_matched) == [0, 0, 0]


def test_rule_predicted_class_defaults_to_benign():
    root = parse_tree_text(TINY_TREE)
    rules = compile_attack_rules(root)
    df = pd.DataFrame({
        "feat_a": [1.0, 10.0],
        "feat_b": [0.5, 0.0],
        "Fwd Packet Length Max": [0.0, 0.0],
    })
    pred = rule_predicted_class(rules, df, benign_class="Benign")
    assert list(pred) == ["Benign", "PortScan"]


def test_first_matching_rule_reports_id_and_class():
    root = parse_tree_text(TINY_TREE)
    rules = compile_attack_rules(root)
    df = pd.DataFrame({
        "feat_a": [1.0, 10.0, 1.0],
        "feat_b": [0.5, 0.0, 2.0],
        "Fwd Packet Length Max": [0.0, 0.0, 0.0],
    })
    rule_id, predicted_class = first_matching_rule(rules, df)
    hulk_rule = next(r for r in rules if r.predicted_class == "DoS Hulk")
    assert rule_id[0] == -1 and predicted_class[0] == ""  # BENIGN row, no rule fires
    assert rule_id[1] != -1 and predicted_class[1] == "PortScan"
    assert rule_id[2] == hulk_rule.id and predicted_class[2] == "DoS Hulk"


def test_depends_on_derived_flags_rate_and_mean_features():
    root = parse_tree_text(TINY_TREE)
    rules = compile_attack_rules(root)
    hulk_rule = next(r for r in rules if r.predicted_class == "DoS Hulk")
    assert hulk_rule.depends_on_derived == []  # feat_b isn't a real derived feature name

    # Directly check the real derived-feature set is exactly as documented
    assert DERIVED_FEATURES == frozenset({"Bwd Packets/s", "Flow IAT Mean", "Packet Length Mean"})
    assert "Bwd Packets/s" in DERIVED_FEATURES and "Bwd Packets/s" not in INTEGER_FEATURES
