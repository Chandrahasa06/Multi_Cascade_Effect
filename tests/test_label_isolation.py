"""Verifies the CSV `Label` (ground truth) column can never reach any
escalation decision — it is used only for SCORING (eval/escalation_eval.py's
sampled-run report), never as an input to Priority 1/2/3 or the meter
composition. Two independent checks: static (no compiled rule ever
constrains on "Label") and behavioural (scrambling Label leaves every
decision-relevant output bit-for-bit identical).
"""
import numpy as np
import pandas as pd

from dataplane.dt_rules import compile_attack_rules, evaluate_rules_union, first_matching_rule
from dataplane.escalation_policy import SignatureTable, priority2_escalate, priority3_sample, run_policy_reserved_thirds
from eval.parse_tree import parse_tree_text

TINY_TREE = """\
|--- feat_a <= 5.00
|   |--- feat_b <= 1.50
|   |   |--- class: BENIGN
|   |--- feat_b >  1.50
|   |   |--- class: DoS Hulk
|--- feat_a >  5.00
|   |--- class: PortScan
"""

PRIORITY2_FEATURES = ("flows_per_src", "distinct_dst_ports_per_src", "syn_without_synack_count",
                      "bwd_pkt_len_mean", "pkt_len_range")


def _make_df(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "flow_uid": [f"flow-{i}" for i in range(n)],
        "first_ts": np.arange(n, dtype=float),
        "feat_a": rng.random(n) * 10,
        "feat_b": rng.random(n) * 3,
        "Fwd Packet Length Max": rng.random(n) * 100,
        "flows_per_src": rng.integers(1, 50, n).astype(float),
        "distinct_dst_ports_per_src": rng.integers(1, 10, n).astype(float),
        "syn_without_synack_count": np.zeros(n),
        "bwd_pkt_len_mean": rng.random(n) * 100,
        "pkt_len_range": rng.random(n) * 50,
        "Label": rng.choice(["BENIGN", "DoS Hulk", "PortScan"], n),
    })


class TestStaticLabelIsolation:
    def test_no_compiled_rule_constrains_on_label(self):
        root = parse_tree_text(TINY_TREE)
        rules = compile_attack_rules(root)
        assert rules, "sanity: rules must exist for this check to mean anything"
        for r in rules:
            assert "Label" not in r.intervals

    def test_rule_matching_never_reads_label_column(self):
        """Dropping Label entirely must not break rule evaluation --
        proof it's never accessed."""
        root = parse_tree_text(TINY_TREE)
        rules = compile_attack_rules(root)
        df = _make_df(50)
        df_no_label = df.drop(columns=["Label"])
        matched_with, _ = evaluate_rules_union(rules, df)
        matched_without, _ = evaluate_rules_union(rules, df_no_label)
        assert list(matched_with) == list(matched_without)


class TestBehaviouralLabelIsolation:
    def test_scrambling_label_does_not_change_any_decision(self):
        n = 2000
        df = _make_df(n, seed=1)
        root = parse_tree_text(TINY_TREE)
        rules = compile_attack_rules(root)
        table = SignatureTable(
            bin_edges={f: np.array([10.0]) for f in PRIORITY2_FEATURES},
            common_signatures=frozenset({(0, 0, 0, 0, 0)}),
            features=PRIORITY2_FEATURES, floor=1, fit_n=1,
        )

        def run(frame):
            rule_id, predicted_class = first_matching_rule(rules, frame)
            p1_matched = rule_id != -1
            m2 = priority2_escalate(frame, table)
            m3 = priority3_sample(frame["flow_uid"].tolist(), tau=2000)
            result = run_policy_reserved_thirds(
                frame, p1_matched, table, meter="per_day", tau=2000, global_budget_fraction=0.05,
            )
            return rule_id, predicted_class, m2, m3, result

        rule_id_a, class_a, m2_a, m3_a, result_a = run(df)

        scrambled = df.copy()
        rng = np.random.default_rng(99)
        scrambled["Label"] = rng.permutation(scrambled["Label"].to_numpy())
        # also try nonsense labels entirely, not just a permutation of the same values
        garbage = df.copy()
        garbage["Label"] = [f"not-a-real-label-{i}" for i in range(n)]

        for variant, name in ((scrambled, "permuted"), (garbage, "garbage")):
            rule_id_b, class_b, m2_b, m3_b, result_b = run(variant)
            assert list(rule_id_a) == list(rule_id_b), f"Priority 1 rule_id changed under {name} Label"
            assert list(class_a) == list(class_b), f"Priority 1 predicted_class changed under {name} Label"
            assert list(m2_a) == list(m2_b), f"Priority 2 match changed under {name} Label"
            assert list(m3_a) == list(m3_b), f"Priority 3 sampling changed under {name} Label"
            assert list(result_a.escalated) == list(result_b.escalated), f"escalation decision changed under {name} Label"
            assert list(result_a.priority) == list(result_b.priority), f"admitting priority changed under {name} Label"

    def test_dropping_label_entirely_does_not_break_the_policy(self):
        """The full policy composition must not even require a Label
        column to be present -- the strongest form of the isolation
        guarantee: it's not just unused, it's not needed."""
        n = 500
        df = _make_df(n, seed=2).drop(columns=["Label"])
        root = parse_tree_text(TINY_TREE)
        rules = compile_attack_rules(root)
        table = SignatureTable(
            bin_edges={f: np.array([10.0]) for f in PRIORITY2_FEATURES},
            common_signatures=frozenset(),
            features=PRIORITY2_FEATURES, floor=1, fit_n=1,
        )
        rule_id, _ = first_matching_rule(rules, df)
        p1_matched = rule_id != -1
        result = run_policy_reserved_thirds(df, p1_matched, table, meter="per_day", tau=100, global_budget_fraction=0.1)
        assert result.escalated.dtype == bool
