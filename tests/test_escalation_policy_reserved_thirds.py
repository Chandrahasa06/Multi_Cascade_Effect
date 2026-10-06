"""Tests for dataplane.escalation_policy.run_policy_reserved_thirds -- the
additive three-reserved-meter composition built for eval/escalation_eval.py's
decision-tree-rule-based Priority 1 (see that module's docstring). Existing
tests/test_escalation_policy.py covers run_policy()/priority1_escalate()
(the SpliDT-classification-based path) and is left untouched.
"""
import numpy as np
import pandas as pd
import pytest

from dataplane.escalation_policy import (
    SignatureTable,
    build_signature_table,
    run_policy_reserved_thirds,
)

PRIORITY2_FEATURES = ("flows_per_src", "distinct_dst_ports_per_src", "syn_without_synack_count",
                      "bwd_pkt_len_mean", "pkt_len_range")


def _make_df(n, first_ts=None):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "flow_uid": [f"flow-{i}" for i in range(n)],
        "first_ts": first_ts if first_ts is not None else np.arange(n, dtype=float),
        "flows_per_src": rng.integers(1, 50, n).astype(float),
        "distinct_dst_ports_per_src": rng.integers(1, 10, n).astype(float),
        "syn_without_synack_count": np.zeros(n),
        "bwd_pkt_len_mean": rng.random(n) * 100,
        "pkt_len_range": rng.random(n) * 50,
    })


class TestThreeWayMeterEnforcement:
    def test_p1_burst_cannot_starve_p2_or_p3(self):
        """The first half of flows (chronologically earliest) all match
        Priority 1 and outnumber the global budget on their own -- with
        no reservation, a naive single-meter design would let P1 alone
        exhaust the whole global budget before the second half (P2/P3-
        eligible, P1 non-matching) is ever reached. With the reservation,
        P1's own cap is held below global capacity, and P2/P3 both still
        get admitted flows from the second half."""
        n = 2000
        df = _make_df(n)  # first_ts = 0..n-1, so first half is chronologically first
        # Built directly (not via build_signature_table's quantile fit) so
        # "uncommon" is unambiguous -- with quantile bins, coarse top/bottom
        # bins can conflate "far outside the fit range" with "merely past
        # the nearest edge" (see STATUS.md's own PortScan-collapse finding),
        # which made an earlier version of this test flaky. Second half's
        # features are pushed to >=10 so their signature (1,1,1,1,1) is
        # guaranteed absent from common_signatures.
        fit_table = SignatureTable(
            bin_edges={f: np.array([10.0]) for f in PRIORITY2_FEATURES},
            common_signatures=frozenset({(0, 0, 0, 0, 0)}),
            features=PRIORITY2_FEATURES, floor=1, fit_n=1,
        )
        # second quarter: uncommon signature (P2-eligible); last quarter:
        # exactly the common signature (0,0,0,0,0), so P2 does NOT fire
        # and only P3's tau=9999 sampling can catch them.
        df.loc[n // 2: 3 * n // 4 - 1, list(PRIORITY2_FEATURES)] = 20.0
        df.loc[3 * n // 4:, list(PRIORITY2_FEATURES)] = 0.0
        p1_matched = np.zeros(n, dtype=bool)
        p1_matched[: n // 2] = True  # first half matches P1, chronologically first
        result = run_policy_reserved_thirds(
            df, p1_matched, fit_table, meter="per_day", tau=9999,  # tau=9999 -> second half also P3-sampled
            global_budget_fraction=0.10, priority_fractions=(0.10, 0.70, 0.20),
        )
        # P1's cap is what's left after P2+P3's reservations, not the full budget
        assert result.priority_capacity[1] < result.global_capacity
        assert result.priority_used[1] == result.priority_capacity[1]  # P1 demand exceeds its cap
        assert result.priority_used[2] > 0
        assert result.priority_used[3] > 0

    def test_global_meter_is_the_hard_cap(self):
        n = 1000
        df = _make_df(n)
        fit_table = build_signature_table(
            pd.DataFrame({
                "Label": ["Benign"] * 500,
                **{f: np.random.default_rng(2).integers(1, 50, 500).astype(float) for f in PRIORITY2_FEATURES},
            }),
            n_bins=3, floor=1,
        )
        p1_matched = np.ones(n, dtype=bool)
        result = run_policy_reserved_thirds(df, p1_matched, fit_table, meter="per_day", tau=1, global_budget_fraction=0.05)
        assert result.escalated.sum() <= result.global_capacity


class TestReportedBitSuppression:
    def test_each_escalated_flow_has_exactly_one_priority(self):
        n = 500
        df = _make_df(n)
        fit_table = SignatureTable(
            bin_edges={f: np.array([10.0]) for f in PRIORITY2_FEATURES},
            common_signatures=frozenset(),  # nothing is "common" -> everything not caught by P1 is P2-uncommon
            features=PRIORITY2_FEATURES, floor=1, fit_n=1,
        )
        p1_matched = np.zeros(n, dtype=bool)
        p1_matched[:100] = True  # first 100 flows match P1
        result = run_policy_reserved_thirds(df, p1_matched, fit_table, meter="per_day", tau=5000, global_budget_fraction=1.0)
        escalated_priorities = result.priority[result.escalated]
        assert np.all(np.isin(escalated_priorities, [1, 2, 3]))
        # every flow that matched P1 is reported as priority 1 if escalated at all, never 2 or 3
        assert np.all(result.priority[p1_matched & result.escalated] == 1)

    def test_p1_match_refused_by_its_meter_is_not_reported_as_p2_or_p3(self):
        """A flow whose P1 match is refused by P1's own (tiny) sub-meter
        must be forwarded normally, never re-checked against P2/P3 --
        matching the PDF's 'forward normally without another
        notification' semantics."""
        n = 100
        df = _make_df(n)
        fit_table = SignatureTable(
            bin_edges={f: np.array([10.0]) for f in PRIORITY2_FEATURES},
            common_signatures=frozenset(),
            features=PRIORITY2_FEATURES, floor=1, fit_n=1,
        )
        p1_matched = np.ones(n, dtype=bool)  # all match P1 -- P1's tiny cap admits only a few
        result = run_policy_reserved_thirds(
            df, p1_matched, fit_table, meter="per_day", tau=1,
            global_budget_fraction=0.5, priority_fractions=(0.10, 0.80, 0.10),
        )
        n_p1_matched_not_escalated = int((p1_matched & ~result.escalated).sum())
        # those flows must never show up as priority 2 or 3
        assert np.all(result.priority[p1_matched & ~result.escalated] == 0)
        assert n_p1_matched_not_escalated > 0  # sanity: P1's cap really is smaller than its match volume


class TestClassifierAgnosticInput:
    def test_accepts_precomputed_boolean_match_array(self):
        """priority1_matched is a plain boolean array (e.g. from
        dataplane.dt_rules.evaluate_rules_union), not a predicted-class
        array -- no string comparison against any 'Benign'/'BENIGN'
        spelling happens inside this function."""
        n = 50
        df = _make_df(n)
        fit_table = build_signature_table(
            pd.DataFrame({
                "Label": ["Benign"] * 200,
                **{f: np.random.default_rng(3).integers(1, 50, 200).astype(float) for f in PRIORITY2_FEATURES},
            }),
            n_bins=3, floor=1,
        )
        p1_matched = np.array([True, False] * 25)
        result = run_policy_reserved_thirds(df, p1_matched, fit_table, meter="per_day", tau=1, global_budget_fraction=0.2)
        assert list(result.matched_priority1) == list(p1_matched)
