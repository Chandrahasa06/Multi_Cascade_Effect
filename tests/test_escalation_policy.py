import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from dataplane.escalation_policy import (
    Meter,
    build_signature_table,
    priority1_escalate,
    priority2_escalate,
    priority3_sample,
    run_policy,
    stable_hash,
)


class TestPriority1:
    def test_escalates_only_non_benign(self):
        pred = np.array(["Benign", "PortScan", "Benign", "SSH-Patator"])
        assert list(priority1_escalate(pred)) == [False, True, False, True]


class TestPriority2BenignOnlyAssertion:
    def make_df(self, labels, n_bins=3):
        n = len(labels)
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "Label": labels,
            "flows_per_src": rng.integers(1, 100, n).astype(float),
            "distinct_dst_ports_per_src": rng.integers(1, 20, n).astype(float),
            "syn_without_synack_count": rng.integers(0, 5, n).astype(float),
            "bwd_pkt_len_mean": rng.random(n) * 1000,
            "pkt_len_range": rng.random(n) * 500,
        })

    def test_rejects_non_benign_fit_data(self):
        df = self.make_df(["Benign"] * 5 + ["DoS Hulk"])
        with pytest.raises(AssertionError):
            build_signature_table(df, n_bins=3, floor=1)

    def test_accepts_all_benign_fit_data(self):
        df = self.make_df(["Benign"] * 20)
        table = build_signature_table(df, n_bins=3, floor=1)
        assert table.fit_n == 20

    def test_escalates_unseen_signature(self):
        fit_df = self.make_df(["Benign"] * 50)
        table = build_signature_table(fit_df, n_bins=4, floor=2)
        # a wildly out-of-range flow should not match any common signature
        outlier = pd.DataFrame({
            "Label": ["DoS Hulk"],
            "flows_per_src": [1e9],
            "distinct_dst_ports_per_src": [1e9],
            "syn_without_synack_count": [1e9],
            "bwd_pkt_len_mean": [1e9],
            "pkt_len_range": [1e9],
        })
        esc = priority2_escalate(outlier, table)
        assert esc[0] == True  # noqa: E712 (numpy bool)


class TestPriority3SamplingDeterminism:
    def test_stable_hash_is_deterministic_within_process(self):
        assert stable_hash("192.168.1.1-10.0.0.1-1234-80-6") == stable_hash("192.168.1.1-10.0.0.1-1234-80-6")

    def test_sample_decision_matches_expected_rate_roughly(self):
        flow_ids = [f"flow-{i}" for i in range(200_000)]
        sampled = priority3_sample(flow_ids, tau=20, mod=10_000)  # ~0.2%
        rate = sampled.mean()
        assert 0.001 < rate < 0.005

    def test_deterministic_across_separate_processes(self):
        """Priority 3 must use a STABLE hash, not Python's salted built-in
        hash() (randomized per-process via PYTHONHASHSEED by default) --
        verified by running the same computation in two subprocesses with
        different, explicit PYTHONHASHSEED values and checking they agree."""
        script = (
            "import sys; sys.path.insert(0, '.'); "
            "from dataplane.escalation_policy import stable_hash; "
            "print(stable_hash('flow-42-fixed') % 10000)"
        )
        import os

        env0 = dict(os.environ, PYTHONHASHSEED="0")
        env1 = dict(os.environ, PYTHONHASHSEED="12345")
        out0 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env0, cwd=".")
        out1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env1, cwd=".")
        assert out0.returncode == 0 and out1.returncode == 0, (out0.stderr, out1.stderr)
        assert out0.stdout.strip() == out1.stdout.strip()

    def test_decided_once_per_flow_not_per_packet(self):
        """Calling priority3_sample twice on the same flow id must give
        the same decision -- there is no per-call/per-packet randomness."""
        fid = ["stable-flow-id-xyz"]
        first = priority3_sample(fid, tau=5000)  # 50%, high enough that a
        second = priority3_sample(fid, tau=5000)  # flaky implementation would show it
        assert first[0] == second[0]


class TestMeter:
    def test_admits_up_to_capacity_then_refuses(self):
        meter = Meter(capacity=3)
        results = [meter.try_admit() for _ in range(5)]
        assert results == [True, True, True, False, False]
        assert meter.used == 3
        assert meter.remaining == 0

    def test_zero_capacity_admits_nothing(self):
        meter = Meter(capacity=0)
        assert meter.try_admit() is False


class TestPolicyComposition:
    def _make_df(self, n, labels, first_ts=None):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "Flow ID": [f"flow-{i}" for i in range(n)],
            "Label": labels,
            "first_ts": first_ts if first_ts is not None else np.arange(n, dtype=float),
            "flows_per_src": rng.integers(1, 50, n).astype(float),
            "distinct_dst_ports_per_src": rng.integers(1, 10, n).astype(float),
            "syn_without_synack_count": np.zeros(n),
            "bwd_pkt_len_mean": rng.random(n) * 100,
            "pkt_len_range": rng.random(n) * 50,
        })
        return df

    def test_global_meter_caps_total_escalations(self):
        n = 1000
        df = self._make_df(n, ["Benign"] * n)
        fit_table = build_signature_table(self._make_df(500, ["Benign"] * 500), n_bins=3, floor=1)
        predicted = np.array(["PortScan"] * n)  # everyone matches Priority 1
        result = run_policy(df, predicted, fit_table, tau=1, global_budget_fraction=0.05)
        assert result.escalated.sum() <= result.global_capacity
        assert result.escalated.sum() == result.global_capacity  # demand exceeds budget here

    def test_reported_bit_one_priority_per_flow_never_more(self):
        """Every escalated flow gets exactly one priority; a flow that
        would match multiple checks is still reported only once (the
        PDF's per-flow reported-bit suppression, collapsed here into the
        mutually-exclusive if/elif/else processing order)."""
        n = 300
        df = self._make_df(n, ["PortScan"] * n)  # matches P1 AND (trivially) may hash-sample too
        fit_table = build_signature_table(self._make_df(500, ["Benign"] * 500), n_bins=3, floor=1)
        predicted = np.array(["PortScan"] * n)
        result = run_policy(df, predicted, fit_table, tau=9999, global_budget_fraction=1.0)
        escalated_priorities = result.priority[result.escalated]
        # every escalated row has exactly one nonzero priority value (int, not a set/list)
        assert np.all(np.isin(escalated_priorities, [1, 2, 3]))
        # rows that matched priority 1 are never separately counted for 2 or 3
        p1_matched = result.matched_priority1
        assert np.all(result.priority[p1_matched & result.escalated] == 1)

    def test_priority2_reservation_survives_priority1_saturation(self):
        """Priority 1 consuming its own sub-budget must not be able to
        starve Priority 2's reserved capacity.

        Builds the SignatureTable directly (rather than via
        build_signature_table's quantile fit) for full determinism: with
        quantile bins, "extreme" isn't automatically "uncommon" (a value
        far above everything else still lands in the same top bin as any
        other above-median value — that coarse-top-bin collapse is the
        real, documented PortScan-recovery question this project already
        tracks, exercised separately by the bin-resolution sweep in
        eval/splidt_escalation_eval.py, not by this composition test)."""
        n = 1000
        from dataplane.escalation_policy import PRIORITY2_FEATURES, SignatureTable

        table = SignatureTable(
            bin_edges={f: np.array([10.0]) for f in PRIORITY2_FEATURES},
            common_signatures=frozenset({(0, 0, 0, 0, 0)}),
            features=PRIORITY2_FEATURES, floor=1, fit_n=1,
        )

        # first 500 rows match Priority 1 (predicted PortScan) and stay
        # "common" (all features < 10, signature (0,0,0,0,0)); last 500
        # are Benign-predicted with flows_per_src >= 10 -- a signature
        # never in `common_signatures`, unambiguously uncommon.
        df = pd.DataFrame({
            "Flow ID": [f"flow-{i}" for i in range(n)],
            "Label": ["PortScan"] * 500 + ["Benign"] * 500,
            "first_ts": np.arange(n, dtype=float),
            "flows_per_src": np.full(n, 1.0),
            "distinct_dst_ports_per_src": np.full(n, 1.0),
            "syn_without_synack_count": np.zeros(n),
            "bwd_pkt_len_mean": np.full(n, 1.0),
            "pkt_len_range": np.full(n, 1.0),
        })
        df.loc[500:, "flows_per_src"] = 20.0

        predicted = np.array(["PortScan"] * 500 + ["Benign"] * 500)
        result = run_policy(
            df, predicted, table, tau=1,
            global_budget_fraction=0.05, priority2_reserved_fraction=0.01,
        )
        assert result.priority_used[2] >= 1  # P2 got SOME budget despite P1 demand


class TestSignatureTableSaturationDiagnostic:
    def test_check_bin_tie_saturation_flags_degenerate_feature(self):
        from dataplane.escalation_policy import check_bin_tie_saturation

        n = 500
        df = pd.DataFrame({
            "Label": ["Benign"] * n,
            "flows_per_src": np.random.default_rng(2).integers(1, 50, n).astype(float),
            "distinct_dst_ports_per_src": np.random.default_rng(3).integers(1, 10, n).astype(float),
            "syn_without_synack_count": np.zeros(n),  # entirely degenerate
            "bwd_pkt_len_mean": np.random.default_rng(4).random(n) * 100,
            "pkt_len_range": np.random.default_rng(5).random(n) * 50,
        })
        table = build_signature_table(df, n_bins=6, floor=1)
        report = check_bin_tie_saturation(df, table)
        degenerate = report[report["feature"] == "syn_without_synack_count"]
        assert (degenerate["tied_mass"] == 1.0).any()
