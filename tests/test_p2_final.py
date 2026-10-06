"""Tests for the close-out wiring: refit never reads Label, the token bucket is the
default meter, the per-day mode still works, the n_bins decision is wired, and the
v1 and v2 CSVs are unchanged.
"""
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataplane.controller_verdict import AgentVerdict, LabelStandInVerdict
from dataplane.escalation_policy import (
    PRIORITY2_FEATURES,
    OnlineSignatureTable,
    SignatureTable,
    run_policy_reserved_thirds,
    run_policy_token,
    run_policy_token_refit,
)

T0 = 1_499_072_158_000_000  # microseconds, 2017-07-03 11:55:58 UTC
HOUR = 3_600_000_000


class AllBenignVerdict:
    """Ignores labels entirely. Used to show the walk never needs one."""

    def __init__(self):
        self.calls = []

    def judge(self, escalated):
        self.calls.append(len(escalated))
        return np.ones(len(escalated), dtype=bool)


def _frame(n=600, seed=0, hours=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.integers(0, 12, n).astype(float) for f in PRIORITY2_FEATURES})
    df["first_ts"] = T0 + rng.integers(0, hours * HOUR, n).astype(np.int64)
    df["flow_uid"] = [f"d::{i}::x" for i in range(n)]
    df["Label"] = np.where(rng.random(n) < 0.2, "PortScan", "BENIGN")
    return df


def _fit_table(seed=1):
    rng = np.random.default_rng(seed)
    fit = pd.DataFrame({f: rng.integers(0, 12, 800).astype(float) for f in PRIORITY2_FEATURES})
    fit["Label"] = "Benign"
    return OnlineSignatureTable.from_benign_fit(fit, n_bins=6, floor=5)


def test_update_has_no_label_parameter():
    params = inspect.signature(OnlineSignatureTable.update).parameters
    assert not [p for p in params if "label" in p.lower()]
    assert list(params) == ["self", "df", "benign_judged"]


def test_refit_never_reads_labels_except_through_the_verdict():
    """Scramble every label. With a verdict that ignores labels, the admissions,
    the K sizes and the updates must be identical. If the walk read Label, they
    would differ."""
    df = _frame(seed=2)
    p1 = np.zeros(len(df), dtype=bool)
    scrambled = df.copy()
    scrambled["Label"] = np.random.default_rng(99).permutation(df["Label"].to_numpy())
    kwargs = dict(tau=20, global_budget_fraction=0.05, burst_seconds=3600)
    r1, t1, k1 = run_policy_token_refit(df, p1, _fit_table(), AllBenignVerdict(), **kwargs)
    r2, t2, k2 = run_policy_token_refit(scrambled, p1, _fit_table(), AllBenignVerdict(), **kwargs)
    assert (r1.escalated == r2.escalated).all()
    assert (r1.priority == r2.priority).all()
    assert k1 == k2
    assert t1.common == t2.common


def test_refit_judges_only_p2_admitted_flows():
    df = _frame(seed=3)
    p1 = np.zeros(len(df), dtype=bool)
    verdict = AllBenignVerdict()
    result, _, _ = run_policy_token_refit(df, p1, _fit_table(), verdict,
                                          tau=20, global_budget_fraction=0.05, burst_seconds=3600)
    assert sum(verdict.calls) == int((result.priority == 2).sum())


def test_agent_verdict_is_not_wired_and_refuses_to_run():
    with pytest.raises(NotImplementedError):
        AgentVerdict().judge(_frame(n=5))


def test_label_standin_is_marked_as_an_upper_bound():
    assert LabelStandInVerdict.upper_bound is True
    assert AgentVerdict.upper_bound is False


def test_fit_rejects_non_benign_labels():
    fit = pd.DataFrame({f: np.arange(10, dtype=float) for f in PRIORITY2_FEATURES})
    fit["Label"] = "PortScan"
    with pytest.raises(AssertionError):
        OnlineSignatureTable.from_benign_fit(fit, n_bins=6, floor=5)


def test_update_rejects_length_mismatch():
    table = _fit_table()
    df = _frame(n=10)
    with pytest.raises(ValueError):
        table.update(df, np.ones(9, dtype=bool))


def test_token_bucket_is_the_default_meter():
    default = inspect.signature(run_policy_reserved_thirds).parameters["meter"].default
    assert default == "token"
    df = _frame(seed=4)
    p1 = np.zeros(len(df), dtype=bool)
    table = _fit_table().as_signature_table()
    via_default = run_policy_reserved_thirds(df, p1, table, tau=20, global_budget_fraction=0.05, burst_seconds=3600)
    via_token = run_policy_token(df, p1, table, tau=20, global_budget_fraction=0.05, burst_seconds=3600)
    assert (via_default.escalated == via_token.escalated).all()
    assert (via_default.priority == via_token.priority).all()


def test_per_day_mode_is_still_reachable_and_uses_the_meter_caps():
    df = _frame(seed=5)
    p1 = np.zeros(len(df), dtype=bool)
    table = _fit_table().as_signature_table()
    res = run_policy_reserved_thirds(df, p1, table, tau=20, global_budget_fraction=0.05, meter="per_day")
    g = int(np.ceil(0.05 * len(df)))
    f2 = int(np.ceil(0.70 * g))
    f3 = int(np.ceil(0.20 * g))
    assert res.global_capacity == g
    assert res.priority_capacity[2] == g - min(f3, g - min(f2, g))


def test_unknown_meter_is_rejected():
    df = _frame(seed=6)
    p1 = np.zeros(len(df), dtype=bool)
    table = _fit_table().as_signature_table()
    with pytest.raises(ValueError):
        run_policy_reserved_thirds(df, p1, table, tau=20, global_budget_fraction=0.05, meter="weekly")


def test_production_n_bins_is_the_measured_decision():
    import eval.escalation_eval as ee
    assert ee.P2_N_BINS == 6
    assert ee.P2_FLOOR == 5


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_v1_and_v2_csvs_are_unchanged():
    assert _md5(Path("results/escalated_flows.csv")) == "8a598a25a075c237a003e978ccc8dc39"
    assert _md5(Path("results/escalated_flows_v2.csv")) == "7398dec2518606cec09c8ae568dd4d7f"
