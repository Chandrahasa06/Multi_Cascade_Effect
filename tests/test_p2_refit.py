"""Tests for eval/p2_refit.py.

Covers: rule B never reads labels; rule A only ever sees escalated flows;
evaluation never uses data already folded into K; the walk is deterministic;
the sliding window forgets; the day granularity defers updates to day end.
"""
from collections import Counter

import numpy as np
import pytest

import eval.p2_refit as pr
from eval.p2_refit import (
    CountState,
    FLOOR,
    SLOTS_PER_DAY,
    WalkInputs,
    run_walk,
)


def _synthetic(n=800, seed=0, R=30):
    rng = np.random.default_rng(seed)
    weekday = rng.integers(0, 5, n)
    first_ts = weekday * 86400.0 + rng.integers(0, 86400, n).astype(float)
    hour = np.floor(first_ts / 3600.0).astype(np.int64)
    codes = rng.integers(0, R, n).astype(np.int64)
    is_attack = (rng.random(n) < 0.2) & (weekday != 0)  # Monday is benign by construction
    eligible = rng.random(n) < 0.9
    monday_benign = (weekday == 0) & ~is_attack
    return WalkInputs(
        codes=codes, weekday=weekday, hour=hour, first_ts=first_ts,
        row_idx=np.arange(n), eligible=eligible, is_attack=is_attack,
        fit_codes=codes[monday_benign], fit_hours=hour[monday_benign], R=R,
    )


def _reference_unbounded(inp, rule, granularity):
    """Brute-force reference: Counter-based, unbounded, no window. Independent
    of CountState and of run_walk's bookkeeping."""
    counts = Counter(inp.fit_codes.tolist())
    n = len(inp.codes)
    flagged = np.zeros(n, dtype=bool)
    admitted = np.zeros(n, dtype=bool)
    eval_rows = np.nonzero(np.isin(inp.weekday, [1, 2, 3, 4]))[0]
    order = eval_rows[np.lexsort((inp.row_idx[eval_rows], inp.first_ts[eval_rows]))]
    keys = [(int(inp.weekday[r]), int(inp.hour[r]) if granularity == "hour" else 0) for r in order]
    groups = []
    for r, k in zip(order, keys):
        if groups and groups[-1][0] == k:
            groups[-1][1].append(r)
        else:
            groups.append((k, [r]))
    pending = []
    current_day = None
    remaining = 0
    for (day, _h), rows in groups:
        if day != current_day:
            for c in pending:
                counts[c] += 1
            pending = []
            current_day = day
            remaining = SLOTS_PER_DAY
        K = {c for c, v in counts.items() if v >= FLOOR}
        for r in rows:
            flagged[r] = int(inp.codes[r]) not in K
        cand = [r for r in rows if flagged[r] and inp.eligible[r]]
        take = cand[:remaining]
        remaining -= len(take)
        for r in take:
            admitted[r] = True
        if rule == "A":
            upd = [r for r in take if not inp.is_attack[r]]
        else:
            upd = [r for r in rows if inp.eligible[r] and not admitted[r]]
        newcodes = [int(inp.codes[r]) for r in upd]
        if granularity == "hour":
            for c in newcodes:
                counts[c] += 1
        else:
            pending.extend(newcodes)
    for c in pending:
        counts[c] += 1
    return flagged, admitted


@pytest.mark.parametrize("granularity", ["day", "hour"])
def test_rule_b_never_reads_labels(granularity):
    inp = _synthetic()
    scrambled = WalkInputs(**{**inp.__dict__, "is_attack": np.random.default_rng(99).random(len(inp.codes)) < 0.5})
    a = run_walk(inp, "B", granularity, "unbounded")
    b = run_walk(scrambled, "B", granularity, "unbounded")
    assert (a.flagged == b.flagged).all()
    assert (a.admitted == b.admitted).all()
    assert (a.in_update == b.in_update).all()


@pytest.mark.parametrize("granularity", ["day", "hour"])
def test_rule_a_only_ever_sees_escalated_flows(granularity):
    inp = _synthetic(seed=3)
    res = run_walk(inp, "A", granularity, "unbounded")
    assert not (res.in_update & ~res.admitted).any(), "rule A updated from a non-escalated flow"
    assert not (res.in_update & inp.is_attack).any(), "rule A updated from an attack flow"
    assert (res.in_update == (res.admitted & ~inp.is_attack)).all()


def test_rule_a_standin_is_called_only_with_admitted_rows(monkeypatch):
    inp = _synthetic(seed=4)
    seen = []
    real = pr._standin_perfect_controller_verdict

    def spy(is_attack_subset):
        seen.append(len(is_attack_subset))
        return real(is_attack_subset)

    monkeypatch.setattr(pr, "_standin_perfect_controller_verdict", spy)
    res = run_walk(inp, "A", "hour", "unbounded")
    assert sum(seen) == int(res.admitted.sum())


@pytest.mark.parametrize("granularity", ["day", "hour"])
@pytest.mark.parametrize("rule", ["A", "B"])
def test_evaluation_uses_only_K_built_from_earlier_windows(granularity, rule):
    inp = _synthetic(seed=5)
    res = run_walk(inp, rule, granularity, "unbounded")
    ref_flag, ref_adm = _reference_unbounded(inp, rule, granularity)
    assert (res.flagged == ref_flag).all()
    assert (res.admitted == ref_adm).all()


def test_walk_is_deterministic():
    inp = _synthetic(seed=6)
    a = run_walk(inp, "B", "hour", "window_24h")
    b = run_walk(inp, "B", "hour", "window_24h")
    assert (a.flagged == b.flagged).all()
    assert (a.admitted == b.admitted).all()
    assert a.ksize_log == b.ksize_log


def test_window_forgets_signatures_not_seen_recently():
    st = CountState(R=4, policy="window_6h")
    codes = np.array([1, 1, 1, 1, 1], dtype=np.int64)  # five sightings at hour 0
    st.add(codes, np.zeros(5, dtype=np.int64))
    assert st.common_mask()[1]
    st.expire(now_hour=5)  # still inside the window (cutoff is 5-6 = -1)
    assert st.common_mask()[1]
    st.expire(now_hour=6)  # cutoff 0: hour-0 sightings drop out
    assert not st.common_mask()[1]
    assert st.total[1] == 0


def test_unbounded_never_forgets():
    st = CountState(R=4, policy="unbounded")
    st.add(np.array([2] * 5, dtype=np.int64), np.zeros(5, dtype=np.int64))
    st.expire(now_hour=10_000)
    assert st.common_mask()[2]


def test_day_granularity_holds_K_constant_within_a_day():
    inp = _synthetic(seed=7)
    res = run_walk(inp, "B", "day", "unbounded")
    for day in (1, 2, 3, 4):
        rows = np.nonzero(inp.weekday == day)[0]
        k_by_hour = {}
        for r in rows:
            k_by_hour.setdefault(int(inp.hour[r]), []).append(r)
        # K at the start of the day is the same for every hour of the day, so
        # the flags for all rows of the day come from one K. Check: rows with
        # identical codes within the day are flagged identically.
        by_code = {}
        for r in rows:
            by_code.setdefault(int(inp.codes[r]), set()).add(bool(res.flagged[r]))
        assert all(len(v) == 1 for v in by_code.values())


def test_budget_resets_each_day_and_is_respected():
    inp = _synthetic(n=4000, seed=8)
    res = run_walk(inp, "A", "day", "unbounded")
    for day in (1, 2, 3, 4):
        assert int(res.admitted[inp.weekday == day].sum()) <= SLOTS_PER_DAY


def test_clock_hour_uses_microseconds():
    from eval.p2_refit import clock_hour, MICROS_PER_HOUR
    # 2017-07-03 11:55:58 UTC in microseconds, the pool's real unit
    ts = np.array([1499072158000000.0, 1499072158000000.0 + MICROS_PER_HOUR])
    h = clock_hour(ts)
    assert h[1] - h[0] == 1, "one real hour must be one bucket"
    assert h[0] == 1499072158000000 // MICROS_PER_HOUR


def test_clock_hour_refuses_seconds():
    from eval.p2_refit import clock_hour
    with pytest.raises(ValueError):
        clock_hour(np.array([1499072158.0]))


def test_hour_budget_admits_at_most_100_per_clock_hour():
    inp = _synthetic(n=4000, seed=9)
    res = run_walk(inp, "A", "hour", "unbounded", budget="hour")
    for day in (1, 2, 3, 4):
        for h in np.unique(inp.hour[inp.weekday == day]):
            m = (inp.weekday == day) & (inp.hour == h)
            assert int(res.admitted[m].sum()) <= SLOTS_PER_DAY // 24


def test_hour_budget_requires_hour_granularity():
    inp = _synthetic(seed=10)
    with pytest.raises(ValueError):
        run_walk(inp, "A", "day", "unbounded", budget="hour")
