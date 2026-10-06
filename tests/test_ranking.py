"""Tests for borrowing with reserved floors, the unusualness scores, and the
ranking buffer. Also checks that the scores never read a label and that
first-come-first-served stays reachable as the baseline."""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from dataplane.escalation_policy import (
    MICROS_PER_SECOND,
    PriorityTokenBuckets,
    RESERVED_FLOOR_FRACTION,
    SignatureTable,
    benign_signature_counts,
    frequency_unusualness,
    nearest_known_distance,
    run_policy_token,
    run_policy_token_buffered,
)

T0 = 1_499_072_158_000_000
FEATS = ("a", "b", "c", "d")


def _pb(rate=1.0, cap=100.0, borrow=True):
    return PriorityTokenBuckets.from_budget(rate, cap, borrow=borrow)


def _set_tokens(pb, priority, tokens):
    pb.buckets[priority].tokens = float(tokens)


def test_borrowing_takes_surplus_only_and_keeps_the_floor():
    pb = _pb()
    t = T0
    pb.admit(t, 3)  # advance everything once
    _set_tokens(pb, 1, 0.0)
    floor2 = pb.floors[2]
    _set_tokens(pb, 2, floor2)  # exactly at the floor: no surplus
    _set_tokens(pb, 3, pb.floors[3])
    assert pb.admit(t, 1) is False, "P1 must not borrow a lender that is at its floor"
    _set_tokens(pb, 2, floor2 + 1.0)  # one spare token above the floor
    assert pb.admit(t, 1) is True
    assert pb.buckets[2].tokens == pytest.approx(floor2), "lender drawn exactly down to its floor"


def test_lender_is_never_drawn_below_its_floor_under_a_flood():
    pb = _pb()
    t = T0
    pb.admit(t, 1)
    _set_tokens(pb, 1, 0.0)
    _set_tokens(pb, 2, pb.buckets[2].capacity)
    _set_tokens(pb, 3, pb.buckets[3].capacity)
    admitted = sum(pb.admit(t, 1) for _ in range(500))  # P1 floods at one instant
    assert admitted > 0
    assert pb.buckets[2].tokens >= pb.floors[2] - 1e-9
    assert pb.buckets[3].tokens >= pb.floors[3] - 1e-9


def test_partitioned_mode_never_borrows():
    pb = _pb(borrow=False)
    t = T0
    pb.admit(t, 1)
    _set_tokens(pb, 1, 0.0)
    _set_tokens(pb, 2, pb.buckets[2].capacity)
    assert pb.admit(t, 1) is False


def test_floor_is_a_stated_fraction_of_capacity():
    pb = _pb()
    for p, b in pb.buckets.items():
        assert pb.floors[p] == pytest.approx(RESERVED_FLOOR_FRACTION * b.capacity)


def test_frequency_score_ranks_rarer_signatures_higher():
    counts = Counter({(0, 0, 0, 0): 4, (1, 1, 1, 1): 1})
    sc = frequency_unusualness([(0, 0, 0, 0), (1, 1, 1, 1), (9, 9, 9, 9)], counts)
    assert sc[2] == pytest.approx(1.0)  # never seen
    assert sc[1] == pytest.approx(0.5)  # seen once
    assert sc[0] == pytest.approx(0.2)  # seen four times: just under the floor
    assert sc[2] > sc[1] > sc[0]


def test_distance_score_is_bins_differing_from_nearest_known():
    common = frozenset({(0, 0, 0, 0)})
    d = nearest_known_distance([(1, 0, 0, 0), (2, 2, 0, 0), (3, 3, 3, 0)], common)
    assert list(d) == [1.0, 2.0, 3.0]


def test_scores_never_read_labels():
    fit = pd.DataFrame({f: np.arange(40, dtype=float) % 4 for f in FEATS})
    fit["Label"] = "BENIGN"
    edges = {f: np.array([1.0, 2.0, 3.0]) for f in FEATS}
    counts = benign_signature_counts(fit, FEATS, edges)
    sigs = [(1, 0, 0, 0), (3, 3, 0, 0)]
    a = frequency_unusualness(sigs, counts)
    b = frequency_unusualness(sigs, counts)  # no label argument exists to change
    assert (a == b).all()
    import inspect
    for fn in (frequency_unusualness, nearest_known_distance, benign_signature_counts):
        assert not [p for p in inspect.signature(fn).parameters if "label" in p.lower()]


def _synthetic_table():
    edges = {f: np.array([1.0, 2.0, 3.0]) for f in FEATS}
    common = frozenset({(0, 0, 0, 0)})
    return SignatureTable(bin_edges=edges, common_signatures=common, features=FEATS, floor=5, fit_n=0)


def _flows(rows):
    """rows: list of (ts_offset_us, feature tuple, priority_matched_by_P1)."""
    data = {f: [] for f in FEATS}
    ts, flow_uid, lab = [], [], []
    for i, (off, sig, _) in enumerate(rows):
        for f, v in zip(FEATS, sig):
            data[f].append(float(v))
        ts.append(T0 + off)
        flow_uid.append(f"d::{i}::x")
        lab.append("PortScan")
    df = pd.DataFrame(data)
    df["first_ts"] = ts
    df["flow_uid"] = flow_uid
    df["Label"] = lab
    p1 = np.array([bool(m) for _, _, m in rows])
    return df, p1


def test_buffer_admits_highest_scoring_first_when_only_one_token_is_available():
    # Three P2-flagged flows arrive in one one-second window; one token is
    # available at the window's close. Distance 3 must beat distance 1 and
    # distance 2 even though it arrives last.
    rows = [
        (0, (1, 0, 0, 0), False),   # distance 1 from the common signature
        (1, (2, 2, 0, 0), False),   # distance 2
        (2, (3, 3, 3, 0), False),   # distance 3, arrives last
        (3_600 * MICROS_PER_SECOND, (0, 0, 0, 0), True),  # a P1 flow an hour later
    ]
    df, p1 = _flows(rows)
    res, lat, stats = run_policy_token_buffered(
        df, p1, _synthetic_table(), tau=0, global_budget_fraction=1.0, burst_seconds=1.0,
        window_us=1.0 * MICROS_PER_SECOND, scorer="distance")
    p2_admitted = np.nonzero(res.priority == 2)[0]
    assert list(p2_admitted) == [2], "the highest-distance flow should take the single token"


def test_buffer_ranking_is_deterministic():
    rows = [(i * 1000, (i % 4, (i * 3) % 4, 0, 0), False) for i in range(60)]
    rows.append((3_600 * MICROS_PER_SECOND, (0, 0, 0, 0), True))
    df, p1 = _flows(rows)
    args = dict(tau=0, global_budget_fraction=0.2, burst_seconds=2.0,
                window_us=0.5 * MICROS_PER_SECOND, scorer="distance")
    a, la, _ = run_policy_token_buffered(df, p1, _synthetic_table(), **args)
    b, lb, _ = run_policy_token_buffered(df, p1, _synthetic_table(), **args)
    assert (a.priority == b.priority).all()
    assert np.array_equal(la, lb, equal_nan=True)


def test_fcfs_window_zero_matches_the_plain_token_policy():
    rows = [(i * 2000, (i % 4, (i * 3) % 4, 0, 0), i % 7 == 0) for i in range(80)]
    df, p1 = _flows(rows)
    plain = run_policy_token(df, p1, _synthetic_table(), tau=0, global_budget_fraction=0.3, burst_seconds=3.0)
    buffered, lat, _ = run_policy_token_buffered(df, p1, _synthetic_table(), tau=0, global_budget_fraction=0.3,
                                                 burst_seconds=3.0, window_us=0.0, scorer="fcfs")
    assert (plain.priority == buffered.priority).all()
    assert np.all(lat[np.isfinite(lat)] == 0.0), "window zero adds no latency"


def test_buffered_fcfs_adds_latency_but_not_ranking():
    rows = [(i * 300_000, (i % 4, (i * 3) % 4, 0, 0), False) for i in range(20)]
    df, p1 = _flows(rows)
    res, lat, _ = run_policy_token_buffered(df, p1, _synthetic_table(), tau=0, global_budget_fraction=0.5,
                                            burst_seconds=2.0, window_us=2.0 * MICROS_PER_SECOND, scorer="fcfs")
    admitted = np.isfinite(lat)
    assert admitted.any()
    assert np.all(lat[admitted] >= 0)
    assert np.all(lat[admitted] < 2.0 * MICROS_PER_SECOND)


def test_frequency_scorer_requires_counts():
    rows = [(0, (1, 0, 0, 0), False)]
    df, p1 = _flows(rows)
    with pytest.raises(ValueError):
        run_policy_token_buffered(df, p1, _synthetic_table(), tau=0, global_budget_fraction=0.5,
                                  window_us=MICROS_PER_SECOND, scorer="frequency")
