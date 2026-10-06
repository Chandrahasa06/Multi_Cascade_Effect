"""Tests for TokenBucket and PriorityTokenBuckets in dataplane/escalation_policy.py.

Covers: long-run rate matches the target; burst caps accumulation; tokens accrue
by time not by flow count; microsecond units asserted; per-priority buckets
cannot be starved; deterministic. The existing Meter is not exercised here and
must stay unchanged.
"""
import numpy as np
import pytest

from dataplane.escalation_policy import (
    MICROS_PER_SECOND,
    PriorityTokenBuckets,
    TokenBucket,
    assert_microseconds,
)

T0 = 1_499_072_158_000_000  # 2017-07-03 11:55:58 UTC, microseconds


def test_long_run_rate_matches_target():
    rate = 0.5  # tokens per second
    b = TokenBucket(rate_per_second=rate, capacity=10)
    admitted = 0
    # demand far above the rate: one attempt every 0.1 s for 10,000 s
    for k in range(100_000):
        if b.take(T0 + k * 100_000):
            admitted += 1
    expected = rate * 10_000 + b.capacity
    assert abs(admitted - expected) <= 1, (admitted, expected)


def test_burst_caps_accumulation():
    b = TokenBucket(rate_per_second=1.0, capacity=5)
    # a long quiet period cannot bank more than the capacity
    assert b.advance(T0 + 3600 * MICROS_PER_SECOND) == pytest.approx(5.0)
    admitted = sum(b.take(T0 + 3600 * MICROS_PER_SECOND) for _ in range(20))
    assert admitted == 5


def test_tokens_accrue_by_time_not_by_flow_count():
    # Same wall-clock span, very different flow counts: same tokens.
    sparse = TokenBucket(rate_per_second=2.0, capacity=1000, tokens=0.0)
    dense = TokenBucket(rate_per_second=2.0, capacity=1000, tokens=0.0)
    sparse.advance(T0)
    dense.advance(T0)
    span = 30 * MICROS_PER_SECOND
    sparse.advance(T0 + span)  # one event
    for k in range(1, 3001):  # 3,000 events over the same span
        dense.advance(T0 + int(span * k / 3000))
    assert sparse.tokens == pytest.approx(60.0)
    assert dense.tokens == pytest.approx(60.0)


def test_empty_bucket_refills_without_traffic():
    b = TokenBucket(rate_per_second=1.0, capacity=10, tokens=0.0)
    b.advance(T0)
    assert not b.has_token(T0)
    assert b.has_token(T0 + 2 * MICROS_PER_SECOND)  # two seconds later: two tokens


def test_microseconds_are_asserted_at_the_boundary():
    b = TokenBucket(rate_per_second=1.0, capacity=10)
    with pytest.raises(ValueError):
        b.advance(1_499_072_158.0)  # seconds, not microseconds
    with pytest.raises(ValueError):
        assert_microseconds(1.5e20)
    assert_microseconds(T0)  # accepted


def test_time_must_not_go_backwards():
    b = TokenBucket(rate_per_second=1.0, capacity=10)
    b.advance(T0 + MICROS_PER_SECOND)
    with pytest.raises(ValueError):
        b.advance(T0)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_second=0.0, capacity=10)
    with pytest.raises(ValueError):
        TokenBucket(rate_per_second=1.0, capacity=0.5)


def test_per_priority_buckets_cannot_be_starved_by_a_p1_burst():
    pb = PriorityTokenBuckets.from_budget(global_rate_per_second=1.0, global_capacity=100)
    # P1 floods: attempts every 1 ms for 100 s
    for k in range(100_000):
        pb.admit(T0 + k * 1000, 1)
    # P2 and P3 still have their reserved refill: at their rate they admit
    ts = T0 + 100 * MICROS_PER_SECOND
    admitted2 = admitted3 = 0
    for k in range(10_000):  # one time-ordered stream: P2 and P3 interleaved
        t = ts + k * 10_000
        admitted2 += pb.admit(t, 2)
        admitted3 += pb.admit(t, 3)
    assert admitted2 >= 60, admitted2  # reserved 0.70 of 1.0/s over ~100 s
    assert admitted3 >= 15, admitted3  # reserved 0.20 of 1.0/s over ~100 s


def test_global_bucket_is_enforced():
    pb = PriorityTokenBuckets.from_budget(global_rate_per_second=0.1, global_capacity=3)
    admitted = sum(pb.admit(T0, p) for p in (1, 2, 3) * 10)
    assert admitted <= 3


def test_priority_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        PriorityTokenBuckets.from_budget(1.0, 10, fractions=(0.1, 0.1, 0.1))


def test_deterministic_for_the_same_event_sequence():
    rng = np.random.default_rng(3)
    ts = np.sort(rng.integers(T0, T0 + 3600 * MICROS_PER_SECOND, 2000))
    pri = rng.integers(1, 4, 2000)
    runs = []
    for _ in range(2):
        pb = PriorityTokenBuckets.from_budget(0.05, 8)
        runs.append([pb.admit(int(t), int(p)) for t, p in zip(ts, pri)])
    assert runs[0] == runs[1]


def test_lazy_advance_equals_eager_advance_including_the_cap():
    # admit() advances only the buckets it serves, so an idle bucket is advanced
    # late. That must give the same tokens as advancing it at every step.
    eager = TokenBucket(rate_per_second=0.5, capacity=3.0, tokens=0.0)
    lazy = TokenBucket(rate_per_second=0.5, capacity=3.0, tokens=0.0)
    eager.advance(T0)
    lazy.advance(T0)
    for k in range(1, 201):
        eager.advance(T0 + k * 2 * MICROS_PER_SECOND)  # advanced at every step
    lazy.advance(T0 + 200 * 2 * MICROS_PER_SECOND)  # advanced once, at the end
    assert lazy.tokens == pytest.approx(eager.tokens)
