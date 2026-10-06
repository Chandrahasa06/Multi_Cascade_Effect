import itertools

import pytest

from eval.auc import bootstrap_auc_ci, compute_auc


def _brute_force_auc(y_true, y_score):
    """Reference implementation: literal pairwise comparison, ties count
    as half a win -- used to check compute_auc's rank-sum shortcut agrees
    on small, hand-checkable inputs."""
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    total = 0.0
    for p, n in itertools.product(pos, neg):
        if p > n:
            total += 1.0
        elif p == n:
            total += 0.5
    return total / (len(pos) * len(neg))


def test_perfect_separation_is_one():
    y_true = [1, 1, 1, 0, 0, 0]
    y_score = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert compute_auc(y_true, y_score) == pytest.approx(1.0)


def test_perfectly_wrong_is_zero():
    y_true = [1, 1, 1, 0, 0, 0]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert compute_auc(y_true, y_score) == pytest.approx(0.0)


def test_all_tied_scores_is_half():
    y_true = [1, 1, 0, 0, 0]
    y_score = [0.5, 0.5, 0.5, 0.5, 0.5]
    assert compute_auc(y_true, y_score) == pytest.approx(0.5)


def test_matches_brute_force_on_random_small_cases():
    import random

    rng = random.Random(1234)
    for _ in range(20):
        n = rng.randint(4, 12)
        y_true = [rng.randint(0, 1) for _ in range(n)]
        if len(set(y_true)) < 2:
            continue
        # include some ties deliberately
        y_score = [round(rng.uniform(0, 1), 1) for _ in range(n)]
        assert compute_auc(y_true, y_score) == pytest.approx(_brute_force_auc(y_true, y_score))


def test_order_invariant_to_input_order():
    y_true = [1, 0, 1, 0, 0]
    y_score = [0.9, 0.1, 0.4, 0.6, 0.2]
    base = compute_auc(y_true, y_score)
    paired = list(zip(y_true, y_score))
    import random

    rng = random.Random(7)
    for _ in range(5):
        rng.shuffle(paired)
        yt, ys = zip(*paired)
        assert compute_auc(list(yt), list(ys)) == pytest.approx(base)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_auc([1, 0], [0.5])


def test_rejects_single_class():
    with pytest.raises(ValueError):
        compute_auc([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        compute_auc([0, 0, 0], [0.1, 0.2, 0.3])


def test_bootstrap_ci_contains_point_estimate_and_is_ordered():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_score = [0.9, 0.7, 0.6, 0.4, 0.5, 0.3, 0.2, 0.35, 0.1]
    point = compute_auc(y_true, y_score)
    lo, hi = bootstrap_auc_ci(y_true, y_score, n_resamples=500, seed=0)
    assert lo <= hi
    # a 95% percentile CI need not strictly contain the point estimate in
    # every resample-dependent case, but for this well-separated, n=9
    # example it should comfortably bracket it
    assert lo <= point <= hi


def test_bootstrap_ci_is_deterministic_given_seed():
    y_true = [1, 1, 0, 0, 0]
    y_score = [0.8, 0.6, 0.5, 0.4, 0.2]
    a = bootstrap_auc_ci(y_true, y_score, n_resamples=300, seed=42)
    b = bootstrap_auc_ci(y_true, y_score, n_resamples=300, seed=42)
    assert a == b
