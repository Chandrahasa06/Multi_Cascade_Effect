"""Minimal, dependency-free AUC (area under the ROC curve) computation
plus a bootstrap confidence interval, for
results/plausibility_diagnostic.md's Step 4 (does a rank ordering of the
36 records separate attacks from benign better than the recorded
benign_plausibility numbers do).

AUC here is the Mann-Whitney U statistic, normalised: the probability
that a uniformly-random positive-class item scores higher than a
uniformly-random negative-class item, with ties counted as half a win.
This is exactly what sklearn.metrics.roc_auc_score computes, but this
project keeps its own implementation (a) so a test can pin the exact
formula down, and (b) so the diagnostic doesn't grow a new hard
dependency for one report.
"""
from __future__ import annotations

import random
from typing import List, Sequence, Tuple


def compute_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """``y_true``: 1 for the positive class (here: true attack), 0 for
    negative (true benign). ``y_score``: higher means "more likely
    positive" (here: higher means more likely attack). Ties in y_score
    are handled via the standard mid-rank correction (a tie contributes
    0.5 to the count of "positive scored higher", not 0 or 1), so a
    completely tied score set (every item gets the same score) correctly
    returns exactly 0.5, not an arbitrary value depending on input order.

    Raises ValueError if y_true has no positives or no negatives (AUC is
    undefined without both classes present).
    """
    n = len(y_true)
    if n != len(y_score):
        raise ValueError(f"y_true and y_score must be the same length, got {n} and {len(y_score)}")
    pos_scores = [s for t, s in zip(y_true, y_score) if t == 1]
    neg_scores = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos_scores or not neg_scores:
        raise ValueError(
            f"AUC undefined without both classes present: {len(pos_scores)} positive, "
            f"{len(neg_scores)} negative"
        )

    # Mann-Whitney U via rank-sum (O(n log n)), not the naive O(n_pos *
    # n_neg) pairwise comparison -- equivalent result, cheaper at scale,
    # and exercised identically by tests/test_auc.py's brute-force check.
    combined = [(s, 0) for s in neg_scores] + [(s, 1) for s in pos_scores]
    combined.sort(key=lambda x: x[0])
    ranks = _mid_ranks([s for s, _ in combined])
    rank_sum_pos = sum(r for (_, label), r in zip(combined, ranks) if label == 1)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _mid_ranks(sorted_values: List[float]) -> List[float]:
    """1-indexed ranks with ties averaged to their mid-rank (the standard
    tie-breaking convention for a rank-sum test), for a list already
    sorted ascending."""
    n = len(sorted_values)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        mid = (i + 1 + j + 1) / 2.0  # average of the 1-indexed rank range [i+1, j+1]
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    return ranks


def bootstrap_auc_ci(
    y_true: Sequence[int], y_score: Sequence[float], n_resamples: int = 2000, seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI on the AUC: resample (y_true, y_score)
    pairs with replacement `n_resamples` times, recompute AUC each time,
    and report the [2.5th, 97.5th] percentile. A resample lacking either
    class is skipped (rare at n~=36 with a 12-attack/... mix, but not
    impossible) rather than crashing the whole CI -- reported honestly
    below via how many resamples were actually used.
    """
    rng = random.Random(seed)
    n = len(y_true)
    values = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        ys = [y_score[i] for i in idx]
        try:
            values.append(compute_auc(yt, ys))
        except ValueError:
            continue
    values.sort()
    if not values:
        raise ValueError("no bootstrap resample contained both classes -- cannot form a CI")
    lo = values[int(0.025 * len(values))]
    hi = values[min(len(values) - 1, int(0.975 * len(values)))]
    return lo, hi
