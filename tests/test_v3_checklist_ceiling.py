"""Scoring and sweep logic for eval/v3_checklist_ceiling.py."""
import pytest

from eval import v3_checklist_ceiling as vc


def test_held_from_checklist_reads_only_yes():
    rows = [{"condition_id": "C1", "holds": "yes"}, {"condition_id": "C2", "holds": "no"},
            {"condition_id": "C3", "holds": "not_applicable"}]
    assert vc.held_from_checklist(rows) == {"C1"}


def test_held_from_code_reads_only_yes():
    assert vc.held_from_code({"C1": "yes", "C2": "no", "C12": "not_applicable"}) == {"C1"}


def test_count_score_counts_listed_conditions_only():
    assert vc.count_score({"C1", "C5", "C99"}) == 2
    assert vc.count_score({"C1", "C5"}, conditions=("C5",)) == 1


def test_auc_perfect_reversed_and_ties():
    assert vc.auc([3, 2, 0, 0], [True, True, False, False]) == 1.0
    assert vc.auc([0, 0, 3, 2], [True, True, False, False]) == 0.0
    assert vc.auc([1, 1], [True, False]) == 0.5


def test_sweep_counts_at_each_cutoff():
    out = vc.sweep([0, 1, 2, 3], [True, True, False, True])
    assert [r["cutoff"] for r in out] == [1, 2, 3]
    assert [(r["tp"], r["fp"]) for r in out] == [(2, 1), (1, 1), (1, 0)]
    assert out[0]["n_att"] == 3 and out[0]["n_ben"] == 1


def test_sweep_with_all_zero_scores_still_reports_cutoff_one():
    out = vc.sweep([0, 0], [True, False])
    assert out == [{"cutoff": 1, "tp": 0, "n_att": 1, "fp": 0, "n_ben": 1}]


def test_subset_flags_is_any_of():
    held = [{"C1"}, {"C2"}, set()]
    assert vc.subset_flags(held, ("C1", "C3")) == [True, False, False]


def test_search_subsets_ranks_by_recall_minus_benign_rate():
    held = [{"C5"}, {"C5"}, {"C1"}, {"C1"}]
    labels = [True, True, False, False]
    top = vc.search_subsets(held, labels, max_size=1)
    assert top[0]["subset"] == ["C5"] and top[0]["tp"] == 2 and top[0]["fp"] == 0
    assert top[-1]["youden"] == pytest.approx(-1.0)


def test_wilson_bounds():
    lo, hi = vc.wilson(0, 9)
    assert lo == 0.0 and 0.29 < hi < 0.31
