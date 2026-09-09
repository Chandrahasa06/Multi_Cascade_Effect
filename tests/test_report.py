import pandas as pd

from eval.report import (
    ablation_comparison_table,
    build_recall_curve_frame,
    per_class_ablation_delta,
    per_class_recall_table,
    plot_recall_curve,
    write_results,
)
from eval.sweep import SweepPoint, SweepResult


def make_recall(rows):
    return pd.DataFrame(rows, columns=["label", "escalated_count", "total_count", "recall"])


def make_result(points):
    return SweepResult(points=points)


class TestPerClassRecallTable:
    def test_picks_nearest_percentile_per_target_rate(self):
        result = make_result(
            [
                SweepPoint(90.0, 0.20, 0.20, make_recall([["DoS", 8, 10, 0.8]]), 0.0, 0.0, 5),
                SweepPoint(99.0, 0.05, 0.05, make_recall([["DoS", 5, 10, 0.5]]), 0.0, 0.0, 5),
                SweepPoint(99.9, 0.01, 0.01, make_recall([["DoS", 2, 10, 0.2]]), 0.0, 0.0, 5),
            ]
        )
        table = per_class_recall_table(result, target_rates=(0.05,))
        assert table.iloc[0]["recall"] == 0.5
        assert table.iloc[0]["percentile"] == 99.0

    def test_flags_small_sample_classes(self):
        result = make_result(
            [SweepPoint(99.0, 0.05, 0.05, make_recall([["Heartbleed", 1, 11, 1 / 11]]), 0.0, 0.0, 5)]
        )
        table = per_class_recall_table(result, target_rates=(0.05,))
        assert bool(table.iloc[0]["small_sample"]) is True


class TestBuildRecallCurveFrame:
    def test_flattens_all_points_and_classes(self):
        result = make_result(
            [
                SweepPoint(90.0, 0.2, 0.2, make_recall([["DoS", 8, 10, 0.8], ["Bot", 1, 5, 0.2]]), 0, 0, 5),
                SweepPoint(99.0, 0.05, 0.05, make_recall([["DoS", 5, 10, 0.5], ["Bot", 0, 5, 0.0]]), 0, 0, 5),
            ]
        )
        curve = build_recall_curve_frame(result)
        assert len(curve) == 4
        assert set(curve["label"]) == {"DoS", "Bot"}


class TestPlotRecallCurve:
    def test_writes_a_png(self, tmp_path):
        curve = pd.DataFrame(
            {
                "percentile": [90.0, 99.0, 90.0, 99.0],
                "overall_escalation_rate": [0.2, 0.05, 0.2, 0.05],
                "label": ["DoS", "DoS", "Bot", "Bot"],
                "recall": [0.8, 0.5, 0.3, 0.1],
                "escalated_count": [8, 5, 3, 1],
                "total_count": [10, 10, 10, 10],
            }
        )
        path = tmp_path / "curve.png"
        plot_recall_curve(curve, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_empty_frame_without_crashing(self, tmp_path):
        path = tmp_path / "empty.png"
        plot_recall_curve(pd.DataFrame(columns=["label", "overall_escalation_rate", "recall", "total_count"]), path)
        assert not path.exists()  # nothing to plot, nothing written


class TestAblationComparisonTable:
    def test_one_row_per_ablation(self):
        baseline = make_result(
            [SweepPoint(99.0, 0.05, 0.049, make_recall([["DoS", 5, 10, 0.5]]), 0.0, 0.01, 20)]
        )
        flow_only = make_result(
            [SweepPoint(98.0, 0.052, 0.05, make_recall([["DoS", 3, 10, 0.3]]), 0.0, 0.0, 16)]
        )
        table = ablation_comparison_table({"baseline": baseline, "flow_only": flow_only}, target_rate=0.05)
        assert len(table) == 2
        assert set(table["ablation"]) == {"baseline", "flow_only"}
        baseline_row = table[table["ablation"] == "baseline"].iloc[0]
        assert baseline_row["overall_attack_recall"] == 0.5


class TestPerClassAblationDelta:
    def test_computes_delta_between_two_sweeps(self):
        baseline = make_result(
            [SweepPoint(99.0, 0.05, 0.05, make_recall([["Bot", 8, 10, 0.8]]), 0.0, 0.0, 20)]
        )
        flow_only = make_result(
            [SweepPoint(98.0, 0.05, 0.05, make_recall([["Bot", 2, 10, 0.2]]), 0.0, 0.0, 16)]
        )
        delta = per_class_ablation_delta(baseline, flow_only).set_index("label")
        assert delta.loc["Bot", "recall_baseline"] == 0.8
        assert delta.loc["Bot", "recall_ablation"] == 0.2
        assert abs(delta.loc["Bot", "recall_delta"] - 0.6) < 1e-9


class TestWriteResults:
    def test_writes_all_expected_files(self, tmp_path):
        result = make_result(
            [
                SweepPoint(90.0, 0.2, 0.2, make_recall([["DoS", 8, 10, 0.8]]), 0.0, 0.01, 20),
                SweepPoint(99.0, 0.05, 0.05, make_recall([["DoS", 5, 10, 0.5]]), 0.0, 0.02, 20),
            ]
        )
        ablations = {"baseline": result}
        trigger_freq = pd.DataFrame(
            {"feature": ["flow_bytes_per_sec"], "crossing_rate": [0.1], "trigger_share": [0.9]}
        )
        join_rates = pd.DataFrame(
            {"day": ["monday"], "key_mode": ["eval"], "join_rate": [1.0]}
        )
        write_results(result, ablations, trigger_freq, join_rates, results_dir=tmp_path)

        assert (tmp_path / "recall_curve.csv").exists()
        assert (tmp_path / "recall_curve.png").exists()
        assert (tmp_path / "per_class_recall.csv").exists()
        assert (tmp_path / "sweep_summary.csv").exists()
        assert (tmp_path / "ablation_comparison.csv").exists()
        assert (tmp_path / "trigger_frequency.csv").exists()
        assert (tmp_path / "join_rates.csv").exists()
        assert (tmp_path / "summary.md").exists()
        assert "Per-class recall" in (tmp_path / "summary.md").read_text()
