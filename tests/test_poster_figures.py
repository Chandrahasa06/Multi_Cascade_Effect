"""Smoke tests for eval/poster_figures.py against a small SYNTHETIC
multi-condition dataset built from the real (but schema-older) v4
20-record clean results.

v4 predates the self-reported trust schema and has only one condition
(clean) -- neither is enough to exercise Fig 1-3's cross-condition
comparisons for real. These tests fabricate placeholder self-report
values (copied from v4's own code-computed C/E/V, since that's the
closest real number available) and four deterministically-perturbed
"fault" variants per record, purely so every code path (loading,
binning, regression, plotting, CSV writing) runs end to end without
crashing and produces sane output shapes. This is NOT a validation that
the fault-injection NUMBERS are meaningful -- only that poster_figures.py
itself is correct and ready for the real
results/fault_injection_20.jsonl once it exists.
"""
import json

import pytest

from eval import fault_injection as fi
from eval import poster_figures as pf

V4_RESULTS = "results/agent_pipeline_20_v4.jsonl"
V4_MANIFEST = "results/escalation_records_sample_manifest.INTERNAL.v1_50each.jsonl"


def _synthesize_self_report(resp: dict, trust: dict, jitter: float) -> dict:
    """Placeholder self-report: code-computed C/E/V plus a small
    deterministic offset so different conditions don't look identical."""
    resp = dict(resp)
    resp["confidence"] = max(0.0, min(1.0, (trust.get("C") or 0.7) + jitter))
    resp["evidence_support"] = max(0.0, min(1.0, (trust.get("E") or 0.7) - jitter))
    resp["verification"] = max(0.0, min(1.0, (trust.get("V") or 0.7) - 2 * jitter))
    return resp


@pytest.fixture(scope="module")
def placeholder_dataset(tmp_path_factory):
    v4_rows = []
    with open(V4_RESULTS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                v4_rows.append(json.loads(line))
    assert v4_rows, "v4 results file must exist and be non-empty for this placeholder test"

    out_path = tmp_path_factory.mktemp("poster_figures") / "placeholder.jsonl"
    #: deterministic per-condition jitter so conditions differ in a
    #: fixed, reproducible (if scientifically meaningless) way.
    jitter_by_condition = {
        "clean": 0.0,
        fi.MISSING_EVIDENCE: 0.10,
        fi.INCORRECT_BEHAVIOR: 0.15,
        fi.HALLUCINATED_HYPOTHESIS: 0.20,
        fi.FAULTY_VERIFICATION: 0.30,
    }
    bp_shift_by_condition = {
        "clean": 0.0,
        fi.MISSING_EVIDENCE: -0.05,
        fi.INCORRECT_BEHAVIOR: -0.10,
        fi.HALLUCINATED_HYPOTHESIS: -0.15,
        fi.FAULTY_VERIFICATION: -0.25,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        for condition in ("clean", *fi.ALL_CONDITIONS):
            jitter = jitter_by_condition[condition]
            bp_shift = bp_shift_by_condition[condition]
            for row in v4_rows:
                new_row = dict(row)
                new_row["condition"] = condition
                for agent in pf.AGENTS:
                    resp = dict(row[agent])
                    trust = (row.get("trust_scores") or {}).get(agent, {}) or {}
                    new_row[agent] = _synthesize_self_report(resp, trust, jitter)
                a5 = dict(new_row["a5"])
                a5["benign_plausibility"] = max(0.0, min(1.0, a5["benign_plausibility"] + bp_shift))
                new_row["a5"] = a5
                f.write(json.dumps(new_row) + "\n")

    return str(out_path)


def test_load_jsonl_and_manifest_work(placeholder_dataset):
    rows = pf.load_jsonl(placeholder_dataset)
    manifest = pf.load_manifest(V4_MANIFEST)
    assert len(rows) == 20 * 5  # 20 records x 5 conditions
    assert len(manifest) >= 20


def test_fig1_data_has_all_conditions_and_agents(placeholder_dataset):
    rows = pf.load_jsonl(placeholder_dataset)
    fig1 = pf.build_fig1_data(rows)
    conditions = {d["condition"] for d in fig1}
    agents = {d["agent"] for d in fig1}
    assert conditions == {"clean", *fi.ALL_CONDITIONS}
    assert agents == {"A1", "A2", "A3", "A4", "A5"}
    for d in fig1:
        assert d["self_reported_trust"] is not None
        assert 0.0 <= d["self_reported_trust"] <= 1.0


def test_fig2_data_covers_all_four_degraded_agents(placeholder_dataset):
    rows = pf.load_jsonl(placeholder_dataset)
    manifest = pf.load_manifest(V4_MANIFEST)
    fig2 = pf.build_fig2_data(rows, manifest)
    degraded = {d["degraded_agent"] for d in fig2}
    assert degraded == {"A1", "A2", "A3", "A4"}
    for d in fig2:
        assert d["n_paired_records"] == 20  # all 20 v4 records in this condition
        for key in ("decision_error_rate", "false_escalation_rate", "missed_detection_rate"):
            assert d[key] is None or 0.0 <= d[key] <= 1.0


def test_fig3_points_and_binning(placeholder_dataset):
    rows = pf.load_jsonl(placeholder_dataset)
    manifest = pf.load_manifest(V4_MANIFEST)
    points = pf.build_fig3_points(rows, manifest)
    # 20 records x 5 conditions x 5 agents, all with self-report present
    assert len(points) == 20 * 5 * 5
    assert all(e in (0, 1) for _, e in points)

    bins = pf.bin_fig3_points(points, n_bins=9)
    assert len(bins) == 9
    assert sum(b["n_points"] for b in bins) == len(points)


def test_fig3_linear_fit_has_expected_fields(placeholder_dataset):
    rows = pf.load_jsonl(placeholder_dataset)
    manifest = pf.load_manifest(V4_MANIFEST)
    points = pf.build_fig3_points(rows, manifest)
    bins = pf.bin_fig3_points(points, n_bins=9)
    fit = pf.fit_linear(bins)
    assert fit is not None
    assert {"slope", "intercept", "r2", "n_bins_used"} <= set(fit)
    assert -1e6 < fit["slope"] < 1e6  # sane, not NaN/inf


def test_divergence_report_runs_and_is_near_zero_for_synthetic_data(placeholder_dataset):
    # placeholder self-report was derived FROM the code-computed score
    # with a small offset -- divergence should be small but (thanks to
    # the offset) not exactly zero, proving the comparison actually runs.
    rows = pf.load_jsonl(placeholder_dataset)
    divergence = pf.build_divergence_data(rows)
    assert len(divergence) == len(pf.AGENTS) * 2  # evidence_support + verification per agent
    for d in divergence:
        if d["agent"] == "A5":
            # A5's code-computed E/V are undefined by design (no claims
            # to check/corroborate for a verdict-only response) -- see
            # agents/pipeline.py::_a5_trust_score -- so there is nothing
            # to compare self-report against for A5 specifically.
            assert d["n"] == 0
            continue
        assert d["n"] > 0
        assert d["mean_abs_diff"] is not None


def test_main_produces_all_figures_and_csvs(placeholder_dataset, tmp_path):
    out_dir = tmp_path / "figures"
    rc = pf.main([
        "--input", placeholder_dataset,
        "--manifest", V4_MANIFEST,
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    expected = [
        "fig1_trust_by_stage.png", "fig1_trust_by_stage.csv",
        "fig2_failure_modes.png", "fig2_failure_modes.csv",
        "fig3_gap_vs_error.png", "fig3_gap_vs_error.csv",
        "fig4_self_vs_code.png", "fig4_self_vs_code_divergence.csv",
    ]
    for name in expected:
        path = out_dir / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0, f"{name} is empty"
