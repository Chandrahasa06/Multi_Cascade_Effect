"""Presentation figures + summary, built entirely from already-written
results -- zero API calls. Figures 2-5 (agent-pipeline data) read
results/agent_pipeline_200.jsonl, whatever fraction of it is done as of
this run -- every figure states its own n on the figure itself (legend,
title, or axis tick labels), never just in the returned data or the
summary doc. Figure 1 (data-plane recall vs escalation) reads the
corrected step-1 sweep CSVs in results/ (see PRESENTATION_SUMMARY.md for
why the pre-existing ones on disk had to be regenerated first: they
predated two documented fitting bugfixes and read PortScan recall as
~0%, a known-stale artifact, not a finding).

Every input path is a CLI flag (see --help); the defaults already point
at the same files eval.run_agent_pipeline appends to in place, so
re-running this with no arguments at all picks up however many of the
200 records have landed by then -- "change one path" only matters if
you want to point at a different file entirely (e.g. a frozen snapshot).

Run: python -m eval.presentation_figures
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.schema import VerdictLabel, derive_verdict
from agents.trust import DEFAULT_WEIGHTS
from eval import ablation_v200 as abl

AGENTS = ("a1", "a2", "a3", "a4", "a5")

FIGURES_DIR = Path("results/figures")
RECALL_CURVE_CSV = Path("results/recall_curve.csv")
PER_CLASS_BY_BENIGN_RATE_CSV = Path("results/per_class_recall_by_benign_rate.csv")
PIPELINE_JSONL = Path("results/agent_pipeline_200.jsonl")
MANIFEST_JSONL = Path("results/escalation_records_sample_manifest.INTERNAL.jsonl")

HEADLINE_CLASSES = ("Bot", "DDoS", "PortScan")

_POSTER_RC = {
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
}

_CLASS_COLORS = {"BENIGN": "#4C72B0", "DDoS": "#C44E52", "PortScan": "#DD8452", "Bot": "#55A868"}


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_manifest(path: Path = MANIFEST_JSONL) -> Dict[str, str]:
    return {r["flow_id"]: r["label"] for r in load_jsonl(path)}


# ---------- Fig 1: recall vs escalation rate, per class (step 1) ----------


def plot_fig1(
    recall_curve_path: Path = RECALL_CURVE_CSV,
    per_class_benign_rate_path: Path = PER_CLASS_BY_BENIGN_RATE_CSV,
    out_dir: Path = FIGURES_DIR,
) -> dict:
    df = pd.read_csv(recall_curve_path)
    out_path = out_dir / "fig1_recall_vs_escalation.png"

    # class total_count (n) up front, from the per-class table, so the
    # legend can state each headline class's n directly -- every figure
    # must state its n somewhere on the figure itself, not just in the
    # returned data or the summary doc.
    bdf = pd.read_csv(per_class_benign_rate_path)
    n_by_label = {row["label"]: int(row["total_count"]) for _, row in bdf.drop_duplicates("label").iterrows()}

    with plt.rc_context(_POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        for label, g in df.groupby("label"):
            g = g.sort_values("overall_escalation_rate")
            if label in HEADLINE_CLASSES:
                n = n_by_label.get(label)
                legend_label = f"{label} (n={n:,})" if n is not None else label
                ax.plot(g["overall_escalation_rate"] * 100, g["recall"] * 100,
                        marker="o", linewidth=2.8, markersize=7,
                        label=legend_label, color=_CLASS_COLORS.get(label), zorder=5)
            else:
                ax.plot(g["overall_escalation_rate"] * 100, g["recall"] * 100,
                        linewidth=1.0, color="lightgray", alpha=0.8, zorder=1)
        ax.plot([], [], color="lightgray", linewidth=1.0, label="other classes")
        ax.set_xlabel("Overall escalation rate (%)")
        ax.set_ylabel("Recall (%)")
        ax.set_title("Selector recall vs. escalation rate, by class (data-plane sweep)")
        ax.set_xlim(0, 100)
        ax.set_ylim(-2, 102)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", frameon=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    # Headline numbers: recall anchored on held-out BENIGN escalation rate
    # (1%/5%/10%), not overall escalation rate -- STATUS.md's own established
    # convention. Overall-rate anchoring collapses to the same single
    # (tightest) percentile point for every target here, since attack
    # traffic is ~22% of the operational week's volume (see summary doc);
    # the benign-anchored numbers are the ones that actually vary and are
    # comparable to every other benign-rate-anchored table in this project.
    headline: Dict[str, dict] = {}
    for label in HEADLINE_CLASSES:
        g = bdf[bdf["label"] == label]
        if g.empty:
            continue
        headline[label] = {
            str(row["target_escalation_rate"]): {
                "recall": float(row["recall"]), "escalated_count": int(row["escalated_count"]),
                "total_count": int(row["total_count"]), "actual_benign_escalation_rate": float(row["actual_escalation_rate"]),
            }
            for _, row in g.iterrows()
        }
    return headline


# ---------- Fig 2: benign_plausibility distribution by class ----------


def plot_fig2(rows: List[dict], manifest: Dict[str, str], out_dir: Path = FIGURES_DIR) -> dict:
    by_class: Dict[str, List[float]] = {}
    for r in rows:
        label = manifest.get(r["record_id"])
        if label is None:
            continue
        by_class.setdefault(label, []).append(r["a5"]["benign_plausibility"])

    classes = [c for c in ("PortScan", "DDoS", "BENIGN") if c in by_class]
    out_path = out_dir / "fig2_benign_plausibility_by_class.png"

    with plt.rc_context(_POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        data = [by_class[c] for c in classes]
        bp = ax.boxplot(data, positions=range(len(classes)), widths=0.5, patch_artist=True,
                         showfliers=False, zorder=2)
        for patch, c in zip(bp["boxes"], classes):
            patch.set_facecolor(_CLASS_COLORS.get(c, "gray"))
            patch.set_alpha(0.35)
        rng = np.random.default_rng(0)
        for i, c in enumerate(classes):
            vals = by_class[c]
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=_CLASS_COLORS.get(c, "gray"),
                       alpha=0.7, s=45, edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f"{c}\n(n={len(by_class[c])})" for c in classes])
        ax.set_ylabel("benign_plausibility (A5)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.3, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(0.7, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title("Agent pipeline's benign_plausibility, by true class")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    return {c: {"n": len(by_class[c]), "mean": statistics.mean(by_class[c]),
                "median": statistics.median(by_class[c]),
                "min": min(by_class[c]), "max": max(by_class[c])} for c in classes}


# ---------- Fig 3: detector comparison bar chart ----------


def build_fig3(rows: List[dict], manifest: Dict[str, str], out_dir: Path = FIGURES_DIR) -> dict:
    table_rows = abl.build_table(rows, [], manifest)
    table_rows = abl.add_combined_score(table_rows)
    classes = [c for c in abl.classes_present(table_rows) if c != "BENIGN"]

    detectors = [
        ("nn_dist alone", abl.sweep_high_side(table_rows, "nn_dist", abl.classes_present(table_rows)), "high"),
        ("close_to_observed_count alone",
         abl.sweep_low_side(table_rows, "max_close_to_observed_count", abl.classes_present(table_rows)), "low"),
        ("combined", abl.sweep_high_side(table_rows, "combined_score", abl.classes_present(table_rows)), "high"),
        ("full pipeline",
         abl.sweep_bp(table_rows, "benign_plausibility", abl.classes_present(table_rows)), "low"),
    ]
    # Primary operating point: 5% benign FPR -- this project's own
    # established default (the selector itself was fitted to "the
    # 5%-benign-rate operating point" throughout STATUS.md), not picked
    # after the fact for flattering numbers. 0% FPR is stricter but
    # reads as an empty DDoS group at n=69 (every detector, including
    # the full pipeline, hits 0% DDoS recall right at a zero-tolerance
    # bar) -- that's a real, reportable finding on its own, kept in the
    # returned dict and the presentation summary text, just not forced
    # into a chart where it looks like a rendering gap rather than a
    # result.
    PRIMARY_TARGET = 0.05
    at_primary, at_zero = {}, {}
    for name, sweep, direction in detectors:
        matched_p = abl.at_matched_benign_fpr(sweep, [PRIMARY_TARGET], abl.classes_present(table_rows), direction)[0]
        matched_0 = abl.at_matched_benign_fpr(sweep, [0.0], abl.classes_present(table_rows), direction)[0]
        at_primary[name] = {c: matched_p.get(c) for c in classes}
        at_zero[name] = {c: matched_0.get(c) for c in classes}

    n_benign = sum(1 for r in table_rows if r["label"] == "BENIGN")
    n_by_class = {c: sum(1 for r in table_rows if r["label"] == c) for c in classes}

    out_path = out_dir / "fig3_detector_comparison.png"
    detector_names = [d[0] for d in detectors]
    with plt.rc_context(_POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        x = np.arange(len(classes))
        width = 0.8 / len(detector_names)
        for i, name in enumerate(detector_names):
            y = [(at_primary[name].get(c) or 0.0) * 100 for c in classes]
            ax.bar(x + (i - (len(detector_names) - 1) / 2) * width, y, width, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(n={n_by_class[c]})" for c in classes])
        ax.set_ylabel(f"Recall at {PRIMARY_TARGET:.0%} benign FPR (%)")
        ax.set_title(f"Detector comparison at {PRIMARY_TARGET:.0%} benign false-positive rate (n_benign={n_benign})")
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", frameon=True, fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    return {
        "n_benign": n_benign, "n_by_class": n_by_class,
        "primary_target_fpr": PRIMARY_TARGET, "at_primary_fpr": at_primary, "at_zero_fpr": at_zero,
    }


# ---------- Fig 4: per-agent trust scores ----------


def plot_fig4(rows: List[dict], out_dir: Path = FIGURES_DIR) -> dict:
    agents = ("a1", "a2", "a3", "a4", "a5")
    components = ("C", "E", "V", "T")
    means: Dict[str, Dict[str, Optional[float]]] = {}
    for agent in agents:
        means[agent] = {}
        for comp in components:
            vals = [r["trust_scores"][agent][comp] for r in rows
                    if r.get("trust_scores", {}).get(agent, {}).get(comp) is not None]
            means[agent][comp] = statistics.mean(vals) if vals else None

    out_path = out_dir / "fig4_agent_trust_scores.png"
    with plt.rc_context(_POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        x = np.arange(len(agents))
        width = 0.8 / len(components)
        for i, comp in enumerate(components):
            y = [means[a][comp] if means[a][comp] is not None else 0.0 for a in agents]
            bars = ax.bar(x + (i - (len(components) - 1) / 2) * width, y, width, label=comp)
            for j, a in enumerate(agents):
                if means[a][comp] is None:
                    bars[j].set_hatch("//")
                    bars[j].set_alpha(0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in agents])
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Per-agent trust scores, code-computed (n={len(rows)} records)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", frameon=True, ncol=4)
        ax.text(0.5, -0.16, "hatched bars = undefined (A5 has no claims to check E/V against)",
                transform=ax.transAxes, ha="center", fontsize=11, color="dimgray")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    return means


# ---------- Fig 5: chain_vs_independent distribution ----------


def plot_fig5(rows: List[dict], out_dir: Path = FIGURES_DIR) -> dict:
    vals = [r["chain_vs_independent"] for r in rows if r.get("chain_vs_independent") is not None]
    out_path = out_dir / "fig5_chain_vs_independent.png"

    with plt.rc_context(_POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        ax.hist(vals, bins=20, color="#4C72B0", edgecolor="black", alpha=0.85)
        mean_v = statistics.mean(vals)
        ax.axvline(0, color="black", linewidth=1.5, linestyle="-")
        ax.axvline(mean_v, color="firebrick", linewidth=2.5, linestyle="--",
                   label=f"mean = {mean_v:+.3f}")
        ax.set_xlabel("chain_vs_independent  (T4 − mean(T1,T2,T3))")
        ax.set_ylabel("Number of records")
        n_pos = sum(1 for v in vals if v > 0)
        ax.set_title(
            f"Chain vs. independent replica (A4), n={len(vals)}\n"
            f"positive (chain underperforms) in {n_pos}/{len(vals)} ({n_pos/len(vals):.0%})",
            fontsize=17,
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    return {
        "n": len(vals), "mean": mean_v, "median": statistics.median(vals),
        "n_positive": n_pos, "frac_positive": n_pos / len(vals) if vals else None,
        "min": min(vals), "max": max(vals),
    }


# ---------- Fig 6: SELF-REPORTED trust vs agent stage ----------
#
# Only a fraction of agent_pipeline_200.jsonl carries the self-reported
# confidence/evidence_support/verification triad: it was added to the
# schema and every prompt mid-run (agents/schema.py, prompt versions
# bumped to a1_evidence_v5/a2_behaviour_v2/a3_hypotheses_v5/
# a4_replication_v5/a5_verdict_v5 -- see results/ARCHITECTURE.md section
# 5), so records processed before that bump simply don't have it, and
# records processed after do, all-or-nothing per record across all five
# agents (the cache key changed with the prompt version, so a record is
# either entirely pre-bump or entirely post-bump). This is NOT the
# multi-condition fault-injection figure from the original poster spec --
# eval/run_fault_injection.py has never been run (zero API calls spent on
# it), so there is only ever one line here: whatever's in
# agent_pipeline_200.jsonl is all "clean" (un-corrupted) operation.


def self_reported_rows(rows: List[dict]) -> List[dict]:
    """Rows where every one of A1-A5 has all three self-reported fields
    populated -- see module docstring above for why this is a strict
    subset, not "most of the file with a few gaps"."""
    fields = ("confidence", "evidence_support", "verification")
    out = []
    for r in rows:
        if all(r.get(a) and all(r[a].get(f) is not None for f in fields) for a in AGENTS):
            out.append(r)
    return out


def _self_reported_weighted(resp: dict) -> Optional[float]:
    wc, we, wv = DEFAULT_WEIGHTS
    c, e, v = resp.get("confidence"), resp.get("evidence_support"), resp.get("verification")
    if c is None or e is None or v is None:
        return None
    return wc * c + we * e + wv * v


def plot_fig6_self_reported_trust(rows: List[dict], manifest: Dict[str, str], out_dir: Path = FIGURES_DIR) -> dict:
    sr_rows = self_reported_rows(rows)
    out_path = out_dir / "fig6_self_reported_trust_by_stage.png"

    class_counts: Dict[str, int] = {}
    for r in sr_rows:
        label = manifest.get(r["record_id"], "UNKNOWN")
        class_counts[label] = class_counts.get(label, 0) + 1

    means: Dict[str, Optional[float]] = {}
    for agent in AGENTS:
        vals = [_self_reported_weighted(r[agent]) for r in sr_rows]
        vals = [v for v in vals if v is not None]
        means[agent] = statistics.mean(vals) if vals else None

    composition = ", ".join(f"{k}={v}" for k, v in sorted(class_counts.items(), key=lambda kv: -kv[1]))

    with plt.rc_context(_POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        x = list(range(len(AGENTS)))
        y = [means[a] for a in AGENTS]
        ax.plot(x, y, marker="o", linewidth=2.8, markersize=10, color="#4C72B0",
                label=f"n={len(sr_rows)} ({composition})")
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in AGENTS])
        ax.set_xlabel("Agent stage")
        ax.set_ylabel("Self-reported trust score (weighted C/E/V)")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Self-reported trust vs. agent stage (n={len(sr_rows)} of {len(rows)} total records)\n"
            f"only records processed after the self-report schema was added -- see caption",
            fontsize=16,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    return {"n": len(sr_rows), "n_total": len(rows), "class_composition": class_counts, "mean_by_agent": means}


# ---------- Fig 7: self-reported confidence-verification gap vs decision error ----------


def build_fig7_gap_vs_error(rows: List[dict], manifest: Dict[str, str], out_dir: Path = FIGURES_DIR, n_bins: int = 8) -> dict:
    """Decision error here means plain correctness against ground truth
    (derive_verdict(benign_plausibility) vs true label) -- NOT a
    clean-vs-fault-injected transition, because no fault-injection data
    exists yet (see Fig 6's docstring). With only one condition, "did the
    verdict change" isn't a question that has an answer; "was the verdict
    right" is the only decision-error signal available from this data."""
    sr_rows = self_reported_rows(rows)
    out_path = out_dir / "fig7_gap_vs_decision_error.png"

    points: List[Tuple[float, int]] = []
    for r in sr_rows:
        label = manifest.get(r["record_id"])
        if label is None:
            continue
        verdict = derive_verdict(r["a5"]["benign_plausibility"])
        consistent = verdict == VerdictLabel.CONSISTENT_WITH_BENIGN
        correct = consistent if label == "BENIGN" else not consistent
        error = 0 if correct else 1
        for agent in AGENTS:
            resp = r[agent]
            gap = resp["confidence"] - resp["verification"]
            points.append((gap, error))

    if not points:
        return {"n_points": 0, "n_records": len(sr_rows)}

    gaps = np.array([p[0] for p in points])
    errors = np.array([p[1] for p in points], dtype=float)
    lo, hi = float(gaps.min()), float(gaps.max())
    if lo == hi:
        edges = np.array([lo - 0.5, hi + 0.5])
        n_bins = 1
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(gaps, edges[1:-1], right=True), 0, n_bins - 1)

    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        mid = float((edges[b] + edges[b + 1]) / 2)
        bins.append({
            "bin": b, "gap_mid": mid, "n_points": n,
            "decision_error_rate": float(errors[mask].mean()) if n else None,
        })

    usable = [b for b in bins if b["decision_error_rate"] is not None and b["n_points"] > 0]
    fit = None
    if len(usable) >= 2:
        x = np.array([b["gap_mid"] for b in usable])
        y = np.array([b["decision_error_rate"] for b in usable])
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fit = {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n_bins_used": len(usable)}

    with plt.rc_context(_POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        x = [b["gap_mid"] for b in usable]
        y = [b["decision_error_rate"] for b in usable]
        sizes = [max(60, b["n_points"] * 15) for b in usable]
        ax.scatter(x, y, s=sizes, alpha=0.75, edgecolor="black", linewidth=0.5, zorder=3)
        if fit is not None:
            xs = np.linspace(min(x), max(x), 100)
            ys = fit["slope"] * xs + fit["intercept"]
            weak = fit["r2"] < 0.3
            ax.plot(xs, ys, "--" if weak else "-", color="firebrick", linewidth=2.5, zorder=2,
                     label=f"fit: y={fit['slope']:.2f}x+{fit['intercept']:.2f}, R²={fit['r2']:.2f}"
                           + ("  (weak fit)" if weak else ""))
            ax.legend(loc="best", frameon=True)
        ax.set_xlabel("Self-reported confidence − verification (gap)")
        ax.set_ylabel("Decision error rate (vs. ground truth)")
        n_records = len(sr_rows)
        ax.set_title(
            f"Overconfidence gap vs. decision error -- n={n_records} records "
            f"({sum(1 for r in sr_rows if manifest.get(r['record_id'])=='PortScan')} PortScan, "
            f"{sum(1 for r in sr_rows if manifest.get(r['record_id'])=='BENIGN')} BENIGN)\n"
            f"single-condition data (no fault injection run) -- error = wrong verdict, not a fault-induced flip",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    return {"n_records": len(sr_rows), "n_points": len(points), "bins": bins, "fit": fit}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", default=str(PIPELINE_JSONL),
                         help="agent-pipeline results JSONL (figs 2-5) -- the one path to change once "
                              "the full 200-record run lands; same file, so usually no change needed at all")
    parser.add_argument("--manifest", default=str(MANIFEST_JSONL))
    parser.add_argument("--recall-curve", default=str(RECALL_CURVE_CSV))
    parser.add_argument("--per-class-benign-rate", default=str(PER_CLASS_BY_BENIGN_RATE_CSV))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.manifest))
    rows = load_jsonl(Path(args.pipeline))
    print(f"loaded {len(rows)} pipeline records (of {len(manifest)} in manifest) from {args.pipeline}")

    results = {}
    print("fig1: recall vs escalation rate ...")
    results["fig1"] = plot_fig1(Path(args.recall_curve), Path(args.per_class_benign_rate), out_dir)
    print("fig2: benign_plausibility by class ...")
    results["fig2"] = plot_fig2(rows, manifest, out_dir)
    print("fig3: detector comparison ...")
    results["fig3"] = build_fig3(rows, manifest, out_dir)
    print("fig4: per-agent trust scores ...")
    results["fig4"] = plot_fig4(rows, out_dir)
    print("fig5: chain_vs_independent distribution ...")
    results["fig5"] = plot_fig5(rows, out_dir)
    print("fig6: self-reported trust vs agent stage ...")
    results["fig6"] = plot_fig6_self_reported_trust(rows, manifest, out_dir)
    print("fig7: self-reported gap vs decision error ...")
    results["fig7"] = build_fig7_gap_vs_error(rows, manifest, out_dir)

    with open(out_dir / "_presentation_figures_data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))
    print(f"\nwrote figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
