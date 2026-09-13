"""Poster figures for the trust-propagation fault-injection study.

Reads the combined JSONL eval/run_fault_injection.py writes (one line
per (record, condition), same RecordResult shape as the main pipeline
plus a "condition" field) and produces:

  - Fig 1: trust score vs agent stage (A1..A5), one line per condition.
  - Fig 2: grouped bars, x = which agent was degraded (A1..A4), three
    bars each (decision error / false escalation / missed detection).
  - Fig 3: scatter of confidence-verification gap vs decision error
    rate (binned), with a fitted line.
  - Fig 4 (bonus, not asked for in the three-figure spec but flagged as
    likely warranted by Part 1's self-report-vs-code-computed
    comparison): self-reported vs code-computed evidence_support/
    verification, per agent.

Every figure's underlying numbers are also written to CSV in the same
output directory, so they can be replotted without re-running anything.

"Primary" trust signal is self-reported (confidence, evidence_support,
verification -- agents/schema.py), per the professor's design; the
code-computed agents/trust.py scores are reported alongside as a
validation check, and used as a graceful fallback wherever self-report
is missing (e.g. if ever run against pre-schema-change data).

Run (once eval/run_fault_injection.py has produced real data)::

    python -m eval.poster_figures --input results/fault_injection_20.jsonl

Against placeholder data for a dry run (no fault-injection data exists
yet) -- see tests/test_poster_figures.py, which builds a small
synthetic multi-condition dataset out of the existing v4 20-record
clean results (agents/fault_injection.py's own pure corruption
functions applied to real cached v4 claims, purely to exercise this
module's plotting/statistics code path -- NOT a real fault-injection
result; v4 predates the self-reported schema and has only one
condition).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents.schema import VerdictLabel, derive_verdict
from agents.trust import DEFAULT_WEIGHTS
from eval import fault_injection as fi

DEFAULT_INPUT = "results/fault_injection_20.jsonl"
DEFAULT_MANIFEST = "results/escalation_records_sample_manifest.INTERNAL.v1_50each.jsonl"
DEFAULT_FIGURES_DIR = "results/figures"

AGENTS = ("a1", "a2", "a3", "a4", "a5")
AGENT_LABELS = {"a1": "A1", "a2": "A2", "a3": "A3", "a4": "A4", "a5": "A5"}

CONDITION_ORDER = ("clean", *fi.ALL_CONDITIONS)
CONDITION_LABELS = {
    "clean": "Clean",
    fi.MISSING_EVIDENCE: "Missing evidence",
    fi.INCORRECT_BEHAVIOR: "Incorrect behavior",
    fi.HALLUCINATED_HYPOTHESIS: "Hallucinated hypothesis",
    fi.FAULTY_VERIFICATION: "Faulty verification",
}
#: which agent a condition's injection degrades -- Fig 2's x-axis.
CONDITION_TO_DEGRADED_AGENT = {
    fi.MISSING_EVIDENCE: "A1",
    fi.INCORRECT_BEHAVIOR: "A2",
    fi.HALLUCINATED_HYPOTHESIS: "A3",
    fi.FAULTY_VERIFICATION: "A4",
}

#: poster-readable defaults -- a laptop-screen figure this is not.
_POSTER_RCPARAMS = {
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 100,  # on-screen; savefig dpi is set separately at 300
}


# ---------- loading ----------


def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_manifest(path: str) -> Dict[str, str]:
    return {row["flow_id"]: row["label"] for row in load_jsonl(path)}


# ---------- per-row derived quantities ----------


@dataclass
class AgentTrust:
    self_reported: Optional[float]  # weighted combination of self-reported C/E/V (DEFAULT_WEIGHTS)
    code_computed: Optional[float]  # trust_scores[agent]["T"]
    self_confidence: Optional[float]
    self_evidence_support: Optional[float]
    self_verification: Optional[float]
    code_evidence_support: Optional[float]
    code_verification: Optional[float]


def _weighted(c: Optional[float], e: Optional[float], v: Optional[float]) -> Optional[float]:
    if c is None or e is None or v is None:
        return None
    wc, we, wv = DEFAULT_WEIGHTS
    return wc * c + we * e + wv * v


def agent_trust(row: dict, agent: str) -> AgentTrust:
    resp = row.get(agent, {}) or {}
    self_c = resp.get("confidence")
    self_e = resp.get("evidence_support")
    self_v = resp.get("verification")
    ts = (row.get("trust_scores") or {}).get(agent, {}) or {}
    code_t = ts.get("T")
    code_e = ts.get("E")
    code_v = ts.get("V")
    return AgentTrust(
        self_reported=_weighted(self_c, self_e, self_v),
        code_computed=code_t,
        self_confidence=self_c,
        self_evidence_support=self_e,
        self_verification=self_v,
        code_evidence_support=code_e,
        code_verification=code_v,
    )


def derived_verdict_for_row(row: dict) -> VerdictLabel:
    bp = row["a5"]["benign_plausibility"]
    return derive_verdict(bp)


def is_correct(verdict: VerdictLabel, true_label: str) -> bool:
    consistent = verdict == VerdictLabel.CONSISTENT_WITH_BENIGN
    return consistent if true_label == "BENIGN" else not consistent


# ---------- Fig 1: trust score vs agent stage, one line per condition ----------


def build_fig1_data(rows: List[dict]) -> List[dict]:
    """One row per (condition, agent): mean self-reported and
    code-computed trust across every record in that condition."""
    out = []
    for condition in CONDITION_ORDER:
        condition_rows = [r for r in rows if r["condition"] == condition]
        if not condition_rows:
            continue
        for agent in AGENTS:
            trusts = [agent_trust(r, agent) for r in condition_rows]
            self_vals = [t.self_reported for t in trusts if t.self_reported is not None]
            code_vals = [t.code_computed for t in trusts if t.code_computed is not None]
            out.append({
                "condition": condition,
                "agent": AGENT_LABELS[agent],
                "self_reported_trust": float(np.mean(self_vals)) if self_vals else None,
                "code_computed_trust": float(np.mean(code_vals)) if code_vals else None,
                "n": len(condition_rows),
                "n_with_self_report": len(self_vals),
            })
    return out


def plot_fig1(fig1_data: List[dict], out_path: Path) -> None:
    with plt.rc_context(_POSTER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        x = list(range(len(AGENTS)))
        for condition in CONDITION_ORDER:
            pts = [d for d in fig1_data if d["condition"] == condition]
            if not pts:
                continue
            by_agent = {d["agent"]: d["self_reported_trust"] for d in pts}
            y = [by_agent.get(AGENT_LABELS[a]) for a in AGENTS]
            if all(v is None for v in y):
                continue
            ax.plot(
                x, y, marker="o", linewidth=2.5, markersize=9,
                label=CONDITION_LABELS.get(condition, condition),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABELS[a] for a in AGENTS])
        ax.set_xlabel("Agent stage")
        ax.set_ylabel("Self-reported trust score (weighted C/E/V)")
        ax.set_title("Trust propagation across the agent chain, by fault condition")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


# ---------- Fig 2: decision error / false escalation / missed detection, by degraded agent ----------


def build_fig2_data(rows: List[dict], manifest: Dict[str, str]) -> List[dict]:
    clean_by_id = {r["record_id"]: r for r in rows if r["condition"] == "clean"}
    out = []
    for condition, agent_label in CONDITION_TO_DEGRADED_AGENT.items():
        condition_rows = [r for r in rows if r["condition"] == condition]
        benign_total = benign_false_escalations = 0
        attack_total = attack_missed = 0
        decision_errors = 0
        n_paired = 0
        for r in condition_rows:
            clean_row = clean_by_id.get(r["record_id"])
            label = manifest.get(r["record_id"])
            if clean_row is None or label is None:
                continue
            n_paired += 1
            clean_verdict = derived_verdict_for_row(clean_row)
            fault_verdict = derived_verdict_for_row(r)
            clean_correct = is_correct(clean_verdict, label)
            fault_correct = is_correct(fault_verdict, label)
            if clean_correct and not fault_correct:
                decision_errors += 1

            clean_benign = clean_verdict == VerdictLabel.CONSISTENT_WITH_BENIGN
            fault_benign = fault_verdict == VerdictLabel.CONSISTENT_WITH_BENIGN
            if label == "BENIGN":
                benign_total += 1
                if clean_benign and not fault_benign:
                    benign_false_escalations += 1
            else:
                attack_total += 1
                if not clean_benign and fault_benign:
                    attack_missed += 1

        out.append({
            "degraded_agent": agent_label,
            "condition": condition,
            "n_paired_records": n_paired,
            "decision_error_rate": (decision_errors / n_paired) if n_paired else None,
            "false_escalation_rate": (benign_false_escalations / benign_total) if benign_total else None,
            "missed_detection_rate": (attack_missed / attack_total) if attack_total else None,
        })
    return out


def plot_fig2(fig2_data: List[dict], out_path: Path) -> None:
    metrics = [
        ("decision_error_rate", "Decision error rate"),
        ("false_escalation_rate", "False escalation rate"),
        ("missed_detection_rate", "Missed detection rate"),
    ]
    agents_order = [CONDITION_TO_DEGRADED_AGENT[c] for c in fi.ALL_CONDITIONS]
    by_agent = {d["degraded_agent"]: d for d in fig2_data}

    with plt.rc_context(_POSTER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        x = np.arange(len(agents_order))
        width = 0.25
        for i, (key, label) in enumerate(metrics):
            y = [((by_agent.get(a) or {}).get(key) or 0.0) for a in agents_order]
            ax.bar(x + (i - 1) * width, y, width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(agents_order)
        ax.set_xlabel("Degraded agent")
        ax.set_ylabel("Rate")
        ax.set_title("Failure modes by degraded agent")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


# ---------- Fig 3: confidence-verification gap vs decision error rate ----------


def build_fig3_points(rows: List[dict], manifest: Dict[str, str]) -> List[Tuple[float, int]]:
    """One (gap, decision_error) point per (record, agent, condition),
    pooling clean and all fault-injected conditions. decision_error is
    shared across all 5 agents of the same (record, condition) -- it's
    a property of that record+condition's final verdict, not of any
    one agent; the per-agent gap is what varies and supplies the x-axis
    resolution."""
    clean_by_id = {r["record_id"]: r for r in rows if r["condition"] == "clean"}
    points: List[Tuple[float, int]] = []
    for r in rows:
        label = manifest.get(r["record_id"])
        clean_row = clean_by_id.get(r["record_id"])
        if label is None or clean_row is None:
            continue
        if r["condition"] == "clean":
            decision_error = 0  # no transition possible against itself
        else:
            clean_correct = is_correct(derived_verdict_for_row(clean_row), label)
            fault_correct = is_correct(derived_verdict_for_row(r), label)
            decision_error = int(clean_correct and not fault_correct)

        for agent in AGENTS:
            resp = r.get(agent, {}) or {}
            c, v = resp.get("confidence"), resp.get("verification")
            if c is None or v is None:
                continue
            points.append((c - v, decision_error))
    return points


def bin_fig3_points(points: List[Tuple[float, int]], n_bins: int = 9) -> List[dict]:
    if not points:
        return []
    gaps = np.array([p[0] for p in points])
    errors = np.array([p[1] for p in points], dtype=float)
    lo, hi = float(gaps.min()), float(gaps.max())
    if lo == hi:
        edges = np.array([lo - 0.5, hi + 0.5])
        n_bins = 1
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(gaps, edges[1:-1], right=True), 0, n_bins - 1)

    out = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        mid = float((edges[b] + edges[b + 1]) / 2)
        out.append({
            "bin": b,
            "gap_lo": float(edges[b]),
            "gap_hi": float(edges[b + 1]),
            "gap_mid": mid,
            "n_points": n,
            "decision_error_rate": float(errors[mask].mean()) if n else None,
        })
    return out


def fit_linear(bins: List[dict]) -> Optional[dict]:
    usable = [b for b in bins if b["decision_error_rate"] is not None and b["n_points"] > 0]
    if len(usable) < 2:
        return None
    x = np.array([b["gap_mid"] for b in usable])
    y = np.array([b["decision_error_rate"] for b in usable])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n_bins_used": len(usable)}


def plot_fig3(bins: List[dict], fit: Optional[dict], out_path: Path) -> None:
    usable = [b for b in bins if b["decision_error_rate"] is not None]
    with plt.rc_context(_POSTER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        x = [b["gap_mid"] for b in usable]
        y = [b["decision_error_rate"] for b in usable]
        sizes = [max(40, b["n_points"] * 8) for b in usable]
        ax.scatter(x, y, s=sizes, alpha=0.75, edgecolor="black", linewidth=0.5, zorder=3)

        if fit is not None:
            xs = np.linspace(min(x), max(x), 100) if x else np.array([])
            ys = fit["slope"] * xs + fit["intercept"]
            weak = fit["r2"] < 0.3
            style = "--" if weak else "-"
            ax.plot(xs, ys, style, color="firebrick", linewidth=2.5, zorder=2,
                     label=f"fit: y={fit['slope']:.2f}x+{fit['intercept']:.2f}, R²={fit['r2']:.2f}"
                           + ("  (weak linear fit)" if weak else ""))
            ax.legend(loc="best", frameon=True)

        ax.set_xlabel("Confidence − verification (self-reported gap)")
        ax.set_ylabel("Decision error rate")
        ax.set_title("Overconfidence gap vs. decision error rate")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


# ---------- Part 1 / Fig 4 (bonus): self-reported vs code-computed ----------


def build_divergence_data(rows: List[dict]) -> List[dict]:
    out = []
    for agent in AGENTS:
        pairs_e, pairs_v = [], []
        for r in rows:
            t = agent_trust(r, agent)
            if t.self_evidence_support is not None and t.code_evidence_support is not None:
                pairs_e.append((t.self_evidence_support, t.code_evidence_support))
            if t.self_verification is not None and t.code_verification is not None:
                pairs_v.append((t.self_verification, t.code_verification))
        for metric_name, pairs in (("evidence_support", pairs_e), ("verification", pairs_v)):
            if len(pairs) >= 2:
                self_vals = np.array([p[0] for p in pairs])
                code_vals = np.array([p[1] for p in pairs])
                corr = float(np.corrcoef(self_vals, code_vals)[0, 1]) if np.std(self_vals) and np.std(code_vals) else None
                mad = float(np.mean(np.abs(self_vals - code_vals)))
            else:
                corr = mad = None
            out.append({
                "agent": AGENT_LABELS[agent], "metric": metric_name,
                "n": len(pairs), "correlation": corr, "mean_abs_diff": mad,
            })
    return out


def plot_fig4(rows: List[dict], out_path: Path) -> None:
    with plt.rc_context(_POSTER_RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for ax, metric, self_attr, code_attr in (
            (axes[0], "evidence_support", "self_evidence_support", "code_evidence_support"),
            (axes[1], "verification", "self_verification", "code_verification"),
        ):
            for agent in AGENTS:
                xs, ys = [], []
                for r in rows:
                    t = agent_trust(r, agent)
                    sv, cv = getattr(t, self_attr), getattr(t, code_attr)
                    if sv is not None and cv is not None:
                        xs.append(sv)
                        ys.append(cv)
                if xs:
                    ax.scatter(xs, ys, label=AGENT_LABELS[agent], alpha=0.6, s=50)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.5)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel(f"self-reported {metric}")
            ax.set_ylabel(f"code-computed {metric[0].upper()}")
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=11)
        fig.suptitle("Self-reported vs. code-computed trust components")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


# ---------- CSV writing ----------


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------- main ----------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--bins", type=int, default=9)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.input)
    manifest = load_manifest(args.manifest)
    print(f"loaded {len(rows)} (record, condition) rows from {args.input}")
    by_condition = {}
    for r in rows:
        by_condition[r["condition"]] = by_condition.get(r["condition"], 0) + 1
    print(f"by condition: {by_condition}")

    fig1_data = build_fig1_data(rows)
    write_csv(fig1_data, out_dir / "fig1_trust_by_stage.csv")
    plot_fig1(fig1_data, out_dir / "fig1_trust_by_stage.png")

    fig2_data = build_fig2_data(rows, manifest)
    write_csv(fig2_data, out_dir / "fig2_failure_modes.csv")
    plot_fig2(fig2_data, out_dir / "fig2_failure_modes.png")

    fig3_points = build_fig3_points(rows, manifest)
    fig3_bins = bin_fig3_points(fig3_points, n_bins=args.bins)
    fit = fit_linear(fig3_bins)
    write_csv(fig3_bins, out_dir / "fig3_gap_vs_error.csv")
    if fit is not None:
        fit_path = out_dir / "fig3_fit.csv"
        write_csv([fit], fit_path)
        print(f"fig3 linear fit: slope={fit['slope']:.4f} intercept={fit['intercept']:.4f} "
              f"R2={fit['r2']:.4f} (n_bins_used={fit['n_bins_used']})")
        if fit["r2"] < 0.3:
            print("[note] R2 is low -- the gap/error relationship may not be well described by a "
                  "straight line; the figure marks the fit as weak rather than hiding this.")
    else:
        print("[warn] not enough populated bins to fit a line for fig3")
    plot_fig3(fig3_bins, fit, out_dir / "fig3_gap_vs_error.png")

    divergence = build_divergence_data(rows)
    write_csv(divergence, out_dir / "fig4_self_vs_code_divergence.csv")
    plot_fig4(rows, out_dir / "fig4_self_vs_code.png")
    print("self-report vs code-computed divergence (Part 1):")
    for d in divergence:
        if d["n"]:
            print(f"  {d['agent']} {d['metric']}: n={d['n']} corr={d['correlation']!s:>6} "
                  f"mean_abs_diff={d['mean_abs_diff']:.3f}" if d["mean_abs_diff"] is not None
                  else f"  {d['agent']} {d['metric']}: n={d['n']} (insufficient data)")

    print(f"wrote figures + CSVs to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
