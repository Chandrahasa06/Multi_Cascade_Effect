"""Ablation report at scale: nn_dist alone, close_to_observed_count
alone, combined, single-LLM baseline, and the full five-agent pipeline,
compared head to head at n=200 (or however much of it has completed so
far -- every input file this reads is append-only and safe to read
mid-run).

Generalizes eval/ablation_v4.py (n=20, 5/class) to:
  - an arbitrary, unequal per-class stratification (70/50/50/30)
  - the single-LLM baseline (results/baseline_200.jsonl), not available
    at v4 time
  - a full operating-point sweep per class, not just the 0%-FPR point
  - per-agent trust and chain_vs_independent broken down by true class

No new API calls -- pure re-analysis of already-written JSONL.

Run: python -m eval.ablation_v200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_PIPELINE = Path("results/agent_pipeline_200.jsonl")
DEFAULT_BASELINE = Path("results/baseline_200.jsonl")
DEFAULT_MANIFEST = Path("results/escalation_records_sample_manifest.INTERNAL.jsonl")

AGENTS = ("a1", "a2", "a3", "a4", "a5")


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


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Dict[str, str]:
    out = {}
    for row in load_jsonl(path):
        out[row["flow_id"]] = row["label"]
    return out


def build_table(
    pipeline_records: List[dict], baseline_records: List[dict], manifest: Dict[str, str]
) -> List[dict]:
    baseline_by_id = {r["record_id"]: r for r in baseline_records}
    rows = []
    for r in pipeline_records:
        fid = r["record_id"]
        label = manifest.get(fid, "UNKNOWN")
        nn = r["nearest_neighbours"]
        nn_dist = nn.get("nearest_distance")
        all_supports = list(r["hypothesis_support"].values())
        coc_values = [
            s["close_to_observed_count"] for s in all_supports if s["close_to_observed_count"] is not None
        ]
        max_coc = max(coc_values) if coc_values else None
        bp = r["a5"]["benign_plausibility"]

        baseline_row = baseline_by_id.get(fid)
        baseline_bp = baseline_row["response"]["benign_plausibility"] if baseline_row else None

        row = {
            "flow_id": fid,
            "label": label,
            "nn_dist": nn_dist,
            "max_close_to_observed_count": max_coc,
            "benign_plausibility": bp,
            "baseline_benign_plausibility": baseline_bp,
            "chain_vs_independent": r["chain_vs_independent"],
        }
        for agent in AGENTS:
            ts = r["trust_scores"].get(agent, {})
            for comp in ("C", "E", "V", "T"):
                row[f"{agent}_{comp}"] = ts.get(comp)
        rows.append(row)
    return rows


def classes_present(rows: List[dict]) -> List[str]:
    return sorted({r["label"] for r in rows if r["label"] != "UNKNOWN"})


def sweep_high_side(rows: List[dict], key: str, classes: Sequence[str]) -> List[dict]:
    """escalate if value > threshold (nn_dist, combined_score)."""
    values = sorted({r[key] for r in rows if r[key] is not None})
    thresholds = ([v - 1e-9 for v in values] + [values[-1] + 1e-9]) if values else []
    out = []
    for t in thresholds:
        row = {"threshold": t}
        for cls in classes:
            members = [r for r in rows if r["label"] == cls and r[key] is not None]
            row[cls] = (sum(1 for r in members if r[key] > t) / len(members)) if members else None
        out.append(row)
    return out


def sweep_low_side(rows: List[dict], key: str, classes: Sequence[str]) -> List[dict]:
    """escalate if value <= threshold; None treated as always-flagged
    (close_to_observed_count)."""
    values = sorted({r[key] for r in rows if r[key] is not None})
    thresholds = [-1] + values
    out = []
    for t in thresholds:
        row = {"threshold": t}
        for cls in classes:
            members = [r for r in rows if r["label"] == cls]
            row[cls] = (
                sum(1 for r in members if r[key] is None or r[key] <= t) / len(members)
            ) if members else None
        out.append(row)
    return out


def sweep_bp(rows: List[dict], key: str, classes: Sequence[str]) -> List[dict]:
    """escalate if bp < threshold (full pipeline or baseline)."""
    values = sorted({r[key] for r in rows if r[key] is not None})
    if not values:
        return []
    thresholds = values + [values[-1] + 1e-9]
    out = []
    for t in thresholds:
        row = {"threshold": t}
        for cls in classes:
            members = [r for r in rows if r["label"] == cls and r[key] is not None]
            row[cls] = (sum(1 for r in members if r[key] < t) / len(members)) if members else None
        out.append(row)
    return out


def zscore(values: List[Optional[float]]) -> List[Optional[float]]:
    present = [v for v in values if v is not None]
    if not present:
        return values
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / len(present)
    std = var ** 0.5 or 1.0
    return [None if v is None else (v - mean) / std for v in values]


def add_combined_score(rows: List[dict]) -> List[dict]:
    nn_vals = [r["nn_dist"] for r in rows]
    coc_raw = [r["max_close_to_observed_count"] for r in rows]
    worst = min((v for v in coc_raw if v is not None), default=0)
    neg_coc = [-(v if v is not None else worst) for v in coc_raw]
    nn_z = zscore(nn_vals)
    coc_z = zscore(neg_coc)
    for r, a, b in zip(rows, nn_z, coc_z):
        r["combined_score"] = (a or 0) + (b or 0)
    return rows


def at_matched_benign_fpr(
    sweep: List[dict], targets: Sequence[float], classes: Sequence[str], direction: str = "high"
) -> List[dict]:
    """For each target BENIGN FPR, the *most sensitive* sweep row whose
    BENIGN rate still satisfies the target -- the highest-recall point
    available within the false-positive budget (mirrors STATUS.md's
    existing 'anchored on benign_escalation_rate' convention).

    ``direction`` MUST match how the sweep's escalation rule moves with
    its own ascending threshold list, or this silently picks the wrong
    end and reports 0% recall everywhere -- a real bug caught while
    building this exact figure (see PRESENTATION_SUMMARY.md):

    - "high" (sweep_high_side: nn_dist, combined -- escalate if
      value > threshold): BENIGN's rate is NON-INCREASING as threshold
      rises (a higher cutoff flags fewer records), so the most sensitive
      point satisfying the budget is the SMALLEST qualifying threshold
      -> candidates[0].
    - "low" (sweep_low_side: close_to_observed_count -- escalate if
      value <= threshold -- and sweep_bp: full pipeline / baseline --
      escalate if bp < threshold): BENIGN's rate is NON-DECREASING as
      threshold rises (a higher cutoff flags MORE records), so the most
      sensitive point satisfying the budget is the LARGEST qualifying
      threshold -> candidates[-1]. Using candidates[0] here picks the
      smallest threshold in the whole sweep, which is exactly the
      dataset's own minimum observed value -- at that point NOTHING is
      strictly less than it, so both BENIGN and every attack class read
      0% simultaneously, regardless of what recall was actually
      achievable at a real 0%-FPR operating point.
    """
    if direction not in ("high", "low"):
        raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")
    out = []
    for target in targets:
        candidates = [row for row in sweep if row.get("BENIGN") is not None and row["BENIGN"] <= target + 1e-9]
        if candidates:
            chosen = candidates[0] if direction == "high" else candidates[-1]
        else:
            chosen = sweep[0] if sweep else {"threshold": None}
        out.append({"target_benign_fpr": target, **{k: v for k, v in chosen.items() if k != "threshold"}})
    return out


def render_matched_table(matched: List[dict], classes: Sequence[str]) -> str:
    header = "| target benign FPR | " + " | ".join(classes) + " |"
    sep = "|---" * (len(classes) + 1) + "|"
    lines = [header, sep]
    for row in matched:
        vals = " | ".join((f"{row[c]*100:.0f}%" if row.get(c) is not None else "-") for c in classes)
        lines.append(f"| {row['target_benign_fpr']*100:.0f}% | {vals} |")
    return "\n".join(lines)


def disagreement_report(rows: List[dict], nn_threshold: float, bp_threshold: float = 0.5) -> dict:
    fp_corrected = []  # distance flags, truth is BENIGN, agents call it benign -- correct override
    induced_misses = []  # distance flags, truth is an attack, agents call it benign -- wrong override
    reverse = []  # agents flag anomalous, distance doesn't
    for r in rows:
        if r["nn_dist"] is None:
            continue
        dist_anom = r["nn_dist"] > nn_threshold
        agent_anom = r["benign_plausibility"] < bp_threshold
        if dist_anom and not agent_anom:
            if r["label"] == "BENIGN":
                fp_corrected.append(r["flow_id"])
            else:
                induced_misses.append(r["flow_id"])
        elif agent_anom and not dist_anom:
            reverse.append(r["flow_id"])
    return {
        "nn_threshold": nn_threshold,
        "bp_threshold": bp_threshold,
        "fp_corrected": fp_corrected,
        "induced_misses": induced_misses,
        "agents_anomalous_distance_benign": reverse,
    }


def zero_fpr_threshold(sweep: List[dict]) -> Optional[float]:
    """Smallest swept threshold at which BENIGN's rate is exactly 0 (or
    the smallest observed if none hit exactly 0)."""
    zero_rows = [row for row in sweep if row.get("BENIGN") == 0.0]
    if not zero_rows:
        return None
    return min(row["threshold"] for row in zero_rows)


def trust_by_class(rows: List[dict], classes: Sequence[str]) -> str:
    lines = []
    for cls in classes:
        members = [r for r in rows if r["label"] == cls]
        if not members:
            continue
        lines.append(f"  {cls} (n={len(members)}):")
        for agent in AGENTS:
            comps = {}
            for comp in ("C", "E", "V", "T"):
                vals = [r[f"{agent}_{comp}"] for r in members if r[f"{agent}_{comp}"] is not None]
                comps[comp] = (sum(vals) / len(vals)) if vals else None
            comp_str = "  ".join(
                f"{c}={comps[c]:.3f}" if comps[c] is not None else f"{c}=  -" for c in ("C", "E", "V", "T")
            )
            lines.append(f"    {agent}: {comp_str}")
        cvis = [r["chain_vs_independent"] for r in members if r["chain_vs_independent"] is not None]
        if cvis:
            lines.append(
                f"    chain_vs_independent: mean={sum(cvis)/len(cvis):+.4f} "
                f"(n={len(cvis)}, positive in {sum(1 for v in cvis if v > 0)}/{len(cvis)})"
            )
    return "\n".join(lines)


def render_sweep_table(sweep: List[dict], classes: Sequence[str]) -> str:
    header = "| threshold | " + " | ".join(classes) + " |"
    sep = "|---" * (len(classes) + 1) + "|"
    lines = [header, sep]
    for row in sweep:
        vals = " | ".join((f"{row[c]*100:.0f}%" if row.get(c) is not None else "-") for c in classes)
        lines.append(f"| {row['threshold']:.4g} | {vals} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", default=str(DEFAULT_PIPELINE))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--full-sweeps", action="store_true", help="also print the full per-threshold tables")
    args = parser.parse_args(argv)

    manifest = load_manifest(Path(args.manifest))
    pipeline_records = load_jsonl(Path(args.pipeline))
    baseline_records = load_jsonl(Path(args.baseline))
    rows = build_table(pipeline_records, baseline_records, manifest)
    classes = classes_present(rows)
    n_baseline = sum(1 for r in rows if r["baseline_benign_plausibility"] is not None)

    print(f"pipeline records loaded: {len(rows)} (of manifest total {len(manifest)})")
    print(f"  by class: " + ", ".join(f"{c}={sum(1 for r in rows if r['label']==c)}" for c in classes))
    print(f"baseline records loaded: {n_baseline}")
    print()

    if not rows:
        print("no pipeline records yet -- nothing to report.")
        return 0

    rows = add_combined_score(rows)

    detectors = [
        ("nn_dist alone", sweep_high_side(rows, "nn_dist", classes), "high"),
        ("close_to_observed_count alone", sweep_low_side(rows, "max_close_to_observed_count", classes), "low"),
        ("combined (nn_dist + close_to_observed_count)", sweep_high_side(rows, "combined_score", classes), "high"),
        ("full agent pipeline (benign_plausibility)", sweep_bp(rows, "benign_plausibility", classes), "low"),
    ]
    if n_baseline:
        detectors.append((
            "single-LLM baseline (benign_plausibility)",
            sweep_bp(rows, "baseline_benign_plausibility", classes), "low",
        ))

    n_benign = sum(1 for r in rows if r["label"] == "BENIGN")
    step = (1.0 / n_benign) if n_benign else 0.05
    # 0.114 is threshold_sweep.py's own Youden's-J-optimal operating point
    # (T=0.275, benign_fpr=8/70=11.43%) -- included explicitly so every
    # detector here gets compared at the exact same real operating point
    # that number is quoted from, not just the nearest coarser grid step.
    matched_target = round(round(0.114 * n_benign) / n_benign, 6) if n_benign else 0.114
    targets = sorted({0.0, round(step, 4), round(2 * step, 4), 0.05, 0.10, matched_target, 0.20})

    print("=" * 90)
    print(f"Per-class recall at matched benign FPR (n_benign={n_benign}, so the finest achievable step is {step:.2%})")
    print("=" * 90)
    for name, sweep, direction in detectors:
        print(f"\n{name}:")
        print(render_matched_table(at_matched_benign_fpr(sweep, targets, classes, direction), classes))
        if args.full_sweeps:
            print("\nfull sweep:")
            print(render_sweep_table(sweep, classes))

    print()
    print("=" * 90)
    print("Disagreement: distance-implied anomaly vs agents' benign_plausibility")
    print("=" * 90)
    nn_sweep = [d for n, d, _dir in detectors if n == "nn_dist alone"][0]
    nn_t = zero_fpr_threshold(nn_sweep)
    if nn_t is None:
        print("no 0%-benign-FPR threshold found for nn_dist in this data yet.")
    else:
        report = disagreement_report(rows, nn_t, bp_threshold=0.5)
        print(f"nn_dist threshold used (smallest value achieving 0% benign FPR): {nn_t:.4g}")
        print(f"distance-anomalous / agents-benign, and truth=BENIGN (agents correctly filtered a FP): "
              f"{len(report['fp_corrected'])}  {report['fp_corrected']}")
        print(f"distance-anomalous / agents-benign, and truth=attack (agents induced a miss): "
              f"{len(report['induced_misses'])}  {report['induced_misses']}")
        print(f"agents-anomalous / distance-benign (reverse direction): "
              f"{len(report['agents_anomalous_distance_benign'])}  {report['agents_anomalous_distance_benign']}")
        n_fp, n_miss = len(report['fp_corrected']), len(report['induced_misses'])
        if n_fp + n_miss:
            print(f"ratio: {n_fp}/{n_fp+n_miss} = {n_fp/(n_fp+n_miss):.1%} of overrides were correct FP corrections")

    print()
    print("=" * 90)
    print("chain_vs_independent and per-agent trust, by true class")
    print("=" * 90)
    print(trust_by_class(rows, classes))

    return 0


if __name__ == "__main__":
    main()
