"""Zero-cost ablation over the existing v4 agent-pipeline results
(results/agent_pipeline_20_v4.jsonl) -- no new API calls.

Question: does nn_dist (mechanical nearest-benign-flow distance,
controlplane/reference.py) separate classes about as well as the full
five-agent pipeline's benign_plausibility? If so, the agents aren't
where the detection signal comes from.

Run: python -m eval.ablation_v4
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

RECORDS_PATH = Path("results/agent_pipeline_20_v4.jsonl")
MANIFEST_PATH = Path("results/escalation_records_sample_manifest.INTERNAL.jsonl")


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["flow_id"]] = row["label"]
    return out


def load_records(path: Path = RECORDS_PATH) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_table(records: List[dict], manifest: Dict[str, str]) -> List[dict]:
    rows = []
    for r in records:
        fid = r["record_id"]
        label = manifest.get(fid, "UNKNOWN")
        nn = r["nearest_neighbours"]
        nn_dist = nn.get("nearest_distance")
        all_supports = list(r["hypothesis_support"].values())
        coc_values = [s["close_to_observed_count"] for s in all_supports if s["close_to_observed_count"] is not None]
        max_coc = max(coc_values) if coc_values else None
        bp = r["a5"]["benign_plausibility"]
        cred = r["a5"]["credited_hypothesis_id"]
        cred_support = r["hypothesis_support"].get(cred) if cred else None
        rows.append({
            "flow_id": fid,
            "label": label,
            "nn_dist": nn_dist,
            "max_close_to_observed_count": max_coc,
            "credited_close_to_observed_count": cred_support["close_to_observed_count"] if cred_support else None,
            "benign_plausibility": bp,
            "chain_vs_independent": r["chain_vs_independent"],
            "a2_V": r["trust_scores"]["a2"]["V"],
            "a4_V": r["trust_scores"]["a4"]["V"],
        })
    return rows


def sweep_high_side(rows: List[dict], key: str, classes: List[str]) -> List[dict]:
    """Threshold sweep for a 'higher = more anomalous' feature (nn_dist).
    escalate (flag as attack) if value > threshold."""
    values = sorted({r[key] for r in rows if r[key] is not None})
    thresholds = [v - 1e-9 for v in values] + ([values[-1] + 1e-9] if values else [])
    out = []
    for t in thresholds:
        row = {"threshold": t}
        for cls in classes:
            members = [r for r in rows if r["label"] == cls and r[key] is not None]
            if not members:
                row[cls] = None
                continue
            flagged = sum(1 for r in members if r[key] > t)
            row[cls] = flagged / len(members)
        out.append(row)
    return out


def sweep_low_side(rows: List[dict], key: str, classes: List[str]) -> List[dict]:
    """Threshold sweep for a 'lower = more anomalous' feature
    (close_to_observed_count). escalate if value <= threshold. None is
    treated as maximally anomalous (no benign explanation found close
    to this flow at all) -- always flagged regardless of threshold."""
    values = sorted({r[key] for r in rows if r[key] is not None})
    thresholds = [-1] + values
    out = []
    for t in thresholds:
        row = {"threshold": t}
        for cls in classes:
            members = [r for r in rows if r["label"] == cls]
            if not members:
                row[cls] = None
                continue
            flagged = sum(1 for r in members if r[key] is None or r[key] <= t)
            row[cls] = flagged / len(members)
        out.append(row)
    return out


def sweep_bp(rows: List[dict], classes: List[str]) -> List[dict]:
    """Agent pipeline: escalate (flag as attack) if benign_plausibility < threshold."""
    values = sorted({r["benign_plausibility"] for r in rows})
    thresholds = values + [values[-1] + 1e-9]
    out = []
    for t in thresholds:
        row = {"threshold": t}
        for cls in classes:
            members = [r for r in rows if r["label"] == cls]
            flagged = sum(1 for r in members if r["benign_plausibility"] < t)
            row[cls] = flagged / len(members)
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


def combined_score(rows: List[dict]) -> List[dict]:
    """Simple, transparent combination: z-score nn_dist (high=anomalous)
    and z-score of -max_close_to_observed_count (low count=anomalous,
    so negate before z-scoring so high=anomalous consistently), sum the
    two. Missing max_close_to_observed_count (no hypothesis had any
    checkable feature) is imputed at the worst (most anomalous) observed
    value for that side before z-scoring, matching sweep_low_side's
    treatment of None."""
    nn_vals = [r["nn_dist"] for r in rows]
    coc_raw = [r["max_close_to_observed_count"] for r in rows]
    worst = min((v for v in coc_raw if v is not None), default=0)
    neg_coc = [-(v if v is not None else worst) for v in coc_raw]
    nn_z = zscore(nn_vals)
    coc_z = zscore(neg_coc)
    out = []
    for r, a, b in zip(rows, nn_z, coc_z):
        r2 = dict(r)
        r2["combined_score"] = (a or 0) + (b or 0)
        out.append(r2)
    return out


def sweep_combined(rows: List[dict], classes: List[str]) -> List[dict]:
    values = sorted({r["combined_score"] for r in rows})
    thresholds = [values[0] - 1] + values
    out = []
    for t in thresholds:
        row = {"threshold": t}
        for cls in classes:
            members = [r for r in rows if r["label"] == cls]
            flagged = sum(1 for r in members if r["combined_score"] > t)
            row[cls] = flagged / len(members)
        out.append(row)
    return out


def render_sweep_table(sweep: List[dict], classes: List[str]) -> str:
    header = "| threshold | " + " | ".join(classes) + " |"
    sep = "|---" * (len(classes) + 1) + "|"
    lines = [header, sep]
    for row in sweep:
        vals = " | ".join(
            (f"{row[c]*100:.0f}%" if row[c] is not None else "-") for c in classes
        )
        lines.append(f"| {row['threshold']:.4g} | {vals} |")
    return "\n".join(lines)


def main() -> None:
    manifest = load_manifest()
    records = load_records()
    rows = build_table(records, manifest)
    classes = ["BENIGN", "Bot", "DDoS", "PortScan"]

    print(f"n={len(rows)} records loaded from {RECORDS_PATH}")
    print()

    print("=" * 80)
    print("1. nn_dist alone vs benign_plausibility (full pipeline)")
    print("=" * 80)
    print("nn_dist sweep (escalate if nn_dist > threshold):")
    print(render_sweep_table(sweep_high_side(rows, "nn_dist", classes), classes))
    print()
    print("benign_plausibility sweep (escalate if bp < threshold):")
    print(render_sweep_table(sweep_bp(rows, classes), classes))
    print()

    print("=" * 80)
    print("2. close_to_observed_count alone, and nn_dist + close_to_observed_count combined")
    print("=" * 80)
    print("close_to_observed_count sweep (escalate if count <= threshold; None treated as always-flagged):")
    print(render_sweep_table(sweep_low_side(rows, "max_close_to_observed_count", classes), classes))
    print()
    rows_c = combined_score(rows)
    print("combined (z(nn_dist) + z(-max_close_to_observed_count)) sweep (escalate if score > threshold):")
    print(render_sweep_table(sweep_combined(rows_c, classes), classes))
    print()

    print("=" * 80)
    print("3. Per-record disagreement: nn_dist-implied anomaly vs agents' benign_plausibility")
    print("=" * 80)
    print(f"{'flow_id':14s} {'label':9s} {'nn_dist':>9s} {'bp':>5s}  disagreement")
    dist_says_anom_agents_say_benign = 0
    agents_say_anom_dist_says_benign = 0
    both_agree = 0
    for r in sorted(rows, key=lambda r: r["nn_dist"] if r["nn_dist"] is not None else -1):
        dist_anom = (r["nn_dist"] or 0) > 1.0  # roughly the DDoS/PortScan floor, see below
        agent_anom = r["benign_plausibility"] < 0.5
        if dist_anom and not agent_anom:
            tag = "DISTANCE=anomalous, AGENTS=benign"
            dist_says_anom_agents_say_benign += 1
        elif agent_anom and not dist_anom:
            tag = "AGENTS=anomalous, DISTANCE=benign"
            agents_say_anom_dist_says_benign += 1
        else:
            tag = ""
            both_agree += 1
        if tag:
            print(f"{r['flow_id'][:12]:14s} {r['label']:9s} {r['nn_dist']:9.3f} {r['benign_plausibility']:5.2f}  {tag}")
    print()
    print(f"distance-anomalous / agents-benign: {dist_says_anom_agents_say_benign}")
    print(f"agents-anomalous / distance-benign: {agents_say_anom_dist_says_benign}")
    print(f"both agree: {both_agree}")
    print()

    print("=" * 80)
    print("4. chain_vs_independent (T4 - mean(T1,T2,T3)) for this run")
    print("=" * 80)
    cvis = [r["chain_vs_independent"] for r in rows if r["chain_vs_independent"] is not None]
    print(f"n with defined chain_vs_independent: {len(cvis)}/{len(rows)}")
    if cvis:
        print(f"mean: {sum(cvis)/len(cvis):+.4f}")
        print(f"min:  {min(cvis):+.4f}   max: {max(cvis):+.4f}")
        n_positive = sum(1 for v in cvis if v > 0)
        print(f"positive (chain underperforms A4's blind pass) in {n_positive}/{len(cvis)} records")
    print()
    print("per-record:")
    for r in rows:
        cvi = r["chain_vs_independent"]
        if cvi is not None:
            print(f"  {r['flow_id'][:12]} {r['label']:9s} chain_vs_independent={cvi:+.4f}")
        else:
            print(f"  {r['flow_id'][:12]} {r['label']:9s} chain_vs_independent=undefined")
    print()
    a2_Vs = [r["a2_V"] for r in rows if r["a2_V"] is not None]
    if a2_Vs:
        print(f"A2's V across records: mean={sum(a2_Vs)/len(a2_Vs):.3f} (n={len(a2_Vs)}), "
              f"min={min(a2_Vs):.3f}, max={max(a2_Vs):.3f}")


if __name__ == "__main__":
    main()
