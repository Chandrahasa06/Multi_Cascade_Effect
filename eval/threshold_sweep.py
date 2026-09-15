"""Sweep benign_plausibility's verdict threshold(s) over whatever is in
results/agent_pipeline_200.jsonl -- no API calls, pure re-analysis.

derive_verdict (agents/schema.py) uses two thresholds:
  - LOW  (default 0.3): bp < LOW  -> ANOMALOUS_AND_UNEXPLAINED (flagged)
  - HIGH (default 0.7): bp >= HIGH -> CONSISTENT_WITH_BENIGN (cleared)
  - in between: ANOMALOUS_BUT_EXPLICABLE (neither flagged nor cleared)

This sweeps a single cut T over every observed bp value and reports, for
each T:
  - per-class detection rate: P(bp < T | true class) for each attack class
  - benign FPR: P(bp < T | BENIGN)
  - Youden's J = pooled-attack detection rate - benign FPR, the standard
    single-number separation criterion for a binary cut -- maximised at
    the empirically best "flag as anomalous" operating point.

Then evaluates the CURRENT 0.3/0.7 defaults against that same sweep,
directly, rather than asserting they're fine.

Run: python -m eval.threshold_sweep
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PIPELINE_JSONL = Path("results/agent_pipeline_200.jsonl")
MANIFEST_JSONL = Path("results/escalation_records_sample_manifest.INTERNAL.jsonl")

LOW_DEFAULT = 0.3
HIGH_DEFAULT = 0.7


def load_jsonl(path: Path) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    manifest = {r["flow_id"]: r["label"] for r in load_jsonl(MANIFEST_JSONL)}
    rows = load_jsonl(PIPELINE_JSONL)

    by_class: Dict[str, List[float]] = {}
    for r in rows:
        label = manifest.get(r["record_id"])
        if label is None:
            continue
        by_class.setdefault(label, []).append(r["a5"]["benign_plausibility"])

    n_by_class = {c: len(v) for c, v in by_class.items()}
    print(f"n={len(rows)} records loaded, by class: {n_by_class}")
    attack_classes = [c for c in by_class if c != "BENIGN"]
    for c in ("Bot",):
        if c not in by_class or not by_class[c]:
            print(f"  [note] {c}: n=0 in this data -- cannot report a detection rate for it, not silently omitted")
    print()

    benign_vals = by_class.get("BENIGN", [])
    all_attack_vals = [v for c in attack_classes for v in by_class[c]]

    thresholds = sorted(set(v for vals in by_class.values() for v in vals))
    # candidate cuts: midpoints between consecutive observed values, plus
    # the extremes -- avoids ties landing exactly on an observed value,
    # which would make "<" ambiguous about which side a real record falls.
    cuts = [thresholds[0] - 0.05] + [
        (thresholds[i] + thresholds[i + 1]) / 2 for i in range(len(thresholds) - 1)
    ] + [thresholds[-1] + 0.05]

    def rate_below(vals: List[float], t: float) -> float:
        return sum(1 for v in vals if v < t) / len(vals) if vals else float("nan")

    print("=" * 100)
    print("Full sweep: escalate (\"flagged anomalous\") if benign_plausibility < T")
    print("=" * 100)
    header = f"{'T':>7} | " + " | ".join(f"{c:>9}" for c in attack_classes) + f" | {'BENIGN FPR':>10} | {'Youden J':>9}"
    print(header)
    print("-" * len(header))

    best = None  # (J, T, per_class_rates, benign_fpr)
    rows_out = []
    for t in cuts:
        per_class = {c: rate_below(by_class[c], t) for c in attack_classes}
        benign_fpr = rate_below(benign_vals, t)
        pooled_attack_rate = rate_below(all_attack_vals, t)
        J = pooled_attack_rate - benign_fpr
        rows_out.append({"T": t, **per_class, "benign_fpr": benign_fpr, "pooled_attack_rate": pooled_attack_rate, "J": J})
        line = f"{t:7.3f} | " + " | ".join(f"{per_class[c]*100:8.1f}%" for c in attack_classes) + \
               f" | {benign_fpr*100:9.1f}% | {J:+8.3f}"
        print(line)
        if best is None or J > best[0]:
            best = (J, t, dict(per_class), benign_fpr, pooled_attack_rate)

    print()
    print("=" * 100)
    print("Optimal single cut (maximises Youden's J = pooled-attack detection rate - benign FPR)")
    print("=" * 100)
    J, t_star, per_class_star, benign_fpr_star, pooled_star = best
    print(f"T* = {t_star:.3f}   J = {J:+.3f}")
    print(f"  pooled attack detection rate: {pooled_star:.1%}  (n={len(all_attack_vals)}, classes: {attack_classes})")
    for c in attack_classes:
        print(f"    {c:10s} detection rate: {per_class_star[c]:.1%}  (n={n_by_class[c]})")
    print(f"  benign FPR at T*: {benign_fpr_star:.1%}  (n={n_by_class.get('BENIGN', 0)})")

    # ties: report the full plateau, not just the first found
    tie_ts = [r["T"] for r in rows_out if abs(r["J"] - J) < 1e-9]
    if len(tie_ts) > 1:
        print(f"  [note] J is tied (within float precision) across T in [{min(tie_ts):.3f}, {max(tie_ts):.3f}] "
              f"({len(tie_ts)} candidate cuts) -- the sweep grid is coarse (only as many distinct cuts as "
              f"distinct observed values), report the plateau, not a spuriously precise single T*")

    print()
    print("=" * 100)
    print("The current defaults (LOW=0.3, HIGH=0.7) evaluated directly against this same sweep")
    print("=" * 100)
    low_row = min(rows_out, key=lambda r: abs(r["T"] - LOW_DEFAULT))
    high_row = min(rows_out, key=lambda r: abs(r["T"] - HIGH_DEFAULT))
    print(f"At T=LOW=0.3 (\"flag as anomalous\" cut, ANOMALOUS_AND_UNEXPLAINED boundary):")
    for c in attack_classes:
        print(f"    {c:10s} detection rate: {rate_below(by_class[c], LOW_DEFAULT):.1%}")
    print(f"    benign FPR: {rate_below(benign_vals, LOW_DEFAULT):.1%}")
    print(f"    pooled attack detection rate: {rate_below(all_attack_vals, LOW_DEFAULT):.1%}")
    print(f"    J at T=0.3: {rate_below(all_attack_vals, LOW_DEFAULT) - rate_below(benign_vals, LOW_DEFAULT):+.3f}")
    print(f"    (best possible J in this sweep: {J:+.3f} at T={t_star:.3f})")
    print()
    print(f"At T=HIGH=0.7 (\"cleared as benign\" boundary -- bp>=0.7 means CONSISTENT_WITH_BENIGN, "
          f"so bp<0.7 means \"not cleared\", a MUCH more aggressive cut than LOW):")
    for c in attack_classes:
        print(f"    {c:10s} \"not cleared\" rate: {rate_below(by_class[c], HIGH_DEFAULT):.1%}")
    print(f"    benign \"not cleared\" (false-non-clear) rate: {rate_below(benign_vals, HIGH_DEFAULT):.1%}")
    print(f"    pooled attack \"not cleared\" rate: {rate_below(all_attack_vals, HIGH_DEFAULT):.1%}")
    print(f"    J at T=0.7: {rate_below(all_attack_vals, HIGH_DEFAULT) - rate_below(benign_vals, HIGH_DEFAULT):+.3f}")

    with open("results/threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_by_class": n_by_class, "sweep": rows_out,
            "optimal": {"T": t_star, "J": J, "per_class": per_class_star, "benign_fpr": benign_fpr_star,
                        "pooled_attack_rate": pooled_star, "tie_range": [min(tie_ts), max(tie_ts)] if len(tie_ts) > 1 else None},
        }, f, indent=2, default=str)
    print()
    print("wrote results/threshold_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
