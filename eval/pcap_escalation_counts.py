"""Exact escalation counts from the PCAP-path sweep (Friday + Monday
held-out), at the 1%/5%/10% held-out-benign-rate-anchored operating
points. Pure re-analysis of already-cached PCAP feature parquets
(results/cache/*_pcap__fidelity__adapter2__features4.parquet) -- no
simulation, no API calls. Same fitting discipline as
eval/run_pcap_sweep.py (Monday PCAP chronological fit/holdout split,
k_of_n(k=2), exclude_low_confidence=False).

Run: python -m eval.pcap_escalation_counts
"""
from __future__ import annotations

import json

import pandas as pd

from dataplane.fitting import fit_thresholds_from_frame
from dataplane.selector import EscalationRule
from eval.run_pcap_sweep import load_friday_pcap, load_monday_pcap
from eval.sweep import (
    PERCENTILE_SWEEP,
    compute_crossings,
    escalate_from_crossings,
    split_monday_chronologically,
)

TARGETS = (0.01, 0.05, 0.10)
LABELS = ("PortScan", "DDoS", "Bot", "BENIGN")


def fit_at_benign_rate(monday_fit: pd.DataFrame, monday_holdout: pd.DataFrame, target: float):
    """Sweep percentiles, fit on monday_fit, measure benign_escalation_rate
    on monday_holdout, return the (percentile, config, measured_rate) whose
    measured rate is closest to target."""
    best = None
    for pct in PERCENTILE_SWEEP:
        fit = fit_thresholds_from_frame(
            monday_fit, percentile=pct, rule=EscalationRule.K_OF_N, k=2,
            exclude_low_confidence=False,
        )
        crossings = compute_crossings(monday_holdout, fit.config.thresholds)
        escalated = escalate_from_crossings(crossings, EscalationRule.K_OF_N, 2)
        rate = float(escalated.mean())
        dist = abs(rate - target)
        if best is None or dist < best[0]:
            best = (dist, pct, fit.config, rate)
    _, pct, config, rate = best
    return pct, config, rate


def counts_for(df: pd.DataFrame, config, label_col: str = "label") -> dict:
    crossings = compute_crossings(df, config.thresholds)
    escalated = escalate_from_crossings(crossings, EscalationRule.K_OF_N, 2)
    total = len(df)
    total_escalated = int(escalated.sum())

    by_label_total = df[label_col].value_counts().to_dict()
    by_label_escalated = df.loc[escalated, label_col].value_counts().to_dict()

    return {
        "total_flows": total,
        "total_escalated": total_escalated,
        "by_label_total": {k: int(v) for k, v in by_label_total.items()},
        "by_label_escalated": {k: int(by_label_escalated.get(k, 0)) for k in by_label_total},
    }


def main() -> int:
    print("loading cached PCAP feature parquets ...")
    monday_df, _ = load_monday_pcap()
    friday_df, _ = load_friday_pcap()
    monday_fit, monday_holdout = split_monday_chronologically(monday_df)
    print(f"Monday total={len(monday_df)}  fit_half={len(monday_fit)}  holdout_half={len(monday_holdout)}")
    print(f"Friday total={len(friday_df)}  by label: {friday_df['label'].value_counts().to_dict()}")
    print()

    report = {}
    for target in TARGETS:
        pct, config, measured_rate = fit_at_benign_rate(monday_fit, monday_holdout, target)
        print(f"=== target benign rate {target:.0%} -> percentile {pct:.4f}, "
              f"measured held-out rate {measured_rate:.4%}, config_hash={config.config_hash} ===")

        friday = counts_for(friday_df, config)
        monday_h = counts_for(monday_holdout, config)

        attack_labels = [l for l in LABELS if l != "BENIGN"]
        friday_attacks_escalated = sum(friday["by_label_escalated"].get(l, 0) for l in attack_labels)
        friday_attacks_total = sum(friday["by_label_total"].get(l, 0) for l in attack_labels)
        friday_benign_escalated = friday["by_label_escalated"].get("BENIGN", 0)
        precision = friday_attacks_escalated / friday["total_escalated"] if friday["total_escalated"] else None
        missed_by_label = {
            l: friday["by_label_total"].get(l, 0) - friday["by_label_escalated"].get(l, 0)
            for l in attack_labels
        }

        print(f"  Friday: total={friday['total_flows']:,}  escalated={friday['total_escalated']:,}")
        for l in LABELS:
            tot = friday["by_label_total"].get(l, 0)
            esc = friday["by_label_escalated"].get(l, 0)
            print(f"    {l:10s} total={tot:>7,}  escalated={esc:>6,}  "
                  f"({esc/tot:.2%} of that class)" if tot else f"    {l:10s} total=0")
        print(f"  Friday precision (attacks / all escalated): "
              f"{friday_attacks_escalated:,}/{friday['total_escalated']:,} = "
              f"{precision:.2%}" if precision is not None else "  Friday precision: n/a (0 escalated)")
        print(f"  Friday attacks MISSED (not escalated): "
              f"{friday_attacks_total - friday_attacks_escalated:,} of {friday_attacks_total:,} total attacks")
        for l, m in missed_by_label.items():
            tot = friday["by_label_total"].get(l, 0)
            print(f"    {l:10s} missed={m:>6,} of {tot:,}")
        print()
        print(f"  Monday holdout: total={monday_h['total_flows']:,}  "
              f"escalated={monday_h['total_escalated']:,}  "
              f"(all false alarms, benign_escalation_rate={monday_h['total_escalated']/monday_h['total_flows']:.4%})")
        print()

        combined_total = friday["total_flows"] + monday_h["total_flows"]
        combined_escalated = friday["total_escalated"] + monday_h["total_escalated"]
        combined_attacks_escalated = friday_attacks_escalated
        combined_attacks_total = friday_attacks_total
        combined_benign_escalated = friday_benign_escalated + monday_h["total_escalated"]
        combined_benign_total = friday["by_label_total"].get("BENIGN", 0) + monday_h["total_flows"]
        combined_precision = combined_attacks_escalated / combined_escalated if combined_escalated else None
        print(f"  COMBINED (Friday + Monday holdout): total={combined_total:,}  "
              f"escalated={combined_escalated:,}")
        print(f"    attacks escalated={combined_attacks_escalated:,} of {combined_attacks_total:,} total attacks "
              f"({combined_attacks_escalated/combined_attacks_total:.2%} recall)")
        print(f"    benign escalated={combined_benign_escalated:,} of {combined_benign_total:,} total benign "
              f"({combined_benign_escalated/combined_benign_total:.4%} FPR)")
        print(f"    combined precision (attacks/all escalated): "
              f"{combined_precision:.2%}" if combined_precision is not None else "    n/a")
        print("=" * 100)
        print()

        report[f"{target:.0%}"] = {
            "percentile": pct, "config_hash": config.config_hash, "measured_monday_holdout_rate": measured_rate,
            "friday": friday, "monday_holdout": monday_h,
            "friday_precision": precision, "friday_attacks_missed": friday_attacks_total - friday_attacks_escalated,
            "friday_attacks_total": friday_attacks_total,
            "combined": {
                "total_flows": combined_total, "total_escalated": combined_escalated,
                "attacks_escalated": combined_attacks_escalated, "attacks_total": combined_attacks_total,
                "benign_escalated": combined_benign_escalated, "benign_total": combined_benign_total,
                "precision": combined_precision,
            },
        }

    with open("results/pcap_escalation_counts.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print("wrote results/pcap_escalation_counts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
