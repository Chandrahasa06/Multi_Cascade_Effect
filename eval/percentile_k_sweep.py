"""2D sweep over fitting percentile x k_of_n's k, hunting for an
operating point that reaches ~1% total escalation without giving up
attack recall.

eval/sweep.py's PERCENTILE_SWEEP tops out at p99.99 and k has always
been fixed at 2 (STATUS.md's own default, chosen for a different reason
-- union-bound blowup under `any`). Two live questions motivate the
2nd axis: (1) does raising k cut volume harder than tightening the
percentile, given ~27 correlated Tier-1/per-source comparisons where
k=2 often just means "two symptoms of the same underlying signal"; and
(2) does the Monday-to-Friday FPR gap (1.18% vs 7.79% at the current
default operating point) persist as the operating point tightens, which
would mean Monday-fitted thresholds don't generalize -- a fitting
problem, not a volume problem.

Per percentile, thresholds are fit ONCE (percentile is the only fitting
parameter -- k only changes how per-feature crossings combine into an
escalation decision), then every k is evaluated against the same
crossings via escalate_from_crossings. This also means "which features
are still firing" is a per-percentile property, not a per-(percentile,k)
one, and is reported that way (once per percentile) rather than
redundantly 4x per row.

Pure re-analysis of already-cached PCAP feature parquets -- no
simulation, no API calls. Same fitting discipline as
eval/run_pcap_sweep.py: Monday PCAP chronological fit/holdout split,
exclude_low_confidence=False.

Run: python -m eval.percentile_k_sweep
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dataplane.fitting import fit_thresholds_from_frame
from dataplane.selector import EscalationRule
from eval.run_pcap_sweep import load_friday_pcap, load_monday_pcap
from eval.sweep import (
    compute_crossings,
    escalate_from_crossings,
    split_monday_chronologically,
)

PERCENTILES = (99.5, 99.9, 99.95, 99.99, 99.995, 99.999)
KS = (2, 3, 4, 5)
ATTACK_LABELS = ("PortScan", "DDoS", "Bot")
COMBINED_TARGET = 0.01

OUT_CSV = Path("results/percentile_k_sweep.csv")
OUT_FIG = Path("results/figures/percentile_k_sweep.png")

_POSTER_RC = {
    "font.size": 15, "axes.titlesize": 19, "axes.labelsize": 17,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
}
_K_COLORS = {2: "#4C72B0", 3: "#DD8452", 4: "#55A868", 5: "#C44E52"}


def per_feature_crossing_rates(df: pd.DataFrame, crossings: pd.DataFrame) -> dict:
    if len(df) == 0 or crossings.shape[1] == 0:
        return {}
    rates = crossings.mean(axis=0)
    return {k: float(v) for k, v in rates.items() if v > 0}


def counts_by_label(df: pd.DataFrame, escalated: pd.Series, label_col: str = "label") -> dict:
    total = df[label_col].value_counts().to_dict()
    esc = df.loc[escalated, label_col].value_counts().to_dict()
    return {"total": {k: int(v) for k, v in total.items()}, "escalated": {k: int(esc.get(k, 0)) for k in total}}


def main() -> int:
    print("loading cached PCAP feature parquets ...")
    monday_df, _ = load_monday_pcap()
    friday_df, _ = load_friday_pcap()
    monday_fit, monday_holdout = split_monday_chronologically(monday_df)
    combined_holdout = pd.concat([monday_holdout, friday_df], ignore_index=True)
    print(f"Monday total={len(monday_df)}  fit_half={len(monday_fit)}  holdout_half={len(monday_holdout)}")
    print(f"Friday total={len(friday_df)}  by label: {friday_df['label'].value_counts().to_dict()}")
    print()

    rows = []  # flat table, one per (percentile, k)
    firing_by_pct = {}  # percentile -> {all, escalated_only(k=2 as reference), monday}

    for pct in PERCENTILES:
        fit = fit_thresholds_from_frame(
            monday_fit, percentile=pct, rule=EscalationRule.K_OF_N, k=2,
            exclude_low_confidence=False,
        )
        config = fit.config
        mon_crossings = compute_crossings(monday_holdout, config.thresholds)
        fri_crossings = compute_crossings(friday_df, config.thresholds)
        comb_crossings = compute_crossings(combined_holdout, config.thresholds)

        firing_by_pct[pct] = {
            "n_thresholds_fit": len(config.thresholds),
            "friday_all": per_feature_crossing_rates(friday_df, fri_crossings),
            "monday_holdout": per_feature_crossing_rates(monday_holdout, mon_crossings),
        }

        for k in KS:
            mon_esc = escalate_from_crossings(mon_crossings, EscalationRule.K_OF_N, k)
            fri_esc = escalate_from_crossings(fri_crossings, EscalationRule.K_OF_N, k)
            comb_esc = escalate_from_crossings(comb_crossings, EscalationRule.K_OF_N, k)

            mon_counts = counts_by_label(monday_holdout, mon_esc)
            fri_counts = counts_by_label(friday_df, fri_esc)
            comb_counts = counts_by_label(combined_holdout, comb_esc)

            mon_fpr = mon_counts["escalated"].get("BENIGN", 0) / mon_counts["total"].get("BENIGN", 1)
            fri_fpr = fri_counts["escalated"].get("BENIGN", 0) / fri_counts["total"].get("BENIGN", 1)
            fri_esc_rate = int(fri_esc.sum()) / len(friday_df) if len(friday_df) else None
            comb_esc_rate = int(comb_esc.sum()) / len(combined_holdout) if len(combined_holdout) else None

            recall = {
                l: (fri_counts["escalated"].get(l, 0) / fri_counts["total"][l]) if fri_counts["total"].get(l) else None
                for l in ATTACK_LABELS
            }
            attacks_escalated = sum(fri_counts["escalated"].get(l, 0) for l in ATTACK_LABELS)
            attacks_total = sum(fri_counts["total"].get(l, 0) for l in ATTACK_LABELS)
            pooled_recall = (attacks_escalated / attacks_total) if attacks_total else None

            fri_precision = (attacks_escalated / int(fri_esc.sum())) if int(fri_esc.sum()) else None
            comb_attacks_escalated = sum(comb_counts["escalated"].get(l, 0) for l in ATTACK_LABELS)
            comb_precision = (comb_attacks_escalated / int(comb_esc.sum())) if int(comb_esc.sum()) else None

            rows.append({
                "percentile": pct, "k": k, "config_hash": config.config_hash,
                "monday_fpr": mon_fpr, "friday_escalation_rate": fri_esc_rate, "friday_fpr": fri_fpr,
                "fpr_gap": fri_fpr - mon_fpr, "combined_escalation_rate": comb_esc_rate,
                "recall_portscan": recall.get("PortScan"), "recall_ddos": recall.get("DDoS"),
                "recall_bot": recall.get("Bot"), "pooled_attack_recall": pooled_recall,
                "friday_precision": fri_precision, "combined_precision": comb_precision,
                "friday_escalated_n": int(fri_esc.sum()), "combined_escalated_n": int(comb_esc.sum()),
                "combined_n": len(combined_holdout),
            })

    # ---- console table ----
    print("=" * 150)
    print("percentile x k grid")
    print("=" * 150)
    hdr = (f"{'pct':>8} {'k':>2} | {'mon_fpr':>8} {'fri_esc':>8} {'fri_fpr':>8} {'gap':>8} "
           f"{'comb_esc':>9} | {'PortScan':>9} {'DDoS':>7} {'Bot':>7} {'pooled':>7} | "
           f"{'fri_prec':>8} {'comb_prec':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def pct_or_dash(v):
            return f"{v:.2%}" if v is not None else "  n/a"
        print(f"{r['percentile']:>8} {r['k']:>2} | {pct_or_dash(r['monday_fpr']):>8} "
              f"{pct_or_dash(r['friday_escalation_rate']):>8} {pct_or_dash(r['friday_fpr']):>8} "
              f"{r['fpr_gap']*100:>+7.2f}% {pct_or_dash(r['combined_escalation_rate']):>9} | "
              f"{pct_or_dash(r['recall_portscan']):>9} {pct_or_dash(r['recall_ddos']):>7} "
              f"{pct_or_dash(r['recall_bot']):>7} {pct_or_dash(r['pooled_attack_recall']):>7} | "
              f"{pct_or_dash(r['friday_precision']):>8} {pct_or_dash(r['combined_precision']):>9}")
    print()

    # ---- firing features per percentile ----
    print("=" * 150)
    print("Features still firing, per percentile (k-independent -- k only changes the combine rule)")
    print("=" * 150)
    for pct in PERCENTILES:
        info = firing_by_pct[pct]
        print(f"\np{pct}  (n_thresholds_fit={info['n_thresholds_fit']}):")
        print("  Friday (all flows), crossing rate desc:")
        for name, rate in sorted(info["friday_all"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:35s} {rate:.4%}")
        if not info["friday_all"]:
            print("    (none)")
        print("  Monday holdout (false-alarm side), crossing rate desc:")
        for name, rate in sorted(info["monday_holdout"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:35s} {rate:.4%}")
        if not info["monday_holdout"]:
            print("    (none)")

    # ---- cells reaching <=1% combined escalation ----
    print()
    print("=" * 150)
    print(f"Cells with combined escalation rate <= {COMBINED_TARGET:.0%}")
    print("=" * 150)
    under_target = [r for r in rows if r["combined_escalation_rate"] is not None and r["combined_escalation_rate"] <= COMBINED_TARGET]
    if not under_target:
        print(f"  none of the {len(rows)} cells reach <= {COMBINED_TARGET:.0%} combined escalation.")
    else:
        under_target_sorted = sorted(under_target, key=lambda r: -(r["pooled_attack_recall"] or -1))
        for r in under_target_sorted:
            print(f"  p{r['percentile']} k={r['k']}: combined_esc={r['combined_escalation_rate']:.3%}  "
                  f"pooled_recall={r['pooled_attack_recall']:.1%}  "
                  f"PortScan={r['recall_portscan']:.1%} DDoS={r['recall_ddos']:.1%} Bot={r['recall_bot']:.1%}  "
                  f"fpr_gap={r['fpr_gap']*100:+.2f}pp")
        best = under_target_sorted[0]
        print()
        print(f"BEST under target: p{best['percentile']} k={best['k']}")
        print(f"  combined escalation: {best['combined_escalation_rate']:.3%} "
              f"({best['combined_escalated_n']:,}/{best['combined_n']:,})")
        print(f"  Monday holdout FPR: {best['monday_fpr']:.3%}   Friday FPR: {best['friday_fpr']:.3%}   "
              f"gap: {best['fpr_gap']*100:+.2f}pp")
        print(f"  Friday escalation rate: {best['friday_escalation_rate']:.3%}")
        print(f"  recall -- PortScan={best['recall_portscan']:.1%}  DDoS={best['recall_ddos']:.1%}  "
              f"Bot={best['recall_bot']:.1%}  pooled={best['pooled_attack_recall']:.1%}")
        print(f"  precision -- Friday={best['friday_precision']:.1%}  combined={best['combined_precision']:.1%}")

    # ---- write CSV ----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT_CSV}")

    # ---- plot: x=combined escalation rate, y=pooled attack recall, one line per k ----
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_POSTER_RC):
        fig, ax = plt.subplots(figsize=(11, 7.5))
        for k in KS:
            kr = sorted((r for r in rows if r["k"] == k), key=lambda r: r["combined_escalation_rate"])
            xs = [r["combined_escalation_rate"] * 100 for r in kr]
            ys = [r["pooled_attack_recall"] * 100 for r in kr]
            ax.plot(xs, ys, marker="o", linewidth=2.8, markersize=8,
                     label=f"k={k}", color=_K_COLORS[k], zorder=5)
            for r, x, y in zip(kr, xs, ys):
                ax.annotate(f"p{r['percentile']}", (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=9, alpha=0.75)
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
        ax.text(1.0, ax.get_ylim()[0] if False else 2, " 1% target", color="gray", fontsize=12, rotation=90, va="bottom")
        ax.set_xlabel("Combined escalation rate (%) -- Monday holdout + Friday")
        ax.set_ylabel("Pooled attack recall (%) -- PortScan+DDoS+Bot, Friday")
        ax.set_title("Escalation rate vs. attack recall, by k_of_n's k (percentile x k sweep)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True, title="k_of_n")
        fig.tight_layout()
        fig.savefig(OUT_FIG, dpi=300)
        plt.close(fig)
    print(f"wrote {OUT_FIG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
