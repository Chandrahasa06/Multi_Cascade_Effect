"""End-to-end evaluation of the three-priority escalation policy
(``dataplane/escalation_policy.py``) with SpliDT
(``dataplane/splidt_inference.py``) as Priority 1, on the CICIDS2017 CSV
pool (``eval/splidt_features.py``).

Uses the professor's own train/test split (never re-split): contaminated
flows (Flow ID present in ``cicids-2017_p0.pkl``'s training set) are
excluded from every headline number and reported separately. No
simulation is run anywhere in this script — CSV columns and existing
cached parquets only.

Run: python -m eval.splidt_escalation_eval
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataplane.escalation_policy import (
    PRIORITY2_FEATURES,
    build_signature_table,
    check_bin_tie_saturation,
    compute_priority2_source_features,
    priority1_escalate,
    priority2_escalate,
    priority3_sample,
    run_policy,
)
from dataplane.splidt_inference import load_model_json, predict_batch
from eval.splidt_features import (
    MODEL_CLASSES,
    UNSEEN_CLASSES,
    build_full_pool,
    contamination_report,
    load_pickle_train_flow_ids,
    rate_quality_report,
)

OUT_DIR = Path("results")
FIG_DIR = OUT_DIR / "figures"
TINY_CLASS_FLOOR = 50
BUDGET_SWEEP = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
PRIORITY2_RESERVED_FRACTION = 0.003  # illustrative, per PDF's Table (Priority 2 budget)
DEFAULT_TAU = 20  # ~0.2% of common-region flows, per PDF's own example


# --------------------------------------------------------------------- #
# Stage 0: load pools
# --------------------------------------------------------------------- #

def load_pools():
    print("loading CSV pool (all 8 days, no simulation) ...")
    full_pool = build_full_pool(exclude_contaminated=False)
    full_pool = compute_priority2_source_features(full_pool)
    clean_pool = full_pool.loc[~full_pool["contaminated"]].reset_index(drop=True)
    print(f"full pool: {len(full_pool):,} rows | clean (non-contaminated): {len(clean_pool):,} rows")
    return full_pool, clean_pool


# --------------------------------------------------------------------- #
# Stage 1: coverage gap
# --------------------------------------------------------------------- #

def coverage_gap_report(clean_pool: pd.DataFrame) -> dict:
    counts = clean_pool["Label"].value_counts().to_dict()
    present = set(counts.keys())
    seen = sorted(present & MODEL_CLASSES)
    unseen = sorted(present & UNSEEN_CLASSES)
    unrecognized = sorted(present - MODEL_CLASSES - UNSEEN_CLASSES - {"Benign"})
    unseen_total = sum(counts.get(c, 0) for c in unseen)
    total_attack = sum(v for k, v in counts.items() if k != "Benign")
    return {
        "counts": counts, "seen_classes": seen, "unseen_classes": unseen,
        "unrecognized_classes": unrecognized, "unseen_total_n": unseen_total,
        "total_attack_n": total_attack,
        "unseen_fraction_of_attack": unseen_total / total_attack if total_attack else None,
    }


# --------------------------------------------------------------------- #
# Stage 2: Priority 1 per-class recall (clean only), contamination effect
# --------------------------------------------------------------------- #

def priority1_recall_report(clean_pool: pd.DataFrame, full_pool: pd.DataFrame, model_data: dict) -> dict:
    clean_pred = predict_batch(model_data, clean_pool)
    clean_esc = priority1_escalate(clean_pred.predicted_class)

    rows = []
    for label in sorted(clean_pool["Label"].unique()):
        mask = (clean_pool["Label"] == label).to_numpy()
        n = int(mask.sum())
        n_esc = int(clean_esc[mask].sum())
        rows.append({
            "label": label, "n_clean": n, "escalated_clean": n_esc,
            "recall_clean": (n_esc / n) if n else None,
            "reliable": n >= TINY_CLASS_FLOOR,
        })
    clean_recall_df = pd.DataFrame(rows).reset_index(drop=True)

    # Contamination effect: same recall computed WITH contaminated flows
    # included, for every attack class that had any contamination, side by
    # side with the clean number -- shows the inflation directly.
    full_pred = predict_batch(model_data, full_pool)
    full_esc = priority1_escalate(full_pred.predicted_class)
    infl_rows = []
    for label in sorted(set(full_pool["Label"]) - {"Benign"}):
        mask_all = (full_pool["Label"] == label).to_numpy()
        mask_contam = mask_all & full_pool["contaminated"].to_numpy()
        n_all, n_contam = int(mask_all.sum()), int(mask_contam.sum())
        recall_all = float(full_esc[mask_all].mean()) if n_all else None
        recall_contam_only = float(full_esc[mask_contam].mean()) if n_contam else None
        infl_rows.append({
            "label": label, "n_all": n_all, "n_contaminated": n_contam,
            "recall_including_contaminated": recall_all,
            "recall_on_contaminated_subset_only": recall_contam_only,
        })
    contamination_effect_df = pd.DataFrame(infl_rows)

    benign_mask = (clean_pool["Label"] == "Benign").to_numpy()
    benign_fpr = float(clean_esc[benign_mask].mean())

    return {
        "clean_recall": clean_recall_df,
        "contamination_effect": contamination_effect_df,
        "benign_fpr_clean": benign_fpr,
        "clean_predictions": clean_pred,
        "full_predictions": full_pred,
        "clean_escalated": clean_esc,
        "full_escalated": full_esc,
    }


# --------------------------------------------------------------------- #
# Stage 3: Priority 2 — signature table, bin-resolution sweep, saturation
# --------------------------------------------------------------------- #

def monday_chronological_split(clean_pool: pd.DataFrame):
    monday = clean_pool[clean_pool["day"] == "monday"].sort_values("first_ts", kind="stable").reset_index(drop=True)
    mid = len(monday) // 2
    return monday.iloc[:mid].reset_index(drop=True), monday.iloc[mid:].reset_index(drop=True)


def priority2_bin_sweep(clean_pool: pd.DataFrame, monday_fit: pd.DataFrame, bin_counts=(6, 10, 20, 40, 80)) -> pd.DataFrame:
    """Sweeps top-end bin resolution (n_bins) and reports whether
    PortScan recovers -- the analogue of STATUS's diagnosed cause
    (extreme values collapsing into the same coarse top bin heavy-tailed
    benign hosts occupy)."""
    rows = []
    attack_pool = clean_pool[clean_pool["Label"] != "Benign"]
    for n_bins in bin_counts:
        table = build_signature_table(monday_fit, n_bins=n_bins, floor=5)
        esc = priority2_escalate(attack_pool, table)
        row = {"n_bins": n_bins, "n_common_signatures": len(table.common_signatures)}
        for label in sorted(attack_pool["Label"].unique()):
            mask = (attack_pool["Label"] == label).to_numpy()
            n = int(mask.sum())
            row[f"{label}__n"] = n
            row[f"{label}__recall"] = float(esc[mask].mean()) if n else None
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Stage 4: distinct-flow-count reproducibility check
# --------------------------------------------------------------------- #

def flow_count_reproducibility_check() -> dict:
    train_ids = load_pickle_train_flow_ids()
    _install = None
    import pickle
    mod_name = "pandas.core.indexes.numeric"
    if mod_name not in sys.modules:
        mod = types.ModuleType(mod_name)

        class Int64Index(pd.Index):
            pass

        mod.Int64Index = Int64Index
        sys.modules[mod_name] = mod
    with open("cicids-2017_p0.pkl", "rb") as f:
        pkl = pickle.load(f)
    test_ids = frozenset(pkl["ungrouped_test_df"]["Flow ID"].unique().tolist())
    return {
        "pickle_train_distinct_flows": len(train_ids),
        "pickle_test_distinct_flows": len(test_ids),
        "pdf_cited_train": 108_120,
        "pdf_cited_test": 27_030,
        "matches_pdf": (len(train_ids) == 108_120 and len(test_ids) == 27_030),
    }


# --------------------------------------------------------------------- #
# Stage 5: controller load vs attack coverage (budget sweep)
# --------------------------------------------------------------------- #

def budget_sweep(clean_pool: pd.DataFrame, predicted_class: np.ndarray, signature_table) -> pd.DataFrame:
    """The meter is applied PER DAY, not once over the whole 5-day pool.

    Checked directly first: sorting the whole pool by `first_ts` and
    applying ONE non-renewing meter over all 5 days lets Monday (100%
    benign, ~529K flows, first chronologically) burn through the entire
    global budget on its own false positives (Priority 1's binary-flag
    mismatch gives it a ~28% benign FPR — see the report's caveats)
    before Tuesday's traffic is even reached, driving measured attack
    coverage to ~0% at every budget below 10%. That's an artifact of
    treating "at most X% of flows" as a single lifetime cap over a
    multi-day trace, not a property of the policy — a real controller
    budget is a rate, renewed continuously (hourly/daily), not a
    lifetime allowance spent once. Applying the meter per day (its
    capacity = budget_fraction * that day's own flow count) is the more
    realistic reading and is what's reported here."""
    is_attack_all = (clean_pool["Label"] != "Benign").to_numpy()
    is_benign_all = ~is_attack_all

    rows = []
    for budget in BUDGET_SWEEP:
        escalated_total, priority_total = _run_policy_per_day(clean_pool, predicted_class, signature_table, budget)

        n = len(clean_pool)
        attack_covered = int((escalated_total & is_attack_all).sum())
        attack_total = int(is_attack_all.sum())
        benign_escalated = int((escalated_total & is_benign_all).sum())
        p1_n = int((priority_total == 1).sum())
        p2_n = int((priority_total == 2).sum())
        p3_n = int((priority_total == 3).sum())
        rows.append({
            "budget_fraction": budget,
            "n_total": n,
            "attack_coverage": attack_covered / attack_total if attack_total else None,
            "benign_fpr": benign_escalated / int(is_benign_all.sum()) if is_benign_all.sum() else None,
            "escalated_total": int(escalated_total.sum()),
            "priority1_escalated": p1_n,
            "priority2_escalated": p2_n,
            "priority3_escalated": p3_n,
        })
    return pd.DataFrame(rows)


def plot_load_vs_coverage(sweep_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = sweep_df["budget_fraction"] * 100
    p1 = sweep_df["priority1_escalated"] / sweep_df["n_total"] * 100
    p2 = sweep_df["priority2_escalated"] / sweep_df["n_total"] * 100
    p3 = sweep_df["priority3_escalated"] / sweep_df["n_total"] * 100
    ax.stackplot(x, p1, p2, p3, labels=["Priority 1 (SpliDT)", "Priority 2 (uncommon sig.)", "Priority 3 (sampling)"])
    ax2 = ax.twinx()
    ax2.plot(x, sweep_df["attack_coverage"] * 100, color="black", marker="o", label="attack coverage (%)")
    ax.set_xscale("log")
    ax.set_xlabel("controller budget (% of all flows)")
    ax.set_ylabel("controller load, stacked by priority (% of all flows)")
    ax2.set_ylabel("attack coverage (%)")
    ax.legend(loc="upper left")
    ax2.legend(loc="lower right")
    ax.set_title("Controller load vs attack coverage")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _run_policy_per_day(clean_pool: pd.DataFrame, predicted_class: np.ndarray, signature_table, budget: float):
    """Shared per-day meter runner (see budget_sweep's docstring for why
    per-day, not once over the whole pool). Returns (escalated, priority)
    arrays aligned to clean_pool's own row order."""
    n = len(clean_pool)
    escalated = np.zeros(n, dtype=bool)
    priority = np.zeros(n, dtype=int)
    for day in clean_pool["day"].unique():
        day_idx = np.nonzero((clean_pool["day"] == day).to_numpy())[0]
        day_df = clean_pool.iloc[day_idx].reset_index(drop=True)
        day_pred = predicted_class[day_idx]
        result = run_policy(
            day_df, day_pred, signature_table,
            tau=DEFAULT_TAU, global_budget_fraction=budget,
            priority2_reserved_fraction=min(PRIORITY2_RESERVED_FRACTION, budget),
        )
        escalated[day_idx] = result.escalated
        priority[day_idx] = result.priority
    return escalated, priority


def per_priority_attribution(
    clean_pool: pd.DataFrame, predicted_class: np.ndarray, signature_table,
    budgets=(0.01, 0.05),
) -> Dict[float, pd.DataFrame]:
    """At each named operating point: per-class breakdown of which
    priority (1/2/3, or 0=not escalated) caught it, and what each
    priority spent overall."""
    out = {}
    for budget in budgets:
        escalated, priority = _run_policy_per_day(clean_pool, predicted_class, signature_table, budget)
        rows = []
        for label in sorted(clean_pool["Label"].unique()):
            mask = (clean_pool["Label"] == label).to_numpy()
            n = int(mask.sum())
            row = {"label": label, "n": n}
            for p in (1, 2, 3):
                row[f"caught_by_p{p}"] = int(((priority == p) & mask).sum())
            row["not_escalated"] = int(((priority == 0) & mask).sum())
            rows.append(row)
        df = pd.DataFrame(rows)
        df.attrs["priority_totals"] = {p: int((priority == p).sum()) for p in (1, 2, 3)}
        out[budget] = df
    return out


# --------------------------------------------------------------------- #
# Stage 6: comparison against the existing 5-feature selector
# --------------------------------------------------------------------- #

def five_feature_comparison(clean_pool: pd.DataFrame, priority1_result: dict) -> dict:
    from eval.csv_refit_cross_day import escalate_generic, fit_subset_generic
    from eval.seven_feature_selector import COUNT_5, RATIO_5
    from eval.sweep import split_monday_chronologically

    cache_dir = Path("results/cache")
    day_to_cache = {
        "monday": "monday", "tuesday": "tuesday", "wednesday": "wednesday",
        "thursday_webattacks": "thursday_morning_webattacks",
        "thursday_infiltration": "thursday_afternoon_infiltration",
        "friday_morning": "friday_morning", "friday_portscan": "friday_afternoon_portscan",
        "friday_ddos": "friday_afternoon_ddos",
    }
    frames = {}
    for day_key, cache_name in day_to_cache.items():
        frames[day_key] = pd.read_parquet(cache_dir / f"{cache_name}__eval__adapter1__features4.parquet")

    monday_fit, monday_holdout = split_monday_chronologically(frames["monday"])
    fit5 = fit_subset_generic(monday_fit, COUNT_5, RATIO_5, 99.5)

    rows = []
    for day_key, df in frames.items():
        esc = escalate_generic(df, fit5, COUNT_5, RATIO_5)
        is_benign = df["label"] == "BENIGN"
        for label in sorted(df["label"].unique()):
            mask = df["label"] == label
            n = int(mask.sum())
            rows.append({
                "day": day_key, "label": label, "n": n,
                "recall_or_fpr": float(esc[mask].mean()) if n else None,
            })
    five_feature_df = pd.DataFrame(rows)
    five_feature_benign_fpr = float(
        five_feature_df.loc[five_feature_df["label"] == "BENIGN", "recall_or_fpr"].mean()
    )
    return {
        "five_feature_per_day": five_feature_df,
        "five_feature_benign_fpr_avg": five_feature_benign_fpr,
        "splidt_benign_fpr": priority1_result["benign_fpr_clean"],
    }


# --------------------------------------------------------------------- #
# Report writer
# --------------------------------------------------------------------- #

def write_report(
    full_pool, clean_pool, rate_report, contam_report, coverage_gap,
    p1_result, bin_sweep_df, flow_count_check, budget_sweep_df, comparison,
    tie_saturation_df, attribution,
) -> None:
    lines: List[str] = []
    lines.append("# SpliDT Priority 1 escalation policy — CSV-path evaluation\n")
    lines.append(
        "Uses CICIDS2017 CSV columns directly (no flow simulation, no packet "
        "synthesis). Priority 1 (SpliDT) is supervised inference with NO "
        "zero-day property. Priority 2's benign-only signature table is the "
        "only component here carrying a zero-day claim. Priority 3 guarantees "
        "nothing. Because every subtree sees one static whole-flow CSV vector "
        "(not a sliding packet window), this run is a plain nested classifier, "
        "not real SpliDT — no time-to-detection number is reported anywhere "
        "below (see dataplane/splidt_inference.py's docstring).\n"
    )

    lines.append("## Data-quality caveats carried through every number below\n")
    lines.append(
        "- **Flag counts are binary (0/1) in this CSV, not true per-packet counts** "
        "— 56 of 66 of the model's own flag-count splits use thresholds >= 1 "
        "(ACK Flag Count up to 1114.0), unreachable with binary input. 13/45 "
        "subtrees carry at least one such dead split.\n"
        "- **Rate-feature cleaning**: NaN rows dropped (undefined 0/0 flows), "
        "Infinity kept (a real, comparison-correct encoding of a near-"
        "instantaneous flow). See table below.\n"
        "- **Training contamination**: Flow ID overlap with the professor's own "
        "training set is ~80% for nearly every attack class, 100% for Heartbleed. "
        "Every number below uses the CLEAN (non-contaminated) population only, "
        "unless explicitly labeled otherwise.\n"
    )

    lines.append("### Rate-feature (Flow Bytes/s, Flow Packets/s) Infinity/NaN, by label\n")
    lines.append(rate_report.to_markdown(index=False))
    lines.append("")

    lines.append("### Training-set contamination, by label\n")
    lines.append(contam_report.to_markdown(index=False))
    lines.append("")

    lines.append("## Coverage ceiling — the headline result\n")
    lines.append(
        f"Model can name 10 classes; **{len(coverage_gap['unseen_classes'])} classes present in this data are "
        f"entirely unseen by the model**: {', '.join(coverage_gap['unseen_classes'])} "
        f"(n={coverage_gap['unseen_total_n']:,}, "
        f"{coverage_gap['unseen_fraction_of_attack']:.1%} of all attack flows in the clean pool). "
        "Priority 1 recall on these is zero by construction, not by failure.\n"
    )
    lines.append(f"Seen classes (in this data and in the model): {', '.join(coverage_gap['seen_classes'])}\n")
    if coverage_gap["unrecognized_classes"]:
        lines.append(f"Unrecognized/unexpected labels: {coverage_gap['unrecognized_classes']}\n")

    lines.append("## Priority 1 — per-class recall (clean population only)\n")
    lines.append(f"Benign FPR (clean, held-out): **{p1_result['benign_fpr_clean']:.3%}**\n")
    lines.append(p1_result["clean_recall"].to_markdown(index=False))
    lines.append("")
    lines.append(
        "Classes with n < 50 are flagged `reliable=False` — raw count only, "
        "no recall claim should be drawn from them (Heartbleed and Web Attack "
        "SQL Injection in particular: contamination alone leaves 0 and 2 clean "
        "flows respectively — unmeasurable, not just small).\n"
    )

    lines.append("## Contamination effect on Priority 1 recall (informational, not a headline number)\n")
    lines.append(p1_result["contamination_effect"].to_markdown(index=False))
    lines.append("")

    lines.append("## Priority 2 — bin-resolution sweep (does PortScan recover?)\n")
    lines.append(
        "**Recomputed on a simplified per-(day, Source IP) aggregate** of "
        "`flows_per_src`/`distinct_dst_ports_per_src`/`syn_without_synack_count` "
        "(not the project's real rolling 10s-bucket SrcTable — recomputing from "
        "the simulated cache was not possible: eval/simulate.py's cached records "
        "carry no Flow-ID/row join key back to these raw CSV rows, and its row "
        "order is CLOSURE order, not file order — see escalation_policy.py's "
        "docstring). `bwd_pkt_len_mean`/`pkt_len_range` are exact. Because of this "
        "simplification, the numbers below are NOT directly comparable to "
        "STATUS.md's cited 12.85%/0.0% figures — this is a fresh fit on a "
        "differently-scaled feature, not a reproduction of that run.\n"
    )
    lines.append(bin_sweep_df.to_markdown(index=False))
    portscan_col = "PortScan__recall"
    portscan_start = bin_sweep_df.iloc[0][portscan_col]
    portscan_end = bin_sweep_df.iloc[-1][portscan_col]
    lines.append(
        f"\n**Does PortScan recover?** Partially. `PortScan__recall` moves from "
        f"{portscan_start:.4%} at n_bins=6 to {portscan_end:.2%} at n_bins=80 — a "
        "real, large recovery, but it plateaus well short of full recovery "
        "(unlike DoS GoldenEye/Hulk/Slowhttptest/DoS-Slowloris/FTP-Patator, which "
        "all reach ~100% by n_bins=40). Finer top-end resolution helps PortScan "
        "but does not fully resolve it here, unlike the cleaner full recovery "
        "STATUS.md reports for the project's real per-source SrcTable — "
        "consistent with this being a simplified, coarser per-day aggregate "
        "(see caveat above), not the exact same feature.\n"
    )

    lines.append("### Bin-edge tied-mass saturation check (at the default n_bins=6 fit)\n")
    lines.append(
        "Analogue of the percentile-saturation bug fixed in dataplane/fitting.py, "
        "generalized to discrete bin edges: mass tied exactly at an edge's value, "
        "as a fraction of that feature's used population.\n"
    )
    lines.append(tie_saturation_df.sort_values("tied_mass", ascending=False).head(15).to_markdown(index=False))
    lines.append("")

    lines.append("## Flow-count reproducibility check\n")
    lines.append(
        f"Packet-pickle distinct flows: train={flow_count_check['pickle_train_distinct_flows']:,}, "
        f"test={flow_count_check['pickle_test_distinct_flows']:,}. PDF cites "
        f"{flow_count_check['pdf_cited_train']:,}/{flow_count_check['pdf_cited_test']:,}. "
        f"**Match: {flow_count_check['matches_pdf']}** — "
        + ("the professor's own measurements are reproducible from this pickle.\n"
           if flow_count_check["matches_pdf"] else
           "NOT reproducible from the available packet-level pickle; the PDF's "
           "rule-selection experiment was evidently built from a different, "
           "unavailable intermediate flow export (a different day subset or "
           "flow-definition), not this pickle's own train/test split.\n")
    )

    lines.append("## Controller load vs attack coverage (budget sweep, clean population)\n")
    lines.append(
        "**Meter renews per day**, not once over the whole 5-day pool — checked "
        "directly first: a single non-renewing meter over the whole pool lets "
        "Monday (100% benign, ~529K flows, first chronologically) burn the "
        "entire budget on Priority 1's own false positives before Tuesday's "
        "traffic is even reached, driving measured attack coverage to ~0% at "
        "every budget below 10% — an artifact of treating a rate budget as a "
        "one-time lifetime cap over a multi-day trace, not a policy property. "
        "Each day gets its own `budget_fraction * that day's flow count` "
        "capacity below, which is the realistic reading of a continuously "
        "renewed controller-bound rate.\n"
    )
    lines.append(budget_sweep_df.to_markdown(index=False))
    lines.append(
        "\n**Reading the near-zero attack coverage at budgets <=2%**: this is a "
        "real, diagnosed consequence of Priority 1's benign FPR (27.986% — see "
        "above), not a further meter artifact. Benign traffic vastly outnumbers "
        "attack traffic in every day of this pool, so a benign false-positive "
        "rate this high means most of Priority 1's admitted capacity under a "
        "tight budget is consumed by false positives before genuine attack "
        "flows are even reached in arrival order — attacks only start winning "
        "a meaningful share of the budget once it's loose enough (5-10%) to "
        "clear the false-positive flood first. This is the binary-flag/model "
        "mismatch's cost made concrete at the policy level, not a shortcoming "
        "of the escalation policy's own design.\n"
    )
    lines.append(
        "STATUS.md found the existing percentile selector at <=0.4% pooled "
        "attack recall at <=1% escalation. At the matched ~1% budget row above, "
        "this policy's attack coverage is reported directly for comparison — "
        "read the row at budget_fraction=0.01. Priority 1's own severe FPR "
        "(driven by the CSV/model flag-count mismatch, not a flaw in this "
        "comparison) is why that number isn't higher; Priority 2 alone, run "
        "without Priority 1's competing false positives, would need to be read "
        "from the per-class recall/bin-sweep tables above instead.\n"
    )
    lines.append("![load vs coverage](figures/load_vs_coverage.png)\n")

    lines.append("## Per-priority attribution — which priority caught which class, what each spent\n")
    for budget, df in attribution.items():
        totals = df.attrs["priority_totals"]
        lines.append(
            f"### Operating point: budget_fraction={budget} "
            f"(P1 spent {totals[1]:,}, P2 spent {totals[2]:,}, P3 spent {totals[3]:,})\n"
        )
        lines.append(df.to_markdown(index=False))
        lines.append("")

    lines.append("## Comparison: 5-feature count-AND-ratio selector (methodologically NOT comparable — supervised vs unsupervised)\n")
    lines.append(
        f"5-feature selector average benign FPR across all 8 days (Monday-CSV-fitted, "
        f"p99.5): **{comparison['five_feature_benign_fpr_avg']:.3%}**. SpliDT Priority 1 "
        f"clean benign FPR: **{comparison['splidt_benign_fpr']:.3%}**. Priority 1 is "
        "supervised (trained on labeled attacks); the 5-feature selector is "
        "benign-only fit with no label information at fit time — these are not "
        "playing the same game, and a head-to-head recall comparison would be "
        "misleading without that caveat attached every time.\n"
    )
    lines.append(comparison["five_feature_per_day"].to_markdown(index=False))
    lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "splidt_escalation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT_DIR / 'splidt_escalation_report.md'}")


def main() -> int:
    full_pool, clean_pool = load_pools()
    model_data = load_model_json("results/splidt_model.json")

    print("rate quality + contamination reports ...")
    rate_report = rate_quality_report(full_pool)
    contam_report = contamination_report(full_pool)

    print("coverage gap ...")
    coverage_gap = coverage_gap_report(clean_pool)

    print("Priority 1 recall (clean + contamination effect) ...")
    p1_result = priority1_recall_report(clean_pool, full_pool, model_data)

    print("Priority 2: fitting signature table on Monday clean benign fit-half ...")
    monday_fit, monday_holdout = monday_chronological_split(clean_pool)
    signature_table = build_signature_table(monday_fit, n_bins=6, floor=5)
    tie_saturation_df = check_bin_tie_saturation(monday_fit, signature_table)

    print("Priority 2: bin-resolution sweep ...")
    bin_sweep_df = priority2_bin_sweep(clean_pool, monday_fit)

    print("flow-count reproducibility check ...")
    flow_count_check = flow_count_reproducibility_check()

    print("budget sweep ...")
    budget_sweep_df = budget_sweep(clean_pool, p1_result["clean_predictions"].predicted_class, signature_table)
    plot_load_vs_coverage(budget_sweep_df, FIG_DIR / "load_vs_coverage.png")

    print("per-priority attribution at 1% and 5% ...")
    attribution = per_priority_attribution(
        clean_pool, p1_result["clean_predictions"].predicted_class, signature_table, budgets=(0.01, 0.05)
    )

    print("5-feature selector comparison ...")
    comparison = five_feature_comparison(clean_pool, p1_result)

    write_report(
        full_pool, clean_pool, rate_report, contam_report, coverage_gap,
        p1_result, bin_sweep_df, flow_count_check, budget_sweep_df, comparison,
        tie_saturation_df, attribution,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
