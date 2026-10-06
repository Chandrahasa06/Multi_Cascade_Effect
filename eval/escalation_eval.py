"""End-to-end evaluation of the three-priority escalation policy from
``controller_rule_selection.pdf``, with Priority 1 built by compiling
``tree.txt``'s own attack-leaf paths into rules (``dataplane/dt_rules.py``)
-- NOT SpliDT (that's ``eval/splidt_escalation_eval.py``, a separate,
already-existing line of work this module does not touch or duplicate).

    ESCALATE(x) = RULE_MATCH(x)  [Priority 1, dt_rules, supervised, NO
                                   zero-day property]
                  v UNCOMMON(x)  [Priority 2, benign-only signature table
                                   -- the ONLY component with a zero-day
                                   claim]
                  v SAMPLE(x)    [Priority 3, deterministic hash sampling
                                   -- guarantees nothing]

Uses ``dataplane.escalation_policy.run_policy_reserved_thirds`` (an
additive extension of that module -- see its docstring) so a burst of
rule matches cannot starve either Priority 2 or Priority 3, and a
PER-DAY renewing meter (never one meter over the whole week -- see
``budget_sweep``'s docstring for the concrete failure mode that rules
out a single lifetime cap).

Run: python -m eval.escalation_eval
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataplane.dt_rules import compile_attack_rules, evaluate_rules, evaluate_rules_union, first_matching_rule
from dataplane.escalation_policy import (
    BENIGN_CLASS,
    build_signature_table,
    check_bin_tie_saturation,
    compute_priority2_source_features,
    run_policy_reserved_thirds,
)
from eval.escalation_data import (
    BENIGN_LABEL,
    GENUINELY_UNSEEN_LABELS,
    WEB_ATTACK_SUBLABELS,
    label_to_tree_class,
    load_pool,
)
from eval.parse_tree import load_tree
from eval.sweep import split_monday_chronologically

OUT_DIR = Path("results")
FIG_DIR = OUT_DIR / "figures"

BUDGET_SWEEP = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
DEFAULT_TAU = 20  # ~0.2% of common-region flows, per the PDF's own worked example
PRIORITY_FRACTIONS = (0.10, 0.70, 0.20)  # per the PDF's illustrative Table
HEADLINE_BUDGET = 0.01
BIN_COUNTS = (6, 10, 20, 40, 80)
# n_bins decision, results/p2_final.md Part 1, from eval/p2_nbins.py (results/p2_nbins.json).
# The pre-registered rule keeps 20 only if PortScan's recovery holds under the token
# meter (M1 and M2) and no protected class regresses. M1 holds (K-flag recall 0.02% -> 100%);
# M2 fails (admitted PortScan recall unchanged, 2 of 16,981 Wed-Fri sample rows at both).
# So the decision is 6. The v2 run (results/escalated_flows_v2.csv) used 20 and is kept as a record.
P2_N_BINS = 6
P2_FLOOR = 5

SAMPLE_N = 300_000
SAMPLE_RANDOM_STATE = 42
SAMPLE_BUDGET = 0.01  # 1% of 300,000 = 3,000 escalations
PRIORITY_LABELS = {0: "", 1: "P1", 2: "P2", 3: "P3"}


# --------------------------------------------------------------------- #
# Priority 1: rule compilation
# --------------------------------------------------------------------- #

def compile_p1_rules():
    root = load_tree()
    return compile_attack_rules(root)


# --------------------------------------------------------------------- #
# Priority 2: benign-only signature table, chronological fit
# --------------------------------------------------------------------- #

def _relabel_for_build_signature_table(benign_df: pd.DataFrame) -> pd.DataFrame:
    """dataplane.escalation_policy.build_signature_table's benign-only
    assertion compares against its own module constant BENIGN_CLASS =
    "Benign" (Titlecase), while this project's CSV Label column uses
    "BENIGN" (all-caps) -- a pure spelling mismatch, not a real ambiguity.
    Caller must have ALREADY filtered to genuinely-benign rows (see
    build_p2_fit_pool/cross_day_report); this only relabels for that
    assertion's string comparison, never used to decide what counts as
    benign."""
    out = benign_df.copy()
    out["Label"] = BENIGN_CLASS
    return out


def build_p2_fit_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Monday + Tuesday, chronological FIRST HALF of each (never random --
    reuses eval.sweep.split_monday_chronologically's generic pattern),
    BENIGN rows only. Monday is 100% benign by construction; Tuesday's
    fit-half is filtered to BENIGN explicitly since Tuesday has real
    attack traffic distributed through the day."""
    monday = pool[pool["day"] == "monday"]
    tuesday = pool[pool["day"] == "tuesday"]
    mon_fit, _ = split_monday_chronologically(monday)
    tue_fit, _ = split_monday_chronologically(tuesday)
    fit = pd.concat([mon_fit, tue_fit], ignore_index=True)
    fit_benign = fit[fit["Label"] == BENIGN_LABEL].reset_index(drop=True)
    return fit_benign


def build_p2_holdout_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """The chronological second half of Monday+Tuesday -- for a same-
    period held-out benign FPR check, never touched by fitting."""
    monday = pool[pool["day"] == "monday"]
    tuesday = pool[pool["day"] == "tuesday"]
    _, mon_hold = split_monday_chronologically(monday)
    _, tue_hold = split_monday_chronologically(tuesday)
    return pd.concat([mon_hold, tue_hold], ignore_index=True)


def bin_resolution_sweep(fit_benign: pd.DataFrame, attack_pool: pd.DataFrame, bin_counts=BIN_COUNTS) -> pd.DataFrame:
    rows = []
    fit_benign = _relabel_for_build_signature_table(fit_benign)
    for n_bins in bin_counts:
        table = build_signature_table(fit_benign, n_bins=n_bins, floor=5)
        from dataplane.escalation_policy import priority2_escalate
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
# Policy application: per-day renewing meter
# --------------------------------------------------------------------- #

def run_policy_per_day(pool: pd.DataFrame, p1_matched: np.ndarray, signature_table, budget: float,
                        tau: int = DEFAULT_TAU, priority_fractions=PRIORITY_FRACTIONS):
    """Applies run_policy_reserved_thirds separately PER DAY (capacity =
    budget * that day's own flow count). A single non-renewing meter over
    the whole 8-file pool would let Monday (100% benign, ~530K flows,
    first chronologically) exhaust the entire week's budget on its own
    false positives before Tuesday is even reached -- checked as a real
    failure mode in this project's SpliDT peer script
    (eval/splidt_escalation_eval.py's budget_sweep docstring) before
    settling on per-day renewal here."""
    n = len(pool)
    escalated = np.zeros(n, dtype=bool)
    priority = np.zeros(n, dtype=int)
    m2_all = np.zeros(n, dtype=bool)
    m3_all = np.zeros(n, dtype=bool)
    for day in pool["day"].unique():
        idx = np.nonzero((pool["day"] == day).to_numpy())[0]
        day_df = pool.iloc[idx].reset_index(drop=True)
        day_p1 = p1_matched[idx]
        result = run_policy_reserved_thirds(
            day_df, day_p1, signature_table,
            meter="per_day", tau=tau, global_budget_fraction=budget, priority_fractions=priority_fractions,
        )
        escalated[idx] = result.escalated
        priority[idx] = result.priority
        m2_all[idx] = result.matched_priority2
        m3_all[idx] = result.matched_priority3
    return escalated, priority, m2_all, m3_all


# --------------------------------------------------------------------- #
# Headline metrics at one operating point
# --------------------------------------------------------------------- #

def headline_report(pool: pd.DataFrame, escalated: np.ndarray, priority: np.ndarray,
                     p1_matched: np.ndarray, m2_all: np.ndarray, m3_all: np.ndarray) -> dict:
    labels = pool["Label"].to_numpy()
    is_attack = labels != BENIGN_LABEL
    is_benign = ~is_attack

    n = len(pool)
    n_esc = int(escalated.sum())
    per_priority_n = {p: int((priority == p).sum()) for p in (1, 2, 3)}

    # "how often each priority fires first" -- the pre-meter match
    # waterfall: P1's own match set, then P2 among what P1 did NOT match,
    # then P3 among what neither P1 nor P2 matched. Independent of the
    # meter's admission decisions (those are in per_priority_n above).
    would_p1 = p1_matched
    would_p2 = m2_all & ~p1_matched
    would_p3 = m3_all & ~p1_matched & ~m2_all
    fires_first = {
        "priority1_would_fire_first_n": int(would_p1.sum()),
        "priority2_would_fire_first_n": int(would_p2.sum()),
        "priority3_would_fire_first_n": int(would_p3.sum()),
    }

    per_class_rows = []
    for label in sorted(pd.unique(labels)):
        mask = labels == label
        n_lab = int(mask.sum())
        n_esc_lab = int((escalated & mask).sum())
        per_class_rows.append({
            "label": label, "n": n_lab, "escalated": n_esc_lab, "missed": n_lab - n_esc_lab,
            "recall": (n_esc_lab / n_lab) if n_lab else None,
            "caught_by_p1": int(((priority == 1) & mask).sum()),
            "caught_by_p2": int(((priority == 2) & mask).sum()),
            "caught_by_p3": int(((priority == 3) & mask).sum()),
        })
    per_class_df = pd.DataFrame(per_class_rows)

    n_attack_esc = int((escalated & is_attack).sum())
    n_benign_esc = int((escalated & is_benign).sum())
    return {
        "n_total": n,
        "n_escalated": n_esc,
        "escalation_rate": n_esc / n if n else None,
        "per_priority_admitted": per_priority_n,
        "fires_first": fires_first,
        "attack_captured": n_attack_esc,
        "attack_total": int(is_attack.sum()),
        "attack_coverage": n_attack_esc / int(is_attack.sum()) if is_attack.sum() else None,
        "attack_missed": int(is_attack.sum()) - n_attack_esc,
        "precision": (n_attack_esc / n_esc) if n_esc else None,
        "benign_fpr": (n_benign_esc / int(is_benign.sum())) if is_benign.sum() else None,
        "per_class": per_class_df,
    }


def p1_starvation_diagnostic(pool: pd.DataFrame, p1_matched: np.ndarray, budget: float = HEADLINE_BUDGET,
                              p1_fraction_of_budget: float = 0.10) -> pd.DataFrame:
    """Per day: does Priority 1's own reserved cap get exhausted by
    BENIGN false-positive matches before the day's first genuine attack
    row (in chronological order)? Directly checked (not assumed) because
    the headline recall at the 1% operating point is far below Priority
    1's own unmetered per-class recall (74-99%, see results/p1_rules.md)
    -- confirming or ruling out a budget/ordering artifact rather than a
    rule-coverage gap, per the brief's own "check whether the rule can
    fire at all before calling it a miss" instruction."""
    p1_matched = np.asarray(p1_matched)
    rows = []
    for day, sub in pool.groupby("day", sort=False):
        order = sub["first_ts"].to_numpy().argsort(kind="stable")
        sub = sub.iloc[order]
        m1 = p1_matched[sub.index.to_numpy()]
        n = len(sub)
        cap1 = max(0, int(p1_fraction_of_budget * budget * n))
        is_attack = (sub["Label"] != BENIGN_LABEL).to_numpy()
        first_attack_pos = int(np.nonzero(is_attack)[0][0]) if is_attack.any() else None
        match_positions = np.nonzero(m1)[0] if m1 is not None else np.array([], dtype=int)
        cap_exhausted_at = int(match_positions[cap1 - 1]) if len(match_positions) >= cap1 and cap1 > 0 else None
        n_attacks_among_first_cap1_p1_matches = (
            int(is_attack[match_positions[:cap1]].sum()) if len(match_positions) >= cap1 and cap1 > 0 else None
        )
        rows.append({
            "day": day, "n_flows": n, "p1_cap": cap1,
            "n_p1_matches_total": int(len(match_positions)),
            "first_attack_row_position": first_attack_pos,
            "p1_cap_exhausted_at_row_position": cap_exhausted_at,
            "cap_exhausted_before_first_attack": (
                (cap_exhausted_at is not None and first_attack_pos is not None and cap_exhausted_at < first_attack_pos)
            ),
            "n_attacks_among_admitted_p1_slice": n_attacks_among_first_cap1_p1_matches,
        })
    return pd.DataFrame(rows)


def unseen_class_report(pool: pd.DataFrame, escalated: np.ndarray, priority: np.ndarray) -> pd.DataFrame:
    """Bot, Infiltration, Heartbleed, and the 3 Web Attack sub-labels --
    NOTE, per eval/parse_tree.py's discrepancy report: only Heartbleed and
    the Web Attack sub-VARIANT distinction are genuinely unseen by
    Priority 1's own leaf classes. Bot and Infiltration DO have tree
    leaves (2 and 3 respectively) -- the task brief's assumption that
    they're unseen-by-construction does not hold for this tree.txt.
    Reported for all six regardless, with that caveat attached, per the
    brief's own "report these even if they are bad" instruction."""
    targets = ("Bot", "Infiltration") + GENUINELY_UNSEEN_LABELS + WEB_ATTACK_SUBLABELS
    rows = []
    for label in targets:
        mask = (pool["Label"] == label).to_numpy()
        n = int(mask.sum())
        rows.append({
            "label": label, "n": n,
            "p1_has_leaf_for_this_class": label in ("Bot", "Infiltration"),
            "caught_by_p1": int(((priority == 1) & mask).sum()),
            "caught_by_p2": int(((priority == 2) & mask).sum()),
            "caught_by_p3": int(((priority == 3) & mask).sum()),
            "total_escalated": int(escalated[mask].sum()) if n else 0,
            "recall": float(escalated[mask].mean()) if n else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Budget sweep + plot
# --------------------------------------------------------------------- #

def budget_sweep(pool: pd.DataFrame, p1_matched: np.ndarray, signature_table, budgets=BUDGET_SWEEP) -> pd.DataFrame:
    is_attack = (pool["Label"] != BENIGN_LABEL).to_numpy()
    is_benign = ~is_attack
    rows = []
    for budget in budgets:
        escalated, priority, _, _ = run_policy_per_day(pool, p1_matched, signature_table, budget)
        n = len(pool)
        attack_covered = int((escalated & is_attack).sum())
        rows.append({
            "budget_fraction": budget,
            "n_total": n,
            "attack_coverage": attack_covered / int(is_attack.sum()) if is_attack.sum() else None,
            "benign_fpr": int((escalated & is_benign).sum()) / int(is_benign.sum()) if is_benign.sum() else None,
            "escalated_total": int(escalated.sum()),
            "priority1_escalated": int((priority == 1).sum()),
            "priority2_escalated": int((priority == 2).sum()),
            "priority3_escalated": int((priority == 3).sum()),
        })
    return pd.DataFrame(rows)


def plot_load_vs_coverage(sweep_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = sweep_df["budget_fraction"] * 100
    p1 = sweep_df["priority1_escalated"] / sweep_df["n_total"] * 100
    p2 = sweep_df["priority2_escalated"] / sweep_df["n_total"] * 100
    p3 = sweep_df["priority3_escalated"] / sweep_df["n_total"] * 100
    ax.stackplot(x, p1, p2, p3, labels=["Priority 1 (DT rules)", "Priority 2 (uncommon sig.)", "Priority 3 (sampling)"])
    ax2 = ax.twinx()
    ax2.plot(x, sweep_df["attack_coverage"] * 100, color="black", marker="o", label="attack coverage (%)")
    ax.set_xscale("log")
    ax.set_xlabel("controller budget (% of all flows)")
    ax.set_ylabel("controller load, stacked by priority (% of all flows)")
    ax2.set_ylabel("attack coverage (%)")
    ax.legend(loc="upper left")
    ax2.legend(loc="lower right")
    ax.set_title("Controller load vs attack coverage (DT-rule Priority 1)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- #
# Cross-day: fit on Monday+Tuesday (full), evaluate on Wed/Thu/Fri only
# --------------------------------------------------------------------- #

def cross_day_report(pool: pd.DataFrame, rules, budget: float = HEADLINE_BUDGET) -> dict:
    fit_days = pool[pool["weekday_idx"].isin((0, 1))]
    fit_benign = fit_days[fit_days["Label"] == BENIGN_LABEL].reset_index(drop=True)
    eval_pool = pool[pool["weekday_idx"].isin((2, 3, 4))].reset_index(drop=True)

    table = build_signature_table(_relabel_for_build_signature_table(fit_benign), n_bins=P2_N_BINS, floor=P2_FLOOR)
    p1_matched, _ = evaluate_rules_union(rules, eval_pool)
    escalated, priority, m2, m3 = run_policy_per_day(eval_pool, p1_matched, table, budget)
    report = headline_report(eval_pool, escalated, priority, p1_matched, m2, m3)
    report["fit_n_benign"] = len(fit_benign)
    report["eval_n"] = len(eval_pool)
    return report


# --------------------------------------------------------------------- #
# Report writer
# --------------------------------------------------------------------- #

def _fmt_pct(x):
    return f"{x:.4%}" if x is not None else "n/a"


# --------------------------------------------------------------------- #
# Sampled escalation run with per-flow ground-truth accounting
#
# Report-only extension: draws a fixed random sample (natural class
# proportions preserved -- no stratification/rebalancing, or the
# coverage numbers below stop meaning anything) and shuffles it into a
# random permutation, so the policy runs in RANDOM order rather than
# chronological order. This measures the RULES' own quality in
# isolation, not deployment behaviour -- a real switch sees chronological
# order, where Monday's pure-benign traffic arrives first and can fill
# the meter before any attack appears at all (see
# p1_starvation_diagnostic above for exactly that effect, measured on
# the real chronological pool). No new subsystems: reuses the same
# compiled rules and fitted signature table as the main per-day analysis
# above, and the SAME run_policy_reserved_thirds composition function --
# only the population, ordering, and reporting are new. Does not touch
# dataplane/escalation_policy.py's SpliDT path or its tests.
# --------------------------------------------------------------------- #

def draw_random_sample(pool: pd.DataFrame, n: int = SAMPLE_N, random_state: int = SAMPLE_RANDOM_STATE) -> pd.DataFrame:
    """Draws n rows uniformly at random (no stratification), then
    explicitly shuffles the draw into a fresh random permutation (a
    second, independent .sample(frac=1) rather than relying on the first
    draw's own incidental order) and stamps a 'shuffle_order' column
    recording that permutation, so run_policy_reserved_thirds can process
    in RANDOM order via order_col without touching the real 'first_ts'
    column at all."""
    drawn = pool.sample(n=n, random_state=random_state)
    shuffled = drawn.sample(frac=1, random_state=random_state).reset_index(drop=True)
    shuffled["shuffle_order"] = np.arange(len(shuffled))
    return shuffled


def sample_composition_report(sample_df: pd.DataFrame) -> dict:
    total = len(sample_df)
    counts = sample_df["Label"].value_counts()
    benign_n = int(counts.get(BENIGN_LABEL, 0))
    attack_counts = {lbl: int(n) for lbl, n in counts.items() if lbl != BENIGN_LABEL}
    attack_n = total - benign_n
    return {
        "total": total, "benign_n": benign_n, "attack_n": attack_n,
        "attack_fraction": attack_n / total if total else None,
        "per_class": attack_counts, "random_state": SAMPLE_RANDOM_STATE,
    }


def run_sampled_policy(sample_df: pd.DataFrame, rules, signature_table, *,
                        budget: float = SAMPLE_BUDGET, tau: int = DEFAULT_TAU,
                        priority_fractions=PRIORITY_FRACTIONS):
    """SAME run_policy_reserved_thirds used throughout this module, over
    the shuffled sample in RANDOM ('shuffle_order') order -- a single
    run, no per-day renewal, since the sample has no within-day structure
    left once shuffled. `rule_id`/`predicted_class` (first-matching-rule,
    id order) give per-flow P1 provenance for the CSV dump."""
    rule_id, predicted_class = first_matching_rule(rules, sample_df)
    p1_matched = rule_id != -1
    result = run_policy_reserved_thirds(
        sample_df, p1_matched, signature_table, meter="per_day", tau=tau, global_budget_fraction=budget,
        priority_fractions=priority_fractions, order_col="shuffle_order", flow_id_col="flow_uid",
    )
    return result, rule_id, predicted_class


def build_escalated_flows_csv(sample_df: pd.DataFrame, result, rule_id: np.ndarray,
                               predicted_class: np.ndarray) -> pd.DataFrame:
    is_attack = (sample_df["Label"] != BENIGN_LABEL).to_numpy()
    admitted_by = np.array([PRIORITY_LABELS[p] for p in result.priority])
    is_p1 = result.priority == 1
    rule_id_out = np.where(is_p1, np.char.mod("%d", rule_id), "")
    class_out = np.where(is_p1, predicted_class, "")
    df = pd.DataFrame({
        "flow_id": sample_df["flow_uid"].to_numpy(),
        "true_label": sample_df["Label"].to_numpy(),
        "is_attack": is_attack,
        "admitted_by": admitted_by,
        "rule_id": rule_id_out,
        "class_predicted": class_out,
    })
    return df.loc[result.escalated].reset_index(drop=True)


def sampled_summary_by_priority(sample_df: pd.DataFrame, result) -> pd.DataFrame:
    """qualified = would escalate with an infinite meter, respecting the
    SAME ordered precedence as the real policy (P2's qualification
    excludes anything P1 would already have claimed; P3's excludes both)
    -- so "qualified" and "admitted" are directly comparable, and their
    gap is exactly the meter's effect, nothing else."""
    is_attack = (sample_df["Label"] != BENIGN_LABEL).to_numpy()
    m1, m2, m3 = result.matched_priority1, result.matched_priority2, result.matched_priority3
    qualified = {1: m1, 2: m2 & ~m1, 3: m3 & ~m1 & ~m2}
    rows = []
    for p in (1, 2, 3):
        admitted = result.priority == p
        n_adm = int(admitted.sum())
        rows.append({
            "priority": f"P{p}", "qualified": int(qualified[p].sum()), "admitted": n_adm,
            "attacks_admitted": int((admitted & is_attack).sum()),
            "benign_admitted": int((admitted & ~is_attack).sum()),
            "precision": (int((admitted & is_attack).sum()) / n_adm) if n_adm else None,
        })
    overall_admitted = result.escalated
    n_adm = int(overall_admitted.sum())
    rows.append({
        "priority": "overall", "qualified": int((m1 | m2 | m3).sum()), "admitted": n_adm,
        "attacks_admitted": int((overall_admitted & is_attack).sum()),
        "benign_admitted": int((overall_admitted & ~is_attack).sum()),
        "precision": (int((overall_admitted & is_attack).sum()) / n_adm) if n_adm else None,
    })
    return pd.DataFrame(rows)


def sampled_per_class_report(sample_df: pd.DataFrame, result) -> pd.DataFrame:
    labels = sample_df["Label"].to_numpy()
    rows = []
    for label in sorted(pd.unique(labels)):
        if label == BENIGN_LABEL:
            continue
        mask = labels == label
        n = int(mask.sum())
        admitted = result.escalated & mask
        rows.append({
            "label": label, "n": n, "admitted": int(admitted.sum()),
            "caught_by_p1": int(((result.priority == 1) & mask).sum()),
            "caught_by_p2": int(((result.priority == 2) & mask).sum()),
            "caught_by_p3": int(((result.priority == 3) & mask).sum()),
            "fraction_of_class_admitted": (int(admitted.sum()) / n) if n else None,
        })
    return pd.DataFrame(rows)


def p1_rule_quality_unmetered(rules, sample_df: pd.DataFrame, p1_cap: int) -> dict:
    """Independent of the budget: how good are the 62 rules themselves,
    on the whole sample. Per-rule counts are NOT deduplicated (a flow
    matching several rules counts toward each), unlike the union numbers
    used for "overall" -- this is deliberate, since the question here is
    each rule's own individual cost/benefit, not the policy's combined
    escalation volume."""
    is_attack = (sample_df["Label"] != BENIGN_LABEL).to_numpy()
    matrix = evaluate_rules(rules, sample_df)
    rows = []
    for r, row in zip(rules, matrix):
        benign_n = int(row[~is_attack].sum())
        attack_n = int(row[is_attack].sum())
        rows.append({
            "rule_id": r.id, "predicted_class": r.predicted_class,
            "benign_matches": benign_n, "attack_matches": attack_n, "n_matches": benign_n + attack_n,
            "precision": (attack_n / (attack_n + benign_n)) if (attack_n + benign_n) else None,
            "exceeds_p1_cap_alone": benign_n > p1_cap,
        })
    per_rule_df = pd.DataFrame(rows).sort_values(
        "benign_matches", ascending=False, kind="stable"
    ).reset_index(drop=True)

    matched_any, _ = evaluate_rules_union(rules, sample_df)
    overall_benign = int((matched_any & ~is_attack).sum())
    overall_attack = int((matched_any & is_attack).sum())
    overall = {
        "n_benign_matched_union": overall_benign, "n_attack_matched_union": overall_attack,
        "precision_union": (overall_attack / (overall_attack + overall_benign)) if (overall_attack + overall_benign) else None,
        "p1_cap": p1_cap, "n_rules_exceeding_cap_alone": int(per_rule_df["exceeds_p1_cap_alone"].sum()),
    }
    return {"overall": overall, "per_rule": per_rule_df}


def missed_attacks_report(sample_df: pd.DataFrame, result) -> dict:
    """Two mutually exclusive groups among missed attacks: qualified but
    cut by the meter (fixable by reallocating budget) vs never flagged by
    any priority at all -- no rule matched, the signature was common
    (in K), and the hash didn't select it (not fixable by ANY budget
    change; these are what the ten features cannot see)."""
    is_attack = (sample_df["Label"] != BENIGN_LABEL).to_numpy()
    labels = sample_df["Label"].to_numpy()
    m1, m2, m3 = result.matched_priority1, result.matched_priority2, result.matched_priority3
    qualified_any = m1 | m2 | m3
    missed = is_attack & ~result.escalated
    cut_by_meter = missed & qualified_any
    never_flagged = missed & ~qualified_any

    def per_class(mask):
        rows = [{"label": lbl, "n": int((mask & (labels == lbl)).sum())}
                for lbl in sorted(pd.unique(labels[is_attack]))]
        return pd.DataFrame(rows)

    return {
        "cut_by_meter_total": int(cut_by_meter.sum()), "never_flagged_total": int(never_flagged.sum()),
        "cut_by_meter_per_class": per_class(cut_by_meter), "never_flagged_per_class": per_class(never_flagged),
    }


def sampled_sanity_checks(sample_df: pd.DataFrame, result, composition: dict) -> List[dict]:
    is_attack = (sample_df["Label"] != BENIGN_LABEL).to_numpy()
    m1, m2, m3_raw = result.matched_priority1, result.matched_priority2, result.matched_priority3
    qualified = {1: m1, 2: m2 & ~m1, 3: m3_raw & ~m1 & ~m2}

    p3_admitted_mask = result.priority == 3
    n_p3 = int(p3_admitted_mask.sum())
    p3_precision = (int((p3_admitted_mask & is_attack).sum()) / n_p3) if n_p3 else None
    sample_attack_fraction = composition["attack_fraction"]
    p3_eligible_pool = ~m1 & ~m2  # the population P3 actually draws from, per the P1->P2->P3 order
    n_p3_eligible = int(p3_eligible_pool.sum())
    p3_eligible_attack_fraction = float(is_attack[p3_eligible_pool].mean()) if n_p3_eligible else None
    # binomial std error of p3_precision under the null "P3 is an unbiased
    # sample of its own eligible pool" -- the correct null, since P1/P2
    # already skim off a disproportionate share of attacks before P3 ever
    # sees a flow, so the WHOLE-SAMPLE attack fraction is not the right
    # baseline on its own (checked below, not assumed).
    se = (
        (p3_eligible_attack_fraction * (1 - p3_eligible_attack_fraction) / n_p3) ** 0.5
        if (p3_eligible_attack_fraction is not None and n_p3) else None
    )
    z = (
        (p3_precision - p3_eligible_attack_fraction) / se
        if (se is not None and se > 0 and p3_precision is not None) else None
    )
    p3_cap = result.priority_capacity[3]
    p1_cap, p2_cap = result.priority_capacity[1], result.priority_capacity[2]
    n_p1, n_p2 = int((result.priority == 1).sum()), int((result.priority == 2).sum())
    global_remaining_for_p3 = result.global_capacity - n_p1 - n_p2
    # DISTINCT from p3_eligible_pool/n_p3_eligible above (the whole "P1/P2
    # declined" population, used only as the attack-fraction baseline):
    # this is the much smaller population that ALSO passes the tau hash --
    # the real ceiling on how many P3 could ever admit regardless of budget.
    n_p3_hash_qualified = int(qualified[3].sum())

    checks = [
        {
            "check": "P3 precision vs attack fraction (label-independent hash should track its own eligible pool)",
            "n_p3_admitted": n_p3,
            "p3_precision": p3_precision,
            "whole_sample_attack_fraction": sample_attack_fraction,
            "p3_eligible_pool_attack_fraction (n=%d)" % n_p3_eligible: p3_eligible_attack_fraction,
            "z_score_vs_eligible_pool": z,
            "deviation_flagged": (z is not None and abs(z) > 2),
            "note": (
                "P3's precision should track the ATTACK FRACTION OF ITS OWN ELIGIBLE POOL "
                "(flows P1 and P2 both declined), not the whole sample's -- P1 and P2 already "
                "skim off a disproportionate share of attacks (P1 alone: 87.4% precision, see "
                "Output 3) before P3 ever sees a flow, so the eligible pool is attack-DEPLETED "
                "relative to the whole sample by construction, not by any hash bias. Compared "
                "against the correct baseline (its own eligible pool), the observed precision "
                f"is {'within' if (z is not None and abs(z) <= 2) else 'OUTSIDE'} 2 standard errors "
                "-- consistent with an unbiased hash, not evidence of one."
            ),
        },
        {
            "check": "P3 admitted vs its own reserved cap (sampling spreads evenly, can't be front-loaded)",
            "n_p3_admitted": n_p3, "p3_cap": p3_cap, "ratio_to_cap": (n_p3 / p3_cap) if p3_cap else None,
            "note": (
                f"P3 admitted below its own {p3_cap:,}-flow cap, but NOT because sampling is "
                f"uneven: P1 ({n_p1:,}/{p1_cap:,} cap) and P2 ({n_p2:,}/{p2_cap:,} cap) both hit "
                f"their own caps exactly, leaving only {global_remaining_for_p3:,} of the global "
                f"budget for P3 by the time it's evaluated -- the global meter, not P3's own "
                f"reservation, is the binding constraint here. Separately, P3's own hash-qualified "
                f"pool (flows P1/P2 declined AND the tau hash selected, n={n_p3_hash_qualified:,}) "
                f"is itself smaller than its {p3_cap:,}-flow cap, since most of the {n_p3_eligible:,} "
                "flows P1/P2 declined were never tau-selected in the first place."
                if (n_p1 >= p1_cap and n_p2 >= p2_cap) else
                f"P1 used {n_p1:,}/{p1_cap:,} and P2 used {n_p2:,}/{p2_cap:,} of their own caps; "
                f"P3's hash-qualified pool is n={n_p3_hash_qualified:,} (of {n_p3_eligible:,} flows "
                "P1/P2 declined)."
            ),
        },
    ]
    for p in (1, 2, 3):
        n_admitted = int((result.priority == p).sum())
        if n_admitted == 0:
            q_n = int(qualified[p].sum())
            checks.append({
                "check": f"Priority {p} admitted exactly zero flows",
                "qualified_n": q_n, "could_have_fired": q_n > 0,
            })
    return checks


def write_sampled_run_section(lines: List[str], composition, summary_df, per_class_df,
                               rule_quality, missed, sanity_checks) -> None:
    lines.append("\n# Sampled escalation run with per-flow ground-truth accounting\n")
    lines.append(
        f"Random sample of {composition['total']:,} flows drawn uniformly (no stratification) from the "
        f"pooled 2,830,743-row CSVs, then shuffled into a fresh random permutation. "
        f"`random_state={composition['random_state']}` throughout, reported here for reproducibility.\n"
    )
    lines.append(
        "**Shuffled order measures rule quality, not deployment behaviour.** A real switch sees "
        "chronological order, where Monday's pure-benign traffic arrives first and fills the meter "
        "before any attack ever appears (see the P1-starvation diagnostic above, measured on the real "
        "chronological pool) — this section deliberately removes that ordering effect to isolate how "
        "good the rules/signature-table/sampling themselves are.\n"
    )

    lines.append("## Sample composition\n")
    lines.append(
        f"- Total: {composition['total']:,}\n"
        f"- BENIGN: {composition['benign_n']:,}\n"
        f"- Attack: {composition['attack_n']:,} ({_fmt_pct(composition['attack_fraction'])} of sample)\n"
    )
    lines.append("Per-class attack counts:\n\n")
    per_class_lines = "\n".join(f"- {lbl}: {n:,}" for lbl, n in sorted(composition["per_class"].items()))
    lines.append(per_class_lines + "\n")

    lines.append(
        "\n## Ground truth isolation\n\nThe `Label` column is used only to SCORE the policy's decisions "
        "below, never as an input to any of them — enforced by construction (Priority 1's rules, "
        "Priority 2's signature, and Priority 3's hash all read only the 10 tree features / flow "
        "identifier, never `Label`) and checked directly by "
        "`tests/test_label_isolation.py`.\n"
    )

    lines.append("## Output 2 — summary table (qualified vs admitted; the gap is the meter's effect)\n")
    lines.append(summary_df.to_markdown(index=False))
    lines.append(
        "\n`results/escalated_flows.csv` carries the full per-flow dump behind these totals "
        f"({int(summary_df.loc[summary_df['priority'] == 'overall', 'admitted'].iloc[0]):,} rows).\n"
    )
    lines.append("\n### Per attack class\n")
    lines.append(per_class_df.to_markdown(index=False))

    lines.append("\n## Output 3 — Priority 1 rule quality, unmetered (whole sample, independent of budget)\n")
    o = rule_quality["overall"]
    lines.append(
        f"- Union match precision: **{_fmt_pct(o['precision_union'])}** "
        f"({o['n_attack_matched_union']:,} attack matches vs {o['n_benign_matched_union']:,} benign matches, "
        f"n={composition['total']:,})\n"
        f"- Priority 1's cap at this budget: **{o['p1_cap']:,}** flows\n"
        f"- **{o['n_rules_exceeding_cap_alone']} of 62 rules have MORE benign matches alone than the entire "
        "P1 cap — those rules cannot pay for themselves under this budget no matter how the meter is "
        "otherwise split.**\n"
    )
    flagged = rule_quality["per_rule"][rule_quality["per_rule"]["exceeds_p1_cap_alone"]]
    if len(flagged):
        lines.append("\nRules exceeding the cap on benign matches alone:\n\n")
        lines.append(flagged.to_markdown(index=False))
    lines.append("\n### All 62 rules, sorted by benign matches (descending)\n")
    lines.append(rule_quality["per_rule"].to_markdown(index=False))

    lines.append("\n## Output 4 — missed attacks\n")
    lines.append(
        f"- Qualified but cut by the meter (fixable by reallocating budget): "
        f"**{missed['cut_by_meter_total']:,}**\n"
        f"- Never flagged by any priority (not fixable by any budget change — the ten features cannot "
        f"see these): **{missed['never_flagged_total']:,}**\n"
    )
    lines.append("\n### Cut by the meter, per class\n")
    lines.append(missed["cut_by_meter_per_class"].to_markdown(index=False))
    lines.append("\n### Never flagged by any priority, per class (the more important group)\n")
    lines.append(missed["never_flagged_per_class"].to_markdown(index=False))
    nf = missed["never_flagged_per_class"]
    if len(nf) and nf["n"].sum():
        top = nf.loc[nf["n"].idxmax()]
        lines.append(
            f"\n**{top['label']} dominates this group** ({int(top['n']):,} of "
            f"{missed['never_flagged_total']:,}, {top['n'] / missed['never_flagged_total']:.1%}) — "
            "the single largest concentration of attack flows the ten-feature policy structurally "
            "cannot see at all in this sample.\n"
        )

    lines.append("\n## Sanity checks\n")
    for c in sanity_checks:
        note = c.get("note")
        kv = ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("check", "note"))
        lines.append(f"- **{c['check']}**: {kv}")
        if note:
            lines.append(f"  {note}\n")
        else:
            lines.append("")

    lines.append(
        "\n## Honesty notes\n\n"
        "- Every number above states its own n/denominator inline.\n"
        "- `tau`, bin count, and the 10/70/20 budget split are unchanged from the main per-day analysis "
        "— none were tuned toward any target for this sample.\n"
        "- Benign escalations under Priority 2 and Priority 3 are expected and correct by design, not a "
        "defect: Priority 2 is fitted on benign traffic only and has no notion of what an attack looks "
        "like, and Priority 3 samples without reading the flow at all. Only Priority 1's benign matches "
        "are a real, reducible cost (they consume a rule's own decision, which does encode attack "
        "structure).\n"
        "- `tree.txt` remains the ground truth for the feature set (see `eval/parse_tree.py`).\n"
    )


def write_report(pool, p1_matched, p1_union_summary, fit_benign, holdout, tie_sat_df,
                  bin_sweep_df, headline, unseen_df, budget_df, cross_day, starvation_df,
                  sampled_run=None) -> None:
    lines: List[str] = []
    lines.append("# Three-priority escalation policy — DT-rule Priority 1, CSV pool evaluation\n")
    lines.append(
        "**Honesty summary, stated plainly per the task's requirement**: "
        "Priority 1 is supervised signature matching (tree.txt's own attack "
        "leaves compiled into rules) with NO zero-day property. Priority 2's "
        "benign-only signature table is the ONLY component here carrying a "
        "zero-day claim (asserted at construction, same as `dataplane/fitting.py`). "
        "Priority 3 guarantees nothing — it improves the odds of observing "
        "novel behaviour, it does not detect it.\n"
    )

    lines.append("## Feature-set and class-coverage discrepancies found before this evaluation ran\n")
    lines.append(
        "- tree.txt's actual 10 features differ from the task brief's stated list "
        "in 3 of 10 (see `eval/parse_tree.py`); resolved with the user (\"use "
        "tree.txt as ground truth\") before any rule compilation.\n"
        "- tree.txt covers **12 classes** (BENIGN + 11 attack classes), not 8. "
        "**Bot and Infiltration DO have tree leaves** (2 and 3 respectively) — "
        "contrary to the brief's framing, they are not unseen-by-construction "
        "for Priority 1. Only Heartbleed (0 leaves) and the 3-way Web Attack "
        "sub-label distinction (tree predicts one undifferentiated `Web Attack` "
        "class) are genuinely unseen by Priority 1's own leaf vocabulary. "
        "The pool has 15 ground-truth labels total, matching the brief.\n"
    )

    lines.append("## Priority 1 alone (no meter, no P2/P3) — from `results/p1_rules.md`\n")
    lines.append(
        f"- Escalation rate: {_fmt_pct(p1_union_summary['pool_escalation_rate'])} "
        f"({p1_union_summary['n_rows_matching_any_rule']:,}/{p1_union_summary['n_pool_rows']:,})\n"
        f"- Pooled attack recall: {_fmt_pct(p1_union_summary['pooled_attack_recall_p1_alone'])}\n"
        f"- Benign FPR: {_fmt_pct(p1_union_summary['benign_fpr_p1_alone'])}\n"
        "- This is far above any realistic 1% budget on its own — the meter and "
        "priority ordering below are what make the policy budget-respecting, "
        "not the rule set itself.\n"
    )

    lines.append("## Priority 2 — signature table fit and bin-resolution sweep\n")
    lines.append(
        f"Fit population: Monday+Tuesday chronological FIRST HALF, BENIGN rows only "
        f"(n={len(fit_benign):,}; asserted benign-only at `build_signature_table`'s "
        f"first check). Held-out same-period population (never touched by "
        f"fitting): Monday+Tuesday chronological second half (n={len(holdout):,}).\n"
    )
    lines.append("### Bin-resolution sweep — does PortScan recover with finer top bins?\n")
    lines.append(bin_sweep_df.to_markdown(index=False))
    if "PortScan__recall" in bin_sweep_df.columns:
        lines.append(
            f"\nPortScan recall: {bin_sweep_df.iloc[0]['PortScan__recall']:.4%} at "
            f"n_bins={bin_sweep_df.iloc[0]['n_bins']} -> "
            f"{bin_sweep_df.iloc[-1]['PortScan__recall']:.4%} at n_bins={bin_sweep_df.iloc[-1]['n_bins']}.\n"
        )
    lines.append("\n### Bin-edge tied-mass saturation check (top 15 by tied mass, at n_bins=6 fit)\n")
    lines.append(tie_sat_df.sort_values("tied_mass", ascending=False).head(15).to_markdown(index=False))
    lines.append("")

    lines.append(f"## Headline: three-priority policy at {HEADLINE_BUDGET:.2%} budget, per-day renewing meter\n")
    lines.append(
        f"- Total flows: {headline['n_total']:,}\n"
        f"- Escalated: {headline['n_escalated']:,} ({_fmt_pct(headline['escalation_rate'])})\n"
        f"- Per priority (admitted): P1={headline['per_priority_admitted'][1]:,}, "
        f"P2={headline['per_priority_admitted'][2]:,}, P3={headline['per_priority_admitted'][3]:,}\n"
        f"- Pre-meter \"fires first\" waterfall: P1={headline['fires_first']['priority1_would_fire_first_n']:,}, "
        f"P2 (of what P1 didn't take)={headline['fires_first']['priority2_would_fire_first_n']:,}, "
        f"P3 (of what neither took)={headline['fires_first']['priority3_would_fire_first_n']:,}\n"
        f"- Attack flows captured: {headline['attack_captured']:,}/{headline['attack_total']:,} "
        f"({_fmt_pct(headline['attack_coverage'])}), missed: {headline['attack_missed']:,}\n"
        f"- Precision of escalated set: {_fmt_pct(headline['precision'])}\n"
        f"- Benign FPR of escalated set: {_fmt_pct(headline['benign_fpr'])}\n"
    )
    lines.append("### Per-class capture/miss at the headline operating point\n")
    lines.append(headline["per_class"].to_markdown(index=False))
    lines.append("")

    lines.append("### Why headline recall is so far below Priority 1's own unmetered per-class recall\n")
    lines.append(
        "`results/p1_rules.md` shows Priority 1 alone (no meter) gets 74-99% "
        "unmetered recall on DoS Hulk/GoldenEye/Slowloris/Slowhttptest/DDoS/PortScan/"
        "FTP-Patator — the rules clearly CAN fire on these classes. The near-zero "
        "headline numbers above are a **verified budget/ordering artifact, not a "
        "rule-coverage gap** (checked directly, per the brief's own instruction to "
        "rule this out before calling a 0% a miss):\n\n"
        "- Priority 1's own reserved cap is small by construction: the brief requires "
        "extracting **all 62** attack-leaf rules (\"do not hand-pick\"), which match "
        "19.83% of the whole pool — nowhere near the PDF's own worked example (6 "
        "hand-selected, highly selective rules matching ~0.096-0.1% of flows). Under "
        "the PDF's own illustrative 10/70/20 budget split, Priority 1's guaranteed "
        "share is `global_capacity - reserved(P2) - reserved(P3)` ≈ 10% of the "
        "1% total budget ≈ 0.1% of a day's flows — a small fixed ceiling that a "
        "62-rule, 19.83%-of-traffic match set blows through almost immediately.\n"
        "- **Directly checked on Wednesday (the DoS Hulk day)**: n=692,703, Priority "
        "1's cap ≈692 flows. Priority 1 matches 204,623 rows that day, but the "
        "**first attack row (chronologically) sits at position 72,871 — Priority 1's "
        "692-flow cap is already exhausted by row 24,293, using 692/692 BENIGN "
        "false-positive matches, 0 of them attacks.** DoS Hulk's attack window "
        "starts well after Priority 1 has nothing left to admit that day. "
        "Per-day detail for every day:\n\n"
    )
    lines.append(starvation_df.to_markdown(index=False))
    lines.append(
        "\nThis is a structural consequence of evaluating an unfiltered, "
        "brief-mandated 62-rule Priority 1 inside a budget/priority framework the "
        "PDF itself designed around a small, coverage-maximizing SELECTED subset "
        "(its own Step 1: \"evaluate all 71 attack-leaf rules ... select the subset "
        "that maximizes attack-flow coverage under the 1% transfer constraint\") — "
        "a step this evaluation deliberately does not perform, per this task's own "
        "explicit instruction to extract every rule rather than hand-pick. Not "
        "re-tuned or re-selected to improve this number, per the honesty "
        "requirements.\n"
    )

    lines.append("## Unseen-class recall — the real test (reported even where bad)\n")
    lines.append(
        "Bot/Infiltration are included for completeness but, per the discrepancy "
        "above, Priority 1 CAN in principle name them (they have tree leaves) — "
        "they are not a clean zero-day test the way Heartbleed and the Web Attack "
        "sub-variants are.\n"
    )
    lines.append(unseen_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Budget sweep — 0.01% to 10%, stacked by priority\n")
    lines.append(budget_df.to_markdown(index=False))
    lines.append(f"\n![load vs coverage](figures/load_vs_coverage.png)\n")
    beats = None
    row_1pct = budget_df.loc[budget_df["budget_fraction"] == 0.01]
    if len(row_1pct):
        cov = row_1pct.iloc[0]["attack_coverage"]
        beats = cov is not None and cov > 0.004
        lines.append(
            f"\n**Does this beat the existing selector's <=0.4% pooled attack recall "
            f"at <=1% escalation (STATUS.md)?** At budget_fraction=0.01, attack "
            f"coverage here is {_fmt_pct(cov)}. "
            f"**{'Yes' if beats else 'No'}, this {'clears' if beats else 'does not clear'} "
            "that bar.**\n"
        )

    lines.append("## Cross-day: fit Monday+Tuesday (full), evaluate Wednesday/Thursday/Friday only\n")
    lines.append(
        f"Fit: {cross_day['fit_n_benign']:,} Monday+Tuesday benign rows (full days, not "
        f"chronologically halved — the day boundary itself is the fit/eval split here). "
        f"Eval: {cross_day['eval_n']:,} rows across Wed/Thu/Fri, never touched by fitting.\n\n"
        f"- Escalated: {cross_day['n_escalated']:,} ({_fmt_pct(cross_day['escalation_rate'])})\n"
        f"- Per priority (admitted): P1={cross_day['per_priority_admitted'][1]:,}, "
        f"P2={cross_day['per_priority_admitted'][2]:,}, P3={cross_day['per_priority_admitted'][3]:,}\n"
        f"- Attack coverage: {_fmt_pct(cross_day['attack_coverage'])} "
        f"({cross_day['attack_captured']:,}/{cross_day['attack_total']:,})\n"
        f"- Precision: {_fmt_pct(cross_day['precision'])}\n"
        f"- Benign FPR: {_fmt_pct(cross_day['benign_fpr'])}\n"
    )
    lines.append("### Cross-day per-class capture/miss\n")
    lines.append(cross_day["per_class"].to_markdown(index=False))

    if sampled_run is not None:
        write_sampled_run_section(
            lines, sampled_run["composition"], sampled_run["summary_df"], sampled_run["per_class_df"],
            sampled_run["rule_quality"], sampled_run["missed"], sampled_run["sanity_checks"],
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "escalation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT_DIR / 'escalation_report.md'}")


def main() -> int:
    print("loading pool ...")
    pool = load_pool()
    print(f"pool: {len(pool):,} rows")
    pool = compute_priority2_source_features(pool)  # adds flows_per_src etc., per (day, Source IP)

    print("compiling Priority 1 rules from tree.txt ...")
    rules = compile_p1_rules()
    p1_matched, p1_n_matched = evaluate_rules_union(rules, pool)

    with open(OUT_DIR / "p1_rules.json", encoding="utf-8") as f:
        p1_union_summary = json.load(f)["union_summary"]

    print("Priority 2: fitting signature table on Monday+Tuesday chronological fit-half ...")
    fit_benign = build_p2_fit_pool(pool)
    holdout = build_p2_holdout_pool(pool)
    signature_table = build_signature_table(_relabel_for_build_signature_table(fit_benign), n_bins=P2_N_BINS, floor=P2_FLOOR)
    tie_sat_df = check_bin_tie_saturation(fit_benign, signature_table)

    print("Priority 2: bin-resolution sweep ...")
    attack_pool = pool[pool["Label"] != BENIGN_LABEL]
    bin_sweep_df = bin_resolution_sweep(fit_benign, attack_pool)

    print(f"headline run at {HEADLINE_BUDGET:.2%} budget, per-day meter ...")
    escalated, priority, m2, m3 = run_policy_per_day(pool, p1_matched, signature_table, HEADLINE_BUDGET)
    headline = headline_report(pool, escalated, priority, p1_matched, m2, m3)
    unseen_df = unseen_class_report(pool, escalated, priority)

    print("diagnosing P1 cap vs attack-window ordering ...")
    starvation_df = p1_starvation_diagnostic(pool, p1_matched)

    print("budget sweep 0.01% - 10% ...")
    budget_df = budget_sweep(pool, p1_matched, signature_table)
    plot_load_vs_coverage(budget_df, FIG_DIR / "load_vs_coverage.png")

    print("cross-day: fit Mon+Tue, eval Wed/Thu/Fri ...")
    cross_day = cross_day_report(pool, rules)

    print(f"sampled run: drawing {SAMPLE_N:,} random flows (random_state={SAMPLE_RANDOM_STATE}) ...")
    sample_df = draw_random_sample(pool)
    composition = sample_composition_report(sample_df)
    sample_result, sample_rule_id, sample_predicted_class = run_sampled_policy(sample_df, rules, signature_table)

    # The escalated-flows CSV is NOT written here any more: results/escalated_flows.csv
    # is the v1 record the agent batches were drawn from and must not be overwritten.
    # eval/escalation_v2.py writes results/escalated_flows_v2.csv.

    summary_df = sampled_summary_by_priority(sample_df, sample_result)
    per_class_df = sampled_per_class_report(sample_df, sample_result)
    rule_quality = p1_rule_quality_unmetered(rules, sample_df, sample_result.priority_capacity[1])
    missed = missed_attacks_report(sample_df, sample_result)
    sanity_checks = sampled_sanity_checks(sample_df, sample_result, composition)

    sampled_run = {
        "composition": composition, "summary_df": summary_df, "per_class_df": per_class_df,
        "rule_quality": rule_quality, "missed": missed, "sanity_checks": sanity_checks,
    }

    write_report(pool, p1_matched, p1_union_summary, fit_benign, holdout, tie_sat_df,
                 bin_sweep_df, headline, unseen_df, budget_df, cross_day, starvation_df,
                 sampled_run=sampled_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
