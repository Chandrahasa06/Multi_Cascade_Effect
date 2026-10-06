"""Escalation policy v3: the production chronological pipeline.

Three P2 decisions, all wired:
  - n_bins = P2_N_BINS (6, from results/p2_final.md Part 1)
  - token-bucket meter, burst DEFAULT_BURST_SECONDS (the default of the policy)
  - rule-A refit: hourly, unbounded K, updated from P2-admitted flows only,
    with the verdict from dataplane.controller_verdict.

The 300k sample is the same draw as v1 and v2 (SAMPLE_RANDOM_STATE 42), sorted
by time and NOT shuffled. Chronology is the point: shuffling is what produced
the false n_bins conclusion.

UPPER BOUND: the verdict is LabelStandInVerdict, a perfect controller. Every
v3 number is an upper bound on what refit can give. The agent verdict will
replace it later through the same seam.

Writes results/escalated_flows_v3.csv (never touches v1 or v2) and
results/escalation_v3_k_sizes.csv (K size per clock hour).

Run: python -m eval.escalation_v3
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataplane.controller_verdict import LabelStandInVerdict
from dataplane.dt_rules import evaluate_rules_union, first_matching_rule
from dataplane.escalation_policy import (
    PRIORITY2_FEATURES,
    OnlineSignatureTable,
    compute_priority2_source_features,
    run_policy_token_refit,
)
from eval.escalation_data import load_pool
from eval.escalation_eval import (
    P2_FLOOR,
    P2_N_BINS,
    SAMPLE_BUDGET,
    DEFAULT_TAU,
    PRIORITY_FRACTIONS,
    _relabel_for_build_signature_table,
    build_escalated_flows_csv,
    build_p2_fit_pool,
    compile_p1_rules,
    draw_random_sample,
)

V3_CSV = Path("results/escalated_flows_v3.csv")
K_SIZES_CSV = Path("results/escalation_v3_k_sizes.csv")


def run_v3(verdict=None):
    """Returns (result, table, k_sizes, chronological sample, rule_id, predicted_class)."""
    verdict = verdict or LabelStandInVerdict()
    pool = compute_priority2_source_features(load_pool())
    rules = compile_p1_rules()
    fit_benign = build_p2_fit_pool(pool)
    sample = draw_random_sample(pool)
    sample = sample.sort_values("first_ts", kind="stable").reset_index(drop=True)
    p1 = evaluate_rules_union(rules, sample)[0]
    rule_id, predicted_class = first_matching_rule(rules, sample)
    table = OnlineSignatureTable.from_benign_fit(
        _relabel_for_build_signature_table(fit_benign), n_bins=P2_N_BINS, floor=P2_FLOOR,
        features=PRIORITY2_FEATURES)
    result, table, k_sizes = run_policy_token_refit(
        sample, p1, table, verdict, tau=DEFAULT_TAU, global_budget_fraction=SAMPLE_BUDGET,
        priority_fractions=PRIORITY_FRACTIONS)
    return result, table, k_sizes, sample, rule_id, predicted_class


def main() -> int:
    result, table, k_sizes, sample, rule_id, predicted_class = run_v3()
    csv = build_escalated_flows_csv(sample, result, rule_id, predicted_class)
    csv.to_csv(V3_CSV, index=False)
    pd.DataFrame(k_sizes, columns=["clock_hour", "K_size_at_hour_start"]).to_csv(K_SIZES_CSV, index=False)
    print(f"wrote {V3_CSV} ({len(csv)} rows); K size {len(table.common)} at end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
