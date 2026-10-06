"""Compiles every non-Benign leaf of ``tree.txt`` into a standalone
interval rule (``dataplane/dt_rules.py``) and reports empirical support
and purity for each, measured against the full CICIDS2017 CSV pool
(``eval/escalation_data.py``).

tree.txt carries no per-node sample counts at all (plain
``export_text``, no ``show_weights`` -- see eval/parse_tree.py's
docstring), so "leaf support/purity" cannot be read off the tree itself.
It is instead computed empirically here: for each rule, how many pool
rows match its interval conjunction, and what fraction of those matches
carry the label the tree predicts for that leaf (label coarsened to the
tree's own 12-class vocabulary via
``eval.escalation_data.label_to_tree_class``, since e.g. "Web Attack -
XSS" can never literally equal the tree's undifferentiated "Web Attack"
class otherwise).

Run: python -m eval.p1_rules
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from dataplane.dt_rules import DERIVED_FEATURES, INTEGER_FEATURES, Rule, compile_attack_rules, evaluate_rules_union
from eval.escalation_data import BENIGN_LABEL, label_to_tree_class, load_pool
from eval.parse_tree import load_tree

OUT_JSON = Path("results/p1_rules.json")
OUT_MD = Path("results/p1_rules.md")


def rule_to_row(rule: Rule, df: pd.DataFrame, tree_classes: pd.Series) -> dict:
    mask = rule.matches(df)
    support = int(mask.sum())
    if support:
        purity = float((tree_classes[mask] == rule.predicted_class).mean())
    else:
        purity = None
    return {
        "rule_id": rule.id,
        "leaf_id": rule.leaf_id,
        "predicted_class": rule.predicted_class,
        "intervals": {f: iv.describe() for f, iv in sorted(rule.intervals.items())},
        "depends_on_derived": rule.depends_on_derived,
        "n_features_constrained": len(rule.intervals),
        "empirical_support_n": support,
        "empirical_support_frac_of_pool": support / len(df) if len(df) else None,
        "empirical_purity": purity,
    }


def build_report(rules: List[Rule], df: pd.DataFrame) -> dict:
    tree_classes = df["Label"].map(label_to_tree_class)
    rows = [rule_to_row(r, df, tree_classes) for r in rules]

    matched_any, n_matched = evaluate_rules_union(rules, df)
    is_attack = (df["Label"] != BENIGN_LABEL).to_numpy()
    n_multi_match = int((n_matched > 1).sum())

    union_summary = {
        "n_pool_rows": len(df),
        "n_rows_matching_any_rule": int(matched_any.sum()),
        "pool_escalation_rate": float(matched_any.mean()) if len(df) else None,
        "n_rows_matching_multiple_rules": n_multi_match,
        "benign_rows_matching_any_rule": int((matched_any & ~is_attack).sum()),
        "attack_rows_matching_any_rule": int((matched_any & is_attack).sum()),
        "n_attack_rows_total": int(is_attack.sum()),
        "pooled_attack_recall_p1_alone": (
            float((matched_any & is_attack).sum() / is_attack.sum()) if is_attack.sum() else None
        ),
        "benign_fpr_p1_alone": (
            float((matched_any & ~is_attack).sum() / (~is_attack).sum()) if (~is_attack).sum() else None
        ),
    }
    return {"rules": rows, "union_summary": union_summary}


def write_markdown(report: dict, rules: List[Rule]) -> None:
    lines: List[str] = []
    lines.append("# Priority 1 — decision-tree rule extraction (`tree.txt`)\n")
    lines.append(
        "62 non-Benign leaves of `tree.txt`, each compiled into a standalone "
        "interval rule via `dataplane/dt_rules.py`. **Extraction is exhaustive "
        "— every attack-leaf path, not a hand-picked subset.** tree.txt carries "
        "no per-node sample counts, so support/purity below are measured "
        "empirically against the full 2,830,743-row CICIDS2017 pool "
        "(`eval/escalation_data.py`), not read from the tree.\n"
    )
    lines.append(
        "**Feature-set note**: tree.txt's actual 10 features differ from the "
        "task brief's original list in 3 of 10 (it uses `Packet Length Mean`, "
        "`Subflow Fwd Packets`, `Total Backward Packets` instead of `Avg Bwd "
        "Segment Size`, `Bwd Header Length`, `Fwd Packet Length Std`) — "
        "confirmed with the user, tree.txt is treated as ground truth "
        "throughout (see `eval/parse_tree.py`'s docstring).\n"
    )
    lines.append(
        f"**Derived features requiring cross-multiplication in a real switch "
        f"deployment**: {sorted(DERIVED_FEATURES)} (a rate and two means — "
        "`b/dt <= t` becomes `b <= t*dt`; `mean <= t` becomes `sum <= t*n`). "
        "Rules that constrain any of these are flagged in `depends_on_derived` "
        "below; every other feature is a raw integer register "
        f"({sorted(INTEGER_FEATURES)}) needing no rearrangement.\n"
    )

    u = report["union_summary"]
    lines.append("## Union summary — Priority 1 alone, no P2/P3, no meter\n")
    lines.append(
        f"- Pool: {u['n_pool_rows']:,} rows ({u['n_attack_rows_total']:,} attack, "
        f"{u['n_pool_rows'] - u['n_attack_rows_total']:,} benign)\n"
        f"- Rows matching >=1 rule: {u['n_rows_matching_any_rule']:,} "
        f"({u['pool_escalation_rate']:.4%} of pool)\n"
        f"- Rows matching >1 rule (deduplicated in the union above): "
        f"{u['n_rows_matching_multiple_rules']:,}\n"
        f"- Pooled attack recall (P1 alone): **{u['pooled_attack_recall_p1_alone']:.4%}** "
        f"({u['attack_rows_matching_any_rule']:,}/{u['n_attack_rows_total']:,})\n"
        f"- Benign FPR (P1 alone, no meter): **{u['benign_fpr_p1_alone']:.4%}** "
        f"({u['benign_rows_matching_any_rule']:,}/{u['n_pool_rows'] - u['n_attack_rows_total']:,})\n"
    )

    lines.append("## Rule table\n")
    lines.append(
        "| rule | predicted class | leaf id | support (n) | support (% pool) | purity | "
        "derived features | intervals |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    for row in report["rules"]:
        ivs = "; ".join(f"{f}: {v}" for f, v in row["intervals"].items())
        purity_s = f"{row['empirical_purity']:.1%}" if row["empirical_purity"] is not None else "n/a (0 support)"
        derived_s = ", ".join(row["depends_on_derived"]) or "none"
        lines.append(
            f"| {row['rule_id']} | {row['predicted_class']} | {row['leaf_id']} | "
            f"{row['empirical_support_n']:,} | {row['empirical_support_frac_of_pool']:.4%} | "
            f"{purity_s} | {derived_s} | {ivs} |"
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = load_tree()
    rules = compile_attack_rules(root)
    print(f"compiled {len(rules)} attack-leaf rules")

    print("loading CSV pool (cached if available) ...")
    df = load_pool()
    print(f"pool: {len(df):,} rows")

    report = build_report(rules, df)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, rules)

    u = report["union_summary"]
    print(f"P1 alone: {u['pool_escalation_rate']:.4%} escalation, "
          f"{u['pooled_attack_recall_p1_alone']:.4%} pooled attack recall, "
          f"{u['benign_fpr_p1_alone']:.4%} benign FPR")
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
