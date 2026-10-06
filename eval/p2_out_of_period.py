"""Out-of-period validation of two Priority 2 feature changes (see
results/p2_out_of_period.md for the pre-registered decision rule).

Evaluates each configuration on Wednesday, Thursday and Friday, per day and
pooled. K is always built from Monday+Tuesday first-half benign rows, as in
production, with n_bins=20 and floor=5 (not tuned here).

Writes results/p2_out_of_period.csv, one row per (configuration, day).

Run: python -m eval.p2_out_of_period
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from eval.p2_feature_search import P2_SLOTS_SAMPLE, build_fit_table, escalate, signature_space

#: The production five, written out explicitly so this evaluation does not
#: drift when PRIORITY2_FEATURES changes.
ORIGINAL_FIVE: Sequence[str] = (
    "flows_per_src",
    "distinct_dst_ports_per_src",
    "syn_without_synack_count",
    "bwd_pkt_len_mean",
    "pkt_len_range",
)
DAY_NUMBERS: Dict[str, int] = {"Wed": 2, "Thu": 3, "Fri": 4}
N_BINS = 20
FLOOR = 5
OUT_CSV = Path("results/p2_out_of_period.csv")


def configurations() -> Dict[str, List[str]]:
    c1 = [f for f in ORIGINAL_FIVE if f != "syn_without_synack_count"]
    return {
        "R_five_production": list(ORIGINAL_FIVE),
        "C1_four_minus_syn": c1,
        "C2_minus_pkt_len_range": [f for f in c1 if f != "pkt_len_range"],
        "C3_minus_bwd_pkt_len_mean": [f for f in c1 if f != "bwd_pkt_len_mean"],
    }


def out_of_period_mask(weekday_idx: np.ndarray) -> np.ndarray:
    """Wednesday, Thursday and Friday rows only. Monday and Tuesday are never
    evaluated here (they are the fit population)."""
    return np.isin(weekday_idx, [2, 3, 4])


def chronological_cap(rows: np.ndarray, first_ts: np.ndarray, row_idx: np.ndarray,
                      eligible: np.ndarray, flagged: np.ndarray, is_attack: np.ndarray) -> dict:
    """Admit the first 2,400 P2-eligible flagged rows in time order. Returns
    admitted count, fill rate (admitted / 2,400), and attacks admitted."""
    order = rows[np.lexsort((row_idx[rows], first_ts[rows]))]
    order = order[eligible[order]]
    fl = order[flagged[order]]
    adm = fl[:P2_SLOTS_SAMPLE]
    return {
        "cap_eligible_n": int(len(order)),
        "flagged_total": int(len(fl)),
        "admitted": int(len(adm)),
        "fill_rate": len(adm) / P2_SLOTS_SAMPLE,
        "attacks_admitted": int(is_attack[adm].sum()),
        "attacks_per_100_slots": (100.0 * is_attack[adm].sum() / len(adm)) if len(adm) else None,
    }


def evaluate_unit(rows: np.ndarray, flagged: np.ndarray, labels: np.ndarray, is_attack: np.ndarray,
                  first_ts: np.ndarray, row_idx: np.ndarray, eligible: np.ndarray) -> dict:
    """Benign escalation, per-class recall with counts, and the filled-cap
    admission for one evaluation unit (a day, or Wed-Fri pooled)."""
    benign = rows[~is_attack[rows]]
    att = rows[is_attack[rows]]
    out = {
        "benign_n": int(len(benign)),
        "benign_escalated": int(flagged[benign].sum()),
        "benign_rate": float(flagged[benign].mean()) if len(benign) else None,
    }
    for c in sorted(pd.unique(labels[att])):
        m = att[labels[att] == c]
        out[f"recall__{c}__n"] = int(len(m))
        out[f"recall__{c}__caught"] = int(flagged[m].sum())
        out[f"recall__{c}"] = float(flagged[m].mean()) if len(m) else None
    out.update(chronological_cap(rows, first_ts, row_idx, eligible, flagged, is_attack))
    return out


def evaluate_configurations(ev, pool: pd.DataFrame) -> pd.DataFrame:
    """ev: an eval.p2_feature_search.P2Evaluator built over `pool`."""
    wd = pool["weekday_idx"].to_numpy()
    first_ts = pool["first_ts"].to_numpy()
    row_idx = pool["row_idx"].to_numpy()
    labels = ev.labels
    is_attack = ev.is_attack
    eligible = ~ev.p1_pool
    wf = out_of_period_mask(wd)
    rows_out = []
    for name, feats in configurations().items():
        table = build_fit_table(ev.fit_benign, feats, N_BINS, FLOOR)
        flagged = escalate({f: ev.features_all[f] for f in feats}, table)
        sp = signature_space(feats, table.edges)
        base = {"config": name, "n_features": len(feats), "K_size": int(len(table.common)),
                "signature_space": sp}
        for day, dn in DAY_NUMBERS.items():
            rows = np.nonzero(wd == dn)[0]
            rows_out.append({**base, "day": day, **evaluate_unit(rows, flagged, labels, is_attack, first_ts, row_idx, eligible)})
        rows = np.nonzero(wf)[0]
        rows_out.append({**base, "day": "pooled_Wed_Fri", **evaluate_unit(rows, flagged, labels, is_attack, first_ts, row_idx, eligible)})
    return pd.DataFrame(rows_out)


def main() -> int:
    from eval.p2_feature_search import make_context
    ev, pool, _ = make_context()
    df = evaluate_configurations(ev, pool)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
