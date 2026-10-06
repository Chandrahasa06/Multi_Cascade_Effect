"""Shared metrics for comparing P2 escalation runs (v1, v2, v3, n_bins probes).

Headline: per-(source, attack class) incident coverage. A controller needs one
flow per incident to identify an attack and act; it does not need all 231,073
DoS Hulk flows. Per-flow coverage is kept alongside it.

The per-source-IP metric is deliberately absent. It counts any escalated flow
from an attacking IP, benign flows included, so it is saturated (10/10) under
every meter and measures nothing.

Denominators are always reported with each number. Metrics are computed over a
frame whose rows are in the order the meter saw them, and every function takes
the admitted mask in that same order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.escalation_data import BENIGN_LABEL

WED_FRI = (2, 3, 4)
MICROS_PER_MINUTE = 60 * 1_000_000


def class_recall(admitted: np.ndarray, df: pd.DataFrame, days=WED_FRI) -> dict:
    """Per attack class over the given days: n, admitted count, recall."""
    wd = df["weekday_idx"].to_numpy()
    lab = df["Label"].to_numpy()
    in_days = np.isin(wd, list(days))
    out = {}
    for c in sorted(pd.unique(lab[in_days & (lab != BENIGN_LABEL)])):
        m = in_days & (lab == c)
        n = int(m.sum())
        k = int((admitted & m).sum())
        out[c] = {"n": n, "admitted": k, "recall": (k / n) if n else None}
    return out


def incident_coverage(admitted: np.ndarray, df: pd.DataFrame, days=WED_FRI) -> dict:
    """Per-(source, class) incident coverage, plus per-flow coverage, over the
    given days. A pair is covered if any admitted flow shares its source IP and
    attack class. Time to first escalation is measured per class from that
    class's first attack flow to the first admitted flow of that class."""
    wd = df["weekday_idx"].to_numpy()
    lab = df["Label"].to_numpy()
    src = df["Source IP"].astype(str).to_numpy()
    t = df["first_ts"].to_numpy(dtype=float)
    in_days = np.isin(wd, list(days))
    att = in_days & (lab != BENIGN_LABEL)
    adm = admitted & in_days

    pairs_att = {}
    for s, c, ts in zip(src[att], lab[att], t[att]):
        key = (s, c)
        pairs_att[key] = min(ts, pairs_att.get(key, np.inf))
    pairs_adm = set(zip(src[adm & (lab != BENIGN_LABEL)], lab[adm & (lab != BENIGN_LABEL)]))
    covered = [k for k in pairs_att if k in pairs_adm]

    first_att_class = {}
    for c, ts in zip(lab[att], t[att]):
        first_att_class[c] = min(ts, first_att_class.get(c, np.inf))
    first_adm_class = {}
    for c, ts in zip(lab[adm & (lab != BENIGN_LABEL)], t[adm & (lab != BENIGN_LABEL)]):
        first_adm_class[c] = min(ts, first_adm_class.get(c, np.inf))
    latencies = {c: (first_adm_class[c] - first_att_class[c]) / MICROS_PER_MINUTE
                 for c in first_att_class if c in first_adm_class}

    n_att_flows = int(att.sum())
    n_att_adm = int((adm & att).sum())
    return {
        "attack_pairs_n": len(pairs_att),
        "attack_pairs_covered": len(covered),
        "pair_coverage": (len(covered) / len(pairs_att)) if pairs_att else None,
        "attack_flows_n": n_att_flows,
        "attack_flows_admitted": n_att_adm,
        "flow_coverage": (n_att_adm / n_att_flows) if n_att_flows else None,
        "classes_with_attack_n": len(first_att_class),
        "classes_escalated_n": len(latencies),
        "median_first_escalation_min": float(np.median(list(latencies.values()))) if latencies else None,
        "latency_min_by_class": latencies,
    }


def precision(admitted: np.ndarray, df: pd.DataFrame, days=WED_FRI) -> dict:
    """Over the given days: admitted, attacks admitted, precision (of admitted)."""
    wd = df["weekday_idx"].to_numpy()
    lab = df["Label"].to_numpy()
    in_days = np.isin(wd, list(days))
    adm = admitted & in_days
    n_adm = int(adm.sum())
    n_att = int((adm & (lab != BENIGN_LABEL)).sum())
    return {"admitted_n": n_adm, "attacks_admitted": n_att,
            "precision": (n_att / n_adm) if n_adm else None}
