"""Logistic-regression control on the same four benign-only signals the agents get.

Why this control exists: if the agents match a simple function of the same inputs,
the reasoning layer adds nothing a linear model cannot. If they beat it, they weigh
something the linear model cannot, and the disagreement records show what.

Fitting without attack data. A two-class logistic model needs two classes, and the
pipeline may not use attack rows. So the control is fitted to separate REAL benign
flows (class 1) from SYNTHETIC flows (class 0). A synthetic flow takes each escalation
feature independently from the benign marginal, so it keeps every value's marginal
distribution and loses the joint structure. The model learns "how typical is this
combination", which is a benign-only density-ratio classifier. It is not trained to
recognise attacks, and its output is scored against the same 0.30 rule as the agents.

No attack row is read in fitting. The 45 evaluation records are excluded from the
training sample by feature identity, so the control is not fitted on the records it is
scored on. The sample size (2000 real, 2000 synthetic) and the regularisation (C=1.0)
are fixed in advance and are not tuned.

Writes results/logistic_baseline_45.json (per-record probabilities, the fit summary).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from agents import evidence_signals as es
from eval.escalation_data import CACHE_PATH
from eval.run_blind_pipeline_45_v2 import load_45

OUT = Path("results/logistic_baseline_45.json")
N_PER_CLASS = 2000
SEED = 0
C = 1.0
THRESHOLD = 0.30  # the pipeline's verdict threshold, unchanged
NO_NEIGHBOUR_DISTANCE = 10.0  # placeholder when no non-identical benign neighbour exists


def signal_vector(b: es.EvidenceBundle) -> np.ndarray:
    """Fixed-length vector built only from the four signals and the count."""
    pct = np.array([b.percentiles[f] / 100.0 for f in es.FEATURES])
    high = sum(1 for f in es.FEATURES if b.percentiles[f] >= es.TAIL_HIGH_PCT)
    low = sum(1 for f in es.FEATURES if b.percentiles[f] <= es.TAIL_LOW_PCT)
    count = np.log1p(b.count_excl)
    if b.neighbours:
        d = [nb["distance"] for nb in b.neighbours]
        d_first, d_mean = np.log1p(d[0]), np.log1p(np.mean(d))
    else:
        d_first = d_mean = np.log1p(NO_NEIGHBOUR_DISTANCE)
    if b.history is None:
        hist = np.array([0.0, 0.0, 0.0, 0.0])
    else:
        hist = np.array([
            np.log1p(sum(h["flows"] for h in b.history)),
            np.log1p(sum(h["distinct_dst_ports"] for h in b.history)),
            np.log1p(sum(h["packets"] for h in b.history)),
            1.0,
        ])
    return np.concatenate([pct, [high, low, count, d_first, d_mean], hist])


def _bundle_for(ref, values: Dict[str, float], source) -> es.EvidenceBundle:
    return es.bundle(ref, values, source)


def main() -> int:
    records = load_45()
    eval_ids = {r["record_id"] for r in records}
    pool = es.load_pool_slim(extra_columns=records[0]["features"].keys())
    ref = es.build_reference(pool)
    match_cols = [c for c in records[0]["features"] if c in pool.columns]

    # training sample: benign pool rows, excluding any row that is one of the 45 evaluation flows
    benign = pool.loc[pool["Label"] == es.BENIGN].reset_index(drop=True)
    eval_x = np.array([[float(r["features"][c]) for c in match_cols] for r in records])
    bx = benign[match_cols].to_numpy(dtype=np.float64)
    in_eval = np.zeros(len(benign), dtype=bool)
    for row in eval_x:
        in_eval |= np.all(bx == row, axis=1)
    candidates = np.flatnonzero(~in_eval)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(candidates, size=N_PER_CLASS, replace=False)
    sample = benign.iloc[pick].reset_index(drop=True)

    esc = list(es.FEATURES)
    real_rows, real_src, X, y = [], [], [], []
    for _, row in sample.iterrows():
        values = {f: float(row[f]) for f in esc}
        src = (str(row["Source IP"]), int(row["first_ts"]))
        b = _bundle_for(ref, values, src)
        real_rows.append(values)
        real_src.append(src)
        X.append(signal_vector(b))
        y.append(1)

    # synthetic: each escalation feature independently permuted across the sampled real rows
    shuffled = {f: rng.permutation(sample[f].to_numpy(dtype=np.float64)) for f in esc}
    for i in range(N_PER_CLASS):
        values = {f: float(shuffled[f][i]) for f in esc}
        b = _bundle_for(ref, values, real_src[i])
        X.append(signal_vector(b))
        y.append(0)
    X = np.vstack(X)
    y = np.array(y)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(X_tr), y_tr)
    val_acc = float(model.score(scaler.transform(X_va), y_va))

    eval_scores = {}
    for r in records:
        values = {f: float(r["features"][f]) for f in esc}
        src, status = es.locate_source(pool, r["features"], match_cols)
        b = _bundle_for(ref, values, src)
        p = float(model.predict_proba(scaler.transform(signal_vector(b)[None, :]))[0, 1])
        eval_scores[r["record_id"]] = {"p_real": p, "source_status": status}

    OUT.write_text(json.dumps({
        "n_per_class": N_PER_CLASS, "seed": SEED, "C": C, "threshold": THRESHOLD,
        "validation_accuracy_real_vs_synthetic": val_acc,
        "coefficients": dict(zip(
            [f"pct:{f}" for f in esc] + ["n_tail_high", "n_tail_low", "log_count_excl",
                                         "log_nn_distance_first", "log_nn_distance_mean",
                                         "log_hist_flows", "log_hist_ports", "log_hist_packets", "hist_available"],
            model.coef_[0].tolist())),
        "scores": eval_scores,
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT}: validation accuracy real-vs-synthetic {val_acc:.3f} (n_val={len(y_va)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
