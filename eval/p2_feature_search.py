"""Priority 2 feature diagnosis and search harness.

Everything here measures Priority 2 (benign-only signature table K, uncommon =
signature not in K) under the production meter. It changes no production call
site. Two kinds of number are kept apart:

- SELECTION objective: computed on the fit split only (Mon+Tue chronological
  first halves). It is the expected number of attacks admitted under the
  per-population cap, given the fit population's P2-eligible flagged rows.
- REPORT metrics: holdout benign escalation, the 300k sample's attacks per 100
  P2 slots under the real 2,400 cap, and per-class recall over the whole pool.
  These are never used to rank subsets.

Caveat that bounds every number here: the fit split contains SSH-Patator
attacks only (see results/escalation_report_v2.md, Part 2). The selection
objective therefore sees one attack class. Recall on the other classes comes
from data the search never selected on, but it is not a fit-split measurement.

Signatures are encoded as one int64 per row (mixed radix over the per-feature
digitize indices, with NaN as its own sentinel). This is equivalent to
dataplane.escalation_policy.signatures_for; tests/test_p2_feature_search.py
checks that equivalence on the production five features.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dataplane.escalation_policy import PRIORITY2_FEATURES, compute_priority2_source_features
from eval.escalation_data import BENIGN_LABEL
from eval.generalization_experiments import compute_bin_edges

#: The sample's Priority 2 cap: 2,400 slots out of a 300,000-flow sample.
P2_CAP_FRACTION = 2400 / 300_000
P2_SLOTS_SAMPLE = 2400

#: Identifiers and bookkeeping columns, never candidate signature features.
EXCLUDED_CANDIDATES = {
    "Source IP", "Destination IP", "Source Port", "Destination Port", "Dst Port",
    "row_idx", "first_ts", "weekday_idx", "day", "Label", "flow_uid", "timestamp_resolution_us",
    "_p2_source_ip", "_p2_bwd_pkt_len_mean", "_p2_pkt_len_range", "_p2_unanswered_syn",
}


def fast_codes(values: Dict[str, np.ndarray], features: Sequence[str], edges: Dict[str, np.ndarray]) -> np.ndarray:
    """One int64 per row. Per feature: digitize index in [0, len(edges)],
    with NaN mapped to its own sentinel, then combined in mixed radix."""
    n = len(next(iter(values.values())))
    code = np.zeros(n, dtype=np.int64)
    for f in features:
        v = values[f]
        idx = np.digitize(v, edges[f]).astype(np.int64) + 1  # 1..len(edges)+1
        idx[np.isnan(v)] = 0  # sentinel, distinct from every bin
        radix = len(edges[f]) + 2
        code = code * radix + idx
    return code


def signature_space(features: Sequence[str], edges: Dict[str, np.ndarray]) -> int:
    """Upper bound on distinct signatures: product of effective bin counts
    (edges + 1; NaN sentinels not counted)."""
    return int(math.prod(len(edges[f]) + 1 for f in features))


@dataclass(frozen=True)
class FitTable:
    """K built from benign fit rows, for one (features, n_bins, floor)."""
    features: Tuple[str, ...]
    n_bins: int
    floor: int
    edges: Dict[str, np.ndarray]
    common: np.ndarray  # sorted int64 codes in K


def build_fit_table(fit_benign_values: Dict[str, np.ndarray], features: Sequence[str], n_bins: int, floor: int) -> FitTable:
    edges = {}
    for f in features:
        vals = fit_benign_values[f]
        vals = vals[~np.isnan(vals)]
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges[f] = np.unique(np.quantile(vals, qs))
    codes = fast_codes(fit_benign_values, features, edges)
    uniq, counts = np.unique(codes, return_counts=True)
    common = np.sort(uniq[counts >= floor])
    return FitTable(tuple(features), n_bins, floor, edges, common)


def escalate(values: Dict[str, np.ndarray], table: FitTable) -> np.ndarray:
    codes = fast_codes(values, table.features, table.edges)
    return ~np.isin(codes, table.common, assume_unique=False)


class P2Evaluator:
    """Holds the splits once; every subset evaluation reuses them.

    fit_*     : Mon+Tue first halves (selection population)
    hold_*    : Mon+Tue second halves (report only)
    sample_*  : the 300k production sample, in shuffle order (report only)
    pool_*    : the whole pool, for per-class recall (report only)
    """

    def __init__(self, pool: pd.DataFrame, fit_mask: np.ndarray, hold_mask: np.ndarray,
                 sample_index: np.ndarray, p1_pool: np.ndarray, p1_sample: np.ndarray):
        """sample_index: pool row positions of the 300k sample, in shuffle
        order. p1_sample: P1 match mask aligned to sample_index."""
        self.pool = pool
        self.labels = pool["Label"].to_numpy()
        self.is_attack = self.labels != BENIGN_LABEL
        self.fit_mask = fit_mask
        self.hold_mask = hold_mask
        self.p1_pool = p1_pool
        self.sample_index = sample_index
        self.p1_sample = p1_sample
        self.n_fit = int(fit_mask.sum())
        self.cap_fit = P2_CAP_FRACTION * self.n_fit
        self.features_all = {c: pool[c].to_numpy(dtype=float) for c in candidate_columns(pool)}
        self.fit_benign = {k: v[fit_mask & ~self.is_attack] for k, v in self.features_all.items()}
        self.fit_elig = fit_mask & ~p1_pool
        self.hold_benign_mask = hold_mask & ~self.is_attack
        self.class_names = sorted(pd.unique(self.labels[self.is_attack]))
        self.class_n = {c: int((self.labels == c).sum()) for c in self.class_names}

    def _vals(self, idx, features):
        return {f: self.features_all[f][idx] for f in features}

    def fit_objective(self, features: Sequence[str], n_bins: int, floor: int) -> dict:
        """SELECTION. Uses fit rows only: expected attacks admitted under the
        fit-population cap, over P2-eligible flows (not P1-matched)."""
        table = build_fit_table(self.fit_benign, features, n_bins, floor)
        elig_idx = np.nonzero(self.fit_elig)[0]
        esc = escalate(self._vals(elig_idx, features), table)
        flagged = int(esc.sum())
        att_flagged = int((esc & self.is_attack[elig_idx]).sum())
        share = min(1.0, self.cap_fit / flagged) if flagged else 0.0
        return {
            "fit_flagged": flagged, "fit_attacks_flagged": att_flagged,
            "fit_expected_attacks_admitted": att_flagged * share,
            "fit_expected_attacks_per_100_slots": (100.0 * att_flagged / flagged) if flagged else None,
            "_table": table,
        }

    def evaluate(self, features: Sequence[str], n_bins: int, floor: int, *, report: bool = True) -> dict:
        """Full row. Selection columns from fit_objective; report columns from
        holdout, sample, and pool. Report columns never feed back into search."""
        fo = self.fit_objective(features, n_bins, floor)
        table = fo.pop("_table")
        row = {
            "features": "|".join(features), "n_features": len(features), "n_bins": n_bins, "floor": floor,
            "signature_space": signature_space(features, table.edges),
            "K_size": int(len(table.common)),
        }
        row.update(fo)
        if not report:
            return row
        row["K_over_space"] = row["K_size"] / row["signature_space"] if row["signature_space"] else None
        # holdout benign escalation (P2 on all holdout benign flows, as in v2)
        hb = np.nonzero(self.hold_benign_mask)[0]
        row["hold_benign_n"] = int(len(hb))
        row["hold_benign_p2_rate"] = float(escalate(self._vals(hb, features), table).mean())
        # sample: P2-eligible flows in shuffle order, first 2,400 flagged are admitted
        s_idx = self.sample_index  # already in shuffle order
        s_elig = ~self.p1_sample
        s_esc = escalate(self._vals(s_idx, features), table) & s_elig
        s_att = self.is_attack[s_idx]
        flagged_pos = np.nonzero(s_esc)[0][:P2_SLOTS_SAMPLE]
        row["sample_p2_flagged"] = int(s_esc.sum())
        row["sample_p2_admitted"] = int(len(flagged_pos))
        row["sample_p2_attacks_admitted"] = int(s_att[flagged_pos].sum())
        row["sample_attacks_per_100_slots"] = (
            100.0 * row["sample_p2_attacks_admitted"] / row["sample_p2_admitted"] if row["sample_p2_admitted"] else None)
        # per-class recall over the whole pool (K flags regardless of P1)
        all_idx = np.arange(len(self.labels))
        pool_esc = escalate(self._vals(all_idx, features), table)
        for c in self.class_names:
            m = self.labels == c
            row[f"recall__{c}"] = float(pool_esc[m].mean())
            row[f"n__{c}"] = self.class_n[c]
        return row


def candidate_columns(pool: pd.DataFrame) -> List[str]:
    """Numeric per-flow columns usable as signature features."""
    out = []
    for c in pool.columns:
        if c in EXCLUDED_CANDIDATES:
            continue
        if not pd.api.types.is_numeric_dtype(pool[c]):
            continue
        out.append(c)
    return out


def prepare_pool(full_pool: pd.DataFrame) -> pd.DataFrame:
    """Adds the five production P2 features (per day and source IP)."""
    from eval.escalation_data import add_priority2_columns
    p = add_priority2_columns(full_pool)
    p = compute_priority2_source_features(p)
    return p


def greedy_forward(ev: P2Evaluator, candidates: Sequence[str], n_bins: int, floor: int, max_size: int,
                   log=None) -> Tuple[Dict[int, List[str]], List[dict]]:
    """Greedy forward selection on the FIT objective only. Returns the path
    (size -> subset) and every evaluated (candidate, size) fit row."""
    chosen: List[str] = []
    path: Dict[int, List[str]] = {}
    seen: List[dict] = []
    for size in range(1, max_size + 1):
        best = None
        for f in candidates:
            if f in chosen:
                continue
            trial = chosen + [f]
            fo = ev.fit_objective(trial, n_bins, floor)
            fo.pop("_table")
            fo.update({"features": "|".join(trial), "n_features": len(trial), "n_bins": n_bins, "floor": floor,
                       "stage": "forward"})
            seen.append(fo)
            key = (fo["fit_expected_attacks_admitted"], -candidates.index(f))
            if best is None or key > best[0]:
                best = (key, f)
        chosen.append(best[1])
        path[size] = list(chosen)
        if log:
            log(f"forward size {size}: +{best[1]} (fit expected attacks {best[0][0]:.1f})")
    return path, seen


def backward_eliminate(ev: P2Evaluator, start: Sequence[str], n_bins: int, floor: int, min_size: int,
                       log=None) -> Tuple[Dict[int, List[str]], List[dict]]:
    """Backward pass from `start`: at each size, drop the feature whose removal
    keeps the FIT objective highest. Fit only."""
    cur = list(start)
    path = {len(cur): list(cur)}
    seen: List[dict] = []
    while len(cur) > min_size:
        best = None
        for f in cur:
            trial = [x for x in cur if x != f]
            fo = ev.fit_objective(trial, n_bins, floor)
            fo.pop("_table")
            fo.update({"features": "|".join(trial), "n_features": len(trial), "n_bins": n_bins, "floor": floor,
                       "stage": "backward"})
            seen.append(fo)
            key = (fo["fit_expected_attacks_admitted"], f)
            if best is None or key > best[0]:
                best = (key, f)
        cur = [x for x in cur if x != best[1]]
        path[len(cur)] = list(cur)
        if log:
            log(f"backward size {len(cur)}: -{best[1]}")
    return path, seen


# ---------------------------------------------------------------------------
# Full-column pool (all 87 CSV columns), built once and cached.
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

FULL_POOL_PATH = Path("results/cache/p2_full_pool.parquet")
_ID_COLS = {"Flow ID", "Timestamp", "Label", "Source IP", "Destination IP", "Source Port", "Destination Port"}


def build_full_pool(path: Path = FULL_POOL_PATH) -> pd.DataFrame:
    """Every numeric CICFlowMeter column from the eight CSVs, with the same
    label-drop and timestamp-drop rules as eval.escalation_data.load_pool, so
    rows line up with the production pool. Verified against it in
    tests/test_p2_feature_search.py."""
    from adapters.csv_flow_adapter import load_csv, parse_timestamp
    from eval.escalation_data import CSV_DIR, DAY_FILES

    frames = []
    for day, fn, wd in DAY_FILES:
        raw = load_csv(CSV_DIR / fn).dropna(subset=["Label"]).reset_index(drop=True)
        feats = [c for c in raw.columns if c not in _ID_COLS]
        num = raw[feats].apply(pd.to_numeric, errors="coerce")
        first = raw["Timestamp"].apply(parse_timestamp).apply(lambda t: t[0] if t else np.nan)
        keep = first.notna().to_numpy()
        df = pd.concat([
            raw.loc[keep, ["Label", "Source IP", "Destination IP", "Source Port", "Destination Port"]].reset_index(drop=True),
            num.loc[keep].reset_index(drop=True),
        ], axis=1)
        df["first_ts"] = first[keep].to_numpy()
        df["day"] = day
        df["weekday_idx"] = wd
        df["row_idx"] = np.arange(len(df))
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(path, index=False)
    return full


def load_full_pool(path: Path = FULL_POOL_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return build_full_pool(path)


# ---------------------------------------------------------------------------
# Context and runner.
# ---------------------------------------------------------------------------

from eval.sweep import split_monday_chronologically  # noqa: E402
from eval.escalation_data import BENIGN_LABEL as _BENIGN  # noqa: E402

RESULTS = Path("results")
TREE_FEATURES = (
    "Bwd Packets/s", "Flow IAT Mean", "Fwd Packet Length Max", "Init_Win_bytes_backward",
    "Init_Win_bytes_forward", "Packet Length Mean", "Subflow Fwd Packets", "Total Backward Packets",
    "act_data_pkt_fwd", "min_seg_size_forward",
)
SAMPLE_N_RUN = 300_000
SAMPLE_SEED = 42


def make_context():
    """Loads the full pool, splits, P1 matches, and the production-equivalent
    300k sample (same draw as eval.escalation_eval.draw_random_sample: the
    draw depends only on pool length and random_state)."""
    from dataplane.dt_rules import evaluate_rules_union
    from eval.escalation_eval import compile_p1_rules

    pool = prepare_pool(load_full_pool()).reset_index(drop=True)
    n = len(pool)
    mon = pool[pool["day"] == "monday"]
    tue = pool[pool["day"] == "tuesday"]
    fit_mask = np.zeros(n, dtype=bool)
    hold_mask = np.zeros(n, dtype=bool)
    mf, mh = split_monday_chronologically(mon)
    tf, th = split_monday_chronologically(tue)
    fit_mask[np.concatenate([mf.index.to_numpy(), tf.index.to_numpy()])] = True
    hold_mask[np.concatenate([mh.index.to_numpy(), th.index.to_numpy()])] = True

    rules = compile_p1_rules()
    p1_pool, _ = evaluate_rules_union(rules, pool)
    drawn = pd.Series(np.arange(n)).sample(n=SAMPLE_N_RUN, random_state=SAMPLE_SEED)
    sample_index = drawn.sample(frac=1, random_state=SAMPLE_SEED).to_numpy()
    p1_sample = p1_pool[sample_index]
    ev = P2Evaluator(pool, fit_mask, hold_mask, sample_index, p1_pool, p1_sample)
    return ev, pool, rules


def candidate_list(ev: P2Evaluator) -> Tuple[List[str], List[str], List[str]]:
    """(production five, tree ten, other per-flow numeric columns)."""
    five = list(PRIORITY2_FEATURES)
    tree = [f for f in TREE_FEATURES if f in ev.features_all]
    other = sorted(c for c in ev.features_all if c not in set(five) | set(tree))
    return five, tree, other


def sample_p2_precision(ev: P2Evaluator, features: Sequence[str], n_bins: int, floor: int,
                        all_eligible: bool = False) -> dict:
    """Exact P2 admission on the 300k sample: P2-eligible flows in shuffle
    order, first 2,400 flagged are admitted. all_eligible=True disables P1."""
    table = build_fit_table(ev.fit_benign, features, n_bins, floor)
    s_idx = ev.sample_index
    elig = np.ones(len(s_idx), dtype=bool) if all_eligible else ~ev.p1_sample
    esc = escalate({f: ev.features_all[f][s_idx] for f in features}, table) & elig
    pos = np.nonzero(esc)[0][:P2_SLOTS_SAMPLE]
    att = ev.is_attack[s_idx][pos]
    adm = len(pos)
    return {"eligible_n": int(elig.sum()), "admitted": adm, "attacks_admitted": int(att.sum()),
            "precision": (att.sum() / adm) if adm else None}


def part1_baseline(ev: P2Evaluator) -> dict:
    s_idx = ev.sample_index
    s_att = ev.is_attack[s_idx]
    p1 = ev.p1_sample
    elig = ~p1
    n = len(s_idx)
    elig_att = int((elig & s_att).sum())
    base = elig_att / int(elig.sum())
    prod = sample_p2_precision(ev, list(PRIORITY2_FEATURES), 20, 5)
    off = sample_p2_precision(ev, list(PRIORITY2_FEATURES), 20, 5, all_eligible=True)
    k = P2_SLOTS_SAMPLE
    rnd_sd = (base * (1 - base) / k) ** 0.5
    return {
        "sample_n": n, "sample_attacks": int(s_att.sum()), "sample_attack_frac": float(s_att.mean()),
        "p1_matched_n": int(p1.sum()), "p1_matched_attacks": int((p1 & s_att).sum()),
        "p2_eligible_n": int(elig.sum()), "p2_eligible_attacks": elig_att,
        "p2_eligible_attack_frac": float(base),
        "random_sampler_precision_on_eligible": float(base),
        "random_sampler_sd_at_2400_slots": float(rnd_sd),
        "p2_production": prod,
        "p2_production_ratio_vs_eligible_baseline": (prod["precision"] / base) if prod["precision"] is not None else None,
        "p2_p1_disabled": off,
        "p2_p1_disabled_ratio_vs_sample_attack_frac": (off["precision"] / float(s_att.mean())) if off["precision"] else None,
    }


def discrimination_table(ev: P2Evaluator, candidates: Sequence[str]) -> pd.DataFrame:
    """Per candidate and attack class: AUC (P(attack > benign), ties half) and
    KS statistic, benign reference = Mon+Tue first-half benign (the K
    population). Attack rows = every pool row of that class."""
    from scipy.stats import ks_2samp
    rows = []
    for f in candidates:
        b_all = ev.fit_benign[f]
        b = np.sort(b_all[~np.isnan(b_all)])
        full = ev.features_all[f]
        for c in ev.class_names:
            m = (ev.labels == c) & ~np.isnan(full)
            a = full[m]
            if len(a) == 0 or len(b) == 0:
                continue
            lo = np.searchsorted(b, a, side="left")
            hi = np.searchsorted(b, a, side="right")
            auc = float(((lo + hi) / 2.0).mean() / len(b))
            ks = float(ks_2samp(b, a).statistic)
            rows.append({"feature": f, "attack_class": c, "n_attack": int(len(a)), "n_benign_ref": int(len(b)),
                         "auc": auc, "auc_separation": abs(auc - 0.5) * 2, "ks": ks})
    return pd.DataFrame(rows)


def redundancy_table(ev: P2Evaluator, features: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame({f: ev.fit_benign[f] for f in features})
    return df.corr(method="spearman")


def run_search(ev: P2Evaluator, five: List[str], tree: List[str], other: List[str],
               log=print) -> Tuple[pd.DataFrame, dict]:
    """Greedy forward (size 1..8) on fit at n_bins=20, floor=5, then backward
    from the size-8 set. Per size, the chosen subset is the better of the
    forward and backward subsets on the fit objective. Top three by fit
    objective get an n_bins re-sweep. Returns all report rows and a summary."""
    cands = list(five) + list(tree) + list(other)
    log(f"candidates: {len(cands)} (five {len(five)}, tree {len(tree)}, other {len(other)})")
    fwd_path, fwd_seen = greedy_forward(ev, cands, 20, 5, 8, log=log)
    bwd_path, bwd_seen = backward_eliminate(ev, fwd_path[8], 20, 5, 3, log=log)

    per_size = {}
    for s in range(3, 9):
        opts = [("forward", fwd_path[s])] + ([("backward", bwd_path[s])] if s in bwd_path else [])
        scored = []
        for stage, feats in opts:
            fo = ev.fit_objective(feats, 20, 5)
            fo.pop("_table")
            scored.append((fo["fit_expected_attacks_admitted"], stage, feats))
        scored.sort(key=lambda t: (-t[0], t[1]))
        per_size[s] = {"fit_expected_attacks_admitted": scored[0][0], "stage": scored[0][1], "features": scored[0][2]}
    top3 = sorted(per_size.items(), key=lambda kv: -kv[1]["fit_expected_attacks_admitted"])[:3]

    # Full report rows for every evaluated subset (fit objective recomputed in evaluate).
    rows: Dict[Tuple[str, int, int], dict] = {}

    def add(features, nb, fl, stage):
        key = ("|".join(features), nb, fl)
        if key in rows:
            rows[key]["stage"] = rows[key]["stage"] + "," + stage if stage not in rows[key]["stage"] else rows[key]["stage"]
            return
        r = ev.evaluate(list(features), nb, fl)
        r["stage"] = stage
        rows[key] = r

    add(five, 20, 5, "baseline_production_five")
    for stage, seen in (("forward", fwd_seen), ("backward", bwd_seen)):
        for fo in seen:
            add(fo["features"].split("|"), fo["n_bins"], fo["floor"], stage)
    for s in range(3, 9):
        add(per_size[s]["features"], 20, 5, f"chosen_size_{s}")
    log(f"full report rows so far: {len(rows)}")
    for size, info in top3:
        for nb in (6, 10, 20, 40):
            add(info["features"], nb, 5, f"nbins_sweep_size_{size}")
    log(f"total rows: {len(rows)}")
    df = pd.DataFrame(list(rows.values()))
    summary = {
        "per_size_chosen": {str(s): v for s, v in per_size.items()},
        "top3_by_fit_objective": [{"size": s, **v} for s, v in top3],
        "forward_path": {str(s): v for s, v in fwd_path.items()},
        "backward_path": {str(s): v for s, v in bwd_path.items()},
    }
    return df, summary


def main() -> int:
    import json
    import time

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log("building context ...")
    ev, pool, rules = make_context()
    five, tree, other = candidate_list(ev)

    log("Part 1: baselines ...")
    p1 = part1_baseline(ev)

    log("Part 2: leave-one-out on the five ...")
    loo_rows = [ev.evaluate(list(PRIORITY2_FEATURES), 20, 5)]
    loo_rows[0]["stage"] = "baseline_production_five"
    for f in five:
        feats = [x for x in five if x != f]
        r = ev.evaluate(feats, 20, 5)
        r["stage"] = f"loo_without_{f}"
        loo_rows.append(r)

    log("Part 2: per-feature discrimination ...")
    disc = discrimination_table(ev, five + tree + other)
    disc.to_csv(RESULTS / "p2_feature_discrimination.csv", index=False)

    log("Part 2: redundancy among the five ...")
    redundancy_table(ev, five).to_csv(RESULTS / "p2_feature_redundancy.csv")

    log("Part 3: greedy forward + backward search ...")
    search_df, summary = run_search(ev, five, tree, other, log=log)

    loo_df = pd.DataFrame(loo_rows)
    all_df = pd.concat([loo_df, search_df], ignore_index=True)
    all_df.to_csv(RESULTS / "p2_feature_search.csv", index=False)
    summary["part1"] = p1
    summary["candidates"] = {"five": five, "tree": tree, "other": other}
    with open(RESULTS / "p2_feature_search_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"wrote results/p2_feature_search.csv ({len(all_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
