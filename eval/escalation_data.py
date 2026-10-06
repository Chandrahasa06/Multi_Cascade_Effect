"""Loads and pools the 8 CICIDS2017 TrafficLabelling CSVs for the
three-priority escalation evaluation (``eval/escalation_eval.py``).

Column mapping is NOT a guess: every one of tree.txt's 10 actual features
(see ``eval/parse_tree.py`` for the feature-set discrepancy already
surfaced and resolved with the user) matches a real CSV header
byte-for-byte after ``adapters.csv_flow_adapter.load_csv``'s
leading-space strip -- printed and checked in ``check_feature_columns``
below, no fuzzy/renamed mapping needed.

Feature values are read directly from the CSV columns (not derived via
the flow-table simulation/adapter packet synthesis) -- the brief is
explicit that the adapter's documented 100-600x distortion only affects
per-source COUNT features, none of which are among these 10.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from adapters.csv_flow_adapter import load_csv, parse_timestamp

CSV_DIR = Path("data/csv/TrafficLabelling")
CACHE_PATH = Path("results/cache/escalation_pool.parquet")

#: (day label, filename, weekday-order index -- used for the Monday+Tuesday
#: fit / Wed-Thu-Fri eval cross-day split). Thursday's two files and
#: Friday's three files share a weekday index; ordering within a weekday
#: doesn't matter for that split, only which weekday a row belongs to.
DAY_FILES: List[Tuple[str, str, int]] = [
    ("monday", "Monday-WorkingHours.pcap_ISCX.csv", 0),
    ("tuesday", "Tuesday-WorkingHours.pcap_ISCX.csv", 1),
    ("wednesday", "Wednesday-workingHours.pcap_ISCX.csv", 2),
    ("thursday_webattacks", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", 3),
    ("thursday_infiltration", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", 3),
    ("friday_morning", "Friday-WorkingHours-Morning.pcap_ISCX.csv", 4),
    ("friday_portscan", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 4),
    ("friday_ddos", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 4),
]

WEEKDAY_NAMES = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}

#: tree.txt's 10 features -- confirmed to match these exact CSV headers,
#: see check_feature_columns().
TREE_FEATURE_COLUMNS = [
    "Bwd Packets/s",
    "Flow IAT Mean",
    "Fwd Packet Length Max",
    "Init_Win_bytes_backward",
    "Init_Win_bytes_forward",
    "Packet Length Mean",
    "Subflow Fwd Packets",
    "Total Backward Packets",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
]

#: extra columns needed for Priority 2 (its own 5 signature features,
#: recomputed per-day-per-source, see dataplane/escalation_policy.py's
#: existing PRIORITY2_FEATURES machinery) and for identifiers/labels/
#: rate-feature-audit columns.
EXTRA_COLUMNS = [
    "Flow ID", "Source IP", "Destination IP", "Source Port", "Destination Port",
    "Timestamp", "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
    "Total Fwd Packets", "Label",
]

#: raw CSV columns needed to build Priority 2's 5 signature features via
#: dataplane.escalation_policy.compute_priority2_source_features (which
#: expects columns named "_p2_source_ip"/"Dst Port"/"_p2_unanswered_syn"/
#: "_p2_bwd_pkt_len_mean"/"_p2_pkt_len_range" -- see that module's
#: docstring). The unanswered-SYN proxy (SYN seen, no ACK, no backward
#: packet at all) matches eval/splidt_features.py's own definition
#: exactly, reused here for consistency rather than inventing a second
#: one.
PRIORITY2_RAW_COLUMNS = [
    "Bwd Packet Length Mean", "Max Packet Length", "Min Packet Length",
    "SYN Flag Count", "ACK Flag Count",
]


def add_priority2_columns(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["_p2_source_ip"] = out["Source IP"].astype(str)
    out["Dst Port"] = pd.to_numeric(out["Destination Port"], errors="coerce")
    out["_p2_bwd_pkt_len_mean"] = pd.to_numeric(out["Bwd Packet Length Mean"], errors="coerce")
    pkt_max = pd.to_numeric(out["Max Packet Length"], errors="coerce")
    pkt_min = pd.to_numeric(out["Min Packet Length"], errors="coerce")
    out["_p2_pkt_len_range"] = pkt_max - pkt_min
    syn = pd.to_numeric(out["SYN Flag Count"], errors="coerce")
    ack = pd.to_numeric(out["ACK Flag Count"], errors="coerce")
    out["_p2_unanswered_syn"] = ((syn >= 1) & (ack == 0) & (out["Total Backward Packets"] == 0)).astype(float)
    return out

BENIGN_LABEL = "BENIGN"

#: CSV ground-truth labels the tree's own leaves cannot produce even in
#: principle -- checked against tree.json's actual class list, NOT
#: assumed from the task brief (which claimed Bot/Infiltration were also
#: absent; they are not -- see eval/p1_rules.py's report). Heartbleed has
#: no leaf at all; the three Web Attack CSV sub-labels collapse to one
#: undifferentiated "Web Attack" tree leaf class.
GENUINELY_UNSEEN_LABELS = ("Heartbleed",)
WEB_ATTACK_SUBLABELS = ("Web Attack – Brute Force", "Web Attack – XSS", "Web Attack – Sql Injection")


def check_feature_columns(columns: List[str]) -> None:
    missing = [f for f in TREE_FEATURE_COLUMNS if f not in columns]
    if missing:
        raise ValueError(
            f"tree feature(s) not found verbatim in CSV columns: {missing}. "
            "Do not guess a rename -- STOP and report (see task brief)."
        )


@dataclass
class ZeroDurationReport:
    per_day: pd.DataFrame  # day, n_rows, n_zero_duration, n_zero_duration_pct
    per_day_label: pd.DataFrame  # day, label, n_zero_duration
    rate_feature_inf_nan: pd.DataFrame  # day, feature, n_inf, n_nan


def _clean_one_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    check_feature_columns(list(df.columns))
    df = df.dropna(subset=["Label"]).reset_index(drop=True)  # WebAttacks' 62.9% blank rows
    numeric_cols = TREE_FEATURE_COLUMNS + PRIORITY2_RAW_COLUMNS + [
        "Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Total Fwd Packets",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ts = df["Timestamp"].apply(parse_timestamp)
    df["first_ts"] = ts.apply(lambda t: t[0] if t else np.nan)
    df["timestamp_resolution_us"] = ts.apply(lambda t: t[1] if t else np.nan)
    n_unparsed = int(df["first_ts"].isna().sum())
    if n_unparsed:
        df = df.dropna(subset=["first_ts"]).reset_index(drop=True)
    df["day"] = day
    df["row_idx"] = np.arange(len(df))
    df["flow_uid"] = df["day"] + "::" + df["row_idx"].astype(str) + "::" + df["Flow ID"].astype(str)
    base_keep = ["day", "row_idx", "flow_uid", "Label", "first_ts", "timestamp_resolution_us",
                 "Source IP", "Destination IP", "Source Port", "Destination Port",
                 "Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Total Fwd Packets"]
    keep = list(dict.fromkeys(base_keep + TREE_FEATURE_COLUMNS + PRIORITY2_RAW_COLUMNS))
    return df[keep], n_unparsed


def zero_duration_report(pool: pd.DataFrame) -> ZeroDurationReport:
    rows_day, rows_day_label, rows_rate = [], [], []
    for day, sub in pool.groupby("day", sort=False):
        zero = sub["Flow Duration"] == 0
        rows_day.append({
            "day": day, "n_rows": len(sub), "n_zero_duration": int(zero.sum()),
            "pct_zero_duration": (100.0 * zero.sum() / len(sub)) if len(sub) else 0.0,
        })
        for label, cnt in sub.loc[zero, "Label"].value_counts().items():
            rows_day_label.append({"day": day, "label": label, "n_zero_duration": int(cnt)})
        for feat in ("Flow Bytes/s", "Flow Packets/s", "Bwd Packets/s"):
            v = sub[feat].to_numpy(dtype=float)
            rows_rate.append({
                "day": day, "feature": feat,
                "n_inf": int(np.isinf(v).sum()), "n_nan": int(np.isnan(v).sum()),
                "n_inf_or_nan_at_zero_duration": int(((np.isinf(v) | np.isnan(v)) & zero.to_numpy()).sum()),
            })
    return ZeroDurationReport(
        per_day=pd.DataFrame(rows_day),
        per_day_label=pd.DataFrame(rows_day_label),
        rate_feature_inf_nan=pd.DataFrame(rows_rate),
    )


def load_pool(*, use_cache: bool = True, csv_dir: Path = CSV_DIR) -> pd.DataFrame:
    if use_cache and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    frames = []
    for day, filename, weekday_idx in DAY_FILES:
        path = csv_dir / filename
        raw = load_csv(path)
        clean, n_unparsed = _clean_one_day(raw, day)
        clean["weekday_idx"] = weekday_idx
        clean["weekday_name"] = WEEKDAY_NAMES[weekday_idx]
        if n_unparsed:
            print(f"  {day}: dropped {n_unparsed} rows with unparseable Timestamp")
        frames.append(clean)
        del raw

    pool = pd.concat(frames, ignore_index=True)
    # Bwd Packets/s never carries Inf/NaN empirically (checked across all
    # 8 days, unlike Flow Bytes/s and Flow Packets/s which do at zero
    # duration) -- still guard explicitly rather than trust that finding
    # silently forever.
    bad = ~np.isfinite(pool["Bwd Packets/s"].to_numpy(dtype=float))
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} rows have non-finite Bwd Packets/s -- the "
            "'never Inf/NaN' finding this loader relies on doesn't hold; "
            "handle explicitly rather than silently dropping/filling."
        )

    pool = add_priority2_columns(pool)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(CACHE_PATH)
    return pool


def label_to_tree_class(label: str) -> str:
    """Coarsens the 15 CSV ground-truth labels to the tree's own 12-class
    vocabulary (the 3 Web Attack sub-labels collapse to one; Heartbleed
    has no tree equivalent and is passed through unchanged so it can
    never spuriously match a tree class)."""
    if label.startswith("Web Attack"):
        return "Web Attack"
    return label


def split_fit_eval_by_weekday(pool: pd.DataFrame, fit_weekdays=(0, 1), eval_weekdays=(2, 3, 4)):
    fit = pool[pool["weekday_idx"].isin(fit_weekdays)].reset_index(drop=True)
    ev = pool[pool["weekday_idx"].isin(eval_weekdays)].reset_index(drop=True)
    return fit, ev
