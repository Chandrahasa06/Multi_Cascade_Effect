"""CSV-path feature extraction for the SpliDT model (see
``dataplane/splidt_inference.py`` for the traversal reimplementation and
its "what we lose running this way" caveats — read that module's
docstring first, this one assumes it).

Reads CICIDS2017 TrafficLabelling CSVs directly via
``adapters.csv_flow_adapter.load_csv`` (leading-space stripping, cp1252
fallback, already handled there) and renames columns into the model's own
CICIDS2018-era feature-name space. Does NOT run the flow simulation or
packet synthesis in that adapter module — a CSV row's own already-computed
CICFlowMeter columns are used directly, unlike every other CSV-path
consumer in this project (``eval/simulate.py`` and friends), which feed
synthesized packets through ``FlowTable`` to derive this project's own
Tier-1 features instead. That machinery is irrelevant here: SpliDT wants
CICFlowMeter-shaped features, and these CSVs already are CICFlowMeter
output.

=== Feature name mapping (pre-flight check 1) ===

All 41 features referenced anywhere in ``model.pkl`` map to a CICIDS2017
CSV column with high confidence — verified programmatically against the
real header list, not eyeballed. See ``MODEL_FEATURE_TO_CSV_COLUMN``.

=== Flag counts are BINARY in this CSV, not true counts (pre-flight check 2) ===

`PSH/FIN/ACK/SYN/RST/URG Flag Count` are 0/1 presence indicators in this
CICIDS2017 release (checked: max=1, n_unique<=2 for all six, every day).
Cross-checked against ``model.pkl``'s own fitted thresholds: 56 of the
66 splits on these six features use thresholds >= 1 (ACK Flag Count up
to 1114.0, PSH up to 49.5) — reachable only with genuine per-packet
counts. Feeding this CSV's binary values makes those 56 splits dead
(permanently forced down the "<=" branch; 13 of 45 subtrees contain at
least one). This proves the model was trained on genuine per-packet flag
counts (matching ``cicids-2017_p0.pkl``'s semantics — see the earlier,
timestamp-blocked attempt at that path), not this CSV's own flag columns.
Carried through downstream as a real accuracy ceiling, not silently
patched over: no correction is applied here, the raw binary values are
used as-is (converting a 0/1 flag column into a synthetic multi-valued
count would itself be a bigger, unjustified assumption).

=== Rate columns: Infinity is kept, NaN is dropped and reported ===

``Flow Bytes/s`` and ``Flow Packets/s`` are undefined at zero duration.
Checked directly, both columns, all 8 days:

- ``Flow Packets/s`` is NEVER NaN (a valid flow always has >=1 packet, so
  packets/0-duration is +Infinity, never 0/0).
- ``Flow Bytes/s`` is NaN exactly when a flow has BOTH zero duration AND
  zero payload bytes (0/0) — rare (64 rows on Monday, 1,008 on Wednesday,
  concentrated in DoS Hulk and BENIGN) — and Infinity when duration is
  zero but bytes are nonzero.

These are handled differently on purpose. Decision-tree threshold splits
use ``<=``/`>`; under IEEE-754, ``+inf > threshold`` is True for any
finite threshold, so **Infinity is kept as-is** — it is the single most
faithful encoding available of "an extremely fast, near-instantaneous
flow," which is exactly what a zero-duration nonzero-byte flow is. NaN
has no such well-defined reading (``NaN <= t`` and ``NaN > t`` are both
False, which would silently and arbitrarily route every NaN row down the
">" branch of any split touching that feature) and represents a
genuinely different condition (a flow that moved zero bytes, e.g. a bare
SYN with no response) — so NaN rows are **dropped, per-day and per-label
counts reported** (see :func:`rate_quality_report`), never silently
``dropna``'d without that report existing.

=== Training-set contamination (pre-flight check 3) ===

Flow ID format matches exactly between this CSV pool and
``cicids-2017_p0.pkl`` (``srcIP-dstIP-srcPort-dstPort-protocol``).
:func:`load_pickle_train_flow_ids` extracts the pickle's distinct
training Flow IDs (needs the ``Int64Index -> pd.Index`` shim, applied
here since pandas 3.0.5 dropped that module entirely — pandas <2.0 at
pickle time still referenced it). :func:`mark_contamination` flags, never
silently drops, every CSV row whose Flow ID was in that set — measured at
15.07% of the full CSV pool overall but ~80% for nearly every attack
class (attack tooling reuses a narrow 5-tuple far more than benign
browsing does) and 100% for Heartbleed (11/11). Callers MUST exclude
``contaminated=True`` rows before computing any recall/FPR number, or
report contaminated and clean populations separately — never blend them.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Dict, FrozenSet, List

import numpy as np
import pandas as pd

from adapters.csv_flow_adapter import load_csv, parse_timestamp_us

CSV_DIR = Path("data/csv/TrafficLabelling")
PICKLE_PATH = Path("cicids-2017_p0.pkl")

#: (day key -> CSV filename). Matches the 8 files actually shipped in
#: data/csv/TrafficLabelling/ (note Wednesday's lowercase "workingHours"
#: and the original release's "Infilteration" misspelling — the path is
#: used as given, not "fixed").
DAY_FILES: Dict[str, str] = {
    "monday": "Monday-WorkingHours.pcap_ISCX.csv",
    "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
    "thursday_webattacks": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "thursday_infiltration": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "friday_morning": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "friday_portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "friday_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}

#: model feature name -> CICIDS2017 CSV column name. All 41 features
#: referenced anywhere in model.pkl; verified 41/41 against the real
#: header list (see module docstring).
MODEL_FEATURE_TO_CSV_COLUMN: Dict[str, str] = {
    "Dst Port": "Destination Port",
    "ACK Flag Count": "ACK Flag Count",
    "Average Packet Size": "Average Packet Size",
    "Bwd Header Length": "Bwd Header Length",
    "Bwd IAT Max": "Bwd IAT Max",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "Bwd IAT Min": "Bwd IAT Min",
    "Bwd IAT Total": "Bwd IAT Total",
    "Bwd Packet Length Max": "Bwd Packet Length Max",
    "Bwd Packets/s": "Bwd Packets/s",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Down/Up Ratio": "Down/Up Ratio",
    "FIN Flag Count": "FIN Flag Count",
    "Flow Bytes/s": "Flow Bytes/s",
    "Flow Duration": "Flow Duration",
    "Flow IAT Max": "Flow IAT Max",
    "Flow IAT Mean": "Flow IAT Mean",
    "Flow IAT Min": "Flow IAT Min",
    "Flow Packets/s": "Flow Packets/s",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Header Length": "Fwd Header Length",
    "Fwd IAT Max": "Fwd IAT Max",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Fwd IAT Min": "Fwd IAT Min",
    "Fwd IAT Total": "Fwd IAT Total",
    "Fwd Packet Length Max": "Fwd Packet Length Max",
    "Fwd Packet Length Mean": "Fwd Packet Length Mean",
    "Fwd Packet Length Min": "Fwd Packet Length Min",
    "Fwd Packets/s": "Fwd Packets/s",
    "Fwd Seg Size Min": "min_seg_size_forward",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "PSH Flag Count": "PSH Flag Count",
    "Packet Length Max": "Max Packet Length",
    "Packet Length Mean": "Packet Length Mean",
    "RST Flag Count": "RST Flag Count",
    "SYN Flag Count": "SYN Flag Count",
    "Total Bwd packets": "Total Backward Packets",
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "URG Flag Count": "URG Flag Count",
}

FLAG_COUNT_FEATURES: FrozenSet[str] = frozenset(
    {"PSH Flag Count", "FIN Flag Count", "ACK Flag Count", "SYN Flag Count", "RST Flag Count", "URG Flag Count"}
)

RATE_FEATURES_WITH_INF_NAN = ("Flow Bytes/s", "Flow Packets/s")

#: model.pkl's own classes_ (union across all 45 subtrees) — the 10
#: labels this model can name at all.
MODEL_CLASSES = frozenset(
    {
        "Benign", "DoS-Slowloris", "FTP-Patator", "Heartbleed", "Infiltration",
        "PortScan", "SSH-Patator", "WebAttack-BruteForce", "WebAttack-SQLInjection", "WebAttack-XSS",
    }
)

#: present in CICIDS2017 but absent from the model's classes_ entirely —
#: Priority 1 recall on these is zero by construction (see eval/splidt_escalation_eval.py).
UNSEEN_CLASSES = frozenset({"Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest"})


def normalize_label(raw_label: str) -> str:
    """CICIDS2017's raw CSV label -> the model's own class-name spelling
    where the model has that class at all; unseen classes keep their own
    CSV-native spelling (own name, not forced into a made-up model-style
    name) so the coverage-gap analysis can name them directly."""
    label = str(raw_label).strip()
    if label == "BENIGN":
        return "Benign"
    if label == "DoS slowloris":
        return "DoS-Slowloris"
    low = label.lower()
    if "brute force" in low:
        return "WebAttack-BruteForce"
    if "sql injection" in low:
        return "WebAttack-SQLInjection"
    if "xss" in low:
        return "WebAttack-XSS"
    return label  # FTP-Patator, SSH-Patator, Heartbleed, Infiltration, PortScan,
    # and the 5 unseen classes (Bot, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest)
    # already match their CSV spelling exactly.


def load_day(day_key: str) -> pd.DataFrame:
    path = CSV_DIR / DAY_FILES[day_key]
    return load_csv(path)


def load_all_days() -> Dict[str, pd.DataFrame]:
    return {day: load_day(day) for day in DAY_FILES}


#: Priority 2 (dataplane/escalation_policy.py) needs a handful of raw
#: CSV columns that are NOT part of the SpliDT model's own 41 features
#: (Source IP identifies a signature's source; the other three feed the
#: 5-feature signature table carried over from
#: eval/generalization_experiments.py's Experiment 2 — see
#: escalation_policy.py's module docstring for why they're recomputed
#: directly from this same CSV pool instead of joined against the
#: project's simulated parquet cache). Prefixed `_p2_` so they can never
#: collide with a model feature name.
_PRIORITY2_RAW_COLUMNS = {
    "_p2_source_ip": "Source IP",
    "_p2_bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "_p2_pkt_len_max": "Max Packet Length",
    "_p2_pkt_len_min": "Min Packet Length",
    "_p2_syn_flag": "SYN Flag Count",
    "_p2_ack_flag": "ACK Flag Count",
    "_p2_bwd_packets": "Total Backward Packets",
}


def extract_model_features(df: pd.DataFrame, day_key: str) -> pd.DataFrame:
    """Rename CSV columns into model feature-name space; carry Flow ID,
    normalized Label, day_key, and the raw Priority-2 helper columns
    through unchanged. No cleaning here — see :func:`clean_rate_features`
    for the Inf/NaN decision."""
    out = pd.DataFrame({name: df[col].astype(float) for name, col in MODEL_FEATURE_TO_CSV_COLUMN.items()})
    out["Flow ID"] = df["Flow ID"].astype(str)
    out["Label"] = df["Label"].astype(str).str.strip().map(normalize_label)
    out["day"] = day_key
    out["_p2_source_ip"] = df["Source IP"].astype(str)
    for name, col in _PRIORITY2_RAW_COLUMNS.items():
        if name == "_p2_source_ip":
            continue
        out[name] = pd.to_numeric(df[col], errors="coerce")
    out["_p2_unanswered_syn"] = (
        (out["_p2_syn_flag"] >= 1) & (out["_p2_ack_flag"] == 0) & (out["_p2_bwd_packets"] == 0)
    ).astype(float)
    out["_p2_pkt_len_range"] = out["_p2_pkt_len_max"] - out["_p2_pkt_len_min"]
    # needed for chronological (not random) splits, e.g. Monday fit/holdout
    # (eval/sweep.py::split_monday_chronologically's pattern) — parsed the
    # same way adapters/csv_flow_adapter.py does for the simulated path.
    out["first_ts"] = df["Timestamp"].map(parse_timestamp_us)
    return out


def rate_quality_report(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Per-label counts of Infinity and NaN in the two rate features,
    for the report — computed BEFORE any cleaning."""
    rows = []
    for label, group in feature_df.groupby("Label"):
        row = {"label": label, "n": len(group)}
        for feat in RATE_FEATURES_WITH_INF_NAN:
            vals = group[feat].to_numpy(dtype=float)
            row[f"{feat}__n_inf"] = int(np.isinf(vals).sum())
            row[f"{feat}__n_nan"] = int(np.isnan(vals).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def clean_rate_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN in either rate feature (genuinely undefined —
    see module docstring); Infinity is left as-is (a real, tree-comparison
    -correct value). Returns the cleaned frame; caller should have already
    called :func:`rate_quality_report` on the uncleaned frame to know what
    was dropped."""
    nan_mask = np.zeros(len(feature_df), dtype=bool)
    for feat in RATE_FEATURES_WITH_INF_NAN:
        nan_mask |= feature_df[feat].isna().to_numpy()
    return feature_df.loc[~nan_mask].reset_index(drop=True)


def _install_pickle_shim() -> None:
    """pandas 3.0.5 removed pandas.core.indexes.numeric entirely; the
    pickle was written under pandas <2.0, which still had Int64Index
    there. Register a minimal stand-in so pickle.load can resolve it."""
    mod_name = "pandas.core.indexes.numeric"
    if mod_name in sys.modules and hasattr(sys.modules[mod_name], "Int64Index"):
        return
    mod = types.ModuleType(mod_name)

    class Int64Index(pd.Index):
        pass

    mod.Int64Index = Int64Index
    sys.modules[mod_name] = mod


def load_pickle_train_flow_ids(path=PICKLE_PATH) -> FrozenSet[str]:
    """Distinct Flow IDs in cicids-2017_p0.pkl's ungrouped_train_df — the
    professor's own training set. Format matches this project's CSV Flow
    ID exactly (checked): "srcIP-dstIP-srcPort-dstPort-protocol"."""
    _install_pickle_shim()
    import pickle

    with open(path, "rb") as f:
        pkl = pickle.load(f)
    return frozenset(pkl["ungrouped_train_df"]["Flow ID"].unique().tolist())


def mark_contamination(feature_df: pd.DataFrame, train_flow_ids: FrozenSet[str]) -> pd.DataFrame:
    """Adds a `contaminated` column — never drops rows itself. Callers
    must exclude contaminated=True before computing recall/FPR, or report
    the two populations separately (see module docstring)."""
    out = feature_df.copy()
    out["contaminated"] = out["Flow ID"].isin(train_flow_ids)
    return out


def contamination_report(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in feature_df.groupby("Label"):
        n = len(group)
        n_contaminated = int(group["contaminated"].sum())
        rows.append({
            "label": label, "n": n, "n_contaminated": n_contaminated,
            "contamination_rate": n_contaminated / n if n else None,
            "n_clean": n - n_contaminated,
        })
    return pd.DataFrame(rows)


def build_full_pool(exclude_contaminated: bool = False) -> pd.DataFrame:
    """Load all 8 days, extract+rename features, clean rates, mark
    contamination. Does NOT drop contaminated rows unless asked — the
    eval script decides how to report them."""
    train_flow_ids = load_pickle_train_flow_ids()
    frames = []
    for day_key in DAY_FILES:
        raw = load_day(day_key)
        feats = extract_model_features(raw, day_key)
        feats = clean_rate_features(feats)
        feats = mark_contamination(feats, train_flow_ids)
        frames.append(feats)
    pool = pd.concat(frames, ignore_index=True)
    if exclude_contaminated:
        pool = pool.loc[~pool["contaminated"]].reset_index(drop=True)
    return pool
