"""PCAP-path fit/sweep/comparison — the real-packet-timing counterpart
to the CSV-path pipeline (eval/simulate.py + eval/sweep.py + eval/report.py),
and the honest version of the same question the CSV path answers.

Fitting is Monday PCAP only, chronologically split (same discipline as
eval/sweep.py's CSV-path split): the first half fits benign-only
thresholds, the second half measures held-out benign escalation
out-of-sample. Friday PCAP (joined to its three CICIDS2017 CSV sessions
via eval/labels_pcap.py) plus Monday's own holdout half form the
operational set per-class recall is measured against — mirroring the
CSV path's "week_df" concept, just scoped to the two days we have real
captures for.

``exclude_low_confidence`` is passed through explicitly rather than
left at fit_thresholds_from_frame's default: real packet timing means
every PCAP-derived flow's timestamp_resolution_us is 1 (microsecond),
so low_confidence is never True for these features in the first place —
see the docstring on run_pcap_sweep for why this is expected to be a
no-op relative to the CSV path, and how that's checked, not assumed.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from dataplane.fitting import FittingResult
from dataplane.selector import EscalationRule
from eval.labels_pcap import join_friday_sessions, load_session_index
from eval.report import per_class_recall_table
from eval.simulate_pcap import PCAP_PATHS, extract_day_features_pcap, label_all_benign
from eval.sweep import (
    PERCENTILE_SWEEP,
    SweepResult,
    run_sweep,
    split_monday_chronologically,
)

FRIDAY_SESSION_CSVS = {
    "bot": "data/csv/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "portscan": "data/csv/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "ddos": "data/csv/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}


#: see eval/labels_pcap.py's load_session_index docstring: these two
#: Friday CSVs record afternoon Timestamps in unmarked 12-hour format.
FRIDAY_SESSION_PM_CORRECTION = {"bot": False, "portscan": True, "ddos": True}


def load_friday_pcap(force: bool = False) -> Tuple[pd.DataFrame, dict]:
    sessions = [
        load_session_index(name, path, pm_hour_correction=FRIDAY_SESSION_PM_CORRECTION[name])
        for name, path in FRIDAY_SESSION_CSVS.items()
    ]

    def label_friday(closed_flows):
        result = join_friday_sessions(closed_flows, sessions)
        return result.labels, result.to_meta_dict()

    return extract_day_features_pcap("friday", PCAP_PATHS["friday"], label_friday, force=force)


def load_monday_pcap(force: bool = False) -> Tuple[pd.DataFrame, dict]:
    return extract_day_features_pcap("monday", PCAP_PATHS["monday"], label_all_benign, force=force)


def run_pcap_sweep(
    exclude_low_confidence: bool = False,
) -> Tuple[SweepResult, Dict[float, FittingResult], pd.DataFrame, pd.DataFrame, dict]:
    """Returns (sweep_result, fitting_results_by_percentile,
    monday_fit_half, monday_holdout_half, friday_meta)."""
    monday_df, _monday_meta = load_monday_pcap()
    friday_df, friday_meta = load_friday_pcap()

    monday_fit, monday_holdout = split_monday_chronologically(monday_df)
    week_df = pd.concat([monday_holdout, friday_df], ignore_index=True)

    result, fitting_results = run_sweep(
        monday_fit,
        monday_holdout,
        week_df,
        percentiles=PERCENTILE_SWEEP,
        rule=EscalationRule.K_OF_N,
        k=2,
        exclude_low_confidence=exclude_low_confidence,
        flow_table_eviction_rate=0.0,  # confirmed 0 evictions on both full runs
        src_table_eviction_rate=0.0,
        ablation_name=f"pcap_exclude_low_confidence={exclude_low_confidence}",
    )
    return result, fitting_results, monday_fit, monday_holdout, friday_meta


def bot_portscan_ddos_recall(
    result: SweepResult, target_rates=(0.01, 0.05, 0.10)
) -> pd.DataFrame:
    table = per_class_recall_table(result, target_rates=target_rates, by="benign_escalation_rate")
    return table[table["label"].isin(["Bot", "PortScan", "DDoS"])].reset_index(drop=True)
