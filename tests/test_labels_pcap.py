import csv

import pytest

from dataplane.flow_state import FlowState, canonical_key
from eval.labels_pcap import (
    join_flow_to_session,
    join_friday_sessions,
    load_session_index,
)


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _row(**overrides):
    row = {
        "Source IP": "10.0.0.1",
        "Destination IP": "10.0.0.2",
        "Source Port": "1234",
        "Destination Port": "80",
        "Protocol": "6",
        "Timestamp": "7/7/2017 12:00",
        "Label": "BENIGN",
    }
    row.update(overrides)
    return row


class TestLoadSessionIndex:
    def test_exact_5tuple_lookup(self, tmp_path):
        path = tmp_path / "s.csv"
        _write_csv(path, [_row(Label="Bot")])
        idx = load_session_index("s", path)
        key = canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 6)
        assert key in idx.by_key

    def test_pm_hour_correction_shifts_by_12h(self, tmp_path):
        path = tmp_path / "s.csv"
        _write_csv(path, [_row(Timestamp="7/7/2017 1:00")])
        uncorrected = load_session_index("s", path)
        corrected = load_session_index("s", path, pm_hour_correction=True)
        assert corrected.min_ts - uncorrected.min_ts == 12 * 3_600 * 1_000_000


class TestJoinFlowToSession:
    def test_exact_key_match_wins(self, tmp_path):
        path = tmp_path / "s.csv"
        _write_csv(path, [_row(Label="Bot")])
        idx = load_session_index("s", path)
        key = canonical_key("10.0.0.2", "10.0.0.1", 80, 1234, 6)  # reversed direction
        flow = FlowState(key=key, fwd_ip="10.0.0.2", fwd_port=80, protocol=6)
        flow.first_ts = idx.min_ts
        label, kind = join_flow_to_session(flow, idx)
        assert (label, kind) == ("Bot", "5tuple")

    def test_no_key_match_falls_back_to_nearest_timestamp(self, tmp_path):
        path = tmp_path / "s.csv"
        _write_csv(
            path,
            [
                _row(Label="Bot", Timestamp="7/7/2017 12:00"),
                _row(
                    Label="BENIGN",
                    Timestamp="7/7/2017 12:05",
                    **{"Source Port": "9999"},
                ),
            ],
        )
        idx = load_session_index("s", path)
        key = canonical_key("9.9.9.9", "8.8.8.8", 1, 2, 17)  # matches nothing
        flow = FlowState(key=key, fwd_ip="9.9.9.9", fwd_port=1, protocol=17)
        flow.first_ts = idx.min_ts + 1_000_000  # close to the Bot row
        label, kind = join_flow_to_session(flow, idx)
        assert (label, kind) == ("Bot", "nearest_timestamp")


class TestJoinFridaySessionsIdentityFirst:
    """Regression test for the bug found while building the PCAP sweep:
    a flow's 5-tuple can legitimately be reopened by FlowTable well
    outside the CSV-observed time window of the session it actually
    belongs to (FIDELITY mode runs continuously across the whole day,
    no per-session reset). Naive time-window-first filtering silently
    dropped these — on the real Friday capture, only 2 of 52,514
    time-windowed "Bot session" flows resolved to Bot even though all
    1,228 distinct Bot 5-tuples existed somewhere in the full day's flow
    set. Identity (exact 5-tuple) must be checked before time window."""

    def test_flow_outside_its_sessions_time_window_still_matches_by_identity(self, tmp_path):
        morning_path = tmp_path / "morning.csv"
        afternoon_path = tmp_path / "afternoon.csv"
        _write_csv(
            morning_path,
            [_row(Label="Bot", Timestamp="7/7/2017 9:00")],
        )
        _write_csv(
            afternoon_path,
            [
                {
                    "Source IP": "10.0.0.5",
                    "Destination IP": "10.0.0.6",
                    "Source Port": "5555",
                    "Destination Port": "443",
                    "Protocol": "6",
                    "Timestamp": "7/7/2017 14:00",
                    "Label": "BENIGN",
                }
            ],
        )
        morning = load_session_index("morning", morning_path)
        afternoon = load_session_index("afternoon", afternoon_path)

        # a flow with the Bot 5-tuple, but whose first_ts (when our table
        # happened to reopen it) falls in the afternoon window instead.
        bot_key = canonical_key("10.0.0.1", "10.0.0.2", 1234, 80, 6)
        flow = FlowState(key=bot_key, fwd_ip="10.0.0.1", fwd_port=1234, protocol=6)
        flow.first_ts = afternoon.min_ts + 1_000_000  # squarely in afternoon's window

        result = join_friday_sessions([flow], [morning, afternoon])
        assert result.labels == ["Bot"]
        assert result.session_of_flow == ["morning"]
        assert result.n_outside_all_windows == 0

    def test_flow_matching_no_identity_falls_back_to_window(self, tmp_path):
        morning_path = tmp_path / "morning.csv"
        _write_csv(morning_path, [_row(Label="Bot", Timestamp="7/7/2017 9:00")])
        morning = load_session_index("morning", morning_path)

        unrelated_key = canonical_key("9.9.9.9", "8.8.8.8", 1, 2, 17)
        flow = FlowState(key=unrelated_key, fwd_ip="9.9.9.9", fwd_port=1, protocol=17)
        flow.first_ts = morning.min_ts  # inside the window, no identity match

        result = join_friday_sessions([flow], [morning])
        assert result.labels == ["Bot"]  # only row in the file, nearest-timestamp fallback
        assert result.session_of_flow == ["morning"]

    def test_flow_outside_every_window_and_no_identity_match_is_excluded(self, tmp_path):
        morning_path = tmp_path / "morning.csv"
        _write_csv(morning_path, [_row(Label="Bot", Timestamp="7/7/2017 9:00")])
        morning = load_session_index("morning", morning_path)

        unrelated_key = canonical_key("9.9.9.9", "8.8.8.8", 1, 2, 17)
        flow = FlowState(key=unrelated_key, fwd_ip="9.9.9.9", fwd_port=1, protocol=17)
        flow.first_ts = morning.max_ts + 100_000_000_000  # way outside

        result = join_friday_sessions([flow], [morning])
        assert result.labels == [None]
        assert result.n_outside_all_windows == 1
