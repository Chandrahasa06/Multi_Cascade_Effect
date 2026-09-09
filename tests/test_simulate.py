import csv

import pytest

from dataplane.flow_table import KeyMode
from eval.simulate import extract_day_features


def make_row(**overrides):
    row = {
        "Source IP": "10.0.0.1",
        "Destination IP": "10.0.0.2",
        "Source Port": 1234,
        "Destination Port": 80,
        "Protocol": 6,
        "Timestamp": "03/07/2017 08:55:58",
        "Flow Duration": 5000,
        "Total Fwd Packets": 3,
        "Total Backward Packets": 2,
        "Total Length of Fwd Packets": 300,
        "Total Length of Bwd Packets": 200,
        "Fwd Packet Length Min": 50,
        "Fwd Packet Length Max": 150,
        "Bwd Packet Length Min": 80,
        "Bwd Packet Length Max": 120,
        "SYN Flag Count": 1,
        "FIN Flag Count": 1,
        "RST Flag Count": 0,
        "ACK Flag Count": 2,
        "Fwd PSH Flags": 0,
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Init_Win_bytes_forward": 65535,
        "Init_Win_bytes_backward": 29200,
        "Label": "BENIGN",
    }
    row.update(overrides)
    return row


def write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestExtractDayFeatures:
    def test_eval_mode_one_row_per_flow(self, tmp_path):
        rows = [
            make_row(**{"Destination Port": 80}),
            make_row(**{"Destination Port": 443}),
            make_row(**{"Destination Port": 22, "Label": "PortScan"}),
        ]
        csv_path = tmp_path / "day.csv"
        write_csv(csv_path, rows)

        df, meta = extract_day_features(
            csv_path, "testday", KeyMode.EVAL, cache_dir=tmp_path / "cache"
        )
        assert len(df) == 3
        assert meta["flows_total"] == 3
        assert meta["flow_table_eviction_count"] == 0
        assert set(df["label"]) == {"BENIGN", "PortScan"}
        assert "flow_duration" in df.columns
        assert "flow_duration__low_confidence" in df.columns
        assert "flows_per_src" in df.columns  # per-source feature present
        assert "first_ts" in df.columns
        assert "last_ts" in df.columns
        assert "closure_reason" in df.columns

    def test_cache_hit_skips_resimulation(self, tmp_path, monkeypatch):
        rows = [make_row()]
        csv_path = tmp_path / "day.csv"
        write_csv(csv_path, rows)
        cache_dir = tmp_path / "cache"

        df1, meta1 = extract_day_features(csv_path, "testday", KeyMode.EVAL, cache_dir=cache_dir)

        def _boom(*args, **kwargs):
            raise AssertionError("should not re-run simulation on cache hit")

        monkeypatch.setattr("eval.simulate._run_simulation", _boom)
        df2, meta2 = extract_day_features(csv_path, "testday", KeyMode.EVAL, cache_dir=cache_dir)
        assert df1.equals(df2)
        assert meta1 == meta2

    def test_force_bypasses_cache(self, tmp_path):
        rows = [make_row()]
        csv_path = tmp_path / "day.csv"
        write_csv(csv_path, rows)
        cache_dir = tmp_path / "cache"

        extract_day_features(csv_path, "testday", KeyMode.EVAL, cache_dir=cache_dir)
        df2, meta2 = extract_day_features(
            csv_path, "testday", KeyMode.EVAL, cache_dir=cache_dir, force=True
        )
        assert meta2["flows_total"] == 1

    def test_fidelity_mode_reports_mixed_labels(self, tmp_path):
        # two rows, same 5-tuple, different labels, close in time -> merge
        rows = [
            make_row(Label="BENIGN", Timestamp="03/07/2017 08:55:58"),
            make_row(Label="DoS Hulk", Timestamp="03/07/2017 08:55:59"),
        ]
        csv_path = tmp_path / "day.csv"
        write_csv(csv_path, rows)

        df, meta = extract_day_features(
            csv_path, "testday", KeyMode.FIDELITY, cache_dir=tmp_path / "cache"
        )
        assert meta["flows_total"] == 1  # merged into one flow
        assert df.iloc[0]["mixed_label"] == True  # noqa: E712
        assert meta["join_report"]["merged_mixed_label_flows"] == 1

    def test_different_key_modes_get_different_cache_files(self, tmp_path):
        rows = [make_row()]
        csv_path = tmp_path / "day.csv"
        write_csv(csv_path, rows)
        cache_dir = tmp_path / "cache"

        extract_day_features(csv_path, "testday", KeyMode.EVAL, cache_dir=cache_dir)
        extract_day_features(csv_path, "testday", KeyMode.FIDELITY, cache_dir=cache_dir)
        cached_files = list(cache_dir.glob("*.parquet"))
        assert len(cached_files) == 2


class TestPerSourceFeatureSnapshotTiming:
    def test_early_flow_reflects_its_own_closure_time_not_end_of_file(self, tmp_path):
        # Same source touches 2 distinct dest ports early (t=0s, t=1s,
        # both closing cleanly via a two-sided FIN so they leave the
        # table immediately), then the SAME source appears again much
        # later (t=201s — past the 60s sliding window) touching a third
        # port. If per-source features were snapshotted after the whole
        # file is processed (the bug), the early flows would see the
        # LATE bucket state (source_table has since rotated past the
        # early ports) instead of their own historical context.
        rows = [
            make_row(
                **{
                    "Destination Port": 100,
                    "Timestamp": "03/07/2017 08:55:58",
                    "Flow Duration": 1000,
                    "FIN Flag Count": 2,
                },
            ),
            make_row(
                **{
                    "Destination Port": 200,
                    "Timestamp": "03/07/2017 08:55:59",
                    "Flow Duration": 1000,
                    "FIN Flag Count": 2,
                }
            ),
            make_row(
                **{
                    "Destination Port": 300,
                    "Timestamp": "03/07/2017 08:59:19",  # +201s from row 0
                    "Flow Duration": 1000,
                    "FIN Flag Count": 2,
                }
            ),
        ]
        csv_path = tmp_path / "day.csv"
        write_csv(csv_path, rows)

        df, meta = extract_day_features(
            csv_path, "testday", KeyMode.EVAL, cache_dir=tmp_path / "cache"
        )
        assert meta["flows_total"] == 3  # all three closed via FIN, none merged/lost

        # row 1 (port 200) closes right after row 0 (port 100): at that
        # instant this source has touched 2 distinct ports.
        row1_distinct_ports = df.iloc[1]["distinct_dst_ports_per_src"]
        assert row1_distinct_ports == pytest.approx(2.0, abs=0.3)

        # row 0's OWN snapshot, taken at its own closure (before row 1
        # even exists), must see exactly 1 distinct port — never more,
        # regardless of what this source does later.
        row0_distinct_ports = df.iloc[0]["distinct_dst_ports_per_src"]
        assert row0_distinct_ports == pytest.approx(1.0, abs=0.3)
