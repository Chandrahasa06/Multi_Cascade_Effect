import csv
import math

from adapters.csv_flow_adapter import (
    AdapterStats,
    _flag_positions,
    _interleave_directions,
    _synth_lengths,
    iter_packets_from_csv,
    load_csv,
    parse_timestamp,
    parse_timestamp_us,
    synthesize_packets,
)
from dataplane.flow_state import UNSET, Direction


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
        "FIN Flag Count": 0,
        "RST Flag Count": 0,
        "ACK Flag Count": 2,
        "Fwd PSH Flags": 0,
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Init_Win_bytes_forward": 65535,
        "Init_Win_bytes_backward": 29200,
    }
    row.update(overrides)
    return row


class TestTimestampParsing:
    def test_second_resolution_format(self):
        assert parse_timestamp_us("03/07/2017 08:55:58") is not None

    def test_minute_resolution_format(self):
        assert parse_timestamp_us("7/7/2017 3:30") is not None

    def test_unparsable_returns_none(self):
        assert parse_timestamp_us("not a date") is None

    def test_relative_ordering_is_correct(self):
        t1 = parse_timestamp_us("03/07/2017 08:55:58")
        t2 = parse_timestamp_us("03/07/2017 08:56:22")
        assert t2 > t1
        assert t2 - t1 == 24_000_000

    def test_second_resolution_reports_1_second_tick(self):
        _, resolution_us = parse_timestamp("03/07/2017 08:55:58")
        assert resolution_us == 1_000_000

    def test_minute_resolution_reports_1_minute_tick(self):
        _, resolution_us = parse_timestamp("7/7/2017 3:30")
        assert resolution_us == 60_000_000

    def test_unparsable_returns_none_tuple(self):
        assert parse_timestamp("garbage") is None


class TestInterleaveDirections:
    def test_counts_match(self):
        seq = _interleave_directions(3, 2)
        assert seq.count(Direction.FWD) == 3
        assert seq.count(Direction.BWD) == 2

    def test_first_is_forward_when_any_forward_exists(self):
        seq = _interleave_directions(1, 5)
        assert seq[0] == Direction.FWD

    def test_zero_backward(self):
        seq = _interleave_directions(4, 0)
        assert seq == [Direction.FWD] * 4


class TestSynthLengths:
    def test_sum_matches_total_exactly(self):
        lengths = _synth_lengths(5, 300, 20, 100)
        assert sum(lengths) == 300
        assert lengths[0] == 20
        assert lengths[-1] == 100

    def test_single_packet(self):
        assert _synth_lengths(1, 77, 0, 0) == [77]

    def test_two_packets(self):
        lengths = _synth_lengths(2, 100, 30, 70)
        assert lengths == [30, 70]
        assert sum(lengths) == 100

    def test_inconsistent_min_max_falls_back_to_even_split_without_negatives(self):
        # min + max alone exceed the reported total: real, messy data.
        lengths = _synth_lengths(4, 10, 50, 50)
        assert sum(lengths) == 10
        assert all(x >= 0 for x in lengths)

    def test_zero_packets(self):
        assert _synth_lengths(0, 0, 0, 0) == []


class TestFlagPositions:
    def test_start_bias(self):
        assert _flag_positions(5, 2, "start") == {0, 1}

    def test_end_bias(self):
        assert _flag_positions(5, 2, "end") == {3, 4}

    def test_k_clamped_to_n(self):
        assert _flag_positions(3, 10, "start") == {0, 1, 2}

    def test_zero_k(self):
        assert _flag_positions(5, 0, "spread") == set()


class TestSynthesizePackets:
    def test_valid_row_produces_expected_packet_count(self):
        result = synthesize_packets(make_row())
        assert result.skip_reason is None
        assert len(result.packets) == 5

    def test_byte_sums_reconstruct_exactly(self):
        result = synthesize_packets(make_row())
        fwd_bytes = sum(p.length for p in result.packets if p.src_ip == "10.0.0.1")
        bwd_bytes = sum(p.length for p in result.packets if p.src_ip == "10.0.0.2")
        assert fwd_bytes == 300
        assert bwd_bytes == 200

    def test_duration_reconstructs_exactly(self):
        result = synthesize_packets(make_row(**{"Flow Duration": 8000}))
        ts = [p.timestamp_us for p in result.packets]
        assert max(ts) - min(ts) == 8000

    def test_init_window_bytes_on_first_packet_of_each_direction_only(self):
        result = synthesize_packets(make_row())
        fwd_pkts = [p for p in result.packets if p.src_ip == "10.0.0.1"]
        bwd_pkts = [p for p in result.packets if p.src_ip == "10.0.0.2"]
        assert fwd_pkts[0].win_size == 65535
        assert all(p.win_size == UNSET for p in fwd_pkts[1:])
        assert bwd_pkts[0].win_size == 29200
        assert all(p.win_size == UNSET for p in bwd_pkts[1:])

    def test_missing_required_field_is_skipped(self):
        row = make_row()
        del row["Source IP"]
        result = synthesize_packets(row)
        assert result.packets is None
        assert result.skip_reason == "missing_field:Source IP"

    def test_unparsable_timestamp_is_skipped(self):
        result = synthesize_packets(make_row(Timestamp="garbage"))
        assert result.packets is None
        assert result.skip_reason == "unparsable_timestamp"

    def test_zero_forward_packets_is_skipped(self):
        result = synthesize_packets(make_row(**{"Total Fwd Packets": 0}))
        assert result.packets is None
        assert result.skip_reason == "no_forward_packets"

    def test_negative_duration_is_clamped_not_skipped(self):
        result = synthesize_packets(make_row(**{"Flow Duration": -1}))
        assert result.packets is not None
        assert result.duration_clamped
        assert all(p.timestamp_us == result.packets[0].timestamp_us for p in result.packets)

    def test_zero_duration_flagged_and_all_packets_share_timestamp(self):
        result = synthesize_packets(make_row(**{"Flow Duration": 0}))
        assert result.zero_duration
        ts = {p.timestamp_us for p in result.packets}
        assert len(ts) == 1

    def test_nan_infinity_in_unused_rate_columns_does_not_break_synthesis(self):
        result = synthesize_packets(
            make_row(**{"Flow Bytes/s": float("inf"), "Flow Packets/s": float("nan")})
        )
        assert result.skip_reason is None
        assert len(result.packets) == 5

    def test_second_resolution_row_tags_packets_as_1s_resolution(self):
        result = synthesize_packets(make_row(Timestamp="03/07/2017 08:55:58"))
        assert result.timestamp_resolution_us == 1_000_000
        assert all(p.timestamp_resolution_us == 1_000_000 for p in result.packets)

    def test_minute_resolution_row_tags_packets_as_60s_resolution(self):
        result = synthesize_packets(make_row(Timestamp="7/7/2017 3:30"))
        assert result.timestamp_resolution_us == 60_000_000
        assert all(p.timestamp_resolution_us == 60_000_000 for p in result.packets)

    def test_single_fwd_packet_no_backward(self):
        row = make_row(**{
            "Total Fwd Packets": 1, "Total Backward Packets": 0,
            "Total Length of Fwd Packets": 40,
            "Fwd Packet Length Min": 40, "Fwd Packet Length Max": 40,
        })
        result = synthesize_packets(row)
        assert len(result.packets) == 1
        assert result.packets[0].length == 40


class TestSourceRowId:
    def test_default_row_id_is_zero(self):
        result = synthesize_packets(make_row())
        assert all(p.source_row_id == 0 for p in result.packets)

    def test_explicit_row_id_stamped_on_every_packet(self):
        result = synthesize_packets(make_row(), row_id=99)
        assert all(p.source_row_id == 99 for p in result.packets)

    def test_iter_packets_from_csv_stamps_sequential_row_ids(self, tmp_path):
        rows = [make_row(), make_row(**{"Destination Port": 443})]
        csv_path = tmp_path / "day.csv"
        _write_csv(csv_path, rows)
        packets = list(iter_packets_from_csv(csv_path))
        row_ids_seen = {p.source_row_id for p in packets}
        assert row_ids_seen == {0, 1}


class TestLoadCsvEncodingFallback:
    def test_falls_back_to_cp1252_on_undecodable_utf8(self, tmp_path):
        # real quirk: the WebAttacks file's Label column has an en-dash
        # ("Web Attack \x96 Brute Force") encoded as Windows-1252, which
        # is not valid UTF-8 (0x96 is a continuation-byte-shaped value
        # with no valid lead byte before it here).
        path = tmp_path / "cp1252.csv"
        header = "Source IP,Label\r\n".encode("utf-8")
        row = b"1.2.3.4,Web Attack \x96 Brute Force\r\n"  # raw byte 0x96, not U+0096
        path.write_bytes(header + row)

        df = load_csv(path)
        assert df["Label"].iloc[0] == "Web Attack – Brute Force"

    def test_plain_utf8_still_reads_normally(self, tmp_path):
        path = tmp_path / "utf8.csv"
        path.write_text("Source IP,Label\r\n1.2.3.4,BENIGN\r\n", encoding="utf-8")
        df = load_csv(path)
        assert df["Label"].iloc[0] == "BENIGN"


def _write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestAdapterStatsResolutionTracking:
    def test_uniform_second_resolution_file_reports_single_resolution(self, tmp_path):
        rows = [make_row(Timestamp="03/07/2017 08:55:58"), make_row(Timestamp="03/07/2017 08:56:22")]
        csv_path = tmp_path / "day.csv"
        _write_csv(csv_path, rows)

        stats = AdapterStats()
        list(iter_packets_from_csv(csv_path, stats=stats))

        assert stats.resolution_counts == {1_000_000: 2}
        assert stats.timestamp_resolution_us == 1_000_000

    def test_mixed_resolution_file_reports_none_for_single_resolution(self, tmp_path):
        rows = [make_row(Timestamp="03/07/2017 08:55:58"), make_row(Timestamp="7/7/2017 3:30")]
        csv_path = tmp_path / "day.csv"
        _write_csv(csv_path, rows)

        stats = AdapterStats()
        list(iter_packets_from_csv(csv_path, stats=stats))

        assert stats.resolution_counts == {1_000_000: 1, 60_000_000: 1}
        assert stats.timestamp_resolution_us is None
