import numpy as np
import pandas as pd

from eval.splidt_features import (
    MODEL_FEATURE_TO_CSV_COLUMN,
    UNSEEN_CLASSES,
    MODEL_CLASSES,
    clean_rate_features,
    mark_contamination,
    normalize_label,
    rate_quality_report,
)


class TestLabelNormalization:
    def test_benign_uppercase_maps_to_titlecase(self):
        assert normalize_label("BENIGN") == "Benign"

    def test_slowloris_case_fix(self):
        assert normalize_label("DoS slowloris") == "DoS-Slowloris"

    def test_web_attack_variants_mojibake_safe(self):
        assert normalize_label("Web Attack \x96 Brute Force") == "WebAttack-BruteForce"
        assert normalize_label("Web Attack \x96 XSS") == "WebAttack-XSS"
        assert normalize_label("Web Attack \x96 Sql Injection") == "WebAttack-SQLInjection"

    def test_unseen_and_exact_match_labels_pass_through(self):
        for label in ("FTP-Patator", "SSH-Patator", "Heartbleed", "Infiltration", "PortScan",
                      "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest"):
            assert normalize_label(label) == label

    def test_model_classes_and_unseen_classes_disjoint(self):
        assert MODEL_CLASSES.isdisjoint(UNSEEN_CLASSES)


class TestFeatureMapping:
    def test_all_41_features_mapped(self):
        assert len(MODEL_FEATURE_TO_CSV_COLUMN) == 41

    def test_no_duplicate_csv_columns_mapped_twice_by_accident(self):
        # duplicates ARE allowed in principle (none expected here) -- this
        # documents that expectation rather than assuming it silently.
        values = list(MODEL_FEATURE_TO_CSV_COLUMN.values())
        assert len(values) == len(set(values))


class TestRateCleaning:
    def make_df(self):
        return pd.DataFrame({
            "Label": ["Benign", "Benign", "DoS Hulk", "DoS Hulk"],
            "Flow Bytes/s": [1.0, np.nan, np.inf, 5.0],
            "Flow Packets/s": [1.0, 2.0, np.inf, 3.0],
        })

    def test_nan_rows_dropped_inf_rows_kept(self):
        df = self.make_df()
        cleaned = clean_rate_features(df)
        assert len(cleaned) == 3  # only the NaN row is dropped
        assert np.isinf(cleaned["Flow Bytes/s"]).any()  # Infinity survives

    def test_rate_quality_report_counts_by_label(self):
        df = self.make_df()
        report = rate_quality_report(df)
        hulk = report[report["label"] == "DoS Hulk"].iloc[0]
        assert hulk["Flow Bytes/s__n_inf"] == 1
        assert hulk["Flow Bytes/s__n_nan"] == 0
        benign = report[report["label"] == "Benign"].iloc[0]
        assert benign["Flow Bytes/s__n_nan"] == 1


class TestContaminationMarking:
    def test_marks_matching_flow_ids_only(self):
        df = pd.DataFrame({"Flow ID": ["a-b-1-2-6", "c-d-3-4-6", "a-b-1-2-6"]})
        train_ids = frozenset({"a-b-1-2-6"})
        marked = mark_contamination(df, train_ids)
        assert list(marked["contaminated"]) == [True, False, True]

    def test_does_not_mutate_or_drop_rows(self):
        df = pd.DataFrame({"Flow ID": ["x", "y"]})
        marked = mark_contamination(df, frozenset({"x"}))
        assert len(marked) == len(df)
        assert "Flow ID" in df.columns  # original untouched
