import numpy as np
import pandas as pd
import pytest

from dataplane.splidt_inference import (
    build_model_data,
    load_model_json,
    load_raw_model,
    predict_batch,
    predict_single_subtree,
    union_features,
)


def _leaf(node_id, predicted):
    return node_id, predicted


def make_toy_model():
    """Two subtrees: root splits on f0 <= 5 -> leaf 'A'; the f0 > 5 branch
    (node 2) is a `children` handoff key to subtree 1, which always
    predicts 'C'. Node 2 ALSO carries its own (unused-if-handoff-correct)
    leaf class 'B' in the parent tree's own value array — this is exactly
    the shape real handoff nodes have (sklearn requires every node to
    have SOME value), and lets a test distinguish "handoff replaces the
    node's own decision" from "handoff only applies after reaching a
    leaf"."""
    subtree0 = {
        "features": ["f0", "f1"],
        "partition_depth": 1,
        "classes": ["A", "B"],
        "n_nodes": 3,
        "feature_index": [0, -1, -1],
        "children_left": [1, -1, -1],
        "children_right": [2, -1, -1],
        "threshold": [5.0, 0.0, 0.0],
        "predicted_class": ["_internal_", "A", "B"],
        "children": {"2": 1},
    }
    subtree1 = {
        "features": ["f2"],
        "partition_depth": 1,
        "classes": ["C"],
        "n_nodes": 1,
        "feature_index": [-1],
        "children_left": [-1],
        "children_right": [-1],
        "threshold": [0.0],
        "predicted_class": ["C"],
        "children": {},
    }
    return {
        "global_classes": ["A", "B", "C"],
        "root_index": 0,
        "n_subtrees": 2,
        "subtrees": {"0": subtree0, "1": subtree1},
    }


class TestPartitionHandoff:
    def test_handoff_bypasses_parent_node_own_leaf(self):
        """The key behavioral claim in splidt_inference.py's docstring:
        reaching a `children`-keyed node hands off immediately to the
        child subtree, never using that node's own stored leaf class."""
        model_data = make_toy_model()
        df = pd.DataFrame({"f0": [10.0], "f1": [0.0], "f2": [0.0]})
        result = predict_batch(model_data, df)
        assert result.predicted_class[0] == "C"  # NOT "B"
        assert result.final_subtree[0] == 1
        assert result.final_leaf_node[0] == 0

    def test_non_handoff_leaf_still_used_directly(self):
        model_data = make_toy_model()
        df = pd.DataFrame({"f0": [1.0], "f1": [0.0], "f2": [0.0]})
        result = predict_batch(model_data, df)
        assert result.predicted_class[0] == "A"
        assert result.final_subtree[0] == 0
        assert result.final_leaf_node[0] == 1

    def test_batch_mixes_handoff_and_non_handoff_rows(self):
        model_data = make_toy_model()
        df = pd.DataFrame({"f0": [1.0, 10.0, 2.0, 9.0], "f1": [0, 0, 0, 0], "f2": [0, 0, 0, 0]})
        result = predict_batch(model_data, df)
        assert list(result.predicted_class) == ["A", "C", "A", "C"]

    def test_single_subtree_predict_ignores_handoff_entirely(self):
        """predict_single_subtree must reproduce plain sklearn semantics
        (no knowledge of `children` at all) -- this is what the
        correctness gate against real sklearn .predict() depends on."""
        model_data = make_toy_model()
        df = pd.DataFrame({"f0": [10.0], "f1": [0.0]})
        pred = predict_single_subtree(model_data["subtrees"]["0"], df)
        assert pred[0] == "B"  # the node's own leaf, unlike predict_batch's "C"


class TestWindowSemanticsMoot:
    def test_same_feature_vector_feeds_every_subtree_in_a_chain(self):
        """CSV-path assumption (see splidt_inference.py docstring): every
        subtree in a handoff chain sees the SAME static feature vector --
        there is no sliding window, no window-advancement step. A flow
        with only one row of features must still traverse a multi-hop
        handoff chain correctly using that one vector for every hop."""
        model_data = make_toy_model()
        df = pd.DataFrame({"f0": [10.0], "f1": [0.0], "f2": [0.0]})
        result = predict_batch(model_data, df)
        # reached subtree 1 (a second hop) from the SAME single input row
        assert result.final_subtree[0] == 1
        assert result.predicted_class[0] == "C"


class TestRealModelCorrectnessGate:
    """Reproduces every one of the 45 subtrees' own sklearn .predict()
    exactly, on real CICIDS2017 rows (the sklearn estimators load fine —
    only the packetdt.splidt.PartitionSubtree wrapper is missing)."""

    @classmethod
    @pytest.fixture(scope="class")
    def raw_and_json(cls):
        raw = load_raw_model("model.pkl")
        model_data = build_model_data(raw)
        return raw, model_data

    @classmethod
    @pytest.fixture(scope="class")
    def sample_features(cls):
        from adapters.csv_flow_adapter import load_csv
        from eval.splidt_features import MODEL_FEATURE_TO_CSV_COLUMN

        df = load_csv("data/csv/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv")
        sample = df.sample(n=2000, random_state=0).reset_index(drop=True)
        feats = pd.DataFrame({k: sample[v].astype(float) for k, v in MODEL_FEATURE_TO_CSV_COLUMN.items()})
        feats = feats.replace([np.inf, -np.inf], np.nan)
        finite_mask = ~feats.isna().any(axis=1)
        return feats[finite_mask].reset_index(drop=True)

    def test_every_subtree_matches_sklearn_predict(self, raw_and_json, sample_features):
        from dataplane.splidt_inference import _flatten_subtrees

        raw, model_data = raw_and_json
        order, _ = _flatten_subtrees(raw["root"])
        assert len(order) == 45

        total_mismatches = 0
        for sid_str, st in model_data["subtrees"].items():
            sid = int(sid_str)
            live_st = order[sid]
            X = sample_features[st["features"]].to_numpy(dtype=float)
            sklearn_pred = live_st.model.predict(X)
            my_pred = predict_single_subtree(st, sample_features)
            total_mismatches += int((sklearn_pred.astype(str) != my_pred.astype(str)).sum())
        assert total_mismatches == 0

    def test_json_roundtrip_matches_freshly_built(self, raw_and_json, tmp_path):
        from dataplane.splidt_inference import save_model_json

        _, model_data = raw_and_json
        path = tmp_path / "model.json"
        save_model_json(model_data, path)
        reloaded = load_model_json(path)
        assert reloaded == model_data

    def test_union_features_has_41_entries(self, raw_and_json):
        _, model_data = raw_and_json
        assert len(union_features(model_data)) == 41

    def test_full_model_predict_batch_runs_without_error(self, raw_and_json, sample_features):
        _, model_data = raw_and_json
        result = predict_batch(model_data, sample_features)
        assert len(result.predicted_class) == len(sample_features)
        assert set(result.predicted_class).issubset(set(model_data["global_classes"]))
