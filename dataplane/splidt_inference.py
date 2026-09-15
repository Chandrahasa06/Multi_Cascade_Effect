"""Standalone reimplementation of SpliDT (arXiv 2509.00397) partitioned
decision tree inference, with no dependency on the ``packetdt`` package
(not installed, not obtainable — never imported here).

``model.pkl`` is a dict: ``root`` (a ``packetdt.splidt.PartitionSubtree``)
and ``features`` (45 four-element feature-name lists, one per subtree).
Each subtree wraps an independently-trained sklearn ``DecisionTreeClassifier``
over its own 4 features. ``children`` on a subtree maps an INTERNAL split
node id of that subtree's own ``tree_`` to another ``PartitionSubtree`` —
never a leaf id (checked: 0 of the 45 subtrees' handoff keys are leaves).

WHAT THIS MODULE ASSUMES ABOUT CSV-PATH USAGE — READ BEFORE TRUSTING ANY
DOWNSTREAM RECALL NUMBER: SpliDT's paper design partitions a *sliding
window of packets* — each subtree in a handoff chain is meant to see the
*next* window, which is what gives the architecture both its
early-detection property and its ability to use more stateful features
than a single hardware stage could hold at once. Here, every feature
comes from one CICIDS2017 CSV row — one already-completed flow, one
static feature vector. Every subtree in a chain is handed the SAME
vector. This makes the model run as a plain nested/cascaded classifier,
NOT as SpliDT — there is no window, no early decision, and
time-to-detection is meaningless in this configuration. Do not report a
time-to-detection or early-escalation metric anywhere downstream of this
module; only the terminal-leaf-after-full-traversal decision is
meaningful here.

=== Traversal semantics — determined empirically, not assumed ===

Two readings were possible for what a ``children`` entry means: (a) the
node's own feature/threshold still decides which of its two ordinary
sklearn children to visit, and the handoff to the new subtree happens
*after* that, at whichever leaf is eventually reached under it; or (b)
reaching the node hands off immediately, bypassing that node's own
split and its two sklearn children entirely.

Checked directly against every one of the 13 handoff points in the root
subtree (and spot-checked elsewhere): a handoff node's own
``tree_.n_node_samples`` is (up to a handful of rows — see below) exactly
equal to the CHILD subtree's ``tree_.n_node_samples[0]`` (its own root's
sample count). E.g. root's node 11 has 17,144 samples; its child subtree's
own root was fit on 17,144 samples. If reading (a) were correct, the
child would instead have been fit on whichever of node 11's two ordinary
children (17,053 or 91 samples) the ORIGINAL split would have sent
training data to — it doesn't match either. The child was fit on the
node's FULL population, before that node's own split was ever applied.

This proves reading (b): **on reaching a node that is a `children` key,
inference hands off immediately to the child subtree's own root,
evaluated on the same feature vector** — the node's own stored
feature/threshold (an artifact of whatever tree existed at that spot
before SpliDT's partitioning cut it there) and its two ordinary sklearn
children are never used. A handful of subtrees (e.g. node 19: parent
40,392 vs child 40,387; node 22: 6,961 vs 6,958) are short by single-digit
counts — almost certainly exact-duplicate feature vectors deduplicated
during that child's own fit, not evidence against the above; every
observed gap is <0.02% of the node's population, none is a fraction
resembling either ordinary-child split.

Structural facts checked and relied on elsewhere in this module:
45 distinct subtrees are reachable from root (matches ``len(features)``);
maximum subtree-to-subtree handoff chain length is 3 hops (root + 3);
maximum single subtree's own sklearn tree depth is 15.

=== Window size ===

Every subtree object's attributes were enumerated
(``root_attrs`` = ``children``, ``features``, ``model``, ``partition_depth``)
— there is no window/size/k attribute anywhere on any of the 45
subtrees. ``partition_depth`` is NOT a window size: it does not vary
monotonically along a handoff chain (root itself carries
``partition_depth=4`` while some of its direct children carry 3), so it
reads as a fitting-time hyperparameter (e.g. an sklearn ``max_depth``
budget passed to that subtree's own fit), not a packet-count parameter.
Window size is not recoverable from this pickle. On the CSV path this is
moot by construction (see above — one static vector, no window), but
flagged here for whoever next runs this model against a real packet
window: window size is an unknown to be swept, not a value this module
can supply.
"""
from __future__ import annotations

import json
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: safety bound on total node-hops (within-subtree steps + subtree handoffs)
#: any single row can take. Structurally bounded by (max hop depth 3 + 1
#: subtrees) * (max sklearn tree depth 15) = 60; generous headroom kept
#: here rather than tuning it to the exact observed structure.
MAX_TRAVERSAL_STEPS = 200


def _install_packetdt_stub() -> None:
    """Register a minimal inert stand-in for ``packetdt.splidt.PartitionSubtree``
    so ``pickle.load`` can resolve the class without the real (unavailable)
    package. Attribute container only — no methods, nothing that could
    silently run PartitionSubtree's real (unknown) behavior."""
    if "packetdt" in sys.modules and hasattr(sys.modules.get("packetdt.splidt", None), "PartitionSubtree"):
        return
    splidt_mod = types.ModuleType("packetdt.splidt")

    class PartitionSubtree:  # attribute container only
        pass

    splidt_mod.PartitionSubtree = PartitionSubtree
    packetdt_mod = types.ModuleType("packetdt")
    packetdt_mod.splidt = splidt_mod
    sys.modules["packetdt"] = packetdt_mod
    sys.modules["packetdt.splidt"] = splidt_mod


def load_raw_model(path) -> dict:
    """Load ``model.pkl`` with the inert stub installed. Returns the raw
    dict ({"root": PartitionSubtree, "features": [...]}) with live sklearn
    DecisionTreeClassifier objects still attached — used only to BUILD the
    serialized form below; nothing downstream of this module should load
    the pickle again."""
    _install_packetdt_stub()
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------- #
# Flattening the live pickle structure into plain, pickle-free JSON data
# --------------------------------------------------------------------- #

def _flatten_subtrees(root) -> Tuple[List, Dict[int, Dict[int, int]]]:
    """BFS over the live PartitionSubtree graph. Returns
    (subtrees_in_discovery_order, children_by_index) where
    children_by_index[i] maps a node id in subtrees[i]'s own tree_ to the
    discovery-order index of the child subtree."""
    order: List = []
    index_of_id: Dict[int, int] = {}
    children_by_index: Dict[int, Dict[int, int]] = {}
    queue = [root]
    while queue:
        st = queue.pop(0)
        if id(st) in index_of_id:
            continue
        idx = len(order)
        index_of_id[id(st)] = idx
        order.append(st)
        queue.extend(st.children.values())

    for st in order:
        idx = index_of_id[id(st)]
        children_by_index[idx] = {
            int(node_id): index_of_id[id(child)] for node_id, child in st.children.items()
        }
    return order, children_by_index


def build_model_data(raw_obj: dict) -> dict:
    """Convert the live pickle structure into a plain-data dict, JSON-ready
    (no pickle, no PartitionSubtree, no sklearn objects — plain lists/ints/
    floats/strings only)."""
    root = raw_obj["root"]
    order, children_by_index = _flatten_subtrees(root)

    subtrees_json = {}
    for idx, st in enumerate(order):
        tree = st.model.tree_
        classes = [str(c) for c in st.model.classes_]
        n_nodes = tree.node_count

        feature_index = [int(f) for f in tree.feature]  # -1 at leaves
        children_left = [int(c) for c in tree.children_left]
        children_right = [int(c) for c in tree.children_right]
        threshold = [float(t) for t in tree.threshold]
        # majority class per node (meaningful at leaves; harmless elsewhere)
        value = tree.value  # shape (n_nodes, 1, n_classes)
        predicted_class = [classes[int(np.argmax(value[n, 0]))] for n in range(n_nodes)]

        subtrees_json[str(idx)] = {
            "features": list(st.features),
            "partition_depth": int(st.partition_depth),
            "classes": classes,
            "n_nodes": n_nodes,
            "feature_index": feature_index,
            "children_left": children_left,
            "children_right": children_right,
            "threshold": threshold,
            "predicted_class": predicted_class,
            # handoff: node id (as string, JSON-object-key-safe) -> child subtree index
            "children": {str(k): v for k, v in children_by_index[idx].items()},
        }

    global_classes = sorted({c for st in subtrees_json.values() for c in st["classes"]})

    return {
        "global_classes": global_classes,
        "root_index": 0,
        "n_subtrees": len(order),
        "subtrees": subtrees_json,
    }


def save_model_json(model_data: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)


def load_model_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def union_features(model_data: dict) -> List[str]:
    """Every distinct feature name referenced anywhere in the model,
    sorted — the set eval/splidt_features.py must be able to supply."""
    names = set()
    for st in model_data["subtrees"].values():
        names.update(st["features"])
    return sorted(names)


# --------------------------------------------------------------------- #
# Pure single-subtree traversal (no handoff) — used by the correctness
# gate to reproduce each subtree's own sklearn .predict() independently.
# --------------------------------------------------------------------- #

def predict_single_subtree(subtree: dict, feature_df: pd.DataFrame) -> np.ndarray:
    """Walk ONE subtree's own tree_ to a leaf, exactly as sklearn's
    DecisionTreeClassifier.predict() would (``<=`` goes left, ``>`` goes
    right, argmax of leaf class distribution) — handoff is irrelevant
    here since sklearn's own .predict() knows nothing about it."""
    n = len(feature_df)
    cols = [feature_df[name].to_numpy(dtype=float) for name in subtree["features"]]
    feature_index = np.asarray(subtree["feature_index"])
    threshold = np.asarray(subtree["threshold"])
    children_left = np.asarray(subtree["children_left"])
    children_right = np.asarray(subtree["children_right"])
    predicted_class = np.asarray(subtree["predicted_class"], dtype=object)

    node = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    result = np.empty(n, dtype=object)

    for _ in range(MAX_TRAVERSAL_STEPS):
        active = np.nonzero(~done)[0]
        if active.size == 0:
            break
        nodes = node[active]
        # sklearn leaf sentinel is children_left == -1 (TREE_LEAF); the
        # feature array uses -2 (TREE_UNDEFINED) at leaves, not -1.
        is_leaf = children_left[nodes] == -1
        fidx = feature_index[nodes]
        leaf_idx = active[is_leaf]
        if leaf_idx.size:
            result[leaf_idx] = predicted_class[nodes[is_leaf]]
            done[leaf_idx] = True

        internal_idx = active[~is_leaf]
        if internal_idx.size:
            internal_nodes = nodes[~is_leaf]
            internal_fidx = fidx[~is_leaf]
            vals = np.empty(internal_idx.size)
            for k, col in enumerate(cols):
                sel = internal_fidx == k
                if sel.any():
                    vals[sel] = col[internal_idx[sel]]
            go_left = vals <= threshold[internal_nodes]
            next_nodes = np.where(go_left, children_left[internal_nodes], children_right[internal_nodes])
            node[internal_idx] = next_nodes

    if not done.all():
        raise RuntimeError(f"{int((~done).sum())} rows did not reach a leaf within {MAX_TRAVERSAL_STEPS} steps")
    return result


# --------------------------------------------------------------------- #
# Full-model traversal, with handoff — the real inference path.
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class SplidtPrediction:
    predicted_class: np.ndarray  # str per row
    final_subtree: np.ndarray  # int per row — which subtree produced the leaf
    final_leaf_node: np.ndarray  # int per row — which node id in that subtree


def predict_batch(model_data: dict, feature_df: pd.DataFrame) -> SplidtPrediction:
    """Full-model inference: start at the root subtree, walk its tree_,
    and whenever a node is a `children` key, hand off immediately to that
    child subtree's own root (see module docstring for the evidence this
    is the correct semantics) — repeat until a genuine leaf (no handoff
    registered for it) is reached."""
    subtrees = model_data["subtrees"]
    n = len(feature_df)

    unique_feature_names = union_features(model_data)
    col_arrays = {name: feature_df[name].to_numpy(dtype=float) for name in unique_feature_names}

    cur_subtree = np.full(n, model_data["root_index"], dtype=int)
    cur_node = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    result_class = np.empty(n, dtype=object)
    result_subtree = np.zeros(n, dtype=int)
    result_node = np.zeros(n, dtype=int)

    for _ in range(MAX_TRAVERSAL_STEPS):
        active = np.nonzero(~done)[0]
        if active.size == 0:
            break
        for sid in np.unique(cur_subtree[active]):
            st = subtrees[str(int(sid))]
            mask = (~done) & (cur_subtree == sid)
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue
            nodes = cur_node[idxs]

            children_map = st["children"]
            if children_map:
                is_handoff = np.array([str(int(nd)) in children_map for nd in nodes])
            else:
                is_handoff = np.zeros(nodes.size, dtype=bool)

            if is_handoff.any():
                ho_idxs = idxs[is_handoff]
                ho_nodes = nodes[is_handoff]
                new_sub = np.array([children_map[str(int(nd))] for nd in ho_nodes])
                cur_subtree[ho_idxs] = new_sub
                cur_node[ho_idxs] = 0

            rest_idxs = idxs[~is_handoff]
            rest_nodes = nodes[~is_handoff]
            if rest_idxs.size == 0:
                continue

            feature_index = np.asarray(st["feature_index"])
            children_left_arr = np.asarray(st["children_left"])
            # sklearn leaf sentinel is children_left == -1 (TREE_LEAF); the
            # feature array uses -2 (TREE_UNDEFINED) at leaves, not -1.
            is_leaf = children_left_arr[rest_nodes] == -1
            fidx = feature_index[rest_nodes]

            leaf_idxs = rest_idxs[is_leaf]
            if leaf_idxs.size:
                leaf_nodes = rest_nodes[is_leaf]
                predicted_class = np.asarray(st["predicted_class"], dtype=object)
                result_class[leaf_idxs] = predicted_class[leaf_nodes]
                result_subtree[leaf_idxs] = int(sid)
                result_node[leaf_idxs] = leaf_nodes
                done[leaf_idxs] = True

            internal_idxs = rest_idxs[~is_leaf]
            if internal_idxs.size:
                internal_nodes = rest_nodes[~is_leaf]
                internal_fidx = fidx[~is_leaf]
                threshold = np.asarray(st["threshold"])
                children_left = np.asarray(st["children_left"])
                children_right = np.asarray(st["children_right"])
                vals = np.empty(internal_idxs.size)
                for k, fname in enumerate(st["features"]):
                    sel = internal_fidx == k
                    if sel.any():
                        vals[sel] = col_arrays[fname][internal_idxs[sel]]
                go_left = vals <= threshold[internal_nodes]
                next_nodes = np.where(go_left, children_left[internal_nodes], children_right[internal_nodes])
                cur_node[internal_idxs] = next_nodes

    if not done.all():
        raise RuntimeError(f"{int((~done).sum())} rows did not reach a leaf within {MAX_TRAVERSAL_STEPS} steps")

    return SplidtPrediction(predicted_class=result_class, final_subtree=result_subtree, final_leaf_node=result_node)


def build_and_save(pickle_path, json_path) -> dict:
    raw = load_raw_model(pickle_path)
    model_data = build_model_data(raw)
    save_model_json(model_data, json_path)
    return model_data


if __name__ == "__main__":
    data = build_and_save("model.pkl", "results/splidt_model.json")
    print(f"wrote results/splidt_model.json: {data['n_subtrees']} subtrees")
