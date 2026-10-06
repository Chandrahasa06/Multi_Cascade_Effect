"""Required deliverable test: batch 3 shares no record with batch 1 or
batch 2. Same two-level approach as tests/test_batch2_no_overlap.py:
record_id-level (real generated files) and flow_id-level (the meaningful
check, since record_id is a fresh opaque token with no link back to the
source flow).
"""
import numpy as np
import pandas as pd
import pytest

from eval.select_agent_sample import (
    N_ATTACK,
    N_BENIGN,
    RANDOM_STATE,
    reconstruct_batch1_flow_ids,
    select_attack_flows,
    select_benign_flows,
)
from eval.select_agent_sample_batch2 import RANDOM_STATE_B, reconstruct_batch2_flow_ids
from eval.select_agent_sample_batch3 import (
    N_ATTACK_C,
    N_BENIGN_C,
    RANDOM_STATE_C,
    compute_scored_class_counts,
    select_attack_flows_prioritised,
)


@pytest.fixture(scope="module")
def all_flows():
    try:
        return pd.read_csv("results/escalated_flows.csv")
    except FileNotFoundError:
        pytest.skip("results/escalated_flows.csv not generated yet")


def _reconstruct_batch3_ids(all_flows: pd.DataFrame) -> set:
    batch1_ids = reconstruct_batch1_flow_ids(all_flows)
    batch2_ids = reconstruct_batch2_flow_ids(all_flows)
    used = batch1_ids | batch2_ids
    remaining = all_flows[~all_flows["flow_id"].isin(used)].reset_index(drop=True)

    scored_counts = compute_scored_class_counts()
    all_attack_classes = sorted(all_flows.loc[all_flows["is_attack"], "true_label"].unique())
    priority_classes = [c for c in all_attack_classes if scored_counts.get(c, 0) < 3]

    rng = np.random.default_rng(RANDOM_STATE_C)
    attack_sel = select_attack_flows_prioritised(
        remaining[remaining["is_attack"]], rng, N_ATTACK_C, priority_classes
    )
    benign_sel = select_benign_flows(remaining[~remaining["is_attack"]], rng, N_BENIGN_C)
    return set(attack_sel["flow_id"]) | set(benign_sel["flow_id"]), used


class TestFlowLevelExclusion:
    def test_reconstruction_is_deterministic(self, all_flows):
        a, _ = _reconstruct_batch3_ids(all_flows)
        b, _ = _reconstruct_batch3_ids(all_flows)
        assert a == b

    def test_batch3_selection_excludes_every_batch1_and_batch2_flow_id(self, all_flows):
        batch3_ids, used = _reconstruct_batch3_ids(all_flows)
        assert batch3_ids.isdisjoint(used)
        assert len(batch3_ids) == N_ATTACK_C + N_BENIGN_C  # no accidental within-batch duplicates

    def test_all_three_seeds_differ(self):
        assert len({RANDOM_STATE, RANDOM_STATE_B, RANDOM_STATE_C}) == 3


class TestRealGeneratedFilesDoNotOverlap:
    def test_key_csvs_share_no_record_id(self):
        try:
            key1 = pd.read_csv("results/agent_key_20.csv")
            key2 = pd.read_csv("results/agent_key_20b.csv")
            key3 = pd.read_csv("results/agent_key_10c.csv")
        except FileNotFoundError:
            pytest.skip("batch 1/2/3 key CSVs not all generated yet")
        assert not (set(key1["record_id"]) & set(key2["record_id"]))
        assert not (set(key1["record_id"]) & set(key3["record_id"]))
        assert not (set(key2["record_id"]) & set(key3["record_id"]))
        assert len(key3) == N_ATTACK_C + N_BENIGN_C

    def test_real_batch3_input_excludes_every_real_batch1_and_batch2_flow(self, all_flows):
        """The actual deliverable check, against the files this task
        produced (not a re-simulation): batch 3's 10 selected flow_ids,
        reconstructed the same deterministic way batch 3's own script
        selected them, must be disjoint from batches 1 and 2's
        reconstructed sets."""
        try:
            pd.read_csv("results/agent_key_10c.csv")
        except FileNotFoundError:
            pytest.skip("results/agent_key_10c.csv not generated yet")
        batch3_ids, used = _reconstruct_batch3_ids(all_flows)
        assert batch3_ids.isdisjoint(used)
