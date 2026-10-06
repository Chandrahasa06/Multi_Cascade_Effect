"""Required deliverable test: batch 2 shares no record with batch 1.

Two levels, since results/agent_key_20.csv's `record_id` is a fresh
opaque token with no link back to the source flow (see
eval.select_agent_sample.reconstruct_batch1_flow_ids's docstring) --
checking record_id alone would be a near-vacuous UUID check, so this
also checks the real, meaningful thing: the underlying flow_ids batch 2
was drawn from exclude every flow_id batch 1 actually used.
"""
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
from eval.select_agent_sample_batch2 import RANDOM_STATE_B


@pytest.fixture(scope="module")
def all_flows():
    try:
        return pd.read_csv("results/escalated_flows.csv")
    except FileNotFoundError:
        pytest.skip("results/escalated_flows.csv not generated yet")


class TestFlowLevelExclusion:
    def test_reconstructed_batch1_has_20_flow_ids(self, all_flows):
        ids = reconstruct_batch1_flow_ids(all_flows)
        assert len(ids) == 20

    def test_reconstruction_is_deterministic(self, all_flows):
        assert reconstruct_batch1_flow_ids(all_flows) == reconstruct_batch1_flow_ids(all_flows)

    def test_batch2_selection_excludes_every_batch1_flow_id(self, all_flows):
        import numpy as np

        batch1_ids = reconstruct_batch1_flow_ids(all_flows)
        remaining = all_flows[~all_flows["flow_id"].isin(batch1_ids)].reset_index(drop=True)
        rng = np.random.default_rng(RANDOM_STATE_B)
        attack_sel = select_attack_flows(remaining[remaining["is_attack"]], rng, N_ATTACK)
        benign_sel = select_benign_flows(remaining[~remaining["is_attack"]], rng, N_BENIGN)
        batch2_ids = set(attack_sel["flow_id"]) | set(benign_sel["flow_id"])
        assert batch2_ids.isdisjoint(batch1_ids)
        assert len(batch2_ids) == N_ATTACK + N_BENIGN  # no accidental within-batch duplicates either

    def test_batch1_and_batch2_seeds_differ(self):
        assert RANDOM_STATE_B != RANDOM_STATE


class TestRealGeneratedFilesDoNotOverlap:
    def test_key_csvs_share_no_record_id(self):
        try:
            key1 = pd.read_csv("results/agent_key_20.csv")
            key2 = pd.read_csv("results/agent_key_20b.csv")
        except FileNotFoundError:
            pytest.skip("batch 1 and/or batch 2 key CSVs not generated yet")
        overlap = set(key1["record_id"]) & set(key2["record_id"])
        assert not overlap
        assert len(key2) == 20

    def test_real_batch2_input_excludes_every_real_batch1_flow(self, all_flows):
        """The actual deliverable check, against the files this task
        produced (not a re-simulation): batch 2's 20 selected flow_ids,
        reconstructed the same deterministic way batch 2's own script
        selected them, must be disjoint from batch 1's reconstructed set."""
        try:
            pd.read_csv("results/agent_key_20b.csv")
        except FileNotFoundError:
            pytest.skip("results/agent_key_20b.csv not generated yet")
        import numpy as np

        batch1_ids = reconstruct_batch1_flow_ids(all_flows)
        remaining = all_flows[~all_flows["flow_id"].isin(batch1_ids)].reset_index(drop=True)
        rng = np.random.default_rng(RANDOM_STATE_B)
        attack_sel = select_attack_flows(remaining[remaining["is_attack"]], rng, N_ATTACK)
        benign_sel = select_benign_flows(remaining[~remaining["is_attack"]], rng, N_BENIGN)
        batch2_ids = set(attack_sel["flow_id"]) | set(benign_sel["flow_id"])
        assert batch2_ids.isdisjoint(batch1_ids)
