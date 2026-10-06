"""Batch 2: a second, non-overlapping 20-record selection from
`results/escalated_flows.csv`, replicating `eval/select_agent_sample.py`'s
protocol exactly except for the seed and the exclusion of every flow
batch 1 already used. SELECTION ONLY -- no agent is run here.

Excludes batch 1's 20 flows by REPLAYING batch 1's own selection
(`reconstruct_batch1_flow_ids` -- a pure function of the unchanged
`escalated_flows.csv` and batch 1's own RANDOM_STATE, not a lookup: see
that function's docstring for why `results/agent_key_20.csv` alone
can't answer "which flows did batch 1 draw", since it only carries the
fresh opaque `record_id`, not the source `flow_id`), then draws batch 2
with `random_state=21` from what's left. Overlap is asserted zero, not
just assumed.

Run: python -m eval.select_agent_sample_batch2
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eval.select_agent_sample import (
    ESCALATED_FLOWS_CSV,
    N_ATTACK,
    N_BENIGN,
    build_records,
    compute_feature_bins,
    fit_signature_table_for_bins,
    full_cicflowmeter_features,
    reconstruct_batch1_flow_ids,
    select_attack_flows,
    select_benign_flows,
    write_and_verify_jsonl,
)

RANDOM_STATE_B = 21
AGENT_INPUT_PATH_B = Path("results/agent_input_20b.jsonl")
AGENT_KEY_PATH_B = Path("results/agent_key_20b.csv")


def reconstruct_batch2_flow_ids(all_flows: pd.DataFrame) -> set:
    """Same replay principle as `reconstruct_batch1_flow_ids` (batch 2's
    own key CSV only carries a fresh opaque `record_id`, not the source
    `flow_id`): batch 2 is a pure function of (all_flows minus batch 1's
    flows, RANDOM_STATE_B), so replaying it here -- excluding batch 1
    first, exactly as `main()` above does -- reproduces the same 20
    flow_ids batch 2 actually drew. Used by batch 3 to exclude both
    batches' flows without re-selecting either."""
    batch1_ids = reconstruct_batch1_flow_ids(all_flows)
    remaining = all_flows[~all_flows["flow_id"].isin(batch1_ids)].reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_STATE_B)
    attack_sel = select_attack_flows(remaining[remaining["is_attack"]], rng, N_ATTACK)
    benign_sel = select_benign_flows(remaining[~remaining["is_attack"]], rng, N_BENIGN)
    return set(attack_sel["flow_id"]) | set(benign_sel["flow_id"])


def main() -> int:
    print(f"reading {ESCALATED_FLOWS_CSV} ...")
    all_flows = pd.read_csv(ESCALATED_FLOWS_CSV)
    print(f"total {len(all_flows):,}: {int(all_flows['is_attack'].sum())} attacks, "
          f"{int((~all_flows['is_attack']).sum())} benign")

    batch1_ids = reconstruct_batch1_flow_ids(all_flows)
    print(f"reconstructed {len(batch1_ids)} batch-1 flow_ids to exclude")

    remaining = all_flows[~all_flows["flow_id"].isin(batch1_ids)].reset_index(drop=True)
    print(f"remaining pool after excluding batch 1: {len(remaining):,} "
          f"({int(remaining['is_attack'].sum())} attacks, {int((~remaining['is_attack']).sum())} benign)")

    rng = np.random.default_rng(RANDOM_STATE_B)
    attack_sel = select_attack_flows(remaining[remaining["is_attack"]], rng, N_ATTACK)
    benign_sel = select_benign_flows(remaining[~remaining["is_attack"]], rng, N_BENIGN)
    selected = pd.concat([attack_sel, benign_sel], ignore_index=True)
    assert len(selected) == N_ATTACK + N_BENIGN

    overlap = set(selected["flow_id"]) & batch1_ids
    assert not overlap, f"batch 2 overlaps batch 1 on {len(overlap)} flow_id(s): {overlap}"
    print(f"selected {len(attack_sel)} attacks (classes: "
          f"{dict(attack_sel['true_label'].value_counts())}) + {len(benign_sel)} benign -- zero overlap with batch 1 confirmed")

    print("fitting signature table (reused, not re-fit) for feature-bin digests ...")
    pool, signature_table = fit_signature_table_for_bins()

    flow_ids = selected["flow_id"].tolist()
    pool_rows = pool.loc[pool["flow_uid"].isin(set(flow_ids)), ["flow_uid", "Label"]].set_index("flow_uid")
    for _, row in selected.iterrows():
        assert pool_rows.loc[row["flow_id"], "Label"] == row["true_label"], (
            f"Label mismatch for {row['flow_id']} -- flow_uid join is unreliable"
        )

    bins_by_id = compute_feature_bins(flow_ids, pool, signature_table)

    print("extracting full CICFlowMeter-style features from each flow's original CSV row ...")
    features_by_id = {fid: full_cicflowmeter_features(fid) for fid in flow_ids}
    n_feat = len(next(iter(features_by_id.values())))
    print(f"{n_feat} feature columns per record")

    rng2 = np.random.default_rng(RANDOM_STATE_B + 1)  # separate stream for output-order shuffling
    records, key_df = build_records(selected, features_by_id, bins_by_id, rng2)

    write_and_verify_jsonl(records, AGENT_INPUT_PATH_B)
    print(f"wrote {AGENT_INPUT_PATH_B} ({len(records)} records) -- no ground truth, verified by reading back")

    key_df.to_csv(AGENT_KEY_PATH_B, index=False)
    print(f"wrote {AGENT_KEY_PATH_B} ({len(key_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
