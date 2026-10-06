"""Batch 3: a third, non-overlapping selection from
`results/escalated_flows.csv` -- 8 attack + 2 benign (not the 12/8 split
batches 1-2 used; see module docstring in the task write-up: specificity
is already 13/13 across both batches, recall at 12/23 is the uncertain
number, so this batch is deliberately attack-weighted). SELECTION ONLY --
no agent is run here.

Excludes every flow batches 1 AND 2 used by REPLAYING both selections
(`reconstruct_batch1_flow_ids`, `reconstruct_batch2_flow_ids` -- pure
functions of the unchanged `escalated_flows.csv` and each batch's own
seed, not a lookup -- see those functions' docstrings). Overlap with
either prior batch is asserted zero, not just assumed.

Unlike batches 1-2's flat "2 per class, alphabetical, then largest
top-up" rule, batch 3 first PRIORITISES any attack class with fewer than
3 SCORED observations across batches 1+2 (`compute_scored_class_counts`,
counted from the actual run files, not the selection files, so a
class's own scoring failures -- e.g. batch 2's 4 A3 failures -- count
against it) up to 2 per such class (same cap as the normal rule, so a
priority class isn't over-represented either), THEN fills any remaining
attack slots via the normal batches-1/2 rule
(`eval.select_agent_sample.select_attack_flows`) restricted to whatever
classes are left. A priority class already fully exhausted in the pool
(0 remaining rows) cannot be represented no matter how few scored
observations it has -- reported plainly by `main()`, not silently
skipped.

Run: python -m eval.select_agent_sample_batch3
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from eval.select_agent_sample import (
    ESCALATED_FLOWS_CSV,
    build_records,
    compute_feature_bins,
    fit_signature_table_for_bins,
    full_cicflowmeter_features,
    reconstruct_batch1_flow_ids,
    select_attack_flows,
    select_benign_flows,
    write_and_verify_jsonl,
)
from eval.select_agent_sample_batch2 import reconstruct_batch2_flow_ids

RANDOM_STATE_C = 22
N_ATTACK_C = 8
N_BENIGN_C = 2
MAX_PER_CLASS = 2
MIN_SCORED_TO_NOT_PRIORITISE = 3  # a class with fewer than this many SCORED observations is prioritised

AGENT_INPUT_PATH_C = Path("results/agent_input_10c.jsonl")
AGENT_KEY_PATH_C = Path("results/agent_key_10c.csv")

KEY_PATH_A = Path("results/agent_key_20.csv")
KEY_PATH_B = Path("results/agent_key_20b.csv")
RUN_PATH_A = Path("results/agent_run_20_v2.jsonl")
RUN_PATH_B = Path("results/agent_run_20b.jsonl")


def _scored_record_ids(run_path: Path) -> set:
    ids = set()
    with open(run_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["record_id"])
    return ids


def compute_scored_class_counts() -> "pd.Series[int]":
    """Per attack class, how many records were actually SCORED (present
    in the run's own output jsonl, not just selected) across batches 1
    and 2 combined. Batch 1 scored all 20 it selected; batch 2 lost 4 of
    20 to A3 schema failures (results/agent_scoring_20b.md) -- a class
    that lost its only scored record there counts as 0 from batch 2, not
    1, since the point of prioritising is "how much real evidence do we
    have", not "how many were drawn"."""
    key_a = pd.read_csv(KEY_PATH_A)
    key_b = pd.read_csv(KEY_PATH_B)
    scored_a = _scored_record_ids(RUN_PATH_A)
    scored_b = _scored_record_ids(RUN_PATH_B)

    a_scored = key_a[key_a["record_id"].isin(scored_a) & key_a["is_attack"]]
    b_scored = key_b[key_b["record_id"].isin(scored_b) & key_b["is_attack"]]
    counts = a_scored["true_label"].value_counts().add(b_scored["true_label"].value_counts(), fill_value=0)
    return counts.astype(int).sort_index()


def select_attack_flows_prioritised(
    attacks: pd.DataFrame, rng: np.random.Generator, n: int, priority_classes: Iterable[str],
    max_per_class: int = MAX_PER_CLASS,
) -> pd.DataFrame:
    """Phase A: up to `max_per_class` from each priority class (sorted
    alphabetically among themselves, same neutral-order convention as
    `select_attack_flows`), skipping any priority class with zero rows
    left in `attacks` (exhausted -- cannot be represented, caller reports
    this). Phase B: whatever slots remain are filled by
    `select_attack_flows`'s normal 2-per-class-then-largest-topup rule,
    applied ONLY to the non-priority-class remainder, so a priority class
    is never drawn a second time beyond its Phase A allocation (keeping
    the same 2-per-class ceiling as batches 1/2, not stacking two
    allocations on the same class)."""
    priority_classes = sorted(set(priority_classes))
    picked_idx: List[int] = []
    for c in priority_classes:
        if len(picked_idx) >= n:
            break
        idx = attacks.index[attacks["true_label"] == c].to_numpy()
        if len(idx) == 0:
            continue  # exhausted in the pool -- cannot represent, not an error
        shuffled = idx[rng.permutation(len(idx))]
        take = min(max_per_class, len(shuffled), n - len(picked_idx))
        picked_idx.extend(shuffled[:take].tolist())

    remaining_needed = n - len(picked_idx)
    if remaining_needed > 0:
        rest_pool = attacks.drop(index=picked_idx)
        rest_pool = rest_pool[~rest_pool["true_label"].isin(priority_classes)]
        topped_up = select_attack_flows(rest_pool, rng, remaining_needed)
        picked_idx.extend(topped_up.index.tolist())

    assert len(picked_idx) == n, f"could only fill {len(picked_idx)}/{n} attack slots"
    return attacks.loc[picked_idx]


def main() -> int:
    print(f"reading {ESCALATED_FLOWS_CSV} ...")
    all_flows = pd.read_csv(ESCALATED_FLOWS_CSV)

    batch1_ids = reconstruct_batch1_flow_ids(all_flows)
    batch2_ids = reconstruct_batch2_flow_ids(all_flows)
    used_ids = batch1_ids | batch2_ids
    print(f"reconstructed {len(batch1_ids)} batch-1 + {len(batch2_ids)} batch-2 flow_ids to exclude "
          f"({len(used_ids)} total, overlap between them: {len(batch1_ids & batch2_ids)})")

    remaining = all_flows[~all_flows["flow_id"].isin(used_ids)].reset_index(drop=True)
    print(f"remaining pool after excluding batches 1+2: {len(remaining):,} "
          f"({int(remaining['is_attack'].sum())} attacks, {int((~remaining['is_attack']).sum())} benign)")

    scored_counts = compute_scored_class_counts()
    all_attack_classes = sorted(all_flows.loc[all_flows["is_attack"], "true_label"].unique())
    priority_classes = [
        c for c in all_attack_classes if scored_counts.get(c, 0) < MIN_SCORED_TO_NOT_PRIORITISE
    ]
    print(f"scored counts across batches 1+2: {scored_counts.to_dict()}")
    print(f"priority classes (scored < {MIN_SCORED_TO_NOT_PRIORITISE}): {priority_classes}")

    remaining_attack_pool = remaining[remaining["is_attack"]]
    exhausted_priority = [
        c for c in priority_classes if (remaining_attack_pool["true_label"] == c).sum() == 0
    ]
    if exhausted_priority:
        print(f"[note] priority class(es) with ZERO remaining rows in the pool -- cannot be "
              f"represented no matter the priority: {exhausted_priority}")

    rng = np.random.default_rng(RANDOM_STATE_C)
    attack_sel = select_attack_flows_prioritised(remaining_attack_pool, rng, N_ATTACK_C, priority_classes)
    benign_sel = select_benign_flows(remaining[~remaining["is_attack"]], rng, N_BENIGN_C)
    selected = pd.concat([attack_sel, benign_sel], ignore_index=True)
    assert len(selected) == N_ATTACK_C + N_BENIGN_C

    overlap = set(selected["flow_id"]) & used_ids
    assert not overlap, f"batch 3 overlaps batch 1/2 on {len(overlap)} flow_id(s): {overlap}"
    print(f"selected {len(attack_sel)} attacks (classes: "
          f"{dict(attack_sel['true_label'].value_counts())}) + {len(benign_sel)} benign -- "
          "zero overlap with batches 1/2 confirmed")

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

    rng2 = np.random.default_rng(RANDOM_STATE_C + 1)  # separate stream for output-order shuffling
    records, key_df = build_records(selected, features_by_id, bins_by_id, rng2)

    write_and_verify_jsonl(records, AGENT_INPUT_PATH_C)
    print(f"wrote {AGENT_INPUT_PATH_C} ({len(records)} records) -- no ground truth, verified by reading back")

    key_df.to_csv(AGENT_KEY_PATH_C, index=False)
    print(f"wrote {AGENT_KEY_PATH_C} ({len(key_df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
