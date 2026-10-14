"""Draws the fresh 20-record evaluation sample (10 attack, 10 benign) from the escalated
flows of the existing policy run, results/escalated_flows.csv. SELECTION ONLY -- no model
is run, and no previous evaluation output is read: the 50 flows already used as agent
input (batches 20 / 20b / 10c) are excluded by matching their INPUT feature vectors
(results/agent_input_*.jsonl, what the agents were shown) against this pool's raw rows,
never by reading any run, score or key file.

Draw, per group (attack / benign) independently, one seeded generator:
  1. eligible capture days = days with >= 1 unused row of that group, sorted by name
  2. rows within each day shuffled; day order shuffled
  3. round-robin over that day order, one row per day per pass, until 10 are taken
So every eligible capture day is represented before any day gets a second record.

Writes results/eval20_records.json -- the KEY file (record_id, flow_id, day, protocol,
true_label). It must never be reachable from a decision path; tests assert that.

Run: python -m eval.draw_eval20
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from eval.select_agent_sample import ESCALATED_FLOWS_CSV, full_cicflowmeter_features, parse_flow_uid

SEED = 20261014
N_PER_GROUP = 10
OUT = Path("results/eval20_records.json")
PRIOR_INPUTS = [Path("results/agent_input_20.jsonl"), Path("results/agent_input_20b.jsonl"),
                Path("results/agent_input_10c.jsonl")]
PROTOCOLS = {"6": "TCP", "17": "UDP", "0": "HOPOPT"}


def _vec(features: dict) -> tuple:
    return tuple(sorted((k, None if v != v else float(v)) for k, v in features.items()))


def used_flow_ids(flows: pd.DataFrame) -> set:
    prior = set()
    for p in PRIOR_INPUTS:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prior.add(_vec(json.loads(line)["features"]))
    used = {fid for fid in flows["flow_id"] if _vec(full_cicflowmeter_features(fid)) in prior}
    return used


def draw_group(rows: pd.DataFrame, rng: np.random.Generator, n: int) -> list:
    days = sorted(rows["day"].unique())
    queues = {}
    for d in days:
        ids = rows.loc[rows["day"] == d, "flow_id"].to_numpy()
        queues[d] = list(ids[rng.permutation(len(ids))])
    order = [days[i] for i in rng.permutation(len(days))]
    picked = []
    while len(picked) < n:
        progressed = False
        for d in order:
            if len(picked) < n and queues[d]:
                picked.append(queues[d].pop(0))
                progressed = True
        assert progressed, f"pool exhausted at {len(picked)}/{n}"
    return picked


def main() -> int:
    flows = pd.read_csv(ESCALATED_FLOWS_CSV)
    flows["day"] = flows["flow_id"].map(lambda f: parse_flow_uid(f)[0])
    used = used_flow_ids(flows)
    print(f"{len(flows)} escalated flows; {len(used)} already used as agent input, excluded")
    assert len(used) == 50, f"expected the 50 prior input flows to match exactly, got {len(used)}"
    pool = flows[~flows["flow_id"].isin(used)]

    rng = np.random.default_rng(SEED)
    attack = draw_group(pool[pool["is_attack"]], rng, N_PER_GROUP)
    benign = draw_group(pool[~pool["is_attack"]], rng, N_PER_GROUP)
    chosen = pool.set_index("flow_id").loc[attack + benign]

    order = rng.permutation(len(chosen))  # position must not encode the group
    records = []
    for i in order:
        fid = chosen.index[i]
        row = chosen.iloc[i]
        proto = fid.rsplit("-", 1)[1]
        records.append({
            "record_id": rng.bytes(16).hex(),
            "flow_id": fid,
            "day": row["day"],
            "protocol": PROTOCOLS.get(proto, proto),
            "true_label": row["true_label"],
            "is_attack": bool(row["is_attack"]),
        })
    assert len({r["record_id"] for r in records}) == 20

    attack_days = Counter(r["day"] for r in records if r["is_attack"])
    benign_days = Counter(r["day"] for r in records if not r["is_attack"])
    OUT.write_text(json.dumps({
        "seed": SEED,
        "source": str(ESCALATED_FLOWS_CSV).replace("\\", "/"),
        "excluded_prior_input_flows": len(used),
        "day_distribution": {"attack": dict(sorted(attack_days.items())),
                             "benign": dict(sorted(benign_days.items()))},
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print("attack days:", dict(sorted(attack_days.items())))
    print("benign days:", dict(sorted(benign_days.items())))
    print("protocols:", Counter(r["protocol"] for r in records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
