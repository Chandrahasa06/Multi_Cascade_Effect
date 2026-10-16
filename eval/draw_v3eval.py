"""Draws the v3eval sample from results/escalated_flows_v3.csv (the v3 policy run). Separate from
eval20: a different policy run, never merged with it. SELECTION ONLY -- no model is run.

  - attacks: all 42 escalated v3 attacks (all TCP), no sampling. Flows already evaluated before
    (same CSV row in eval20 or the earlier 50) are kept, as instructed, and flagged per record.
  - benign: 42 TCP flows by the same seeded day-cycling draw as eval20 (eval/draw_eval20.py's
    draw_group, same seed), from benign TCP flows never evaluated before.

Writes results/v3eval_records.json -- the KEY file. No decision path may read it.

Run: python -m eval.draw_v3eval
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from eval.draw_eval20 import SEED, draw_group, protocol_of, used_flow_ids
from eval.select_agent_sample import ESCALATED_FLOWS_CSV, parse_flow_uid

V3_CSV = Path("results/escalated_flows_v3.csv")
EVAL20_KEY = Path("results/eval20_records.json")
OUT = Path("results/v3eval_records.json")


def previously_evaluated() -> set:
    v1 = pd.read_csv(ESCALATED_FLOWS_CSV)
    prior = used_flow_ids(v1)
    assert len(prior) == 50
    e20 = {r["flow_id"] for r in json.loads(EVAL20_KEY.read_text(encoding="utf-8"))["records"]}
    return prior | e20


def main() -> int:
    flows = pd.read_csv(V3_CSV)
    flows["day"] = flows["flow_id"].map(lambda f: parse_flow_uid(f)[0])
    flows["protocol"] = flows["flow_id"].map(protocol_of)
    seen = previously_evaluated()

    attacks = flows[flows["is_attack"]]
    assert len(attacks) == 42 and (attacks["protocol"] == "TCP").all()
    benign_pool = flows[~flows["is_attack"] & (flows["protocol"] == "TCP") & ~flows["flow_id"].isin(seen)]

    rng = np.random.default_rng(SEED)
    benign = draw_group(benign_pool, rng, len(attacks))
    assert len(benign) == len(attacks), f"benign shortfall: {len(benign)}/{len(attacks)}"
    chosen = pd.concat([attacks.set_index("flow_id"), benign_pool.set_index("flow_id").loc[benign]])

    order = rng.permutation(len(chosen))  # position must not encode the group
    records = []
    for i in order:
        fid = chosen.index[i]
        row = chosen.iloc[i]
        records.append({"record_id": rng.bytes(16).hex(), "flow_id": fid, "day": row["day"],
                        "protocol": row["protocol"], "true_label": row["true_label"],
                        "is_attack": bool(row["is_attack"]), "previously_evaluated": fid in seen})
    assert len({r["record_id"] for r in records}) == len(records) == 84

    att = [r for r in records if r["is_attack"]]
    ben = [r for r in records if not r["is_attack"]]
    OUT.write_text(json.dumps({
        "seed": SEED, "source": str(V3_CSV).replace("\\", "/"),
        "benign_pool_size": int(len(benign_pool)),
        "excluded_from_benign_pool_as_previously_evaluated": int((~flows["is_attack"] & flows["flow_id"].isin(seen)).sum()),
        "attacks_previously_evaluated": sum(r["previously_evaluated"] for r in att),
        "class_distribution": dict(sorted(Counter(r["true_label"] for r in att).items())),
        "day_distribution": {"attack": dict(sorted(Counter(r["day"] for r in att).items())),
                             "benign": dict(sorted(Counter(r["day"] for r in ben).items()))},
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT}: {len(att)} attacks ({sum(r['previously_evaluated'] for r in att)} previously evaluated), "
          f"{len(ben)} benign from a pool of {len(benign_pool)}")
    print("attack classes:", dict(sorted(Counter(r["true_label"] for r in att).items())))
    print("previously evaluated attacks by class:",
          dict(Counter(r["true_label"] for r in att if r["previously_evaluated"])))
    print("benign days:", dict(sorted(Counter(r["day"] for r in ben).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
