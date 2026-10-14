"""Recovers each blind record's own IP protocol (6 TCP, 17 UDP) by matching its feature
values to rows of the raw TrafficLabelling CSVs. Reads Protocol only; never Label.
Writes results/record_protocols.json: {record_id: {"protocol": "TCP"|"UDP"|None, "status": ...}}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.escalation_data import CSV_DIR, DAY_FILES
from eval.run_blind_pipeline_45_v2 import load_45

OUT = Path("results/record_protocols.json")
MATCH = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
         "Total Length of Bwd Packets", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
         "act_data_pkt_fwd", "min_seg_size_forward", "Destination Port", "Flow IAT Mean"]
NAMES = {6: "TCP", 17: "UDP"}


def protocols_for(blinds) -> dict:
    keep = set(MATCH) | {"Protocol"}
    frames = []
    for _day, fname, _w in DAY_FILES:
        try:
            df = pd.read_csv(CSV_DIR / fname, usecols=lambda c: c.strip() in keep, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_DIR / fname, usecols=lambda c: c.strip() in keep, encoding="cp1252", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df.apply(pd.to_numeric, errors="coerce"))
    pool = pd.concat(frames, ignore_index=True)
    X = pool[MATCH].to_numpy(dtype=np.float64)
    proto = pool["Protocol"].to_numpy()
    out = {}
    for b in blinds:
        x = np.array([float(b["features"][c]) for c in MATCH])
        idx = np.flatnonzero(np.all(X == x, axis=1))
        ps = sorted({int(p) for p in proto[idx] if p == p})
        if len(ps) == 1:
            out[b["record_id"]] = {"protocol": NAMES.get(ps[0], str(ps[0])), "status": "unique", "matches": int(len(idx))}
        else:
            out[b["record_id"]] = {"protocol": None, "status": "none" if not ps else f"ambiguous {ps}", "matches": int(len(idx))}
    return out


def main() -> int:
    out = protocols_for(load_45())
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    from collections import Counter
    print(Counter((v["protocol"], v["status"]) for v in out.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
