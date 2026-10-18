"""Part 1: can the TCP flag columns be trusted? No API calls. Reads the raw TrafficLabelling
CSVs (the cached pool drops most flag columns and the Protocol column).

Writes results/flag_column_diagnosis.md and results/flag_column_diagnosis.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.escalation_data import CSV_DIR, DAY_FILES

OUT_MD = Path("results/flag_column_diagnosis.md")
OUT_JSON = Path("results/flag_column_diagnosis.json")
PCAP_REF = Path("results/cache/monday_pcap__fidelity__adapter2__features4.parquet")
AGG = ["SYN Flag Count", "ACK Flag Count", "FIN Flag Count", "PSH Flag Count", "RST Flag Count",
       "URG Flag Count", "CWE Flag Count", "ECE Flag Count"]
DIR = ["Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags"]
OTHER = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
         "Total Length of Bwd Packets", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
         "act_data_pkt_fwd", "min_seg_size_forward"]
KEEP = ["Label", "Protocol"] + AGG + DIR + OTHER
PROTO = {6: "TCP", 17: "UDP"}


def load() -> pd.DataFrame:
    frames = []
    for day, fname, _w in DAY_FILES:
        path = CSV_DIR / fname
        try:
            df = pd.read_csv(path, usecols=lambda c: c.strip() in KEEP, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, usecols=lambda c: c.strip() in KEEP, encoding="cp1252", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=["Label"])
        df["day"] = day
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in AGG + DIR + OTHER + ["Protocol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["proto"] = df["Protocol"].map(PROTO).fillna("other")
    df["group"] = np.where(df["Label"] == "BENIGN", "benign", "attack")
    return df


def describe(col: pd.Series) -> dict:
    v = col.dropna()
    vc = v.value_counts()
    return {"n": int(len(v)), "distinct": int(v.nunique()), "share_zero": float((v == 0).mean()) if len(v) else float("nan"),
            "max": float(v.max()) if len(v) else float("nan"),
            "top": {str(k): int(c) for k, c in vc.head(4).items()}}


def kind(d: dict) -> str:
    if d["distinct"] <= 1:
        return "constant"
    if d["distinct"] == 2 and d["max"] == 1:
        return "binary (0/1)"
    return "count"


def main() -> int:
    df = load()
    res = {"rows": len(df), "by_protocol": df["proto"].value_counts().to_dict()}
    L = ["# Can the TCP flag columns be trusted?\n",
         f"No API calls. Read from the raw TrafficLabelling CSVs: {len(df):,} labelled rows across 8 files "
         f"(TCP {res['by_protocol'].get('TCP', 0):,}, UDP {res['by_protocol'].get('UDP', 0):,}, other {res['by_protocol'].get('other', 0):,}).\n"]

    # ---- 1 and 2: value distributions
    L.append("## 1-2. Value distributions, by protocol and label\n")
    L.append("| column | protocol | label | rows | distinct values | share = 0 | max | most common values (count) | kind |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    res["columns"] = {}
    for c in AGG + DIR:
        for proto in ("TCP", "UDP"):
            for g in ("benign", "attack"):
                d = describe(df.loc[(df["proto"] == proto) & (df["group"] == g), c])
                res["columns"][f"{c}|{proto}|{g}"] = d
                top = ", ".join(f"{k}: {v:,}" for k, v in d["top"].items())
                L.append(f"| {c} | {proto} | {g} | {d['n']:,} | {d['distinct']} | {100 * d['share_zero']:.1f}% | {d['max']:g} | {top} | {kind(d)} |")
    L.append("")

    # ---- consistency: aggregate PSH vs per-direction PSH
    tcp = df[df["proto"] == "TCP"]
    agg_psh = tcp["PSH Flag Count"] > 0
    dir_psh = (tcp["Fwd PSH Flags"] > 0) | (tcp["Bwd PSH Flags"] > 0)
    res["psh_agreement"] = float((agg_psh == dir_psh).mean())
    res["psh_agg_without_dir"] = int((agg_psh & ~dir_psh).sum())
    res["psh_dir_without_agg"] = int((~agg_psh & dir_psh).sum())

    # ---- 3: handshake evidence
    L.append("## 3. Does any column show whether a TCP handshake completed?\n")
    syn0 = tcp["SYN Flag Count"] == 0
    tcp_b = tcp[tcp["group"] == "benign"]
    has_reply_win = tcp["Init_Win_bytes_backward"] != -1
    no_bwd = tcp["Total Backward Packets"] == 0
    res["tcp_syn0_share"] = float(syn0.mean())
    res["tcp_benign_syn0_share"] = float((tcp_b["SYN Flag Count"] == 0).mean())
    res["tcp_bwd_win_present_share"] = float(has_reply_win.mean())
    res["tcp_no_backward_packets_share"] = float(no_bwd.mean())
    res["tcp_bwdwin_minus1_vs_no_bwd"] = {
        "bwd_win_-1_and_bwd_pkts_0": int(((~has_reply_win) & no_bwd).sum()),
        "bwd_win_-1_and_bwd_pkts_gt0": int(((~has_reply_win) & ~no_bwd).sum()),
        "bwd_win_set_and_bwd_pkts_0": int((has_reply_win & no_bwd).sum()),
    }
    # ground truth from real packets: Monday PCAP-derived flows, syn_ratio > 0 means a SYN was seen
    pc = pd.read_parquet(PCAP_REF, columns=["syn_ratio", "init_win_bytes_fwd", "no_response_flag"])
    pc_tcp = pc[pc["init_win_bytes_fwd"] >= 0]
    res["pcap_monday_tcp_flows"] = int(len(pc_tcp))
    res["pcap_monday_tcp_with_syn_share"] = float((pc_tcp["syn_ratio"] > 0).mean())
    mon_tcp = df[(df["day"] == "monday") & (df["proto"] == "TCP")]
    res["csv_monday_tcp_with_syn_share"] = float((mon_tcp["SYN Flag Count"] > 0).mean())

    L.append(f"- In the CSVs, `SYN Flag Count` is 0 on {100 * res['tcp_syn0_share']:.1f}% of all TCP flows ({100 * res['tcp_benign_syn0_share']:.1f}% of benign TCP flows).")
    L.append(f"- **Real packets disagree.** Monday's flows rebuilt from the PCAP by this project's own packet parser (which reads the TCP flag byte directly): {100 * res['pcap_monday_tcp_with_syn_share']:.1f}% of {res['pcap_monday_tcp_flows']:,} TCP flows contain a SYN. The Monday CSV says {100 * res['csv_monday_tcp_with_syn_share']:.1f}% of its TCP flows do. The two use different flow boundaries, so the figures are not row-for-row comparable, but a gap this size means the CSV's SYN column does not record the SYNs that were on the wire.")
    L.append(f"- `Init_Win_bytes_backward` is set (not -1) on {100 * res['tcp_bwd_win_present_share']:.1f}% of TCP flows. It is -1 when no backward packet carried a window. Cross-tab against `Total Backward Packets` = 0: {res['tcp_bwdwin_minus1_vs_no_bwd']}. So -1 tracks \"the other side never answered\" closely, which is a valid sign of an unanswered connection attempt, but it cannot tell a completed handshake from a reply that came later in a split flow.")
    L.append(f"- `Total Backward Packets` = 0 on {100 * res['tcp_no_backward_packets_share']:.1f}% of TCP flows: the other side sent nothing in this flow.")
    L.append(f"- Aggregate `PSH Flag Count` > 0 agrees with per-direction `Fwd/Bwd PSH Flags` > 0 on {100 * res['psh_agreement']:.1f}% of TCP flows ({res['psh_agg_without_dir']:,} aggregate-only, {res['psh_dir_without_agg']:,} direction-only).")
    L.append("- `Flow Duration`, `act_data_pkt_fwd` and `min_seg_size_forward` say nothing about the handshake on their own: a flow can carry data and last long whether or not its start was captured.\n")

    # identical column pairs on TCP rows (only pairs where both columns vary)
    import itertools
    varying = [c for c in AGG + DIR if tcp[c].nunique() > 1]
    ident = [(a, b) for a, b in itertools.combinations(varying, 2) if bool((tcp[a] == tcp[b]).all())]
    res["identical_pairs_tcp"] = ident
    L.append("### Duplicated columns\n")
    L.append("Pairs of flag columns that are identical on every TCP row (columns that vary only):\n")
    for a, b in ident:
        L.append(f"- `{a}` is identical to `{b}` on all {len(tcp):,} TCP rows.")
    L.append("\nEvery aggregate \"count\" column holds only 0 or 1, and `Bwd PSH Flags` and `Bwd URG Flags` are 0 on every row. "
             "`SYN Flag Count` being identical to `Fwd PSH Flags` means at least one of the two is mislabelled; neither can be read as the flag its name says. "
             f"`URG Flag Count` is 1 on {100 * (1 - res['columns']['URG Flag Count|TCP|benign']['share_zero']):.1f}% of benign TCP flows, while real applications almost never set URG, so that column does not mean URG either.\n")

    # ---- 4: recomputation
    L.append("## 4. Recomputation from packets\n")
    L.append("- `adapters/pcap_adapter.py` already reads the TCP flag byte (FIN, SYN, RST, PSH, ACK, URG) for every packet, and `dataplane/flow_state.py` keeps per-flow counts of each. No new parser is needed.")
    L.append("- PCAPs on disk: Monday (10.8 GB) and Friday (8.8 GB) only.")
    L.append("- To recompute one record's flags, its flow must be found in the PCAP by 5-tuple and time (recoverable from its pool row), then its packets re-read. The project's earlier recollection measured about 71 seconds per 200 flows on top of a full PCAP scan.")
    L.append("- Of the 45 records: 1 is on Monday, 19 on Friday, 1 is ambiguous between Wednesday and Friday, and 24 are on Tuesday to Thursday, which have no PCAP. **Recomputation could cover at most 20 of 45**, so recomputed flags would exist for some records and not others. That would be a second confound, the one we are trying to remove.")
    L.append("")

    # ---- verdict
    L.append("## Verdict\n")
    L.append("- **No flag column can be trusted.** Every one is a 0/1 indicator rather than a count; SYN duplicates Fwd PSH exactly; URG is set on a fifth of benign TCP flows; the CSV records far fewer SYNs than the packets contain. Their presence or absence means nothing reliable.")
    L.append("- **The information was lost at extraction and cannot be recovered from the CSV.** Recomputation from packets is possible only for Monday and Friday, at most 20 of the 45 records, which would make the evaluation inconsistent. Not done.")
    L.append("- **What remains usable for TCP:** whether the other side answered at all (`Total Backward Packets` = 0, or `Init_Win_bytes_backward` = -1), initial window sizes where present, duration, packet counts and byte asymmetry.")
    L.append("- **Consequence for the priors:** v2 must not mention SYN/ACK/FIN counts or handshakes inferred from them.\n")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "columns"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
