"""Condition family v4: internal-consistency and machine-signature conditions, each checkable
from a single flow row with arithmetic only. No reference distribution, no TCP flag column
(results/flag_column_diagnosis.md: all twelve untrustworthy).

This module is the single source of truth: results/conditions_v4_preregistered.md is
generated from it (python -m eval.conditions_v4) and committed before any measurement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional

PREREG = Path("results/conditions_v4_preregistered.md")


class Condition(NamedTuple):
    cid: str
    scope: str          # "TCP", "UDP" or "any"
    expression: str     # exact boolean expression over CSV column names, as registered
    test: Callable[[Dict[str, float]], bool]
    justification: str
    disclosure: Optional[str] = None


def _pk(f):
    return f["Total Fwd Packets"] + f["Total Backward Packets"]


CONDITIONS: List[Condition] = [
    Condition("D1", "TCP", "min_seg_size_forward < 20",
              lambda f: f["min_seg_size_forward"] < 20,
              "A TCP header is at least 20 bytes, so a smaller minimum forward segment header on a TCP flow cannot come from a well-formed TCP stack."),
    Condition("D2", "any", "Fwd Packet Length Max == Fwd Packet Length Min AND Total Fwd Packets > 5",
              lambda f: f["Fwd Packet Length Max"] == f["Fwd Packet Length Min"] and f["Total Fwd Packets"] > 5,
              "Human-driven clients send requests of varying size; more than five client packets of exactly one size is the output of a fixed generator."),
    Condition("D3", "any", "Flow IAT Std == 0 AND (Total Fwd Packets + Total Backward Packets) > 3",
              lambda f: f["Flow IAT Std"] == 0 and _pk(f) > 3,
              "Network and human timing jitter make perfectly identical inter-arrival times across more than three packets possible only for a machine on a fixed clock."),
    Condition("D4", "any", "act_data_pkt_fwd == 0 AND Total Fwd Packets > 3",
              lambda f: f["act_data_pkt_fwd"] == 0 and f["Total Fwd Packets"] > 3,
              "A client that sends more than three packets without any of them carrying application data is probing or holding a connection, not using a service."),
    Condition("D5", "any", "Init_Win_bytes_backward > 0 AND Total Backward Packets == 0",
              lambda f: f["Init_Win_bytes_backward"] > 0 and f["Total Backward Packets"] == 0,
              "A receive window cannot be advertised in a direction that carried no packet, so this combination is an internal inconsistency in the record.",
              "Earlier measurement (results/flag_column_diagnosis.md) found Init_Win_bytes_backward = -1 exactly when Total Backward Packets = 0 on TCP, so this is expected never to hold. Kept as a consistency check."),
    Condition("D6", "any", "(Total Fwd Packets + Total Backward Packets) > 10 AND Flow Duration < 1000",
              lambda f: _pk(f) > 10 and f["Flow Duration"] < 1000,
              "More than ten packets inside one millisecond is a burst no request-response exchange paced by round trips produces; it is machine-generated sending."),
    Condition("D7", "any", "Packet Length Variance == 0 AND (Total Fwd Packets + Total Backward Packets) > 10",
              lambda f: f["Packet Length Variance"] == 0 and _pk(f) > 10,
              "More than ten packets of exactly the same size in both directions means neither side carried variable content, which is the signature of a fixed generator."),
    Condition("D8", "TCP", "Init_Win_bytes_forward == 0 AND act_data_pkt_fwd > 0",
              lambda f: f["Init_Win_bytes_forward"] == 0 and f["act_data_pkt_fwd"] > 0,
              "A client that opens by advertising a zero receive window and then sends data is not behaving like a standard TCP stack.",
              "The author has seen, in earlier reports, that some evaluation attacks have Init_Win_bytes_forward = 0. This condition is not independent of that knowledge."),
    Condition("D9", "any", "Total Length of Fwd Packets > 0 AND act_data_pkt_fwd == 0",
              lambda f: f["Total Length of Fwd Packets"] > 0 and f["act_data_pkt_fwd"] == 0,
              "If forward lengths count application bytes, forward bytes with no forward data packet is an internal inconsistency in the record."),
    Condition("D10", "any", "Fwd Packet Length Mean > Fwd Packet Length Max",
              lambda f: f["Fwd Packet Length Mean"] > f["Fwd Packet Length Max"],
              "A mean cannot exceed the maximum it is taken over, so this marks an arithmetically corrupt record."),
    Condition("D11", "any", "Flow Duration == 0 AND (Total Fwd Packets + Total Backward Packets) > 1",
              lambda f: f["Flow Duration"] == 0 and _pk(f) > 1,
              "Several packets with zero elapsed time between first and last can only come from a sender emitting them back-to-back faster than the clock resolves."),
    Condition("D12", "any", "Bwd Packet Length Max == Bwd Packet Length Min AND Total Backward Packets > 5",
              lambda f: f["Bwd Packet Length Max"] == f["Bwd Packet Length Min"] and f["Total Backward Packets"] > 5,
              "A server answering with content sends responses of varying size; more than five server packets of exactly one size means fixed replies, such as refusals or acknowledgements only."),
    Condition("D13", "any", "Fwd IAT Std == 0 AND Total Fwd Packets > 3",
              lambda f: f["Fwd IAT Std"] == 0 and f["Total Fwd Packets"] > 3,
              "Perfectly regular spacing of more than three client packets requires a machine timer; a human-driven client has jitter."),
    Condition("D14", "any", "Destination Port == 0",
              lambda f: f["Destination Port"] == 0,
              "Port 0 is reserved and hosts no service, so traffic addressed to it is never ordinary service use."),
]

COLUMNS = sorted({
    "min_seg_size_forward", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Total Fwd Packets", "Total Backward Packets", "Flow IAT Std", "act_data_pkt_fwd",
    "Init_Win_bytes_backward", "Init_Win_bytes_forward", "Flow Duration", "Packet Length Variance",
    "Total Length of Fwd Packets", "Bwd Packet Length Max", "Bwd Packet Length Min", "Fwd IAT Std",
    "Destination Port",
})

#: seed conditions considered and their fate
SEED_NOTES = [
    "D1-D8 are the seed list, kept as given. D1 and D8 are scoped to TCP: a UDP header is 8 bytes, and the window columns mean nothing for UDP.",
    "None of the conditions uses any of the twelve TCP flag columns.",
    "Considered and not added: comparisons of Subflow columns with Total columns (CICFlowMeter subflow semantics are not documented well enough to justify a consistency claim); any threshold on rates or sizes (those would need a reference distribution).",
]


def evaluate(features: Dict[str, float], protocol: Optional[str]) -> Dict[str, bool]:
    """True/False per condition; a condition scoped to the other protocol is False."""
    out = {}
    for c in CONDITIONS:
        if c.scope != "any" and c.scope != protocol:
            out[c.cid] = False
        else:
            out[c.cid] = bool(c.test(features))
    return out


def write_prereg() -> None:
    L = ["# Conditions v4, pre-registered before measurement\n",
         "Written and committed before any attack-labelled row was read for this task. Every condition is a boolean "
         "expression over one flow row's CSV columns, checkable with arithmetic only: no reference distribution, no "
         "comparison with other flows, no TCP flag column. Justifications appeal only to protocol semantics or to how "
         "machine-generated traffic differs from human-driven traffic.\n",
         "The expressions are defined once, in `eval/conditions_v4.py`; this file is generated from it.\n",
         "| id | scope | expression | justification |", "|---|---|---|---|"]
    for c in CONDITIONS:
        L.append(f"| {c.cid} | {c.scope} | `{c.expression}` | {c.justification} |")
    L.append("\n## Disclosures\n")
    L.append("The author is not blind to this data. Where earlier work bears on a condition, it is stated here.\n")
    for c in CONDITIONS:
        if c.disclosure:
            L.append(f"- {c.cid}: {c.disclosure}")
    L.append("- General: earlier reports showed the author the feature values of the 45 evaluation records, including the 7 attacks no v3 condition caught. The seed list came from the task brief; D9-D14 were added from protocol semantics.\n")
    L.append("## Notes on the list\n")
    for n in SEED_NOTES:
        L.append(f"- {n}")
    L.append("\n## Measurement plan (fixed now)\n")
    L.append("- Data: all eight pooled TrafficLabelling CSVs. Every attack row; every benign row (2.27M, more than the 200,000 minimum). Protocol from the CSV `Protocol` column (6 TCP, 17 UDP, other).")
    L.append("- Per condition: benign hold rate by protocol; attack hold rate by protocol and by attack class; separation = attack rate minus benign rate within protocol.")
    L.append("- Unusable: any condition whose benign hold rate exceeds 1% within the protocol it applies to.")
    L.append("- Per record: each D-condition evaluated on the 7 attacks that no v3 condition catches.")
    L.append("- No best subset is reported as a result. Any subset computed is labelled a selection-on-evaluation ceiling.")
    PREREG.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_prereg()
    print(f"wrote {PREREG}")
