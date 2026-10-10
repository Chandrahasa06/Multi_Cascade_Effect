# Conditions v4, pre-registered before measurement

Written and committed before any attack-labelled row was read for this task. Every condition is a boolean expression over one flow row's CSV columns, checkable with arithmetic only: no reference distribution, no comparison with other flows, no TCP flag column. Justifications appeal only to protocol semantics or to how machine-generated traffic differs from human-driven traffic.

The expressions are defined once, in `eval/conditions_v4.py`; this file is generated from it.

| id | scope | expression | justification |
|---|---|---|---|
| D1 | TCP | `min_seg_size_forward < 20` | A TCP header is at least 20 bytes, so a smaller minimum forward segment header on a TCP flow cannot come from a well-formed TCP stack. |
| D2 | any | `Fwd Packet Length Max == Fwd Packet Length Min AND Total Fwd Packets > 5` | Human-driven clients send requests of varying size; more than five client packets of exactly one size is the output of a fixed generator. |
| D3 | any | `Flow IAT Std == 0 AND (Total Fwd Packets + Total Backward Packets) > 3` | Network and human timing jitter make perfectly identical inter-arrival times across more than three packets possible only for a machine on a fixed clock. |
| D4 | any | `act_data_pkt_fwd == 0 AND Total Fwd Packets > 3` | A client that sends more than three packets without any of them carrying application data is probing or holding a connection, not using a service. |
| D5 | any | `Init_Win_bytes_backward > 0 AND Total Backward Packets == 0` | A receive window cannot be advertised in a direction that carried no packet, so this combination is an internal inconsistency in the record. |
| D6 | any | `(Total Fwd Packets + Total Backward Packets) > 10 AND Flow Duration < 1000` | More than ten packets inside one millisecond is a burst no request-response exchange paced by round trips produces; it is machine-generated sending. |
| D7 | any | `Packet Length Variance == 0 AND (Total Fwd Packets + Total Backward Packets) > 10` | More than ten packets of exactly the same size in both directions means neither side carried variable content, which is the signature of a fixed generator. |
| D8 | TCP | `Init_Win_bytes_forward == 0 AND act_data_pkt_fwd > 0` | A client that opens by advertising a zero receive window and then sends data is not behaving like a standard TCP stack. |
| D9 | any | `Total Length of Fwd Packets > 0 AND act_data_pkt_fwd == 0` | If forward lengths count application bytes, forward bytes with no forward data packet is an internal inconsistency in the record. |
| D10 | any | `Fwd Packet Length Mean > Fwd Packet Length Max` | A mean cannot exceed the maximum it is taken over, so this marks an arithmetically corrupt record. |
| D11 | any | `Flow Duration == 0 AND (Total Fwd Packets + Total Backward Packets) > 1` | Several packets with zero elapsed time between first and last can only come from a sender emitting them back-to-back faster than the clock resolves. |
| D12 | any | `Bwd Packet Length Max == Bwd Packet Length Min AND Total Backward Packets > 5` | A server answering with content sends responses of varying size; more than five server packets of exactly one size means fixed replies, such as refusals or acknowledgements only. |
| D13 | any | `Fwd IAT Std == 0 AND Total Fwd Packets > 3` | Perfectly regular spacing of more than three client packets requires a machine timer; a human-driven client has jitter. |
| D14 | any | `Destination Port == 0` | Port 0 is reserved and hosts no service, so traffic addressed to it is never ordinary service use. |

## Disclosures

The author is not blind to this data. Where earlier work bears on a condition, it is stated here.

- D5: Earlier measurement (results/flag_column_diagnosis.md) found Init_Win_bytes_backward = -1 exactly when Total Backward Packets = 0 on TCP, so this is expected never to hold. Kept as a consistency check.
- D8: The author has seen, in earlier reports, that some evaluation attacks have Init_Win_bytes_forward = 0. This condition is not independent of that knowledge.
- General: earlier reports showed the author the feature values of the 45 evaluation records, including the 7 attacks no v3 condition caught. The seed list came from the task brief; D9-D14 were added from protocol semantics.

## Notes on the list

- D1-D8 are the seed list, kept as given. D1 and D8 are scoped to TCP: a UDP header is 8 bytes, and the window columns mean nothing for UDP.
- None of the conditions uses any of the twelve TCP flag columns.
- Considered and not added: comparisons of Subflow columns with Total columns (CICFlowMeter subflow semantics are not documented well enough to justify a consistency claim); any threshold on rates or sizes (those would need a reference distribution).

## Measurement plan (fixed now)

- Data: all eight pooled TrafficLabelling CSVs. Every attack row; every benign row (2.27M, more than the 200,000 minimum). Protocol from the CSV `Protocol` column (6 TCP, 17 UDP, other).
- Per condition: benign hold rate by protocol; attack hold rate by protocol and by attack class; separation = attack rate minus benign rate within protocol.
- Unusable: any condition whose benign hold rate exceeds 1% within the protocol it applies to.
- Per record: each D-condition evaluated on the 7 attacks that no v3 condition catches.
- No best subset is reported as a result. Any subset computed is labelled a selection-on-evaluation ceiling.
