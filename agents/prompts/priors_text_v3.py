"""Written priors, version 3: checkable conditions.

Fixed before any priors-v3 call. Each condition is a yes/no test on one or two of the flow's
own measured values (no reference data, no comparison with other flows, no statistics), with
a stable id, a protocol scope, a note on what it means, and a source. Thresholds are round
numbers from general networking knowledge, not fitted to any data.

No condition uses a TCP flag column: results/flag_column_diagnosis.md found all twelve
untrustworthy, and v1's apparent gain came from reading them.

Disclosure: the author had seen some of these 45 records' feature values in earlier reports
when writing the thresholds. They were chosen as round general-knowledge values, but their
independence from this sample is not complete.

v1 and v2 stay importable.
"""
from __future__ import annotations

PRIORS_VERSION = "priors_text_v3"

#: (id, scope, test, meaning, source). scope: "TCP", "UDP" or "any".
CONDITIONS = (
    ("C1", "TCP", "Init_Win_bytes_backward = -1",
     "The other side never answered: no packet came back in this flow. For TCP this is an unanswered "
     "connection attempt (a scan, an unreachable host, or a dropped connection). One can be accidental; "
     "it is not ordinary for a working service.",
     "RFC 9293 (TCP); CICFlowMeter convention that -1 means no packet in that direction"),
    ("C2", "any", "Total Backward Packets = 0",
     "Nothing came back at all. Expected for some one-way UDP services (multicast DNS, SSDP, syslog); "
     "for TCP it means the same as C1.",
     "RFC 9293, RFC 768 (UDP), RFC 6762 (mDNS)"),
    ("C3", "any", "Total Length of Fwd Packets > 0 and Total Length of Bwd Packets = 0",
     "Data was sent and no data came back. A request with no answer.",
     "general practice"),
    ("C4", "TCP", "Init_Win_bytes_forward = 0",
     "The client advertised a zero receive window at the start. Ordinary operating systems advertise "
     "several kilobytes or more; a zero window at connection start is not ordinary client behaviour.",
     "RFC 9293, RFC 7323 (window scaling)"),
    ("C5", "any", "Flow IAT Mean > 10000000 (10 seconds between packets on average)",
     "The flow was mostly silent: held open with long gaps. Idle keepalive connections do this; so does "
     "slow-rate denial of service, which ties up a server by sending almost nothing.",
     "RFC 9112 (persistent connections), general security practice"),
    ("C6", "any", "Total Length of Fwd Packets > 10 x Total Length of Bwd Packets, and Total Length of Bwd Packets > 0",
     "The client sent far more data than it received. Normal for uploads; unexpected for request/response "
     "services, which usually send back more than they receive.",
     "RFC 9110 (HTTP semantics), general practice"),
    ("C7", "any", "Total Fwd Packets + Total Backward Packets >= 3, and Packet Length Std = 0",
     "Every packet in the flow had the same size. Typical of automated probes, keepalives and beacons; "
     "human-driven exchanges vary in size.",
     "general security practice"),
    ("C8", "any", "Total Length of Fwd Packets = 0 and Total Length of Bwd Packets = 0",
     "No application data in either direction: control packets only. Probes and liveness checks look "
     "like this; so do benign keepalives.",
     "general security practice; CICFlowMeter lengths count application data"),
    ("C9", "any", "Destination Port >= 1024 and Destination Port not in {8000, 8080, 8443}",
     "The destination is not a standard service port or a common web alternate. Ordinary client traffic "
     "goes to a service port; registered and ephemeral ports need a known application to explain them.",
     "IANA port registry, RFC 6335"),
    ("C10", "TCP", "Destination Port in {21, 22, 23, 3389} and Total Fwd Packets + Total Backward Packets <= 10",
     "A login service (FTP, SSH, Telnet, RDP) with too few packets to complete a login session. Usually a "
     "failed or scripted login attempt, a banner check, or a scan.",
     "RFC 959 (FTP), RFC 4253 (SSH), general security practice"),
    ("C11", "any", "Flow Packets/s > 1000 and Average Packet Size < 100",
     "Many tiny packets at a high rate: flooding. Bulk transfers reach high rates with large packets, not "
     "small ones.",
     "general security practice"),
    ("C12", "UDP", "Destination Port = 53, Total Length of Bwd Packets > 10 x Total Length of Fwd Packets, and Total Length of Bwd Packets > 512",
     "A DNS answer far larger than its query and above the classic 512-byte limit: the shape of DNS "
     "amplification.",
     "RFC 1035 (DNS), general security practice"),
)

CONDITION_IDS = tuple(c[0] for c in CONDITIONS)

EXCULPATION_RULE = (
    "Resembling ordinary traffic is not by itself exculpatory. Capable attackers deliberately produce "
    "flows that look ordinary. A flow is benign only if you can say which conditions you checked, what "
    "values you read, and why the ones that hold have a benign explanation here."
)

APPLICABILITY_RULE = (
    "A condition scoped to TCP is \"not applicable\" for a UDP flow, and one scoped to UDP is \"not "
    "applicable\" for a TCP flow. A condition scoped \"any\" applies to every flow. No condition looks at "
    "TCP flags: the measurements do not record them, so do not reason about a handshake."
)

PRIORS_TEXT = (
    "Conditions to check on this flow. Each is a yes/no test on the flow's own measured values "
    "(general networking knowledge; no condition is measured on this network):\n"
    + "\n".join(f"[{cid}] ({scope}) {test}.\n     Meaning: {meaning}" for cid, scope, test, meaning, _src in CONDITIONS)
    + "\n\n" + APPLICABILITY_RULE + "\n\n" + EXCULPATION_RULE + "\n"
)
