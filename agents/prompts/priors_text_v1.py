"""Written priors about ordinary enterprise traffic, version 1.

Fixed before any priors-run call. General networking knowledge of the kind a security
engineer writes from experience and from the protocol specifications. Nothing here is
derived from CICIDS2017 or from this project's data or results. Each statement carries an
id and a source label (PRIOR_SOURCES). The numbers are typical ranges from general
knowledge, not measurements.
"""
from __future__ import annotations

PRIORS_VERSION = "priors_text_v1"

#: (id, source, statement). The prompt shows the statements; the report shows all three.
PRIOR_STATEMENTS = (
    # how to read the measurements
    ("U1", "CICFlowMeter feature definitions (tool documentation)",
     "Durations and inter-arrival times are in microseconds. Lengths are in bytes. Rates are per second."),
    ("U2", "CICFlowMeter feature definitions (tool convention)",
     "Init_Win_bytes_forward and Init_Win_bytes_backward are the TCP receive window advertised in the "
     "first packet of each direction; -1 means no such packet was seen in that direction."),
    ("U3", "CICFlowMeter flow definition (tool documentation)",
     "A flow is one connection (or, for UDP, one exchange) between two endpoints. It ends at a FIN or "
     "RST, or after an idle or active timeout, so a long connection can be split into several flows."),
    # ordinary services
    ("S1", "RFC 1035 (DNS), general practice",
     "DNS on port 53 is mostly UDP: one query and one response, usually 1-2 packets each way, each under "
     "about 512 bytes for classic DNS, completing in milliseconds to tens of milliseconds."),
    ("S2", "RFC 9110/9112 (HTTP), general practice",
     "HTTP on port 80 (and alternates such as 8080 and 8000): the client sends requests of a few hundred "
     "bytes to a few kilobytes; responses are usually larger, from hundreds of bytes to megabytes. "
     "Server-to-client volume exceeding client-to-server volume is normal for browsing. Persistent "
     "connections may sit idle for seconds to minutes between requests."),
    ("S3", "RFC 8446 (TLS 1.3), general practice",
     "HTTPS on port 443 begins with a TLS handshake of a few kilobytes, then follows the same "
     "request/response pattern as HTTP. Long-lived, mostly idle HTTPS connections are common for web "
     "applications and push services."),
    ("S4", "RFC 4253 (SSH transport), general practice",
     "SSH on port 22: completing key exchange and authentication takes dozens of packets. Interactive "
     "sessions last minutes to hours with many small packets in both directions; file copies move large "
     "volumes one way. An SSH connection that ends after only a handful of packets is usually a failed "
     "login, a banner check or a scan."),
    ("S5", "RFC 959 (FTP), general practice",
     "FTP control on port 21 is a short exchange of text commands and replies; a login takes several "
     "round trips."),
    ("S6", "IANA port registry, RFC 6335",
     "Ports 0-1023 host standard services, 1024-49151 are registered, and 49152-65535 are ephemeral "
     "(client-side) ports. Ordinary client traffic is addressed to a service port; a destination in the "
     "ephemeral range is unusual unless a known application uses it. 8000, 8080 and 8443 are common for "
     "proxies, development servers and administration consoles."),
    # TCP behaviour
    ("T1", "RFC 9293 (TCP)",
     "An ordinary TCP connection opens with SYN, SYN-ACK, ACK; almost every later packet carries ACK; "
     "data segments often carry PSH; it closes with FIN in each direction, or with RST."),
    ("T2", "RFC 9293 (TCP), Nmap documentation (scan types)",
     "URG is almost never used by modern applications. SYN together with FIN, the FIN+PSH+URG "
     "combination, and packets with no flags do not occur in legitimate traffic."),
    ("T3", "Nmap documentation (scan types), general practice",
     "A SYN with no reply, a handshake followed by immediate close with no data, or a connection reset "
     "before any data is what port scans and refused connections produce."),
    ("T4", "RFC 9293, RFC 7323 (TCP window scaling)",
     "Ordinary operating systems advertise an initial receive window of several kilobytes to tens of "
     "kilobytes (often 8,192 to 65,535 bytes before scaling). A zero initial window is not ordinary "
     "client behaviour."),
    # suspicious on its own
    ("X1", "general security practice (slow-rate denial of service)",
     "A very long connection that carries very few packets is how slow-rate denial-of-service holds a "
     "server's connection slots open while sending almost nothing."),
    ("X2", "general security practice",
     "Repeated packets carrying no application data, especially to non-standard ports, are typical of "
     "scans, liveness checks and command-and-control polling. Benign keepalives exist too, so the port "
     "and the regularity matter."),
    ("X3", "general security practice",
     "Strong asymmetry in the unexpected direction is suspicious: a client pushing much more data to a "
     "web server than it receives, or a large one-way transfer on an unusual port."),
    ("X4", "general security practice",
     "Many tiny packets at a high rate in a short flow is flooding. Many requests in rapid succession to "
     "one web server from one client is request flooding."),
    ("X5", "general security practice",
     "Highly regular timing (near-identical inter-arrival times) means automation: legitimate for "
     "monitoring, but also how bots beacon to their controllers."),
    ("X6", "general security practice",
     "Behaviour that does not match the port is suspicious: web-like exchanges on a high port, or "
     "interactive-looking traffic on a web port."),
)

EXCULPATION_RULE = (
    "Resembling ordinary traffic is not by itself exculpatory. Capable attackers deliberately produce "
    "flows that look ordinary. A flow is benign only if you can say what, specifically, makes it benign, "
    "and nothing above that marks it as suspicious is present."
)

PRIORS_TEXT = (
    "What ordinary enterprise network traffic looks like (general knowledge, not measured on this "
    "network):\n"
    + "\n".join(f"[{sid}] {text}" for sid, _src, text in PRIOR_STATEMENTS)
    + "\n\n"
    + EXCULPATION_RULE
    + "\n"
)

PRIOR_IDS = tuple(sid for sid, _src, _text in PRIOR_STATEMENTS)
