"""Written priors about ordinary enterprise traffic, version 2: protocol-aware.

Fixed before any priors-v2 call. General networking knowledge from protocol specifications
and security practice; nothing derived from CICIDS2017 or this project's results. Each
statement carries an id and a source.

Changes from v1, and why:
  - Separate guidance for TCP, UDP, and protocol-independent signs. v1 treated all flows
    alike, so the agents sorted TCP from UDP.
  - No statement about TCP flags or the handshake. results/flag_column_diagnosis.md found
    that no flag column in these measurements can be trusted (they are 0/1 indicators, the
    SYN column duplicates another column exactly, URG is set on a fifth of benign TCP
    flows). The flag columns are also removed from what the agents see.
  - What remains for TCP: whether the other side answered at all, initial window sizes,
    duration, packet counts and byte direction.
v1 (priors_text_v1.py) stays importable.
"""
from __future__ import annotations

PRIORS_VERSION = "priors_text_v2"

PRIOR_STATEMENTS = (
    # how to read the measurements
    ("U1", "CICFlowMeter feature definitions (tool documentation)",
     "Durations and inter-arrival times are in microseconds. Lengths are in bytes. Rates are per second."),
    ("U2", "CICFlowMeter feature definitions (tool convention)",
     "Init_Win_bytes_forward and Init_Win_bytes_backward are the TCP receive window advertised in the "
     "first packet of each direction; -1 means no such packet was seen in that direction. For UDP they "
     "are always -1 and mean nothing."),
    ("U3", "CICFlowMeter flow definition (tool documentation)",
     "A flow is one connection (TCP) or one exchange (UDP) between two endpoints. A long connection can be "
     "split into several flows by timeouts, so a flow may begin or end partway through a connection."),
    ("U4", "this run's measurements (see the note on what is not shown)",
     "TCP flag indicators are not shown. These measurements do not record whether a TCP handshake took "
     "place, so do not infer a missing or completed handshake either way."),
    # TCP
    ("T1", "RFC 9293 (TCP), general practice",
     "TCP is connection-oriented: both sides exchange packets for the life of the connection. An ordinary "
     "TCP flow has packets in both directions."),
    ("T2", "RFC 9293, Nmap documentation (scan types)",
     "A TCP flow in which the other side sent nothing back (Total Backward Packets = 0, "
     "Init_Win_bytes_backward = -1) is an unanswered connection attempt: a scan, an unreachable host, or a "
     "dropped connection. One such flow can be accidental; it is not ordinary for a working service."),
    ("T3", "RFC 9293, RFC 7323 (TCP window scaling)",
     "Ordinary operating systems advertise an initial receive window of several kilobytes to tens of "
     "kilobytes (often 8,192 to 65,535 bytes before scaling). A zero initial window from the client is "
     "not ordinary client behaviour."),
    ("T4", "RFC 9110/9112 (HTTP), general practice",
     "HTTP on TCP port 80 (and alternates such as 8080 and 8000): requests of a few hundred bytes to a few "
     "kilobytes; responses usually larger, from hundreds of bytes to megabytes. Server-to-client volume "
     "exceeding client-to-server volume is normal for browsing. Persistent connections may sit idle for "
     "seconds to minutes between requests."),
    ("T5", "RFC 8446 (TLS 1.3), general practice",
     "HTTPS on TCP port 443 begins with a TLS exchange of a few kilobytes, then follows the HTTP pattern. "
     "Long-lived, mostly idle HTTPS connections are common for web applications and push services."),
    ("T6", "RFC 4253 (SSH transport), general practice",
     "SSH on TCP port 22: key exchange and authentication take dozens of packets. Interactive sessions "
     "last minutes to hours with many small packets both ways; file copies move large volumes one way. An "
     "SSH flow that carries only a handful of packets is usually a failed login, a banner check or a scan."),
    ("T7", "RFC 959 (FTP), general practice",
     "FTP control on TCP port 21 is a short exchange of text commands and replies; a login takes several "
     "round trips."),
    ("T8", "general security practice (slow-rate denial of service)",
     "A TCP connection held open for a very long time while carrying very few packets is how slow-rate "
     "denial of service ties up a server's connection slots."),
    # UDP
    ("D1", "RFC 768 (UDP)",
     "UDP is connectionless: there is no handshake and no connection state, so the absence of a handshake "
     "means nothing for UDP. Many UDP exchanges are one packet each way."),
    ("D2", "RFC 1035 (DNS), general practice",
     "DNS on UDP port 53: one query and one response, usually 1-2 packets each way, each under about 512 "
     "bytes for classic DNS, completing in milliseconds to tens of milliseconds."),
    ("D3", "RFC 5905 (NTP)",
     "NTP on UDP port 123: small fixed-size packets (48 bytes of NTP data), one request and one reply, "
     "repeated at intervals of minutes."),
    ("D4", "RFC 2131 (DHCP), RFC 6762 (mDNS), UPnP SSDP specification, NetBIOS (RFC 1002)",
     "Other ordinary UDP on a local network: DHCP (ports 67/68), multicast DNS (5353), SSDP (1900) and "
     "NetBIOS name and datagram services (137/138) send small packets, often with no reply."),
    ("D5", "general security practice",
     "Suspicious for UDP: large or many packets to a service that should answer briefly, very large "
     "responses to small queries (amplification), or sustained streams to unusual ports."),
    # protocol-independent
    ("P1", "IANA port registry, RFC 6335",
     "Ports 0-1023 host standard services, 1024-49151 are registered, and 49152-65535 are ephemeral "
     "(client-side) ports. Ordinary client traffic is addressed to a service port; a destination in the "
     "ephemeral range is unusual unless a known application uses it. 8000, 8080 and 8443 are common for "
     "proxies, development servers and administration consoles."),
    ("P2", "general security practice",
     "Repeated packets carrying no application data, especially to non-standard ports, are typical of "
     "scans, liveness checks and command-and-control polling. Benign keepalives exist too, so the port and "
     "the regularity matter."),
    ("P3", "general security practice",
     "Strong asymmetry in the unexpected direction is suspicious: a client pushing much more data to a "
     "server than it receives on a service that normally answers with more, or a large one-way transfer on "
     "an unusual port."),
    ("P4", "general security practice",
     "Many tiny packets at a high rate in a short flow is flooding. Many requests in rapid succession to "
     "one server from one client is request flooding."),
    ("P5", "general security practice",
     "Highly regular timing (near-identical inter-arrival times) means automation: legitimate for "
     "monitoring, but also how bots beacon to their controllers."),
    ("P6", "general security practice",
     "Behaviour that does not match the port is suspicious: web-like exchanges on a high port, or "
     "interactive-looking traffic on a web port."),
)

HANDSHAKE_RULE = (
    "A missing handshake is meaningful only for TCP, and these measurements do not record the TCP "
    "handshake at all. Never treat a flow as suspicious or as ordinary because of its protocol alone."
)

EXCULPATION_RULE = (
    "Resembling ordinary traffic is not by itself exculpatory. Capable attackers deliberately produce "
    "flows that look ordinary. A flow is benign only if you can say what, specifically, makes it benign, "
    "and nothing above that marks it as suspicious is present."
)

_SECTIONS = (
    ("How to read the measurements", "U"),
    ("TCP flows", "T"),
    ("UDP flows", "D"),
    ("Any protocol", "P"),
)

PRIORS_TEXT = (
    "What ordinary enterprise network traffic looks like (general knowledge, not measured on this "
    "network):\n"
    + "".join(
        f"\n{title}:\n" + "\n".join(f"[{sid}] {text}" for sid, _src, text in PRIOR_STATEMENTS if sid.startswith(prefix)) + "\n"
        for title, prefix in _SECTIONS
    )
    + "\n" + HANDSHAKE_RULE + "\n\n" + EXCULPATION_RULE + "\n"
)

PRIOR_IDS = tuple(sid for sid, _src, _text in PRIOR_STATEMENTS)
