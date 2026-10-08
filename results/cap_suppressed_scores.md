# What A5 said before the cap overwrote it

No API calls. Read from the saved runs.

## The answer

| arm | records the cap caught | of which attacks | raw A5 score already below 0.30 |
|---|---|---|---|
| M (Monday) | 20 | 20 | **0 of 20 (95% CI 0.0-16.1%)** |
| MT (Monday+Tuesday) | 18 | 18 | **0 of 18 (95% CI 0.0-17.6%)** |
| A (all-days) | 15 | 15 | **0 of 15 (95% CI 0.0-20.4%)** |

**In every arm, A5 on its own would have caught none of the flows the cap caught.** Its raw scores on them were 0.70 to 0.95. Every attack the arms caught, they caught because the cap overwrote A5. Removing the cap with these prompts would drop recall in Arm M from 20 of 29 to 0 of 29.

Per the brief's decision rule, Part 2 (the uncapped run) was **not run**. See the caveat below before deciding whether to run it anyway.

## Raw-score distribution, by true label

Every record the cap caught in any arm was an attack, so there is no benign row.

- Arm M: attacks (n=20) [0.72, 0.82, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.88, 0.88, 0.88, 0.92, 0.95, 0.95]; benign (n=0) []
- Arm MT: attacks (n=18) [0.72, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.88, 0.88, 0.88, 0.9, 0.92, 0.92, 0.95]; benign (n=0) []
- Arm A: attacks (n=15) [0.7, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.88, 0.88, 0.88, 0.88, 0.92, 0.92]; benign (n=0) []

## Caveat: the raw scores were given with counts in view

The arm prompts (`a5_verdict_v7` with the reference-window sentences) showed A5 the real neighbourhood count and per-hypothesis profile counts. On every one of the 20 Arm M records the cap caught, the ten-feature neighbourhood count was 0 to 13, yet at least one hypothesis sentence said 6,531 or more benign flows were "within 2x" (closeness on only the one or two features that hypothesis named). A5 went with the large number. Its 0.70-0.95 is its reading of that mixed evidence, not its judgement without counts.

The earlier v2 and v3 runs show the opposite behaviour, and why:

- v2: 15 ungrounded records; cap fired on 0; A5 itself scored below 0.30 on 9 without any cap. Stored scores: {0.25: 9, 0.3: 6}.
- v3: 14 ungrounded records; cap fired on 6; A5 itself scored below 0.30 on 8 without any cap. Stored scores: {0.1: 1, 0.25: 7, 0.29: 6}.

In v2 and v3, A5 scored ungrounded flows low itself, without the cap. But in those runs `patch_hypothesis_support` replaced every hypothesis's closeness count with **0** for ungrounded flows, so every hypothesis line A5 read said "of which 0 are within 2x". A5 was following that suppressed zero. Once the arms showed the real numbers, the same kind of flow scored 0.70-0.95.

So in no run has A5 flagged a low-count flow from its own judgement. It either followed a count it was shown or was overwritten by the cap. Part 2 would test the one untested case: low-count flows with no count shown at all. The grounded-only run (eighth attempt, count >= 30 flows, no count shown) called 22 of 22 benign, which suggests the same here, but it is a different set of flows.

## Per record

### Arm M (Monday): 20 records the cap caught

| record | true label | count (leave-one-out) | raw A5 | clamped |
|---|---|---|---|---|
| 1b21af7fb2 | PortScan | 0 | 0.88 | 0.29 |
| 25323bd708 | PortScan | 0 | 0.85 | 0.29 |
| 2948aa4982 | DoS Hulk | 0 | 0.85 | 0.29 |
| 3aedc4e3b0 | DDoS | 0 | 0.92 | 0.29 |
| 405408a974 | FTP-Patator | 0 | 0.85 | 0.29 |
| 421c39412b | DDoS | 0 | 0.85 | 0.29 |
| 6846713a33 | DoS Hulk | 0 | 0.85 | 0.29 |
| 7437f3e358 | DDoS | 0 | 0.85 | 0.29 |
| a2123a0401 | DoS Hulk | 0 | 0.88 | 0.29 |
| aa23704f6b | DoS Hulk | 0 | 0.85 | 0.29 |
| b32f2676f5 | PortScan | 0 | 0.85 | 0.29 |
| dd7d54642f | Bot | 0 | 0.88 | 0.29 |
| 49bcc3b140 | DoS Slowhttptest | 1 | 0.85 | 0.29 |
| 56d0f28196 | DoS Slowhttptest | 1 | 0.95 | 0.29 |
| 681c501fd3 | DoS Hulk | 4 | 0.85 | 0.29 |
| fb0ed3f72f | DoS GoldenEye | 7 | 0.95 | 0.29 |
| 70448e2e53 | DDoS | 10 | 0.72 | 0.29 |
| d07e7212eb | DDoS | 10 | 0.82 | 0.29 |
| 2c27e52883 | DDoS | 11 | 0.85 | 0.29 |
| d67a88f41a | SSH-Patator | 13 | 0.85 | 0.29 |

### Arm MT (Monday+Tuesday): 18 records the cap caught

| record | true label | count (leave-one-out) | raw A5 | clamped |
|---|---|---|---|---|
| 1b21af7fb2 | PortScan | 0 | 0.90 | 0.29 |
| 25323bd708 | PortScan | 0 | 0.92 | 0.29 |
| 2948aa4982 | DoS Hulk | 0 | 0.95 | 0.29 |
| 3aedc4e3b0 | DDoS | 0 | 0.85 | 0.29 |
| 421c39412b | DDoS | 0 | 0.85 | 0.29 |
| 6846713a33 | DoS Hulk | 0 | 0.85 | 0.29 |
| 7437f3e358 | DDoS | 0 | 0.72 | 0.29 |
| aa23704f6b | DoS Hulk | 0 | 0.85 | 0.29 |
| b32f2676f5 | PortScan | 0 | 0.85 | 0.29 |
| dd7d54642f | Bot | 0 | 0.85 | 0.29 |
| 49bcc3b140 | DoS Slowhttptest | 1 | 0.85 | 0.29 |
| 56d0f28196 | DoS Slowhttptest | 1 | 0.85 | 0.29 |
| a2123a0401 | DoS Hulk | 4 | 0.88 | 0.29 |
| 681c501fd3 | DoS Hulk | 5 | 0.92 | 0.29 |
| 70448e2e53 | DDoS | 10 | 0.85 | 0.29 |
| d07e7212eb | DDoS | 10 | 0.85 | 0.29 |
| 2c27e52883 | DDoS | 11 | 0.88 | 0.29 |
| fb0ed3f72f | DoS GoldenEye | 11 | 0.88 | 0.29 |

### Arm A (all-days): 15 records the cap caught

| record | true label | count (leave-one-out) | raw A5 | clamped |
|---|---|---|---|---|
| 2948aa4982 | DoS Hulk | 0 | 0.92 | 0.29 |
| 3aedc4e3b0 | DDoS | 0 | 0.88 | 0.29 |
| 405408a974 | FTP-Patator | 0 | 0.85 | 0.29 |
| 421c39412b | DDoS | 0 | 0.85 | 0.29 |
| 6846713a33 | DoS Hulk | 0 | 0.85 | 0.29 |
| 7437f3e358 | DDoS | 0 | 0.70 | 0.29 |
| aa23704f6b | DoS Hulk | 0 | 0.85 | 0.29 |
| dd7d54642f | Bot | 0 | 0.85 | 0.29 |
| 49bcc3b140 | DoS Slowhttptest | 1 | 0.85 | 0.29 |
| 56d0f28196 | DoS Slowhttptest | 1 | 0.88 | 0.29 |
| 25323bd708 | PortScan | 2 | 0.92 | 0.29 |
| a2123a0401 | DoS Hulk | 4 | 0.88 | 0.29 |
| 681c501fd3 | DoS Hulk | 9 | 0.85 | 0.29 |
| fb0ed3f72f | DoS GoldenEye | 17 | 0.88 | 0.29 |
| 1b21af7fb2 | PortScan | 20 | 0.85 | 0.29 |

### A5's rationale on each Arm M record the cap caught

- **1b21af7fb2** (PortScan, count 0, raw 0.88): The credited hypothesis a3_h1 represents a lightweight health check or reachability probe polling a service to verify port availability and responsiveness. It shows strong empirical support with 201,845 matching benign flows in the reference window, and 51,633 of those falling within 2x of this flow's own values on trigger_flow_duration. Its prediction of an extremely short, rapid minimal packet exchange is fully confirmed by the independent review claims such as a4_c1, a4_c2, and a4_c3, as well as chain claims a1_c1, a2_c1, and a2_c3. There are no contradicting claims flagged against this hypothesis, and mechanical agreement between the chain and independent reviews is extremely high.

- **25323bd708** (PortScan, count 0, raw 0.85): The credited hypothesis a3_h1 proposes a database client or monitoring script executing an immediate poll or lightweight probe against the MySQL port. This hypothesis has robust empirical support with 95,510 matching benign flows in the reference window, of which 80,570 fall within 2x of this flow's values on trigger_flow_duration and trigger_flow_pkts_per_sec. Its predictions regarding extremely short flow duration, high packets per second, and minimal payload bytes are fully confirmed by the independent review claims (such as a4_c1, a4_c2, a4_c3, a4_c6) and chain claims. There are no contradicting claims flagged for this hypothesis, supporting a high benign plausibility score.

- **2948aa4982** (DoS Hulk, count 0, raw 0.85): The credited hypothesis a3_h1 proposes an HTTP connection fetching a small static file with a long idle keep-alive period before closing. This hypothesis has strong empirical support: 71,292 benign flows match its predicted profile in the reference window, with 53,203 of those falling within 2x of this flow's specific trigger_flow_duration value (8.62141e+07). Its prediction of an extended duration with a low packet rate is well-confirmed by independent review claims such as a4_c4, which notes the extended flow duration and long idle periods, as well as a4_c1 (destination port 80) and a4_c5 (normal FIN teardown). There are no contradicting claims for this hypothesis. Therefore, the benign plausibility is high, reflecting strong empirical backing and close alignment with independent evidence.

- **3aedc4e3b0** (DDoS, count 0, raw 0.92): The credited hypothesis a4_h1 suggests a standard HTTP request-response transaction over port 80 where a small client request is met with a larger server response. This is strongly supported by the empirical data, matching 279,248 benign flows in the reference window, with 42,349 flows falling within 2x of this flow's values on the predicted feature. The independent review and chain review fully corroborate the destination port 80 (a1_c1, a4_c5), the flow duration of approximately 8 seconds (a4_c1), and the asymmetric data transfer where backward payload exceeds forward payload (a4_c4). Furthermore, max packet length reaches 5840 bytes (a4_c6), consistent with normal web traffic payloads. No contradicting claims were flagged against a4_h1, and the high empirical support confirms this is a common, ordinary pattern on this network.

- **405408a974** (FTP-Patator, count 0, raw 0.85): The credited hypothesis a3_h2 (An automated monitoring system or health checker rapidly probing port 21 to verify if the FTP service is responding) is strongly supported by empirical evidence, matching 164,036 benign flows in the reference window, with 94,674 of those within 2x of this flow's duration. The independent review fully corroborates the basic flow characteristics through claims a4_c1 and a4_c4, confirming short duration and destination port 21. Furthermore, chain claims a1_c1, a2_c2, and a2_c4 support the targeted port, one-sided payload, and isolated transient burst nature of the flow. There are no contradicting claims flagged against this hypothesis, leading to a high benign plausibility score.

- **421c39412b** (DDoS, count 0, raw 0.85): The network flow exhibits standard characteristics of a brief HTTP request-response transaction over port 80, showing small forward packet sizes and significantly larger backward payload returns. We credit hypothesis a4_h1, which posits a standard web client sending a small HTTP GET request and receiving a moderately sized response payload. Its predicted feature profile for trigger_flow_duration matches 79,730 benign flows in the reference window, with 13,802 of those falling within 2x of this flow's own duration of 1,177,050 microseconds. The independent review confirms the asymmetric packet lengths and short duration, as captured in a4_c1, a4_c2, and a4_c3, aligning well with chain observations a1_c1 through a1_c4 and a2_c1 through a2_c4. There are no contradicting claims flagged by any generating reviewer, and the strong empirical support combined with consistent independent measurements reinforces a high benign plausibility.

- **6846713a33** (DoS Hulk, count 0, raw 0.85): The credited hypothesis a3_h1 has strong empirical support with 66,126 matching benign flows in the reference window and 9,974 within 2x of this flow's values. Its prediction of an exceptionally high flow duration with very low packet rates and sparse inter-arrival times is fully consistent with independent observations confirming the ~98.38 second flow duration (a4_c1), standard HTTP port 80 traffic (a4_c6), and asymmetric packet/byte distribution (a4_c4, a4_c5). There are no contradicting claims flagged against this hypothesis, and the empirical data strongly confirms the plausibility of this benign long-polling or keep-alive session.

- **7437f3e358** (DDoS, count 0, raw 0.85): The credited hypothesis a4_h1 describes a standard web request over port 80 where a small request payload results in a large response file. Although its direct empirical support count cannot be queried here via the feature profile template, its core claims are strongly corroborated by independent review claims a4_c4, a4_c5, a4_c6, and a4_c7, showing small forward packet length mean (8.67 bytes) and large backward packet length mean (2321.4 bytes) over destination port 80. Furthermore, a3_h3 provides additional backing with 21,819 matching benign flows and 762 within 2x of observed values. Regarding the required claim to address (a2_c1, which noted that the backward direction carries vastly more data than the forward direction), this does not undermine the benign hypothesis at all; rather, it directly reinforces the asymmetric client request/server response pattern characteristic of normal web traffic.

- **a2123a0401** (DoS Hulk, count 0, raw 0.88): The credited hypothesis a3_h1 represents an automated monitoring or health-checking tool polling a web service over HTTP with occasional small requests and responses. This hypothesis is supported by strong empirical evidence: 45,241 matching benign flows in the reference window, with 6,026 falling within 2x of this flow's values on the predicted features (trigger_flow_pkts_per_sec and trigger_flow_duration). Furthermore, independent review claims such as a4_c1, a4_c5, and a4_c6 confirm the long flow duration (~83.17 seconds), destination port 80 (HTTP), and long idle times. There are no contradicting claims flagged against this hypothesis, and all relevant independent observations corroborate the low-frequency, persistent HTTP polling behavior.

- **aa23704f6b** (DoS Hulk, count 0, raw 0.85): The credited hypothesis a3_h1 describes an HTTP client fetching a small request and receiving a moderately sized web resource over a slow or idle connection with long pauses. This explanation has strong empirical support with 35,730 matching benign flows in the reference window, out of which 15,768 fall within 2x of this flow's values on duration and byte rate. The hypothesis's prediction of extended flow duration with low packet and byte rates over standard HTTP destination port 80 is fully confirmed by the independent review claims (a4_c1, a4_c2, a4_c4, a4_c7). There are no contradicting claims that undermine this explanation, making it highly plausible as standard benign web traffic.

- **b32f2676f5** (PortScan, count 0, raw 0.85): The credited hypothesis a3_h1 represents a scheduled or automated health-check probe that exchanges a single request-response message pair. It possesses strong empirical support with 197,106 matching benign flows in the reference window and 4,897 flows within 2x of this flow's values on duration and packet rate. Its prediction of extremely short duration and minimal packet counts is fully confirmed by independent review claims such as a4_c1 and a4_c2, which verify the 10 microsecond duration and 800,000 bytes per second rate. There are no contradicting claims flagged for this hypothesis, supporting a high benign plausibility score.

- **dd7d54642f** (Bot, count 0, raw 0.88): The credited hypothesis a3_h1 proposes a short-lived client request or file upload transmitting a moderate batch of data to a service running on port 8080, which is strongly supported by the empirical data showing 2617 matching benign flows in the reference window and 388 flows within 2x of this flow's specific duration and byte rate values. Hypothesis a3_h1's prediction of a brief duration and high forward burst is confirmed by independent claims such as a4_c2 and a4_c3. Although a2_c1 was flagged as contradicting a2 hypotheses regarding asymmetric data distribution, it actually reinforces a3_h1 by highlighting the heavy forward data volume compared to backward acknowledgement, aligning completely with the high forward packet length mean and transfer profile.

- **49bcc3b140** (DoS Slowhttptest, count 1, raw 0.85): The credited hypothesis a3_h1 describes an HTTP client or keep-alive connection experiencing severe network delays or intermittent keep-alive probes over a long duration. It has strong empirical support with 11092 matching benign flows in the reference window and 1262 within 2x of this flow's own values on duration, packets per second, and bytes per second. Its prediction of an exceptionally long flow duration with low packets and bytes per second is fully corroborated by independent claims such as a4_c1, a4_c5, and a4_c6. There are no contradicting claims flagged for this hypothesis, supporting a high benign plausibility score.

- **56d0f28196** (DoS Slowhttptest, count 1, raw 0.95): The credited hypothesis a3_h1 proposes an HTTP client or monitoring script polling a web service or health endpoint very infrequently. This hypothesis has robust empirical support on this network, with 46,843 matching benign flows in the reference window and 7,769 of those falling within 2x of this flow's values on duration and packet rate. Its prediction of long flow duration with low packet rates and minimal bytes per second is fully confirmed by independent review claims such as a4_c1, a4_c2, and a4_c3, which corroborate the duration, packet counts, and low overall traffic characteristics. Furthermore, multi-signal benign evidence demonstrates close nearest neighbours sharing identical behavioural profiles. There are no contradicting claims listed for this hypothesis.

- **681c501fd3** (DoS Hulk, count 4, raw 0.85): The credited hypothesis a3_h1 proposes a standard web client performing a quick HTTP GET request for a small asset or file on port 80. This hypothesis has strong empirical support with 48,113 matching benign flows in the reference window, out of which 3,717 are within 2x of this flow's values on the predicted features (trigger_flow_duration and trigger_flow_bytes_per_sec). The independent review fully confirms the core aspects of this explanation, specifically through claims a4_c1 (short flow duration of 144602 microseconds) and a4_c6 (destination port 80 indicating standard HTTP traffic). There are no contradicting claims flagged against this hypothesis, and the mechanical agreement shows consistent findings between both review tracks regarding port and duration.

- **fb0ed3f72f** (DoS GoldenEye, count 7, raw 0.95): The credited hypothesis a3_h3 suggests an asymmetrical data transfer where a client triggers a bulk download of configuration updates or logs over port 80. Empirical support for this hypothesis is strong, with 13,978 matching benign flows in the reference window and 10,414 of those within 2x of this flow's specific values for trigger_bwd_pkt_len_mean. Furthermore, independent review claims a4_c4, a4_c5, and a4_c6 fully confirm the HTTP request-response structure characterized by a small forward request and a large backward payload (mean 2326.4 bytes). There are no contradicting claims or flags, leading to a high benign plausibility score.

- **70448e2e53** (DDoS, count 10, raw 0.72): The credited hypothesis is a4_h1, which posits a client sending small keep-alive or beacon packets to a web server without expecting or receiving any response data back. This hypothesis has strong empirical support, with 256,767 matching benign flows in the reference window and 36,490 of those falling within 2x of this flow's values on the predicted feature profile. Its prediction of a short to moderate flow duration with extremely low packet count and zero backward packets is fully confirmed by independent review claims a4_c1, a4_c2, and a4_c6. Regarding the contradicting claim a2_c2 (which notes that uniformly tiny 6-byte packets suggest probing rather than standard web data transfer), this does not undermine the benign client beacon/keep-alive hypothesis; repetitive tiny packets are entirely characteristic of certain application-layer keep-alives, heartbeat beacons, or minimalist diagnostic signals routinely observed in ordinary network traffic, as further corroborated by the high multi-signal benign neighborhood count.

- **d07e7212eb** (DDoS, count 10, raw 0.82): The credited hypothesis a4_h1 (monitoring or health-check system polling a service with a tiny probe packet) has robust empirical support, matching 227,038 total benign flows in the reference window with 14,076 falling within 2x of this flow's values on trigger_flow_duration. Furthermore, multi-signal benign evidence shows 10 benign flows closely resembling this traffic across all ten features. The independent review's claims (a4_c1, a4_c2, a4_c3) strongly confirm the unidirectional nature, low packet count (4 forward packets), and short duration (~2.6 seconds) of the connection, matching the chain's observations (a1_c1, a1_c2, a1_c4, a2_c2, a3_c1). There are no contradicting claims to address for this hypothesis.

- **2c27e52883** (DDoS, count 11, raw 0.85): The credited hypothesis a3_h1 ('A scheduled health-check or beacon client sending small, periodic keep-alive probes that receive no response') has strong empirical support, with 31084 matching benign flows in the reference window and 14759 of those within 2x of this flow's own values on trigger_flow_pkts_per_sec and trigger_flow_duration. Its prediction of slow, sparse forward packets with no backward response, low packet rate, and long idle periods is fully corroborated by independent review claims such as a4_c1, a4_c3, and a4_c6. There are no contradicting claims flagged against this hypothesis, and multiple independent observations confirm the unidirectional nature and extended duration of the flow.

- **d67a88f41a** (SSH-Patator, count 13, raw 0.85): The credited hypothesis a3_h3 proposes an automated configuration management tool or script connecting over SSH. It has strong empirical support with 466015 matching benign flows in the reference window, and 17058 of those within 2x of this flow's values on trigger_flow_duration. Its prediction of a short duration and low packet counts is fully supported by the independent review's claims regarding flow duration (a4_c1) and SSH destination port (a4_c4), as well as chain claims (a1_c1, a2_c3, a3_c1). There are no contradicting claims that need to be addressed.

