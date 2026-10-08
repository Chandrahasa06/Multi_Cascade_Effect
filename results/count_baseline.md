# Count-only baseline vs the agent pipeline, on the same 45 records

No API calls were made. The agent side is re-scored from recorded outputs in `results/agent_run_45_v2.jsonl`, with the corrected cap (0.29) re-applied to A5's recorded `benign_plausibility`. The baseline reads features only and uses the same `agents/escalation_grounding.py` neighbourhood path the pipeline uses.

**What the comparison is.** The agents receive the same grounding code, so this compares 'count plus agents' against 'count alone', not two different detectors. The agents' ungrounded verdict is also a function of the count: the cap forces `benign_plausibility` to at most the cap whenever the neighbourhood is ungrounded. So on ungrounded flows the agents can only agree with the count. The agents can differ only on grounded flows, where the cap does not apply.

n = 45 (30 attacks, 15 benign). The cutoff is the pipeline's own, 30. It is not tuned.

## 1. Side by side

| | agents (corrected cap 0.29) | agents (recorded, cap 0.30) | count-only baseline |
|---|---|---|---|
| attacks caught of 30 | 15/30 = 50.0% (95% CI 33.2-66.8%) | 9/30 = 30.0% (95% CI 16.7-47.9%) | 15/30 = 50.0% (95% CI 33.2-66.8%) |
| benign cleared of 15 | 15/15 = 100.0% (95% CI 79.6-100.0%) | 15/15 = 100.0% (95% CI 79.6-100.0%) | 15/15 = 100.0% (95% CI 79.6-100.0%) |
| benign wrongly flagged | 0 of 15 | 0 of 15 | 0 of 15 |
| records where agents and baseline disagree | 0 of 45 | | |

Confusion matrices (counts):

| | caught (predicted attack) | not caught (predicted benign) |
|---|---|---|
| agents, corrected cap 0.29: attack | TP 15 | FN 15 |
| agents, corrected cap 0.29: benign | FP 0 | TN 15 |
| count-only baseline: attack | TP 15 | FN 15 |
| count-only baseline: benign | FP 0 | TN 15 |

Reference figures from the earlier reports: 13/30 caught before the fixes, 9/30 after the fixes (both at the 0.30 cap, before the cap change).

## 2. Per-class recall (count n)

| true label | n | agents (corrected) caught | baseline caught |
|---|---|---|---|
| Bot | 5 | 1 of 5 | 1 of 5 |
| DDoS | 6 | 3 of 6 | 3 of 6 |
| DoS GoldenEye | 3 | 1 of 3 | 1 of 3 |
| DoS Hulk | 6 | 5 of 6 | 5 of 6 |
| DoS Slowhttptest | 4 | 2 of 4 | 2 of 4 |
| FTP-Patator | 1 | 1 of 1 | 1 of 1 |
| PortScan | 3 | 2 of 3 | 2 of 3 |
| SSH-Patator | 2 | 0 of 2 | 0 of 2 |
| BENIGN | 15 | 0 of 15 | 0 of 15 |

## 3. Disagreements

The agents and the baseline agree on every one of the 45 records. No disagreement record exists to inspect, so the question of whether the agents add judgment is not answered by this sample.


## 4. Cutoff sweep (baseline)

Baseline flags a flow as attack when its count is below the cutoff. Recall and specificity are reported for each cutoff. The cutoff is not tuned to the agents; this table is the whole sweep.

| cutoff | attacks caught | benign cleared |
|---|---|---|
| 0 | 0/30 = 0.0% (95% CI 0.0-11.4%) | 15/15 = 100.0% (95% CI 79.6-100.0%) |
| 1 | 8/30 = 26.7% (95% CI 14.2-44.4%) | 15/15 = 100.0% (95% CI 79.6-100.0%) |
| 5 | 12/30 = 40.0% (95% CI 24.6-57.7%) | 15/15 = 100.0% (95% CI 79.6-100.0%) |
| 10 | 13/30 = 43.3% (95% CI 27.4-60.8%) | 15/15 = 100.0% (95% CI 79.6-100.0%) |
| 30 (pipeline cutoff) | 15/30 = 50.0% (95% CI 33.2-66.8%) | 15/15 = 100.0% (95% CI 79.6-100.0%) |
| 100 | 19/30 = 63.3% (95% CI 45.5-78.1%) | 15/15 = 100.0% (95% CI 79.6-100.0%) |
| 500 | 21/30 = 70.0% (95% CI 52.1-83.3%) | 15/15 = 100.0% (95% CI 79.6-100.0%) |

Compare to the agents (corrected): 15 of 30 attacks caught, 0 of 15 benign wrongly flagged.

**Cutoffs that beat the agents outright on this sample.** Any cutoff at or above 100 catches more attacks than the agents, and none flags a benign record. The sweep was run after the 45-record results were known, so it is post-hoc. The pipeline cutoff stays 30. Nothing here is adopted.

### Disagreements at cutoff 0: 15 of 45 records

- 1b21af7fb2 (PortScan): count 20; A5 corrected 0.25; agents attack, baseline benign. A5: The evaluated flow is an extremely short, two-packet interaction consisting of a forward packet and a backward response of 6 bytes with a PSH flag set, completing in 49 microseconds. Hypothesis a3_h1 suggests a lightweight health check or reachability probe. While its predicted feature profile matches 236,954 benign flows in terms of duration, the empirical support data shows 0 flows within 2x of 
- 25323bd708 (PortScan): count 2; A5 corrected 0.25; agents attack, baseline benign. A5: The credited hypothesis a4_h1 has high broad profile match count (239169 benign flows), but zero flows are within 2x of this specific flow's observed values across multiple features such as packet rates and window sizes. The independent evidence confirms the duration, packet counts, and destination port. However, because zero real benign flows match the exact feature profile within 2x, the empiric
- 2948aa4982 (DoS Hulk): count 0; A5 corrected 0.29; agents attack, baseline benign. A5: I am crediting hypothesis a3_h1, which posits an HTTP connection fetching a small static file with a long idle keep-alive period before closing. The empirical support indicates that while 45,561 benign flows match the broad profile of trigger_flow_duration, exactly 0 flows fall within 2x of this specific flow's observed values across multiple metrics (such as Bwd Packets/s, Flow IAT Mean, etc.). C
- 3aedc4e3b0 (DDoS): count 0; A5 corrected 0.25; agents attack, baseline benign. A5: The evaluated benign hypotheses such as a4_h1, a3_h1, and a3_h2 describe standard HTTP request-response behavior over port 80 with asymmetric traffic volumes. Although they match broad criteria with high raw benign profile counts (e.g., 302,495 flows for a4_h1), empirical support indicates that zero matching benign flows are within 2x of this specific flow's observed values across multiple key met
- 405408a974 (FTP-Patator): count 0; A5 corrected 0.25; agents attack, baseline benign. A5: The evaluated flow represents a brief connection attempt directed toward port 21 with a duration of 204 microseconds, consisting of 2 forward packets and 1 backward packet. Hypothesis a3_h2 suggests an automated monitoring system or health checker probing port 21. While the profile matches 109,033 benign flows generally, it has zero empirical support for flows within 2x of this specific flow's obs
- 421c39412b (DDoS): count 0; A5 corrected 0.29; agents attack, baseline benign. A5: I credited hypothesis a3_h1, which posits a brief HTTP request and response interaction. The empirical support shows 41,113 benign flows matching the predicted profile, but 0 within 2x of this flow's specific observed values for key features such as Bwd Packets/s, Flow IAT Mean, and Packet Length Mean. Consequently, per the grounding rules, the benign plausibility is capped at 0.3 due to zero clos
- 49bcc3b140 (DoS Slowhttptest): count 1; A5 corrected 0.29; agents attack, baseline benign. A5: The evaluated flow exhibits characteristics of a long-duration, low-throughput HTTP connection with sparse packet transmission, consistent with hypothesis a3_h1. Although a3_h1 matches 5,753 benign flows on its broader profile, it has 0 flows within 2x of this specific flow's observed values for features such as Bwd Packets/s and Flow IAT Mean. Because of this zero close-match empirical support, t
- 56d0f28196 (DoS Slowhttptest): count 1; A5 corrected 0.29; agents attack, baseline benign. A5: I credited hypothesis a3_h1, which posits an infrequent HTTP polling or monitoring script. It has empirical support of 25868 matching benign flows, but 0 close-to-observed flows within 2x of this specific flow's values across several features. Because zero benign flows matched closely to this exact feature profile, the mechanical empirical plausibility cap applies, limiting the benign plausibility
- 681c501fd3 (DoS Hulk): count 9; A5 corrected 0.29; agents attack, baseline benign. A5: We examine hypothesis a3_h1, which describes a standard web client performing a quick HTTP GET request on port 80. While the broad profile matches 78,208 benign flows in terms of duration and general traffic volume, the empirical support reveals that exactly zero benign flows are within 2x of this specific flow's observed values across multiple features such as bwd packets/s, flow IAT mean, packet
- 6846713a33 (DoS Hulk): count 0; A5 corrected 0.25; agents attack, baseline benign. A5: The credited hypothesis a3_h1 describes an HTTP long-polling or keep-alive connection over port 80 with extended duration and asymmetrical packet payloads. Although the broad profile matches tens of thousands of benign flows globally, the empirical support query shows zero benign flows within 2x of this specific flow's observed values across multiple features such as backward packet rate, flow IAT
- 7437f3e358 (DDoS): count 0; A5 corrected 0.25; agents attack, baseline benign. A5: The credited hypothesis a3_h1 suggests an ordinary client requesting a small resource or sending a minimal command over port 80, receiving a substantial file in return. This hypothesis has 6312 matching benign flows in the dataset, but 0 matching flows within 2x of this specific flow's observed values across multiple features such as Bwd Packets/s, Flow IAT Mean, and Fwd Packet Length Max. Consequ
- a2123a0401 (DoS Hulk): count 4; A5 corrected 0.25; agents attack, baseline benign. A5: The credited hypothesis a4_h1 suggests an infrequent HTTP GET request with a small payload and a moderately sized response over a persistent connection with long idle gaps. While 45561 benign flows match its broad predicted profile, zero flows match within 2x of this specific flow's observed values across multiple features such as Bwd Packets/s, Flow IAT Mean, and Packet Length Mean. Furthermore, 
- aa23704f6b (DoS Hulk): count 0; A5 corrected 0.25; agents attack, baseline benign. A5: The evaluated flow exhibits characteristics of a slow HTTP request-response transaction over port 80, marked by prolonged duration, an asymmetrical packet size distribution with larger backward responses, and a normal termination indicated by a FIN flag. Both the chain and independent reviews consistently describe these features through claims such as a1_c3, a4_c7, a2_c1, and a4_c4. Hypothesis a4_
- dd7d54642f (Bot): count 0; A5 corrected 0.25; agents attack, baseline benign. A5: The credited hypothesis a4_h2 suggests an API request transmitting a structured payload with a minimal acknowledgment response. While it has 284,350 matching benign flows globally, it has zero flows within 2x of this flow's specific observed values on key fine-grained metrics like Bwd Packets/s, Flow IAT Mean, and Packet Length Mean. The mandatory claim to address is a2_c1, which asserts that the 
- fb0ed3f72f (DoS GoldenEye): count 17; A5 corrected 0.29; agents attack, baseline benign. A5: We examine hypothesis a3_h1, which posits a standard HTTP client request over port 80 receiving a larger server response. This hypothesis possesses strong contextual and qualitative backing from the chain and independent review, with claims such as a1_c1, a3_c1, a4_c4, and a4_c5 all corroborating the asymmetric byte ratio and destination port 80. However, the empirical support query shows that whi

### Disagreements at cutoff 1: 7 of 45 records

- 1b21af7fb2 (PortScan): count 20; A5 corrected 0.25; agents attack, baseline benign. A5: The evaluated flow is an extremely short, two-packet interaction consisting of a forward packet and a backward response of 6 bytes with a PSH flag set, completing in 49 microseconds. Hypothesis a3_h1 suggests a lightweight health check or reachability probe. While its predicted feature profile matches 236,954 benign flows in terms of duration, the empirical support data shows 0 flows within 2x of 
- 25323bd708 (PortScan): count 2; A5 corrected 0.25; agents attack, baseline benign. A5: The credited hypothesis a4_h1 has high broad profile match count (239169 benign flows), but zero flows are within 2x of this specific flow's observed values across multiple features such as packet rates and window sizes. The independent evidence confirms the duration, packet counts, and destination port. However, because zero real benign flows match the exact feature profile within 2x, the empiric
- 49bcc3b140 (DoS Slowhttptest): count 1; A5 corrected 0.29; agents attack, baseline benign. A5: The evaluated flow exhibits characteristics of a long-duration, low-throughput HTTP connection with sparse packet transmission, consistent with hypothesis a3_h1. Although a3_h1 matches 5,753 benign flows on its broader profile, it has 0 flows within 2x of this specific flow's observed values for features such as Bwd Packets/s and Flow IAT Mean. Because of this zero close-match empirical support, t
- 56d0f28196 (DoS Slowhttptest): count 1; A5 corrected 0.29; agents attack, baseline benign. A5: I credited hypothesis a3_h1, which posits an infrequent HTTP polling or monitoring script. It has empirical support of 25868 matching benign flows, but 0 close-to-observed flows within 2x of this specific flow's values across several features. Because zero benign flows matched closely to this exact feature profile, the mechanical empirical plausibility cap applies, limiting the benign plausibility
- 681c501fd3 (DoS Hulk): count 9; A5 corrected 0.29; agents attack, baseline benign. A5: We examine hypothesis a3_h1, which describes a standard web client performing a quick HTTP GET request on port 80. While the broad profile matches 78,208 benign flows in terms of duration and general traffic volume, the empirical support reveals that exactly zero benign flows are within 2x of this specific flow's observed values across multiple features such as bwd packets/s, flow IAT mean, packet
- a2123a0401 (DoS Hulk): count 4; A5 corrected 0.25; agents attack, baseline benign. A5: The credited hypothesis a4_h1 suggests an infrequent HTTP GET request with a small payload and a moderately sized response over a persistent connection with long idle gaps. While 45561 benign flows match its broad predicted profile, zero flows match within 2x of this specific flow's observed values across multiple features such as Bwd Packets/s, Flow IAT Mean, and Packet Length Mean. Furthermore, 
- fb0ed3f72f (DoS GoldenEye): count 17; A5 corrected 0.29; agents attack, baseline benign. A5: We examine hypothesis a3_h1, which posits a standard HTTP client request over port 80 receiving a larger server response. This hypothesis possesses strong contextual and qualitative backing from the chain and independent review, with claims such as a1_c1, a3_c1, a4_c4, and a4_c5 all corroborating the asymmetric byte ratio and destination port 80. However, the empirical support query shows that whi

### Disagreements at cutoff 5: 3 of 45 records

- 1b21af7fb2 (PortScan): count 20; A5 corrected 0.25; agents attack, baseline benign. A5: The evaluated flow is an extremely short, two-packet interaction consisting of a forward packet and a backward response of 6 bytes with a PSH flag set, completing in 49 microseconds. Hypothesis a3_h1 suggests a lightweight health check or reachability probe. While its predicted feature profile matches 236,954 benign flows in terms of duration, the empirical support data shows 0 flows within 2x of 
- 681c501fd3 (DoS Hulk): count 9; A5 corrected 0.29; agents attack, baseline benign. A5: We examine hypothesis a3_h1, which describes a standard web client performing a quick HTTP GET request on port 80. While the broad profile matches 78,208 benign flows in terms of duration and general traffic volume, the empirical support reveals that exactly zero benign flows are within 2x of this specific flow's observed values across multiple features such as bwd packets/s, flow IAT mean, packet
- fb0ed3f72f (DoS GoldenEye): count 17; A5 corrected 0.29; agents attack, baseline benign. A5: We examine hypothesis a3_h1, which posits a standard HTTP client request over port 80 receiving a larger server response. This hypothesis possesses strong contextual and qualitative backing from the chain and independent review, with claims such as a1_c1, a3_c1, a4_c4, and a4_c5 all corroborating the asymmetric byte ratio and destination port 80. However, the empirical support query shows that whi

### Disagreements at cutoff 10: 2 of 45 records

- 1b21af7fb2 (PortScan): count 20; A5 corrected 0.25; agents attack, baseline benign. A5: The evaluated flow is an extremely short, two-packet interaction consisting of a forward packet and a backward response of 6 bytes with a PSH flag set, completing in 49 microseconds. Hypothesis a3_h1 suggests a lightweight health check or reachability probe. While its predicted feature profile matches 236,954 benign flows in terms of duration, the empirical support data shows 0 flows within 2x of 
- fb0ed3f72f (DoS GoldenEye): count 17; A5 corrected 0.29; agents attack, baseline benign. A5: We examine hypothesis a3_h1, which posits a standard HTTP client request over port 80 receiving a larger server response. This hypothesis possesses strong contextual and qualitative backing from the chain and independent review, with claims such as a1_c1, a3_c1, a4_c4, and a4_c5 all corroborating the asymmetric byte ratio and destination port 80. However, the empirical support query shows that whi

### Disagreements at cutoff 100: 4 of 45 records

- 2c27e52883 (DDoS): count 88; A5 corrected 0.85; baseline attack, agents benign. A5: The credited hypothesis a4_h2 (a monitoring or diagnostic probe sending periodic heartbeat test packets where replies are dropped) is strongly supported by empirical data, showing 54845 matching benign flows in the network, with 88 matching closely on key features. Its prediction of a small number of forward packets distributed over a long duration with zero backward traffic is fully corroborated 
- 70448e2e53 (DDoS): count 62; A5 corrected 0.85; baseline attack, agents benign. A5: I am crediting hypothesis a4_h1, which describes a client sending small keep-alive or beacon packets to a web server without expecting or receiving any response data back. This hypothesis has strong empirical support with 284350 matching benign flows, of which 62 are within 2x of this flow's own observed values. Its prediction of a short-to-moderate flow duration with extremely low packet count an
- d07e7212eb (DDoS): count 76; A5 corrected 0.85; baseline attack, agents benign. A5: The credited hypothesis a3_h1 proposes an application layer keepalive or heartbeat ping sent periodically over a unidirectional channel. It is strongly backed by empirical support, matching 821 benign flows on this network, with 76 falling within 2x of this flow's specific observed values. The independent review's claims (such as a4_c1, a4_c2, a4_c3, and a4_c4) fully confirm the predicted feature 
- d67a88f41a (SSH-Patator): count 30; A5 corrected 0.85; baseline attack, agents benign. A5: The credited hypothesis a3_h3 represents an automated configuration management tool or script connecting over SSH. It has strong empirical support on this network with 530,306 matching benign flows, of which 30 fall within 2x of this flow's observed values across key features. Its prediction of short duration and low packet counts is fully confirmed by independent review claims such as a4_c1, a4_c

### Disagreements at cutoff 500: 6 of 45 records

- 2c27e52883 (DDoS): count 88; A5 corrected 0.85; baseline attack, agents benign. A5: The credited hypothesis a4_h2 (a monitoring or diagnostic probe sending periodic heartbeat test packets where replies are dropped) is strongly supported by empirical data, showing 54845 matching benign flows in the network, with 88 matching closely on key features. Its prediction of a small number of forward packets distributed over a long duration with zero backward traffic is fully corroborated 
- 5b39770724 (DoS GoldenEye): count 114; A5 corrected 0.95; baseline attack, agents benign. A5: The credited hypothesis a3_h1 proposes a standard HTTP client request to port 80 with a moderate-sized response payload. It has robust empirical support with 48517 matching benign flows and 114 flows closely matching the observed values. Its prediction of typical web traffic characteristics with small forward requests and larger backward responses is fully confirmed by the independent review claim
- 70448e2e53 (DDoS): count 62; A5 corrected 0.85; baseline attack, agents benign. A5: I am crediting hypothesis a4_h1, which describes a client sending small keep-alive or beacon packets to a web server without expecting or receiving any response data back. This hypothesis has strong empirical support with 284350 matching benign flows, of which 62 are within 2x of this flow's own observed values. Its prediction of a short-to-moderate flow duration with extremely low packet count an
- b32f2676f5 (PortScan): count 117; A5 corrected 0.85; baseline attack, agents benign. A5: I credited hypothesis a3_h1, which represents a scheduled or automated health-check probe that exchanges a single request-response message pair very quickly to verify service responsiveness. This hypothesis has strong empirical support with 111,176 matching benign flows, of which 117 are within 2x of this flow's observed values. Its prediction of extremely short flow duration and minimal packet co
- d07e7212eb (DDoS): count 76; A5 corrected 0.85; baseline attack, agents benign. A5: The credited hypothesis a3_h1 proposes an application layer keepalive or heartbeat ping sent periodically over a unidirectional channel. It is strongly backed by empirical support, matching 821 benign flows on this network, with 76 falling within 2x of this flow's specific observed values. The independent review's claims (such as a4_c1, a4_c2, a4_c3, and a4_c4) fully confirm the predicted feature 
- d67a88f41a (SSH-Patator): count 30; A5 corrected 0.85; baseline attack, agents benign. A5: The credited hypothesis a3_h3 represents an automated configuration management tool or script connecting over SSH. It has strong empirical support on this network with 530,306 matching benign flows, of which 30 fall within 2x of this flow's observed values across key features. Its prediction of short duration and low packet counts is fully confirmed by independent review claims such as a4_c1, a4_c

## 5. Cap change: what moved

Records whose benign_plausibility changes under the 0.29 cap: 6 of 45.
- 2948aa4982 (DoS Hulk): 0.30 -> 0.29, neighbourhood 0
- 421c39412b (DDoS): 0.30 -> 0.29, neighbourhood 0
- 49bcc3b140 (DoS Slowhttptest): 0.30 -> 0.29, neighbourhood 1
- 56d0f28196 (DoS Slowhttptest): 0.30 -> 0.29, neighbourhood 1
- 681c501fd3 (DoS Hulk): 0.30 -> 0.29, neighbourhood 9
- fb0ed3f72f (DoS GoldenEye): 0.30 -> 0.29, neighbourhood 17

The earlier pipeline had no clamp firing on these records. The 0.30 values were A5's own output, because the prompt tells A5 the cap is 0.3. A clamp only fires when a value is strictly above the cap, so a 0.30 output passed through unchanged. Under 0.29 it is clamped. Applying the corrected cap is therefore a real change to A5's recorded outputs.

## 6. Checks

- Baseline count equals the count the pipeline recorded: 45 of 45 records (mismatches: 0).
- Re-score at the old 0.30 caps reproduces the recorded benign_plausibility exactly: 45 of 45 (mismatches: 0). This shows the re-scoring path is faithful.

## 7. Conclusion

**At the pipeline's own cutoff (30), the baseline matches the agents.** On these 45 records the count-only rule gets the same verdict on every record. That is a negative result about the agents at this cutoff: the pipeline reproduces what the neighbourhood count already says. The design gives the agents nothing to add on ungrounded flows, and no grounded attack was caught, so the sample cannot show they add judgment on grounded ones. **But the sweep shows the baseline beats the agents at cutoffs 100, 500** (more attacks caught, no benign flagged). That result is post-hoc, from the same 45 records, and is not adopted. A cutoff needs its own held-out check before anyone should trust it. It is a sample of 45, and it is not an independent test.

## Limits

- n = 45. One record moves a rate by 2.2 points.
- The agents' verdicts on ungrounded flows are fixed by the cap, so agreement on those is by construction, not evidence the agents match the count's judgment. The discriminating test is on grounded flows, and this sample has no grounded attack that the agents caught.
- The baseline is one rule on one feature space. A different baseline could differ.

