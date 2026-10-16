# v3eval findings — post-run analysis (no API calls)

Written after the v3eval run, from saved files only. Nothing here changes the pre-registered
metric. Per-record J stays the headline; the per-behaviour figures sit beside it as the honest
interpretation.

## 1. Per-behaviour headline

**Method** (`eval/behaviour_groups.py`):

- Each of the 10 escalation features is min-max scaled over the dataset's own records.
- Distance is the Euclidean distance divided by √10, so 0 means identical and 1 the full range
  on every feature.
- Records of the same attack class form one behaviour when they are linked by a chain of pairs,
  each within **0.05** (single linkage).
- A behaviour counts as caught when a strict majority of its records were caught.

**Choosing the threshold.** The threshold was fixed before looking at verdicts: 0.05 is below
the 10th-percentile pair distance across all 84 records (0.061).

| threshold | attack behaviours (min-max scaling) | attack behaviours (rank scaling) |
|---|---|---|
| 0.01 | 21 | 30 (Hulk splits on timing) |
| **0.05** | **19** | **19** (Hulk 2, Slowhttptest 3) |
| 0.10 | 19 | 19 |

At 0.05 both scalings give the same groups.

| v3eval, Haiku 5.5, `holdout3_noquota` | per record (pre-registered) | per behaviour |
|---|---|---|
| Attacks caught | 20 / 42 | **7 / 19** |
| Benign flagged | 13 / 42 | 13 / 42 (unchanged) |
| **J** | **+0.167** | **+0.059** |

**Behaviour groups** (records per group):

| class | groups |
|---|---|
| DoS Hulk | [14, 1] |
| DoS Slowhttptest | [6, 1, 1] |
| DoS slowloris | [3, 1] |
| Bot | [2, 1, 1] |
| SSH-Patator | [2, 2] |
| PortScan | [1, 1] |
| FTP-Patator | [1, 1] |
| DDoS | [1] |
| Infiltration | [1] |
| XSS | [1] |

Two groups are mixed:
- The Hulk group of 14 had 13 caught, so it counts as caught.
- One SSH-Patator group of 2 had 1 caught, which is not a majority, so it counts as not caught.

**What it means:** most of the per-record J comes from one Hulk flow pattern counted 15
times. By behaviour, the pipeline catches 7 of 19 distinct attack behaviours.

### Benign flows are not duplicated (checked)

- All 42 benign records have distinct feature vectors.
- Under min-max scaling at 0.05 they *appear* to collapse into 18 groups. That is a scaling
  artifact: extreme attack and benign values stretch each feature's range, which squashes
  ordinary flows together.
- The largest benign "group" (12 records) spans 9 different source/destination/port
  combinations and 6 capture days.
- Under rank scaling at 0.05 the 42 benign flows stay 35 groups.

Benign records are separate connections, unlike the 15 Hulk records (one host, one victim,
one port, one day, 7 of 10 features constant). Benign is therefore kept per record.

## 2. The same check on eval20

eval20 is mostly *not* duplicated:
- The 3 DoS Hulk records are 3 distinct behaviours.
- 2 of the 3 PortScan records collapse into one, so eval20 has **9 attack behaviours from 10
  records**.
- All 10 benign records are distinct.

| eval20 | caught | J per record | behaviours caught | J per behaviour |
|---|---|---|---|---|
| Haiku 5.5 | 3 / 10 | +0.300 | 2 / 9 | +0.222 |
| DeepSeek V4.1 Flash | 0 / 10 | +0.000 | 0 / 9 | +0.000 |
| GLM 5.2 | 2 / 10 | +0.200 | 1 / 9 | +0.111 |
| Gemini 3.5 Flash-Lite | 3 / 10 | −0.100 | 2 / 9 | −0.178 |
| Kimi K2.6 (endpoint did not honour the schema) | 0 / 10 | −1.000 | 0 / 9 | −1.000 |
| Haiku, A3 hold-out | 4 / 10 | +0.300 | 3 / 9 | +0.233 |
| Haiku, A3+A4 hold-out | 4 / 10 | +0.300 | 3 / 9 | +0.233 |
| Haiku, no quota | 3 / 10 | +0.200 | 3 / 9 | +0.233 |

The caveat applies to eval20, but mildly. Per-behaviour J is lower by 0 to 0.09, and the
four-model ranking and the winner (Haiku 5.5) are unchanged.

## 3. Nondeterminism within a run

Missed Hulk record `e0c1fbf4e41b7931aff44e2934b808ec` (verdict `anomalous_but_explicable`)
is **identical on all 10 escalation features** to three Hulk records that were caught
(`anomalous_and_unexplained`):

- `36ab60a27fdba90c8eb642827d4d7c77`
- `0834753c0518eceab46251f1fbe2960a`
- `3ff9152349b688845c6e3dd87040b2fd`

It also matches 11 other Hulk records on every feature except the two timing fields.

Same model, same prompts, same run, same escalation features, and a different verdict. The
difference comes from sampled agent output: here, a benign hypothesis that survived and
refuted the conditions.

This is direct evidence of within-run nondeterminism. It sits alongside the across-run flip of
eval20's `8d754291d1`.

## 4. Limitations

**One attacker machine, not a population of attackers.**

- 37 of the 42 v3eval attacks come from one source, `172.16.0.1`, and target one victim,
  `192.168.10.50`. That covers all DoS (Hulk, Slowhttptest, slowloris), DDoS, SSH-Patator,
  FTP-Patator, PortScan and XSS records.
- The other 5 are Bot (three infected hosts reaching one C2 server, `205.174.165.73:8080`) and
  Infiltration (one host).
- The evaluation therefore measures how the pipeline handles one attacker's traffic, mostly
  against one web server. It does not show how the pipeline would handle other attackers,
  tools or targets.

**Breadth, not power.** As pre-registered, per-class results are coverage only.
