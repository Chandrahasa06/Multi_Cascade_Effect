# A Five-Agent LLM Pipeline Did Not Beat a Single Threshold on This Task

Written for: readers evaluating LLM agents for network-flow triage.

## Summary

A five-agent language-model pipeline with mechanical grounding and a structurally blind independent reviewer was built to judge whether an escalated network flow has a benign explanation the data supports.

On 44 scored flows it catches 14 of 29 attacks and clears 15 of 15 benign flows.

A single threshold on the benign-neighbourhood count, with no model calls, catches 14 of 29 and clears 15 of 15. The two disagree on zero records.

Seven attempts to make the agents add to the count were tested on the same records. None outperformed it. The failure mode is stable on this sample and is described below.

## Setup

**Data.** CICIDS2017, all capture days pooled, 2,830,743 flows. Escalated flows were drawn from a three-priority data-plane policy under a 1% controller budget.

**Records.** 45 escalated flows across three batches, sampled to be attack-weighted (30 attack, 15 benign) so that rare classes appear at all. Escalated flows are not 30% attacks in deployment. The evaluation therefore measures discrimination, not deployment calibration. *The 9.4% attack rate quoted in an earlier draft was not re-checked for this write-up.*

**Blind protocol.** True labels were withheld from every agent and held in a separate key file. The matched rule's class prediction, escalation priority and reason were stripped, since each correlates strongly with ground truth.

**Threshold.** A `benign_plausibility` below 0.30 is an attack verdict. The 0.30 threshold was set before these records were scored and was not changed. **The grounding cap that clamps `benign_plausibility` was changed from 0.30 to 0.29 during this work, after the 0.30 boundary effect was observed on six records.** That change is post-hoc and is disclosed here. It is discussed under Limitations.

**Pipeline.** A1 states quantitative deviations. A2 describes connection behaviour. A3 proposes 3–5 hypotheses, at least two benign, each with a machine-checkable predicted feature range. A4 performs the three tasks independently in a single pass; its interface accepts no chain output. A5 issues the verdict.

**Grounding.** Each hypothesis's predicted range is queried in code against 2.27M real benign flows before any agent reads the result. Agents see counts and percentiles, not the flows. A hypothesis matching zero benign flows is marked unsupported, and the score is capped regardless of what A5 reports.

## The baseline

For each flow, compute the benign-neighbourhood count from its own features using the grounding path the agents receive. Flag the flow if the count is below 30.

This is not a different detector. It is the same evidence the agents receive, without the reasoning layer.

| | attacks caught (of 29) | benign cleared (of 15) |
|---|---|---|
| agent pipeline (v3, cap 0.29) | 14 | 15 |
| count-only threshold | 14 | 15 |
| logistic regression on the same signals | 8 | 15 |

Zero disagreements between the agents and the count across 44 records.

**At higher cutoffs the count rule pulls ahead.** On the 44 scored records, a cutoff of 100 catches 18 of 29 attacks and a cutoff of 500 catches 20 of 29, with no benign flow flagged in either case. An earlier draft quoted 19 of 30 and 21 of 30; those figures included the unscored PortScan record. The sweep was performed after the results were known. It is post-hoc and is not adopted.

## Seven attempts to make the agents add to the count

Each targeted a different hypothesis about why the agents underperform.

**1. Repair the grounding path.** An early blind run scored 0 of 11, with every record at 0.75–0.99, because `close_to_observed_count` was undefined for every hypothesis. After repair the agents reached 6 of 12 on that earlier set. *These figures come from an earlier session and were not re-run for this write-up. They are on a different record set from the 14 of 29 below and are not a paired comparison.*

**2. Fix A3's schema filter.** The flow-level filter rejected any prediction containing "payload", including "zero payload", a legitimate claim about byte counts. The fix narrowed the filter. In the 45-record re-run, the 44 records that had produced output scored 0 schema failures. The five records that failed in the original run produced no output and were not re-run, so the fix's effect on them is unmeasured. Later runs still had failures: one A3 failure in v3 and seven advocate failures in the reframing arm.

**3. Give A2 the grounding block.** A2 was the only agent without grounding. Trust component V moved from median 0.125 to 0.143, and mean paired change was +0.022. `chain_vs_independent` moved from −0.002 to −0.021. Input starvation was not the cause.

**4. Four richer signals.** Per-feature percentiles, nearest benign neighbours with their values, source history over earlier time, and directional deviation, all computed from benign traffic only. Result: zero verdict flips across 44 records. 190 calls.

**5. Adversarial framing.** A5 asked whether a capable attacker would plausibly produce this flow, with an explicit instruction that resembling ordinary traffic is not exculpatory. Result: 14 of 29 attacks caught and 15 of 15 benign cleared. The benign-plausibility score moved on 19 of 44 records, but no verdict changed.

**6. Devil's-advocate agent.** A separate agent argued only the attack reading, under the same flow-level and grounding constraints. A5 weighed both sides. Result on the 37 records where the advocate validated: 11 of 23 attacks caught, 14 of 14 benign cleared. The count rule and the v3 agents score the same 11 of 23 on those 37 records, so this arm matched them rather than beating them.

**7. Two-sided scoring.** A5 reported benign and attack plausibility separately, with the verdict derived by a rule fixed before the run (attack if attack-plausibility exceeds benign-plausibility). Result: 13 of 29 attacks caught, 15 of 15 benign cleared. One record flipped, from attack to benign. Of 44 records, 30 were benign-leaning, 13 attack-leaning, and 1 neither plausible. No record was both plausible.

## The failure mechanism

Six attacks have a real but thin benign neighbourhood: counts of 30 to 117. The v3 agents score them 0.85 to 0.95 and clear them.

A5's reasoning on one of them reads "114 benign flows closely matching" as support for the benign case. The inference is locally valid: with 114 matching flows, a benign explanation does exist in the data. But a capable attacker also produces flows that look ordinary, and that consideration did not enter the reasoning in any arm.

No arm changed the call on any of the six. Two arms moved some scores on these records (the adversarial arm moved one from 0.85 to 0.72), but none crossed the threshold.

## A5's output is nearly bimodal

Across runs, benign plausibility took 8 to 9 distinct values, clustering at about 0.25–0.29 and 0.85–0.95. **Zero of 44 benign-plausibility scores fell in 0.35–0.65 in any arm.**

The second score in the two-sided arm (attack plausibility) did use the middle band: 13 of 44 values fell in 0.35–0.65, at 0.35, 0.40, 0.45 and 0.60. So the middle is not entirely unused, but the benign score that drives the verdict never used it.

When offered a ten-point anchored scale (0.05 through 0.95, each point described in words), A5 used five points and never used the middle band for benign plausibility.

## What the tests show

| Hypothesis | Test | Result |
|---|---|---|
| Insufficient evidence | four signals added | no verdict flips |
| Question framing | adversarial reframe | no call changed |
| Missing counter-argument | devil's advocate | matched the count on 37 records |
| Single score too coarse | two-sided scoring | no gain; one flip, in the wrong direction |
| One agent starved of input | A2 given grounding | V +0.018 median |
| Output format too coarse | ten-point anchored scale | middle band unused for benign score |

These results rule out the particular changes tested. They do not establish that no reframing could help. `chain_vs_independent` (about −0.02) measures agreement between the chain and the blind replica. It is not a test of alternative agent topologies, and the write-up does not claim one.

## Limitations

- n = 44 scored records (29 attacks, 15 benign). One record moves a rate by 2.3 points. Every rate carries a wide Wilson interval.
- The 30 attack / 15 benign mix is not the deployment rate, so this measures discrimination, not calibration.
- Later batches were selected after earlier results were seen. Pooling is replication, not an independent trial.
- **The 0.29 cap was chosen after the boundary effect was seen.** It changed the scores of six records, all attacks, relative to the 0.30 cap. The 0.30 verdict threshold was not changed. A cap of 0.30 would have left those six records at exactly the threshold, where they are not flagged. The prompts still state the cap as 0.3 while the code clamps at 0.29. This mismatch is not resolved.
- A single model family and a single prompt lineage were tested.
- The higher-cutoff count results are post-hoc and unvalidated.
- One record (1b21af7f, PortScan) failed A3 validation in v3 and is unscored in every arm.
- The "neither plausible" state occurred once and "both plausible" never occurred, so the two-sided arm's extra states were barely used.

## Conclusion

On this task, with this evidence and these prompts, the reasoning layer did not add to a threshold on the count it was given. Across six tested variants, no arm exceeded the count rule on attack recall, and none reduced benign specificity.

The mechanical grounding is the component that matters. The count-only rule alone reproduces the pipeline's verdicts exactly on this sample. Whether the agents help an analyst by explaining a verdict was not measured.

Open questions that this sample cannot answer: whether agent explanations are more actionable than a bare count, and whether the agents would help on grounded flows if asked a question that statistics cannot answer.
