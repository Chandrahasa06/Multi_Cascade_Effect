# Reference arms: all-days, Monday, Monday+Tuesday

**Setup.** A5 is the only model call per record. Upstream A1 to A4 are the recorded v2 outputs; their grounding block is the Monday Tier-1 reference for every arm, so the upstream is identical across arms. A5 uses the corrected prompt (`a5_verdict_v7`, cap 0.29), the real leave-one-out count, and profile sentences drawn from the arm's reference window. The evidence block is rebuilt from the same window.

**Honesty.** Counts primary; Wilson 95% intervals on every rate. Arms are smaller than 44. Monday is one day of reference tested up to four days later, so Arm M is a lower bound, not a deployment estimate. All-days grounding was not deployment-realistic: it includes the same days as the attacks it is tested on.

## 1. Qualifying records per arm

| arm | window | evaluated | dropped | attacks | benign |
|---|---|---|---|---|---|
| A | all-days | 45 of 45 | 0 | 30 | 15 |
| M | monday | 42 of 45 | 3 | 29 | 13 |
| MT | monday+tuesday | 39 of 45 | 6 | 26 | 13 |

Dropped from arm M:

- 9721796b2e (DoS Hulk): ambiguous day: matching rows span weekdays [2, 3]
- e209f4618e (BENIGN): ambiguous day: matching rows span weekdays [2, 4]
- e9324f3833 (BENIGN): inside the reference window (weekday 0)

Dropped from arm MT:

- 3da627f972 (SSH-Patator): inside the reference window (weekday 1)
- 405408a974 (FTP-Patator): inside the reference window (weekday 1)
- 9721796b2e (DoS Hulk): ambiguous day: matching rows span weekdays [2, 3]
- d67a88f41a (SSH-Patator): inside the reference window (weekday 1)
- e209f4618e (BENIGN): ambiguous day: matching rows span weekdays [2, 4]
- e9324f3833 (BENIGN): inside the reference window (weekday 0)

**Common subset** present in all three arms: 39 records (26 attacks, 13 benign).

## 2. Agents and count-only control, per arm

| arm | attacks caught, agents | attacks caught, count-only | benign cleared, agents | benign cleared, count-only | disagreements |
|---|---|---|---|---|---|
| A | 15/30 = 50.0% (95% CI 33.2-66.8%) | 15/30 = 50.0% (95% CI 33.2-66.8%) | 15/15 = 100.0% (95% CI 79.6-100.0%) | 15/15 = 100.0% (95% CI 79.6-100.0%) | 0 |
| M | 20/29 = 69.0% (95% CI 50.8-82.7%) | 20/29 = 69.0% (95% CI 50.8-82.7%) | 13/13 = 100.0% (95% CI 77.2-100.0%) | 13/13 = 100.0% (95% CI 77.2-100.0%) | 0 |
| MT | 18/26 = 69.2% (95% CI 50.0-83.5%) | 18/26 = 69.2% (95% CI 50.0-83.5%) | 13/13 = 100.0% (95% CI 77.2-100.0%) | 13/13 = 100.0% (95% CI 77.2-100.0%) | 0 |

## 3. Common subset, identical flows in all three arms

| method | attacks caught | benign cleared |
|---|---|---|
| arm A agents | 14/26 = 53.8% (95% CI 35.5-71.2%) | 13/13 = 100.0% (95% CI 77.2-100.0%) |
| arm A count-only | 14/26 = 53.8% (95% CI 35.5-71.2%) | 13/13 = 100.0% (95% CI 77.2-100.0%) |
| arm M agents | 18/26 = 69.2% (95% CI 50.0-83.5%) | 13/13 = 100.0% (95% CI 77.2-100.0%) |
| arm M count-only | 18/26 = 69.2% (95% CI 50.0-83.5%) | 13/13 = 100.0% (95% CI 77.2-100.0%) |
| arm MT agents | 18/26 = 69.2% (95% CI 50.0-83.5%) | 13/13 = 100.0% (95% CI 77.2-100.0%) |
| arm MT count-only | 18/26 = 69.2% (95% CI 50.0-83.5%) | 13/13 = 100.0% (95% CI 77.2-100.0%) |

## 4. Disagreements, printed in full

### Arm A (0)

None.

### Arm M (0)

None.

### Arm MT (0)

None.

## 5. Six previously-missed grounded attacks (old counts 30 to 117)

| record | old count | v3 score | arm A | arm M | arm MT |
|---|---|---|---|---|---|
| 2c27e52883 | 88 | 0.95 | 0.88 (benign, count 88) | 0.29 (attack, count 11) | 0.29 (attack, count 11) |
| 5b39770724 | 114 | 0.95 | 0.95 (benign, count 114) | 0.95 (benign, count 31) | 0.88 (benign, count 65) |
| 70448e2e53 | 62 | 0.90 | 0.75 (benign, count 62) | 0.29 (attack, count 10) | 0.29 (attack, count 10) |
| b32f2676f5 | 117 | 0.85 | 0.90 (benign, count 116) | 0.29 (attack, count 0) | 0.29 (attack, count 0) |
| d07e7212eb | 76 | 0.85 | 0.85 (benign, count 76) | 0.29 (attack, count 10) | 0.29 (attack, count 10) |
| d67a88f41a | 30 | 0.90 | 0.85 (benign, count 30) | 0.29 (attack, count 13) | not evaluated |

## 6. Does the real count change anything? (Arm A against the recorded all-days result)

Scores that changed between the recorded v3 run and arm A: 29 of 44 records in both. Verdict flips: 0.

## 7. Score spread per arm

| arm | distinct benign-plausibility values | scores in 0.35-0.65 | value counts |
|---|---|---|---|
| A | 8 | 0 of 45 | {0.29: 15, 0.75: 2, 0.85: 7, 0.88: 4, 0.9: 2, 0.95: 11, 0.98: 2, 0.99: 2} |
| M | 9 | 0 of 42 | {0.29: 20, 0.75: 1, 0.82: 1, 0.85: 5, 0.88: 2, 0.9: 2, 0.92: 1, 0.95: 9, 0.99: 1} |
| MT | 7 | 0 of 39 | {0.29: 18, 0.85: 5, 0.88: 3, 0.9: 2, 0.92: 3, 0.95: 7, 0.98: 1} |

## 8. Specificity trade

- Arm A: benign cleared by agents 15/15 = 100.0% (95% CI 79.6-100.0%). Recall 15/30 = 50.0% (95% CI 33.2-66.8%).
- Arm M: benign cleared by agents 13/13 = 100.0% (95% CI 77.2-100.0%). Recall 20/29 = 69.0% (95% CI 50.8-82.7%).
- Arm MT: benign cleared by agents 13/13 = 100.0% (95% CI 77.2-100.0%). Recall 18/26 = 69.2% (95% CI 50.0-83.5%).

## Limits

- Small arms. One record moves a rate by a few points. Wilson intervals are wide.
- The mapping leaves seven profile features unevaluable (not loaded in the cached pool) and four per-source features unevaluable per flow. Predictions that name them are reported as unevaluable, with no support count.
- The zero-support cap cannot fire on an unevaluable hypothesis. The ungrounded cap still fires on the count.
- Source IP and time for the evidence history come from matching the flow's features against the pool (see the v3 report). Five records have ambiguous source matches and no history.
- A1 to A4 are the v2 outputs. The v3 evidence block is not reused in the arms; the arms rebuild it per window.
