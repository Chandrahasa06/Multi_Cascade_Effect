# v3eval — preregistration (written and committed before any model call)

A separate evaluation on the **v3 policy run** (`results/escalated_flows_v3.csv`). It is not
eval20 and is never merged with eval20. The two are different policy runs, and their numbers
are reported side by side, never pooled.

**Code:** commit `f8f1e51`. That commit holds the draw, the blind input and the runner and
scorer changes. No prompt, condition, validator or scoring rule changed.

## Purpose: breadth, not power

eval20 covers 4 attack classes; v3 covers 10. This run tests **coverage**: does the pipeline
catch each kind of attack at all.

With 1 Infiltration, 1 XSS, 1 DDoS and 2 PortScan records, **per-class results cannot
generalise.** Each class is reported as coverage only (caught / missed), never as a rate.

## Sample

- **Attacks:** all 42 escalated v3 attacks, with no sampling. All are TCP.

  | class | n |
  |---|---|
  | DoS Hulk | 15 |
  | DoS Slowhttptest | 8 |
  | Bot | 4 |
  | DoS slowloris | 4 |
  | SSH-Patator | 4 |
  | FTP-Patator | 2 |
  | PortScan | 2 |
  | DDoS | 1 |
  | Infiltration | 1 |
  | Web Attack – XSS | 1 |

- **6 of the 42 attacks were evaluated before.** They are the same CSV rows used in eval20 or
  the earlier 50: 2 SSH-Patator, 2 DoS Hulk, 1 DoS Slowhttptest and 1 Bot. They are kept, as
  instructed, and flagged per record (`previously_evaluated`). Results are reported both with
  and without them.
- **Benign:** 42 TCP flows, drawn by eval20's seeded day-cycling method (`draw_group`, seed
  `20261014`). The pool is the 1,345 escalated benign TCP flows in v3 that were never
  evaluated before; 2 previously evaluated benign flows were excluded. All 8 capture days
  are covered (5–6 each).
- **Files:** the key file is `results/v3eval_records.json`, which no decision path can reach
  (asserted in tests). The blind input is `results/v3eval_input_blind.jsonl`, built by
  eval20's identical build, strip and verify steps. Protocol was recovered from the raw CSV
  without labels: all 84 records are TCP, each matched uniquely.

## Configuration (the current one, unchanged)

- Haiku 5.5 (`anthropic/claude-haiku-5.5`), pinned to the `anthropic` endpoint, no
  temperature set, concurrency 8.
- Variant `holdout3_noquota` (prompts v6):
  - A3 and A4 both withhold 3 of the 10 escalation features per record, chosen by seed
    `20261016` from the opaque record id, and withhold those fields' exact twins too.
  - There is no minimum number of benign hypotheses.
  - The SUPPORTED / CONTRADICTED / UNCHECKABLE rule is unchanged.
- **Twins:** the twin list was checked on eval20's records, and re-checked on these 84 before
  any call. `Subflow Fwd Packets` = `Total Fwd Packets` and `Total Backward Packets` =
  `Subflow Bwd Packets` on all 84 records, and no other escalation feature has an exact twin
  here.
- **Unchanged:** prompts, conditions, validators, the scoring map and the J criterion.

## Scoring (pre-registered, unchanged)

- **Caught:** an attack given `anomalous_and_unexplained`.
- **Flagged:** a benign flow given `anomalous_and_unexplained`.
- **Cleared:** a benign flow given `consistent_with_benign` or `anomalous_but_explicable`.
- **Neither caught nor cleared:** `insufficient_evidence`, or a record lost to validation.
- **J** = caught/42 + cleared/42 − 1.
- **Inconclusive:** a run with more than 30% `insufficient_evidence` is reported as
  inconclusive.

**Per class, reported for every attack class:**
- caught;
- missed;
- lost;
- **how many of the class's records had at least one condition marked yes.**

That last column separates "the agents failed" from "nothing was ever flagged to fail on". On
a record with no condition marked yes, the verdict rule clears the flow whatever the
hypotheses say.

## Expectations, recorded before any call

- **Run-to-run variance:** about ±0.10 in J on eval20, measured from one PortScan record
  (`8d754291d1`) that flipped between the holdout and no-quota runs. One record is 0.10 at
  10 per group, but about 0.024 at 42 per group. So a difference of a few records between
  this run and any other is within noise.
- **Bot and DDoS are predicted to be missed.** `results/conditions_v4_measured.md` found no
  usable condition holding on them: DDoS at most 0.9%, and Bot's only substantial hold is
  D9, which is unusable. A catch of either would be a surprise, worth inspecting record by
  record.

## Stop and cost

- Hard limit **$2.00** for this run. The guard trips at $1.95, so calls in flight
  (at most 8, the largest Haiku stage about $0.004) cannot carry spend past $2.00.
- **Expected about $0.75.** Haiku costs $0.0088 per record with all six agents fresh, × 84.
- The run will not start unless the OpenRouter key has at least the remaining ceiling plus a
  $0.25 margin (for the key's reporting lag). It refuses at the start rather than failing
  midway.
