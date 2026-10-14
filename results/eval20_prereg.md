# eval20 -- preregistration (written and committed before any model run)

## Sample

- **Source:** `results/escalated_flows.csv` -- the escalated flows of the existing policy
  run (3,000 flows, 282 attack / 2,718 benign). This is the same run every earlier agent
  evaluation record was drawn from.
- **Same run as the evaluated system:** the selfcheck pipeline reads no escalated-flows
  file directly. Its records come from `load_45` (in `eval/run_blind_pipeline_45_v2.py`),
  which reads the blind inputs of batches 20 / 20b / 10c, and `eval/select_agent_sample*`
  drew all three from `escalated_flows.csv`. No part of the evaluated pipeline reads
  `escalated_flows_v3.csv`.
- **Exclusions:** the 50 flows already given to the agents as input (batches
  `agent_input_20`, `agent_input_20b` and `agent_input_10c`). They were identified by
  matching their input feature vectors against this pool's raw CSV rows. No run output,
  score, tally or key file was read when drawing. Exactly 50 of 50 matched.
- **Seed:** `20261014` (`numpy.random.default_rng`), one generator for the whole draw.
  Attacks are drawn first, then benign.
- **Draw method** (`eval/draw_eval20.py`): the attack group and the benign group are drawn
  separately, 10 each. For each group:
  1. eligible capture days are the days with at least one unused row of that group;
  2. the rows within each day are shuffled, then the order of the days is shuffled;
  3. records are taken round-robin over that day order, one per day per pass, until 10
     are taken, or until the pool runs out. Any shortfall is recorded, not padded.

  Every eligible day is represented before any day gets a second record. The final list
  is then shuffled and given opaque `record_id`s, so a record's position does not reveal
  its group.
- **Benign candidates are TCP only** (see "Redraw" below). 1,286 unused escalated TCP
  benign flows were available, so there is **no shortfall**.

### Redraw (before any run): benign group restricted to TCP

The first draw (commit `a07165b`) had 10 TCP attacks and 10 benign records, 5 of them
UDP. On that sample, a trivial "UDP means benign" rule catches 0 of 10 attacks and clears
5 of 10 benign. Youden's J for any classifier is at most its own benign-cleared share,
so that rule scores the full J = 0.50. It could beat the real system on protocol
recognition alone, so the comparison would not measure detection.

The benign 10 were redrawn from TCP-only candidates with the same seed and the same
day-cycling method. The attack draw comes first from the same generator, so the attack
10 are identical to the first draw (checked by flow id). One benign record from the
first draw recurs. All `record_id`s are new.

### Day distribution

| capture day | attack | benign |
|---|---|---|
| friday_ddos | 2 | 1 |
| friday_morning | 2 | 1 |
| friday_portscan | 3 | 1 |
| monday | 0 | 2 |
| thursday_infiltration | 0 | 1 |
| thursday_webattacks | 0 | 1 |
| tuesday | 0 | 1 |
| wednesday | 3 | 2 |
| **total** | **10** | **10** |

Benign records cover all 8 capture days. Attack records cover the 4 days that still have
escalated attacks after the exclusions:

- Monday, both Thursday captures and Tuesday have none left. Monday and Thursday had no
  escalated attacks in this run at all, and Tuesday's 4 were all in the excluded 50.
- `friday_morning` had only 2 attacks left, and both were taken.

Protocols: all 20 records are TCP.

### Attack class mix and expected ceiling (recorded in advance of any result)

| class | records |
|---|---|
| DoS Hulk | 3 |
| PortScan | 3 |
| DDoS | 2 |
| Bot | 2 |

`results/conditions_v4_measured.md` measured DDoS and Bot as having no condition
holding:

- On the earlier evaluation records, none of the 14 pre-registered conditions holds on
  any Bot or DDoS record.
- Across the full pool, the most any condition reaches on DDoS is 0.9% (D2). Bot's only
  substantial hold is D9, at 37.4%, and D9 is unusable because it also holds on 12.45% of
  benign TCP. Apart from D9, Bot's highest hold is D11 at 0.5%.

The **expected ceiling on this sample is therefore roughly 6 of 10 attacks**: the 3 DoS
Hulk and 3 PortScan. This expectation is written down before any model has been run on
these 20 records.

## Files

- `results/eval20_records.json` is the **key file**: `record_id`, `flow_id`, `day`,
  `protocol`, `true_label`, `is_attack`, plus the seed, the benign protocol filter, the
  pool size, the shortfall and the day distribution.
- No decision path may read it. The existing assertions in `tests/test_selfcheck.py` and
  `tests/test_priors_v3.py` check that no label field or key filename appears in decision
  code. They are kept, and `eval20_records` has been added to them.
- `results/eval20_input_blind.jsonl` is the agent-facing input, built by
  `eval/build_eval20_input.py`. The build reuses the earlier batches' own steps: the
  feature extraction, bins, digest and ground-truth check in `eval/select_agent_sample`,
  then `strip_to_blind` and `verify_blind_file` in `eval/build_blind_input`. On top of
  those, the file is asserted to contain no label, priority, reason, rule id or rule
  class prediction, checked both by key and by string value.
- One difference from the earlier batches: `feature_bins` has 4 keys rather than 5.
  `syn_without_synack_count` was later removed from `PRIORITY2_FEATURES` because its bin
  edges collapse to a single bin. It is 1 on all 50 earlier records, so it carries no
  information, and no decision code reads `feature_bins`.

## Selection rule (fixed now)

    Winner = highest Youden's J = (attacks caught / 10) + (benign cleared / 10) - 1

- **Ties:** first, fewer records lost to validation failure; then, lower cost.
- **Raw recall is not the criterion.**

How the terms are counted, as the formula implies:

- The denominators are always 10.
- A record lost to validation failure (no valid final verdict after the condition's own
  retries) counts as neither caught nor cleared. It lowers J for its group and is also
  counted for the first tie-break.
- A record whose final verdict is neither "attack" nor "benign" (for example
  uncertain/abstain) is likewise neither caught nor cleared.
- **Cost** means the total API requests made for the 20 records under that condition.
