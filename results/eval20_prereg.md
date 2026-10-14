# eval20 -- preregistration (written and committed before any model run)

## Sample

- **Source:** `results/escalated_flows.csv` -- the escalated flows of the existing policy
  run (3,000 flows, 282 attack / 2,718 benign). This is the same run every earlier agent
  evaluation record was drawn from.
- **Exclusions:** the 50 flows already given to the agents as input (batches
  `agent_input_20`, `agent_input_20b` and `agent_input_10c`). They were identified by
  matching their input feature vectors against this pool's raw CSV rows. No run output,
  score, tally or key file was read when drawing. Exactly 50 of 50 matched.
- **Seed:** `20261014` (`numpy.random.default_rng`), one generator for the whole draw.
- **Draw method** (`eval/draw_eval20.py`): the attack group and the benign group are drawn
  separately, 10 each. For each group:
  1. eligible capture days are the days with at least one unused row of that group;
  2. the rows within each day are shuffled, then the order of the days is shuffled;
  3. records are taken round-robin over that day order, one per day per pass, until 10
     are taken.

  Every eligible day is represented before any day gets a second record. The final list
  is then shuffled and given opaque `record_id`s, so a record's position does not reveal
  its group.

### Day distribution

| capture day | attack | benign |
|---|---|---|
| friday_ddos | 2 | 2 |
| friday_morning | 2 | 1 |
| friday_portscan | 3 | 1 |
| monday | 0 | 2 |
| thursday_infiltration | 0 | 1 |
| thursday_webattacks | 0 | 1 |
| tuesday | 0 | 1 |
| wednesday | 3 | 1 |
| **total** | **10** | **10** |

Benign records cover all 8 capture days. Attack records cover the 4 days that still have
escalated attacks after the exclusions:

- Monday, both Thursday captures and Tuesday have none left. Monday and Thursday had no
  escalated attacks in this run at all, and Tuesday's 4 were all in the excluded 50.
- `friday_morning` had only 2 attacks left, and both were taken.

Protocols: 15 TCP, 5 UDP (all 10 attacks are TCP).

## Files

- `results/eval20_records.json` is the **key file**: `record_id`, `flow_id`, `day`,
  `protocol`, `true_label`, `is_attack`, plus the seed and the day distribution.
- No decision path may read it. The existing assertions in `tests/test_selfcheck.py` and
  `tests/test_priors_v3.py` check that no label field or key filename appears in decision
  code. They are kept, and `eval20_records` has been added to them. The agent-facing
  input for these 20 records will be built separately, without labels, by the existing
  blind-input stripping and verification step.

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
