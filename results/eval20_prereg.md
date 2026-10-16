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

## Pipeline changes after the shakedown, before any comparison run

A 3-record shakedown on `anthropic/claude-haiku-5.5` (OpenRouter) lost one record at A1. A1 was
rejected 3 times by the claim validators. The rejections were diagnosed from the validator
messages and the rule code alone. No label, verdict or score was consulted in the diagnosis.
(The shakedown's console output had displayed the two completed records' verdicts; they were
not used.) The rules were then changed as follows, before any comparison run:

1. **Rejected attempts are saved in full.** Each rejected attempt's text, its complete
   validator message, and its cost/provider/generation id are written to disk: in the call
   cache, in each record's `stage_meta`, and, for a record lost to validation, in
   `<run>/records_failed.jsonl` as soon as it fails.
2. **The comparison rule replaces the banned-word list** (`selfcheck_core.comparison_violations`):
   - These words judge a flow against outside traffic and are always rejected: unusual(ly),
     abnormal(ly), typical(ly), atypical, normal(ly), anomalous(ly), excessive(ly), rare(ly),
     ordinary, extreme(ly), significant(ly), elevated.
   - Bare size judgements are always rejected, because they have no second field to compare
     with: high, low, large, small, highest, lowest, largest, smallest.
   - larger, smaller, higher, lower, longer and shorter are allowed only when the same
     sentence names two of the fields it cites. That makes it a comparison within the flow.
   - Comparisons whose target is outside the flow are rejected whatever the word, for example
     "than most", "than usual", "than typical", "compared to other flows", "than other
     connections" or "for a connection". Under the old list, "Flow Duration is far longer
     than most connections" was accepted; it is now rejected.
   - Every violation in every claim is reported in a single retry message.
3. **Field-name rule:** a sentence may write underscores as spaces, and Fwd/Forward or
   Bwd/Backward interchangeably. A paraphrase that names no field is still rejected.
4. **The retry count is unchanged at 3.**

**Prompt version:** the comparison runs on `selfcheck_prompts_v3`, with all six agents at
`*_selfcheck_v3`.
- A1's prompt now shows the complete rule, generated from the same constants the validator
  applies. Under v1 and v2 it showed only the first 12 of 30 banned words.
- The other five prompts are textually identical to v2. v2 differs from v1 only in removing
  the JSON-shape prose, since a native strict JSON schema now carries the shape.

**Applied identically:** all five models run on v3 with these validators.
- The shakedown outputs and their cache entries were deleted before model 1.
- A1–A4 replayed from the upstream cache must match the run's prompt version, or the run
  refuses to start.

## Code and models the comparison runs at

**Code:** the comparison runs at commit `211e15c`.

- `5297e39` holds the selfcheck pipeline, the OpenRouter backend, the v3 prompts, the
  rewritten validators and their tests. A clean checkout of it imports, and its tests pass,
  except two fixtures that need gitignored `results/` data files.
- `211e15c` adds the cost stop, concurrency of 8 within a model, and the scorer. It changes
  no prompt, validator or schema.

**Models**, in run order (`config/openrouter.json` `sequence`). Every slug was resolved from
the full `/api/v1/models` catalog, fetched with curl, and verified against its
`/endpoints` list before any call:

| # | model | pinned endpoint | precision | $ in / out per 1M |
|---|---|---|---|---|
| 1 | `anthropic/claude-haiku-5.5` | `anthropic` (first party) | undisclosed | 0.100 / 0.500 |
| 2 | `deepseek/deepseek-v4.1-flash` | `deepinfra/fp8` | fp8 | 0.140 / 0.420 |
| 3 | `z-ai/glm-5.2` | `streamlake/fp8` | fp8 | 0.643 / 2.020 |
| 4 | `google/gemini-3.5-flash-lite` | `google-ai-studio` (first party) | undisclosed | 0.300 / 2.500 |
| 5 | `moonshotai/kimi-k2.6` | `moonshotai/int4` (first party) | int4 | 0.950 / 4.000 |

How each model was resolved:

- **GLM:** `z-ai/glm-5.2` is in the catalog, so it was kept. `z-ai/glm-5.3-prime` was the
  fallback, but it does not support structured outputs and could not run this pipeline.
- **Gemini:** `google/gemini-3.5-flash-lite` is the same model as the historical Gemini
  baseline, and is served via Google AI Studio, the API that baseline used.
- **Kimi:** `moonshotai/kimi-k2.6` is used, not K3.

**Endpoint rule** (fixed before any run):

1. Use the model's first-party endpoint if it supports strict structured outputs.
2. Otherwise use the cheapest active endpoint with structured outputs at a disclosed
   precision of fp8 or better. Endpoints at fp4/int4, endpoints with undisclosed precision,
   and endpoints whose tag is duplicated (so it cannot be pinned unambiguously) are excluded.

DeepSeek's and Z.AI's own endpoints do not offer structured outputs, so rule 2 applies to
those two models. Every request pins the endpoint (`only`/`order`, `allow_fallbacks: false`)
and also its disclosed quantization.

**Sampling: no temperature is set for any model.** Every model runs at its provider's default.

- Why: no single value is accepted by all five endpoints. Haiku 5.5 takes no temperature at
  all, and Moonshot's endpoint does not either, so a uniform provider default is the only
  consistent choice.
- The historical Gemini run used 0.7, so it differs from this comparison in temperature as
  well as in prompt version and API path.

**Runaway guards** (hard stops set above the long-reasoning estimates; not budgets):

| model | per-run stop |
|---|---|
| `anthropic/claude-haiku-5.5` | $0.50 |
| `deepseek/deepseek-v4.1-flash` | $0.75 |
| `z-ai/glm-5.2` | $2.50 |
| `google/gemini-3.5-flash-lite` | $2.50 |
| `moonshotai/kimi-k2.6` | $4.50 |
| global, all five | $10.00 |

- How it works: every priced call is appended to `results/eval20_runs/_spend_ledger.jsonl`.
  Before each call, its model's total and the global total are checked against the stops.
  When one trips, no further call is sent, and the tally records where the run stopped and
  which records were not run.
- Calls already in flight complete, so a stop can overshoot by at most 8 calls' cost.
- A run that trips a stop is treated as a bug, not as a slow model.

**Concurrency:** up to 8 concurrent calls within a model. Models run one at a time, in the
order above.

**Scoring** (`eval/score_eval20.py`, the only code that opens the key). The verdict-to-call
mapping is the one pre-registered for the selfcheck run (`results/selfcheck_preregistered.md`):

- **Caught:** an attack given `anomalous_and_unexplained`.
- **Flagged:** a benign flow given `anomalous_and_unexplained`.
- **Cleared:** a benign flow given `consistent_with_benign` or `anomalous_but_explicable`.
- **Neither caught nor cleared:** `insufficient_evidence`, or a record lost to validation.
- **Inconclusive:** a run with more than 30% `insufficient_evidence` is reported as
  inconclusive, alongside its J.

**Cost estimate (before any run):**

- The estimate assumes 150 calls per model: 20 records × 6 agents × 1.21 attempts, using the
  measured retry rate. Each call is taken as about 2.8k input tokens, and 1.5k ("base") or
  5k ("heavy reasoning") output tokens.
- Haiku is extrapolated from its measured shakedown ($0.0251 for 3 records).

| model | base | heavy |
|---|---|---|
| Haiku 5.5 | $0.17 | $0.17 |
| DeepSeek V4.1 Flash | $0.15 | $0.37 |
| GLM 5.2 | $0.72 | $1.78 |
| Gemini 3.5 Flash-Lite | $0.69 | $2.00 |
| Kimi K2.6 | $1.30 | **$3.40** |
| **whole sequence** | **$3.03** | **$7.73** |

Kimi K2.6, and Gemini at the margin, could exceed the $2.00 per-run ceiling if they reason at
length.

## Operational change during the run (commit `932c2d5`)

**What happened.** Model 2 (`deepseek/deepseek-v4.1-flash`) stopped after 5 of 20 records.
DeepInfra, the pinned endpoint, returned HTTP 429, which OpenRouter reported as an upstream
rate limit on a pool shared with other DeepInfra users. The client gave up after its old
retry limit of 6 attempts, about 31 seconds of backoff in total. Three records were cut off
mid-stage, and 12 had not started. This was a property of the provider's capacity, not of
the model's behaviour.

**What changed:**

- **429 backoff, for all models:** the client honours `retry-after` when given, and otherwise
  backs off exponentially. It waits for up to 300 seconds in total per call before giving
  up. Other retryable failures keep 6 attempts.
- **Every HTTP retry and its wait is now recorded.** Latency is reported both raw and
  excluding retry waits.
- **DeepSeek concurrency:** dropped from 8 to 3, for DeepSeek only. Models 3–5 stay at 8,
  unless they hit the same limit, in which case they also drop to 3, and that is noted.
  The concurrency each model actually used is reported.

**What did not change:** no prompt, schema, temperature, validator or scoring rule.

**Haiku is unaffected and is not re-run.** Its run printed no HTTP retry line of any kind,
so it saw zero 429s.

**How DeepSeek resumes:**
- It continues from its 5 completed records.
- Stages that completed inside the 3 interrupted records are reused from the call cache.
- Its raw latency includes rate-limit waits, so it is not comparable to Haiku's.
- Its first 5 records predate retry-wait recording, so they are excluded from the
  latency-excluding-waits figure.

## Follow-up variant `holdout3`: A3 predicts withheld fields (commit `a4b0dfb`, written before any call)

**Why.** In the selfcheck run, 197 of 204 hypotheses were SUPPORTED and only 4 were struck
out. Haiku's own eval20 run had A3 at 79 SUPPORTED, 1 CONTRADICTED and 0 UNCHECKABLE. A3
predicts fields it can see, so the check cannot reject anything.

**Change** (Haiku 5.5 only, pinned `anthropic` endpoint, the same 20 records):

- **What is withheld:**
  - For each record, 3 of the 10 escalation features (`ESCALATION_FEATURES`) are withheld
    from A3. They are chosen by `random.Random(f"20261016:{record_id}")`, so the choice
    depends on the opaque record id only, and is the same for every run.
  - Their exact twins are withheld too, because those carry the same value:
    `Total Fwd Packets` for `Subflow Fwd Packets`, and `Subflow Bwd Packets` for
    `Total Backward Packets`. Both pairs were checked to be equal on all 20 records.
  - So are A1/A2 claims that cite or name a withheld field, and A2 absences that read one.
    Building the prompts dropped 63 such claims in total.
  - A code assertion refuses any A3 prompt whose record-specific text names a withheld field.
- **What A3 must predict:** every A3 hypothesis must give at least one prediction for each
  withheld field, and predictions only about withheld fields. This is enforced by the
  validator; retries stay at 3.
- **How each hypothesis is graded**, against the real withheld values:
  - **CONTRADICTED** if any prediction is false. It is struck out, and A5b cannot credit it.
  - **SUPPORTED** only if every withheld field has a true prediction that could have failed.
  - **UNCHECKABLE** otherwise. A prediction "cannot fail" if the existing vacuity test says
    so, or if it is a bound every legal value meets: `Init_Win_bytes_*` lie in [-1, 65535]
    by protocol, so a prediction such as ≤ 65535 cannot fail. No population data is used.
- **A4 is unchanged** and still checked against the full row.

**Unchanged:** A1, A2, A4, A5a (v3 prompts, verbatim, and their cached outputs from the
eval20 Haiku run are reused, so only A3 and A5b are new calls), the conditions, the other
validators, the scoring rule and the J criterion.

**Prompt versions:** A3 is `a3_selfcheck_v4_holdout`. A5b's text is v3's, but it is versioned
`a5b_selfcheck_v4_holdout` because its input changes.

**Known residual leak.** Some withheld fields are partly derivable by arithmetic from visible
ones; for example, `Bwd Packets/s` from `Total Backward Packets` and `Flow Duration`, and
`Packet Length Mean` from length totals and packet counts. Exact twins are removed; derivable
fields are not.

**Stop and estimate:** cost stop $0.50 for this run. Expected $0.05–0.10 (Haiku's A3+A5b cost
$0.0495 on eval20).

**Reading the result:** if the strike-out rate stays near zero, the hypotheses were vacuous
rather than circular, and that is reported as the finding, not tuned.

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
