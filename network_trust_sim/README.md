# Network Trust Simulation

Simulates a swarm of LLM-driven network-ops agents, each managing one segment
of a backbone, and detects when one of them has been compromised via a
poisoned telemetry feed (a stand-in for a novel/"0-day" prompt-injection
attack) using a per-turn **TrustScore**.

## Setup

```
ollama pull llama3.1:8b
```

Leave Ollama running in the background.

```
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

Run:

```
python main.py
```

Output prints a live table to the console and writes a full per-turn JSON log
to `logs/run_<timestamp>.json` (every claim, every checker verdict, every
score component -- useful for your write-up's evidence/appendix).

## Folder structure

```
network_trust_sim/
  config.py                # all tunables: weights, thresholds, agent count, injection turn
  main.py                  # orchestrates the run, prints table, writes JSON log
  requirements.txt
  core/
    ollama_client.py       # HTTP wrapper for Ollama /api/chat and /api/embeddings
    telemetry.py           # synthetic per-segment telemetry generator (source of truth)
    injection.py           # the simulated poisoned-feed attack
    agent.py                # the agent: telemetry -> {claims, action, justification, confidence}
  trust/
    grounding.py            # Tier 1: are the claims supported by the telemetry?
    entailment.py           # Tier 1: does the action follow from the claims?
    behavior.py              # Tier 1: z-score anomaly vs. this agent's own history
    consistency.py           # Tier 2: resample N=5, only runs on escalation
    trust_score.py           # combines tiers + exponential decay across turns
  logs/                     # per-run JSON output (created automatically)
```

## How the attack is modeled

`core/injection.py` doesn't touch the agent's code or prompt. Instead, from
`INJECTION_TURN` onward, `COMPROMISED_AGENT_ID`'s telemetry has a hidden
instruction appended to it -- exactly like an attacker tampering with an
upstream log/sensor feed the agent trusts. This matches your problem
statement's "polished but deceptive justification" scenario: the agent isn't
buggy, its input is poisoned, and it produces a technically-formatted but
logically-disconnected response (`no_action_required` regardless of actual
readings). Because there's no signature to match against, detection has to
come from the agent's *behavior*, which is exactly what the trust score
measures.

## How TrustScore is computed

**Tier 1 (every agent, every turn -- fair by construction):**

```
S1 = w_grounding * grounding_rate + w_entailment * entailment_score + w_behavior * (1 - behavior_deviation)
```

- `grounding_rate`: Ollama LLM-judge checks each claim against the raw
  telemetry, returns fraction verified.
- `entailment_score`: Ollama LLM-judge checks whether the action + justification
  actually follows from the claims (entailed / unsupported_leap / contradicted).
- `behavior_deviation`: z-score of this turn's claim count / justification
  length / confidence against that specific agent's own rolling baseline.

**Tier 2 (only when `S1 < ESCALATION_THRESHOLD`):**

Resample the agent 5x on the same input, measure agreement (action agreement,
confidence coefficient-of-variation, claim-set overlap), and **multiply**
into S1 -- so consistency can only lower a suspicious score, never inflate a
clean one. This keeps agents that never escalate on equal footing with those
that do (see the fairness bug this fixes, discussed below).

**Decay across turns:**

```
TrustScore_t = lambda * S_final + (1 - lambda) * TrustScore_{t-1}
```

**Two separate flags, on purpose:**

- `turn_flagged` = `S_final < TURN_FLAG_THRESHOLD` -- an immediate, single-turn
  signal. This is what tells you *which turn* an agent was compromised on.
  It ignores history entirely.
- `trust_flagged` = `TrustScore_t < FLAG_THRESHOLD` -- the decayed, sustained
  signal. Slower to trip (by design -- one bad turn shouldn't take an agent
  offline), so don't use it to localize the incident, only to decide whether
  accumulated evidence justifies acting on it.

Trying to use `trust_flagged` alone to answer "which turn was it compromised"
will always lag the real incident turn by however many turns it takes the
decay to overcome the `1.0` starting prior -- that's expected behavior, not a
bug in the decay math, but it's the wrong tool for localization. Use
`turn_flagged` for that instead.

All weights/thresholds live in `config.py` and start at reasonable but
**uncalibrated** defaults (uniform-ish weights, `lambda=0.3`) -- worth stating
explicitly in your write-up rather than presenting them as tuned.

## If it's running slow

Each agent-turn can cost up to 8 Ollama calls (1 decision + grounding +
entailment + up to 5 resamples on escalation). With an 8B model on CPU that
adds up fast. Levers, roughly in order of impact:

- `CONSISTENCY_RESAMPLES` (default 3) -- the single biggest cost driver, cut
  it further to 2 while iterating.
- `NUM_TURNS` (default 6) / `NUM_AGENTS` (default 5) -- fewer agent-turns,
  less wall-clock time, linearly.
- `ESCALATION_THRESHOLD` -- lower it (e.g. 0.4) so fewer clean turns trigger
  Tier-2 resampling at all.
- Swap `OLLAMA_MODEL` to something smaller (e.g. `llama3.2:3b`) for fast
  iteration, then switch back to `llama3.1:8b` for your real evaluation run.

## Known limitations worth naming in your write-up

- **Adaptive baseline leakage**: `BehaviorTracker.update()` runs every turn,
  including flagged ones, so a sustained compromise slowly drags the
  "normal" baseline toward itself. Mitigation: stop updating the baseline for
  turns where `flagged=True`, or freeze the baseline once an agent is flagged
  until a human clears it.
- **LLM-judge as both grounding and entailment checker** means a single
  model's blind spots can affect two of your three Tier-1 signals at once.
  Using a second, different model (or a lightweight local NLI model) for one
  of the two would decorrelate the failure modes.
- **Detection vs. attribution stays split**: this system tells you *an agent
  looks compromised*, not *why*. True attribution (adversarial injection vs.
  genuine hallucination) needs input-side analysis (perplexity spikes, known
  injection markers) which this repo doesn't implement yet.
- **Cost is escalation-gated but not agent-count-gated**: with 5 agents x 8
  turns x (grounding + entailment + maybe 5x resample), a single run is
  ~13-30+ Ollama calls per agent-turn in the worst case. Fine for a class
  project; would need batching/caching for a larger swarm.

## Ideas for extending / improving the trust score

1. **Cross-agent consensus as a fifth signal.** Right now each agent is only
   judged against its own history and its own telemetry. Your problem
   statement is specifically about *swarm* contagion -- add a check where
   neighboring agents (with overlapping/adjacent segments) cross-verify a
   surprising claim before it's acted on. This turns detection into the
   "collective truth" mechanism your problem statement calls for, not just
   isolated per-agent scoring.
2. **Calibrate the weights instead of guessing them.** Once you have a batch
   of labeled runs (compromised vs. clean turns you know the ground truth
   for), fit `w_grounding`, `w_entailment`, `w_behavior`, and `lambda` with a
   simple logistic regression or grid search against the flagged/not-flagged
   label instead of hand-picked 0.4/0.4/0.2.
3. **Freeze-on-flag for the behavior baseline** (see limitation above) --
   cheap fix, meaningfully improves detection persistence once an agent is
   already under suspicion.
4. **Second, independent judge model** for entailment specifically (e.g. a
   smaller/different Ollama model, or a local NLI classifier) so grounding and
   entailment aren't both single points of failure from one model's blind spots.
5. **Vary the injection itself.** Right now there's one fixed payload. Add a
   small library of injection variants (subtle vs. blatant, single-turn vs.
   slow-drift over several turns, injected in different telemetry fields) and
   report TrustScore's detection latency/false-positive rate per variant --
   this is the kind of robustness table that makes a "0-day" claim credible
   rather than just testing one known pattern.
6. **False-positive baseline.** Run several fully clean multi-agent sessions
   and report how often TrustScore flags an innocent agent anyway (noisy
   telemetry, model being verbose one turn, etc.). A detector is only as
   good as its false-positive rate, and this is easy to generate with the
   current harness -- just set `INJECTION_TURN` past `NUM_TURNS`.