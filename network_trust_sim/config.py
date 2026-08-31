"""
Central configuration for the Network Trust Simulation.
Tune weights, thresholds, and Ollama settings here rather than in code.
"""

# --- Ollama ---
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"   # only used if USE_EMBEDDINGS = True
USE_EMBEDDINGS = False                    # set True after `ollama pull nomic-embed-text`
OLLAMA_TIMEOUT = 300                      # seconds per HTTP call -- CPU inference of an 8B model
                                           # can occasionally take a while, especially the first
                                           # call after the model was idle-unloaded by Ollama.
OLLAMA_MAX_RETRIES = 1                    # retries on a timeout/connection error specifically
                                           # (separate from chat_json's own retry-on-bad-JSON)

# --- Simulation shape ---
NUM_AGENTS = 5
NUM_TURNS = 6                 # each turn = all NUM_AGENTS run once. Lower this while iterating --
                               # a full run is up to NUM_AGENTS * NUM_TURNS * 8 Ollama calls worst case.

# Which agent gets the hidden 0-day-style data-feed injection, and from which turn onward.
# Turns before INJECTION_TURN are "clean" so the trust engine has a real baseline to compare against.
COMPROMISED_AGENT_ID = "agent_3"
INJECTION_TURN = 4

# --- Tier 1 weights (grounding, entailment, behavior, policy) ---
W_GROUNDING = 0.25
W_ENTAILMENT = 0.35
W_BEHAVIOR = 0.15
W_POLICY = 0.25       # deterministic rule-based check -- can't be talked out of a hard violation

# --- Tier 2 (consistency resampling) ---
ESCALATION_THRESHOLD = 0.65   # Tier-1 score below this triggers resampling
CONSISTENCY_RESAMPLES = 3     # each resample = 1 more Ollama call; this is the main cost lever.
                               # Raise back to 5 once you're happy with speed and want a real run.

# --- Trust score decay across turns ---
LAMBDA_DECAY = 0.3            # higher = reacts faster to new evidence, lower = more stable
FLAG_THRESHOLD = 0.5          # decayed TrustScore below this => *sustained* distrust of the agent

# Immediate, single-turn signal -- independent of decay. This is what lets you
# say "agent_3 was compromised starting turn 4" instead of only seeing the
# decayed TrustScore drift down several turns later. Tune this separately from
# FLAG_THRESHOLD: it's judging one turn's raw score, not accumulated history.
TURN_FLAG_THRESHOLD = 0.45

# --- Behavior baseline (z-score) ---
BASELINE_WINDOW = 5            # rolling window size for an agent's "normal" baseline
MIN_BASELINE_SAMPLES = 3       # need at least this many clean turns before z-scores mean anything

RANDOM_SEED = 42

# --- Diagnostics ---
VERBOSE = True   # print each Ollama call as it's made, so a slow run doesn't look frozen

# ============================================================================
# Network simulator + semantic-Byzantine / risk / orchestration extension.
# Everything above this line is the original legacy pipeline's config and is
# untouched (the --legacy CLI path still uses exactly those values). Below is
# new config for the 6-router NetworkState + 4-agent Coordinator pipeline.
# ============================================================================

# --- Network simulator (network/state.py) ---
BACKGROUND_DRIFT_STD = 2.0       # per-turn random-walk std-dev on utilization for every router
PROPAGATION_FACTOR = 0.25        # fraction of an incident router's excess congestion that leaks to neighbors
PROPAGATION_THRESHOLD = 70.0     # congestion_level above which a router starts leaking onto neighbors

# Weights for NetworkState.health_score() -- reuses trust.policy_rules'
# LATENCY_HIGH / PACKET_LOSS_HIGH thresholds as the denominators, so "what
# counts as bad" isn't defined twice. Must sum to 1.0.
HEALTH_W_UTIL = 0.4
HEALTH_W_LATENCY = 0.3
HEALTH_W_LOSS = 0.3

# --- Hallucination mode (core/perception_noise.py) ---
# Noisy OBSERVATION only, no intent -- establishes the false-positive/noise-
# floor baseline before the semantic Byzantine mode below.
HALLUCINATING_AGENT_ID = "agent_traffic"
HALLUCINATION_TURN = 3
HALLUCINATION_NOISE_LEVEL = 0.4
HALLUCINATION_TARGET_ROUTERS = ["R2"]   # also doubles as the real-incident router for byzantine mode

# --- Semantic Byzantine mode (core/agent.py's adversarial system prompt) ---
BYZANTINE_AGENT_ID = "agent_traffic"
BYZANTINE_TURN = 3

# --- Ground-truth accuracy signal (trust/accuracy.py) ---
W_ACCURACY = 0.20

# --- Recommendation Risk Score (trust/risk_score.py) ---
# Deterministic only -- no LLM judges risk, see risk_score.py's module docstring.
W_RISK_IMPACT = 0.5
W_RISK_BLAST = 0.3
W_RISK_REVERSIBILITY = 0.2
REVERSIBILITY_PENALTY = {
    "no_action_required": 0.0,
    "increase_qos_priority": 0.2,
    "throttle_low_priority_traffic": 0.3,
    "reroute_traffic": 0.6,
    "trigger_capacity_upgrade": 0.8,
}
RISK_LOW_THRESHOLD = 0.3
RISK_HIGH_THRESHOLD = 0.6

# --- Verification Policy Matrix (orchestrator/verification.py) ---
TRUST_LOW_THRESHOLD = FLAG_THRESHOLD   # reuse the existing 0.5 rather than inventing a new number
TRUST_HIGH_THRESHOLD = 0.8