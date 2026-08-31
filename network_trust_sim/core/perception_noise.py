"""
Hallucination-mode sensor corruption: jitters an agent's OBSERVED numeric
readings on specific routers, with no hidden instruction and no intent. Kept
separate from core/injection.py on purpose -- injection.py specifically
models a hidden-instruction/prompt-injection attack (the "0-day" scenario);
mixing that narrative with plain sensor noise would blur what each demo mode
is actually supposed to prove.

Because only the *perceived* observation is corrupted -- the ground-truth
NetworkState, and therefore what grounding/policy checks compare against,
stay clean -- a hallucinating agent's claims will fail to ground even though
its own reasoning from what it (wrongly) perceived is completely honest.
That's the intended signature: grounding_rate drops while entailment/policy
stay roughly normal, which is exactly what separates "hallucinating" from
"semantically Byzantine" using the existing Tier-1 decomposition.
"""

import copy
import random


def corrupt_observation(observation: dict, target_routers: list, noise_level: float, rng: random.Random) -> dict:
    corrupted = copy.deepcopy(observation)
    for router_id in target_routers:
        if router_id not in corrupted:
            continue
        r = corrupted[router_id]
        r["utilization_pct"] = round(_jitter(r["utilization_pct"], noise_level, 0, 100, rng), 1)
        r["latency_ms"] = round(_jitter(r["latency_ms"], noise_level, 0, 200, rng), 1)
        r["packet_loss_pct"] = round(_jitter(r["packet_loss_pct"], noise_level, 0, 10, rng), 3)
        r["error_rate_pct"] = round(_jitter(r["error_rate_pct"], noise_level, 0, 5, rng), 3)
    return corrupted


def _jitter(value: float, noise_level: float, lo: float, hi: float, rng: random.Random) -> float:
    span = (hi - lo) * noise_level
    return max(lo, min(hi, value + rng.uniform(-span, span)))
