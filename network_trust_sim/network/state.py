"""
Stateful ground-truth model of the 6-router backbone. Agents only ever see a
restricted view of this via observation_for(); the simulator itself always
holds the full truth, which is what grounding/policy/accuracy checks are
verified against -- exactly the same clean-vs-seen split the original
telemetry.py/injection.py pair used, just now backed by a real shared,
persistent network instead of independent i.i.d. draws per segment per turn.
"""

import copy
import random

from config import (
    RANDOM_SEED,
    BACKGROUND_DRIFT_STD,
    PROPAGATION_FACTOR,
    PROPAGATION_THRESHOLD,
    HEALTH_W_UTIL,
    HEALTH_W_LATENCY,
    HEALTH_W_LOSS,
)
from trust.policy_rules import LATENCY_HIGH, PACKET_LOSS_HIGH
from network.topology import ROUTER_IDS, AGENT_DOMAINS, neighbors
from network.actions import apply_effect

CAPACITY_GBPS = 500.0


def _initial_router(rng: random.Random) -> dict:
    utilization = rng.uniform(30.0, 55.0)
    packet_loss = rng.uniform(0.0, 0.3)
    error_rate = rng.uniform(0.01, 0.15)
    latency = rng.uniform(3.0, 10.0)
    return {
        "utilization_pct": round(utilization, 1),
        "capacity_gbps": CAPACITY_GBPS,
        "throughput_gbps": round(CAPACITY_GBPS * utilization / 100, 1),
        "latency_ms": round(latency, 1),
        "packet_loss_pct": round(packet_loss, 3),
        "error_rate_pct": round(error_rate, 3),
        "congestion_level": round(utilization, 1),
        "routing_cost": round(1.0 + utilization / 100, 2),
    }


class NetworkState:
    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = random.Random(seed)
        self.turn = 0
        self.routers = {r: _initial_router(self.rng) for r in ROUTER_IDS}

    def snapshot(self) -> dict:
        return copy.deepcopy(self.routers)

    def observation_for(self, agent_id: str) -> dict:
        """Restricted view: only the router readings this agent is authorized
        to see, per network.topology.AGENT_DOMAINS. This IS the partial-
        observability model -- an agent literally cannot construct a view of
        routers outside its domain."""
        routers = AGENT_DOMAINS.get(agent_id, [])
        return {r: copy.deepcopy(self.routers[r]) for r in routers}

    def health_score(self, routers: dict = None) -> float:
        """Pure function of a routers dict (defaults to current committed
        state), so hypothetical/counterfactual states can be scored too.
        Reuses the exact thresholds trust.policy_rules already defines for
        'what counts as bad', instead of inventing a second vocabulary."""
        routers = routers if routers is not None else self.routers
        if not routers:
            return 1.0
        scores = []
        for r in routers.values():
            util_penalty = min(1.0, r["utilization_pct"] / 100.0)
            latency_penalty = min(1.0, r["latency_ms"] / LATENCY_HIGH)
            loss_penalty = min(1.0, r["packet_loss_pct"] / PACKET_LOSS_HIGH)
            per_router = 1.0 - (
                HEALTH_W_UTIL * util_penalty
                + HEALTH_W_LATENCY * latency_penalty
                + HEALTH_W_LOSS * loss_penalty
            )
            scores.append(max(0.0, min(1.0, per_router)))
        return sum(scores) / len(scores)

    def step(self, incident: dict = None) -> None:
        """Advance the network one turn: bounded random walk on every router,
        plus congestion propagation to topology-adjacent routers when a
        router is under real, developing stress (incident != None). This
        replaces telemetry.py's stateless i.i.d. draws with genuine
        cause-and-effect between segments."""
        self.turn += 1
        incident_router = incident.get("router_id") if incident else None
        severity = incident.get("severity", 0) if incident else 0

        for router_id, r in self.routers.items():
            drift = self.rng.gauss(0, BACKGROUND_DRIFT_STD)
            if router_id == incident_router and severity > 0:
                target_util = min(97.0, 70.0 + severity * 6.0)
                r["utilization_pct"] = round(max(r["utilization_pct"], target_util) + drift, 1)
                r["latency_ms"] = round(min(60.0, 14.0 + severity * 4.0), 1)
                r["packet_loss_pct"] = round(min(3.0, 0.4 + severity * 0.2), 3)
                r["error_rate_pct"] = round(min(1.5, 0.25 + severity * 0.15), 3)
            else:
                r["utilization_pct"] = round(max(5.0, min(98.0, r["utilization_pct"] + drift)), 1)
                r["latency_ms"] = round(max(1.0, r["latency_ms"] + self.rng.gauss(0, 0.5)), 1)
                r["packet_loss_pct"] = round(max(0.0, r["packet_loss_pct"] + self.rng.gauss(0, 0.02)), 3)
                r["error_rate_pct"] = round(max(0.0, r["error_rate_pct"] + self.rng.gauss(0, 0.01)), 3)

            r["congestion_level"] = round(0.7 * r["congestion_level"] + 0.3 * r["utilization_pct"], 1)
            r["throughput_gbps"] = round(r["capacity_gbps"] * r["utilization_pct"] / 100, 1)
            r["routing_cost"] = round(1.0 + r["congestion_level"] / 100, 2)

        if incident_router and severity > 0:
            source = self.routers[incident_router]
            if source["congestion_level"] > PROPAGATION_THRESHOLD:
                excess = source["congestion_level"] - PROPAGATION_THRESHOLD
                for nb in neighbors(incident_router):
                    if nb not in self.routers:
                        continue
                    bump = excess * PROPAGATION_FACTOR
                    nbr = self.routers[nb]
                    nbr["utilization_pct"] = round(min(98.0, nbr["utilization_pct"] + bump * 0.5), 1)
                    nbr["congestion_level"] = round(min(100.0, nbr["congestion_level"] + bump), 1)

    def apply_action(self, action: dict, committing: bool = False) -> dict:
        """committing=False -> operates on a deep copy (the counterfactual
        simulator, used for risk scoring). committing=True -> mutates the
        real state (the actual commit, only ever called when a recommendation
        is allowed to propagate)."""
        working = copy.deepcopy(self.routers)
        before_health = self.health_score(self.routers)
        mutated, affected = apply_effect(working, action)
        after_health = self.health_score(mutated)
        if committing:
            self.routers = mutated
        return {
            "before_health": before_health,
            "after_health": after_health,
            "affected_routers": affected,
            "state_after": mutated,
        }
