"""
Discrete-event replacement for orchestrator/coordinator.py's fixed-order
turn loop. The old Coordinator processes every agent sequentially in a
single global order each turn (self.agent_order), so a compromised agent's
"cascade pathway" only exists because agents earlier in that order are seen
as peer context by agents later in it -- not because of anything that would
generalize to a real deployment, where agents act concurrently on their own
schedules.

This module keeps everything single-threaded (a plain Python heapq event
queue, no asyncio/threading) but decouples WHEN each agent decides from a
global lockstep loop: every agent runs on its own jittered cadence, and
reads peer state from a shared, timestamped bus rather than "whoever
happens to be earlier in a fixed list this same turn." Network ground-truth
evolution (NetworkState.step()) is its own independent event stream in the
same queue, not something re-run once per agent's turn.

Works against any topologies.base.Topology instance, not just diamond6 --
see topologies/ for the registry.

check_contagion/TrustScoreEngine/recommendation_risk/decide_verification/
gate_recommendation etc. are reused completely unchanged -- only the
scheduling and peer-visibility mechanism differs from Coordinator.
"""

import heapq
import itertools
import random
import time

from config import (
    TRUST_LOW_THRESHOLD,
    HALLUCINATION_NOISE_LEVEL,
    AGENT_DECISION_INTERVAL,
    AGENT_DECISION_JITTER_STD,
    AGENT_START_JITTER_STD,
    NETWORK_TICK_INTERVAL,
    ACCURACY_GRADE_DELAY,
    PEER_STALENESS_HORIZON,
)
from network.state import NetworkState
from network.observation import observation_to_source_text, pick_focus_router
from core.agent import Agent, format_peer_context
from core.perception_noise import corrupt_observation
from trust.grounding import GroundingChecker
from trust.entailment import EntailmentChecker
from trust.behavior import BehaviorTracker
from trust.consistency import ConsistencyChecker
from trust.trust_score import TrustScoreEngine
from trust.accuracy import AccuracyTracker, check_prediction
from trust.risk_score import recommendation_risk
from trust.cascade import check_contagion, gate_recommendation
from orchestrator.verification import decide_verification
from orchestrator.verifier_selection import select_verifiers, evidence_claim_verification
from orchestrator.coordinator import build_action, fault_router_for

NETWORK_TICK = "network_tick"
AGENT_DECISION = "agent_decision"
ACCURACY_GRADE = "accuracy_grade"

# priority is the tie-break when two events land on the exact same
# logical_time: ground truth always advances before any agent reads it,
# and a decision is always graded strictly after it was made.
_PRIORITY = {NETWORK_TICK: 0, AGENT_DECISION: 1, ACCURACY_GRADE: 2}


def turn_to_logical_time(turn: int) -> float:
    """The one place the "1 logical-time-unit ~= 1 old-style turn"
    convention lives -- used to translate a scenario's integer fault_turn
    into the DES clock."""
    return round(turn * AGENT_DECISION_INTERVAL, 6)


class EventCoordinator:
    def __init__(self, client, topology, mode: str = "world2_gated", seed: int = None):
        self.client = client
        self.topology = topology
        self.mode = mode  # "world1_undefended" | "world2_gated"
        self.state = NetworkState(topology, seed=seed) if seed is not None else NetworkState(topology)
        self.agents = {
            agent_id: Agent(
                agent_id,
                agent_id.replace("agent_", ""),
                client,
                role=topology.role_of(agent_id),
                observed_routers=topology.domain_of(agent_id),
            )
            for agent_id in topology.all_agent_ids()
        }
        self.grounding_checker = GroundingChecker(client)
        self.entailment_checker = EntailmentChecker(client)
        self.behavior_tracker = BehaviorTracker()
        self.accuracy_tracker = AccuracyTracker()
        self.engine = TrustScoreEngine(self.grounding_checker, self.entailment_checker, self.behavior_tracker)

        base_seed = seed or 0
        self.hallucination_rng = random.Random(base_seed + 999)
        # One independent jitter stream per agent, decorrelated from
        # construction/iteration order -- deterministic given the same seed.
        self.jitter_rngs = {
            agent_id: random.Random(f"{base_seed}:{agent_id}:jitter") for agent_id in topology.all_agent_ids()
        }

        self.bus = {}                 # agent_id -> {"decision", "action", "published_at", "decision_seq"}
        self.decision_counts = {agent_id: 0 for agent_id in topology.all_agent_ids()}
        self.pending_accuracy = {}    # (agent_id, decision_seq) -> {"decision", "health_before"}
        self.tick_count = 0
        self._seq = itertools.count()
        self.event_queue = []

    def _role(self, agent_id: str) -> str:
        return self.topology.role_of(agent_id)

    def _schedule(self, logical_time: float, event_type: str, agent_id: str = None, payload: dict = None):
        seq = next(self._seq)
        heapq.heappush(
            self.event_queue,
            (round(logical_time, 6), _PRIORITY[event_type], seq, event_type, agent_id, payload),
        )
        return seq

    def _peers_visible_to(self, agent_id: str, now: float) -> list:
        """Read-from-the-bus replacement for the old "agents earlier in this
        same turn's fixed list" logic. A peer counts as visible only if it
        has actually published something, and that publication isn't stale
        -- without the staleness filter a stalled agent's old opinion would
        be cited as "current" peer context forever."""
        if self._role(agent_id) == "orchestrator":
            candidates = self.topology.field_agent_ids()
        else:
            candidates = [
                p for p in self.topology.field_agent_ids() if p != agent_id and self.topology.overlap(agent_id, p)
            ]
        return [
            p for p in candidates
            if p in self.bus and (now - self.bus[p]["published_at"]) <= PEER_STALENESS_HORIZON
        ]

    def _handle_network_tick(self, now: float, scenario) -> dict:
        self.tick_count += 1
        fault_router = fault_router_for(self.topology, scenario.fault_agent_id)
        incident = None
        if scenario.fault_mode in ("hallucinating", "byzantine") and self.tick_count >= scenario.fault_turn and fault_router:
            incident = {
                "router_id": fault_router,
                "severity": self.tick_count - scenario.fault_turn + 1,
            }
        self.state.step(incident=incident)
        ledger_entry = {
            "turn": self.tick_count,
            "logical_time": now,
            "state": self.state.snapshot(),
            "health": self.state.health_score(),
        }
        if self.tick_count < scenario.num_turns:
            self._schedule(now + NETWORK_TICK_INTERVAL, NETWORK_TICK)
        return ledger_entry

    def _handle_accuracy_grade(self, agent_id: str, decision_seq: int):
        key = (agent_id, decision_seq)
        pending = self.pending_accuracy.pop(key, None)
        if pending is None:
            return
        current_obs = self.state.observation_for(agent_id)
        health_after = self.state.health_score(current_obs) if current_obs else 1.0
        outcome = check_prediction(pending["decision"], pending["health_before"], health_after)
        self.accuracy_tracker.update(agent_id, outcome["correct"])

    def _handle_agent_decision(self, now: float, agent_id: str, event_seq: int, scenario) -> dict:
        agent = self.agents[agent_id]
        self.decision_counts[agent_id] += 1
        decision_seq = self.decision_counts[agent_id]  # this agent's own Nth decision -- the new "turn"

        is_fault_turn = (
            scenario.fault_mode != "healthy"
            and agent_id == scenario.fault_agent_id
            and decision_seq >= scenario.fault_turn
        )

        if self._role(agent_id) == "orchestrator":
            clean_observation = {}
            domain_health_before = 1.0
        else:
            clean_observation = self.state.observation_for(agent_id)
            domain_health_before = self.state.health_score(clean_observation)

        overlapping_peers = self._peers_visible_to(agent_id, now)
        peer_context = [format_peer_context(p, p, self.bus[p]["decision"]) for p in overlapping_peers]
        peer_actions = {p: self.bus[p]["decision"].get("action") for p in overlapping_peers}

        observation = clean_observation
        if is_fault_turn and scenario.fault_mode == "hallucinating":
            fault_router = fault_router_for(self.topology, agent_id)
            if fault_router:
                observation = corrupt_observation(
                    clean_observation, [fault_router], HALLUCINATION_NOISE_LEVEL, self.hallucination_rng
                )
        adversarial = is_fault_turn and scenario.fault_mode == "byzantine"

        if self._role(agent_id) == "orchestrator":
            source_text = "\n\n".join(peer_context) if peer_context else "(no peer reports yet this turn)"
            observation_text = source_text
        else:
            source_text = observation_to_source_text(clean_observation)
            observation_text = observation_to_source_text(observation)

        t0 = time.time()
        decision = agent.decide(observation_text, peer_context=peer_context, adversarial=adversarial)

        focus_telemetry = pick_focus_router(clean_observation) if self._role(agent_id) != "orchestrator" else {}
        accuracy = self.accuracy_tracker.score(agent_id)

        consistency_checker = ConsistencyChecker(agent, self.client, adversarial=adversarial)
        result = self.engine.evaluate(
            agent_id, source_text, decision, focus_telemetry,
            consistency_checker=consistency_checker,
            telemetry_text=observation_text,
            peer_context=peer_context,
            accuracy=accuracy,
        )

        contagion = check_contagion(decision, peer_actions, overlapping_peers, result["detail"]["policy_alignment"])

        action = build_action(decision, agent_id, decision_seq, self.topology, clean_observation)
        if self._role(agent_id) == "orchestrator":
            current_health = self.state.health_score()
            risk_detail = {
                "risk_score": 0.0, "impact": 0.0, "blast_radius": 0.0, "reversibility": 0.0,
                "propagation_risk": 0.0, "before_health": current_health,
                "after_health": current_health, "affected_routers": [],
            }
        else:
            risk_detail = recommendation_risk(self.state, action, agent_id)

        verifiers_selected = []
        evidence_result = None
        if self.mode == "world1_undefended":
            verification = {"verification_action": "execute", "trust_bucket": "n/a", "risk_bucket": "n/a"}
            gate = {"propagate": True, "reason": "world1_undefended: no gating"}
        else:
            verification = decide_verification(result["trust_score"], risk_detail["risk_score"])
            if verification["verification_action"] != "execute":
                verifiers_selected = select_verifiers(action["target_routers"], agent_id, self.topology)
                evidence_result = evidence_claim_verification(action["target_routers"], self.state, decision)
            gate = gate_recommendation(
                decision, verification["verification_action"], contagion,
                result["trust_score"], TRUST_LOW_THRESHOLD,
            )

        if gate["propagate"] and self._role(agent_id) != "orchestrator":
            self.state.apply_action(action, committing=True)

        if self._role(agent_id) != "orchestrator":
            self.pending_accuracy[(agent_id, decision_seq)] = {
                "decision": decision, "health_before": domain_health_before,
            }
            self._schedule(now + ACCURACY_GRADE_DELAY, ACCURACY_GRADE, agent_id, {"decision_seq": decision_seq})

        self.bus[agent_id] = {
            "decision": decision, "action": action, "published_at": now, "decision_seq": decision_seq,
        }

        record = {
            "turn": decision_seq,
            "event_seq": event_seq,
            "logical_time": now,
            "agent_id": agent_id,
            "role": self.topology.role_of(agent_id),
            "observed_routers": self.topology.domain_of(agent_id),
            "fault_mode": scenario.fault_mode if is_fault_turn else "healthy",
            "decision": decision,
            "result": {k: v for k, v in result.items() if k != "detail"},
            "detail": result["detail"],
            "risk": risk_detail,
            "verification_action": verification["verification_action"],
            "verifiers_selected": verifiers_selected,
            "evidence_check": evidence_result,
            "contagion": contagion,
            "propagate": gate["propagate"],
            "gate_reason": gate["reason"],
            "peer_ids_seen": overlapping_peers,
            "peer_context_seen": peer_context,
            "elapsed_sec": round(time.time() - t0, 2),
        }

        sim_horizon = scenario.num_turns * AGENT_DECISION_INTERVAL
        jitter_rng = self.jitter_rngs[agent_id]
        step = max(0.1, AGENT_DECISION_INTERVAL + jitter_rng.gauss(0, AGENT_DECISION_JITTER_STD))
        next_time = now + step
        if next_time <= sim_horizon:
            self._schedule(next_time, AGENT_DECISION, agent_id)

        return record

    def run(self, scenario) -> dict:
        sim_horizon = scenario.num_turns * AGENT_DECISION_INTERVAL

        self._schedule(NETWORK_TICK_INTERVAL, NETWORK_TICK)
        for agent_id in self.topology.all_agent_ids():
            jitter_rng = self.jitter_rngs[agent_id]
            start = max(0.05, AGENT_DECISION_INTERVAL + jitter_rng.gauss(0, AGENT_START_JITTER_STD))
            self._schedule(start, AGENT_DECISION, agent_id)

        run_log = []
        ledger = []
        while self.event_queue and self.event_queue[0][0] <= sim_horizon:
            now, _priority, seq, event_type, agent_id, payload = heapq.heappop(self.event_queue)
            if event_type == NETWORK_TICK:
                ledger.append(self._handle_network_tick(now, scenario))
            elif event_type == AGENT_DECISION:
                run_log.append(self._handle_agent_decision(now, agent_id, seq, scenario))
            elif event_type == ACCURACY_GRADE:
                self._handle_accuracy_grade(agent_id, payload["decision_seq"])

        # run_log is already in strict chronological (logical_time, priority,
        # seq) order -- that's exactly the order heapq.heappop() produced it
        # in above. event_seq is a SCHEDULE-time id (assigned when an event
        # was queued, not when it ran) and must never be used to re-sort the
        # log -- doing so would scramble true execution order back into
        # schedule order, which is what an earlier version of this function
        # did by mistake.
        return {"run_log": run_log, "ledger": ledger}
