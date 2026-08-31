"""
Single Coordinator class implementing BOTH worlds via a `mode` flag -- never
two separate pipelines -- so a measured damage difference between worlds can
be attributed to the defense itself, not to environment drift. Both worlds
run against an identically-seeded NetworkState and identical fault schedule.

Per turn: state.step() advances ground truth -> each agent (in an order that
recreates the intended contagion pathway: traffic -> routing -> safety ->
orchestrator, since agent_traffic/agent_routing overlap on R4 and
agent_routing/agent_safety overlap on R5) builds its observation, decides,
gets scored by the existing Tier-1/Tier-2 trust engine, gets a deterministic
Recommendation Risk Score, and -- in world2_gated -- gets a verification
action from the Trust x Risk matrix that can block it from propagating or
committing to the network at all.
"""

import random
import time

from config import TRUST_LOW_THRESHOLD, HALLUCINATION_NOISE_LEVEL, HALLUCINATION_TARGET_ROUTERS
from network.state import NetworkState
from network.topology import AGENT_DOMAINS, AGENT_ROLES, all_agent_ids, field_agent_ids, overlap
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

AGENT_ORDER = ["agent_traffic", "agent_routing", "agent_safety", "agent_orchestrator"]


def build_action(decision: dict, agent_id: str, turn: int, clean_observation: dict = None) -> dict:
    """Coordinator-constructed action object -- the agent itself keeps
    returning exactly {claims, action, justification, confidence}, unchanged.
    Target-router inference is a deliberate simplification: defaults to the
    issuing agent's full observed_routers, except reroute_traffic (picks the
    single most-congested router in-domain as the reroute source) and
    no_action_required (no routers touched, by definition)."""
    action_type = decision.get("action", "no_action_required")
    domain = list(AGENT_DOMAINS.get(agent_id, []))
    action = {"type": action_type, "issuing_agent": agent_id, "target_routers": domain, "turn": turn}

    if action_type == "no_action_required":
        action["target_routers"] = []
    elif action_type == "reroute_traffic" and domain:
        if clean_observation:
            source = max(domain, key=lambda r: clean_observation.get(r, {}).get("utilization_pct", 0))
        else:
            source = domain[0]
        action["target_routers"] = [source]

    return action


class Coordinator:
    def __init__(self, client, mode: str = "world2_gated", seed: int = None):
        self.client = client
        self.mode = mode  # "world1_undefended" | "world2_gated"
        self.state = NetworkState(seed=seed) if seed is not None else NetworkState()
        self.agents = {
            agent_id: Agent(
                agent_id,
                agent_id.replace("agent_", ""),
                client,
                role=AGENT_ROLES[agent_id],
                observed_routers=AGENT_DOMAINS[agent_id],
            )
            for agent_id in all_agent_ids()
        }
        self.grounding_checker = GroundingChecker(client)
        self.entailment_checker = EntailmentChecker(client)
        self.behavior_tracker = BehaviorTracker()
        self.accuracy_tracker = AccuracyTracker()
        self.engine = TrustScoreEngine(self.grounding_checker, self.entailment_checker, self.behavior_tracker)
        self.pending_accuracy = {}   # agent_id -> {"decision", "health_before"}
        self.rng = random.Random((seed or 0) + 999)

    def _resolve_pending_accuracy(self):
        """Called at the top of each turn (after state.step()), so a pending
        decision from last turn is graded against ground truth one turn
        later -- exactly the lag the accuracy signal is meant to measure."""
        for agent_id, pending in list(self.pending_accuracy.items()):
            current_obs = self.state.observation_for(agent_id)
            health_after = self.state.health_score(current_obs) if current_obs else 1.0
            outcome = check_prediction(pending["decision"], pending["health_before"], health_after)
            self.accuracy_tracker.update(agent_id, outcome["correct"])
        self.pending_accuracy = {}

    def run_turn(self, turn: int, scenario) -> list:
        incident = None
        if scenario.fault_mode in ("hallucinating", "byzantine") and turn >= scenario.fault_turn:
            incident = {
                "router_id": HALLUCINATION_TARGET_ROUTERS[0],
                "severity": turn - scenario.fault_turn + 1,
            }
        self.state.step(incident=incident)
        self._resolve_pending_accuracy()

        decisions = {}
        peer_actions = {}
        turn_records = []

        for agent_id in AGENT_ORDER:
            agent = self.agents[agent_id]
            is_fault_turn = (
                scenario.fault_mode != "healthy"
                and agent_id == scenario.fault_agent_id
                and turn >= scenario.fault_turn
            )

            if agent_id == "agent_orchestrator":
                clean_observation = {}
                domain_health_before = 1.0
                overlapping_peers = [p for p in field_agent_ids() if p in decisions]
            else:
                clean_observation = self.state.observation_for(agent_id)
                domain_health_before = self.state.health_score(clean_observation)
                overlapping_peers = [
                    p for p in field_agent_ids()
                    if p != agent_id and overlap(agent_id, p) and p in decisions
                ]

            peer_context = [format_peer_context(p, p, decisions[p]) for p in overlapping_peers]

            observation = clean_observation
            if is_fault_turn and scenario.fault_mode == "hallucinating":
                observation = corrupt_observation(
                    clean_observation, HALLUCINATION_TARGET_ROUTERS, HALLUCINATION_NOISE_LEVEL, self.rng
                )
            adversarial = is_fault_turn and scenario.fault_mode == "byzantine"

            if agent_id == "agent_orchestrator":
                source_text = "\n\n".join(peer_context) if peer_context else "(no peer reports yet this turn)"
                observation_text = source_text
            else:
                source_text = observation_to_source_text(clean_observation)
                observation_text = observation_to_source_text(observation)

            t0 = time.time()
            decision = agent.decide(observation_text, peer_context=peer_context, adversarial=adversarial)

            focus_telemetry = pick_focus_router(clean_observation) if agent_id != "agent_orchestrator" else {}
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

            action = build_action(decision, agent_id, turn, clean_observation)
            if agent_id == "agent_orchestrator":
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
                    verifiers_selected = select_verifiers(action["target_routers"], agent_id)
                    evidence_result = evidence_claim_verification(action["target_routers"], self.state, decision)
                gate = gate_recommendation(
                    decision, verification["verification_action"], contagion,
                    result["trust_score"], TRUST_LOW_THRESHOLD,
                )

            if gate["propagate"] and agent_id != "agent_orchestrator":
                self.state.apply_action(action, committing=True)

            if agent_id != "agent_orchestrator":
                self.pending_accuracy[agent_id] = {"decision": decision, "health_before": domain_health_before}

            decisions[agent_id] = decision
            peer_actions[agent_id] = decision.get("action")

            turn_records.append({
                "turn": turn,
                "agent_id": agent_id,
                "role": AGENT_ROLES[agent_id],
                "observed_routers": AGENT_DOMAINS[agent_id],
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
                # Which overlapping peers had already decided this turn, and the
                # exact peer-note text this agent was shown -- audit the "what
                # did the swarm say about this" wiring here, and cross-reference
                # peer_ids_seen against those peers' OWN turn records (same
                # `turn`) to see how they subsequently reacted.
                "peer_ids_seen": overlapping_peers,
                "peer_context_seen": peer_context,
                "elapsed_sec": round(time.time() - t0, 2),
            })

        return turn_records

    def run(self, scenario) -> dict:
        run_log = []
        ledger = []
        for turn in range(1, scenario.num_turns + 1):
            turn_records = self.run_turn(turn, scenario)
            run_log.extend(turn_records)
            ledger.append({"turn": turn, "state": self.state.snapshot(), "health": self.state.health_score()})
        return {"run_log": run_log, "ledger": ledger}
