import json
import os
import time
from datetime import datetime

from config import NUM_AGENTS, NUM_TURNS, COMPROMISED_AGENT_ID, INJECTION_TURN, OLLAMA_MODEL
from core.ollama_client import OllamaClient
from core.telemetry import SEGMENT_NAMES, generate_telemetry, telemetry_to_source_text
from core.injection import inject
from core.agent import Agent, format_neighbor_context
from trust.grounding import GroundingChecker
from trust.entailment import EntailmentChecker
from trust.behavior import BehaviorTracker
from trust.consistency import ConsistencyChecker
from trust.trust_score import TrustScoreEngine
from trust.cascade import check_contagion

LOG_DIR = "logs"


def build_agents(client):
    """Agents are built in topology order (agent_1 -> agent_2 -> ... -> agent_5,
    matching SEGMENT_NAMES). This order IS the cascade chain: each turn, every
    agent after the first sees its immediate upstream neighbor's most recent
    decision as context, so a compromised agent's output has an actual pathway
    to influence the next agent -- otherwise there is nothing for a "cascade"
    or "contagion" experiment to even test."""
    agents = {}
    for i in range(NUM_AGENTS):
        agent_id = f"agent_{i + 1}"
        segment = SEGMENT_NAMES[i % len(SEGMENT_NAMES)]
        agents[agent_id] = Agent(agent_id, segment, client)
    return agents


def _truncate(text, n=180):
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def print_result_block(turn, agent, decision, telemetry, result, contagion_suspected, neighbor_id, elapsed):
    detail = result["detail"]
    grounding = detail["grounding"]["grounding_rate"]
    entailment = detail["entailment"]
    behavior_dev = detail["behavior"]["behavior_deviation"]
    policy = detail["policy_alignment"]

    flags = []
    if result["turn_flagged"]:
        flags.append("SUSPECT-THIS-TURN")
    if result["trust_flagged"]:
        flags.append("SUSTAINED-DISTRUST")
    if contagion_suspected:
        flags.append(f"CASCADE-FROM-{neighbor_id}")
    flag_str = (" [" + " | ".join(flags) + "]") if flags else ""

    print(
        f"\n[Turn {turn}] {agent.agent_id} ({agent.segment})  "
        f"util={telemetry['bandwidth_utilization_pct']}%  errors={telemetry['error_rate_pct']}%  "
        f"({elapsed:.1f}s){flag_str}"
    )
    print(f"  action: {decision.get('action')}  (confidence {decision.get('confidence')})")
    print(f"  justification: \"{_truncate(decision.get('justification'))}\"")
    print(
        f"  scores: tier1={result['tier1_score']:.2f} "
        f"[grounding={grounding:.2f} entailment={entailment['score']:.2f}({entailment['verdict']}) "
        f"policy={policy:.2f} behavior_dev={behavior_dev:.2f}]  "
        f"escalated={'Y' if result['escalated'] else 'N'}  trust={result['trust_score']:.2f}"
    )


def run():
    os.makedirs(LOG_DIR, exist_ok=True)
    client = OllamaClient()
    agents = build_agents(client)

    grounding_checker = GroundingChecker(client)
    entailment_checker = EntailmentChecker(client)
    behavior_tracker = BehaviorTracker()
    engine = TrustScoreEngine(grounding_checker, entailment_checker, behavior_tracker)

    run_log = []

    print(
        f"Model: {OLLAMA_MODEL} | Agents: {list(agents)} | "
        f"Compromised: {COMPROMISED_AGENT_ID} (from turn {INJECTION_TURN})"
    )

    for turn in range(1, NUM_TURNS + 1):
        neighbor_context = None  # reset each turn -- the first agent in the chain has no upstream neighbor
        neighbor_action = None
        neighbor_id = None

        for agent_id, agent in agents.items():
            is_compromised_turn = (agent_id == COMPROMISED_AGENT_ID) and (turn >= INJECTION_TURN)
            incident_severity = (turn - INJECTION_TURN + 1) if is_compromised_turn else 0

            telemetry = generate_telemetry(agent_id, agent.segment, turn, incident_severity=incident_severity)
            source_text = telemetry_to_source_text(telemetry)

            telemetry_seen_by_agent = inject(source_text) if is_compromised_turn else source_text

            t0 = time.time()
            decision = agent.decide(telemetry_seen_by_agent, neighbor_context=neighbor_context)

            consistency_checker = ConsistencyChecker(agent, client)
            result = engine.evaluate(
                agent_id,
                source_text,
                decision,
                telemetry,
                consistency_checker=consistency_checker,
                telemetry_text=telemetry_seen_by_agent,
                neighbor_context=neighbor_context,
            )

            contagion_suspected = check_contagion(
                decision, neighbor_action, result["detail"]["policy_alignment"]
            )

            print_result_block(
                turn, agent, decision, telemetry, result, contagion_suspected, neighbor_id,
                elapsed=time.time() - t0,
            )

            run_log.append(
                {
                    "turn": turn,
                    "agent_id": agent_id,
                    "segment": agent.segment,
                    "injected": is_compromised_turn,
                    "neighbor_id": neighbor_id,
                    "neighbor_context": neighbor_context,  # exact text this agent was shown -- audit the cascade wiring here
                    "contagion_suspected": contagion_suspected,
                    "decision": decision,
                    "result": {k: v for k, v in result.items() if k != "detail"},
                    "detail": result["detail"],
                }
            )

            # hand this agent's decision forward to the next agent in the chain
            neighbor_context = format_neighbor_context(agent_id, agent.segment, decision)
            neighbor_action = decision.get("action")
            neighbor_id = agent_id

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(LOG_DIR, f"run_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"\nFull run log saved to {out_path}")


if __name__ == "__main__":
    run()