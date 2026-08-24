import json
import os
import time
from datetime import datetime

from config import NUM_AGENTS, NUM_TURNS, COMPROMISED_AGENT_ID, INJECTION_TURN, OLLAMA_MODEL
from core.ollama_client import OllamaClient
from core.telemetry import SEGMENT_NAMES, generate_telemetry, telemetry_to_source_text
from core.injection import inject
from core.agent import Agent
from trust.grounding import GroundingChecker
from trust.entailment import EntailmentChecker
from trust.behavior import BehaviorTracker
from trust.consistency import ConsistencyChecker
from trust.trust_score import TrustScoreEngine

LOG_DIR = "logs"


def build_agents(client):
    agents = {}
    for i in range(NUM_AGENTS):
        agent_id = f"agent_{i + 1}"
        segment = SEGMENT_NAMES[i % len(SEGMENT_NAMES)]
        agents[agent_id] = Agent(agent_id, segment, client)
    return agents


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
        f"Compromised: {COMPROMISED_AGENT_ID} (from turn {INJECTION_TURN})\n"
    )
    header = f"{'Turn':<5}{'Agent':<10}{'Action':<28}{'Tier1':<8}{'Escl':<6}{'Trust':<8}{'Flags'}"
    print(header)
    print("-" * len(header))

    for turn in range(1, NUM_TURNS + 1):
        for agent_id, agent in agents.items():
            is_compromised_turn = (agent_id == COMPROMISED_AGENT_ID) and (turn >= INJECTION_TURN)
            incident_severity = (turn - INJECTION_TURN + 1) if is_compromised_turn else 0

            telemetry = generate_telemetry(agent_id, agent.segment, turn, incident_severity=incident_severity)
            source_text = telemetry_to_source_text(telemetry)

            telemetry_seen_by_agent = inject(source_text) if is_compromised_turn else source_text

            t0 = time.time()
            decision = agent.decide(telemetry_seen_by_agent)

            consistency_checker = ConsistencyChecker(agent, client)
            result = engine.evaluate(
                agent_id,
                source_text,
                decision,
                telemetry,
                consistency_checker=consistency_checker,
                telemetry_text=telemetry_seen_by_agent,
            )

            flags = []
            if result["turn_flagged"]:
                flags.append("SUSPECT-THIS-TURN")
            if result["trust_flagged"]:
                flags.append("SUSTAINED-DISTRUST")
            flag_str = " | ".join(flags)

            print(
                f"{turn:<5}{agent_id:<10}{decision.get('action', ''):<28}"
                f"{result['tier1_score']:.2f}    {'Y' if result['escalated'] else 'N':<6}"
                f"{result['trust_score']:.2f}    {flag_str}  ({time.time() - t0:.1f}s)",
                flush=True
            )

            run_log.append(
                {
                    "turn": turn,
                    "agent_id": agent_id,
                    "segment": agent.segment,
                    "injected": is_compromised_turn,
                    "decision": decision,
                    "result": {k: v for k, v in result.items() if k != "detail"},
                    "detail": result["detail"],
                }
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(LOG_DIR, f"run_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"\nFull run log saved to {out_path}")


if __name__ == "__main__":
    run()