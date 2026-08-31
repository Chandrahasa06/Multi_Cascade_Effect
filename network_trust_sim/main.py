import json
import os
import time
from datetime import datetime

from config import NUM_AGENTS, NUM_TURNS, COMPROMISED_AGENT_ID, INJECTION_TURN, OLLAMA_MODEL
from core.ollama_client import OllamaClient
from core.telemetry import SEGMENT_NAMES, generate_telemetry, telemetry_to_source_text
from core.injection import inject
from core.agent import Agent, format_peer_context
from trust.grounding import GroundingChecker
from trust.entailment import EntailmentChecker
from trust.behavior import BehaviorTracker
from trust.consistency import ConsistencyChecker
from trust.trust_score import TrustScoreEngine
from trust.cascade import check_contagion
from trust.contagion_metrics import contagion_report
from scenarios import from_args
from orchestrator.coordinator import Coordinator
from eval.comparison import (
    network_damage, damage_reduction, detection_latency, false_positive_rate, verification_overhead,
)

LOG_DIR = "logs"


# ---------------------------------------------------------------------------
# Legacy pipeline: the original 5-segment linear-chain flow. Kept working
# behind --legacy, untouched apart from adapting to trust/cascade.py's
# generalized check_contagion() signature (peer-set instead of one fixed
# neighbor -- a chain of one neighbor is just a peer-set of size <= 1).
# ---------------------------------------------------------------------------

def build_agents_legacy(client):
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


def run_legacy():
    os.makedirs(LOG_DIR, exist_ok=True)
    client = OllamaClient()
    agents = build_agents_legacy(client)

    grounding_checker = GroundingChecker(client)
    entailment_checker = EntailmentChecker(client)
    behavior_tracker = BehaviorTracker()
    engine = TrustScoreEngine(grounding_checker, entailment_checker, behavior_tracker)

    run_log = []

    print(
        f"[legacy] Model: {OLLAMA_MODEL} | Agents: {list(agents)} | "
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
            decision = agent.decide(telemetry_seen_by_agent, peer_context=neighbor_context)

            consistency_checker = ConsistencyChecker(agent, client)
            result = engine.evaluate(
                agent_id, source_text, decision, telemetry,
                consistency_checker=consistency_checker,
                telemetry_text=telemetry_seen_by_agent,
                peer_context=neighbor_context,
            )

            contagion = check_contagion(
                decision,
                {neighbor_id: neighbor_action} if neighbor_id else {},
                [neighbor_id] if neighbor_id else [],
                result["detail"]["policy_alignment"],
            )

            print_result_block(
                turn, agent, decision, telemetry, result, contagion["suspected"], neighbor_id,
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
                    "contagion_suspected": contagion["suspected"],
                    "decision": decision,
                    "result": {k: v for k, v in result.items() if k != "detail"},
                    "detail": result["detail"],
                }
            )

            # hand this agent's decision forward to the next agent in the chain
            neighbor_context = format_peer_context(agent_id, agent.segment, decision)
            neighbor_action = decision.get("action")
            neighbor_id = agent_id

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(LOG_DIR, f"run_legacy_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"\nFull run log saved to {out_path}")


# ---------------------------------------------------------------------------
# New pipeline: 6-router network simulator + 4-agent Coordinator, world1
# (undefended baseline) vs world2 (risk+trust-gated orchestration).
# ---------------------------------------------------------------------------

def print_new_result_block(record):
    flags = []
    if record["result"]["turn_flagged"]:
        flags.append("SUSPECT-THIS-TURN")
    if record["result"]["trust_flagged"]:
        flags.append("SUSTAINED-DISTRUST")
    if record["contagion"]["suspected"]:
        flags.append(f"CASCADE-FROM-{record['contagion']['matched_peer_id']}")
    if not record["propagate"]:
        flags.append(f"BLOCKED({record['gate_reason']})")
    flag_str = (" [" + " | ".join(flags) + "]") if flags else ""

    decision = record["decision"]
    print(
        f"\n[Turn {record['turn']}] {record['agent_id']} ({record['role']}, routers={record['observed_routers']})"
        f"  ({record['elapsed_sec']:.1f}s){flag_str}"
    )
    if record.get("peer_ids_seen"):
        print(f"  saw peer notes from: {record['peer_ids_seen']}")
        for note in record.get("peer_context_seen") or []:
            print(f"    | {note.splitlines()[0]}  {' '.join(note.splitlines()[1:])}"[:180])
    print(f"  action: {decision.get('action')}  (confidence {decision.get('confidence')})")
    print(f"  justification: \"{(decision.get('justification') or '').strip()[:180]}\"")
    print(
        f"  trust={record['result']['trust_score']:.2f}  risk={record['risk']['risk_score']:.2f} "
        f"(impact={record['risk']['impact']:.2f} blast={record['risk']['blast_radius']:.2f})  "
        f"verification={record['verification_action']}"
    )
    if record.get("verifiers_selected"):
        print(f"  verifiers selected: {record['verifiers_selected']}  evidence: {record['evidence_check']}")


def run_scenario(scenario, client, verbose=True):
    coordinator = Coordinator(
        client,
        mode="world1_undefended" if scenario.world == 1 else "world2_gated",
        seed=scenario.seed,
    )
    print(
        f"World {scenario.world} | fault={scenario.fault_mode} "
        f"(agent={scenario.fault_agent_id}, from turn {scenario.fault_turn}) | turns={scenario.num_turns}"
    )
    result = coordinator.run(scenario)
    if verbose:
        for record in result["run_log"]:
            print_new_result_block(record)
    return result


def save_log(payload, prefix="run"):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(LOG_DIR, f"{prefix}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nFull run log saved to {out_path}")
    return out_path


def run_comparison(scenario, client):
    """Same fault/seed, World 1 then World 2 -- reports damage reduction and
    the other headline metrics. A single Coordinator class runs both worlds
    via its `mode` flag, so this is a fair comparison, not two pipelines."""
    base_kwargs = dict(scenario.__dict__)
    scenario_w1 = type(scenario)(**{**base_kwargs, "world": 1})
    scenario_w2 = type(scenario)(**{**base_kwargs, "world": 2})

    print("=== World 1 (undefended baseline) ===")
    result_w1 = run_scenario(scenario_w1, client, verbose=False)
    print("=== World 2 (risk+trust-gated) ===")
    result_w2 = run_scenario(scenario_w2, client, verbose=False)

    damage_w1 = network_damage(result_w1["ledger"])
    damage_w2 = network_damage(result_w2["ledger"])

    summary = {
        "damage_world1": damage_w1,
        "damage_world2": damage_w2,
        "damage_reduction": damage_reduction(damage_w1, damage_w2),
        "detection_latency_turns": detection_latency(result_w2["run_log"], scenario.fault_turn, scenario.fault_agent_id),
        "false_positive_rate_world2": false_positive_rate(result_w2["run_log"], scenario.fault_agent_id, scenario.fault_turn),
        "verification_overhead": verification_overhead(result_w2["run_log"]),
        "contagion_world1": contagion_report(result_w1["run_log"]),
        "contagion_world2": contagion_report(result_w2["run_log"]),
    }

    print("\n=== Two-world comparison ===")
    print(json.dumps(summary, indent=2))

    save_log({"world1": result_w1["run_log"], "world2": result_w2["run_log"], "summary": summary}, prefix="compare")
    return summary


def run():
    scenario = from_args()

    if scenario.legacy:
        run_legacy()
        return

    client = OllamaClient()

    if scenario.compare:
        run_comparison(scenario, client)
        return

    result = run_scenario(scenario, client, verbose=True)
    save_log(result["run_log"], prefix=f"run_world{scenario.world}_{scenario.fault_mode}")


if __name__ == "__main__":
    run()
