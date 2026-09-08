"""
Phase A validation harness: runs the same seeded fault scenario once through
the original turn-based Coordinator, once through the discrete-event
EventCoordinator, and reports whether the qualitative detection story holds
under both -- NOT numerical equality, which isn't expected given the two
execution models schedule agents differently. The claim being checked is:
"does the fault agent still get flagged and gated, regardless of which
scheduler drove the run" -- not "do the two produce identical logs."
"""

from orchestrator.coordinator import Coordinator
from orchestrator.event_coordinator import EventCoordinator
from eval.comparison import network_damage


def _detection_summary(run_log: list, fault_agent_id: str, fault_turn: int) -> dict:
    fault_records = [r for r in run_log if r["agent_id"] == fault_agent_id and r["turn"] >= fault_turn]
    first_turn_flagged = next((r["turn"] for r in fault_records if r["result"]["turn_flagged"]), None)
    final_trust = fault_records[-1]["result"]["trust_score"] if fault_records else None
    return {
        "fault_turns_seen": len(fault_records),
        "turn_flagged_ever": any(r["result"]["turn_flagged"] for r in fault_records),
        "trust_flagged_ever": any(r["result"]["trust_flagged"] for r in fault_records),
        "gated_ever": any(not r["propagate"] for r in fault_records),
        "first_turn_flagged": first_turn_flagged,
        "final_trust_score": final_trust,
    }


def compare_execution_modes(scenario, client) -> dict:
    topology = scenario.resolve_topology()
    mode = "world1_undefended" if scenario.world == 1 else "world2_gated"

    turn_based = Coordinator(client, topology, mode=mode, seed=scenario.seed)
    result_turn = turn_based.run(scenario)

    event_based = EventCoordinator(client, topology, mode=mode, seed=scenario.seed)
    result_event = event_based.run(scenario)

    turn_detection = _detection_summary(result_turn["run_log"], scenario.fault_agent_id, scenario.fault_turn)
    event_detection = _detection_summary(result_event["run_log"], scenario.fault_agent_id, scenario.fault_turn)

    return {
        "turn_based": {
            "detection": turn_detection,
            "network_damage": network_damage(result_turn["ledger"]),
            "run_log": result_turn["run_log"],
        },
        "event_based": {
            "detection": event_detection,
            "network_damage": network_damage(result_event["ledger"]),
            "run_log": result_event["run_log"],
        },
        "qualitative_agreement": {
            "turn_flagged_ever_matches": turn_detection["turn_flagged_ever"] == event_detection["turn_flagged_ever"],
            "gated_ever_matches": turn_detection["gated_ever"] == event_detection["gated_ever"],
        },
    }
