"""
Post-hoc, run-log-level contagion measurement -- no live-loop changes needed.
This is the actual "did the bad call spread" evidence: correlated_action_rate
checks whether an overlapping peer's OWN action starts matching a flagged
agent's action more often after the flagged turn than before, gated on real
topology overlap so coincidental agreement on a genuinely shared symptom
doesn't get counted as contagion.
"""

from network.topology import overlap, field_agent_ids


def correlated_action_rate(run_log: list, source_agent_id: str, target_agent_id: str, window: int = 3) -> dict:
    if not overlap(source_agent_id, target_agent_id):
        return {
            "rate": 0.0,
            "reason": "no domain overlap -- not a meaningful contagion pathway",
            "matches": 0,
            "checked": 0,
        }

    source_turns = {r["turn"]: r for r in run_log if r["agent_id"] == source_agent_id}
    target_turns = {r["turn"]: r for r in run_log if r["agent_id"] == target_agent_id}

    flagged_turns = sorted(t for t, r in source_turns.items() if r["result"]["turn_flagged"])
    if not flagged_turns:
        return {"rate": 0.0, "reason": "source agent was never flagged", "matches": 0, "checked": 0}

    checked = 0
    matches = 0
    for flagged_turn in flagged_turns:
        for t in range(flagged_turn, flagged_turn + window + 1):
            if t in target_turns and t in source_turns:
                checked += 1
                if target_turns[t]["decision"].get("action") == source_turns[t]["decision"].get("action"):
                    matches += 1

    rate = matches / checked if checked else 0.0
    return {"rate": rate, "matches": matches, "checked": checked, "flagged_turns": flagged_turns}


def contagion_report(run_log: list) -> dict:
    agents = field_agent_ids()
    report = {}
    for source in agents:
        for target in agents:
            if source == target:
                continue
            report[f"{source}->{target}"] = correlated_action_rate(run_log, source, target)
    return report
