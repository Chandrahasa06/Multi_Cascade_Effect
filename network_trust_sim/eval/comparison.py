"""
Two-world comparison: same scenario, same seed, same NetworkState evolution,
run once through World 1 (undefended -- everything always propagates and
commits, mirroring the original main.py's behavior as the control group) and
once through World 2 (risk+trust-gated). The only fair way to attribute a
damage difference to the defense is holding everything else constant, which
is why orchestrator.coordinator.Coordinator implements both worlds via a
single `mode` flag rather than two separate pipelines.
"""


def network_damage(ledger: list) -> float:
    """Sum of (1 - health) across turns -- higher means more accumulated
    network unhealthiness over the whole run, not just the final state."""
    return sum(max(0.0, 1.0 - entry["health"]) for entry in ledger)


def damage_reduction(damage_baseline: float, damage_protected: float) -> float:
    if damage_baseline <= 0:
        return 0.0
    return 1.0 - (damage_protected / damage_baseline)


def detection_latency(run_log: list, fault_turn: int, fault_agent_id: str):
    flagged = [
        r["turn"] for r in run_log
        if r["agent_id"] == fault_agent_id and r["turn"] >= fault_turn and r["result"]["turn_flagged"]
    ]
    return (min(flagged) - fault_turn) if flagged else None


def false_positive_rate(run_log: list, fault_agent_id: str, fault_turn: int) -> float:
    """Fraction of turns that were NOT the actual fault (any agent before the
    fault started, or any agent other than the fault agent at any turn) that
    still got flagged -- the noise floor a real detector has to beat."""
    clean_turns = [
        r for r in run_log
        if not (r["agent_id"] == fault_agent_id and r["turn"] >= fault_turn)
    ]
    if not clean_turns:
        return 0.0
    flagged = sum(1 for r in clean_turns if r["result"]["turn_flagged"])
    return flagged / len(clean_turns)


def verification_overhead(run_log_w2: list) -> dict:
    verified = [r for r in run_log_w2 if r.get("verification_action") not in (None, "execute")]
    escalated = [r for r in run_log_w2 if r["result"]["escalated"]]
    return {
        "turns_requiring_verification": len(verified),
        "turns_escalated_tier2": len(escalated),
        "total_turns": len(run_log_w2),
    }
