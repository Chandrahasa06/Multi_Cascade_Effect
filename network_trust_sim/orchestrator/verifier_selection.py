"""
Evidence-ownership verifier selection: when an agent makes a claim about a
router, only agents that can INDEPENDENTLY observe that router should be
asked to verify it -- not an arbitrary/all-peer vote. This is deliberately
split into two different checks, not merged into one: does a peer's own
telemetry agree with the claim (evidence-claim verification), versus does the
proposed action actually help per the ground-truth simulator (action-impact
verification -- see trust.risk_score.predicted_impact, which already runs the
same counterfactual simulation this would need, so it isn't duplicated here).
"""

def select_verifiers(claim_router_ids: list, issuing_agent_id: str, topology) -> list:
    verifiers = set()
    for router_id in claim_router_ids:
        verifiers.update(topology.observers_of(router_id))
    verifiers.discard(issuing_agent_id)
    # An orchestrator-role agent has an empty router domain by construction --
    # never a legitimate verifier. Role-based, not a hardcoded literal id, so
    # this keeps working across any topology.
    verifiers -= {a for a in verifiers if topology.role_of(a) == "orchestrator"}
    return sorted(verifiers)


def evidence_claim_verification(claim_router_ids: list, state, issuing_decision: dict) -> dict:
    """Coarse, deterministic sanity check: if the agent escalated to a
    corrective action, at least one of its claimed routers should genuinely
    be under load per ground truth; if it claimed no action was needed, none
    of its claimed routers should be critically loaded. This is what an
    independent verifying peer's own telemetry would show -- a cheap
    deterministic stand-in for "ask a peer to double-check," not a second
    LLM-judge call per verifier."""
    action = issuing_decision.get("action", "no_action_required")
    escalating = action != "no_action_required"
    checked = [r for r in claim_router_ids if r in state.routers]
    if not checked:
        return {"evidence_agreement": 1.0, "checked_routers": [], "high_load_routers": []}

    high_load_routers = [r for r in checked if state.routers[r]["utilization_pct"] >= 70.0]
    agrees = (len(high_load_routers) > 0) if escalating else (len(high_load_routers) == 0)
    return {
        "evidence_agreement": 1.0 if agrees else 0.0,
        "checked_routers": checked,
        "high_load_routers": high_load_routers,
    }
