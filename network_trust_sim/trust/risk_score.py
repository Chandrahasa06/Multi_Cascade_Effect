"""
Recommendation Risk Score -- "how dangerous would it be if THIS recommendation
is wrong" -- kept deliberately separate from Agent Trust ("how reliable has
this agent been"). Deterministic only: no LLM client is imported anywhere in
this file. Letting an LLM judge "how risky is this recommendation" would just
create a second Byzantine-shaped attack surface, exactly the thing this whole
project is trying to guard against -- the attacker could learn to fool the
risk judge the same way it fools grounding/entailment.
"""

from config import W_RISK_IMPACT, W_RISK_BLAST, W_RISK_REVERSIBILITY, REVERSIBILITY_PENALTY


def blast_radius(affected_routers: list, router_ids: list) -> float:
    if not affected_routers or not router_ids:
        return 0.0
    return len(set(affected_routers)) / len(router_ids)


def predicted_impact(state, action: dict) -> dict:
    """Counterfactual: how much would network health DROP if this action were
    taken but turns out to be unnecessary/wrong. Scoped deliberately to this
    one direction (cost of acting) rather than also pricing cost-of-inaction,
    which would need a second simulation call and isn't needed for the risk
    score's purpose here -- flag as a future extension, not silently folded
    in."""
    result = state.apply_action(action, committing=False)
    drop = max(0.0, result["before_health"] - result["after_health"])
    return {
        "impact": min(1.0, drop),
        "before_health": result["before_health"],
        "after_health": result["after_health"],
        "affected_routers": result["affected_routers"],
    }


def reversibility_penalty(action_type: str) -> float:
    return REVERSIBILITY_PENALTY.get(action_type, 0.5)


def propagation_risk(action: dict, agent_id: str, topology) -> float:
    """How much of the swarm would plausibly see this recommendation as peer
    context this turn -- fraction of OTHER agents whose observed domain
    overlaps this agent's. Reported alongside the risk score as a diagnostic,
    not folded into the weighted sum (kept to impact/blast/reversibility for
    this first pass, per the minimal-signal build order)."""
    peers = [a for a in topology.all_agent_ids() if a != agent_id]
    if not peers:
        return 0.0
    reach = sum(1 for peer in peers if topology.overlap(agent_id, peer))
    return reach / len(peers)


def recommendation_risk(state, action: dict, agent_id: str) -> dict:
    impact_detail = predicted_impact(state, action)
    blast = blast_radius(impact_detail["affected_routers"], state.topology.router_ids)
    reversibility = reversibility_penalty(action.get("type", "no_action_required"))
    propagation = propagation_risk(action, agent_id, state.topology)

    risk_score = (
        W_RISK_IMPACT * impact_detail["impact"]
        + W_RISK_BLAST * blast
        + W_RISK_REVERSIBILITY * reversibility
    )
    risk_score = max(0.0, min(1.0, risk_score))

    return {
        "risk_score": risk_score,
        "impact": impact_detail["impact"],
        "blast_radius": blast,
        "reversibility": reversibility,
        "propagation_risk": propagation,
        "before_health": impact_detail["before_health"],
        "after_health": impact_detail["after_health"],
        "affected_routers": impact_detail["affected_routers"],
    }
