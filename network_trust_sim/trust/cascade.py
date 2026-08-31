"""
check_contagion() keeps its original core logic -- "matches a peer's action"
+ "own ground truth disagrees" -- just generalized from one fixed neighbor to
a set of overlapping peers, since agents no longer sit in a fixed linear
chain. That combination is what agentic contagion looks like, as opposed to
two agents independently and correctly agreeing that the same action is
warranted.

gate_recommendation() is what's new: it's the function that can actually stop
a recommendation from propagating (becoming another agent's peer_context, or
being committed to the network), instead of only flagging it after the fact
in the log the way the original diagnostic-only version did.
"""


def check_contagion(decision: dict, peer_actions: dict, overlapping_peer_ids: list, policy_alignment: float) -> dict:
    """peer_actions: {agent_id: action} for this turn's other decisions.
    overlapping_peer_ids: agents whose observed domain overlaps this agent's
    -- only agreement with THESE peers counts as potential contagion, not
    coincidental agreement with an unrelated agent."""
    own_action = decision.get("action")
    for peer_id in overlapping_peer_ids:
        peer_action = peer_actions.get(peer_id)
        if peer_action is not None and peer_action == own_action and policy_alignment == 0.0:
            return {"suspected": True, "matched_peer_id": peer_id}
    return {"suspected": False, "matched_peer_id": None}


def gate_recommendation(
    decision: dict, verification_action: str, contagion: dict, trust_score: float, trust_low_threshold: float
) -> dict:
    """The actual containment step. A recommendation is blocked from
    propagating when verification says to reject/quarantine it, OR when it
    looks like contagion (copying a peer while failing its own policy check)
    AND the issuing agent's trust has already dropped -- a single
    coincidental match on a healthy, trusted agent should NOT be blocked."""
    if verification_action == "reject_quarantine":
        return {"propagate": False, "reason": "risk_trust_gate: reject_quarantine"}
    if contagion.get("suspected") and trust_score < trust_low_threshold:
        return {"propagate": False, "reason": f"contagion_suspected_from_{contagion.get('matched_peer_id')}"}
    return {"propagate": True, "reason": "ok"}
