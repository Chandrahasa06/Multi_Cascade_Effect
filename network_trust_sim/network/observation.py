"""
Bridges NetworkState's multi-router observation dicts to the per-router text
format agents/graders expect, and reduces a multi-router observation down to
a single flat dict for legacy single-router consumers like
trust.policy_rules.policy_alignment(), which was written for the old
single-segment telemetry shape and doesn't need to change.
"""

from core.telemetry import telemetry_to_source_text


def observation_to_source_text(observation: dict) -> str:
    """observation: {router_id: {field: value, ...}, ...} -> multi-block
    text, one block per router, reusing telemetry_to_source_text() per
    router unchanged."""
    blocks = []
    for router_id, fields in observation.items():
        block = f"router: {router_id}\n" + telemetry_to_source_text(fields)
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else "(no router state visible to this agent)"


def pick_focus_router(observation: dict) -> dict:
    """Reduce a multi-router observation to the single worst-affected
    router's flat field dict (highest utilization), for callers that still
    expect one flat telemetry-shaped dict -- specifically
    trust.policy_rules.policy_alignment(), which is reused with ZERO code
    changes and expects these exact key names."""
    if not observation:
        return {}
    worst_id = max(observation, key=lambda r: observation[r].get("utilization_pct", 0))
    f = observation[worst_id]
    return {
        "segment": worst_id,
        "bandwidth_utilization_pct": f.get("utilization_pct", 0),
        "error_rate_pct": f.get("error_rate_pct", 0),
        "packet_loss_pct": f.get("packet_loss_pct", 0),
        "avg_latency_ms": f.get("latency_ms", 0),
    }
