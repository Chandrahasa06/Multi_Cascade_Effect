"""
Deterministic per-action-type effect model over router state. Every delta
here is illustrative/uncalibrated -- exactly like the trust-score weights in
config.py -- not a calibrated network-engineering model. What matters for the
experiment is that effects are DETERMINISTIC and REPEATABLE (same action on
the same state always produces the same before/after health delta), so
counterfactual risk scoring and the two-world comparison are reproducible.

"Affected routers" is defined as exactly what this function mutates, and is
returned as a side-channel -- that keeps blast-radius calculations (and
anything else that needs "what did this action touch") using one single
consistent definition instead of two possibly-diverging ones.
"""

ACTION_TYPES = {
    "no_action_required",
    "increase_qos_priority",
    "trigger_capacity_upgrade",
    "reroute_traffic",
    "throttle_low_priority_traffic",
}


def _recompute_derived(r: dict) -> None:
    r["throughput_gbps"] = round(r["capacity_gbps"] * r["utilization_pct"] / 100, 1)
    r["congestion_level"] = round(0.5 * r["congestion_level"] + 0.5 * r["utilization_pct"], 1)
    r["routing_cost"] = round(1.0 + r["congestion_level"] / 100, 2)


def apply_effect(routers: dict, action: dict, topology) -> tuple:
    """Mutates `routers` in place and returns (routers, affected_router_ids).
    `routers` is expected to already be a working copy the caller owns --
    this function never decides committing vs. counterfactual, the caller
    (NetworkState.apply_action) does. `topology` supplies neighbors() for
    reroute_traffic -- this function doesn't assume any particular network
    shape."""
    action_type = action.get("type", "no_action_required")
    targets = [r for r in action.get("target_routers", []) if r in routers]
    affected = []

    if action_type == "no_action_required" or not targets:
        return routers, affected

    if action_type == "increase_qos_priority":
        for rid in targets:
            r = routers[rid]
            r["latency_ms"] = round(r["latency_ms"] * 0.85, 1)
            r["packet_loss_pct"] = round(r["packet_loss_pct"] * 0.80, 3)
            _recompute_derived(r)
            affected.append(rid)

    elif action_type == "throttle_low_priority_traffic":
        for rid in targets:
            r = routers[rid]
            r["utilization_pct"] = round(max(5.0, r["utilization_pct"] * 0.85), 1)
            r["packet_loss_pct"] = round(r["packet_loss_pct"] * 0.90, 3)
            _recompute_derived(r)
            affected.append(rid)

    elif action_type == "trigger_capacity_upgrade":
        for rid in targets:
            r = routers[rid]
            r["utilization_pct"] = round(max(5.0, r["utilization_pct"] * 0.75), 1)
            r["latency_ms"] = round(r["latency_ms"] * 0.90, 1)
            _recompute_derived(r)
            affected.append(rid)

    elif action_type == "reroute_traffic":
        for rid in targets:
            candidates = [n for n in topology.neighbors(rid) if n in routers]
            if not candidates:
                continue
            requested = action.get("reroute_to")
            dest = requested if requested in candidates else min(
                candidates, key=lambda n: routers[n]["utilization_pct"]
            )
            src, dst = routers[rid], routers[dest]
            shift = min(20.0, src["utilization_pct"] * 0.30)
            src["utilization_pct"] = round(max(5.0, src["utilization_pct"] - shift), 1)
            dst["utilization_pct"] = round(min(98.0, dst["utilization_pct"] + shift), 1)
            _recompute_derived(src)
            _recompute_derived(dst)
            affected.extend([rid, dest])

    return routers, sorted(set(affected))
