"""
Mock network topology - a tiny stand-in for Confucius's real topology database.
Two regions (ATN, PRN), each with a couple of nodes, linked by backbone edges.
Capacities are in arbitrary units (e.g., Gbps).
"""

MOCK_TOPOLOGY = {
    "regions": ["ATN", "PRN"],
    "nodes": [
        {"id": "ATN1", "region": "ATN", "role": "L1"},
        {"id": "ATN2", "region": "ATN", "role": "L1"},
        {"id": "PRN1", "region": "PRN", "role": "L1"},
        {"id": "PRN2", "region": "PRN", "role": "L1"},
    ],
    "edges": [
        {"src": "ATN1", "dst": "PRN1", "capacity_gbps": 400, "current_util_gbps": 280},
        {"src": "ATN2", "dst": "PRN2", "capacity_gbps": 400, "current_util_gbps": 150},
        {"src": "ATN1", "dst": "ATN2", "capacity_gbps": 800, "current_util_gbps": 300},
        {"src": "PRN1", "dst": "PRN2", "capacity_gbps": 800, "current_util_gbps": 200},
    ],
    # The "what-if" scenario every run is built around
    "scenario": {
        "description": "A new AI training datacenter is being added to region PRN, "
                        "expected to add significant east-west traffic between ATN and PRN "
                        "over the next 90 days.",
        "new_dc_id": "PRN3",
        "new_dc_region": "PRN",
    },
    # Hard invariants the Validator Agent must enforce - analogous to Confucius's
    # graph validator (full connectivity, min-path requirements, etc.)
    "invariants": {
        "min_capacity_gbps": 0,
        "max_single_link_increase_gbps": 500,   # a single proposed change can't exceed this
        "min_remaining_paths_per_region": 1,     # a region can never be fully cut off
    },
}


def format_topology_for_prompt(topology: dict) -> str:
    """Render the topology as a readable string for LLM prompts."""
    lines = ["Current network topology:"]
    for edge in topology["edges"]:
        lines.append(
            f"  {edge['src']} <-> {edge['dst']}: "
            f"capacity={edge['capacity_gbps']}Gbps, "
            f"current_utilization={edge['current_util_gbps']}Gbps"
        )
    lines.append(f"\nScenario: {topology['scenario']['description']}")
    return "\n".join(lines)