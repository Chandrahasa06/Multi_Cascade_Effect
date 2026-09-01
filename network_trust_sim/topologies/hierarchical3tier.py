"""
A real core/aggregation/edge datacenter/ISP pattern -- structurally distinct
from mesh_large (an explicit hierarchy, not a flat ring), so the
generalization claim is tested against genuinely different shapes, not just
different sizes.

  Core:  RC1 --- RC2                       (fully meshed, 2 routers)
  Agg:   RA1, RA2, RA3, each dual-homed to BOTH RC1 and RC2 (redundant uplinks)
  Edge:  RE1,RE2 -> RA1   RE3 -> RA2   RE4 -> RA3   (single-homed)

Deviation from the plan's original one-line sketch, worth stating: a single
"agent_edge covering all 4 edge routers" would have ZERO overlap with any
other field agent (edge routers aren't shared with anyone) and fail the
topology's own isolation check. Split into two edge agents instead, each
keeping its immediate upstream aggregation router in its domain too -- a
realistic "local + one-hop-up" visibility an edge NOC agent would actually
have, and what gives it a real overlapping peer (agent_agg1/2/3).
"""

from topologies.base import Topology

TOPOLOGY = Topology(
    name="hierarchical3tier",
    router_ids=["RC1", "RC2", "RA1", "RA2", "RA3", "RE1", "RE2", "RE3", "RE4"],
    edges=[
        ("RC1", "RC2"),
        ("RA1", "RC1"), ("RA1", "RC2"),
        ("RA2", "RC1"), ("RA2", "RC2"),
        ("RA3", "RC1"), ("RA3", "RC2"),
        ("RE1", "RA1"), ("RE2", "RA1"),
        ("RE3", "RA2"),
        ("RE4", "RA3"),
    ],
    agent_domains={
        "agent_orchestrator": [],
        "agent_core": ["RC1", "RC2"],
        "agent_agg1": ["RA1", "RC1", "RC2"],
        "agent_agg2": ["RA2", "RC1", "RC2"],
        "agent_agg3": ["RA3", "RC1", "RC2"],
        "agent_edge_a": ["RE1", "RE2", "RA1"],
        "agent_edge_b": ["RE3", "RE4", "RA2", "RA3"],
    },
    agent_roles={
        "agent_orchestrator": "orchestrator",
        "agent_core": "field",
        "agent_agg1": "field",
        "agent_agg2": "field",
        "agent_agg3": "field",
        "agent_edge_a": "field",
        "agent_edge_b": "field",
    },
)
