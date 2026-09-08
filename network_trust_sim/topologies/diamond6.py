"""
The original 6-router demo topology, relocated verbatim from
network/topology.py into a Topology instance -- this is a pure data move,
not a redesign (see topologies/base.py's Topology for the shared logic all
topologies now use).

        R1
       /  \\
     R2    R3
      \\    /
       R4
       |
       R5 --- R6

R4 is shared by agent_traffic/agent_routing; R5 is shared by
agent_routing/agent_safety -- deliberate overlaps so cross-agent
verification and contagion have a real pathway.
"""

from topologies.base import Topology

TOPOLOGY = Topology(
    name="diamond6",
    router_ids=["R1", "R2", "R3", "R4", "R5", "R6"],
    edges=[
        ("R1", "R2"),
        ("R1", "R3"),
        ("R2", "R4"),
        ("R3", "R4"),
        ("R4", "R5"),
        ("R5", "R6"),
    ],
    agent_domains={
        "agent_orchestrator": [],
        "agent_traffic": ["R1", "R2", "R4"],
        "agent_routing": ["R3", "R4", "R5"],
        "agent_safety": ["R5", "R6"],
    },
    agent_roles={
        "agent_orchestrator": "orchestrator",
        "agent_traffic": "field",
        "agent_routing": "field",
        "agent_safety": "field",
    },
)
