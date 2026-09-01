"""
A larger, structurally different topology from diamond6: two 5-router
clusters joined by only 2 bridge edges (a deliberate cross-cluster
bottleneck), 6 field agents + 1 orchestrator, each field agent covering 3
routers with a sliding-window overlap onto its ring-neighbors -- a loose
ring of overlapping domains rather than diamond6's single branching chain.
Stress-tests peer-visibility/staleness at higher agent count with no single
dominant chain.

  Cluster A (ring)        bridges        Cluster B (ring)
   R1-R2-R3-R4-R5   --R5-R6--  R6-R7-R8-R9-R10  --R10-R1--  (closes the loop)

Agent domains slide by 2 routers around the combined ring:
  agent_a: R1,R2,R3   agent_b: R2,R3,R4   agent_c: R4,R5,R6
  agent_d: R6,R7,R8   agent_e: R8,R9,R10  agent_f: R1,R9,R10
Each overlaps both its ring-neighbors (agent_a<->agent_b<->agent_c<->...
<->agent_f<->agent_a), including overlap right at both bridge routers
(R6: agent_c/agent_d; R1: agent_a/agent_f).
"""

from topologies.base import Topology

TOPOLOGY = Topology(
    name="mesh_large",
    router_ids=["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"],
    edges=[
        # cluster A ring
        ("R1", "R2"), ("R2", "R3"), ("R3", "R4"), ("R4", "R5"), ("R5", "R1"),
        # cluster B ring
        ("R6", "R7"), ("R7", "R8"), ("R8", "R9"), ("R9", "R10"), ("R10", "R6"),
        # bridges -- the only two edges connecting the clusters
        ("R5", "R6"), ("R10", "R1"),
    ],
    agent_domains={
        "agent_orchestrator": [],
        "agent_a": ["R1", "R2", "R3"],
        "agent_b": ["R2", "R3", "R4"],
        "agent_c": ["R4", "R5", "R6"],
        "agent_d": ["R6", "R7", "R8"],
        "agent_e": ["R8", "R9", "R10"],
        "agent_f": ["R1", "R9", "R10"],
    },
    agent_roles={
        "agent_orchestrator": "orchestrator",
        "agent_a": "field",
        "agent_b": "field",
        "agent_c": "field",
        "agent_d": "field",
        "agent_e": "field",
        "agent_f": "field",
    },
)
