"""
Registry of available network topologies. network/topology.py is retired as
source of truth -- not kept as a shim, since a shim would let --topology
silently do nothing wherever a call site still imported the old module
functions. Every consumer is threaded explicitly instead (see
network/state.py, network/actions.py, trust/risk_score.py,
orchestrator/verifier_selection.py, trust/contagion_metrics.py,
orchestrator/coordinator.py, orchestrator/event_coordinator.py).
"""

from topologies.diamond6 import TOPOLOGY as DIAMOND6
from topologies.mesh_large import TOPOLOGY as MESH_LARGE
from topologies.hierarchical3tier import TOPOLOGY as HIERARCHICAL3TIER

TOPOLOGY_REGISTRY = {
    "diamond6": DIAMOND6,
    "mesh_large": MESH_LARGE,
    "hierarchical3tier": HIERARCHICAL3TIER,
}


def load_topology(name: str):
    if name not in TOPOLOGY_REGISTRY:
        raise ValueError(f"Unknown topology '{name}'. Available: {sorted(TOPOLOGY_REGISTRY)}")
    return TOPOLOGY_REGISTRY[name]
