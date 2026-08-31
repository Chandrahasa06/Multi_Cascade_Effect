"""
Static router graph + which agent observes which routers. This is the single
source of truth for network *structure* -- config.py stays the place for
tunable numbers, not the graph itself.

        R1
       /  \\
     R2    R3
      \\    /
       R4
       |
       R5 --- R6
"""

ROUTER_IDS = ["R1", "R2", "R3", "R4", "R5", "R6"]

EDGES = [
    ("R1", "R2"),
    ("R1", "R3"),
    ("R2", "R4"),
    ("R3", "R4"),
    ("R4", "R5"),
    ("R5", "R6"),
]

# Which routers each agent is authorized to observe. agent_orchestrator has no
# routers of its own -- it only ever sees peer agents' decisions, never raw
# network state. The overlap between agent_traffic/agent_routing (R4) and
# agent_routing/agent_safety (R5) is deliberate: it's what makes cross-agent
# verification and contagion measurement meaningful instead of trivial.
AGENT_DOMAINS = {
    "agent_orchestrator": [],
    "agent_traffic": ["R1", "R2", "R4"],
    "agent_routing": ["R3", "R4", "R5"],
    "agent_safety": ["R5", "R6"],
}

AGENT_ROLES = {
    "agent_orchestrator": "orchestrator",
    "agent_traffic": "field",
    "agent_routing": "field",
    "agent_safety": "field",
}

_NEIGHBORS = {r: set() for r in ROUTER_IDS}
for _a, _b in EDGES:
    _NEIGHBORS[_a].add(_b)
    _NEIGHBORS[_b].add(_a)

_OBSERVERS = {r: [] for r in ROUTER_IDS}
for _agent_id, _routers in AGENT_DOMAINS.items():
    for _r in _routers:
        _OBSERVERS[_r].append(_agent_id)


def neighbors(router_id: str) -> list:
    return sorted(_NEIGHBORS.get(router_id, set()))


def observers_of(router_id: str) -> list:
    """Which agents can independently observe this router -- the basis for
    evidence-ownership verifier selection (only ask agents who actually have
    relevant evidence, not an arbitrary/all-peer vote)."""
    return list(_OBSERVERS.get(router_id, []))


def overlap(agent_x: str, agent_y: str) -> set:
    return set(AGENT_DOMAINS.get(agent_x, [])) & set(AGENT_DOMAINS.get(agent_y, []))


def all_agent_ids() -> list:
    return list(AGENT_DOMAINS.keys())


def field_agent_ids() -> list:
    return [a for a in AGENT_DOMAINS if AGENT_ROLES[a] == "field"]
