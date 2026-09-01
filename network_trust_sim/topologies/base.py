"""
Topology as data, not module-level constants. Promotes what
network/topology.py used to hardcode (ROUTER_IDS/EDGES/AGENT_DOMAINS/
AGENT_ROLES as free-standing module globals, usable by exactly one network
shape) into an instantiable object, so the IDS machinery -- TrustScoreEngine,
risk score, verification matrix, cascade gate -- can run against ANY network
shape without code changes, only a different Topology instance.
"""

from dataclasses import dataclass, field


@dataclass
class Topology:
    name: str
    router_ids: list
    edges: list                # list[tuple[str, str]]
    agent_domains: dict        # agent_id -> list[router_id]
    agent_roles: dict          # agent_id -> "field" | "orchestrator"
    _neighbors: dict = field(default=None, repr=False, compare=False)
    _observers: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        neighbors = {r: set() for r in self.router_ids}
        for a, b in self.edges:
            neighbors[a].add(b)
            neighbors[b].add(a)
        self._neighbors = neighbors

        observers = {r: [] for r in self.router_ids}
        for agent_id, routers in self.agent_domains.items():
            for r in routers:
                observers[r].append(agent_id)
        self._observers = observers

    def neighbors(self, router_id: str) -> list:
        return sorted(self._neighbors.get(router_id, set()))

    def observers_of(self, router_id: str) -> list:
        """Which agents can independently observe this router -- the basis
        for evidence-ownership verifier selection."""
        return list(self._observers.get(router_id, []))

    def overlap(self, agent_x: str, agent_y: str) -> set:
        return set(self.agent_domains.get(agent_x, [])) & set(self.agent_domains.get(agent_y, []))

    def all_agent_ids(self) -> list:
        return list(self.agent_domains.keys())

    def field_agent_ids(self) -> list:
        return [a for a in self.agent_domains if self.agent_roles[a] == "field"]

    def role_of(self, agent_id: str) -> str:
        return self.agent_roles[agent_id]

    def domain_of(self, agent_id: str) -> list:
        return list(self.agent_domains.get(agent_id, []))

    def validate(self):
        """Every field agent must have at least one overlapping peer --
        check_contagion/select_verifiers depend on this (an isolated field
        agent can never be cross-verified or shown as a contagion pathway).
        Raises ValueError with the offending agent id(s) rather than failing
        silently deep in the pipeline."""
        isolated = []
        for a in self.field_agent_ids():
            if not any(self.overlap(a, b) for b in self.field_agent_ids() if b != a):
                isolated.append(a)
        if isolated:
            raise ValueError(
                f"Topology '{self.name}': field agent(s) {isolated} have no overlapping peer -- "
                "cross-verification and contagion detection have no pathway to/from them."
            )
