"""
Scenario/CLI layer: selects world (1=undefended baseline, 2=risk+trust-gated),
fault mode (healthy/hallucinating/byzantine), which agent/turn the fault
starts on, execution model (turn_based=the original fixed-order loop,
event_based=the discrete-event scheduler), and topology (which network
shape/agent set to run against) -- without editing config.py by hand for
every run.
"""

import argparse
from dataclasses import dataclass

from config import (
    NUM_TURNS,
    RANDOM_SEED,
    HALLUCINATING_AGENT_ID,
    HALLUCINATION_TURN,
    BYZANTINE_AGENT_ID,
    BYZANTINE_TURN,
)
from topologies import TOPOLOGY_REGISTRY, load_topology


@dataclass
class ScenarioConfig:
    world: int = 2
    fault_mode: str = "healthy"        # healthy | hallucinating | byzantine
    fault_agent_id: str = BYZANTINE_AGENT_ID
    fault_turn: int = BYZANTINE_TURN
    num_turns: int = NUM_TURNS
    seed: int = RANDOM_SEED
    legacy: bool = False
    compare: bool = False
    execution_mode: str = "turn_based"  # turn_based | event_based
    compare_execution_modes: bool = False
    topology_name: str = "diamond6"

    def resolve_topology(self):
        """Resolve topology_name -> Topology instance and validate the
        scenario is coherent against it, BEFORE any coordinator is
        constructed -- replaces a silent no-op/KeyError deep in the
        pipeline with a clear error naming exactly what's wrong."""
        topology = load_topology(self.topology_name)
        if self.fault_mode != "healthy" and self.fault_agent_id not in topology.all_agent_ids():
            raise ValueError(
                f"fault_agent_id '{self.fault_agent_id}' does not exist in topology "
                f"'{self.topology_name}'. Available agents: {topology.all_agent_ids()}"
            )
        return topology


PRESET_SCENARIOS = {
    "clean": ScenarioConfig(world=2, fault_mode="healthy", topology_name="diamond6"),
    "hallucination-w1": ScenarioConfig(
        world=1, fault_mode="hallucinating",
        fault_agent_id=HALLUCINATING_AGENT_ID, fault_turn=HALLUCINATION_TURN,
        topology_name="diamond6",
    ),
    "hallucination-w2": ScenarioConfig(
        world=2, fault_mode="hallucinating",
        fault_agent_id=HALLUCINATING_AGENT_ID, fault_turn=HALLUCINATION_TURN,
        topology_name="diamond6",
    ),
    "byzantine-w1": ScenarioConfig(
        world=1, fault_mode="byzantine",
        fault_agent_id=BYZANTINE_AGENT_ID, fault_turn=BYZANTINE_TURN,
        topology_name="diamond6",
    ),
    "byzantine-w2": ScenarioConfig(
        world=2, fault_mode="byzantine",
        fault_agent_id=BYZANTINE_AGENT_ID, fault_turn=BYZANTINE_TURN,
        topology_name="diamond6",
    ),
}


def from_args(argv=None) -> ScenarioConfig:
    parser = argparse.ArgumentParser(description="Network trust simulation runner")
    parser.add_argument("--preset", choices=sorted(PRESET_SCENARIOS), default=None)
    parser.add_argument("--world", type=int, choices=[1, 2], default=2)
    parser.add_argument("--fault", choices=["healthy", "hallucinating", "byzantine"], default="healthy")
    parser.add_argument("--fault-agent", default=BYZANTINE_AGENT_ID)
    parser.add_argument("--fault-turn", type=int, default=BYZANTINE_TURN)
    parser.add_argument("--turns", type=int, default=NUM_TURNS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--legacy", action="store_true",
        help="run the original single-chain telemetry pipeline instead of the network simulator",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="run both worlds back-to-back on the same seed and report the two-world comparison",
    )
    parser.add_argument(
        "--event-based", action="store_true",
        help="use the discrete-event scheduler (independent, jittered per-agent cadence, "
             "shared bus for peer visibility) instead of the fixed-order turn loop",
    )
    parser.add_argument(
        "--compare-execution-modes", action="store_true",
        help="run the same seeded fault scenario once turn_based, once event_based, and report "
             "whether the qualitative detection story (flagging, gating) holds under both",
    )
    parser.add_argument(
        "--topology", choices=sorted(TOPOLOGY_REGISTRY), default="diamond6",
        help="which network shape/agent set to run against",
    )
    args = parser.parse_args(argv)

    execution_mode = "event_based" if args.event_based else "turn_based"

    if args.preset:
        base = PRESET_SCENARIOS[args.preset]
        return ScenarioConfig(**{
            **base.__dict__,
            "legacy": args.legacy,
            "compare": args.compare,
            "execution_mode": execution_mode,
            "compare_execution_modes": args.compare_execution_modes,
        })

    return ScenarioConfig(
        world=args.world,
        fault_mode=args.fault,
        fault_agent_id=args.fault_agent,
        fault_turn=args.fault_turn,
        num_turns=args.turns,
        seed=args.seed,
        legacy=args.legacy,
        compare=args.compare,
        execution_mode=execution_mode,
        compare_execution_modes=args.compare_execution_modes,
        topology_name=args.topology,
    )
