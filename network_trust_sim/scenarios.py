"""
Scenario/CLI layer: selects world (1=undefended baseline, 2=risk+trust-gated),
fault mode (healthy/hallucinating/byzantine), and which agent/turn the fault
starts on -- without editing config.py by hand for every run.
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


PRESET_SCENARIOS = {
    "clean": ScenarioConfig(world=2, fault_mode="healthy"),
    "hallucination-w1": ScenarioConfig(
        world=1, fault_mode="hallucinating",
        fault_agent_id=HALLUCINATING_AGENT_ID, fault_turn=HALLUCINATION_TURN,
    ),
    "hallucination-w2": ScenarioConfig(
        world=2, fault_mode="hallucinating",
        fault_agent_id=HALLUCINATING_AGENT_ID, fault_turn=HALLUCINATION_TURN,
    ),
    "byzantine-w1": ScenarioConfig(
        world=1, fault_mode="byzantine",
        fault_agent_id=BYZANTINE_AGENT_ID, fault_turn=BYZANTINE_TURN,
    ),
    "byzantine-w2": ScenarioConfig(
        world=2, fault_mode="byzantine",
        fault_agent_id=BYZANTINE_AGENT_ID, fault_turn=BYZANTINE_TURN,
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
    args = parser.parse_args(argv)

    if args.preset:
        base = PRESET_SCENARIOS[args.preset]
        return ScenarioConfig(**{**base.__dict__, "legacy": args.legacy, "compare": args.compare})

    return ScenarioConfig(
        world=args.world,
        fault_mode=args.fault,
        fault_agent_id=args.fault_agent,
        fault_turn=args.fault_turn,
        num_turns=args.turns,
        seed=args.seed,
        legacy=args.legacy,
        compare=args.compare,
    )
