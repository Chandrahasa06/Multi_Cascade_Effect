"""
Generates synthetic but plausible telemetry for each network segment/turn.
This is the source-of-truth data: grounding checks verify agent claims against it.
"""

import random

from config import RANDOM_SEED

random.seed(RANDOM_SEED)

SEGMENT_NAMES = [
    "core-router-1",
    "edge-switch-2",
    "core-router-3",
    "edge-switch-4",
    "backbone-link-5",
]


def generate_telemetry(agent_id: str, segment: str, turn: int) -> dict:
    """Synthetic telemetry reading for one segment on one turn."""
    base_util = 40 + (turn * 3) % 30
    util = round(base_util + random.uniform(-5, 5), 1)
    return {
        "segment": segment,
        "turn": turn,
        "bandwidth_utilization_pct": util,
        "link_capacity_gbps": 500,
        "current_throughput_gbps": round(500 * util / 100, 1),
        "error_rate_pct": round(random.uniform(0.01, 0.3), 3),
        "avg_latency_ms": round(random.uniform(2, 15), 1),
        "active_connections": random.randint(800, 5000),
        "packet_loss_pct": round(random.uniform(0.0, 0.5), 3),
    }


def telemetry_to_source_text(telemetry: dict) -> str:
    """Render telemetry as the plain-text 'source of truth' string given to the
    agent and later used for grounding verification."""
    return "\n".join(f"{k}: {v}" for k, v in telemetry.items())