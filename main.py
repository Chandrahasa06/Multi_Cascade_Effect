"""
Run this to produce the traces for the meeting:
  1. baseline    - clean run, no injection
  2. compromised - forecast injection active

Applies the grounding check (see detector.py) to the Forecast Agent's
output. Other methods explored during development are documented in
README.md but not run here - see README for why.

Usage:
    python main.py
"""

import json
import os
from datetime import datetime

from crewai import Crew, Process

from agents import build_llm, build_agents
from tasks import build_tasks
from detector import grounding_check

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def run_pipeline(inject_attack: bool, run_name: str) -> dict:
    llm = build_llm()
    agents = build_agents(llm)
    tasks = build_tasks(agents, inject_attack=inject_attack)

    main_crew = Crew(
        agents=[
            agents["topology_agent"],
            agents["forecast_agent"],
            agents["optimization_agent"],
            agents["validator_agent"],
        ],
        tasks=[
            tasks["topology_task"],
            tasks["forecast_task"],
            tasks["optimization_task"],
            tasks["validation_task"],
        ],
        process=Process.sequential,
        verbose=True,
    )
    main_crew.kickoff()

    result = {
        "run_name": run_name,
        "inject_attack": inject_attack,
        "timestamp": datetime.now().isoformat(),
        "topology_output": str(tasks["topology_task"].output),
        "forecast_output": str(tasks["forecast_task"].output),
        "optimization_output": str(tasks["optimization_task"].output),
        "validation_output": str(tasks["validation_task"].output),
    }

    result["grounding_check"] = grounding_check(
        result["forecast_output"], result["topology_output"]
    )

    log_path = os.path.join(LOG_DIR, f"{run_name}.json")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_summary(result: dict):
    print("\n" + "=" * 70)
    print(f"RUN: {result['run_name']}  (injected={result['inject_attack']})")
    print("=" * 70)
    print(f"\n[Forecast Agent]\n{result['forecast_output']}")
    print(f"\n[Optimization Agent]\n{result['optimization_output']}")
    print(f"\n[Grounding check]\n{result['grounding_check']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("\n>>> Running BASELINE (no injection)...")
    baseline = run_pipeline(inject_attack=False, run_name="baseline")
    print_summary(baseline)

    print("\n>>> Running COMPROMISED (forecast injection active)...")
    compromised = run_pipeline(inject_attack=True, run_name="compromised")
    print_summary(compromised)

    print("\n>>> DONE. Full JSON traces saved in ./logs/")
    print("Compare logs/baseline.json vs logs/compromised.json for your writeup.")