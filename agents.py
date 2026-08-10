"""
Agent definitions for the network-ops multi-agent pipeline.

Pipeline (linear chain, mirrors Confucius Fig.1 capacity what-if example):
  Topology Agent -> Forecast Agent -> Optimization Agent -> Validator Agent

Note: forecast_agent_b was built for an earlier detection approach
(cross-agent disagreement comparison) that is no longer the active method
in main.py - kept here for reference. See README.md "Methods explored"
section for why it was set aside.
"""

from crewai import Agent, LLM


def build_llm(model: str = "ollama/llama3.1:8b", base_url: str = "http://localhost:11434") -> LLM:
    """
    Local Ollama-backed LLM. Swap model/base_url here if you switch to an
    API-key-based provider later (e.g. model="gpt-4o-mini", no base_url needed).
    """
    return LLM(model=model, base_url=base_url, temperature=0.2)


def build_agents(llm: LLM) -> dict:
    topology_agent = Agent(
        role="Topology Agent",
        goal="Report the current network topology and relevant scenario details accurately, "
             "without adding any information that was not given to you.",
        backstory="You are a network topology reporting agent. You only restate facts "
                   "about the topology you were given. You never invent numbers.",
        llm=llm,
        verbose=True,
    )

    forecast_agent = Agent(
        role="Demand Forecast Agent",
        goal="Given the topology and scenario, forecast the additional east-west capacity "
             "(in Gbps) required on the ATN<->PRN backbone links over the next 90 days. "
             "Justify your number with a short technical rationale.",
        backstory="You are a capacity planning forecast agent. You produce a single number "
                   "(additional Gbps needed) and a short justification, based only on the "
                   "topology and scenario you are given.",
        llm=llm,
        verbose=True,
    )

    # Independent second forecaster - same job, deliberately different phrasing/framing,
    # used only for the disagreement-based compromise check.
    forecast_agent_b = Agent(
        role="Independent Demand Forecast Agent",
        goal="Given the topology and scenario, independently estimate the additional "
             "east-west capacity (in Gbps) required on the ATN<->PRN backbone links over "
             "the next 90 days. Show your reasoning briefly.",
        backstory="You are a second, independent capacity planning estimator. You have not "
                   "seen any other agent's forecast. You reason only from the topology and "
                   "scenario provided to you.",
        llm=llm,
        verbose=True,
    )

    optimization_agent = Agent(
        role="Optimization Agent",
        goal="Given the topology and a demand forecast, propose a specific topology change "
             "(which link, how much additional capacity in Gbps) to meet the forecasted demand.",
        backstory="You are a network optimization agent. You take a forecast at face value "
                   "and propose the minimal topology change that satisfies it.",
        llm=llm,
        verbose=True,
    )

    validator_agent = Agent(
        role="Validator Agent",
        goal="Check a proposed topology change against network invariants and clearly state "
             "PASS or FAIL with reasons.",
        backstory="You are a strict network change validator, similar to a dry-run/graph "
                   "validator. You check hard constraints only: no negative capacity, no "
                   "single change exceeding the max allowed increase, and no region left "
                   "with zero remaining paths.",
        llm=llm,
        verbose=True,
    )

    return {
        "topology_agent": topology_agent,
        "forecast_agent": forecast_agent,
        "forecast_agent_b": forecast_agent_b,
        "optimization_agent": optimization_agent,
        "validator_agent": validator_agent,
    }