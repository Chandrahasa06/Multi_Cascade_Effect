"""Single-LLM control: same input as the five-agent pipeline (full
feature set, trigger reasons, and the same empirical grounding against
Monday's real benign traffic that A1/A3/A4/A5 see) -- one call, asked
directly for benign_plausibility on the same [0,1] scale A5 reports. No
chain, no blind replication, no trust scoring.

Exists to answer the question a reviewer will ask first: does the
five-agent structure earn its cost over a single call? This is a
different question from ``agents.trust.chain_vs_independent`` (T4 -
mean(T1,T2,T3)), which compares the chain to A4 -- and A4 still runs the
full structured claims/hypotheses apparatus, just without seeing the
chain's output first. This baseline strips all of that away too.
"""
from __future__ import annotations

from controlplane.record import EscalationRecord
from controlplane.reference import get_reference_distribution, load_benign_dataframe

from agents import base
from agents.grounding import compute_nearest_neighbours, render_empirical_grounding_block
from agents.prompts import baseline_v2 as prompt_module
from agents.schema import BaselineResponse

AGENT = "baseline"


def run(
    record: EscalationRecord,
    *,
    model: str = base.DEFAULT_MODEL,
    temperature: float = 0.7,
    run_index: int = 0,
    use_cache: bool = True,
) -> tuple[BaselineResponse, base.CallMetadata]:
    reference = get_reference_distribution()
    benign_df = load_benign_dataframe()
    nn_result = compute_nearest_neighbours(record, benign_df, reference)
    empirical_grounding_block = render_empirical_grounding_block(record, reference, nn_result)

    prompt = prompt_module.build_prompt(record, empirical_grounding_block)
    return base.call_structured(
        record_id=record.flow_id,
        agent=AGENT,
        prompt_version=prompt_module.PROMPT_VERSION,
        prompt=prompt,
        response_schema=BaselineResponse,
        model=model,
        temperature=temperature,
        run_index=run_index,
        use_cache=use_cache,
    )
