"""A4 -- blind replication. Receives the raw record only, no chain
output, and has no dependency on A1-A3 -- run this concurrently with
the chain, never after it."""
from __future__ import annotations

from controlplane.record import EscalationRecord

from agents import base
from agents.prompts import a4_replication_v5 as prompt_module
from agents.schema import HypothesisResponse
from agents.validators import validate_hypothesis_response

AGENT = "a4"


def run(
    record: EscalationRecord,
    empirical_grounding_block: str,
    available_features: str,
    known_reference_features,
    *,
    model: str = base.DEFAULT_MODEL,
    temperature: float = 0.7,
    run_index: int = 0,
    use_cache: bool = True,
    fault_condition: str = base.CLEAN_FAULT_CONDITION,
) -> tuple[HypothesisResponse, base.CallMetadata]:
    prompt = prompt_module.build_prompt(record, empirical_grounding_block, available_features)
    return base.call_structured(
        record_id=record.flow_id,
        agent=AGENT,
        prompt_version=prompt_module.PROMPT_VERSION,
        prompt=prompt,
        response_schema=HypothesisResponse,
        model=model,
        temperature=temperature,
        run_index=run_index,
        use_cache=use_cache,
        extra_validate=lambda r: validate_hypothesis_response(AGENT, r, known_reference_features),
        fault_condition=fault_condition,
    )
