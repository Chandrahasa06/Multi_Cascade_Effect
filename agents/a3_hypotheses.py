"""A3 -- hypothesis generation. Third stage of the sequential chain;
receives A1 and A2's full output."""
from __future__ import annotations

from controlplane.record import EscalationRecord

from agents import base
from agents.prompts import a3_hypotheses_v5 as prompt_module
from agents.schema import ClaimsResponse, HypothesisResponse
from agents.validators import validate_hypothesis_response

AGENT = "a3"


def run(
    record: EscalationRecord,
    a1: ClaimsResponse,
    a2: ClaimsResponse,
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
    prompt = prompt_module.build_prompt(
        record, a1.claims, a2.claims, empirical_grounding_block, available_features
    )
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
