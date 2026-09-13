"""A2 -- behavioural characterisation. Second stage of the sequential
chain; receives A1's full output."""
from __future__ import annotations

from controlplane.record import EscalationRecord

from agents import base
from agents.prompts import a2_behaviour_v2 as prompt_module
from agents.schema import ClaimsResponse
from agents.validators import validate_claim_ids

AGENT = "a2"


def run(
    record: EscalationRecord,
    a1: ClaimsResponse,
    *,
    model: str = base.DEFAULT_MODEL,
    temperature: float = 0.7,
    run_index: int = 0,
    use_cache: bool = True,
    fault_condition: str = base.CLEAN_FAULT_CONDITION,
) -> tuple[ClaimsResponse, base.CallMetadata]:
    prompt = prompt_module.build_prompt(record, a1.claims)
    return base.call_structured(
        record_id=record.flow_id,
        agent=AGENT,
        prompt_version=prompt_module.PROMPT_VERSION,
        prompt=prompt,
        response_schema=ClaimsResponse,
        model=model,
        temperature=temperature,
        run_index=run_index,
        use_cache=use_cache,
        extra_validate=lambda r: validate_claim_ids(AGENT, r),
        fault_condition=fault_condition,
    )
