"""A1 -- evidence interpretation. First stage of the sequential chain."""
from __future__ import annotations

from typing import Optional

from controlplane.record import EscalationRecord

from agents import base
from agents.prompts import a1_evidence_v5 as prompt_module
from agents.prompts import a1_evidence_v6
from agents.schema import ClaimsResponse
from agents.validators import validate_claim_ids

AGENT = "a1"


def run(
    record: EscalationRecord,
    empirical_grounding_block: str,
    *,
    model: str = base.DEFAULT_MODEL,
    temperature: float = 0.7,
    run_index: int = 0,
    use_cache: bool = True,
    fault_condition: str = base.CLEAN_FAULT_CONDITION,
    evidence_block: Optional[str] = None,
) -> tuple[ClaimsResponse, base.CallMetadata]:
    # evidence_block None keeps the v5 prompt and its cache keys; a block selects v6
    if evidence_block is None:
        module = prompt_module
        prompt = module.build_prompt(record, empirical_grounding_block)
    else:
        module = a1_evidence_v6
        prompt = module.build_prompt(record, empirical_grounding_block, evidence_block)
    return base.call_structured(
        record_id=record.flow_id,
        agent=AGENT,
        prompt_version=module.PROMPT_VERSION,
        prompt=prompt,
        response_schema=ClaimsResponse,
        model=model,
        temperature=temperature,
        run_index=run_index,
        use_cache=use_cache,
        extra_validate=lambda r: validate_claim_ids(AGENT, r),
        fault_condition=fault_condition,
    )
