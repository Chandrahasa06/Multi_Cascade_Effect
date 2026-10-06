"""Devil's-advocate agent (reframe arm B). Argues only the attack reading.

Its output is checked by the same flow-level filter as A3 and against the same
feature vocabulary, and its profiles are queried through the same grounding path
(see eval/run_reframe_45.py). Its hypotheses are never added to A5's
required-claim set and never credited as benign explanations.
"""
from __future__ import annotations

from typing import Iterable, Set

from controlplane.record import EscalationRecord

from agents import base
from agents.prompts import da_advocate_v1 as prompt_module
from agents.schema import Claim, ClaimsResponse, HypothesisResponse
from agents.validators import validate_claim_ids, validate_hypothesis_predictions_are_flow_level

AGENT = "da"
MIN_ADVOCATE_HYPOTHESES = 2


def validate_advocate_response(response: HypothesisResponse, known_feature_names: Set[str]) -> None:
    """Same contract as A3 where it applies: id prefix, uniqueness, flow-level predictions.
    Differs in two ways: every hypothesis is an attack (benign=False), and no
    contradicting claim ids are allowed, because the advocate does not weigh the benign side."""
    validate_claim_ids(AGENT, response)
    seen = set()
    for hyp in response.hypotheses:
        if not hyp.hypothesis_id.startswith(f"{AGENT}_h"):
            raise ValueError(f"hypothesis_id {hyp.hypothesis_id!r} must start with {AGENT}_h")
        if hyp.hypothesis_id in seen:
            raise ValueError(f"duplicate hypothesis_id {hyp.hypothesis_id!r}")
        seen.add(hyp.hypothesis_id)
        if hyp.benign:
            raise ValueError(f"advocate hypothesis {hyp.hypothesis_id!r} must have benign=false")
        if hyp.contradicting_claim_ids:
            raise ValueError(f"advocate hypothesis {hyp.hypothesis_id!r} must leave contradicting_claim_ids empty")
    if len(response.hypotheses) < MIN_ADVOCATE_HYPOTHESES:
        raise ValueError(f"expected at least {MIN_ADVOCATE_HYPOTHESES} advocate hypotheses, got {len(response.hypotheses)}")
    if known_feature_names:
        validate_hypothesis_predictions_are_flow_level(response, known_feature_names)


def run(
    record: EscalationRecord,
    a1: ClaimsResponse,
    empirical_grounding_block: str,
    available_features: str,
    known_reference_features,
    evidence_block: str,
    *,
    model: str = base.DEFAULT_MODEL,
    temperature: float = 0.7,
    run_index: int = 0,
    use_cache: bool = False,
    fault_condition: str = base.CLEAN_FAULT_CONDITION,
):
    prompt = prompt_module.build_prompt(
        record, a1.claims, empirical_grounding_block, available_features, evidence_block,
    )
    known = set(known_reference_features)
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
        extra_validate=lambda r: validate_advocate_response(r, known),
        fault_condition=fault_condition,
    )
