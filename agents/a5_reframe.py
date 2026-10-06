"""A5 for the three reframe arms. Same validation and cap order as agents/a5_verdict.run.

Arms:
  adversarial  a5_verdict_adversarial_v1, single score, A5Response
  devil        a5_verdict_devil_v1, single score, with the advocate's hypotheses shown
  twosided     a5_verdict_twosided_v1, benign and attack scores, TwoSidedA5Response

Advocate hypotheses are displayed to A5 but are NOT part of its required-claim set and
cannot be credited as credited_hypothesis_id. Credit stays restricted to the chain and
A4 hypotheses, as in the single-score pipeline.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from controlplane.record import EscalationRecord

from agents import base
from agents.a5_verdict import _validate_cites_something
from agents.grounding import HypothesisSupport, apply_empirical_plausibility_cap
from agents.prompts import a5_verdict_adversarial_v1, a5_verdict_devil_v1, a5_verdict_twosided_v1
from agents.schema import A5Response, ClaimsResponse, HypothesisResponse, TwoSidedA5Response
from agents.validators import validate_a5_addresses_all_contradictions, validate_a5_credits_a_real_hypothesis
from agents.verification import counts_by_agent, match_claims

AGENT = "a5"
ARMS = ("adversarial", "devil", "twosided")


def run_arm(
    arm: str,
    record: EscalationRecord,
    a1: ClaimsResponse,
    a2: ClaimsResponse,
    a3: HypothesisResponse,
    a4: HypothesisResponse,
    empirical_support_lines: Dict[str, str],
    supports_by_id: Dict[str, HypothesisSupport],
    evidence_block: str,
    *,
    da_hypotheses: Optional[Iterable] = None,
    da_support_lines: Optional[Dict[str, str]] = None,
    model: str = base.DEFAULT_MODEL,
    temperature: float = 0.7,
    run_index: int = 0,
    use_cache: bool = False,
    fault_condition: str = base.CLEAN_FAULT_CONDITION,
):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    chain_claims: List = [*a1.claims, *a2.claims, *a3.claims]
    chain_hypotheses: List = list(a3.hypotheses)
    all_hypotheses: List = [*chain_hypotheses, *a4.hypotheses]
    verification_by_agent = counts_by_agent(match_claims(chain_claims, a4.claims))
    required: Set[str] = set()
    for hyp in all_hypotheses:
        required.update(hyp.contradicting_claim_ids)
    known_hypothesis_ids: Set[str] = {h.hypothesis_id for h in all_hypotheses}
    known_claim_ids: Set[str] = {c.claim_id for c in chain_claims} | {c.claim_id for c in a4.claims}

    if arm == "adversarial":
        module, schema = a5_verdict_adversarial_v1, A5Response
        prompt = module.build_prompt(
            record, chain_claims, chain_hypotheses, a4.claims, a4.hypotheses,
            verification_by_agent, required, empirical_support_lines, evidence_block,
        )
    elif arm == "devil":
        module, schema = a5_verdict_devil_v1, A5Response
        prompt = module.build_prompt(
            record, chain_claims, chain_hypotheses, a4.claims, a4.hypotheses,
            verification_by_agent, required, empirical_support_lines, evidence_block,
            da_hypotheses=list(da_hypotheses or []), da_support_lines=da_support_lines or {},
        )
    else:
        module, schema = a5_verdict_twosided_v1, TwoSidedA5Response
        prompt = module.build_prompt(
            record, chain_claims, chain_hypotheses, a4.claims, a4.hypotheses,
            verification_by_agent, required, empirical_support_lines, evidence_block,
        )

    def validate(response) -> None:
        _validate_cites_something(response, known_claim_ids)
        validate_a5_addresses_all_contradictions(response, required)
        validate_a5_credits_a_real_hypothesis(response, known_hypothesis_ids)

    response, meta = base.call_structured(
        record_id=record.flow_id,
        agent=AGENT,
        prompt_version=module.PROMPT_VERSION,
        prompt=prompt,
        response_schema=schema,
        model=model,
        temperature=temperature,
        run_index=run_index,
        use_cache=use_cache,
        extra_validate=validate,
        fault_condition=fault_condition,
    )
    clamped, was_clamped = apply_empirical_plausibility_cap(response, supports_by_id)
    return clamped, meta, was_clamped
