"""A5 -- verdict. Receives the chain's full output, A4's independent
output, the mechanical agreement between them, and the empirical support
for every hypothesis (all computed here, in code, before the prompt is
built -- never left for the model itself to eyeball or restate)."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from controlplane.record import EscalationRecord

from agents import base
from agents.grounding import HypothesisSupport, apply_empirical_plausibility_cap
from agents.prompts import a5_verdict_v5 as prompt_module
from agents.schema import A5Response, Claim, ClaimsResponse, Hypothesis, HypothesisResponse
from agents.validators import validate_a5_addresses_all_contradictions, validate_a5_credits_a_real_hypothesis
from agents.verification import VerificationCounts, counts_by_agent, match_claims

AGENT = "a5"


def run(
    record: EscalationRecord,
    a1: ClaimsResponse,
    a2: ClaimsResponse,
    a3: HypothesisResponse,
    a4: HypothesisResponse,
    empirical_support_lines: Dict[str, str],
    supports_by_id: Dict[str, HypothesisSupport],
    *,
    model: str = base.DEFAULT_MODEL,
    temperature: float = 0.7,
    run_index: int = 0,
    use_cache: bool = True,
    fault_condition: str = base.CLEAN_FAULT_CONDITION,
    verification_override: Optional[Dict[str, VerificationCounts]] = None,
) -> Tuple[A5Response, base.CallMetadata, bool]:
    """``verification_override``: for eval/fault_injection.py's
    ``faulty_verification`` condition only -- A5's prompt normally shows
    the REAL mechanical corroboration between the chain and A4
    (computed below), but that condition needs A5 to see a deliberately
    flipped version instead, without touching A1-A4 at all. None (the
    default) preserves the real computation for every other caller."""
    chain_claims: List[Claim] = [*a1.claims, *a2.claims, *a3.claims]
    chain_hypotheses: List[Hypothesis] = list(a3.hypotheses)
    all_hypotheses: List[Hypothesis] = [*chain_hypotheses, *a4.hypotheses]

    verification_by_agent = (
        verification_override if verification_override is not None
        else counts_by_agent(match_claims(chain_claims, a4.claims))
    )

    required_claim_ids: Set[str] = set()
    for hyp in all_hypotheses:
        required_claim_ids.update(hyp.contradicting_claim_ids)

    known_hypothesis_ids: Set[str] = {h.hypothesis_id for h in all_hypotheses}

    prompt = prompt_module.build_prompt(
        record, chain_claims, chain_hypotheses, a4.claims, a4.hypotheses,
        verification_by_agent, required_claim_ids, empirical_support_lines,
    )
    known_ids: Set[str] = {c.claim_id for c in chain_claims} | {c.claim_id for c in a4.claims}

    def validate(response: A5Response) -> None:
        _validate_cites_something(response, known_ids)
        validate_a5_addresses_all_contradictions(response, required_claim_ids)
        validate_a5_credits_a_real_hypothesis(response, known_hypothesis_ids)

    response, meta = base.call_structured(
        record_id=record.flow_id,
        agent=AGENT,
        prompt_version=prompt_module.PROMPT_VERSION,
        prompt=prompt,
        response_schema=A5Response,
        model=model,
        temperature=temperature,
        run_index=run_index,
        use_cache=use_cache,
        extra_validate=validate,
        fault_condition=fault_condition,
    )

    # Fix 4 (redesign spec): a hypothesis with zero empirically matching
    # benign flows cannot support a plausibility above the cap, enforced
    # here in code against the REAL computed support, regardless of what
    # the model reported -- see agents/grounding.py's docstring.
    clamped_response, was_clamped = apply_empirical_plausibility_cap(response, supports_by_id)
    return clamped_response, meta, was_clamped


def _validate_cites_something(response: A5Response, known_ids: Set[str]) -> None:
    if not response.cited_claim_ids:
        raise ValueError("A5 must cite at least one claim_id or hypothesis_id it relied on")
