"""A5 -- FIXED-SCALE diagnostic variant of agents/a5_verdict.py.

Same orchestration (build the same prompt inputs, validate the same two
mechanical requirements, apply the same empirical-support-based
plausibility floor) but against agents.schema.A5ScaleResponse and
agents.prompts.a5_verdict_scale_v1 instead of the real pipeline's
A5Response/a5_verdict_v5. Never imported by agents/pipeline.py -- see
results/plausibility_diagnostic.md Step 3 and eval/run_a5_scale.py, the
only caller.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from controlplane.record import EscalationRecord

from agents import base
from agents.grounding import HypothesisSupport
from agents.prompts import a5_verdict_scale_v1 as prompt_module
from agents.schema import A5_SCALE_VALUES, A5ScaleResponse, Claim, ClaimsResponse, Hypothesis, HypothesisResponse
from agents.validators import validate_a5_addresses_all_contradictions, validate_a5_credits_a_real_hypothesis
from agents.verification import VerificationCounts, counts_by_agent, match_claims

AGENT = "a5_scale"

#: nearest-below-0.3 anchor -- the fixed-scale analogue of
#: agents/grounding.py's ZERO_SUPPORT_PLAUSIBILITY_CAP (0.3) and
#: agents/escalation_grounding.py's UNGROUNDED_PLAUSIBILITY_CAP (0.3).
#: 0.3 itself is not a valid scale point (see A5_SCALE_VALUES), and
#: snapping down to 0.25 rather than up to 0.35 preserves the original
#: caps' intent ("do not let this be rated benign") -- see
#: results/plausibility_diagnostic.md's note on the original 0.3 cap
#: coinciding exactly with the 0.3 verdict threshold.
SCALE_CAP = 0.25


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
) -> Tuple[A5ScaleResponse, base.CallMetadata, bool]:
    chain_claims: List[Claim] = [*a1.claims, *a2.claims, *a3.claims]
    chain_hypotheses: List[Hypothesis] = list(a3.hypotheses)
    all_hypotheses: List[Hypothesis] = [*chain_hypotheses, *a4.hypotheses]

    verification_by_agent = counts_by_agent(match_claims(chain_claims, a4.claims))

    required_claim_ids: Set[str] = set()
    for hyp in all_hypotheses:
        required_claim_ids.update(hyp.contradicting_claim_ids)

    known_hypothesis_ids: Set[str] = {h.hypothesis_id for h in all_hypotheses}

    prompt = prompt_module.build_prompt(
        record, chain_claims, chain_hypotheses, a4.claims, a4.hypotheses,
        verification_by_agent, required_claim_ids, empirical_support_lines,
    )
    known_ids: Set[str] = {c.claim_id for c in chain_claims} | {c.claim_id for c in a4.claims}

    def validate(response: A5ScaleResponse) -> None:
        _validate_cites_something(response, known_ids)
        validate_a5_addresses_all_contradictions(response, required_claim_ids)
        validate_a5_credits_a_real_hypothesis(response, known_hypothesis_ids)

    response, meta = base.call_structured(
        record_id=record.flow_id,
        agent=AGENT,
        prompt_version=prompt_module.PROMPT_VERSION,
        prompt=prompt,
        response_schema=A5ScaleResponse,
        model=model,
        temperature=temperature,
        run_index=run_index,
        use_cache=use_cache,
        extra_validate=validate,
    )

    clamped_response, was_clamped = _apply_scale_plausibility_floor(response, supports_by_id)
    return clamped_response, meta, was_clamped


def _apply_scale_plausibility_floor(
    response: A5ScaleResponse, supports_by_id: Dict[str, HypothesisSupport],
) -> Tuple[A5ScaleResponse, bool]:
    """Fixed-scale analogue of agents.grounding.apply_empirical_plausibility_cap:
    a hypothesis with zero empirically matching benign flows cannot
    support a scale point above SCALE_CAP (0.25)."""
    if response.credited_hypothesis_id is None:
        return response, False
    support = supports_by_id.get(response.credited_hypothesis_id)
    if support is None:
        return response, False
    if support.matching_profile_count == 0 and response.benign_plausibility > SCALE_CAP:
        return response.model_copy(update={"benign_plausibility": SCALE_CAP}), True
    return response, False


def _validate_cites_something(response: A5ScaleResponse, known_ids: Set[str]) -> None:
    if not response.cited_claim_ids:
        raise ValueError("A5 must cite at least one claim_id or hypothesis_id it relied on")
