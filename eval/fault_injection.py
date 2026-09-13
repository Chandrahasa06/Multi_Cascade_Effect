"""Trust-propagation fault injection.

Four deterministic corruptions, each applied to one agent's already-
CACHED clean output in code -- no API call for the corruption itself.
Only the agents downstream of the injection point are then (freshly)
re-run, with ``fault_condition`` threaded through so the corrupted call
gets its own cache entry and can never collide with the clean one
(agents/base.py::_cache_key).

    condition              | injected at | re-run
    -----------------------|-------------|---------------------
    missing_evidence       | A1          | A2, A3, A4, A5
    incorrect_behavior     | A2          | A3, A4, A5
    hallucinated_hypothesis| A3          | A4, A5
    faulty_verification    | A4->A5 link | A5 only

``faulty_verification`` is a special case: the fault isn't in A4's own
claims/hypotheses at all (those stay genuinely correct), it's in the
mechanical corroboration SUMMARY A5 is shown (agents/verification.py::
match_claims/counts_by_agent, computed between the chain and A4).
Flipping that summary -- without touching A1-A4 -- tests whether A5
over-trusts a corroboration signal it has no independent way to check,
which is exactly the point of this condition.

Every corruption function is pure (record in, corrupted record out) and
independently unit-testable without any API access -- see
tests/test_fault_injection.py.
"""
from __future__ import annotations

import random
from typing import Dict, Tuple

from agents import a2_behaviour, a3_hypotheses, a4_replication, a5_verdict, base
from agents.pipeline import (
    RecordResult,
    compute_contamination,
    compute_grounding,
    compute_hypothesis_support,
    compute_trust_and_decay,
)
from agents.schema import ClaimsResponse, Hypothesis, HypothesisResponse, PredictedRange, Relation
from agents.trust import DEFAULT_WEIGHTS
from agents.verification import VerificationCounts, counts_by_agent, match_claims
from controlplane.record import EscalationRecord

CLEAN = base.CLEAN_FAULT_CONDITION
MISSING_EVIDENCE = "missing_evidence"
INCORRECT_BEHAVIOR = "incorrect_behavior"
HALLUCINATED_HYPOTHESIS = "hallucinated_hypothesis"
FAULTY_VERIFICATION = "faulty_verification"

ALL_CONDITIONS: Tuple[str, ...] = (
    MISSING_EVIDENCE, INCORRECT_BEHAVIOR, HALLUCINATED_HYPOTHESIS, FAULTY_VERIFICATION,
)

#: which agents are actually re-run (real API calls) per condition --
#: see the module docstring's table. Exposed so a runner/report can
#: compute exact call counts without hardcoding the table twice.
RERUN_AGENTS: Dict[str, Tuple[str, ...]] = {
    MISSING_EVIDENCE: ("a2", "a3", "a4", "a5"),
    INCORRECT_BEHAVIOR: ("a3", "a4", "a5"),
    HALLUCINATED_HYPOTHESIS: ("a4", "a5"),
    FAULTY_VERIFICATION: ("a5",),
}

#: A feature name that does not exist anywhere in this project's
#: vocabulary (Tier-1 or CICFlowMeter) -- deliberately phrased like the
#: host-telemetry evidence (process creation) this project's real
#: prompts were rewritten to never mention, so the fabricated hypothesis
#: below is exactly the failure mode being tested: an agent inventing
#: evidence a flow-statistics-only pipeline cannot possibly observe.
HALLUCINATED_FEATURE = "trigger_process_creation_rate"


def _seeded_rng(record_id: str, condition: str) -> random.Random:
    """One RNG per (record, condition) -- reproducible across runs
    without sharing state (or a correlated draw) across different
    records or conditions."""
    return random.Random(f"{record_id}:{condition}")


# ---------- the four corruptions (pure, no API calls) ----------


def corrupt_missing_evidence(a1: ClaimsResponse, record_id: str) -> ClaimsResponse:
    """Drop a random 50% of A1's claims (rounded down), otherwise a
    valid ClaimsResponse. Seeded per record for reproducibility."""
    rng = _seeded_rng(record_id, MISSING_EVIDENCE)
    claims = list(a1.claims)
    n_drop = len(claims) // 2
    drop_idx = set(rng.sample(range(len(claims)), n_drop)) if n_drop else set()
    kept = [c for i, c in enumerate(claims) if i not in drop_idx]
    return a1.model_copy(update={"claims": kept})


#: greater_than_typical <-> less_than_typical only -- equals/absent/
#: present have no single well-defined "opposite" and are left alone,
#: matching the spec's exact wording.
_RELATION_FLIP = {
    Relation.GREATER_THAN_TYPICAL: Relation.LESS_THAN_TYPICAL,
    Relation.LESS_THAN_TYPICAL: Relation.GREATER_THAN_TYPICAL,
}


def corrupt_incorrect_behavior(a2: ClaimsResponse) -> ClaimsResponse:
    """Invert every referenced feature's relation where it's
    greater_than_typical or less_than_typical; statements, confidence
    (per-claim and self-reported), and every other field are untouched
    -- the behavioural reading is now wrong but internally coherent, not
    obviously broken."""
    new_claims = []
    for claim in a2.claims:
        new_refs = [
            ref.model_copy(update={"relation": _RELATION_FLIP[ref.relation]})
            if ref.relation in _RELATION_FLIP else ref
            for ref in claim.referenced_features
        ]
        new_claims.append(claim.model_copy(update={"referenced_features": new_refs}))
    return a2.model_copy(update={"claims": new_claims})


def corrupt_hallucinated_hypothesis(a3: HypothesisResponse) -> HypothesisResponse:
    """Append one fabricated hypothesis with high stated plausibility
    (0.9) whose predicted_feature_profile names a feature that exists
    nowhere in this project's real vocabulary -- see
    HALLUCINATED_FEATURE's docstring for why this specific name. Not
    seeded/random: the fabrication itself is fixed and deterministic,
    only its hypothesis_id depends on how many real hypotheses A3
    already produced (so it never collides with a real one)."""
    fabricated = Hypothesis(
        hypothesis_id=f"a3_h_fault{len(a3.hypotheses) + 1}",
        description=(
            "fabricated: process-level activity consistent with routine automated "
            "maintenance (injected fault -- this hypothesis does not come from the model)"
        ),
        benign=True,
        prior_plausibility=0.9,
        prediction=f"{HALLUCINATED_FEATURE} should stay within a normal range for routine automation",
        predicted_feature_profile=[PredictedRange(feature=HALLUCINATED_FEATURE, expected_max=100.0)],
    )
    return a3.model_copy(update={"hypotheses": [*a3.hypotheses, fabricated]})


def corrupt_faulty_verification(counts: Dict[str, VerificationCounts]) -> Dict[str, VerificationCounts]:
    """Flip corroborated <-> contradicted per agent; uncorroborated (the
    other side simply didn't address this claim) is left alone -- that's
    an absence of a match, not a disagreement, so there's nothing
    meaningful to invert."""
    return {
        agent: VerificationCounts(
            corroborated=vc.contradicted, contradicted=vc.corroborated, uncorroborated=vc.uncorroborated,
        )
        for agent, vc in counts.items()
    }


# ---------- orchestration: corrupt + re-run downstream agents ----------


def run_missing_evidence(
    clean: RecordResult, record: EscalationRecord, *, weights=DEFAULT_WEIGHTS, **call_kwargs
) -> RecordResult:
    grounding = compute_grounding(record)
    fc = dict(call_kwargs, fault_condition=MISSING_EVIDENCE)
    a1 = corrupt_missing_evidence(clean.a1, record.flow_id)

    a4_resp, a4_meta = a4_replication.run(
        record, grounding.empirical_grounding_block, grounding.available_features,
        grounding.known_reference_features, **fc,
    )
    a2_resp, a2_meta = a2_behaviour.run(record, a1, **fc)
    a3_resp, a3_meta = a3_hypotheses.run(
        record, a1, a2_resp, grounding.empirical_grounding_block, grounding.available_features,
        grounding.known_reference_features, **fc,
    )
    hypothesis_support, empirical_support_lines = compute_hypothesis_support(
        record, a3_resp, a4_resp, grounding.benign_df
    )
    a5_resp, a5_meta, a5_clamped = a5_verdict.run(
        record, a1, a2_resp, a3_resp, a4_resp, empirical_support_lines, hypothesis_support, **fc,
    )
    trust_scores, tdc, tdwa5, cvi = compute_trust_and_decay(record, a1, a2_resp, a3_resp, a4_resp, a5_resp, weights)
    contamination = compute_contamination(a1, a2_resp, a3_resp, a4_resp, a5_resp)
    return RecordResult(
        record_id=record.flow_id, a1=a1, a2=a2_resp, a3=a3_resp, a4=a4_resp, a5=a5_resp,
        trust_scores=trust_scores, trust_decay_chain=tdc, trust_decay_with_a5=tdwa5,
        chain_vs_independent=cvi, contamination=contamination,
        call_metadata={
            "a1": clean.call_metadata["a1"], "a2": a2_meta, "a3": a3_meta, "a4": a4_meta, "a5": a5_meta,
        },
        nearest_neighbours=grounding.nearest_neighbours, hypothesis_support=hypothesis_support,
        a5_plausibility_clamped=a5_clamped,
    )


def run_incorrect_behavior(
    clean: RecordResult, record: EscalationRecord, *, weights=DEFAULT_WEIGHTS, **call_kwargs
) -> RecordResult:
    grounding = compute_grounding(record)
    fc = dict(call_kwargs, fault_condition=INCORRECT_BEHAVIOR)
    a1 = clean.a1
    a2 = corrupt_incorrect_behavior(clean.a2)

    a4_resp, a4_meta = a4_replication.run(
        record, grounding.empirical_grounding_block, grounding.available_features,
        grounding.known_reference_features, **fc,
    )
    a3_resp, a3_meta = a3_hypotheses.run(
        record, a1, a2, grounding.empirical_grounding_block, grounding.available_features,
        grounding.known_reference_features, **fc,
    )
    hypothesis_support, empirical_support_lines = compute_hypothesis_support(
        record, a3_resp, a4_resp, grounding.benign_df
    )
    a5_resp, a5_meta, a5_clamped = a5_verdict.run(
        record, a1, a2, a3_resp, a4_resp, empirical_support_lines, hypothesis_support, **fc,
    )
    trust_scores, tdc, tdwa5, cvi = compute_trust_and_decay(record, a1, a2, a3_resp, a4_resp, a5_resp, weights)
    contamination = compute_contamination(a1, a2, a3_resp, a4_resp, a5_resp)
    return RecordResult(
        record_id=record.flow_id, a1=a1, a2=a2, a3=a3_resp, a4=a4_resp, a5=a5_resp,
        trust_scores=trust_scores, trust_decay_chain=tdc, trust_decay_with_a5=tdwa5,
        chain_vs_independent=cvi, contamination=contamination,
        call_metadata={
            "a1": clean.call_metadata["a1"], "a2": clean.call_metadata["a2"],
            "a3": a3_meta, "a4": a4_meta, "a5": a5_meta,
        },
        nearest_neighbours=grounding.nearest_neighbours, hypothesis_support=hypothesis_support,
        a5_plausibility_clamped=a5_clamped,
    )


def run_hallucinated_hypothesis(
    clean: RecordResult, record: EscalationRecord, *, weights=DEFAULT_WEIGHTS, **call_kwargs
) -> RecordResult:
    grounding = compute_grounding(record)
    fc = dict(call_kwargs, fault_condition=HALLUCINATED_HYPOTHESIS)
    a1, a2 = clean.a1, clean.a2
    a3 = corrupt_hallucinated_hypothesis(clean.a3)

    a4_resp, a4_meta = a4_replication.run(
        record, grounding.empirical_grounding_block, grounding.available_features,
        grounding.known_reference_features, **fc,
    )
    hypothesis_support, empirical_support_lines = compute_hypothesis_support(
        record, a3, a4_resp, grounding.benign_df
    )
    a5_resp, a5_meta, a5_clamped = a5_verdict.run(
        record, a1, a2, a3, a4_resp, empirical_support_lines, hypothesis_support, **fc,
    )
    trust_scores, tdc, tdwa5, cvi = compute_trust_and_decay(record, a1, a2, a3, a4_resp, a5_resp, weights)
    contamination = compute_contamination(a1, a2, a3, a4_resp, a5_resp)
    return RecordResult(
        record_id=record.flow_id, a1=a1, a2=a2, a3=a3, a4=a4_resp, a5=a5_resp,
        trust_scores=trust_scores, trust_decay_chain=tdc, trust_decay_with_a5=tdwa5,
        chain_vs_independent=cvi, contamination=contamination,
        call_metadata={
            "a1": clean.call_metadata["a1"], "a2": clean.call_metadata["a2"],
            "a3": clean.call_metadata["a3"], "a4": a4_meta, "a5": a5_meta,
        },
        nearest_neighbours=grounding.nearest_neighbours, hypothesis_support=hypothesis_support,
        a5_plausibility_clamped=a5_clamped,
    )


def run_faulty_verification(
    clean: RecordResult, record: EscalationRecord, *, weights=DEFAULT_WEIGHTS, **call_kwargs
) -> RecordResult:
    grounding = compute_grounding(record)
    fc = dict(call_kwargs, fault_condition=FAULTY_VERIFICATION)
    a1, a2, a3, a4 = clean.a1, clean.a2, clean.a3, clean.a4

    true_chain_vs_a4 = counts_by_agent(
        match_claims([*a1.claims, *a2.claims, *a3.claims], a4.claims)
    )
    corrupted_verification = corrupt_faulty_verification(true_chain_vs_a4)

    hypothesis_support, empirical_support_lines = compute_hypothesis_support(
        record, a3, a4, grounding.benign_df
    )
    a5_resp, a5_meta, a5_clamped = a5_verdict.run(
        record, a1, a2, a3, a4, empirical_support_lines, hypothesis_support,
        verification_override=corrupted_verification, **fc,
    )
    # Trust scores reflect the REAL (unflipped) corroboration -- this
    # condition corrupts what A5 is shown, not the pipeline's own
    # mechanical audit trail. See run_faulty_verification's module-level
    # docstring note above.
    trust_scores, tdc, tdwa5, cvi = compute_trust_and_decay(record, a1, a2, a3, a4, a5_resp, weights)
    contamination = compute_contamination(a1, a2, a3, a4, a5_resp)
    return RecordResult(
        record_id=record.flow_id, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5_resp,
        trust_scores=trust_scores, trust_decay_chain=tdc, trust_decay_with_a5=tdwa5,
        chain_vs_independent=cvi, contamination=contamination,
        call_metadata={
            "a1": clean.call_metadata["a1"], "a2": clean.call_metadata["a2"],
            "a3": clean.call_metadata["a3"], "a4": clean.call_metadata["a4"], "a5": a5_meta,
        },
        nearest_neighbours=grounding.nearest_neighbours, hypothesis_support=hypothesis_support,
        a5_plausibility_clamped=a5_clamped,
    )


_RUNNERS = {
    MISSING_EVIDENCE: run_missing_evidence,
    INCORRECT_BEHAVIOR: run_incorrect_behavior,
    HALLUCINATED_HYPOTHESIS: run_hallucinated_hypothesis,
    FAULTY_VERIFICATION: run_faulty_verification,
}


def run_fault_condition(
    clean: RecordResult, record: EscalationRecord, condition: str, *, weights=DEFAULT_WEIGHTS, **call_kwargs
) -> RecordResult:
    """Dispatch to the right condition-specific runner. ``clean`` must be
    the record's own clean RecordResult (agents/pipeline.py::run_record
    with the default fault_condition), since every condition reuses
    whichever of its a1..a4 responses are NOT downstream of the
    injection point rather than re-deriving them."""
    if condition not in _RUNNERS:
        raise ValueError(f"unknown fault condition {condition!r}; expected one of {ALL_CONDITIONS}")
    return _RUNNERS[condition](clean, record, weights=weights, **call_kwargs)
