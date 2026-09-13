"""Orchestrates the five-agent pipeline over EscalationRecords.

A1->A2->A3 is sequential (each stage's prompt is built from the last
stage's full output, so it can't be parallelised). A4 has no dependency
on the chain at all, so it runs concurrently with it, not after.

Every call is routed through agents/base.py's disk cache, so this
module is resumable for free: killing a run and restarting it re-does
nothing that already succeeded.

Ground truth never enters this module. ``EscalationRecord`` has no
label field to begin with (controlplane/record.py); nothing here reads
one, and tests/test_pipeline.py asserts no label-shaped field can reach
a prompt.
"""
from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from controlplane.record import EscalationRecord, PacketWindowSummary
from dataplane.selector import TriggerReason

from controlplane.reference import (
    NearestNeighbourResult,
    get_reference_distribution,
    load_benign_dataframe,
)

from agents import a1_evidence, a2_behaviour, a3_hypotheses, a4_replication, a5_verdict, base
from agents.contamination import ContaminationReport, scan_agent_response
from agents.grounding import (
    HypothesisSupport,
    compute_nearest_neighbours,
    evaluate_all,
    observed_tier1_values,
    reference_feature_names,
    render_available_profile_features,
    render_empirical_grounding_block,
    render_hypothesis_support,
)
from agents.prompts.render import TRIGGER_FEATURE_PREFIX
from agents.schema import A5Response, ClaimsResponse, HypothesisResponse
from agents.trust import (
    DEFAULT_WEIGHTS,
    TrustDecay,
    TrustScore,
    chain_vs_independent,
    compute_trust_decay,
    compute_trust_score,
)
from agents.verification import VerificationCounts, counts_by_agent, match_claims


def load_records(path: Union[str, Path]) -> List[EscalationRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(_record_from_dict(json.loads(line)))
    return records


def _record_from_dict(data: dict) -> EscalationRecord:
    return EscalationRecord(
        flow_id=data["flow_id"],
        trigger_reasons=[
            TriggerReason(
                feature=tr["feature"],
                observed_value=tr["observed_value"],
                threshold=tr["threshold"],
                direction=tr["direction"],
                ratio=tr["ratio"],
                low_confidence=tr.get("low_confidence", False),
            )
            for tr in data["trigger_reasons"]
        ],
        features=data["features"],
        packet_window=PacketWindowSummary(**data["packet_window"]),
        selector_config_hash=data["selector_config_hash"],
    )


@dataclass
class RecordResult:
    record_id: str
    a1: ClaimsResponse
    a2: ClaimsResponse
    a3: HypothesisResponse
    a4: HypothesisResponse
    a5: A5Response
    trust_scores: Dict[str, TrustScore]
    trust_decay_chain: Optional[TrustDecay]
    trust_decay_with_a5: Optional[TrustDecay]
    chain_vs_independent: Optional[float]
    contamination: Dict[str, ContaminationReport]
    call_metadata: Dict[str, base.CallMetadata]
    nearest_neighbours: NearestNeighbourResult
    hypothesis_support: Dict[str, HypothesisSupport]
    a5_plausibility_clamped: bool

    def to_dict(self) -> dict:
        def meta_dict(m: base.CallMetadata) -> dict:
            return {
                "cached": m.cached,
                "schema_retries": m.schema_retries,
                "retry_reasons": m.retry_reasons,
                "elapsed_s": m.elapsed_s,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
            }

        return {
            "record_id": self.record_id,
            "a1": self.a1.model_dump(mode="json"),
            "a2": self.a2.model_dump(mode="json"),
            "a3": self.a3.model_dump(mode="json"),
            "a4": self.a4.model_dump(mode="json"),
            "a5": self.a5.model_dump(mode="json"),
            "trust_scores": {k: v.to_dict() for k, v in self.trust_scores.items()},
            "trust_decay_chain": self.trust_decay_chain.to_dict() if self.trust_decay_chain else None,
            "trust_decay_with_a5": (
                self.trust_decay_with_a5.to_dict() if self.trust_decay_with_a5 else None
            ),
            "chain_vs_independent": self.chain_vs_independent,
            "contamination": {k: v.to_dict() for k, v in self.contamination.items()},
            "call_metadata": {k: meta_dict(m) for k, m in self.call_metadata.items()},
            "nearest_neighbours": self.nearest_neighbours.to_dict(),
            "hypothesis_support": {k: v.to_dict() for k, v in self.hypothesis_support.items()},
            "a5_plausibility_clamped": self.a5_plausibility_clamped,
        }


@dataclass
class Grounding:
    """Empirical grounding against Monday's real benign traffic -- a pure
    function of the record (plus the cached reference data), so it's
    identical whether the run downstream of it is clean or fault-
    injected. Factored out of run_record so eval/fault_injection.py's
    reruns can build the exact same grounding without duplicating this
    wiring."""

    reference: Dict[str, "FeatureReference"]
    benign_df: "pd.DataFrame"
    nearest_neighbours: NearestNeighbourResult
    empirical_grounding_block: str
    available_features: str
    known_reference_features: set


def compute_grounding(record: EscalationRecord) -> Grounding:
    reference = get_reference_distribution()
    benign_df = load_benign_dataframe()
    nn_result = compute_nearest_neighbours(record, benign_df, reference)
    empirical_grounding_block = render_empirical_grounding_block(record, reference, nn_result)
    available_features = render_available_profile_features(reference)
    known_reference_features = reference_feature_names(reference)
    return Grounding(
        reference=reference,
        benign_df=benign_df,
        nearest_neighbours=nn_result,
        empirical_grounding_block=empirical_grounding_block,
        available_features=available_features,
        known_reference_features=known_reference_features,
    )


def compute_hypothesis_support(
    record: EscalationRecord, a3: HypothesisResponse, a4: HypothesisResponse, benign_df
) -> Tuple[Dict[str, HypothesisSupport], Dict[str, str]]:
    """Empirical hypothesis testing (fix 2 of the redesign): query
    Monday's real benign flows against every hypothesis's
    predicted_feature_profile BEFORE A5 ever sees them. Factored out of
    run_record for the same reuse reason as compute_grounding."""
    all_hypotheses = [*a3.hypotheses, *a4.hypotheses]
    trigger_reason_values_prefixed = observed_tier1_values(record, prefixed=True)
    hypothesis_support = evaluate_all(all_hypotheses, trigger_reason_values_prefixed, benign_df)
    empirical_support_lines = {
        hid: render_hypothesis_support(support) for hid, support in hypothesis_support.items()
    }
    return hypothesis_support, empirical_support_lines


def compute_trust_and_decay(
    record: EscalationRecord,
    a1: ClaimsResponse,
    a2: ClaimsResponse,
    a3: HypothesisResponse,
    a4: HypothesisResponse,
    a5: A5Response,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> Tuple[Dict[str, TrustScore], Optional[TrustDecay], Optional[TrustDecay], Optional[float]]:
    """Trust scores for all five agents, chain trust-decay, and
    chain_vs_independent -- factored out of run_record for the same
    reuse reason as compute_grounding. Takes whichever a1..a5 responses
    the caller has (clean or fault-corrupted/rerun), and mechanically
    recomputes verification (agents/verification.py::match_claims)
    against exactly those inputs -- correctness under fault injection
    falls straight out of that: corrupted claims simply match/mismatch
    differently, no separate fault-aware branch needed here.
    """
    chain_claims = [*a1.claims, *a2.claims, *a3.claims]
    chain_vs_a4 = counts_by_agent(match_claims(chain_claims, a4.claims))
    a4_vs_chain = counts_by_agent(match_claims(a4.claims, chain_claims))

    features = record.features
    trigger_reason_values = {
        f"{TRIGGER_FEATURE_PREFIX}{tr.feature}": tr.observed_value for tr in record.trigger_reasons
    }
    trust_scores: Dict[str, TrustScore] = {
        "a1": compute_trust_score(
            agent="a1", record_id=record.flow_id, claims=a1.claims, features=features,
            verification_counts=chain_vs_a4.get("a1", VerificationCounts()),
            trigger_reason_values=trigger_reason_values, weights=weights,
        ),
        "a2": compute_trust_score(
            agent="a2", record_id=record.flow_id, claims=a2.claims, features=features,
            verification_counts=chain_vs_a4.get("a2", VerificationCounts()),
            trigger_reason_values=trigger_reason_values, weights=weights,
        ),
        "a3": compute_trust_score(
            agent="a3", record_id=record.flow_id, claims=a3.claims, features=features,
            verification_counts=chain_vs_a4.get("a3", VerificationCounts()),
            hypotheses=a3.hypotheses, trigger_reason_values=trigger_reason_values, weights=weights,
        ),
        "a4": compute_trust_score(
            agent="a4", record_id=record.flow_id, claims=a4.claims, features=features,
            verification_counts=a4_vs_chain.get("a4", VerificationCounts()),
            hypotheses=a4.hypotheses, trigger_reason_values=trigger_reason_values, weights=weights,
        ),
        "a5": _a5_trust_score(record.flow_id, a5, weights),
    }

    chain_scores = [trust_scores["a1"], trust_scores["a2"], trust_scores["a3"]]
    trust_decay_chain = compute_trust_decay(chain_scores)
    trust_decay_with_a5 = compute_trust_decay([*chain_scores, trust_scores["a5"]])
    cvi = chain_vs_independent(chain_scores, trust_scores["a4"])
    return trust_scores, trust_decay_chain, trust_decay_with_a5, cvi


def compute_contamination(
    a1: ClaimsResponse, a2: ClaimsResponse, a3: HypothesisResponse, a4: HypothesisResponse, a5: A5Response
) -> Dict[str, ContaminationReport]:
    return {
        "a1": scan_agent_response(a1),
        "a2": scan_agent_response(a2),
        "a3": scan_agent_response(a3),
        "a4": scan_agent_response(a4),
        "a5": scan_agent_response(a5),
    }


def _a5_trust_score(record_id: str, a5_resp: A5Response, weights: Tuple[float, float, float]) -> TrustScore:
    # A5 emits a verdict, not claims -- there is nothing of its own to
    # check for evidence-survival or corroborate against A4, so E and V
    # are undefined by construction (not a scoring failure); C is its
    # self-reported verdict confidence directly.
    return TrustScore(
        agent="a5",
        record_id=record_id,
        weights=weights,
        C=a5_resp.confidence,
        E=None,
        V=None,
        corroborated=0,
        contradicted=0,
        uncorroborated=0,
    )


def run_record(
    record: EscalationRecord,
    *,
    model: str = base.DEFAULT_MODEL,
    temperature: float = 0.7,
    run_index: int = 0,
    use_cache: bool = True,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> RecordResult:
    call_kwargs = dict(model=model, temperature=temperature, run_index=run_index, use_cache=use_cache)

    # Empirical grounding against Monday's real benign traffic -- computed
    # once per record, shared by every agent that needs it. Both the
    # reference distribution and the benign DataFrame are cached
    # in-process (controlplane/reference.py), so this costs nothing on the
    # 2nd..Nth call of a batch.
    grounding = compute_grounding(record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a4_future = pool.submit(
            a4_replication.run, record, grounding.empirical_grounding_block, grounding.available_features,
            grounding.known_reference_features, **call_kwargs,
        )

        a1_resp, a1_meta = a1_evidence.run(record, grounding.empirical_grounding_block, **call_kwargs)
        a2_resp, a2_meta = a2_behaviour.run(record, a1_resp, **call_kwargs)
        a3_resp, a3_meta = a3_hypotheses.run(
            record, a1_resp, a2_resp, grounding.empirical_grounding_block, grounding.available_features,
            grounding.known_reference_features, **call_kwargs,
        )

        a4_resp, a4_meta = a4_future.result()

    hypothesis_support, empirical_support_lines = compute_hypothesis_support(
        record, a3_resp, a4_resp, grounding.benign_df
    )

    a5_resp, a5_meta, a5_clamped = a5_verdict.run(
        record, a1_resp, a2_resp, a3_resp, a4_resp,
        empirical_support_lines, hypothesis_support, **call_kwargs,
    )

    trust_scores, trust_decay_chain, trust_decay_with_a5, cvi = compute_trust_and_decay(
        record, a1_resp, a2_resp, a3_resp, a4_resp, a5_resp, weights
    )
    contamination = compute_contamination(a1_resp, a2_resp, a3_resp, a4_resp, a5_resp)

    return RecordResult(
        record_id=record.flow_id,
        a1=a1_resp,
        a2=a2_resp,
        a3=a3_resp,
        a4=a4_resp,
        a5=a5_resp,
        trust_scores=trust_scores,
        trust_decay_chain=trust_decay_chain,
        trust_decay_with_a5=trust_decay_with_a5,
        chain_vs_independent=cvi,
        contamination=contamination,
        call_metadata={"a1": a1_meta, "a2": a2_meta, "a3": a3_meta, "a4": a4_meta, "a5": a5_meta},
        nearest_neighbours=grounding.nearest_neighbours,
        hypothesis_support=hypothesis_support,
        a5_plausibility_clamped=a5_clamped,
    )


def run_batch(
    records: List[EscalationRecord],
    *,
    model: str = base.DEFAULT_MODEL,
    temperature: float = 0.7,
    run_index: int = 0,
    use_cache: bool = True,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> Iterator[RecordResult]:
    for record in records:
        yield run_record(
            record,
            model=model,
            temperature=temperature,
            run_index=run_index,
            use_cache=use_cache,
            weights=weights,
        )


def append_result_jsonl(path: Union[str, Path], result: RecordResult) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict()) + "\n")
