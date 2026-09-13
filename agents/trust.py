"""Per-agent trust scoring.

Three components, all defined for one agent on one record:

- C_i -- confidence. Self-reported by the model, mean across its own
  claims. The only component the model controls.
- E_i -- evidence. Fraction of claims that survive programmatic
  checking against the record: the referenced feature exists, and its
  asserted_value matches within tolerance. Computed in code.
- V_i -- verification. Corroboration rate against the blind-replication
  agent (agents/verification.py). Computed in code.

T_i = w_c*C_i + w_e*E_i + w_v*V_i, weights configurable
(agents/trust.py's DEFAULT_WEIGHTS), summing to 1. Evidence carries the
most weight by default because it's the most objective of the three;
confidence the least because it's the only self-reported term. Weights
are a free parameter -- see eval/agent_eval.py's sensitivity grid over
at least {equal, evidence-dominant, confidence-dominant, the defaults}.

Trust decay (TD) is computed over the chain only (A1->A2->A3, optionally
->A5) and must never include A4: A4 doesn't inherit from A3, so a
"drop" between A3 and A4 wouldn't measure decay, it would be a category
error. See compute_trust_decay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from agents.schema import Claim, Hypothesis
from agents.verification import VerificationCounts

#: (w_c, w_e, w_v) -- must sum to 1.
DEFAULT_WEIGHTS: Tuple[float, float, float] = (0.2, 0.5, 0.3)

_DEFAULT_TOLERANCE = 1e-3  # relative tolerance for asserted_value vs the record


@dataclass
class ClaimCheck:
    claim_id: str
    hallucinated_features: List[str] = field(default_factory=list)
    factual_errors: List[Tuple[str, float, float]] = field(default_factory=list)  # (feature, asserted, actual)

    @property
    def survives(self) -> bool:
        return not self.hallucinated_features and not self.factual_errors

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "hallucinated_features": self.hallucinated_features,
            "factual_errors": [
                {"feature": f, "asserted": a, "actual": v} for f, a, v in self.factual_errors
            ],
            "survives": self.survives,
        }


def _approx_equal(a: float, b: float, tol: float) -> bool:
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tol


def check_claim(
    claim: Claim,
    features: Dict[str, float],
    trigger_reason_values: Optional[Dict[str, float]] = None,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> ClaimCheck:
    """A claim referencing a feature absent from the record is a
    hallucination; a claim whose asserted_value doesn't match the
    record (for relations that carry a value at all) is a factual
    error. Both detectable without any LLM judgement -- this is the
    entire point of forcing claims into a structured, checkable shape.

    ``trigger_reason_values`` (``{TriggerReason.feature: observed_value}``,
    built by the caller from the same record's trigger reasons) is
    checked as a second, independent source of truth alongside
    ``features``. This is load-bearing, not cosmetic: the selector's
    trigger-reason feature names (Tier-1/per-source, e.g.
    ``flow_duration``, ``flow_iat_mean``) and the CICFlowMeter-derived
    ``features`` dict can share an identical name while disagreeing by
    up to ~1e6x in scale (microseconds vs seconds -- confirmed live,
    ratio exactly 1,000,000.0 for ``flow_duration`` on one stop-point-2
    record) and, beyond the unit gap, by a further nontrivial margin
    even after correcting for it (the data plane's O(1) approximation
    vs the control plane's full CICFlowMeter recomputation are not
    required to agree exactly). The prompt shows both namespaces to the
    agent under the same names (see agents/prompts/render.py), so a
    claim quoting the trigger reason's own number verbatim is accurate,
    not hallucinated -- checking only ``features`` flagged ~20 correct
    A1 claims as hallucinations/factual errors on the stop-point-2
    sample before this was found and fixed. A claim survives if its
    asserted_value matches *either* source; a name absent from *both*
    is the real hallucination case.
    """
    trigger_reason_values = trigger_reason_values or {}
    check = ClaimCheck(claim_id=claim.claim_id)
    for ref in claim.referenced_features:
        in_features = ref.name in features
        in_triggers = ref.name in trigger_reason_values
        if not in_features and not in_triggers:
            check.hallucinated_features.append(ref.name)
            continue
        if ref.asserted_value is not None:
            candidates = []
            if in_features:
                candidates.append(features[ref.name])
            if in_triggers:
                candidates.append(trigger_reason_values[ref.name])
            if not any(_approx_equal(ref.asserted_value, actual, tolerance) for actual in candidates):
                # report against the CICFlowMeter value when both exist
                # (it's the more complete/precise source), else whichever
                # single source the name came from.
                reported_actual = features[ref.name] if in_features else trigger_reason_values[ref.name]
                check.factual_errors.append((ref.name, ref.asserted_value, reported_actual))
    return check


def check_claims(
    claims: Sequence[Claim],
    features: Dict[str, float],
    trigger_reason_values: Optional[Dict[str, float]] = None,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> List[ClaimCheck]:
    return [check_claim(c, features, trigger_reason_values, tolerance) for c in claims]


def evidence_score(
    claims: Sequence[Claim],
    features: Dict[str, float],
    trigger_reason_values: Optional[Dict[str, float]] = None,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> Optional[float]:
    """E_i. None (undefined) only when the agent made zero claims at
    all -- see confidence_score's docstring for when that can happen."""
    if not claims:
        return None
    checks = check_claims(claims, features, trigger_reason_values, tolerance)
    return sum(1 for c in checks if c.survives) / len(checks)


def confidence_score(claims: Sequence[Claim], hypotheses: Sequence[Hypothesis] = ()) -> Optional[float]:
    """C_i: mean self-reported confidence across the agent's claims.

    Falls back to the mean hypothesis prior_plausibility only when the
    agent emitted zero claims (A3's new claims are explicitly optional
    in its prompt, so this is a real, if hopefully rare, case) --
    prior_plausibility is the closest self-reported number available
    then, but it answers a different question (how plausible is this
    hypothesis in general, not how sure am I of this specific
    observation), so a record scored via this fallback is a weaker
    signal and should be flagged, not treated identically to a normal
    claims-based C_i (see TrustScore.degenerate).
    """
    if claims:
        return sum(c.confidence for c in claims) / len(claims)
    if hypotheses:
        return sum(h.prior_plausibility for h in hypotheses) / len(hypotheses)
    return None


@dataclass
class TrustScore:
    agent: str
    record_id: str
    weights: Tuple[float, float, float]
    C: Optional[float]
    E: Optional[float]
    V: Optional[float]
    corroborated: int
    contradicted: int
    uncorroborated: int
    used_hypothesis_fallback_for_C: bool = False

    @property
    def T(self) -> Optional[float]:
        w_c, w_e, w_v = self.weights
        parts, weight_sum = [], 0.0
        for w, x in ((w_c, self.C), (w_e, self.E), (w_v, self.V)):
            if x is not None:
                parts.append(w * x)
                weight_sum += w
        if weight_sum == 0.0:
            return None
        # Renormalise over whichever components are actually defined,
        # rather than treating a missing component as 0 -- a missing
        # component means "nothing to check" (e.g. zero claims that
        # record), not "checked and failed", and scoring it as 0 would
        # conflate the two. Only bites in the rare zero-claims case;
        # flagged via `degenerate` so it's visible in aggregate reporting.
        return sum(parts) / weight_sum

    @property
    def CTG(self) -> Optional[float]:
        """Confidence-trust gap: C_i - T_i. Positive means the agent was
        more confident than its claims turned out to justify."""
        if self.C is None or self.T is None:
            return None
        return self.C - self.T

    @property
    def degenerate(self) -> bool:
        return self.C is None or self.E is None or self.V is None

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "record_id": self.record_id,
            "weights": {"w_c": self.weights[0], "w_e": self.weights[1], "w_v": self.weights[2]},
            "C": self.C,
            "E": self.E,
            "V": self.V,
            "T": self.T,
            "CTG": self.CTG,
            "corroborated": self.corroborated,
            "contradicted": self.contradicted,
            "uncorroborated": self.uncorroborated,
            "degenerate": self.degenerate,
            "used_hypothesis_fallback_for_C": self.used_hypothesis_fallback_for_C,
        }


def compute_trust_score(
    *,
    agent: str,
    record_id: str,
    claims: Sequence[Claim],
    features: Dict[str, float],
    verification_counts: VerificationCounts,
    hypotheses: Sequence[Hypothesis] = (),
    trigger_reason_values: Optional[Dict[str, float]] = None,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> TrustScore:
    C = confidence_score(claims, hypotheses)
    E = evidence_score(claims, features, trigger_reason_values, tolerance)
    V = verification_counts.V if verification_counts.total else None
    return TrustScore(
        agent=agent,
        record_id=record_id,
        weights=weights,
        C=C,
        E=E,
        V=V,
        corroborated=verification_counts.corroborated,
        contradicted=verification_counts.contradicted,
        uncorroborated=verification_counts.uncorroborated,
        used_hypothesis_fallback_for_C=(not claims and bool(hypotheses)),
    )


@dataclass
class TrustDecay:
    drops: List[float]
    stage_labels: List[str]
    TD: float
    TD_normalised: float
    max_single_drop: float
    max_single_drop_stage: Optional[str]

    def to_dict(self) -> dict:
        return {
            "drops": self.drops,
            "stage_labels": self.stage_labels,
            "TD": self.TD,
            "TD_normalised": self.TD_normalised,
            "max_single_drop": self.max_single_drop,
            "max_single_drop_stage": self.max_single_drop_stage,
        }


def compute_trust_decay(chain_scores: Sequence[TrustScore]) -> Optional[TrustDecay]:
    """TD = sum of max(0, T_{i-1} - T_i) over adjacent chain stages.

    ``chain_scores`` must be given in chain order (e.g. [T_1, T_2, T_3]
    or [T_1, T_2, T_3, T_5]) and must NEVER include A4 -- A4 is a blind
    control with no dependency on A3, so a gap ending or starting at A4
    doesn't measure decay of anything. Returns None if fewer than 2
    scores are given, or any score's T is undefined (e.g. a degenerate
    zero-claims record) -- decay isn't meaningfully computable then.

    NOTE: because A3 inherits A1 and A2's framing (it's the third stage
    of a sequential chain fed the prior stages' full output), a low E_3
    may reflect inherited bad evidence rather than A3's own reasoning.
    That's exactly what decay is meant to expose, not a flaw in this
    metric -- but it means "T_3 dropped" should be read as "the chain's
    evidence quality dropped by stage 3", not necessarily "A3 reasoned
    worse than A1/A2 would have from the same fresh input".
    """
    if len(chain_scores) < 2:
        return None
    Ts = [s.T for s in chain_scores]
    if any(t is None for t in Ts):
        return None
    drops = [max(0.0, Ts[i - 1] - Ts[i]) for i in range(1, len(Ts))]
    labels = [f"{chain_scores[i-1].agent}->{chain_scores[i].agent}" for i in range(1, len(chain_scores))]
    TD = sum(drops)
    TD_norm = TD / len(drops)
    max_drop = max(drops)
    max_stage = labels[drops.index(max_drop)]
    return TrustDecay(
        drops=drops,
        stage_labels=labels,
        TD=TD,
        TD_normalised=TD_norm,
        max_single_drop=max_drop,
        max_single_drop_stage=max_stage,
    )


def chain_vs_independent(chain_scores: Sequence[TrustScore], a4_score: TrustScore) -> Optional[float]:
    """T_4 - mean(T_1, T_2, T_3): whether the whole evidence-gathering
    chain underperformed A4's single blind pass. ``chain_scores`` should
    be exactly [T_1, T_2, T_3] -- A5 excluded, since this compares
    evidence-gathering to evidence-gathering, not to the final verdict.
    If this is consistently positive across records, the chain is
    costing accuracy rather than adding depth -- a genuine negative
    result worth reporting as such, not something to explain away.
    """
    if a4_score.T is None:
        return None
    Ts = [s.T for s in chain_scores]
    if not Ts or any(t is None for t in Ts):
        return None
    return a4_score.T - (sum(Ts) / len(Ts))


#: sensitivity-analysis grid: at minimum equal weights, evidence-dominant,
#: confidence-dominant, plus the defaults -- see eval/agent_eval.py.
WEIGHT_GRID: Dict[str, Tuple[float, float, float]] = {
    "default": DEFAULT_WEIGHTS,
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "evidence_dominant": (0.1, 0.8, 0.1),
    "confidence_dominant": (0.8, 0.1, 0.1),
}
