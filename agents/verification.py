"""Mechanical chain-vs-A4 claim matching -- no LLM involvement.

The matching unit is a single (claim, referenced feature) pair, not a
whole claim: relation is asserted per-feature, so "same feature" only
makes sense checked at that granularity. For a source claim's reference
to feature X:

- some claim on the other side also references X with a *compatible*
  relation -> corroborated
- some claim on the other side references X with an *opposing*
  relation -> contradicted
- nothing on the other side references X at all -> uncorroborated

A high uncorroborated count means the other side simply looked
elsewhere, not that it disagreed -- keep the two apart when reading
results (see trust.py / eval/agent_eval.py).

``match_claims`` is symmetric in its two arguments, so it's used both
directions: chain claims (grouped by claim_id prefix -- a1/a2/a3)
checked against A4, and A4's own claims checked against the combined
chain, giving A4 its own V score for trust.py's T_4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from agents.schema import Claim, Relation


class MatchOutcome(str, Enum):
    CORROBORATED = "corroborated"
    CONTRADICTED = "contradicted"
    UNCORROBORATED = "uncorroborated"


#: relations directly opposed in the sense that both can't hold at once.
_OPPOSITES: Dict[Relation, Relation] = {
    Relation.GREATER_THAN_TYPICAL: Relation.LESS_THAN_TYPICAL,
    Relation.LESS_THAN_TYPICAL: Relation.GREATER_THAN_TYPICAL,
    Relation.PRESENT: Relation.ABSENT,
    Relation.ABSENT: Relation.PRESENT,
}
#: EQUALS conflicts with a claimed direction away from typical, but has
#: no single fixed "opposite" relation of its own.
_EQUALS_CONFLICTS = frozenset({Relation.GREATER_THAN_TYPICAL, Relation.LESS_THAN_TYPICAL})


def _compare_relations(a: Relation, b: Relation) -> MatchOutcome:
    if a == b:
        return MatchOutcome.CORROBORATED
    if _OPPOSITES.get(a) == b:
        return MatchOutcome.CONTRADICTED
    if a == Relation.EQUALS and b in _EQUALS_CONFLICTS:
        return MatchOutcome.CONTRADICTED
    if b == Relation.EQUALS and a in _EQUALS_CONFLICTS:
        return MatchOutcome.CONTRADICTED
    # e.g. PRESENT vs GREATER_THAN_TYPICAL: related but not a direct
    # logical clash -- treated as "didn't address the same question",
    # not as agreement or disagreement.
    return MatchOutcome.UNCORROBORATED


@dataclass(frozen=True)
class MatchResult:
    source_claim_id: str
    feature: str
    source_relation: Relation
    outcome: MatchOutcome
    matched_claim_id: Optional[str] = None
    matched_relation: Optional[Relation] = None


def match_claims(source_claims: Sequence[Claim], reference_claims: Sequence[Claim]) -> List[MatchResult]:
    """Match every (claim, feature) pair in ``source_claims`` against
    ``reference_claims``. Not symmetric in its *output* (matching X
    against Y answers "does Y corroborate X", not vice versa), but
    callable either direction on the same two claim sets."""
    by_feature: Dict[str, List[Tuple[str, Relation]]] = {}
    for claim in reference_claims:
        for ref in claim.referenced_features:
            by_feature.setdefault(ref.name, []).append((claim.claim_id, ref.relation))

    results: List[MatchResult] = []
    for claim in source_claims:
        for ref in claim.referenced_features:
            candidates = by_feature.get(ref.name, [])
            if not candidates:
                results.append(
                    MatchResult(claim.claim_id, ref.name, ref.relation, MatchOutcome.UNCORROBORATED)
                )
                continue
            best_outcome = MatchOutcome.UNCORROBORATED
            best_id = best_rel = None
            for cand_id, cand_rel in candidates:
                outcome = _compare_relations(ref.relation, cand_rel)
                if outcome == MatchOutcome.CORROBORATED:
                    best_outcome, best_id, best_rel = outcome, cand_id, cand_rel
                    break  # one corroboration is enough, prefer it over a mixed bag
                if outcome == MatchOutcome.CONTRADICTED and best_outcome != MatchOutcome.CORROBORATED:
                    best_outcome, best_id, best_rel = outcome, cand_id, cand_rel
            results.append(
                MatchResult(claim.claim_id, ref.name, ref.relation, best_outcome, best_id, best_rel)
            )
    return results


@dataclass
class VerificationCounts:
    corroborated: int = 0
    contradicted: int = 0
    uncorroborated: int = 0

    @property
    def total(self) -> int:
        return self.corroborated + self.contradicted + self.uncorroborated

    @property
    def V(self) -> float:
        return self.corroborated / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "corroborated": self.corroborated,
            "contradicted": self.contradicted,
            "uncorroborated": self.uncorroborated,
            "V": self.V if self.total else None,
        }


def counts_by_agent(results: Sequence[MatchResult]) -> Dict[str, VerificationCounts]:
    """Groups match results by the source claim's agent prefix
    (``claim_id.split("_", 1)[0]``, e.g. "a1_c3" -> "a1")."""
    out: Dict[str, VerificationCounts] = {}
    for r in results:
        agent = r.source_claim_id.split("_", 1)[0]
        counts = out.setdefault(agent, VerificationCounts())
        if r.outcome == MatchOutcome.CORROBORATED:
            counts.corroborated += 1
        elif r.outcome == MatchOutcome.CONTRADICTED:
            counts.contradicted += 1
        else:
            counts.uncorroborated += 1
    return out


def render_verification_summary(by_agent: Dict[str, VerificationCounts]) -> str:
    """Plain-text rendering for A5's prompt."""
    if not by_agent:
        return "  (no checkable claims)"
    lines = []
    for agent in sorted(by_agent):
        c = by_agent[agent]
        if c.total == 0:
            lines.append(f"  {agent}: no checkable claims")
            continue
        lines.append(
            f"  {agent}: {c.corroborated} corroborated, {c.contradicted} contradicted, "
            f"{c.uncorroborated} not addressed by the other side (of {c.total} checkable claims)"
        )
    return "\n".join(lines)
