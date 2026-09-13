"""Structured-claim schema shared by every agent.

Trust scoring works by checking an agent's claims against the record in
code. That's only possible if claims are machine-readable — hence a
closed ``relation`` vocabulary (never free-text comparison words) and a
``referenced_features`` list naming exactly which feature each claim is
about. Every agent response is validated against one of the Pydantic
models below via the model provider's structured-output mode
(``agents/base.py``); a response that doesn't parse is rejected and
retried, never silently coerced.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Relation(str, Enum):
    GREATER_THAN_TYPICAL = "greater_than_typical"
    LESS_THAN_TYPICAL = "less_than_typical"
    EQUALS = "equals"
    ABSENT = "absent"
    PRESENT = "present"


#: relations that make a claim about a specific numeric value, and so
#: require `asserted_value` to be checkable against the record.
_VALUE_RELATIONS = frozenset(
    {Relation.GREATER_THAN_TYPICAL, Relation.LESS_THAN_TYPICAL, Relation.EQUALS}
)


class FeatureReference(BaseModel):
    name: str
    asserted_value: Optional[float] = None
    relation: Relation

    @model_validator(mode="after")
    def _value_required_for_value_relations(self) -> "FeatureReference":
        if self.relation in _VALUE_RELATIONS and self.asserted_value is None:
            raise ValueError(
                f"relation {self.relation.value!r} on feature {self.name!r} "
                "requires asserted_value"
            )
        return self


class Claim(BaseModel):
    claim_id: str
    statement: str
    referenced_features: List[FeatureReference] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class PredictedRange(BaseModel):
    """One (feature, expected range) pair in a hypothesis's predicted
    feature profile -- the mechanically checkable form of "if this
    hypothesis is true, this feature should look like X" (agents/
    grounding.py queries Monday's benign reference data for how many real
    benign flows actually match this profile). ``feature`` must be the
    TRIGGER_FEATURE_PREFIX-prefixed name of a Tier-1/per-source feature
    this project has a benign reference distribution for (agents/
    validators.py checks this dynamically) -- not any of the 77
    CICFlowMeter features, since there's no benign reference for that
    space (see controlplane/reference.py's module docstring for why)."""

    feature: str
    expected_min: Optional[float] = None
    expected_max: Optional[float] = None

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> "PredictedRange":
        if self.expected_min is None and self.expected_max is None:
            raise ValueError(f"{self.feature}: predicted range needs expected_min and/or expected_max")
        if self.expected_min is not None and self.expected_max is not None and self.expected_min > self.expected_max:
            raise ValueError(f"{self.feature}: expected_min > expected_max")
        return self


class Hypothesis(BaseModel):
    hypothesis_id: str
    description: str
    benign: bool
    prior_plausibility: float = Field(ge=0.0, le=1.0)
    #: A concrete, checkable prediction this hypothesis makes about the
    #: evidence if it's true (e.g. "if this is a monitoring system,
    #: inter-arrival times should be regular") -- kept as the
    #: human-readable statement of what predicted_feature_profile encodes
    #: mechanically; the two should agree, but only the profile below is
    #: actually queried against real data.
    prediction: str = Field(min_length=10)
    #: The mechanically checkable form of the prediction above: which
    #: Tier-1 feature(s), and what range each should fall in, if this
    #: hypothesis is true. agents/grounding.py queries Monday's benign
    #: flows against this profile BEFORE A5 ever sees the hypothesis,
    #: attaching how many real benign flows actually match it -- a
    #: hypothesis with zero matches is empirically unsupported, a fact,
    #: not an opinion A5 has to form on its own. Replaces the earlier
    #: referenced_features field (a bare name list) with something
    #: actually testable: a live stop-point-2 record showed a hypothesis
    #: reframing a source opening 4,614 flows as "automated polling"
    #: while citing that exact number as *supporting* evidence -- an
    #: empirical check against real benign traffic (Monday's observed max
    #: for flows_per_src is 1,993) would have caught that immediately.
    predicted_feature_profile: List[PredictedRange] = Field(min_length=1)
    supporting_claim_ids: List[str] = Field(default_factory=list)
    contradicting_claim_ids: List[str] = Field(default_factory=list)


#: Self-reported counterparts to agents/trust.py's code-computed C_i/E_i/V_i
#: -- same [0,1] scale and the same intent, but asked of the model
#: directly instead of derived mechanically. Kept side by side with the
#: code-computed scores everywhere (never replacing them): the
#: comparison between the two -- does the model's own sense of how well
#: its evidence holds up, or how well it expects to be corroborated,
#: actually track the mechanical check? -- is itself a finding, not
#: just plumbing for the poster figures that consume the self-reported
#: side as the primary trust signal.
_SELF_REPORT_DOC = (
    "Self-reported {name}, [0,1] -- your own estimate, not computed for "
    "you. Distinct from agents/trust.py's code-computed counterpart, "
    "which checks this mechanically after the fact; report your honest "
    "estimate here even if you expect the two to disagree."
)


class ClaimsResponse(BaseModel):
    """A1 / A2's response shape: claims, plus this call's self-reported
    trust triad (mirrors A3/A4/A5's own confidence field, extended to
    all three components -- see _SELF_REPORT_DOC above)."""

    claims: List[Claim]
    #: overall self-rated confidence in this analysis as a whole (distinct
    #: from each individual claim's own `confidence` field).
    confidence: float = Field(ge=0.0, le=1.0)
    #: self-rated estimate of how well YOUR OWN claims above would survive
    #: a mechanical check against the record's actual data (mirrors
    #: agents/trust.py's code-computed E_i).
    evidence_support: float = Field(ge=0.0, le=1.0)
    #: self-rated estimate of how well your claims would be corroborated
    #: by an independent, blind re-analysis of this same flow (mirrors
    #: agents/trust.py's code-computed V_i, which checks this mechanically
    #: against A4's actual output -- something you cannot see yet).
    verification: float = Field(ge=0.0, le=1.0)


class HypothesisResponse(BaseModel):
    """A3's response shape (and A4's, since A4 performs all three
    analyses — evidence, behaviour, hypotheses — in one pass and must
    be directly comparable to the chain's combined output). Same
    self-reported trust triad as ClaimsResponse -- see its docstring."""

    claims: List[Claim]
    hypotheses: List[Hypothesis]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_support: float = Field(ge=0.0, le=1.0)
    verification: float = Field(ge=0.0, le=1.0)


class VerdictLabel(str, Enum):
    CONSISTENT_WITH_BENIGN = "consistent_with_benign"
    ANOMALOUS_BUT_EXPLICABLE = "anomalous_but_explicable"
    ANOMALOUS_AND_UNEXPLAINED = "anomalous_and_unexplained"


class A5Response(BaseModel):
    """A5's response shape. No categorical verdict: A5 reports
    ``benign_plausibility``, a continuous [0,1] estimate of how plausible
    the strongest benign hypothesis that survived scrutiny is, given the
    chain's and A4's evidence -- and now the empirical support counts
    computed against Monday's real benign traffic (agents/grounding.py),
    attached to every hypothesis before A5 ever sees it. The three-way
    ``VerdictLabel`` is derived from this score in code (``derive_verdict``,
    threshold-swept), never asserted by the model directly.

    ``credited_hypothesis_id`` names the single hypothesis (from either
    side) A5's plausibility estimate is actually resting on, or None if no
    hypothesis survives well enough to be credited. This is not just for
    the rationale's sake: agents/grounding.py::apply_empirical_plausibility_cap
    looks up that hypothesis's already-computed empirical support (never
    A5's own restatement of it, which can't be trusted to be accurate) and
    clamps benign_plausibility to at most 0.3 in code if it has zero
    matching benign flows -- enforced mechanically, not requested in the
    prompt.
    """

    benign_plausibility: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    #: self-rated estimate of how well your verdict's cited evidence
    #: would survive a mechanical check against the record (mirrors
    #: agents/trust.py's code-computed E_i; see ClaimsResponse's
    #: docstring for the self-report/code-computed distinction).
    evidence_support: float = Field(ge=0.0, le=1.0)
    #: self-rated estimate of how well your verdict is actually
    #: supported by the corroboration you were shown between the chain
    #: and A4 (mirrors agents/trust.py's code-computed V_i -- unlike
    #: A1-A4, A5 DOES see the real mechanical corroboration summary
    #: before answering, so this one has real evidence to be self-rated
    #: against, not a blind guess).
    verification: float = Field(ge=0.0, le=1.0)
    credited_hypothesis_id: Optional[str] = None
    cited_claim_ids: List[str]
    rationale: str


#: Default two-threshold mapping from benign_plausibility to the
#: three-way VerdictLabel, for reporting/confusion-matrix purposes only.
#: These are a starting point, not a finding -- explicitly meant to be
#: swept (eval/agent_eval.py), the same way step 1 swept the selector's
#: own escalation percentile rather than trusting one fixed cut.
DEFAULT_LOW_PLAUSIBILITY_THRESHOLD = 0.3
DEFAULT_HIGH_PLAUSIBILITY_THRESHOLD = 0.7


def derive_verdict(
    benign_plausibility: float,
    low_threshold: float = DEFAULT_LOW_PLAUSIBILITY_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_PLAUSIBILITY_THRESHOLD,
) -> "VerdictLabel":
    if benign_plausibility < low_threshold:
        return VerdictLabel.ANOMALOUS_AND_UNEXPLAINED
    if benign_plausibility >= high_threshold:
        return VerdictLabel.CONSISTENT_WITH_BENIGN
    return VerdictLabel.ANOMALOUS_BUT_EXPLICABLE


class BaselineVerdictResponse(BaseModel):
    """v1 single-LLM control response shape -- a categorical verdict
    asked directly, rather than a continuous score. Superseded by
    ``BaselineResponse`` (v2): asking for a category directly isn't
    comparable to A5's continuous ``benign_plausibility``, only to a
    label *derived* from it, which throws away resolution the head-to-
    head comparison needs. Kept importable, not used by any live code
    path (agents/baseline.py now targets BaselineResponse)."""

    verdict: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    cited_features: List[str]


class BaselineResponse(BaseModel):
    """v2 single-LLM control response shape: continuous
    ``benign_plausibility`` on the same [0,1] scale as ``A5Response``,
    so its output is directly comparable to A5's own number -- not just
    to a three-way label derived from it. ``VerdictLabel`` can still be
    derived from this score with the same ``derive_verdict`` function,
    for confusion-matrix reporting, but the raw score is the primary
    output (agents/baseline.py). No claim_ids -- a single pass produces
    no structured claims to cite, so it cites feature names directly,
    same as v1."""

    benign_plausibility: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    cited_features: List[str]
    rationale: str


#: agents whose response is a bare claim list.
CLAIM_ONLY_AGENTS = frozenset({"a1", "a2"})
#: agents whose response is claims + hypotheses (A3 chain-fed, A4 blind).
HYPOTHESIS_AGENTS = frozenset({"a3", "a4"})
