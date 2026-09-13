"""Empirical grounding for hypotheses: query Monday's real benign traffic
(controlplane/reference.py) to find out, mechanically, whether a
hypothesis's predicted feature profile is actually supported by any real
benign flow -- computed in code, before A5 ever sees the hypothesis, and
enforced in code afterward (a hypothesis with zero matching benign flows
cannot support a plausibility above a fixed cap, regardless of what the
prompt asked for).

This is the direct fix for what earlier stop points kept finding: agents
inventing plausible-sounding benign stories the data doesn't actually
support (a source opening 4,614 flows reframed as "automated polling",
credited at 0.85 plausibility, when Monday's observed benign max for
flows_per_src is 1,993 -- zero real benign flows look anything like that).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from controlplane.record import EscalationRecord
from controlplane.reference import (
    FeatureReference,
    NearestNeighbourResult,
    find_nearest_benign_flows,
    render_nearest_neighbour_line,
    render_reference_line,
)
from agents.prompts.render import TRIGGER_FEATURE_PREFIX
from agents.schema import A5Response, Hypothesis

#: A5's benign_plausibility may not exceed this when the hypothesis it
#: credits has zero empirically-matching benign flows -- fixed, not
#: prompt-negotiable (agents/schema.py::A5Response's own docstring).
ZERO_SUPPORT_PLAUSIBILITY_CAP = 0.3


def _strip_prefix(feature: str) -> str:
    return feature[len(TRIGGER_FEATURE_PREFIX):] if feature.startswith(TRIGGER_FEATURE_PREFIX) else feature


def observed_tier1_values(record: EscalationRecord, prefixed: bool = True) -> Dict[str, float]:
    """This flow's own observed Tier-1 values, from its trigger_reasons --
    the only Tier-1 values this flow actually has (the record doesn't
    carry the full ~21-feature Tier-1 vector, only whatever crossed a
    threshold). ``prefixed=True`` (the default) matches
    predicted_feature_profile's feature naming and the existing
    trigger_reason_values convention used elsewhere in the pipeline for
    claim evidence-checking; ``prefixed=False`` gives bare names, which
    is what controlplane.reference's DataFrame-based functions expect
    (their columns are the selector's own bare Tier-1 names)."""
    if prefixed:
        return {f"{TRIGGER_FEATURE_PREFIX}{tr.feature}": tr.observed_value for tr in record.trigger_reasons}
    return {tr.feature: tr.observed_value for tr in record.trigger_reasons}


def reference_feature_names(reference: Dict[str, FeatureReference]) -> Set[str]:
    """The closed, TRIGGER_FEATURE_PREFIX-prefixed vocabulary a
    predicted_feature_profile entry may legally name -- exactly the
    features this project has real benign data for. Independent of any
    particular record (unlike agents/prompts/render.py::known_feature_names,
    which also covers a record's own CICFlowMeter features for claim
    evidence-checking, a separate mechanism this doesn't touch)."""
    return {f"{TRIGGER_FEATURE_PREFIX}{f}" for f in reference}


@dataclass(frozen=True)
class HypothesisSupport:
    hypothesis_id: str
    matching_profile_count: int
    #: None when this flow has no observed value for any predicted
    #: feature (so "within 2x of observed" isn't a checkable question at
    #: all) -- distinct from 0, which means it WAS checked and none of
    #: the matching flows were close.
    close_to_observed_count: Optional[int]
    checked_features: List[str]
    benign_population_size: int

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "matching_profile_count": self.matching_profile_count,
            "close_to_observed_count": self.close_to_observed_count,
            "checked_features": self.checked_features,
            "benign_population_size": self.benign_population_size,
        }


def evaluate_hypothesis_support(
    hypothesis: Hypothesis,
    trigger_reason_values: Dict[str, float],
    benign_df: pd.DataFrame,
) -> HypothesisSupport:
    """``trigger_reason_values``: TRIGGER_FEATURE_PREFIX-prefixed feature
    name -> this flow's own observed value (from its trigger_reasons --
    the only Tier-1 values this flow actually has; see
    controlplane/reference.py::NearestNeighbourResult's docstring for why
    that's the full extent of what's available)."""
    profile = hypothesis.predicted_feature_profile
    mask = pd.Series(True, index=benign_df.index)
    any_feature_present = False
    for p in profile:
        bare = _strip_prefix(p.feature)
        if bare not in benign_df.columns:
            continue
        any_feature_present = True
        col = benign_df[bare]
        col_mask = col.notna()
        if p.expected_min is not None:
            col_mask &= col >= p.expected_min
        if p.expected_max is not None:
            col_mask &= col <= p.expected_max
        mask &= col_mask

    if not any_feature_present:
        return HypothesisSupport(hypothesis.hypothesis_id, 0, None, [], len(benign_df))

    matching = benign_df[mask]
    matching_count = int(len(matching))

    checked_features: List[str] = []
    close_mask = pd.Series(True, index=matching.index)
    for p in profile:
        bare = _strip_prefix(p.feature)
        observed = trigger_reason_values.get(p.feature)
        if observed is None or bare not in matching.columns:
            continue
        checked_features.append(p.feature)
        col = matching[bare]
        if observed == 0:
            close_mask &= col == 0
        else:
            lo, hi = sorted([observed / 2.0, observed * 2.0])
            close_mask &= col.between(lo, hi)

    close_count = int(close_mask.sum()) if checked_features else None

    return HypothesisSupport(
        hypothesis_id=hypothesis.hypothesis_id,
        matching_profile_count=matching_count,
        close_to_observed_count=close_count,
        checked_features=checked_features,
        benign_population_size=int(len(benign_df)),
    )


def evaluate_all(
    hypotheses: List[Hypothesis], trigger_reason_values: Dict[str, float], benign_df: pd.DataFrame
) -> Dict[str, HypothesisSupport]:
    return {
        h.hypothesis_id: evaluate_hypothesis_support(h, trigger_reason_values, benign_df)
        for h in hypotheses
    }


def compute_nearest_neighbours(
    record: EscalationRecord, benign_df: pd.DataFrame, reference: Dict[str, FeatureReference], k: int = 10
) -> NearestNeighbourResult:
    return find_nearest_benign_flows(observed_tier1_values(record, prefixed=False), benign_df, reference, k=k)


def render_available_profile_features(reference: Dict[str, FeatureReference]) -> str:
    """The exact closed vocabulary predicted_feature_profile entries may
    use -- shown directly in A3/A4's prompts so the model isn't guessing
    at what's checkable (agents/validators.py::
    validate_hypothesis_predictions_are_flow_level rejects anything else
    mechanically, but showing the list up front means fewer wasted
    retries)."""
    return ", ".join(sorted(reference_feature_names(reference)))


def render_empirical_grounding_block(
    record: EscalationRecord,
    reference: Dict[str, FeatureReference],
    nn_result: NearestNeighbourResult,
) -> str:
    """The block shown in every agent prompt (A1, A3, A4, A5 -- not A2,
    which never sees trigger_reasons and reasons purely from the full
    feature list + upstream claims): what does normal look like on this
    network, for every feature this flow actually has an observed
    Tier-1 value for (its own trigger_reasons -- the record doesn't carry
    the full Tier-1 vector, only what crossed a threshold, so grounding
    is necessarily scoped to that subset; see controlplane/reference.py
    and NearestNeighbourResult's docstrings for the full reasoning)."""
    if not record.trigger_reasons:
        return "  (no trigger-flagged features to ground against benign reference data)"
    lines = ["  Benign reference (Monday's real network traffic, Tier-1 measurements only):"]
    for tr in record.trigger_reasons:
        ref = reference.get(tr.feature)
        lines.append("  " + render_reference_line(f"{TRIGGER_FEATURE_PREFIX}{tr.feature}", tr.observed_value, ref))
    lines.append("")
    lines.append("  Nearest-neighbour grounding (10 closest real benign flows, normalised distance):")
    lines.append(render_nearest_neighbour_line(nn_result))
    return "\n".join(lines)


def render_hypothesis_support(support: HypothesisSupport) -> str:
    if support.matching_profile_count == 0:
        verdict = (
            f"EMPIRICALLY UNSUPPORTED -- zero of {support.benign_population_size} benign flows "
            "observed on this network match this predicted profile"
        )
    else:
        verdict = f"{support.matching_profile_count} benign flow(s) match this predicted profile"
        if support.checked_features:
            verdict += (
                f", of which {support.close_to_observed_count} are within 2x of this flow's own "
                f"observed values on {', '.join(support.checked_features)}"
            )
        else:
            verdict += (
                " (this flow has no observed value for any predicted feature, so closeness to it "
                "could not be checked)"
            )
    return f"empirical support: {verdict}"


def apply_empirical_plausibility_cap(
    response: A5Response,
    supports_by_id: Dict[str, HypothesisSupport],
    cap: float = ZERO_SUPPORT_PLAUSIBILITY_CAP,
) -> Tuple[A5Response, bool]:
    """Enforced in code, per spec: a hypothesis with zero empirically
    matching benign flows cannot support a plausibility above `cap`. Looks
    up the ALREADY-COMPUTED support for whichever hypothesis A5 names as
    credited_hypothesis_id -- never A5's own restatement of the numbers,
    which can't be trusted to be accurate. Returns (possibly-clamped
    response, whether a clamp was applied) so callers can report the rate
    honestly rather than silently overriding the model.
    """
    if response.credited_hypothesis_id is None:
        return response, False
    support = supports_by_id.get(response.credited_hypothesis_id)
    if support is None:
        return response, False
    if support.matching_profile_count == 0 and response.benign_plausibility > cap:
        clamped = response.model_copy(update={"benign_plausibility": cap})
        return clamped, True
    return response, False
