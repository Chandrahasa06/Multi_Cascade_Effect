"""Restores empirical grounding for the DT-rule-based escalation
policy's blind agent runs, WITHOUT reintroducing the escalation hint
(class_predicted/rule_id/priority/reason) and WITHOUT depending on
dataplane/selector.py's Tier-1 trigger-reason concept, which this
policy's EscalationRecords don't have (see eval/run_blind_pipeline.py --
every record built there carries trigger_reasons=[]).

STATUS.md's "fifth silent-disable": the first blind run
(results/agent_scoring_20.md v1) found close_to_observed_count undefined
for all 116 hypotheses evaluated, because that quantity is derived
entirely from record.trigger_reasons (agents/grounding.py::
observed_tier1_values). A generic support number
(matching_profile_count) survived and LOOKED like grounding while
answering a different, uninformative question: "does any benign flow
fit this profile" (258,258 of 566,864 did, for the traced DoS Hulk
record) instead of "is this SPECIFIC flow unusual".

Fix: build the benign comparison set directly in this policy's own
10-feature CICFlowMeter escalation-feature space (dataplane/dt_rules.py's
own features, compiled from tree.txt), using real BENIGN rows from the
SAME CICIDS2017 CSV pool the escalation policy itself reads
(eval.escalation_data.load_pool()) -- 2,273,097 real benign flows,
already cached, no new extraction needed. This function's only input is
the flow's own `features` dict (its real CICFlowMeter values); it never
reads class_predicted, rule_id, priority, or reason, and cannot
(tests/test_escalation_grounding.py asserts this both by signature
inspection and by checking identical output when those fields are
scrambled upstream).

An empty or too-small neighbourhood is reported as UNGROUNDED, never
silently treated as "close to benign" -- see
apply_ungrounded_neighbourhood_cap, an ADDITIONAL cap layered alongside
(not instead of) agents.grounding.apply_empirical_plausibility_cap.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import pandas as pd

from agents.grounding import HypothesisSupport
from agents.schema import A5Response

#: dataplane/dt_rules.py's own 10 features (compiled from tree.txt) --
#: these are the ONLY features this module ever reads from a record, and
#: they come straight from record.features (real CICFlowMeter values),
#: never from the escalation digest.
ESCALATION_FEATURES = (
    "Bwd Packets/s",
    "Flow IAT Mean",
    "Fwd Packet Length Max",
    "Init_Win_bytes_backward",
    "Init_Win_bytes_forward",
    "Packet Length Mean",
    "Subflow Fwd Packets",
    "Total Backward Packets",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
)

#: a flow is "close" to a benign value on one feature if it's within this
#: multiplicative band of it (or both are exactly 0) -- the SAME "within
#: 2x" convention agents/grounding.py already uses for its own (Tier-1)
#: close-to-observed check, reused here for consistency rather than
#: inventing a different band width.
BAND_LOW_MULT = 0.5
BAND_HIGH_MULT = 2.0

#: a neighbourhood smaller than this is reported ungrounded rather than
#: trusted as a real "close to benign" reading.
MIN_NEIGHBOURHOOD_SIZE = 30

#: benign_plausibility may not exceed this when the flow's own
#: escalation-feature neighbourhood is ungrounded -- an empty
#: neighbourhood is evidence the flow is unusual, not a free pass.
#: 0.29, not 0.3: the verdict boundary is bp < 0.3 (ANOMALOUS_AND_UNEXPLAINED),
#: so a clamp to exactly 0.3 would read as explicable. Note the cap only clamps
#: values strictly above it, so a model output of exactly 0.30 passes through
#: unchanged under a 0.3 cap; 0.29 catches it.
UNGROUNDED_PLAUSIBILITY_CAP = 0.29

_benign_reference_cache: Optional[pd.DataFrame] = None


def load_benign_escalation_reference(force_reload: bool = False) -> pd.DataFrame:
    """Real BENIGN rows from the same pooled CICIDS2017 CSVs the
    escalation policy itself uses, restricted to the 10 escalation
    features. Cached in-process (this project's pool is 2.83M rows;
    re-filtering it on every one of a run's ~20 records is wasted work)."""
    global _benign_reference_cache
    if _benign_reference_cache is not None and not force_reload:
        return _benign_reference_cache
    from eval.escalation_data import BENIGN_LABEL, load_pool  # lazy: this module has no hard dependency on the CSV pool otherwise

    pool = load_pool()
    benign = pool.loc[pool["Label"] == BENIGN_LABEL, list(ESCALATION_FEATURES)].reset_index(drop=True)
    _benign_reference_cache = benign
    return benign


def observed_escalation_values(features: Dict[str, float]) -> Dict[str, float]:
    """The flow's own observed values on the ten escalation features,
    read directly from its real CICFlowMeter feature dict -- the ONLY
    input this function accepts. It has no parameter through which
    class_predicted, rule_id, priority, or reason could reach it even by
    accident (tests/test_escalation_grounding.py checks this by calling
    it with only a bare `features` dict and confirming identical output
    regardless of what escalation metadata existed upstream)."""
    return {f: features[f] for f in ESCALATION_FEATURES if f in features}


@dataclass(frozen=True)
class FeatureNeighbourhood:
    """One flow's own empirical neighbourhood in real benign traffic,
    computed directly from its feature values -- never from the
    escalation mechanism that flagged it."""

    features_used: List[str]
    band_description: str
    neighbourhood_size: int
    benign_population_size: int
    ungrounded: bool
    min_required: int

    def to_dict(self) -> dict:
        return {
            "features_used": self.features_used,
            "band_description": self.band_description,
            "neighbourhood_size": self.neighbourhood_size,
            "benign_population_size": self.benign_population_size,
            "ungrounded": self.ungrounded,
            "min_required": self.min_required,
        }


def compute_feature_neighbourhood(
    observed: Dict[str, float],
    benign_df: Optional[pd.DataFrame] = None,
    min_neighbours: int = MIN_NEIGHBOURHOOD_SIZE,
) -> FeatureNeighbourhood:
    """Benign flows within [0.5x, 2x] of this flow's own value (or ==0
    when the flow's own value is 0), jointly across every escalation
    feature both sides have -- drawn from `benign_df` (real BENIGN rows
    only; the caller is responsible for that filter, and
    load_benign_escalation_reference's default already applies it)."""
    benign_df = benign_df if benign_df is not None else load_benign_escalation_reference()
    features = [f for f in ESCALATION_FEATURES if f in observed and f in benign_df.columns]

    mask = pd.Series(True, index=benign_df.index)
    for f in features:
        val = observed[f]
        col = benign_df[f]
        if val == 0:
            mask &= col == 0
        else:
            lo, hi = sorted([val * BAND_LOW_MULT, val * BAND_HIGH_MULT])
            mask &= col.between(lo, hi)

    size = int(mask.sum()) if features else 0
    return FeatureNeighbourhood(
        features_used=features,
        band_description=(
            f"[{BAND_LOW_MULT}x, {BAND_HIGH_MULT}x] of this flow's own observed value per feature "
            f"(or ==0 when observed is 0), joint AND across all {len(features)} feature(s) used"
        ),
        neighbourhood_size=size,
        benign_population_size=int(len(benign_df)),
        ungrounded=(size < min_neighbours),
        min_required=min_neighbours,
    )


def patch_hypothesis_support(
    supports: Dict[str, HypothesisSupport], neighbourhood: FeatureNeighbourhood,
) -> Dict[str, HypothesisSupport]:
    """Replaces every hypothesis's close_to_observed_count/checked_features
    for this record with the SAME record-level neighbourhood -- "how many
    real benign flows resemble this flow" is a property of the flow, not
    of any one hypothesis, so every hypothesis on the same record gets
    the same, now-always-DEFINED number (never None again for this
    reason). `matching_profile_count` (the existing Tier-1 profile check)
    is left untouched -- this only repairs the closeness refinement,
    it does not replace the generic check."""
    close_count = 0 if neighbourhood.ungrounded else neighbourhood.neighbourhood_size
    return {
        hid: replace(s, close_to_observed_count=close_count, checked_features=list(neighbourhood.features_used))
        for hid, s in supports.items()
    }


def apply_ungrounded_neighbourhood_cap(
    response: A5Response, neighbourhood: FeatureNeighbourhood, cap: float = UNGROUNDED_PLAUSIBILITY_CAP,
) -> "tuple[A5Response, bool]":
    """An ADDITIONAL cap, independent of and layered alongside
    agents.grounding.apply_empirical_plausibility_cap (which only fires
    on matching_profile_count==0 -- a different, Tier-1-scoped signal
    this repair does not touch). If this flow has NO real benign
    neighbours of its own in escalation-feature space, that emptiness is
    itself evidence of an unusual flow and must not be silently read as
    "benign" just because some generic Tier-1 profile still matched
    broadly -- so benign_plausibility is capped regardless of which
    hypothesis A5 credited or what its generic match count says."""
    if not neighbourhood.ungrounded:
        return response, False
    if response.benign_plausibility > cap:
        return response.model_copy(update={"benign_plausibility": cap}), True
    return response, False
