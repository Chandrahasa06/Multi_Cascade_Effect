"""Tier-1 feature computation, threshold comparison, and escalation
decision.

Everything here runs at *check time* — thresholds are compared against
quantities derived from the stored integer counters (FlowState, SrcTable),
never stored themselves, matching the hard constraint that only counters
live in the data plane.

Trigger reasons are returned as a structured list (feature, observed
value, threshold, direction, ratio), never a formatted string: this gets
handed to the agent pipeline later, whose first stage needs the raw
numbers, not prose describing them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from dataplane.flow_state import UNSET, FlowState
from dataplane.src_table import SrcTable

#: bump whenever compute_tier1_features/compute_src_features change what
#: they return (new/removed/redefined features) — cached extractions key
#: on this so a code change invalidates the cache. See eval/simulate.py.
FEATURE_SCHEMA_VERSION = "2"

#: Timing-derived Tier-1 features whose confidence depends on how
#: precisely the flow's timestamps are known. A flow that passed through
#: the CSV adapter on a minute-resolution day (see
#: adapters.csv_flow_adapter) carries a coarse timestamp_resolution_us;
#: values computed from first_ts/last_ts/iat_* on such a flow are flagged
#: low-confidence rather than silently treated as exact.
TIMING_DERIVED_FEATURES = frozenset(
    {
        "flow_duration",
        "flow_bytes_per_sec",
        "flow_pkts_per_sec",
        "flow_iat_mean",
        "flow_iat_max",
        "flow_iat_min",
    }
)

#: a flow whose coarsest contributing packet resolution is coarser than
#: this (i.e. worse than 1-second ticks) gets its timing features flagged.
LOW_CONFIDENCE_RESOLUTION_THRESHOLD_US = 1_000_000

US_PER_SEC = 1_000_000


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    """One computed feature value. ``value`` is None when the quantity is
    undefined for this flow (e.g. a rate over zero duration) — undefined
    features are never compared to a threshold and must never be fed to
    percentile fitting as if they were zero."""

    value: Optional[float]
    low_confidence: bool = False


def _combined_pkt_len_min_max(flow: FlowState):
    mins: List[int] = []
    maxs: List[int] = []
    if flow.fwd_pkt_count > 0:
        mins.append(flow.fwd_pkt_len_min)
        maxs.append(flow.fwd_pkt_len_max)
    if flow.bwd_pkt_count > 0:
        mins.append(flow.bwd_pkt_len_min)
        maxs.append(flow.bwd_pkt_len_max)
    if not mins:
        return None, None
    return min(mins), max(maxs)


def compute_tier1_features(flow: FlowState) -> Dict[str, FeatureObservation]:
    """Per-flow selector features, computed from FlowState counters."""
    low_conf = flow.timestamp_resolution_us > LOW_CONFIDENCE_RESOLUTION_THRESHOLD_US

    def obs(name: str, value: Optional[float]) -> FeatureObservation:
        return FeatureObservation(value, low_conf and name in TIMING_DERIVED_FEATURES)

    total_pkts = flow.total_pkt_count
    total_bytes = flow.total_byte_sum
    duration = flow.last_ts - flow.first_ts

    features: Dict[str, FeatureObservation] = {}

    features["flow_duration"] = obs("flow_duration", float(duration))
    features["flow_bytes_per_sec"] = obs(
        "flow_bytes_per_sec", (total_bytes / duration) * US_PER_SEC if duration > 0 else None
    )
    features["flow_pkts_per_sec"] = obs(
        "flow_pkts_per_sec", (total_pkts / duration) * US_PER_SEC if duration > 0 else None
    )

    features["fwd_pkt_len_mean"] = obs(
        "fwd_pkt_len_mean",
        flow.fwd_byte_sum / flow.fwd_pkt_count if flow.fwd_pkt_count > 0 else None,
    )
    features["bwd_pkt_len_mean"] = obs(
        "bwd_pkt_len_mean",
        flow.bwd_byte_sum / flow.bwd_pkt_count if flow.bwd_pkt_count > 0 else None,
    )

    pkt_min, pkt_max = _combined_pkt_len_min_max(flow)
    features["pkt_len_range"] = obs(
        "pkt_len_range", float(pkt_max - pkt_min) if pkt_min is not None else None
    )

    features["down_up_pkt_ratio"] = obs(
        "down_up_pkt_ratio",
        flow.bwd_pkt_count / flow.fwd_pkt_count if flow.fwd_pkt_count > 0 else None,
    )
    features["bwd_fwd_byte_ratio"] = obs(
        "bwd_fwd_byte_ratio",
        flow.bwd_byte_sum / flow.fwd_byte_sum if flow.fwd_byte_sum > 0 else None,
    )

    features["flow_iat_mean"] = obs(
        "flow_iat_mean", flow.iat_sum / (total_pkts - 1) if total_pkts > 1 else None
    )
    # iat_max/iat_min are "stored directly" per spec, not a derived ratio;
    # iat_max's default of 0 is a valid value (no larger gap ever seen),
    # iat_min's UNSET sentinel is the only truly-undefined case.
    features["flow_iat_max"] = obs("flow_iat_max", float(flow.iat_max))
    features["flow_iat_min"] = obs(
        "flow_iat_min", float(flow.iat_min) if flow.iat_min != UNSET else None
    )

    features["syn_ratio"] = obs(
        "syn_ratio", flow.syn_count / total_pkts if total_pkts > 0 else None
    )
    features["rst_ratio"] = obs(
        "rst_ratio", flow.rst_count / total_pkts if total_pkts > 0 else None
    )

    features["no_response_flag"] = obs("no_response_flag", 1.0 if flow.bwd_pkt_count == 0 else 0.0)

    features["init_win_bytes_fwd"] = obs(
        "init_win_bytes_fwd",
        float(flow.init_win_bytes_fwd) if flow.init_win_bytes_fwd != UNSET else None,
    )
    features["init_win_bytes_bwd"] = obs(
        "init_win_bytes_bwd",
        float(flow.init_win_bytes_bwd) if flow.init_win_bytes_bwd != UNSET else None,
    )

    return features


def compute_src_features(
    src_table: SrcTable, src_ip: str, now_us: int
) -> Dict[str, FeatureObservation]:
    """Per-source sliding-window features, computed from SrcTable."""
    return {
        "flows_per_src": FeatureObservation(float(src_table.flows_per_src(src_ip, now_us))),
        "distinct_dst_ports_per_src": FeatureObservation(
            src_table.distinct_dst_ports_per_src(src_ip, now_us)
        ),
        "distinct_dst_ips_per_src": FeatureObservation(
            src_table.distinct_dst_ips_per_src(src_ip, now_us)
        ),
        "syn_without_synack_count": FeatureObservation(
            float(src_table.syn_without_synack_count(src_ip, now_us))
        ),
    }


def should_run_selector(flow: FlowState, check_every: int = 8) -> bool:
    """Selector cadence: every N packets, and always at flow close."""
    return flow.terminated or flow.total_pkt_count % check_every == 0


class EscalationRule(str, Enum):
    ANY = "any"
    K_OF_N = "k_of_n"


@dataclass(frozen=True, slots=True)
class FeatureThreshold:
    """A fitted, benign-only threshold for one feature. Either side may
    be absent: a feature that's only ever anomalous when high sets `low`
    to None, and vice versa."""

    high: Optional[float] = None
    low: Optional[float] = None

    def to_dict(self) -> dict:
        return {"high": self.high, "low": self.low}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureThreshold":
        return cls(high=data.get("high"), low=data.get("low"))


@dataclass(frozen=True, slots=True)
class TriggerReason:
    """One feature's threshold crossing. Kept as fields, not a formatted
    string, since this is handed to the agent pipeline as evidence."""

    feature: str
    observed_value: float
    threshold: float
    direction: str  # "high" or "low"
    ratio: float  # multiple of the threshold the observation represents
    low_confidence: bool = False

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "observed_value": self.observed_value,
            "threshold": self.threshold,
            "direction": self.direction,
            "ratio": self.ratio,
            "low_confidence": self.low_confidence,
        }


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    thresholds: Dict[str, FeatureThreshold]
    rule: EscalationRule = EscalationRule.ANY
    k: int = 1  # only used when rule == K_OF_N
    config_hash: str = ""  # identifies the fitted threshold set that produced this

    def to_dict(self) -> dict:
        return {
            "thresholds": {name: t.to_dict() for name, t in self.thresholds.items()},
            "rule": self.rule.value,
            "k": self.k,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SelectorConfig":
        return cls(
            thresholds={
                name: FeatureThreshold.from_dict(t) for name, t in data["thresholds"].items()
            },
            rule=EscalationRule(data.get("rule", "any")),
            k=data.get("k", 1),
            config_hash=data.get("config_hash", ""),
        )


@dataclass(frozen=True, slots=True)
class SelectorDecision:
    escalate: bool
    trigger_reasons: List[TriggerReason]
    features: Dict[str, FeatureObservation] = field(default_factory=dict)


class Selector:
    """Compares live flow/source features against fitted, benign-only
    thresholds and decides whether to escalate to the control plane."""

    def __init__(self, config: SelectorConfig):
        self.config = config

    def evaluate(
        self,
        flow: FlowState,
        src_table: Optional[SrcTable] = None,
        now_us: Optional[int] = None,
    ) -> SelectorDecision:
        features = compute_tier1_features(flow)
        if src_table is not None:
            features.update(
                compute_src_features(src_table, flow.fwd_ip, now_us if now_us is not None else flow.last_ts)
            )

        trigger_reasons: List[TriggerReason] = []
        for name, observation in features.items():
            if observation.value is None:
                continue
            threshold = self.config.thresholds.get(name)
            if threshold is None:
                continue
            if threshold.high is not None and observation.value > threshold.high:
                trigger_reasons.append(
                    self._make_reason(name, observation, threshold.high, "high")
                )
            if threshold.low is not None and observation.value < threshold.low:
                trigger_reasons.append(
                    self._make_reason(name, observation, threshold.low, "low")
                )

        if self.config.rule == EscalationRule.ANY:
            escalate = len(trigger_reasons) >= 1
        else:
            distinct_features = {r.feature for r in trigger_reasons}
            escalate = len(distinct_features) >= self.config.k

        return SelectorDecision(escalate=escalate, trigger_reasons=trigger_reasons, features=features)

    @staticmethod
    def _make_reason(
        feature: str, observation: FeatureObservation, threshold: float, direction: str
    ) -> TriggerReason:
        value = observation.value
        if direction == "high":
            ratio = value / threshold if threshold != 0 else float("inf")
        else:
            ratio = threshold / value if value != 0 else float("inf")
        return TriggerReason(
            feature=feature,
            observed_value=value,
            threshold=threshold,
            direction=direction,
            ratio=ratio,
            low_confidence=observation.low_confidence,
        )
