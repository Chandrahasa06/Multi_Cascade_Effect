"""The handoff object from the data plane's selector to the agent
pipeline: everything an escalated flow's *evidence* consists of, and
nothing that identifies who was talking to whom or when.

Ground truth label deliberately has no field here — it lives in the
eval harness, keyed by ``flow_id``, so the agent pipeline is tested the
same way it will run in production: without an answer key. Putting a
label on this object would make it trivial to accidentally leak into a
prompt or a log line; leaving it out entirely is the allow-list
philosophy applied to the object's own shape, not just its field
values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from dataplane.selector import TriggerReason


@dataclass(frozen=True, slots=True)
class PacketWindowSummary:
    """Provenance for the packet window an escalation record's features
    were computed from — relative/count-based only, see
    controlplane/extractor.py's anonymisation pass for what's excluded
    and why."""

    packet_count: int
    byte_count: int
    duration_us: int  # relative (last packet - first packet), never absolute

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EscalationRecord:
    """One escalated flow, packaged for the agent pipeline.

    ``flow_id`` is an opaque token (see
    ``controlplane.extractor.make_flow_id``) — never derived from the
    flow's real IP/port identity, so it can't be reversed even in
    principle; the mapping from ``flow_id`` back to the real flow (and
    its ground-truth label, for anyone evaluating the pipeline) is kept
    separately by whoever built the record, never inside it.

    ``features`` is the full anonymised CICFlowMeter feature set — see
    ``controlplane/extractor.py``'s module docstring for the exact
    count and why it isn't exactly the traditional "78" — sent whole,
    never pruned by importance (pruning by supervised feature
    importance selects for *known* attacks, the wrong prior for a
    zero-day selector to encode).
    """

    flow_id: str
    trigger_reasons: List[TriggerReason]
    features: Dict[str, float]
    packet_window: PacketWindowSummary
    selector_config_hash: str

    def to_dict(self) -> dict:
        return {
            "flow_id": self.flow_id,
            "trigger_reasons": [r.to_dict() for r in self.trigger_reasons],
            "features": self.features,
            "packet_window": self.packet_window.to_dict(),
            "selector_config_hash": self.selector_config_hash,
        }
