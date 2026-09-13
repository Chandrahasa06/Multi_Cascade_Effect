"""Join simulated flows back to CICIDS2017 ground-truth labels.

See README's "Known Limitations": ground truth is defined at
CICFlowMeter's flow granularity, and a data plane with a different flow
definition has an irreducible join ambiguity. This module doesn't solve
that; it characterises it.

EVAL mode (FlowTable.KeyMode.EVAL): Packet.source_row_id maps 1:1 to a
CSV row, which maps 1:1 to a label. Trivial — asserted to hold, not just
assumed.

FIDELITY mode: a merged flow (``len(flow.source_row_ids) > 1``) inherits
labels from every composing row. When those rows disagree, resolve to a
single label by nearest timestamp — the composing row whose own CSV
Timestamp is closest to the flow's ``first_ts``. This is a best-effort
resolution, not a guarantee: by construction, rows only merge in
FIDELITY mode when they're close enough in time that the table's own
idle/active timeout didn't separate them — exactly the situation where
"nearest timestamp" is least able to disambiguate. A flow whose two
closest, differently-labeled composing rows sit within the day's own
timestamp resolution of each other is flagged residually ambiguous:
proximity can't disambiguate below the precision the data actually has.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from adapters.csv_flow_adapter import load_csv, parse_timestamp_us
from dataplane.flow_state import FlowState

BENIGN_LABEL = "BENIGN"

#: rows closer together than this can't be meaningfully ordered by
#: "nearest timestamp" — it's the finest resolution CICIDS2017 timestamps
#: ever have (Monday; other days are coarser — see the CSV adapter).
AMBIGUITY_THRESHOLD_US = 1_000_000


@dataclass(frozen=True, slots=True)
class RowGroundTruth:
    label: str
    timestamp_us: Optional[int]


def load_row_ground_truth(csv_path) -> List[RowGroundTruth]:
    """One entry per CSV row, in file order — the same order/indexing the
    adapter uses for Packet.source_row_id, so row index lines up exactly."""
    df = load_csv(csv_path)
    labels = df["Label"].astype(str).str.strip().tolist()
    timestamps = df["Timestamp"].tolist()
    return [
        RowGroundTruth(label=label, timestamp_us=parse_timestamp_us(str(ts)))
        for label, ts in zip(labels, timestamps)
    ]


@dataclass(frozen=True, slots=True)
class LabelResolution:
    label: str
    ambiguous: bool
    mixed: bool  # more than one distinct label among this flow's composing rows
    mixing_kind: Optional[str] = None  # "benign_attack" | "attack_attack" | None


def resolve_eval_label(flow: FlowState, row_truth: Sequence[RowGroundTruth]) -> str:
    """EVAL mode: source_row_id is 1:1 with a CSV row. Asserted, not
    assumed — a violation here means KeyMode.EVAL's keying is broken."""
    assert len(flow.source_row_ids) == 1, (
        f"EVAL-mode flow {flow.key} has {len(flow.source_row_ids)} source rows, "
        "expected exactly 1 — the 1:1 keying guarantee is violated"
    )
    (row_id,) = flow.source_row_ids
    return row_truth[row_id].label


def resolve_fidelity_label(
    flow: FlowState, row_truth: Sequence[RowGroundTruth]
) -> LabelResolution:
    """FIDELITY mode: a flow may have merged several rows. Resolve by
    nearest timestamp when they disagree; flag residual ambiguity when
    that resolution isn't trustworthy at the data's own precision."""
    row_ids = sorted(flow.source_row_ids)
    assert row_ids, f"flow {flow.key} has no source_row_ids — cannot join to ground truth"

    entries = [(rid, row_truth[rid].label, row_truth[rid].timestamp_us) for rid in row_ids]
    distinct_labels = {label for _, label, _ in entries}

    if len(distinct_labels) == 1:
        return LabelResolution(label=entries[0][1], ambiguous=False, mixed=False)

    mixing_kind = "benign_attack" if BENIGN_LABEL in distinct_labels else "attack_attack"

    dated = [(rid, label, ts) for rid, label, ts in entries if ts is not None]
    if not dated:
        # no usable timestamp on any composing row: nothing to resolve
        # by, so any pick is untrustworthy by definition.
        return LabelResolution(
            label=entries[0][1], ambiguous=True, mixed=True, mixing_kind=mixing_kind
        )

    dated.sort(key=lambda e: abs(e[2] - flow.first_ts))
    _, nearest_label, nearest_ts = dated[0]

    # residual ambiguity: another composing row, with a DIFFERENT label,
    # sits within the ambiguity threshold of the winning row.
    ambiguous = any(
        label != nearest_label and abs(ts - nearest_ts) < AMBIGUITY_THRESHOLD_US
        for _, label, ts in dated[1:]
    )

    return LabelResolution(
        label=nearest_label, ambiguous=ambiguous, mixed=True, mixing_kind=mixing_kind
    )


@dataclass
class JoinReport:
    total_flows: int = 0
    single_row_flows: int = 0
    merged_flows: int = 0
    merged_same_label_flows: int = 0
    merged_mixed_label_flows: int = 0
    benign_attack_mixed: int = 0
    attack_attack_mixed: int = 0
    residual_ambiguous: int = 0

    @property
    def join_rate(self) -> float:
        """Fraction of flows that got a trustworthy single label —
        everything except a residually-ambiguous resolution."""
        if self.total_flows == 0:
            return 1.0
        return 1.0 - (self.residual_ambiguous / self.total_flows)

    def to_dict(self) -> dict:
        return {
            "total_flows": self.total_flows,
            "single_row_flows": self.single_row_flows,
            "merged_flows": self.merged_flows,
            "merged_same_label_flows": self.merged_same_label_flows,
            "merged_mixed_label_flows": self.merged_mixed_label_flows,
            "benign_attack_mixed": self.benign_attack_mixed,
            "attack_attack_mixed": self.attack_attack_mixed,
            "residual_ambiguous": self.residual_ambiguous,
            "join_rate": self.join_rate,
        }


def join_eval_flows(
    flows: Sequence[FlowState], row_truth: Sequence[RowGroundTruth]
) -> Tuple[List[str], JoinReport]:
    """EVAL mode is join-trivial by construction: report is included for
    a uniform interface with join_fidelity_flows, not because it can
    meaningfully be below 100%."""
    labels = [resolve_eval_label(flow, row_truth) for flow in flows]
    report = JoinReport(total_flows=len(flows), single_row_flows=len(flows))
    return labels, report


def resolve_fidelity_labels(
    flows: Sequence[FlowState], row_truth: Sequence[RowGroundTruth]
) -> List[LabelResolution]:
    """Per-flow resolutions, exposed separately from join_fidelity_flows
    so a caller that needs the full LabelResolution (e.g. eval/simulate.py,
    to record mixed/ambiguous flags per flow) doesn't resolve twice."""
    return [resolve_fidelity_label(flow, row_truth) for flow in flows]


def build_fidelity_join_report(
    flows: Sequence[FlowState], resolutions: Sequence[LabelResolution]
) -> JoinReport:
    report = JoinReport(total_flows=len(flows))
    for flow, resolution in zip(flows, resolutions):
        if len(flow.source_row_ids) <= 1 or not resolution.mixed:
            if len(flow.source_row_ids) <= 1:
                report.single_row_flows += 1
            else:
                report.merged_flows += 1
                report.merged_same_label_flows += 1
            continue

        report.merged_flows += 1
        report.merged_mixed_label_flows += 1
        if resolution.mixing_kind == "benign_attack":
            report.benign_attack_mixed += 1
        else:
            report.attack_attack_mixed += 1
        if resolution.ambiguous:
            report.residual_ambiguous += 1
    return report


def join_fidelity_flows(
    flows: Sequence[FlowState], row_truth: Sequence[RowGroundTruth]
) -> Tuple[List[str], JoinReport]:
    resolutions = resolve_fidelity_labels(flows, row_truth)
    labels = [r.label for r in resolutions]
    report = build_fidelity_join_report(flows, resolutions)
    return labels, report
