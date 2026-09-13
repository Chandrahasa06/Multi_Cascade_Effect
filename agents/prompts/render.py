"""Shared formatting helpers for turning an EscalationRecord or an
upstream agent's structured output into prompt text. Kept separate from
the versioned prompt files: a wording tweak to instructions shouldn't
force touching how numbers get rendered, and vice versa.

Ground truth never enters this module, or anything it touches --
EscalationRecord has no label field at all (see controlplane/record.py),
so there is nothing here that could leak one.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Set

from controlplane.record import EscalationRecord

from agents.schema import Claim, Hypothesis

FRAMING_PREAMBLE = """\
Context: this flow was selected for review by an automated system using \
purely statistical thresholds fitted on ordinary traffic from this \
network. Nothing about being selected implies a conclusion -- treat \
this as one flow's evidence, to be examined on its own terms.
"""

#: Grounded in documented project facts (this sample was drawn at the
#: "5%-benign-rate operating point" under the project's default
#: k_of_n(k=2) escalation rule -- see STATUS.md), not an invented number.
#: Deliberately does NOT assert a specific per-feature percentile (e.g.
#: "99.5th") -- that number isn't stored per-threshold anywhere in this
#: project and would be a guess dressed up as a fact. What's true and
#: checkable is the *combined* calibration target; stated as such.
CALIBRATION_NOTE = """\
Calibration context: every "typical-range threshold" you see (here and \
in any claim using greater_than_typical/less_than_typical) was fit \
using ONLY ordinary (benign) traffic from this network -- never from \
the kind of traffic this review process exists to catch, so a \
threshold reflects what is statistically rare for ordinary use, not a \
guess. This record's flagged measurements were \
selected so that, combined, only about 5% of ordinary traffic would \
ever cross two or more such thresholds at once (this project's default \
rule); crossing a single feature's own threshold alone is markedly \
rarer than that for most features. A "ratio" of 1.5x means the \
observed value is 1.5 times the fitted threshold itself -- values near \
1.0x are common right at the boundary and not remarkable on their own; \
larger ratios represent increasingly rare deviations from ordinary \
traffic on this specific network.
"""

#: Self-reported trust triad, asked of every agent (agents/schema.py's
#: ClaimsResponse/HypothesisResponse/A5Response all carry these three
#: fields now) -- distinct from agents/trust.py's code-computed C_i/E_i/
#: V_i, which check the same three things mechanically after the fact.
#: Kept as shared text here (not duplicated per prompt file) so the
#: three fields mean the same thing everywhere they're asked. A1-A4
#: have not seen any independent replica's output when asked this, so
#: their `verification` self-report is a genuine blind estimate, not
#: something they can check -- SELF_REPORT_INSTRUCTIONS_A5 below is the
#: one exception, since A5 is shown the real mechanical corroboration
#: summary before answering.
SELF_REPORT_INSTRUCTIONS = """\
Also report three overall self-assessments for this entire response, \
each from 0 to 1 -- distinct from any individual claim's own confidence \
field, which is about one specific claim, not the response as a whole:
- confidence: your overall confidence in this analysis as a whole.
- evidence_support: your own estimate of how well the claims above \
would hold up if checked directly against the record's actual data. \
This is not computed for you -- give your honest estimate, even if you \
expect it to differ from what a mechanical check would find.
- verification: your own estimate of how well these claims would be \
corroborated by a separate, independent, blind re-analysis of this same \
flow, performed by someone who has not seen your work and cannot see it \
now. You have no way to check this directly -- it is a genuine estimate \
of your own reliability, not something you can verify here.
Report your real estimate for each of the three, even where you expect \
them to diverge from each other.
"""

#: A5 sees the real mechanical corroboration between the chain and A4
#: before answering, so its verification self-report has something
#: concrete to be grounded in, not a blind guess like A1-A4's.
SELF_REPORT_INSTRUCTIONS_A5 = """\
Also report three overall self-assessments for this verdict, each from \
0 to 1 -- distinct from cited_claim_ids and the rationale, which explain \
your reasoning, not rate it:
- confidence: your overall confidence in this benign_plausibility \
estimate.
- evidence_support: your own estimate of how well the evidence you \
cited above would hold up if checked directly against the record's \
actual data.
- verification: your own estimate of how well this verdict is actually \
supported by the mechanical corroboration between the chain and the \
independent review shown to you above -- unlike A1-A4, you have already \
been shown that comparison, so base this on it, not a blind guess.
Report your real estimate for each of the three, even where you expect \
them to diverge from each other.
"""

#: prefix distinguishing the selector's own live Tier-1/per-source
#: measurements (data-plane, O(1)/approximate, native units -- typically
#: microseconds for anything duration/inter-arrival-time-shaped) from the
#: full CICFlowMeter feature set below (control-plane, precise
#: recomputation from the raw packets, typically seconds). The two can
#: share a bare name (e.g. "flow_duration") while disagreeing by orders
#: of magnitude -- confirmed live, exactly 1,000,000x for flow_duration
#: on one stop-point-2 record (microseconds vs seconds). Prefixing keeps
#: them permanently distinct strings so a claim can unambiguously say
#: which one it means, instead of two different numbers silently sharing
#: one name.
TRIGGER_FEATURE_PREFIX = "trigger_"


def known_feature_names(record: EscalationRecord) -> Set[str]:
    """The complete closed vocabulary of real feature names for this
    record -- the CICFlowMeter feature dict plus the selector's own
    trigger-reason names under their TRIGGER_FEATURE_PREFIX. Used to
    validate that a hypothesis's ``referenced_features`` (agents/
    validators.py::validate_hypothesis_predictions_are_flow_level) names
    something this pipeline actually observes, never an
    application-layer concept it has no data for."""
    names = set(record.features)
    names.update(f"{TRIGGER_FEATURE_PREFIX}{tr.feature}" for tr in record.trigger_reasons)
    return names


def render_features(record: EscalationRecord) -> str:
    lines = [f"  {name} = {value:g}" for name, value in sorted(record.features.items())]
    return "\n".join(lines)


def render_trigger_reasons(record: EscalationRecord) -> str:
    if not record.trigger_reasons:
        return "  (none recorded)"
    header = (
        "  (these are the automated system's own live measurements, in its own native units "
        "-- durations and inter-arrival times here are typically MICROSECONDS, not the SECONDS "
        "used in the full feature list below. A name below may look similar to a name in the "
        "full feature list but is a DIFFERENT measurement, on a different scale -- always use "
        f"the '{TRIGGER_FEATURE_PREFIX}' prefix shown here when referencing one of these, never "
        "the bare name, so it's never confused with the full-feature-list value of the same "
        "bare name.)"
    )
    lines = [header]
    for tr in record.trigger_reasons:
        tag = " [coarse timing resolution -- treat this specific comparison cautiously]" if tr.low_confidence else ""
        # observed_value and ratio_to_threshold go on separate labelled
        # lines, not one sentence -- a live stop-point-2 run showed A1
        # asserting the ratio (e.g. "1.71") as if it were the
        # observed_value (e.g. "2361.4") in 2/20 records, plausibly
        # because both numbers sat close together in one line ending
        # "...1.71x the threshold". ratio_to_threshold is explicitly
        # labelled as a multiplier, never a candidate value on its own.
        lines.append(f"  {TRIGGER_FEATURE_PREFIX}{tr.feature}:{tag}")
        lines.append(f"      observed_value: {tr.observed_value:g}")
        lines.append(f"      threshold: {tr.threshold:g} ({tr.direction} side)")
        lines.append(
            f"      ratio_to_threshold: {tr.ratio:.2f}x  "
            "(a multiplier, NOT the observed_value -- use observed_value above for that)"
        )
    return "\n".join(lines)


def _render_feature_ref(r) -> str:
    value_part = f" (asserted {r.asserted_value:g})" if r.asserted_value is not None else ""
    return f"{r.name} {r.relation.value}{value_part}"


def render_claims(claims: Iterable[Claim]) -> str:
    lines = []
    for c in claims:
        refs = "; ".join(_render_feature_ref(r) for r in c.referenced_features)
        ref_part = f" -- refs: {refs}" if refs else ""
        lines.append(f"  [{c.claim_id}] (confidence {c.confidence:.2f}) {c.statement}{ref_part}")
    return "\n".join(lines) if lines else "  (no claims)"


def render_required_claim_ids(claim_ids: Iterable[str]) -> str:
    """The set of claim_ids A5 must dispose of -- see a5_verdict_v3's
    contradiction-disposal requirement (agents/validators.py::
    validate_a5_addresses_all_contradictions, which enforces this exact
    set mechanically, so what's rendered here and what's checked in code
    are computed from the same set, never allowed to drift apart)."""
    ids = sorted(set(claim_ids))
    if not ids:
        return "  (none -- no hypothesis below has flagged any claim as contradicting it)"
    return "  " + ", ".join(ids)


def _render_predicted_range(p) -> str:
    if p.expected_min is not None and p.expected_max is not None:
        bounds = f"[{p.expected_min:g}, {p.expected_max:g}]"
    elif p.expected_min is not None:
        bounds = f">= {p.expected_min:g}"
    else:
        bounds = f"<= {p.expected_max:g}"
    return f"{p.feature} {bounds}"


def render_hypotheses(
    hypotheses: Iterable[Hypothesis], empirical_support_lines: Optional[Dict[str, str]] = None
) -> str:
    """``empirical_support_lines``: optional {hypothesis_id: pre-rendered
    empirical-support string} (agents.grounding.render_hypothesis_support,
    computed by the caller -- kept out of this module's own imports to
    avoid a render.py <-> grounding.py cycle, since grounding.py already
    imports TRIGGER_FEATURE_PREFIX from here)."""
    empirical_support_lines = empirical_support_lines or {}
    lines = []
    for h in hypotheses:
        tag = "benign" if h.benign else "non-benign"
        support = ", ".join(h.supporting_claim_ids) or "(none)"
        contra = ", ".join(h.contradicting_claim_ids) or "(none)"
        profile = "; ".join(_render_predicted_range(p) for p in h.predicted_feature_profile) or "(none)"
        block = (
            f"  [{h.hypothesis_id}] ({tag}, prior plausibility {h.prior_plausibility:.2f}) "
            f"{h.description}\n      prediction: {h.prediction}"
            f"\n      predicted_feature_profile: {profile}"
            f"\n      supporting: {support}\n      contradicting: {contra}"
        )
        if h.hypothesis_id in empirical_support_lines:
            block += f"\n      {empirical_support_lines[h.hypothesis_id]}"
        lines.append(block)
    return "\n".join(lines) if lines else "  (no hypotheses)"
