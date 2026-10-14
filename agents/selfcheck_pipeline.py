"""Self-check pipeline: A1, A2, A3, A4 (blind), A5a (checklist), A5b (verdict).
Rules: results/selfcheck_preregistered.md. Definitions: agents/selfcheck_core.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from agents import base
from agents import selfcheck_core as core
from agents.prompts import selfcheck_prompts_v1 as P
from agents.schema import (
    SCBehaviourResponse, SCBlindResponse, SCChecklistResponse, SCClaimsResponse, SCHypothesisResponse,
    SCVerdictResponse,
)

ABSENCE_KINDS = ("no_payload_sent", "no_reply", "no_teardown", "no_size_variation")

#: which model and prompt module the run_* calls use. Defaults are the original Gemini run;
#: configure() switches both for a run (e.g. an OpenRouter model from config/openrouter.json).
MODEL = base.DEFAULT_MODEL
TEMPERATURE = 0.7


def configure(model: str, prompts=None) -> None:
    """Point every agent call at `model` (and optionally another prompt module). An
    OpenRouter model takes its temperature from config/openrouter.json."""
    from agents import openrouter

    global MODEL, TEMPERATURE, P
    MODEL = model
    TEMPERATURE = openrouter.temperature_for(model) if openrouter.is_configured(model) else 0.7
    if prompts is not None:
        P = prompts


# ------------------------------------------------------------------ validators
def validate_claims(claims, prefix: str, ban: bool = False) -> None:
    """Checks every claim and reports every problem found in one message, so a retry can fix
    them all at once. `ban` applies the comparison rule (core.comparison_violations)."""
    seen = set()
    errors = []
    for c in claims:
        if not c.claim_id.startswith(prefix):
            errors.append(f"claim_id {c.claim_id!r} must start with {prefix!r}")
        if c.claim_id in seen:
            errors.append(f"duplicate claim_id {c.claim_id!r}")
        seen.add(c.claim_id)
        if not any(core.names_field(c.statement, v.name) for v in c.cited):
            errors.append(f"{c.claim_id}: the sentence must name at least one of the fields it cites "
                          f"({', '.join(v.name for v in c.cited)})")
        bad = [x for x in c.condition_ids if x not in core.CONDITION_IDS]
        if bad:
            errors.append(f"{c.claim_id}: unknown condition id(s) {bad}")
        if ban:
            errors += [f"{c.claim_id}: {v}" for v in core.comparison_violations(c.statement, [v.name for v in c.cited])]
    if errors:
        raise ValueError("; ".join(errors))


def validate_absences(absences) -> None:
    kinds = [a.absence.value for a in absences]
    if sorted(kinds) != sorted(ABSENCE_KINDS):
        raise ValueError(f"absences must be exactly one each of {list(ABSENCE_KINDS)}; got {kinds}")
    for a in absences:
        if a.holds.value != "cannot_tell" and not a.cited:
            raise ValueError(f"{a.absence.value}: cite the field(s) you read")


def validate_hypotheses(hyps, prefix: str) -> None:
    if not 3 <= len(hyps) <= 5:
        raise ValueError(f"expected 3 to 5 hypotheses, got {len(hyps)}")
    if sum(h.benign for h in hyps) < 2:
        raise ValueError("at least 2 hypotheses must be benign")
    seen = set()
    for h in hyps:
        if not h.hypothesis_id.startswith(prefix):
            raise ValueError(f"hypothesis_id {h.hypothesis_id!r} must start with {prefix!r}")
        if h.hypothesis_id in seen:
            raise ValueError(f"duplicate hypothesis_id {h.hypothesis_id!r}")
        seen.add(h.hypothesis_id)
        bad = [x for x in h.condition_ids if x not in core.CONDITION_IDS]
        if bad:
            raise ValueError(f"{h.hypothesis_id}: unknown condition id(s) {bad}")


def validate_checklist(checklist, value_names: Set[str]) -> None:
    ids = [r.condition_id for r in checklist]
    known = set(core.CONDITION_IDS)
    missing = [c for c in core.CONDITION_IDS if c not in ids]
    if missing:
        raise ValueError(f"checklist is missing condition(s) {missing}; every condition needs one entry")
    extra = [i for i in ids if i not in known]
    if extra:
        raise ValueError(f"checklist has unknown condition id(s) {extra}")
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"checklist has duplicate entries for {dupes}")
    allowed = value_names | {"Protocol"}
    for r in checklist:
        bad = [v.name for v in r.observed if v.name not in allowed]
        if bad:
            raise ValueError(f"{r.condition_id}: {bad} are not names of this flow's fields")


def validate_verdict(r: SCVerdictResponse, yes_ids: List[str], creditable: Set[str]) -> None:
    disp_ids = [d.condition_id for d in r.dispositions]
    if sorted(disp_ids) != sorted(yes_ids):
        raise ValueError(f"give exactly one disposition for each condition marked yes: {sorted(yes_ids)}; got {sorted(disp_ids)}")
    for d in r.dispositions:
        if d.refuted:
            if not d.refuted_by:
                raise ValueError(f"{d.condition_id}: a refutation must cite a SUPPORTED benign explanation")
            bad = [h for h in d.refuted_by if h not in creditable]
            if bad:
                raise ValueError(f"{d.condition_id}: {bad} cannot refute (only SUPPORTED benign explanations shown to you can): {sorted(creditable) or 'none'}")
    allowed = core.allowed_verdicts(yes_ids, r.dispositions)
    if r.verdict.value not in allowed:
        raise ValueError(f"verdict {r.verdict.value!r} is inconsistent with the dispositions; allowed: {sorted(allowed)}")


# ------------------------------------------------------------------ calls
def _call(record_id, agent, prompt, schema, validate):
    return base.call_structured(
        record_id=record_id, agent=agent, prompt_version=P.VERSIONS[agent], prompt=prompt,
        response_schema=schema, model=MODEL, temperature=TEMPERATURE, run_index=0,
        use_cache=True, extra_validate=validate,
    )


def run_a1(rid, flow):
    prompt = P.a1(flow)
    return _call(rid, "a1", prompt, SCClaimsResponse, lambda r: validate_claims(r.claims, "a1_c", ban=True)), prompt


def run_a2(rid, flow, a1):
    prompt = P.a2(flow, P.claims_text(a1.claims))

    def v(r):
        validate_claims(r.claims, "a2_c")
        validate_absences(r.absences)

    return _call(rid, "a2", prompt, SCBehaviourResponse, v), prompt


def run_a3(rid, flow, a1, a2):
    prompt = P.a3(flow, P.claims_text(a1.claims), P.claims_text(a2.claims) + "\n" + P.absences_text(a2.absences))

    def v(r):
        validate_claims(r.claims, "a3_c")
        validate_hypotheses(r.hypotheses, "a3_h")

    return _call(rid, "a3", prompt, SCHypothesisResponse, v), prompt


def run_a4(rid, flow):
    prompt = P.a4(flow)

    def v(r):
        validate_claims(r.claims, "a4_c")
        validate_absences(r.absences)
        validate_hypotheses(r.hypotheses, "a4_h")

    return _call(rid, "a4", prompt, SCBlindResponse, v), prompt


def run_a5a(rid, flow, value_names: Set[str]):
    prompt = P.a5a(flow)
    return _call(rid, "a5a", prompt, SCChecklistResponse, lambda r: validate_checklist(r.checklist, value_names)), prompt


def run_a5b(rid, checklist_text: str, justifications_text: str, yes_ids: List[str], creditable: Set[str]):
    prompt = P.a5b(checklist_text, justifications_text, yes_ids)
    return _call(rid, "a5b", prompt, SCVerdictResponse, lambda r: validate_verdict(r, yes_ids, creditable)), prompt
