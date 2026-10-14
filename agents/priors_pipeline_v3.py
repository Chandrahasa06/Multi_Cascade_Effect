"""Five-agent pipeline for the priors-v3 run (forced citation). Same structure as
agents/priors_pipeline.py; different schemas and these extra validators:
  - A5 checklist: every condition id exactly once, no unknown ids, at least one observed value
    per row, value names from the flow's shown values or "Protocol"
  - A1: every claim cites at least one known condition id
  - A3, A4: every explanation cites at least one known condition id
  - any claim's condition ids must be known
No grounding, no cap.
"""
from __future__ import annotations

from typing import Iterable, Set

from agents import base
from agents.priors_pipeline import validate_claims, validate_explanations
from agents.prompts import a1_priors_v3, a2_priors_v3, a3_priors_v3, a4_priors_v3, a5_priors_v3
from agents.prompts.priors_text_v3 import CONDITION_IDS
from agents.schema import V3A5Response, V3ClaimsResponse, V3HypothesisResponse

KNOWN = set(CONDITION_IDS)


def validate_condition_refs(claims, require: bool) -> None:
    for c in claims:
        unknown = [x for x in c.condition_ids if x not in KNOWN]
        if unknown:
            raise ValueError(f"{c.claim_id}: unknown condition id(s) {unknown}; use {sorted(KNOWN)}")
        if require and not c.condition_ids:
            raise ValueError(f"{c.claim_id}: every claim must cite at least one condition id")


def validate_hypothesis_conditions(hyps) -> None:
    for h in hyps:
        if not h.condition_ids:
            raise ValueError(f"{h.hypothesis_id}: every explanation must cite at least one condition id")
        unknown = [x for x in h.condition_ids if x not in KNOWN]
        if unknown:
            raise ValueError(f"{h.hypothesis_id}: unknown condition id(s) {unknown}")


def validate_checklist(checklist, value_names: Set[str]) -> None:
    ids = [row.condition_id for row in checklist]
    missing = sorted(KNOWN - set(ids), key=lambda s: int(s[1:]))
    extra = [i for i in ids if i not in KNOWN]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if missing:
        raise ValueError(f"checklist is missing condition(s) {missing}; every condition needs one entry")
    if extra:
        raise ValueError(f"checklist has unknown condition id(s) {extra}")
    if dupes:
        raise ValueError(f"checklist has duplicate entries for {dupes}")
    allowed = value_names | {"Protocol"}
    for row in checklist:
        if not row.observed:
            raise ValueError(f"{row.condition_id}: report the value(s) you read")
        bad = [v.name for v in row.observed if v.name not in allowed]
        if bad:
            raise ValueError(f"{row.condition_id}: {bad} are not names of this flow's measured values")


def _call(record_id, agent, module, prompt, schema, validate, *, use_cache, model):
    return base.call_structured(
        record_id=record_id, agent=agent, prompt_version=module.PROMPT_VERSION, prompt=prompt,
        response_schema=schema, model=model, temperature=0.7, run_index=0,
        use_cache=use_cache, extra_validate=validate,
    )


def run_a1(record_id, flow_text, names, *, use_cache=True, model=base.DEFAULT_MODEL):
    prompt = a1_priors_v3.build_prompt(flow_text)

    def validate(r):
        validate_claims(r.claims, "a1_c", names)
        validate_condition_refs(r.claims, require=True)

    return _call(record_id, "a1", a1_priors_v3, prompt, V3ClaimsResponse, validate, use_cache=use_cache, model=model), prompt


def run_a2(record_id, flow_text, names, a1, *, use_cache=True, model=base.DEFAULT_MODEL):
    prompt = a2_priors_v3.build_prompt(flow_text, a1.claims)

    def validate(r):
        validate_claims(r.claims, "a2_c", names)
        validate_condition_refs(r.claims, require=False)

    return _call(record_id, "a2", a2_priors_v3, prompt, V3ClaimsResponse, validate, use_cache=use_cache, model=model), prompt


def run_a3(record_id, flow_text, names, profile, a1, a2, *, use_cache=True, model=base.DEFAULT_MODEL):
    prompt = a3_priors_v3.build_prompt(flow_text, a1.claims, a2.claims, sorted(profile))

    def validate(r):
        validate_claims(r.claims, "a3_c", names)
        validate_condition_refs(r.claims, require=False)
        validate_explanations(r, "a3_h", set(profile))
        validate_hypothesis_conditions(r.hypotheses)

    return _call(record_id, "a3", a3_priors_v3, prompt, V3HypothesisResponse, validate, use_cache=use_cache, model=model), prompt


def run_a4(record_id, flow_text, names, profile, *, use_cache=True, model=base.DEFAULT_MODEL):
    prompt = a4_priors_v3.build_prompt(flow_text, sorted(profile))

    def validate(r):
        validate_claims(r.claims, "a4_c", names)
        validate_condition_refs(r.claims, require=False)
        validate_explanations(r, "a4_h", set(profile))
        validate_hypothesis_conditions(r.hypotheses)

    return _call(record_id, "a4", a4_priors_v3, prompt, V3HypothesisResponse, validate, use_cache=use_cache, model=model), prompt


def run_a5(record_id, flow_text, names, a1, a2, a3, a4, *, use_cache=True, model=base.DEFAULT_MODEL):
    chain_claims = [*a1.claims, *a2.claims, *a3.claims]
    prompt = a5_priors_v3.build_prompt(flow_text, chain_claims, a3.hypotheses, a4.claims, a4.hypotheses)
    known_claims = {c.claim_id for c in chain_claims} | {c.claim_id for c in a4.claims}
    known_hyps = {h.hypothesis_id for h in [*a3.hypotheses, *a4.hypotheses]}

    def validate(r: V3A5Response):
        validate_checklist(r.checklist, names)
        if not r.cited_claim_ids or not (set(r.cited_claim_ids) & (known_claims | known_hyps)):
            raise ValueError("cite at least one claim_id from the reviews")
        if r.credited_hypothesis_id is not None and r.credited_hypothesis_id not in known_hyps:
            raise ValueError(f"credited_hypothesis_id {r.credited_hypothesis_id!r} is not one of {sorted(known_hyps)}")

    return _call(record_id, "a5", a5_priors_v3, prompt, V3A5Response, validate, use_cache=use_cache, model=model), prompt
