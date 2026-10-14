"""Five-agent pipeline for the priors run. Each stage sees only the flow's own properties,
the written priors, and (for A2, A3, A5) earlier agents' output. No grounding, no cap.

Validators specific to this run:
  - claims: id prefix, unique ids, feature names from the flow's own measured values. The
    claim schema (agents/schema.PriorsClaim) has no relation field, so the *_typical
    relations, which this run forbids, cannot be produced
  - explanations: id prefix, unique ids, 3-5 of them with at least one benign and one
    hostile, profile names from the flow's own list, predictions checked by the existing
    flow-level text filter
  - verdict: at least one cited id that exists; credited explanation, if any, must exist
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from agents import base
from agents.prompts import a1_priors_v1, a2_priors_v1, a3_priors_v1, a4_priors_v1, a5_priors_v1
from agents.prompts import a1_priors_v2, a2_priors_v2, a3_priors_v2, a4_priors_v2, a5_priors_v2
from agents.schema import A5Response, PriorsClaimsResponse as ClaimsResponse, PriorsHypothesisResponse as HypothesisResponse
from agents.validators import validate_hypothesis_predictions_are_flow_level

MIN_EXPLANATIONS, MAX_EXPLANATIONS = 3, 5


def validate_claims(claims, prefix: str, feature_names: Set[str]) -> None:
    seen = set()
    for c in claims:
        if not c.claim_id.startswith(prefix):
            raise ValueError(f"claim_id {c.claim_id!r} must start with {prefix!r}")
        if c.claim_id in seen:
            raise ValueError(f"duplicate claim_id {c.claim_id!r}")
        seen.add(c.claim_id)
        for r in c.referenced_features:
            if r.name not in feature_names:
                raise ValueError(f"{c.claim_id}: {r.name!r} is not one of this flow's measured values")


def validate_explanations(resp: HypothesisResponse, prefix: str, profile_names: Set[str]) -> None:
    hyps = resp.hypotheses
    if not (MIN_EXPLANATIONS <= len(hyps) <= MAX_EXPLANATIONS):
        raise ValueError(f"expected {MIN_EXPLANATIONS} to {MAX_EXPLANATIONS} explanations, got {len(hyps)}")
    seen = set()
    for h in hyps:
        if not h.hypothesis_id.startswith(prefix):
            raise ValueError(f"hypothesis_id {h.hypothesis_id!r} must start with {prefix!r}")
        if h.hypothesis_id in seen:
            raise ValueError(f"duplicate hypothesis_id {h.hypothesis_id!r}")
        seen.add(h.hypothesis_id)
    if not any(h.benign for h in hyps) or all(h.benign for h in hyps):
        raise ValueError("need at least one benign and at least one hostile explanation")
    validate_hypothesis_predictions_are_flow_level(resp, profile_names)


def _call(record_id: str, agent: str, module, prompt: str, schema, validate, *, use_cache: bool, model: str):
    return base.call_structured(
        record_id=record_id, agent=agent, prompt_version=module.PROMPT_VERSION, prompt=prompt,
        response_schema=schema, model=model, temperature=0.7, run_index=0,
        use_cache=use_cache, extra_validate=validate,
    )


#: prompt sets: v1 (priors_text_v1) and v2 (protocol-aware priors_text_v2)
PROMPT_SETS = {
    "v1": {"a1": a1_priors_v1, "a2": a2_priors_v1, "a3": a3_priors_v1, "a4": a4_priors_v1, "a5": a5_priors_v1},
    "v2": {"a1": a1_priors_v2, "a2": a2_priors_v2, "a3": a3_priors_v2, "a4": a4_priors_v2, "a5": a5_priors_v2},
}


def run_a1(record_id, flow_text, feature_names, *, use_cache=True, model=base.DEFAULT_MODEL, prompts="v1"):
    prompt = PROMPT_SETS[prompts]["a1"].build_prompt(flow_text)
    return _call(record_id, "a1", PROMPT_SETS[prompts]["a1"], prompt, ClaimsResponse,
                 lambda r: validate_claims(r.claims, "a1_c", feature_names), use_cache=use_cache, model=model), prompt


def run_a2(record_id, flow_text, feature_names, a1, *, use_cache=True, model=base.DEFAULT_MODEL, prompts="v1"):
    prompt = PROMPT_SETS[prompts]["a2"].build_prompt(flow_text, a1.claims)
    return _call(record_id, "a2", PROMPT_SETS[prompts]["a2"], prompt, ClaimsResponse,
                 lambda r: validate_claims(r.claims, "a2_c", feature_names), use_cache=use_cache, model=model), prompt


def run_a3(record_id, flow_text, feature_names, profile_names, a1, a2, *, use_cache=True, model=base.DEFAULT_MODEL, prompts="v1"):
    prompt = PROMPT_SETS[prompts]["a3"].build_prompt(flow_text, a1.claims, a2.claims, sorted(profile_names))

    def validate(r):
        validate_claims(r.claims, "a3_c", feature_names)
        validate_explanations(r, "a3_h", set(profile_names))

    return _call(record_id, "a3", PROMPT_SETS[prompts]["a3"], prompt, HypothesisResponse, validate, use_cache=use_cache, model=model), prompt


def run_a4(record_id, flow_text, feature_names, profile_names, *, use_cache=True, model=base.DEFAULT_MODEL, prompts="v1"):
    prompt = PROMPT_SETS[prompts]["a4"].build_prompt(flow_text, sorted(profile_names))

    def validate(r):
        validate_claims(r.claims, "a4_c", feature_names)
        validate_explanations(r, "a4_h", set(profile_names))

    return _call(record_id, "a4", PROMPT_SETS[prompts]["a4"], prompt, HypothesisResponse, validate, use_cache=use_cache, model=model), prompt


def run_a5(record_id, flow_text, a1, a2, a3, a4, *, use_cache=True, model=base.DEFAULT_MODEL, prompts="v1"):
    chain_claims = [*a1.claims, *a2.claims, *a3.claims]
    prompt = PROMPT_SETS[prompts]["a5"].build_prompt(flow_text, chain_claims, a3.hypotheses, a4.claims, a4.hypotheses)
    known_claims = {c.claim_id for c in chain_claims} | {c.claim_id for c in a4.claims}
    known_hyps = {h.hypothesis_id for h in [*a3.hypotheses, *a4.hypotheses]}

    def validate(r: A5Response):
        if not r.cited_claim_ids or not (set(r.cited_claim_ids) & (known_claims | known_hyps)):
            raise ValueError("cite at least one claim_id from the reviews")
        if r.credited_hypothesis_id is not None and r.credited_hypothesis_id not in known_hyps:
            raise ValueError(f"credited_hypothesis_id {r.credited_hypothesis_id!r} is not one of {sorted(known_hyps)}")

    return _call(record_id, "a5", PROMPT_SETS[prompts]["a5"], prompt, A5Response, validate, use_cache=use_cache, model=model), prompt
