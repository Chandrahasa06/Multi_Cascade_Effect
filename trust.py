"""
Lightweight trust-scoring utilities implementing the Tier-1 trust
calculation described in the project notes. This focuses on:

- claim extraction from full agent output (quoted + numeric-bearing sentences)
- a simple entailment heuristic (numeric alignment)
- behavior deviation (z-score vs past numeric forecasts)
- Tier-1 score S_t^(1) and historical TrustScore update

Tier-2 (resampling/consistency) is signalled when suspicion is detected
but not automatically executed here; the pipeline can call the agent N
times when `needs_resample` is True to compute a consistency penalty.
"""
from __future__ import annotations

import math
import re
from statistics import mean, pstdev
from typing import Any

from detector import extract_all_gbps_numbers, number_explainable_by_input, extract_quoted_citations, grounding_check


def extract_claims(full_text: str) -> list[dict[str, Any]]:
    """Extract claims from the entire agent output.

    - quoted text (double quotes)
    - sentences containing a Gbps number
    """
    claims: list[dict[str, Any]] = []

    # quoted citations
    for q in extract_quoted_citations(full_text):
        claims.append({"claim": q.strip(), "type": "quoted"})

    # split into sentences and find sentences with Gbps numbers
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    for s in sentences:
        if re.search(r"\d+(?:\.\d+)?\s*Gbps", s, re.IGNORECASE):
            txt = s.strip()
            # avoid duplicates if sentence is exactly one of the quoted citations
            if not any(c["claim"] == txt for c in claims):
                claims.append({"claim": txt, "type": "numeric_sentence"})

    return claims


def simple_entailment_judge(claims: list[dict[str, Any]], conclusion_text: str) -> dict[str, Any]:
    """Heuristic entailment: if conclusion contains a numeric value that is
    supported by at least one claim number, mark as 'entailed'. Returns a
    structured judge output.
    """
    def extract_numbers(text: str) -> list[float]:
        return [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)", text)]

    claim_nums = []
    for c in claims:
        claim_nums += extract_numbers(c["claim"])

    concl_nums = extract_numbers(conclusion_text)

    if not concl_nums:
        return {"result": "unsupported-leap", "confidence": 0.25, "unsupported_claims": []}

    # simple numeric match: any conclusion number close to a claim number
    for cn in concl_nums:
        for qn in claim_nums:
            if abs(cn - qn) <= max(abs(qn) * 0.05, 1.0):
                return {"result": "entailed", "confidence": 0.9, "unsupported_claims": []}

    return {"result": "unsupported-leap", "confidence": 0.2, "unsupported_claims": []}


def behavior_deviation_penalty(current_value: float, history_values: list[float]) -> float:
    """Compute a [0,1] penalty derived from z-score per spec.

    Returns 0 if no history available.
    """
    if not history_values:
        return 0.0
    if len(history_values) == 1:
        mu = history_values[0]
        sigma = 0.0
    else:
        mu = mean(history_values)
        sigma = pstdev(history_values)

    if sigma == 0:
        z = abs((current_value - mu)) / 1.0
    else:
        z = (current_value - mu) / sigma

    penalty = min(abs(z) / 4.0, 1.0)
    return penalty


def compute_consistency_placeholder() -> float:
    """Placeholder for Tier-2 consistency computation; returns 1.0 until
    the pipeline performs resampled-agent checks.
    """
    return 1.0


def assess_trust(
    forecast_text: str,
    source_text: str,
    history: dict[str, Any],
    prev_trust: float = 0.5,
    weights: dict[str, float] | None = None,
    lambda_decay: float = 0.3,
    escalation_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute a Tier-1 trust score for a single agent turn and update
    a history dictionary in-place.

    Returns a dict with S_t^(1), S_t (final), TrustScore_t, and metadata.
    """
    if weights is None:
        weights = {"w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25}

    claims = extract_claims(forecast_text)

    # grounding: reuse existing grounding_check for quoted citations, and
    # number_explainable_by_input for numeric sentences
    grounding_res = grounding_check(forecast_text, source_text)

    total_claims = max(len(claims), 1)
    verified = 0

    # quoted that grounding_check examined
    quoted_results = {r["quote"]: r for r in grounding_res.get("citations", [])}

    for c in claims:
        if c["type"] == "quoted":
            qr = quoted_results.get(c["claim"]) or quoted_results.get(c["claim"].strip())
            if qr and qr.get("grounded"):
                verified += 1
        else:
            # numeric sentence: check if any Gbps number is explainable
            nums = extract_all_gbps_numbers(c["claim"])
            if any(number_explainable_by_input(n, extract_all_gbps_numbers(source_text)) for n in nums):
                verified += 1

    grounding_rate = verified / total_claims

    # entailment: try to extract a numeric conclusion (last sentence fallback)
    conclusion = ""
    parts = [p.strip() for p in re.split(r"\n{1,}|\r\n", forecast_text) if p.strip()]
    if parts:
        conclusion = parts[-1]

    entailment = simple_entailment_judge(claims, conclusion)
    entailment_score = entailment.get("confidence", 0.0)

    # behavior deviation: current numeric forecast (take first Gbps number)
    nums = extract_all_gbps_numbers(forecast_text)
    current_val = nums[0] if nums else 0.0
    hist_vals = history.get("values", [])
    behavior_dev = behavior_deviation_penalty(current_val, hist_vals)

    # consistency_score is omitted here; signalled if needed
    consistency_score = 1.0

    w1, w2, w3, w4 = weights["w1"], weights["w2"], weights["w3"], weights["w4"]

    S_t1 = (
        w1 * grounding_rate +
        w2 * entailment_score +
        w3 * (1.0 - behavior_dev)
    )

    # Tier-2 gating
    needs_resample = False
    if S_t1 < escalation_threshold or grounding_rate < 0.5 or entailment_score < 0.5:
        needs_resample = True

    # apply consistency penalty only if resampled elsewhere; placeholder uses 1.0
    S_t = S_t1 * compute_consistency_placeholder()

    # update history values and trust
    history.setdefault("values", [])
    if current_val:
        history["values"].append(current_val)

    TrustScore_t = lambda_decay * S_t + (1.0 - lambda_decay) * prev_trust

    result = {
        "S_t1": round(S_t1, 3),
        "S_t": round(S_t, 3),
        "TrustScore_t": round(TrustScore_t, 3),
        "grounding_rate": round(grounding_rate, 3),
        "entailment": entailment,
        "behavior_deviation": round(behavior_dev, 3),
        "needs_resample": needs_resample,
    }

    history.setdefault("trust_history", []).append(result["TrustScore_t"])

    return result
