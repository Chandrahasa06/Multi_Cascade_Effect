"""
Verification Policy Matrix: combine Agent Trust (how reliable has this agent
been) with Recommendation Risk (how dangerous would this specific call be if
wrong) into a verification action. This is what lets the swarm avoid
verifying every single recommendation -- the "without grinding the control
plane to a halt" requirement -- by only spending verification budget where
low trust meets high risk, instead of running full cross-verification
unconditionally on every turn.
"""

from config import (
    TRUST_LOW_THRESHOLD,
    TRUST_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    RISK_HIGH_THRESHOLD,
)

VERIFICATION_MATRIX = {
    ("high", "low"): "execute",
    ("high", "medium"): "lightweight_verification",
    ("high", "high"): "independent_verification",
    ("medium", "low"): "lightweight_verification",
    ("medium", "medium"): "multiple_verification",
    ("medium", "high"): "strong_verification",
    ("low", "low"): "lightweight_verification",
    ("low", "medium"): "multiple_verification",
    ("low", "high"): "reject_quarantine",
}


def trust_bucket(trust_score: float) -> str:
    if trust_score >= TRUST_HIGH_THRESHOLD:
        return "high"
    if trust_score >= TRUST_LOW_THRESHOLD:
        return "medium"
    return "low"


def risk_bucket(risk_score: float) -> str:
    if risk_score <= RISK_LOW_THRESHOLD:
        return "low"
    if risk_score <= RISK_HIGH_THRESHOLD:
        return "medium"
    return "high"


def decide_verification(trust_score: float, risk_score: float) -> dict:
    t_bucket, r_bucket = trust_bucket(trust_score), risk_bucket(risk_score)
    action = VERIFICATION_MATRIX[(t_bucket, r_bucket)]
    return {
        "verification_action": action,
        "trust_bucket": t_bucket,
        "risk_bucket": r_bucket,
    }
