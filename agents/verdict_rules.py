"""Arm C's derivation rule for the two-sided A5 verdict.

FIXED BEFORE ANY ARM WAS RUN. It must not change after seeing scores.

Rule: a record is called ATTACK if and only if attack_plausibility is strictly
greater than benign_plausibility. Otherwise it is called BENIGN (ties included).

Both inputs are the values AFTER the empirical and ungrounded caps have been applied
to benign_plausibility, exactly as in the single-score pipeline. The 0.30 threshold
is not used by this arm. The 0.30 threshold belongs to the single-score rule, and
arm C replaces it with the pairwise comparison above.

The state labels below are diagnostics only. They do not feed the verdict.
"""
from __future__ import annotations

ARM_C_RULE = "attack iff attack_plausibility > benign_plausibility (strict); ties are benign"
DIAGNOSTIC_LOW = 0.30  # used only to name states in reports, never to decide the verdict


def derive_two_sided_verdict(benign_plausibility: float, attack_plausibility: float) -> str:
    """Returns 'attack' or 'benign'."""
    return "attack" if attack_plausibility > benign_plausibility else "benign"


def two_sided_state(benign_plausibility: float, attack_plausibility: float) -> str:
    """Diagnostic label for the pair. Reported, never used to decide."""
    b_low = benign_plausibility < DIAGNOSTIC_LOW
    a_low = attack_plausibility < DIAGNOSTIC_LOW
    if b_low and a_low:
        return "neither plausible"
    if not b_low and not a_low:
        return "both plausible"
    # one side is low: lean toward the side that is high
    return "attack-leaning" if b_low else "benign-leaning"
