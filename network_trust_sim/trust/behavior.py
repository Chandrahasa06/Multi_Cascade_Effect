"""
Tracks each agent's own rolling baseline (claim count, justification length,
confidence) and scores how anomalous the current turn looks relative to that
agent's own history -- a domain-agnostic anomaly signal that catches injected
instructions changing an agent's output *shape*, not just its content.
"""

import statistics

from config import BASELINE_WINDOW, MIN_BASELINE_SAMPLES


class BehaviorTracker:
    def __init__(self):
        self.history = {}  # agent_id -> {"claim_count": [...], "length": [...], "confidence": [...]}

    def _get(self, agent_id):
        return self.history.setdefault(
            agent_id, {"claim_count": [], "length": [], "confidence": []}
        )

    @staticmethod
    def _zscore(x, series):
        if len(series) < MIN_BASELINE_SAMPLES:
            return 0.0
        mu = statistics.mean(series)
        sigma = statistics.pstdev(series) or 1e-6
        return (x - mu) / sigma

    def score(self, agent_id: str, decision: dict) -> dict:
        h = self._get(agent_id)
        claim_count = len(decision.get("claims", []))
        length = len(decision.get("justification", ""))
        confidence = float(decision.get("confidence", 0.5))

        z_claims = self._zscore(claim_count, h["claim_count"])
        z_length = self._zscore(length, h["length"])
        z_conf = self._zscore(confidence, h["confidence"])

        z_max = max(abs(z_claims), abs(z_length), abs(z_conf))
        deviation = min(z_max / 4.0, 1.0)  # normalize to [0,1], saturates at |z| = 4

        return {
            "behavior_deviation": deviation,
            "z_claim_count": z_claims,
            "z_length": z_length,
            "z_confidence": z_conf,
        }

    def update(self, agent_id: str, decision: dict):
        """Grow the rolling baseline. Called every turn after scoring; a genuinely
        compromised turn will nudge the baseline slightly, which is a known
        trade-off of an adaptive baseline -- see README for mitigation options."""
        h = self._get(agent_id)
        h["claim_count"].append(len(decision.get("claims", [])))
        h["length"].append(len(decision.get("justification", "")))
        h["confidence"].append(float(decision.get("confidence", 0.5)))
        for key in h:
            if len(h[key]) > BASELINE_WINDOW:
                h[key].pop(0)