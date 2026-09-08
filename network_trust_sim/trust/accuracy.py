"""
Ground-truth-accuracy signal: did this agent's action, checked one turn later
against the real (uncorrupted) NetworkState, actually make sense? This is a
deliberately coarse, DETERMINISTIC directional check -- not a full claim-by-
claim comparison -- so it doesn't add a second LLM-judge surface next to
grounding/entailment. It's the "historical reliability" signal: an agent that
used to be accurate and suddenly isn't should lose trust even if any single
turn's justification still reads as plausible.
"""

from config import BASELINE_WINDOW, MIN_BASELINE_SAMPLES


def check_prediction(decision: dict, health_before: float, health_after: float) -> dict:
    """Directional check: did this agent's own domain get worse by next turn
    despite 'no_action_required', or hold steady/improve after the agent
    escalated to a corrective action? health_before/health_after are the
    agent's OWN-domain health (not global network health), one turn apart."""
    action = decision.get("action", "no_action_required")
    degraded = health_after < health_before - 0.02  # small tolerance for background noise
    improved = health_after > health_before + 0.02

    if action == "no_action_required":
        correct = not degraded
    else:
        correct = improved or not degraded  # escalating and holding steady still counts as reasonable
    return {"correct": correct, "health_before": health_before, "health_after": health_after}


class AccuracyTracker:
    """Rolling per-agent accuracy rate, same shape as BehaviorTracker's
    baseline. Returns None until an agent has any resolved turns, so
    TrustScoreEngine.tier1() can cleanly exclude the signal (rather than
    guessing a default) until there's real history."""

    def __init__(self):
        self.history = {}  # agent_id -> [bool, bool, ...]

    def score(self, agent_id: str):
        series = self.history.get(agent_id, [])
        if not series:
            return None
        return sum(series) / len(series)

    def update(self, agent_id: str, correct: bool):
        series = self.history.setdefault(agent_id, [])
        series.append(correct)
        if len(series) > BASELINE_WINDOW:
            series.pop(0)
