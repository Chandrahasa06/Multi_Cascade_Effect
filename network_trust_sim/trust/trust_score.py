"""
Combines grounding + entailment + behavior (Tier 1, runs every turn for every
agent -- fair by construction) with consistency resampling (Tier 2, only runs
when Tier 1 is uncertain -- a gate that can only lower the score, never
inflate it), then applies exponential decay across turns so one bad turn
matters without being permanently diluted, and one good turn doesn't erase
history immediately.
"""

from config import (
    W_GROUNDING,
    W_ENTAILMENT,
    W_BEHAVIOR,
    ESCALATION_THRESHOLD,
    LAMBDA_DECAY,
    FLAG_THRESHOLD,
    TURN_FLAG_THRESHOLD,
)


class TrustScoreEngine:
    def __init__(self, grounding_checker, entailment_checker, behavior_tracker):
        self.grounding_checker = grounding_checker
        self.entailment_checker = entailment_checker
        self.behavior_tracker = behavior_tracker
        self.trust_history = {}  # agent_id -> last TrustScore

    def tier1(self, agent_id, source_text, decision):
        grounding = self.grounding_checker.check(source_text, decision.get("claims", []))
        entailment = self.entailment_checker.check(
            decision.get("claims", []), decision.get("action", ""), decision.get("justification", "")
        )
        behavior = self.behavior_tracker.score(agent_id, decision)

        s1 = (
            W_GROUNDING * grounding["grounding_rate"]
            + W_ENTAILMENT * entailment["score"]
            + W_BEHAVIOR * (1 - behavior["behavior_deviation"])
        )
        return s1, {"grounding": grounding, "entailment": entailment, "behavior": behavior}

    def evaluate(self, agent_id, source_text, decision, consistency_checker=None, telemetry_text=None):
        s1, detail = self.tier1(agent_id, source_text, decision)

        escalated = s1 < ESCALATION_THRESHOLD
        consistency = None
        if escalated and consistency_checker is not None:
            consistency = consistency_checker.run(telemetry_text)
            s_final = s1 * consistency["consistency_score"]  # can only lower s1, never raise it
        else:
            s_final = s1

        prev = self.trust_history.get(agent_id, 1.0)  # innocent-until-proven-otherwise prior
        trust_score = LAMBDA_DECAY * s_final + (1 - LAMBDA_DECAY) * prev
        self.trust_history[agent_id] = trust_score

        self.behavior_tracker.update(agent_id, decision)

        return {
            "tier1_score": s1,
            "escalated": escalated,
            "consistency": consistency,
            "turn_score": s_final,
            # Immediate signal: this turn alone looked bad. Use this to localize
            # *when* an agent was compromised -- it doesn't wait for decay to catch up.
            "turn_flagged": s_final < TURN_FLAG_THRESHOLD,
            "trust_score": trust_score,
            # Sustained signal: accumulated distrust across turns. Slower to trip,
            # more robust to one noisy turn -- use this to decide whether to actually
            # take the agent offline, not just to spot the incident turn.
            "trust_flagged": trust_score < FLAG_THRESHOLD,
            "detail": detail,
        }