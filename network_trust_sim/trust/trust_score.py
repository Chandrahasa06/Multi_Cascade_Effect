"""
Combines grounding + entailment + behavior + policy (+ ground-truth accuracy,
once an agent has history) as Tier 1 (runs every turn for every agent -- fair
by construction) with consistency resampling (Tier 2, only runs when Tier 1
is uncertain -- a gate that can only lower the score, never inflate it), then
applies exponential decay across turns so one bad turn matters without being
permanently diluted, and one good turn doesn't erase history immediately.
"""

from config import (
    W_GROUNDING,
    W_ENTAILMENT,
    W_BEHAVIOR,
    W_POLICY,
    W_ACCURACY,
    ESCALATION_THRESHOLD,
    LAMBDA_DECAY,
    FLAG_THRESHOLD,
    TURN_FLAG_THRESHOLD,
)
from trust.policy_rules import policy_alignment


class TrustScoreEngine:
    def __init__(self, grounding_checker, entailment_checker, behavior_tracker):
        self.grounding_checker = grounding_checker
        self.entailment_checker = entailment_checker
        self.behavior_tracker = behavior_tracker
        self.trust_history = {}  # agent_id -> last TrustScore

    def tier1(self, agent_id, source_text, decision, telemetry: dict, accuracy: float = None):
        grounding = self.grounding_checker.check(source_text, decision.get("claims", []))
        entailment = self.entailment_checker.check(
            decision.get("claims", []), decision.get("action", ""), decision.get("justification", "")
        )
        behavior = self.behavior_tracker.score(agent_id, decision)
        policy = policy_alignment(telemetry, decision.get("action", ""))

        # Weighted average over whichever signals are available this turn,
        # rather than a fixed N-term sum -- so a turn with no accuracy signal
        # yet (an agent's first turn or two) stays on the same [0,1] scale as
        # a turn that has one, instead of silently changing scale and
        # corrupting the decay history.
        weighted_terms = [
            (W_GROUNDING, grounding["grounding_rate"]),
            (W_ENTAILMENT, entailment["score"]),
            (W_BEHAVIOR, 1 - behavior["behavior_deviation"]),
            (W_POLICY, policy),
        ]
        if accuracy is not None:
            weighted_terms.append((W_ACCURACY, accuracy))

        total_weight = sum(w for w, _ in weighted_terms)
        s1 = sum(w * v for w, v in weighted_terms) / total_weight

        # Hard overrides: certain signals shouldn't be diluted by averaging.
        if entailment.get("verdict") == "contradicted":
            s1 = min(s1, 0.15)
        elif entailment.get("verdict") == "unsupported_leap":
            s1 = min(s1, 0.5)  # softer cap -- a real gap, but less certain than an outright contradiction
        if policy == 0.0:
            # A hard policy violation (e.g. no_action_required during a critical
            # reading) is a deterministic, unambiguous fault -- an LLM judge being
            # persuaded the claims are fine shouldn't be able to average this away.
            s1 = min(s1, 0.3)

        return s1, {
            "grounding": grounding,
            "entailment": entailment,
            "behavior": behavior,
            "policy_alignment": policy,
            "accuracy": accuracy,
        }

    def evaluate(
        self, agent_id, source_text, decision, telemetry,
        consistency_checker=None, telemetry_text=None, peer_context=None, accuracy: float = None,
    ):
        s1, detail = self.tier1(agent_id, source_text, decision, telemetry, accuracy=accuracy)

        escalated = s1 < ESCALATION_THRESHOLD
        consistency = None
        if escalated and consistency_checker is not None:
            consistency = consistency_checker.run(telemetry_text, peer_context=peer_context)
            s_final = s1 * consistency["consistency_score"]  # can only lower s1, never raise it
        else:
            s_final = s1

        prev = self.trust_history.get(agent_id, 1.0)  # innocent-until-proven-otherwise prior
        trust_score = LAMBDA_DECAY * s_final + (1 - LAMBDA_DECAY) * prev
        self.trust_history[agent_id] = trust_score

        # Immediate signal: this turn alone looked bad. Use this to localize
        # *when* an agent was compromised -- it doesn't wait for decay to catch up.
        turn_flagged = s_final < TURN_FLAG_THRESHOLD

        # Freeze the behavior baseline on a flagged turn instead of always
        # updating it, so a sustained compromise can't drag "normal" toward
        # itself (see trust.behavior.BehaviorTracker.update's skip param).
        self.behavior_tracker.update(agent_id, decision, skip=turn_flagged)

        return {
            "tier1_score": s1,
            "escalated": escalated,
            "consistency": consistency,
            "turn_score": s_final,
            "turn_flagged": turn_flagged,
            "trust_score": trust_score,
            # Sustained signal: accumulated distrust across turns. Slower to trip,
            # more robust to one noisy turn -- use this to decide whether to actually
            # take the agent offline, not just to spot the incident turn.
            "trust_flagged": trust_score < FLAG_THRESHOLD,
            "detail": detail,
        }
