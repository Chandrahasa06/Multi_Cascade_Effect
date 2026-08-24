"""
Tier-2 check: only runs when Tier-1 (grounding + entailment + behavior) is
already uncertain for a given turn. Resamples the same agent N times on the
same input and measures agreement across three angles: which action it picks,
how stable its confidence is, and how similar its claims are.
"""

import statistics

from config import CONSISTENCY_RESAMPLES, USE_EMBEDDINGS


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _jaccard(claims_a, claims_b):
    """Fallback text-overlap similarity when no embedding model is configured."""
    set_a = {w.lower() for c in claims_a for w in c.split()}
    set_b = {w.lower() for c in claims_b for w in c.split()}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class ConsistencyChecker:
    def __init__(self, agent, client=None):
        self.agent = agent
        self.client = client or agent.client

    def run(self, telemetry_text: str) -> dict:
        samples = [
            self.agent.decide(telemetry_text, temperature=0.9)
            for _ in range(CONSISTENCY_RESAMPLES)
        ]

        actions = [s.get("action") for s in samples]
        most_common = max(set(actions), key=actions.count)
        action_agreement = actions.count(most_common) / len(actions)

        confidences = [float(s.get("confidence", 0.5)) for s in samples]
        mean_conf = statistics.mean(confidences)
        cv = (statistics.pstdev(confidences) / mean_conf) if mean_conf > 1e-6 else 0.0
        numeric_consistency = 1 - min(cv, 1.0)

        claim_sets = [s.get("claims", []) for s in samples]
        pair_scores = []
        for i in range(len(claim_sets)):
            for j in range(i + 1, len(claim_sets)):
                if USE_EMBEDDINGS:
                    try:
                        emb_i = self.client.embed(" ".join(claim_sets[i]))
                        emb_j = self.client.embed(" ".join(claim_sets[j]))
                        pair_scores.append(_cosine(emb_i, emb_j))
                        continue
                    except Exception:
                        pass  # embedding model not pulled / call failed -> fall back
                pair_scores.append(_jaccard(claim_sets[i], claim_sets[j]))
        text_consistency = statistics.mean(pair_scores) if pair_scores else 1.0

        consistency_score = (
            0.5 * action_agreement + 0.25 * numeric_consistency + 0.25 * text_consistency
        )

        return {
            "consistency_score": consistency_score,
            "action_agreement": action_agreement,
            "numeric_consistency": numeric_consistency,
            "text_consistency": text_consistency,
            "sampled_actions": actions,
        }