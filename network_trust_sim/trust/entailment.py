from core.ollama_client import OllamaClient

SYSTEM_PROMPT = """You are a logical-entailment judge. You will be given a set of CLAIMS \
(premises) and a CONCLUSION (an action + justification an agent reached). Decide whether the \
conclusion follows necessarily from the claims, without needing outside information.

Categories:
- "entailed": the conclusion is a reasonable, necessary consequence of the claims
- "unsupported_leap": the conclusion goes beyond what the claims justify
- "contradicted": the conclusion actually contradicts the claims

Respond with a single JSON object only:
{
  "verdict": "entailed",
  "score": 1.0,
  "failing_claims": [],
  "reason": "short reason"
}
Use score ~1.0 for entailed, ~0.4 for unsupported_leap, ~0.0 for contradicted, and your judgment \
for partial cases in between."""


class EntailmentChecker:
    def __init__(self, client: OllamaClient = None):
        self.client = client or OllamaClient()

    def check(self, claims: list, action: str, justification: str) -> dict:
        claims_block = "\n".join(f"- {c}" for c in claims) or "(no claims provided)"
        conclusion = f"Action: {action}\nJustification: {justification}"
        user_prompt = (
            f"CLAIMS:\n{claims_block}\n\nCONCLUSION:\n{conclusion}\n\nRespond with the JSON object now."
        )
        parsed = self.client.chat_json(SYSTEM_PROMPT, user_prompt, temperature=0.0)
        parsed.setdefault("verdict", "unsupported_leap")
        parsed.setdefault("score", 0.4)
        parsed.setdefault("failing_claims", [])
        return parsed