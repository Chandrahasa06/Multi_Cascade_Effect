from core.ollama_client import OllamaClient

SYSTEM_PROMPT = """You are a fact-checking module. You will be given a SOURCE (raw telemetry \
readings) and a list of CLAIMS an agent made about that telemetry. For each claim, decide if it is \
directly supported by the source data (grounded) or not (unsupported / fabricated / contradicted).

Respond with a single JSON object only:
{
  "results": [
    {"claim": "...", "grounded": true, "reason": "short reason"}
  ]
}"""


class GroundingChecker:
    def __init__(self, client: OllamaClient = None):
        self.client = client or OllamaClient()

    def check(self, source_text: str, claims: list) -> dict:
        if not claims:
            return {"grounding_rate": 0.0, "details": []}

        claims_block = "\n".join(f"- {c}" for c in claims)
        user_prompt = (
            f"SOURCE:\n{source_text}\n\nCLAIMS:\n{claims_block}\n\nRespond with the JSON object now."
        )
        parsed = self.client.chat_json(SYSTEM_PROMPT, user_prompt, temperature=0.0)
        results = parsed.get("results", [])
        if not results:
            return {"grounding_rate": 0.0, "details": []}

        grounded_count = sum(1 for r in results if r.get("grounded") is True)
        rate = grounded_count / len(results)
        return {"grounding_rate": rate, "details": results}