"""
Thin wrapper around Ollama's local HTTP API (/api/chat, /api/embeddings).
Nothing here talks to the internet -- it only ever hits OLLAMA_HOST.
"""

import json
import re

import requests
from json_repair import repair_json

from config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_EMBED_MODEL, VERBOSE


def extract_json(raw: str) -> dict:
    """Pull the first {...} block out of a model response and parse it.
    Local models (especially smaller ones like llama3.1:8b) routinely produce
    JSON with unescaped quotes/colons inside string values, trailing commas,
    or stray commentary -- json_repair handles all of that far more reliably
    than hand-rolled regex fixes."""
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{raw[:500]}")

    text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json(text)
        parsed = json.loads(repaired)
        if not isinstance(parsed, dict):
            raise ValueError(f"Repaired JSON was not an object:\n{raw[:500]}")
        return parsed


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host.rstrip("/")
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.7, label: str = "") -> str:
        """Single-turn chat call. Returns the raw text response."""
        if VERBOSE and label:
            # A single dot per call -- just enough to show a long escalation
            # (initial decision + resamples) is still alive, without the noise
            # of a "label...done" line for every one of them.
            print(".", end="", flush=True)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    def chat_json(self, system: str, user: str, temperature: float = 0.7, label: str = "") -> dict:
        """Chat call that expects a JSON object back, with extraction/repair applied.
        Retries once with a stricter reminder if parsing fails."""
        raw = self.chat(system, user, temperature=temperature, label=label)
        try:
            return extract_json(raw)
        except (ValueError, json.JSONDecodeError):
            strict_user = user + "\n\nIMPORTANT: reply with ONLY the JSON object, nothing else."
            raw2 = self.chat(system, strict_user, temperature=temperature, label=f"{label} (retry)")
            return extract_json(raw2)

    def embed(self, text: str, model: str = OLLAMA_EMBED_MODEL):
        payload = {"model": model, "prompt": text}
        resp = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("embedding", [])