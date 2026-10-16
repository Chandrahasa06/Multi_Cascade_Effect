"""OpenRouter backend for ``agents.base.call_structured``.

- Model slugs come only from config/openrouter.json; nothing here names a model.
- Every request pins the configured provider: ``provider.only = order = [provider]``,
  ``allow_fallbacks: false``, ``require_parameters: true`` (a provider that cannot honour
  the strict schema is refused, never silently substituted).
- ``openrouter/free`` (a router that picks an arbitrary free model) is rejected loudly.
- Output is constrained natively: ``response_format: json_schema, strict: true``. The schema
  is the Pydantic response model reduced to the subset strict mode accepts across providers
  (see ``strict_schema``); the callers' own validators still run on every response, as a
  second check, so constraints the schema cannot carry are still enforced.
- Every call returns provider, model, generation id and cost, for the cache record.
- ``verify`` checks each configured slug against /api/v1/models and its pinned provider
  against /api/v1/models/<slug>/endpoints before any call is made.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

from pydantic import BaseModel

CONFIG_PATH = Path("config/openrouter.json")
FORBIDDEN_SLUGS = frozenset({"openrouter/free"})
MAX_HTTP_RETRIES = 6
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
REQUEST_TIMEOUT_S = 300.0

#: keywords strict structured output does not accept on every provider (Anthropic rejects
#: numeric and string bounds and array bounds other than minItems 0/1). Dropped from the
#: schema sent; the Pydantic model and the callers' validators still enforce them.
_DROP_KEYS = ("title", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
              "multipleOf", "minLength", "maxLength", "pattern", "maxItems")


class OpenRouterConfigError(RuntimeError):
    pass


class OpenRouterError(RuntimeError):
    pass


class CostCeilingExceeded(OpenRouterError):
    """A runaway guard tripped. Raised before a call would be sent, never after."""


# ------------------------------------------------------------------ cost guard
#: Every priced call is appended to the ledger, so the per-run and global totals survive
#: across processes (and a resumed run counts what it already spent). Calls in flight when a
#: ceiling trips still complete, so a stop can overshoot by at most the concurrent calls' cost.
_guard_lock = threading.Lock()
_guard: Dict = {}
_sem = threading.BoundedSemaphore(1)


def ledger_totals(ledger: Path) -> Tuple[float, Dict[str, float]]:
    total, by_model = 0.0, {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                total += e["cost_usd"]
                by_model[e["model"]] = by_model.get(e["model"], 0.0) + e["cost_usd"]
    return total, by_model


def set_cost_guard(model: str, run_ceiling: float, global_ceiling: float, ledger: Path, max_concurrent: int) -> None:
    global _sem
    total, by_model = ledger_totals(ledger)
    with _guard_lock:
        _guard.clear()
        _guard.update(model=model, run_ceiling=float(run_ceiling), global_ceiling=float(global_ceiling),
                      ledger=Path(ledger), run_spent=by_model.get(model, 0.0), global_spent=total, tripped=None)
    _sem = threading.BoundedSemaphore(max_concurrent)


def cost_guard_state() -> Dict:
    with _guard_lock:
        return dict(_guard)


def _check_ceiling(model: str) -> None:
    with _guard_lock:
        if not _guard:
            return
        g = _guard
        if g["tripped"] is None:
            if g["run_spent"] >= g["run_ceiling"]:
                g["tripped"] = f"per-run ceiling ${g['run_ceiling']:.2f} for {g['model']} reached (spent ${g['run_spent']:.4f})"
            elif g["global_spent"] >= g["global_ceiling"]:
                g["tripped"] = f"global ceiling ${g['global_ceiling']:.2f} reached (spent ${g['global_spent']:.4f})"
        if g["tripped"]:
            raise CostCeilingExceeded(g["tripped"])


def _charge(meta: Dict, pricing: Dict) -> None:
    cost = meta.get("cost_usd")
    estimated = cost is None
    if estimated:  # never let an unpriced call slip past the guard
        cost = ((meta.get("input_tokens") or 0) * float(pricing.get("prompt", 0))
                + (meta.get("output_tokens") or 0) * float(pricing.get("completion", 0)))
    with _guard_lock:
        if not _guard:
            return
        _guard["run_spent"] += cost
        _guard["global_spent"] += cost
        with open(_guard["ledger"], "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "model": meta.get("requested_model"),
                                "generation_id": meta.get("generation_id"), "cost_usd": cost,
                                "estimated": estimated}) + "\n")


# ------------------------------------------------------------------ config
@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    named = set(cfg.get("models", {})) | set(cfg.get("sequence", [])) | {cfg.get("shakedown_model")}
    bad = sorted(s for s in named if s and (s in FORBIDDEN_SLUGS or s.startswith("openrouter/free")))
    if bad:
        raise OpenRouterConfigError(
            f"{path}: {bad} is the OpenRouter free router, which picks an arbitrary model per request. "
            "A comparison needs a fixed model and provider. Remove it from the config."
        )
    missing = sorted(s for s in cfg.get("sequence", []) if s not in cfg.get("models", {}))
    if missing:
        raise OpenRouterConfigError(f"{path}: sequence names {missing} with no entry under 'models'")
    for slug, m in cfg.get("models", {}).items():
        if not m.get("provider"):
            raise OpenRouterConfigError(f"{path}: {slug} has no pinned provider")
    return cfg


def is_configured(model: str) -> bool:
    try:
        return model in load_config()["models"]
    except FileNotFoundError:
        return False


def model_config(model: str) -> dict:
    cfg = load_config()
    if model not in cfg["models"]:
        raise OpenRouterConfigError(f"{model!r} is not in {CONFIG_PATH}")
    return cfg["models"][model]


def temperature_for(model: str) -> Optional[float]:
    return model_config(model).get("temperature")


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise OpenRouterConfigError("OPENROUTER_API_KEY not set (environment or .env)")
    return key


# ------------------------------------------------------------------ verification
@lru_cache(maxsize=None)
def verify(model: str) -> dict:
    """Live check, once per process per slug. Returns the pinned endpoint's details.
    Fails loudly if the slug, the provider, or a required parameter is not there."""
    import httpx

    m = model_config(model)
    base = load_config()["api_base"]
    listed = httpx.get(f"{base}/models", timeout=60).json()["data"]
    entry = next((x for x in listed if x["id"] == model), None)
    if entry is None:
        raise OpenRouterConfigError(f"{model!r} is not listed by {base}/models -- check the slug")
    eps = httpx.get(f"{base}/models/{model}/endpoints", timeout=60).json()["data"]["endpoints"]
    ep = next((e for e in eps if e.get("tag") == m["provider"]), None)
    if ep is None:
        raise OpenRouterConfigError(
            f"{model}: provider {m['provider']!r} not among its endpoints {sorted(e.get('tag') for e in eps)}")
    need = {"response_format", "structured_outputs"} | ({"temperature"} if m.get("temperature") is not None else set())
    lacking = sorted(need - set(ep.get("supported_parameters", [])))
    if lacking:
        raise OpenRouterConfigError(f"{model} via {m['provider']}: endpoint does not support {lacking}")
    return {"model": model, "provider_tag": m["provider"], "provider_name": ep.get("provider_name"),
            "endpoint_name": ep.get("name"), "pricing": ep.get("pricing", {}), "context_length": entry.get("context_length"),
            "quantization": ep.get("quantization")}


# ------------------------------------------------------------------ schema
def strict_schema(model_cls: Type[BaseModel]) -> dict:
    schema = model_cls.model_json_schema()

    def walk(node):
        if isinstance(node, dict):
            for k in _DROP_KEYS:
                node.pop(k, None)
            if node.get("minItems", 0) > 1:
                node["minItems"] = 1
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for v in list(node.values()):
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


# ------------------------------------------------------------------ request
def _post(payload: dict) -> dict:
    import httpx

    base = load_config()["api_base"]
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    last = None
    for attempt in range(MAX_HTTP_RETRIES):
        try:
            r = httpx.post(f"{base}/chat/completions", headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S)
        except httpx.TransportError as exc:
            last = exc
            time.sleep(min(60.0, 2.0 ** attempt) + random.uniform(0, 1))
            continue
        body = r.json() if r.content else {}
        err = body.get("error") if isinstance(body, dict) else None
        code = r.status_code if r.status_code != 200 else (int(err.get("code", 0) or 0) if err else 200)
        if code == 200 and body.get("choices"):
            return body
        last = OpenRouterError(f"HTTP {r.status_code}: {json.dumps(err or body)[:800]}")
        if code in RETRYABLE_STATUS and attempt < MAX_HTTP_RETRIES - 1:
            wait = float(r.headers.get("retry-after", 0) or 0) or min(60.0, 2.0 ** attempt)
            print(f"[openrouter] {code}, retry {attempt + 1}/{MAX_HTTP_RETRIES - 1} in {wait:.1f}s", flush=True)
            time.sleep(wait + random.uniform(0, 1))
            continue
        raise last
    raise last


def _cost_from_generation(gen_id: str) -> Optional[float]:
    import httpx

    base = load_config()["api_base"]
    for _ in range(5):
        r = httpx.get(f"{base}/generation", params={"id": gen_id},
                      headers={"Authorization": f"Bearer {_api_key()}"}, timeout=60)
        if r.status_code == 200:
            return r.json()["data"].get("total_cost")
        time.sleep(2.0)
    return None


def generate(model: str, prompt: str, response_schema: Type[BaseModel], temperature: Optional[float]) -> Tuple[str, Dict]:
    """One completion. Returns (raw text, call metadata). Raises on any provider mismatch."""
    m = model_config(model)
    if temperature != m.get("temperature"):
        raise OpenRouterConfigError(f"{model}: temperature {temperature!r} differs from config {m.get('temperature')!r}")
    pinned = verify(model)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": m.get("max_tokens", 16000),
        "response_format": {"type": "json_schema", "json_schema": {
            "name": response_schema.__name__, "strict": True, "schema": strict_schema(response_schema)}},
        "provider": {"order": [m["provider"]], "only": [m["provider"]],
                     "allow_fallbacks": False, "require_parameters": True},
        "usage": {"include": True},
    }
    if pinned.get("quantization") not in (None, "unknown"):
        # the provider name alone does not tell an fp8 deployment from an fp4 one
        payload["provider"]["quantizations"] = [pinned["quantization"]]
    if m.get("temperature") is not None:
        payload["temperature"] = m["temperature"]
    if m.get("reasoning") is not None:
        payload["reasoning"] = m["reasoning"]
    _check_ceiling(model)
    t0 = time.monotonic()
    with _sem:
        body = _post(payload)
    latency = time.monotonic() - t0
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    cost = usage.get("cost")
    if cost is None and body.get("id"):
        cost = _cost_from_generation(body["id"])
    meta = {
        "generation_id": body.get("id"),
        "provider": body.get("provider"),
        "model": body.get("model"),
        "requested_model": model,
        "pinned_provider": m["provider"],
        "cost_usd": cost,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "latency_s": latency,
    }
    _charge(meta, pinned.get("pricing", {}))
    if pinned["provider_name"] and meta["provider"] != pinned["provider_name"]:
        raise OpenRouterError(f"{model}: served by {meta['provider']!r}, pinned {pinned['provider_name']!r}")
    return choice["message"].get("content") or "", meta
