"""Gemini API client: rate limiting, retry/backoff, daily-quota guard,
and disk caching.

Provider is intentionally behind this one module — every agent calls
``call_structured`` and never touches ``google.genai`` directly, so
swapping models/providers later doesn't touch agent code.

Free-tier rate limits are the binding constraint, so this is built in
from the start rather than patched on later. RPM_LIMIT below is set from
a real measured 429 against this key/model (see its own comment), not
from documentation -- the originally assumed ~10 RPM was wrong (real cap
is 5 RPM for gemini-3.8-flash on this key):

- a token-bucket limiter under the RPM cap, shared process-wide
- backoff with jitter on 429/RESOURCE_EXHAUSTED and transient 503s,
  honoring the API's own `retryDelay` hint when the error body carries
  one rather than guessing, capped retries
- a persistent daily request counter (survives process restarts) that
  warns approaching the daily cap and refuses once at it, so a run
  can't silently die three-quarters through
- disk caching keyed on (record_id, agent, prompt_version, model,
  temperature, run_index) so a killed/interrupted run resumes for free
  and a prompt-version bump invalidates only its own cache entries
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

#: "gemini-3-flash" (the originally requested name) doesn't exist on the
#: API. Model selection was verified live against this project's key, not
#: assumed from docs (see STATUS.md's stop-point-1/2 verification runs).
#: Tested and rejected: gemini-2.5-flash / gemini-2.0-flash / their
#: -lite variants are all 404 "no longer available" on this key (retired,
#: not quota-limited); gemini-3.5-flash, gemini-3.6-flash, gemini-3.8-flash
#: are all throttled to 5 RPM, and gemini-3.5-flash additionally hit a
#: hard 20-requests/day wall mid-run (confirmed exhausted, not guessed).
#: gemini-3.5-flash-lite measured 15 RPM and at least 88 successful calls
#: in one day with zero 429s -- 3x the RPM and >4x the daily volume of
#: every other option, the one model on this key that isn't either
#: retired or artificially throttled. Do not rotate models at runtime:
#: one model, used consistently, so results are comparable across agents
#: and records.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

CACHE_DIR = Path("results/agent_cache")
DAILY_COUNTER_PATH = CACHE_DIR / "_daily_request_count.json"

#: Measured directly against this key (not assumed from docs): a live
#: 429 on gemini-3.5-flash-lite reported
#: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier ... quotaValue: 15"
#: -- 12 leaves headroom under that. (gemini-3.5-flash/3.6-flash/3.8-flash
#: measured 5 RPM each; not used as DEFAULT_MODEL, see above.)
RPM_LIMIT = 12

#: Measured directly against this key, NOT assumed (the originally
#: planned ~1500/day was wrong by ~75x for gemini-3.5-flash specifically,
#: which hit a live 429 with quotaId
#: "GenerateRequestsPerDayPerProjectPerModel-FreeTier", quotaValue 20).
#: gemini-3.5-flash-lite's cap was pinned down for real on 2026-09-14: a
#: live 429 during a resumed run named quotaValue 500 explicitly
#: (previously just "comfortably above 400, not pinned down further" --
#: 400 was a placeholder, not a measurement). NOTE, not yet resolved:
#: that 429 fired after this process's own tracked count for the new
#: UTC day was only 104 (well under even the old 400 guess), following
#: a prior UTC day that had reached 397 -- 397+104=501, just over 500.
#: This strongly suggests the real quota window does NOT reset at UTC
#: midnight the way `_today()` assumes (a different reset boundary, e.g.
#: US Pacific midnight, would explain the two days' counts effectively
#: summing against one server-side window instead of resetting between
#: them). The 500 ceiling below is real; the exact reset boundary isn't
#: -- until it's pinned down, DAILY_REFUSE_HEADROOM is the only thing
#: standing between a resumed run and repeating this same live 429
#: (harmless either way, see _is_daily_quota_error: fails fast, no
#: wasted retries). Unknown/other models fall back to the harsher 20/day
#: figure actually measured for gemini-3.5-flash, rather than assuming
#: this model's better number generalizes.
#:
#: SECOND confirming data point, same day: a fresh call attempt at
#: 2026-09-14 06:12 UTC hit the identical live 429 with this process's
#: tracked count at only 104/500 -- not a one-off. 06:12 UTC is ~48 min
#: before 07:00 UTC, which is US Pacific midnight during PDT (UTC-7) --
#: a live Pacific-midnight-boundary hypothesis, not yet confirmed by a
#: successful call just after that time. If a call succeeds shortly
#: after 07:00 UTC on a day when it was refused just before, that
#: confirms the boundary; record the result here either way once known.
DAILY_QUOTA_BY_MODEL = {"gemini-3.5-flash-lite": 500, "gemini-3.5-flash": 20}
_UNKNOWN_MODEL_DAILY_QUOTA = 20
DAILY_WARN_HEADROOM = 15  # warn once within this many calls of the cap
DAILY_REFUSE_HEADROOM = 3  # refuse once within this many calls of the cap

MAX_RATE_LIMIT_RETRIES = 6
MAX_SCHEMA_RETRIES = 3

T = TypeVar("T", bound=BaseModel)


class DailyQuotaExceeded(RuntimeError):
    pass


class SchemaValidationFailed(RuntimeError):
    def __init__(self, agent: str, record_id: str, attempts: int, last_error: Exception):
        super().__init__(
            f"{agent}/{record_id}: response failed schema validation after "
            f"{attempts} attempts: {last_error}"
        )
        self.agent = agent
        self.record_id = record_id
        self.attempts = attempts
        self.last_error = last_error


class _TokenBucket:
    """Thread-safe limiter: at most `rpm` acquisitions per rolling minute,
    enforced by spacing acquisitions `60/rpm` seconds apart."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + self._interval
            wait = start - now
        if wait > 0:
            time.sleep(wait)


_bucket = _TokenBucket(RPM_LIMIT)
_counter_lock = threading.Lock()


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _load_daily_counts() -> dict:
    """{"date": "...", "counts": {model_name: n, ...}} -- per-model,
    because the API's own free-tier daily quota is per-model (a live 429
    named quotaId "GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
    not a project-wide pool. A single shared counter would let calls
    against one model mask how close another model is to its own cap."""
    if DAILY_COUNTER_PATH.exists():
        try:
            data = json.loads(DAILY_COUNTER_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}
    if data.get("date") != _today():
        data = {"date": _today(), "counts": {}}
    data.setdefault("counts", {})
    return data


def _bump_daily_count(model: str) -> int:
    with _counter_lock:
        data = _load_daily_counts()
        data["counts"][model] = data["counts"].get(model, 0) + 1
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        DAILY_COUNTER_PATH.write_text(json.dumps(data))
        return data["counts"][model]


def _daily_quota_for(model: str) -> int:
    return DAILY_QUOTA_BY_MODEL.get(model, _UNKNOWN_MODEL_DAILY_QUOTA)


def _check_daily_quota(model: str) -> None:
    count = _load_daily_counts()["counts"].get(model, 0)
    cap = _daily_quota_for(model)
    if count >= cap - DAILY_REFUSE_HEADROOM:
        raise DailyQuotaExceeded(
            f"{model}: daily request count {count} is within {DAILY_REFUSE_HEADROOM} of its "
            f"observed free-tier cap ({cap}/day); refusing rather than burning the rest of "
            "today's budget on a call the API would 429 anyway. Resume tomorrow (or another "
            "model) -- the cache means already-completed records won't be re-billed."
        )
    if count >= cap - DAILY_WARN_HEADROOM:
        print(f"[warn] {model}: daily request count at {count}/{cap}")


def daily_request_count(model: Optional[str] = None) -> int:
    counts = _load_daily_counts()["counts"]
    if model is not None:
        return counts.get(model, 0)
    return sum(counts.values())


#: default fault_condition for every call -- an ordinary, uncorrupted
#: pipeline run. Never included in the cache key explicitly (see
#: _cache_key) so every already-cached clean call from before
#: fault_condition existed keeps resolving to the exact same key --
#: fault injection (eval/fault_injection.py) is purely additive, it
#: cannot invalidate a single byte of prior work.
CLEAN_FAULT_CONDITION = "clean"


def _cache_key(
    record_id: str, agent: str, prompt_version: str, model: str, temperature: float, run_index: int,
    fault_condition: str = CLEAN_FAULT_CONDITION,
) -> str:
    raw = f"{record_id}|{agent}|{prompt_version}|{model}|{temperature}|{run_index}"
    # Only append fault_condition when it's non-default: this is the
    # entire backward-compatibility mechanism -- every call made before
    # fault injection existed was implicitly "clean", and must keep
    # hashing to the same key it always did, or every already-paid-for
    # cached response in results/agent_cache/ becomes unreachable.
    if fault_condition != CLEAN_FAULT_CONDITION:
        raw += f"|{fault_condition}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def cache_lookup(
    *, record_id: str, agent: str, prompt_version: str, model: str, temperature: float, run_index: int,
    fault_condition: str = CLEAN_FAULT_CONDITION,
) -> Optional[dict]:
    """Raw cached call record, or None. Exposed separately from
    ``call_structured`` so the pipeline can check "is this already done"
    without importing a response schema."""
    key = _cache_key(record_id, agent, prompt_version, model, temperature, run_index, fault_condition)
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


@dataclass
class CallMetadata:
    cached: bool
    schema_retries: int
    elapsed_s: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    #: one entry per rejected attempt, the validation failure's own
    #: message (schema ValidationError or an extra_validate ValueError,
    #: e.g. agents/validators.py's FLOW_LEVEL_FILTER_PREFIX-tagged ones)
    #: -- lets eval reporting break retries down by *why*, not just count
    #: them. Empty when schema_retries is 0.
    retry_reasons: List[str] = field(default_factory=list)


_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai  # imported lazily so schema-only tests don't need the SDK

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it before making live agent calls "
                "(never hardcode it)."
            )
        _client = genai.Client(api_key=api_key)
    return _client


#: matches both a per-minute quota 429 and a transient server-side 503 --
#: both observed live against this key in the same burst test that set
#: RPM_LIMIT (a 503 "high demand" storm hit before the 429s did), so
#: both are treated as retryable. Deliberately does NOT include the
#: daily-quota case -- see _is_daily_quota_error, checked first and
#: never retried, since no amount of waiting within a single run fixes it.
_TRANSIENT_ERROR_MARKERS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "503", "UNAVAILABLE", "high demand")

#: the API's 429 body distinguishes per-minute quota
#: ("GenerateRequestsPerMinute...") from per-day quota
#: ("GenerateRequestsPerDay...") via quotaId -- only the former is worth
#: retrying inside one run. Retrying a per-day 429 just burns the retry
#: budget waiting out its ~60s retryDelay for nothing, since the quota
#: resets at the next UTC day, not 60 seconds from now.
_DAILY_QUOTA_MARKER = "RequestsPerDay"

#: the API's 429 body includes a concrete `retryDelay`/"Please retry in
#: Xs" hint -- honoring it beats blind exponential backoff against a
#: hard 5 RPM ceiling, where guessing too short just re-triggers 429.
_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def _is_network_error(exc: Exception) -> bool:
    """Connection-level failures (DNS resolution, timeouts, resets) --
    distinct from API-level errors, matched by exception type rather than
    message text since the underlying OS/socket error text varies by
    platform (e.g. Windows' "[Errno 11001] getaddrinfo failed"). These
    are exactly as transient as a 503 in practice (a live run died on one
    of these mid-batch) and deserve the same backoff-and-retry, not an
    immediate crash.
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with google-genai
        return False
    return isinstance(exc, httpx.TransportError)


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc)
    if any(marker.lower() in msg.lower() for marker in _TRANSIENT_ERROR_MARKERS):
        return True
    return _is_network_error(exc)


def _is_daily_quota_error(exc: Exception) -> bool:
    return _DAILY_QUOTA_MARKER.lower() in str(exc).lower()


def _retry_delay_for(exc: Exception, attempt: int) -> float:
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        return float(match.group(1)) + random.uniform(0.1, 1.0)
    return min(60.0, float(2 ** attempt)) + random.uniform(0, 1.0)


def _raw_generate(client, model: str, prompt: str, response_schema: Type[BaseModel], temperature: float) -> tuple[str, Optional[int], Optional[int]]:
    from google.genai import types

    _check_daily_quota(model)
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        _bucket.acquire()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            _bump_daily_count(model)
            usage = getattr(resp, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", None) if usage else None
            out_tok = getattr(usage, "candidates_token_count", None) if usage else None
            return resp.text, in_tok, out_tok
        except Exception as exc:  # noqa: BLE001 - re-raised below if not retryable
            last_exc = exc
            if _is_daily_quota_error(exc):
                raise DailyQuotaExceeded(
                    f"{model}: hit the live daily quota 429 despite the pre-check "
                    f"(count was below the tracked cap -- the tracked count may be stale, "
                    f"e.g. after calls from another process). Not retrying: {exc}"
                ) from exc
            if _is_transient_error(exc) and attempt < MAX_RATE_LIMIT_RETRIES - 1:
                time.sleep(_retry_delay_for(exc, attempt))
                continue
            raise
    raise last_exc  # pragma: no cover - loop always returns or raises


def call_structured(
    *,
    record_id: str,
    agent: str,
    prompt_version: str,
    prompt: str,
    response_schema: Type[T],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    run_index: int = 0,
    use_cache: bool = True,
    extra_validate: Optional[Callable[[T], None]] = None,
    fault_condition: str = CLEAN_FAULT_CONDITION,
) -> tuple[T, CallMetadata]:
    """Call the model, enforcing the given Pydantic response schema.

    Cache key is ``(record_id, agent, prompt_version, model, temperature,
    run_index)`` plus ``fault_condition`` -- every one of those inputs
    changes the key, so a prompt-version bump, a repeated ``run_index``
    (step 3's stability runs), or a fault-injection condition
    (eval/fault_injection.py) never collides with a previous entry.
    ``fault_condition`` defaults to ``CLEAN_FAULT_CONDITION`` and is
    deliberately NOT part of the hashed string at that default (see
    ``_cache_key``), so every call made before fault injection existed
    still resolves to the exact same key it always did.

    On a schema-validation failure (either Pydantic's own parse, or the
    caller's ``extra_validate``, e.g. checking claim-id prefixes match
    the calling agent) the failure is fed back into the prompt and
    retried up to ``MAX_SCHEMA_RETRIES`` times; a call that never
    validates raises ``SchemaValidationFailed`` rather than returning a
    best-effort guess.
    """
    key = _cache_key(record_id, agent, prompt_version, model, temperature, run_index, fault_condition)
    path = _cache_path(key)

    if use_cache and path.exists():
        cached = json.loads(path.read_text())
        parsed = response_schema.model_validate(cached["response"])
        return parsed, CallMetadata(
            cached=True,
            schema_retries=cached.get("schema_retries", 0),
            elapsed_s=cached.get("elapsed_s", 0.0),
            input_tokens=cached.get("input_tokens"),
            output_tokens=cached.get("output_tokens"),
            retry_reasons=cached.get("retry_reasons", []),
        )

    client = _get_client()
    start = time.monotonic()
    working_prompt = prompt
    parsed_obj: Optional[T] = None
    last_err: Optional[Exception] = None
    schema_retries = 0
    retry_reasons: List[str] = []
    input_tokens = output_tokens = None

    for attempt in range(MAX_SCHEMA_RETRIES):
        raw_text, input_tokens, output_tokens = _raw_generate(
            client, model, working_prompt, response_schema, temperature
        )
        try:
            candidate = response_schema.model_validate_json(raw_text)
            if extra_validate is not None:
                extra_validate(candidate)
            parsed_obj = candidate
            break
        except (ValidationError, ValueError) as exc:
            last_err = exc
            schema_retries += 1
            retry_reasons.append(str(exc)[:500])
            working_prompt = (
                prompt
                + f"\n\nYour previous response failed validation: {exc}\n"
                "Return ONLY JSON matching the required schema, with no other text."
            )

    if parsed_obj is None:
        raise SchemaValidationFailed(agent, record_id, MAX_SCHEMA_RETRIES, last_err)

    elapsed = time.monotonic() - start
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "response": parsed_obj.model_dump(mode="json"),
                    "schema_retries": schema_retries,
                    "retry_reasons": retry_reasons,
                    "elapsed_s": elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model": model,
                    "agent": agent,
                    "record_id": record_id,
                    "prompt_version": prompt_version,
                    "temperature": temperature,
                    "run_index": run_index,
                    "fault_condition": fault_condition,
                }
            )
        )
    return parsed_obj, CallMetadata(
        cached=False,
        schema_retries=schema_retries,
        retry_reasons=retry_reasons,
        elapsed_s=elapsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
