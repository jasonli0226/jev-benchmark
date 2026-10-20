"""Minimal client for the OpenRouter Decisions API (Jev).

Requests are cached on disk keyed by (request body, repeat index), so
re-running a finished eval costs nothing, while repeat > 0 forces fresh
calls for variance measurement. Jev is not deterministic: the docs note
probabilities differ slightly from run to run.
"""

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

from jev_bench.config import CACHE_DIR

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
REQUEST_TIMEOUT_S = 60.0
MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0
DEFAULT_CONCURRENCY = 8


class JevError(RuntimeError):
    """Non-retryable API failure (bad request, auth, out of credits)."""


class _Retryable(Exception):
    def __init__(self, message: str, retry_after: float | None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class JevResponse:
    body: dict
    latency_s: float  # wall time of the successful HTTP call
    attempts: int
    cached: bool


def require_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise JevError("OPENROUTER_API_KEY is not set. Put it in .env (see .env.example).")
    return key


def _is_inflight_budget(body: dict) -> bool:
    meta = (body.get("error") or {}).get("metadata") or {}
    return meta.get("limit_source") == "openrouter_in_flight_budget"


class JevClient:
    def __init__(self, concurrency: int = DEFAULT_CONCURRENCY, use_cache: bool = True):
        self._headers = {"Authorization": f"Bearer {require_api_key()}", "Content-Type": "application/json"}
        self._http = httpx.Client(timeout=REQUEST_TIMEOUT_S)
        self.concurrency = concurrency
        self.use_cache = use_cache
        self.cache_dir = CACHE_DIR / "jev"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Shared pause so one 429 slows every worker, not just the one that hit it.
        self._pause_until = 0.0
        self._lock = threading.Lock()

    def _cache_path(self, body: dict, repeat: int):
        key = hashlib.sha256(json.dumps([body, repeat], sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _post_once(self, body: dict) -> tuple[dict, float]:
        with self._lock:
            wait = self._pause_until - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        start = time.perf_counter()
        try:
            res = self._http.post(DECISIONS_URL, headers=self._headers, json=body)
        except httpx.TransportError as exc:
            raise _Retryable(f"transport error: {exc}", None) from exc
        elapsed = time.perf_counter() - start
        try:
            payload = res.json()
        except ValueError:
            payload = {"raw": res.text[:500]}
        if res.status_code == 200:
            return payload, elapsed
        retry_after = res.headers.get("Retry-After")
        retry_after = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
        message = f"HTTP {res.status_code}: {json.dumps(payload)[:300]}"
        if res.status_code == 429 or res.status_code >= 500 or (res.status_code == 402 and _is_inflight_budget(payload)):
            raise _Retryable(message, retry_after)
        raise JevError(message)

    def decide(self, body: dict, repeat: int = 0) -> JevResponse:
        path = self._cache_path(body, repeat)
        if self.use_cache and path.exists():
            hit = json.loads(path.read_text())
            return JevResponse(hit["body"], hit["latency_s"], hit["attempts"], cached=True)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                payload, elapsed = self._post_once(body)
                break
            except _Retryable as exc:
                if attempt == MAX_ATTEMPTS:
                    raise JevError(f"gave up after {attempt} attempts: {exc}") from exc
                delay = exc.retry_after or min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (attempt - 1))
                delay *= 1 + random.random() * 0.25
                with self._lock:
                    self._pause_until = max(self._pause_until, time.monotonic() + delay)
        path.write_text(json.dumps({"body": payload, "latency_s": elapsed, "attempts": attempt}))
        return JevResponse(payload, elapsed, attempt, cached=False)

    def decide_many(self, bodies: list[dict], repeat: int = 0) -> list[JevResponse]:
        """Run requests concurrently; results keep input order. A JevError on any
        request stops the batch (e.g. credits ran out) rather than being hidden."""
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda b: self.decide(b, repeat), bodies))
