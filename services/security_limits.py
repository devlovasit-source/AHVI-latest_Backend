import asyncio
import logging
import os
import time
from typing import Tuple

from services.settings import settings

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis_async
except Exception:  # pragma: no cover - optional dependency at runtime
    redis_async = None


_redis_client = None
_redis_lock = asyncio.Lock()
_redis_disabled: bool | None = None  # None=unknown, True=off (no config / unavailable)
_redis_next_retry = 0.0
_REDIS_COOLDOWN_SECONDS = 60.0
_local_lock = asyncio.Lock()
_local_windows: dict[str, tuple[int, float]] = {}
_LOCAL_WINDOW_MAX_BUCKETS = 10000


def _redis_configured() -> bool:
    """True only when a real Redis URL is provided. The bare localhost default
    means "not configured" — never dial localhost:6379 on Cloud Run."""
    for var in ("REDIS_URL", "UPSTASH_REDIS_URL", "RAILWAY_REDIS_URL"):
        if str(os.getenv(var) or "").strip():
            return True
    return False


async def get_redis_client():
    """Return a live Redis client, or None. Never dials on every request:
    unconfigured -> permanently disabled (one warning); a live failure trips a
    cooldown circuit breaker. Callers fall back to uncached auth (still fully
    enforced), so a missing/broken Redis never blocks or floods logs."""
    global _redis_client, _redis_disabled, _redis_next_retry
    if _redis_client is not None:
        return _redis_client
    if _redis_disabled:
        return None
    if redis_async is None:
        _redis_disabled = True
        return None
    if not _redis_configured():
        if _redis_disabled is None:
            logger.warning(
                "Redis not configured (no REDIS_URL); auth cache disabled, "
                "using uncached authentication."
            )
        _redis_disabled = True
        return None
    if time.monotonic() < _redis_next_retry:
        return None  # circuit breaker: still cooling down after a failure
    async with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            client = redis_async.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await client.ping()
            _redis_client = client
            return _redis_client
        except Exception:
            _redis_client = None
            _redis_next_retry = time.monotonic() + _REDIS_COOLDOWN_SECONDS
            logger.warning(
                "Redis unavailable; falling back to uncached auth for %ss.",
                int(_REDIS_COOLDOWN_SECONDS),
            )
            return None


async def is_redis_rate_limit_ready() -> bool:
    client = await get_redis_client()
    return client is not None


def extract_client_ip(headers, client_host: str | None) -> str:
    """Return the originating client IP.

    On Cloud Run we are behind exactly one trusted hop (the Google front-end),
    which appends the real client IP as the *last* entry in X-Forwarded-For.
    Trusting the leftmost entry is spoofable — clients can set XFF themselves.

    Operators can override the trusted-hop count via XFF_TRUSTED_HOPS (default 1).
    """
    forwarded = ""
    try:
        forwarded = headers.get("x-forwarded-for", "")
    except Exception:
        forwarded = ""
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            try:
                hops = max(1, int(os.getenv("XFF_TRUSTED_HOPS", "1")))
            except ValueError:
                hops = 1
            # The rightmost `hops` entries are appended by trusted proxies.
            # The client-attributable IP is at index -hops.
            idx = -hops
            if -idx > len(parts):
                idx = 0
            return parts[idx]
    return str(client_host or "unknown")


async def _check_local_window(
    key: str, max_requests: int, window_seconds: int
) -> Tuple[bool, int]:
    now = time.time()
    async with _local_lock:
        if len(_local_windows) > _LOCAL_WINDOW_MAX_BUCKETS:
            expired = [
                k for k, (_, reset_at) in _local_windows.items() if now >= reset_at
            ]
            for k in expired:
                _local_windows.pop(k, None)
            if len(_local_windows) > _LOCAL_WINDOW_MAX_BUCKETS:
                oldest = sorted(_local_windows.items(), key=lambda kv: kv[1][1])[
                    : max(0, len(_local_windows) - _LOCAL_WINDOW_MAX_BUCKETS)
                ]
                for k, _ in oldest:
                    _local_windows.pop(k, None)
        count, reset_at = _local_windows.get(key, (0, now + window_seconds))
        if now >= reset_at:
            count = 0
            reset_at = now + window_seconds
        count += 1
        _local_windows[key] = (count, reset_at)
        allowed = count <= max_requests
        remaining = max(0, max_requests - count)
        return allowed, remaining


async def check_rate_limit(
    *,
    bucket_key: str,
    max_requests: int | None = None,
    window_seconds: int | None = None,
) -> Tuple[bool, int]:
    if not settings.rate_limit_enabled:
        return True, 999999

    max_requests = int(max_requests or settings.rate_limit_max_requests)
    window_seconds = int(window_seconds or settings.rate_limit_window_seconds)

    redis_client = await get_redis_client()
    if redis_client is None:
        return await _check_local_window(bucket_key, max_requests, window_seconds)

    try:
        key = f"rl:{bucket_key}"
        current = await redis_client.incr(key)
        if current == 1:
            await redis_client.expire(key, window_seconds)
        allowed = int(current) <= max_requests
        remaining = max(0, max_requests - int(current))
        return allowed, remaining
    except Exception:
        return await _check_local_window(bucket_key, max_requests, window_seconds)
