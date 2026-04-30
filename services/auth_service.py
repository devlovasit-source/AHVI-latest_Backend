import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from services.appwrite_service import build_account_for_jwt
from services.security_limits import get_redis_client
from services.settings import settings


# =========================
# IN-MEMORY CACHE (FALLBACK)
# =========================
_mem_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_mem_lock = asyncio.Lock()


# =========================
# HELPERS
# =========================
def _extract_bearer_token(auth_header: str) -> str:
    parts = str(auth_header or "").split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    return parts[1].strip()


def _token_cache_key(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"auth:token:{digest}"


# =========================
# CACHE GET
# =========================
async def _cache_get(token: str) -> Optional[Dict[str, Any]]:
    key = _token_cache_key(token)
    redis_client = await get_redis_client()

    # 🔹 Redis first
    if redis_client:
        try:
            value = await redis_client.get(key)
            if value:
                parsed = json.loads(value)
                if isinstance(parsed, dict) and "user_id" in parsed:
                    return parsed
        except Exception as e:
            print("[AUTH CACHE REDIS GET ERROR]", e)

    # 🔹 In-memory fallback
    now = time.time()
    async with _mem_lock:
        row = _mem_cache.get(key)
        if not row:
            return None

        expires_at, payload = row
        if now >= expires_at:
            _mem_cache.pop(key, None)
            return None

        return dict(payload)


# =========================
# CACHE SET
# =========================
async def _cache_set(token: str, payload: Dict[str, Any], negative: bool = False) -> None:
    key = _token_cache_key(token)
    ttl = int(settings.auth_cache_ttl_seconds)

    # 🔥 shorter TTL for invalid tokens
    if negative:
        ttl = min(30, ttl)

    redis_client = await get_redis_client()

    if redis_client:
        try:
            await redis_client.setex(key, ttl, json.dumps(payload))
            return
        except Exception as e:
            print("[AUTH CACHE REDIS SET ERROR]", e)

    async with _mem_lock:
        _mem_cache[key] = (time.time() + ttl, dict(payload))


# =========================
# VALIDATION (SYNC → THREAD)
# =========================
def _validate_token_sync(token: str) -> Dict[str, Any]:
    account = build_account_for_jwt(token)
    user = account.get()

    if not user or "$id" not in user:
        raise HTTPException(status_code=401, detail="Invalid user payload")

    return {
        "user_id": user["$id"],
        "email": user.get("email"),
        "name": user.get("name"),
    }


# =========================
# MAIN AUTH DEPENDENCY
# =========================
async def get_current_user(request: Request):
    auth_header = request.headers.get("authorization", "")

    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = _extract_bearer_token(auth_header)

    # 🔹 CACHE HIT
    cached = await _cache_get(token)
    if cached:
        request.state.user = cached
        return cached

    try:
        payload = await asyncio.to_thread(_validate_token_sync, token)

        # 🔹 cache success
        await _cache_set(token, payload)

        request.state.user = payload
        return payload

    except HTTPException:
        # 🔥 negative caching (invalid token)
        await _cache_set(token, {"invalid": True}, negative=True)
        raise

    except Exception as e:
        error_str = str(e).lower()

        if any(x in error_str for x in ["timeout", "connection", "name or service not known"]):
            raise HTTPException(status_code=503, detail="Auth service temporarily unavailable")

        # 🔥 negative caching
        await _cache_set(token, {"invalid": True}, negative=True)

        raise HTTPException(status_code=401, detail="Invalid or expired token")
