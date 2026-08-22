import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from services.appwrite_service import build_account_for_jwt
from services.security_limits import get_redis_client
from services.settings import settings

logger = logging.getLogger(__name__)

_MEM_CACHE_MAX = 10_000
_mem_cache: "OrderedDict[str, tuple[float, Dict[str, Any]]]" = OrderedDict()
_mem_lock = asyncio.Lock()


def _is_valid_positive_payload(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    user_id = parsed.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        return False
    for field in ("email", "name"):
        value = parsed.get(field)
        if value is not None and not isinstance(value, str):
            return False
    return True


def _is_negative_payload(parsed: Any) -> bool:
    return isinstance(parsed, dict) and parsed.get("invalid") is True


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
    """Returns either a validated positive payload, a negative-cache marker
    {"invalid": True}, or None on miss. Callers must distinguish."""
    key = _token_cache_key(token)
    redis_client = await get_redis_client()

    if redis_client:
        try:
            value = await redis_client.get(key)
            if value:
                try:
                    parsed = json.loads(value)
                except (TypeError, ValueError):
                    parsed = None
                if _is_negative_payload(parsed):
                    return {"invalid": True}
                if _is_valid_positive_payload(parsed):
                    return parsed
        except Exception:
            logger.exception("auth cache redis get failed")

    now = time.time()
    async with _mem_lock:
        row = _mem_cache.get(key)
        if not row:
            return None

        expires_at, payload = row
        if now >= expires_at:
            _mem_cache.pop(key, None)
            return None

        _mem_cache.move_to_end(key)

        if _is_negative_payload(payload):
            return {"invalid": True}
        if _is_valid_positive_payload(payload):
            return dict(payload)

        _mem_cache.pop(key, None)
        return None


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
        except Exception:
            logger.exception("auth cache redis set failed")

    async with _mem_lock:
        _mem_cache[key] = (time.time() + ttl, dict(payload))
        _mem_cache.move_to_end(key)
        while len(_mem_cache) > _MEM_CACHE_MAX:
            _mem_cache.popitem(last=False)


# =========================
# VALIDATION
# =========================
def _user_field(user: Any, *names: str) -> Any:
    """Pull a field from either a dict (older Appwrite SDK shape) or a
    Pydantic model (Appwrite Python SDK >=17 returns User / Session model
    instances). Returns the first non-empty match.
    """
    if user is None:
        return None
    for name in names:
        # Dict / mapping path
        if isinstance(user, dict):
            value = user.get(name)
            if value not in (None, ""):
                return value
            continue
        # Pydantic / dataclass path
        if hasattr(user, name):
            value = getattr(user, name)
            if value not in (None, ""):
                return value
        # Pydantic 2 sometimes exposes only model_dump
        if hasattr(user, "model_dump"):
            try:
                dumped = user.model_dump()
                if isinstance(dumped, dict) and name in dumped:
                    value = dumped[name]
                    if value not in (None, ""):
                        return value
            except Exception:
                pass
    return None


def _validate_token_sync(token: str) -> Dict[str, Any]:
    account = build_account_for_jwt(token)
    user = account.get()

    # Appwrite Python SDK 17.x returns a User model (field `id`); older SDKs
    # returned a dict (field `$id`). Support both.
    user_id = _user_field(user, "$id", "id")
    if not user or not user_id:
        raise HTTPException(status_code=401, detail="Invalid user payload")

    return {
        "user_id": str(user_id),
        "email": _user_field(user, "email"),
        "name": _user_field(user, "name"),
    }


# =========================
# MAIN AUTH
# =========================
async def get_current_user(request: Request):
    auth_header = request.headers.get("authorization", "")

    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = _extract_bearer_token(auth_header)

    # 🔹 CACHE HIT
    cached = await _cache_get(token)
    if cached is not None:
        if _is_negative_payload(cached):
            raise HTTPException(status_code=401, detail="Invalid or expired token")
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
