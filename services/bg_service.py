import os
import asyncio
import hashlib
from typing import Optional

import httpx
from redis.asyncio import Redis


# =========================
# CONFIG
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
HF_BG_URL = "https://api-inference.huggingface.co/models/briaai/RMBG-2.0"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 60 * 60  # 1 hour
LOCK_TTL = 30  # seconds


# =========================
# REDIS CLIENT (GLOBAL)
# =========================
redis_client = Redis.from_url(
    REDIS_URL,
    decode_responses=False  # IMPORTANT: we store raw bytes
)


# =========================
# HELPERS
# =========================
def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _get_cached(cache_key: str) -> Optional[bytes]:
    try:
        return await redis_client.get(cache_key)
    except Exception as e:
        print("[REDIS GET ERROR]", e)
        return None


async def _set_cached(cache_key: str, value: bytes):
    try:
        await redis_client.set(cache_key, value, ex=CACHE_TTL)
    except Exception as e:
        print("[REDIS SET ERROR]", e)


async def _acquire_lock(lock_key: str) -> bool:
    try:
        return await redis_client.set(lock_key, b"1", ex=LOCK_TTL, nx=True)
    except Exception as e:
        print("[REDIS LOCK ERROR]", e)
        return True  # fail-open (don’t block processing)


async def _release_lock(lock_key: str):
    try:
        await redis_client.delete(lock_key)
    except Exception:
        pass


# =========================
# MAIN FUNCTION
# =========================
async def remove_bg_bytes(image_bytes: bytes) -> bytes:
    if not image_bytes:
        return image_bytes

    cache_key = f"bg:{_hash_bytes(image_bytes)}"
    lock_key = f"{cache_key}:lock"

    # =========================
    # ✅ CACHE HIT
    # =========================
    cached = await _get_cached(cache_key)
    if cached:
        print("[BG CACHE HIT]")
        return cached

    # =========================
    # 🔒 LOCK (DEDUP)
    # =========================
    has_lock = await _acquire_lock(lock_key)

    if not has_lock:
        # Someone else is processing → wait briefly and retry cache
        for _ in range(5):
            await asyncio.sleep(0.4)
            cached = await _get_cached(cache_key)
            if cached:
                print("[BG CACHE HIT AFTER WAIT]")
                return cached

        # fallback → proceed anyway (avoid deadlock)

    # =========================
    # ❌ NO TOKEN
    # =========================
    if not HF_TOKEN:
        print("[BG] HF token missing ❌")
        return image_bytes

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/octet-stream"
    }

    try:
        # =========================
        # 🔁 RETRY LOOP
        # =========================
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    res = await client.post(
                        HF_BG_URL,
                        headers=headers,
                        content=image_bytes
                    )

                print(f"[BG STATUS] {res.status_code} (attempt {attempt+1})")

                if res.status_code == 200:
                    result = res.content

                    # ✅ CACHE STORE
                    await _set_cached(cache_key, result)

                    return result

                # HF cold start / overload
                if res.status_code in (503, 504):
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue

                print("[BG ERROR]", res.text)
                return image_bytes

            except Exception as e:
                print(f"[BG EXCEPTION] attempt {attempt+1}:", e)
                await asyncio.sleep(1)

        return image_bytes

    finally:
        # 🔓 always release lock
        if has_lock:
            await _release_lock(lock_key)
# =========================
# BACKWARD COMPATIBILITY
# =========================

async def remove_bg_external(image_bytes: bytes) -> bytes:
    return await remove_bg_bytes(image_bytes)


def remove_bg_external_sync(image_bytes: bytes) -> bytes:
    import asyncio
    return asyncio.run(remove_bg_bytes(image_bytes))
