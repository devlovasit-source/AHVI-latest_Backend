import base64
import asyncio
from typing import Any, Optional

import numpy as np
import httpx
from PIL import Image
import io


# =========================
# CACHE (LRU STYLE)
# =========================
_URL_HASH_CACHE: dict[str, str] = {}
_URL_HASH_CACHE_MAX = 2048


def _cache_get(url: str) -> str:
    return _URL_HASH_CACHE.get(url, "")


def _cache_set(url: str, value: str):
    if not url or not value:
        return

    _URL_HASH_CACHE[url] = value

    if len(_URL_HASH_CACHE) > _URL_HASH_CACHE_MAX:
        _URL_HASH_CACHE.pop(next(iter(_URL_HASH_CACHE)), None)


# =========================
# IMAGE DECODE (PIL)
# =========================
def _decode_bytes(image_bytes: bytes):
    if not image_bytes:
        return None

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception:
        return None


def _foreground_crop(img: Image.Image):
    if img is None:
        return None

    img_np = np.array(img)

    # Handle alpha channel
    if img_np.shape[-1] == 4:
        alpha = img_np[:, :, 3]
        ys, xs = np.where(alpha > 10)

        if len(xs) and len(ys):
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            img_np = img_np[y1:y2, x1:x2]

        img_np = img_np[:, :, :3]

    return Image.fromarray(img_np)


# =========================
# CORE HASH FUNCTION (DHash)
# =========================
def compute_hash_from_bytes(image_bytes: bytes, size: int = 8) -> str:
    try:
        img = _decode_bytes(image_bytes)
        img = _foreground_crop(img)

        if img is None:
            return ""

        gray = img.convert("L")
        resized = gray.resize((size + 1, size), Image.Resampling.LANCZOS)

        pixels = np.array(resized)

        diff = pixels[:, 1:] > pixels[:, :-1]
        bits = diff.flatten()

        if bits.size == 0:
            return ""

        value = 0
        for b in bits:
            value = (value << 1) | int(b)

        width = (size * size + 3) // 4
        return f"{value:0{width}x}"

    except Exception:
        return ""


# =========================
# BASE64
# =========================
def compute_hash_from_base64(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    if "," in text:
        text = text.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(text, validate=True)
    except Exception:
        return ""

    return compute_hash_from_bytes(image_bytes)


# =========================
# URL (ASYNC)
# =========================
async def compute_hash_from_url(url: str, timeout: float = 6.0) -> str:
    if not url:
        return ""

    cached = _cache_get(url)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            h = compute_hash_from_bytes(resp.content)

            if h:
                _cache_set(url, h)

            return h

    except Exception:
        return ""


# =========================
# HAMMING DISTANCE
# =========================
def hamming_distance_hex(h1: Any, h2: Any) -> Optional[int]:
    a = str(h1 or "").strip().lower()
    b = str(h2 or "").strip().lower()

    if not a or not b or len(a) != len(b):
        return None

    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return None


# =========================
# BACKWARD COMPATIBILITY
# =========================
def compute_pixel_hash_from_bytes(image_bytes: bytes, size: int = 8) -> str:
    return compute_hash_from_bytes(image_bytes, size)


def compute_pixel_hash_from_base64(value: Any) -> str:
    return compute_hash_from_base64(value)


async def compute_pixel_hash_from_url(url: str, timeout: float = 6.0) -> str:
    return await compute_hash_from_url(url, timeout)
