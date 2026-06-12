"""Gemini Vision multi-garment bbox detection + crop for AHVI capture preview.

MVP addition on top of the existing capture pipeline:

- Detect up to ``WARDROBE_CAPTURE_MAX_ITEMS`` (default 6) distinct fashion /
  wearable items in one uploaded image using Gemini Vision.
- Validate + clamp each item's normalized bbox ``[xmin, ymin, xmax, ymax]``.
- Convert the normalized bbox to pixel coordinates (with a small padding
  margin) and crop the original image with PIL.

This module does NOT call GroundingDINO, does NOT touch
``services/hybrid_detection_service.py``, and does NOT perform background
removal, R2 upload, or taxonomy normalization itself — that orchestration
stays in ``routers/wardrobe_capture.py`` so the existing preview pipeline
(RMBG, R2 upload, taxonomy/suitability guards) is reused unchanged.

Gated entirely by ``ENABLE_GEMINI_MULTI_GARMENT_PREVIEW`` (default off). Any
failure (disabled, missing client, timeout, invalid JSON, <2 valid items)
results in an empty/short list so the caller can fall back to the existing
single-garment flow unchanged.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:  # google-genai may be absent in local/dev until requirements are installed
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - import guard
    genai = None
    types = None

from services.ai_gateway import parse_json_array

logger = logging.getLogger("ahvi.capture.gemini_multi")


def _env_enabled(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


ENABLE_FLAG = "ENABLE_GEMINI_MULTI_GARMENT_PREVIEW"

GEMINI_MULTI_GARMENT_MODEL = os.getenv(
    "GEMINI_MULTI_GARMENT_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
)
GOOGLE_CLOUD_PROJECT = (
    os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or "ahvi-485510"
)
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
GEMINI_MULTI_GARMENT_TIMEOUT_SECONDS = _env_float(
    "GEMINI_MULTI_GARMENT_TIMEOUT_SECONDS", 20.0
)

MAX_ITEMS = max(1, _env_int("WARDROBE_CAPTURE_MAX_ITEMS", 6))
MIN_VALID_ITEMS = 2
BBOX_PAD_RATIO = _env_float("GEMINI_MULTI_GARMENT_BBOX_PAD_RATIO", 0.04)


# Supported fashion/wearable item types — used only to steer the prompt.
# Final categorization is still done by the existing wardrobe taxonomy guard.
SUPPORTED_ITEMS = [
    "dress", "saree", "lehenga", "top", "shirt", "blouse", "kurta", "jeans",
    "trousers", "skirt", "shorts", "blazer", "jacket", "footwear", "sandals",
    "sneakers", "handbag", "sunglasses", "watch", "jewelry",
]

EXCLUDED_ITEMS = [
    "background", "text", "phone UI", "furniture", "hanger",
    "person/body/face", "room", "mirror frame",
]


def is_enabled() -> bool:
    return _env_enabled(ENABLE_FLAG, "false")


def _build_prompt() -> str:
    supported = ", ".join(SUPPORTED_ITEMS)
    excluded = ", ".join(EXCLUDED_ITEMS)
    return f"""
You are a fashion item detector for a wardrobe app.

Detect up to {MAX_ITEMS} distinct fashion / wearable items visible in this
image.

Supported item types: {supported}.

Exclude: {excluded}.

Rules:
- For mirror selfies or single-person photos, detect ONLY visible
  clothing/accessories worn by the main person.
- For group photos, detect ONLY the central/largest person's outfit.
- If the image is ambiguous (unclear single subject, heavy occlusion, or you
  cannot confidently separate items), set "needs_review": true on the
  affected item(s).

Return JSON ONLY - a JSON array, no markdown, no commentary - in this exact
shape:
[
  {{
    "name": "Red Eyelet Dress",
    "category": "Dresses",
    "sub_category": "Mini Dress",
    "color": "Red",
    "confidence": 0.9,
    "bbox": [0.10, 0.05, 0.55, 0.95],
    "needs_review": false
  }}
]

bbox is [xmin, ymin, xmax, ymax], normalized 0.0-1.0 relative to the full
image, top-left origin. confidence is 0.0-1.0.

If you cannot confidently detect any supported items, return [].
""".strip()


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if genai is None or types is None:
        return None
    if _gemini_client is not None:
        return _gemini_client
    try:
        _gemini_client = genai.Client(
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
            http_options=types.HttpOptions(api_version="v1"),
        )
        return _gemini_client
    except Exception as exc:
        logger.warning("ahvi.capture.gemini_multi.client_init_failed err=%s", exc)
        return None


def _call_gemini_vision(image_bytes: bytes, *, request_id: str = "") -> Optional[str]:
    """Synchronous Gemini Vision call. Run via asyncio.to_thread by callers."""
    client = _get_gemini_client()
    if client is None:
        return None
    try:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1536,
            response_mime_type="application/json",
        )
        response = client.models.generate_content(
            model=GEMINI_MULTI_GARMENT_MODEL,
            contents=[image_part, _build_prompt()],
            config=config,
        )
        return (response.text or "").strip()
    except Exception as exc:
        logger.warning(
            "ahvi.capture.gemini_multi.call_failed request_id=%s err_type=%s err=%s",
            request_id,
            type(exc).__name__,
            str(exc)[:300],
        )
        return None


def _validate_bbox(raw: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in raw)
    except Exception:
        return None
    # Allow a tiny bit of slack for models that slightly overshoot 0/1.
    for v in (xmin, ymin, xmax, ymax):
        if v < -0.05 or v > 1.05:
            return None
    xmin = min(max(xmin, 0.0), 1.0)
    ymin = min(max(ymin, 0.0), 1.0)
    xmax = min(max(xmax, 0.0), 1.0)
    ymax = min(max(ymax, 0.0), 1.0)
    if xmax - xmin < 0.02 or ymax - ymin < 0.02:
        return None
    if xmin >= xmax or ymin >= ymax:
        return None
    return (xmin, ymin, xmax, ymax)


def _validate_item(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    bbox = _validate_bbox(raw.get("bbox"))
    if bbox is None:
        return None
    try:
        confidence = float(raw.get("confidence"))
    except Exception:
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)
    return {
        "name": name,
        "category": str(raw.get("category") or "").strip(),
        "sub_category": str(raw.get("sub_category") or "").strip(),
        "color": str(raw.get("color") or "").strip(),
        "confidence": confidence,
        "bbox": list(bbox),
        "needs_review": bool(raw.get("needs_review") or False),
    }


def _bbox_to_pixels(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    pad_ratio: float = BBOX_PAD_RATIO,
) -> List[int]:
    xmin, ymin, xmax, ymax = bbox
    x1 = xmin * width
    y1 = ymin * height
    x2 = xmax * width
    y2 = ymax * height
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    px1 = max(0, int(round(x1 - pad_x)))
    py1 = max(0, int(round(y1 - pad_y)))
    px2 = min(width, int(round(x2 + pad_x)))
    py2 = min(height, int(round(y2 + pad_y)))
    if px2 <= px1:
        px2 = min(width, px1 + 1)
    if py2 <= py1:
        py2 = min(height, py1 + 1)
    return [px1, py1, px2, py2]


def _crop_to_png_bytes(image: Image.Image, box: List[int]) -> bytes:
    crop = image.crop(tuple(box))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


async def detect_and_crop(
    image: Image.Image,
    image_bytes: bytes = b"",
    *,
    request_id: str = "",
) -> List[Dict[str, Any]]:
    """Detect + crop up to MAX_ITEMS fashion items from ``image``.

    Returns a list of dicts with keys: name, category, sub_category, color,
    confidence, bbox (normalized), bbox_px (pixel ints), crop_bytes (PNG),
    needs_review. Returns [] on any failure (disabled, no client, timeout,
    invalid JSON, no valid items). Never raises.
    """
    if not is_enabled():
        return []

    logger.info("ahvi.capture.gemini_multi.start request_id=%s", request_id)

    payload_bytes = image_bytes
    if not payload_bytes:
        try:
            buf = io.BytesIO()
            rgb = image.convert("RGB") if image.mode != "RGB" else image
            rgb.save(buf, format="PNG")
            payload_bytes = buf.getvalue()
        except Exception as exc:
            logger.info(
                "ahvi.capture.gemini_multi.fallback reason=encode_failed:%s request_id=%s",
                exc,
                request_id,
            )
            return []

    try:
        raw_text = await asyncio.wait_for(
            asyncio.to_thread(_call_gemini_vision, payload_bytes, request_id=request_id),
            timeout=GEMINI_MULTI_GARMENT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.info(
            "ahvi.capture.gemini_multi.fallback reason=timeout request_id=%s", request_id
        )
        return []
    except Exception as exc:
        logger.info(
            "ahvi.capture.gemini_multi.fallback reason=exception:%s request_id=%s",
            exc,
            request_id,
        )
        return []

    if not raw_text:
        logger.info(
            "ahvi.capture.gemini_multi.fallback reason=empty_response request_id=%s",
            request_id,
        )
        return []

    logger.info(
        "ahvi.capture.gemini_multi.raw_result request_id=%s raw=%s",
        request_id,
        raw_text[:2000],
    )

    try:
        parsed = parse_json_array(raw_text)
    except Exception as exc:
        logger.info(
            "ahvi.capture.gemini_multi.fallback reason=invalid_json:%s request_id=%s",
            exc,
            request_id,
        )
        return []

    width, height = image.size
    results: List[Dict[str, Any]] = []
    for raw_item in parsed[:MAX_ITEMS]:
        valid = _validate_item(raw_item)
        if valid is None:
            continue
        bbox_px = _bbox_to_pixels(tuple(valid["bbox"]), width, height)
        try:
            crop_bytes = _crop_to_png_bytes(image, bbox_px)
        except Exception as exc:
            logger.info(
                "ahvi.capture.gemini_multi.fallback reason=crop_failed:%s request_id=%s",
                exc,
                request_id,
            )
            continue
        valid["bbox_px"] = bbox_px
        valid["crop_bytes"] = crop_bytes
        results.append(valid)
        logger.info(
            "ahvi.capture.gemini_multi.crop request_id=%s item=%s bbox_px=%s",
            request_id,
            valid["name"],
            bbox_px,
        )

    return results


__all__ = [
    "ENABLE_FLAG",
    "MIN_VALID_ITEMS",
    "SUPPORTED_ITEMS",
    "EXCLUDED_ITEMS",
    "is_enabled",
    "detect_and_crop",
]
