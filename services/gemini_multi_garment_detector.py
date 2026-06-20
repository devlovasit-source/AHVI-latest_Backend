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
import hashlib
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:  # google-genai may be absent in local/dev until requirements are installed
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - import guard
    genai = None
    types = None

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
_GEMINI_INT32_MAX = 2_147_483_647


# Supported fashion/wearable item types — used only to steer the prompt.
# Final categorization is still done by the existing wardrobe taxonomy guard.
SUPPORTED_ITEMS = [
    "dress", "saree", "lehenga", "top", "shirt", "blouse", "kurta", "jeans",
    "trousers", "skirt", "shorts", "blazer", "jacket", "footwear", "sandals",
    "sneakers", "handbag", "sunglasses", "watch", "jewelry", "cap", "hat",
    "headwear", "baseball cap", "snapback", "dad cap", "visor", "beanie",
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

Return exactly one JSON object, no markdown, no commentary, no prose, in this
exact shape:
{{
  "items": [
    {{
      "name": "Red Eyelet Dress",
      "category": "Dresses",
      "sub_category": "Mini Dress",
      "color": "Red",
      "confidence": 0.9,
      "bbox": [0.10, 0.05, 0.55, 0.95],
      "needs_review": false,
      "reason": null
    }}
  ]
}}

bbox is [xmin, ymin, xmax, ymax], normalized 0.0-1.0 relative to the full
image, top-left origin. confidence is 0.0-1.0.

If you cannot confidently detect any supported items, return {{"items": []}}.
""".strip()


_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "sub_category": {"type": "string"},
                    "color": {"type": "string"},
                    "confidence": {"type": "number"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "needs_review": {"type": "boolean"},
                    "reason": {"type": ["string", "null"]},
                },
                "required": ["name", "category", "sub_category", "confidence", "bbox"],
            },
        }
    },
    "required": ["items"],
}


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


def _stable_int32_seed(image_bytes: bytes) -> int:
    raw_seed = int(hashlib.sha256(image_bytes or b"").hexdigest()[:8], 16)
    seed = raw_seed % _GEMINI_INT32_MAX
    if seed <= 0:
        seed = 1
    return seed


def _is_seed_invalid_argument(exc: Exception) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return (
        "invalid_argument" in blob
        and "seed" in blob
        and (
            "generation_config.seed" in blob
            or "generation config.seed" in blob
            or "type_int32" in blob
        )
    )


def _call_gemini_vision(image_bytes: bytes, *, request_id: str = "") -> Optional[str]:
    """Synchronous Gemini Vision call. Run via asyncio.to_thread by callers."""
    client = _get_gemini_client()
    if client is None:
        return None
    try:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        config_kwargs = {
            "temperature": 0,
            "max_output_tokens": 1536,
            "response_mime_type": "application/json",
            "response_schema": _ITEM_SCHEMA,
        }
        # Keep Gemini 2.5 Flash, but make the request as deterministic as the
        # SDK/model supports. Unsupported fields are intentionally omitted.
        try:
            config_kwargs["candidate_count"] = 1
            config_kwargs["seed"] = _stable_int32_seed(image_bytes)
            config = types.GenerateContentConfig(**config_kwargs)
        except Exception:
            config_kwargs.pop("seed", None)
            try:
                config = types.GenerateContentConfig(**config_kwargs)
            except Exception:
                config_kwargs.pop("candidate_count", None)
                config_kwargs.pop("response_schema", None)
                config = types.GenerateContentConfig(**config_kwargs)
        try:
            response = client.models.generate_content(
                model=GEMINI_MULTI_GARMENT_MODEL,
                contents=[image_part, _build_prompt()],
                config=config,
            )
        except Exception as exc:
            if "seed" not in config_kwargs or not _is_seed_invalid_argument(exc):
                raise
            logger.warning(
                "ahvi.capture.gemini_multi.seed_retry request_id=%s err=%s",
                request_id,
                str(exc)[:300],
            )
            config_kwargs.pop("seed", None)
            retry_config = types.GenerateContentConfig(**config_kwargs)
            response = client.models.generate_content(
                model=GEMINI_MULTI_GARMENT_MODEL,
                contents=[image_part, _build_prompt()],
                config=retry_config,
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


def _strip_markdown_fence(raw_text: str) -> Optional[str]:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _extract_balanced_json(raw_text: str) -> Optional[str]:
    start_positions = [
        pos for pos in (raw_text.find("{"), raw_text.find("["))
        if pos >= 0
    ]
    if not start_positions:
        return None
    start = min(start_positions)
    opener = raw_text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw_text)):
        char = raw_text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return raw_text[start:idx + 1].strip()
    return None


def _items_from_payload(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    items = payload.get("items")
    if isinstance(items, list):
        return items
    if "name" in payload and "bbox" in payload:
        return [payload]
    return []


def _parse_gemini_items(raw_text: str) -> Tuple[List[Any], str, str]:
    """Parse Gemini JSON-ish output into an item list.

    Returns (items, parse_mode, error). parse_mode is one of object, array,
    fenced, repaired, failed. error is empty on success.
    """
    text = str(raw_text or "").strip()
    if not text or text in {"[", "{", "```", "```json"}:
        return [], "failed", "invalid_json:truncated_or_empty"

    fenced = _strip_markdown_fence(text)
    candidates: List[Tuple[str, str]] = []
    if fenced:
        candidates.append(("fenced", fenced))
    candidates.append(("direct", text))
    extracted = _extract_balanced_json(text)
    if extracted and extracted not in {text, fenced}:
        candidates.append(("repaired", extracted))

    last_error = ""
    for mode_hint, candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception as exc:
            last_error = str(exc)
            continue
        if isinstance(payload, dict):
            mode = "fenced" if mode_hint == "fenced" else "object"
        elif isinstance(payload, list):
            mode = "fenced" if mode_hint == "fenced" else "array"
        else:
            return [], "failed", "invalid_json:unsupported_json_shape"
        if mode_hint == "repaired":
            mode = "repaired"
        return _items_from_payload(payload), mode, ""

    if text[0] in "[{" and text[-1] not in "]}":
        return [], "failed", "invalid_json:truncated_or_empty"
    summary = last_error[:120] if last_error else "no valid JSON found in model response"
    return [], "failed", f"invalid_json:{summary}"


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
        "reason": str(raw.get("reason") or raw.get("review_reason") or "").strip(),
    }


def detection_config_summary(image_bytes: bytes | None = None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "model": GEMINI_MULTI_GARMENT_MODEL,
        "temperature": 0,
        "candidate_count": 1,
        "seed_source": "image_sha256",
        "response_mime_type": "application/json",
        "response_schema": "items[]",
    }
    if image_bytes:
        summary["seed_hash"] = hashlib.sha256(image_bytes).hexdigest()[:12]
        summary["seed"] = _stable_int32_seed(image_bytes)
    return summary


# Category-aware crop padding. Gemini bboxes hug the garment tightly, which
# makes maxi dresses lose hems/straps and accessories look cramped. Pads are
# fractions of the bbox size: (pad_x, pad_top, pad_bottom). Top padding is
# slightly larger than bottom for tall garments so necklines/straps survive.
_PAD_TALL_GARMENT = (0.12, 0.16, 0.13)   # dress/saree/lehenga/gown/coat/jacket
_PAD_TOP_GARMENT = (0.10, 0.12, 0.10)    # top/shirt/kurta/blouse
_PAD_ACCESSORY = (0.13, 0.13, 0.13)      # bag/footwear/sunglasses/jewelry/watch

_TALL_GARMENT_TERMS = (
    "dress", "saree", "sari", "lehenga", "gown", "coat", "jacket",
    "jumpsuit", "maxi", "anarkali", "abaya", "kaftan",
)
_TOP_GARMENT_TERMS = (
    "top", "shirt", "blouse", "kurta", "t-shirt", "tee", "sweater",
    "hoodie", "cardigan", "tunic",
)
_ACCESSORY_TERMS = (
    "bag", "handbag", "tote", "purse", "clutch", "backpack",
    "shoe", "sneaker", "sandal", "slipper", "heel", "boot", "footwear",
    "sunglasses", "glasses", "eyewear", "jewelry", "jewellery",
    "necklace", "earring", "bracelet", "watch", "belt", "hat", "cap",
    "scarf", "accessory", "accessories",
)


def _pad_ratios_for_item(
    name: str = "", category: str = "", sub_category: str = ""
) -> Tuple[float, float, float]:
    """Return (pad_x, pad_top, pad_bottom) for the detected item."""
    blob = " ".join(
        str(v or "").lower() for v in (name, sub_category, category)
    )

    def _has(terms: Tuple[str, ...]) -> bool:
        return any(t in blob for t in terms)

    if _has(_TALL_GARMENT_TERMS):
        return _PAD_TALL_GARMENT
    if _has(_TOP_GARMENT_TERMS):
        return _PAD_TOP_GARMENT
    if _has(_ACCESSORY_TERMS):
        return _PAD_ACCESSORY
    return (BBOX_PAD_RATIO, BBOX_PAD_RATIO, BBOX_PAD_RATIO)


def _bbox_to_pixels(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    pad_ratio: float = BBOX_PAD_RATIO,
    *,
    pads: Optional[Tuple[float, float, float]] = None,
) -> List[int]:
    xmin, ymin, xmax, ymax = bbox
    x1 = xmin * width
    y1 = ymin * height
    x2 = xmax * width
    y2 = ymax * height
    if pads is None:
        pads = (pad_ratio, pad_ratio, pad_ratio)
    pad_x_ratio, pad_top_ratio, pad_bottom_ratio = pads

    box_w = x2 - x1
    box_h = y2 - y1
    # Tall bbox (e.g. full-length dress shot at an angle): bump top padding a
    # touch more so straps/neckline are never clipped.
    if box_w > 0 and box_h > 1.4 * box_w:
        pad_top_ratio += 0.02

    pad_x = box_w * pad_x_ratio
    pad_top = box_h * pad_top_ratio
    pad_bottom = box_h * pad_bottom_ratio
    px1 = max(0, int(round(x1 - pad_x)))
    py1 = max(0, int(round(y1 - pad_top)))
    px2 = min(width, int(round(x2 + pad_x)))
    py2 = min(height, int(round(y2 + pad_bottom)))
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

    parsed, parse_mode, parse_error = _parse_gemini_items(raw_text)
    logger.info(
        "ahvi.capture.gemini_multi.raw_result request_id=%s raw_response_length=%d raw_response_first_500=%s json_parse_mode=%s parsed_item_count=%d parse_error_summary=%s",
        request_id,
        len(raw_text),
        raw_text[:500],
        parse_mode,
        len(parsed),
        parse_error,
    )

    if parse_error:
        logger.info(
            "ahvi.capture.gemini_multi.fallback reason=%s request_id=%s",
            parse_error,
            request_id,
        )
        return []

    width, height = image.size
    results: List[Dict[str, Any]] = []
    for raw_item in parsed[:MAX_ITEMS]:
        valid = _validate_item(raw_item)
        if valid is None:
            continue
        bbox_px = _bbox_to_pixels(
            tuple(valid["bbox"]),
            width,
            height,
            pads=_pad_ratios_for_item(
                name=valid.get("name") or "",
                category=valid.get("category") or "",
                sub_category=valid.get("sub_category") or "",
            ),
        )
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
