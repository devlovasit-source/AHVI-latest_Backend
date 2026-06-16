"""Deterministic catalog image pipeline (RMBG-first, no AI regeneration).

Turns an uploaded / model / selfie garment cutout into a clean, straightened,
centered catalog-style product image on a neutral background.

Scope of THIS module (Phase 1-7 + P0 straighten):
- deterministic only: EXIF straighten -> sideways rotate heuristic -> alpha/
  whitespace trim -> center on a 1600x1600 off-white canvas -> validate -> encode.
- NO Imagen / Flux Kontext yet. ``generate_catalog_image_ai`` is a designed but
  unimplemented hook (Phase 8).

Everything here is pure CPU (PIL) and never raises into the caller: callers fail
open to the existing maskedUrl. Gated by env flags (default OFF) at the call site.
"""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from services.image_normalizer import (
    _flatten_rgba,
    _open_rgba,
    _trim_near_white_bounds,
    _trim_transparent_bounds,
)

logger = logging.getLogger("ahvi.catalog")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANVAS_SIZE = 1600
OBJECT_FILL = 0.84  # ~8% padding each side, inside the 8-12% target band.
OFF_WHITE = (247, 247, 245, 255)  # #F7F7F5
OFF_WHITE_RGB = (247, 247, 245)

# Categories that get a catalog image at all.
CATALOG_CATEGORIES = {
    "top",
    "bottom",
    "dress",
    "ethnic",
    "outerwear",
    "footwear",
    "accessory",
    "bag",
    "jewellery",
    "jewelry",
}
# Categories that must never get one (defence in depth — non-fashion / private).
SKIP_CATEGORIES = {
    "skincare",
    "makeup",
    "grooming",
    "innerwear",
    "documents",
    "document",
    "gadgets",
    "gadget",
}

# Only these garment categories may be auto-rotated when they look sideways.
_ROTATE_CATEGORIES = {"top", "dress", "outerwear", "ethnic"}
# bbox aspect ratio (w/h) above which a rotate-category garment is "clearly
# sideways". Kept conservative so a wide-cut top is not flipped needlessly.
_SIDEWAYS_RATIO = 1.15

# Phase 8 — AI regeneration prompt (designed, not used yet).
CATALOG_AI_PROMPT = (
    "Clean this garment into a premium catalog product photo on a neutral "
    "background. Preserve the exact garment design, color, pattern, fabric "
    "texture, neckline, sleeves, length, and shape. Remove any person, body, "
    "face, hands, hanger, mannequin, clutter, or background. Do not add a "
    "model. Do not change the garment."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_category(category: Any) -> str:
    return str(category or "").strip().lower()


def category_allowed(category: Any) -> bool:
    """True if a catalog image should be attempted for this category."""
    cat = _norm_category(category)
    if not cat:
        return False
    if cat in SKIP_CATEGORIES:
        return False
    return cat in CATALOG_CATEGORIES


# ---------------------------------------------------------------------------
# Core deterministic canvas build (straighten + rotate + trim + center)
# ---------------------------------------------------------------------------
def _foreground_bbox(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Return (left, top, right, bottom) of visible content, alpha-first then
    near-white fallback. None when nothing visible."""
    rgba = img.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox:
        return bbox
    # No usable alpha (opaque photo) — fall back to non-white bounds.
    trimmed = _trim_near_white_bounds(rgba)
    if trimmed.size != rgba.size:
        # _trim_near_white_bounds already cropped; recover bbox via difference.
        return rgba.getbbox()
    return rgba.getbbox()


def _build_catalog_canvas(
    image_bytes: bytes,
    category: Any,
) -> Tuple[Optional[Image.Image], int, Dict[str, int], str]:
    """Straighten -> sideways-rotate -> trim -> center on off-white canvas.

    Returns (canvas_rgba | None, rotation_applied, foreground_bounds, reason).
    canvas is None when the foreground is empty/undecodable.
    """
    try:
        img = _open_rgba(image_bytes)  # applies EXIF orientation
    except Exception as exc:  # noqa: BLE001
        return None, 0, {}, f"decode_failed:{exc}"

    bbox = _foreground_bbox(img)
    if not bbox:
        return None, 0, {}, "empty_foreground"

    left, top, right, bottom = bbox
    fg_w, fg_h = right - left, bottom - top
    if fg_w <= 0 or fg_h <= 0:
        return None, 0, {}, "empty_foreground"

    foreground_bounds = {"x": int(left), "y": int(top), "w": int(fg_w), "h": int(fg_h)}

    # Sideways rotation: only rotate-eligible categories, only when clearly wide.
    rotation_applied = 0
    cat = _norm_category(category)
    if cat in _ROTATE_CATEGORIES and fg_w > fg_h * _SIDEWAYS_RATIO:
        # Rotate 90 clockwise, expand so nothing clips. Deterministic direction.
        img = img.rotate(-90, expand=True)
        rotation_applied = -90

    # Trim margins (alpha first; near-white fallback for opaque inputs).
    trimmed = _trim_transparent_bounds(img)
    if trimmed.getchannel("A").getbbox() is None:
        trimmed = _trim_near_white_bounds(img)

    canvas = _center_to_fill(trimmed)
    return canvas, rotation_applied, foreground_bounds, ""


def _center_to_fill(img: Image.Image) -> Image.Image:
    """Scale garment (up OR down) to OBJECT_FILL of the canvas, keep aspect,
    center on off-white. Unlike thumbnail-based centering this upscales small
    crops so the catalog frame is consistently filled."""
    img = img.convert("RGBA")
    max_w = int(CANVAS_SIZE * OBJECT_FILL)
    max_h = int(CANVAS_SIZE * OBJECT_FILL)
    w, h = img.size
    if w <= 0 or h <= 0:
        scale = 1.0
    else:
        scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), OFF_WHITE)
    x = (CANVAS_SIZE - new_w) // 2
    y = (CANVAS_SIZE - new_h) // 2
    canvas.alpha_composite(resized, (x, y))
    return canvas


def _encode(canvas: Image.Image, output_format: str = "JPEG") -> bytes:
    out = io.BytesIO()
    fmt = (output_format or "JPEG").upper()
    if fmt in {"JPG", "JPEG"}:
        flat = _flatten_rgba(canvas, OFF_WHITE)
        flat.save(out, format="JPEG", quality=92, optimize=True)
    elif fmt == "WEBP":
        flat = _flatten_rgba(canvas, OFF_WHITE)
        flat.save(out, format="WEBP", quality=92, method=4)
    else:
        canvas.save(out, format="PNG", optimize=True)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _avg_color_rgba(img: Image.Image) -> Optional[Tuple[float, float, float]]:
    """Average color of visible (alpha>16) pixels. None if nothing visible."""
    rgba = img.convert("RGBA")
    small = rgba.resize((64, 64))
    px = small.load()
    r = g = b = n = 0
    for y in range(64):
        for x in range(64):
            cr, cg, cb, ca = px[x, y]
            if ca <= 16:
                continue
            r += cr
            g += cg
            b += cb
            n += 1
    if n == 0:
        return None
    return (r / n, g / n, b / n)


def _is_skin(r: float, g: float, b: float) -> bool:
    # Loose skin-tone heuristic (RGB rule-of-thumb).
    return (
        r > 95
        and g > 40
        and b > 20
        and r > g
        and r > b
        and (max(r, g, b) - min(r, g, b)) > 15
        and abs(r - g) > 15
    )


def _visible_and_skin_ratio(canvas: Image.Image) -> Tuple[float, float, bool]:
    """Return (visible_ratio, skin_ratio_of_visible, touches_edge).

    visible = pixels meaningfully different from the off-white background.
    """
    rgb = canvas.convert("RGB")
    w, h = rgb.size
    small = rgb.resize((80, 80))
    px = small.load()
    total = 80 * 80
    visible = skin = 0
    min_x, min_y, max_x, max_y = 80, 80, -1, -1
    for y in range(80):
        for x in range(80):
            cr, cg, cb = px[x, y]
            dist = abs(cr - OFF_WHITE_RGB[0]) + abs(cg - OFF_WHITE_RGB[1]) + abs(cb - OFF_WHITE_RGB[2])
            if dist <= 24:  # background
                continue
            visible += 1
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
            if _is_skin(cr, cg, cb):
                skin += 1
    visible_ratio = visible / total
    skin_ratio = (skin / visible) if visible else 0.0
    touches_edge = max_x >= 79 or min_x <= 0 or max_y >= 79 or min_y <= 0
    return visible_ratio, skin_ratio, touches_edge


def validate_catalog_image(
    canvas: Image.Image,
    *,
    original_bytes: Optional[bytes] = None,
    category: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic acceptance checks. Returns {ok, reason, ...metrics}."""
    visible_ratio, skin_ratio, touches_edge = _visible_and_skin_ratio(canvas)
    result: Dict[str, Any] = {
        "ok": True,
        "reason": "",
        "visible_ratio": round(visible_ratio, 4),
        "skin_ratio": round(skin_ratio, 4),
        "touches_edge": touches_edge,
        "color_match": None,
    }

    # 2. garment visible area > 20% of canvas
    if visible_ratio < 0.20:
        result.update(ok=False, reason="visible_area_below_20pct")
        return result
    # 3. garment not cropped at edges (padding should keep it off the border)
    if touches_edge:
        result.update(ok=False, reason="garment_touches_edge")
        return result
    # 7. no person/skin-dominant region (best-effort; only reject when extreme)
    if skin_ratio > 0.60:
        result.update(ok=False, reason="skin_region_dominant")
        return result
    # 5. dominant color roughly matches original crop
    if original_bytes:
        try:
            orig = _avg_color_rgba(_open_rgba(original_bytes))
            cat_color = _avg_color_rgba(canvas)
            if orig and cat_color:
                d = sum(abs(a - b) for a, b in zip(orig, cat_color))
                result["color_match"] = round(d, 2)
                if d > 180:  # very loose; only catches gross mismatch
                    result.update(ok=False, reason="color_mismatch")
                    return result
        except Exception:  # noqa: BLE001
            pass  # color check is best-effort, never blocks on its own error
    return result


# ---------------------------------------------------------------------------
# Public: P0 normalize (straighten + center)  — returns encoded bytes
# ---------------------------------------------------------------------------
def normalize_catalog_image(
    image_bytes: bytes,
    category: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    output_format: str = "JPEG",
) -> Dict[str, Any]:
    """Deterministic straighten + center for a (preferably masked) garment image.

    Returns:
        {success, image_bytes, rotation_applied, canvas_size, foreground_bounds, reason}
    """
    t0 = time.monotonic()
    logger.info("ahvi.catalog.normalize.start category=%s", _norm_category(category))
    canvas, rotation, bounds, reason = _build_catalog_canvas(image_bytes, category)
    if canvas is None:
        logger.info("ahvi.catalog.normalize.failed reason=%s", reason)
        return {
            "success": False,
            "image_bytes": None,
            "rotation_applied": 0,
            "canvas_size": [CANVAS_SIZE, CANVAS_SIZE],
            "foreground_bounds": {},
            "reason": reason or "normalize_failed",
        }
    if rotation:
        logger.info("ahvi.catalog.normalize.rotation rotation=%d", rotation)
    out_bytes = _encode(canvas, output_format)
    logger.info(
        "ahvi.catalog.normalize.centered rotation=%d bounds=%s elapsed_ms=%d",
        rotation,
        bounds,
        int((time.monotonic() - t0) * 1000),
    )
    return {
        "success": True,
        "image_bytes": out_bytes,
        "rotation_applied": rotation,
        "canvas_size": [CANVAS_SIZE, CANVAS_SIZE],
        "foreground_bounds": bounds,
        "reason": "",
    }


# ---------------------------------------------------------------------------
# Public: Phase 1 full pipeline (RMBG-first) — generate + validate
# ---------------------------------------------------------------------------
def generate_catalog_image(
    image_bytes: bytes,
    item_metadata: Optional[Dict[str, Any]] = None,
    mode: str = "rmbg_first",
    *,
    output_format: str = "JPEG",
) -> Dict[str, Any]:
    """Deterministic catalog image from an already-cut (masked) garment image.

    Input should be the masked/cutout bytes (RMBG already run upstream). No AI.

    Returns:
        {success, catalog_image_bytes, method, reason, validation}
    """
    meta = item_metadata or {}
    category = meta.get("category")
    t0 = time.monotonic()
    logger.info(
        "ahvi.catalog.start item_id=%s category=%s mode=%s",
        meta.get("item_id"),
        _norm_category(category),
        mode,
    )

    if not category_allowed(category):
        logger.info("ahvi.catalog.summary status=skipped_category category=%s", _norm_category(category))
        return {
            "success": False,
            "catalog_image_bytes": None,
            "method": "rmbg",
            "reason": "category_not_eligible",
            "validation": {},
        }

    canvas, rotation, bounds, reason = _build_catalog_canvas(image_bytes, category)
    if canvas is None:
        logger.info("ahvi.catalog.failed reason=%s", reason)
        return {
            "success": False,
            "catalog_image_bytes": None,
            "method": "rmbg",
            "reason": reason or "build_failed",
            "validation": {},
        }
    logger.info("ahvi.catalog.rmbg_complete rotation=%d", rotation)
    logger.info("ahvi.catalog.centered bounds=%s", bounds)

    validation = validate_catalog_image(
        canvas, original_bytes=image_bytes, category=category, metadata=meta
    )
    if not validation.get("ok"):
        logger.info(
            "ahvi.catalog.validation_failed reason=%s metrics=%s",
            validation.get("reason"),
            validation,
        )
        return {
            "success": False,
            "catalog_image_bytes": None,
            "method": "rmbg",
            "reason": f"validation:{validation.get('reason')}",
            "validation": validation,
        }

    out_bytes = _encode(canvas, output_format)
    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(
        "ahvi.catalog.summary status=ok item_id=%s category=%s method=rmbg rotation=%d elapsed_ms=%d",
        meta.get("item_id"),
        _norm_category(category),
        rotation,
        elapsed,
    )
    return {
        "success": True,
        "catalog_image_bytes": out_bytes,
        "method": "rmbg",
        "reason": "",
        "rotation_applied": rotation,
        "foreground_bounds": bounds,
        "validation": validation,
        "generated_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Phase 8 — AI regeneration hook (designed, NOT implemented)
# ---------------------------------------------------------------------------
def generate_catalog_image_ai(
    image_bytes: bytes,
    prompt: str = CATALOG_AI_PROMPT,
    provider: str = "imagen",
) -> Dict[str, Any]:
    """Future: regenerate a poor RMBG result into a catalog photo via Imagen /
    Flux Kontext. Interface only — deliberately not implemented yet.

    Use ONLY when RMBG output is poor, a selfie/model body remains, the garment
    is partially occluded, or the background is bad.
    """
    raise NotImplementedError(
        "AI catalog regeneration (provider=%s) is not implemented in P0/Phase-1. "
        "Deterministic RMBG + center only." % provider
    )
