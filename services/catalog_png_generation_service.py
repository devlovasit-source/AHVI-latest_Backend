"""Premium transparent wardrobe catalog PNG generation.

This module is intentionally fail-open. It builds a clean centered transparent
PNG from the existing RMBG cutout, scores it, and optionally calls an external
provider only when the deterministic cutout is not catalog quality.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests
from PIL import Image, ImageChops, ImageFilter, ImageOps

from services.catalog_image_service import category_allowed, normalize_catalog_category
from services.image_normalizer import _open_rgba, _trim_near_white_bounds, _trim_transparent_bounds

try:  # google-genai is present in production, but local/dev must fail open.
    from google import genai
    from google.genai import types
except Exception:  # noqa: BLE001
    genai = None
    types = None

logger = logging.getLogger("ahvi.catalog_png")

CATALOG_GENERATION_VERSION = "catalog_png_v1"
CANVAS_SIZE = 1600
OBJECT_FILL = 0.86
QUALITY_READY_THRESHOLD = 82

CATALOG_PROMPT = """You are a professional fashion e-commerce image editor specializing in premium catalog product photography, garment cutouts, and accurate clothing preservation.

Transform the provided garment image into a clean premium e-commerce catalog product image.

The garment is the only product.

The hanger, hook, rod, mannequin, human body parts,
hands, arms, legs, face, shadows and background
are not part of the product.

Remove them completely.

Reconstruct any garment areas hidden by the hanger,
hook or body.

Preserve exactly:
- color
- pattern
- fabric
- silhouette
- neckline
- straps
- sleeves
- hemline

Output a premium online retail catalog image.

Requirements:
- garment only
- centered
- front-facing
- upright
- evenly lit
- clean edges
- PNG-friendly clean edges
- no hanger
- no hook
- no rod
- no mannequin
- no model
- no props
- no accessories
- transparent background with a clean alpha channel
- no baked background fill of any colour
- garment only, centered, natural catalog lighting
- no black border
- no outline
- no rectangular frame
- no product card
- no box
- no mat
- no screenshot frame
- no template border
- no graphic layout
- no text
- no watermark

Quality priority:
- exact garment preservation over creativity
- realistic fabric detail
- accurate pattern retention
- premium product presentation"""


# Ghost / invisible-mannequin variant (behind CATALOG_GHOST_MANNEQUIN). Produces
# the premium "worn 3D form on an invisible body" look for wearable garments,
# instead of the flat repair/cleanup the default prompt yields.
GHOST_MANNEQUIN_PROMPT = """Create a premium fashion e-commerce catalog photo of this garment on an INVISIBLE (ghost) mannequin.

Render the garment in its natural worn 3D form — structured shoulders, filled sleeves, natural drape and volume from neckline to hem — as if worn on a body, but with NO visible body, person, face, hands, mannequin, hanger, hook, or prop. The garment holds a realistic worn shape on its own.

Reconstruct any parts hidden, flattened, or distorted in the source (collar, shoulders, sleeves, hemline) so the COMPLETE garment is shown front-facing, upright, centered, top to hem.

Preserve EXACTLY: the same garment, its color, print/pattern, fabric texture, silhouette, proportions, neckline, sleeves, straps, and hemline length. Do NOT redesign, recolor, or alter the print. Never convert full-length to short.

Background: fully transparent with a clean alpha channel, soft even lighting, a subtle natural contact shadow under the garment only.

Output ONE clean catalog image: garment only on an invisible mannequin. No text, watermark, border, frame, card, template, person, visible mannequin, hanger, or background clutter."""


def _ghost_mannequin_enabled() -> bool:
    return str(os.getenv("CATALOG_GHOST_MANNEQUIN", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_catalog_prompt(category: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
    meta = metadata or {}
    cat = normalize_catalog_category(category)
    blob = _text_blob(meta)
    subcategory = str(meta.get("sub_category") or meta.get("subcategory") or "").strip()
    color = str(meta.get("color_name") or meta.get("color") or "").strip()
    pattern = str(meta.get("pattern") or meta.get("print") or "").strip()
    name = str(meta.get("name") or meta.get("label") or "").strip()
    anchor_parts = []
    if cat:
        anchor_parts.append(f"category = {cat}")
    if subcategory:
        anchor_parts.append(f"subcategory = {subcategory}")
    if color:
        anchor_parts.append(f"color = {color}")
    if pattern:
        anchor_parts.append(f"pattern = {pattern}")
    if name:
        anchor_parts.append(f"item = {name}")
    anchor = ""
    if anchor_parts:
        anchor = "\n\nMETADATA ANCHOR:\n" + "\n".join(anchor_parts)
    # Ghost / invisible-mannequin look for wearable garments (flag-gated). This
    # supersedes the flat repair/cleanup prompt so the garment renders in a
    # premium worn 3D form on an invisible body.
    if _ghost_mannequin_enabled() and cat in {
        "dress",
        "top",
        "outerwear",
        "ethnic",
        "bottom",
    }:
        return GHOST_MANNEQUIN_PROMPT + anchor
    details = []
    if cat == "dress" or "dress" in blob:
        descriptor_parts = []
        if color:
            descriptor_parts.append(color)
        if subcategory:
            descriptor_parts.append(subcategory)
        elif "sleeveless" in blob:
            descriptor_parts.append("sleeveless dress")
        else:
            descriptor_parts.append("dress")
        if pattern:
            descriptor_parts.append(pattern)
        descriptor = " ".join(descriptor_parts).strip()
        final_anchor = (
            f"\n\nThis is a {descriptor}.\n"
            f"The final image must still be a {descriptor}."
            if descriptor
            else ""
        )
        return f"""Create a premium fashion e-commerce product image.

PRIMARY SUBJECT:
The dress is the product and must remain fully visible.

Keep exactly:
- same dress
- same color
- same fabric
- same print pattern
- same silhouette
- same proportions
- same hemline length
- same neckline shape

The dress must remain the dominant object in the image.

REPAIR TASK:
The input image contains a hanger, hook and support rod.

Remove:
- hanger
- hook
- clothing rod
- support hardware

After removal, reconstruct any hidden garment areas naturally.

Specifically reconstruct:
- shoulder seams
- neckline
- armholes
- upper bodice

The reconstructed areas must match the existing fabric pattern and color.

DO NOT:
- crop the garment
- remove the garment
- replace the garment
- redesign the garment
- change fit
- change color
- change pattern
- generate a mannequin
- generate a model
- generate accessories

OUTPUT:
Single garment product photo.
Centered.
Upright.
Clean studio lighting.
Transparent background with a clean alpha channel.
No black border, no outline, no rectangular frame, no product card, no box, no mat, no template border.
Entire dress visible from top to hem.
Fashion catalog quality.{anchor}{final_anchor}"""
    elif cat in {"top", "outerwear", "ethnic"}:
        details.append(
            "This item is an upper-body garment. Reconstruct the collar, shoulder, "
            "sleeve, neckline, and upper edge areas hidden by any hanger, hook, rod, or body."
        )
    elif cat == "bottom":
        details.append(
            "This item is a bottom garment. Reconstruct the waistband and upper garment "
            "boundary if the crop is incomplete or occluded."
        )
        blob_bottom = f"{subcategory} {name}".lower()
        is_full_length = any(t in blob_bottom for t in (
            "trouser", "trousers", "pant", "pants", "chino", "chinos",
            "jean", "jeans", "slack", "slacks", "cargo", "cargos",
            "legging", "leggings", "jogger", "joggers"
        ))
        if is_full_length:
            details.append(
                "Keep exactly the same hemline length and pant length. "
                "NEVER convert full-length pants, trousers, or jeans into shorts."
            )
    elif cat in {"accessory", "bag", "footwear", "jewellery", "jewelry"}:
        details.append(
            "This item is an accessory. Preserve the exact shape, proportions, color, "
            "material, and product identity. Do not add people, outfits, scenes, or lifestyle imagery."
        )
    if not details:
        return CATALOG_PROMPT + anchor
    return CATALOG_PROMPT + anchor + "\n\nCategory-specific instruction:\n" + "\n".join(details)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_enabled(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _text_blob(metadata: Optional[Dict[str, Any]]) -> str:
    meta = metadata or {}
    parts = [
        meta.get("name"),
        meta.get("category"),
        meta.get("sub_category"),
        meta.get("subcategory"),
        meta.get("label"),
        meta.get("source"),
        meta.get("label_source"),
        meta.get("crop_source"),
        meta.get("crop_quality"),
        meta.get("review_reason"),
        " ".join(str(x) for x in (meta.get("tags") or []) if x),
    ]
    return " ".join(str(x or "") for x in parts).lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _forced_provider_reason(metadata: Optional[Dict[str, Any]], category: str) -> str:
    meta = metadata or {}
    blob = _text_blob(meta)
    # Explicit boolean risk signals win over text sniffing — a person/mirror
    # source must never fall through to a plain cutout.
    unsafe_reason = str(
        meta.get("unsafe_reason") or meta.get("unsafeSourceReason") or ""
    ).strip()
    if (
        _truthy(meta.get("unsafe_source"))
        or _truthy(meta.get("source_contains_person"))
        or unsafe_reason
    ):
        return "unsafe_person_source"
    if _truthy(meta.get("needs_review")) or _truthy(meta.get("requires_manual_entry")):
        return "needs_review"
    crop_quality = str(meta.get("crop_quality") or meta.get("cropQuality") or "").strip().lower()
    crop_source = str(meta.get("crop_source") or meta.get("cropSource") or "").strip().lower()
    if crop_quality == "full_image_person_risk":
        return "full_image_person_risk"
    if any(tok in blob for tok in ("hanger", "hook", "human", "person", "mannequin", "mirror", "selfie", "body", "torso", "arm", "leg", "feet", "face", "hand")):
        return "human_or_hanger_metadata"
    if normalize_catalog_category(category) == "bottom" and (
        crop_quality in {"full_image", "broad", "broad_crop"}
        or crop_source == "full_image_fallback"
    ):
        return "bottoms_broad_crop"
    return ""


_UNSAFE_SOURCE_REASON_TOKENS = (
    "human_or_mannequin_remnants",
    "human",
    "person",
    "face",
    "body",
    "hands",
    "hand",
    "group_photo",
    "mannequin",
)


def _unsafe_source_reason(metadata: Optional[Dict[str, Any]], validation: Optional[Dict[str, Any]]) -> str:
    meta_blob = _text_blob(metadata)
    validation_reason = str((validation or {}).get("reason") or "").strip().lower()
    checks = (validation or {}).get("checks") or {}
    combined = f"{validation_reason} {meta_blob}"
    if checks.get("no_human") is False or checks.get("no_face") is False or checks.get("no_mannequin") is False:
        return validation_reason or "human_or_mannequin_remnants"
    for token in _UNSAFE_SOURCE_REASON_TOKENS:
        if token in combined:
            return validation_reason or token
    return ""


# Explicit human / mannequin / hanger evidence tokens. Colour or skin-ratio
# heuristics are deliberately NOT here: a clean flat-lay garment (e.g. an orange
# tee that a colour test reads as "skin") must never be treated as a human
# remnant unless capture metadata actually carries one of these tokens.
_EXPLICIT_HUMAN_TOKENS = (
    "human",
    "person",
    "selfie",
    "mirror",
    "mannequin",
    "face",
    "body",
    "torso",
    "arm",
    "leg",
    "feet",
    "hand",
    "hanger",
    "hook",
)

_APPAREL_CATEGORIES = {"top", "bottom", "outerwear", "dress"}


def _explicit_unsafe_evidence(metadata: Optional[Dict[str, Any]]) -> str:
    """Return the explicit human/mannequin/hanger token found in capture
    metadata, or "" when there is none. Used to gate the
    human_or_mannequin_remnants classification so colour alone never blocks a
    clean garment."""
    blob = _text_blob(metadata)
    for token in _EXPLICIT_HUMAN_TOKENS:
        if token in blob:
            return token
    return ""


def _is_apparel_category(category: Any) -> bool:
    return normalize_catalog_category(category) in _APPAREL_CATEGORIES


def _bbox_area_ratio_and_full_frame(
    bbox: Any,
) -> Tuple[Optional[float], bool]:
    """(area_ratio, is_full_frame) from a bbox. Normalized (0..1) bboxes give a
    real ratio; a [0,0,w,h]-style box that starts at the origin and spans the
    frame is flagged full_frame so it is never scored as a tight crop."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None, False
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None, False
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    if w <= 0 or h <= 0:
        return None, False
    normalized = max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5
    if normalized:
        ratio = max(0.0, min(1.0, w * h))
        is_full = x1 <= 0.02 and y1 <= 0.02 and x2 >= 0.98 and y2 >= 0.98
        return round(ratio, 4), is_full
    # Pixel bbox: ratio needs frame dims we don't have here, but a box anchored
    # at the origin and spanning a large area is still a full-frame signal.
    is_full = x1 <= 1 and y1 <= 1 and w >= 64 and h >= 64
    return None, is_full


def _provider_validation_metadata(meta: Dict[str, Any], category: str) -> Dict[str, Any]:
    cleaned = {**meta, "category": category}
    for key in (
        "needs_review",
        "requires_manual_entry",
        "crop_quality",
        "cropQuality",
        "crop_source",
        "cropSource",
        "review_reason",
        "source",
        "label_source",
    ):
        cleaned.pop(key, None)
    return cleaned


def _foreground_bbox(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    rgba = img.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox:
        return bbox
    trimmed = _trim_near_white_bounds(rgba)
    if trimmed.size != rgba.size:
        return trimmed.getbbox()
    return rgba.getbbox()


def _transparent_catalog_canvas(image_bytes: bytes, category: Any) -> Tuple[Optional[Image.Image], int, Dict[str, int], str]:
    try:
        img = _open_rgba(image_bytes)
    except Exception as exc:  # noqa: BLE001
        return None, 0, {}, f"decode_failed:{exc}"

    bbox = _foreground_bbox(img)
    if not bbox:
        return None, 0, {}, "empty_foreground"

    left, top, right, bottom = bbox
    fg_w, fg_h = max(1, right - left), max(1, bottom - top)
    rotation = 0
    cat = normalize_catalog_category(category)
    if cat in {"top", "dress", "outerwear", "ethnic"} and fg_w > fg_h * 1.15:
        img = img.rotate(-90, expand=True)
        rotation = -90

    trimmed = _trim_transparent_bounds(img)
    if trimmed.getchannel("A").getbbox() is None:
        trimmed = _trim_near_white_bounds(img)
    trimmed = trimmed.convert("RGBA")

    w, h = trimmed.size
    max_w = int(CANVAS_SIZE * OBJECT_FILL)
    max_h = int(CANVAS_SIZE * OBJECT_FILL)
    scale = min(max_w / max(1, w), max_h / max(1, h))
    resized = trimmed.resize(
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    x = (CANVAS_SIZE - resized.size[0]) // 2
    y = (CANVAS_SIZE - resized.size[1]) // 2
    canvas.alpha_composite(resized, (x, y))
    bounds = {"x": int(x), "y": int(y), "w": int(resized.size[0]), "h": int(resized.size[1])}
    return canvas, rotation, bounds, ""


def _provider_output_to_transparent(
    image_bytes: bytes, category: Any, item_id: Any = ""
) -> Tuple[bytes, str]:
    """Provider output -> RMBG -> transparent catalogue canvas.

    Returns (png_bytes, reason). On any failure returns (b"", reason) so the
    caller can discard the opaque provider output. RMBG stays the enforcement
    step because the model may ignore the prompt's transparency instruction."""
    try:
        if not is_effectively_transparent(image_bytes):
            from services.bg_service import remove_bg_external_sync

            image_bytes = remove_bg_external_sync(image_bytes)
            # bg_service fails OPEN (returns the input unchanged when RMBG is
            # unset or errors). Check BEFORE canvasing: the canvas adds
            # transparent padding around the garment, which would mask a
            # still-white cutout and reproduce the white-box bug.
            if not is_effectively_transparent(image_bytes):
                logger.info(
                    "ahvi.catalog.transparency.still_opaque item_id=%s", item_id
                )
                return b"", "still_opaque"
        canvas, _rot, _bounds, reason = _transparent_catalog_canvas(image_bytes, category)
        if canvas is None:
            logger.info(
                "ahvi.catalog.transparency.canvas_failed item_id=%s reason=%s",
                item_id,
                reason,
            )
            return b"", reason or "canvas_failed"
        out = _encode_png(canvas)
        if not is_effectively_transparent(out):
            logger.info(
                "ahvi.catalog.transparency.still_opaque item_id=%s", item_id
            )
            return b"", "still_opaque"
        logger.info("ahvi.catalog.transparency.ok item_id=%s", item_id)
        return out, "ok"
    except Exception as exc:  # noqa: BLE001 - never break generation
        logger.info(
            "ahvi.catalog.transparency.failed item_id=%s err=%s",
            item_id,
            str(exc)[:120],
        )
        return b"", "rmbg_failed"


def is_effectively_transparent(image_bytes: bytes) -> bool:
    """True when the image carries real transparency (>=1% transparent pixels)."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in {"RGBA", "LA", "P"}:
            return False
        alpha = img.convert("RGBA").getchannel("A")
        values = list(alpha.resize((64, 64)).getdata())
        if not values:
            return False
        return (sum(1 for v in values if v < 16) / float(len(values))) >= 0.01
    except Exception:  # noqa: BLE001
        return False


def _encode_png(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _alpha_metrics(img: Image.Image) -> Dict[str, Any]:
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return {
            "visible_ratio": 0.0,
            "center_offset": 1.0,
            "touches_edge": False,
            "edge_quality": 0,
            "completeness": 0,
        }
    w, h = rgba.size
    visible = sum(1 for v in alpha.resize((96, 96)).getdata() if v > 16)
    visible_ratio = visible / float(96 * 96)
    l, t, r, b = bbox
    cx = (l + r) / 2.0
    cy = (t + b) / 2.0
    center_offset = (abs(cx - w / 2) / (w / 2) + abs(cy - h / 2) / (h / 2)) / 2
    touches = l <= 2 or t <= 2 or r >= w - 2 or b >= h - 2
    # Edge fuzz: compare alpha to a blurred alpha; very noisy edges score lower.
    edge = ImageChops.difference(alpha, alpha.filter(ImageFilter.GaussianBlur(1.4)))
    edge_values = list(edge.resize((64, 64)).getdata())
    edge_noise = sum(edge_values) / max(1, len(edge_values))
    edge_quality = max(0, min(100, int(100 - edge_noise * 1.8)))
    completeness = 100
    if visible_ratio < 0.06:
        completeness = 25
    elif visible_ratio < 0.10:
        completeness = 55
    elif touches:
        completeness = 60
    return {
        "visible_ratio": round(visible_ratio, 4),
        "center_offset": round(center_offset, 4),
        "touches_edge": touches,
        "edge_quality": edge_quality,
        "completeness": completeness,
    }


def _foreground_avg(img: Image.Image) -> Optional[Tuple[float, float, float]]:
    small = img.convert("RGBA").resize((64, 64))
    r = g = b = n = 0
    for cr, cg, cb, ca in small.getdata():
        if ca <= 16:
            continue
        r += cr
        g += cg
        b += cb
        n += 1
    if not n:
        return None
    return r / n, g / n, b / n


def _skin_ratio(img: Image.Image) -> float:
    small = img.convert("RGBA").resize((80, 80))
    visible = skin = 0
    for r, g, b, a in small.getdata():
        if a <= 16:
            continue
        visible += 1
        if r > 95 and g > 40 and b > 35 and r > g > b and (r - g) > 12 and (g - b) < 48:
            skin += 1
    return round(skin / visible, 4) if visible else 0.0


def _is_jewelry_or_accessory(category: Any) -> bool:
    cat = normalize_catalog_category(category)
    return cat in {"accessory", "jewellery", "jewelry", "bag", "watch"}


def _catalog_blank_reason(image_bytes: bytes, category: Any) -> str:
    """Reject empty/flat provider outputs before they become wardrobe assets."""
    try:
        rgba = _open_rgba(image_bytes)
    except Exception as exc:  # noqa: BLE001
        return f"decode_failed:{exc}"

    small = rgba.resize((96, 96)).convert("RGBA")
    pixels = list(small.getdata())
    visible = [p for p in pixels if p[3] > 16]
    if not visible:
        return "blank_transparent_catalog"

    # Opaque provider outputs can be valid on a white studio background. Treat
    # the image as blank only when there is effectively no non-background detail.
    non_white = [
        p for p in visible
        if not (p[0] >= 244 and p[1] >= 244 and p[2] >= 244)
    ]
    non_white_ratio = len(non_white) / max(1, len(visible))
    alpha_visible_ratio = len(visible) / float(96 * 96)

    if alpha_visible_ratio < 0.002:
        return "blank_transparent_catalog"
    if non_white_ratio < 0.003:
        return "blank_flat_catalog"

    if _is_jewelry_or_accessory(category):
        bbox = small.getchannel("A").getbbox()
        if bbox:
            l, t, r, b = bbox
            bbox_ratio = ((r - l) * (b - t)) / float(96 * 96)
        else:
            bbox_ratio = 0.0
        if non_white_ratio < 0.01 or bbox_ratio < 0.006:
            return "tiny_accessory_catalog"

    return ""


_BLACK_FRAME_N = 288
_BLACK_FRAME_DARK_T = 55          # pixel is "dark" if max(r,g,b) < this
_BLACK_FRAME_LINE_T = 0.80        # row/col is a "line" if >= this fraction dark
_BLACK_FRAME_BAND_RATIO = 0.14    # outer/inset band scanned for frame lines


def _black_frame_metrics(image_bytes: bytes) -> Dict[str, Any]:
    """Detect baked-in black frames/borders/side bars on a generated catalog
    image. Multi-band, line-structure based so it catches inset frames and thin
    edges WITHOUT false-positiving on dark garments (which are thick blobs, not
    long thin lines along the edges)."""
    empty = {
        "detected": False,
        "frame_type": "",
        "border_dark_ratio": 0.0,
        "center_dark_ratio": 0.0,
        "candidate_crop_box": None,
    }
    try:
        # NEAREST keeps thin 1-2px borders crisp instead of blurring them away.
        rgba = _open_rgba(image_bytes).resize(
            (_BLACK_FRAME_N, _BLACK_FRAME_N), Image.Resampling.NEAREST
        ).convert("RGBA")
    except Exception:
        return empty

    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    rgb = white.convert("RGB")
    n = _BLACK_FRAME_N
    pixels = list(rgb.getdata())
    dark_t = _BLACK_FRAME_DARK_T

    # dark_mask[y][x] flattened: 1 if pixel dark.
    dark = [1 if max(p) < dark_t else 0 for p in pixels]

    row_dark = [0.0] * n
    col_dark = [0.0] * n
    col_sum = [0] * n
    for y in range(n):
        base = y * n
        s = 0
        for x in range(n):
            d = dark[base + x]
            s += d
            col_sum[x] += d
        row_dark[y] = s / n
    for x in range(n):
        col_dark[x] = col_sum[x] / n

    band = max(2, int(n * _BLACK_FRAME_BAND_RATIO))
    line_t = _BLACK_FRAME_LINE_T

    # Largest dark idx near the low edge, smallest dark idx near the high edge.
    def _deep_low(values):
        idx = -1
        for i in range(0, band):
            if values[i] >= line_t:
                idx = i
        return idx

    def _deep_high(values):
        idx = -1
        for i in range(n - 1, n - band - 1, -1):
            if values[i] >= line_t:
                idx = i
        return idx

    top_idx = _deep_low(row_dark)
    bottom_idx = _deep_high(row_dark)
    left_idx = _deep_low(col_dark)
    right_idx = _deep_high(col_dark)

    top_line = top_idx >= 0
    bottom_line = bottom_idx >= 0
    left_line = left_idx >= 0
    right_line = right_idx >= 0

    c0, c1 = int(n * 0.30), int(n * 0.70)
    center_total = center_dark_n = 0
    for y in range(c0, c1):
        base = y * n
        for x in range(c0, c1):
            center_total += 1
            center_dark_n += dark[base + x]
    center_dark = center_dark_n / max(1, center_total)

    edge_band = [
        v
        for i, v in enumerate(row_dark + col_dark)
        if (i % n) < band or (i % n) >= n - band
    ]
    border_dark = sum(edge_band) / max(1, len(edge_band))

    sides = left_line and right_line
    horiz = top_line and bottom_line
    ring = sides and horiz

    # Structural frame = opposite dark LINES (both sides, or top+bottom, or full
    # ring) with a light center. Dark lines are >=LINE_T dark by construction and
    # the center is light, so no extra border>center margin gate is needed (which
    # would wrongly reject thin 1px frames whose single line averages low).
    structural = ring or sides or horiz
    detected = structural and center_dark < 0.45

    frame_type = ""
    candidate_crop_box = None
    if detected:
        if ring:
            edge_dark = (
                row_dark[0] >= line_t
                or row_dark[n - 1] >= line_t
                or col_dark[0] >= line_t
                or col_dark[n - 1] >= line_t
            )
            frame_type = "outer_border" if edge_dark else "inset_frame"
        elif sides:
            frame_type = "vertical_side_bars"
        elif horiz:
            frame_type = "horizontal_bars"

        inner_l = (left_idx + 1) if left_line else 0
        inner_t = (top_idx + 1) if top_line else 0
        inner_r = (right_idx - 1) if right_line else n - 1
        inner_b = (bottom_idx - 1) if bottom_line else n - 1
        if inner_r - inner_l > n * 0.30 and inner_b - inner_t > n * 0.30:
            candidate_crop_box = (
                round(inner_l / n, 4),
                round(inner_t / n, 4),
                round((inner_r + 1) / n, 4),
                round((inner_b + 1) / n, 4),
            )

    return {
        "detected": bool(detected),
        "frame_type": frame_type,
        "border_dark_ratio": round(border_dark, 4),
        "center_dark_ratio": round(center_dark, 4),
        "candidate_crop_box": candidate_crop_box,
    }


def _has_visible_content(image_bytes: bytes) -> bool:
    """True if image still has meaningful non-white content (garment survived)."""
    try:
        rgba = _open_rgba(image_bytes).resize((96, 96)).convert("RGBA")
    except Exception:
        return False
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    pixels = list(white.convert("RGB").getdata())
    non_white = sum(
        1 for r, g, b in pixels if not (r >= 244 and g >= 244 and b >= 244)
    )
    return (non_white / max(1, len(pixels))) > 0.003


def _crop_black_frame(image_bytes: bytes) -> Tuple[bytes, bool]:
    """Crop out a detected black frame/border/side bars to the inner content.

    Uses the candidate crop box from ``_black_frame_metrics`` (inner region just
    inside the frame). Falls back to a near-white bounding box for plain
    borders. Only accepts the crop if the result no longer has a frame and the
    garment is still visible. Returns (bytes, cropped)."""
    try:
        rgba = _open_rgba(image_bytes).convert("RGBA")
    except Exception:
        return image_bytes, False

    w, h = rgba.size
    metrics = _black_frame_metrics(image_bytes)
    box: Optional[Tuple[int, int, int, int]] = None

    candidate = metrics.get("candidate_crop_box")
    if candidate:
        cl, ct, cr, cb = candidate
        # Inset a hair further so the (anti-aliased) frame line itself is gone.
        inset = 0.006
        l = max(0, int(round((cl + inset) * w)))
        t = max(0, int(round((ct + inset) * h)))
        r = min(w, int(round((cr - inset) * w)))
        b = min(h, int(round((cb - inset) * h)))
        if r - l > w * 0.30 and b - t > h * 0.30:
            box = (l, t, r, b)

    if box is None:
        # Fallback: trim toward the non-dark content bounding box.
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.alpha_composite(rgba)
        gray = white.convert("L")
        mask = gray.point(lambda p: 255 if p > 38 else 0)
        bbox = mask.getbbox()
        if not bbox:
            return image_bytes, False
        l, t, r, b = bbox
        if r - l < w * 0.30 or b - t < h * 0.30:
            return image_bytes, False
        box = (l, t, r, b)

    pad = max(6, int(min(w, h) * 0.012))
    l, t, r, b = box
    box = (max(0, l - pad), max(0, t - pad), min(w, r + pad), min(h, b + pad))
    cropped = rgba.crop(box)
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    scale = min(
        (w * 0.92) / max(1, cropped.size[0]),
        (h * 0.92) / max(1, cropped.size[1]),
        1.0,
    )
    resized = cropped.resize(
        (max(1, int(cropped.size[0] * scale)), max(1, int(cropped.size[1] * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas.alpha_composite(resized, ((w - resized.size[0]) // 2, (h - resized.size[1]) // 2))
    out = _encode_png(canvas)

    # Only accept if the frame is gone AND the garment survived.
    if _black_frame_metrics(out).get("detected"):
        return image_bytes, False
    if not _has_visible_content(out):
        return image_bytes, False
    return out, True


def score_catalog_quality(image_bytes: bytes, *, original_bytes: bytes = b"", item_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        img = _open_rgba(image_bytes)
    except Exception as exc:  # noqa: BLE001
        return {"score": 0, "ok": False, "reason": f"decode_failed:{exc}"}

    meta = item_metadata or {}
    cat = normalize_catalog_category(meta.get("category"))
    metrics = _alpha_metrics(img)
    skin = _skin_ratio(img)
    blob = _text_blob(meta)
    if metrics["visible_ratio"] <= 0:
        return {
            "score": 0,
            "ok": False,
            "reason": "missing_garment_sections",
            "edge_quality": 0,
            "orientation_quality": 0,
            "cleanliness": 0,
            "garment_completeness": 0,
            "color_preservation": 0,
            "pattern_preservation": 0,
            "visible_ratio": metrics["visible_ratio"],
            "center_offset": metrics["center_offset"],
            "touches_edge": metrics["touches_edge"],
            "skin_ratio": skin,
            "color_distance": None,
        }

    human_penalty = 0
    # Explicit evidence (capture metadata) is the ONLY thing that may classify a
    # garment as a human/mannequin remnant. Skin/colour ratio stays a soft score
    # penalty but must not hard-label — an orange flat-lay tee reads as "skin".
    unsafe_evidence = _explicit_unsafe_evidence(meta)
    remnant_signal = bool(unsafe_evidence)
    if remnant_signal:
        human_penalty += 54
    if cat not in {"bag", "footwear", "accessory", "jewellery", "jewelry"} and skin > 0.25:
        human_penalty += 30

    original_avg = _foreground_avg(_open_rgba(original_bytes)) if original_bytes else None
    generated_avg = _foreground_avg(img)
    color_distance = None
    color_score = 88
    if original_avg and generated_avg:
        color_distance = sum(abs(a - b) for a, b in zip(original_avg, generated_avg))
        color_score = max(0, min(100, int(100 - color_distance / 2)))

    orientation_quality = 100 if metrics["center_offset"] <= 0.08 else max(40, int(100 - metrics["center_offset"] * 220))
    cleanliness = 100 - human_penalty
    if metrics["touches_edge"]:
        cleanliness -= 20
    edge_quality = int(metrics["edge_quality"])
    completeness = int(metrics["completeness"])
    pattern_preservation = color_score
    score = int(
        edge_quality * 0.18
        + orientation_quality * 0.18
        + cleanliness * 0.22
        + completeness * 0.18
        + color_score * 0.14
        + pattern_preservation * 0.10
    )
    score = max(0, min(100, score))
    reason = ""
    if metrics["visible_ratio"] < 0.05:
        reason = "missing_garment_sections"
    elif remnant_signal:
        # Only explicit human/mannequin/hanger evidence — never skin colour.
        reason = "human_or_mannequin_remnants"
    elif metrics["touches_edge"]:
        reason = "bad_crop"
    elif orientation_quality < 70:
        reason = "crooked_orientation"
    elif edge_quality < 55:
        reason = "distorted_edges"

    return {
        "score": score,
        "ok": score >= QUALITY_READY_THRESHOLD and not reason,
        "reason": reason,
        "edge_quality": edge_quality,
        "orientation_quality": orientation_quality,
        "cleanliness": max(0, cleanliness),
        "garment_completeness": completeness,
        "color_preservation": color_score,
        "pattern_preservation": pattern_preservation,
        "visible_ratio": metrics["visible_ratio"],
        "center_offset": metrics["center_offset"],
        "touches_edge": metrics["touches_edge"],
        "skin_ratio": skin,
        "color_distance": round(color_distance, 2) if color_distance is not None else None,
        "unsafe_source_evidence": unsafe_evidence,
    }


def validate_catalog_png(image_bytes: bytes, *, original_bytes: bytes = b"", item_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    score = score_catalog_quality(image_bytes, original_bytes=original_bytes, item_metadata=item_metadata)
    blob = _text_blob(item_metadata)
    checks = {
        # Evidence-based, not colour-based: skin_ratio alone (e.g. an orange
        # tee) must not fail no_face. Only an explicit "face" token does.
        "no_face": "face" not in blob,
        "no_human": score.get("reason") != "human_or_mannequin_remnants",
        "no_mannequin": "mannequin" not in blob,
        "no_hanger": "hanger" not in blob,
        "category_matches_original": bool(normalize_catalog_category((item_metadata or {}).get("category"))),
        "color_matches_original": (score.get("color_distance") is None or score.get("color_distance", 0) <= 140),
        "orientation_upright": score.get("orientation_quality", 0) >= 70,
        "garment_centered": score.get("center_offset", 1) <= 0.16,
        "image_dimensions_valid": True,
    }
    image_info = {"image_width": 0, "image_height": 0, "image_mode": ""}
    try:
        img = Image.open(io.BytesIO(image_bytes))
        image_info = {
            "image_width": int(img.size[0]),
            "image_height": int(img.size[1]),
            "image_mode": str(img.mode or ""),
        }
        checks["image_size_valid"] = img.size[0] >= 512 and img.size[1] >= 512
        checks["alpha_or_palette_mode"] = img.mode in {"RGBA", "LA", "P"}
        checks["image_dimensions_valid"] = checks["image_size_valid"] and checks["alpha_or_palette_mode"]
    except Exception:
        checks["image_size_valid"] = False
        checks["alpha_or_palette_mode"] = False
        checks["image_dimensions_valid"] = False

    # Hard backstop: an unresolved baked-in black frame can never pass and is
    # score-capped below the save threshold so demo relaxation cannot accept it.
    frame = _black_frame_metrics(image_bytes)
    checks["no_black_frame"] = not bool(frame.get("detected"))
    result = {**score, **image_info, "checks": checks}
    if frame.get("detected"):
        result["score"] = min(int(result.get("score") or 0), 44)
        result["reason"] = "black_frame_unresolved"
        result["frame_type"] = frame.get("frame_type")
        result["ok"] = False
    else:
        result["ok"] = bool(score.get("ok")) and all(checks.values())

    # The full-length-bottom shape guard (h/w <= 1.25 -> "shortened") rejects
    # folded/square denim even when the generated render is fine, forcing a
    # cutout fallback. Env-gated so it can be turned off without a redeploy.
    if (
        _env_enabled("CATALOG_ENFORCE_FULL_LENGTH_BOTTOM", "true")
        and normalize_catalog_category((item_metadata or {}).get("category")) == "bottom"
    ):
        meta = item_metadata or {}
        meta_blob = f"{meta.get('sub_category') or ''} {meta.get('name') or ''}".lower()
        is_full_length = any(t in meta_blob for t in (
            "trouser", "trousers", "pant", "pants", "chino", "chinos",
            "jean", "jeans", "slack", "slacks", "cargo", "cargos",
            "legging", "leggings", "jogger", "joggers"
        ))
        if is_full_length:
            try:
                from services.image_normalizer import _trim_near_white_bounds

                rgba = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                width, height = rgba.size
                alpha_bbox = rgba.getchannel("A").getbbox()
                # Nano Banana returns a WHITE-BACKGROUND (opaque) image, so the
                # alpha bbox is the whole canvas — measuring it gives the image
                # aspect ratio, not the garment, and false-rejects square output.
                # When the alpha bbox covers (nearly) the full frame, trim the
                # near-white background to find the real garment extent.
                if alpha_bbox and alpha_bbox != (0, 0, width, height):
                    gen_w = max(1, alpha_bbox[2] - alpha_bbox[0])
                    gen_h = max(1, alpha_bbox[3] - alpha_bbox[1])
                else:
                    trimmed = _trim_near_white_bounds(rgba)
                    gen_w, gen_h = (max(1, trimmed.width), max(1, trimmed.height))
                if (gen_h / gen_w) <= 1.25:
                    result["ok"] = False
                    result["reason"] = "full_length_bottom_shortened"
            except Exception:
                pass

    return result


def _vertex_demo_accepts_generated_validation(validation: Dict[str, Any]) -> bool:
    checks = validation.get("checks") or {}
    reason = str(validation.get("reason") or "").strip()
    # Never accept a baked-in black frame, regardless of other signals.
    if (
        "black_frame" in reason
        or reason in {"dark_side_bars", "inset_frame"}
        or checks.get("no_black_frame") is False
    ):
        return False
    if reason.startswith("decode_failed") or reason in {
        "missing_garment_sections",
        "human_or_mannequin_remnants",
    }:
        return False
    for key in (
        "no_face",
        "no_human",
        "no_mannequin",
        "category_matches_original",
        "image_size_valid",
    ):
        if checks.get(key) is False:
            return False
    # Demo relaxation: accept useful Imagen output when the remaining problem is
    # an opaque/non-transparent background or edge contact from that background.
    if checks.get("alpha_or_palette_mode") is False or checks.get("image_dimensions_valid") is False:
        return reason in {"", "bad_crop"} and int(validation.get("score") or 0) >= 45
    return reason == "bad_crop" and int(validation.get("score") or 0) >= 45


@dataclass
class CatalogProviderResult:
    success: bool
    image_bytes: bytes = b""
    reason: str = ""
    provider: str = "disabled"


class CatalogProvider:
    name = "disabled"

    def generate(self, *, cutout_bytes: bytes, prompt: str, item_metadata: Dict[str, Any], timeout: int) -> CatalogProviderResult:
        return CatalogProviderResult(False, reason="provider_disabled", provider=self.name)


class DisabledCatalogProvider(CatalogProvider):
    name = "disabled"


class HttpCatalogProvider(CatalogProvider):
    def __init__(self, name: str, endpoint_env: str, token_env: str):
        self.name = name
        self.endpoint = os.getenv(endpoint_env, "").strip()
        self.token = os.getenv(token_env, "").strip()

    def generate(self, *, cutout_bytes: bytes, prompt: str, item_metadata: Dict[str, Any], timeout: int) -> CatalogProviderResult:
        if not self.endpoint:
            return CatalogProviderResult(False, reason=f"{self.name}_endpoint_missing", provider=self.name)
        payload = {
            "prompt": prompt,
            "image_base64": base64.b64encode(cutout_bytes).decode("ascii"),
            "metadata": item_metadata,
            "output_format": "png",
            "transparent": True,
        }
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout)
            if resp.status_code >= 400:
                return CatalogProviderResult(False, reason=f"http_{resp.status_code}:{resp.text[:120]}", provider=self.name)
            data = resp.json()
            b64 = str(data.get("image_base64") or data.get("png_base64") or "").strip()
            url = str(data.get("image_url") or data.get("url") or "").strip()
            if b64:
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                return CatalogProviderResult(True, image_bytes=base64.b64decode(b64), provider=self.name)
            if url.startswith("http"):
                img_resp = requests.get(url, timeout=timeout)
                if img_resp.status_code == 200 and img_resp.content:
                    return CatalogProviderResult(True, image_bytes=img_resp.content, provider=self.name)
            return CatalogProviderResult(False, reason="provider_returned_no_image", provider=self.name)
        except Exception as exc:  # noqa: BLE001
            return CatalogProviderResult(False, reason=repr(exc)[:180], provider=self.name)


_vertex_imagen_client = None


def _vertex_project() -> str:
    return os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or "ahvi-485510"


def _vertex_location() -> str:
    return os.getenv("GOOGLE_CLOUD_LOCATION", "global")


def _image_to_png_bytes(image_obj: Any) -> bytes:
    if image_obj is None:
        return b""
    raw = getattr(image_obj, "image_bytes", None) or getattr(image_obj, "imageBytes", None)
    if raw:
        try:
            im = Image.open(io.BytesIO(raw))
            return _encode_png(im.convert("RGBA"))
        except Exception:
            return bytes(raw)
    pil = getattr(image_obj, "_pil_image", None)
    if pil is not None:
        return _encode_png(pil.convert("RGBA"))
    save = getattr(image_obj, "save", None)
    if callable(save):
        buf = io.BytesIO()
        try:
            save(buf)
            buf.seek(0)
            im = Image.open(buf)
            return _encode_png(im.convert("RGBA"))
        except Exception:
            return buf.getvalue()
    return b""


class CatalogProviderVertexImagen(CatalogProvider):
    name = "vertex_imagen"

    def __init__(self):
        self.model = os.getenv("CATALOG_IMAGEN_MODEL", "imagen-3.0-capability-001").strip()

    def _client(self):
        global _vertex_imagen_client
        if genai is None or types is None:
            return None
        if _vertex_imagen_client is not None:
            return _vertex_imagen_client
        _vertex_imagen_client = genai.Client(
            vertexai=True,
            project=_vertex_project(),
            location=_vertex_location(),
            http_options=types.HttpOptions(api_version="v1"),
        )
        return _vertex_imagen_client

    def _edit_config(self):
        fields = getattr(types.EditImageConfig, "model_fields", {}) or {}
        kwargs: Dict[str, Any] = {}
        if not fields or "number_of_images" in fields:
            kwargs["number_of_images"] = 1
        if "output_mime_type" in fields:
            kwargs["output_mime_type"] = "image/png"
        if "add_watermark" in fields:
            kwargs["add_watermark"] = False
        return types.EditImageConfig(**kwargs)

    def generate(self, *, cutout_bytes: bytes, prompt: str, item_metadata: Dict[str, Any], timeout: int) -> CatalogProviderResult:
        del timeout  # Vertex SDK call timeout is controlled by client/http options.
        try:
            client = self._client()
            if client is None:
                return CatalogProviderResult(False, reason="google_genai_unavailable", provider=self.name)
            if not self.model:
                return CatalogProviderResult(False, reason="catalog_imagen_model_missing", provider=self.name)
            reference = types.RawReferenceImage(
                reference_image=types.Image(image_bytes=cutout_bytes, mime_type="image/png"),
                reference_id=1,
            )
            config = self._edit_config()
            response = client.models.edit_image(
                model=self.model,
                prompt=prompt,
                reference_images=[reference],
                config=config,
            )
            generated = (
                getattr(response, "generated_images", None)
                or getattr(response, "images", None)
                or []
            )
            for candidate in generated:
                image_obj = getattr(candidate, "image", None) or candidate
                image_bytes = _image_to_png_bytes(image_obj)
                if image_bytes:
                    return CatalogProviderResult(True, image_bytes=image_bytes, provider=self.name)
            return CatalogProviderResult(False, reason="vertex_imagen_returned_no_image", provider=self.name)
        except Exception as exc:  # noqa: BLE001
            return CatalogProviderResult(False, reason=repr(exc), provider=self.name)


class CatalogProviderNanoBanana(CatalogProvider):
    name = "nanobanana"

    def __init__(self):
        self.model = (
            os.getenv("NANO_BANANA_CATALOG_MODEL")
            or os.getenv("WARDROBE_NANO_BANANA_MODEL")
            or os.getenv("NANO_BANANA_MODEL")
            or "gemini-2.5-flash-image-preview"
        ).strip()

    def _client(self, timeout_s: Optional[int] = None):
        if genai is None or types is None:
            return None
        http_kwargs: Dict[str, Any] = {"api_version": "v1"}
        # Bound the Vertex image-gen call. Without this the SDK call is
        # unbounded, so a cold/slow generation stretches save-selected toward
        # its 120s client timeout. HttpOptions.timeout is milliseconds; only
        # set it when the installed SDK supports the field.
        if timeout_s and timeout_s > 0:
            fields = getattr(types.HttpOptions, "model_fields", {}) or {}
            if not fields or "timeout" in fields:
                http_kwargs["timeout"] = int(timeout_s * 1000)
        return genai.Client(
            vertexai=True,
            project=_vertex_project(),
            location=_vertex_location(),
            http_options=types.HttpOptions(**http_kwargs),
        )

    def _config(self):
        fields = getattr(types.GenerateContentConfig, "model_fields", {}) or {}
        kwargs: Dict[str, Any] = {}
        if not fields or "temperature" in fields:
            kwargs["temperature"] = 0
        if not fields or "candidate_count" in fields:
            kwargs["candidate_count"] = 1
        if not fields or "response_modalities" in fields:
            kwargs["response_modalities"] = ["IMAGE"]
        return types.GenerateContentConfig(**kwargs)

    def generate(self, *, cutout_bytes: bytes, prompt: str, item_metadata: Dict[str, Any], timeout: int) -> CatalogProviderResult:
        # Bound the Vertex image-gen call so a cold/slow generation can't drag
        # save-selected to its 120s tail. A normal success is ~35-40s, so use a
        # nanobanana-specific cap (default 75s) well above that; the generic
        # CATALOG_TIMEOUT_SECONDS (~30s) would kill healthy generations.
        try:
            nb_timeout = int(os.getenv("NANO_BANANA_TIMEOUT_SECONDS", "75") or 75)
        except (TypeError, ValueError):
            nb_timeout = 75
        if timeout and timeout > nb_timeout:
            nb_timeout = timeout
        try:
            client = self._client(timeout_s=nb_timeout)
            if client is None:
                return CatalogProviderResult(False, reason="google_genai_unavailable", provider=self.name)
            if not self.model:
                return CatalogProviderResult(False, reason="nano_banana_model_missing", provider=self.name)
            image_part = types.Part.from_bytes(data=cutout_bytes, mime_type="image/png")
            response = client.models.generate_content(
                model=self.model,
                contents=[prompt, image_part],
                config=self._config(),
            )
            image_bytes = _extract_generated_image_bytes(response)
            if image_bytes:
                return CatalogProviderResult(True, image_bytes=image_bytes, provider=self.name)
            logger.warning(
                "ahvi.catalog.nanobanana.no_image model=%s diag=%s",
                self.model,
                _no_image_diagnostics(response),
            )
            return CatalogProviderResult(False, reason="nanobanana_returned_no_image", provider=self.name)
        except Exception as exc:  # noqa: BLE001
            return CatalogProviderResult(False, reason=repr(exc), provider=self.name)


def _no_image_diagnostics(response: Any) -> str:
    """Best-effort summary of WHY an image-gen response carried no image
    (safety block, text-only/refusal, empty) so nanobanana_returned_no_image
    is debuggable. Never raises."""
    bits: list = []
    try:
        pf = getattr(response, "prompt_feedback", None)
        br = getattr(pf, "block_reason", None) if pf is not None else None
        if br:
            bits.append(f"prompt_block={br}")
    except Exception:  # noqa: BLE001
        pass
    try:
        for cand in (getattr(response, "candidates", None) or []):
            fr = getattr(cand, "finish_reason", None)
            if fr:
                bits.append(f"finish={fr}")
            srs = getattr(cand, "safety_ratings", None) or []
            blocked = [
                str(getattr(s, "category", "?"))
                for s in srs
                if getattr(s, "blocked", False)
            ]
            if blocked:
                bits.append(f"safety_blocked={blocked}")
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                txt = getattr(part, "text", None)
                if txt:
                    bits.append(f"text={str(txt)[:200]!r}")
    except Exception:  # noqa: BLE001
        pass
    return " ".join(bits) or "no_diagnostics"


def _extract_generated_image_bytes(response: Any) -> bytes:
    generated = (
        getattr(response, "generated_images", None)
        or getattr(response, "images", None)
        or []
    )
    for candidate in generated:
        image_obj = getattr(candidate, "image", None) or candidate
        image_bytes = _image_to_png_bytes(image_obj)
        if image_bytes:
            return image_bytes

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if data:
                if isinstance(data, str):
                    try:
                        data = base64.b64decode(data)
                    except Exception:
                        data = data.encode("latin1", errors="ignore")
                try:
                    im = Image.open(io.BytesIO(data))
                    return _encode_png(im.convert("RGBA"))
                except Exception:
                    return bytes(data)
    return b""


def _provider_for(name: str) -> CatalogProvider:
    key = str(name or "disabled").strip().lower()
    if key in {"nanobanana", "nano_banana", "gemini_image", "gemini_image_edit"}:
        return CatalogProviderNanoBanana()
    if key == "flux_kontext":
        return HttpCatalogProvider("flux_kontext", "FLUX_KONTEXT_CATALOG_URL", "FLUX_KONTEXT_API_KEY")
    if key in {"vertex_imagen", "imagen_vertex"}:
        return CatalogProviderVertexImagen()
    if key == "imagen":
        return HttpCatalogProvider("imagen", "IMAGEN_CATALOG_URL", "IMAGEN_API_KEY")
    return DisabledCatalogProvider()


def _provider_key(name: str) -> str:
    return str(name or "").strip().lower()


def _provider_allows_quality_gate_cutout(name: str) -> bool:
    return _provider_key(name) in {"", "disabled", "none", "off", "false", "cutout"}


def _selected_provider_name(provider: Optional[str] = None) -> str:
    return (
        str(provider or "").strip()
        or os.getenv("WARDROBE_CATALOG_PROVIDER", "").strip()
        or os.getenv("CATALOG_PROVIDER", "").strip()
        or "nanobanana"
    )


def _legacy_fallback_provider_name(primary_provider: str) -> str:
    fallback = (
        os.getenv("WARDROBE_CATALOG_FALLBACK_PROVIDER", "").strip()
        or os.getenv("CATALOG_FALLBACK_PROVIDER", "").strip()
    )
    if fallback:
        return fallback
    legacy = os.getenv("CATALOG_PROVIDER", "").strip()
    if legacy and legacy.lower() not in {"nanobanana", "nano_banana"} and legacy.lower() != primary_provider.lower():
        return legacy
    return ""


def generate_catalog_png(
    cutout_bytes: bytes,
    *,
    item_metadata: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    fallback_to_cutout: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return a transparent catalog PNG and quality metadata.

    The returned bytes are always safe to upload when ``success`` is true.
    Provider generation is only attempted when deterministic quality is weak.
    """
    meta = dict(item_metadata or {})
    category = normalize_catalog_category(meta.get("category"))
    t0 = time.monotonic()
    if not category_allowed(category):
        return {"success": False, "status": "catalog_failed", "reason": "category_not_eligible"}

    fallback = _env_enabled("CATALOG_FALLBACK_TO_CUTOUT", "true") if fallback_to_cutout is None else bool(fallback_to_cutout)
    timeout = int(timeout_seconds or os.getenv("CATALOG_TIMEOUT_SECONDS", "30") or 30)
    provider_name = _selected_provider_name(provider)
    forced_reason = _forced_provider_reason(meta, category)
    if forced_reason:
        provider_name = _selected_provider_name(provider)
        if _provider_allows_quality_gate_cutout(provider_name):
            provider_name = "nanobanana"
        logger.info(
            "ahvi.catalog.force_provider reason=%s item_id=%s category=%s",
            forced_reason,
            meta.get("item_id"),
            category,
        )

    nb_from_raw = provider_name == "nanobanana" and str(
        os.getenv("CATALOG_NANOBANANA_FROM_RAW", "false")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if nb_from_raw:
        # Feed Nano Banana the clean ORIGINAL image (the caller supplies raw
        # bytes when this flag is on) instead of the RMBG-cutout-on-canvas, so
        # it renders a fresh product shot from scratch (matching manual Gemini
        # quality) rather than polishing a degraded cutout. Skip the
        # deterministic canvas + its quality gate so generation always runs.
        deterministic_bytes = cutout_bytes
        rotation, bounds = 0, None
        deterministic_validation = {
            "ok": False,
            "score": 0,
            "reason": "nanobanana_from_raw",
        }
        logger.info(
            "ahvi.catalog.nanobanana_from_raw item_id=%s category=%s raw_bytes=%s",
            meta.get("item_id"),
            category,
            len(cutout_bytes or b""),
        )
    else:
        canvas, rotation, bounds, reason = _transparent_catalog_canvas(cutout_bytes, category)
        if canvas is None:
            return {"success": False, "status": "catalog_failed", "reason": reason or "transparent_canvas_failed"}
        deterministic_bytes = _encode_png(canvas)
        deterministic_validation = validate_catalog_png(
            deterministic_bytes, original_bytes=cutout_bytes, item_metadata={**meta, "category": category}
        )
    unsafe_source_reason = _unsafe_source_reason(meta, deterministic_validation)
    # Allow masked-cutout fallback for clean apparel even if a downstream
    # provider validation flags a remnant, as long as there is NO explicit
    # human/mannequin/hanger evidence in the capture metadata. This stops the
    # colour-only false block (orange tee -> human_or_mannequin_remnants).
    unsafe_evidence = _explicit_unsafe_evidence(meta)
    bbox_area_ratio, is_full_frame_bbox = _bbox_area_ratio_and_full_frame(
        meta.get("bbox")
    )
    fallback_allowed = (not unsafe_evidence) and _is_apparel_category(category)
    fallback_allowed_reason = (
        "apparel_no_human_evidence"
        if fallback_allowed
        else ("explicit_human_evidence" if unsafe_evidence else "non_apparel_category")
    )
    if unsafe_source_reason:
        logger.warning(
            "ahvi.capture.catalog.unsafe_source_detected item_id=%s reason=%s",
            meta.get("item_id"),
            unsafe_source_reason,
        )
    logger.info(
        "ahvi.capture.catalog.source_risk item_id=%s bbox_area_ratio=%s is_full_frame_bbox=%s "
        "source_risk_reason=%s unsafe_source_evidence=%s fallback_allowed_reason=%s",
        meta.get("item_id"),
        bbox_area_ratio,
        is_full_frame_bbox,
        unsafe_source_reason or "none",
        unsafe_evidence or "none",
        fallback_allowed_reason,
    )
    logger.info(
        "ahvi.catalog_png.quality_gate item_id=%s category=%s score=%s ok=%s reason=%s",
        meta.get("item_id"),
        category,
        deterministic_validation.get("score"),
        deterministic_validation.get("ok"),
        deterministic_validation.get("reason"),
    )
    logger.info(
        "ahvi.capture.catalog.provider_select item_id=%s provider_env=%s legacy_provider_env=%s provider_selected=%s quality_gate_ok=%s fallback_cutout=%s",
        meta.get("item_id"),
        os.getenv("WARDROBE_CATALOG_PROVIDER", "").strip(),
        os.getenv("CATALOG_PROVIDER", "").strip(),
        provider_name,
        bool(deterministic_validation.get("ok")),
        fallback,
    )

    # Clean-cutout short-circuit is allowed ONLY for cutout/disabled providers.
    # When a real generator (nanobanana) is selected, every saveable garment —
    # including clean flat-lays — must go through generation; cutout is used only
    # as a fallback if generation fails (handled below).
    if (
        deterministic_validation.get("ok")
        and not forced_reason
        and not unsafe_source_reason
        and _provider_allows_quality_gate_cutout(provider_name)
    ):
        return {
            "success": True,
            "status": "catalog_ready",
            "catalog_png_bytes": deterministic_bytes,
            "catalog_provider": "cutout",
            "catalog_quality_score": int(deterministic_validation.get("score") or 0),
            "catalog_generation_version": CATALOG_GENERATION_VERSION,
            "catalog_generated_at": _now_iso(),
            "rotation_applied": rotation,
            "foreground_bounds": bounds,
            "validation": deterministic_validation,
            "reason": "",
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
        }

    provider_obj = _provider_for(provider_name)
    logger.info(
        "ahvi.capture.catalog.provider provider=%s item_id=%s validation_status=%s masked_png=%s",
        provider_obj.name,
        meta.get("item_id"),
        meta.get("validation_status"),
        bool(cutout_bytes),
    )
    catalog_prompt = _build_catalog_prompt(category, meta)
    if provider_obj.name == "vertex_imagen":
        logger.info(
            "ahvi.catalog.vertex.start item_id=%s category=%s reason=%s",
            meta.get("item_id"),
            category,
            forced_reason or deterministic_validation.get("reason") or "quality_gate_failed",
        )
        logger.info(
            "ahvi.catalog.vertex.prompt item_id=%s category=%s prompt=%s",
            meta.get("item_id"),
            category,
            catalog_prompt[:1200],
        )
    if provider_obj.name == "nanobanana":
        logger.info(
            "ahvi.capture.catalog.nanobanana.start item_id=%s category=%s reason=%s",
            meta.get("item_id"),
            category,
            forced_reason or deterministic_validation.get("reason") or "quality_gate_failed",
        )
    provider_result = provider_obj.generate(
        cutout_bytes=deterministic_bytes,
        prompt=catalog_prompt,
        item_metadata={**meta, "category": category},
        timeout=timeout,
    )
    fallback_used = False
    if (
        not provider_result.success
        and provider_obj.name == "nanobanana"
        and _legacy_fallback_provider_name(provider_obj.name)
    ):
        fallback_name = _legacy_fallback_provider_name(provider_obj.name)
        logger.warning(
            "ahvi.capture.catalog.nanobanana.failed item_id=%s err=%s fallback=%s",
            meta.get("item_id"),
            provider_result.reason,
            fallback_name,
        )
        fallback_provider = _provider_for(fallback_name)
        provider_result = fallback_provider.generate(
            cutout_bytes=deterministic_bytes,
            prompt=catalog_prompt,
            item_metadata={**meta, "category": category},
            timeout=timeout,
        )
        fallback_used = True
    if provider_result.success and provider_result.image_bytes:
        blank_reason = _catalog_blank_reason(provider_result.image_bytes, category)
        if blank_reason:
            logger.warning(
                "ahvi.capture.catalog.blank_image_detected item_id=%s provider=%s reason=%s",
                meta.get("item_id"),
                provider_result.provider,
                blank_reason,
            )
            provider_result = CatalogProviderResult(
                False,
                reason=blank_reason,
                provider=provider_result.provider or provider_obj.name,
            )
        else:
            black_metrics = _black_frame_metrics(provider_result.image_bytes)
            if black_metrics.get("detected"):
                logger.warning(
                    "ahvi.capture.catalog.black_frame_detected item_id=%s provider=%s frame_type=%s border_dark_ratio=%s center_dark_ratio=%s",
                    meta.get("item_id"),
                    provider_result.provider,
                    black_metrics.get("frame_type"),
                    black_metrics.get("border_dark_ratio"),
                    black_metrics.get("center_dark_ratio"),
                )
                cropped_bytes, cropped = _crop_black_frame(provider_result.image_bytes)
                if cropped and not _catalog_blank_reason(cropped_bytes, category):
                    provider_result = CatalogProviderResult(
                        True,
                        image_bytes=cropped_bytes,
                        provider=provider_result.provider,
                    )
                    logger.info(
                        "ahvi.capture.catalog.black_frame_cropped item_id=%s provider=%s frame_type=%s",
                        meta.get("item_id"),
                        provider_result.provider,
                        black_metrics.get("frame_type"),
                    )
                else:
                    if not cropped:
                        logger.warning(
                            "ahvi.capture.catalog.black_frame_crop_failed item_id=%s provider=%s frame_type=%s",
                            meta.get("item_id"),
                            provider_result.provider,
                            black_metrics.get("frame_type"),
                        )
                    logger.warning(
                        "ahvi.capture.catalog.black_frame_unresolved item_id=%s provider=%s frame_type=%s",
                        meta.get("item_id"),
                        provider_result.provider,
                        black_metrics.get("frame_type"),
                    )
                    provider_result = CatalogProviderResult(
                        False,
                        reason="black_frame_rejected",
                        provider=provider_result.provider or provider_obj.name,
                    )

    if provider_result.success and provider_result.image_bytes:
        # The provider returns an OPAQUE image (white studio background), and
        # convert("RGBA") does not remove it — that is what produced white boxes
        # in Wardrobe/Style Boards. Run the same RMBG service used everywhere
        # else, then recentre on the transparent catalogue canvas, BEFORE
        # validation so an opaque result can never be demo-accepted.
        transparent_bytes, _rmbg_reason = _provider_output_to_transparent(
            provider_result.image_bytes, category, meta.get("item_id")
        )
        if transparent_bytes:
            provider_result = CatalogProviderResult(
                success=True,
                image_bytes=transparent_bytes,
                provider=provider_result.provider,
            )
        else:
            provider_result = CatalogProviderResult(
                success=False,
                reason=_rmbg_reason or "rmbg_failed",
                provider=provider_result.provider,
            )

    if provider_result.success and provider_result.image_bytes:
        if provider_result.provider == "nanobanana":
            logger.info(
                "ahvi.capture.catalog.nanobanana.success item_id=%s category=%s",
                meta.get("item_id"),
                category,
            )
        if provider_result.provider == "vertex_imagen":
            logger.info(
                "ahvi.catalog.vertex.success item_id=%s category=%s",
                meta.get("item_id"),
                category,
            )
        generated_validation = validate_catalog_png(
            provider_result.image_bytes,
            original_bytes=cutout_bytes,
            item_metadata=_provider_validation_metadata(meta, category),
        )
        vertex_demo_accepted = (
            provider_result.provider in {"vertex_imagen", "nanobanana"}
            and not generated_validation.get("ok")
            and _vertex_demo_accepts_generated_validation(generated_validation)
        )
        if generated_validation.get("ok") or vertex_demo_accepted:
            return {
                "success": True,
                "status": "catalog_generated",
                "catalog_png_bytes": provider_result.image_bytes,
                "catalog_provider": provider_result.provider,
                "catalog_quality_score": int(generated_validation.get("score") or 0),
                "catalog_generation_version": CATALOG_GENERATION_VERSION,
                "catalog_generated_at": _now_iso(),
                "rotation_applied": rotation,
                "foreground_bounds": bounds,
                "validation": generated_validation,
                "reason": "demo_accept_background" if vertex_demo_accepted else "",
                "fallback_used": fallback_used,
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
            }
        if provider_result.provider == "vertex_imagen":
            logger.info(
                "ahvi.catalog.vertex.validation_failed score=%s reason=%s dimensions=%sx%s mode=%s",
                generated_validation.get("score"),
                generated_validation.get("reason"),
                generated_validation.get("image_width"),
                generated_validation.get("image_height"),
                generated_validation.get("image_mode"),
            )
        logger.info(
            "ahvi.catalog_png.provider_validation_failed provider=%s reason=%s",
            provider_result.provider,
            generated_validation.get("reason"),
        )

    if provider_obj.name == "nanobanana" and not fallback_used:
        logger.warning(
            "ahvi.capture.catalog.nanobanana.failed item_id=%s err=%s",
            meta.get("item_id"),
            provider_result.reason or "validation_failed",
        )
    if provider_obj.name == "vertex_imagen":
        logger.warning(
            "ahvi.catalog.vertex.failed item_id=%s category=%s err=%s",
            meta.get("item_id"),
            category,
            provider_result.reason or "validation_failed",
        )

    if unsafe_source_reason and provider_obj.name == "nanobanana" and fallback_allowed:
        logger.info(
            "ahvi.capture.catalog.unsafe_fallback_allowed item_id=%s source_risk=%s reason=%s",
            meta.get("item_id"),
            unsafe_source_reason,
            fallback_allowed_reason,
        )
    if (
        unsafe_source_reason
        and provider_obj.name == "nanobanana"
        and not fallback_allowed
    ):
        logger.warning(
            "ahvi.capture.catalog.unsafe_fallback_blocked item_id=%s reason=%s",
            meta.get("item_id"),
            provider_result.reason or "validation_failed",
        )
        return {
            "success": False,
            "status": "blocked_unsafe_fallback",
            "catalog_provider": "nanobanana",
            "catalog_quality_score": int(deterministic_validation.get("score") or 0),
            "catalog_generation_version": CATALOG_GENERATION_VERSION,
            "catalog_generated_at": _now_iso(),
            "rotation_applied": rotation,
            "foreground_bounds": bounds,
            "validation": deterministic_validation,
            "reason": "unsafe_source_nanobanana_failed",
            "provider_reason": provider_result.reason or "validation_failed",
            "unsafe_source_reason": unsafe_source_reason,
            "fallback_used": False,
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
        }

    if provider_obj.name == "nanobanana" and provider_result.reason in {
        "blank_transparent_catalog",
        "blank_flat_catalog",
        "tiny_accessory_catalog",
        "black_frame_rejected",
    }:
        logger.warning(
            "ahvi.capture.catalog.blank_fallback_blocked item_id=%s reason=%s",
            meta.get("item_id"),
            provider_result.reason,
        )
        return {
            "success": False,
            "status": "blocked_blank_catalog",
            "catalog_provider": "nanobanana",
            "catalog_quality_score": int(deterministic_validation.get("score") or 0),
            "catalog_generation_version": CATALOG_GENERATION_VERSION,
            "catalog_generated_at": _now_iso(),
            "rotation_applied": rotation,
            "foreground_bounds": bounds,
            "validation": deterministic_validation,
            "reason": provider_result.reason,
            "fallback_used": False,
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
        }

    if fallback:
        if provider_result.reason == "black_frame_rejected":
            logger.info(
                "ahvi.capture.catalog.black_frame_fallback_masked item_id=%s provider=%s",
                meta.get("item_id"),
                provider_result.provider or provider_obj.name,
            )
        logger.info(
            "ahvi.capture.catalog.fallback_cutout item_id=%s provider=%s used=true reason=%s",
            meta.get("item_id"),
            provider_result.provider or provider_obj.name,
            provider_result.reason or deterministic_validation.get("reason") or "quality_gate_failed",
        )
        return {
            "success": True,
            "status": "fallback_cutout",
            "catalog_png_bytes": deterministic_bytes,
            "catalog_provider": (
                "cutout" if provider_obj.name == "nanobanana" else provider_result.provider or "cutout"
            ),
            "catalog_quality_score": int(deterministic_validation.get("score") or 0),
            "catalog_generation_version": CATALOG_GENERATION_VERSION,
            "catalog_generated_at": _now_iso(),
            "rotation_applied": rotation,
            "foreground_bounds": bounds,
            "validation": deterministic_validation,
            "reason": provider_result.reason or deterministic_validation.get("reason") or "quality_gate_failed",
            "fallback_used": fallback_used,
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
        }

    return {
        "success": False,
        "status": "catalog_failed",
        "reason": provider_result.reason or deterministic_validation.get("reason") or "quality_gate_failed",
        "validation": deterministic_validation,
        "catalog_quality_score": int(deterministic_validation.get("score") or 0),
    }
