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

logger = logging.getLogger("ahvi.catalog_png")

CATALOG_GENERATION_VERSION = "catalog_png_v1"
CANVAS_SIZE = 1600
OBJECT_FILL = 0.86
QUALITY_READY_THRESHOLD = 82

CATALOG_PROMPT = """Preserve the garment exactly.

Remove all humans, body parts, mannequins, hangers and background.

Restore missing garment edges naturally.

Reduce wrinkles and folds while preserving garment structure.

Keep original:
- color
- fabric
- texture
- pattern
- silhouette

Generate a premium wardrobe garment asset.

Output must be:
- transparent PNG
- centered garment
- upright orientation
- realistic garment
- no model
- no props
- no accessories
- no background
- no text
- no watermark

The garment must remain visually identical to the original item."""


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
        " ".join(str(x) for x in (meta.get("tags") or []) if x),
    ]
    return " ".join(str(x or "") for x in parts).lower()


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
    remnant_signal = any(
        tok in blob
        for tok in ("human", "selfie", "mirror", "person", "body", "face", "hand", "mannequin", "hanger")
    )
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
    elif human_penalty >= 30 or remnant_signal:
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
    }


def validate_catalog_png(image_bytes: bytes, *, original_bytes: bytes = b"", item_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    score = score_catalog_quality(image_bytes, original_bytes=original_bytes, item_metadata=item_metadata)
    checks = {
        "no_face": score.get("skin_ratio", 0) < 0.25,
        "no_human": score.get("reason") != "human_or_mannequin_remnants",
        "no_mannequin": "mannequin" not in _text_blob(item_metadata),
        "no_hanger": "hanger" not in _text_blob(item_metadata),
        "category_matches_original": bool(normalize_catalog_category((item_metadata or {}).get("category"))),
        "color_matches_original": (score.get("color_distance") is None or score.get("color_distance", 0) <= 140),
        "orientation_upright": score.get("orientation_quality", 0) >= 70,
        "garment_centered": score.get("center_offset", 1) <= 0.16,
        "image_dimensions_valid": True,
    }
    try:
        img = Image.open(io.BytesIO(image_bytes))
        checks["image_dimensions_valid"] = img.size[0] >= 512 and img.size[1] >= 512 and img.mode in {"RGBA", "LA", "P"}
    except Exception:
        checks["image_dimensions_valid"] = False
    return {**score, "ok": bool(score.get("ok")) and all(checks.values()), "checks": checks}


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


def _provider_for(name: str) -> CatalogProvider:
    key = str(name or "disabled").strip().lower()
    if key == "flux_kontext":
        return HttpCatalogProvider("flux_kontext", "FLUX_KONTEXT_CATALOG_URL", "FLUX_KONTEXT_API_KEY")
    if key == "imagen":
        return HttpCatalogProvider("imagen", "IMAGEN_CATALOG_URL", "IMAGEN_API_KEY")
    return DisabledCatalogProvider()


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
    provider_name = provider or os.getenv("CATALOG_PROVIDER", "disabled")

    canvas, rotation, bounds, reason = _transparent_catalog_canvas(cutout_bytes, category)
    if canvas is None:
        return {"success": False, "status": "catalog_failed", "reason": reason or "transparent_canvas_failed"}
    deterministic_bytes = _encode_png(canvas)
    deterministic_validation = validate_catalog_png(
        deterministic_bytes, original_bytes=cutout_bytes, item_metadata={**meta, "category": category}
    )
    logger.info(
        "ahvi.catalog_png.quality_gate item_id=%s category=%s score=%s ok=%s reason=%s",
        meta.get("item_id"),
        category,
        deterministic_validation.get("score"),
        deterministic_validation.get("ok"),
        deterministic_validation.get("reason"),
    )

    if deterministic_validation.get("ok"):
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
    provider_result = provider_obj.generate(
        cutout_bytes=deterministic_bytes,
        prompt=CATALOG_PROMPT,
        item_metadata={**meta, "category": category},
        timeout=timeout,
    )
    if provider_result.success and provider_result.image_bytes:
        generated_validation = validate_catalog_png(
            provider_result.image_bytes,
            original_bytes=cutout_bytes,
            item_metadata={**meta, "category": category},
        )
        if generated_validation.get("ok"):
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
                "reason": "",
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
            }
        logger.info(
            "ahvi.catalog_png.provider_validation_failed provider=%s reason=%s",
            provider_result.provider,
            generated_validation.get("reason"),
        )

    if fallback:
        return {
            "success": True,
            "status": "fallback_cutout",
            "catalog_png_bytes": deterministic_bytes,
            "catalog_provider": provider_result.provider or "cutout",
            "catalog_quality_score": int(deterministic_validation.get("score") or 0),
            "catalog_generation_version": CATALOG_GENERATION_VERSION,
            "catalog_generated_at": _now_iso(),
            "rotation_applied": rotation,
            "foreground_bounds": bounds,
            "validation": deterministic_validation,
            "reason": provider_result.reason or deterministic_validation.get("reason") or "quality_gate_failed",
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
        }

    return {
        "success": False,
        "status": "catalog_failed",
        "reason": provider_result.reason or deterministic_validation.get("reason") or "quality_gate_failed",
        "validation": deterministic_validation,
        "catalog_quality_score": int(deterministic_validation.get("score") or 0),
    }
