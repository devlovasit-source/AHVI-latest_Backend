import asyncio
import base64
import io
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: str = "false") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}

from services import ai_gateway
from services.bg_service import remove_bg_bytes
from services.embedding_service import encode_metadata
from services.hybrid_detection_service import run_hybrid_detection
from services.image_embedding_service import encode_image_url
from services.image_fingerprint import compute_hash_from_base64, compute_hash_from_url
from services.qdrant_service import qdrant_service
from services.r2_storage import R2Storage
from services.wardrobe_persistence_service import (
    delete_wardrobe_item,
    persist_selected_items,
    update_item_labels,
    update_wardrobe_item_images,
)
from services.wardrobe_suitability import apply_metadata_guard
from services.wardrobe_taxonomy import (
    normalize_item as _taxonomy_normalize_item,
    build_review_card as _taxonomy_review_card,
    enforce_preview_taxonomy as _enforce_preview_taxonomy,
)
from prompts.core_prompts import WARDROBE_CAPTURE_PROMPT
from services import gemini_multi_garment_detector as _gemini_multi
from services.wardrobe_intelligence_service import enrich_wardrobe_item

router = APIRouter(prefix="/api/wardrobe/capture", tags=["wardrobe-capture"])
wardrobe_router = APIRouter(prefix="/api/wardrobe", tags=["wardrobe"])


class UpdateLabelsRequest(BaseModel):
    user_id: str
    item_id: str
    name: str | None = None
    category: str | None = None
    subcategory: str | None = None
    color: str | None = None
    material: str | None = None
    tags: List[str] | None = None
    # Client-supplied Appwrite location. When the client passes these,
    # backend uses them directly instead of guessing from env vars —
    # eliminates 'Update failed: Not Found' from env mismatch.
    collection_id: str | None = None
    database_id: str | None = None


@wardrobe_router.post("/update-labels")
def update_labels(http_request: Request, request: UpdateLabelsRequest):
    user_id = _effective_user_id(http_request, request.user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user_id")
    if not request.item_id.strip():
        raise HTTPException(status_code=400, detail="Missing item_id")

    try:
        result = update_item_labels(
            user_id=user_id,
            item_id=request.item_id.strip(),
            name=request.name,
            category=request.category,
            subcategory=request.subcategory,
            color=request.color,
            material=request.material,
            tags=request.tags,
            override_collection_id=request.collection_id,
            override_database_id=request.database_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.exception("update_labels_failed")
        raise HTTPException(status_code=502, detail=str(exc))

    logger.info(
        "ahvi.update_labels user_id=%s item_id=%s category=%s",
        user_id,
        request.item_id,
        request.category,
    )
    return result


@wardrobe_router.get("/diagnostics")
def wardrobe_diagnostics(http_request: Request):
    """Public health check — confirms backend sees the same Appwrite
    DB + collection ids the frontend uses. Compare values to your
    client's Env.appwrite* constants. Returns only public configuration,
    never API keys.
    """
    from services.wardrobe_persistence_service import (
        APPWRITE_ENDPOINT,
        APPWRITE_PROJECT_ID,
        APPWRITE_DATABASE_ID,
        APPWRITE_COLLECTION_ID,
        _KNOWN_COLLECTIONS,
        _appwrite_ready,
    )
    # Read user_id WITHOUT triggering the 401 raise.
    user_id_optional = ""
    try:
        user_id_optional = _request_user_id(http_request)
    except Exception:
        user_id_optional = ""
    return {
        "appwrite_ready": _appwrite_ready(),
        "endpoint": APPWRITE_ENDPOINT,
        "project_id": APPWRITE_PROJECT_ID,
        "database_id": APPWRITE_DATABASE_ID,
        "primary_collection_id": APPWRITE_COLLECTION_ID,
        "all_known_collection_ids": list(_KNOWN_COLLECTIONS),
        "user_id": user_id_optional,
    }


@wardrobe_router.post("/rmbg-warm")
async def rmbg_warm(http_request: Request):
    """Keep-warm ping for the remote RMBG (GCE) model. Whitelisted from JWT in
    main.py auth_guard; gated by RMBG_WARM_SECRET when set. Meant to be called
    by Cloud Scheduler every few minutes so the model never goes cold (a cold
    model adds ~37s to the first save, blowing the 30s target)."""
    secret = os.getenv("RMBG_WARM_SECRET", "").strip()
    if secret and http_request.headers.get("x-warm-secret", "") != secret:
        raise HTTPException(status_code=403, detail="forbidden")
    from services.bg_service import warm_rmbg

    return await warm_rmbg()


@wardrobe_router.delete("/{item_id}")
def delete_wardrobe_item_route(item_id: str, http_request: Request):
    log = logging.getLogger("ahvi.wardrobe.delete")
    user_id = _request_user_id(http_request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    clean_item_id = str(item_id or "").strip()
    if not clean_item_id:
        raise HTTPException(status_code=400, detail="Missing item_id")

    log.info(
        "ahvi.wardrobe.delete.started user_id=%s item_id=%s",
        user_id,
        clean_item_id,
    )

    images_deleted = False
    try:
        result = delete_wardrobe_item(user_id=user_id, item_id=clean_item_id)
        r2_result = _ahvi_delete_r2_images_for_item(result.get("item") or {})
        images_deleted = any(
            bool(r2_result.get(key))
            for key in ("raw_deleted", "masked_deleted", "normalized_deleted")
        )
        log.info(
            "ahvi.wardrobe.delete.images_deleted user_id=%s item_id=%s deleted=%s status=%s",
            user_id,
            clean_item_id,
            images_deleted,
            r2_result.get("status"),
        )
        return {
            "success": True,
            "deleted_item_id": clean_item_id,
            "metadata_deleted": bool(result.get("metadata_deleted")),
            "images_deleted": images_deleted,
        }
    except LookupError as exc:
        log.warning(
            "ahvi.wardrobe.delete.failed user_id=%s item_id=%s err=%s",
            user_id,
            clean_item_id,
            exc,
        )
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        log.warning(
            "ahvi.wardrobe.delete.failed user_id=%s item_id=%s err=%s",
            user_id,
            clean_item_id,
            exc,
        )
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        log.warning(
            "ahvi.wardrobe.delete.failed user_id=%s item_id=%s err=%s",
            user_id,
            clean_item_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        log.exception(
            "ahvi.wardrobe.delete.failed user_id=%s item_id=%s",
            user_id,
            clean_item_id,
        )
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        log.exception(
            "ahvi.wardrobe.delete.failed user_id=%s item_id=%s",
            user_id,
            clean_item_id,
        )
        raise HTTPException(status_code=500, detail=str(exc))


class CaptureAnalyzeRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., min_length=20)
    auto_save: bool = False
    save_duplicates: bool = False


class SaveSelectedRequest(BaseModel):
    user_id: str
    selected_item_ids: List[str]
    detected_items: List[Dict[str, Any]]


class CaptureAnalyzeBatchRequest(BaseModel):
    user_id: str
    image_base64s: List[str] = Field(default_factory=list)
    auto_save: bool = False
    save_duplicates: bool = False


class DeleteSelectedRequest(BaseModel):
    user_id: str
    item_ids: List[str] = Field(default_factory=list)
    items: List[Dict[str, Any]] = Field(default_factory=list)
    delete_r2: bool = True


def _request_user_id(http_request: Request) -> str:
    user = getattr(http_request.state, "user", None)
    if not isinstance(user, dict):
        return ""
    return str(user.get("user_id") or user.get("$id") or user.get("id") or "").strip()


def _effective_user_id(http_request: Request, supplied_user_id: str) -> str:
    authed_user_id = _request_user_id(http_request)
    if not authed_user_id:
        # Auth is mandatory now; the bypass in main.py was removed.
        raise HTTPException(status_code=401, detail="Authentication required")
    supplied = str(supplied_user_id or "").strip()
    if supplied and supplied != authed_user_id:
        raise HTTPException(
            status_code=403, detail="user_id does not match authenticated user"
        )
    return authed_user_id


def _decode_image_base64(value: str) -> Image.Image:
    text = (value or "").strip()
    if "," in text:
        text = text.split(",", 1)[1]

    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}")

    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 15MB)")

    try:
        opened = Image.open(io.BytesIO(data))
        # Normalize EXIF orientation BEFORE anything downstream (Gemini bbox,
        # crop, RMBG, catalog) sees the image. Camera photos carry an EXIF
        # orientation tag with sideways-stored pixels; without this the crop is
        # re-encoded rotated and cannot be recovered later. exif_transpose is a
        # no-op when there is no orientation tag (idempotent).
        had_orientation = False
        try:
            exif = opened.getexif()
            had_orientation = bool(exif) and 0x0112 in exif and int(exif.get(0x0112) or 1) != 1
        except Exception:  # noqa: BLE001 — orientation probe must never break decode.
            had_orientation = False
        transposed = ImageOps.exif_transpose(opened)
        if had_orientation:
            logger.info(
                "ahvi.image.orientation.applied src=%dx%d out=%dx%d",
                opened.size[0],
                opened.size[1],
                transposed.size[0],
                transposed.size[1],
            )
        else:
            logger.info("ahvi.image.orientation.skipped size=%dx%d", opened.size[0], opened.size[1])
        return transposed.convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image bytes: {exc}")


def _bytes_from_image_base64(value: str) -> bytes:
    text = (value or "").strip()
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}")


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    rgb.save(buf, format="PNG")
    return buf.getvalue()


from services.category_taxonomy import (
    CANONICAL_CATEGORIES as _CANONICAL_CATEGORIES,
    CANONICAL_CATEGORY_KEYWORDS as _CANONICAL_CATEGORY_KEYWORDS,
    infer_style_attributes as _infer_style_attributes,
    normalize_category_from_label as _shared_normalize_category_from_label,
)


def _normalize_category_from_label(label: str) -> tuple[str, str]:
    # Delegated to shared taxonomy. Kept as a thin wrapper because the symbol
    # is referenced inside _guardrail_category below.
    return _shared_normalize_category_from_label(label)


def _guardrail_category(
    *,
    raw_label: str,
    vision_name: str,
    vision_category: str,
    vision_sub_category: str,
    fallback_category: str,
    fallback_sub_category: str,
) -> tuple[str, str, bool]:
    """Keep vision useful, but never let it misclassify obvious garments."""
    primary_text = " ".join(
        str(v or "").lower()
        for v in [raw_label, vision_name, vision_sub_category, fallback_sub_category]
    )
    # Ethnic menswear Gemini/Ollama often misfiles as shirt/jacket/"royal
    # attire". Pin the unambiguous ones to Outerwear (a canonical category).
    # saree/lehenga/kurta intentionally excluded — handled elsewhere.
    _ETHNIC_OUTERWEAR = {
        "sherwani": "Sherwani",
        "bandhgala": "Bandhgala",
        "nehru": "Nehru Jacket",
    }
    _incoming_cat = str(vision_category or fallback_category or "").strip().title()
    for _key, _sub in _ETHNIC_OUTERWEAR.items():
        if _key in primary_text:
            return "Outerwear", _sub, _incoming_cat != "Outerwear"

    category_text = str(vision_category or fallback_category or "").lower()
    category = str(vision_category or fallback_category or "Item").strip().title()
    sub_category = str(
        vision_sub_category or fallback_sub_category or category or "Item"
    ).strip()
    corrected = False

    matched = False
    for canonical, default_sub, keywords in _CANONICAL_CATEGORY_KEYWORDS:
        if any(keyword in primary_text for keyword in keywords):
            if category != canonical:
                corrected = True
            category = canonical
            if not sub_category or sub_category.lower() in {
                "item",
                "unknown",
                "none",
                "accessory",
                "accessories",
                "top",
                "tops",
            }:
                match = next((kw for kw in keywords if kw in primary_text), default_sub)
                sub_category = match.replace("tshirt", "t-shirt").title()
            matched = True
            break

    if not matched:
        for canonical, default_sub, keywords in _CANONICAL_CATEGORY_KEYWORDS:
            if category == canonical or any(
                keyword in category_text for keyword in keywords
            ):
                if category != canonical:
                    corrected = True
                category = canonical
                if not sub_category or sub_category.lower() in {
                    "item",
                    "unknown",
                    "none",
                }:
                    sub_category = default_sub
                break

    if category not in _CANONICAL_CATEGORIES and category != "Item":
        normalized_category, normalized_sub = _normalize_category_from_label(
            category or sub_category or raw_label
        )
        corrected = corrected or normalized_category != category
        category = normalized_category
        if not sub_category or sub_category.lower() in {"item", "unknown", "none"}:
            sub_category = normalized_sub

    return category, sub_category, corrected


def _hex_to_name(color_hex: str) -> str:
    named = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "red": (220, 20, 60),
        "blue": (30, 90, 200),
        "green": (34, 139, 34),
        "yellow": (240, 200, 40),
        "brown": (120, 80, 55),
        "beige": (220, 200, 170),
        "pink": (230, 130, 170),
        "purple": (130, 70, 170),
        "orange": (230, 140, 40),
        "navy": (20, 35, 90),
    }
    try:
        h = str(color_hex or "#000000").strip().lstrip("#")
        if len(h) != 6:
            return "unknown"
        rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return "unknown"

    best_name = "unknown"
    best_dist = 10**9
    for name, ref in named.items():
        dist = (rgb[0] - ref[0]) ** 2 + (rgb[1] - ref[1]) ** 2 + (rgb[2] - ref[2]) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _embeddings_enabled() -> bool:
    return str(os.getenv("ENABLE_IMAGE_EMBEDDINGS", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _duplicate_result(
    *,
    checked: bool,
    is_duplicate: bool,
    reason: str | None = None,
    confidence: float = 0.0,
    matched_item_id: Any = None,
    distance: Any = None,
    score: Any = None,
    payload: Any = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "checked": bool(checked),
        "is_duplicate": bool(is_duplicate),
        "reason": reason,
        "confidence": float(confidence or 0.0),
        "matched_item_id": str(matched_item_id or "") or None,
    }
    if distance is not None:
        result["distance"] = distance
    if score is not None:
        result["score"] = float(score or 0.0)
    if isinstance(payload, dict):
        result["payload"] = payload
    return result


def _same_metadata_family(item: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    item_category = str(item.get("category") or "").strip().lower()
    item_sub = str(item.get("sub_category") or item.get("type") or "").strip().lower()
    item_color = str(item.get("color_code") or item.get("color") or "").strip().lower()
    payload_category = str(payload.get("category") or "").strip().lower()
    payload_type = str(payload.get("type") or payload.get("sub_category") or "").strip().lower()
    payload_color = str(payload.get("color") or payload.get("color_code") or "").strip().lower()
    category_ok = bool(item_category and payload_category and item_category == payload_category)
    sub_ok = bool(item_sub and payload_type and item_sub == payload_type)
    color_ok = bool(item_color and payload_color and item_color == payload_color)
    return category_ok and (sub_ok or color_ok)


def _find_upload_duplicate(
    *,
    user_id: str,
    item: Dict[str, Any],
    pixel_hash: str,
    image_embedding: List[float],
) -> Dict[str, Any]:
    checked_any = False

    if pixel_hash:
        duplicate = qdrant_service.find_pixel_duplicate(
            user_id, pixel_hash, max_distance=6
        )
        checked_any = checked_any or bool(duplicate.get("checked"))
        if duplicate.get("is_duplicate"):
            return _duplicate_result(
                checked=True,
                is_duplicate=True,
                reason="pixel_hash",
                confidence=float(duplicate.get("confidence") or 1.0),
                matched_item_id=duplicate.get("matched_item_id") or duplicate.get("id"),
                distance=duplicate.get("distance"),
                payload=duplicate.get("payload"),
            )

    if image_embedding:
        duplicate = qdrant_service.find_image_duplicate(
            image_embedding, user_id, threshold=0.985
        )
        checked_any = checked_any or bool(duplicate.get("checked"))
        if duplicate.get("is_duplicate"):
            return _duplicate_result(
                checked=True,
                is_duplicate=True,
                reason="image_vector",
                confidence=float(duplicate.get("confidence") or duplicate.get("score") or 0.0),
                matched_item_id=duplicate.get("matched_item_id") or duplicate.get("id"),
                score=duplicate.get("score"),
                payload=duplicate.get("payload"),
            )

    metadata_vector = encode_metadata(
        {
            "category": item.get("category"),
            "sub_category": item.get("sub_category"),
            "color_code": item.get("color_code"),
            "pattern": item.get("pattern"),
            "occasions": item.get("occasions") or [],
        }
    )
    if metadata_vector:
        duplicate = qdrant_service.find_duplicate(
            metadata_vector, user_id, threshold=0.995
        )
        checked_any = checked_any or bool(duplicate.get("checked"))
        payload = duplicate.get("payload") if isinstance(duplicate.get("payload"), dict) else {}
        if duplicate.get("is_duplicate") and _same_metadata_family(item, payload):
            score = float(duplicate.get("score") or 0.0)
            return _duplicate_result(
                checked=True,
                is_duplicate=True,
                reason="metadata",
                confidence=min(score, 0.75),
                matched_item_id=duplicate.get("matched_item_id") or duplicate.get("id"),
                score=score,
                payload=payload,
            )

    return _duplicate_result(checked=checked_any, is_duplicate=False)


def _dominant_color_hex_from_image(image: Image.Image) -> str:
    try:
        img = image.convert("RGB").resize((64, 64))
        pixels = list(img.getdata())
        if not pixels:
            return "#000000"
        r = int(sum(p[0] for p in pixels) / len(pixels))
        g = int(sum(p[1] for p in pixels) / len(pixels))
        b = int(sum(p[2] for p in pixels) / len(pixels))
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "#000000"


def _dominant_color_hex_from_url(url: str) -> str:
    try:
        from services.auth_helpers import is_safe_outbound_url

        clean_url = str(url or "").strip()
        if not clean_url or not is_safe_outbound_url(clean_url):
            return "#000000"
        response = requests.get(clean_url, timeout=(2, 8))
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB").resize((64, 64))
        pixels = list(img.getdata())
        if not pixels:
            return "#000000"
        r = int(sum(p[0] for p in pixels) / len(pixels))
        g = int(sum(p[1] for p in pixels) / len(pixels))
        b = int(sum(p[2] for p in pixels) / len(pixels))
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "#000000"


def _clean_label_text(value: Any, default: str, max_len: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return text[:max_len]


def _normalize_occasions(value: Any) -> List[str]:
    if isinstance(value, str):
        raw = [p.strip() for p in value.split(",")]
    elif isinstance(value, list):
        raw = [str(p).strip() for p in value]
    else:
        raw = []
    occasions = []
    blocked = {"unknown", "none", "n/a", "na", "null", "undefined"}
    for item in raw:
        key = item.lower()
        if key and key not in blocked and key not in occasions:
            occasions.append(key)
    return occasions[:8]


def _extract_first_vision_item(ai_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ai_json, dict):
        return {}
    items = ai_json.get("items")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return dict(items[0])
    return ai_json


def _vision_says_item_is_known(label_source: str, category: str, name: str) -> bool:
    """Return true when vision produced enough signal to skip manual review."""
    if not str(label_source or "").startswith("vision"):
        return False
    category_key = str(category or "").strip().lower()
    name_key = str(name or "").strip().lower()
    if category_key in {"", "item", "unknown", "needs review", "needs_review", "review"}:
        return False
    if name_key in {
        "",
        "item",
        "unknown",
        "review item",
        "reviewed item",
        "needs review",
        "needs_review",
    }:
        return False
    return True


def _best_effort_item_name(item: Dict[str, Any], original: Dict[str, Any] | None = None) -> str:
    original = original or {}
    raw_name = str(item.get("name") or item.get("label") or "").strip()
    if raw_name.lower() not in {
        "",
        "item",
        "unknown",
        "review item",
        "reviewed item",
        "needs review",
        "needs_review",
    }:
        return raw_name.title()
    color = str(item.get("color_name") or "").strip()
    sub = str(
        original.get("sub_category")
        or original.get("subcategory")
        or item.get("sub_category")
        or item.get("subcategory")
        or ""
    ).strip()
    category = str(item.get("category") or "").strip()
    noun = sub if sub.lower() not in {"", "item", "unknown", "needs review"} else category
    if noun.lower() in {"", "item", "unknown", "needs review", "needs_review"}:
        noun = "Wardrobe Item"
    label = " ".join(part for part in [color, noun] if part).strip()
    return label.title() or "Wardrobe Item"


_HEADWEAR_CAP_TERMS = (
    "baseball cap",
    "dad cap",
    "snapback",
    "visor",
    "cap",
    "hat",
)
_HEADWEAR_BEANIE_TERMS = ("beanie",)
_LOGO_BRANDS = {
    "adidas": "Adidas",
    "google": "Google",
    "ibm": "IBM",
    "nike": "Nike",
    "watsonx": "Watsonx",
}
_WRONG_LOGO_GARMENT_TERMS = (
    "sock",
    "socks",
    "top",
    "shirt",
    "tee",
    "t-shirt",
    "tshirt",
)


def _text_has_token(text: str, *tokens: str) -> bool:
    clean = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", clean)
        for token in tokens
    )


def _known_logo_brand(text: str) -> str:
    clean = f" {re.sub(r'[^a-z0-9]+', ' ', str(text or '').lower())} "
    for key, label in _LOGO_BRANDS.items():
        if f" {key} " in clean:
            return label
    return ""


def _headwear_subcategory_from_text(text: str) -> str:
    if _text_has_token(text, *_HEADWEAR_BEANIE_TERMS):
        return "Beanie"
    if _text_has_token(text, *_HEADWEAR_CAP_TERMS):
        return "Cap"
    return ""


def _clean_headwear_label(item: Dict[str, Any], sub_category: str) -> str:
    color = str(item.get("color_name") or item.get("color") or "").strip().title()
    if color.lower() in {"", "unknown", "none", "null"}:
        color = ""
    if sub_category == "Beanie":
        return f"{color} Beanie".strip() or "Beanie"
    if color:
        return f"{color} Baseball Cap"
    brand = _known_logo_brand(str(item.get("name") or item.get("label") or ""))
    return f"{brand} Cap".strip() if brand else "Baseball Cap"


def _apply_headwear_ocr_guard(
    item: Dict[str, Any],
    *,
    context_text: str = "",
    reason_prefix: str = "cap_ocr_guard",
) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return item
    out = dict(item)
    before = {
        "name": out.get("name"),
        "category": out.get("category"),
        "sub_category": out.get("sub_category") or out.get("subcategory"),
    }
    blob = " ".join(
        str(v or "")
        for v in (
            context_text,
            out.get("name"),
            out.get("label"),
            out.get("title"),
            out.get("category"),
            out.get("sub_category"),
            out.get("subcategory"),
            out.get("description"),
            out.get("reasoning"),
        )
    )
    sub = _headwear_subcategory_from_text(blob)
    if sub:
        out["category"] = "Accessories"
        out["sub_category"] = sub
        out["subcategory"] = sub
        out["subCategory"] = sub
        if _known_logo_brand(str(out.get("name") or "")) and _text_has_token(
            str(out.get("name") or ""), *_WRONG_LOGO_GARMENT_TERMS
        ):
            out["name"] = _clean_headwear_label(out, sub)
        elif not str(out.get("name") or "").strip() or _text_has_token(
            str(out.get("name") or ""), "item", "unknown", "review item"
        ):
            out["name"] = _clean_headwear_label(out, sub)
        out["privateWear"] = False
        out["publicWear"] = True
        out["styleEligible"] = True
        after = {
            "name": out.get("name"),
            "category": out.get("category"),
            "sub_category": out.get("sub_category"),
        }
        if after != before:
            logger.info(
                "ahvi.capture.taxonomy_corrected from=%s to=%s reason=%s",
                before,
                after,
                reason_prefix,
            )
        return out

    category_key = str(out.get("category") or "").strip().lower()
    sub_key = str(out.get("sub_category") or out.get("subcategory") or "").strip().lower()
    name = str(out.get("name") or out.get("label") or "")
    try:
        confidence = float(out.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    if (
        _known_logo_brand(name)
        and _text_has_token(name, *_WRONG_LOGO_GARMENT_TERMS)
        and (category_key in {"tops", "top", ""} or sub_key in {"top", "tops", ""})
        and confidence < 0.75
    ):
        out["category"] = "Needs Review"
        out["sub_category"] = "Needs Review"
        out["subcategory"] = "Needs Review"
        out["subCategory"] = "Needs Review"
        out["requires_manual_entry"] = True
        out["needs_review"] = True
        logger.info(
            "ahvi.capture.taxonomy_corrected from=%s to=%s reason=logo_only_weak_top",
            before,
            {
                "name": out.get("name"),
                "category": out.get("category"),
                "sub_category": out.get("sub_category"),
            },
        )
    return out


def _apply_full_image_person_risk_guard(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return item
    out = dict(item)
    crop_source = str(out.get("crop_source") or "").strip().lower()
    category = str(out.get("category") or "").strip().lower()
    sub = str(out.get("sub_category") or out.get("subcategory") or "").strip().lower()
    name = str(out.get("name") or out.get("label") or "").strip().lower()
    bottom_signal = (
        category in {"bottoms", "bottom"}
        or any(token in sub for token in ("short", "pant", "trouser", "jean"))
        or any(token in name for token in ("short", "pant", "trouser", "jean"))
    )
    if crop_source == "full_image_fallback" and bottom_signal:
        out["crop_quality"] = "full_image_person_risk"
        out["needs_review"] = True
        out["requires_manual_entry"] = True
        out["review_reason"] = "Needs cleaner photo"
        logger.info(
            "ahvi.capture.full_image_person_risk item_id=%s category=%s",
            out.get("item_id"),
            out.get("category"),
        )
    return out


_ACCESSORY_REVIEW_TERMS = {
    "necklace",
    "bracelet",
    "ring",
    "earring",
    "earrings",
    "watch",
}
_BOTTOM_REVIEW_TERMS = {
    "trouser",
    "trousers",
    "pant",
    "pants",
    "short",
    "shorts",
    "jean",
    "jeans",
    "skirt",
    "skirts",
}


_SOURCE_PERSON_TERMS = {
    "person",
    "human",
    "body",
    "full_body",
    "full body",
    "selfie",
    "mirror",
    "model",
    "mannequin",
    "worn",
    "wearing",
    "torso",
    "arm",
    "leg",
    "feet",
    "face",
    "hand",
}
_SCREENSHOT_COLLAGE_TERMS = {
    "screenshot",
    "style collage",
    "social",
    "pinterest",
    "instagram",
    "save button",
    "remix",
    "like",
    "share",
    "comment",
    "watermark",
    "app ui",
    "interface",
    "editorial board",
    "inspiration board",
}
_WOMENSWEAR_STRONG_MISMATCH_TERMS = {
    "saree",
    "sari",
    "lehenga",
    "anarkali",
    "sundress",
    "dress",
    "gown",
    "skirt",
    "ethnic blouse",
    "saree blouse",
    "women's blouse",
    "womens blouse",
    "heel",
    "heels",
    "bangle",
    "bangles",
}
_FEMININE_ETHNIC_JEWELRY_TERMS = {
    "earring",
    "earrings",
    "necklace",
    "choker",
    "bangle",
    "bangles",
}
_UNISEX_ALLOW_TERMS = {
    "shirt",
    "t-shirt",
    "tshirt",
    "tee",
    "jeans",
    "trousers",
    "pants",
    "sneaker",
    "sneakers",
    "sandal",
    "sandals",
    "jacket",
    "cap",
    "sunglasses",
    "watch",
}


def _item_text_blob(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    parts = []
    for key in (
        "name",
        "label",
        "category",
        "sub_category",
        "subcategory",
        "description",
        "reasoning",
        "review_reason",
        "rejection_reason",
        "source",
        "input_type",
        "label_source",
        "crop_source",
        "crop_quality",
        "detected_text",
        "ocr_text",
        "logo_text",
        "tags",
    ):
        value = item.get(key)
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(v or "") for v in value)
        elif isinstance(value, dict):
            parts.extend(str(v or "") for v in value.values())
        else:
            parts.append(str(value or ""))
    return " ".join(parts).lower()


_SOURCE_PERSON_PHRASES = {
    "full body",
    "full_body",
    "mirror selfie",
    "group selfie",
    "group photo",
    "person body crop",
    "human body",
    "body crop",
    "body remnants",
    "human_or_mannequin_remnants",
}
_SOURCE_PERSON_WORDS = {
    "person",
    "human",
    "selfie",
    "mirror",
    "model",
    "mannequin",
    "worn",
    "wearing",
    "torso",
    "arm",
    "arms",
    "leg",
    "legs",
    "feet",
    "face",
    "hand",
    "hands",
}


def _source_person_reason(item: Dict[str, Any]) -> str:
    blob = _item_text_blob(item)
    for phrase in _SOURCE_PERSON_PHRASES:
        if phrase in blob:
            return phrase
    tokens = set(re.findall(r"[a-z0-9]+", blob))
    for word in _SOURCE_PERSON_WORDS:
        if word in tokens:
            return word
    return ""


def _source_contains_person_item(item: Dict[str, Any]) -> bool:
    return bool(_source_person_reason(item))


def _source_contains_person_batch(items: List[Dict[str, Any]]) -> bool:
    if any(_source_contains_person_item(i) for i in items if isinstance(i, dict)):
        return True
    cats = {
        str(i.get("category") or "").strip().lower()
        for i in items
        if isinstance(i, dict)
    }
    names = " ".join(str(i.get("name") or i.get("label") or "") for i in items if isinstance(i, dict)).lower()
    has_top = bool(cats & {"tops", "top", "outerwear"})
    has_bottom = bool(cats & {"bottoms", "bottom"})
    has_footwear = "footwear" in cats
    # A top+bottom+shoe set from capture is most often a worn full-body photo.
    # Keep this as a source risk, not a label correction.
    return (has_top and has_bottom and has_footwear) or "wearing" in names


def _is_screenshot_or_style_collage(item: Dict[str, Any]) -> bool:
    blob = _item_text_blob(item)
    return any(term in blob for term in _SCREENSHOT_COLLAGE_TERMS)


def _is_strong_male_profile_mismatch(item: Dict[str, Any]) -> bool:
    blob = _item_text_blob(item)
    if any(term in blob for term in _UNISEX_ALLOW_TERMS) and not any(
        term in blob for term in _WOMENSWEAR_STRONG_MISMATCH_TERMS
    ):
        return False
    if any(term in blob for term in _WOMENSWEAR_STRONG_MISMATCH_TERMS):
        return True
    ethnic_context = any(term in blob for term in ("saree", "lehenga", "anarkali", "ethnic", "traditional"))
    return ethnic_context and any(term in blob for term in _FEMININE_ETHNIC_JEWELRY_TERMS)


def _apply_preview_rejection(
    item: Dict[str, Any],
    *,
    reason: str,
    mismatch: bool = False,
) -> Dict[str, Any]:
    out = dict(item)
    out["validation_status"] = "rejected"
    out["needs_review"] = True
    out["requires_manual_entry"] = True
    out["selected_by_default"] = False
    out["rejection_reason"] = reason
    out["review_reason"] = reason
    if mismatch:
        out["wardrobe_profile_mismatch"] = True
        out["needs_wearer_confirmation"] = True
        out["mismatch_reason"] = "outside_current_wardrobe_profile"
    return out


def _apply_capture_source_and_profile_safety(
    items: List[Dict[str, Any]],
    *,
    user_id: str = "",
) -> List[Dict[str, Any]]:
    if not items:
        return items
    source_has_person = _source_contains_person_batch(items)
    gender = _fetch_wardrobe_profile_gender(user_id) if user_id else "unknown"
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cur = dict(item)
        if source_has_person or _source_contains_person_item(cur):
            cur["source_contains_person"] = True
            cur["unsafe_source"] = True
        if _is_screenshot_or_style_collage(cur):
            logger.warning(
                "ahvi.capture.preview.rejected_screenshot_collage item_id=%s",
                cur.get("item_id"),
            )
            cur = _apply_preview_rejection(cur, reason="screenshot_or_style_collage")
        if gender == "male" and _is_strong_male_profile_mismatch(cur):
            logger.warning(
                "ahvi.capture.preview.profile_mismatch item_id=%s gender=%s name=%s category=%s",
                cur.get("item_id"),
                gender,
                cur.get("name"),
                cur.get("category"),
            )
            cur = _apply_preview_rejection(
                cur,
                reason="outside_current_wardrobe_profile",
                mismatch=True,
            )
        out.append(cur)
    return out


def _bbox_area_ratio(bbox: Any) -> float | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width <= 0 or height <= 0:
        return 0.0
    # Normalized Gemini bboxes are 0..1. Pixel bboxes need only a rough
    # small-crop guard; screenshots vary wildly in size.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        return width * height
    return min((width * height) / (1200.0 * 1600.0), 1.0)


def _is_accessory_review_item(item: Dict[str, Any]) -> bool:
    text = " ".join(
        str(item.get(k) or "")
        for k in ("name", "label", "category", "sub_category", "subcategory")
    ).lower()
    return any(term in text for term in _ACCESSORY_REVIEW_TERMS)


def _is_bottom_review_item(item: Dict[str, Any]) -> bool:
    text = " ".join(
        str(item.get(k) or "")
        for k in ("name", "label", "category", "sub_category", "subcategory")
    ).lower()
    category = str(item.get("category") or "").strip().lower()
    return category in {"bottom", "bottoms"} or any(term in text for term in _BOTTOM_REVIEW_TERMS)


# Safe single-garment fallback approval (P0). When Gemini multi returns no
# valid items, the backend full-image fallback produces exactly one item with
# crop_source == "full_image_fallback". Approve it ONLY if it is a safe public
# fashion garment — never accessories, private wear, skin/body/person false
# positives, or non-fashion objects.
_FALLBACK_BLOCKED_TERMS = {
    "necklace", "bracelet", "ring", "watch", "earring", "earrings",
    "bag", "handbag", "skin", "body", "face", "hair", "person",
    "underwear", "brief", "briefs", "boxer", "boxers", "lingerie",
    "nightwear", "pajama", "pyjama", "charger", "bottle", "cable",
    "phone", "adapter",
}
_FALLBACK_SAFE_CATEGORIES = {
    "dresses", "dress", "tops", "top", "outerwear",
    "traditional", "indian wear", "ethnic wear",
}
_FALLBACK_SAFE_TERMS = {
    "saree", "sari", "kurti", "kurta", "dress", "one-piece", "one piece",
    "shirt", "t-shirt", "tshirt", "tee", "blazer", "jacket", "blouse",
    "top", "gown", "lehenga", "anarkali",
}
_FALLBACK_SAFE_BOTTOM_TERMS = {"trouser", "trousers", "pant", "pants", "jean", "jeans"}
_FALLBACK_BOTTOM_MIN_CONFIDENCE = 0.80


def _is_safe_public_fallback_garment(
    item: Dict[str, Any], confidence: float, crop_quality: str
) -> bool:
    """True if a full_image_fallback item is a safe public fashion garment.

    Conservative by design: blocked terms, private wear, and person/skin risk
    all veto approval. Bottoms approved only at high confidence.
    """
    if not isinstance(item, dict):
        return False
    category = str(item.get("category") or "").strip().lower()
    sub = str(item.get("sub_category") or item.get("subcategory") or "").strip().lower()
    name = str(item.get("name") or item.get("label") or "").strip().lower()
    blob = " ".join((name, category, sub))

    # Hard blocks: non-fashion, accessories, skin/body/person, private wear.
    if any(term in blob for term in _FALLBACK_BLOCKED_TERMS):
        return False
    if bool(item.get("privateWear")) or bool(item.get("private_wear")):
        return False
    if str(item.get("publicWear")).strip().lower() == "false":
        return False
    # Person/skin risk already flagged upstream by the full_image person guard.
    if crop_quality in {"full_image_person_risk", "broad"}:
        return False

    # Safe public tops/dresses/outerwear/traditional.
    if category in _FALLBACK_SAFE_CATEGORIES:
        return True
    if any(term in blob for term in _FALLBACK_SAFE_TERMS):
        return True

    # Bottoms: allowed only at high confidence and no person/skin risk.
    is_bottom = category in {"bottoms", "bottom"} or any(
        term in blob for term in _FALLBACK_SAFE_BOTTOM_TERMS
    )
    if is_bottom and confidence >= _FALLBACK_BOTTOM_MIN_CONFIDENCE:
        return True

    return False


def _standardize_preview_validation(item: Dict[str, Any]) -> Dict[str, Any]:
    """Add stable validation fields without removing legacy preview fields."""
    if not isinstance(item, dict):
        return item
    out = dict(item)
    status = str(out.get("validation_status") or "").strip().lower()
    reason = str(
        out.get("rejection_reason")
        or out.get("review_reason")
        or out.get("reason")
        or ""
    ).strip()
    crop_source = str(out.get("crop_source") or out.get("cropSource") or "").strip().lower()
    crop_quality = str(out.get("crop_quality") or out.get("cropQuality") or "").strip().lower()
    try:
        confidence = float(out.get("confidence") or out.get("score") or 0.0)
    except Exception:
        confidence = 0.0
    bbox_area = _bbox_area_ratio(out.get("bbox"))

    if bool(out.get("needs_review")) and status not in {"rejected", "needs_review"}:
        status = "needs_review"

    safe_fallback_approved = False
    if crop_source == "full_image_fallback":
        if _is_safe_public_fallback_garment(out, confidence, crop_quality):
            safe_fallback_approved = True
            status = "ok"
            reason = ""
            out["needs_review"] = False
            out["requires_manual_entry"] = False
            out["crop_quality"] = crop_quality or "full_image"
            out["cropQuality"] = out["crop_quality"]
        else:
            status = "needs_review"
            reason = "detector_fallback_full_image"
            out["needs_review"] = True
            out["requires_manual_entry"] = True
            out["crop_quality"] = crop_quality or "full_image_person_risk"
            out["cropQuality"] = out["crop_quality"]

    if _is_accessory_review_item(out):
        if confidence < 0.72:
            status = "rejected"
            reason = "accessory_low_confidence"
        elif bbox_area is not None and bbox_area < 0.012:
            status = "rejected"
            reason = "accessory_bbox_too_small"
        elif crop_source in {"full_image_fallback", "hybrid"} and crop_quality in {
            "broad",
            "full_image",
            "full_image_person_risk",
        }:
            status = "rejected"
            reason = "accessory_body_region_false_positive"
        elif str(out.get("reasoning") or out.get("upload_error") or "").lower().find("skin") >= 0:
            status = "rejected"
            reason = "accessory_not_clearly_visible"

    if _is_bottom_review_item(out):
        partial_signal = (
            crop_quality in {"broad", "full_image", "full_image_person_risk", "partial"}
            or crop_source == "full_image_fallback"
            or (bbox_area is not None and bbox_area < 0.08)
            or any(
                token in str(out.get(k) or "").lower()
                for k in ("review_reason", "reasoning", "upload_error")
                for token in ("partial", "waistband", "body", "person", "cleaner photo")
            )
        )
        if partial_signal and status != "rejected" and not safe_fallback_approved:
            status = "needs_review"
            reason = "partial_bottomwear_visible"
            out["needs_review"] = True
            out["requires_manual_entry"] = True

    if status not in {"ok", "needs_review", "rejected"}:
        status = "needs_review" if bool(out.get("needs_review") or out.get("requires_manual_entry")) else "ok"
    if status == "ok":
        reason = ""
        out["needs_review"] = False
        out["requires_manual_entry"] = False
    elif status == "needs_review":
        out["needs_review"] = True
        out["requires_manual_entry"] = True
    elif status == "rejected":
        out["needs_review"] = True
        out["requires_manual_entry"] = True

    out["validation_status"] = status
    out["rejection_reason"] = reason or None
    out["review_reason"] = reason or out.get("review_reason")
    out["selected_by_default"] = status == "ok"
    out["crop_quality_score"] = out.get("crop_quality_score")
    if out.get("crop_quality") and not out.get("cropQuality"):
        out["cropQuality"] = out.get("crop_quality")
    out["detection_mode"] = out.get("detection_mode") or out.get("source") or crop_source or None
    out["regen_provider"] = out.get("regen_provider")
    out["input_type"] = out.get("input_type")
    return out


def _is_preview_item_save_approved(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    if bool(item.get("wardrobe_profile_mismatch")) or bool(item.get("needs_wearer_confirmation")):
        return False
    if _is_screenshot_or_style_collage(item):
        return False
    status = str(item.get("validation_status") or "").strip().lower()
    if status:
        return status == "ok"
    return not bool(item.get("needs_review") or item.get("requires_manual_entry"))


def _normalize_capture_preview_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize taxonomy and keep successful vision results out of review state."""
    normalized = _taxonomy_normalize_item(item)
    if not isinstance(normalized, dict):
        return normalized
    normalized = _apply_headwear_ocr_guard(normalized)
    normalized["name"] = _best_effort_item_name(normalized, item)
    blob = _item_text_blob({**item, **normalized})
    if "blouse" in blob and any(term in blob for term in ("saree", "sari", "ethnic", "traditional")):
        normalized["category"] = "Tops"
        normalized["sub_category"] = "Saree Blouse" if "saree" in blob or "sari" in blob else "Ethnic Blouse"
        normalized["subcategory"] = normalized["sub_category"]
        normalized["style_context"] = "ethnic"
        normalized["traditional_wear"] = True
    normalized = apply_metadata_guard(normalized, source="capture_preview")

    if _vision_says_item_is_known(
        str(normalized.get("label_source") or ""),
        str(normalized.get("category") or ""),
        str(normalized.get("name") or ""),
    ):
        normalized["requires_manual_entry"] = False
        normalized["needs_review"] = False
    return _standardize_preview_validation(
        _apply_full_image_person_risk_guard(_enforce_preview_taxonomy(normalized))
    )


def _strip_internal_preview_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal pipeline fields the review screen renders as debug
    chips (e.g. 'vision:gemini_multi', 'heuristic'). Logic that needs them
    (validator gating, review-state checks) runs before this point; logs are
    unaffected.
    """
    if not isinstance(item, dict):
        return item
    out = dict(item)
    out.pop("label_source", None)
    out.pop("metadata_validator", None)
    reasoning = str(out.get("reasoning") or "")
    if "heuristic" in reasoning or "vision:gemini_multi" in reasoning:
        out["reasoning"] = "auto_detected"
    return out


def _vision_extract_attributes(
    masked_url: str, fallback_label: str, image_base64: str = ""
) -> Dict[str, Any]:
    base = {
        "name": str(fallback_label or "Item").strip().title() or "Item",
        "category": "",
        "sub_category": "",
        "pattern": "plain",
        "color_name": "",
        "occasions": [],
        "label_source": "heuristic",
        "requires_manual_entry": True,
    }
    if not masked_url and not image_base64:
        return base
    if str(os.getenv("ENABLE_VISION", "false")).strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return base

    try:
        from services.auth_helpers import assert_safe_outbound_url

        if image_base64:
            image_b64 = str(image_base64 or "").split(",")[-1]
        else:
            assert_safe_outbound_url(masked_url)
            image_resp = requests.get(masked_url, timeout=(2, 8))
            image_resp.raise_for_status()
            image_b64 = base64.b64encode(image_resp.content).decode("utf-8")

        ai_json, _ = ai_gateway.ollama_vision_json(
            prompt=(
                WARDROBE_CAPTURE_PROMPT
                + "\nThis image is a single garment crop. Return exactly one item in items. "
                + "Use the garment only; ignore background, body, hanger, mannequin, room, or shadows."
            ),
            image_base64=image_b64,
            timeout_seconds=int(
                os.getenv(
                    "WARDROBE_CAPTURE_VISION_TIMEOUT_SECONDS",
                    os.getenv("OLLAMA_VISION_TIMEOUT_SECONDS", "8"),
                )
            ),
            usecase="vision",
        )
        if isinstance(ai_json, dict):
            ai_item = _extract_first_vision_item(ai_json)
            logger.info(
                "ahvi.capture_vision_item name=%s category=%s sub_category=%s color=%s confidence=%s",
                _clean_label_text(ai_item.get("name"), "", 80),
                _clean_label_text(ai_item.get("category"), "", 50),
                _clean_label_text(ai_item.get("sub_category"), "", 50),
                _clean_label_text(ai_item.get("color_name"), "", 40),
                ai_item.get("confidence"),
            )
            base.update(
                {
                    "name": _clean_label_text(ai_item.get("name"), base["name"]),
                    "category": _clean_label_text(ai_item.get("category"), "", 50),
                    "sub_category": _clean_label_text(
                        ai_item.get("sub_category"), "", 50
                    ),
                    "pattern": _clean_label_text(
                        ai_item.get("pattern"), base["pattern"], 40
                    ).lower(),
                    "color_name": _clean_label_text(
                        ai_item.get("color_name"), "", 40
                    ).lower(),
                    "occasions": _normalize_occasions(ai_item.get("occasions")),
                    "confidence": float(ai_item.get("confidence") or 0.0),
                    "reasoning": _clean_label_text(ai_item.get("reasoning"), "", 160),
                    "label_source": "vision",
                    "requires_manual_entry": False,
                }
            )
    except Exception:
        logger.exception("vision item enrichment failed; falling back to heuristic")

    return _enforce_preview_taxonomy(
        apply_metadata_guard(base, source="vision_extract")
    )


async def _full_image_fallback_item(
    image: Image.Image, source_bytes: bytes, reason: str
) -> Dict[str, Any]:
    masked_bytes = b""
    image_status = "rmbg_pending"
    if _env_enabled("WARDROBE_CAPTURE_FAST_MODE", "true"):
        reason = f"{reason}; fast_mode_bg_skipped"
    else:
        try:
            masked_bytes = await remove_bg_bytes(source_bytes)
            if not masked_bytes:
                raise RuntimeError("rmbg_returned_empty_image")
            if masked_bytes == source_bytes:
                raise RuntimeError("rmbg_returned_original_image")
            image_status = "rmbg_complete"
        except Exception as exc:
            reason = f"{reason}; bg_fallback:{exc}"
            masked_bytes = b""
            image_status = "rmbg_failed"

    return {
        "item_id": str(uuid.uuid4()),
        "label": "item",
        "score": 0.35,
        "bbox": [0, 0, image.size[0], image.size[1]],
        "crop_source": "full_image_fallback",
        "crop_quality": "full_image",
        "orientation_corrected": True,
        "raw_url": None,
        "masked_url": None,
        "raw_image_base64": "data:image/png;base64,"
        + base64.b64encode(source_bytes).decode("ascii"),
        "masked_image_base64": (
            "data:image/png;base64," + base64.b64encode(masked_bytes).decode("ascii")
            if masked_bytes
            else ""
        ),
        "imageStatus": image_status,
        "image_status": image_status,
        "upload_error": reason,
    }


def _decode_inline_image(value: Any) -> bytes:
    text = str(value or "").strip()
    if not text:
        return b""
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except Exception:
        return b""


def _catalog_generation_enabled() -> bool:
    return (
        _env_enabled("ENABLE_CATALOG_GENERATION", "false")
        or _env_enabled("ENABLE_CATALOG_IMAGE_GENERATION", "false")
        or _env_enabled("ENABLE_CATALOG_NORMALIZATION", "false")
    )


_BLOCKED_CATALOG_STATUSES = {
    "blocked_unsafe_fallback",
    "failed_unsafe_catalog",
    "blocked_blank_catalog",
    "blocked_black_frame",
}


def _apply_display_image_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Expose catalog-first display hints without requiring new Appwrite fields."""
    if not isinstance(item, dict):
        return item
    status = str(item.get("catalogStatus") or item.get("catalog_status") or "").strip()
    provider = str(item.get("catalogProvider") or item.get("catalog_provider") or item.get("regen_provider") or "").strip()
    normalized = str(item.get("normalized_url") or item.get("normalizedUrl") or "").strip()
    masked = str(item.get("masked_url") or item.get("maskedUrl") or "").strip()
    original = str(
        item.get("image_url")
        or item.get("imageUrl")
        or item.get("raw_url")
        or item.get("rawUrl")
        or item.get("url")
        or ""
    ).strip()

    display_url = ""
    display_source = ""
    if status == "catalog_generated" and normalized:
        display_url = normalized
        display_source = "catalog"
    elif status == "catalog_ready" and normalized:
        display_url = normalized
        display_source = "cutout"
    elif masked:
        display_url = masked
        display_source = "masked_fallback" if status in {"fallback_cutout", "catalog_ready", "catalog_failed", "catalog_skipped_category"} else "masked"
    elif normalized:
        display_url = normalized
        display_source = "catalog" if status == "catalog_generated" else "normalized"
    elif original:
        display_url = original
        display_source = "original"

    if display_url:
        item["display_image_url"] = display_url
        item["displayImageUrl"] = display_url
        item["display_image_source"] = display_source
        item["displayImageSource"] = display_source
    if status:
        item["catalog_status"] = status
    if provider:
        item["catalog_provider"] = provider
    return item


def _catalog_quality_score(item: Dict[str, Any]) -> float | None:
    if not isinstance(item, dict):
        return None
    for key in (
        "catalogQualityScore",
        "catalog_quality_score",
        "catalog_score",
        "catalogScore",
        "quality_score",
        "qualityScore",
    ):
        value = item.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _catalog_black_frame_unresolved(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    detected = bool(item.get("black_frame_detected") or item.get("blackFrameDetected"))
    cropped = bool(item.get("black_frame_cropped") or item.get("blackFrameCropped"))
    rejected = bool(item.get("black_frame_rejected") or item.get("blackFrameRejected"))
    reason = str(item.get("catalog_reason") or item.get("catalogReason") or "").lower()
    return rejected or (detected and not cropped) or ("black_frame" in reason and not cropped)


def _catalog_provider_name(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("catalogProvider")
        or item.get("catalog_provider")
        or item.get("regen_provider")
        or ""
    ).strip().lower()


def _bbox_is_full_image(item: Dict[str, Any]) -> bool:
    bbox = item.get("bbox") if isinstance(item, dict) else None
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return False
    return x1 <= 0 and y1 <= 0 and x2 > 0 and y2 > 0


def _fallback_cutout_flag(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    status = str(item.get("catalogStatus") or item.get("catalog_status") or "").strip()
    return (
        status == "fallback_cutout"
        or bool(item.get("fallback_cutout"))
        or bool(item.get("fallback_used"))
    )


def _log_source_risk_debug(item: Dict[str, Any], *, unsafe: bool, unsafe_reason: str) -> None:
    score = _catalog_quality_score(item)
    status = str(item.get("catalogStatus") or item.get("catalog_status") or "").strip()
    provider = _catalog_provider_name(item)
    display = _apply_display_image_fields(dict(item))
    display_source = str(display.get("display_image_source") or "").strip()
    quality_gate_ok = bool(item.get("quality_gate_ok") or item.get("qualityGateOk"))
    if not quality_gate_ok and score is not None:
        quality_gate_ok = score >= 75 and status in {"catalog_ready", "catalog_generated"}
    logger.info(
        "ahvi.capture.save_selected.source_risk_debug item_id=%s name=%s catalog_status=%s provider=%s display_source=%s quality_score=%s quality_gate_ok=%s validation_status=%s rejection_reason=%s selected_by_default=%s detection_state=%s bbox_is_full_image=%s fallback_cutout=%s unsafe_source=%s unsafe_reason=%s",
        item.get("item_id"),
        item.get("name") or item.get("label"),
        status,
        provider,
        display_source,
        score,
        quality_gate_ok,
        item.get("validation_status"),
        item.get("rejection_reason") or item.get("review_reason"),
        item.get("selected_by_default"),
        item.get("detection_state") or item.get("source") or item.get("label_source"),
        _bbox_is_full_image(item),
        _fallback_cutout_flag(item),
        unsafe,
        unsafe_reason,
    )


# --- Demo catalog save thresholds (category-aware, env-overridable) ----------
# Lower the save floor ONLY for generated Nano Banana catalog outputs so safe
# apparel that scores 62-70 still saves for the investor demo. Small/risky
# categories (footwear/jewelry/accessories) stay strict. Hard safety/quality
# blockers below are NEVER relaxed by score.
_CATALOG_READY_SAVE_THRESHOLD = 75  # cutout / fallback stay production-strict


def _catalog_env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _catalog_save_threshold(item: Dict[str, Any]) -> int:
    """Category-aware save threshold for generated catalog output."""
    cat = str(item.get("category") or "").strip().lower()
    blob = _item_text_blob(item)

    def _has(*toks: str) -> bool:
        return any(t in blob for t in toks)

    if cat in {"footwear", "shoes"} or _has("footwear", "sandal", "sneaker", "loafer", "heels"):
        return _catalog_env_int("CATALOG_FOOTWEAR_SAVE_THRESHOLD", 75)
    if cat in {"jewelry", "jewellery"} or _has(
        "jewelry", "jewellery", "necklace", "earring", "bangle", "bracelet", "pendant", "anklet"
    ):
        return _catalog_env_int("CATALOG_JEWELRY_SAVE_THRESHOLD", 80)
    if cat in {"accessories", "accessory", "bags", "bag", "watches"} or _has(
        "handbag", "clutch", "sunglasses", "wristwatch"
    ):
        return _catalog_env_int("CATALOG_ACCESSORY_SAVE_THRESHOLD", 75)
    # Apparel: tops, bottoms, dresses, outerwear, traditional/indian/ethnic.
    return _catalog_env_int(
        "CATALOG_APPAREL_SAVE_THRESHOLD",
        _catalog_env_int("CATALOG_DEMO_SAVE_THRESHOLD", 62),
    )


def _catalog_validation_obj(item: Dict[str, Any]) -> Dict[str, Any]:
    v = item.get("catalog_validation") or item.get("validation")
    return v if isinstance(v, dict) else {}


def _catalog_has_human_remnant(item: Dict[str, Any]) -> bool:
    if any(
        _truthy_flag(item.get(k))
        for k in (
            "human_remnants",
            "human_remnant",
            "has_human_remnant",
            "catalog_human_remnant",
            "mannequin_remnant",
        )
    ):
        return True
    v = _catalog_validation_obj(item)
    checks = v.get("checks") or {}
    if (
        checks.get("no_human") is False
        or checks.get("no_face") is False
        or checks.get("no_mannequin") is False
    ):
        return True
    return str(v.get("reason") or "").strip() == "human_or_mannequin_remnants"


def _catalog_orientation_invalid(item: Dict[str, Any]) -> bool:
    if _truthy_flag(item.get("orientation_invalid")) or _truthy_flag(item.get("sideways")):
        return True
    v = _catalog_validation_obj(item)
    if (v.get("checks") or {}).get("orientation_upright") is False:
        return True
    return str(v.get("reason") or "").strip() == "crooked_orientation"


def _catalog_identity_drift(item: Dict[str, Any]) -> bool:
    # NOTE: color_distance is intentionally NOT used here. Nano Banana
    # regenerates the garment, so a moderate color/pattern shift is expected and
    # legitimate (e.g. paisley prints, distressed denim). Gating identity_drift
    # on color_distance falsely rejected good patterned garments, so identity
    # drift now fires only on explicit wrong-garment signals.
    if _truthy_flag(item.get("identity_drift")) or _truthy_flag(item.get("wrong_garment")):
        return True
    v = _catalog_validation_obj(item)
    return str(v.get("reason") or "").strip() in {"identity_drift", "wrong_garment_type"}


def _catalog_generated_hard_block(item: Dict[str, Any]) -> str:
    """Hard blockers for a generated catalog output — never relaxed by score."""
    normalized = str(item.get("normalized_url") or item.get("normalizedUrl") or "").strip()
    if not normalized:
        return "missing_normalized_url"
    if _catalog_black_frame_unresolved(item):
        return "black_frame_unresolved"
    if _catalog_has_human_remnant(item):
        return "human_remnants"
    if _catalog_orientation_invalid(item):
        return "orientation_invalid"
    if _catalog_identity_drift(item):
        return "identity_drift"
    return ""


def _catalog_generated_block_reason(item: Dict[str, Any], score: Optional[float], provider: str) -> str:
    """Decide save for a catalog_generated output: hard blockers first, then a
    category-aware demo score floor (generated Nano Banana output only). Logs
    the decision. Returns "" to allow."""
    hard = _catalog_generated_hard_block(item)
    threshold = (
        _catalog_save_threshold(item)
        if provider == "nanobanana"
        else _CATALOG_READY_SAVE_THRESHOLD
    )
    if hard:
        reason = hard
    elif provider == "nanobanana" and score is None:
        # Missing score on a generated Nano Banana output hides provider/
        # validation failures — never silently accept for demo.
        reason = "missing_catalog_quality_score"
    elif score is not None and score < threshold:
        reason = "low_quality_catalog"
    else:
        reason = ""
    logger.info(
        "ahvi.capture.save_selected.catalog_threshold_decision item_id=%s name=%s category=%s catalog_status=catalog_generated provider=%s score=%s threshold_used=%s hard_checks_passed=%s accepted=%s rejection_reason=%s",
        item.get("item_id"),
        item.get("name") or item.get("label"),
        item.get("category"),
        provider,
        score,
        threshold,
        not bool(hard),
        not bool(reason),
        reason or "",
    )
    return reason


def _save_selected_block_reason(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return "invalid_item"
    if bool(item.get("wardrobe_profile_mismatch")) or bool(item.get("needs_wearer_confirmation")):
        return "outside_current_wardrobe_profile"
    if _is_screenshot_or_style_collage(item):
        return "screenshot_or_style_collage"
    status = str(item.get("catalogStatus") or item.get("catalog_status") or "").strip()
    normalized = str(item.get("normalized_url") or item.get("normalizedUrl") or "").strip()
    display = _apply_display_image_fields(dict(item))
    display_source = str(display.get("display_image_source") or "").strip()
    score = _catalog_quality_score(item)
    provider = _catalog_provider_name(item)
    validation_status = str(item.get("validation_status") or "").strip().lower()
    if validation_status and validation_status != "ok":
        return "validation_not_ok"
    if status == "catalog_pending":
        # WARDROBE_ASYNC_CATALOG: catalog is generated post-response. The item
        # already carries a valid rmbg/raw display image, so don't gate on
        # catalog quality here — _run_bg_finalize_catalog patches it (or leaves
        # the cutout) once it lands.
        return ""
    if status == "catalog_ready":
        if provider and provider not in {"cutout", "disabled", "none"}:
            return "unsupported_catalog_status"
        if not normalized:
            return "missing_normalized_url"
        if _catalog_black_frame_unresolved(item):
            return "black_frame_unresolved"
        if score is None or score < _CATALOG_READY_SAVE_THRESHOLD:
            return "low_quality_catalog"
    elif status == "fallback_cutout" and provider in {"cutout", ""}:
        if score is not None and score < _CATALOG_READY_SAVE_THRESHOLD:
            return "low_quality_catalog"
    elif status == "catalog_generated":
        # Demo path: hard blockers + category-aware score floor (generated only).
        generated_reason = _catalog_generated_block_reason(item, score, provider)
        if generated_reason:
            return generated_reason
    elif status == "catalog_skipped_full_frame":
        # No-detection full-frame fallback — never auto-save a board-like image.
        return "full_frame_needs_review"
    elif status and status not in {
        "catalog_failed",
        "catalog_skipped_category",
    }:
        return "unsupported_catalog_status"
    unsafe_reason = (
        str(item.get("unsafe_reason") or item.get("unsafeSourceReason") or "").strip()
        or _source_person_reason(item)
    )
    unsafe = bool(item.get("unsafe_source") or item.get("source_contains_person") or unsafe_reason)
    _log_source_risk_debug(item, unsafe=unsafe, unsafe_reason=unsafe_reason)
    if unsafe:
        logger.info(
            "ahvi.capture.save_selected.unsafe_source_requires_catalog item_id=%s catalog_status=%s display_source=%s",
            item.get("item_id"),
            status,
            display_source,
        )
        # Person/mirror/selfie sources may only save as a clean generated catalog
        # image. Raw/cutout/fallback person images still reject. Hard checks and
        # the score floor for catalog_generated already ran above.
        if status != "catalog_generated" or not normalized or display_source != "catalog":
            return "unsafe_non_catalog"
    blob = _item_text_blob(item)
    is_footwear = "footwear" in blob or any(t in blob for t in ("sandal", "shoe", "sneaker", "loafer", "boot"))
    is_jewelry = any(t in blob for t in ("jewelry", "jewellery", "necklace", "earring", "bangle", "bracelet", "ring"))
    is_bottom = any(t in blob for t in ("bottom", "jean", "trouser", "pant", "short"))
    body_terms = any(t in blob for t in ("leg", "legs", "pant leg", "feet", "foot", "skin", "wrist", "hand", "arm", "neck", "torso", "body", "person"))
    if unsafe and body_terms and status != "catalog_generated":
        if is_footwear:
            return "footwear_body_remnant"
        if is_jewelry:
            return "jewelry_body_remnant"
        if is_bottom:
            return "body_remnant"
    return ""


def _resolve_catalog_source_bytes(item: Dict[str, Any]) -> tuple[bytes, str]:
    """Find usable image bytes for catalog generation, regardless of whether
    RMBG cleanup ran. Order: inline masked b64 -> inline raw b64 -> fetch a
    resolved image URL (masked/normalized/raw). Returns (bytes, source).

    When CATALOG_NANOBANANA_FROM_RAW is on, prefer the RAW image so Nano Banana
    renders a clean product shot from the original photo instead of polishing
    the degraded RMBG cutout."""
    _from_raw = str(
        os.getenv("CATALOG_NANOBANANA_FROM_RAW", "false")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if _from_raw:
        b = _decode_inline_image(item.get("raw_image_base64"))
        if b:
            return b, "raw_b64_nb_from_raw"
    b = _decode_inline_image(item.get("masked_image_base64"))
    if b:
        return b, "masked_b64"
    b = _decode_inline_image(item.get("raw_image_base64"))
    if b:
        return b, "raw_b64"
    for key in (
        "masked_url",
        "maskedUrl",
        "normalized_url",
        "normalizedUrl",
        "image_url",
        "imageUrl",
        "raw_url",
        "rawUrl",
        "url",
    ):
        url = str(item.get(key) or "").strip()
        if url.startswith("http"):
            try:
                resp = requests.get(url, timeout=(2, 8))
                if resp.status_code == 200 and resp.content:
                    return resp.content, f"fetch:{key}"
            except Exception:  # noqa: BLE001 — fetch is best-effort.
                continue
    return b"", "none"


def _maybe_generate_catalog_image(item: Dict[str, Any]) -> None:
    """Build a clean centered catalog image whenever catalog flags are ON and
    usable bytes exist — independent of the RMBG cleanup path. NEVER raises;
    save must not depend on it. Idempotent via item['_catalog_done']."""
    if item.get("_catalog_done"):
        return
    file_id = str(item.get("item_id") or "")
    category_raw = str(item.get("category") or "").strip()
    from services.catalog_image_service import category_allowed, generate_catalog_image, normalize_catalog_category
    try:
        from services.catalog_png_generation_service import generate_catalog_png
    except Exception:  # noqa: BLE001
        generate_catalog_png = None

    category = normalize_catalog_category(category_raw)
    flags_on = _catalog_generation_enabled()
    # Hard entry log — always fires so we can see WHY catalog did/didn't run.
    logger.info(
        "ahvi.catalog.hook_enter item_id=%s category_raw=%r category_norm=%s flags_on=%s "
        "has_masked_b64=%s has_raw_b64=%s has_masked_url=%s has_raw_url=%s",
        file_id,
        category_raw,
        category,
        flags_on,
        bool(item.get("masked_image_base64")),
        bool(item.get("raw_image_base64")),
        bool(item.get("masked_url") or item.get("maskedUrl")),
        bool(item.get("raw_url") or item.get("rawUrl")),
    )
    if not flags_on:
        logger.info("ahvi.catalog.skip_flag_off item_id=%s", file_id)
        return
    try:
        if not category_allowed(category):
            item["catalogStatus"] = "catalog_skipped_category"
            item["_catalog_done"] = True
            logger.info(
                "ahvi.catalog.skip_category item_id=%s category_raw=%r category_norm=%s",
                file_id,
                category_raw,
                category,
            )
            return
        # Full-image fallback = Gemini detected no garment, so the whole photo is
        # the "item". Sending that to Nano Banana stylizes the entire scene into a
        # style-board-like image. Don't catalog it — mark needs_review so the user
        # re-shoots a clean single-garment photo instead of auto-saving a board.
        _crop_source = str(item.get("crop_source") or item.get("cropSource") or "").strip().lower()
        if _crop_source == "full_image_fallback":
            item["catalogStatus"] = "catalog_skipped_full_frame"
            item["catalog_status"] = "catalog_skipped_full_frame"
            item["needs_review"] = True
            item["requires_manual_entry"] = True
            item["_catalog_done"] = True
            logger.info(
                "ahvi.catalog.skip_full_frame item_id=%s category=%s reason=no_detection_full_image_fallback",
                file_id,
                category,
            )
            return
        src_bytes, src = _resolve_catalog_source_bytes(item)
        if not src_bytes:
            item["catalogStatus"] = "catalog_failed"
            item["catalog_status"] = "catalog_failed"
            item["_catalog_done"] = True
            logger.info(
                "ahvi.catalog.skip_no_bytes item_id=%s category=%s", file_id, category
            )
            return
        logger.info(
            "ahvi.catalog.start item_id=%s category=%s source=%s", file_id, category, src
        )
        catalog_png_enabled = _env_enabled("ENABLE_CATALOG_GENERATION", "false")
        if catalog_png_enabled and generate_catalog_png is not None:
            catalog_metadata = {
                "category": category,
                "item_id": file_id,
                "name": item.get("name") or item.get("label"),
                "sub_category": item.get("sub_category") or item.get("subcategory"),
                "crop_source": item.get("crop_source") or item.get("cropSource"),
                "crop_quality": item.get("crop_quality") or item.get("cropQuality"),
                "needs_review": item.get("needs_review"),
                "review_reason": item.get("review_reason"),
                "requires_manual_entry": item.get("requires_manual_entry"),
                "source": item.get("source"),
                "label_source": item.get("label_source"),
                "validation_status": item.get("validation_status"),
                # Explicit risk signals: the generator must force Imagen for
                # person/mirror/selfie sources, and text sniffing alone can miss
                # them.
                "unsafe_source": bool(item.get("unsafe_source")),
                "source_contains_person": bool(item.get("source_contains_person")),
                "unsafe_reason": (
                    item.get("unsafe_reason")
                    or item.get("unsafeSourceReason")
                    or ""
                ),
            }
            logger.info(
                "ahvi.catalog.metadata item_id=%s crop_quality=%s needs_review=%s review_reason=%s",
                file_id,
                catalog_metadata.get("crop_quality"),
                catalog_metadata.get("needs_review"),
                catalog_metadata.get("review_reason"),
            )
            result = generate_catalog_png(
                src_bytes,
                item_metadata=catalog_metadata,
            )
            if not result.get("success") or not result.get("catalog_png_bytes"):
                status = str(result.get("status") or "catalog_failed")
                item["catalogStatus"] = status
                item["catalog_status"] = status
                item["catalogProvider"] = result.get("catalog_provider")
                item["catalog_provider"] = result.get("catalog_provider")
                item["_catalog_done"] = True
                logger.info(
                    "ahvi.catalog_png.failed item_id=%s category=%s reason=%s",
                    file_id,
                    category,
                    result.get("reason"),
                )
                return
            storage = R2Storage()
            if hasattr(storage, "upload_catalog_png"):
                upload = storage.upload_catalog_png(
                    file_id=file_id, image_bytes=result["catalog_png_bytes"]
                )
                catalog_png_url = upload.get("normalized_url") or upload.get("catalog_png_url")
            else:
                # Test/backward compatibility for older fake R2 helpers.
                upload = storage.upload_catalog_image(
                    file_id=file_id, image_bytes=result["catalog_png_bytes"], extension="png"
                )
                catalog_png_url = (
                    upload.get("normalized_url")
                    or upload.get("catalog_png_url")
                    or upload.get("catalog_url")
                )
            status = str(result.get("status") or "catalog_ready")
            if catalog_png_url:
                item["normalized_url"] = catalog_png_url
                item["normalizedUrl"] = catalog_png_url
            item["catalogStatus"] = status
            item["catalog_status"] = status
            item["catalog_ready"] = status in {"catalog_ready", "catalog_generated", "fallback_cutout"}
            item["catalogQualityScore"] = result.get("catalog_quality_score")
            item["catalogProvider"] = result.get("catalog_provider")
            item["catalog_provider"] = result.get("catalog_provider")
            item["regen_provider"] = result.get("catalog_provider")
            item["catalogRotationApplied"] = int(result.get("rotation_applied") or 0)
            # Stamp the generated-output validation so save-side hard blockers
            # (human_remnants / orientation_invalid / identity_drift / black
            # frame) can enforce on real signals, not just the score.
            validation = result.get("validation")
            if isinstance(validation, dict):
                item["catalog_validation"] = validation
            item["_catalog_done"] = True
            _apply_display_image_fields(item)
            logger.info(
                "ahvi.catalog_png.uploaded item_id=%s category=%s status=%s provider=%s score=%s normalized_url=%s masked_url=%s",
                file_id,
                category,
                status,
                result.get("catalog_provider"),
                item.get("catalogQualityScore"),
                catalog_png_url,
                item.get("masked_url") or item.get("maskedUrl"),
            )
            return

        result = generate_catalog_image(
            src_bytes,
            item_metadata={"category": category, "item_id": file_id},
            mode="rmbg_first",
        )
        if not result.get("success") or not result.get("catalog_image_bytes"):
            reason = str(result.get("reason") or "")
            status = (
                "catalog_validation_failed"
                if reason.startswith("validation:")
                else "catalog_failed"
            )
            item["catalogStatus"] = status
            item["catalog_status"] = status
            item["_catalog_done"] = True
            logger.info(
                "ahvi.catalog.failed item_id=%s category=%s reason=%s",
                file_id,
                category,
                reason,
            )
            return
        upload = R2Storage().upload_catalog_image(
            file_id=file_id, image_bytes=result["catalog_image_bytes"], extension="jpg"
        )
        catalog_url = upload.get("catalog_url")  # deterministic catalog_{item_id}.jpg
        if catalog_url:
            item["normalized_url"] = catalog_url
            item["normalizedUrl"] = catalog_url
        item["catalogStatus"] = "catalog_ready"
        item["catalog_ready"] = True  # API convenience flag
        item["catalogMethod"] = "rmbg_center_normalize"
        item["catalogRotationApplied"] = int(result.get("rotation_applied") or 0)
        item["_catalog_done"] = True
        logger.info(
            "ahvi.catalog.uploaded item_id=%s category=%s url=%s rotation=%d",
            file_id,
            category,
            catalog_url,
            int(result.get("rotation_applied") or 0),
        )
    except Exception as exc:  # noqa: BLE001 — catalog is non-blocking.
        item["catalogStatus"] = "catalog_failed"
        item["catalog_status"] = "catalog_failed"
        item["_catalog_done"] = True
        logger.warning(
            "ahvi.catalog.failed item_id=%s category=%s err=%s",
            file_id,
            category,
            repr(exc)[:160],
        )


def _try_upload_inline_images(
    item: Dict[str, Any],
    *,
    allow_fast_mode_skip: bool = True,
    prefer_inline: bool = False,
) -> Dict[str, Any]:
    if _privacy_catalog_only():
        from services.category_taxonomy import is_face_risk_category

        if is_face_risk_category(
            item.get("category"), item.get("sub_category"), item.get("name")
        ):
            # Privacy: for face-risk items (worn apparel, head/neck), never
            # upload the raw crop or RMBG cutout — they can contain the user's
            # face. The catalog generator reads the in-memory *_image_base64 and
            # uploads its own face-free PNG, which becomes the only stored image.
            # No catalog -> item not saved (handled at persist), never a face.
            # Non-face-risk accessories (footwear, bags, belts, watches) fall
            # through to the normal upload below.
            item["_save_image_source"] = "privacy_catalog_only_skip_upload"
            return item

    if allow_fast_mode_skip and _env_enabled("WARDROBE_CAPTURE_FAST_MODE", "true"):
        item["upload_error"] = (
            str(item.get("upload_error") or "") + "; fast_mode_upload_skipped"
        ).strip("; ")
        return item

    if item.get("masked_url") and not prefer_inline:
        return item

    raw_bytes = _decode_inline_image(item.get("raw_image_base64"))
    masked_bytes = _decode_inline_image(item.get("masked_image_base64"))
    if not raw_bytes or not masked_bytes:
        item["_save_image_source"] = (
            "masked_url"
            if item.get("masked_url")
            else "image_url"
            if item.get("image_url") or item.get("imageUrl")
            else "raw_url"
            if item.get("raw_url") or item.get("rawUrl")
            else "missing"
        )
        return item

    original_image_url = item.get("image_url") or item.get("imageUrl")
    preserve_original_image_url = bool(original_image_url) and (
        item.get("source") == "gemini_multi"
        or item.get("label_source") == "vision:gemini_multi"
    )

    try:
        file_id = str(item.get("item_id") or uuid.uuid4())
        upload = R2Storage().upload_wardrobe_images(
            file_id=file_id,
            raw_image_bytes=raw_bytes,
            masked_image_bytes=masked_bytes,
        )
        item["item_id"] = file_id
        item["raw_url"] = upload.get("raw_image_url")
        item["rawUrl"] = item.get("raw_url")
        item["masked_url"] = upload.get("masked_image_url")
        item["maskedUrl"] = item.get("masked_url")
        item["normalized_url"] = (
            upload.get("normalized_image_url")
            or upload.get("normalized_url")
            or upload.get("image_url")
            or upload.get("masked_image_url")
        )
        processed_image_url = (
            item.get("normalized_url") or item.get("masked_url") or item.get("raw_url")
        )
        item["image_url"] = (
            original_image_url if preserve_original_image_url else processed_image_url
        )
        item["imageUrl"] = item.get("image_url")
        item["processed_image_url"] = processed_image_url
        item["normalizedUrl"] = item.get("normalized_url")
        item["raw_file_name"] = upload.get("raw_file_name")
        item["masked_file_name"] = upload.get("masked_file_name")
        item["normalized_file_name"] = upload.get("normalized_file_name")
        item["_save_image_source"] = "inline_crop_upload"
    except Exception as exc:
        item["upload_error"] = str(exc)
        item["_save_image_source"] = "existing_url_after_inline_upload_failure"
    return item


# ---------------------------------------------------------------------------
# Preview-stage Gemini metadata validation
# ---------------------------------------------------------------------------
# The Metadata Validator agent already runs on the save/persistence flow, but
# the preview the user sees first was Ollama + local taxonomy only — so a
# saree could preview as "Accessories" and only get corrected after save.
# These helpers run the SAME validator (services.agent_metadata_validator)
# on risky / low-confidence items before the preview is returned.

# Terms whose presence makes the local pipeline unreliable enough to warrant
# a Gemini pass even at high local confidence.
_PREVIEW_RISKY_TERMS: tuple[str, ...] = (
    "saree", "sari", "lehenga", "gown", "dress", "one-piece", "one piece",
    "boxer", "boxers", "brief", "briefs", "underwear", "innerwear",
    "pajama", "pajamas", "pyjama", "pyjamas", "nightwear", "sleepwear",
    "hanger", "mannequin", "mirror", "selfie", "person", "human", "body",
)


def _should_validate_preview_with_gemini(
    item: Dict[str, Any],
    vision: Dict[str, Any],
    raw_label: str,
    category: str,
    sub_category: str,
    confidence: float,
) -> "tuple[bool, str]":
    """Returns (should_validate, reason)."""
    try:
        from services.agent_metadata_validator import (
            is_enabled as _metadata_validator_enabled,
        )

        if not _metadata_validator_enabled():
            return False, "disabled"
    except Exception:
        return False, "disabled"
    try:
        threshold = float(
            os.getenv("AGENT_METADATA_LOW_CONFIDENCE_THRESHOLD", "0.65")
        )
    except Exception:
        threshold = 0.65
    # Trusted Gemini multi-garment metadata: skip the low-confidence trigger
    # (Gemini already classified the crop) but keep the risky/private-wear
    # term check below so saree/boxers/sleepwear still get the full pass.
    gemini_trusted = (
        str(item.get("label_source") or "") == "vision:gemini_multi"
    )
    if confidence < threshold and not gemini_trusted:
        return True, f"low_confidence:{confidence:.2f}<{threshold:.2f}"
    blob = " ".join(
        [
            str(raw_label or ""),
            str(item.get("name") or ""),
            str(vision.get("name") or ""),
            str(category or ""),
            str(sub_category or ""),
        ]
    ).lower()
    for term in _PREVIEW_RISKY_TERMS:
        if term in blob:
            return True, f"risky_term:{term}"
    return False, "high_confidence_safe"


def _merge_validator_into_preview(
    detected: Dict[str, Any],
    validated: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge validator output into the preview item.

    Only meaningful values overwrite; image fields and item_id are never
    touched. The caller re-runs apply_metadata_guard afterwards so
    privateWear/publicWear/styleEligible stay consistent.
    """
    if not isinstance(validated, dict) or not validated:
        return detected
    out = dict(detected)

    def _meaningful(value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip().lower()
        return text not in {"", "unknown", "none", "null"}

    category = validated.get("category")
    if _meaningful(category):
        out["category"] = str(category)
    subcategory = validated.get("subcategory")
    if _meaningful(subcategory):
        out["sub_category"] = str(subcategory)
        out["subcategory"] = str(subcategory)
    confidence = validated.get("confidence")
    if isinstance(confidence, (int, float)) and confidence > 0:
        out["gemini_confidence"] = float(confidence)
        out["metadata_confidence"] = float(confidence)
    allowed = validated.get("allowed_occasions")
    if isinstance(allowed, list) and allowed and not out.get("occasions"):
        out["occasions"] = [str(o) for o in allowed if str(o).strip()]
    blocked = validated.get("blocked_occasions")
    if isinstance(blocked, list) and blocked:
        out["excludedContexts"] = [str(o) for o in blocked if str(o).strip()]
    risk_flags = validated.get("risk_flags")
    if isinstance(risk_flags, list) and risk_flags:
        out["risk_flags"] = [str(f) for f in risk_flags if str(f).strip()]
    formality = validated.get("formality")
    if _meaningful(formality):
        out["formality"] = str(formality)
    style_role = validated.get("style_role")
    if _meaningful(style_role):
        out["style_role"] = str(style_role)
    notes = validated.get("styling_notes")
    if isinstance(notes, list) and notes:
        out["metadata_notes"] = [str(n) for n in notes if str(n).strip()]
    return out


async def _apply_preview_metadata_validator(
    detected: Dict[str, Any],
    *,
    user_id: str,
    vision: Dict[str, Any],
    raw_label: str,
) -> "tuple[Dict[str, Any], str]":
    """Run the existing Gemini metadata validator on a preview item.

    Returns (item, state) where state is used | skipped | disabled | failed.
    Never raises — a validator failure leaves the local metadata intact.
    """
    should, reason = _should_validate_preview_with_gemini(
        detected,
        vision,
        raw_label,
        str(detected.get("category") or ""),
        str(detected.get("sub_category") or ""),
        float(detected.get("confidence") or 0.0),
    )
    if not should:
        state = "disabled" if reason == "disabled" else "skipped"
        detected["metadata_validator"] = {
            "used": False,
            "reason": reason,
            "confidence": None,
        }
        return _enforce_preview_taxonomy(detected), state
    try:
        from services.agent_metadata_validator import validate_wardrobe_metadata

        validated = await validate_wardrobe_metadata(
            item=detected,
            user_id=user_id,
            vision_result=vision,
            context={
                "stage": "capture_preview",
                "raw_label": raw_label,
                "source": "wardrobe_capture.analyze",
                "bbox": detected.get("bbox"),
                "upload_error": detected.get("upload_error"),
            },
        )
        merged = _merge_validator_into_preview(detected, validated)
        # Re-apply the metadata guard so privateWear/publicWear/styleEligible
        # are recomputed against the (possibly corrected) category.
        merged = apply_metadata_guard(merged, source="capture_preview_validator")
        # Deterministic taxonomy override wins over an empty/failed validator.
        merged = _enforce_preview_taxonomy(merged)
        merged["metadata_validator"] = {
            "used": True,
            "reason": reason,
            "confidence": (validated or {}).get("confidence"),
        }
        logger.info(
            "ahvi.capture_preview.validator_used user_id=%s reason=%s category=%s->%s",
            user_id,
            reason,
            detected.get("category"),
            merged.get("category"),
        )
        return merged, "used"
    except Exception as exc:  # noqa: BLE001 - preview must never break
        logger.warning(
            "ahvi.capture_preview.validator_failed user_id=%s err=%s",
            user_id,
            str(exc)[:200],
        )
        detected["metadata_validator"] = {
            "used": False,
            "reason": f"failed:{str(exc)[:80]}",
            "confidence": None,
        }
        return _enforce_preview_taxonomy(detected), "failed"


@router.post("/analyze")
async def analyze_capture(http_request: Request, request: CaptureAnalyzeRequest):
    started = time.perf_counter()
    user_id = _effective_user_id(http_request, request.user_id)
    image = _decode_image_base64(request.image_base64)
    source_bytes = _bytes_from_image_base64(request.image_base64)
    corrected_source_bytes = _image_to_png_bytes(image)
    logger.info(
        "ahvi.capture.orientation source_size=%s corrected_size=%s",
        len(source_bytes),
        len(corrected_source_bytes),
    )

    detection_state = "single_garment_demo"
    request_id = str(getattr(http_request.state, "request_id", "") or "")

    # --- Gemini multi-garment preview (MVP) -------------------------------
    # Runs BEFORE the existing single-garment fallback. Only takes over when
    # it returns 2+ valid items; any failure falls through to the existing
    # flow below, unchanged.
    detected_items: list = []
    if _gemini_multi.is_enabled():
        try:
            gemini_multi_items = await _gemini_multi.detect_and_crop(
                image, corrected_source_bytes, request_id=request_id
            )
        except Exception as exc:
            logger.info(
                "ahvi.capture.gemini_multi.fallback reason=exception:%s request_id=%s",
                exc,
                request_id,
            )
            gemini_multi_items = []

        if len(gemini_multi_items) >= 1:
            detection_state = (
                "gemini_multi_garment"
                if len(gemini_multi_items) >= _gemini_multi.MIN_VALID_ITEMS
                else "gemini_single_garment"
            )
            logger.info(
                "ahvi.capture.gemini_multi.result request_id=%s count=%d labels=%s",
                request_id,
                len(gemini_multi_items),
                [g.get("name") for g in gemini_multi_items],
            )
            for g in gemini_multi_items:
                crop_bytes = g.get("crop_bytes") or b""
                # Fast path: no RMBG at preview time. The raw crop doubles as
                # the preview cutout; save-selected runs the real cleanup for
                # the items the user actually keeps.
                crop_b64 = (
                    base64.b64encode(crop_bytes).decode("utf-8")
                    if crop_bytes
                    else ""
                )
                detected_items.append(
                    {
                        "item_id": str(uuid.uuid4()),
                        "label": g.get("name") or "Item",
                        "score": g.get("confidence") or 0.8,
                        # Carry Gemini metadata through so the per-item loop
                        # below does not rebuild it from heuristics and land
                        # the item in Needs Review / "Review item".
                        "source": "gemini_multi",
                        "gemini_name": g.get("name") or "",
                        "gemini_category": g.get("category") or "",
                        "gemini_sub_category": g.get("sub_category") or "",
                        "gemini_color": g.get("color") or "",
                        "gemini_needs_review": bool(g.get("needs_review") or False),
                        "gemini_review_reason": g.get("reason") or g.get("review_reason") or "",
                        "crop_source": "gemini",
                        "crop_quality": "tight",
                        "orientation_corrected": True,
                        "preview_cutout_pending": True,
                        "bbox": g.get("bbox_px") or [],
                        "raw_image_base64": crop_b64,
                        "masked_image_base64": crop_b64,
                        "upload_error": "",
                    }
                )
                if detection_state == "gemini_single_garment":
                    logger.info(
                        "ahvi.capture.gemini_single.accepted item_id=%s category=%s bbox=%s",
                        detected_items[-1].get("item_id"),
                        detected_items[-1].get("gemini_category"),
                        detected_items[-1].get("bbox"),
                    )
            logger.info(
                "ahvi.capture.preview_fast_path detection_state=%s items=%d request_id=%s",
                detection_state,
                len(detected_items),
                request_id,
            )
        elif gemini_multi_items:
            logger.info(
                "ahvi.capture.gemini_multi.fallback reason=insufficient_items count=%d request_id=%s",
                len(gemini_multi_items),
                request_id,
            )
        else:
            logger.info(
                "ahvi.capture.gemini_multi.fallback reason=no_valid_items request_id=%s",
                request_id,
            )

    if not detected_items:
        if str(
            os.getenv("WARDROBE_CAPTURE_SINGLE_GARMENT_MODE", "false")
        ).strip().lower() in {"1", "true", "yes", "on"}:
            detected_items = [
                await _full_image_fallback_item(image, corrected_source_bytes, "single_garment_mode")
            ]
        else:
            try:
                detected_items = await run_hybrid_detection(image)
            except Exception as e:
                detection_state = f"fallback:{e}"
                detected_items = [
                    await _full_image_fallback_item(image, corrected_source_bytes, str(e))
                ]

    if not detected_items:
        detection_state = "fallback:no_detection"
        detected_items = [
            await _full_image_fallback_item(image, corrected_source_bytes, "no_detection")
        ]

    items = []
    validator_states: list[str] = []
    for item in detected_items:
        item = _try_upload_inline_images(dict(item))
        raw_label = str(item.get("label") or "Item")
        category, sub_category = _normalize_category_from_label(raw_label)
        fallback_color_code = _dominant_color_hex_from_image(image)

        masked_b64 = str(item.get("masked_image_base64") or "")
        # Sync helpers — offload so the event loop is free under concurrency.
        import asyncio as _asyncio

        gemini_trusted = (
            item.get("source") == "gemini_multi"
            and bool(str(item.get("gemini_category") or "").strip())
        )
        if gemini_trusted:
            # Gemini multi-garment already produced trusted metadata for this
            # crop — use it directly as the vision signal and skip the
            # per-crop Ollama enrichment call entirely. Deterministic
            # taxonomy + suitability guards still run downstream.
            vision = {
                "name": str(item.get("gemini_name") or raw_label),
                "category": str(item.get("gemini_category") or ""),
                "sub_category": str(item.get("gemini_sub_category") or ""),
                "pattern": "plain",
                "color_name": str(item.get("gemini_color") or ""),
                "occasions": [],
                "label_source": "vision:gemini_multi",
                "requires_manual_entry": bool(item.get("gemini_needs_review")),
                "needs_review": bool(item.get("gemini_needs_review")),
                "review_reason": str(item.get("gemini_review_reason") or ""),
                "confidence": float(item.get("score") or 0.8),
                "reasoning": "gemini_multi_garment_detection",
            }
        elif _env_enabled("WARDROBE_CAPTURE_VISION_ENRICHMENT", "false"):
            vision = await _asyncio.to_thread(
                _vision_extract_attributes,
                str(item.get("masked_url") or ""),
                raw_label,
                masked_b64,
            )
        else:
            vision = _vision_extract_attributes("", raw_label, "")

        # Gemini item without a usable category (edge case): keep at least
        # the Gemini name/color as vision signal when enrichment was
        # heuristic, so the item does not degrade to "Review item".
        if (
            not gemini_trusted
            and item.get("source") == "gemini_multi"
            and not str(vision.get("label_source") or "").startswith("vision")
        ):
            if item.get("gemini_name"):
                vision["name"] = str(item["gemini_name"])
            if item.get("gemini_sub_category"):
                vision["sub_category"] = str(item["gemini_sub_category"])
            if item.get("gemini_color"):
                vision["color_name"] = str(item["gemini_color"])
            vision["label_source"] = "vision:gemini_multi"
            vision["requires_manual_entry"] = False
            vision["confidence"] = float(item.get("score") or 0.8)
            vision["reasoning"] = "gemini_multi_garment_detection"

        logger.info(
            "ahvi.capture.vision_raw name=%s category=%s sub_category=%s",
            vision.get("name"),
            vision.get("category"),
            vision.get("sub_category"),
        )

        category, sub_category, category_corrected = _guardrail_category(
            raw_label=raw_label,
            vision_name=str(vision.get("name") or ""),
            vision_category=str(vision.get("category") or ""),
            vision_sub_category=str(vision.get("sub_category") or ""),
            fallback_category=category,
            fallback_sub_category=sub_category,
        )

        color_code = (
            await _asyncio.to_thread(
                _dominant_color_hex_from_url, str(item.get("masked_url") or "")
            )
            or fallback_color_code
        )
        if color_code == "#000000" and fallback_color_code != "#000000":
            color_code = fallback_color_code
        color_name = str(vision.get("color_name") or _hex_to_name(color_code))
        label_source = str(vision.get("label_source") or "heuristic")
        if category_corrected and label_source == "vision":
            label_source = "vision+rules"
        requires_manual_entry = bool(
            vision.get("requires_manual_entry")
            or not _vision_says_item_is_known(label_source, category, vision.get("name") or "")
        )

        embedding = []
        if _embeddings_enabled():
            try:
                embedding = (
                    encode_image_url(item.get("masked_url"))
                    if item.get("masked_url")
                    else []
                )
            except Exception:
                embedding = []

        pixel_hash = ""
        duplicate = _duplicate_result(checked=False, is_duplicate=False)
        try:
            if item.get("masked_url"):
                pixel_hash = await compute_hash_from_url(str(item.get("masked_url") or ""))
            if not pixel_hash and item.get("masked_image_base64"):
                pixel_hash = compute_hash_from_base64(item.get("masked_image_base64"))

            duplicate_item = {
                "category": category,
                "sub_category": sub_category,
                "color_code": color_code,
                "pattern": str(vision.get("pattern") or "plain"),
                "occasions": vision.get("occasions") or [],
            }
            duplicate = _find_upload_duplicate(
                user_id=user_id,
                item=duplicate_item,
                pixel_hash=pixel_hash,
                image_embedding=embedding,
            )
        except Exception as exc:
            logger.warning("wardrobe duplicate check failed user_id=%s error=%s", user_id, exc)
            duplicate = _duplicate_result(checked=False, is_duplicate=False)

        confidence = float(item.get("score") or 0.8)
        if vision.get("confidence"):
            try:
                confidence = max(confidence, float(vision.get("confidence") or 0.0))
            except Exception:
                pass

        detected = {
            "item_id": item.get("item_id") or str(uuid.uuid4()),
            "name": vision.get("name") or sub_category or raw_label or "Item",
            "category": category,
            "sub_category": sub_category,
            "color_code": color_code,
            "color_name": color_name,
            "pattern": str(vision.get("pattern") or "plain"),
            "occasions": vision.get("occasions") or [],
            "confidence": confidence,
            "label_source": label_source,
            "crop_source": item.get("crop_source") or (
                "gemini" if item.get("source") == "gemini_multi" else "hybrid"
            ),
            "crop_quality": item.get("crop_quality") or "tight",
            "orientation_corrected": bool(item.get("orientation_corrected") or True),
            "preview_cutout_pending": bool(item.get("preview_cutout_pending")),
            "requires_manual_entry": requires_manual_entry,
            "needs_review": bool(vision.get("needs_review") or item.get("needs_review")),
            "review_reason": str(
                vision.get("review_reason")
                or item.get("gemini_review_reason")
                or item.get("review_reason")
                or ""
            ),
            "reasoning": vision.get("reasoning")
            or f"hybrid_detection+{label_source}",
            "bbox": item.get("bbox") or [],
            "raw_url": item.get("raw_url"),
            "rawUrl": item.get("raw_url"),
            "masked_url": item.get("masked_url"),
            "maskedUrl": item.get("masked_url"),
            "normalized_url": (
                item.get("normalized_url")
                or item.get("normalizedUrl")
                or item.get("image_url")
                or item.get("masked_url")
                or item.get("raw_url")
            ),
            "normalizedUrl": (
                item.get("normalized_url")
                or item.get("normalizedUrl")
                or item.get("image_url")
                or item.get("masked_url")
                or item.get("raw_url")
            ),

            # Compatibility for frontend paths that still read image_url/imageUrl.
            # Prefer normalized 1024x1024 transparent PNG first, then masked, then raw.
            "image_url": (
                item.get("normalized_url")
                or item.get("normalizedUrl")
                or item.get("image_url")
                or item.get("masked_url")
                or item.get("raw_url")
            ),
            "imageUrl": (
                item.get("normalized_url")
                or item.get("normalizedUrl")
                or item.get("image_url")
                or item.get("masked_url")
                or item.get("raw_url")
            ),
            "raw_file_name": item.get("raw_file_name"),
            "masked_file_name": item.get("masked_file_name"),
            "normalized_file_name": item.get("normalized_file_name"),
            "raw_image_base64": item.get("raw_image_base64"),
            "masked_image_base64": item.get("masked_image_base64"),
            "upload_error": item.get("upload_error") or "",
            "pixel_hash": pixel_hash,
            "duplicate": duplicate,
            "image_embedding": embedding,
        }
        detected = _apply_headwear_ocr_guard(
            detected,
            context_text=" ".join(
                str(v or "")
                for v in (
                    raw_label,
                    item.get("label"),
                    item.get("gemini_name"),
                    item.get("gemini_category"),
                    item.get("gemini_sub_category"),
                    vision.get("name"),
                    vision.get("category"),
                    vision.get("sub_category"),
                    vision.get("reasoning"),
                )
            ),
        )
        logger.info(
            "ahvi.capture.crop_source item_id=%s source=%s bbox=%s",
            detected.get("item_id"),
            detected.get("crop_source"),
            detected.get("bbox"),
        )
        detected.update(_infer_style_attributes(detected))
        # Preview-stage Gemini validation: correct risky / low-confidence
        # labels (saree→Accessories, one-piece→top, boxers→shorts) BEFORE
        # the user sees the preview, using the same validator the save flow
        # already trusts.
        detected, validator_state = await _apply_preview_metadata_validator(
            detected,
            user_id=user_id,
            vision=vision,
            raw_label=raw_label,
        )
        detected = _apply_headwear_ocr_guard(
            detected,
            context_text=" ".join(
                str(v or "")
                for v in (
                    raw_label,
                    item.get("label"),
                    item.get("gemini_name"),
                    item.get("gemini_category"),
                    item.get("gemini_sub_category"),
                    vision.get("name"),
                    vision.get("category"),
                    vision.get("sub_category"),
                    vision.get("reasoning"),
                )
            ),
            reason_prefix="cap_ocr_guard_post_validator",
        )
        detected = _apply_full_image_person_risk_guard(detected)
        validator_states.append(validator_state)
        items.append(detected)

    if not items:
        # No vision detection — preserve the image and surface a manual
        # review card. Never dead-end with "No items detected".
        review = _taxonomy_review_card(
            image_url="",
            image_base64=str(request.image_base64 or ""),
            reason="no_detection",
        )
        items = [
            {
                "item_id": str(uuid.uuid4()),
                "color_code": "#000000",
                "color_name": "black",
                "pattern": "plain",
                "occasions": [],
                "confidence": 0.3,
                "label_source": "manual_fallback",
                "reasoning": "fallback_no_detection",
                "bbox": [],
                "raw_url": None,
                "masked_url": None,
                "normalized_url": None,
                "image_url": None,
                "imageUrl": None,
                "raw_image_base64": None,
                "upload_error": "",
                "pixel_hash": "",
                "duplicate": _duplicate_result(checked=False, is_duplicate=False),
                "image_embedding": [],
                **review,
            }
        ]

    # Canonical taxonomy normalization for capture preview.
    # Ensures the frontend never sees: Sari→Accessories, polo→Bottoms,
    # one-piece→Tops, or unknowns silently labeled Accessories.
    items = [_normalize_capture_preview_item(item) for item in items if isinstance(item, dict)]
    items = _apply_capture_source_and_profile_safety(items, user_id=user_id)
    # Strip internal pipeline fields so the review screen never shows debug
    # chips like "vision:gemini_multi" — runs AFTER all logic that reads them.
    items = [_strip_internal_preview_fields(item) for item in items]
    validation_counts = {
        "ok": sum(1 for i in items if str(i.get("validation_status") or "") == "ok"),
        "needs_review": sum(
            1 for i in items if str(i.get("validation_status") or "") == "needs_review"
        ),
        "rejected": sum(
            1 for i in items if str(i.get("validation_status") or "") == "rejected"
        ),
    }
    selected_default_count = sum(1 for i in items if bool(i.get("selected_by_default")))

    save_result = None
    save_state = "skipped"
    if bool(request.auto_save):
        try:
            save_candidates = [
                i
                for i in items
                if _is_preview_item_save_approved(i)
                if bool(request.save_duplicates)
                or not bool((i.get("duplicate") or {}).get("is_duplicate"))
            ]
            selected_ids = [str(i["item_id"]) for i in save_candidates]
            save_result = persist_selected_items(
                user_id=user_id,
                selected_item_ids=selected_ids,
                detected_items=save_candidates,
            )
            save_state = "ok"
        except Exception as exc:
            save_state = f"failed:{exc}"
            raise HTTPException(status_code=503, detail=f"Wardrobe save failed: {exc}")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "ahvi.capture.preview_final request_id=%s items=%d names=%s categories=%s detection_state=%s",
        request_id,
        len(items),
        [i.get("name") for i in items],
        [i.get("category") for i in items],
        detection_state,
    )
    logger.info(
        "ahvi.capture.preview_validation request_id=%s ok=%d needs_review=%d rejected=%d selected_default=%d",
        request_id,
        validation_counts["ok"],
        validation_counts["needs_review"],
        validation_counts["rejected"],
        selected_default_count,
    )
    logger.info(
        "ahvi.capture_analyze user_id=%s items=%s elapsed_ms=%s fast_mode=%s vision_enrichment=%s",
        user_id,
        len(items),
        elapsed_ms,
        _env_enabled("WARDROBE_CAPTURE_FAST_MODE", "true"),
        _env_enabled("WARDROBE_CAPTURE_VISION_ENRICHMENT", "false"),
    )

    # Cache each crop server-side so the SAVE can reference it by token instead
    # of re-uploading the base64. base64 stays in this response for the preview.
    if _image_cache_enabled():
        for _it in items:
            if not isinstance(_it, dict):
                continue
            _b64 = _it.get("raw_image_base64") or _it.get("masked_image_base64")
            if _b64:
                _tok = await _image_cache_put_async(str(_b64))
                if _tok:
                    _it["image_cache_token"] = _tok

    return {
        "success": True,
        "count": len(items),
        "items": items,
        "stage_trace": {
            "total_items": len(items),
            "ok_count": validation_counts["ok"],
            "needs_review_count": validation_counts["needs_review"],
            "rejected_count": validation_counts["rejected"],
            "selected_default_count": selected_default_count,
            "regen_attempted_count": 0,
            "regen_skipped_count": validation_counts["needs_review"] + validation_counts["rejected"],
            "fallback_reason": (
                "none"
                if not str(detection_state).startswith("fallback")
                else str(detection_state)
            ),
            "detection_state": detection_state,
            "detection_config": (
                _gemini_multi.detection_config_summary(corrected_source_bytes)
                if hasattr(_gemini_multi, "detection_config_summary")
                else {"model": getattr(_gemini_multi, "GEMINI_MULTI_GARMENT_MODEL", "")}
            ),
            "detection": detection_state,
            "background_removal": (
                "ok"
                if any(
                    i.get("masked_url") or i.get("masked_image_base64") for i in items
                )
                else "fallback"
            ),
            "r2_upload": (
                "ok"
                if all(i.get("masked_url") for i in items)
                else "not_configured_or_failed"
            ),
            "vision_analyze": (
                "ok"
                if any(
                    str(i.get("label_source") or "").startswith("vision") for i in items
                )
                else "fallback"
            ),
            "duplicate_detection": (
                "ok"
                if any((i.get("duplicate") or {}).get("checked") for i in items)
                else "skipped"
            ),
            "metadata_validator": (
                "used"
                if "used" in validator_states
                else "failed"
                if "failed" in validator_states
                else "disabled"
                if validator_states and all(s == "disabled" for s in validator_states)
                else "skipped"
            ),
            "save_to_wardrobe": save_state,
        },
        "save_result": save_result,
        "request_meta": {
            "request_id": str(getattr(http_request.state, "request_id", "") or ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_image_bytes": len(source_bytes),
            "duration_ms": elapsed_ms,
        },
    }


# Wardrobe save/delete helpers — backed by AppwriteProxy (replaces the
# legacy raw `requests` calls that bypassed our shared session, retries,
# and timeouts).

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError


def _normalize_profile_gender(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"male", "man", "men", "m"}:
        return "male"
    if raw in {"female", "woman", "women", "f"}:
        return "female"
    return "unknown"


def _extract_profile_gender(profile: Dict[str, Any]) -> str:
    if not isinstance(profile, dict):
        return "unknown"
    for key in ("style_gender", "gender", "preferred_gender", "target_gender"):
        gender = _normalize_profile_gender(profile.get(key))
        if gender != "unknown":
            return gender
    nested = profile.get("profile") if isinstance(profile.get("profile"), dict) else {}
    for key in ("style_gender", "gender", "preferred_gender", "target_gender"):
        gender = _normalize_profile_gender(nested.get(key))
        if gender != "unknown":
            return gender
    return "unknown"


def _fetch_wardrobe_profile_gender(user_id: str) -> str:
    clean_user = str(user_id or "").strip()
    if not clean_user:
        return "unknown"
    try:
        doc = AppwriteProxy().get_document("users", clean_user)
        gender = _extract_profile_gender(doc)
        if gender != "unknown":
            return gender
    except Exception:
        pass
    try:
        docs = AppwriteProxy().list_documents("users", limit=100)
        rows = docs if isinstance(docs, list) else (docs or {}).get("documents", [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("$id") or row.get("userId") or row.get("user_id") or "").strip() != clean_user:
                continue
            gender = _extract_profile_gender(row)
            if gender != "unknown":
                return gender
    except Exception:
        pass
    return "unknown"


def _ahvi_item_doc_id(item: Dict[str, Any]) -> str:
    return str(
        item.get("$id")
        or item.get("document_id")
        or item.get("documentId")
        or item.get("id")
        or item.get("item_id")
        or item.get("itemId")
        or ""
    ).strip()


def _ahvi_fetch_outfit_doc(document_id: str) -> Dict[str, Any]:
    if not document_id:
        return {}
    try:
        doc = AppwriteProxy().get_document("outfits", document_id)
        return doc if isinstance(doc, dict) else {}
    except AppwriteProxyError as exc:
        msg = str(exc).lower()
        if "404" in msg or "not found" in msg:
            return {}
        logger.warning("ahvi_fetch_outfit_doc failed id=%s err=%s", document_id, exc)
        return {}


def _ahvi_delete_outfit_doc(document_id: str) -> Dict[str, Any]:
    if not document_id:
        return {"ok": False, "status": "missing_document_id"}
    try:
        AppwriteProxy().delete_document("outfits", document_id)
        return {"ok": True, "status": 200}
    except AppwriteProxyError as exc:
        msg = str(exc)
        if "404" in msg.lower():
            # Treat already-deleted as success.
            return {"ok": True, "status": 404}
        logger.warning("ahvi_delete_outfit_doc failed id=%s err=%s", document_id, exc)
        return {"ok": False, "status": "exception", "error": msg[:400]}


def _ahvi_file_name_from_url(url: Any) -> str:
    try:
        from urllib.parse import urlparse, unquote

        path = urlparse(str(url or "")).path
        name = unquote(path.rsplit("/", 1)[-1])
        return name.strip()
    except Exception:
        return ""


def _ahvi_extract_r2_file_names(item: Dict[str, Any]) -> Dict[str, str]:
    raw_file_name = str(
        item.get("raw_file_name")
        or item.get("rawFileName")
        or item.get("raw_key")
        or item.get("rawKey")
        or ""
    ).strip()

    masked_file_name = str(
        item.get("masked_file_name")
        or item.get("maskedFileName")
        or item.get("masked_key")
        or item.get("maskedKey")
        or ""
    ).strip()

    if not raw_file_name:
        raw_file_name = _ahvi_file_name_from_url(
            item.get("raw_url")
            or item.get("rawUrl")
            or item.get("image_url")
            or item.get("imageUrl")
        )

    normalized_file_name = str(
        item.get("normalized_file_name")
        or item.get("normalizedFileName")
        or item.get("normalized_key")
        or item.get("normalizedKey")
        or ""
    ).strip()

    if not masked_file_name:
        masked_file_name = _ahvi_file_name_from_url(
            item.get("masked_url") or item.get("maskedUrl") or item.get("url")
        )

    if not normalized_file_name:
        normalized_file_name = _ahvi_file_name_from_url(
            item.get("normalized_url")
            or item.get("normalizedUrl")
            or item.get("image_url")
            or item.get("imageUrl")
        )

    return {
        "raw_file_name": raw_file_name,
        "masked_file_name": masked_file_name,
        "normalized_file_name": normalized_file_name,
    }


def _ahvi_delete_r2_images_for_item(item: Dict[str, Any]) -> Dict[str, Any]:
    names = _ahvi_extract_r2_file_names(item)

    if (
        not names.get("raw_file_name")
        and not names.get("masked_file_name")
        and not names.get("normalized_file_name")
    ):
        return {
            "raw_deleted": False,
            "masked_deleted": False,
            "normalized_deleted": False,
            "status": "no_r2_file_names",
        }

    try:
        storage = R2Storage()
        try:
            result = storage.delete_wardrobe_images(
                raw_file_name=names.get("raw_file_name") or "",
                masked_file_name=names.get("masked_file_name") or "",
                normalized_file_name=names.get("normalized_file_name") or "",
            )
        except TypeError:
            # Backward compatible fallback if an older r2_storage.py is still deployed.
            result = storage.delete_wardrobe_images(
                raw_file_name=names.get("raw_file_name") or "",
                masked_file_name=names.get("masked_file_name") or "",
            )
            result.setdefault("normalized_deleted", False)
        result["status"] = "ok"
        result.update(names)
        return result
    except Exception as exc:
        return {
            "raw_deleted": False,
            "masked_deleted": False,
            "normalized_deleted": False,
            "status": "exception",
            "error": str(exc),
            **names,
        }


def _ahvi_doc_belongs_to_user(doc: Dict[str, Any], user_id: str) -> bool:
    if not isinstance(doc, dict) or not doc:
        return True

    owner = str(
        doc.get("userId") or doc.get("user_id") or doc.get("owner_id") or ""
    ).strip()

    if not owner:
        return True

    return owner == str(user_id or "").strip()


# Deterministic complementary categories — picks real items, never invents.
_WORKS_WITH_CATEGORIES = {
    "tops": ["Bottoms", "Outerwear", "Footwear"],
    "bottoms": ["Tops", "Outerwear", "Footwear"],
    "dresses": ["Outerwear", "Footwear", "Bags"],
    "outerwear": ["Tops", "Bottoms", "Footwear"],
    "footwear": ["Tops", "Bottoms", "Dresses"],
    "bags": ["Dresses", "Tops", "Outerwear"],
    "jewelry": ["Tops", "Dresses", "Outerwear", "Traditional", "Ethnic Wear"],
    "jewellery": ["Tops", "Dresses", "Outerwear", "Traditional", "Ethnic Wear"],
    "accessories": ["Tops", "Dresses", "Outerwear", "Traditional", "Ethnic Wear", "Bags"],
    "traditional": ["Jewelry", "Accessories", "Bags", "Footwear", "Outerwear"],
    "ethnic wear": ["Jewelry", "Accessories", "Bags", "Footwear", "Outerwear"],
}


def _works_with_blob(item: Dict[str, Any]) -> str:
    return " ".join(
        str(item.get(k) or "")
        for k in (
            "name",
            "category",
            "sub_category",
            "subcategory",
            "style_context",
            "traditional_wear",
            "tags",
            "style_tags",
            "occasion",
        )
    ).lower()


def _works_with_is_ethnic(item: Dict[str, Any]) -> bool:
    blob = _works_with_blob(item)
    return any(
        token in blob
        for token in (
            "ethnic",
            "traditional",
            "saree",
            "lehenga",
            "kurta",
            "sherwani",
            "bandhgala",
            "nehru",
            "dupatta",
        )
    )


def _works_with_bad_ethnic_pair(item: Dict[str, Any]) -> bool:
    blob = _works_with_blob(item)
    return any(
        token in blob
        for token in (
            "loafer",
            "leather shoe",
            "sneaker",
            "trainer",
            "running",
            "oxford",
            "derby",
            "office belt",
            "laptop",
            "backpack",
            "jeans",
            "trouser",
            "trousers",
            "pants",
        )
    )


def _works_with_score(anchor: Dict[str, Any], candidate: Dict[str, Any]) -> int:
    score = 0
    blob = _works_with_blob(candidate)
    if _works_with_is_ethnic(anchor):
        if any(t in blob for t in ("jewelry", "jewellery", "necklace", "earring", "bangle", "bracelet")):
            score += 40
        if any(t in blob for t in ("clutch", "potli", "pouch")):
            score += 35
        if any(t in blob for t in ("jutti", "jutti", "mojari", "kolhapuri", "ethnic sandal", "dressy flat")):
            score += 35
        if any(t in blob for t in ("dupatta", "stole", "brooch")):
            score += 25
    if any(t in blob for t in ("formal", "dressy", "polished", "classic")):
        score += 8
    return score


@wardrobe_router.get("/{item_id}/works-with")
def get_works_with(item_id: str, http_request: Request):
    """Pairing suggestions drawn from the user's ACTUAL saved wardrobe."""
    user_id = _effective_user_id(http_request, "")
    doc = _ahvi_fetch_outfit_doc(item_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Item not found")
    if not _ahvi_doc_belongs_to_user(doc, user_id):
        raise HTTPException(status_code=403, detail="Not allowed")

    cat = str(doc.get("category") or "").strip().lower()
    want = {c.lower() for c in _WORKS_WITH_CATEGORIES.get(cat, [])}
    anchor_is_ethnic = _works_with_is_ethnic(doc)
    anchor_is_accessory = cat in {"accessories", "jewelry", "jewellery"}
    if anchor_is_ethnic:
        want.update({"tops", "dresses", "traditional", "ethnic wear", "jewelry", "jewellery", "accessories", "bags", "footwear"})

    matches: List[Dict[str, Any]] = []
    try:
        docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=100)
        rows = docs if isinstance(docs, list) else (docs or {}).get("documents", [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            rid = _ahvi_item_doc_id(row)
            if not rid or rid == item_id:
                continue
            rcat = str(row.get("category") or "").strip().lower()
            if want and rcat not in want:
                continue
            if anchor_is_accessory and rcat == "bottoms":
                continue
            if anchor_is_ethnic and _works_with_bad_ethnic_pair(row):
                continue
            image_url = (
                row.get("normalized_url")
                or row.get("masked_url")
                or row.get("image_url")
            )
            matches.append({
                "item_id": rid,
                "name": row.get("name") or row.get("sub_category") or row.get("category"),
                "category": row.get("category"),
                "image_url": image_url,
                "_score": _works_with_score(doc, row),
            })
    except Exception as exc:
        logger.warning("works_with lookup failed user_id=%s err=%s", user_id, exc)

    matches.sort(key=lambda item: int(item.get("_score") or 0), reverse=True)
    for item in matches:
        item.pop("_score", None)
    return {"matches": matches[:6], "count": len(matches)}


@wardrobe_router.get("/{item_id}/best-for")
def get_best_for(item_id: str, http_request: Request):
    """Occasions from real item metadata affinity; generic fallback only when empty."""
    user_id = _effective_user_id(http_request, "")
    doc = _ahvi_fetch_outfit_doc(item_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Item not found")
    if not _ahvi_doc_belongs_to_user(doc, user_id):
        raise HTTPException(status_code=403, detail="Not allowed")

    occ_map = {
        "office": "Office", "date": "Dinner", "coffee_run": "Coffee Date",
        "gym": "Gym", "beach": "Beach", "daily": "Everyday Wear",
        "cocktail": "Cocktail", "business_formal": "Formal",
    }

    meta = {}
    try:
        meta = enrich_wardrobe_item(doc) or {}
    except Exception as exc:
        logger.warning("best_for enrich failed user_id=%s err=%s", user_id, exc)

    def _disp(o):
        return occ_map.get(str(o), str(o).replace("_", " ").title())

    best_for = [_disp(o) for o in (meta.get("occasion_affinity") or [])]
    avoid = [_disp(o) for o in (meta.get("avoid_for") or [])]

    if not best_for:  # fallback ONLY when item metadata yields nothing
        best_for = ["Everyday Wear"]

    return {"best_for": best_for[:3], "avoid": avoid[:3]}


@router.post("/analyze-batch")
async def analyze_capture_batch(
    http_request: Request, request: CaptureAnalyzeBatchRequest
):
    user_id = _effective_user_id(http_request, request.user_id)

    images = list(request.image_base64s or [])[:6]
    all_items: List[Dict[str, Any]] = []
    per_image: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if not images:
        raise HTTPException(status_code=400, detail="image_base64s is required")

    import asyncio as _asyncio

    sem = _asyncio.Semaphore(
        max(1, int(os.getenv("CAPTURE_BATCH_PARALLELISM", "2")))
    )

    async def _run_one(index: int, image_base64: str):
        async with sem:
            single_request = CaptureAnalyzeRequest(
                user_id=user_id,
                image_base64=image_base64,
                auto_save=False,
                save_duplicates=request.save_duplicates,
            )
            return await analyze_capture(http_request, single_request)

    results = await _asyncio.gather(
        *[_run_one(i, b) for i, b in enumerate(images)],
        return_exceptions=True,
    )

    for index, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append({"index": index, "error": str(result)})
            review = _taxonomy_review_card(
                image_base64=str(images[index] or ""),
                reason=f"analyze_failed:{result}",
            )
            review["item_id"] = str(uuid.uuid4())
            review["source_image_index"] = index
            review["batch_index"] = index
            all_items.append(review)
            per_image.append(
                {
                    "index": index,
                    "success": False,
                    "count": 1,
                    "items": [review],
                    "error": str(result),
                }
            )
            continue

        items = result.get("items") if isinstance(result, dict) else []
        if not isinstance(items, list):
            items = []

        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item["source_image_index"] = index
            item["batch_index"] = index
            normalized.append(item)
            all_items.append(item)

        per_image.append(
            {
                "index": index,
                "success": True,
                "count": len(normalized),
                "items": normalized,
                "stage_trace": (
                    result.get("stage_trace") if isinstance(result, dict) else {}
                ),
            }
        )

    try:
        import logging

        logging.getLogger("ahvi.wardrobe_capture").info(
            "ahvi.capture_analyze_batch user_id=%s images=%s items=%s errors=%s",
            user_id,
            len(images),
            len(all_items),
            len(errors),
        )
    except Exception:
        pass

    return {
        "success": len(all_items) > 0,
        "count": len(all_items),
        "items": all_items,
        "per_image": per_image,
        "errors": errors,
        "max_selectable": 6,
        "stage_trace": {
            "batch_images": len(images),
            "batch_items": len(all_items),
            "batch_errors": len(errors),
        },
    }


def _needs_save_rmbg_cleanup(item: Dict[str, Any]) -> bool:
    """Gemini multi preview returned a raw crop without RMBG — clean it now."""
    if not isinstance(item, dict):
        return False
    is_gemini_crop = (
        bool(item.get("preview_cutout_pending"))
        or item.get("source") == "gemini_multi"
        or item.get("label_source") == "vision:gemini_multi"
    )
    if not is_gemini_crop:
        return False
    if (
        item.get("imageStatus") == "rmbg_complete"
        or item.get("image_status") == "rmbg_complete"
    ):
        return False
    return bool(str(item.get("raw_image_base64") or "").strip())


async def _save_rmbg_cleanup(items: List[Dict[str, Any]]) -> "tuple[int, int]":
    """Run RMBG on selected raw crops without labelling failures as cutouts."""
    sem = asyncio.Semaphore(
        max(1, int(os.getenv("WARDROBE_SAVE_RMBG_PARALLELISM", "6")))
    )

    async def _one(item: Dict[str, Any]) -> bool:
        raw_bytes = _decode_inline_image(item.get("raw_image_base64"))
        if not raw_bytes:
            item.pop("masked_image_base64", None)
            item["preview_cutout_pending"] = False
            item["imageStatus"] = "rmbg_failed"
            item["image_status"] = "rmbg_failed"
            item["_rmbg_failure_reason"] = "missing_raw_crop"
            return False

        logger.info(
            "ahvi.capture.save_selected.rmbg_start item_id=%s name=%s category=%s",
            item.get("item_id"),
            item.get("name") or item.get("label"),
            item.get("category"),
        )
        try:
            async with sem:
                masked_bytes = await remove_bg_bytes(raw_bytes)
            if not masked_bytes:
                raise RuntimeError("rmbg_returned_empty_image")
            if masked_bytes == raw_bytes:
                raise RuntimeError("rmbg_returned_original_crop")

            item["masked_image_base64"] = (
                "data:image/png;base64,"
                + base64.b64encode(masked_bytes).decode("utf-8")
            )
            item["preview_cutout_pending"] = False
            item["imageStatus"] = "rmbg_complete"
            item["image_status"] = "rmbg_complete"
            item.pop("_rmbg_failure_reason", None)
            return True
        except Exception as exc:
            reason = str(exc)[:160] or exc.__class__.__name__
            item.pop("masked_image_base64", None)
            item["preview_cutout_pending"] = False
            item["imageStatus"] = "rmbg_failed"
            item["image_status"] = "rmbg_failed"
            item["_rmbg_failure_reason"] = reason
            logger.warning(
                "ahvi.capture.save_selected.rmbg_failed item_id=%s reason=%s",
                item.get("item_id"),
                reason,
            )
            return False

    results = await asyncio.gather(*[_one(i) for i in items])
    success = sum(1 for r in results if r)
    return success, len(results) - success


def _run_save_rmbg_cleanup_sync(items: List[Dict[str, Any]]) -> "tuple[int, int]":
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_save_rmbg_cleanup(items))
    # Called from within a running loop (shouldn't happen for the sync
    # endpoint, but stay safe): run in a dedicated thread.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _save_rmbg_cleanup(items)).result()


def _async_rmbg_enabled() -> bool:
    return str(os.getenv("WARDROBE_ASYNC_RMBG", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _async_catalog_enabled() -> bool:
    """Defer the ~37s catalog PNG generation to a post-response background task
    so save-selected returns after persist (~1s). The item is saved with its
    rmbg/raw display image and the catalog is patched in when it lands.

    Hard-disabled under WARDROBE_PRIVACY_CATALOG_ONLY: there, only the face-free
    catalog image may be stored, so it cannot be deferred past persist.
    """
    if _privacy_catalog_only():
        return False
    return str(os.getenv("WARDROBE_ASYNC_CATALOG", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _privacy_catalog_only() -> bool:
    """When on, only the regenerated catalog image (face-free) may be stored.
    The raw crop and RMBG cutout can contain the user's face on worn/selfie
    photos, so they are never uploaded to R2 or written to the wardrobe doc."""
    return str(os.getenv("WARDROBE_PRIVACY_CATALOG_ONLY", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# --- WARDROBE_ANALYZE_IMAGE_CACHE -------------------------------------------
# Cache the analyzed crop bytes server-side (Redis, short TTL) keyed by a token,
# so the SAVE call can reference the image by token instead of re-uploading the
# ~MB base64 (the slow client-side step). Privacy-safe: Redis is private and
# transient — the image never lands on the public R2 bucket.
def _image_cache_enabled() -> bool:
    return str(os.getenv("WARDROBE_ANALYZE_IMAGE_CACHE", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _image_cache_ttl() -> int:
    try:
        return max(60, int(os.getenv("WARDROBE_ANALYZE_IMAGE_CACHE_TTL", "900")))
    except (TypeError, ValueError):
        return 900


async def _image_cache_put_async(data_b64: str) -> str:
    """Cache a base64 image string, return a token (or '' if unavailable)."""
    if not data_b64:
        return ""
    try:
        from services.bg_service import redis_client as _rc

        if _rc is None:
            return ""
        token = uuid.uuid4().hex
        await _rc.setex(f"imgcache:{token}", _image_cache_ttl(), data_b64)
        return token
    except Exception:  # noqa: BLE001 — caching is best-effort
        return ""


def _image_cache_get_sync(token: str) -> str:
    """Fetch a cached base64 image by token from the sync save path."""
    token = str(token or "").strip()
    if not token:
        return ""
    try:
        from services.bg_service import redis_client as _rc

        if _rc is None:
            return ""
        val = asyncio.run(_rc.get(f"imgcache:{token}"))
        if val is None:
            return ""
        return val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)
    except Exception:  # noqa: BLE001
        return ""


def _run_bg_finalize_rmbg(user_id: str, cleanup_items: List[Dict[str, Any]]) -> None:
    """Background (post-response) finalize for WARDROBE_ASYNC_RMBG: run RMBG on
    the raw crops, upload the cutout to R2, then patch each already-saved doc's
    masked_url. The item is already persisted with its catalog image; this only
    fills the cutout fallback. Best-effort; never raises."""
    try:
        _run_save_rmbg_cleanup_sync(cleanup_items)  # sets masked_image_base64
        for it in cleanup_items:
            if not isinstance(it, dict):
                continue
            item_id = str(it.get("item_id") or "").strip()
            if not item_id or str(it.get("imageStatus") or "") != "rmbg_complete":
                continue
            try:
                _try_upload_inline_images(
                    it, allow_fast_mode_skip=False, prefer_inline=True
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ahvi.async_rmbg.upload_failed item_id=%s err=%s",
                    item_id,
                    str(exc)[:160],
                )
                continue
            masked_url = str(it.get("masked_url") or it.get("maskedUrl") or "").strip()
            if not masked_url:
                continue
            update_wardrobe_item_images(
                user_id=user_id,
                item_id=item_id,
                masked_url=masked_url,
                image_status="rmbg_complete",
            )
    except Exception as exc:  # noqa: BLE001 — background must never raise
        logger.warning("ahvi.async_rmbg.finalize_failed err=%s", str(exc)[:200])


def _run_bg_finalize_catalog(
    user_id: str, catalog_items: List[Dict[str, Any]]
) -> None:
    """Background (post-response) finalize for WARDROBE_ASYNC_CATALOG: generate
    the Nano Banana catalog PNG per item (concurrently), upload it, then patch
    each already-saved doc's normalized_url + catalog_status. The item is saved
    with its rmbg/raw display image; this swaps in the polished catalog when it
    lands. Best-effort; never raises."""
    try:
        items = [i for i in catalog_items if isinstance(i, dict)]
        if not items:
            return
        from concurrent.futures import ThreadPoolExecutor

        workers = max(
            1,
            min(len(items), int(os.getenv("WARDROBE_SAVE_CATALOG_PARALLELISM", "6") or 6)),
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_maybe_generate_catalog_image, items))
        for it in items:
            item_id = str(it.get("item_id") or "").strip()
            if not item_id:
                continue
            _apply_display_image_fields(it)
            status = str(it.get("catalogStatus") or it.get("catalog_status") or "").strip()
            normalized = str(it.get("normalized_url") or it.get("normalizedUrl") or "").strip()
            # Only patch when a real catalog landed; otherwise leave the cutout.
            if status in {"catalog_ready", "catalog_generated"} and normalized:
                patched = update_wardrobe_item_images(
                    user_id=user_id,
                    item_id=item_id,
                    normalized_url=normalized,
                    catalog_status=status,
                )
                if patched:
                    logger.info("AHVI_ASYNC_CATALOG_COMPLETED item_id=%s", item_id)
                else:
                    logger.warning(
                        "AHVI_ASYNC_CATALOG_FAILED item_id=%s reason=patch_failed",
                        item_id,
                    )
            else:
                update_wardrobe_item_images(
                    user_id=user_id,
                    item_id=item_id,
                    catalog_status="catalog_failed",
                )
                logger.warning(
                    "AHVI_ASYNC_CATALOG_FAILED item_id=%s reason=%s",
                    item_id,
                    status or "generation_failed",
                )
    except Exception as exc:  # noqa: BLE001 — background must never raise
        logger.warning("AHVI_ASYNC_CATALOG_FAILED reason=finalize_error err=%s", str(exc)[:200])


@router.post("/save-selected")
def save_selected(
    http_request: Request,
    request: SaveSelectedRequest,
    background_tasks: BackgroundTasks,
):
    user_id = _effective_user_id(http_request, request.user_id)
    _async_rmbg = _async_rmbg_enabled()
    _async_catalog = _async_catalog_enabled()

    # Per-stage latency timers (RMBG -> catalog -> persist). Initialized to the
    # start so the summary log is safe even when a stage is skipped.
    _t0 = time.perf_counter()
    _t_rmbg = _t_catalog = _t_persist = _t0

    max_selectable = 6
    selected_item_ids = list(request.selected_item_ids or [])[:max_selectable]
    detected_items = [
        dict(i) if isinstance(i, dict) else i
        for i in (request.detected_items or [])
    ]
    detected_items = _apply_capture_source_and_profile_safety(
        [i for i in detected_items if isinstance(i, dict)],
        user_id=user_id,
    )

    # WARDROBE_ANALYZE_IMAGE_CACHE: when the client sent an image_cache_token
    # instead of the base64 (to avoid re-uploading ~MB), restore the crop bytes
    # from the server-side Redis cache so all downstream steps (RMBG, catalog,
    # persist) work unchanged.
    if _image_cache_enabled():
        _cache_hits = 0
        for _it in detected_items:
            if not isinstance(_it, dict):
                continue
            _tok = str(_it.get("image_cache_token") or "").strip()
            if _tok and not _it.get("raw_image_base64"):
                _cached_b64 = _image_cache_get_sync(_tok)
                if _cached_b64:
                    _it["raw_image_base64"] = _cached_b64
                    _cache_hits += 1
        if _cache_hits:
            logger.info(
                "ahvi.capture.save_selected.image_cache_restored items=%d", _cache_hits
            )

    # Deferred RMBG cleanup for Gemini multi fast-path previews: the preview
    # returned raw crops; clean only the items the user actually selected.
    selected_set = {str(x).strip() for x in selected_item_ids if str(x or "").strip()}
    selected_items = [
        i
        for i in detected_items
        if isinstance(i, dict)
        and str(i.get("item_id") or "").strip() in selected_set
        and _is_preview_item_save_approved(i)
    ]
    approved_selected_ids = [
        str(i.get("item_id") or "").strip()
        for i in selected_items
        if str(i.get("item_id") or "").strip()
    ][:max_selectable]
    rejected_selected_count = max(0, len(selected_set) - len(approved_selected_ids))
    regen_attempted_count = 0
    regen_skipped_count = max(0, len(detected_items) - len(selected_items))
    cleanup_items = [
        i
        for i in selected_items
        if isinstance(i, dict)
        and _needs_save_rmbg_cleanup(i)
    ]
    cleanup_ok = 0
    cleanup_failed = 0
    # WARDROBE_ASYNC_RMBG: defer the ~37s RMBG cutout to a post-response
    # background task so the save returns with the catalog image in ~22s. The
    # cutout (masked_url) is only the display fallback, so deferring it is safe.
    if cleanup_items and _async_rmbg:
        for i in cleanup_items:
            if isinstance(i, dict):
                i.pop("masked_image_base64", None)
                i["imageStatus"] = "rmbg_pending"
                i["image_status"] = "rmbg_pending"
        logger.info(
            "ahvi.capture.save_selected.rmbg_deferred items=%d", len(cleanup_items)
        )
    elif cleanup_items:
        try:
            cleanup_ok, cleanup_failed = _run_save_rmbg_cleanup_sync(cleanup_items)
        except Exception as exc:  # noqa: BLE001 - save must never fail on RMBG
            cleanup_ok, cleanup_failed = 0, len(cleanup_items)
            for i in cleanup_items:
                i.pop("masked_image_base64", None)
                i["preview_cutout_pending"] = False
                i["imageStatus"] = "rmbg_failed"
                i["image_status"] = "rmbg_failed"
                i["_rmbg_failure_reason"] = str(exc)[:160]
            logger.warning(
                "ahvi.capture.save_selected.rmbg_failed item_id=batch reason=%s",
                str(exc)[:160],
            )

    _t_rmbg = time.perf_counter()

    normalized_items: List[Dict[str, Any]] = []
    upload_fixed = 0
    skipped_invalid = 0
    catalog_succeeded_count = 0
    catalog_failed_count = 0
    catalog_fallback_count = 0
    unsafe_catalog_skipped_ids: set[str] = set()
    unsafe_catalog_skipped_items: Dict[str, Dict[str, Any]] = {}

    # Phase 1 (sequential, cheap): per-item prep — inline upload, RMBG status
    # reconciliation, bottom-length guard. No provider/network catalog work here.
    prepared_items: List[Dict[str, Any]] = []
    for original in selected_items:
        if not isinstance(original, dict):
            skipped_invalid += 1
            continue

        item = dict(original)

        had_url = bool(
            item.get("raw_url")
            or item.get("rawUrl")
            or item.get("image_url")
            or item.get("imageUrl")
            or item.get("masked_url")
            or item.get("maskedUrl")
            or item.get("url")
        )

        try:
            item = _try_upload_inline_images(
                item,
                allow_fast_mode_skip=False,
                prefer_inline=True,
            )
        except Exception as exc:
            item["upload_error"] = str(exc)

        if item.get("imageStatus") == "rmbg_complete":
            if item.get("masked_url"):
                item["maskedUrl"] = item.get("masked_url")
                logger.info(
                    "ahvi.capture.save_selected.rmbg_complete item_id=%s masked_url=%s",
                    item.get("item_id"),
                    item.get("masked_url"),
                )
            elif item.get("_save_image_source") == "privacy_catalog_only_skip_upload":
                # Privacy-only deliberately skips uploading the person-bearing
                # RMBG cutout (never stored). KEEP masked_image_base64 in memory
                # so the catalog generator builds its face-free PNG from the
                # CLEAN cutout, not the raw image. Popping it (as the failure
                # branch below did) forced catalog gen onto raw_b64, which fails
                # the quality gate (QUALITY_READY_THRESHOLD) and drops the item.
                logger.info(
                    "ahvi.capture.save_selected.privacy_cutout_retained item_id=%s",
                    item.get("item_id"),
                )
            else:
                reason = str(item.get("upload_error") or "masked_upload_failed")[:160]
                item.pop("masked_image_base64", None)
                item["imageStatus"] = "rmbg_failed"
                item["image_status"] = "rmbg_failed"
                item["_rmbg_failure_reason"] = reason
                cleanup_ok = max(0, cleanup_ok - 1)
                cleanup_failed += 1
                logger.warning(
                    "ahvi.capture.save_selected.rmbg_failed item_id=%s reason=%s",
                    item.get("item_id"),
                    reason,
                )

        has_url = bool(
            item.get("raw_url")
            or item.get("rawUrl")
            or item.get("image_url")
            or item.get("imageUrl")
            or item.get("masked_url")
            or item.get("maskedUrl")
            or item.get("url")
        )

        if not had_url and has_url:
            upload_fixed += 1

        # Bottoms pants/shorts sanity guard — correct a misdetected shorts/pants
        # label from the garment cutout aspect ratio. Never blocks save.
        try:
            _guard_bytes = _decode_inline_image(
                item.get("masked_image_base64")
            ) or _decode_inline_image(item.get("raw_image_base64"))
            if _guard_bytes:
                from services.wardrobe_taxonomy import apply_bottom_length_guard

                item["_orig_capture_name"] = item.get("name")
                item["_orig_capture_sub"] = item.get("sub_category")
                item["_orig_capture_gemini_name"] = item.get("gemini_name")
                item["_orig_capture_gemini_sub"] = item.get("gemini_sub_category")
                item["_orig_capture_label"] = item.get("label")
                item = apply_bottom_length_guard(item, _guard_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ahvi.taxonomy.bottom_length_guard_error item_id=%s err=%s",
                item.get("item_id"),
                repr(exc)[:160],
            )

        prepared_items.append(item)

    # Phase 2 (parallel): Nano Banana catalog generation per item runs
    # CONCURRENTLY so a multi-item save costs ~one item's provider latency, not
    # N x. Each call mutates its own item dict in place, is idempotent, and never
    # raises. Cap via WARDROBE_SAVE_CATALOG_PARALLELISM (default 6 = max items).
    if prepared_items and _async_catalog:
        # WARDROBE_ASYNC_CATALOG: defer the ~37s catalog gen to a post-response
        # background task. Mark pending so save-gating doesn't reject on catalog
        # quality; the item persists with its rmbg/raw display image and the
        # catalog is patched in by _run_bg_finalize_catalog when it lands.
        for _it in prepared_items:
            if isinstance(_it, dict):
                _it["catalogStatus"] = "catalog_pending"
                _it["catalog_status"] = "catalog_pending"
        logger.info(
            "ahvi.capture.save_selected.catalog_deferred items=%d", len(prepared_items)
        )
    elif prepared_items:
        regen_attempted_count += len(prepared_items)
        from concurrent.futures import ThreadPoolExecutor

        _cat_workers = max(
            1,
            min(
                len(prepared_items),
                int(os.getenv("WARDROBE_SAVE_CATALOG_PARALLELISM", "6") or 6),
            ),
        )
        with ThreadPoolExecutor(max_workers=_cat_workers) as _cat_pool:
            list(_cat_pool.map(_maybe_generate_catalog_image, prepared_items))

    _t_catalog = time.perf_counter()

    # Phase 3 (sequential, cheap): catalog-status accounting, save-gating, append.
    for item in prepared_items:
        # Catalog generation already ran (Phase 2). Never blocks save.
        try:
            status = str(item.get("catalogStatus") or "").strip()
            if status in {"catalog_ready", "catalog_generated"}:
                catalog_succeeded_count += 1
            elif status == "fallback_cutout":
                catalog_fallback_count += 1
            elif status in _BLOCKED_CATALOG_STATUSES:
                catalog_failed_count += 1
                item_id = str(item.get("item_id") or "").strip()
                if item_id:
                    unsafe_catalog_skipped_ids.add(item_id)
                item["validation_status"] = "rejected"
                item["rejection_reason"] = (
                    "unsafe_catalog_generation_failed"
                    if status in {"blocked_unsafe_fallback", "failed_unsafe_catalog"}
                    else "blank_catalog_image"
                )
                logger.warning(
                    "ahvi.capture.save_selected.skipped_unsafe_catalog item_id=%s status=%s",
                    item.get("item_id"),
                    status,
                )
                if item_id:
                    unsafe_catalog_skipped_items[item_id] = dict(item)
                continue
            elif status == "catalog_pending":
                # Deferred to _run_bg_finalize_catalog; not a failure.
                pass
            elif status:
                catalog_failed_count += 1
            item["regen_provider"] = (
                item.get("catalogProvider")
                or item.get("catalog_provider")
                or item.get("regen_provider")
            )
            item = _apply_display_image_fields(item)
            block_reason = _save_selected_block_reason(item)
            if block_reason:
                catalog_failed_count += 1
                item_id = str(item.get("item_id") or "").strip()
                if item_id:
                    unsafe_catalog_skipped_ids.add(item_id)
                item["validation_status"] = "rejected"
                item["rejection_reason"] = block_reason
                if block_reason == "unsafe_non_catalog":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_unsafe_non_catalog item_id=%s",
                        item.get("item_id"),
                    )
                elif block_reason == "footwear_body_remnant":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_footwear_body_remnant item_id=%s",
                        item.get("item_id"),
                    )
                elif block_reason == "jewelry_body_remnant":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_jewelry_body_remnant item_id=%s",
                        item.get("item_id"),
                    )
                elif block_reason == "body_remnant":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_body_remnant item_id=%s",
                        item.get("item_id"),
                    )
                elif block_reason == "screenshot_or_style_collage":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_screenshot_collage item_id=%s",
                        item.get("item_id"),
                    )
                elif block_reason == "low_quality_catalog":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_low_quality_catalog item_id=%s score=%s",
                        item.get("item_id"),
                        _catalog_quality_score(item),
                    )
                elif block_reason == "black_frame_unresolved":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_black_frame item_id=%s",
                        item.get("item_id"),
                    )
                elif block_reason in {
                    "human_remnants",
                    "orientation_invalid",
                    "identity_drift",
                    "missing_catalog_quality_score",
                }:
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_hard_blocker item_id=%s reason=%s",
                        item.get("item_id"),
                        block_reason,
                    )
                elif block_reason == "full_frame_needs_review":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_full_frame_needs_review item_id=%s",
                        item.get("item_id"),
                    )
                elif block_reason == "missing_normalized_url":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_missing_normalized_url item_id=%s status=%s provider=%s",
                        item.get("item_id"),
                        item.get("catalogStatus") or item.get("catalog_status"),
                        _catalog_provider_name(item),
                    )
                elif block_reason == "unsupported_catalog_status":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_unsupported_catalog_status item_id=%s status=%s provider=%s",
                        item.get("item_id"),
                        item.get("catalogStatus") or item.get("catalog_status"),
                        _catalog_provider_name(item),
                    )
                elif block_reason == "validation_not_ok":
                    logger.warning(
                        "ahvi.capture.save_selected.skipped_validation_not_ok item_id=%s validation_status=%s",
                        item.get("item_id"),
                        item.get("validation_status"),
                    )
                if item_id:
                    unsafe_catalog_skipped_items[item_id] = dict(item)
                continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ahvi.catalog.failed item_id=%s err=%s",
                item.get("item_id"),
                repr(exc)[:160],
            )

        processed_item = _apply_display_image_fields(item)
        cat_lower = str(processed_item.get("category") or "").strip().lower()
        if cat_lower in {"bottoms", "bottom"}:
            from services.wardrobe_taxonomy import _TROUSER_TOKENS, _SHORTS_TOKENS
            orig_blob = " ".join(str(v or "") for v in (
                item.get("_orig_capture_name"),
                item.get("_orig_capture_sub"),
                item.get("_orig_capture_gemini_name"),
                item.get("_orig_capture_gemini_sub"),
                item.get("_orig_capture_label")
            )).lower()
            
            is_full_length = any(t in orig_blob for t in _TROUSER_TOKENS)
            was_shorts = any(t in orig_blob for t in _SHORTS_TOKENS)
            final_blob = f"{processed_item.get('sub_category') or ''} {processed_item.get('name') or ''}".lower()
            
            if is_full_length and not was_shorts and any(t in final_blob for t in _SHORTS_TOKENS):
                processed_item["needs_review"] = True
                if processed_item.get("validation_status") == "ok":
                    processed_item["validation_status"] = "needs_review"
                logger.info(
                    "ahvi.capture.save_selected.bottom_length_mismatch item_id=%s orig_blob=%r final_blob=%r",
                    processed_item.get("item_id"), orig_blob, final_blob
                )

        normalized_items.append(
            apply_metadata_guard(processed_item, source="save_selected_request")
        )

    if unsafe_catalog_skipped_ids:
        approved_selected_ids = [
            item_id for item_id in approved_selected_ids if item_id not in unsafe_catalog_skipped_ids
        ]
        rejected_selected_count += len(unsafe_catalog_skipped_ids)
        regen_skipped_count += len(unsafe_catalog_skipped_ids)

    selected_total = len(selected_set)
    logger.info(
        "ahvi.capture.save_selected.rmbg_summary total=%d success=%d failed=%d skipped=%d",
        selected_total,
        cleanup_ok,
        cleanup_failed,
        max(0, selected_total - len(cleanup_items)),
    )
    logger.info(
        "ahvi.capture.save_selected.validation_summary selected=%d approved=%d rejected_or_review=%d regen_attempted=%d regen_skipped=%d",
        len(selected_set),
        len(approved_selected_ids),
        rejected_selected_count,
        regen_attempted_count,
        regen_skipped_count,
    )
    logger.info(
        "ahvi.capture.save_selected.catalog_summary selected=%d attempted=%d succeeded=%d failed=%d fallback_used=%d",
        len(selected_set),
        regen_attempted_count,
        catalog_succeeded_count,
        catalog_failed_count,
        catalog_fallback_count,
    )

    result = persist_selected_items(
        user_id=user_id,
        selected_item_ids=approved_selected_ids,
        detected_items=normalized_items,
    )

    _t_persist = time.perf_counter()

    saved_ids: set[str] = set()
    if isinstance(result, dict):
        if isinstance(result.get("items"), list):
            result["items"] = [
                _apply_display_image_fields(dict(item)) if isinstance(item, dict) else item
                for item in result.get("items", [])
            ]
        result.setdefault("max_selectable", max_selectable)
        result.setdefault("selected_count", len(approved_selected_ids))
        result.setdefault("input_item_count", len(detected_items))
        result.setdefault("normalized_item_count", len(normalized_items))
        result.setdefault("upload_fixed_count", upload_fixed)
        result.setdefault("skipped_invalid_count", skipped_invalid)
        result.setdefault("regen_attempted_count", regen_attempted_count)
        result.setdefault("regen_skipped_count", regen_skipped_count)
        result.setdefault("rejected_selected_count", rejected_selected_count)
        result.setdefault("catalog_processing", False)
        result.setdefault("catalog_processing_semantics", "best_effort")
        result.setdefault("catalog_scheduled_count", 0)

        # Explicit save accounting so callers never see a silent drop.
        requested_count = len(selected_set)
        raw_saved_count = result.get("saved_count")
        try:
            saved_count = int(raw_saved_count) if raw_saved_count is not None else len(result.get("items") or [])
        except Exception:
            saved_count = len(result.get("items") or [])
        dropped_count = max(0, requested_count - saved_count)
        approved_set = set(approved_selected_ids)
        saved_rows = result.get("items") if isinstance(result.get("items"), list) else []
        saved_ids = {
            str(
                row.get("item_id")
                or row.get("$id")
                or row.get("id")
                or row.get("image_id")
                or ""
            ).strip()
            for row in saved_rows
            if isinstance(row, dict)
        }
        saved_ids.discard("")
        normalized_by_id = {
            str(i.get("item_id") or "").strip(): i
            for i in normalized_items
            if isinstance(i, dict) and str(i.get("item_id") or "").strip()
        }
        items_by_id = {
            str(i.get("item_id") or "").strip(): i
            for i in detected_items
            if isinstance(i, dict) and str(i.get("item_id") or "").strip()
        }
        items_by_id.update(
            {
                item_id: item
                for item_id, item in unsafe_catalog_skipped_items.items()
            }
        )
        dropped_reasons: List[Dict[str, Any]] = []
        for item_id in selected_set:
            if saved_ids:
                if item_id in saved_ids:
                    continue
            elif item_id in approved_set and saved_count >= len(approved_set):
                continue
            dropped = (
                unsafe_catalog_skipped_items.get(item_id)
                or normalized_by_id.get(item_id)
                or items_by_id.get(item_id)
                or {}
            )
            reason = str(
                dropped.get("rejection_reason")
                or dropped.get("review_reason")
                or ""
            )
            if not reason:
                reason = "persistence_error" if item_id in approved_set else "not_save_approved"
            dropped_reasons.append(
                {
                    "item_id": item_id,
                    "validation_status": str(dropped.get("validation_status") or "unknown"),
                    "reason": reason,
                }
            )
            logger.warning(
                "ahvi.capture.save_selected.selected_not_saved item_id=%s reason=%s",
                item_id,
                reason,
            )
        result["requested_count"] = requested_count
        result["saved_count"] = saved_count
        result["dropped_count"] = dropped_count
        result["dropped_reasons"] = dropped_reasons
        # Never report success when nothing was durably saved.
        if requested_count > 0 and saved_count == 0:
            result["success"] = False
            result["message"] = "Selected wardrobe items could not be saved."
            result["retryable"] = True
        elif dropped_count > 0:
            result["partial_success"] = True
            result["message"] = f"Saved {saved_count} of {requested_count} selected items."
        if dropped_count and not int(result.get("skipped") or 0) and not result.get("errors"):
            result["skipped"] = dropped_count
        try:
            from services.agent_metadata_validator import is_enabled as _md_on
            if _md_on():
                result.setdefault("metadata_validated", True)
        except Exception:
            pass

    try:
        import logging

        logging.getLogger("ahvi.wardrobe_capture").info(
            "ahvi.save_selected_v4 user_id=%s selected=%s input_items=%s normalized_items=%s upload_fixed=%s skipped_invalid=%s saved=%s skipped=%s errors=%s",
            user_id,
            len(selected_item_ids),
            len(detected_items),
            len(normalized_items),
            upload_fixed,
            skipped_invalid,
            result.get("saved_count") if isinstance(result, dict) else None,
            result.get("skipped") if isinstance(result, dict) else None,
            result.get("errors") if isinstance(result, dict) else None,
        )
    except Exception:
        pass

    # WARDROBE_ASYNC_RMBG: finalize the deferred cutout off the response path.
    if _async_rmbg and cleanup_items:
        _approved = set(approved_selected_ids)
        _to_finalize = [
            i
            for i in cleanup_items
            if isinstance(i, dict)
            and str(i.get("item_id") or "").strip() in _approved
        ]
        if _to_finalize:
            background_tasks.add_task(
                _run_bg_finalize_rmbg, user_id, _to_finalize
            )
            logger.info(
                "ahvi.async_rmbg.scheduled user_id=%s items=%d",
                user_id,
                len(_to_finalize),
            )

    # WARDROBE_ASYNC_CATALOG: generate + patch the catalog PNG off the response
    # path for the items that actually saved.
    if _async_catalog and prepared_items:
        _cat_finalize = [
            i
            for i in prepared_items
            if isinstance(i, dict)
            and str(i.get("item_id") or "").strip() in saved_ids
        ]
        if _cat_finalize:
            background_tasks.add_task(
                _run_bg_finalize_catalog, user_id, _cat_finalize
            )
            if isinstance(result, dict):
                result["catalog_scheduled_count"] = len(_cat_finalize)
                result["catalog_processing"] = True
            logger.info(
                "AHVI_ASYNC_CATALOG_SCHEDULED user_id=%s items=%d semantics=best_effort",
                user_id, len(_cat_finalize),
            )

    # Per-stage latency for the 30s-save target: upload+RMBG -> catalog -> persist.
    logger.info(
        "ahvi.save_selected.latency user_id=%s rmbg_ms=%s catalog_ms=%s persist_ms=%s total_ms=%s items=%s",
        user_id,
        int((_t_rmbg - _t0) * 1000),
        int((_t_catalog - _t_rmbg) * 1000),
        int((_t_persist - _t_catalog) * 1000),
        int((time.perf_counter() - _t0) * 1000),
        len(approved_selected_ids),
    )

    return result


@router.post("/delete-selected")
def delete_selected(http_request: Request, request: DeleteSelectedRequest):
    user_id = _effective_user_id(http_request, request.user_id)

    ids: List[str] = []
    by_id: Dict[str, Dict[str, Any]] = {}

    for item_id in request.item_ids or []:
        clean = str(item_id or "").strip()
        if clean and clean not in ids:
            ids.append(clean)

    for item in request.items or []:
        if not isinstance(item, dict):
            continue

        clean = _ahvi_item_doc_id(item)
        if clean and clean not in ids:
            ids.append(clean)

        if clean:
            by_id[clean] = dict(item)

    ids = ids[:24]

    deleted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for doc_id in ids:
        supplied_item = by_id.get(doc_id) or {}
        appwrite_doc = _ahvi_fetch_outfit_doc(doc_id)
        source_item = (
            {**supplied_item, **appwrite_doc} if appwrite_doc else supplied_item
        )

        if appwrite_doc and not _ahvi_doc_belongs_to_user(appwrite_doc, user_id):
            skipped.append(
                {
                    "id": doc_id,
                    "reason": "user_mismatch",
                }
            )
            continue

        r2_result = {"status": "skipped"}
        if bool(request.delete_r2):
            r2_result = _ahvi_delete_r2_images_for_item(source_item)

        appwrite_result = _ahvi_delete_outfit_doc(doc_id)

        if appwrite_result.get("ok"):
            deleted.append(
                {
                    "id": doc_id,
                    "appwrite": appwrite_result,
                    "r2": r2_result,
                }
            )
        else:
            errors.append(
                {
                    "id": doc_id,
                    "appwrite": appwrite_result,
                    "r2": r2_result,
                }
            )

    try:
        import logging

        logging.getLogger("ahvi.wardrobe_capture").info(
            "ahvi.delete_selected_v4 user_id=%s requested=%s deleted=%s skipped=%s errors=%s",
            user_id,
            len(ids),
            len(deleted),
            len(skipped),
            errors,
        )
    except Exception:
        pass

    return {
        "success": len(errors) == 0,
        "requested_count": len(ids),
        "deleted_count": len(deleted),
        "skipped_count": len(skipped),
        "error_count": len(errors),
        "deleted": deleted,
        "skipped": skipped,
        "errors": errors,
    }


# ================= AHVI CAPTURE SAVE DELETE PATCH V4 END =================
