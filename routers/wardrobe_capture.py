import asyncio
import base64
import io
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
from fastapi import APIRouter, HTTPException, Request
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


def _normalize_capture_preview_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize taxonomy and keep successful vision results out of review state."""
    normalized = _taxonomy_normalize_item(item)
    if not isinstance(normalized, dict):
        return normalized
    normalized["name"] = _best_effort_item_name(normalized, item)
    normalized = apply_metadata_guard(normalized, source="capture_preview")

    if _vision_says_item_is_known(
        str(normalized.get("label_source") or ""),
        str(normalized.get("category") or ""),
        str(normalized.get("name") or ""),
    ):
        normalized["requires_manual_entry"] = False
        normalized["needs_review"] = False
    return _enforce_preview_taxonomy(normalized)


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
    masked_bytes = source_bytes
    masked_mime = "image/png"
    if _env_enabled("WARDROBE_CAPTURE_FAST_MODE", "true"):
        reason = f"{reason}; fast_mode_bg_skipped"
    else:
        try:
            masked_bytes = await remove_bg_bytes(source_bytes)
        except Exception as exc:
            reason = f"{reason}; bg_fallback:{exc}"
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            masked_bytes = buf.getvalue()

    return {
        "item_id": str(uuid.uuid4()),
        "label": "item",
        "score": 0.35,
        "bbox": [0, 0, image.size[0], image.size[1]],
        "raw_url": None,
        "masked_url": None,
        "raw_image_base64": "data:image/png;base64,"
        + base64.b64encode(source_bytes).decode("ascii"),
        "masked_image_base64": f"data:{masked_mime};base64,"
        + base64.b64encode(masked_bytes).decode("ascii"),
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


def _resolve_catalog_source_bytes(item: Dict[str, Any]) -> tuple[bytes, str]:
    """Find usable image bytes for catalog generation, regardless of whether
    RMBG cleanup ran. Order: inline masked b64 -> inline raw b64 -> fetch a
    resolved image URL (masked/normalized/raw). Returns (bytes, source)."""
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
            result = generate_catalog_png(
                src_bytes,
                item_metadata={
                    "category": category,
                    "item_id": file_id,
                    "name": item.get("name") or item.get("label"),
                    "sub_category": item.get("sub_category") or item.get("subcategory"),
                    "source": item.get("source"),
                    "label_source": item.get("label_source"),
                },
            )
            if not result.get("success") or not result.get("catalog_png_bytes"):
                status = str(result.get("status") or "catalog_failed")
                item["catalogStatus"] = status
                item["catalog_status"] = status
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
            item["catalog_ready"] = status in {"catalog_ready", "catalog_generated", "fallback_cutout"}
            item["catalogQualityScore"] = result.get("catalog_quality_score")
            item["catalogRotationApplied"] = int(result.get("rotation_applied") or 0)
            item["_catalog_done"] = True
            logger.info(
                "ahvi.catalog_png.uploaded item_id=%s category=%s status=%s provider=%s score=%s url=%s",
                file_id,
                category,
                status,
                result.get("catalog_provider"),
                item.get("catalogQualityScore"),
                catalog_png_url,
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
                image, source_bytes, request_id=request_id
            )
        except Exception as exc:
            logger.info(
                "ahvi.capture.gemini_multi.fallback reason=exception:%s request_id=%s",
                exc,
                request_id,
            )
            gemini_multi_items = []

        if len(gemini_multi_items) >= _gemini_multi.MIN_VALID_ITEMS:
            detection_state = "gemini_multi_garment"
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
                        "preview_cutout_pending": True,
                        "bbox": g.get("bbox_px") or [],
                        "raw_image_base64": crop_b64,
                        "masked_image_base64": crop_b64,
                        "upload_error": "",
                    }
                )
            logger.info(
                "ahvi.capture.preview_fast_path detection_state=gemini_multi_garment items=%d request_id=%s",
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
                await _full_image_fallback_item(image, source_bytes, "single_garment_mode")
            ]
        else:
            try:
                detected_items = await run_hybrid_detection(image)
            except Exception as e:
                detection_state = f"fallback:{e}"
                detected_items = [
                    await _full_image_fallback_item(image, source_bytes, str(e))
                ]

    if not detected_items:
        detection_state = "fallback:no_detection"
        detected_items = [
            await _full_image_fallback_item(image, source_bytes, "no_detection")
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
                "requires_manual_entry": False,
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
            "preview_cutout_pending": bool(item.get("preview_cutout_pending")),
            "requires_manual_entry": requires_manual_entry,
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
    # Strip internal pipeline fields so the review screen never shows debug
    # chips like "vision:gemini_multi" — runs AFTER all logic that reads them.
    items = [_strip_internal_preview_fields(item) for item in items]

    save_result = None
    save_state = "skipped"
    if bool(request.auto_save):
        try:
            save_candidates = [
                i
                for i in items
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
        "ahvi.capture_analyze user_id=%s items=%s elapsed_ms=%s fast_mode=%s vision_enrichment=%s",
        user_id,
        len(items),
        elapsed_ms,
        _env_enabled("WARDROBE_CAPTURE_FAST_MODE", "true"),
        _env_enabled("WARDROBE_CAPTURE_VISION_ENRICHMENT", "false"),
    )

    return {
        "success": True,
        "count": len(items),
        "items": items,
        "stage_trace": {
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
}


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
            matches.append({
                "item_id": rid,
                "name": row.get("name") or row.get("sub_category") or row.get("category"),
                "category": row.get("category"),
                "image_url": row.get("masked_url") or row.get("image_url") or row.get("normalized_url"),
            })
    except Exception as exc:
        logger.warning("works_with lookup failed user_id=%s err=%s", user_id, exc)

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
    """Run RMBG on selected raw crops and retain the crop on failure."""
    sem = asyncio.Semaphore(
        max(1, int(os.getenv("WARDROBE_SAVE_RMBG_PARALLELISM", "6")))
    )

    async def _one(item: Dict[str, Any]) -> bool:
        raw_bytes = _decode_inline_image(item.get("raw_image_base64"))
        if not raw_bytes:
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
            item["masked_image_base64"] = item.get("raw_image_base64")
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


@router.post("/save-selected")
def save_selected(http_request: Request, request: SaveSelectedRequest):
    user_id = _effective_user_id(http_request, request.user_id)

    max_selectable = 6
    selected_item_ids = list(request.selected_item_ids or [])[:max_selectable]
    detected_items = [
        dict(i) if isinstance(i, dict) else i
        for i in (request.detected_items or [])
    ]

    # Deferred RMBG cleanup for Gemini multi fast-path previews: the preview
    # returned raw crops; clean only the items the user actually selected.
    selected_set = {str(x).strip() for x in selected_item_ids if str(x or "").strip()}
    cleanup_items = [
        i
        for i in detected_items
        if isinstance(i, dict)
        and _needs_save_rmbg_cleanup(i)
        and (not selected_set or str(i.get("item_id") or "").strip() in selected_set)
    ]
    cleanup_ok = 0
    cleanup_failed = 0
    if cleanup_items:
        try:
            cleanup_ok, cleanup_failed = _run_save_rmbg_cleanup_sync(cleanup_items)
        except Exception as exc:  # noqa: BLE001 - save must never fail on RMBG
            cleanup_ok, cleanup_failed = 0, len(cleanup_items)
            for i in cleanup_items:
                i["preview_cutout_pending"] = False
                i["imageStatus"] = "rmbg_failed"
                i["image_status"] = "rmbg_failed"
                i["_rmbg_failure_reason"] = str(exc)[:160]
            logger.warning(
                "ahvi.capture.save_selected.rmbg_failed item_id=batch reason=%s",
                str(exc)[:160],
            )

    normalized_items: List[Dict[str, Any]] = []
    upload_fixed = 0
    skipped_invalid = 0

    for original in detected_items:
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
            else:
                reason = str(item.get("upload_error") or "masked_upload_failed")[:160]
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

                item = apply_bottom_length_guard(item, _guard_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ahvi.taxonomy.bottom_length_guard_error item_id=%s err=%s",
                item.get("item_id"),
                repr(exc)[:160],
            )

        # Catalog generation — runs whenever flags are ON and bytes are
        # resolvable, independent of the RMBG cleanup path. Never blocks save.
        try:
            _maybe_generate_catalog_image(item)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ahvi.catalog.failed item_id=%s err=%s",
                item.get("item_id"),
                repr(exc)[:160],
            )

        normalized_items.append(apply_metadata_guard(item, source="save_selected_request"))

    selected_total = (
        len(selected_set)
        if selected_set
        else sum(1 for item in detected_items if isinstance(item, dict))
    )
    logger.info(
        "ahvi.capture.save_selected.rmbg_summary total=%d success=%d failed=%d skipped=%d",
        selected_total,
        cleanup_ok,
        cleanup_failed,
        max(0, selected_total - len(cleanup_items)),
    )

    result = persist_selected_items(
        user_id=user_id,
        selected_item_ids=selected_item_ids,
        detected_items=normalized_items,
    )

    if isinstance(result, dict):
        result.setdefault("max_selectable", max_selectable)
        result.setdefault("selected_count", len(selected_item_ids))
        result.setdefault("input_item_count", len(detected_items))
        result.setdefault("normalized_item_count", len(normalized_items))
        result.setdefault("upload_fixed_count", upload_fixed)
        result.setdefault("skipped_invalid_count", skipped_invalid)
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
