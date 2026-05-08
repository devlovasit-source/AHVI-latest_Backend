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
from PIL import Image

logger = logging.getLogger(__name__)

from services import ai_gateway
from services.bg_service import remove_bg_bytes
from services.embedding_service import encode_metadata
from services.hybrid_detection_service import run_hybrid_detection
from services.image_embedding_service import encode_image_url
from services.image_fingerprint import compute_hash_from_base64, compute_hash_from_url
from services.qdrant_service import qdrant_service
from services.r2_storage import R2Storage
from services.wardrobe_persistence_service import persist_selected_items
from prompts.core_prompts import WARDROBE_CAPTURE_PROMPT

router = APIRouter(prefix="/api/wardrobe/capture", tags=["wardrobe-capture"])


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
        return Image.open(io.BytesIO(data)).convert("RGB")
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
            usecase="vision",
        )
        if isinstance(ai_json, dict):
            ai_item = _extract_first_vision_item(ai_json)
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

    return base


async def _full_image_fallback_item(
    image: Image.Image, source_bytes: bytes, reason: str
) -> Dict[str, Any]:
    masked_bytes = source_bytes
    masked_mime = "image/png"
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


def _try_upload_inline_images(item: Dict[str, Any]) -> Dict[str, Any]:
    if item.get("masked_url"):
        return item

    raw_bytes = _decode_inline_image(item.get("raw_image_base64"))
    masked_bytes = _decode_inline_image(item.get("masked_image_base64"))
    if not raw_bytes or not masked_bytes:
        return item

    try:
        file_id = str(item.get("item_id") or uuid.uuid4())
        upload = R2Storage().upload_wardrobe_images(
            file_id=file_id,
            raw_image_bytes=raw_bytes,
            masked_image_bytes=masked_bytes,
        )
        item["item_id"] = file_id
        item["raw_url"] = upload.get("raw_image_url")
        item["masked_url"] = upload.get("masked_image_url")
        item["normalized_url"] = (
            upload.get("normalized_image_url")
            or upload.get("normalized_url")
            or upload.get("image_url")
            or upload.get("masked_image_url")
        )
        item["image_url"] = item.get("normalized_url") or item.get("masked_url") or item.get("raw_url")
        item["imageUrl"] = item.get("image_url")
        item["normalizedUrl"] = item.get("normalized_url")
        item["raw_file_name"] = upload.get("raw_file_name")
        item["masked_file_name"] = upload.get("masked_file_name")
        item["normalized_file_name"] = upload.get("normalized_file_name")
    except Exception as exc:
        item["upload_error"] = str(exc)
    return item


@router.post("/analyze")
async def analyze_capture(http_request: Request, request: CaptureAnalyzeRequest):
    user_id = _effective_user_id(http_request, request.user_id)
    image = _decode_image_base64(request.image_base64)
    source_bytes = _bytes_from_image_base64(request.image_base64)

    detection_state = "single_garment_demo"
    if str(
        os.getenv("WARDROBE_CAPTURE_SINGLE_GARMENT_MODE", "true")
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
    for item in detected_items:
        item = _try_upload_inline_images(dict(item))
        raw_label = str(item.get("label") or "Item")
        category, sub_category = _normalize_category_from_label(raw_label)
        fallback_color_code = _dominant_color_hex_from_image(image)

        masked_b64 = str(item.get("masked_image_base64") or "")
        # Sync helpers — offload so the event loop is free under concurrency.
        import asyncio as _asyncio

        vision = await _asyncio.to_thread(
            _vision_extract_attributes,
            str(item.get("masked_url") or ""),
            raw_label,
            masked_b64,
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
            vision.get("requires_manual_entry") or label_source != "vision"
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

        items.append(
            {
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
        )

    if not items:
        items = [
            {
                "item_id": str(uuid.uuid4()),
                "name": "Fallback Item",
                "category": "Tops",
                "sub_category": "Item",
                "color_code": "#000000",
                "color_name": "black",
                "pattern": "plain",
                "occasions": ["casual"],
                "confidence": 0.3,
                "label_source": "manual_fallback",
                "requires_manual_entry": True,
                "reasoning": "fallback_no_detection",
                "bbox": [],
                "raw_url": None,
                "masked_url": None,
                "normalized_url": None,
                "image_url": None,
                "imageUrl": None,
                "raw_image_base64": None,
                "masked_image_base64": None,
                "upload_error": "",
                "pixel_hash": "",
                "duplicate": _duplicate_result(checked=False, is_duplicate=False),
                "image_embedding": [],
            }
        ]

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
            "save_to_wardrobe": save_state,
        },
        "save_result": save_result,
        "request_meta": {
            "request_id": str(getattr(http_request.state, "request_id", "") or ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_image_bytes": len(source_bytes),
            "duration_hint_ms": int(time.time() * 1000) % 100000,
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
            per_image.append(
                {
                    "index": index,
                    "success": False,
                    "count": 0,
                    "items": [],
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


@router.post("/save-selected")
def save_selected(http_request: Request, request: SaveSelectedRequest):
    user_id = _effective_user_id(http_request, request.user_id)

    max_selectable = 6
    selected_item_ids = list(request.selected_item_ids or [])[:max_selectable]
    detected_items = list(request.detected_items or [])

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
            item = _try_upload_inline_images(item)
        except Exception as exc:
            item["upload_error"] = str(exc)

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

        normalized_items.append(item)

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
