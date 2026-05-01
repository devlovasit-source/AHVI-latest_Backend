import base64
import io
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import Image

from services import ai_gateway
from services.bg_service import remove_bg_bytes
from services.hybrid_detection_service import run_hybrid_detection
from services.image_embedding_service import encode_image_url
from services.image_fingerprint import compute_hash_from_url
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


def _request_user_id(http_request: Request) -> str:
    user = getattr(http_request.state, "user", None)
    if not isinstance(user, dict):
        return ""
    return str(user.get("user_id") or user.get("$id") or user.get("id") or "").strip()


def _effective_user_id(http_request: Request, supplied_user_id: str) -> str:
    authed_user_id = _request_user_id(http_request)
    supplied = str(supplied_user_id or "").strip()
    if authed_user_id and supplied and supplied != authed_user_id:
        raise HTTPException(status_code=403, detail="user_id does not match authenticated user")
    return authed_user_id or supplied


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


def _normalize_category_from_label(label: str) -> tuple[str, str]:
    raw = str(label or "").strip().lower()
    if any(x in raw for x in ["saree", "kurta", "lehenga", "dupatta", "sherwani"]):
        return ("Indian Wear", raw.title() or "Indian Wear")
    if any(x in raw for x in ["shirt", "tshirt", "t-shirt", "top", "blouse", "crop", "sweater", "hoodie", "tee"]):
        return ("Tops", raw.title() or "Top")
    if any(x in raw for x in ["pant", "trouser", "jean", "skirt", "short"]):
        return ("Bottoms", raw.title() or "Bottom")
    if any(x in raw for x in ["dress", "gown", "jumpsuit"]):
        return ("Dresses", "Dress")
    if any(x in raw for x in ["jacket", "coat", "blazer", "outerwear"]):
        return ("Outerwear", raw.title() or "Outerwear")
    if any(x in raw for x in ["shoe", "sneaker", "heel", "boot", "sandal"]):
        return ("Footwear", raw.title() or "Footwear")
    if any(x in raw for x in ["bag", "handbag", "backpack", "purse", "tote"]):
        return ("Bags", raw.title() or "Bag")
    if any(x in raw for x in ["watch", "bracelet", "ring", "earring", "necklace"]):
        return ("Jewelry", raw.title() or "Jewelry")
    if any(x in raw for x in ["belt", "scarf", "hat", "cap", "sunglass"]):
        return ("Accessories", raw.title() or "Accessory")
    return ("Tops", "Item")


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
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
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
        if not url:
            return "#000000"
        response = requests.get(str(url).strip(), timeout=8)
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


def _vision_extract_attributes(masked_url: str, fallback_label: str, image_base64: str = "") -> Dict[str, Any]:
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
    if str(os.getenv("ENABLE_VISION", "false")).strip().lower() not in {"1", "true", "yes", "on"}:
        return base

    try:
        if image_base64:
            image_b64 = str(image_base64 or "").split(",")[-1]
        else:
            image_resp = requests.get(masked_url, timeout=8)
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
            base.update({
                "name": _clean_label_text(ai_item.get("name"), base["name"]),
                "category": _clean_label_text(ai_item.get("category"), "", 50),
                "sub_category": _clean_label_text(ai_item.get("sub_category"), "", 50),
                "pattern": _clean_label_text(ai_item.get("pattern"), base["pattern"], 40).lower(),
                "color_name": _clean_label_text(ai_item.get("color_name"), "", 40).lower(),
                "occasions": _normalize_occasions(ai_item.get("occasions")),
                "confidence": float(ai_item.get("confidence") or 0.0),
                "reasoning": _clean_label_text(ai_item.get("reasoning"), "", 160),
                "label_source": "vision",
                "requires_manual_entry": False,
            })
    except Exception:
        pass

    return base


async def _full_image_fallback_item(image: Image.Image, source_bytes: bytes, reason: str) -> Dict[str, Any]:
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
        "raw_image_base64": "data:image/png;base64," + base64.b64encode(source_bytes).decode("ascii"),
        "masked_image_base64": f"data:{masked_mime};base64," + base64.b64encode(masked_bytes).decode("ascii"),
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
        item["raw_file_name"] = upload.get("raw_file_name")
        item["masked_file_name"] = upload.get("masked_file_name")
    except Exception as exc:
        item["upload_error"] = str(exc)
    return item


@router.post("/analyze")
async def analyze_capture(http_request: Request, request: CaptureAnalyzeRequest):
    user_id = _effective_user_id(http_request, request.user_id)
    image = _decode_image_base64(request.image_base64)
    source_bytes = _bytes_from_image_base64(request.image_base64)

    detection_state = "ok"
    try:
        detected_items = await run_hybrid_detection(image)
    except Exception as e:
        detection_state = f"fallback:{e}"
        detected_items = [await _full_image_fallback_item(image, source_bytes, str(e))]

    if not detected_items:
        detection_state = "fallback:no_detection"
        detected_items = [await _full_image_fallback_item(image, source_bytes, "no_detection")]

    items = []
    for item in detected_items:
        item = _try_upload_inline_images(dict(item))
        raw_label = str(item.get("label") or "Item")
        category, sub_category = _normalize_category_from_label(raw_label)
        fallback_color_code = _dominant_color_hex_from_image(image)

        masked_b64 = str(item.get("masked_image_base64") or "")
        vision = _vision_extract_attributes(str(item.get("masked_url") or ""), raw_label, masked_b64)
        if vision.get("category"):
            category = str(vision.get("category"))
        if vision.get("sub_category"):
            sub_category = str(vision.get("sub_category"))

        color_code = _dominant_color_hex_from_url(str(item.get("masked_url") or "")) or fallback_color_code
        if color_code == "#000000" and fallback_color_code != "#000000":
            color_code = fallback_color_code
        color_name = str(vision.get("color_name") or _hex_to_name(color_code))
        label_source = str(vision.get("label_source") or "heuristic")
        requires_manual_entry = bool(vision.get("requires_manual_entry") or label_source != "vision")

        embedding = []
        if str(os.getenv("ENABLE_IMAGE_EMBEDDINGS", "false")).strip().lower() in {"1", "true", "yes", "on"}:
            try:
                embedding = encode_image_url(item.get("masked_url")) if item.get("masked_url") else []
            except Exception:
                embedding = []

        pixel_hash = ""
        duplicate = {"checked": False, "is_duplicate": False}
        try:
            pixel_hash = await compute_hash_from_url(str(item.get("masked_url") or ""))
            duplicate = qdrant_service.find_pixel_duplicate(user_id, pixel_hash, max_distance=6)
        except Exception:
            duplicate = {"checked": False, "is_duplicate": False}

        confidence = float(item.get("score") or 0.8)
        if vision.get("confidence"):
            try:
                confidence = max(confidence, float(vision.get("confidence") or 0.0))
            except Exception:
                pass

        items.append({
            "item_id": item.get("item_id") or str(uuid.uuid4()),
            "name": vision.get("name") or raw_label or "Item",
            "category": category,
            "sub_category": sub_category,
            "color_code": color_code,
            "color_name": color_name,
            "pattern": str(vision.get("pattern") or "plain"),
            "occasions": vision.get("occasions") or [],
            "confidence": confidence,
            "label_source": label_source,
            "requires_manual_entry": requires_manual_entry,
            "reasoning": vision.get("reasoning") or f"hybrid_detection+{label_source}",
            "bbox": item.get("bbox") or [],
            "raw_url": item.get("raw_url"),
            "masked_url": item.get("masked_url"),
            "raw_image_base64": item.get("raw_image_base64"),
            "masked_image_base64": item.get("masked_image_base64"),
            "upload_error": item.get("upload_error") or "",
            "pixel_hash": pixel_hash,
            "duplicate": duplicate,
            "image_embedding": embedding,
        })

    if not items:
        items = [{
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
            "raw_image_base64": None,
            "masked_image_base64": None,
            "upload_error": "",
            "pixel_hash": "",
            "duplicate": {"checked": False, "is_duplicate": False},
            "image_embedding": [],
        }]

    save_result = None
    save_state = "skipped"
    if bool(request.auto_save):
        try:
            save_candidates = [
                i for i in items
                if bool(request.save_duplicates) or not bool((i.get("duplicate") or {}).get("is_duplicate"))
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
            "background_removal": "ok" if any(i.get("masked_url") or i.get("masked_image_base64") for i in items) else "fallback",
            "r2_upload": "ok" if all(i.get("masked_url") for i in items) else "not_configured_or_failed",
            "vision_analyze": "ok" if any(i.get("label_source") == "vision" for i in items) else "fallback",
            "duplicate_detection": "ok" if any((i.get("duplicate") or {}).get("checked") for i in items) else "skipped",
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


@router.post("/save-selected")
def save_selected(http_request: Request, request: SaveSelectedRequest):
    user_id = _effective_user_id(http_request, request.user_id)
    return persist_selected_items(
        user_id=user_id,
        selected_item_ids=request.selected_item_ids,
        detected_items=request.detected_items,
    )
