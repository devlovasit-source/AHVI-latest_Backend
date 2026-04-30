import base64
import io
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import Image

from services import ai_gateway
from services.hybrid_detection_service import run_hybrid_detection
from services.image_embedding_service import encode_image_url
from services.image_fingerprint import compute_hash_from_url
from services.qdrant_service import qdrant_service
from services.wardrobe_persistence_service import persist_selected_items

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
    if any(x in raw for x in ["shirt", "tshirt", "t-shirt", "top", "blouse", "kurta", "sweater", "hoodie"]):
        return ("Tops", "Shirt")
    if any(x in raw for x in ["pant", "trouser", "jean", "skirt", "short"]):
        return ("Bottoms", "Pants")
    if any(x in raw for x in ["dress", "gown", "saree"]):
        return ("Dresses", "Dress")
    if any(x in raw for x in ["shoe", "sneaker", "heel", "boot", "sandal"]):
        return ("Footwear", "Shoes")
    if any(x in raw for x in ["watch", "bracelet", "ring", "earring", "necklace", "bag"]):
        return ("Accessories", "Accessory")
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


def _vision_extract_attributes(masked_url: str, fallback_label: str) -> Dict[str, Any]:
    base = {
        "name": str(fallback_label or "Item").strip().title() or "Item",
        "category": "",
        "sub_category": "",
        "pattern": "plain",
        "color_name": "",
    }
    if not masked_url:
        return base

    try:
        image_resp = requests.get(masked_url, timeout=8)
        image_resp.raise_for_status()
        image_b64 = base64.b64encode(image_resp.content).decode("utf-8")

        ai_json, _ = ai_gateway.ollama_vision_json(
            prompt=(
                "Return strict JSON with keys: name, category, sub_category, pattern, color_name. "
                "Values must be short strings for one clothing item."
            ),
            image_base64=image_b64,
            usecase="vision",
        )
        if isinstance(ai_json, dict):
            base.update({
                "name": str(ai_json.get("name") or base["name"]).strip()[:80] or base["name"],
                "category": str(ai_json.get("category") or "").strip()[:50],
                "sub_category": str(ai_json.get("sub_category") or "").strip()[:50],
                "pattern": str(ai_json.get("pattern") or base["pattern"]).strip().lower()[:40] or "plain",
                "color_name": str(ai_json.get("color_name") or "").strip().lower()[:40],
            })
    except Exception:
        pass

    return base


@router.post("/analyze")
async def analyze_capture(http_request: Request, request: CaptureAnalyzeRequest):
    user_id = _effective_user_id(http_request, request.user_id)
    image = _decode_image_base64(request.image_base64)
    source_bytes = _bytes_from_image_base64(request.image_base64)

    try:
        detected_items = await run_hybrid_detection(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    items = []
    for item in detected_items:
        raw_label = str(item.get("label") or "Item")
        category, sub_category = _normalize_category_from_label(raw_label)

        vision = _vision_extract_attributes(str(item.get("masked_url") or ""), raw_label)
        if vision.get("category"):
            category = str(vision.get("category"))
        if vision.get("sub_category"):
            sub_category = str(vision.get("sub_category"))

        color_code = _dominant_color_hex_from_url(str(item.get("masked_url") or ""))
        color_name = str(vision.get("color_name") or _hex_to_name(color_code))

        try:
            embedding = encode_image_url(item.get("masked_url")) if item.get("masked_url") else []
        except Exception:
            embedding = []

        pixel_hash = ""
        duplicate = {"checked": False, "is_duplicate": False}
        try:
            pixel_hash = await compute_hash_from_url(str(item.get("masked_url") or ""))
            duplicate = qdrant_service.find_pixel_duplicate(request.user_id, pixel_hash, max_distance=6)
        except Exception:
            duplicate = {"checked": False, "is_duplicate": False}

        items.append({
            "item_id": item.get("item_id") or str(uuid.uuid4()),
            "name": vision.get("name") or raw_label or "Item",
            "category": category,
            "sub_category": sub_category,
            "color_code": color_code,
            "color_name": color_name,
            "pattern": str(vision.get("pattern") or "plain"),
            "occasions": [],
            "confidence": float(item.get("score") or 0.8),
            "reasoning": "hybrid_detection+vision",
            "raw_url": item.get("raw_url"),
            "masked_url": item.get("masked_url"),
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
            "reasoning": "fallback_no_detection",
            "raw_url": None,
            "masked_url": None,
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
            "upload_to_rembg": "ok",
            "vision_analyze": "ok",
            "duplicate_detection": "ok",
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
