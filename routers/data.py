from typing import Any, Dict, List, Optional
import uuid
import re
import os
import base64
import time
import hashlib
from threading import Lock
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import requests

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError
from services.embedding_service import encode_metadata
from services.image_embedding_service import encode_image_base64, encode_image_url, encode_image_bytes
from services.image_fingerprint import (
    compute_pixel_hash_from_base64,
    compute_pixel_hash_from_url,
    compute_pixel_hash_from_bytes,
)
from services.qdrant_service import qdrant_service
from services.r2_storage import R2Storage, R2StorageError
from services import data_access_service

router = APIRouter(prefix="/api/data", tags=["data"])
proxy = AppwriteProxy()

_DUPLICATE_ONE_SHOT_TTL_SECONDS = max(10, int(os.getenv("OUTFIT_DUPLICATE_ONE_SHOT_TTL_SECONDS", "90")))
_DUPLICATE_ONE_SHOT_MAX = max(32, int(os.getenv("OUTFIT_DUPLICATE_ONE_SHOT_MAX", "512")))
_DUPLICATE_ONE_SHOT_CACHE: Dict[str, Dict[str, Any]] = {}
_DUPLICATE_ONE_SHOT_LOCK = Lock()

RESOURCE_ALIASES = {
    "meal_planner": "meal_plans",
    "meal": "meal_plans",
    "medicines": "meds",
    "medicine": "meds",
    "calendar": "plans",
    "workout": "workout_outfits",
    "workouts": "workout_outfits",
    "skincare": "skincare_profiles",
    "skin": "skincare_profiles",
    "skincare_profile": "skincare_profiles",
    "contacts": "users",
    "life_board": "life_boards",
    "lifeboard": "life_boards",
}


def _normalize_resource_key(resource: str) -> str:
    key = str(resource or "").strip()
    return RESOURCE_ALIASES.get(key, key)


def _authenticated_user_id(http_request: Request) -> str:
    state_user = getattr(http_request.state, "user", None)
    if isinstance(state_user, dict):
        user_id = str(
            state_user.get("user_id")
            or state_user.get("$id")
            or state_user.get("id")
            or ""
        ).strip()
        if user_id:
            return user_id
    raise HTTPException(status_code=401, detail="Authenticated user is required")


def _assert_user_match(auth_user_id: str, supplied_user_id: Any, *, field: str = "user_id") -> None:
    supplied = str(supplied_user_id or "").strip()
    if supplied and supplied != str(auth_user_id):
        raise HTTPException(
            status_code=403,
            detail=f"{field} does not match authenticated user",
        )


def _assert_payload_user_matches(auth_user_id: str, payload: Dict[str, Any]) -> None:
    for field in ("userId", "user_id", "userid", "userID"):
        if field in payload:
            _assert_user_match(auth_user_id, payload.get(field), field=field)


def _enforce_document_owner(resource: str, doc: Dict[str, Any], auth_user_id: str) -> None:
    if not isinstance(doc, dict):
        raise HTTPException(status_code=403, detail="Document owner could not be verified")
    if resource == "users":
        doc_id = str(doc.get("$id") or doc.get("id") or doc.get("userId") or "").strip()
        if doc_id:
            _assert_user_match(auth_user_id, doc_id, field="document owner")
        return
    user_field = proxy.user_field_map.get(resource)
    if not user_field:
        return
    owner = str(doc.get(user_field) or doc.get("userId") or "").strip()
    if not owner:
        raise HTTPException(status_code=403, detail="Document owner could not be verified")
    _assert_user_match(auth_user_id, owner, field="document owner")


def _stamp_authenticated_user(resource: str, payload: Dict[str, Any], auth_user_id: str) -> Dict[str, Any]:
    _assert_payload_user_matches(auth_user_id, payload)
    user_field = proxy.user_field_map.get(resource)
    if user_field:
        payload[user_field] = auth_user_id
    return payload


def _dup_cache_key(*, user_id: str, pixel_hash: str) -> str:
    uid = str(user_id or "").strip()
    ph = str(pixel_hash or "").strip().lower()
    if not uid or not ph:
        return ""
    digest = hashlib.sha1(f"{uid}|{ph}".encode("utf-8")).hexdigest()
    return f"dup1:{digest}"


def _dup_cache_put(*, key: str, result: Dict[str, Any]) -> None:
    if not key or not isinstance(result, dict):
        return
    now = time.time()
    with _DUPLICATE_ONE_SHOT_LOCK:
        _DUPLICATE_ONE_SHOT_CACHE[key] = {"ts": now, "result": dict(result)}
        if len(_DUPLICATE_ONE_SHOT_CACHE) > _DUPLICATE_ONE_SHOT_MAX:
            victims = sorted(
                _DUPLICATE_ONE_SHOT_CACHE.items(),
                key=lambda kv: float((kv[1] or {}).get("ts") or 0.0),
            )[: max(0, len(_DUPLICATE_ONE_SHOT_CACHE) - _DUPLICATE_ONE_SHOT_MAX)]
            for victim_key, _ in victims:
                _DUPLICATE_ONE_SHOT_CACHE.pop(victim_key, None)


def _dup_cache_pop(*, key: str) -> Dict[str, Any] | None:
    if not key:
        return None
    now = time.time()
    with _DUPLICATE_ONE_SHOT_LOCK:
        row = _DUPLICATE_ONE_SHOT_CACHE.pop(key, None)
    if not isinstance(row, dict):
        return None
    ts = float(row.get("ts") or 0.0)
    if ts <= 0.0 or (now - ts) > _DUPLICATE_ONE_SHOT_TTL_SECONDS:
        return None
    payload = row.get("result")
    return dict(payload) if isinstance(payload, dict) else None


class CreateRequest(BaseModel):
    resource: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    force_save: bool = False


class UpdateRequest(BaseModel):
    resource: str
    data: Dict[str, Any]


class DeleteRequest(BaseModel):
    resource: str
    document_id: str


class OutfitDuplicateCheckRequest(BaseModel):
    data: Dict[str, Any]
    user_id: Optional[str] = None


_HEX_COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{6})\b")
_UNKNOWN_ATTR_RE = re.compile(r'Unknown attribute:\s*"([^"]+)"')
_KNOWN_PATTERNS = {
    "plain",
    "solid",
    "striped",
    "checked",
    "checkered",
    "plaid",
    "floral",
    "printed",
    "polka",
    "graphic",
    "textured",
    "denim",
    "embroidered",
}
_CATEGORY_FALLBACK_SUB = {
    "tops": "Shirt",
    "bottoms": "Trousers",
    "outerwear": "Jacket",
    "footwear": "Shoes",
    "dresses": "Dress",
    "accessories": "Accessory",
    "bags": "Bag",
    "jewelry": "Jewelry",
    "makeup": "Makeup",
    "skincare": "Skincare",
}


def _build_sources(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = [payload]
    for key in ("analysis", "vision", "ai", "detected", "detected_item", "detectedItem", "data"):
        value = payload.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return sources


def _first_text(sources: List[Dict[str, Any]], *keys: str) -> str:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                text = value.strip()
                if text:
                    return text
            elif isinstance(value, (int, float, bool)):
                return str(value)
    return ""


def _to_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return []


def _first_list(sources: List[Dict[str, Any]], *keys: str) -> List[str]:
    for source in sources:
        for key in keys:
            items = _to_string_list(source.get(key))
            if items:
                return items
    return []


def _parse_notes_for_fields(notes: str) -> Dict[str, str]:
    text = str(notes or "").strip()
    if not text:
        return {"color_code": "", "pattern": ""}

    color = ""
    pattern = ""

    m = _HEX_COLOR_RE.search(text)
    if m:
        color = m.group(0).upper()

    lowered = text.lower()
    for token in _KNOWN_PATTERNS:
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            pattern = token
            break

    return {"color_code": color, "pattern": pattern}


def _guess_sub_category(name: str, category: str) -> str:
    text = str(name or "").strip()
    if text:
        words = re.findall(r"[A-Za-z]+", text)
        if words:
            candidate = words[-1].lower()
            if candidate in {
                "shirt",
                "tshirt",
                "blouse",
                "top",
                "jacket",
                "blazer",
                "trousers",
                "pants",
                "jeans",
                "shorts",
                "dress",
                "skirt",
                "shoes",
                "sneakers",
                "heels",
                "boots",
                "sandals",
                "bag",
                "watch",
            }:
                if candidate == "tshirt":
                    return "T-Shirt"
                return candidate.title()

    cat_key = str(category or "").strip().lower()
    return _CATEGORY_FALLBACK_SUB.get(cat_key, "")


def _normalize_outfit_payload(payload: Dict[str, Any], request_user_id: Optional[str]) -> Dict[str, Any]:
    normalized = dict(payload)
    sources = _build_sources(normalized)

    user_id = _first_text(sources, "userId", "user_id", "userid")
    if not user_id and request_user_id:
        user_id = str(request_user_id).strip()
    if user_id:
        normalized["userId"] = user_id

    image_url = _first_text(
        sources,
        "image_url",
        "imageUrl",
        "raw_image_url",
        "rawImageUrl",
        "raw_url",
        "rawUrl",
    )
    masked_url = _first_text(
        sources,
        "masked_url",
        "maskedUrl",
        "image_masked_url",
        "maskedImageUrl",
        "processed_image_url",
        "processedImageUrl",
    )
    image_id = _first_text(sources, "image_id", "imageId", "raw_id", "rawId", "file_id", "fileId")
    masked_id = _first_text(sources, "masked_id", "maskedId", "masked_file_id", "maskedFileId")
    category = _first_text(sources, "category", "type")
    sub_category = _first_text(sources, "sub_category", "subCategory", "subcategory", "subType")
    color_code = _first_text(sources, "color_code", "colorCode", "colour_code")
    pattern = _first_text(sources, "pattern", "fabric_pattern")
    name = _first_text(sources, "name", "title", "item_name", "itemName", "garment_name")
    notes = _first_text(sources, "notes", "note", "description")
    occasions = _first_list(sources, "occasions", "occasion", "occasions_list", "tags")
    qdrant_point_id = _first_text(sources, "qdrant_point_id", "qdrantPointId", "vector_id", "vectorId")

    parsed = _parse_notes_for_fields(notes)
    if not color_code and parsed["color_code"]:
        color_code = parsed["color_code"]
    if not pattern and parsed["pattern"]:
        pattern = parsed["pattern"]

    if not sub_category:
        sub_category = _guess_sub_category(name, category)

    if not name:
        if sub_category:
            name = sub_category
        elif category:
            name = category
        else:
            name = "Outfit"

    if not occasions:
        occasions = []

    # Keep required URL parity resilient across client payload variants.
    if not image_url and masked_url:
        image_url = masked_url
    if not masked_url and image_url:
        masked_url = image_url

    if image_url:
        normalized["image_url"] = image_url
    if masked_url:
        normalized["masked_url"] = masked_url
    if image_id:
        normalized["image_id"] = image_id
    if masked_id:
        normalized["masked_id"] = masked_id
    if category:
        normalized["category"] = category
    if sub_category:
        normalized["sub_category"] = sub_category
    if color_code:
        normalized["color_code"] = color_code.upper()
    if pattern:
        normalized["pattern"] = pattern.lower()
    if qdrant_point_id:
        normalized["qdrant_point_id"] = qdrant_point_id

    normalized["name"] = name
    normalized["occasions"] = occasions
    # Notes attribute is removed from Appwrite outfits schema; consume for parsing only.
    normalized.pop("notes", None)
    normalized.pop("note", None)
    normalized.pop("description", None)
    # Duplicate-check helper fields should not be written to Appwrite.
    normalized.pop("pixel_hash", None)
    normalized.pop("pixelHash", None)
    normalized.pop("masked_pixel_hash", None)
    normalized.pop("maskedPixelHash", None)
    normalized.pop("raw_pixel_hash", None)
    normalized.pop("rawPixelHash", None)
    normalized.pop("processed_image_base64", None)
    normalized.pop("masked_image_base64", None)
    normalized.pop("image_base64", None)
    normalized.pop("image_vector", None)
    normalized.pop("imageVector", None)

    return normalized


def _unknown_attribute_from_error(exc: AppwriteProxyError) -> str:
    msg = str(exc or "")
    matched = _UNKNOWN_ATTR_RE.search(msg)
    return (matched.group(1) if matched else "").strip()


def _create_document_with_schema_retries(
    *,
    resource: str,
    payload: Dict[str, Any],
    document_id: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    safe_payload = dict(payload or {})
    for _ in range(max_retries):
        try:
            return data_access_service.create_document(
                resource=resource,
                payload=safe_payload,
                document_id=document_id,
            )
        except AppwriteProxyError as exc:
            unknown_attr = _unknown_attribute_from_error(exc)
            if unknown_attr and unknown_attr in safe_payload:
                # Schema may drift over time; drop unknown keys and retry.
                safe_payload.pop(unknown_attr, None)
                continue
            raise
    return data_access_service.create_document(
        resource=resource,
        payload=safe_payload,
        document_id=document_id,
    )


def _update_document_with_schema_retries(
    *,
    resource: str,
    document_id: str,
    payload: Dict[str, Any],
    max_retries: int = 5,
) -> Dict[str, Any]:
    safe_payload = dict(payload or {})
    for _ in range(max_retries):
        try:
            return data_access_service.update_document(
                resource=resource,
                document_id=document_id,
                payload=safe_payload,
            )
        except AppwriteProxyError as exc:
            unknown_attr = _unknown_attribute_from_error(exc)
            if unknown_attr and unknown_attr in safe_payload:
                safe_payload.pop(unknown_attr, None)
                continue
            raise
    return data_access_service.update_document(
        resource=resource,
        document_id=document_id,
        payload=safe_payload,
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value or "").strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def _normalize_meal_plan_payload(payload: Dict[str, Any], request_user_id: Optional[str]) -> Dict[str, Any]:
    normalized = dict(payload or {})

    # userId (required)
    user_id = (
        str(
            normalized.get("userId")
            or normalized.get("user_id")
            or normalized.get("userid")
            or request_user_id
            or ""
        ).strip()
    )
    if user_id:
        normalized["userId"] = user_id

    # Required schema fields
    name = str(
        normalized.get("name")
        or normalized.get("title")
        or normalized.get("planName")
        or normalized.get("dietName")
        or ""
    ).strip()
    desc = str(
        normalized.get("desc")
        or normalized.get("description")
        or normalized.get("notes")
        or ""
    ).strip()
    plan_type = str(
        normalized.get("planType")
        or normalized.get("plan_type")
        or normalized.get("type")
        or normalized.get("dietType")
        or "diet"
    ).strip()
    total_cal = _safe_int(
        normalized.get("totalCal")
        if normalized.get("totalCal") is not None
        else normalized.get("total_cal", normalized.get("calories", 0)),
        default=0,
    )

    # meals[] column is a string in your table. Persist as compact JSON string.
    meals_value = normalized.get("meals[]", None)
    if meals_value is None:
        meals_value = normalized.get("meals", normalized.get("items", []))
    if isinstance(meals_value, str):
        meals_str = meals_value.strip()
    else:
        try:
            meals_str = json.dumps(meals_value if meals_value is not None else [], ensure_ascii=False, separators=(",", ":"))
        except Exception:
            meals_str = "[]"

    normalized["name"] = name or "Diet Plan"
    normalized["desc"] = desc or "Diet plan details"
    normalized["planType"] = plan_type or "diet"
    normalized["totalCal"] = int(total_cal)
    normalized["meals[]"] = meals_str

    # Remove aliases/unsupported keys that can break strict schemas.
    normalized.pop("title", None)
    normalized.pop("description", None)
    normalized.pop("notes", None)
    normalized.pop("plan_type", None)
    normalized.pop("dietType", None)
    normalized.pop("total_cal", None)
    normalized.pop("calories", None)
    normalized.pop("meals", None)
    normalized.pop("items", None)
    normalized.pop("user_id", None)
    normalized.pop("userid", None)

    return normalized


def _to_uuid_point_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(raw))
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _basename_from_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        path = parsed.path or text
    except Exception:
        path = text
    return path.rsplit("/", 1)[-1].strip()


def _normalize_object_name(value: Any) -> str:
    text = str(value or "").strip().strip("/")
    if not text:
        return ""
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if "?" in text:
        text = text.split("?", 1)[0]
    if "#" in text:
        text = text.split("#", 1)[0]
    return text.strip()


def _derive_prefixed_png_name(value: Any, prefix: str) -> str:
    base = _normalize_object_name(value)
    if not base:
        return ""
    if "." in base:
        return base
    if not base.startswith(prefix):
        base = f"{prefix}{base}"
    return f"{base}.png"


def _collect_outfit_r2_candidates(doc: Dict[str, Any]) -> Dict[str, List[str]]:
    raw_candidates: List[str] = []
    masked_candidates: List[str] = []

    def add_unique(target: List[str], value: Any):
        name = _normalize_object_name(value)
        if name and name not in target:
            target.append(name)

    image_id = doc.get("image_id")
    masked_id = doc.get("masked_id")
    image_url = doc.get("image_url")
    masked_url = doc.get("masked_url")
    seed = str(doc.get("qdrant_point_id") or "").strip() or str(doc.get("$id") or "").strip()

    add_unique(raw_candidates, image_id)
    add_unique(masked_candidates, masked_id)
    add_unique(raw_candidates, _basename_from_url(image_url))
    add_unique(masked_candidates, _basename_from_url(masked_url))

    add_unique(raw_candidates, _derive_prefixed_png_name(image_id, "raw_"))
    add_unique(masked_candidates, _derive_prefixed_png_name(masked_id, "wardrobe_"))
    if seed:
        add_unique(raw_candidates, _derive_prefixed_png_name(seed, "raw_"))
        add_unique(masked_candidates, _derive_prefixed_png_name(seed, "wardrobe_"))

    return {
        "raw": raw_candidates,
        "masked": masked_candidates,
    }


def _guess_mime_from_name(name: str) -> str:
    lowered = str(name or "").strip().lower()
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return "image/jpeg"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".gif"):
        return "image/gif"
    return "image/png"


def _build_existing_preview_data_url(doc: Dict[str, Any]) -> Optional[str]:
    if not isinstance(doc, dict) or not doc:
        return None
    max_preview_bytes = 8 * 1024 * 1024
    try:
        storage = R2Storage()
        candidates = _collect_outfit_r2_candidates(doc)
        for name in candidates.get("masked", []):
            content = storage.read_object_bytes(
                bucket=storage.wardrobe_bucket,
                object_name=name,
                max_bytes=max_preview_bytes,
            )
            if content:
                mime = _guess_mime_from_name(name)
                encoded = base64.b64encode(content).decode("utf-8")
                return f"data:{mime};base64,{encoded}"
        for name in candidates.get("raw", []):
            content = storage.read_object_bytes(
                bucket=storage.raw_bucket,
                object_name=name,
                max_bytes=max_preview_bytes,
            )
            if content:
                mime = _guess_mime_from_name(name)
                encoded = base64.b64encode(content).decode("utf-8")
                return f"data:{mime};base64,{encoded}"
    except Exception:
        pass

    # Fallback: fetch from public URL server-side and embed as data URL.
    preview_url = _extract_preview_url(doc)
    if preview_url.startswith("http://") or preview_url.startswith("https://"):
        try:
            res = requests.get(preview_url, timeout=10)
            if res.status_code == 200 and res.content:
                content = res.content[: max_preview_bytes + 1]
                if len(content) <= max_preview_bytes:
                    content_type = str(res.headers.get("Content-Type") or "").strip().lower()
                    mime = content_type if content_type.startswith("image/") else _guess_mime_from_name(preview_url)
                    encoded = base64.b64encode(content).decode("utf-8")
                    return f"data:{mime};base64,{encoded}"
        except Exception:
            pass
    return None


def _extract_preview_url(doc: Dict[str, Any]) -> str:
    if not isinstance(doc, dict):
        return ""
    def is_http_url(value: Any) -> bool:
        text = str(value or "").strip()
        return text.startswith("http://") or text.startswith("https://")

    # Prefer actual URL fields first.
    for key in (
        "masked_url",
        "maskedUrl",
        "image_masked_url",
        "maskedImageUrl",
        "image_url",
        "imageUrl",
    ):
        value = doc.get(key)
        if is_http_url(value):
            return str(value).strip()

    return ""


def _hydrate_existing_outfit_document(
    *,
    duplicate_user_id: str,
    duplicate_id: str,
    existing_doc: Dict[str, Any],
    hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = dict(existing_doc or {})
    if _extract_preview_url(merged):
        return merged

    candidate_ids: List[str] = []
    for key in ("$id", "id", "document_id", "documentId"):
        value = str(merged.get(key) or "").strip()
        if value and value not in candidate_ids:
            candidate_ids.append(value)
    normalized_duplicate_id = str(duplicate_id or "").strip()
    if normalized_duplicate_id and normalized_duplicate_id not in candidate_ids:
        candidate_ids.append(normalized_duplicate_id)

    for candidate_id in candidate_ids:
        try:
            app_doc = proxy.get_document("outfits", candidate_id)
        except AppwriteProxyError:
            app_doc = {}
        if not isinstance(app_doc, dict) or not app_doc:
            continue
        merged.update(app_doc)
        if _extract_preview_url(merged):
            return merged

    user_id = str(duplicate_user_id or "").strip()
    point_id = str(duplicate_id or "").strip()
    if not user_id:
        return merged

    docs: List[Dict[str, Any]] = []
    try:
        offset = 0
        scan_cap = 2000
        page_size = 100
        while len(docs) < scan_cap:
            page = proxy.list_documents(
                "outfits",
                user_id=user_id,
                limit=page_size,
                offset=offset,
                return_meta=True,
            )
            if not isinstance(page, dict):
                break
            page_docs = page.get("documents", [])
            if not isinstance(page_docs, list) or not page_docs:
                break
            docs.extend([d for d in page_docs if isinstance(d, dict)])

            meta = page.get("meta", {}) if isinstance(page.get("meta", {}), dict) else {}
            has_more = bool(meta.get("has_more"))
            next_offset = meta.get("next_offset")
            if not has_more or next_offset is None:
                break
            try:
                offset = int(next_offset)
            except Exception:
                break
    except AppwriteProxyError:
        docs = []

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_point_id = str(doc.get("qdrant_point_id") or doc.get("qdrantPointId") or "").strip()
        doc_id = str(doc.get("$id") or "").strip()
        if point_id and point_id in {doc_point_id, doc_id}:
            merged.update(doc)
            masked_url = str(doc.get("masked_url") or "").strip()
            image_url = str(doc.get("image_url") or "").strip()
            print(
                f"[data.duplicate_preview] appwrite_match point_id={point_id} "
                f"doc_id={doc_id} masked_http={masked_url.startswith('http')} "
                f"image_http={image_url.startswith('http')}"
            )
            break

    if _extract_preview_url(merged):
        return merged

    hint_map = dict(hints or {})
    hint_name = str(hint_map.get("name") or "").strip().lower()
    hint_category = str(hint_map.get("category") or "").strip().lower()
    hint_sub_category = str(hint_map.get("sub_category") or hint_map.get("subCategory") or "").strip().lower()
    hint_color_code = str(hint_map.get("color_code") or hint_map.get("colorCode") or "").strip().lower()

    best_doc: Dict[str, Any] = {}
    best_score = -1
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if not _extract_preview_url(doc):
            continue

        score = 0
        doc_name = str(doc.get("name") or "").strip().lower()
        doc_category = str(doc.get("category") or "").strip().lower()
        doc_sub_category = str(doc.get("sub_category") or "").strip().lower()
        doc_color_code = str(doc.get("color_code") or "").strip().lower()

        if hint_name:
            if doc_name == hint_name:
                score += 4
            elif hint_name in doc_name or doc_name in hint_name:
                score += 2
        if hint_category and doc_category == hint_category:
            score += 3
        if hint_sub_category and doc_sub_category == hint_sub_category:
            score += 3
        if hint_color_code and doc_color_code == hint_color_code:
            score += 2

        if score > best_score:
            best_score = score
            best_doc = doc

    if best_doc:
        merged.update(best_doc)
        masked_url = str(best_doc.get("masked_url") or "").strip()
        image_url = str(best_doc.get("image_url") or "").strip()
        print(
            f"[data.duplicate_preview] appwrite_hint_match "
            f"doc_id={best_doc.get('$id')} masked_http={masked_url.startswith('http')} "
            f"image_http={image_url.startswith('http')}"
        )
        return merged

    for doc in docs:
        if isinstance(doc, dict) and _extract_preview_url(doc):
            merged.update(doc)
            masked_url = str(doc.get("masked_url") or "").strip()
            image_url = str(doc.get("image_url") or "").strip()
            print(
                f"[data.duplicate_preview] appwrite_any_match "
                f"doc_id={doc.get('$id')} masked_http={masked_url.startswith('http')} "
                f"image_http={image_url.startswith('http')}"
            )
            return merged

    print(
        f"[data.duplicate_preview] appwrite_no_preview point_id={point_id} user_id={user_id}"
    )

    return merged


def _duplicate_threshold() -> float:
    raw = str(os.getenv("WARDROBE_DUPLICATE_THRESHOLD", "0.97") or "").strip()
    try:
        value = float(raw)
        if value <= 0.0:
            return 0.97
        if value > 1.0:
            return 1.0
        return value
    except Exception:
        return 0.97


def _pixel_duplicate_distance() -> int:
    raw = str(os.getenv("WARDROBE_PIXEL_DUPLICATE_DISTANCE", "6") or "").strip()
    try:
        value = int(raw)
        if value < 0:
            return 0
        if value > 64:
            return 64
        return value
    except Exception:
        return 6


def _image_duplicate_threshold() -> float:
    raw = str(os.getenv("WARDROBE_IMAGE_DUPLICATE_THRESHOLD", "0.985") or "").strip()
    try:
        value = float(raw)
        if value <= 0.0:
            return 0.985
        if value > 1.0:
            return 1.0
        return value
    except Exception:
        return 0.985


def _coerce_vector(value: Any) -> list:
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                return []
        return out
    return []


def _compute_payload_image_vector(payload: Dict[str, Any]) -> list:
    vector = _coerce_vector(payload.get("image_vector"))
    if vector:
        return vector
    vector = _coerce_vector(payload.get("imageVector"))
    if vector:
        return vector

    for key in ("masked_image_base64", "maskedImageBase64", "processed_image_base64", "image_base64", "imageBase64"):
        value = payload.get(key)
        if value:
            vector = encode_image_base64(value)
            if vector:
                return vector

    # Robust fallback: read just-uploaded bytes directly from R2 object IDs
    # to avoid URL propagation/timing issues.
    try:
        storage = R2Storage()
        candidates = _collect_outfit_r2_candidates(payload)
        for name in candidates.get("masked", []):
            image_bytes = storage.read_object_bytes(
                bucket=storage.wardrobe_bucket,
                object_name=name,
            )
            if not image_bytes:
                continue
            vector = encode_image_bytes(image_bytes)
            if vector:
                return vector
        for name in candidates.get("raw", []):
            image_bytes = storage.read_object_bytes(
                bucket=storage.raw_bucket,
                object_name=name,
            )
            if not image_bytes:
                continue
            vector = encode_image_bytes(image_bytes)
            if vector:
                return vector
    except Exception:
        pass

    for key in ("masked_url", "maskedUrl", "image_url", "imageUrl"):
        value = payload.get(key)
        if value:
            vector = encode_image_url(value)
            if vector:
                return vector

    return []


def _compute_payload_pixel_hash(payload: Dict[str, Any]) -> str:
    for key in ("pixel_hash", "pixelHash", "masked_pixel_hash", "maskedPixelHash"):
        value = str(payload.get(key) or "").strip().lower()
        if value:
            return value

    for key in ("masked_image_base64", "maskedImageBase64", "processed_image_base64", "image_base64", "imageBase64"):
        value = payload.get(key)
        if value:
            pixel_hash = compute_pixel_hash_from_base64(value)
            if pixel_hash:
                return pixel_hash

    # Robust fallback: read from R2 objects by file ids before URL download.
    try:
        storage = R2Storage()
        candidates = _collect_outfit_r2_candidates(payload)
        for name in candidates.get("masked", []):
            image_bytes = storage.read_object_bytes(
                bucket=storage.wardrobe_bucket,
                object_name=name,
            )
            if not image_bytes:
                continue
            pixel_hash = compute_pixel_hash_from_bytes(image_bytes)
            if pixel_hash:
                return pixel_hash
        for name in candidates.get("raw", []):
            image_bytes = storage.read_object_bytes(
                bucket=storage.raw_bucket,
                object_name=name,
            )
            if not image_bytes:
                continue
            pixel_hash = compute_pixel_hash_from_bytes(image_bytes)
            if pixel_hash:
                return pixel_hash
    except Exception:
        pass

    for key in ("masked_url", "maskedUrl", "image_url", "imageUrl"):
        value = payload.get(key)
        if value:
            pixel_hash = compute_pixel_hash_from_url(value)
            if pixel_hash:
                return pixel_hash

    return ""


def _new_duplicate_meta(force_save: bool = False) -> Dict[str, Any]:
    return {
        "checked": False,
        "is_duplicate": False,
        "score": 0.0,
        "threshold": _duplicate_threshold(),
        "point_id": None,
        "forced_save": bool(force_save),
        "reason": None,
        "image_checked": False,
        "image_score": 0.0,
        "image_threshold": _image_duplicate_threshold(),
        "pixel_hash": None,
        "pixel_checked": False,
        "pixel_distance": None,
        "pixel_max_distance": _pixel_duplicate_distance(),
        "error": None,
    }


def _run_outfit_duplicate_check(
    *,
    payload: Dict[str, Any],
    duplicate_user_id: str,
    incoming_image_vector: Optional[list] = None,
    incoming_pixel_hash: str = "",
) -> Dict[str, Any]:
    duplicate_meta = _new_duplicate_meta(force_save=False)
    duplicate_meta["checked"] = bool(duplicate_user_id)
    payload_image_vector: List[float] = []
    payload_pixel_hash = ""
    duplicate: Dict[str, Any] = {}

    if not duplicate_user_id:
        return {
            "duplicate_meta": duplicate_meta,
            "duplicate": duplicate,
            "payload_image_vector": payload_image_vector,
            "payload_pixel_hash": payload_pixel_hash,
        }

    try:
        payload_image_vector = incoming_image_vector or _compute_payload_image_vector(payload)
        if payload_image_vector:
            image_duplicate = qdrant_service.find_image_duplicate(
                payload_image_vector,
                duplicate_user_id,
                threshold=duplicate_meta["image_threshold"],
            )
            duplicate_meta["image_checked"] = bool(image_duplicate.get("checked"))
            duplicate_meta["image_score"] = float(image_duplicate.get("score") or 0.0)
            if image_duplicate.get("is_duplicate"):
                duplicate_meta["is_duplicate"] = True
                duplicate_meta["point_id"] = image_duplicate.get("id")
                duplicate_meta["reason"] = "image_vector"
                duplicate = dict(image_duplicate or {})

        payload_pixel_hash = str(incoming_pixel_hash or "").strip().lower() or _compute_payload_pixel_hash(payload)
        duplicate_meta["pixel_hash"] = payload_pixel_hash or None
        if payload_pixel_hash and not duplicate_meta["is_duplicate"]:
            pixel_duplicate = qdrant_service.find_pixel_duplicate(
                duplicate_user_id,
                payload_pixel_hash,
                max_distance=duplicate_meta["pixel_max_distance"],
            )
            duplicate_meta["pixel_checked"] = bool(pixel_duplicate.get("checked"))
            duplicate_meta["pixel_distance"] = pixel_duplicate.get("distance")
            if pixel_duplicate.get("is_duplicate"):
                duplicate_meta["is_duplicate"] = True
                duplicate_meta["point_id"] = pixel_duplicate.get("id")
                duplicate_meta["reason"] = "pixel_hash"
                duplicate = dict(pixel_duplicate or {})

        duplicate_vector_input = {
            "category": payload.get("category", ""),
            "sub_category": payload.get("sub_category", ""),
            "color_code": payload.get("color_code", ""),
            "pattern": payload.get("pattern", ""),
            "occasions": payload.get("occasions", [])
            if isinstance(payload.get("occasions", []), list)
            else [],
        }
        if not duplicate_meta["is_duplicate"]:
            duplicate_vector = encode_metadata(duplicate_vector_input)
            semantic_duplicate = qdrant_service.find_duplicate(
                duplicate_vector,
                duplicate_user_id,
                threshold=duplicate_meta["threshold"],
            )
            duplicate_meta["is_duplicate"] = bool(semantic_duplicate.get("is_duplicate"))
            duplicate_meta["score"] = float(semantic_duplicate.get("score") or 0.0)
            duplicate_meta["point_id"] = semantic_duplicate.get("id")
            if duplicate_meta["is_duplicate"]:
                duplicate_meta["reason"] = "semantic"
                duplicate = dict(semantic_duplicate or {})
        else:
            duplicate_meta["score"] = 1.0
    except Exception as exc:
        duplicate_meta["error"] = str(exc)

    return {
        "duplicate_meta": duplicate_meta,
        "duplicate": duplicate,
        "payload_image_vector": payload_image_vector,
        "payload_pixel_hash": payload_pixel_hash,
    }


def _build_existing_duplicate_preview(
    *,
    duplicate_user_id: str,
    duplicate_meta: Dict[str, Any],
    duplicate: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    duplicate_id = str(duplicate_meta.get("point_id") or "").strip()
    hints: Dict[str, Any] = {}
    payload = (duplicate or {}).get("payload") if isinstance(duplicate, dict) else None
    if isinstance(payload, dict):
        for key in ("name", "category", "sub_category", "color_code"):
            value = payload.get(key)
            if value is not None and str(value).strip():
                hints[key] = value

    existing_doc = _hydrate_existing_outfit_document(
        duplicate_user_id=duplicate_user_id,
        duplicate_id=duplicate_id,
        existing_doc={},
        hints=hints,
    )
    existing_preview_url = _extract_preview_url(existing_doc)
    existing_preview_data_url = _build_existing_preview_data_url(existing_doc)
    print(
        f"[data.duplicate_preview] selected "
        f"doc_id={existing_doc.get('$id')} preview_url={'yes' if existing_preview_url else 'no'} "
        f"preview_http={str(existing_preview_url or '').startswith('http')} "
        f"preview_data={'yes' if existing_preview_data_url else 'no'} "
        f"preview_data_len={len(existing_preview_data_url or '')} "
        f"preview_value={existing_preview_url}"
    )
    return {
        "existing_document": existing_doc,
        "existing_preview_url": existing_preview_url or None,
        "existing_preview_data_url": existing_preview_data_url,
    }


def _http_error_from_proxy(exc: AppwriteProxyError) -> HTTPException:
    msg = str(exc)
    if "connection failed" in msg.lower():
        return HTTPException(status_code=503, detail=msg)
    if "(404)" in msg:
        return HTTPException(status_code=404, detail=msg)
    if "(401)" in msg or "(403)" in msg:
        return HTTPException(status_code=502, detail=msg)
    return HTTPException(status_code=400, detail=msg)


@router.get("/{resource}")
def list_documents(
    http_request: Request,
    resource: str,
    user_id: Optional[str] = None,
    occasion: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    normalized_resource = _normalize_resource_key(resource)
    auth_user_id = _authenticated_user_id(http_request)
    _assert_user_match(auth_user_id, user_id)
    try:
        if normalized_resource == "users":
            doc = proxy.get_document("users", auth_user_id)
            return {
                "documents": [doc],
                "meta": {
                    "limit": 1,
                    "offset": 0,
                    "has_more": False,
                    "next_offset": None,
                    "total": 1,
                    "mode": "owner_document",
                },
            }
        page = proxy.list_documents(
            normalized_resource,
            user_id=auth_user_id,
            occasion=occasion,
            limit=limit,
            offset=offset,
            return_meta=True,
        )
        if isinstance(page, dict):
            docs = page.get("documents", [])
            meta = page.get("meta", {})
            print(
                f"[data.list_documents] resource={resource} normalized={normalized_resource} user_id={auth_user_id} "
                f"offset={offset} limit={limit} returned={len(docs)} "
                f"mode={meta.get('mode')} total={meta.get('total')}"
            )
            return page
        return {
            "documents": page,
            "meta": {
                "limit": max(1, int(limit)),
                "offset": int(offset),
                "has_more": len(page) >= max(1, int(limit)),
                "next_offset": int(offset) + len(page),
                "total": None,
            },
        }
    except AppwriteProxyError as exc:
        print(f"[data.list_documents] resource={resource} normalized={normalized_resource} user_id={auth_user_id} error={exc}")
        raise _http_error_from_proxy(exc)


@router.get("/{resource}/{document_id}")
def get_document(http_request: Request, resource: str, document_id: str):
    normalized_resource = _normalize_resource_key(resource)
    auth_user_id = _authenticated_user_id(http_request)
    if normalized_resource == "users":
        _assert_user_match(auth_user_id, document_id, field="user_id")
    try:
        doc = proxy.get_document(normalized_resource, document_id)
        _enforce_document_owner(normalized_resource, doc, auth_user_id)
        return {"document": doc}
    except AppwriteProxyError as exc:
        if normalized_resource == "users" and "(404)" in str(exc):
            default_doc = {
                "userId": auth_user_id,
                "name": "",
                "username": "",
                "phone": "",
                "gender": "",
                "skinTone": 3,
                "bodyShape": "",
                "stylePreferences": [],
            }
            try:
                created = _create_document_with_schema_retries(
                    resource="users",
                    payload=default_doc,
                    document_id=document_id,
                )
                return {"document": created}
            except AppwriteProxyError as create_exc:
                print(
                    f"[data.get_document] resource={resource} document_id={document_id} "
                    f"create_on_missing error={create_exc}"
                )
                raise _http_error_from_proxy(create_exc)
        print(f"[data.get_document] resource={resource} document_id={document_id} error={exc}")
        raise _http_error_from_proxy(exc)


@router.post("/outfits/duplicate-check")
def check_outfit_duplicate(http_request: Request, request: OutfitDuplicateCheckRequest):
    auth_user_id = _authenticated_user_id(http_request)
    raw_payload = dict(request.data or {})
    _assert_user_match(auth_user_id, request.user_id)
    _assert_payload_user_matches(auth_user_id, raw_payload)
    if not qdrant_service.enabled():
        return {
            "checked": False,
            "duplicate": _new_duplicate_meta(force_save=False),
            "existing_document": {},
            "existing_preview_url": None,
            "existing_preview_data_url": None,
        }
    incoming_image_vector = (
        _coerce_vector(raw_payload.get("image_vector"))
        or _coerce_vector(raw_payload.get("imageVector"))
        or _compute_payload_image_vector(raw_payload)
    )
    incoming_pixel_hash = (
        str(raw_payload.get("pixel_hash") or "")
        or str(raw_payload.get("pixelHash") or "")
        or str(raw_payload.get("masked_pixel_hash") or "")
        or str(raw_payload.get("maskedPixelHash") or "")
    ).strip().lower()

    payload = _normalize_outfit_payload(raw_payload, auth_user_id)
    payload["userId"] = auth_user_id
    duplicate_user_id = auth_user_id
    result = _run_outfit_duplicate_check(
        payload=payload,
        duplicate_user_id=duplicate_user_id,
        incoming_image_vector=incoming_image_vector,
        incoming_pixel_hash=incoming_pixel_hash,
    )
    cache_pixel_hash = str(result.get("payload_pixel_hash") or "").strip().lower()
    cache_key = _dup_cache_key(user_id=duplicate_user_id, pixel_hash=cache_pixel_hash)
    if cache_key:
        _dup_cache_put(key=cache_key, result=result)
    duplicate_meta = dict(result.get("duplicate_meta") or {})
    duplicate = dict(result.get("duplicate") or {})

    print(
        f"[data.duplicate_check] outfits user={duplicate_user_id} "
        f"image_checked={duplicate_meta.get('image_checked')} "
        f"image_score={duplicate_meta.get('image_score')} "
        f"pixel_checked={duplicate_meta.get('pixel_checked')} "
        f"semantic_score={duplicate_meta.get('score')} "
        f"is_duplicate={duplicate_meta.get('is_duplicate')} "
        f"reason={duplicate_meta.get('reason')}"
    )

    existing_payload = {"existing_document": {}, "existing_preview_url": None}
    if duplicate_meta.get("is_duplicate"):
        existing_payload = _build_existing_duplicate_preview(
            duplicate_user_id=duplicate_user_id,
            duplicate_meta=duplicate_meta,
            duplicate=duplicate,
        )

    return {
        "checked": bool(duplicate_user_id),
        "duplicate": duplicate_meta,
        "existing_document": existing_payload.get("existing_document") or {},
        "existing_preview_url": existing_payload.get("existing_preview_url"),
        "existing_preview_data_url": existing_payload.get("existing_preview_data_url"),
    }


@router.post("")
def create_document(http_request: Request, request: CreateRequest):
    auth_user_id = _authenticated_user_id(http_request)
    resource = _normalize_resource_key(request.resource)
    document_id = request.document_id or "unique()"
    try:
        payload = dict(request.data)
        _assert_user_match(auth_user_id, request.user_id)
        _assert_payload_user_matches(auth_user_id, payload)
        if resource == "users":
            _assert_user_match(auth_user_id, request.document_id, field="document_id")
            document_id = auth_user_id
        if resource == "outfits":
            print(
                f"[data.create_document] outfits request "
                f"force_save={bool(request.force_save)} "
                f"user_id={auth_user_id}"
            )
        qdrant_enabled = qdrant_service.enabled()
        incoming_image_vector: list = []
        if resource == "outfits" and qdrant_enabled:
            incoming_image_vector = (
                _coerce_vector(payload.get("image_vector"))
                or _coerce_vector(payload.get("imageVector"))
                or _compute_payload_image_vector(payload)
            )
        incoming_pixel_hash = (
            str(payload.get("pixel_hash") or "")
            or str(payload.get("pixelHash") or "")
            or str(payload.get("masked_pixel_hash") or "")
            or str(payload.get("maskedPixelHash") or "")
        ).strip().lower()
        payload_image_vector: list = []
        payload_pixel_hash = ""
        duplicate_meta: Dict[str, Any] = {
            "checked": False,
            "is_duplicate": False,
            "score": 0.0,
            "threshold": _duplicate_threshold(),
            "point_id": None,
            "forced_save": bool(request.force_save),
            "reason": None,
            "image_checked": False,
            "image_score": 0.0,
            "image_threshold": _image_duplicate_threshold(),
            "pixel_hash": None,
            "pixel_checked": False,
            "pixel_distance": None,
            "pixel_max_distance": _pixel_duplicate_distance(),
            "error": None,
        }

        if resource == "outfits":
            payload = _normalize_outfit_payload(payload, auth_user_id)
            payload["userId"] = auth_user_id

            duplicate_user_id = auth_user_id
            duplicate_meta["checked"] = bool(duplicate_user_id)
            if duplicate_user_id and not request.force_save and qdrant_enabled:
                duplicate_result = None
                if incoming_pixel_hash:
                    cached_key = _dup_cache_key(user_id=duplicate_user_id, pixel_hash=incoming_pixel_hash)
                    duplicate_result = _dup_cache_pop(key=cached_key)
                if not isinstance(duplicate_result, dict):
                    duplicate_result = _run_outfit_duplicate_check(
                        payload=payload,
                        duplicate_user_id=duplicate_user_id,
                        incoming_image_vector=incoming_image_vector,
                        incoming_pixel_hash=incoming_pixel_hash,
                    )
                else:
                    print(f"[data.create_document] outfits duplicate_check cache_hit user={duplicate_user_id}")
                duplicate_meta = dict(duplicate_result.get("duplicate_meta") or duplicate_meta)
                duplicate_meta["forced_save"] = bool(request.force_save)
                payload_image_vector = list(duplicate_result.get("payload_image_vector") or [])
                payload_pixel_hash = str(duplicate_result.get("payload_pixel_hash") or "").strip().lower()
                duplicate = dict(duplicate_result.get("duplicate") or {})

                if duplicate_meta.get("is_duplicate"):
                    print(
                        f"[data.create_document] outfits duplicate_detected "
                        f"user={duplicate_user_id} reason={duplicate_meta.get('reason')} "
                        f"point_id={duplicate_meta.get('point_id')} "
                        f"image_score={duplicate_meta.get('image_score')} "
                        f"pixel_distance={duplicate_meta.get('pixel_distance')} "
                        f"semantic_score={duplicate_meta.get('score')}"
                    )
                    cleanup_meta = {
                        "r2_cleanup_attempted": False,
                        "r2_raw_deleted": False,
                        "r2_masked_deleted": False,
                        "r2_cleanup_error": None,
                    }
                    try:
                        raw_candidates: List[str] = []
                        masked_candidates: List[str] = []

                        def add_unique(target: List[str], value: Any):
                            name = _normalize_object_name(value)
                            if name and name not in target:
                                target.append(name)

                        add_unique(raw_candidates, payload.get("image_id"))
                        add_unique(masked_candidates, payload.get("masked_id"))
                        add_unique(raw_candidates, _basename_from_url(payload.get("image_url")))
                        add_unique(masked_candidates, _basename_from_url(payload.get("masked_url")))
                        add_unique(raw_candidates, _derive_prefixed_png_name(payload.get("image_id"), "raw_"))
                        add_unique(masked_candidates, _derive_prefixed_png_name(payload.get("masked_id"), "wardrobe_"))

                        if raw_candidates or masked_candidates:
                            cleanup_meta["r2_cleanup_attempted"] = True
                            storage = R2Storage()

                            for raw_name in raw_candidates:
                                result = storage.delete_wardrobe_images(raw_file_name=raw_name)
                                if result.get("raw_deleted"):
                                    cleanup_meta["r2_raw_deleted"] = True
                                    break

                            for masked_name in masked_candidates:
                                result = storage.delete_wardrobe_images(masked_file_name=masked_name)
                                if result.get("masked_deleted"):
                                    cleanup_meta["r2_masked_deleted"] = True
                                    break
                    except (R2StorageError, Exception) as cleanup_exc:
                        cleanup_meta["r2_cleanup_error"] = str(cleanup_exc)
                    print(
                        f"[data.create_document] outfits duplicate_cleanup "
                        f"raw_deleted={cleanup_meta.get('r2_raw_deleted')} "
                        f"masked_deleted={cleanup_meta.get('r2_masked_deleted')} "
                        f"error={cleanup_meta.get('r2_cleanup_error')}"
                    )

                    existing_payload = _build_existing_duplicate_preview(
                        duplicate_user_id=duplicate_user_id,
                        duplicate_meta=duplicate_meta,
                        duplicate=duplicate,
                    )
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "message": "Duplicate outfit detected",
                            "saved": False,
                            "duplicate": duplicate_meta,
                            "cleanup": cleanup_meta,
                            "qdrant_saved": False,
                            "qdrant_error": None,
                            "qdrant_point_id": duplicate_meta.get("point_id"),
                            "existing_document": existing_payload.get("existing_document") or {},
                            "existing_preview_url": existing_payload.get("existing_preview_url"),
                            "existing_preview_data_url": existing_payload.get("existing_preview_data_url"),
                        },
                    )
                print(
                    f"[data.create_document] outfits duplicate_check "
                    f"user={duplicate_user_id} image_checked={duplicate_meta.get('image_checked')} "
                    f"image_score={duplicate_meta.get('image_score')} "
                    f"pixel_checked={duplicate_meta.get('pixel_checked')} "
                    f"pixel_hash={'yes' if duplicate_meta.get('pixel_hash') else 'no'} "
                    f"pixel_distance={duplicate_meta.get('pixel_distance')} "
                    f"semantic_score={duplicate_meta.get('score')} "
                    f"is_duplicate={duplicate_meta.get('is_duplicate')} "
                    f"reason={duplicate_meta.get('reason')}"
                )
                if duplicate_meta.get("error"):
                    print(f"[data.create_document] outfits duplicate check failed: {duplicate_meta.get('error')}")
        elif resource == "meal_plans":
            payload = _normalize_meal_plan_payload(payload, auth_user_id)

        payload = _stamp_authenticated_user(resource, payload, auth_user_id)
        if resource == "outfits":
            payload.setdefault("status", "active")
            payload.setdefault("worn", 0)
            payload.setdefault("liked", False)
            payload.setdefault("image_id", str(uuid.uuid4()))
            payload.setdefault("masked_id", str(uuid.uuid4()))
            payload.setdefault("qdrant_point_id", str(uuid.uuid4()))

            # Heal URL fields before write, so strict schemas don't fail intermittently.
            image_url = str(payload.get("image_url") or "").strip()
            masked_url = str(payload.get("masked_url") or "").strip()
            if not image_url and masked_url:
                image_url = masked_url
            if not masked_url and image_url:
                masked_url = image_url
            if not image_url or not masked_url:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Missing wardrobe image URLs",
                        "hint": "Upload raw+masked images via /api/uploads/wardrobe before saving outfit.",
                        "missing": {
                            "image_url": not bool(image_url),
                            "masked_url": not bool(masked_url),
                        },
                    },
                )

            if image_url:
                payload["image_url"] = image_url
            if masked_url:
                payload["masked_url"] = masked_url

        doc = _create_document_with_schema_retries(
            resource=resource,
            payload=payload,
            document_id=document_id,
        )

        qdrant_saved = False
        qdrant_error = None
        image_qdrant_saved = False
        image_qdrant_error = None
        point_id = None
        if resource == "outfits" and qdrant_enabled:
            try:
                if not payload_image_vector:
                    payload_image_vector = incoming_image_vector or _compute_payload_image_vector(payload)
                if not payload_pixel_hash:
                    payload_pixel_hash = incoming_pixel_hash or _compute_payload_pixel_hash(payload)
                point_id = (
                    str(doc.get("qdrant_point_id") or "").strip()
                    or str(payload.get("qdrant_point_id") or "").strip()
                    or _to_uuid_point_id(doc.get("$id") or document_id)
                )
                vector_input = {
                    "category": doc.get("category", ""),
                    "sub_category": doc.get("sub_category", ""),
                    "color_code": doc.get("color_code", ""),
                    "pattern": doc.get("pattern", ""),
                    "occasions": doc.get("occasions", []) if isinstance(doc.get("occasions", []), list) else [],
                }
                vector = encode_metadata(vector_input)
                qdrant_payload = dict(doc)
                qdrant_payload["userId"] = auth_user_id
                if payload_pixel_hash:
                    qdrant_payload["pixel_hash"] = payload_pixel_hash
                qdrant_service.upsert_item(point_id, vector, qdrant_payload)
                qdrant_saved = True

                if payload_image_vector:
                    image_payload = {
                        "userId": qdrant_payload.get("userId"),
                        "category": doc.get("category", ""),
                        "sub_category": doc.get("sub_category", ""),
                        "color_code": doc.get("color_code", ""),
                        "image_url": doc.get("masked_url") or doc.get("image_url") or payload.get("masked_url") or payload.get("image_url") or "",
                        "pixel_hash": payload_pixel_hash or "",
                    }
                    qdrant_service.upsert_image_vector(point_id, payload_image_vector, image_payload)
                    image_qdrant_saved = True

                # Persist the Qdrant point id if schema has this attribute.
                if isinstance(doc, dict) and "qdrant_point_id" in doc and str(doc.get("qdrant_point_id") or "") != point_id:
                    try:
                        doc = data_access_service.update_document(
                            resource="outfits",
                            document_id=doc.get("$id"),
                            payload={"qdrant_point_id": point_id},
                        )
                    except AppwriteProxyError as exc:
                        print(f"[data.create_document] outfits qdrant_point_id update skipped: {exc}")
            except Exception as exc:
                qdrant_error = str(exc)
                image_qdrant_error = str(exc)
                print(f"[data.create_document] outfits qdrant upsert failed: {exc}")

        return {
            "document": doc,
            "meta": {
                "saved": True,
                "duplicate": duplicate_meta if resource == "outfits" else None,
                "resource": resource,
                "qdrant_saved": qdrant_saved,
                "qdrant_error": qdrant_error,
                "image_qdrant_saved": image_qdrant_saved,
                "image_qdrant_error": image_qdrant_error,
                "qdrant_point_id": point_id,
            },
        }
    except HTTPException:
        raise
    except AppwriteProxyError as exc:
        print(f"[data.create_document] resource={request.resource} normalized={resource} error={exc}")
        raise _http_error_from_proxy(exc)


@router.patch("/{document_id}")
def update_document(http_request: Request, document_id: str, request: UpdateRequest):
    auth_user_id = _authenticated_user_id(http_request)
    resource = _normalize_resource_key(request.resource)
    try:
        existing = proxy.get_document(resource, document_id)
        _enforce_document_owner(resource, existing, auth_user_id)
        payload = dict(request.data)
        _assert_payload_user_matches(auth_user_id, payload)
        if resource == "outfits":
            payload = _normalize_outfit_payload(payload, auth_user_id)
        elif resource == "meal_plans":
            payload = _normalize_meal_plan_payload(payload, auth_user_id)
        payload = _stamp_authenticated_user(resource, payload, auth_user_id)
        doc = _update_document_with_schema_retries(
            resource=resource,
            document_id=document_id,
            payload=payload,
        )
        return {"document": doc}
    except AppwriteProxyError as exc:
        print(f"[data.update_document] resource={request.resource} normalized={resource} document_id={document_id} error={exc}")
        raise _http_error_from_proxy(exc)


@router.delete("")
def delete_document(http_request: Request, request: DeleteRequest):
    auth_user_id = _authenticated_user_id(http_request)
    resource = _normalize_resource_key(request.resource)
    try:
        existing_doc = proxy.get_document(resource, request.document_id)
        _enforce_document_owner(resource, existing_doc, auth_user_id)
        delete_meta: Dict[str, Any] = {}
        if resource == "outfits":
            delete_meta = {
                "qdrant_deleted": False,
                "qdrant_error": None,
                "qdrant_point_id": None,
                "r2_raw_deleted": False,
                "r2_masked_deleted": False,
                "r2_error": None,
                "r2_raw_candidates": [],
                "r2_masked_candidates": [],
            }
            doc: Dict[str, Any] = existing_doc

            try:
                point_id = str(doc.get("qdrant_point_id") or "").strip() or _to_uuid_point_id(
                    doc.get("$id") or request.document_id
                )
                qdrant_service.delete_item(point_id)
                delete_meta["qdrant_deleted"] = True
                delete_meta["qdrant_point_id"] = point_id
            except Exception as exc:
                delete_meta["qdrant_error"] = str(exc)
                print(f"[data.delete_document] outfits qdrant delete failed: {exc}")

            try:
                candidates = _collect_outfit_r2_candidates(doc)
                delete_meta["r2_raw_candidates"] = candidates["raw"]
                delete_meta["r2_masked_candidates"] = candidates["masked"]
                if candidates["raw"] or candidates["masked"]:
                    storage = R2Storage()

                    raw_deleted = False
                    for raw_name in candidates["raw"]:
                        result = storage.delete_wardrobe_images(raw_file_name=raw_name)
                        if result.get("raw_deleted"):
                            raw_deleted = True
                            break

                    masked_deleted = False
                    for masked_name in candidates["masked"]:
                        result = storage.delete_wardrobe_images(masked_file_name=masked_name)
                        if result.get("masked_deleted"):
                            masked_deleted = True
                            break

                    delete_meta["r2_raw_deleted"] = raw_deleted
                    delete_meta["r2_masked_deleted"] = masked_deleted
            except (R2StorageError, Exception) as exc:
                delete_meta["r2_error"] = str(exc)
                print(f"[data.delete_document] outfits r2 delete failed: {exc}")

        data_access_service.delete_document(
            resource=resource,
            document_id=request.document_id,
        )
        response: Dict[str, Any] = {"ok": True}
        if resource == "outfits":
            response["meta"] = delete_meta
        return response
    except AppwriteProxyError as exc:
        print(f"[data.delete_document] resource={request.resource} normalized={resource} document_id={request.document_id} error={exc}")
        raise _http_error_from_proxy(exc)


@router.put("/users/{user_id}")
def upsert_user_profile(http_request: Request, user_id: str, body: Dict[str, Any]):
    auth_user_id = _authenticated_user_id(http_request)
    _assert_user_match(auth_user_id, user_id, field="user_id")
    try:
        safe_body = dict(body or {})
        _assert_payload_user_matches(auth_user_id, safe_body)
        safe_body.pop("user_id", None)
        safe_body.pop("userid", None)
        safe_body.pop("userID", None)
        safe_body["userId"] = auth_user_id
        # Write with schema drift tolerance for profile attributes.
        try:
            doc = _update_document_with_schema_retries(
                resource="users",
                document_id=auth_user_id,
                payload=safe_body,
            )
        except AppwriteProxyError as exc:
            if "(404)" in str(exc):
                doc = _create_document_with_schema_retries(
                    resource="users",
                    payload=safe_body,
                    document_id=auth_user_id,
                )
            else:
                raise
        return {"document": doc}
    except AppwriteProxyError as exc:
        print(f"[data.upsert_user_profile] user_id={user_id} error={exc}")
        raise _http_error_from_proxy(exc)
