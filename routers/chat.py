from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import os
import logging
import time
import hashlib
import concurrent.futures
import threading
import re
import json

from deep_translator import GoogleTranslator

try:
    from worker import run_heavy_audio_task
except Exception:
    run_heavy_audio_task = None

from brain.orchestrator import ahvi_orchestrator
from brain.tone.tone_engine import tone_engine
from brain.outfit_pipeline import save_feedback
from services.appwrite_proxy import AppwriteProxy
from services.llm_service import chat_completion
from services.style_flow_service import (
    STYLE_ACTION_CHIPS,
    build_style_flow_response,
    card_signature as style_card_signature,
    finalize_style_response_payload,
    interpret_occasion,
)
from services.module_chat_service import handle_module_chat

try:
    from services.job_tracker import job_tracker
except Exception:
    job_tracker = None
from services.task_queue import enqueue_task

# Ã°Å¸â€Â¥ NEW
from services.weather_service import get_hourly_weather

router = APIRouter()
logger = logging.getLogger("ahvi.routers.chat")

_CHAT_CACHE_MAX_ITEMS = max(64, int(os.getenv("CHAT_CACHE_MAX_ITEMS", "512")))
_CHAT_CACHE_TTL_SECONDS = max(15, int(os.getenv("CHAT_CACHE_TTL_SECONDS", "60")))
_WEATHER_CACHE_MAX_ITEMS = max(32, int(os.getenv("WEATHER_CACHE_MAX_ITEMS", "256")))
_WEATHER_CACHE_TTL_SECONDS = max(60, int(os.getenv("WEATHER_CACHE_TTL_SECONDS", "900")))
_ORCH_TIMEOUT_SECONDS = max(2, int(os.getenv("CHAT_ORCHESTRATOR_TIMEOUT_SECONDS", "8")))
_ORCHESTRATOR_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    # Default reduced from 8 -> 3 to avoid oversubscribing 1-vCPU Cloud Run
    # instances under default concurrency=80. Override via env if needed.
    max_workers=max(2, int(os.getenv("CHAT_ORCHESTRATOR_MAX_WORKERS", "3"))),
    thread_name_prefix="chat-orch",
)
_CHAT_INCLUDE_BASE64_ALLOWED = str(
    os.getenv("AHVI_CHAT_INCLUDE_BASE64_ALLOWED", "")
).lower() in {"1", "true", "yes"}


class _TTLLRUCache:
    """Thread-safe TTL+LRU cache. O(1) get/set, lazy expiry on hit, bounded size."""

    def __init__(self, max_items: int, ttl_seconds: int):
        self._max = int(max_items)
        self._ttl = int(ttl_seconds)
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            row = self._data.get(key)
            if row is None:
                return None
            expires_at, value = row
            if now >= expires_at:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        expires_at = time.time() + self._ttl
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_CHAT_CACHE = _TTLLRUCache(_CHAT_CACHE_MAX_ITEMS, _CHAT_CACHE_TTL_SECONDS)
_WEATHER_CACHE = _TTLLRUCache(_WEATHER_CACHE_MAX_ITEMS, _WEATHER_CACHE_TTL_SECONDS)


def lightweight_chat(text: str) -> str:
    prompt = str(text or "").strip()
    lower = prompt.lower()
    if not prompt:
        return "Hey, what is on your mind today?"
    if "joke" in lower:
        return "Here is a tiny one: Why did the shirt get promoted? Because it had outstanding style."
    if "how are you" in lower or lower in {"hi", "hello", "hey"}:
        return "I am here and ready. Ask me for an outfit, a capsule wardrobe, or just talk to me."
    return "I can help with style, planning, and wardrobe advice. Tell me what you want to solve."


def _wardrobe_hash(wardrobe: Any) -> str:
    if not isinstance(wardrobe, list):
        return "no_wardrobe"
    rows: List[str] = []
    for item in wardrobe:
        if not isinstance(item, dict):
            continue
        rows.append(
            "|".join(
                str(item.get(k) or "").strip().lower()
                for k in ("id", "$id", "name", "category", "sub_category", "subcategory", "color")
            )
        )
    return hashlib.sha1("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def _cache_key(text, user_id, *, module_context: str = "", wardrobe: Any = None, occasion: str = "", weather: Any = None):
    parts = [
        str(user_id or "").strip(),
        str(module_context or "").strip().lower(),
        str(text or "").strip().lower(),
        str(occasion or "").strip().lower(),
        str(weather or "").strip().lower(),
        _wardrobe_hash(wardrobe),
    ]
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()


def _structured_error_response(
    *,
    code: str,
    message: str,
    status_type: str = "error",
    details: Optional[Dict[str, Any]] = None,
    chips: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    safe_message = str(message or "AHVI could not complete this request.").strip()
    return {
        "ok": False,
        "success": False,
        "type": status_type,
        "message": {"role": "assistant", "content": safe_message},
        "message_text": safe_message,
        "response": safe_message,
        "cards": [],
        "style_boards": [],
        "chips": chips or [],
        "board_ids": "",
        "data": {"outfits": [], "rendered_boards": [], "error": {"code": code, "message": safe_message, "details": details or {}}},
        "error": {"code": code, "message": safe_message, "details": details or {}},
        "meta": {"error_code": code, **(details or {})},
        "audio_job_id": "offline",
    }


def _style_clarification_response(query: str, interpretation: Dict[str, Any]) -> Dict[str, Any]:
    chips = interpretation.get("chips") if isinstance(interpretation.get("chips"), list) else []
    if not chips:
        chips = [
            {"label": "Office", "value": "office outfit"},
            {"label": "Casual", "value": "casual outfit"},
            {"label": "Date", "value": "date outfit tonight"},
            {"label": "Party", "value": "party outfit tonight"},
            {"label": "Travel", "value": "airport travel outfit"},
            {"label": "Workout", "value": "workout outfit"},
        ]
    message = (
        "What are we dressing for today? Pick an occasion, or tell me the weather, timing, mood, and any dress code."
    )
    return {
        "success": True,
        "ok": True,
        "type": "style_clarification",
        "message": {"role": "assistant", "content": message},
        "message_text": message,
        "response": message,
        "cards": [],
        "style_boards": [],
        "chips": chips,
        "board_ids": "",
        "data": {
            "outfits": [],
            "rendered_boards": [],
            "clarification": {
                "prompt": query,
                "questions": ["occasion", "weather/timing", "mood/style", "comfort/dress code"],
            },
        },
        "meta": {
            "mode": "style_intent_clarification",
            "intent_status": "clarify",
            "occasion_interpretation": interpretation,
        },
        "audio_job_id": "offline",
    }


def _is_vague_style_prompt(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    vague = {
        "outfit for today",
        "suggest outfit for today",
        "suggest an outfit for today",
        "style me",
        "what should i wear",
        "what to wear",
        "outfit",
        "daily wear",
        "today outfit",
    }
    return normalized in vague


def _weather_cache_key(lat: Any, lon: Any) -> str:
    return f"{float(lat):.4f}:{float(lon):.4f}"


def _get_weather_cached(lat: Any, lon: Any) -> Dict[str, Any]:
    key = _weather_cache_key(lat, lon)
    cached = _WEATHER_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    weather = get_hourly_weather(lat=float(lat), lon=float(lon))
    _WEATHER_CACHE.set(key, weather)
    return weather


def shutdown_chat_resources(wait_seconds: float = 5.0) -> None:
    """Called from app shutdown to drain in-flight orchestrator work."""
    try:
        _ORCHESTRATOR_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    except TypeError:
        # cancel_futures kw added in 3.9; fallback for unusual runtimes
        _ORCHESTRATOR_EXECUTOR.shutdown(wait=True)
    except Exception:
        logger.exception("orchestrator executor shutdown failed")


def _build_history(messages: List["Message"]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for msg in messages[-8:]:
        role = str(getattr(msg, "role", "user")).lower()
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue
        history.append({"role": role, "text": content[:500]})
    return history


def _normalize_memory_history(events: Any, max_items: int = 12) -> List[Dict[str, Any]]:
    if not isinstance(events, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for event in events[-max_items:]:
        if not isinstance(event, dict):
            continue
        row: Dict[str, Any] = {}
        if event.get("intent"):
            row["intent"] = str(event.get("intent"))[:80]
        if isinstance(event.get("slots"), dict):
            row["slots"] = event.get("slots")
        if event.get("role"):
            row["role"] = str(event.get("role"))[:32]
        if event.get("text"):
            row["text"] = str(event.get("text"))[:500]
        if row:
            normalized.append(row)
    return normalized


def _is_fast_wardrobe_count_query(text: str) -> bool:
    lowered = str(text or "").lower()
    count_words = ["how many", "count", "number of", "total", "do i have"]
    wardrobe_words = [
        "wardrobe",
        "closet",
        "outfit",
        "outfits",
        "tops",
        "top",
        "shirts",
        "shirt",
        "pants",
        "trousers",
        "jeans",
        "bottoms",
        "shoes",
        "footwear",
        "dress",
        "dresses",
        "accessories",
        "jewelry",
        "bags",
        "bag",
    ]
    return any(k in lowered for k in count_words) and any(
        k in lowered for k in wardrobe_words
    )


_CHAT_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _chat_tokens(value: Any) -> List[str]:
    return _CHAT_TOKEN_RE.sub(" ", str(value or "").lower()).strip().split()


def _chat_has_any(tokens: List[str], words: List[str]) -> bool:
    return any(word in tokens for word in words)


def _infer_chat_category(item: Dict[str, Any]) -> str:
    # Delegate to the shared taxonomy. Behavior unchanged.
    from services.category_taxonomy import categorize_for_chat

    return categorize_for_chat(item)


def _fast_wardrobe_count_response(user_id: str, query_text: str) -> Dict[str, Any]:
    # Paginate fully so totals are accurate for wardrobes >100 items.
    docs: List[Dict[str, Any]] = []
    try:
        proxy = AppwriteProxy()
        page_size = 100
        offset = 0
        while True:
            page = proxy.list_documents(
                "outfits", user_id=user_id, limit=page_size, offset=offset
            )
            if not page:
                break
            docs.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
            if offset >= 5000:  # safety cap
                break
    except Exception:
        logger.warning("fast_wardrobe_count_response fetch failed", exc_info=True)
        docs = []

    counts = {"tops": 0, "bottoms": 0, "shoes": 0, "dresses": 0, "accessories": 0}
    for d in docs:
        cat = _infer_chat_category(d)

        if cat in {"Tops", "Outerwear"}:
            counts["tops"] += 1
        elif cat == "Bottoms":
            counts["bottoms"] += 1
        elif cat == "Footwear":
            counts["shoes"] += 1
        elif cat == "Dresses":
            counts["dresses"] += 1
        else:
            counts["accessories"] += 1

    lowered = str(query_text or "").lower()
    if any(
        k in lowered for k in ["top", "tops", "shirt", "shirts", "blouse", "blouses"]
    ):
        message = f"You have {counts['tops']} tops in your wardrobe."
    elif any(
        k in lowered
        for k in [
            "bottom",
            "bottoms",
            "pant",
            "pants",
            "trouser",
            "trousers",
            "jean",
            "jeans",
        ]
    ):
        message = f"You have {counts['bottoms']} bottoms in your wardrobe."
    elif any(
        k in lowered for k in ["shoe", "shoes", "footwear", "sneaker", "sneakers"]
    ):
        message = f"You have {counts['shoes']} shoes in your wardrobe."
    else:
        total = len(docs)
        message = (
            f"You currently have {total} items: {counts['tops']} tops, {counts['bottoms']} bottoms, "
            f"{counts['shoes']} shoes, {counts['dresses']} dresses, and {counts['accessories']} accessories."
        )

    return {
        "success": True,
        "message": message,
        "board": "wardrobe",
        "type": "stats",
        "cards": [
            {"id": "tops", "title": "Tops", "kind": "stat", "value": counts["tops"]},
            {
                "id": "bottoms",
                "title": "Bottoms",
                "kind": "stat",
                "value": counts["bottoms"],
            },
            {"id": "shoes", "title": "Shoes", "kind": "stat", "value": counts["shoes"]},
            {
                "id": "dresses",
                "title": "Dresses",
                "kind": "stat",
                "value": counts["dresses"],
            },
            {
                "id": "accessories",
                "title": "Accessories",
                "kind": "stat",
                "value": counts["accessories"],
            },
        ],
        "data": {"counts": counts, "total_items": len(docs)},
        "meta": {"intent": "wardrobe_query", "domain": "wardrobe", "fast_path": True},
        "audio_job_id": "offline",
    }


# ================= AHVI CLEAN CHAT STYLE V2 BEGIN =================
# One clean style adapter for chat.
# Source of truth: brain.outfit_pipeline.get_daily_outfits
# Purpose:
# - remove duplicate _demo_style_board_payload definitions
# - sanitize cards before they reach Flutter
# - one watch max
# - category/role/slot guaranteed for top, bottom, footwear, accessory
# - fallback still works if orchestrator times out

_AHVI_MALE_STYLE_GENDERS = {"m", "male", "man", "men", "mens", "boy"}
_AHVI_FEMALE_STYLE_GENDERS = {
    "f",
    "female",
    "woman",
    "women",
    "womens",
    "girl",
    "ladies",
}
_AHVI_UNISEX_STYLE_GENDERS = {"unisex", "neutral", "genderless", "any"}

_AHVI_FEMININE_ONLY_GARMENTS = {
    "saree",
    "sari",
    "lehenga",
    "gown",
    "skirt",
    "skirts",
    "blouse",
    "kurti",
}
_AHVI_MALE_TRADITIONAL_GARMENTS = {"sherwani", "achkan"}

_AHVI_EXPLICIT_FEMININE_REQUEST = {
    "saree",
    "sari",
    "lehenga",
    "gown",
    "skirt",
    "skirts",
    "female",
    "women",
    "woman",
    "ladies",
    "feminine",
}


def _ahvi_coerce_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            import json as _json

            parsed = _json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _ahvi_normalize_style_gender(value):
    raw = str(value or "").strip().lower()
    if raw in _AHVI_MALE_STYLE_GENDERS:
        return "male"
    if raw in _AHVI_FEMALE_STYLE_GENDERS:
        return "female"
    if raw in _AHVI_UNISEX_STYLE_GENDERS:
        return "unisex"
    return ""


def _ahvi_profile_style_gender(profile):
    profile = profile or {}
    candidates = [
        profile.get("style_gender"),
        profile.get("gender"),
        profile.get("preferred_gender"),
        profile.get("target_gender"),
    ]

    for key in ("preferences", "style_preferences", "stylePreference", "profile"):
        nested = _ahvi_coerce_dict(profile.get(key))
        candidates.extend(
            [
                nested.get("style_gender"),
                nested.get("gender"),
                nested.get("preferred_gender"),
                nested.get("target_gender"),
            ]
        )

    for value in candidates:
        gender = _ahvi_normalize_style_gender(value)
        if gender:
            return gender

    return "unisex"


def _ahvi_resolve_effective_user_profile(user_id, request_profile=None):
    request_profile = request_profile if isinstance(request_profile, dict) else {}
    try:
        from services.data_access_service import get_user_profile, merge_user_profiles

        stored = get_user_profile(user_id=str(user_id or "").strip())
        merged = merge_user_profiles(stored, request_profile)
        merged.setdefault("user_id", user_id)
        return merged
    except Exception:
        out = dict(request_profile)
        out.setdefault("user_id", user_id)
        return out


def _ahvi_query_allows_feminine_item(query_text):
    tokens = set(_chat_tokens(query_text))
    return bool(tokens.intersection(_AHVI_EXPLICIT_FEMININE_REQUEST))


def _ahvi_item_tokens(item):
    blob = " ".join(
        str(item.get(k, "") or "")
        for k in (
            "slot",
            "role",
            "type",
            "category",
            "cat",
            "category_group",
            "sub_category",
            "subcategory",
            "subCategory",
            "name",
            "label",
            "description",
            "gender",
            "style_gender",
            "target_gender",
            "audience",
            "department",
            "intended_for",
            "wearer",
        )
    )
    return set(_chat_tokens(blob))


def _ahvi_item_allowed_for_user_profile(item, user_profile=None, query_text=""):
    if not isinstance(item, dict):
        return False

    target_gender = _ahvi_profile_style_gender(user_profile or {})

    if target_gender != "male":
        return True

    if _ahvi_query_allows_feminine_item(query_text):
        return True

    tokens = _ahvi_item_tokens(item)

    audience_blob = " ".join(
        str(item.get(k, "") or "")
        for k in (
            "gender",
            "style_gender",
            "target_gender",
            "audience",
            "department",
            "intended_for",
            "wearer",
        )
    )
    audience_tokens = set(_chat_tokens(audience_blob))

    if audience_tokens.intersection(_AHVI_FEMALE_STYLE_GENDERS):
        return False

    if tokens.intersection(_AHVI_FEMININE_ONLY_GARMENTS):
        return False

    if _infer_chat_category(item) == "Dresses" and not tokens.intersection(
        _AHVI_MALE_TRADITIONAL_GARMENTS
    ):
        return False

    return True


def _fetch_wardrobe_for_style(
    user_id: str, request_wardrobe: Any
) -> List[Dict[str, Any]]:
    if isinstance(request_wardrobe, list):
        return [dict(i) for i in request_wardrobe if isinstance(i, dict)]

    try:
        docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=100)
        if isinstance(docs, dict):
            rows = docs.get("documents") or docs.get("items") or []
        else:
            rows = docs or []
        return [dict(i) for i in rows if isinstance(i, dict)]
    except Exception as exc:
        logger.warning("style wardrobe fetch failed user_id=%s error=%s", user_id, exc)
        return []


def _ahvi_style_occasion(query_text):
    q = str(query_text or "").lower()
    if any(k in q for k in ["date", "dinner", "night"]):
        return "date night"
    if any(k in q for k in ["office", "meeting", "work", "client"]):
        return "office"
    if any(k in q for k in ["party", "club", "night out"]):
        return "party"
    if any(k in q for k in ["travel", "airport", "trip"]):
        return "travel"
    if any(
        k in q
        for k in ["coffee", "casual", "outing", "weekend", "street", "sport", "outdoor"]
    ):
        return "casual outing"

    generic_daily_style = any(
        k in q
        for k in [
            "suggest an outfit",
            "outfit for today",
            "today outfit",
            "wear today",
            "what should i wear",
        ]
    )
    relaxed_daily_context = any(
        k in q
        for k in [
            "home",
            "wfh",
            "sunday",
            "lazy",
            "errand",
            "grocery",
            "beach",
            "resort",
            "vacation",
        ]
    )
    if generic_daily_style and not relaxed_daily_context:
        return "office"
    return "today"


def _ahvi_style_image(item):
    if not isinstance(item, dict):
        return ""

    # Prefer the cleanest asset for style boards.
    # normalized_url is the 1024x1024 transparent PNG created by the backend
    # image normalizer. raw_url is intentionally last for debugging/fallback only.
    candidates = [
        item.get("normalized_url"),
        item.get("normalizedUrl"),
        item.get("normalized_image_url"),
        item.get("normalizedImageUrl"),
        item.get("transparent_url"),
        item.get("transparentUrl"),
        item.get("processed_url"),
        item.get("processedUrl"),
        item.get("png_url"),
        item.get("pngUrl"),
        item.get("cutout_url"),
        item.get("cutoutUrl"),
        item.get("masked_url"),
        item.get("maskedUrl"),
        item.get("masked_image_url"),
        item.get("maskedImageUrl"),
        item.get("image_url"),
        item.get("imageUrl"),
        item.get("url"),
        item.get("image"),
        item.get("raw_url"),
        item.get("rawUrl"),
        item.get("raw_image_url"),
        item.get("rawImageUrl"),
    ]

    for value in candidates:
        url = str(value or "").strip()
        if url and url.lower() not in {"null", "none", "undefined"}:
            return url

    return ""


def _ahvi_style_blob(item):
    if not isinstance(item, dict):
        return ""
    return " ".join(
        str(item.get(k, "") or "")
        for k in (
            "role",
            "slot",
            "type",
            "category",
            "cat",
            "category_group",
            "sub_category",
            "subcategory",
            "subCategory",
            "name",
            "label",
            "description",
            "pattern",
            "color",
            "color_name",
        )
    ).lower()


def _ahvi_style_key(item):
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("$id")
        or item.get("id")
        or item.get("item_id")
        or item.get("itemId")
        or item.get("image_id")
        or item.get("name")
        or item.get("label")
        or id(item)
    ).lower()


def _ahvi_style_role(item):
    tokens = set(_chat_tokens(_ahvi_style_blob(item)))

    if tokens.intersection(
        {
            "shoe",
            "shoes",
            "sneaker",
            "sneakers",
            "boot",
            "boots",
            "heel",
            "heels",
            "sandal",
            "sandals",
            "loafer",
            "loafers",
            "slipper",
            "slippers",
            "slider",
            "sliders",
            "footwear",
        }
    ):
        return "footwear"

    if tokens.intersection(
        {
            "watch",
            "watches",
            "belt",
            "belts",
            "cap",
            "caps",
            "hat",
            "hats",
            "sunglass",
            "sunglasses",
            "eyewear",
            "glasses",
            "bag",
            "bags",
            "purse",
            "handbag",
            "clutch",
            "tote",
            "jewelry",
            "jewellery",
            "ring",
            "rings",
            "necklace",
            "necklaces",
            "bracelet",
            "bracelets",
            "earring",
            "earrings",
            "scarf",
            "scarves",
            "accessory",
            "accessories",
        }
    ):
        return "accessory"

    if tokens.intersection(
        {"dress", "dresses", "saree", "sari", "lehenga", "gown", "jumpsuit", "sherwani"}
    ):
        return "dress"

    # Tops before bottoms: short-sleeved shirt must not become shorts.
    if tokens.intersection(
        {
            "top",
            "tops",
            "shirt",
            "shirts",
            "tee",
            "tshirt",
            "tshirts",
            "polo",
            "polos",
            "jacket",
            "blazer",
            "sweater",
            "hoodie",
            "kurta",
            "kurti",
            "tunic",
            "tunics",
        }
    ):
        return "top"

    if tokens.intersection(
        {
            "bottom",
            "bottoms",
            "pant",
            "pants",
            "trouser",
            "trousers",
            "jean",
            "jeans",
            "shorts",
            "skirt",
            "skirts",
            "chino",
            "chinos",
        }
    ):
        return "bottom"

    return "unknown"


def _ahvi_style_names(items):
    return [
        str(
            (i or {}).get("name")
            or (i or {}).get("label")
            or (i or {}).get("category")
            or ""
        )
        for i in items or []
        if isinstance(i, dict)
    ]


def _ahvi_router_style_fallback_enabled():
    return os.getenv("AHVI_ENABLE_ROUTER_STYLE_FALLBACK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _ahvi_style_action_chips() -> List[str]:
    return list(STYLE_ACTION_CHIPS)


def _demo_style_board_payload(user_id, query_text, request_wardrobe, user_profile=None):
    profile = _ahvi_resolve_effective_user_profile(user_id, user_profile or {})
    wardrobe = _fetch_wardrobe_for_style(user_id, request_wardrobe)
    wardrobe = [
        item
        for item in wardrobe
        if _ahvi_item_allowed_for_user_profile(item, profile, query_text)
    ]

    occasion = _ahvi_style_occasion(query_text)

    try:
        response = build_style_flow_response(
            user_id=user_id,
            query=query_text,
            wardrobe=wardrobe,
            user_profile=profile,
            context={
                "occasion": occasion,
                "query": query_text,
                "user_profile": profile,
                "style_gender": _ahvi_profile_style_gender(profile),
                "signals": {
                    "source": "routers.chat.style_flow_service_fallback",
                    "style_gender": _ahvi_profile_style_gender(profile),
                },
            },
            include_base64=False,
            cache_bypass=True,
        )
    except Exception as exc:
        logger.warning(
            "ahvi.chat_style_flow_fallback_failed user_id=%s error=%s",
            user_id,
            str(exc)[:180],
        )
        return {
            "success": False,
            "message": (
                "I couldn't build a reliable style board from your wardrobe yet. "
                "Please add at least one top, bottom, and footwear item."
            ),
            "board": "style",
            "type": "missing_outfit_cards",
            "cards": [],
            "style_boards": [],
            "board_ids": "",
            "data": {
                "outfits": [],
                "rendered_boards": [],
                "board_item_ids": [],
            },
            "meta": {
                "mode": "style_flow_service_fallback_failed",
                "fallback_used": False,
                "error": "style_flow_service_failed",
                "error_stage": "routers.chat",
                "occasion": occasion,
                "wardrobe_count": len(wardrobe),
            },
        }

    cards = response.get("cards") if isinstance(response.get("cards"), list) else []
    if not cards and not _ahvi_router_style_fallback_enabled():
        response["success"] = False
        response["message"] = (
            "I couldn't build a complete style board from your wardrobe yet. "
            "Please add at least one top, bottom, and footwear item."
        )
        response["type"] = "missing_outfit_cards"

    try:
        logger.info(
            "ahvi.chat_style_flow_service user_id=%s occasion=%s wardrobe=%s cards=%s first_card_items=%s",
            user_id,
            occasion,
            len(wardrobe),
            len(cards),
            _ahvi_style_names((cards[0].get("items") if cards else []) or [])[:8],
        )
    except Exception:
        pass

    response["meta"] = {
        **(response.get("meta") if isinstance(response.get("meta"), dict) else {}),
        "intent": "style_pipeline_adapter",
        "domain": "style",
        "mode": "style_flow_service_adapter_v1",
        "wardrobe_count": len(wardrobe),
        "style_gender": _ahvi_profile_style_gender(profile),
        "occasion": occasion,
    }
    return response

# ================= AHVI CLEAN CHAT STYLE V2 END =================


def _is_explicit_style_request(text: str, module_context: str | None = None) -> bool:
    """
    True only when user is clearly asking AHVI to build/style an outfit board.

    Important:
    Generic words like "style" or "personal styling" alone must not force the
    wardrobe pipeline. Otherwise normal chat gets misrouted to daily_outfit.
    """
    q = str(text or "").lower().strip()
    module = str(module_context or "").lower().strip()

    if any(
        k in q
        for k in [
            "more looks",
            "more look",
            "next best",
            "next option",
            "next options",
            "other options",
            "other looks",
            "show more",
            "different shoes",
            "different shoe",
            "different footwear",
            "try different shoes",
        ]
    ):
        return True

    if module in {"style", "wardrobe"} and any(
        k in q
        for k in [
            "wear",
            "outfit",
            "look",
            "style me",
            "style this",
            "what should i wear",
            "date night",
            "office outfit",
            "party outfit",
            "build a board",
            "style board",
        ]
    ):
        return True

    occasion_setup_words = [
        "date",
        "dinner",
        "party",
        "club",
        "rave",
        "office",
        "meeting",
        "wedding",
        "reception",
        "event",
        "travel",
        "trip",
        "brunch",
    ]
    setup_phrases = [
        "i have",
        "ive got",
        "i've got",
        "coming up",
        "tonight",
        "tomorrow",
        "this weekend",
        "next week",
        "need to go",
        "going to",
    ]
    if any(o in q for o in occasion_setup_words) and any(
        p in q for p in setup_phrases
    ):
        return True

    if any(
        p in q
        for p in [
            "rave party",
            "club party",
            "birthday party",
            "house party",
            "cocktail party",
            "office party",
            "wedding reception",
        ]
    ):
        return True

    explicit_phrases = [
        "what should i wear",
        "what to wear",
        "what do i wear",
        "help me choose an outfit",
        "choose an outfit",
        "suggest an outfit",
        "show outfits",
        "show me outfits",
        "build an outfit",
        "create an outfit",
        "make an outfit",
        "style me",
        "style this",
        "style my",
        "style board",
        "date night outfit",
        "office outfit",
        "party outfit",
        "travel outfit",
        "airport outfit",
        "wedding outfit",
        "brunch outfit",
        "dinner outfit",
        "gym outfit",
    ]
    if any(p in q for p in explicit_phrases):
        return True

    # Occasion chip support: "date night" alone should create boards.
    occasion_only = {
        "date night",
        "office",
        "party",
        "travel",
        "airport",
        "wedding",
        "brunch",
        "dinner",
        "night out",
    }
    if q in occasion_only:
        return True

    # "I have a date tonight..." is outfit intent only when paired with wear/outfit/look.
    occasion_words = ["date", "dinner", "party", "office", "meeting", "wedding", "travel", "brunch"]
    wardrobe_words = ["wear", "outfit", "look", "clothes", "dress up", "style board"]
    if any(o in q for o in occasion_words) and any(w in q for w in wardrobe_words):
        return True

    return False


def _is_general_chat_request(text: str, module_context: str | None = None) -> bool:
    q = str(text or "").lower().strip()
    module = str(module_context or "").lower().strip()

    if module not in {"", "chat", "general", "home", "assistant", "style", "wardrobe"}:
        return False

    if _is_explicit_style_request(q, module_context):
        return False

    # Anything instructional/question-like that is not explicitly outfit-building
    # should go to the LLM.
    if q in {"hi", "hello", "hey", "chat", "talk", "talk to me"}:
        return True

    general_markers = [
        "reply with",
        "say ",
        "explain",
        "why ",
        "what is",
        "who is",
        "how are",
        "tell me",
        "can you",
        "do you",
        "help me understand",
        "summarize",
        "write",
        "draft",
        "rephrase",
        "just chat",
        "not outfit",
        "do not mention outfits",
        "ai styling should feel personal",
    ]
    if any(k in q for k in general_markers):
        return True

    # If the user mentions fashion/styling conceptually but does not ask for a board,
    # keep it as conversational LLM.
    conceptual_style_markers = [
        "why ai styling",
        "personal styling",
        "fashion advice",
        "style advice",
        "styling should",
        "why style",
    ]
    if any(k in q for k in conceptual_style_markers):
        return True

    return False


def _build_llm_messages(messages: List["Message"], english_input: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in messages[-10:]:
        role = str(getattr(msg, "role", "user") or "user").lower()
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    if not out or out[-1].get("content") != english_input:
        out.append({"role": "user", "content": english_input})
    return out


def _llm_chat_response(
    *,
    messages: List["Message"],
    english_input: str,
    user_id: str,
    user_profile: Dict[str, Any],
    user_message_style: Dict[str, str],
    module_context: str | None = None,
) -> Dict[str, Any]:
    """
    General chat path. This is the path that reaches Gemini/Vertex.
    It must not call outfit pipeline.
    """
    system_instruction = (
        "You are AHVI, a warm premium AI companion. "
        "For normal chat, answer directly and naturally. "
        "Do not create outfit boards unless the user explicitly asks what to wear or asks for an outfit. "
        "Keep replies concise, fresh, and helpful."
    )

    try:
        message = chat_completion(
            _build_llm_messages(messages, english_input),
            system_instruction=system_instruction,
            user_profile=user_profile,
            signals={
                "context_mode": module_context or "chat",
                "user_message_style": user_message_style,
            },
            timeout_seconds=45,
            options={"temperature": 0.65, "max_output_tokens": 320},
            usecase="general_chat",
        )
        mode = "llm_chat"
    except Exception as exc:
        logger.warning("chat.llm_response_failed user_id=%s error=%s", user_id, str(exc)[:180])
        message = lightweight_chat(english_input)
        mode = "llm_chat_fallback"

    try:
        message = tone_engine.apply(
            str(message or "").strip() or lightweight_chat(english_input),
            user_profile=user_profile,
            signals={
                "context_mode": module_context or "chat",
                "user_message_style": user_message_style,
            },
            context={"module_context": module_context},
        )
    except Exception:
        pass

    logger.info(
        "chat.llm_response user_id=%s mode=%s provider=%s",
        user_id,
        mode,
        os.getenv("AI_PROVIDER", ""),
    )

    return {
        "success": True,
        "message": message,
        "board": "general",
        "type": "text",
        "cards": [],
        "board_ids": "",
        "data": {},
        "meta": {
            "mode": mode,
            "intent": "general_chat",
            "provider": os.getenv("AI_PROVIDER", ""),
        },
        "audio_job_id": "offline",
    }


def _detect_mode(text: str, module_context: str | None = None) -> str:
    if _is_general_chat_request(text, module_context):
        return "casual"
    if _is_explicit_style_request(text, module_context):
        return "fashion"

    t = text.lower().strip()
    if t in ["hi", "hello", "hey"]:
        return "greeting"

    # Default must be casual/LLM, not fashion. Otherwise almost every ambiguous
    # prompt falls into wardrobe boards and returns static style fallback.
    return "casual"


def _infer_user_message_style(text: str) -> Dict[str, str]:
    raw = str(text or "")
    lowered = raw.lower()
    length = len(raw.strip())

    emoji_count = sum(1 for ch in raw if ord(ch) > 10000)
    if emoji_count >= 3:
        emoji_density = "high"
    elif emoji_count == 2:
        emoji_density = "medium"
    elif emoji_count == 1:
        emoji_density = "low"
    else:
        emoji_density = "none"

    slang_tokens = ["lowkey", "highkey", "vibe", "it's giving", "main character", "mid"]
    slang_hits = sum(1 for token in slang_tokens if token in lowered)
    if slang_hits >= 3:
        slang_presence = "high"
    elif slang_hits == 2:
        slang_presence = "medium"
    elif slang_hits == 1:
        slang_presence = "low"
    else:
        slang_presence = "none"

    if length <= 80:
        length_bucket = "short"
    elif length <= 220:
        length_bucket = "medium"
    else:
        length_bucket = "long"

    return {
        "message_length_bucket": length_bucket,
        "emoji_density": emoji_density,
        "slang_presence": slang_presence,
    }


# -------------------------
# MODELS
# -------------------------
class Message(BaseModel):
    role: str = Field(..., min_length=1, max_length=24)
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        role = str(value or "").strip().lower()
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be one of user/assistant/system")
        return role


class TextChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1, max_length=30)
    language: str = Field(default="en", min_length=2, max_length=8)
    current_memory: Any = Field(default_factory=dict)
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    userID: str | None = None
    module_context: str | None = None
    include_base64: bool = False
    wardrobe: Any = None
    style_action: str | None = None
    exclude_style_signatures: List[str] = Field(default_factory=list)
    requested_board_count: int | None = None


class OutfitFeedbackRequest(BaseModel):
    user_id: str
    feedback: str
    outfit: Dict[str, Any]


class OrganizeHubRequest(BaseModel):
    user_id: str
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)
    include_counts: bool = False


class PlanPackRequest(BaseModel):
    user_id: str
    prompt: str
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)


class DailyCardsRequest(BaseModel):
    user_id: str
    time_slot: str | None = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)


class ModuleChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=20)
    module: str | None = Field(default=None, min_length=2, max_length=32)
    domain: str | None = Field(default=None, min_length=2, max_length=32)
    context_data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    user_profile: Dict[str, Any] = Field(default_factory=dict)


_MODULE_CHAT_PROMPTS: Dict[str, str] = {
    "medi": (
        "You are AHVI's medicine tracking assistant. Use only the user's "
        "medication inventory and reminder data. Do not give diagnosis or unsafe "
        "dosage advice. For medical uncertainty, advise consulting a doctor."
    ),
    "skincare": (
        "You are AHVI's skincare assistant. Use the user's skin profile and "
        "routine. Be practical, conservative, and avoid diagnosis."
    ),
    "meal": (
        "You are AHVI's meal planning assistant. Use the user's diet, allergies, "
        "goal, calories, and logged meals. Never suggest ingredients listed in allergies."
    ),
    "diet": (
        "You are AHVI's meal planning assistant. Use the user's diet, allergies, "
        "goal, calories, and logged meals. Never suggest ingredients listed in allergies."
    ),
    "calendar": (
        "You are AHVI's planning assistant. Use today's events, tasks, deadlines, "
        "and time context. Prioritize what reduces friction for the user."
    ),
    "bills": (
        "You are AHVI's bills and expense assistant. Use only the provided bill, "
        "transaction, budget, and due-date context. Do not provide investment advice."
    ),
    "fitness": (
        "You are AHVI's fitness preparation assistant. Use the user's workout, "
        "equipment, time, and constraint context. Keep guidance safe, practical, and concise."
    ),
    "daily_wear": (
        "You are AHVI's daily wear assistant. Use the provided style context, but "
        "do not invent wardrobe items. If the user asks for boards, route them to style."
    ),
}


def _normalize_module_name(module: str) -> str:
    value = str(module or "").strip().lower().replace("-", "_")
    allowed = {
        "style",
        "wardrobe",
        "daily_wear",
        "skincare",
        "medi",
        "bills",
        "calendar",
        "meal",
        "diet",
        "fitness",
        "planner",
    }
    if value in {"plan", "planning", "reminder", "reminders"}:
        return "planner"
    return value if value in allowed else "chat"


def _looks_incomplete_module_answer(answer: Any) -> bool:
    text = str(answer or "").strip()
    if not text:
        return True
    if len(text) < 80 and not re.search(r"[.!?)]$", text):
        return True
    tail = text.lower().rstrip(" .,:;")
    incomplete_endings = (
        "about",
        "because",
        "including",
        "such as",
        "based on",
        "depending on",
        "i need",
        "i need a bit more detail about",
    )
    return any(tail.endswith(ending) for ending in incomplete_endings)


def _module_fallback_answer(module_key: str, user_message: str) -> str:
    message = str(user_message or "").lower()
    if module_key == "skincare":
        if any(token in message for token in ("spf", "sunscreen", "sun screen", "sunblock")):
            return (
                "For SPF, choose a broad-spectrum SPF 30 or higher and apply it generously every morning. "
                "Match it to your skin type: if your skin is oily or acne-prone, try a lightweight non-comedogenic gel or fluid. "
                "If your skin is dry or sensitive, look for a fragrance-free moisturizing sunscreen. Reapply "
                "every 2-3 hours outdoors, especially with longer sun exposure."
            )
        if "dry" in message:
            return (
                "For dry skin, keep the routine simple: gentle cleanser, hydrating serum if you use one, "
                "moisturizer, and SPF in the morning. At night, cleanse and use a richer moisturizer. "
                "Avoid harsh scrubs if your skin feels tight or irritated."
            )
        if "oily" in message:
            return (
                "For oily skin, use a gentle cleanser, lightweight moisturizer, and non-comedogenic SPF. "
                "A niacinamide serum can be a reasonable option if your skin tolerates it. Avoid stripping "
                "the skin, because that can make oiliness feel worse."
            )
        if "acne" in message or "pimple" in message or "breakout" in message:
            return (
                "For acne-prone skin, keep it gentle: mild cleanser, lightweight moisturizer, and SPF. "
                "Introduce actives slowly and avoid layering too many strong products at once. For painful, "
                "persistent, or worsening acne, check with a dermatologist."
            )
        if "night" in message:
            return (
                "A simple night routine: cleanse, apply a gentle hydrating step if you use one, then moisturize. "
                "If you use treatments, add only one at a time and watch how your skin reacts."
            )
        return (
            "I can help with a simple skincare routine. A safe baseline is gentle cleanser, moisturizer, "
            "and broad-spectrum SPF in the morning, then cleanser and moisturizer at night. Tell me your "
            "skin type or concern for a more tailored routine."
        )
    if module_key in {"diet", "meal"}:
        if "protein" in message and "breakfast" in message:
            return (
                "For a high-protein breakfast, try Greek yogurt with fruit and nuts, eggs with whole-grain toast, "
                "paneer or tofu scramble, or oats with milk and seeds. Keep portions aligned with your appetite "
                "and dietary preferences."
            )
        if "pre" in message and "workout" in message:
            return (
                "A simple pre-workout meal is easy-to-digest carbs plus a little protein, such as banana with "
                "yogurt, toast with peanut butter, or oats. Give yourself enough time to digest before training."
            )
        if "post" in message and "workout" in message:
            return (
                "Post-workout, aim for protein plus carbs and fluids. Examples: rice with dal or chicken, eggs "
                "and toast, yogurt with fruit, or tofu with a grain bowl."
            )
        if "hydrat" in message or "water" in message:
            return (
                "For hydration, sip water through the day and add electrolytes when you sweat heavily or train "
                "in heat. Use thirst, urine color, and workout intensity as practical cues."
            )
        if "weight loss" in message or "lose weight" in message:
            return (
                "For weight-loss meal ideas, build plates around lean protein, vegetables or fruit, high-fiber "
                "carbs, and a small amount of healthy fat. Keep it sustainable and avoid extreme restriction."
            )
        return (
            "I can help with meal ideas. A balanced option is protein, fiber-rich carbs, vegetables or fruit, "
            "and enough fluids. Tell me your goal, dietary preference, or meal timing for a sharper suggestion."
        )
    if module_key in {"planner", "calendar"}:
        return (
            "I can help structure this plan. Reminder sync is coming next, so for now I will not mark reminders "
            "as saved. Share the tasks, deadline, and rough priority, and I will organize the day."
        )
    return lightweight_chat(user_message)


def _module_response_envelope(
    module_key: str,
    answer: str,
    chips: List[Any] | None = None,
) -> Dict[str, Any]:
    answer_text = str(answer or "").strip()
    return {
        "success": True,
        "type": "module_chat",
        "module": module_key,
        "domain": module_key,
        "response": answer_text,
        "message_text": answer_text,
        "message": {"role": "assistant", "content": answer_text},
        "cards": [],
        "style_boards": [],
        "chips": chips or [],
        "data": {
            "module": module_key,
            "message": answer_text,
            "rendered_boards": [],
            "outfits": [],
        },
        "meta": {
            "mode": module_key,
            "board_count": 0,
        },
    }


def _state_user_id(http_request: Request) -> str:
    state_user = getattr(http_request.state, "user", None)
    if isinstance(state_user, dict):
        return str(
            state_user.get("user_id")
            or state_user.get("$id")
            or state_user.get("id")
            or ""
        ).strip()
    return ""


def _module_llm_response(
    *,
    module: str,
    user_message: str,
    history: List[Dict[str, str]],
    context_data: Dict[str, Any],
    user_profile: Dict[str, Any],
) -> Dict[str, Any]:
    module_key = _normalize_module_name(module)
    system_instruction = _MODULE_CHAT_PROMPTS.get(
        module_key,
        "You are AHVI. Answer directly using the provided context. Do not invent missing data.",
    )

    if context_data:
        system_instruction += (
            "\n\nUse this app context. Do not invent missing data:\n"
            + json.dumps(context_data, ensure_ascii=False, default=str)[:6000]
        )

    messages: List[Dict[str, str]] = []
    for item in list(history or [])[-10:]:
        role = str(item.get("role") or "user").lower().strip()
        content = str(item.get("content") or item.get("text") or "").strip()
        if role not in {"user", "assistant", "system"}:
            role = "user"
        if content:
            messages.append({"role": role, "content": content[:1200]})
    messages.append({"role": "user", "content": user_message})

    answer: str = ""
    started_at = time.perf_counter()
    try:
        answer = chat_completion(
            messages,
            system_instruction=system_instruction,
            user_profile=user_profile,
            signals={"context_mode": module_key},
            # Hard cap LLM call at 30s so the module-chat endpoint always
            # returns within the frontend's 75s budget. If the LLM is slow,
            # we'd rather ship lightweight_chat() than time the user out.
            timeout_seconds=30,
            options={"temperature": 0.45, "max_output_tokens": 320},
            usecase=f"{module_key}_chat",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        logger.warning(
            "chat.module_chat_failed module=%s elapsed=%.2fs error=%s",
            module_key, elapsed, str(exc)[:180],
        )
        answer = ""
    if _looks_incomplete_module_answer(answer):
        # Always return something. lightweight_chat covers greetings and
        # falls through to a generic helpful sentence for anything else.
        answer = _module_fallback_answer(module_key, user_message)
    logger.info(
        "chat.module_chat_ok module=%s elapsed=%.2fs len=%s",
        module_key, time.perf_counter() - started_at, len(answer),
    )

    answer_text = str(answer or "").strip()
    # Canonical AHVI chat response shape. Every chat endpoint
    # (/api/text, /api/chat/module-chat) MUST return this same envelope
    # so the frontend parser doesn't have to special-case the source.
    return {
        "success": True,
        "type": "module_chat",
        "module": module_key,
        # Three keys carrying the same payload — different clients pick
        # different keys. Keep them aligned to avoid empty-message bugs.
        "response": answer_text,
        "message_text": answer_text,
        "message": {"role": "assistant", "content": answer_text},
        "cards": [],
        "style_boards": [],
        "chips": [],
        "data": {
            "module": module_key,
            "message": answer_text,
            "rendered_boards": [],
            "outfits": [],
        },
        "meta": {
            "mode": module_key,
            "board_count": 0,
        },
    }


@router.post("/module-chat")
@router.post("/chat/module-chat")
async def module_chat(request: ModuleChatRequest, http_request: Request):
    module = _normalize_module_name(request.domain or request.module or "")
    profile = dict(request.user_profile or {})
    user_id = _state_user_id(http_request)
    if user_id:
        profile["user_id"] = user_id
    user_message = str(request.message or "").strip()

    if module in {"skincare", "diet", "meal", "planner", "calendar", "medi", "bills", "fitness"}:
        return await handle_module_chat(
            {
                "domain": module,
                "module": module,
                "message": user_message,
                "context": {**(request.context_data or {}), **(request.context or {})},
                "user_profile": profile,
            },
            user_id=user_id,
        )

    return _module_llm_response(
        module=module,
        user_message=user_message,
        history=request.history,
        context_data={**(request.context_data or {}), **(request.context or {})},
        user_profile=profile,
    )


@router.post("/text")
def text_chat(request: TextChatRequest, http_request: Request):

    # -------------------------
    # INPUT VALIDATION
    # -------------------------
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_input = (request.messages[-1].content or "").strip()

    # Backwards-compat: older calendar.dart builds wrap user text as
    # "Occasion: <X>\n\n<user text>". Strip the prefix and promote
    # occasion into user_profile so the orchestrator sees it as a
    # structured slot instead of polluting the LLM prompt.
    import re as _re_occ
    _occ_match = _re_occ.match(r"^\s*Occasion:\s*([^\n]+)\n+", user_input, flags=_re_occ.IGNORECASE)
    if _occ_match:
        _legacy_occasion = _occ_match.group(1).strip().lower()
        user_input = user_input[_occ_match.end():].strip()
        if isinstance(request.user_profile, dict):
            if not request.user_profile.get("occasion"):
                request.user_profile["occasion"] = _legacy_occasion
        else:
            request.user_profile = {"occasion": _legacy_occasion}

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    # SECURITY: user_id MUST come from the authenticated bearer token.
    # Falling back to request body / "user_1" sentinel allows cross-account
    # cache + wardrobe contamination on the same instance.
    auth_user_id = ""
    state_user = getattr(http_request.state, "user", None)
    if isinstance(state_user, dict):
        auth_user_id = str(
            state_user.get("user_id") or state_user.get("$id") or ""
        ).strip()
    if not auth_user_id:
        raise HTTPException(
            status_code=401, detail="Authenticated user is required"
        )
    # If the client sent a user_id, it must match the authed user.
    for supplied in (
        (request.user_id or "").strip(),
        (request.userID or "").strip(),
        (
            str(request.user_profile.get("user_id") or "").strip()
            if isinstance(request.user_profile, dict)
            else ""
        ),
    ):
        if supplied and supplied != auth_user_id:
            raise HTTPException(
                status_code=403, detail="user_id does not match authenticated user"
            )
    user_id = auth_user_id
    user_message_style = _infer_user_message_style(user_input)
    profile_started = time.perf_counter()
    effective_user_profile = _ahvi_resolve_effective_user_profile(
        user_id,
        request.user_profile if isinstance(request.user_profile, dict) else {},
    )
    profile_ms = round((time.perf_counter() - profile_started) * 1000, 2)

    # -------------------------
    # FAST PATH
    # -------------------------
    if _is_fast_wardrobe_count_query(user_input):
        fast = _fast_wardrobe_count_response(user_id, user_input)
        fast["message"] = tone_engine.apply(
            str(fast.get("message") or ""),
            user_profile=effective_user_profile,
            signals={"context_mode": "home", "user_message_style": user_message_style},
            context={},
        )
        return fast

    # -------------------------
    # CACHE
    # -------------------------
    style_query = _is_explicit_style_request(user_input, request.module_context)
    style_action = str(request.style_action or "").strip().lower()
    visual_context = (
        str(request.module_context or "").lower() in {"style", "wardrobe"}
        or style_query
        or bool(style_action)
    )
    cache_key = _cache_key(
        user_input,
        user_id,
        module_context=str(request.module_context or ""),
        wardrobe=request.wardrobe,
        occasion=_ahvi_style_occasion(user_input) if visual_context else "",
    )
    include_base64_for_chat = bool(request.include_base64 and _CHAT_INCLUDE_BASE64_ALLOWED)
    if request.include_base64 and not _CHAT_INCLUDE_BASE64_ALLOWED:
        logger.info(
            "chat.base64_ignored user_id=%s module=%s style_query=%s style_action=%s",
            user_id,
            request.module_context or "",
            bool(style_query),
            style_action or "",
        )
    cache_visual_boards = bool(
        (include_base64_for_chat or style_query or style_action) and visual_context
    )
    cached = None if cache_visual_boards else _CHAT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    logger.info(
        "chat.text_request user_id=%s module=%s visual=%s style_query=%s style_action=%s include_base64=%s",
        user_id,
        request.module_context or "",
        bool(visual_context),
        bool(style_query),
        style_action or "",
        bool(include_base64_for_chat),
    )

    # -------------------------
    # LANGUAGE
    # -------------------------
    try:
        preferred_lang = (request.language or "en").lower()

        if preferred_lang in ("te", "hi"):
            english_input = GoogleTranslator(
                source=preferred_lang, target="en"
            ).translate(user_input)
            target_lang = preferred_lang
        else:
            english_input = user_input
            target_lang = "en"

    except Exception:
        english_input = user_input
        target_lang = "en"

    if visual_context:
        try:
            style_interpretation = interpret_occasion(
                english_input,
                {
                    "module_context": request.module_context or "style",
                    "user_profile": effective_user_profile,
                    "wardrobe": request.wardrobe,
                    "weather": weather_data.get("condition") if "weather_data" in locals() else None,
                },
            )
        except Exception:
            style_interpretation = {"board_generation_notes": {"occasion_kind": _ahvi_style_occasion(english_input)}}
        intent_status = "clarify" if _is_vague_style_prompt(english_input) else "generate"
        logger.info(
            "style_intent user_id=%s intent_status=%s prompt=%s interpreted_occasion=%s",
            user_id,
            intent_status,
            english_input,
            (style_interpretation.get("board_generation_notes") or {}).get("occasion_kind"),
        )
        if intent_status == "clarify":
            return _style_clarification_response(english_input, style_interpretation)

    # -------------------------
    # GENERAL CHAT / LLM ROUTE
    # -------------------------
    # Must happen before orchestrator, because orchestrator can classify broad
    # style/fashion language as daily_outfit and return static wardrobe messages.
    if _is_general_chat_request(english_input, request.module_context):
        response = _llm_chat_response(
            messages=request.messages,
            english_input=english_input,
            user_id=user_id,
            user_profile=effective_user_profile,
            user_message_style=user_message_style,
            module_context=request.module_context,
        )
        if not cache_visual_boards:
            _CHAT_CACHE.set(cache_key, response)
        return response

    # -------------------------
    # HYBRID ROUTING
    # -------------------------
    mode = _detect_mode(english_input, request.module_context)

    if mode == "greeting":
        return _llm_chat_response(
            messages=request.messages,
            english_input=english_input,
            user_id=user_id,
            user_profile=effective_user_profile,
            user_message_style=user_message_style,
            module_context=request.module_context,
        )

    if mode == "casual" and not style_query:
        return _llm_chat_response(
            messages=request.messages,
            english_input=english_input,
            user_id=user_id,
            user_profile=effective_user_profile,
            user_message_style=user_message_style,
            module_context=request.module_context,
        )

    # -------------------------
    # WEATHER
    # -------------------------
    weather_data = {}
    try:
        location = effective_user_profile.get("location") or {}
        lat, lon = location.get("lat"), location.get("lon")

        if lat is not None and lon is not None:
            weather_data = _get_weather_cached(lat=lat, lon=lon)

    except Exception as e:
        logger.warning("weather lookup failed %s", e)

    # -------------------------
    # ORCHESTRATOR (TIMEOUT SAFE)
    # -------------------------
    history = _build_history(request.messages[:-1]) if len(request.messages) > 1 else []
    memory_history = (
        request.current_memory.get("history", [])
        if isinstance(request.current_memory, dict)
        else []
    )
    merged_history = _normalize_memory_history(memory_history) + history

    def run():
        return ahvi_orchestrator.run(
            text=english_input,
            user_id=user_id,
            context={
                "memory": request.current_memory,
                "user_profile": effective_user_profile,
                "module_context": request.module_context,
                # Final style rendering is owned by style_flow_service after
                # card sanitization; avoid stale pre-rendered boards here.
                "include_base64": False,
                "wardrobe": request.wardrobe,
                "history": merged_history[-20:],
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
                "signals": {"user_message_style": user_message_style},
                "style_action": style_action,
                "exclude_style_signatures": [
                    str(x or "").strip().lower()
                    for x in (request.exclude_style_signatures or [])
                    if str(x or "").strip()
                ],
                "requested_board_count": request.requested_board_count,
            },
        )

    try:
        result = _ORCHESTRATOR_EXECUTOR.submit(run).result(
            timeout=_ORCH_TIMEOUT_SECONDS
        )
    except concurrent.futures.TimeoutError:
        logger.exception("chat.orchestrator_timeout user_id=%s prompt=%s", user_id, english_input)
        return _structured_error_response(
            code="style_timeout" if visual_context else "chat_timeout",
            message="AHVI's styling engine timed out. Please try again in a moment.",
            status_type="provider_timeout",
            details={"timeout_seconds": _ORCH_TIMEOUT_SECONDS, "module_context": request.module_context or ""},
            chips=[{"label": "Try again", "value": english_input}],
        )
    except Exception as exc:
        logger.exception("chat.orchestrator_exception user_id=%s prompt=%s", user_id, english_input)
        return _structured_error_response(
            code="style_generation_error" if visual_context else "chat_generation_error",
            message="AHVI hit a backend error while preparing this. Please try again in a moment.",
            status_type="backend_error",
            details={"error": str(exc)[:240], "module_context": request.module_context or ""},
            chips=[{"label": "Try again", "value": english_input}],
        )

    message = result.get("message") or ""
    if isinstance(message, dict):
        message = str(message.get("content") or "")
    else:
        message = str(message or "")

    # -------------------------
    # TRANSLATE BACK
    # -------------------------
    try:
        if target_lang != "en" and message:
            lower_msg = message.strip().lower()
            if lower_msg in ("hi", "hello", "hey", "hi there", "hello there"):
                pass
            else:
                message = GoogleTranslator(source="en", target=target_lang).translate(
                    message
                )
    except Exception:
        pass

    # Final guardrail: every /api/text answer should leave through the tone layer,
    # even if a newer orchestrator branch forgot to apply it internally.
    try:
        message = tone_engine.apply(
            message,
            user_profile=effective_user_profile,
            signals={
                "context_mode": request.module_context or "chat",
                "user_message_style": user_message_style,
            },
            context={
                "module_context": request.module_context,
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
            },
        )
    except Exception:
        pass

    # If this is style chat and the orchestrator came back without a visual payload,
    # build a deterministic wardrobe board so the demo never lands as plain text only.
    data_payload = result.get("data") or {}
    cards_payload = result.get("cards") or []
    style_payload = {}

    # AHVI no visual boards on error responses:
    # If the orchestrator failed, do not attach deterministic fallback boards.
    # This avoids showing "temporary issue" text and outfit boards together.
    result_message_text = str(result.get("message") or "").lower()
    is_error_style_response = any(
        marker in result_message_text
        for marker in (
            "temporary issue",
            "please try again",
            "pipeline temporarily unavailable",
            "no outfits generated",
        )
    )

    if is_error_style_response:
        cards_payload = []
        data_payload = {"outfits": [], "rendered_boards": []}
        result["board_ids"] = ""
    else:
        has_visual_board = bool(
            isinstance(cards_payload, list) and cards_payload
        ) or bool(
            isinstance(data_payload, dict)
            and (data_payload.get("rendered_boards") or data_payload.get("outfits"))
        )
        if visual_context and not has_visual_board:
            style_payload = _demo_style_board_payload(
                user_id, english_input, request.wardrobe, effective_user_profile
            )
            if style_payload.get("cards"):
                cards_payload = style_payload.get("cards") or []
                data_payload = style_payload.get("data") or {}
                result["type"] = style_payload.get("type") or result.get("type")
                result["board_ids"] = (
                    style_payload.get("board_ids") or result.get("board_ids") or ""
                )
            result["meta"] = {
                **(result.get("meta") or {}),
                **(style_payload.get("meta") or {}),
            }
        lower_message = (message or "").lower()
        if (
            not message
            or "clarification" in lower_message
            or "balance isn't quite" in lower_message
            or (
                style_payload.get("cards")
                and "i will assume smart casual" in lower_message
            )
        ):
            replacement = (
                style_payload.get("message")
                or "I will assume smart casual for today: start with a clean hero piece, add a neutral base, and finish with footwear or one accessory. Once your wardrobe has saved items, I will pick the exact pieces from it."
            )
            try:
                message = tone_engine.apply(
                    replacement,
                    user_profile=effective_user_profile,
                    signals={
                        "context_mode": request.module_context or "style",
                        "user_message_style": user_message_style,
                    },
                    context={"module_context": request.module_context},
                )
            except Exception:
                message = replacement

    # -------------------------
    # AUDIO
    # -------------------------
    try:
        audio_job_id = (
            enqueue_task(
                task_func=run_heavy_audio_task,
                args=[message, target_lang],
                kwargs={
                    "request_id": str(
                        getattr(http_request.state, "request_id", "") or ""
                    )
                },
                kind="chat_audio",
                user_id=user_id,
                source="routers.chat.text",
                request_id=str(getattr(http_request.state, "request_id", "") or ""),
            )
            if run_heavy_audio_task
            else "offline"
        )
    except Exception:
        audio_job_id = "offline"

    # -------------------------
    # FINAL RESPONSE
    # -------------------------
    if not isinstance(cards_payload, list):
        cards_payload = []

    board_ids_text = str(result.get("board_ids") or "")
    result_meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    already_finalized_style_payload = (
        result_meta.get("mode") == "style_flow_service_adapter_v1"
        and isinstance(data_payload, dict)
        and bool(data_payload.get("outfits") or data_payload.get("rendered_boards"))
    )

    if (
        visual_context
        and isinstance(cards_payload, list)
        and cards_payload
        and not already_finalized_style_payload
    ):
        canonical_style = finalize_style_response_payload(
            {**result, "cards": cards_payload, "data": data_payload},
            user_id=user_id,
            query=english_input,
            context={
                "query": english_input,
                "occasion": _ahvi_style_occasion(english_input),
                "user_profile": effective_user_profile,
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
            },
            include_base64=include_base64_for_chat,
            style_action=style_action,
            exclude_style_signatures=request.exclude_style_signatures,
            requested_board_count=request.requested_board_count,
            cache_bypass=bool(cache_visual_boards),
        )
        cards_payload = canonical_style["cards"]
        data_payload = canonical_style["data"]
        board_ids_text = str(canonical_style.get("board_ids") or board_ids_text)
        result["meta"] = {
            **(result.get("meta") or {}),
            **(canonical_style.get("meta") or {}),
        }
        result["style_boards"] = canonical_style.get("style_boards") or cards_payload
    elif already_finalized_style_payload:
        result["style_boards"] = result.get("style_boards") or cards_payload
        logger.info(
            "chat.skip_duplicate_style_finalize user_id=%s cards=%d",
            user_id,
            len(cards_payload),
        )

    logger.info(
        "chat.text_response user_id=%s cards=%d signatures=%s style_action=%s profile_ms=%s",
        user_id,
        len(cards_payload),
        [style_card_signature(c) for c in cards_payload],
        style_action or "",
        profile_ms,
    )

    final_signatures = [
        style_card_signature(c) for c in cards_payload if isinstance(c, dict)
    ]
    style_signature = hashlib.sha1(
        "|".join(final_signatures).encode("utf-8")
    ).hexdigest() if final_signatures else ""

    response = {
        "success": True,
        "message": message,
        "board": result.get("board"),
        "type": result.get("type"),
        "style_boards": cards_payload if visual_context else [],
        "cards": cards_payload,
        "chips": (
            result.get("chips")
            if isinstance(result.get("chips"), list)
            else (
                style_payload.get("chips")
                if isinstance(style_payload.get("chips"), list)
                else (_ahvi_style_action_chips() if visual_context and cards_payload else [])
            )
        ),
        "board_ids": board_ids_text,
        "data": data_payload if isinstance(data_payload, dict) else {},
        "meta": {
            **(result.get("meta") or {}),
            "weather": weather_data,
            "history_used": len(merged_history[-20:]),
            "style_action": style_action or None,
            "style_signature": style_signature or None,
            "board_count": len(cards_payload),
            "has_more_style_options": bool(visual_context and len(cards_payload) > 0),
            "style_cache_bypass": bool(cache_visual_boards),
        },
        "audio_job_id": audio_job_id,
    }

    # -------------------------
    # CACHE SAVE
    # -------------------------
    if not cache_visual_boards:
        _CHAT_CACHE.set(cache_key, response)

    return response
