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

from deep_translator import GoogleTranslator

try:
    from worker import run_heavy_audio_task
except Exception:
    run_heavy_audio_task = None

from brain.orchestrator import ahvi_orchestrator
from brain.tone.tone_engine import tone_engine
from brain.outfit_pipeline import save_feedback
from services.appwrite_proxy import AppwriteProxy
try:
    from services.job_tracker import job_tracker
except Exception:
    job_tracker = None
from services.task_queue import enqueue_task

# ðŸ”¥ NEW
from services.weather_service import get_hourly_weather

router = APIRouter()
logger = logging.getLogger("ahvi.routers.chat")

_CHAT_CACHE_MAX_ITEMS = max(64, int(os.getenv("CHAT_CACHE_MAX_ITEMS", "512")))
_CHAT_CACHE_TTL_SECONDS = max(15, int(os.getenv("CHAT_CACHE_TTL_SECONDS", "60")))
_WEATHER_CACHE_MAX_ITEMS = max(32, int(os.getenv("WEATHER_CACHE_MAX_ITEMS", "256")))
_WEATHER_CACHE_TTL_SECONDS = max(60, int(os.getenv("WEATHER_CACHE_TTL_SECONDS", "900")))
_ORCH_TIMEOUT_SECONDS = max(2, int(os.getenv("CHAT_ORCHESTRATOR_TIMEOUT_SECONDS", "8")))
_ORCHESTRATOR_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(2, int(os.getenv("CHAT_ORCHESTRATOR_MAX_WORKERS", "8"))),
    thread_name_prefix="chat-orch",
)


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


def _cache_key(text, user_id):
    return f"{user_id}:{text.lower().strip()}"


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
        "wardrobe", "closet", "outfit", "outfits", "tops", "top", "shirts", "shirt",
        "pants", "trousers", "jeans", "bottoms", "shoes", "footwear", "dress",
        "dresses", "accessories", "jewelry", "bags", "bag",
    ]
    return any(k in lowered for k in count_words) and any(k in lowered for k in wardrobe_words)


def _chat_tokens(value: Any) -> List[str]:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip().split()


def _chat_has_any(tokens: List[str], words: List[str]) -> bool:
    return any(word in tokens for word in words)


def _infer_chat_category(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return "Accessories"

    explicit = str(item.get("category") or item.get("cat") or item.get("type") or "").strip().lower()
    explicit_map = {
        "top": "Tops",
        "tops": "Tops",
        "shirt": "Tops",
        "tshirt": "Tops",
        "t-shirt": "Tops",
        "bottom": "Bottoms",
        "bottoms": "Bottoms",
        "pants": "Bottoms",
        "trousers": "Bottoms",
        "jeans": "Bottoms",
        "shorts": "Bottoms",
        "footwear": "Footwear",
        "shoe": "Footwear",
        "shoes": "Footwear",
        "accessory": "Accessories",
        "accessories": "Accessories",
        "bag": "Accessories",
        "bags": "Accessories",
        "jewelry": "Accessories",
        "jewellery": "Accessories",
        "outerwear": "Outerwear",
        "outer": "Outerwear",
        "dress": "Dresses",
        "dresses": "Dresses",
        "indian wear": "Dresses",
    }

    if explicit in explicit_map:
        return explicit_map[explicit]

    joined = " ".join(
        str(item.get(k, "") or "")
        for k in (
            "category",
            "category_group",
            "cat",
            "type",
            "name",
            "label",
            "sub_category",
            "subcategory",
            "subCategory",
            "description",
        )
    )

    tokens = _chat_tokens(joined)

    # Tops first: Short-Sleeved Shirt must be Tops.
    if _chat_has_any(tokens, [
        "shirt", "shirts", "tee", "tshirt", "tshirts", "top", "tops",
        "blouse", "blouses", "hoodie", "hoodies", "sweater", "sweaters",
        "kurta", "kurtas", "polo", "polos",
    ]):
        return "Tops"

    # Only shorts, never short.
    if _chat_has_any(tokens, [
        "pants", "pant", "trousers", "trouser", "jeans", "jean",
        "shorts", "skirt", "skirts", "legging", "leggings", "chino", "chinos",
    ]):
        return "Bottoms"

    if _chat_has_any(tokens, [
        "shoe", "shoes", "boot", "boots", "sneaker", "sneakers",
        "heel", "heels", "sandal", "sandals", "loafer", "loafers",
        "slipper", "slippers",
    ]):
        return "Footwear"

    if _chat_has_any(tokens, [
        "watch", "watches", "bag", "bags", "belt", "belts",
        "scarf", "scarves", "jewelry", "jewellery", "ring", "rings",
        "necklace", "bracelet", "earring", "earrings", "accessory",
        "accessories", "hat", "cap", "sunglass", "sunglasses",
    ]):
        return "Accessories"

    if _chat_has_any(tokens, ["jacket", "coat", "blazer", "outerwear", "cardigan", "overshirt"]):
        return "Outerwear"

    if _chat_has_any(tokens, ["dress", "dresses", "gown", "jumpsuit", "saree", "lehenga", "sherwani"]):
        return "Dresses"

    return "Accessories"


def _fast_wardrobe_count_response(user_id: str, query_text: str) -> Dict[str, Any]:
    try:
        docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=100)
    except Exception:
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
    if any(k in lowered for k in ["top", "tops", "shirt", "shirts", "blouse", "blouses"]):
        message = f"You have {counts['tops']} tops in your wardrobe."
    elif any(k in lowered for k in ["bottom", "bottoms", "pant", "pants", "trouser", "trousers", "jean", "jeans"]):
        message = f"You have {counts['bottoms']} bottoms in your wardrobe."
    elif any(k in lowered for k in ["shoe", "shoes", "footwear", "sneaker", "sneakers"]):
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
            {"id": "bottoms", "title": "Bottoms", "kind": "stat", "value": counts["bottoms"]},
            {"id": "shoes", "title": "Shoes", "kind": "stat", "value": counts["shoes"]},
            {"id": "dresses", "title": "Dresses", "kind": "stat", "value": counts["dresses"]},
            {"id": "accessories", "title": "Accessories", "kind": "stat", "value": counts["accessories"]},
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
_AHVI_FEMALE_STYLE_GENDERS = {"f", "female", "woman", "women", "womens", "girl", "ladies"}
_AHVI_UNISEX_STYLE_GENDERS = {"unisex", "neutral", "genderless", "any"}

_AHVI_FEMININE_ONLY_GARMENTS = {
    "saree", "sari", "lehenga", "gown", "skirt", "skirts", "blouse", "kurti"
}
_AHVI_MALE_TRADITIONAL_GARMENTS = {"sherwani", "achkan"}

_AHVI_EXPLICIT_FEMININE_REQUEST = {
    "saree", "sari", "lehenga", "gown", "skirt", "skirts",
    "female", "women", "woman", "ladies", "feminine",
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
        candidates.extend([
            nested.get("style_gender"),
            nested.get("gender"),
            nested.get("preferred_gender"),
            nested.get("target_gender"),
        ])

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
            "slot", "role", "type", "category", "cat", "category_group",
            "sub_category", "subcategory", "subCategory",
            "name", "label", "description", "gender",
            "style_gender", "target_gender", "audience",
            "department", "intended_for", "wearer",
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
            "gender", "style_gender", "target_gender",
            "audience", "department", "intended_for", "wearer",
        )
    )
    audience_tokens = set(_chat_tokens(audience_blob))

    if audience_tokens.intersection(_AHVI_FEMALE_STYLE_GENDERS):
        return False

    if tokens.intersection(_AHVI_FEMININE_ONLY_GARMENTS):
        return False

    if _infer_chat_category(item) == "Dresses" and not tokens.intersection(_AHVI_MALE_TRADITIONAL_GARMENTS):
        return False

    return True


def _fetch_wardrobe_for_style(user_id: str, request_wardrobe: Any) -> List[Dict[str, Any]]:
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
    if any(k in q for k in ["coffee", "casual", "outing", "weekend", "street", "sport", "outdoor"]):
        return "casual outing"
    return "today"


def _ahvi_style_image(item):
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("masked_url")
        or item.get("maskedUrl")
        or item.get("image_url")
        or item.get("imageUrl")
        or item.get("raw_url")
        or item.get("rawUrl")
        or item.get("url")
        or item.get("image")
        or ""
    ).strip()


def _ahvi_style_blob(item):
    if not isinstance(item, dict):
        return ""
    return " ".join(
        str(item.get(k, "") or "")
        for k in (
            "role", "slot", "type", "category", "cat", "category_group",
            "sub_category", "subcategory", "subCategory",
            "name", "label", "description", "pattern", "color", "color_name",
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

    if tokens.intersection({
        "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
        "heel", "heels", "sandal", "sandals", "loafer", "loafers",
        "slipper", "slippers", "slider", "sliders", "footwear"
    }):
        return "footwear"

    if tokens.intersection({
        "watch", "watches", "belt", "belts", "cap", "caps", "hat", "hats",
        "sunglass", "sunglasses", "eyewear", "glasses", "bag", "bags",
        "purse", "handbag", "clutch", "tote", "jewelry", "jewellery",
        "ring", "rings", "necklace", "necklaces", "bracelet", "bracelets",
        "earring", "earrings", "scarf", "scarves", "accessory", "accessories"
    }):
        return "accessory"

    if tokens.intersection({
        "dress", "dresses", "saree", "sari", "lehenga", "gown", "jumpsuit", "sherwani"
    }):
        return "dress"

    # Tops before bottoms: short-sleeved shirt must not become shorts.
    if tokens.intersection({
        "top", "tops", "shirt", "shirts", "tee", "tshirt", "tshirts",
        "polo", "polos", "jacket", "blazer", "sweater", "hoodie",
        "kurta", "kurti", "tunic", "tunics"
    }):
        return "top"

    if tokens.intersection({
        "bottom", "bottoms", "pant", "pants", "trouser", "trousers",
        "jean", "jeans", "shorts", "skirt", "skirts", "chino", "chinos"
    }):
        return "bottom"

    return "unknown"


def _ahvi_style_norm(item, role=None):
    row = dict(item or {})
    resolved = role or _ahvi_style_role(row)

    if resolved == "top":
        row["role"] = "top"
        row["slot"] = "top"
        row["category"] = "Tops"
    elif resolved == "bottom":
        row["role"] = "bottom"
        row["slot"] = "bottom"
        row["category"] = "Bottoms"
    elif resolved == "footwear":
        row["role"] = "footwear"
        row["slot"] = "footwear"
        row["category"] = "Footwear"
    elif resolved == "dress":
        row["role"] = "dress"
        row["slot"] = "dress"
        row["category"] = "Dresses"
    elif resolved == "accessory":
        row["role"] = "accessory"
        row["slot"] = "accessory"
        row["category"] = "Accessories"

    image = _ahvi_style_image(row)
    if image:
        row["image_url"] = row.get("image_url") or image
        row["imageUrl"] = row.get("imageUrl") or image
        row["masked_url"] = row.get("masked_url") or image
        row["maskedUrl"] = row.get("maskedUrl") or image

    return row


def _ahvi_style_accessory_type(item):
    blob = _ahvi_style_blob(item)
    if "watch" in blob:
        return "watch"
    if "belt" in blob:
        return "belt"
    if "cap" in blob or "hat" in blob:
        return "headwear"
    if "bag" in blob:
        return "bag"
    if any(k in blob for k in ["ring", "necklace", "bracelet", "earring", "jewelry", "jewellery"]):
        return "jewelry"
    if "sunglass" in blob or "eyewear" in blob or "glasses" in blob:
        return "eyewear"
    if "scarf" in blob:
        return "scarf"
    return "accessory"


def _ahvi_style_pools(wardrobe, query_text, profile):
    buckets = {"top": [], "bottom": [], "dress": [], "footwear": [], "accessory": []}
    seen = {k: set() for k in buckets}

    for item in wardrobe or []:
        if not isinstance(item, dict):
            continue
        if not _ahvi_style_image(item):
            continue
        if not _ahvi_item_allowed_for_user_profile(item, profile, query_text):
            continue

        role = _ahvi_style_role(item)
        if role not in buckets:
            continue

        key = _ahvi_style_key(item)
        if key in seen[role]:
            continue

        seen[role].add(key)
        buckets[role].append(item)

    return buckets


def _ahvi_style_pick(pool, used, fallback=None):
    for item in pool or []:
        key = _ahvi_style_key(item)
        if key and key not in used:
            used.add(key)
            return item

    if fallback is not None:
        key = _ahvi_style_key(fallback)
        if key:
            used.add(key)
        return fallback

    if pool:
        item = pool[0]
        key = _ahvi_style_key(item)
        if key:
            used.add(key)
        return item

    return None


def _ahvi_style_clean_accessories(accessories, query_text):
    q = str(query_text or "").lower()
    headwear_allowed = any(k in q for k in [
        "casual", "street", "travel", "airport", "sport", "gym",
        "sun", "beach", "outdoor", "college", "weekend"
    ])

    out = []
    seen_types = set()
    seen_ids = set()

    for item in accessories or []:
        if not isinstance(item, dict):
            continue

        key = _ahvi_style_key(item)
        if key and key in seen_ids:
            continue

        typ = _ahvi_style_accessory_type(item)

        if typ == "headwear" and not headwear_allowed:
            continue

        # one watch max, one belt max, one cap max, etc.
        if typ in seen_types:
            continue

        out.append(_ahvi_style_norm(item, "accessory"))

        if key:
            seen_ids.add(key)
        seen_types.add(typ)

        max_count = 2 if headwear_allowed else 1
        if len(out) >= max_count:
            break

    return out


def _ahvi_style_names(items):
    return [
        str((i or {}).get("name") or (i or {}).get("label") or (i or {}).get("category") or "")
        for i in items or []
        if isinstance(i, dict)
    ]



def _ahvi_router_style_fallback_enabled():
    return os.getenv("AHVI_ENABLE_ROUTER_STYLE_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}


def _ahvi_style_card_roles(card):
    roles = set()

    if not isinstance(card, dict):
        return roles

    def add_item_role(item):
        if not isinstance(item, dict):
            return
        role = str(item.get("role") or item.get("slot") or "").strip().lower()
        if not role:
            role = _ahvi_style_role(item)
        if role in {"top", "bottom", "footwear", "accessory", "dress"}:
            roles.add(role)

    for key in ("items", "accessories"):
        value = card.get(key)
        if isinstance(value, list):
            for item in value:
                add_item_role(item)

    return roles


def _ahvi_style_card_is_complete(card):
    roles = _ahvi_style_card_roles(card)

    # One-piece outfit path: dress/sherwani/etc + footwear is acceptable.
    if "dress" in roles and "footwear" in roles:
        return True

    # Standard AHVI outfit path: top + bottom + footwear required.
    return "top" in roles and "bottom" in roles and "footwear" in roles


def _ahvi_filter_complete_style_cards(cards, user_id):
    if not isinstance(cards, list):
        return []

    complete = [card for card in cards if _ahvi_style_card_is_complete(card)]

    if len(complete) != len(cards):
        try:
            logger.info(
                "ahvi.router_incomplete_style_cards_blocked user_id=%s before=%s after=%s",
                user_id,
                len(cards),
                len(complete),
            )
        except Exception:
            pass

    return complete


def _ahvi_sanitize_style_cards(cards, user_id, query_text, request_wardrobe=None, user_profile=None):
    if not isinstance(cards, list):
        return []

    profile = _ahvi_resolve_effective_user_profile(user_id, user_profile or {})
    wardrobe = _fetch_wardrobe_for_style(user_id, request_wardrobe)
    buckets = _ahvi_style_pools(wardrobe, query_text, profile)

    if not cards:
        if not _ahvi_router_style_fallback_enabled():
            try:
                logger.info(
                    "ahvi.router_style_fallback_disabled user_id=%s stage=sanitize_empty_cards reason=production_no_fake_boards",
                    user_id,
                )
            except Exception:
                pass
            return []
        # Dev/demo only: build empty shells; sanitizer will fill from wardrobe pools.
        cards = [{"id": f"style_card_{i}", "title": f"Look {i + 1} · Styled Fit", "items": [], "accessories": []} for i in range(3)]

    used_top = set()
    used_bottom = set()
    used_dress = set()
    used_footwear = set()

    cleaned = []

    for idx, card in enumerate(cards[:3]):
        if not isinstance(card, dict):
            continue

        source_items = []
        for key in ("items", "accessories"):
            value = card.get(key)
            if isinstance(value, list):
                source_items.extend([x for x in value if isinstance(x, dict)])

        top = next((x for x in source_items if _ahvi_style_role(x) == "top"), None)
        bottom = next((x for x in source_items if _ahvi_style_role(x) == "bottom"), None)
        dress = next((x for x in source_items if _ahvi_style_role(x) == "dress"), None)
        footwear = next((x for x in source_items if _ahvi_style_role(x) == "footwear"), None)
        accessories = [x for x in source_items if _ahvi_style_role(x) == "accessory"]

        final_items = []

        if dress and not (top and bottom):
            chosen_dress = _ahvi_style_pick(buckets["dress"], used_dress, dress)
            if chosen_dress:
                final_items.append(_ahvi_style_norm(chosen_dress, "dress"))
        else:
            chosen_top = _ahvi_style_pick(buckets["top"], used_top, top)
            chosen_bottom = _ahvi_style_pick(buckets["bottom"], used_bottom, bottom)

            if chosen_top:
                final_items.append(_ahvi_style_norm(chosen_top, "top"))
            if chosen_bottom:
                final_items.append(_ahvi_style_norm(chosen_bottom, "bottom"))

        chosen_footwear = _ahvi_style_pick(buckets["footwear"], used_footwear, footwear)
        if chosen_footwear:
            final_items.append(_ahvi_style_norm(chosen_footwear, "footwear"))

        final_items.extend(_ahvi_style_clean_accessories(accessories or buckets["accessory"], query_text))

        fixed = dict(card)
        fixed["items"] = final_items
        fixed["accessories"] = []

        top_name = next((str(i.get("name") or i.get("label") or "top") for i in final_items if _ahvi_style_role(i) in {"top", "dress"}), "")
        bottom_name = next((str(i.get("name") or i.get("label") or "bottom") for i in final_items if _ahvi_style_role(i) == "bottom"), "")
        footwear_name = next((str(i.get("name") or i.get("label") or "footwear") for i in final_items if _ahvi_style_role(i) == "footwear"), "")
        core = ", ".join([x for x in [top_name, bottom_name, footwear_name] if x])

        q = str(query_text or "").lower()
        if "date" in q:
            why = f"This works for date night because {core} creates a clean smart-casual balance without over-accessorizing."
        elif any(k in q for k in ["office", "meeting", "work"]):
            why = f"This works for office because {core} keeps the outfit structured, neat, and wearable."
        else:
            why = f"This works because {core} creates a balanced top-bottom-footwear structure."

        fixed["why_it_works"] = why
        fixed["explanation"] = why
        fixed["reason"] = why
        fixed["style_reason"] = why

        title = str(fixed.get("title") or fixed.get("name") or "").strip()
        if not title or title.lower() in {"style board", "ahvi styled look"}:
            fixed["title"] = f"Look {idx + 1} · Styled Fit"
            fixed["name"] = fixed["title"]

        cleaned.append(fixed)

    try:
        logger.info(
            "ahvi.clean_chat_style_guard_v2 user_id=%s cards=%s signatures=%s accessory_counts=%s",
            user_id,
            len(cleaned),
            [" | ".join(_ahvi_style_names(c.get("items") or [])) for c in cleaned],
            [len([i for i in (c.get("items") or []) if isinstance(i, dict) and _ahvi_style_role(i) == "accessory"]) for c in cleaned],
        )
    except Exception:
        pass

    return cleaned


def _demo_style_board_payload(user_id, query_text, request_wardrobe, user_profile=None):
    profile = _ahvi_resolve_effective_user_profile(user_id, user_profile or {})
    wardrobe = _fetch_wardrobe_for_style(user_id, request_wardrobe)
    wardrobe = [
        item for item in wardrobe
        if _ahvi_item_allowed_for_user_profile(item, profile, query_text)
    ]

    occasion = _ahvi_style_occasion(query_text)

    try:
        from brain.outfit_pipeline import get_daily_outfits

        result = get_daily_outfits({
            "user_id": user_id,
            "wardrobe": wardrobe,
            "context": {
                "occasion": occasion,
                "query": query_text,
                "user_profile": profile,
                "style_gender": _ahvi_profile_style_gender(profile),
                "signals": {
                    "source": "routers.chat.clean_style_adapter_v2",
                    "style_gender": _ahvi_profile_style_gender(profile),
                },
            },
        })

        if not isinstance(result, dict):
            result = {}

    except Exception as exc:
        logger.warning("ahvi.clean_chat_style_adapter_v2 pipeline_failed user_id=%s error=%s", user_id, str(exc)[:180])
        result = {}

    raw_cards = result.get("cards") if isinstance(result.get("cards"), list) else []

    if not raw_cards and not _ahvi_router_style_fallback_enabled():
        try:
            logger.info(
                "ahvi.router_style_fallback_disabled user_id=%s stage=demo_payload_empty_raw_cards reason=production_no_fake_boards",
                user_id,
            )
        except Exception:
            pass
        return {
            "success": False,
            "message": (
                "I couldn't build a reliable style board from your wardrobe yet. "
                "Please add at least one top, bottom, and footwear item."
            ),
            "board": "style",
            "type": "missing_outfit_cards",
            "cards": [],
            "board_ids": "",
            "data": {
                "outfits": result.get("outfits") if isinstance(result.get("outfits"), list) else [],
                "rendered_boards": result.get("rendered_boards") if isinstance(result.get("rendered_boards"), list) else [],
                "board_item_ids": result.get("board_item_ids") if isinstance(result.get("board_item_ids"), list) else [],
            },
            "meta": {
                "mode": "router_style_fallback_disabled",
                "fallback_used": False,
                "error": "router_deterministic_style_fallback_disabled",
                "error_stage": "routers.chat",
                "occasion": occasion,
                "wardrobe_count": len(wardrobe),
            },
        }

    cards = _ahvi_sanitize_style_cards(raw_cards, user_id, query_text, wardrobe, profile)

    if not _ahvi_router_style_fallback_enabled():
        cards = _ahvi_filter_complete_style_cards(cards, user_id)
        if not cards:
            try:
                logger.info(
                    "ahvi.router_style_fallback_disabled user_id=%s stage=incomplete_cards_blocked reason=missing_required_slots",
                    user_id,
                )
            except Exception:
                pass
            return {
                "success": False,
                "message": (
                    "I couldn't build a complete style board from your wardrobe yet. "
                    "Please add at least one top, bottom, and footwear item."
                ),
                "board": "style",
                "type": "missing_outfit_cards",
                "cards": [],
                "board_ids": "",
                "data": {
                    "outfits": result.get("outfits") if isinstance(result.get("outfits"), list) else [],
                    "rendered_boards": result.get("rendered_boards") if isinstance(result.get("rendered_boards"), list) else [],
                    "board_item_ids": result.get("board_item_ids") if isinstance(result.get("board_item_ids"), list) else [],
                },
                "meta": {
                    "mode": "router_incomplete_cards_blocked",
                    "fallback_used": False,
                    "error": "missing_required_style_slots",
                    "error_stage": "routers.chat",
                    "occasion": occasion,
                    "wardrobe_count": len(wardrobe),
                    "required_slots": ["top", "bottom", "footwear"],
                },
            }

    board_item_ids = result.get("board_item_ids") if isinstance(result.get("board_item_ids"), list) else []
    board_ids = ",".join([str(x) for x in board_item_ids if str(x).strip()])

    first_names = ", ".join(_ahvi_style_names((cards[0].get("items") if cards else []) or [])[:5])
    message = (
        result.get("message")
        or result.get("context")
        or f"Here are {len(cards)} {occasion} boards from your wardrobe. First look: {first_names}."
    )

    try:
        logger.info(
            "ahvi.clean_chat_style_adapter_v2 user_id=%s occasion=%s wardrobe=%s cards=%s first_card_items=%s",
            user_id,
            occasion,
            len(wardrobe),
            len(cards),
            _ahvi_style_names((cards[0].get("items") if cards else []) or [])[:8],
        )
    except Exception:
        pass

    return {
        "success": True,
        "message": str(message),
        "board": "style",
        "type": "cards",
        "cards": cards,
        "board_ids": board_ids,
        "data": {
            "outfits": cards,
            "rendered_boards": [],
            "board_item_ids": board_item_ids,
            "pipeline": result.get("pipeline") if isinstance(result.get("pipeline"), dict) else {},
            "style_gender": _ahvi_profile_style_gender(profile),
        },
        "meta": {
            "intent": "style_pipeline_adapter",
            "domain": "style",
            "mode": "clean_outfit_pipeline_adapter_v2",
            "wardrobe_count": len(wardrobe),
            "style_gender": _ahvi_profile_style_gender(profile),
            "occasion": occasion,
        },
    }


def _ahvi_final_response_style_guard(cards, user_id, query_text, request_wardrobe=None, user_profile=None):
    return _ahvi_sanitize_style_cards(cards, user_id, query_text, request_wardrobe, user_profile)

# ================= AHVI CLEAN CHAT STYLE V2 END =================


def _detect_mode(text: str) -> str:
    t = text.lower().strip()

    if any(k in t for k in ["wear","outfit","dress","style","clothes","wardrobe","look"]):
        return "fashion"

    if t in ["hi","hello","hey"]:
        return "greeting"

    if any(k in t for k in ["how are","what is","who is","tell me","why","joke","explain"]):
        return "casual"

    return "fashion"


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

@router.post("/text")
def text_chat(request: TextChatRequest, http_request: Request):

    # -------------------------
    # INPUT VALIDATION
    # -------------------------
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_input = (request.messages[-1].content or "").strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    # Resolve user_id with this priority:
    #   1. authenticated user attached by auth middleware (most trustworthy)
    #   2. top-level user_id / userID on the request
    #   3. user_profile.user_id (legacy clients put it here)
    #   4. "user_1" sentinel (dev-only fallback; cache key collisions if reached)
    auth_user_id = ""
    state_user = getattr(http_request.state, "user", None)
    if isinstance(state_user, dict):
        auth_user_id = str(state_user.get("user_id") or state_user.get("$id") or "").strip()
    profile_user_id = ""
    if isinstance(request.user_profile, dict):
        profile_user_id = str(request.user_profile.get("user_id") or "").strip()
    user_id = (
        auth_user_id
        or (request.user_id or "").strip()
        or (request.userID or "").strip()
        or profile_user_id
        or "user_1"
    )
    user_message_style = _infer_user_message_style(user_input)

    # -------------------------
    # FAST PATH
    # -------------------------
    if _is_fast_wardrobe_count_query(user_input):
        fast = _fast_wardrobe_count_response(user_id, user_input)
        fast["message"] = tone_engine.apply(
            str(fast.get("message") or ""),
            user_profile=request.user_profile,
            signals={"context_mode": "home", "user_message_style": user_message_style},
            context={},
        )
        return fast

    # -------------------------
    # CACHE
    # -------------------------
    cache_key = _cache_key(user_input, user_id)
    style_query = any(k in user_input.lower() for k in ["wear", "outfit", "dress", "style", "clothes", "wardrobe", "look", "casual", "date night"])
    visual_context = str(request.module_context or "").lower() in {"style", "wardrobe"} or style_query
    cache_visual_boards = bool((request.include_base64 or style_query) and visual_context)
    cached = None if cache_visual_boards else _CHAT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # -------------------------
    # LANGUAGE
    # -------------------------
    try:
        preferred_lang = (request.language or "en").lower()

        if preferred_lang in ("te", "hi"):
            english_input = GoogleTranslator(source=preferred_lang, target="en").translate(user_input)
            target_lang = preferred_lang
        else:
            english_input = user_input
            target_lang = "en"

    except Exception:
        english_input = user_input
        target_lang = "en"

    # -------------------------
    # HYBRID ROUTING
    # -------------------------
    mode = _detect_mode(english_input)

    general_chat_prompt = any(k in english_input.lower() for k in ["joke", "how are you", "how r you", "what are you doing"])
    if general_chat_prompt:
        try:
            casual_message = lightweight_chat(english_input)
        except Exception:
            casual_message = "I am here and ready. I can chat, make you smile, or help style your next look."
        return {
            "success": True,
            "message": tone_engine.apply(
                casual_message,
                user_profile=request.user_profile,
                signals={"context_mode": "home", "user_message_style": user_message_style},
                context={},
            ),
            "cards": [],
            "meta": {"mode": "casual_fast"},
            "audio_job_id": "offline",
        }

    if mode == "greeting":
        return {
            "success": True,
            "message": tone_engine.apply(
                "Hey, I can help you style outfits or just chat.",
                user_profile=request.user_profile,
                signals={"context_mode": "home", "user_message_style": user_message_style},
                context={},
            ),
            "cards": [],
            "meta": {"mode": "greeting"},
            "audio_job_id": "offline",
        }

    if mode == "casual" and str(request.module_context or "").lower() not in {"style", "wardrobe"}:
        try:
            return {
                "success": True,
                "message": tone_engine.apply(
                    lightweight_chat(english_input),
                    user_profile=request.user_profile,
                    signals={"context_mode": "home", "user_message_style": user_message_style},
                    context={},
                ),
                "cards": [],
                "meta": {"mode": "casual"},
                "audio_job_id": "offline",
            }
        except Exception:
            pass

    # -------------------------
    # WEATHER
    # -------------------------
    weather_data = {}
    try:
        location = request.user_profile.get("location") or {}
        lat, lon = location.get("lat"), location.get("lon")

        if lat is not None and lon is not None:
            weather_data = _get_weather_cached(lat=lat, lon=lon)

    except Exception as e:
        logger.warning("weather lookup failed %s", e)

    # -------------------------
    # ORCHESTRATOR (TIMEOUT SAFE)
    # -------------------------
    history = _build_history(request.messages[:-1]) if len(request.messages) > 1 else []
    memory_history = request.current_memory.get("history", []) if isinstance(request.current_memory, dict) else []
    merged_history = _normalize_memory_history(memory_history) + history

    def run():
        return ahvi_orchestrator.run(
            text=english_input,
            user_id=user_id,
            context={
                "memory": request.current_memory,
                "user_profile": _ahvi_resolve_effective_user_profile(user_id, request.user_profile if isinstance(request.user_profile, dict) else {}),
                "module_context": request.module_context,
                "include_base64": bool(request.include_base64),
                "wardrobe": request.wardrobe,
                "history": merged_history[-20:],
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
                "signals": {"user_message_style": user_message_style},
            },
        )

    try:
        result = _ORCHESTRATOR_EXECUTOR.submit(run).result(timeout=_ORCH_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        style_payload = _demo_style_board_payload(user_id, english_input, request.wardrobe) if visual_context else {}
        fallback_message = style_payload.get("message") or (
            "AHVI is still warming the styling engine, but here is a safe look: keep one hero piece, pair it with a clean neutral base, and add one polished accessory."
            if visual_context
            else lightweight_chat(english_input)
        )
        return {
            "success": True,
            "message": tone_engine.apply(
                fallback_message,
                user_profile=request.user_profile,
                signals={"context_mode": request.module_context or "style", "user_message_style": user_message_style},
                context={"module_context": request.module_context},
            ),
            "type": style_payload.get("type") or "style_fallback",
            "cards": style_payload.get("cards") or [],
            "board_ids": style_payload.get("board_ids") or "",
            "data": style_payload.get("data") or {"outfits": [], "rendered_boards": []},
            "meta": {"mode": "timeout_fallback", "timeout_seconds": _ORCH_TIMEOUT_SECONDS, **(style_payload.get("meta") or {})},
            "audio_job_id": "offline",
        }
    except Exception as exc:
        style_payload = _demo_style_board_payload(user_id, english_input, request.wardrobe) if visual_context else {}
        fallback_message = style_payload.get("message") or (
            lightweight_chat(english_input)
            if not visual_context
            else "I will assume smart casual for now: choose one clean hero piece, pair it with a neutral base, and finish with footwear or an accessory that matches the occasion."
        )
        return {
            "success": True,
            "message": tone_engine.apply(
                fallback_message,
                user_profile=request.user_profile,
                signals={"context_mode": request.module_context or "style", "user_message_style": user_message_style},
                context={"module_context": request.module_context},
            ),
            "type": style_payload.get("type") or "style_fallback",
            "cards": style_payload.get("cards") or [],
            "board_ids": style_payload.get("board_ids") or "",
            "data": style_payload.get("data") or {"outfits": [], "rendered_boards": []},
            "meta": {"mode": "error_fallback", "error": str(exc)[:160], **(style_payload.get("meta") or {})},
            "audio_job_id": "offline",
        }

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
                message = GoogleTranslator(source="en", target=target_lang).translate(message)
    except Exception:
        pass

    # Final guardrail: every /api/text answer should leave through the tone layer,
    # even if a newer orchestrator branch forgot to apply it internally.
    try:
        message = tone_engine.apply(
            message,
            user_profile=request.user_profile,
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
            isinstance(cards_payload, list)
            and cards_payload
        ) or bool(
            isinstance(data_payload, dict)
            and (data_payload.get("rendered_boards") or data_payload.get("outfits"))
        )
        if visual_context and not has_visual_board:
            style_payload = _demo_style_board_payload(user_id, english_input, request.wardrobe)
            if style_payload.get("cards"):
                cards_payload = style_payload.get("cards") or []
                data_payload = style_payload.get("data") or {}
                result["type"] = style_payload.get("type") or result.get("type")
                result["board_ids"] = style_payload.get("board_ids") or result.get("board_ids") or ""
            result["meta"] = {**(result.get("meta") or {}), **(style_payload.get("meta") or {})}
        lower_message = (message or "").lower()
        if not message or "clarification" in lower_message or "balance isn't quite" in lower_message:
            replacement = style_payload.get("message") or "I will assume smart casual for today: start with a clean hero piece, add a neutral base, and finish with footwear or one accessory. Once your wardrobe has saved items, I will pick the exact pieces from it."
            try:
                message = tone_engine.apply(
                    replacement,
                    user_profile=request.user_profile,
                    signals={"context_mode": request.module_context or "style", "user_message_style": user_message_style},
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
                kwargs={"request_id": str(getattr(http_request.state, "request_id", "") or "")},
                kind="chat_audio",
                user_id=user_id,
                source="routers.chat.text",
                request_id=str(getattr(http_request.state, "request_id", "") or ""),
            )
            if run_heavy_audio_task else "offline"
        )
    except Exception:
        audio_job_id = "offline"

    # -------------------------
    # FINAL RESPONSE
    # -------------------------
    if not isinstance(cards_payload, list):
        cards_payload = []

    board_ids_text = str(result.get("board_ids") or "")

    if visual_context and isinstance(cards_payload, list) and cards_payload:
        cards_payload = _ahvi_final_response_style_guard(
            cards_payload,
            user_id,
            english_input,
            request.wardrobe,
            request.user_profile if isinstance(request.user_profile, dict) else {},
        )
        if isinstance(data_payload, dict):
            data_payload["outfits"] = cards_payload

    logger.info(
        "chat.text_response user_id=%s cards=%d",
        user_id,
        len(cards_payload),
    )

    response = {
        "success": True,
        "message": message,
        "board": result.get("board"),
        "type": result.get("type"),
        "cards": cards_payload,
        "board_ids": board_ids_text,
        "data": data_payload if isinstance(data_payload, dict) else {},
        "meta": {
            **(result.get("meta") or {}),
            "weather": weather_data,
            "history_used": len(merged_history[-20:])
        },
        "audio_job_id": audio_job_id,
    }

    # -------------------------
    # CACHE SAVE
    # -------------------------
    if not cache_visual_boards:
        _CHAT_CACHE.set(cache_key, response)

    return response
