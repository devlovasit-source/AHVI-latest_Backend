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

def _item_category_blob(item: Dict[str, Any]) -> str:
    parts = [
        item.get("name"),
        item.get("category"),
        item.get("sub_category"),
        item.get("label"),
        item.get("pattern"),
    ]
    return " ".join(str(p).lower() for p in parts if p)


def _fetch_wardrobe_for_style(user_id: str, request_wardrobe: Any) -> List[Dict[str, Any]]:
    if isinstance(request_wardrobe, list):
        return [dict(i) for i in request_wardrobe if isinstance(i, dict)]
    try:
        docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=24)
        if isinstance(docs, dict):
            rows = docs.get("documents") or docs.get("items") or []
        else:
            rows = docs or []
        return [dict(i) for i in rows if isinstance(i, dict)]
    except Exception as exc:
        logger.warning("style fallback wardrobe fetch failed user_id=%s error=%s", user_id, exc)
        return []


def _pick_style_items(items: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
    buckets = {
        "tops": [],
        "bottoms": [],
        "shoes": [],
        "dresses": [],
        "outerwear": [],
        "accessories": [],
    }

    for item in items or []:
        if not isinstance(item, dict):
            continue

        cat = _infer_chat_category(item)

        if cat == "Tops":
            buckets["tops"].append(item)
        elif cat == "Bottoms":
            buckets["bottoms"].append(item)
        elif cat == "Footwear":
            buckets["shoes"].append(item)
        elif cat == "Dresses":
            buckets["dresses"].append(item)
        elif cat == "Outerwear":
            buckets["outerwear"].append(item)
        else:
            buckets["accessories"].append(item)

    selected: List[Dict[str, Any]] = []

    if buckets["dresses"]:
        selected.append(buckets["dresses"][0])
    else:
        selected.extend(buckets["tops"][:1])
        selected.extend(buckets["bottoms"][:1])

    selected.extend(buckets["outerwear"][:1])
    selected.extend(buckets["shoes"][:1])
    selected.extend(buckets["accessories"][:1])

    unique: List[Dict[str, Any]] = []
    seen = set()

    for item in selected:
        item_id = str(item.get("id") or item.get("$id") or item.get("image_id") or item.get("name") or "")
        if item_id and item_id in seen:
            continue
        if item_id:
            seen.add(item_id)
        unique.append(item)

    return unique[:5]

def _demo_style_board_payload(user_id: str, query_text: str, request_wardrobe: Any) -> Dict[str, Any]:
    wardrobe = _fetch_wardrobe_for_style(user_id, request_wardrobe)
    selected = _pick_style_items(wardrobe, query_text)
    if not selected:
        return {
            "message": "I can style this better once you add a few wardrobe pieces. For now, choose one clean hero garment, a neutral base, and one polished accessory.",
            "type": "style_fallback",
            "cards": [],
            "board_ids": "",
            "data": {"outfits": [], "rendered_boards": []},
            "meta": {"wardrobe_count": 0, "mode": "deterministic_style_no_wardrobe"},
        }

    q = (query_text or "").lower()
    if any(k in q for k in ["date", "dinner", "night"]):
        occasion = "date night"
        title = "Date Night Edit"
        note = "soft polish, clean contrast, and one memorable detail"
    elif any(k in q for k in ["coffee", "casual", "outing", "weekend"]):
        occasion = "casual outing"
        title = "Casual Outing Board"
        note = "relaxed structure with a neat finish"
    else:
        occasion = "today"
        title = "AHVI Styled Look"
        note = "balanced, wearable, and intentional"

    def _fallback_image(item: Dict[str, Any]) -> str:
        return str(
            item.get("masked_url")
            or item.get("maskedUrl")
            or item.get("image_url")
            or item.get("imageUrl")
            or item.get("raw_url")
            or item.get("url")
            or item.get("image")
            or ""
        ).strip()

    def _fallback_tokens(item: Dict[str, Any]) -> set:
        blob = " ".join(str(item.get(k, "") or "") for k in (
            "slot", "type", "category", "cat", "category_group",
            "sub_category", "subcategory", "subCategory",
            "name", "label", "description"
        )).lower()
        return set(re.sub(r"[^a-z0-9]+", " ", blob).split())

    def _fallback_role(item: Dict[str, Any]) -> str:
        tokens = _fallback_tokens(item)

        if tokens.intersection({
            "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
            "heel", "heels", "sandal", "sandals", "loafer", "loafers",
            "footwear"
        }):
            return "footwear"

        # Accessories before clothing.
        if tokens.intersection({
            "watch", "watches", "belt", "belts", "cap", "caps", "hat", "hats",
            "sunglass", "sunglasses", "eyewear", "glasses", "bag", "bags",
            "purse", "handbag", "clutch", "tote", "jewelry", "jewellery",
            "ring", "rings", "necklace", "necklaces", "bracelet", "bracelets",
            "earring", "earrings", "scarf", "scarves"
        }):
            return "accessory"

        # Tops before bottoms so short-sleeve shirt never becomes shorts.
        if tokens.intersection({
            "top", "tops", "shirt", "shirts", "tee", "tshirt", "tshirts",
            "blouse", "jacket", "blazer", "sweater", "hoodie", "kurta",
            "kurti", "dress", "dresses", "saree", "sari", "tunic", "tunics"
        }):
            return "top"

        # shorts only; never short.
        if tokens.intersection({
            "bottom", "bottoms", "pant", "pants", "trouser", "trousers",
            "jean", "jeans", "shorts", "skirt", "skirts", "chino", "chinos"
        }):
            return "bottom"

        return "unknown"

    def _fallback_norm(item: Dict[str, Any]) -> Dict[str, Any]:
        image = _fallback_image(item)
        return {
            "id": str(item.get("$id") or item.get("id") or item.get("item_id") or item.get("name") or ""),
            "name": str(item.get("name") or item.get("label") or item.get("category") or "Wardrobe item"),
            "category": str(item.get("category") or item.get("cat") or item.get("sub_category") or "Item"),
            "sub_category": str(item.get("sub_category") or item.get("subcategory") or item.get("subCategory") or ""),
            "color": str(item.get("color_name") or item.get("color") or ""),
            "pattern": str(item.get("pattern") or ""),
            "image_url": image,
            "masked_url": item.get("masked_url") or item.get("maskedUrl") or image,
            "imageUrl": item.get("imageUrl") or image,
            "maskedUrl": item.get("maskedUrl") or item.get("masked_url") or image,
        }

    def _unique_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for item in items:
            key = str(item.get("$id") or item.get("id") or item.get("item_id") or item.get("name") or item.get("label") or id(item)).lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    buckets = {
        "top": [],
        "bottom": [],
        "footwear": [],
        "accessory": [],
    }

    for item in wardrobe or []:
        if not isinstance(item, dict):
            continue
        if not _fallback_image(item):
            continue
        role = _fallback_role(item)
        if role in buckets:
            buckets[role].append(item)

    for key in buckets:
        buckets[key] = _unique_items(buckets[key])

    # Last-resort supplement from selected items only if wardrobe buckets are incomplete.
    if not buckets["top"] or not buckets["bottom"] or not buckets["footwear"]:
        for item in selected:
            if not isinstance(item, dict) or not _fallback_image(item):
                continue
            role = _fallback_role(item)
            if role in buckets:
                buckets[role].append(item)

        for key in buckets:
            buckets[key] = _unique_items(buckets[key])

    cards: List[Dict[str, Any]] = []
    board_ids: List[str] = []
    can_build = bool(buckets["top"] and buckets["bottom"] and buckets["footwear"])

    # ================= AHVI STYLE BOARD VARIETY V9 BEGIN =================
    def _stable_int(value: str) -> int:
        try:
            return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)
        except Exception:
            return abs(hash(value))

    def _item_key(item: Dict[str, Any]) -> str:
        return str(
            item.get("$id")
            or item.get("id")
            or item.get("item_id")
            or item.get("image_id")
            or item.get("name")
            or item.get("label")
            or id(item)
        ).lower()

    def _item_text(item: Dict[str, Any]) -> str:
        return " ".join(
            str(item.get(k, "") or "")
            for k in (
                "name",
                "label",
                "category",
                "cat",
                "sub_category",
                "subcategory",
                "subCategory",
                "color",
                "color_name",
                "pattern",
                "description",
            )
        ).lower()

    def _accessory_type(item: Dict[str, Any]) -> str:
        text_blob = _item_text(item)
        if "watch" in text_blob:
            return "watch"
        if "belt" in text_blob:
            return "belt"
        if "cap" in text_blob or "hat" in text_blob:
            return "headwear"
        if "bag" in text_blob:
            return "bag"
        if any(k in text_blob for k in ["ring", "necklace", "bracelet", "earring", "jewelry", "jewellery"]):
            return "jewelry"
        return "accessory"

    def _combo_signature(top: Dict[str, Any], bottom: Dict[str, Any], shoe: Dict[str, Any]) -> str:
        return "|".join([_item_key(top), _item_key(bottom), _item_key(shoe)])

    def _rotated(items: List[Dict[str, Any]], salt: str) -> List[Dict[str, Any]]:
        if not items:
            return []
        seed = _stable_int(salt) % len(items)
        return items[seed:] + items[:seed]

    def _pick_accessories(idx: int, seed: int) -> List[Dict[str, Any]]:
        if not buckets["accessory"]:
            return []

        q_local = str(query_text or "").lower()
        headwear_allowed = any(
            k in q_local
            for k in ["casual", "street", "travel", "airport", "sport", "gym", "sun", "beach", "outdoor"]
        )

        rotated_accessories = _rotated(
            buckets["accessory"],
            f"{user_id}:{query_text}:accessory:{idx}:{seed}",
        )

        picked: List[Dict[str, Any]] = []
        seen_types = set()

        for accessory in rotated_accessories:
            typ = _accessory_type(accessory)
            if typ == "headwear" and not headwear_allowed:
                continue
            if typ in seen_types:
                continue

            picked.append(accessory)
            seen_types.add(typ)

            if len(picked) >= 2:
                break

        return picked

    def _occasion_score(
        top: Dict[str, Any],
        bottom: Dict[str, Any],
        shoe: Dict[str, Any],
        accs: List[Dict[str, Any]],
    ) -> int:
        q_local = str(query_text or "").lower()
        all_text = " ".join(
            [_item_text(top), _item_text(bottom), _item_text(shoe)]
            + [_item_text(a) for a in accs]
        )

        score = 0

        if any(k in q_local for k in ["date", "dinner", "night"]):
            if any(k in all_text for k in ["black", "dark", "navy", "green", "white", "brown"]):
                score += 8
            if any(k in all_text for k in ["shirt", "polo", "trouser", "jeans", "loafer", "boot", "sneaker"]):
                score += 8
            if any(k in all_text for k in ["cap", "shorts", "slipper"]):
                score -= 8

        elif any(k in q_local for k in ["office", "meeting", "work", "client"]):
            if any(k in all_text for k in ["shirt", "polo", "trouser", "chino", "loafer", "formal", "black", "white", "blue"]):
                score += 10
            if any(k in all_text for k in ["cap", "floral", "vacation", "shorts", "slipper", "slider"]):
                score -= 10

        elif any(k in q_local for k in ["party", "club", "night out"]):
            if any(k in all_text for k in ["black", "dark", "boot", "watch", "jacket"]):
                score += 8

        else:
            if any(k in all_text for k in ["shirt", "tee", "jeans", "sneaker", "trouser"]):
                score += 5

        return score

    if can_build:
        variety_window = max(1, int(os.getenv("AHVI_STYLE_VARIETY_WINDOW_SECONDS", "300")))
        time_bucket = int(time.time() // variety_window)
        seed = _stable_int(f"{user_id}:{query_text}:{time_bucket}:{len(wardrobe)}")

        tops = _rotated(buckets["top"], f"{seed}:tops")
        bottoms = _rotated(buckets["bottom"], f"{seed}:bottoms")
        shoes = _rotated(buckets["footwear"], f"{seed}:shoes")

        candidate_rows: List[Dict[str, Any]] = []

        for ti, top in enumerate(tops[:10]):
            for bi, bottom in enumerate(bottoms[:10]):
                for si, shoe in enumerate(shoes[:10]):
                    accessories = _pick_accessories(len(candidate_rows), seed + ti + bi + si)
                    candidate_rows.append({
                        "top": top,
                        "bottom": bottom,
                        "shoe": shoe,
                        "accessories": accessories,
                        "signature": _combo_signature(top, bottom, shoe),
                        "score": _occasion_score(top, bottom, shoe, accessories)
                            + ((ti + bi + si + seed) % 7),
                    })

        candidate_rows.sort(key=lambda row: row["score"], reverse=True)

        available_variety = (
            len(buckets["top"])
            + len(buckets["bottom"])
            + len(buckets["footwear"])
            + len(buckets["accessory"])
        )
        board_count = 3 if available_variety >= 6 else 2

        used_signatures = set()
        used_tops = set()
        used_bottoms = set()
        used_shoes = set()
        chosen: List[Dict[str, Any]] = []

        # Pass 1: maximize diversity across top/bottom/footwear.
        for row in candidate_rows:
            top_key = _item_key(row["top"])
            bottom_key = _item_key(row["bottom"])
            shoe_key = _item_key(row["shoe"])

            if row["signature"] in used_signatures:
                continue
            if top_key in used_tops and len(buckets["top"]) >= board_count:
                continue
            if bottom_key in used_bottoms and len(buckets["bottom"]) >= board_count:
                continue
            if shoe_key in used_shoes and len(buckets["footwear"]) >= board_count:
                continue

            chosen.append(row)
            used_signatures.add(row["signature"])
            used_tops.add(top_key)
            used_bottoms.add(bottom_key)
            used_shoes.add(shoe_key)

            if len(chosen) >= board_count:
                break

        # Pass 2: relax if wardrobe is small, but never repeat exact combo.
        if len(chosen) < board_count:
            for row in candidate_rows:
                if row["signature"] in used_signatures:
                    continue
                chosen.append(row)
                used_signatures.add(row["signature"])
                if len(chosen) >= board_count:
                    break

        is_date = any(k in str(query_text or "").lower() for k in ["date", "dinner", "night"])
        is_office = any(k in str(query_text or "").lower() for k in ["office", "meeting", "work", "client"])

        for idx, row in enumerate(chosen[:board_count]):
            top = row["top"]
            bottom = row["bottom"]
            shoe = row["shoe"]
            accessories = row["accessories"]

            board_items = [_fallback_norm(x) for x in [top, bottom, shoe] + accessories]
            board_id = f"demo_board_{int(time.time())}_{idx}"
            board_ids.append(board_id)

            if is_date:
                board_title = ["Evening Smart Casual", "Clean Relaxed Date Fit", "Soft Casual Evening"][idx % 3]
            elif is_office:
                board_title = ["Polished Work Fit", "Smart Office Casual", "Client-Ready Look"][idx % 3]
            elif occasion == "casual outing":
                board_title = ["Clean Casual Fit", "Relaxed Weekend Look", "Easy Day Outfit"][idx % 3]
            else:
                board_title = ["Clean Daily Look", "Balanced Smart Casual", "Easy Styled Fit"][idx % 3]

            top_name = str(board_items[0].get("name") or "top")
            bottom_name = str(board_items[1].get("name") or "bottom")
            shoe_name = str(board_items[2].get("name") or "footwear")

            if is_date:
                why = (
                    f"This works for date night because {top_name}, {bottom_name}, and {shoe_name} create a clean smart-casual balance. "
                    "The outfit feels intentional without looking overdone, and accessories are kept minimal."
                )
            elif is_office:
                why = (
                    f"This works for office because {top_name}, {bottom_name}, and {shoe_name} keep the look structured, neat, and wearable through the day."
                )
            else:
                why = (
                    f"This works because {top_name}, {bottom_name}, and {shoe_name} create a balanced outfit with a clear top-bottom-footwear structure."
                )

            cards.append({
                "id": board_id,
                "title": f"Look {idx + 1} · {board_title}",
                "name": f"Look {idx + 1} · {board_title}",
                "kind": "style_board",
                "score": max(82, 94 - idx * 3),
                "vibe": occasion,
                "aesthetic": note,
                "items": board_items,
                "accessories": [_fallback_norm(x) for x in accessories],
                "why_it_works": why,
                "explanation": why,
                "reason": why,
                "style_reason": why,
                "story": {
                    "title": f"Look {idx + 1} · {board_title}",
                    "subtitle": why,
                    "why_it_works": why,
                    "explanation": why,
                },
            })

        try:
            logger.info(
                "ahvi.style_variety_v9 user_id=%s occasion=%s top=%s bottom=%s footwear=%s accessories=%s cards=%s signatures=%s",
                user_id,
                occasion,
                len(buckets["top"]),
                len(buckets["bottom"]),
                len(buckets["footwear"]),
                len(buckets["accessory"]),
                len(cards),
                [c.get("id") for c in cards],
            )
        except Exception:
            pass
    # ================= AHVI STYLE BOARD VARIETY V9 END =================

    if not cards:
        normalized_items: List[Dict[str, Any]] = []
        for item in selected:
            if not isinstance(item, dict) or not _fallback_image(item):
                continue
            normalized_items.append(_fallback_norm(item))

        if normalized_items:
            card_id = f"demo_board_{int(time.time())}"
            board_ids.append(card_id)
            cards.append({
                "id": card_id,
                "title": title,
                "name": title,
                "kind": "style_board",
                "score": 84,
                "vibe": occasion,
                "aesthetic": note,
                "items": normalized_items[:6],
                "accessories": [],
            })

    if not cards:
        return {}

    first_names = ", ".join(i["name"] for i in cards[0]["items"][:6])
    message = (
        f"Here are {len(cards)} {occasion} boards from your wardrobe. "
        f"First look: {first_names}. I kept it {note}, with accessories completing the main look."
    )

    logger.info(
        "chat.rich_fallback user_id=%s wardrobe=%d cards=%d top=%d bottom=%d footwear=%d accessories=%d",
        user_id,
        len(wardrobe or []),
        len(cards),
        len(buckets["top"]),
        len(buckets["bottom"]),
        len(buckets["footwear"]),
        len(buckets["accessory"]),
    )

    return {
        "message": message,
        "type": "style_board",
        "cards": cards,
        "board_ids": ",".join(board_ids),
        "data": {"outfits": cards, "rendered_boards": []},
        "meta": {
            "wardrobe_count": len(wardrobe or []),
            "mode": "deterministic_style_board_v2",
            "fallback_cards": len(cards),
            "accessory_count": len(buckets["accessory"]),
        },
    }

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

# ================= AHVI STYLE CHAT PATCH V2 BEGIN =================

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
            "slot", "type", "category", "cat", "category_group",
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


def _pick_style_items(items, query_text, user_profile=None):
    user_profile = user_profile or {}
    allow_dresses = (
        _ahvi_profile_style_gender(user_profile) != "male"
        or _ahvi_query_allows_feminine_item(query_text)
    )

    buckets = {
        "tops": [],
        "bottoms": [],
        "footwear": [],
        "dresses": [],
        "outerwear": [],
        "accessories": [],
    }

    for item in items or []:
        if not isinstance(item, dict):
            continue

        if not _ahvi_item_allowed_for_user_profile(item, user_profile, query_text):
            continue

        cat = _infer_chat_category(item)

        if cat == "Tops":
            buckets["tops"].append(item)
        elif cat == "Bottoms":
            buckets["bottoms"].append(item)
        elif cat == "Footwear":
            buckets["footwear"].append(item)
        elif cat == "Dresses":
            if allow_dresses:
                buckets["dresses"].append(item)
        elif cat == "Outerwear":
            buckets["outerwear"].append(item)
        else:
            buckets["accessories"].append(item)

    selected = []

    if allow_dresses and buckets["dresses"]:
        selected.append(buckets["dresses"][0])
    else:
        selected.extend(buckets["tops"][:1])
        selected.extend(buckets["bottoms"][:1])

    selected.extend(buckets["outerwear"][:1])
    selected.extend(buckets["footwear"][:1])
    selected.extend(buckets["accessories"][:2])

    unique = []
    seen = set()

    for item in selected:
        item_id = str(
            item.get("id")
            or item.get("$id")
            or item.get("image_id")
            or item.get("name")
            or ""
        )
        if item_id and item_id in seen:
            continue
        if item_id:
            seen.add(item_id)
        unique.append(item)

    return unique[:6]


def _ahvi_fallback_image(item):
    return str(
        item.get("masked_url")
        or item.get("maskedUrl")
        or item.get("image_url")
        or item.get("imageUrl")
        or item.get("raw_url")
        or item.get("url")
        or item.get("image")
        or ""
    ).strip()


def _ahvi_fallback_tokens(item):
    blob = " ".join(
        str(item.get(k, "") or "")
        for k in (
            "slot", "type", "category", "cat", "category_group",
            "sub_category", "subcategory", "subCategory",
            "name", "label", "description", "gender",
            "style_gender", "target_gender", "audience",
            "department", "intended_for", "wearer",
        )
    )
    return set(re.sub(r"[^a-z0-9]+", " ", blob.lower()).split())


def _ahvi_fallback_role(item):
    tokens = _ahvi_fallback_tokens(item)

    if tokens.intersection({
        "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
        "heel", "heels", "sandal", "sandals", "loafer", "loafers",
        "slipper", "slippers", "footwear",
    }):
        return "footwear"

    if tokens.intersection({
        "watch", "watches", "belt", "belts", "cap", "caps", "hat", "hats",
        "sunglass", "sunglasses", "eyewear", "glasses", "bag", "bags",
        "purse", "handbag", "clutch", "tote", "jewelry", "jewellery",
        "ring", "rings", "necklace", "necklaces", "bracelet", "bracelets",
        "earring", "earrings", "scarf", "scarves",
    }):
        return "accessory"

    # Critical: one-piece garments are not tops.
    if tokens.intersection({
        "dress", "dresses", "saree", "sari", "lehenga",
        "gown", "jumpsuit", "sherwani",
    }):
        return "dress"

    if tokens.intersection({
        "top", "tops", "shirt", "shirts", "tee", "tshirt", "tshirts",
        "jacket", "blazer", "sweater", "hoodie", "kurta",
        "polo", "tunic", "tunics",
    }):
        return "top"

    if tokens.intersection({
        "bottom", "bottoms", "pant", "pants", "trouser", "trousers",
        "jean", "jeans", "shorts", "skirt", "skirts", "chino", "chinos",
    }):
        return "bottom"

    return "unknown"


def _ahvi_fallback_norm(item):
    image = _ahvi_fallback_image(item)
    return {
        "id": str(item.get("$id") or item.get("id") or item.get("item_id") or item.get("name") or ""),
        "name": str(item.get("name") or item.get("label") or item.get("category") or "Wardrobe item"),
        "category": str(item.get("category") or item.get("cat") or item.get("sub_category") or "Item"),
        "sub_category": str(item.get("sub_category") or item.get("subcategory") or item.get("subCategory") or ""),
        "color": str(item.get("color_name") or item.get("color") or ""),
        "pattern": str(item.get("pattern") or ""),
        "image_url": image,
        "masked_url": item.get("masked_url") or item.get("maskedUrl") or image,
        "imageUrl": item.get("imageUrl") or image,
        "maskedUrl": item.get("maskedUrl") or item.get("masked_url") or image,
    }


def _ahvi_unique_items(items):
    seen = set()
    out = []

    for item in items or []:
        key = str(
            item.get("$id")
            or item.get("id")
            or item.get("item_id")
            or item.get("name")
            or item.get("label")
            or id(item)
        ).lower()

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


def _demo_style_board_payload(user_id, query_text, request_wardrobe, user_profile=None):
    effective_profile = _ahvi_resolve_effective_user_profile(user_id, user_profile or {})

    wardrobe = _fetch_wardrobe_for_style(user_id, request_wardrobe)
    wardrobe = [
        item for item in wardrobe
        if _ahvi_item_allowed_for_user_profile(item, effective_profile, query_text)
    ]

    selected = _pick_style_items(wardrobe, query_text, effective_profile)

    if not selected:
        return {
            "message": (
                "I can style this better once your wardrobe has enough compatible pieces. "
                "I filtered out items that do not match the saved user style preference."
            ),
            "type": "style_fallback",
            "cards": [],
            "board_ids": "",
            "data": {"outfits": [], "rendered_boards": []},
            "meta": {
                "wardrobe_count": len(wardrobe),
                "mode": "deterministic_style_no_compatible_wardrobe",
                "style_gender": _ahvi_profile_style_gender(effective_profile),
            },
        }

    q = (query_text or "").lower()

    if any(k in q for k in ["date", "dinner", "night"]):
        occasion = "date night"
        title = "Date Night Edit"
        note = "soft polish, clean contrast, and one memorable detail"
    elif any(k in q for k in ["coffee", "casual", "outing", "weekend"]):
        occasion = "casual outing"
        title = "Casual Outing Board"
        note = "relaxed structure with a neat finish"
    else:
        occasion = "today"
        title = "AHVI Styled Look"
        note = "balanced, wearable, and intentional"

    allow_dresses = (
        _ahvi_profile_style_gender(effective_profile) != "male"
        or _ahvi_query_allows_feminine_item(query_text)
    )

    buckets = {
        "top": [],
        "bottom": [],
        "dress": [],
        "footwear": [],
        "accessory": [],
    }

    for item in wardrobe or []:
        if not isinstance(item, dict):
            continue

        if not _ahvi_fallback_image(item):
            continue

        role = _ahvi_fallback_role(item)

        if role == "dress" and not allow_dresses:
            continue

        if role in buckets:
            buckets[role].append(item)

    for key in buckets:
        buckets[key] = _ahvi_unique_items(buckets[key])

    if not buckets["top"] or not buckets["bottom"] or not buckets["footwear"]:
        for item in selected:
            if not isinstance(item, dict):
                continue

            if not _ahvi_fallback_image(item):
                continue

            role = _ahvi_fallback_role(item)

            if role == "dress" and not allow_dresses:
                continue

            if role in buckets:
                buckets[role].append(item)

    for key in buckets:
        buckets[key] = _ahvi_unique_items(buckets[key])

    cards = []
    board_ids = []

    seed = abs(hash(f"{user_id}:{query_text}:{int(time.time() // 60)}"))

    can_build_two_piece = bool(buckets["top"] and buckets["bottom"] and buckets["footwear"])
    can_build_one_piece = bool(allow_dresses and buckets["dress"] and buckets["footwear"])

    if can_build_two_piece or can_build_one_piece:
        variety = sum(len(v) for v in buckets.values())
        board_count = 3 if variety >= 6 else 2

        for idx in range(board_count):
            accessories = []

            if buckets["accessory"]:
                for offset in range(min(3, len(buckets["accessory"]))):
                    accessories.append(
                        buckets["accessory"][(seed + idx + offset) % len(buckets["accessory"])]
                    )

            if can_build_two_piece:
                top = buckets["top"][(seed + idx) % len(buckets["top"])]
                bottom = buckets["bottom"][(seed + idx) % len(buckets["bottom"])]
                footwear = buckets["footwear"][(seed + idx) % len(buckets["footwear"])]
                raw_items = [top, bottom, footwear] + accessories
            else:
                dress = buckets["dress"][(seed + idx) % len(buckets["dress"])]
                footwear = buckets["footwear"][(seed + idx) % len(buckets["footwear"])]
                raw_items = [dress, footwear] + accessories

            board_items = [_ahvi_fallback_norm(x) for x in raw_items]
            board_id = f"demo_board_{int(time.time())}_{idx}"
            board_ids.append(board_id)

            cards.append({
                "id": board_id,
                "title": title if idx == 0 else f"{title} {idx + 1}",
                "name": title if idx == 0 else f"{title} {idx + 1}",
                "kind": "style_board",
                "score": max(82, 91 - idx * 3),
                "vibe": occasion,
                "aesthetic": note,
                "items": board_items,
                "accessories": [_ahvi_fallback_norm(x) for x in accessories],
                "why_chosen": (
                    f"Chosen using the saved {_ahvi_profile_style_gender(effective_profile)} style preference, "
                    "wardrobe compatibility, outfit role balance, and occasion intent."
                ),
            })

    if not cards:
        normalized_items = []

        for item in selected:
            if isinstance(item, dict) and _ahvi_fallback_image(item):
                normalized_items.append(_ahvi_fallback_norm(item))

        if normalized_items:
            board_id = f"demo_board_{int(time.time())}"
            board_ids.append(board_id)
            cards.append({
                "id": board_id,
                "title": title,
                "name": title,
                "kind": "style_board",
                "score": 84,
                "vibe": occasion,
                "aesthetic": note,
                "items": normalized_items[:6],
                "accessories": [],
                "why_chosen": (
                    f"Chosen using the saved {_ahvi_profile_style_gender(effective_profile)} style preference "
                    "and available compatible wardrobe items."
                ),
            })

    if not cards:
        return {}

    first_names = ", ".join(i["name"] for i in cards[0]["items"][:6])

    message = (
        f"Here are {len(cards)} {occasion} boards from your wardrobe. "
        f"First look: {first_names}. "
        "I filtered the wardrobe using the saved user style preference before building these boards."
    )

    return {
        "success": True,
        "message": message,
        "board": "style",
        "type": "cards",
        "cards": cards,
        "board_ids": board_ids[0] if board_ids else "",
        "data": {
            "outfits": [],
            "rendered_boards": [],
            "board_item_ids": board_ids,
            "style_gender": _ahvi_profile_style_gender(effective_profile),
        },
        "meta": {
            "intent": "style_fallback",
            "domain": "style",
            "mode": "deterministic_style_board",
            "wardrobe_count": len(wardrobe),
            "style_gender": _ahvi_profile_style_gender(effective_profile),
        },
    }

# ================= AHVI STYLE CHAT PATCH V2 END =================

# ================= AHVI ACCESSORY POLICY PATCH V3 BEGIN =================

_AHVI_HEADWEAR_EXPLICIT_TOKENS = {
    "cap", "caps", "hat", "hats", "headwear", "beanie"
}

_AHVI_HEADWEAR_CASUAL_TOKENS = {
    "casual", "coffee", "outing", "weekend", "errand", "errands",
    "street", "streetwear", "sport", "sports", "sporty", "gym",
    "walk", "outdoor", "outdoors", "travel", "airport", "college",
    "beach", "summer", "daytime"
}

_AHVI_HEADWEAR_BLOCK_TOKENS = {
    "date", "dinner", "night", "formal", "office", "work", "meeting",
    "interview", "wedding", "party", "business", "professional"
}


def _ahvi_v3_tokens_for_item(item):
    blob = " ".join(
        str(item.get(k, "") or "")
        for k in (
            "role", "slot", "type", "category", "cat", "category_group",
            "sub_category", "subcategory", "subCategory",
            "name", "label", "description",
        )
    )
    return set(_chat_tokens(blob))


def _ahvi_v3_query_tokens(query_text):
    return set(_chat_tokens(query_text or ""))


def _ahvi_v3_is_casual_query(query_text):
    tokens = _ahvi_v3_query_tokens(query_text)
    return bool(tokens.intersection(_AHVI_HEADWEAR_CASUAL_TOKENS))


def _ahvi_v3_allows_headwear(query_text):
    tokens = _ahvi_v3_query_tokens(query_text)

    if tokens.intersection(_AHVI_HEADWEAR_EXPLICIT_TOKENS):
        return True

    if tokens.intersection(_AHVI_HEADWEAR_BLOCK_TOKENS):
        return False

    return bool(tokens.intersection(_AHVI_HEADWEAR_CASUAL_TOKENS))


def _ahvi_v3_accessory_subrole(item):
    tokens = _ahvi_v3_tokens_for_item(item)

    if tokens.intersection({"watch", "watches"}):
        return "watch"

    if tokens.intersection({"cap", "caps", "hat", "hats", "headwear", "beanie"}):
        return "headwear"

    if tokens.intersection({"belt", "belts"}):
        return "belt"

    if tokens.intersection({"sunglass", "sunglasses", "eyewear", "glasses", "shade", "shades"}):
        return "eyewear"

    if tokens.intersection({"bag", "bags", "purse", "clutch", "backpack", "tote", "handbag"}):
        return "bag"

    if tokens.intersection({
        "necklace", "earring", "earrings", "ring", "rings",
        "bracelet", "bracelets", "jewelry", "jewellery"
    }):
        return "jewelry"

    if tokens.intersection({"scarf", "scarves"}):
        return "scarf"

    return "accessory"


def _ahvi_v3_select_accessories(accessories, query_text, seed=0, idx=0):
    """Select clean accessories: one per subrole, cap only for casual intent."""
    if not accessories:
        return []

    allow_headwear = _ahvi_v3_allows_headwear(query_text)
    is_casual = _ahvi_v3_is_casual_query(query_text)

    if is_casual:
        priority = ["headwear", "watch", "eyewear", "bag", "belt", "jewelry", "scarf", "accessory"]
        max_count = 2
    else:
        priority = ["watch", "belt", "eyewear", "bag", "jewelry", "scarf", "accessory"]
        max_count = 2

    by_subrole = {}
    seen_ids = set()

    for offset in range(len(accessories)):
        item = accessories[(seed + idx + offset) % len(accessories)]

        if not isinstance(item, dict):
            continue

        item_id = str(
            item.get("$id")
            or item.get("id")
            or item.get("item_id")
            or item.get("name")
            or item.get("label")
            or ""
        ).lower()

        if item_id and item_id in seen_ids:
            continue

        subrole = _ahvi_v3_accessory_subrole(item)

        if subrole == "headwear" and not allow_headwear:
            continue

        if subrole in by_subrole:
            continue

        if item_id:
            seen_ids.add(item_id)

        by_subrole[subrole] = item

    selected = []

    for subrole in priority:
        if subrole in by_subrole:
            selected.append(by_subrole[subrole])
        if len(selected) >= max_count:
            break

    return selected


def _demo_style_board_payload(user_id, query_text, request_wardrobe, user_profile=None):
    effective_profile = _ahvi_resolve_effective_user_profile(user_id, user_profile or {})

    wardrobe = _fetch_wardrobe_for_style(user_id, request_wardrobe)
    wardrobe = [
        item for item in wardrobe
        if _ahvi_item_allowed_for_user_profile(item, effective_profile, query_text)
    ]

    selected = _pick_style_items(wardrobe, query_text, effective_profile)

    if not selected:
        return {
            "message": (
                "I can style this better once your wardrobe has enough compatible pieces. "
                "I filtered out items that do not match the saved user style preference."
            ),
            "type": "style_fallback",
            "cards": [],
            "board_ids": "",
            "data": {"outfits": [], "rendered_boards": []},
            "meta": {
                "wardrobe_count": len(wardrobe),
                "mode": "deterministic_style_no_compatible_wardrobe",
                "style_gender": _ahvi_profile_style_gender(effective_profile),
                "accessory_policy": "one_per_type_headwear_only_for_casual",
            },
        }

    q = (query_text or "").lower()

    if any(k in q for k in ["date", "dinner", "night"]):
        occasion = "date night"
        title = "Date Night Edit"
        note = "soft polish, clean contrast, and one memorable detail"
    elif any(k in q for k in ["coffee", "casual", "outing", "weekend", "street", "sport", "travel", "outdoor"]):
        occasion = "casual outing"
        title = "Casual Outing Board"
        note = "relaxed structure with a neat finish"
    else:
        occasion = "today"
        title = "AHVI Styled Look"
        note = "balanced, wearable, and intentional"

    allow_dresses = (
        _ahvi_profile_style_gender(effective_profile) != "male"
        or _ahvi_query_allows_feminine_item(query_text)
    )

    buckets = {
        "top": [],
        "bottom": [],
        "dress": [],
        "footwear": [],
        "accessory": [],
    }

    for item in wardrobe or []:
        if not isinstance(item, dict):
            continue

        if not _ahvi_fallback_image(item):
            continue

        role = _ahvi_fallback_role(item)

        if role == "dress" and not allow_dresses:
            continue

        if role in buckets:
            buckets[role].append(item)

    for key in buckets:
        buckets[key] = _ahvi_unique_items(buckets[key])

    if not buckets["top"] or not buckets["bottom"] or not buckets["footwear"]:
        for item in selected:
            if not isinstance(item, dict):
                continue

            if not _ahvi_fallback_image(item):
                continue

            role = _ahvi_fallback_role(item)

            if role == "dress" and not allow_dresses:
                continue

            if role in buckets:
                buckets[role].append(item)

    for key in buckets:
        buckets[key] = _ahvi_unique_items(buckets[key])

    cards = []
    board_ids = []

    seed = abs(hash(f"{user_id}:{query_text}:{int(time.time() // 60)}"))

    can_build_two_piece = bool(buckets["top"] and buckets["bottom"] and buckets["footwear"])
    can_build_one_piece = bool(allow_dresses and buckets["dress"] and buckets["footwear"])

    if can_build_two_piece or can_build_one_piece:
        variety = sum(len(v) for v in buckets.values())
        board_count = 3 if variety >= 6 else 2

        for idx in range(board_count):
            accessories = _ahvi_v3_select_accessories(
                buckets["accessory"],
                query_text,
                seed=seed,
                idx=idx,
            )

            if can_build_two_piece:
                top = buckets["top"][(seed + idx) % len(buckets["top"])]
                bottom = buckets["bottom"][(seed + idx) % len(buckets["bottom"])]
                footwear = buckets["footwear"][(seed + idx) % len(buckets["footwear"])]
                raw_items = [top, bottom, footwear] + accessories
            else:
                dress = buckets["dress"][(seed + idx) % len(buckets["dress"])]
                footwear = buckets["footwear"][(seed + idx) % len(buckets["footwear"])]
                raw_items = [dress, footwear] + accessories

            board_items = [_ahvi_fallback_norm(x) for x in raw_items]
            board_id = f"demo_board_{int(time.time())}_{idx}"
            board_ids.append(board_id)

            cards.append({
                "id": board_id,
                "title": title if idx == 0 else f"{title} {idx + 1}",
                "name": title if idx == 0 else f"{title} {idx + 1}",
                "kind": "style_board",
                "score": max(82, 91 - idx * 3),
                "vibe": occasion,
                "aesthetic": note,
                "items": board_items,

                # Important:
                # Accessories are already included in items.
                # Keep this empty to avoid frontend double-rendering them.
                "accessories": [],

                "why_chosen": (
                    f"Chosen using the saved {_ahvi_profile_style_gender(effective_profile)} style preference, "
                    "wardrobe compatibility, occasion intent, and accessory policy: one item per accessory type."
                ),
            })

    if not cards:
        normalized_items = []

        for item in selected:
            if isinstance(item, dict) and _ahvi_fallback_image(item):
                normalized_items.append(_ahvi_fallback_norm(item))

        if normalized_items:
            board_id = f"demo_board_{int(time.time())}"
            board_ids.append(board_id)
            cards.append({
                "id": board_id,
                "title": title,
                "name": title,
                "kind": "style_board",
                "score": 84,
                "vibe": occasion,
                "aesthetic": note,
                "items": normalized_items[:6],
                "accessories": [],
                "why_chosen": (
                    f"Chosen using the saved {_ahvi_profile_style_gender(effective_profile)} style preference "
                    "and available compatible wardrobe items."
                ),
            })

    if not cards:
        return {}

    first_names = ", ".join(i["name"] for i in cards[0]["items"][:6])

    message = (
        f"Here are {len(cards)} {occasion} boards from your wardrobe. "
        f"First look: {first_names}. "
        "I filtered the wardrobe using the saved user style preference and limited accessories to one per type."
    )

    try:
        logger.info(
            "ahvi.accessory_policy_v3 user_id=%s style_gender=%s cards=%s headwear_allowed=%s accessory_counts=%s",
            user_id,
            _ahvi_profile_style_gender(effective_profile),
            len(cards),
            _ahvi_v3_allows_headwear(query_text),
            [len(card.get("items") or []) - 3 for card in cards],
        )
    except Exception:
        pass

    return {
        "success": True,
        "message": message,
        "board": "style",
        "type": "cards",
        "cards": cards,
        "board_ids": board_ids[0] if board_ids else "",
        "data": {
            "outfits": [],
            "rendered_boards": [],
            "board_item_ids": board_ids,
            "style_gender": _ahvi_profile_style_gender(effective_profile),
            "accessory_policy": "one_per_type_headwear_only_for_casual",
        },
        "meta": {
            "intent": "style_fallback",
            "domain": "style",
            "mode": "deterministic_style_board",
            "wardrobe_count": len(wardrobe),
            "style_gender": _ahvi_profile_style_gender(effective_profile),
            "accessory_policy": "one_per_type_headwear_only_for_casual",
        },
    }

# ================= AHVI ACCESSORY POLICY PATCH V3 END =================


# ================= AHVI CHAT PIPELINE ADAPTER V1 BEGIN =================
# Final definition wins over duplicate _demo_style_board_payload functions above.
# This routes fallback style boards through the existing canonical outfit_pipeline.

try:
    _ahvi_legacy_demo_style_board_payload = _demo_style_board_payload
except Exception:
    _ahvi_legacy_demo_style_board_payload = None


def _ahvi_chat_adapter_occasion(query_text):
    q = str(query_text or "").lower()
    if any(k in q for k in ["date", "dinner", "night"]):
        return "date night"
    if any(k in q for k in ["office", "meeting", "work", "client"]):
        return "office"
    if any(k in q for k in ["party", "club", "night out"]):
        return "party"
    if any(k in q for k in ["travel", "airport", "trip"]):
        return "travel"
    if any(k in q for k in ["coffee", "casual", "outing", "weekend", "street", "sport", "travel", "outdoor"]):
        return "casual outing"
    return "today"


def _demo_style_board_payload(user_id, query_text, request_wardrobe, user_profile=None):
    try:
        from brain.outfit_pipeline import get_daily_outfits as _ahvi_get_daily_outfits

        effective_profile = _ahvi_resolve_effective_user_profile(user_id, user_profile or {})
        wardrobe = _fetch_wardrobe_for_style(user_id, request_wardrobe)
        wardrobe = [
            item for item in wardrobe
            if _ahvi_item_allowed_for_user_profile(item, effective_profile, query_text)
        ]

        occasion = _ahvi_chat_adapter_occasion(query_text)

        result = _ahvi_get_daily_outfits({
            "user_id": user_id,
            "wardrobe": wardrobe,
            "context": {
                "occasion": occasion,
                "query": query_text,
                "user_profile": effective_profile,
                "style_gender": _ahvi_profile_style_gender(effective_profile),
                "signals": {
                    "source": "routers.chat.pipeline_adapter",
                    "style_gender": _ahvi_profile_style_gender(effective_profile),
                },
            },
        })

        if not isinstance(result, dict):
            raise RuntimeError("outfit_pipeline returned non-dict result")

        cards = result.get("cards") if isinstance(result.get("cards"), list) else []
        cards = _ahvi_orchestrator_merge_card_accessories(cards) if "_ahvi_orchestrator_merge_card_accessories" in globals() else cards

        if cards:
            board_item_ids = result.get("board_item_ids") if isinstance(result.get("board_item_ids"), list) else []
            board_ids = ",".join([str(x) for x in board_item_ids if str(x).strip()])

            message = (
                result.get("message")
                or result.get("context")
                or f"Here are {len(cards)} {occasion} boards from your wardrobe."
            )

            try:
                logger.info(
                    "ahvi.chat_pipeline_adapter user_id=%s occasion=%s wardrobe=%s cards=%s first_card_items=%s",
                    user_id,
                    occasion,
                    len(wardrobe),
                    len(cards),
                    [
                        str((i or {}).get("name") or (i or {}).get("label") or "")
                        for i in ((cards[0].get("items") if cards and isinstance(cards[0], dict) else []) or [])
                        if isinstance(i, dict)
                    ][:8],
                )
            except Exception:
                pass

            return {
                "success": True,
                "message": str(message),
                "board": "style",
                "type": "cards",
                "cards": cards,
                "board_ids": board_ids or "",
                "data": {
                    "outfits": result.get("outfits") or [],
                    "rendered_boards": [],
                    "board_item_ids": board_item_ids,
                    "pipeline": result.get("pipeline") or {},
                    "style_gender": _ahvi_profile_style_gender(effective_profile),
                },
                "meta": {
                    "intent": "style_pipeline_adapter",
                    "domain": "style",
                    "mode": "outfit_pipeline_adapter",
                    "wardrobe_count": len(wardrobe),
                    "style_gender": _ahvi_profile_style_gender(effective_profile),
                    "occasion": occasion,
                },
            }

        try:
            logger.warning(
                "ahvi.chat_pipeline_adapter_empty user_id=%s occasion=%s wardrobe=%s result_context=%s",
                user_id,
                occasion,
                len(wardrobe),
                str(result.get("context") or "")[:180],
            )
        except Exception:
            pass

    except Exception as exc:
        try:
            logger.warning(
                "ahvi.chat_pipeline_adapter_failed user_id=%s error=%s",
                user_id,
                str(exc)[:180],
            )
        except Exception:
            pass

    # Legacy fallback remains only as last safety net.
    if callable(_ahvi_legacy_demo_style_board_payload):
        try:
            return _ahvi_legacy_demo_style_board_payload(user_id, query_text, request_wardrobe, user_profile)
        except TypeError:
            return _ahvi_legacy_demo_style_board_payload(user_id, query_text, request_wardrobe)
        except Exception:
            pass

    return {}

# ================= AHVI CHAT PIPELINE ADAPTER V1 END =================

