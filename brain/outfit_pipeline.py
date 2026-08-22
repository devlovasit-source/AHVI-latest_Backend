# ================= AHVI CLEAN STYLE FLOW FIX V1 APPLIED =================
import asyncio
import hashlib
import json
import logging
import math
import os
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from threading import Lock
from typing import Any, Dict, List, Tuple

from brain.engines.styling.style_builder import style_engine
from brain.ml.outfit_ranker import outfit_ranker
from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_scorer import (
    normalize_occasion,
    score_occasion_compatibility,
    score_weather_compatibility,
    style_scorer,
)
from brain.engines.refinement_engine import refinement_engine
from brain.engines.wardrobe_selector import wardrobe_selector
from brain.engines.styling.palette_engine import palette_engine
from services import ai_gateway
from services.appwrite_proxy import AppwriteProxy
from services.embedding_service import embedding_service, get_model
from services.wardrobe_suitability import is_style_eligible

logger = logging.getLogger("ahvi.outfit_pipeline")
from services.qdrant_service import qdrant_service
from brain.engines.outfit_quality_guard import filter_and_guard_outfits
import re


# ---- AHVI Style Orchestrator agent hooks (avoid_items / required_slots) ----
def _ahvi_agent_payload(context: Any) -> Dict[str, Any]:
    if isinstance(context, dict):
        payload = context.get("agent_orchestration")
        if isinstance(payload, dict):
            return payload
    return {}


def _ahvi_item_matches_avoid(item: Dict[str, Any], avoid_terms: List[str]) -> bool:
    if not avoid_terms or not isinstance(item, dict):
        return False
    blob = " ".join(
        str(item.get(k) or "").lower()
        for k in (
            "name",
            "title",
            "label",
            "category",
            "sub_category",
            "subcategory",
            "type",
            "slot",
            "role",
            "garment_type",
            "material",
        )
    )
    for term in avoid_terms:
        token = str(term or "").strip().lower()
        if token and token in blob:
            return True
    return False


def _ahvi_apply_agent_avoid_filter(wardrobe: Any, context: Any) -> Any:
    payload = _ahvi_agent_payload(context)
    avoid = payload.get("avoid_items") if isinstance(payload, dict) else None
    if not avoid:
        return wardrobe
    avoid_terms = [str(t).strip().lower() for t in avoid if str(t).strip()]
    if not avoid_terms:
        return wardrobe
    try:
        if isinstance(wardrobe, dict):
            return {
                slot: [
                    item
                    for item in items
                    if isinstance(item, dict)
                    and not _ahvi_item_matches_avoid(item, avoid_terms)
                ]
                for slot, items in wardrobe.items()
                if isinstance(items, list)
            }
        if isinstance(wardrobe, list):
            return [
                item
                for item in wardrobe
                if isinstance(item, dict)
                and not _ahvi_item_matches_avoid(item, avoid_terms)
            ]
    except Exception:
        logger.warning("ahvi.agent.avoid_filter_failed", exc_info=True)
    return wardrobe


# ---- AHVI demo fix: normalize Appwrite wardrobe records into outfit slots ----
def _ahvi_tokens(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip().split()


def _ahvi_slot_for_item(item):
    """Return canonical role for an Appwrite wardrobe item.

    Important safety rules:
    - Unknown/non-dict items must stay unknown, never accessory.
    - Shirt/top words win before bottom/dress words.
    - "short" is not a bottom; only "shorts" is.
    """
    if not isinstance(item, dict):
        return "unknown"

    raw = " ".join(
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
            "garment_type",
            "name",
            "label",
            "title",
            "description",
        )
    )
    tokens = set(_ahvi_tokens(raw))

    if tokens.intersection(
        {
            "top",
            "tops",
            "shirt",
            "shirts",
            "tee",
            "tshirt",
            "tshirts",
            "blouse",
            "blouses",
            "hoodie",
            "hoodies",
            "sweater",
            "sweaters",
            "kurta",
            "kurtas",
            "polo",
            "polos",
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
            "legging",
            "leggings",
        }
    ):
        return "bottom"

    if tokens.intersection(
        {
            "footwear",
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
        }
    ):
        return "footwear"

    if tokens.intersection(
        {
            "accessory",
            "accessories",
            "watch",
            "watches",
            "bag",
            "bags",
            "belt",
            "belts",
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
            "hat",
            "hats",
            "cap",
            "caps",
            "sunglass",
            "sunglasses",
            "eyewear",
            "glasses",
        }
    ):
        return "accessory"

    if tokens.intersection(
        {"jacket", "coat", "blazer", "outerwear", "cardigan", "overshirt"}
    ):
        return "outerwear"

    if tokens.intersection(
        {"dress", "dresses", "gown", "jumpsuit", "saree", "sari", "lehenga", "sherwani"}
    ):
        return "dress"

    return "unknown"


def _ahvi_image_for_item(item):
    if not isinstance(item, dict):
        return ""
    return (
        item.get("normalized_url")
        or item.get("normalizedUrl")
        or item.get("masked_url")
        or item.get("maskedUrl")
        or item.get("image_url")
        or item.get("imageUrl")
        or item.get("url")
        or item.get("image")
        or ""
    )


def _ahvi_normalize_wardrobe_items(items):
    normalized = []
    for item in items or []:
        if not isinstance(item, dict):
            continue

        slot = _ahvi_slot_for_item(item)
        image_url = _ahvi_image_for_item(item)

        patched = dict(item)
        patched.setdefault("slot", slot)
        patched.setdefault("type", slot)
        patched.setdefault("category_group", slot)
        patched.setdefault("imageUrl", image_url)
        patched.setdefault("image_url", image_url)
        patched.setdefault("maskedUrl", item.get("masked_url") or image_url)
        patched.setdefault("masked_url", item.get("maskedUrl") or image_url)
        patched.setdefault(
            "label",
            item.get("name") or item.get("label") or item.get("category") or slot,
        )

        normalized.append(patched)

    return normalized


# ---- end AHVI demo fix ----


_MEMORY_LOCK = Lock()
_MEMORY_FILE = os.path.join(os.path.dirname(__file__), "data", "outfit_memory.json")


def _tokens(value: Any) -> List[str]:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip().split()


def _contains_word(text: str, words: List[str]) -> bool:
    """
    Token-aware helper. This avoids substring mistakes like:
    "short" inside "short-sleeved".
    """
    parts = set(_tokens(text))
    return any(str(w or "").lower() in parts for w in words)


def _normalize_pipeline_occasion(value: Any, context: Dict[str, Any] | None = None) -> str:
    context = context or {}
    interpretation = context.get("occasion_interpretation") if isinstance(context.get("occasion_interpretation"), dict) else {}
    notes = interpretation.get("board_generation_notes") if isinstance(interpretation.get("board_generation_notes"), dict) else {}
    source = " ".join(
        str(v or "")
        for v in [
            value,
            context.get("occasion"),
            interpretation.get("occasion"),
            notes.get("occasion_kind"),
            interpretation.get("resolved_brief"),
            context.get("query"),
            context.get("user_query"),
            context.get("prompt"),
        ]
    )
    normalized = normalize_occasion(source)
    return normalized if normalized in {
        "client_dinner",
        "date_night",
        "beach",
        "office",
        "brunch",
        "party",
        "house_party",
        "rave",
        "cocktail",
        "travel",
        "workout",
        "wedding",
        "casual",
    } else ""


def _has_any(parts: List[str], words: List[str]) -> bool:
    return any(word in parts for word in words)


def _infer_category(item: Dict[str, Any]) -> str:
    """
    Canonical category inference.

    Critical cases:
    - Formal Dress Shirt -> Tops, not Dresses.
    - White Short-Sleeved Shirt -> Tops.
    - Khaki Shorts -> Bottoms.
    - Brown Boots -> Footwear.
    - Watch -> Accessories.
    - Unknown/non-dict -> Unknown, never Accessories.
    """
    if not isinstance(item, dict):
        return "Unknown"

    explicit = (
        str(
            item.get("role")
            or item.get("slot")
            or item.get("category")
            or item.get("cat")
            or item.get("type")
            or item.get("category_group")
            or item.get("sub_category")
            or item.get("subcategory")
            or item.get("subCategory")
            or ""
        )
        .strip()
        .lower()
    )

    if not explicit:
        explicit = _ahvi_slot_for_item(item)

    explicit_map = {
        "top": "Tops",
        "tops": "Tops",
        "shirt": "Tops",
        "shirts": "Tops",
        "tshirt": "Tops",
        "tshirts": "Tops",
        "t-shirt": "Tops",
        "tee": "Tops",
        "polo": "Tops",
        "kurta": "Tops",
        "bottom": "Bottoms",
        "bottoms": "Bottoms",
        "pants": "Bottoms",
        "pant": "Bottoms",
        "trousers": "Bottoms",
        "trouser": "Bottoms",
        "jeans": "Bottoms",
        "jean": "Bottoms",
        "shorts": "Bottoms",
        "chinos": "Bottoms",
        "chino": "Bottoms",
        "footwear": "Footwear",
        "shoe": "Footwear",
        "shoes": "Footwear",
        "sneaker": "Footwear",
        "sneakers": "Footwear",
        "boot": "Footwear",
        "boots": "Footwear",
        "sandal": "Footwear",
        "sandals": "Footwear",
        "accessory": "Accessories",
        "accessories": "Accessories",
        "bag": "Accessories",
        "bags": "Accessories",
        "watch": "Accessories",
        "watches": "Accessories",
        "belt": "Accessories",
        "belts": "Accessories",
        "jewelry": "Accessories",
        "jewellery": "Accessories",
        "outerwear": "Outerwear",
        "outer": "Outerwear",
        "jacket": "Outerwear",
        "coat": "Outerwear",
        "blazer": "Outerwear",
        "dress": "Dresses",
        "dresses": "Dresses",
        "indian wear": "Dresses",
        "saree": "Dresses",
        "sari": "Dresses",
        "lehenga": "Dresses",
        "unknown": "Unknown",
    }

    if explicit in explicit_map:
        return explicit_map[explicit]

    joined = " ".join(
        str(item.get(k, "") or "")
        for k in (
            "role",
            "slot",
            "category",
            "cat",
            "type",
            "name",
            "label",
            "title",
            "sub_category",
            "subcategory",
            "subCategory",
            "description",
        )
    )
    parts = _tokens(joined)

    if _has_any(
        parts,
        [
            "shirt",
            "shirts",
            "tee",
            "tshirt",
            "tshirts",
            "top",
            "tops",
            "blouse",
            "blouses",
            "hoodie",
            "hoodies",
            "sweater",
            "sweaters",
            "kurta",
            "kurtas",
            "polo",
            "polos",
        ],
    ):
        return "Tops"

    if _has_any(
        parts,
        [
            "pants",
            "pant",
            "trousers",
            "trouser",
            "jeans",
            "jean",
            "shorts",
            "skirt",
            "skirts",
            "legging",
            "leggings",
            "chino",
            "chinos",
            "bottom",
            "bottoms",
        ],
    ):
        return "Bottoms"

    if _has_any(
        parts,
        [
            "shoe",
            "shoes",
            "boot",
            "boots",
            "sneaker",
            "sneakers",
            "heel",
            "heels",
            "sandal",
            "sandals",
            "loafer",
            "loafers",
            "slipper",
            "slippers",
            "footwear",
        ],
    ):
        return "Footwear"

    if _has_any(
        parts,
        [
            "watch",
            "watches",
            "bag",
            "bags",
            "belt",
            "belts",
            "scarf",
            "scarves",
            "jewelry",
            "jewellery",
            "ring",
            "rings",
            "necklace",
            "bracelet",
            "earring",
            "earrings",
            "accessory",
            "accessories",
            "hat",
            "hats",
            "cap",
            "caps",
            "sunglass",
            "sunglasses",
        ],
    ):
        return "Accessories"

    if _has_any(
        parts, ["jacket", "coat", "blazer", "outerwear", "cardigan", "overshirt"]
    ):
        return "Outerwear"

    if _has_any(
        parts,
        [
            "dress",
            "dresses",
            "gown",
            "jumpsuit",
            "saree",
            "sari",
            "lehenga",
            "sherwani",
        ],
    ):
        return "Dresses"

    return "Unknown"


def _bucket_for_category(category: str) -> str:
    return {
        "Tops": "tops",
        "Bottoms": "bottoms",
        "Footwear": "shoes",
        "Accessories": "accessories",
        "Outerwear": "outerwear",
        "Dresses": "dresses",
        "Unknown": "unknown",
    }.get(str(category or ""), "unknown")


def _type_for_category(category: str) -> str:
    return {
        "Tops": "top",
        "Bottoms": "bottom",
        "Footwear": "footwear",
        "Accessories": "accessory",
        "Outerwear": "outerwear",
        "Dresses": "dress",
        "Unknown": "unknown",
    }.get(str(category or ""), "unknown")


def _stable_offset(seed: str, size: int) -> int:
    if size <= 0:
        return 0
    digest = hashlib.sha256(str(seed or "").encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:4], "big") % size


def _rotate(values: List[Dict[str, Any]], offset: int) -> List[Dict[str, Any]]:
    if not values:
        return values
    offset = offset % len(values)
    return values[offset:] + values[:offset]


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_memory() -> Dict[str, Any]:
    if not os.path.exists(_MEMORY_FILE):
        return {"users": {}}
    try:
        with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("users", {})
                return data
    except Exception:
        pass
    return {"users": {}}


def _save_memory(memory: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(_MEMORY_FILE), exist_ok=True)
    with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=True, indent=2)


def _memory_doc_id(user_id: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(user_id or "anonymous")
    )
    return f"outfit_memory_{safe}"[:64]


def _default_user_memory() -> Dict[str, Any]:
    return {
        "recent_outfits": [],
        "liked_outfits": [],
        "disliked_outfits": [],
    }


def _load_user_memory(user_id: str) -> Dict[str, Any]:
    proxy = AppwriteProxy()
    try:
        doc = proxy.get_document("memories", _memory_doc_id(user_id))
        payload = doc.get("payload") if isinstance(doc, dict) else None
        if isinstance(payload, str) and payload.strip():
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                user_memory = _default_user_memory()
                user_memory.update(parsed)
                return user_memory
    except Exception:
        pass

    memory = _load_memory()
    user = _ensure_user_memory(memory, user_id)
    return dict(user)


def _save_user_memory(user_id: str, user_memory: Dict[str, Any]) -> None:
    proxy = AppwriteProxy()
    try:
        payload = {
            "userId": str(user_id),
            "name": "outfit_memory",
            "payload": json.dumps(user_memory, ensure_ascii=True),
        }
        doc_id = _memory_doc_id(user_id)
        try:
            proxy.update_document("memories", doc_id, payload)
        except Exception:
            proxy.create_document("memories", payload, document_id=doc_id)
        return
    except Exception:
        pass

    memory = _load_memory()
    user = _ensure_user_memory(memory, user_id)
    user.clear()
    user.update(user_memory or {})
    _save_memory(memory)


def _ensure_user_memory(memory: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    users = memory.setdefault("users", {})
    user = users.setdefault(
        user_id,
        {
            "recent_outfits": [],
            "liked_outfits": [],
            "disliked_outfits": [],
        },
    )
    user.setdefault("recent_outfits", [])
    user.setdefault("liked_outfits", [])
    user.setdefault("disliked_outfits", [])
    return user


def _normalize_item(item: Dict[str, Any], fallback_type: str) -> Dict[str, Any]:
    item = item or {}
    enriched = dict(item)

    if fallback_type and not enriched.get("category") and not enriched.get("type"):
        enriched["category"] = fallback_type

    category_name = _infer_category(enriched)
    item_type = _type_for_category(category_name)

    raw_tags = item.get("occasion_tags", item.get("occasions", []))
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]

    raw_weather = item.get("weather_tags", item.get("weather", []))
    if isinstance(raw_weather, str):
        raw_weather = [raw_weather]

    return {
        "id": str(
            item.get("id")
            or item.get("$id")
            or item.get("item_id")
            or item.get("image_id")
            or item.get("name")
            or ""
        ),
        "name": str(item.get("name") or item.get("label") or item_type.title()),
        "type": item_type,
        "category": category_name,
        "sub_category": str(
            item.get("sub_category")
            or item.get("subcategory")
            or item.get("label")
            or item_type
        ).strip(),
        "color": str(
            item.get("color") or item.get("color_name") or item.get("color_code") or ""
        ).lower(),
        "image_url": str(
            item.get("image_url")
            or item.get("raw_image_url")
            or item.get("raw_url")
            or item.get("imageUrl")
            or ""
        ).strip(),
        "masked_url": str(
            item.get("masked_url")
            or item.get("masked_image_url")
            or item.get("sticker_url")
            or item.get("maskedUrl")
            or ""
        ).strip(),
        "fabric": str(item.get("fabric") or item.get("pattern") or "").lower(),
        "style": str(item.get("style") or item.get("vibe") or "").lower(),
        "occasion_tags": [
            str(v).strip().lower() for v in raw_tags if str(v or "").strip()
        ],
        "weather_tags": [
            str(v).strip().lower() for v in raw_weather if str(v or "").strip()
        ],
        "layerable": bool(item.get("layerable", False)),
    }


def _normalize_wardrobe(raw_wardrobe: Any) -> Dict[str, List[Dict[str, Any]]]:
    parts = {
        "tops": [],
        "bottoms": [],
        "shoes": [],
        "outerwear": [],
        "dresses": [],
        "accessories": [],
    }

    def _add(raw: Dict[str, Any], forced_category: str = "") -> None:
        if not isinstance(raw, dict):
            return

        enriched = dict(raw)
        if (
            forced_category
            and not enriched.get("category")
            and not enriched.get("type")
        ):
            enriched["category"] = forced_category

        category_name = _infer_category(enriched)
        bucket = _bucket_for_category(category_name)
        if bucket not in parts:
            return
        parts[bucket].append(
            _normalize_item(enriched, _type_for_category(category_name))
        )

    if isinstance(raw_wardrobe, dict):
        for item in raw_wardrobe.get("tops", []) or []:
            _add(item, "Tops")
        for item in raw_wardrobe.get("bottoms", []) or []:
            _add(item, "Bottoms")
        for item in raw_wardrobe.get("shoes", raw_wardrobe.get("footwear", [])) or []:
            _add(item, "Footwear")
        for item in raw_wardrobe.get("outerwear", []) or []:
            _add(item, "Outerwear")
        for item in raw_wardrobe.get("dresses", []) or []:
            _add(item, "Dresses")
        for item in (
            raw_wardrobe.get("accessories", raw_wardrobe.get("jewelry", [])) or []
        ):
            _add(item, "Accessories")
    elif isinstance(raw_wardrobe, list):
        for item in raw_wardrobe:
            _add(item)

    return parts


def _outfit_vector(outfit: Dict[str, Any]) -> List[float]:
    def _hash_fraction(value: str) -> float:
        raw = str(value or "")
        return (sum(ord(ch) for ch in raw) % 100) / 100.0

    top = outfit.get("top", {}) or {}
    bottom = outfit.get("bottom", {}) or {}
    shoes = outfit.get("shoes", {}) or {}
    score = float(outfit.get("score", 0.0))
    return [
        _hash_fraction(top.get("type")),
        _hash_fraction(top.get("color")),
        _hash_fraction(bottom.get("type")),
        _hash_fraction(bottom.get("color")),
        _hash_fraction(shoes.get("type")),
        _hash_fraction(shoes.get("color")),
        max(-1.0, min(1.0, score / 10.0)),
        1.0,
    ]


def _index_outfit_vector(user_id: str, outfit: Dict[str, Any], label: str) -> None:
    if not qdrant_service.enabled():
        return
    try:
        vector = _outfit_vector(outfit)
        combo_id = str(outfit.get("combo_id", ""))
        seed = f"{user_id}:{label}:{combo_id or _utcnow_iso()}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))
        payload = {"user_id": user_id, "label": label, "combo_id": combo_id}
        qdrant_service.upsert_memory_vector(
            point_id=point_id, vector=vector, payload=payload
        )
    except Exception:
        return


def _build_semantic_query_text(context: Dict[str, Any]) -> str:
    query_text = " ".join(
        [
            str(context.get("query", "")),
            str(context.get("occasion", "")),
            str(context.get("weather", "")),
            str(context.get("time_of_day", "")),
            str((context.get("style_dna", {}) or {}).get("style", "")),
            " ".join(
                (context.get("style_dna", {}) or {}).get("preferred_colors", [])
            ),
        ]
    ).strip()
    return query_text or "daily outfit"


def _semantic_results_to_wardrobe(
    results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    wardrobe_items: List[Dict[str, Any]] = []
    semantic_map: Dict[str, float] = {}
    for row in results:
        payload = row.get("payload", {}) if isinstance(row, dict) else {}
        normalized = _normalize_item(payload, str(payload.get("category", "item")))
        item_id = normalized.get("id")
        if not item_id:
            continue
        semantic_map[item_id] = max(
            float(semantic_map.get(item_id, 0.0)), float(row.get("score", 0.0))
        )
        wardrobe_items.append(normalized)
    return wardrobe_items, semantic_map


_QUERY_VECTOR_CACHE: Dict[str, List[float]] = {}
_QUERY_VECTOR_CACHE_LIMIT = 256
_SEMANTIC_RETRIEVAL_LIMIT = int(os.getenv("AHVI_SEMANTIC_LIMIT", "20"))


def _cached_query_vector(query_text: str) -> List[float]:
    """LRU-ish cache for query vectors. Tail-trims when over limit.

    On cache hit we skip the 200ms-3s encode entirely, which is the
    dominant cost of repeated 'Suggest an outfit for today' style flows
    where most users send the same handful of queries.
    """
    cached = _QUERY_VECTOR_CACHE.get(query_text)
    if cached is not None:
        return cached

    if not embedding_service.enabled:
        return []

    try:
        vec = embedding_service.encode_text(query_text)
    except Exception:
        return []

    if vec:
        if len(_QUERY_VECTOR_CACHE) >= _QUERY_VECTOR_CACHE_LIMIT:
            # drop one arbitrary entry — Python 3.7+ dict preserves insertion order
            try:
                _QUERY_VECTOR_CACHE.pop(next(iter(_QUERY_VECTOR_CACHE)))
            except StopIteration:
                pass
        _QUERY_VECTOR_CACHE[query_text] = vec
    return vec


def _semantic_retrieval(
    user_id: str,
    context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Sync semantic retrieval with timing + caching.

    Was previously the dominant slow path: a 62s style-flow latency in
    production was almost entirely the cold-load of sentence-transformers
    plus repeated encode() of the same query text. Now:
      - query-text vector is cached per-process (LRU-ish)
      - retrieval limit reduced from 40 to 20 (env-tunable)
      - timing is logged per stage so future regressions are visible
    """
    if not qdrant_service.enabled():
        return [], {}

    import time as _time
    started = _time.perf_counter()

    try:
        if not embedding_service.enabled:
            logger.debug("ahvi.semantic.skip reason=embeddings_disabled")
            return [], {}

        query_text = _build_semantic_query_text(context)
        encode_started = _time.perf_counter()
        query_vector = _cached_query_vector(query_text)
        encode_ms = round((_time.perf_counter() - encode_started) * 1000, 1)
        if not query_vector:
            logger.info("ahvi.semantic.encode_empty query=%r encode_ms=%s", query_text[:80], encode_ms)
            return [], {}

        fetch_started = _time.perf_counter()
        # search_similar is the live retrieval method; semantic_retrieve was
        # removed and every call threw AttributeError (then wasted the encode).
        # score_threshold=0.0 preserves the old top-N recall — ranking happens
        # downstream.
        results = qdrant_service.search_similar(
            query_vector,
            user_id=user_id,
            limit=_SEMANTIC_RETRIEVAL_LIMIT,
            score_threshold=0.0,
        )
        fetch_ms = round((_time.perf_counter() - fetch_started) * 1000, 1)

        wardrobe_items, semantic_map = _semantic_results_to_wardrobe(results)

        total_ms = round((_time.perf_counter() - started) * 1000, 1)
        logger.info(
            "ahvi.semantic.timing user=%s encode_ms=%s fetch_ms=%s total_ms=%s "
            "limit=%s results=%s cache_size=%s",
            user_id, encode_ms, fetch_ms, total_ms,
            _SEMANTIC_RETRIEVAL_LIMIT, len(wardrobe_items),
            len(_QUERY_VECTOR_CACHE),
        )
        return wardrobe_items, semantic_map
    except Exception as exc:
        total_ms = round((_time.perf_counter() - started) * 1000, 1)
        logger.warning("ahvi.semantic.failed user=%s err=%s total_ms=%s", user_id, exc, total_ms)
        return [], {}


async def _semantic_retrieval_async(
    user_id: str,
    context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Async-friendly semantic retrieval — does NOT freeze the event loop.

    Root cause of the 62s style-flow latency: SentenceTransformer.encode is
    CPU-bound (200ms-3s) but was being called inline from an async route,
    so even though the work was modest the loop was unable to handle the
    request concurrently. asyncio.to_thread() hands the encode to a thread
    so the event loop keeps serving other connections.
    """
    if not qdrant_service.enabled():
        return [], {}

    try:
        from services.embedding_service import embeddings_enabled, encode_text_async

        if not embeddings_enabled():
            return [], {}

        query_text = _build_semantic_query_text(context)
        query_vector = await encode_text_async(query_text)
        if not query_vector:
            return [], {}

        # Run the Qdrant fetch in a thread too — qdrant-client is sync.
        # Lambda keeps limit as a kwarg; search_similar's 3rd positional is
        # `category`, so passing 40 positionally would mis-bind.
        results = await asyncio.to_thread(
            lambda: qdrant_service.search_similar(
                query_vector, user_id, limit=40, score_threshold=0.0
            )
        )
        return _semantic_results_to_wardrobe(results)
    except Exception:
        return [], {}


def _merge_wardrobe(
    base: Dict[str, List[Dict[str, Any]]],
    semantic_items: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    merged = {
        "tops": list(base.get("tops", [])),
        "bottoms": list(base.get("bottoms", [])),
        "shoes": list(base.get("shoes", [])),
        "outerwear": list(base.get("outerwear", [])),
        "dresses": list(base.get("dresses", [])),
        "accessories": list(base.get("accessories", [])),
    }

    seen = set()
    for key in merged:
        for item in merged[key]:
            seen.add(str(item.get("id", "")))

    for item in semantic_items:
        item_id = str(item.get("id", ""))
        if not item_id or item_id in seen:
            continue

        category_name = _infer_category(item)
        bucket = _bucket_for_category(category_name)
        if bucket not in merged:
            seen.add(item_id)
            continue
        merged[bucket].append(_normalize_item(item, _type_for_category(category_name)))
        seen.add(item_id)

    return merged


def _occasion_filter(
    wardrobe: Dict[str, List[Dict[str, Any]]], occasion: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Soft occasion handling:
    - keep usable wardrobe items instead of dropping them
    - move exact occasion matches first
    - date_night/casual/work pieces still produce visual boards
    """
    occ = str(occasion or "").strip().lower()
    if not occ:
        return wardrobe

    reordered: Dict[str, List[Dict[str, Any]]] = {}

    for key, items in wardrobe.items():
        matched: List[Dict[str, Any]] = []
        rest: List[Dict[str, Any]] = []

        for item in items or []:
            tags = item.get("occasion_tags") or item.get("occasions") or []
            if isinstance(tags, str):
                tags = [tags]

            normalized_tags = [
                str(v).strip().lower() for v in tags if str(v or "").strip()
            ]

            if _occasion_match_strength(normalized_tags, occ) > 0.0:
                matched.append(item)
            else:
                rest.append(item)

        reordered[key] = matched + rest

    return reordered


# Vision-tagged items use specific real-world labels (e.g. "work",
# "client presentation", "dinner date"). Pipeline occasions are coarser
# ("office", "date night"). Map each pipeline-occasion to the set of
# vision-style tags that should count as a positive match.
_OCCASION_SYNONYMS: Dict[str, set[str]] = {
    "office": {
        "office",
        "work",
        "workplace",
        "client meeting",
        "client presentation",
        "business meeting",
        "business",
        "interview",
        "professional",
        "formal day",
    },
    "date night": {
        "date night",
        "date",
        "dinner",
        "dinner date",
        "evening",
        "evening out",
        "romantic",
    },
    "date": {
        "date",
        "date night",
        "dinner date",
        "dinner",
        "romantic",
    },
    "casual outing": {
        "casual",
        "casual outing",
        "weekend",
        "weekend coffee run",
        "coffee",
        "everyday",
        "brunch",
        "brunch date",
    },
    "casual": {
        "casual",
        "everyday",
        "weekend",
        "weekend coffee run",
        "coffee",
    },
    "party": {
        "party",
        "club",
        "night out",
        "evening out",
        "celebration",
    },
    "travel": {
        "travel",
        "airport transit",
        "airport",
        "vacation",
        "vacation city walk",
        "trip",
        "city walk",
    },
    "workout": {
        "workout",
        "gym",
        "training",
        "fitness",
        "athletic",
        "sports",
        "running",
        "exercise",
    },
    "business meeting": {
        "office",
        "work",
        "client meeting",
        "client presentation",
        "business meeting",
        "interview",
        "professional",
    },
    "today": {
        "casual",
        "everyday",
        "weekend",
        "work",
    },
    "rainy day": {
        "rainy commute",
        "rainy day",
        "weather",
    },
    "traditional": {
        "traditional",
        "ethnic",
        "wedding",
        "festival",
    },
}


def _occasion_match_strength(tags: List[str], occasion: str) -> float:
    """Return 0.0 / 1.0 / 2.0 based on tag overlap with occasion synonyms.

    2.0 = exact tag match
    1.0 = synonym match OR substring overlap
    0.0 = no overlap
    """
    occ = str(occasion or "").strip().lower()
    if not occ or not tags:
        return 0.0
    tag_set = set(tags)
    if occ in tag_set:
        return 2.0
    synonyms = _OCCASION_SYNONYMS.get(occ, set())
    if synonyms and (synonyms & tag_set):
        return 2.0
    if any(occ in t or t in occ for t in tags):
        return 1.0
    if synonyms:
        for t in tags:
            if any(s in t or t in s for s in synonyms):
                return 1.0
    return 0.0


def _master_piece_score(
    item: Dict[str, Any], occasion: str, semantic_map: Dict[str, float]
) -> float:
    # Vision capture saves the field as `occasions`; legacy code referenced
    # `occasion_tags` only — so every item scored 0 and the picker fell back
    # to semantic similarity, returning the same hero on every prompt.
    raw_tags: List[Any] = []
    for key in ("occasion_tags", "occasions"):
        value = item.get(key)
        if isinstance(value, list):
            raw_tags.extend(value)
        elif isinstance(value, str):
            raw_tags.extend(part.strip() for part in value.split(","))
    tags = [str(v).strip().lower() for v in raw_tags if str(v or "").strip()]

    score = _occasion_match_strength(tags, occasion)
    item_id = str(item.get("id", "")).strip()
    if item_id:
        score += float(semantic_map.get(item_id, 0.0))
    return score


def _pick_master_piece(
    wardrobe: Dict[str, List[Dict[str, Any]]],
    occasion: str,
    semantic_map: Dict[str, float],
) -> Tuple[str, Dict[str, Any]]:
    candidates: List[Tuple[str, Dict[str, Any], float]] = []
    for row in wardrobe.get("tops", []) or []:
        candidates.append(
            ("top", row, _master_piece_score(row, occasion, semantic_map))
        )
    for row in wardrobe.get("dresses", []) or []:
        candidates.append(
            ("dress", row, _master_piece_score(row, occasion, semantic_map) + 0.1)
        )

    if not candidates:
        return "", {}
    candidates.sort(key=lambda x: x[2], reverse=True)
    best = candidates[0]
    return best[0], best[1]


def _build_diverse_combo_pool(
    master_to_combos: List[Tuple[str, List[Dict[str, Any]]]],
    *,
    per_master_min: int = 6,
    global_max: int = 60,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Two-phase combo pool builder used inside get_daily_outfits.

    Phase 1: each master hero is given `per_master_min` slots BEFORE any
    one master can flood the global cap.
    Phase 2: round-robin fill from the overflow until `global_max`.

    Pure function — exposed at module level so it can be unit-tested
    without scaffolding the whole pipeline.

    Returns (combo_pool, per_master_counts).
    """
    combos_out: List[Dict[str, Any]] = []
    seen: set = set()
    counts: Dict[str, int] = {}
    overflow: List[Tuple[str, List[Dict[str, Any]]]] = []

    for hero_name, combos in master_to_combos:
        kept = 0
        spill: List[Dict[str, Any]] = []
        for combo in combos:
            combo_id = str((combo or {}).get("combo_id") or "").strip()
            if not combo_id or combo_id in seen:
                continue
            seen.add(combo_id)
            if kept < per_master_min and len(combos_out) < global_max:
                combos_out.append(combo)
                kept += 1
            else:
                spill.append(combo)
        counts[hero_name] = kept
        overflow.append((hero_name, spill))

    idx = 0
    while len(combos_out) < global_max:
        progressed = False
        for _name, ov in overflow:
            if idx < len(ov):
                combos_out.append(ov[idx])
                progressed = True
                if len(combos_out) >= global_max:
                    break
        if not progressed:
            break
        idx += 1

    return combos_out, counts


def _pick_master_candidates(
    wardrobe: Dict[str, List[Dict[str, Any]]],
    occasion: str,
    semantic_map: Dict[str, float],
    *,
    limit: int = 6,
) -> List[Tuple[str, Dict[str, Any]]]:
    candidates: List[Tuple[str, Dict[str, Any], float]] = []
    for row in wardrobe.get("tops", []) or []:
        candidates.append(
            ("top", row, _master_piece_score(row, occasion, semantic_map))
        )
    for row in wardrobe.get("dresses", []) or []:
        candidates.append(
            ("dress", row, _master_piece_score(row, occasion, semantic_map) + 0.1)
        )

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[2], reverse=True)
    selected: List[Tuple[str, Dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for master_type, item, _ in candidates:
        item_id = str((item or {}).get("id", "")).strip()
        if not item_id or item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        selected.append((master_type, item))
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def _build_master_combos(
    wardrobe: Dict[str, List[Dict[str, Any]]],
    master_type: str,
    master_item: Dict[str, Any],
    *,
    max_combos: int = 36,
) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    if master_type == "top":
        for bottom in wardrobe.get("bottoms", []) or []:
            for shoe in wardrobe.get("shoes", []) or []:
                combos.append(
                    {
                        "combo_id": f"{master_item.get('id')}|{bottom.get('id')}|{shoe.get('id')}",
                        "master_type": "top",
                        "master_piece": master_item,
                        "top": master_item,
                        "bottom": bottom,
                        "shoes": shoe,
                        "dress": {},
                        "outerwear": {},
                        "accessories": [],
                    }
                )
                if len(combos) >= max_combos:
                    return combos
    elif master_type == "dress":
        for shoe in wardrobe.get("shoes", []) or []:
            combos.append(
                {
                    "combo_id": f"{master_item.get('id')}|{shoe.get('id')}",
                    "master_type": "dress",
                    "master_piece": master_item,
                    "top": {},
                    "bottom": {},
                    "shoes": shoe,
                    "dress": master_item,
                    "outerwear": {},
                    "accessories": [],
                }
            )
            if len(combos) >= max_combos:
                return combos
    return combos


def _combo_palette(combo: Dict[str, Any]) -> List[str]:
    colors: List[str] = []
    for part in ("master_piece", "top", "bottom", "shoes", "dress"):
        item = combo.get(part, {}) or {}
        color = str(item.get("color", "")).strip().lower()
        if color:
            colors.append(color)
    return colors


def _combo_patterns(combo: Dict[str, Any]) -> List[str]:
    patterns: List[str] = []
    for part in ("master_piece", "top", "bottom", "shoes", "dress"):
        item = combo.get(part, {}) or {}
        p = str(item.get("fabric", "")).strip().lower()
        if p:
            patterns.append(p)
    return patterns


def _llm_filter_combo_ids(
    *,
    occasion: str,
    stage: str,
    master_type: str,
    master_piece: Dict[str, Any],
    combos: List[Dict[str, Any]],
    max_ids: int = 24,
) -> List[str]:
    if not combos:
        return []

    # Each combo carries its own master (combo['top'] / combo['dress']);
    # we send those into the prompt so the LLM judges each combo on its
    # own merits instead of assuming every combo shares the singular
    # master_piece passed in from the caller. Without this, the LLM
    # filtered out combos that used a different hero than the one it was
    # told the "master" was, collapsing all final cards to one hero.
    compact = []
    for row in combos:
        combo_top = (row.get("top") or {}) or {}
        combo_dress = (row.get("dress") or {}) or {}
        hero = combo_dress if combo_dress else combo_top
        compact.append(
            {
                "combo_id": row.get("combo_id"),
                "palette": _combo_palette(row),
                "patterns": _combo_patterns(row),
                "top": combo_top.get("name"),
                "bottom": (row.get("bottom") or {}).get("name"),
                "shoes": (row.get("shoes") or {}).get("name"),
                "dress": combo_dress.get("name"),
                "hero_color": hero.get("color"),
                "hero_fabric": hero.get("fabric"),
            }
        )

    prompt = f"""
You are a fashion combo evaluator.
Occasion: {occasion}
Stage: {stage}

Each combo below is a complete outfit (top OR dress, plus bottom, shoes,
palette, patterns). Evaluate every combo independently — do NOT assume
all combos share the same hero.

Select best combo_ids from this list:
{json.dumps(compact, ensure_ascii=True)}

Return strict JSON object:
{{
  "selected_combo_ids": ["id1","id2","id3"]
}}
Rules:
- Keep at most {max(1, int(max_ids))} ids.
- Keep only ids present in the input.
- Prioritize practical wearable harmony.
"""
    try:
        parsed = ai_gateway.generate_json_object(
            prompt, signals={"context_mode": "styling"}
        )
        selected = (
            parsed.get("selected_combo_ids", []) if isinstance(parsed, dict) else []
        )
        normalized = []
        valid = {str(c.get("combo_id")) for c in compact}
        for value in selected if isinstance(selected, list) else []:
            cid = str(value).strip()
            if cid and cid in valid and cid not in normalized:
                normalized.append(cid)
        return normalized[: max(1, int(max_ids))]
    except Exception:
        return []


def _rule_color_fallback(
    master_piece: Dict[str, Any],
    combos: List[Dict[str, Any]],
    max_ids: int = 24,
) -> List[str]:
    # Use each combo's own hero color, not just the singular master_piece
    # passed in (which is master_candidates[0] and discriminates against
    # combos built from other masters).
    fallback_master_color = str((master_piece or {}).get("color", "")).strip().lower()
    neutrals = {"black", "white", "beige", "gray", "grey", "navy", "brown"}
    selected: List[str] = []
    for combo in combos:
        palette = _combo_palette(combo)
        uniq = set(palette)
        if not uniq:
            continue
        hero = (combo.get("top") or combo.get("dress") or {}) or {}
        hero_color = str(hero.get("color", "")).strip().lower() or fallback_master_color
        if hero_color and hero_color in uniq:
            selected.append(str(combo.get("combo_id")))
            continue
        if len(uniq) <= 3 and any(c in neutrals for c in uniq):
            selected.append(str(combo.get("combo_id")))
    return selected[: max(1, int(max_ids))]


def _rule_pattern_fallback(
    combos: List[Dict[str, Any]],
    max_ids: int = 24,
) -> List[str]:
    selected: List[str] = []
    for combo in combos:
        pats = [p for p in _combo_patterns(combo) if p]
        if not pats:
            selected.append(str(combo.get("combo_id")))
            continue
        standout = sum(
            1 for p in pats if p in {"striped", "checked", "floral", "printed"}
        )
        if standout <= 1:
            selected.append(str(combo.get("combo_id")))
    return selected[: max(1, int(max_ids))]


def _accessory_tokens(item: Dict[str, Any]) -> set[str]:
    blob = " ".join(
        str((item or {}).get(k, "") or "")
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
        )
    ).lower()
    return set(re.sub(r"[^a-z0-9]+", " ", blob).split())


def _accessory_item_key(item: Dict[str, Any]) -> str:
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
    ).strip().lower()


def _is_accessory_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False

    tokens = _accessory_tokens(item)
    accessory_accept = {
        "accessory",
        "accessories",
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
        "bracelet",
        "bracelets",
        "ring",
        "rings",
        "necklace",
        "necklaces",
        "earring",
        "earrings",
        "jewelry",
        "jewellery",
        "scarf",
        "scarves",
    }
    clothing_reject = {
        "top",
        "tops",
        "shirt",
        "shirts",
        "tee",
        "tshirt",
        "bottom",
        "bottoms",
        "pant",
        "pants",
        "trouser",
        "trousers",
        "jean",
        "jeans",
        "dress",
        "dresses",
        "footwear",
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "boot",
        "boots",
        "sandal",
        "sandals",
        "outerwear",
        "jacket",
        "blazer",
    }
    if tokens.intersection(clothing_reject):
        return False
    return bool(tokens.intersection(accessory_accept))


def _accessory_type(item: Dict[str, Any]) -> str:
    tokens = _accessory_tokens(item)
    if tokens.intersection({"watch", "watches"}):
        return "watch"
    if tokens.intersection({"sunglass", "sunglasses", "eyewear", "glasses"}):
        return "eyewear"
    if tokens.intersection({"bag", "bags", "purse", "handbag", "clutch", "tote"}):
        return "bag"
    if tokens.intersection({"bracelet", "bracelets"}):
        return "bracelet"
    if tokens.intersection({"ring", "rings"}):
        return "ring"
    if tokens.intersection({"necklace", "necklaces"}):
        return "necklace"
    if tokens.intersection({"earring", "earrings"}):
        return "earring"
    if tokens.intersection({"belt", "belts"}):
        return "belt"
    if tokens.intersection({"scarf", "scarves"}):
        return "scarf"
    if tokens.intersection({"cap", "caps", "hat", "hats"}):
        return "headwear"
    if tokens.intersection({"jewelry", "jewellery"}):
        return "jewelry"
    return "accessory"


def _accessory_has_image(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    return bool(
        item.get("masked_url")
        or item.get("maskedUrl")
        or item.get("image_url")
        or item.get("imageUrl")
        or item.get("raw_url")
        or item.get("rawUrl")
        or item.get("url")
        or item.get("image")
    )


def _select_accessories(
    wardrobe: Dict[str, List[Dict[str, Any]]], combo: Dict[str, Any], limit: int = 2
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if isinstance(combo, dict) and isinstance(combo.get("accessories"), list):
        candidates.extend([x for x in combo.get("accessories") if isinstance(x, dict)])
    if isinstance(wardrobe, dict):
        for values in wardrobe.values():
            if isinstance(values, list):
                candidates.extend([x for x in values if isinstance(x, dict)])

    candidates = [item for item in candidates if _is_accessory_item(item)]
    priority = {
        "bag": 0,
        "belt": 1,
        "ring": 2,
        "bracelet": 3,
        "necklace": 4,
        "eyewear": 5,
        "watch": 6,
        "earring": 7,
        "scarf": 8,
        "headwear": 9,
        "jewelry": 10,
        "accessory": 99,
    }
    seed = _outfit_signature(combo) or str(combo.get("combo_id") or "")
    accessory_budget = max(0, min(int(limit), 2))
    # Luxury boards need restraint. Some combinations are cleaner with no accent,
    # and avoiding a guaranteed watch stops every board from feeling templated.
    if seed:
        accessory_budget = _stable_offset(f"accessory-budget:{seed}", accessory_budget + 1)
    if accessory_budget <= 0:
        return []
    # Do not make every generated board accessory-heavy. Five to six items should
    # happen only when the accessories are genuinely useful.
    if accessory_budget > 2 and seed and _stable_offset(f"accessory-restraint:{seed}", 3) == 0:
        accessory_budget = 2
    candidates.sort(
        key=lambda item: (
            _stable_offset(f"accessory-rotate:{seed}:{_accessory_item_key(item)}", 7),
            priority.get(_accessory_type(item), 99),
            0 if _accessory_has_image(item) else 1,
            str(item.get("name") or item.get("label") or ""),
        )
    )

    picked: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_types: set[str] = set()
    for item in candidates:
        key = _accessory_item_key(item)
        typ = _accessory_type(item)
        if key in seen_ids or typ in seen_types:
            continue
        picked.append(item)
        seen_ids.add(key)
        seen_types.add(typ)
        if len(picked) >= accessory_budget:
            break
    return picked


def generate_combinations(
    wardrobe: Dict[str, List[Dict[str, Any]]], max_candidates: int = 600
) -> List[Dict[str, Any]]:
    tops = wardrobe.get("tops", [])
    bottoms = wardrobe.get("bottoms", [])
    shoes = wardrobe.get("shoes", [])
    outerwear = wardrobe.get("outerwear", [])

    combos: List[Dict[str, Any]] = []
    for top, bottom, shoe in product(tops, bottoms, shoes):
        base_combo = {
            "top": top,
            "bottom": bottom,
            "shoes": shoe,
            "outerwear": {},
            "combo_id": "|".join(
                [top.get("id", ""), bottom.get("id", ""), shoe.get("id", "")]
            ),
        }
        combos.append(base_combo)
        if outerwear:
            for layer in outerwear[:4]:
                layered = dict(base_combo)
                layered["outerwear"] = layer
                layered["combo_id"] = f"{base_combo['combo_id']}|{layer.get('id', '')}"
                combos.append(layered)
        if len(combos) >= max_candidates:
            return combos
    return combos


def validate_outfit(outfit: Dict[str, Any], context: Dict[str, Any]) -> bool:
    top = outfit.get("top", {}) or {}
    bottom = outfit.get("bottom", {}) or {}

    top_type = str(top.get("type", "")).lower()
    bottom_type = str(bottom.get("type", "")).lower()
    occasion = _normalize_pipeline_occasion(context.get("occasion"), context)

    if "formal" in top_type and "shorts" in _tokens(bottom_type):
        return False
    if occasion in ("office", "work") and "shorts" in _tokens(bottom_type):
        return False
    return True


def _similarity_score(outfit_a: Dict[str, Any], outfit_b: Dict[str, Any]) -> float:
    if not outfit_a or not outfit_b:
        return 0.0
    score = 0.0
    checks = 0
    for part in ("top", "bottom", "dress", "shoes", "outerwear"):
        a = outfit_a.get(part, {}) or {}
        b = outfit_b.get(part, {}) or {}
        if not a or not b:
            continue
        checks += 1
        if str(a.get("type", "")).lower() == str(b.get("type", "")).lower():
            score += 0.4
        if str(a.get("color", "")).lower() == str(b.get("color", "")).lower():
            score += 0.4
        if str(a.get("fabric", "")).lower() == str(b.get("fabric", "")).lower():
            score += 0.2
    if checks == 0:
        return 0.0
    return min(1.0, score / checks)


def _color_score(colors: List[str], preferred_colors: List[str]) -> float:
    palette = [c for c in colors if c]
    if not palette:
        return 0.0
    unique = set(palette)

    score = 0.4 if len(unique) <= 2 else 0.1
    neutrals = {"black", "white", "beige", "gray", "grey", "navy", "brown"}
    if any(c in neutrals for c in unique):
        score += 0.3

    if preferred_colors:
        hits = sum(1 for c in palette if c in preferred_colors)
        score += min(0.6, hits * 0.2)

    return min(1.5, score)


def score_outfit(
    outfit: Dict[str, Any],
    context: Dict[str, Any],
    memory: Dict[str, Any],
    rules: Dict[str, Any],
    semantic_map: Dict[str, float],
) -> Dict[str, Any]:
    weather = str(context.get("weather", "")).lower()
    occasion = _normalize_pipeline_occasion(context.get("occasion"), context)
    style_dna = context.get("style_dna", {}) or {}

    weather_score = 0.0
    occasion_score = 0.0
    color_intelligence = 0.0
    layering_score = 0.0
    style_graph_bonus = 0.0
    memory_score = 0.0
    feedback_adjustment = 0.0
    semantic_relevance = 0.0

    colors = []
    item_ids = []
    outfit_items = []

    for part in ("master_piece", "top", "bottom", "dress", "shoes", "outerwear"):
        item = outfit.get(part, {}) or {}
        if not item:
            continue
        if isinstance(item, dict):
            outfit_items.append(item)
        color = str(item.get("color", "")).lower()
        colors.append(color)

        item_id = str(item.get("id", ""))
        if item_id:
            item_ids.append(item_id)
            semantic_relevance += float(semantic_map.get(item_id, 0.0))

    outfit_view = {**outfit, "items": outfit.get("items") or outfit_items}
    occasion_result = score_occasion_compatibility(
        outfit_view,
        {
            **context,
            "occasion": occasion,
            "prompt": context.get("query") or context.get("prompt"),
        },
    )
    weather_result = score_weather_compatibility(outfit_view, context)
    occasion_score = (float(occasion_result.get("score") or 0.5) - 0.5) * 8.0
    weather_score = (float(weather_result.get("score") or 0.5) - 0.5) * 3.0
    if occasion_result.get("reject"):
        occasion_score -= 8.0

    color_intelligence = _color_score(
        colors, [str(c).lower() for c in style_dna.get("preferred_colors", [])]
    )

    has_outerwear = bool(outfit.get("outerwear"))
    if weather in ("cold", "rain", "rainy", "chilly") and has_outerwear:
        layering_score += 1.0
    elif weather in ("hot", "warm") and has_outerwear:
        layering_score -= 0.4
    else:
        layering_score += 0.3

    graph = context.get("style_graph", {}) or {}
    top_id = str((outfit.get("top") or {}).get("id", ""))
    bottom_id = str((outfit.get("bottom") or {}).get("id", ""))
    shoes_id = str((outfit.get("shoes") or {}).get("id", ""))
    outer_id = str((outfit.get("outerwear") or {}).get("id", ""))

    style_graph_bonus += style_graph_engine.pair_weight(graph, top_id, bottom_id)
    style_graph_bonus += style_graph_engine.pair_weight(graph, bottom_id, shoes_id)
    if outer_id:
        style_graph_bonus += style_graph_engine.pair_weight(graph, outer_id, top_id)

    recent = memory.get("recent_outfits", [])[:20]
    liked = memory.get("liked_outfits", [])[:30]
    disliked = memory.get("disliked_outfits", [])[:30]

    repetition_penalty = sum(_similarity_score(outfit, r) * 0.9 for r in recent)
    memory_score = max(0.0, 2.0 - min(2.0, repetition_penalty))

    liked_sim = max([_similarity_score(outfit, o) for o in liked], default=0.0)
    disliked_sim = max([_similarity_score(outfit, o) for o in disliked], default=0.0)
    feedback_adjustment = (liked_sim * 1.8) - (disliked_sim * 2.2)

    semantic_relevance = semantic_relevance / max(1, len(item_ids))

    if not validate_outfit(outfit, context):
        occasion_score -= 5.0

    base_score = (
        weather_score
        + occasion_score
        + color_intelligence
        + layering_score
        + style_graph_bonus
        + memory_score
        + feedback_adjustment
        + semantic_relevance
    )

    features = {
        "occasion_rules": round(occasion_score + weather_score, 4),
        "occasion_compatibility": round(float(occasion_result.get("score") or 0.5), 4),
        "weather_compatibility": round(float(weather_result.get("score") or 0.5), 4),
        "occasion_reject": 1.0 if occasion_result.get("reject") else 0.0,
        "color_intelligence": round(color_intelligence, 4),
        "layering": round(layering_score, 4),
        "style_graph": round(style_graph_bonus, 4),
        "memory": round(memory_score, 4),
        "feedback": round(feedback_adjustment, 4),
        "semantic_relevance": round(semantic_relevance, 4),
    }

    scored = deepcopy(outfit)
    scored["score"] = round(base_score, 3)
    scored["ml_features"] = features
    scored["score_breakdown"] = features
    scored.setdefault("score_meta", {})
    scored["score_meta"].update(
        {
            "occasion_profile": occasion_result.get("profile") or {},
            "occasion_compatibility_score": occasion_result.get("score"),
            "occasion_compatibility": occasion_result,
            "occasion_penalties": occasion_result.get("penalties", []),
            "occasion_reject": occasion_result.get("reject", False),
            "weather_compatibility": weather_result,
        }
    )
    return scored


def _explanation_for_outfit(outfit: Dict[str, Any], context: Dict[str, Any]) -> str:
    top = outfit.get("top", {}) or {}
    bottom = outfit.get("bottom", {}) or {}
    dress = outfit.get("dress", {}) or {}
    shoes = outfit.get("shoes", {}) or {}
    outer = outfit.get("outerwear", {}) or {}
    accessories = outfit.get("accessories", []) or []
    master_type = str(outfit.get("master_type", "")).lower()

    if master_type == "dress" or dress:
        lines = [
            f"Occasion fit: {context.get('occasion', 'daily')} look anchored by {dress.get('name', 'dress')}.",
            f"Color harmony: {dress.get('color', 'neutral')} pairs with {shoes.get('color', 'neutral')} footwear.",
        ]
    else:
        lines = [
            f"Occasion fit: {context.get('occasion', 'daily')} look built with {top.get('name', 'top')} and {bottom.get('name', 'bottom')}.",
            f"Color harmony: {top.get('color', 'neutral')} balances with {bottom.get('color', 'neutral')} and {shoes.get('color', 'neutral')}.",
        ]
    if outer:
        lines.append(
            f"Layering: {outer.get('name', 'outerwear')} adds weather-ready structure."
        )
    if accessories:
        lines.append(
            f"Accessories: {', '.join(str(x.get('name', 'accent')) for x in accessories)} complete the style board."
        )
    lines.append(
        "Personalization: ranking boosted using your Style DNA, memory, and feedback signals."
    )
    return " ".join(lines)


def _story_title(score: float) -> str:
    if score >= 9:
        return "Hero Look"
    if score >= 7:
        return "Signature Combo"
    if score >= 5:
        return "Polished Daily"
    return "Easy Win"


def _generate_story(outfit: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    top = outfit.get("top", {}) or {}
    bottom = outfit.get("bottom", {}) or {}
    dress = outfit.get("dress", {}) or {}
    shoes = outfit.get("shoes", {}) or {}
    outer = outfit.get("outerwear", {}) or {}
    master_type = str(outfit.get("master_type", "")).lower()

    setting = str(context.get("occasion") or "the day").replace("_", " ")
    weather = str(context.get("weather") or "today")

    if master_type == "dress" or dress:
        narrative = (
            f"For {setting}, anchor the look with {dress.get('name', 'a dress')} and finish with "
            f"{shoes.get('name', 'your shoes')}."
        )
    else:
        narrative = (
            f"For {setting}, start with {top.get('name', 'a top')} to set the tone, "
            f"ground it with {bottom.get('name', 'a bottom')}, and finish with {shoes.get('name', 'your shoes')}."
        )
    if outer:
        narrative += f" Add {outer.get('name', 'an outer layer')} for {weather} comfort and extra polish."

    confidence = float(outfit.get("rank_score", outfit.get("score", 0.0)))
    return {
        "title": _story_title(confidence),
        "narrative": narrative,
        "why_it_works": _explanation_for_outfit(outfit, context),
    }


def _build_tryon_payload(
    outfit: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    def _safe_id(value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    top_id = _safe_id((outfit.get("top") or {}).get("id"))
    bottom_id = _safe_id((outfit.get("bottom") or {}).get("id"))
    dress_id = _safe_id((outfit.get("dress") or {}).get("id"))
    shoes_id = _safe_id((outfit.get("shoes") or {}).get("id"))
    outerwear_id = _safe_id((outfit.get("outerwear") or {}).get("id"))

    # Try-on supports either top+bottom+shoes or dress+shoes.
    if not (top_id and bottom_id and shoes_id):
        if not (dress_id and shoes_id):
            return {}

    payload = {
        "mode": "virtual_try_on",
        "occasion": context.get("occasion"),
        "weather": context.get("weather"),
        "items": {
            "top_id": top_id,
            "bottom_id": bottom_id,
            "dress_id": dress_id,
            "shoes_id": shoes_id,
            "outerwear_id": outerwear_id,
        },
        "prompt": f"Try on this look for {context.get('occasion', 'daily wear')}.",
    }
    return payload


def _flatten_outfit_items(outfit: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not isinstance(outfit, dict):
        return items

    # master_piece is a meta-pointer to the same dict as top (or dress).
    # Ordering (top, bottom, dress, shoes, outerwear) puts the actual hero
    # first; we then dedupe so master_piece never re-renders as a separate
    # item. Without dedupe, cards showed two shirts (one from master_piece,
    # one from top) which the UI flattened to whichever sorted first.
    seen_ids: set[str] = set()

    def _add(value: Any) -> None:
        if not isinstance(value, dict) or not value:
            return
        item_id = str(
            value.get("id")
            or value.get("$id")
            or value.get("image_id")
            or value.get("name")
            or ""
        ).strip()
        if item_id and item_id in seen_ids:
            return
        if item_id:
            seen_ids.add(item_id)
        items.append(value)

    for part in ("top", "bottom", "dress", "shoes", "outerwear", "master_piece"):
        _add(outfit.get(part, {}))

    accessories = outfit.get("accessories") or []
    if isinstance(accessories, list):
        for acc in accessories:
            _add(acc)

    return items


def _unified_style_snapshot(
    items: List[Dict[str, Any]], context: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        graph = context.get("style_graph", {}) or {}
        return style_scorer.score_outfit(items=items, context=context, graph=graph)
    except Exception:
        return {"score": 0.0, "label": "Basic", "reasons": []}


def _attach_score_meta(outfit: Dict[str, Any], context: Dict[str, Any]) -> None:
    """
    Backward/forward compatible contract:
    - New: outfit['unified_style'] (dict)
    - Legacy/phase-plan: outfit['score_meta'] (dict)
    """
    if not isinstance(outfit, dict):
        return
    items = (
        outfit.get("refined_items")
        if isinstance(outfit.get("refined_items"), list)
        else outfit.get("items")
    )
    if not isinstance(items, list):
        items = _flatten_outfit_items(outfit)
        outfit["items"] = items
    snapshot = _unified_style_snapshot(items, context)
    outfit["unified_style"] = snapshot
    outfit["score_meta"] = snapshot
    try:
        outfit["style_score"] = float(snapshot.get("score") or 0.0)
    except Exception:
        outfit["style_score"] = 0.0

    # Phase plan expects 'score' to reflect the style_scorer output after refinement.
    # Preserve the original pipeline score so we don't lose internal ranking context.
    if "pipeline_score" not in outfit:
        outfit["pipeline_score"] = outfit.get("score")
    outfit["score"] = outfit.get("style_score")


def _swap_part(
    outfit: Dict[str, Any], part: str, candidate: Dict[str, Any]
) -> Dict[str, Any]:
    updated = deepcopy(outfit)
    if not isinstance(candidate, dict) or not candidate:
        return updated
    updated[part] = dict(candidate)
    return updated


def _closed_loop_fix(
    outfit: Dict[str, Any],
    *,
    context: Dict[str, Any],
    wardrobe: List[Dict[str, Any]],
    user_memory: Dict[str, Any],
    rules: Dict[str, Any],
    semantic_map: Dict[str, float],
) -> Dict[str, Any]:
    """
    Closed-loop improvement: score -> identify weakness -> swap 1 part -> re-score.
    Deterministic, bounded (max 2 swaps).
    """
    if not isinstance(outfit, dict) or not wardrobe:
        return outfit

    occasion = _normalize_pipeline_occasion(context.get("occasion"), context)
    style_dna = context.get("style_dna", {}) or {}

    palette_hexes: List[str] = []
    try:
        palette = palette_engine.select_palette(
            {
                "event": occasion or None,
                "microtheme": style_dna.get("primary_aesthetic"),
            }
        )
        palette_hexes = [
            str(x).strip() for x in (palette.get("hex") or []) if str(x).strip()
        ]
    except Exception:
        palette_hexes = []

    preferred_colors = []
    if isinstance(style_dna.get("preferred_colors"), list):
        preferred_colors.extend(
            [
                str(x).strip()
                for x in style_dna.get("preferred_colors")
                if str(x).strip()
            ]
        )
    preferred_colors.extend(palette_hexes)

    current = dict(outfit)

    def _score(o: Dict[str, Any]) -> float:
        try:
            return float(o.get("score") or 0.0)
        except Exception:
            return 0.0

    for _ in range(2):
        breakdown = (
            current.get("score_breakdown")
            if isinstance(current.get("score_breakdown"), dict)
            else {}
        )
        try:
            color_intel = float(breakdown.get("color_intelligence") or 0.0)
        except Exception:
            color_intel = 0.0
        try:
            occ_rules = float(breakdown.get("occasion_rules") or 0.0)
        except Exception:
            occ_rules = 0.0
        try:
            layering = float(breakdown.get("layering") or 0.0)
        except Exception:
            layering = 0.0

        # Pick a deterministic weakness priority.
        weakness = (
            "color"
            if color_intel < 0.5
            else (
                "occasion"
                if occ_rules < 0.5
                else ("layering" if layering < 0.2 else "")
            )
        )
        if not weakness:
            break

        master_type = str(current.get("master_type") or "").strip().lower()
        parts = ["shoes"]
        if master_type == "dress" or current.get("dress"):
            parts = ["dress", "shoes"]
        else:
            parts = ["top", "bottom", "shoes"]

        candidate_outfit = None
        for part in parts:
            target_type = (
                "footwear"
                if part == "shoes"
                else (
                    "dress"
                    if part == "dress"
                    else ("top" if part == "top" else "bottom")
                )
            )

            if weakness == "layering" and part in ("top", "bottom", "dress", "shoes"):
                continue

            candidate = wardrobe_selector.find_best_match(
                target_type,
                {**context, "wardrobe": wardrobe},
                preferred_colors=(
                    preferred_colors if weakness in ("color", "occasion") else None
                ),
                require_occasion=occasion if weakness == "occasion" else None,
            )
            if not candidate:
                continue

            swapped = _swap_part(current, part, candidate)
            swapped["items"] = _flatten_outfit_items(swapped)
            swapped["unified_style"] = _unified_style_snapshot(
                swapped["items"], context
            )

            # Re-score via the pipeline scorer (keeps consistency with ranking features).
            try:
                swapped = score_outfit(
                    swapped, context, user_memory, rules, semantic_map
                )
            except Exception:
                pass

            if _score(swapped) >= _score(current) + 0.15:
                candidate_outfit = swapped
                break

        if candidate_outfit is None:
            break
        current = candidate_outfit

    return current


def _build_cards(
    outfits: List[Dict[str, Any]], context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for idx, outfit in enumerate(outfits):
        story = _generate_story(outfit, context)
        tryon_payload = _build_tryon_payload(outfit, context)
        # Prefer refined_items when present (closed-loop refinement pass), otherwise use the flattened outfit items.
        items = (
            outfit.get("refined_items")
            if isinstance(outfit.get("refined_items"), list)
            else None
        )
        if items is None:
            items = _flatten_outfit_items(outfit)
        pipeline_signature = _outfit_signature(outfit)
        set_profile = _set_diversity_profile(outfit)
        cards.append(
            {
                "id": f"outfit_card_{idx + 1}",
                "title": story.get("title"),
                "score": outfit.get("rank_score", outfit.get("score", 0.0)),
                "ml_score": outfit.get("ml_score", 0.0),
                "items": items,
                "explanation": story.get("why_it_works"),
                "story": story,
                "tryon_payload": tryon_payload,
                "pipeline_style_signature": pipeline_signature,
                "pipeline_set_profile": set_profile,
                "style_metadata": {
                    "pipeline_style_signature": pipeline_signature,
                    "set_diversity_profile": set_profile,
                },
            }
        )
    return cards


def _item_id(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("id")
        or item.get("$id")
        or item.get("image_id")
        or item.get("masked_id")
        or ""
    ).strip()


def _outfit_signature(outfit: Dict[str, Any]) -> str:
    return "|".join(
        [
            _item_id(outfit.get("top") or {}),
            _item_id(outfit.get("bottom") or {}),
            _item_id(outfit.get("dress") or {}),
            _item_id(outfit.get("shoes") or {}),
            _item_id(outfit.get("outerwear") or {}),
        ]
    )


def _different_enough(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (
        _item_id(a.get("top") or a.get("dress") or {})
        != _item_id(b.get("top") or b.get("dress") or {})
        or _item_id(a.get("bottom") or {}) != _item_id(b.get("bottom") or {})
        or _item_id(a.get("shoes") or {}) != _item_id(b.get("shoes") or {})
    )


def _outfit_hero_id(outfit: Dict[str, Any]) -> str:
    """Identifier for the hero piece (dress > top). Used to enforce that the
    final card list shows distinct heroes before allowing repeats."""
    return _item_id(outfit.get("dress") or outfit.get("top") or {})


def _outfit_bottom_id(outfit: Dict[str, Any]) -> str:
    return _item_id(outfit.get("bottom") or {})


def _outfit_footwear_id(outfit: Dict[str, Any]) -> str:
    return _item_id(outfit.get("shoes") or outfit.get("footwear") or {})


def _outfit_accessory_types(outfit: Dict[str, Any]) -> List[str]:
    types: List[str] = []
    for item in outfit.get("accessories") or []:
        if isinstance(item, dict):
            typ = _accessory_type(item)
            if typ:
                types.append(typ)
    return types


def _item_attr(item: Dict[str, Any], *keys: str) -> str:
    if not isinstance(item, dict):
        return ""
    for key in keys:
        value = item.get(key)
        if value is not None:
            text = str(value).strip().lower()
            if text:
                return text
    meta = item.get("style_metadata") if isinstance(item.get("style_metadata"), dict) else {}
    for key in keys:
        value = meta.get(key)
        if value is not None:
            text = str(value).strip().lower()
            if text:
                return text
    return ""


def _outfit_items_for_diversity(outfit: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for key in ("dress", "top", "bottom", "shoes", "footwear", "outerwear"):
        value = outfit.get(key)
        if isinstance(value, dict) and value:
            items.append(value)
    for item in outfit.get("accessories") or []:
        if isinstance(item, dict):
            items.append(item)
    return items


def _outfit_palette_key(outfit: Dict[str, Any]) -> str:
    colors: List[str] = []
    for item in _outfit_items_for_diversity(outfit):
        color = _item_attr(
            item,
            "color",
            "colour",
            "dominant_color",
            "dominantColor",
            "palette",
            "tone",
        )
        if color:
            colors.append(color)
    if not colors and isinstance(outfit.get("palette"), list):
        colors = [str(value).strip().lower() for value in outfit.get("palette") if str(value).strip()]
    return "|".join(sorted(set(colors))[:3])


def _outfit_formality_key(outfit: Dict[str, Any]) -> str:
    values: List[str] = []
    for item in _outfit_items_for_diversity(outfit):
        value = _item_attr(
            item,
            "formality",
            "style_role",
            "styleRole",
            "occasion",
            "vibe",
            "aesthetic",
        )
        if value:
            values.append(value)
    meta = outfit.get("score_meta") if isinstance(outfit.get("score_meta"), dict) else {}
    if meta.get("formality"):
        values.append(str(meta.get("formality")).strip().lower())
    return "|".join(sorted(set(values))[:2])


def _set_diversity_profile(outfit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "signature": _outfit_signature(outfit),
        "hero": _outfit_hero_id(outfit),
        "bottom": _outfit_bottom_id(outfit),
        "footwear": _outfit_footwear_id(outfit),
        "accessories": _outfit_accessory_types(outfit),
        "palette": _outfit_palette_key(outfit),
        "formality": _outfit_formality_key(outfit),
    }


def _occasion_diversity_weights(occasion: str, query: str = "") -> Dict[str, float]:
    text = f"{occasion or ''} {query or ''}".lower()
    weights = {
        "hero": 2.0,
        "bottom": 1.25,
        "footwear": 1.0,
        "accessory": 0.5,
        "palette": 0.5,
        "formality": 0.35,
    }
    if any(token in text for token in ("office", "client", "business", "interview", "meeting", "work")):
        weights.update({"bottom": 1.75, "footwear": 1.25, "hero": 2.0})
    elif any(token in text for token in ("date", "dinner", "evening")):
        weights.update({"hero": 2.0, "palette": 0.75, "footwear": 1.1})
    elif any(token in text for token in ("party", "rave", "night_out", "night out", "club")):
        weights.update({"hero": 2.0, "accessory": 0.85, "palette": 0.75})
    elif any(token in text for token in ("capsule", "minimal", "core wardrobe")):
        weights.update({"formality": 0.8, "palette": 0.7, "hero": 1.6})
    elif any(token in text for token in ("beach", "swim", "pool")):
        weights.update({"footwear": 1.3, "accessory": 0.9, "hero": 1.6})
    elif any(token in text for token in ("workout", "gym", "fitness", "training")):
        weights.update({"footwear": 1.4, "accessory": 0.9, "formality": 0.7})
    elif any(token in text for token in ("casual", "weekend", "coffee")):
        weights.update({"hero": 1.8, "palette": 0.75, "footwear": 1.1})
    return weights


def _outfit_base_score(outfit: Dict[str, Any]) -> float:
    meta = outfit.get("score_meta") if isinstance(outfit.get("score_meta"), dict) else {}
    unified = outfit.get("unified_style") if isinstance(outfit.get("unified_style"), dict) else {}
    for value in (
        meta.get("score"),
        unified.get("score"),
        outfit.get("rank_score"),
        outfit.get("score"),
        outfit.get("ml_score"),
    ):
        try:
            return float(value or 0.0)
        except Exception:
            continue
    return 0.0


def _select_set_diverse_outfits(
    outfits: List[Dict[str, Any]],
    *,
    limit: int,
    occasion: str,
    query: str = "",
) -> List[Dict[str, Any]]:
    """Select a diverse set from already guarded, occasion-valid outfits."""
    pool = [outfit for outfit in outfits or [] if isinstance(outfit, dict)]
    logger.info(
        "ahvi.set_diversity.start occasion=%s limit=%s input=%s",
        occasion,
        limit,
        len(pool),
    )
    if not pool or limit <= 0:
        return []

    profiles = {id(outfit): _set_diversity_profile(outfit) for outfit in pool}
    heroes = {profile["hero"] for profile in profiles.values() if profile.get("hero")}
    bottoms = {profile["bottom"] for profile in profiles.values() if profile.get("bottom")}
    footwear = {profile["footwear"] for profile in profiles.values() if profile.get("footwear")}
    accessory_types = {
        typ
        for profile in profiles.values()
        for typ in (profile.get("accessories") or [])
        if typ
    }
    palettes = {profile["palette"] for profile in profiles.values() if profile.get("palette")}
    logger.info(
        "ahvi.set_diversity.pool heroes=%s bottoms=%s footwear=%s accessory_types=%s palettes=%s",
        len(heroes),
        len(bottoms),
        len(footwear),
        len(accessory_types),
        len(palettes),
    )

    pool.sort(key=_outfit_base_score, reverse=True)
    weights = _occasion_diversity_weights(occasion, query)
    selected: List[Dict[str, Any]] = []
    seen_signatures: set[str] = set()

    def counts(key: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for row in selected:
            profile = profiles.get(id(row), {})
            values = profile.get(key)
            if isinstance(values, list):
                iterable = values
            else:
                iterable = [values]
            for value in iterable:
                if value:
                    out[value] = out.get(value, 0) + 1
        return out

    def profile_score(outfit: Dict[str, Any]) -> float:
        profile = profiles.get(id(outfit), {})
        score = _outfit_base_score(outfit)
        hero_counts = counts("hero")
        bottom_counts = counts("bottom")
        footwear_counts = counts("footwear")
        palette_counts = counts("palette")
        formality_counts = counts("formality")
        accessory_counts = counts("accessories")

        if profile.get("hero") and hero_counts.get(profile["hero"], 0):
            score -= weights["hero"] * hero_counts[profile["hero"]]
        elif profile.get("hero"):
            score += 0.35
        if profile.get("bottom") and bottom_counts.get(profile["bottom"], 0):
            score -= weights["bottom"] * bottom_counts[profile["bottom"]]
        elif profile.get("bottom"):
            score += 0.25
        if profile.get("footwear") and footwear_counts.get(profile["footwear"], 0):
            score -= weights["footwear"] * footwear_counts[profile["footwear"]]
        elif profile.get("footwear"):
            score += 0.2
        if profile.get("palette") and palette_counts.get(profile["palette"], 0):
            score -= weights["palette"] * palette_counts[profile["palette"]]
        elif profile.get("palette"):
            score += 0.15
        if profile.get("formality") and formality_counts.get(profile["formality"], 0):
            score -= weights["formality"] * formality_counts[profile["formality"]]
        accessories = profile.get("accessories") or []
        for typ in accessories:
            if accessory_counts.get(typ, 0):
                score -= weights["accessory"] * accessory_counts[typ]
            else:
                score += 0.08
        return score

    def pick_best(candidates: List[Dict[str, Any]]) -> None:
        nonlocal selected
        viable = []
        for outfit in candidates:
            sig = profiles.get(id(outfit), {}).get("signature") or _outfit_signature(outfit)
            if not sig or sig in seen_signatures:
                continue
            viable.append(outfit)
        if not viable or len(selected) >= limit:
            return
        viable.sort(key=profile_score, reverse=True)
        picked = viable[0]
        sig = profiles.get(id(picked), {}).get("signature") or _outfit_signature(picked)
        selected.append(picked)
        seen_signatures.add(sig)

    for key in ("hero", "bottom", "footwear"):
        values = []
        for outfit in pool:
            value = profiles.get(id(outfit), {}).get(key)
            if value and value not in values:
                values.append(value)
        for value in values:
            if len(selected) >= limit:
                break
            pick_best([outfit for outfit in pool if profiles.get(id(outfit), {}).get(key) == value])

    for typ in sorted(accessory_types):
        if len(selected) >= limit:
            break
        pick_best([outfit for outfit in pool if typ in (profiles.get(id(outfit), {}).get("accessories") or [])])

    while len(selected) < min(limit, len(pool)):
        before = len(selected)
        pick_best(pool)
        if len(selected) == before:
            break

    selected_profiles = [profiles.get(id(outfit), {}) for outfit in selected]
    logger.info(
        "ahvi.set_diversity.selected count=%s heroes=%s bottoms=%s footwear=%s accessories=%s",
        len(selected),
        len({profile.get("hero") for profile in selected_profiles if profile.get("hero")}),
        len({profile.get("bottom") for profile in selected_profiles if profile.get("bottom")}),
        len({profile.get("footwear") for profile in selected_profiles if profile.get("footwear")}),
        len({typ for profile in selected_profiles for typ in (profile.get("accessories") or []) if typ}),
    )
    return selected[:limit]


def _diversify_outfits(
    outfits: List[Dict[str, Any]], limit: int = 6
) -> List[Dict[str, Any]]:
    """Two-pass diversification:

    Pass 1 — one card per unique hero (top / dress). Even if a single hero
    has many highly-scored combos, only its best combo is kept here.

    Pass 2 — fill remaining slots with the next-best combos (allowing hero
    repeats), still deduping by full outfit signature.

    The previous implementation only required (top, bottom, shoes) to differ
    in any one slot, so 'Off White Shirt + jeans + sneakers' and 'Off White
    Shirt + black pants + Birkenstocks' both passed — every card ended up
    with the same hero.
    """
    selected: List[Dict[str, Any]] = []
    seen_signatures: set[str] = set()
    seen_heroes: set[str] = set()
    pool = [o for o in outfits or [] if isinstance(o, dict)]
    unique_bottoms = {b for b in (_outfit_bottom_id(o) for o in pool) if b}
    unique_footwear = {s for s in (_outfit_footwear_id(o) for o in pool) if s}
    bottom_reuse_needed = math.ceil(limit / max(1, len(unique_bottoms)))
    footwear_reuse_needed = math.ceil(limit / max(1, len(unique_footwear)))
    broad_pool_selection = limit > 6 or len(pool) > max(12, limit * 2)
    if broad_pool_selection:
        max_bottom_reuse = (
            max(3, bottom_reuse_needed)
            if len(unique_bottoms) > 1
            else limit
        )
        max_footwear_reuse = (
            max(3, footwear_reuse_needed)
            if len(unique_footwear) > 1
            else limit
        )
    else:
        max_bottom_reuse = 2 if len(unique_bottoms) > 1 else limit
        max_footwear_reuse = 2 if len(unique_footwear) > 1 else limit
    logger.info(
        "outfit_pipeline.diversify_caps limit=%s pool=%s unique_bottoms=%s unique_footwear=%s max_bottom_reuse=%s max_footwear_reuse=%s",
        limit,
        len(pool),
        len(unique_bottoms),
        len(unique_footwear),
        max_bottom_reuse,
        max_footwear_reuse,
    )

    def _count(role_fn, value: str) -> int:
        if not value:
            return 0
        return sum(1 for selected_outfit in selected if role_fn(selected_outfit) == value)

    def _can_add(outfit: Dict[str, Any], *, strict: bool) -> bool:
        sig = _outfit_signature(outfit)
        if not sig or sig in seen_signatures:
            return False
        bottom = _outfit_bottom_id(outfit)
        footwear = _outfit_footwear_id(outfit)
        if bottom and _count(_outfit_bottom_id, bottom) >= max_bottom_reuse:
            return False
        if footwear and _count(_outfit_footwear_id, footwear) >= max_footwear_reuse:
            return False
        if strict:
            hero = _outfit_hero_id(outfit)
            if len(selected) < min(3, limit) and hero and hero in seen_heroes:
                return False
        return True

    def _add(outfit: Dict[str, Any]) -> None:
        selected.append(outfit)
        sig = _outfit_signature(outfit)
        if sig:
            seen_signatures.add(sig)
        hero = _outfit_hero_id(outfit)
        if hero:
            seen_heroes.add(hero)

    # Pass 1: preserve the best ranked look for each available bottom first.
    for bottom in unique_bottoms:
        if len(selected) >= limit:
            break
        candidate = next((o for o in pool if _outfit_bottom_id(o) == bottom and _can_add(o, strict=True)), None)
        if candidate is None:
            candidate = next((o for o in pool if _outfit_bottom_id(o) == bottom and _can_add(o, strict=False)), None)
        if candidate is not None:
            _add(candidate)

    # Pass 2: preserve footwear directions before filling with near-duplicates.
    for footwear in unique_footwear:
        if len(selected) >= limit:
            break
        candidate = next((o for o in pool if _outfit_footwear_id(o) == footwear and _can_add(o, strict=True)), None)
        if candidate is None:
            candidate = next((o for o in pool if _outfit_footwear_id(o) == footwear and _can_add(o, strict=False)), None)
        if candidate is not None:
            _add(candidate)

    # Pass 3: one outfit per unique hero.
    for outfit in pool:
        if len(selected) >= limit:
            break
        hero = _outfit_hero_id(outfit)
        if hero and hero in seen_heroes:
            continue
        if _can_add(outfit, strict=True):
            _add(outfit)

    # Pass 4: fill remaining slots, still respecting bottom/footwear caps.
    for outfit in pool:
        if len(selected) >= limit:
            break
        if _can_add(outfit, strict=False):
            _add(outfit)

    logger.info(
        "outfit_pipeline.diversify_result limit=%s selected=%s",
        limit,
        len(selected),
    )
    return selected


def save_feedback(
    user_id: str, outfit: Dict[str, Any], feedback: str
) -> Dict[str, Any]:
    feedback_value = str(feedback).strip().lower()
    if feedback_value not in ("up", "down"):
        raise ValueError("feedback must be 'up' or 'down'")

    with _MEMORY_LOCK:
        user_memory = _load_user_memory(user_id)
        record = deepcopy(outfit)
        record["feedback"] = feedback_value
        record["saved_at"] = _utcnow_iso()

        if feedback_value == "up":
            user_memory["liked_outfits"] = [record] + user_memory.get(
                "liked_outfits", []
            )
            user_memory["liked_outfits"] = user_memory["liked_outfits"][:100]
        else:
            user_memory["disliked_outfits"] = [record] + user_memory.get(
                "disliked_outfits", []
            )
            user_memory["disliked_outfits"] = user_memory["disliked_outfits"][:100]

        _save_user_memory(user_id, user_memory)
        _index_outfit_vector(user_id=user_id, outfit=record, label=feedback_value)

    outfit_ranker.learn_from_feedback(
        user_id=user_id, features=outfit.get("ml_features", {}), feedback=feedback_value
    )
    return {"ok": True, "feedback": feedback_value}


def get_daily_outfits(user: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(user.get("user_id") or user.get("userId") or "anonymous")
    context = user.get("context", {}) or {}
    style_dna = context.get("style_dna", {}) or {}
    raw_wardrobe = user.get("wardrobe", {}) or {}

    def _bounded_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    raw_candidate_target = _bounded_int(
        context.get("raw_candidate_target") or context.get("candidate_pool_size"),
        48,
    )
    raw_candidate_target = max(24, min(raw_candidate_target, 80))
    requested_board_count = _bounded_int(context.get("requested_board_count"), 6)
    requested_board_count = max(1, min(requested_board_count, 6))
    combo_stage_cap = raw_candidate_target
    color_stage_cap = max(24, raw_candidate_target // 2)
    pattern_stage_cap = max(24, raw_candidate_target // 2)
    filter_cap = max(24, raw_candidate_target // 2)
    widen_cap = raw_candidate_target

    # Wardrobe can arrive as a dict wrapper from various callers; normalize early to avoid silent empty pipelines.
    if isinstance(raw_wardrobe, dict):
        for key in ("items", "documents", "wardrobe", "data"):
            inner = raw_wardrobe.get(key)
            if isinstance(inner, list):
                raw_wardrobe = inner
                break

    normalized = _normalize_wardrobe(raw_wardrobe)
    semantic_items, semantic_map = _semantic_retrieval(user_id=user_id, context=context)
    wardrobe = _merge_wardrobe(normalized, semantic_items)

    # Identity-safe filter (formerly GENDER PATCH V2 wrapper). Block
    # female-only items (saree, sports bra, dress, etc.) from male users
    # before any combo work runs. wardrobe is Dict[slot, List[item]] coming
    # out of _merge_wardrobe — filter inside each slot list, never flatten.
    try:
        if isinstance(wardrobe, dict):
            wardrobe = {
                slot: [
                    item
                    for item in items
                    if isinstance(item, dict)
                    and _ahvi_pipe_item_allowed(item, context)
                    and is_style_eligible(item, context.get("occasion"))
                ]
                for slot, items in wardrobe.items()
                if isinstance(items, list)
            }
        elif isinstance(wardrobe, list):
            wardrobe = [
                item
                for item in wardrobe
                if isinstance(item, dict)
                and _ahvi_pipe_item_allowed(item, context)
                and is_style_eligible(item, context.get("occasion"))
            ]
    except NameError:
        # Helpers loaded later in module — only matters at hot-reload edge cases.
        pass

    # AHVI Style Orchestrator: drop wardrobe items the agent flagged as
    # contextually wrong (e.g. boxers/slides for a client meeting). This is a
    # lightweight, additive filter that runs after the legacy gender/style
    # eligibility filters.
    try:
        wardrobe = _ahvi_apply_agent_avoid_filter(wardrobe, context)
    except Exception:
        logger.warning("ahvi.agent.avoid_filter_outer_failed", exc_info=True)

    occasion = _normalize_pipeline_occasion(context.get("occasion"), context)
    if occasion:
        context = dict(context)
        context["occasion"] = occasion
    closest_requested = (
        str(context.get("style_action") or "").strip().lower()
        in {"show_closest_option", "closest_option", "show_closest"}
        or bool(context.get("show_closest_option"))
        or bool(context.get("allow_closest_option"))
        or bool(context.get("closest"))
    )
    if closest_requested:
        context = dict(context)
        context["style_action"] = "show_closest_option"
        context["show_closest_option"] = True
        context["allow_closest_option"] = True
        context["closest"] = True

    if not occasion:
        return {
            "intent": "daily_outfit",
            "context": "Need occasion clarification before styling.",
            "outfits": [],
            "cards": [],
            "boards": [],
            "normalized_wardrobe": wardrobe,
            "pipeline": {
                "stages": [
                    "clarifying_questions",
                    "occasion_intent_capture",
                ]
            },
            "clarifying_questions": [
                "Which occasion is this for?",
                "Do you want an outfit around a top or a dress?",
            ],
        }

    occasion_filtered = _occasion_filter(wardrobe, occasion)
    try:
        logging.getLogger("ahvi.outfit_pipeline").info(
            "ahvi.combo.slot_counts occasion=%s tops=%s bottoms=%s shoes=%s dresses=%s outerwear=%s accessories=%s",
            occasion,
            len(occasion_filtered.get("tops", []) or []),
            len(occasion_filtered.get("bottoms", []) or []),
            len(occasion_filtered.get("shoes", []) or []),
            len(occasion_filtered.get("dresses", []) or []),
            len(occasion_filtered.get("outerwear", []) or []),
            len(occasion_filtered.get("accessories", []) or []),
        )
    except Exception:
        pass
    master_limit = max(12, min(18, raw_candidate_target // 3))
    master_candidates = _pick_master_candidates(
        occasion_filtered,
        occasion,
        semantic_map,
        limit=master_limit,
    )
    try:
        _diag_picked = [
            {
                "type": t,
                "name": str(item.get("name") or item.get("label") or "?")[:30],
                "tags": (item.get("occasions") or item.get("occasion_tags") or [])[:5],
                "score": _master_piece_score(item, occasion, semantic_map),
            }
            for t, item in master_candidates
        ]
        logging.getLogger("ahvi.outfit_pipeline").info(
            "outfit_pipeline.picked_masters user=%s occ=%s picks=%s",
            user_id,
            occasion,
            _diag_picked,
        )
    except Exception:
        pass
    if not master_candidates:
        return {
            "intent": "daily_outfit",
            "context": f"No suitable top or dress found for occasion '{occasion}'.",
            "outfits": [],
            "cards": [],
            "boards": [],
            "normalized_wardrobe": wardrobe,
            "pipeline": {
                "stages": [
                    "occasion_intent_capture",
                    "occasion_wardrobe_filter",
                    "master_piece_selection",
                ]
            },
        }

    master_type, master_piece = master_candidates[0]

    # Two-phase combo pool build (see _build_diverse_combo_pool).
    # Bumped per_master_min 6→8 and per-master batch 12→20 below so
    # phase-2 round-robin has enough overflow to keep ≥4 unique heroes
    # in the final brief pool after quality-guard rejections.
    per_master_min = max(8, min(12, raw_candidate_target // max(1, master_limit)))
    global_max = raw_candidate_target
    unique_hero_target = 4

    def _hero_label(piece: Dict[str, Any]) -> str:
        if not isinstance(piece, dict):
            return "?"
        return str(piece.get("name") or piece.get("label") or "?")[:25]

    master_to_combos: List[Tuple[str, List[Dict[str, Any]]]] = []
    for candidate_type, candidate_piece in master_candidates:
        candidate_combinations = _build_master_combos(
            occasion_filtered,
            candidate_type,
            candidate_piece,
            # Generous batch per master so phase 2 has overflow to
            # round-robin from; the global cap below still applies.
            max_combos=max(20, raw_candidate_target),
        )
        master_to_combos.append((_hero_label(candidate_piece), candidate_combinations))

    combinations, master_combo_counts = _build_diverse_combo_pool(
        master_to_combos,
        per_master_min=per_master_min,
        global_max=global_max,
    )

    try:
        _pool_logger = logging.getLogger("ahvi.outfit_pipeline")
        _pool_logger.info(
            "outfit_pipeline.master_combo_counts occ=%s counts=%s",
            (context.get("occasion") or context.get("intent") or occasion or ""),
            master_combo_counts,
        )
        _pool_hero_counts: Dict[str, int] = {}
        for combo in combinations:
            t = (combo.get("top") or combo.get("dress") or {}) if isinstance(combo, dict) else {}
            _pool_hero_counts[_hero_label(t)] = _pool_hero_counts.get(_hero_label(t), 0) + 1
        _pool_logger.info(
            "outfit_pipeline.combo_pool heroes=%s total=%d unique_heroes=%d target=%d",
            _pool_hero_counts,
            len(combinations),
            len(_pool_hero_counts),
            unique_hero_target,
        )
    except Exception:
        pass

    def _hero_names(items):
        names = []
        for o in items or []:
            t = (o.get("top") or o.get("dress") or {}) if isinstance(o, dict) else {}
            n = str(t.get("name") or t.get("label") or "?")[:25]
            names.append(n)
        from collections import Counter

        return dict(Counter(names))

    logging.getLogger("ahvi.outfit_pipeline").info(
        "outfit_pipeline.diversity_trace user=%s stage=combinations heroes=%s",
        user_id,
        _hero_names(combinations),
    )

    if not combinations:
        msg = (
            "Need bottoms + footwear for top-based styling."
            if master_type == "top"
            else "Need footwear for dress-based styling."
        )
        return {
            "intent": "daily_outfit",
            "context": msg,
            "outfits": [],
            "cards": [],
            "boards": [],
            "normalized_wardrobe": wardrobe,
            "pipeline": {
                "stages": [
                    "occasion_intent_capture",
                    "occasion_wardrobe_filter",
                    "master_piece_selection",
                    "combo_construction",
                ]
            },
        }

    # Run color/pattern filters PER HERO GROUP so the LLM can't collapse
    # every output to a single hero. Old behavior: filter all 40 combos in
    # one shot -> LLM picks 'best 8' globally -> all from one hero ->
    # downstream cards all share the same shirt.
    def _group_by_hero(combos):
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for combo in combos:
            hero = combo.get("top") or combo.get("dress") or {}
            hero_id = str(
                hero.get("id")
                or hero.get("$id")
                or hero.get("name")
                or "?"
            ).strip()
            groups.setdefault(hero_id, []).append(combo)
        return groups

    def _filter_stage(
        stage_name: str,
        source_combos: List[Dict[str, Any]],
        stage_cap: int,
    ):
        kept_ids: List[str] = []
        groups = _group_by_hero(source_combos)
        # Decide a per-hero cap so the merged set is bounded but each hero
        # is represented. The target is now context-driven so we can score a
        # broad pool and only slice after guard/diversity.
        n_heroes = max(1, len(groups))
        per_hero_cap = max(2, min(8, (stage_cap + n_heroes - 1) // n_heroes))

        group_list = [g for g in groups.values() if g]

        # The per-hero LLM filter routes through the slow "styling" policy
        # (pro model, 45s timeout) and in practice returns Python/prose, not
        # JSON, so it parse-fails and we fall back to the deterministic rules
        # anyway — paying ~55s/request for a discarded result. Default OFF:
        # skip straight to the rules (byte-identical boards, ~55s faster).
        # Flip OUTFIT_LLM_COMBO_FILTER=true to restore the LLM judgment.
        _use_llm_filter = str(
            os.getenv("OUTFIT_LLM_COMBO_FILTER", "false")
        ).strip().lower() in {"1", "true", "yes", "on"}

        def _filter_one_group(group_combos: List[Dict[str, Any]]) -> List[str]:
            hero_master = (
                group_combos[0].get("top") or group_combos[0].get("dress") or {}
            )
            hero_master_type = "dress" if group_combos[0].get("dress") else "top"
            ids = (
                _llm_filter_combo_ids(
                    occasion=occasion,
                    stage=stage_name,
                    master_type=hero_master_type,
                    master_piece=hero_master,
                    combos=group_combos,
                    max_ids=filter_cap,
                )
                if _use_llm_filter
                else []
            )
            if not ids:
                if stage_name == "color_combo":
                    ids = _rule_color_fallback(hero_master, group_combos, max_ids=filter_cap)
                else:
                    ids = _rule_pattern_fallback(group_combos, max_ids=filter_cap)
            if not ids:
                ids = [str(c.get("combo_id")) for c in group_combos[:per_hero_cap]]
            else:
                ids = ids[:per_hero_cap]
            return ids

        # The per-hero LLM filter calls are independent and I/O-bound (Gemini).
        # Running them sequentially was the dominant board-latency cost
        # (~3s x hero_count x stage). Fan them out and preserve hero order.
        if len(group_list) <= 1:
            for group_combos in group_list:
                kept_ids.extend(_filter_one_group(group_combos))
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(8, len(group_list))) as _ex:
                for ids in _ex.map(_filter_one_group, group_list):
                    kept_ids.extend(ids)
        return kept_ids

    color_source = combinations[:combo_stage_cap]
    color_keep = _filter_stage("color_combo", color_source, color_stage_cap)
    color_filtered = [
        c for c in color_source if str(c.get("combo_id")) in set(color_keep)
    ] or color_source[:color_stage_cap]

    pattern_keep = _filter_stage("pattern_combo", color_filtered, pattern_stage_cap)
    pattern_filtered = [
        c for c in color_filtered if str(c.get("combo_id")) in set(pattern_keep)
    ] or color_filtered[:pattern_stage_cap]

    logging.getLogger("ahvi.outfit_pipeline").info(
        "outfit_pipeline.diversity_trace user=%s stage=color_filter heroes=%s",
        user_id,
        _hero_names(color_filtered),
    )
    logging.getLogger("ahvi.outfit_pipeline").info(
        "outfit_pipeline.diversity_trace user=%s stage=pattern_filter heroes=%s",
        user_id,
        _hero_names(pattern_filtered),
    )

    candidate_combos = pattern_filtered[:raw_candidate_target]
    min_candidate_floor = min(24, raw_candidate_target)
    if len(candidate_combos) < min_candidate_floor and len(combinations) >= min_candidate_floor:
        widened: List[Dict[str, Any]] = []
        seen_combo_ids: set[str] = set()
        for source in (pattern_filtered, color_filtered, combinations):
            for combo in source:
                combo_id = str(combo.get("combo_id", "")).strip()
                if not combo_id or combo_id in seen_combo_ids:
                    continue
                seen_combo_ids.add(combo_id)
                widened.append(combo)
                if len(widened) >= widen_cap:
                    break
            if len(widened) >= widen_cap:
                break
        if widened:
            candidate_combos = widened

    logging.getLogger("ahvi.outfit_pipeline").info(
        "ahvi.combo.funnel occasion=%s masters=%s generated=%s color_filtered=%s pattern_filtered=%s candidate_combos=%s raw_target=%s requested=%s",
        occasion,
        len(master_candidates),
        len(combinations),
        len(color_filtered),
        len(pattern_filtered),
        len(candidate_combos),
        raw_candidate_target,
        requested_board_count,
    )

    # Variant seed: repeated asks should not always show the exact same first look.
    query_hint = str(context.get("query") or context.get("prompt") or occasion or "")
    time_bucket = int(datetime.now(timezone.utc).timestamp() // 300)
    offset = _stable_offset(
        f"{user_id}:{query_hint}:{time_bucket}", len(candidate_combos)
    )
    candidate_combos = _rotate(candidate_combos, offset)

    merged_context = dict(context)
    merged_context["style_dna"] = style_dna
    merged_context["style_graph"] = style_graph_engine.build_graph(wardrobe)
    rules = style_engine.get_scoring_rules(style_dna, merged_context)

    with _MEMORY_LOCK:
        user_memory = _load_user_memory(user_id)

        scored = []
        for combo in candidate_combos:
            combo["accessories"] = _select_accessories(
                occasion_filtered, combo, limit=4
            )
            scored_combo = score_outfit(
                combo, merged_context, user_memory, rules, semantic_map
            )
            # Preserve THIS combo's master, do not overwrite with the
            # outer-loop master from master_candidates[0]. Without this guard
            # every card showed the same hero because _flatten_outfit_items
            # surfaces master_piece before top.
            if not scored_combo.get("master_type"):
                scored_combo["master_type"] = combo.get("master_type") or master_type
            if not scored_combo.get("master_piece"):
                scored_combo["master_piece"] = (
                    combo.get("master_piece") or master_piece
                )
            scored_combo["pipeline_tags"] = [
                "occasion_filtered",
                "master_piece",
                "llm_color",
                "llm_pattern",
                "accessories",
            ]
            scored_combo["items"] = _flatten_outfit_items(scored_combo)
            _attach_score_meta(scored_combo, merged_context)
            scored.append(scored_combo)

        logging.getLogger("ahvi.outfit_pipeline").info(
            "ahvi.score.funnel candidate_combos=%s scored=%s top_n=%s",
            len(candidate_combos),
            len(scored),
            min(raw_candidate_target, len(scored)),
        )
        ranked = outfit_ranker.rank(
            user_id=user_id, outfits=scored, top_n=min(raw_candidate_target, len(scored))
        )
        logging.getLogger("ahvi.outfit_pipeline").info(
            "ahvi.rank.funnel ranked=%s",
            len(ranked or []),
        )
        logging.getLogger("ahvi.outfit_pipeline").info(
            "outfit_pipeline.diversity_trace user=%s stage=ranked heroes=%s",
            user_id,
            _hero_names(ranked),
        )

        # Closed-loop (lightweight): if a refinement is requested (or suggested proactively),
        # run a deterministic refinement pass and re-score using the UnifiedStyleScorer.
        refine_mode = (
            str(
                context.get("refinement")
                or (context.get("signals") or {}).get("default_refinement")
                or (context.get("signals") or {}).get("auto_refinement")
                or ""
            )
            .strip()
            .lower()
        )
        if refine_mode:
            merged_context["refinement"] = refine_mode
            merged_context["wardrobe"] = wardrobe
            try:
                refined = (
                    refinement_engine.apply(outfits=ranked, context=merged_context)
                    or ranked
                )
            except Exception:
                refined = ranked

            for idx, outfit in enumerate(ranked):
                refined_items = []
                if (
                    isinstance(refined, list)
                    and idx < len(refined)
                    and isinstance(refined[idx], dict)
                ):
                    refined_items = (
                        refined[idx].get("items")
                        if isinstance(refined[idx].get("items"), list)
                        else []
                    )
                if refined_items:
                    outfit["refined_items"] = refined_items
                    outfit["refinement_mode"] = refine_mode
                    outfit["unified_style_refined"] = _unified_style_snapshot(
                        refined_items, merged_context
                    )
                    outfit["score_meta_refined"] = outfit["unified_style_refined"]

        # Closed-loop fix (weakness-aware): improve the best outfit by addressing the weakest dimension
        # and re-scoring. Keeps the system from being "one-shot".
        if ranked and bool(
            os.getenv("ENABLE_CLOSED_LOOP_FIX", "true").lower()
            in ("1", "true", "yes", "on")
        ):
            try:
                best0 = ranked[0]
                improved = _closed_loop_fix(
                    best0,
                    context=merged_context,
                    wardrobe=wardrobe,
                    user_memory=user_memory,
                    rules=rules,
                    semantic_map=semantic_map,
                )
                if isinstance(improved, dict) and improved:
                    improved["items"] = _flatten_outfit_items(improved)
                    _attach_score_meta(improved, merged_context)
                    ranked[0] = improved
            except Exception:
                pass

        # Phase 4 plan: re-score after refinement and re-sort deterministically.
        for outfit in ranked:
            _attach_score_meta(outfit, merged_context)

        ranked.sort(
            key=lambda o: (
                float(
                    _dict(o.get("score_meta") or o.get("unified_style")).get("score")
                    or 0.0
                ),
                float(o.get("rank_score", o.get("score", 0.0)) or 0.0),
            ),
            reverse=True,
        )

        ranked = _diversify_outfits(ranked, limit=min(raw_candidate_target, len(ranked)))
        logging.getLogger("ahvi.outfit_pipeline").info(
            "outfit_pipeline.diversity_trace user=%s stage=diversified heroes=%s",
            user_id,
            _hero_names(ranked),
        )

        # AHVI editorial quality guard: remove weak/bad combinations before memory, indexing and card rendering.
        pre_guard_ranked = list(ranked or [])
        _guard_occ = (
            merged_context.get("occasion")
            or merged_context.get("intent")
            or locals().get("occasion")
            or ""
        )
        try:
            ranked = filter_and_guard_outfits(
                ranked,
                user_profile=(
                    merged_context.get("user_profile")
                    or merged_context.get("profile")
                    or merged_context.get("user")
                    or {}
                ),
                intent=_guard_occ,
                query=(
                    merged_context.get("user_query")
                    or merged_context.get("query")
                    or locals().get("user_query")
                    or locals().get("query")
                    or ""
                ),
            )
        except Exception as e:
            logging.getLogger(__name__).warning("outfit_quality_guard_failed: %s", e)

        # Diagnostic: surface how the quality guard reshaped the candidate
        # pool. Read-only, never changes behavior.
        try:
            _guard_logger = logging.getLogger("ahvi.outfit_pipeline")
            _guard_logger.info(
                "outfit_pipeline.guard_effect user=%s occ=%s before=%d after=%d",
                user_id,
                _guard_occ,
                len(pre_guard_ranked),
                len(ranked or []),
            )
            _guard_logger.info(
                "outfit_pipeline.guard_heroes user=%s occ=%s before=%s after=%s",
                user_id,
                _guard_occ,
                _hero_names(pre_guard_ranked),
                _hero_names(ranked or []),
            )
        except Exception:
            pass
        if closest_requested and not ranked and pre_guard_ranked:
            closest_outfit = deepcopy(pre_guard_ranked[0])
            closest_outfit.setdefault("score_meta", {})
            closest_score_meta = closest_outfit["score_meta"]
            closest_outfit["title"] = "Closest wardrobe option"
            closest_outfit["badge"] = "CLOSEST OPTION"
            closest_outfit["occasion_label"] = "CLOSEST OPTION"
            closest_score_meta.update(
                {
                    "closest_option": True,
                    "weak_match": True,
                    "needs_refinement": True,
                    "occasion_reject_overridden": True,
                }
            )
            ranked = [closest_outfit]
            logging.getLogger("ahvi.outfit_pipeline").info(
                "style_closest_option_from_rejected user_id=%s occasion=%s score=%s reasons=%s",
                user_id,
                occasion,
                closest_score_meta.get("occasion_compatibility_score")
                or closest_outfit.get("score"),
                closest_score_meta.get("occasion_penalties")
                or closest_score_meta.get("closest_option_reason")
                or [],
            )

        if ranked:
            ranked = _select_set_diverse_outfits(
                ranked,
                limit=requested_board_count,
                occasion=str(_guard_occ or occasion or ""),
                query=str(
                    merged_context.get("user_query")
                    or merged_context.get("query")
                    or locals().get("user_query")
                    or locals().get("query")
                    or ""
                ),
            )
            logging.getLogger("ahvi.outfit_pipeline").info(
                "outfit_pipeline.diversity_trace user=%s stage=post_guard_diversified heroes=%s",
                user_id,
                _hero_names(ranked),
            )

        user_memory["recent_outfits"] = ranked + user_memory.get("recent_outfits", [])
        user_memory["recent_outfits"] = user_memory["recent_outfits"][:30]
        _save_user_memory(user_id, user_memory)

        for outfit in ranked:
            _index_outfit_vector(user_id=user_id, outfit=outfit, label="recent")

    cards = _build_cards(ranked, merged_context)

    try:
        _diag_card_summary = [
            {
                "title": str(c.get("title") or "?")[:30],
                "items": [
                    str(it.get("name") or it.get("label") or "?")[:25]
                    for it in (c.get("items") or [])[:5]
                    if isinstance(it, dict)
                ],
            }
            for c in (cards or [])[:6]
            if isinstance(c, dict)
        ]
        logging.getLogger("ahvi.outfit_pipeline").info(
            "outfit_pipeline.final_cards user=%s occ=%s cards=%s",
            user_id,
            occasion,
            _diag_card_summary,
        )
    except Exception:
        pass

    # Identity-safe filter on outfits + cards (formerly GENDER PATCH V2).
    try:
        ranked = [o for o in ranked if _ahvi_pipe_outfit_allowed(o, context)]
        cards = [c for c in cards if _ahvi_pipe_card_allowed(c, context)]
    except NameError:
        pass

    board_item_ids: List[str] = _ahvi_board_item_ids_from_cards(cards, ranked)

    result_payload = {
        "intent": "daily_outfit",
        "context": "I pulled together wardrobe-based looks that match your request, occasion, and style profile.",
        "outfits": ranked,
        "cards": cards,
        "boards": cards,
        "board_item_ids": board_item_ids,
        "normalized_wardrobe": wardrobe,
        "pipeline": {
            "stages": [
                "clarifying_questions_if_needed",
                "occasion_intent_capture",
                "occasion_wardrobe_filter",
                "master_piece_selection",
                "llm_color_combo_filter",
                "llm_pattern_combo_filter",
                "accessory_completion",
                "style_board_generation",
                "scoring_engine",
                "explanation_generation",
                "tryon_payload",
                "frontend",
            ],
            "scoring_components": [
                "occasion rules",
                "color intelligence",
                "layering",
                "style graph",
                "memory",
                "feedback",
                "ml_ranker",
            ],
        },
        "memory_summary": {
            "recent_count": len(user_memory.get("recent_outfits", [])),
            "liked_count": len(user_memory.get("liked_outfits", [])),
            "disliked_count": len(user_memory.get("disliked_outfits", [])),
        },
        "premium": {
            "outfit_storytelling": True,
            "style_dna_learning": True,
            "ml_ranking": True,
        },
    }

    # Final board ownership now lives in services.style_flow_service. Keep this
    # pipeline focused on candidate generation/scoring so post-score card
    # builders cannot swap tops/bottoms/footwear after style_scorer ranks them.

    if isinstance(result_payload, dict):
        result_payload.setdefault("meta", {})
        if isinstance(result_payload.get("meta"), dict):
            try:
                result_payload["meta"]["style_gender_guard"] = (
                    _ahvi_pipe_context_gender(context)
                )
            except NameError:
                pass

    return result_payload


def _ahvi_board_item_ids_from_cards(cards, fallback_ranked):
    ids = []

    if isinstance(cards, list) and cards and isinstance(cards[0], dict):
        source_items = cards[0].get("items") or []
        for item in source_items:
            if isinstance(item, dict):
                item_id = str(
                    item.get("id") or item.get("$id") or item.get("item_id") or ""
                ).strip()
                if item_id:
                    ids.append(item_id)

    if not ids and isinstance(fallback_ranked, list) and fallback_ranked:
        best = fallback_ranked[0] if isinstance(fallback_ranked[0], dict) else {}
        for part in (
            "top",
            "bottom",
            "dress",
            "footwear",
            "shoe",
            "shoes",
            "outerwear",
        ):
            value = best.get(part)
            if isinstance(value, dict):
                item_id = str(
                    value.get("id") or value.get("$id") or value.get("item_id") or ""
                ).strip()
                if item_id:
                    ids.append(item_id)
        for item in best.get("accessories") or []:
            if isinstance(item, dict):
                item_id = str(
                    item.get("id") or item.get("$id") or item.get("item_id") or ""
                ).strip()
                if item_id:
                    ids.append(item_id)

    deduped = []
    seen = set()
    for item_id in ids:
        if item_id not in seen:
            seen.add(item_id)
            deduped.append(item_id)
    return deduped


# ---- end AHVI style board contract fix ----


# ================= AHVI OUTFIT PIPELINE GENDER PATCH V2 BEGIN =================

_AHVI_PIPE_MALE_GENDERS = {"m", "male", "man", "men", "mens", "boy"}
_AHVI_PIPE_FEMALE_GENDERS = {
    "f",
    "female",
    "woman",
    "women",
    "womens",
    "girl",
    "ladies",
}
_AHVI_PIPE_FEMININE_ONLY = {
    "saree",
    "sari",
    "lehenga",
    "gown",
    "skirt",
    "skirts",
    "blouse",
    "kurti",
}
_AHVI_PIPE_MALE_TRADITIONAL = {"sherwani", "achkan"}
_AHVI_PIPE_EXPLICIT_FEMININE = {
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


def _ahvi_pipe_tokens(value):
    import re as _re

    return _re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip().split()


def _ahvi_pipe_gender(value):
    raw = str(value or "").strip().lower()

    if raw in _AHVI_PIPE_MALE_GENDERS:
        return "male"

    if raw in _AHVI_PIPE_FEMALE_GENDERS:
        return "female"

    if raw in {"unisex", "neutral", "genderless", "any"}:
        return "unisex"

    return ""


def _ahvi_pipe_context_gender(context):
    context = context or {}

    style_dna = (
        context.get("style_dna") if isinstance(context.get("style_dna"), dict) else {}
    )
    profile = (
        context.get("user_profile")
        if isinstance(context.get("user_profile"), dict)
        else {}
    )

    for value in (
        style_dna.get("style_gender"),
        style_dna.get("gender"),
        profile.get("style_gender"),
        profile.get("gender"),
        profile.get("preferred_gender"),
        profile.get("target_gender"),
    ):
        gender = _ahvi_pipe_gender(value)
        if gender:
            return gender

    return "unisex"


def _ahvi_pipe_query_allows_feminine(context):
    tokens = set(_ahvi_pipe_tokens((context or {}).get("query") or ""))
    return bool(tokens.intersection(_AHVI_PIPE_EXPLICIT_FEMININE))


def _ahvi_pipe_item_tokens(item):
    blob = " ".join(
        str(item.get(k, "") or "")
        for k in (
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
            "gender",
            "style_gender",
            "target_gender",
            "audience",
            "department",
            "intended_for",
            "wearer",
        )
    )
    return set(_ahvi_pipe_tokens(blob))


_AHVI_PIPE_PRIVATE_TOKENS = {
    "boxer", "boxers", "brief", "briefs", "underwear", "undergarment",
    "innerwear", "slipper", "slippers", "pajama", "pajamas", "pyjama",
    "pyjamas", "sleepwear", "nightwear", "loungewear", "lounge",
}
# Only when the user explicitly asks for a private/lounge/sleep context do we
# allow these items through.
_AHVI_PIPE_PRIVATE_REQUEST_TERMS = {
    "pajama", "pyjama", "lounge", "loungewear", "sleep", "sleepwear",
    "nightwear", "bed", "home", "relax",
}
# A "boxer fit shirt" is a top, not underwear — never gate top slots.
_AHVI_PIPE_TOP_TOKENS = {
    "shirt", "tshirt", "tee", "top", "tops", "blouse", "kurta", "polo",
    "sweater", "hoodie",
}


def _ahvi_pipe_private_wear_blocked(item, context):
    """Deterministic occasion safety: block private/lounge/sleepwear bottoms
    and footwear unless the user's query explicitly asks for that context.
    Applies regardless of gender. Never gates top/shirt slots."""
    if not isinstance(item, dict):
        return False
    tokens = set(_ahvi_pipe_item_tokens(item))
    if not tokens.intersection(_AHVI_PIPE_PRIVATE_TOKENS):
        return False
    role = str(item.get("type") or item.get("role") or item.get("category") or "").lower()
    if role in {"top", "tops", "shirt", "shirts"} or tokens.intersection(_AHVI_PIPE_TOP_TOKENS):
        return False
    query = str((context or {}).get("query") or "").lower()
    if any(term in query for term in _AHVI_PIPE_PRIVATE_REQUEST_TERMS):
        return False
    return True


def _ahvi_pipe_item_allowed(item, context):
    if not isinstance(item, dict):
        return False

    # Occasion-aware private/lounge safety runs first, for every gender.
    if _ahvi_pipe_private_wear_blocked(item, context):
        return False

    if _ahvi_pipe_context_gender(context) != "male":
        return True

    if _ahvi_pipe_query_allows_feminine(context):
        return True

    tokens = _ahvi_pipe_item_tokens(item)

    audience = set(
        _ahvi_pipe_tokens(
            " ".join(
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
        )
    )

    if audience.intersection(_AHVI_PIPE_FEMALE_GENDERS):
        return False

    if tokens.intersection(_AHVI_PIPE_FEMININE_ONLY):
        return False

    if tokens.intersection({"dress", "dresses"}) and not tokens.intersection(
        _AHVI_PIPE_MALE_TRADITIONAL
    ):
        return False

    return True


def _ahvi_pipe_items_from_outfit(outfit):
    items = []

    if not isinstance(outfit, dict):
        return items

    for key in ("top", "bottom", "dress", "shoes", "footwear", "outerwear"):
        value = outfit.get(key)
        if isinstance(value, dict):
            items.append(value)

    for key in ("items", "refined_items", "accessories"):
        value = outfit.get(key)
        if isinstance(value, list):
            items.extend([x for x in value if isinstance(x, dict)])

    return items


def _ahvi_pipe_outfit_allowed(outfit, context):
    return all(
        _ahvi_pipe_item_allowed(item, context)
        for item in _ahvi_pipe_items_from_outfit(outfit)
    )


def _ahvi_pipe_card_allowed(card, context):
    if not isinstance(card, dict):
        return False

    items = []

    for key in ("items", "accessories"):
        value = card.get(key)
        if isinstance(value, list):
            items.extend([x for x in value if isinstance(x, dict)])

    return all(_ahvi_pipe_item_allowed(item, context) for item in items)


# ================= AHVI OUTFIT PIPELINE GENDER PATCH V2 END =================
# Wrapper rebinding removed — gender filter is now applied directly inside
# the canonical get_daily_outfits() (see calls to _ahvi_pipe_item_allowed /
# _ahvi_pipe_outfit_allowed / _ahvi_pipe_card_allowed). The helpers above
# remain because the canonical function imports them by name.


# ================= AHVI MORE LOOKS V2 BEGIN =================
# Backend style pipeline now allows up to 6 diversified final cards.
AHVI_MORE_LOOKS_V2_ENABLED = True
# ================= AHVI MORE LOOKS V2 END =================
