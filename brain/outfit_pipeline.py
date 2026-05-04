import hashlib
import json
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
from brain.engines.style_scorer import style_scorer
from brain.engines.refinement_engine import refinement_engine
from brain.engines.wardrobe_selector import wardrobe_selector
from brain.engines.styling.palette_engine import palette_engine
from services import ai_gateway
from services.appwrite_proxy import AppwriteProxy
from services.embedding_service import get_model
from services.qdrant_service import qdrant_service
from brain.engines.outfit_quality_guard import filter_and_guard_outfits
import re


# ---- AHVI demo fix: normalize Appwrite wardrobe records into outfit slots ----
def _ahvi_tokens(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip().split()


def _ahvi_slot_for_item(item):
    if not isinstance(item, dict):
        return "accessory"

    raw = " ".join(str(item.get(k, "") or "") for k in (
        "slot", "type", "category", "cat", "category_group",
        "sub_category", "subcategory", "subCategory",
        "name", "label", "description"
    ))
    tokens = _ahvi_tokens(raw)

    # Top first so "Short-Sleeved Shirt" never becomes shorts/bottom.
    if any(t in tokens for t in [
        "top", "tops", "shirt", "shirts", "tee", "tshirt", "tshirts",
        "blouse", "hoodie", "sweater", "kurta", "polo"
    ]):
        return "top"

    # Only shorts, never short.
    if any(t in tokens for t in [
        "bottom", "bottoms", "pant", "pants", "trouser", "trousers",
        "jean", "jeans", "shorts", "skirt", "skirts", "chino", "chinos"
    ]):
        return "bottom"

    if any(t in tokens for t in [
        "footwear", "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
        "heel", "heels", "sandal", "sandals", "loafer", "loafers"
    ]):
        return "footwear"

    if any(t in tokens for t in [
        "accessory", "accessories", "watch", "bag", "belt", "jewelry",
        "jewellery", "ring", "necklace", "bracelet", "earring", "hat", "cap"
    ]):
        return "accessory"

    if any(t in tokens for t in ["jacket", "coat", "blazer", "outerwear", "cardigan"]):
        return "outerwear"

    if any(t in tokens for t in ["dress", "dresses", "gown", "jumpsuit", "saree", "lehenga"]):
        return "dress"

    return "accessory"


def _ahvi_image_for_item(item):
    if not isinstance(item, dict):
        return ""
    return (
        item.get("masked_url")
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
        patched.setdefault("label", item.get("name") or item.get("label") or item.get("category") or slot)

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


def _has_any(parts: List[str], words: List[str]) -> bool:
    return any(word in parts for word in words)


def _infer_category(item: Dict[str, Any]) -> str:
    """
    Canonical category inference.

    Critical demo cases:
    - White Short-Sleeved Shirt -> Tops
    - Khaki Shorts -> Bottoms
    - Brown Boots -> Footwear
    - Watch -> Accessories
    """
    if not isinstance(item, dict):
        return "Accessories"

    explicit = str(item.get("category") or item.get("cat") or _ahvi_slot_for_item(item) or "").strip().lower()
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
    parts = _tokens(joined)

    # Tops first so "Short-Sleeved Shirt" never becomes Bottoms.
    if _has_any(parts, [
        "shirt", "shirts", "tee", "tshirt", "tshirts", "top", "tops",
        "blouse", "blouses", "hoodie", "hoodies", "sweater", "sweaters",
        "kurta", "kurtas", "polo", "polos",
    ]):
        return "Tops"

    # Only "shorts", never "short".
    if _has_any(parts, [
        "pants", "pant", "trousers", "trouser", "jeans", "jean",
        "shorts", "skirt", "skirts", "legging", "leggings", "chino", "chinos",
    ]):
        return "Bottoms"

    if _has_any(parts, [
        "shoe", "shoes", "boot", "boots", "sneaker", "sneakers",
        "heel", "heels", "sandal", "sandals", "loafer", "loafers",
        "slipper", "slippers",
    ]):
        return "Footwear"

    if _has_any(parts, [
        "watch", "watches", "bag", "bags", "belt", "belts",
        "scarf", "scarves", "jewelry", "jewellery", "ring", "rings",
        "necklace", "bracelet", "earring", "earrings", "accessory",
        "accessories", "hat", "cap", "sunglass", "sunglasses",
    ]):
        return "Accessories"

    if _has_any(parts, ["jacket", "coat", "blazer", "outerwear", "cardigan", "overshirt"]):
        return "Outerwear"

    if _has_any(parts, ["dress", "dresses", "gown", "jumpsuit", "saree", "lehenga", "sherwani"]):
        return "Dresses"

    return "Accessories"


def _bucket_for_category(category: str) -> str:
    return {
        "Tops": "tops",
        "Bottoms": "bottoms",
        "Footwear": "shoes",
        "Accessories": "accessories",
        "Outerwear": "outerwear",
        "Dresses": "dresses",
    }.get(str(category or ""), "accessories")


def _type_for_category(category: str) -> str:
    return {
        "Tops": "top",
        "Bottoms": "bottom",
        "Footwear": "shoes",
        "Accessories": "accessory",
        "Outerwear": "outerwear",
        "Dresses": "dress",
    }.get(str(category or ""), "accessory")


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
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(user_id or "anonymous"))
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
        "id": str(item.get("id") or item.get("$id") or item.get("item_id") or item.get("image_id") or item.get("name") or ""),
        "name": str(item.get("name") or item.get("label") or item_type.title()),
        "type": item_type,
        "category": category_name,
        "sub_category": str(item.get("sub_category") or item.get("subcategory") or item.get("label") or item_type).strip(),
        "color": str(item.get("color") or item.get("color_name") or item.get("color_code") or "").lower(),
        "image_url": str(item.get("image_url") or item.get("raw_image_url") or item.get("raw_url") or item.get("imageUrl") or "").strip(),
        "masked_url": str(item.get("masked_url") or item.get("masked_image_url") or item.get("sticker_url") or item.get("maskedUrl") or "").strip(),
        "fabric": str(item.get("fabric") or item.get("pattern") or "").lower(),
        "style": str(item.get("style") or item.get("vibe") or "").lower(),
        "occasion_tags": [str(v).strip().lower() for v in raw_tags if str(v or "").strip()],
        "weather_tags": [str(v).strip().lower() for v in raw_weather if str(v or "").strip()],
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
        if forced_category and not enriched.get("category") and not enriched.get("type"):
            enriched["category"] = forced_category

        category_name = _infer_category(enriched)
        bucket = _bucket_for_category(category_name)
        parts[bucket].append(_normalize_item(enriched, _type_for_category(category_name)))

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
        for item in raw_wardrobe.get("accessories", raw_wardrobe.get("jewelry", [])) or []:
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
        qdrant_service.upsert_memory_vector(point_id=point_id, vector=vector, payload=payload)
    except Exception:
        return


def _semantic_retrieval(
    user_id: str,
    context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if not qdrant_service.enabled():
        return [], {}

    try:
        query_text = " ".join(
            [
                str(context.get("query", "")),
                str(context.get("occasion", "")),
                str(context.get("weather", "")),
                str(context.get("time_of_day", "")),
                str((context.get("style_dna", {}) or {}).get("style", "")),
                " ".join((context.get("style_dna", {}) or {}).get("preferred_colors", [])),
            ]
        ).strip()
        if not query_text:
            query_text = "daily outfit"

        model = get_model()
        query_vector = model.encode(query_text).tolist()
        results = qdrant_service.semantic_retrieve(query_vector, user_id=user_id, limit=40)

        wardrobe_items: List[Dict[str, Any]] = []
        semantic_map: Dict[str, float] = {}
        for row in results:
            payload = row.get("payload", {}) if isinstance(row, dict) else {}
            normalized = _normalize_item(payload, str(payload.get("category", "item")))
            item_id = normalized.get("id")
            if not item_id:
                continue
            semantic_map[item_id] = max(float(semantic_map.get(item_id, 0.0)), float(row.get("score", 0.0)))
            wardrobe_items.append(normalized)

        return wardrobe_items, semantic_map
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
        merged[bucket].append(_normalize_item(item, _type_for_category(category_name)))
        seen.add(item_id)

    return merged


def _occasion_filter(wardrobe: Dict[str, List[Dict[str, Any]]], occasion: str) -> Dict[str, List[Dict[str, Any]]]:
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

            normalized_tags = [str(v).strip().lower() for v in tags if str(v or "").strip()]

            if occ in normalized_tags or any(occ in tag for tag in normalized_tags):
                matched.append(item)
            else:
                rest.append(item)

        reordered[key] = matched + rest

    return reordered


def _master_piece_score(item: Dict[str, Any], occasion: str, semantic_map: Dict[str, float]) -> float:
    tags = [str(v).strip().lower() for v in (item.get("occasion_tags") or []) if str(v).strip()]
    score = 0.0
    if occasion and (occasion in tags or any(occasion in t for t in tags)):
        score += 2.0
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
        candidates.append(("top", row, _master_piece_score(row, occasion, semantic_map)))
    for row in wardrobe.get("dresses", []) or []:
        candidates.append(("dress", row, _master_piece_score(row, occasion, semantic_map) + 0.1))

    if not candidates:
        return "", {}
    candidates.sort(key=lambda x: x[2], reverse=True)
    best = candidates[0]
    return best[0], best[1]


def _pick_master_candidates(
    wardrobe: Dict[str, List[Dict[str, Any]]],
    occasion: str,
    semantic_map: Dict[str, float],
    *,
    limit: int = 6,
) -> List[Tuple[str, Dict[str, Any]]]:
    candidates: List[Tuple[str, Dict[str, Any], float]] = []
    for row in wardrobe.get("tops", []) or []:
        candidates.append(("top", row, _master_piece_score(row, occasion, semantic_map)))
    for row in wardrobe.get("dresses", []) or []:
        candidates.append(("dress", row, _master_piece_score(row, occasion, semantic_map) + 0.1))

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
        for bottom in (wardrobe.get("bottoms", []) or []):
            for shoe in (wardrobe.get("shoes", []) or []):
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
        for shoe in (wardrobe.get("shoes", []) or []):
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
) -> List[str]:
    if not combos:
        return []

    compact = []
    for row in combos:
        compact.append(
            {
                "combo_id": row.get("combo_id"),
                "palette": _combo_palette(row),
                "patterns": _combo_patterns(row),
                "bottom": (row.get("bottom") or {}).get("name"),
                "shoes": (row.get("shoes") or {}).get("name"),
                "dress": (row.get("dress") or {}).get("name"),
            }
        )

    prompt = f"""
You are a fashion combo evaluator.
Occasion: {occasion}
Stage: {stage}
Master type: {master_type}
Master piece: {json.dumps({'name': master_piece.get('name'), 'color': master_piece.get('color'), 'fabric': master_piece.get('fabric')}, ensure_ascii=True)}

Select best combo_ids from this list:
{json.dumps(compact, ensure_ascii=True)}

Return strict JSON object:
{{
  "selected_combo_ids": ["id1","id2","id3"]
}}
Rules:
- Keep at most 8 ids.
- Keep only ids present in the input.
- Prioritize practical wearable harmony.
"""
    try:
        parsed = ai_gateway.generate_json_object(prompt, signals={"context_mode": "styling"})
        selected = parsed.get("selected_combo_ids", []) if isinstance(parsed, dict) else []
        normalized = []
        valid = {str(c.get("combo_id")) for c in compact}
        for value in selected if isinstance(selected, list) else []:
            cid = str(value).strip()
            if cid and cid in valid and cid not in normalized:
                normalized.append(cid)
        return normalized[:8]
    except Exception:
        return []


def _rule_color_fallback(master_piece: Dict[str, Any], combos: List[Dict[str, Any]]) -> List[str]:
    master_color = str((master_piece or {}).get("color", "")).strip().lower()
    neutrals = {"black", "white", "beige", "gray", "grey", "navy", "brown"}
    selected: List[str] = []
    for combo in combos:
        palette = _combo_palette(combo)
        uniq = set(palette)
        if not uniq:
            continue
        if master_color and master_color in uniq:
            selected.append(str(combo.get("combo_id")))
            continue
        if len(uniq) <= 3 and any(c in neutrals for c in uniq):
            selected.append(str(combo.get("combo_id")))
    return selected[:8]


def _rule_pattern_fallback(combos: List[Dict[str, Any]]) -> List[str]:
    selected: List[str] = []
    for combo in combos:
        pats = [p for p in _combo_patterns(combo) if p]
        if not pats:
            selected.append(str(combo.get("combo_id")))
            continue
        standout = sum(1 for p in pats if p in {"striped", "checked", "floral", "printed"})
        if standout <= 1:
            selected.append(str(combo.get("combo_id")))
    return selected[:8]


def _select_accessories(wardrobe: Dict[str, List[Dict[str, Any]]], combo: Dict[str, Any], limit: int = 2) -> List[Dict[str, Any]]:
    accessories = wardrobe.get("accessories", []) or []
    if not accessories:
        return []
    palette = set(_combo_palette(combo))
    picked: List[Dict[str, Any]] = []
    for item in accessories:
        color = str(item.get("color", "")).strip().lower()
        if color and color in palette:
            picked.append(item)
        if len(picked) >= limit:
            return picked
    return accessories[:limit]


def generate_combinations(wardrobe: Dict[str, List[Dict[str, Any]]], max_candidates: int = 600) -> List[Dict[str, Any]]:
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
            "combo_id": "|".join([top.get("id", ""), bottom.get("id", ""), shoe.get("id", "")]),
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
    occasion = str(context.get("occasion", "")).lower()

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
    occasion = str(context.get("occasion", "")).lower()
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

    for part in ("master_piece", "top", "bottom", "dress", "shoes", "outerwear"):
        item = outfit.get(part, {}) or {}
        if not item:
            continue
        color = str(item.get("color", "")).lower()
        colors.append(color)

        item_id = str(item.get("id", ""))
        if item_id:
            item_ids.append(item_id)
            semantic_relevance += float(semantic_map.get(item_id, 0.0))

        weather_tags = [str(v).lower() for v in item.get("weather_tags", [])]
        occasion_tags = [str(v).lower() for v in item.get("occasion_tags", [])]

        if weather and weather in weather_tags:
            weather_score += 1.0
        if occasion and occasion in occasion_tags:
            occasion_score += 1.0

        name = str(item.get("name", "")).lower()
        fabric = str(item.get("fabric", "")).lower()
        if fabric and fabric in rules.get("preferred_fabrics", []):
            occasion_score += 0.4
        if name and name in rules.get("avoided_items", []):
            occasion_score -= 1.0

    color_intelligence = _color_score(colors, [str(c).lower() for c in style_dna.get("preferred_colors", [])])

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
        lines.append(f"Layering: {outer.get('name', 'outerwear')} adds weather-ready structure.")
    if accessories:
        lines.append(f"Accessories: {', '.join(str(x.get('name', 'accent')) for x in accessories)} complete the style board.")
    lines.append("Personalization: ranking boosted using your Style DNA, memory, and feedback signals.")
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


def _build_tryon_payload(outfit: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
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

    for part in ("master_piece", "top", "bottom", "dress", "shoes", "outerwear"):
        value = outfit.get(part, {}) or {}
        if isinstance(value, dict) and value:
            items.append(value)

    accessories = outfit.get("accessories") or []
    if isinstance(accessories, list):
        items.extend([x for x in accessories if isinstance(x, dict)])

    return items


def _unified_style_snapshot(items: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
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
    items = outfit.get("refined_items") if isinstance(outfit.get("refined_items"), list) else outfit.get("items")
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


def _swap_part(outfit: Dict[str, Any], part: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
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

    occasion = str(context.get("occasion") or "").strip().lower()
    style_dna = context.get("style_dna", {}) or {}

    palette_hexes: List[str] = []
    try:
        palette = palette_engine.select_palette(
            {"event": occasion or None, "microtheme": style_dna.get("primary_aesthetic")}
        )
        palette_hexes = [str(x).strip() for x in (palette.get("hex") or []) if str(x).strip()]
    except Exception:
        palette_hexes = []

    preferred_colors = []
    if isinstance(style_dna.get("preferred_colors"), list):
        preferred_colors.extend([str(x).strip() for x in style_dna.get("preferred_colors") if str(x).strip()])
    preferred_colors.extend(palette_hexes)

    current = dict(outfit)

    def _score(o: Dict[str, Any]) -> float:
        try:
            return float(o.get("score") or 0.0)
        except Exception:
            return 0.0

    for _ in range(2):
        breakdown = current.get("score_breakdown") if isinstance(current.get("score_breakdown"), dict) else {}
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
        weakness = "color" if color_intel < 0.5 else ("occasion" if occ_rules < 0.5 else ("layering" if layering < 0.2 else ""))
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
            target_type = "footwear" if part == "shoes" else ("dress" if part == "dress" else ("top" if part == "top" else "bottom"))

            if weakness == "layering" and part in ("top", "bottom", "dress", "shoes"):
                continue

            candidate = wardrobe_selector.find_best_match(
                target_type,
                {**context, "wardrobe": wardrobe},
                preferred_colors=preferred_colors if weakness in ("color", "occasion") else None,
                require_occasion=occasion if weakness == "occasion" else None,
            )
            if not candidate:
                continue

            swapped = _swap_part(current, part, candidate)
            swapped["items"] = _flatten_outfit_items(swapped)
            swapped["unified_style"] = _unified_style_snapshot(swapped["items"], context)

            # Re-score via the pipeline scorer (keeps consistency with ranking features).
            try:
                swapped = score_outfit(swapped, context, user_memory, rules, semantic_map)
            except Exception:
                pass

            if _score(swapped) >= _score(current) + 0.15:
                candidate_outfit = swapped
                break

        if candidate_outfit is None:
            break
        current = candidate_outfit

    return current


def _build_cards(outfits: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for idx, outfit in enumerate(outfits):
        story = _generate_story(outfit, context)
        tryon_payload = _build_tryon_payload(outfit, context)
        # Prefer refined_items when present (closed-loop refinement pass), otherwise use the flattened outfit items.
        items = outfit.get("refined_items") if isinstance(outfit.get("refined_items"), list) else None
        if items is None:
            items = _flatten_outfit_items(outfit)
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
            }
        )
    return cards



def _item_id(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("id") or item.get("$id") or item.get("image_id") or item.get("masked_id") or "").strip()


def _outfit_signature(outfit: Dict[str, Any]) -> str:
    return "|".join([
        _item_id(outfit.get("top") or {}),
        _item_id(outfit.get("bottom") or {}),
        _item_id(outfit.get("dress") or {}),
        _item_id(outfit.get("shoes") or {}),
        _item_id(outfit.get("outerwear") or {}),
    ])


def _different_enough(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (
        _item_id(a.get("top") or a.get("dress") or {}) != _item_id(b.get("top") or b.get("dress") or {}) or
        _item_id(a.get("bottom") or {}) != _item_id(b.get("bottom") or {}) or
        _item_id(a.get("shoes") or {}) != _item_id(b.get("shoes") or {})
    )


def _diversify_outfits(outfits: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for outfit in outfits or []:
        sig = _outfit_signature(outfit)
        if not sig or sig in seen:
            continue

        if not selected or any(_different_enough(existing, outfit) for existing in selected):
            selected.append(outfit)
            seen.add(sig)

        if len(selected) >= limit:
            break

    for outfit in outfits or []:
        if len(selected) >= limit:
            break

        sig = _outfit_signature(outfit)
        if sig and sig not in seen:
            selected.append(outfit)
            seen.add(sig)

    return selected


def save_feedback(user_id: str, outfit: Dict[str, Any], feedback: str) -> Dict[str, Any]:
    feedback_value = str(feedback).strip().lower()
    if feedback_value not in ("up", "down"):
        raise ValueError("feedback must be 'up' or 'down'")

    with _MEMORY_LOCK:
        user_memory = _load_user_memory(user_id)
        record = deepcopy(outfit)
        record["feedback"] = feedback_value
        record["saved_at"] = _utcnow_iso()

        if feedback_value == "up":
            user_memory["liked_outfits"] = [record] + user_memory.get("liked_outfits", [])
            user_memory["liked_outfits"] = user_memory["liked_outfits"][:100]
        else:
            user_memory["disliked_outfits"] = [record] + user_memory.get("disliked_outfits", [])
            user_memory["disliked_outfits"] = user_memory["disliked_outfits"][:100]

        _save_user_memory(user_id, user_memory)
        _index_outfit_vector(user_id=user_id, outfit=record, label=feedback_value)

    outfit_ranker.learn_from_feedback(user_id=user_id, features=outfit.get("ml_features", {}), feedback=feedback_value)
    return {"ok": True, "feedback": feedback_value}


def get_daily_outfits(user: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(user.get("user_id") or user.get("userId") or "anonymous")
    context = user.get("context", {}) or {}
    style_dna = context.get("style_dna", {}) or {}
    raw_wardrobe = user.get("wardrobe", {}) or {}

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
    occasion = str(context.get("occasion", "")).strip().lower()

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
    master_candidates = _pick_master_candidates(
        occasion_filtered,
        occasion,
        semantic_map,
        limit=6,
    )
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

    combinations: List[Dict[str, Any]] = []
    seen_combo_ids: set[str] = set()
    for candidate_type, candidate_piece in master_candidates:
        candidate_combinations = _build_master_combos(
            occasion_filtered,
            candidate_type,
            candidate_piece,
            max_combos=40,
        )
        for combo in candidate_combinations:
            combo_id = str(combo.get("combo_id", "")).strip()
            if not combo_id or combo_id in seen_combo_ids:
                continue
            seen_combo_ids.add(combo_id)
            combinations.append(combo)
            if len(combinations) >= 40:
                break
        if len(combinations) >= 40:
            break

    if not combinations:
        msg = "Need bottoms + footwear for top-based styling." if master_type == "top" else "Need footwear for dress-based styling."
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

    color_keep = _llm_filter_combo_ids(
        occasion=occasion,
        stage="color_combo",
        master_type=master_type,
        master_piece=master_piece,
        combos=combinations,
    )
    if not color_keep:
        color_keep = _rule_color_fallback(master_piece, combinations)
    color_filtered = [c for c in combinations if str(c.get("combo_id")) in set(color_keep)] or combinations[:8]

    pattern_keep = _llm_filter_combo_ids(
        occasion=occasion,
        stage="pattern_combo",
        master_type=master_type,
        master_piece=master_piece,
        combos=color_filtered,
    )
    if not pattern_keep:
        pattern_keep = _rule_pattern_fallback(color_filtered)
    pattern_filtered = [c for c in color_filtered if str(c.get("combo_id")) in set(pattern_keep)] or color_filtered[:5]

    candidate_combos = pattern_filtered
    if len(candidate_combos) < 3:
        widened: List[Dict[str, Any]] = []
        seen_combo_ids: set[str] = set()
        for source in (pattern_filtered, color_filtered, combinations):
            for combo in source:
                combo_id = str(combo.get("combo_id", "")).strip()
                if not combo_id or combo_id in seen_combo_ids:
                    continue
                seen_combo_ids.add(combo_id)
                widened.append(combo)
                if len(widened) >= 12:
                    break
            if len(widened) >= 12:
                break
        if widened:
            candidate_combos = widened

    # Variant seed: repeated asks should not always show the exact same first look.
    query_hint = str(context.get("query") or context.get("prompt") or occasion or "")
    time_bucket = int(datetime.now(timezone.utc).timestamp() // 300)
    offset = _stable_offset(f"{user_id}:{query_hint}:{time_bucket}", len(candidate_combos))
    candidate_combos = _rotate(candidate_combos, offset)

    merged_context = dict(context)
    merged_context["style_dna"] = style_dna
    merged_context["style_graph"] = style_graph_engine.build_graph(wardrobe)
    rules = style_engine.get_scoring_rules(style_dna, merged_context)

    with _MEMORY_LOCK:
        user_memory = _load_user_memory(user_id)

        scored = []
        for combo in candidate_combos:
            combo["accessories"] = _select_accessories(occasion_filtered, combo, limit=2)
            scored_combo = score_outfit(combo, merged_context, user_memory, rules, semantic_map)
            scored_combo["master_type"] = master_type
            scored_combo["master_piece"] = master_piece
            scored_combo["pipeline_tags"] = ["occasion_filtered", "master_piece", "llm_color", "llm_pattern", "accessories"]
            scored_combo["items"] = _flatten_outfit_items(scored_combo)
            _attach_score_meta(scored_combo, merged_context)
            scored.append(scored_combo)

        ranked = outfit_ranker.rank(user_id=user_id, outfits=scored, top_n=min(10, len(scored)))

        # Closed-loop (lightweight): if a refinement is requested (or suggested proactively),
        # run a deterministic refinement pass and re-score using the UnifiedStyleScorer.
        refine_mode = str(context.get("refinement") or (context.get("signals") or {}).get("default_refinement") or (context.get("signals") or {}).get("auto_refinement") or "").strip().lower()
        if refine_mode:
            merged_context["refinement"] = refine_mode
            merged_context["wardrobe"] = wardrobe
            try:
                refined = refinement_engine.apply(outfits=ranked, context=merged_context) or ranked
            except Exception:
                refined = ranked

            for idx, outfit in enumerate(ranked):
                refined_items = []
                if isinstance(refined, list) and idx < len(refined) and isinstance(refined[idx], dict):
                    refined_items = refined[idx].get("items") if isinstance(refined[idx].get("items"), list) else []
                if refined_items:
                    outfit["refined_items"] = refined_items
                    outfit["refinement_mode"] = refine_mode
                    outfit["unified_style_refined"] = _unified_style_snapshot(refined_items, merged_context)
                    outfit["score_meta_refined"] = outfit["unified_style_refined"]

        # Closed-loop fix (weakness-aware): improve the best outfit by addressing the weakest dimension
        # and re-scoring. Keeps the system from being "one-shot".
        if ranked and bool(os.getenv("ENABLE_CLOSED_LOOP_FIX", "true").lower() in ("1", "true", "yes", "on")):
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
                float(_dict(o.get("score_meta") or o.get("unified_style")).get("score") or 0.0),
                float(o.get("rank_score", o.get("score", 0.0)) or 0.0),
            ),
            reverse=True,
        )

        ranked = _diversify_outfits(ranked, limit=3)

        # AHVI editorial quality guard: remove weak/bad combinations before memory, indexing and card rendering.
        try:
            ranked = filter_and_guard_outfits(
                ranked,
                user_profile=(
                    merged_context.get("user_profile")
                    or merged_context.get("profile")
                    or merged_context.get("user")
                    or {}
                ),
                intent=(
                    merged_context.get("occasion")
                    or merged_context.get("intent")
                    or locals().get("occasion")
                    or ""
                ),
                query=(
                    merged_context.get("user_query")
                    or merged_context.get("query")
                    or locals().get("user_query")
                    or locals().get("query")
                    or ""
                ),
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("outfit_quality_guard_failed: %s", e)

        user_memory["recent_outfits"] = ranked + user_memory.get("recent_outfits", [])
        user_memory["recent_outfits"] = user_memory["recent_outfits"][:30]
        _save_user_memory(user_id, user_memory)

        for outfit in ranked:
            _index_outfit_vector(user_id=user_id, outfit=outfit, label="recent")

    cards = _build_cards(ranked, merged_context)
    cards = _ahvi_demo_force_accessories_into_cards(cards, ranked, occasion_filtered, limit=2)
    cards = _ahvi_finalize_style_cards(cards, ranked, occasion_filtered, limit=2)

    board_item_ids: List[str] = _ahvi_board_item_ids_from_cards(cards, ranked)

    return {
        "intent": "daily_outfit",
        "context": "Generated with occasion filtering, master-piece selection, LLM color/pattern filtering, and accessory completion.",
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



# ---- AHVI demo fix: force optional accessories into generated style cards ----
def _ahvi_demo_is_accessory_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False

    blob = " ".join(str(item.get(k, "") or "") for k in (
        "slot", "type", "category", "cat", "category_group",
        "sub_category", "subcategory", "subCategory",
        "name", "label", "description"
    )).lower()

    tokens = set(re.sub(r"[^a-z0-9]+", " ", blob).split())

    return bool(tokens.intersection({
        "accessory", "accessories",
        "watch", "watches",
        "belt", "belts",
        "cap", "caps",
        "hat", "hats",
        "sunglass", "sunglasses",
        "eyewear", "glasses",
        "bag", "bags",
        "jewelry", "jewellery",
        "ring", "rings",
        "necklace", "necklaces",
        "bracelet", "bracelets",
        "earring", "earrings",
        "scarf", "scarves",
    }))


def _ahvi_demo_select_accessories(
    wardrobe: Dict[str, List[Dict[str, Any]]],
    combo: Dict[str, Any],
    limit: int = 2,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    if isinstance(wardrobe, dict):
        for key in (
            "accessories", "accessory",
            "jewelry", "jewellery",
            "bags", "bag",
            "watches", "belts", "caps", "hats",
            "sunglasses", "eyewear",
        ):
            values = wardrobe.get(key, [])
            if isinstance(values, list):
                candidates.extend([x for x in values if isinstance(x, dict)])

        for values in wardrobe.values():
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict) and _ahvi_demo_is_accessory_item(item):
                        candidates.append(item)

    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in candidates:
        key = str(
            item.get("id")
            or item.get("$id")
            or item.get("name")
            or item.get("label")
            or id(item)
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)

    if not unique:
        return []

    core_ids = {
        str(x.get("id") or x.get("$id") or x.get("name") or x.get("label"))
        for x in [
            combo.get("top"),
            combo.get("bottom"),
            combo.get("footwear"),
            combo.get("shoe"),
            combo.get("outerwear"),
            combo.get("dress"),
        ]
        if isinstance(x, dict)
    }

    filtered = [
        item for item in unique
        if str(item.get("id") or item.get("$id") or item.get("name") or item.get("label")) not in core_ids
    ] or unique

    priority = {
        "watch": 0,
        "watches": 0,
        "belt": 1,
        "belts": 1,
        "sunglass": 2,
        "sunglasses": 2,
        "eyewear": 2,
        "glasses": 2,
        "cap": 3,
        "caps": 3,
        "hat": 3,
        "hats": 3,
        "bag": 4,
        "bags": 4,
        "jewelry": 5,
        "jewellery": 5,
        "bracelet": 5,
        "necklace": 5,
        "ring": 5,
    }

    def score(item: Dict[str, Any]) -> tuple:
        blob = " ".join(str(item.get(k, "") or "") for k in (
            "category", "sub_category", "subcategory", "name", "label"
        )).lower()
        tokens = re.sub(r"[^a-z0-9]+", " ", blob).split()
        best = min([priority.get(t, 99) for t in tokens] or [99])
        has_image = bool(
            item.get("masked_url")
            or item.get("maskedUrl")
            or item.get("image_url")
            or item.get("imageUrl")
            or item.get("url")
        )
        return (best, 0 if has_image else 1, str(item.get("name") or item.get("label") or ""))

    filtered.sort(key=score)
    return filtered[:limit]


def _ahvi_demo_force_accessories_into_cards(
    cards: List[Dict[str, Any]],
    outfits: List[Dict[str, Any]],
    wardrobe: Dict[str, List[Dict[str, Any]]],
    limit: int = 2,
) -> List[Dict[str, Any]]:
    if not isinstance(cards, list):
        return cards

    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue

        outfit = outfits[index] if index < len(outfits) and isinstance(outfits[index], dict) else {}

        accessories = outfit.get("accessories")
        if not isinstance(accessories, list) or not accessories:
            accessories = _ahvi_demo_select_accessories(wardrobe, outfit, limit=limit)

        accessories = [x for x in accessories if isinstance(x, dict)][:limit]
        if not accessories:
            continue

        items = card.get("items")
        if not isinstance(items, list):
            items = []

        seen = {
            str(x.get("id") or x.get("$id") or x.get("name") or x.get("label"))
            for x in items
            if isinstance(x, dict)
        }

        for accessory in accessories:
            key = str(
                accessory.get("id")
                or accessory.get("$id")
                or accessory.get("name")
                or accessory.get("label")
            )
            if key not in seen:
                items.append(accessory)
                seen.add(key)

        card["items"] = items
        card["accessories"] = accessories

    return cards
# ---- end AHVI demo accessory fix ----



# ---- AHVI style board contract fix: accessories must live inside card["items"] ----
def _ahvi_card_item_key(item):
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("id")
        or item.get("$id")
        or item.get("item_id")
        or item.get("name")
        or item.get("label")
        or ""
    ).strip().lower()


def _ahvi_is_accessory_item(item):
    if not isinstance(item, dict):
        return False

    blob = " ".join(str(item.get(k, "") or "") for k in (
        "slot", "type", "category", "cat", "category_group",
        "sub_category", "subcategory", "subCategory",
        "name", "label", "description"
    )).lower()

    tokens = set(re.sub(r"[^a-z0-9]+", " ", blob).split())
    return bool(tokens.intersection({
        "accessory", "accessories",
        "watch", "watches",
        "belt", "belts",
        "cap", "caps",
        "hat", "hats",
        "sunglass", "sunglasses",
        "eyewear", "glasses",
        "bag", "bags",
        "jewelry", "jewellery",
        "ring", "rings",
        "necklace", "necklaces",
        "bracelet", "bracelets",
        "earring", "earrings",
        "scarf", "scarves",
    }))


def _ahvi_accessory_candidates(wardrobe, combo, limit=2):
    candidates = []

    if isinstance(combo, dict) and isinstance(combo.get("accessories"), list):
        candidates.extend([x for x in combo.get("accessories") if isinstance(x, dict)])

    if isinstance(wardrobe, dict):
        for key in (
            "accessories", "accessory",
            "jewelry", "jewellery",
            "bags", "bag",
            "watches", "belts", "caps", "hats",
            "sunglasses", "eyewear",
        ):
            values = wardrobe.get(key, [])
            if isinstance(values, list):
                candidates.extend([x for x in values if isinstance(x, dict)])

        for values in wardrobe.values():
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict) and _ahvi_is_accessory_item(item):
                        candidates.append(item)

    seen = set()
    unique = []
    for item in candidates:
        key = _ahvi_card_item_key(item) or str(id(item))
        if key not in seen:
            seen.add(key)
            unique.append(item)

    core_ids = {
        _ahvi_card_item_key(x)
        for x in [
            combo.get("top") if isinstance(combo, dict) else None,
            combo.get("bottom") if isinstance(combo, dict) else None,
            combo.get("footwear") if isinstance(combo, dict) else None,
            combo.get("shoe") if isinstance(combo, dict) else None,
            combo.get("shoes") if isinstance(combo, dict) else None,
            combo.get("dress") if isinstance(combo, dict) else None,
            combo.get("outerwear") if isinstance(combo, dict) else None,
        ]
        if isinstance(x, dict)
    }

    unique = [x for x in unique if _ahvi_card_item_key(x) not in core_ids] or unique

    priority = {
        "watch": 0, "watches": 0,
        "belt": 1, "belts": 1,
        "sunglass": 2, "sunglasses": 2, "eyewear": 2, "glasses": 2,
        "cap": 3, "caps": 3, "hat": 3, "hats": 3,
        "bag": 4, "bags": 4,
        "jewelry": 5, "jewellery": 5,
        "bracelet": 5, "necklace": 5, "ring": 5,
    }

    def score(item):
        blob = " ".join(str(item.get(k, "") or "") for k in (
            "category", "sub_category", "subcategory", "name", "label"
        )).lower()
        tokens = re.sub(r"[^a-z0-9]+", " ", blob).split()
        best = min([priority.get(t, 99) for t in tokens] or [99])
        has_image = bool(
            item.get("masked_url")
            or item.get("maskedUrl")
            or item.get("image_url")
            or item.get("imageUrl")
            or item.get("url")
        )
        return (best, 0 if has_image else 1, str(item.get("name") or item.get("label") or ""))

    unique.sort(key=score)
    return unique[:limit]


def _ahvi_finalize_style_cards(cards, outfits, wardrobe, limit=2):
    if not isinstance(cards, list):
        return cards

    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue

        outfit = outfits[index] if isinstance(outfits, list) and index < len(outfits) and isinstance(outfits[index], dict) else {}

        items = card.get("items")
        if not isinstance(items, list):
            items = []

        accessories = card.get("accessories")
        if not isinstance(accessories, list) or not accessories:
            accessories = _ahvi_accessory_candidates(wardrobe, outfit, limit=limit)

        accessories = [x for x in accessories if isinstance(x, dict)][:limit]

        seen = {
            _ahvi_card_item_key(x)
            for x in items
            if isinstance(x, dict)
        }

        for acc in accessories:
            key = _ahvi_card_item_key(acc)
            if key and key not in seen:
                items.append(acc)
                seen.add(key)

        card["items"] = items
        card["accessories"] = accessories

    return cards


def _ahvi_board_item_ids_from_cards(cards, fallback_ranked):
    ids = []

    if isinstance(cards, list) and cards and isinstance(cards[0], dict):
        source_items = cards[0].get("items") or []
        for item in source_items:
            if isinstance(item, dict):
                item_id = str(item.get("id") or item.get("$id") or item.get("item_id") or "").strip()
                if item_id:
                    ids.append(item_id)

    if not ids and isinstance(fallback_ranked, list) and fallback_ranked:
        best = fallback_ranked[0] if isinstance(fallback_ranked[0], dict) else {}
        for part in ("top", "bottom", "dress", "footwear", "shoe", "shoes", "outerwear"):
            value = best.get(part)
            if isinstance(value, dict):
                item_id = str(value.get("id") or value.get("$id") or value.get("item_id") or "").strip()
                if item_id:
                    ids.append(item_id)
        for item in (best.get("accessories") or []):
            if isinstance(item, dict):
                item_id = str(item.get("id") or item.get("$id") or item.get("item_id") or "").strip()
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



# ---- AHVI final strict accessory override ----
def _ahvi_strict_tokens(item):
    blob = " ".join(str(item.get(k, "") or "") for k in (
        "slot", "type", "category", "cat", "category_group",
        "sub_category", "subcategory", "subCategory",
        "name", "label", "description"
    )).lower()
    return set(re.sub(r"[^a-z0-9]+", " ", blob).split())


def _ahvi_is_accessory_item(item):
    if not isinstance(item, dict):
        return False

    tokens = _ahvi_strict_tokens(item)

    clothing_reject = {
        "top", "tops", "shirt", "shirts", "tee", "tshirt", "tshirts",
        "blouse", "tunic", "tunics", "kurta", "saree", "sari",
        "dress", "dresses", "gown", "jumpsuit",
        "bottom", "bottoms", "pant", "pants", "trouser", "trousers",
        "jean", "jeans", "short", "shorts", "skirt", "skirts",
        "legging", "leggings", "chino", "chinos",
        "footwear", "shoe", "shoes", "sneaker", "sneakers",
        "boot", "boots", "heel", "heels", "sandal", "sandals",
        "outerwear", "jacket", "coat", "blazer",
    }

    accessory_accept = {
        "accessory", "accessories",
        "watch", "watches",
        "belt", "belts",
        "cap", "caps",
        "hat", "hats",
        "sunglass", "sunglasses",
        "eyewear", "glasses",
        "bag", "bags", "purse", "handbag",
        "jewelry", "jewellery",
        "ring", "rings",
        "necklace", "necklaces",
        "bracelet", "bracelets",
        "earring", "earrings",
        "scarf", "scarves",
    }

    if tokens.intersection(clothing_reject):
        # Allow only explicit accessory categories/names, never clothing names.
        if not tokens.intersection(accessory_accept):
            return False
        # If both accessory + clothing words exist, reject obvious clothing.
        if tokens.intersection({
            "shirt", "tunic", "pants", "trousers", "jeans", "saree",
            "dress", "shoes", "sneakers", "boots", "sandals"
        }):
            return False

    return bool(tokens.intersection(accessory_accept))


def _ahvi_accessory_candidates(wardrobe, combo, limit=3):
    candidates = []

    if isinstance(combo, dict) and isinstance(combo.get("accessories"), list):
        candidates.extend([x for x in combo.get("accessories") if isinstance(x, dict)])

    if isinstance(wardrobe, dict):
        for values in wardrobe.values():
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        candidates.append(item)

    # Strict filter only. No clothing fallback.
    candidates = [x for x in candidates if _ahvi_is_accessory_item(x)]

    seen = set()
    unique = []
    for item in candidates:
        key = _ahvi_card_item_key(item) if "_ahvi_card_item_key" in globals() else str(
            item.get("id") or item.get("$id") or item.get("name") or item.get("label") or id(item)
        ).lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    priority = {
        "watch": 0, "watches": 0,
        "belt": 1, "belts": 1,
        "sunglass": 2, "sunglasses": 2, "eyewear": 2, "glasses": 2,
        "bag": 3, "bags": 3, "purse": 3, "handbag": 3,
        "cap": 4, "caps": 4, "hat": 4, "hats": 4,
        "jewelry": 5, "jewellery": 5, "earring": 5, "earrings": 5,
        "bracelet": 5, "necklace": 5, "ring": 5,
        "scarf": 6, "scarves": 6,
    }

    def score(item):
        tokens = _ahvi_strict_tokens(item)
        best = min([priority.get(t, 99) for t in tokens] or [99])
        has_image = bool(
            item.get("masked_url")
            or item.get("maskedUrl")
            or item.get("image_url")
            or item.get("imageUrl")
            or item.get("url")
        )
        return (best, 0 if has_image else 1, str(item.get("name") or item.get("label") or ""))

    unique.sort(key=score)
    return unique[:limit]


def _ahvi_finalize_style_cards(cards, outfits, wardrobe, limit=3):
    if not isinstance(cards, list):
        return cards

    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue

        outfit = outfits[index] if isinstance(outfits, list) and index < len(outfits) and isinstance(outfits[index], dict) else {}

        raw_items = card.get("items") if isinstance(card.get("items"), list) else []

        # Keep core clothing/shoe items, but remove fake accessories like extra pants/sarees.
        core_items = []
        seen_core = set()

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            if _ahvi_is_accessory_item(item):
                continue

            key = _ahvi_card_item_key(item) if "_ahvi_card_item_key" in globals() else str(
                item.get("id") or item.get("$id") or item.get("name") or item.get("label") or id(item)
            ).lower()

            if key not in seen_core:
                seen_core.add(key)
                core_items.append(item)

        # For demo board, keep max 3-4 core pieces.
        core_items = core_items[:4]

        accessories = _ahvi_accessory_candidates(wardrobe, outfit, limit=limit)

        seen = {
            _ahvi_card_item_key(x) if "_ahvi_card_item_key" in globals() else str(
                x.get("id") or x.get("$id") or x.get("name") or x.get("label") or id(x)
            ).lower()
            for x in core_items
            if isinstance(x, dict)
        }

        final_items = list(core_items)
        final_accessories = []

        for acc in accessories:
            key = _ahvi_card_item_key(acc) if "_ahvi_card_item_key" in globals() else str(
                acc.get("id") or acc.get("$id") or acc.get("name") or acc.get("label") or id(acc)
            ).lower()
            if key and key not in seen:
                final_items.append(acc)
                final_accessories.append(acc)
                seen.add(key)

        card["items"] = final_items[:7]
        card["accessories"] = final_accessories

    return cards
# ---- end AHVI final strict accessory override ----

# ================= AHVI OUTFIT PIPELINE GENDER PATCH V2 BEGIN =================

_AHVI_PIPE_MALE_GENDERS = {"m", "male", "man", "men", "mens", "boy"}
_AHVI_PIPE_FEMALE_GENDERS = {"f", "female", "woman", "women", "womens", "girl", "ladies"}
_AHVI_PIPE_FEMININE_ONLY = {"saree", "sari", "lehenga", "gown", "skirt", "skirts", "blouse", "kurti"}
_AHVI_PIPE_MALE_TRADITIONAL = {"sherwani", "achkan"}
_AHVI_PIPE_EXPLICIT_FEMININE = {
    "saree", "sari", "lehenga", "gown", "skirt", "skirts",
    "female", "women", "woman", "ladies", "feminine",
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

    style_dna = context.get("style_dna") if isinstance(context.get("style_dna"), dict) else {}
    profile = context.get("user_profile") if isinstance(context.get("user_profile"), dict) else {}

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
            "slot", "type", "category", "cat", "category_group",
            "sub_category", "subcategory", "subCategory",
            "name", "label", "description", "gender",
            "style_gender", "target_gender", "audience",
            "department", "intended_for", "wearer",
        )
    )
    return set(_ahvi_pipe_tokens(blob))


def _ahvi_pipe_item_allowed(item, context):
    if not isinstance(item, dict):
        return False

    if _ahvi_pipe_context_gender(context) != "male":
        return True

    if _ahvi_pipe_query_allows_feminine(context):
        return True

    tokens = _ahvi_pipe_item_tokens(item)

    audience = set(_ahvi_pipe_tokens(" ".join(
        str(item.get(k, "") or "")
        for k in (
            "gender", "style_gender", "target_gender",
            "audience", "department", "intended_for", "wearer",
        )
    )))

    if audience.intersection(_AHVI_PIPE_FEMALE_GENDERS):
        return False

    if tokens.intersection(_AHVI_PIPE_FEMININE_ONLY):
        return False

    if tokens.intersection({"dress", "dresses"}) and not tokens.intersection(_AHVI_PIPE_MALE_TRADITIONAL):
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


try:
    _AHVI_ORIGINAL_GET_DAILY_OUTFITS = get_daily_outfits
except Exception:
    _AHVI_ORIGINAL_GET_DAILY_OUTFITS = None


if _AHVI_ORIGINAL_GET_DAILY_OUTFITS and not getattr(get_daily_outfits, "_ahvi_gender_guard_v2", False):

    def get_daily_outfits(user):
        user = dict(user or {})
        context = user.get("context") if isinstance(user.get("context"), dict) else {}

        wardrobe = user.get("wardrobe")
        if isinstance(wardrobe, list):
            user["wardrobe"] = [
                item for item in wardrobe
                if _ahvi_pipe_item_allowed(item, context)
            ]

        result = _AHVI_ORIGINAL_GET_DAILY_OUTFITS(user)

        if not isinstance(result, dict):
            return result

        if isinstance(result.get("outfits"), list):
            result["outfits"] = [
                outfit for outfit in result["outfits"]
                if _ahvi_pipe_outfit_allowed(outfit, context)
            ]

        if isinstance(result.get("cards"), list):
            result["cards"] = [
                card for card in result["cards"]
                if _ahvi_pipe_card_allowed(card, context)
            ]

        data = result.get("data")
        if isinstance(data, dict) and isinstance(data.get("outfits"), list):
            data["outfits"] = [
                outfit for outfit in data["outfits"]
                if _ahvi_pipe_outfit_allowed(outfit, context)
            ]

        result.setdefault("meta", {})
        if isinstance(result.get("meta"), dict):
            result["meta"]["style_gender_guard"] = _ahvi_pipe_context_gender(context)

        return result

    get_daily_outfits._ahvi_gender_guard_v2 = True

# ================= AHVI OUTFIT PIPELINE GENDER PATCH V2 END =================


# ================= AHVI OUTFIT PIPELINE FINALIZER V3 BEGIN =================
# Final card quality layer:
# - normalize role/slot/category so Flutter renders top/bottom/footwear correctly
# - remove duplicate accessories, especially two watches
# - cap/headwear only for casual/street/travel/sport/outdoor requests
# - diversify top/bottom/footwear across the 3 cards where wardrobe alternatives exist

try:
    import hashlib as _ahvi_final_hashlib
    import logging as _ahvi_final_logging
except Exception:
    _ahvi_final_hashlib = None
    _ahvi_final_logging = None

_AHVI_FINAL_ORIGINAL_GET_DAILY_OUTFITS = get_daily_outfits


def _ahvi_final_tokens(value):
    import re as _re
    return set(_re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _ahvi_final_blob(item):
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


def _ahvi_final_key(item):
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


def _ahvi_final_role(item):
    blob = _ahvi_final_blob(item)
    tokens = _ahvi_final_tokens(blob)

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
        "earring", "earrings", "scarf", "scarves", "accessory", "accessories",
    }):
        return "accessory"

    if tokens.intersection({
        "dress", "dresses", "saree", "sari", "lehenga",
        "gown", "jumpsuit", "sherwani"
    }):
        return "dress"

    # Tops before bottoms so short-sleeved shirt never becomes shorts.
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


def _ahvi_final_accessory_type(item):
    blob = _ahvi_final_blob(item)
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


def _ahvi_final_image(item):
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


def _ahvi_final_normalize_item(item, role=None):
    row = dict(item or {})
    inferred = role or _ahvi_final_role(row)

    if inferred == "top":
        row["role"] = "top"
        row["slot"] = "top"
        row["category"] = row.get("category") or "Tops"
    elif inferred == "bottom":
        row["role"] = "bottom"
        row["slot"] = "bottom"
        row["category"] = row.get("category") or "Bottoms"
    elif inferred == "footwear":
        row["role"] = "footwear"
        row["slot"] = "footwear"
        row["category"] = row.get("category") or "Footwear"
    elif inferred == "dress":
        row["role"] = "dress"
        row["slot"] = "dress"
        row["category"] = row.get("category") or "Dresses"
    elif inferred == "accessory":
        row["role"] = "accessory"
        row["slot"] = "accessory"
        row["category"] = row.get("category") or "Accessories"

    image = _ahvi_final_image(row)
    if image:
        row["image_url"] = row.get("image_url") or image
        row["imageUrl"] = row.get("imageUrl") or image
        row["masked_url"] = row.get("masked_url") or image
        row["maskedUrl"] = row.get("maskedUrl") or image

    return row


def _ahvi_final_pools(wardrobe):
    pools = {"top": [], "bottom": [], "footwear": [], "dress": [], "accessory": []}
    seen = {k: set() for k in pools}

    for item in wardrobe or []:
        if not isinstance(item, dict):
            continue
        if not _ahvi_final_image(item):
            continue

        role = _ahvi_final_role(item)
        if role not in pools:
            continue

        key = _ahvi_final_key(item)
        if key in seen[role]:
            continue

        seen[role].add(key)
        pools[role].append(item)

    return pools


def _ahvi_final_pick_unused(pool, used, fallback=None):
    for item in pool or []:
        key = _ahvi_final_key(item)
        if key and key not in used:
            used.add(key)
            return item

    if fallback is not None:
        key = _ahvi_final_key(fallback)
        if key:
            used.add(key)
        return fallback

    if pool:
        item = pool[0]
        key = _ahvi_final_key(item)
        if key:
            used.add(key)
        return item

    return None


def _ahvi_final_clean_accessories(accessories, query):
    q = str(query or "").lower()
    headwear_allowed = any(k in q for k in [
        "casual", "street", "travel", "airport", "sport", "gym",
        "sun", "beach", "outdoor", "college", "weekend"
    ])

    selected = []
    seen_types = set()
    seen_ids = set()

    for item in accessories or []:
        if not isinstance(item, dict):
            continue

        key = _ahvi_final_key(item)
        if key and key in seen_ids:
            continue

        typ = _ahvi_final_accessory_type(item)
        if typ == "headwear" and not headwear_allowed:
            continue

        # This is the main two-watch fix.
        if typ in seen_types:
            continue

        selected.append(_ahvi_final_normalize_item(item, "accessory"))

        if key:
            seen_ids.add(key)
        seen_types.add(typ)

        # Keep editorial board clean: max 1 accessory for date/office, max 2 casual.
        max_count = 2 if headwear_allowed else 1
        if len(selected) >= max_count:
            break

    return selected


def _ahvi_final_card_signature(card):
    names = []
    for item in (card.get("items") or []):
        if isinstance(item, dict):
            role = _ahvi_final_role(item)
            if role in {"top", "bottom", "dress", "footwear"}:
                names.append(str(item.get("name") or item.get("label") or item.get("category") or role))
    return " | ".join(names)


def _ahvi_final_postprocess_cards(result, user):
    if not isinstance(result, dict):
        return result

    cards = result.get("cards")
    if not isinstance(cards, list) or not cards:
        return result

    context = user.get("context") if isinstance(user, dict) and isinstance(user.get("context"), dict) else {}
    query = str(context.get("query") or context.get("occasion") or "")

    wardrobe = []
    if isinstance(result.get("normalized_wardrobe"), list):
        wardrobe = result.get("normalized_wardrobe") or []
    elif isinstance(user, dict) and isinstance(user.get("wardrobe"), list):
        wardrobe = user.get("wardrobe") or []

    pools = _ahvi_final_pools(wardrobe)

    used_tops = set()
    used_bottoms = set()
    used_footwear = set()
    used_dresses = set()
    cleaned_cards = []

    for idx, card in enumerate(cards[:3]):
        if not isinstance(card, dict):
            continue

        source_items = []
        for key in ("items", "accessories"):
            value = card.get(key)
            if isinstance(value, list):
                source_items.extend([x for x in value if isinstance(x, dict)])

        top = next((x for x in source_items if _ahvi_final_role(x) == "top"), None)
        bottom = next((x for x in source_items if _ahvi_final_role(x) == "bottom"), None)
        dress = next((x for x in source_items if _ahvi_final_role(x) == "dress"), None)
        footwear = next((x for x in source_items if _ahvi_final_role(x) == "footwear"), None)
        accessories = [x for x in source_items if _ahvi_final_role(x) == "accessory"]

        final_items = []

        if dress and not (top and bottom):
            chosen_dress = _ahvi_final_pick_unused(pools["dress"], used_dresses, dress)
            if chosen_dress:
                final_items.append(_ahvi_final_normalize_item(chosen_dress, "dress"))
        else:
            chosen_top = _ahvi_final_pick_unused(pools["top"], used_tops, top)
            chosen_bottom = _ahvi_final_pick_unused(pools["bottom"], used_bottoms, bottom)

            if chosen_top:
                final_items.append(_ahvi_final_normalize_item(chosen_top, "top"))
            if chosen_bottom:
                final_items.append(_ahvi_final_normalize_item(chosen_bottom, "bottom"))

        chosen_footwear = _ahvi_final_pick_unused(pools["footwear"], used_footwear, footwear)
        if chosen_footwear:
            final_items.append(_ahvi_final_normalize_item(chosen_footwear, "footwear"))

        final_items.extend(_ahvi_final_clean_accessories(accessories or pools["accessory"], query))

        fixed = dict(card)
        fixed["items"] = final_items
        # Keep empty so orchestrator does not double-merge accessories back into items.
        fixed["accessories"] = []

        top_name = next((str(i.get("name") or i.get("label") or "top") for i in final_items if _ahvi_final_role(i) in {"top", "dress"}), "")
        bottom_name = next((str(i.get("name") or i.get("label") or "bottom") for i in final_items if _ahvi_final_role(i) == "bottom"), "")
        footwear_name = next((str(i.get("name") or i.get("label") or "footwear") for i in final_items if _ahvi_final_role(i) == "footwear"), "")

        core = ", ".join([x for x in [top_name, bottom_name, footwear_name] if x])
        if "date" in query.lower():
            why = f"This works for date night because {core} creates a clean smart-casual balance without over-accessorizing."
        elif "office" in query.lower() or "meeting" in query.lower() or "work" in query.lower():
            why = f"This works for office because {core} keeps the look structured, neat, and wearable."
        else:
            why = f"This works because {core} creates a balanced top-bottom-footwear structure."

        # Fix accidental tuple formatting from f-string expression above.
        why = why.replace("('", "").replace("', '", " and ").replace("')", "")

        fixed["why_it_works"] = why
        fixed["explanation"] = why
        fixed["reason"] = why
        fixed["style_reason"] = why

        title = str(fixed.get("title") or fixed.get("name") or "").strip()
        if not title or title.lower() in {"style board", "ahvi styled look"}:
            fixed["title"] = f"Look {idx + 1} · Styled Fit"
            fixed["name"] = fixed["title"]

        cleaned_cards.append(fixed)

    result["cards"] = cleaned_cards
    result["boards"] = cleaned_cards

    try:
        log = _ahvi_final_logging.getLogger("ahvi.outfit_pipeline") if _ahvi_final_logging else None
        if log:
            log.info(
                "ahvi.pipeline_finalizer_v3 cards=%s accessory_counts=%s signatures=%s",
                len(cleaned_cards),
                [len(c.get("accessories") or []) for c in cleaned_cards],
                [_ahvi_final_card_signature(c) for c in cleaned_cards],
            )
    except Exception:
        pass

    return result


def get_daily_outfits(user):
    user = dict(user or {})
    result = _AHVI_FINAL_ORIGINAL_GET_DAILY_OUTFITS(user)
    return _ahvi_final_postprocess_cards(result, user)


get_daily_outfits._ahvi_finalizer_v3 = True

# ================= AHVI OUTFIT PIPELINE FINALIZER V3 END =================

