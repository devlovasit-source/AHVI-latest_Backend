import json
from typing import Any, Dict, List


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v or "").strip()]
    if isinstance(value, (tuple, set)):
        return [str(v) for v in value if str(v or "").strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.replace("|", ",").split(",") if part.strip()]
    return []


def normalize_occasion(query: Any) -> str:
    q = str(query or "").lower().replace("-", "_")

    if any(x in q for x in ["beach", "resort", "pool"]):
        return "beach"

    if any(x in q for x in ["gym", "workout", "fitness", "training", "yoga", "running"]):
        return "gym"

    if any(x in q for x in ["rave", "club"]):
        return "rave"

    if any(x in q for x in ["cocktail"]):
        return "cocktail"

    if any(x in q for x in ["party", "house_party"]):
        return "house_party"

    if any(x in q for x in ["office", "work", "meeting"]):
        return "office"

    if any(x in q for x in ["date", "dinner", "date_night", "tonight"]):
        return "date"

    if any(x in q for x in ["temple", "pooja", "puja", "mandir", "temple_modest"]):
        return "temple"

    return "daily"


def enrich_wardrobe_item(item: dict) -> dict:
    item = item if isinstance(item, dict) else {}
    tags = (
        _as_list(item.get("tags"))
        + _as_list(item.get("style_tags"))
        + _as_list(item.get("occasions"))
        + _as_list(item.get("occasion_tags"))
    )
    colors = _as_list(item.get("colors")) or _as_list(item.get("color")) or _as_list(item.get("color_code"))
    text = " ".join([
        str(item.get("name", "")),
        str(item.get("category", "")),
        str(item.get("subcategory") or item.get("sub_category") or ""),
        " ".join(tags),
        " ".join(colors),
    ]).lower()

    meta = {
        "category": item.get("category"),
        "subcategory": item.get("subcategory") or item.get("sub_category"),
        "formality": "casual",
        "occasion_affinity": [],
        "season_affinity": ["all_season"],
        "avoid_for": [],
        "needs_review": False,
        "confidence": item.get("confidence", 0.7),
    }

    if any(x in text for x in ["belt"]):
        meta.update({
            "category": "Accessories",
            "subcategory": "Belt",
            "formality": "smart_casual",
            "occasion_affinity": ["daily", "office", "date", "coffee_run"],
            "avoid_for": ["gym", "beach", "rave"],
        })

    elif any(x in text for x in ["loafer", "oxford", "derby"]):
        meta.update({
            "category": "Footwear",
            "subcategory": "Loafers",
            "formality": "smart_casual",
            "occasion_affinity": ["office", "date", "cocktail"],
            "avoid_for": ["gym", "beach", "rave"],
        })

    elif any(x in text for x in ["suit", "blazer", "tie"]):
        meta.update({
            "category": item.get("category") or "Outerwear",
            "subcategory": "Formalwear",
            "formality": "formal",
            "occasion_affinity": ["office", "business_formal", "cocktail"],
            "avoid_for": ["gym", "beach", "rave", "house_party"],
        })

    elif any(x in text for x in ["linen"]):
        meta.update({
            "formality": "casual",
            "occasion_affinity": ["beach", "resort", "daily"],
            "season_affinity": ["summer", "spring"],
            "avoid_for": ["business_formal", "rave"],
        })

    elif any(x in text for x in ["jeans", "denim"]):
        meta.update({
            "category": "Bottoms",
            "subcategory": "Jeans",
            "formality": "casual",
            "occasion_affinity": ["daily", "coffee_run", "house_party"],
            "avoid_for": ["business_formal", "temple_formal"],
        })

    elif any(x in text for x in ["sandal", "slides", "flip flop", "slipper"]):
        meta.update({
            "category": "Footwear",
            "subcategory": "Sandals",
            "formality": "casual",
            "occasion_affinity": ["beach", "resort", "daily"],
            "season_affinity": ["summer", "spring"],
            "avoid_for": ["office", "business_formal", "cocktail"],
        })

    elif any(x in text for x in ["watch"]):
        meta.update({
            "category": "Accessories",
            "subcategory": "Watch",
            "formality": "smart_casual",
            "occasion_affinity": ["office", "date", "cocktail", "daily"],
            "avoid_for": ["gym", "beach"],
        })

    elif any(x in text for x in ["saree", "sari"]):
        meta.update({
            "category": "Dresses",
            "subcategory": "Saree",
            "formality": "ethnic_formal",
            "occasion_affinity": ["wedding", "festival", "temple", "traditional"],
            "avoid_for": ["gym", "beach", "rave"],
        })

    if not meta["occasion_affinity"]:
        meta["occasion_affinity"] = ["daily"]

    return meta


def _style_meta(item: dict) -> dict:
    if not isinstance(item, dict):
        return {}
    raw = item.get("style_metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def score_item_for_occasion(item: dict, occasion: Any) -> int:
    meta = _style_meta(item)
    score = 0
    normalized = normalize_occasion(occasion)

    affinity = meta.get("occasion_affinity") or []
    avoid = meta.get("avoid_for") or []

    if normalized in affinity:
        score += 3

    if normalized in avoid:
        score -= 10

    return score


def board_has_occasion_conflict(board: dict, occasion: Any) -> bool:
    if not isinstance(board, dict):
        return False
    normalized = normalize_occasion(occasion)
    items = list(board.get("items") or [])
    for key in ("top", "bottom", "dress", "shoes", "footwear", "outerwear"):
        value = board.get(key)
        if isinstance(value, dict):
            items.append(value)
    for item in items:
        meta = _style_meta(item)
        if normalized in (meta.get("avoid_for") or []):
            return True
    return False
