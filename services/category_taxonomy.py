"""Shared garment-category taxonomy.

Two callers historically duplicated this logic:

- ``routers/chat.py`` collapsed everything into 6 buckets for fast wardrobe
  count responses (Tops / Bottoms / Footwear / Outerwear / Dresses / Accessories).
- ``routers/wardrobe_capture.py`` produced a granular (category, sub_category)
  pair (e.g. Tops / "T-Shirt", Bags / "Bag") for save-to-wardrobe.

Each module has its own behavior; we keep both behaviors intact and only
deduplicate the underlying keyword tables so they cannot drift apart again.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Granular keyword groups — used by wardrobe_capture for save-time labelling.
# Each entry: (canonical_category, default_sub_category, keywords)
# Order matters: more specific buckets must come before more general ones
# (e.g. Footwear before Bags so "boot bag" resolves to Footwear).
# ---------------------------------------------------------------------------
CANONICAL_CATEGORY_KEYWORDS: List[Tuple[str, str, List[str]]] = [
    (
        "Traditional",
        "Saree",
        [
            "saree",
            "sari",
            "lehenga",
            "dupatta",
            "sherwani",
            "kurta",
            "kurti",
            "anarkali",
        ],
    ),
    (
        "Dresses",
        "Dress",
        [
            "one piece dress",
            "one-piece dress",
            "mini dress",
            "midi dress",
            "maxi dress",
            "bodycon dress",
            "shift dress",
            "wrap dress",
            "slip dress",
            "dress",
            "gown",
            "jumpsuit",
        ],
    ),
    (
        "Bags",
        "Bag",
        [
            "handbag",
            "shoulder bag",
            "sling bag",
            "crossbody",
            "tote",
            "clutch",
            "purse",
            "backpack",
            "bag",
        ],
    ),
    (
        "Jewelry",
        "Jewelry",
        [
            "jewelry",
            "jewellery",
            "bracelet",
            "bracelets",
            "ring",
            "rings",
            "earring",
            "earrings",
            "necklace",
            "necklaces",
            "bangle",
            "bangles",
            "pendant",
            "pendants",
            "chain",
            "chains",
        ],
    ),
    (
        "Footwear",
        "Footwear",
        [
            "boot",
            "boots",
            "shoe",
            "shoes",
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
        ],
    ),
    (
        "Bottoms",
        "Bottom",
        [
            "pant",
            "pants",
            "trouser",
            "trousers",
            "jean",
            "jeans",
            "shorts",
            "skirt",
            "chino",
            "chinos",
            "legging",
            "leggings",
        ],
    ),
    (
        "Tops",
        "Top",
        [
            "shirt",
            "shirts",
            "tshirt",
            "t-shirt",
            "tee",
            "top",
            "tops",
            "blouse",
            "crop top",
            "sweater",
            "hoodie",
            "polo",
        ],
    ),
    (
        "Outerwear",
        "Outerwear",
        ["jacket", "coat", "blazer", "outerwear", "cardigan"],
    ),
    (
        "Accessories",
        "Accessory",
        [
            "watch",
            "watches",
            "belt",
            "belts",
            "sunglass",
            "sunglasses",
            "eyewear",
            "glasses",
            "scarf",
            "hat",
            "cap",
        ],
    ),
]

CANONICAL_CATEGORIES: set[str] = {row[0] for row in CANONICAL_CATEGORY_KEYWORDS}


def normalize_category_from_label(label: str) -> Tuple[str, str]:
    """Best-effort label -> (category, sub_category) mapping for capture."""
    raw = (label or "").strip().lower()
    if not raw:
        return ("Item", "Item")
    for category, default_sub, keywords in CANONICAL_CATEGORY_KEYWORDS:
        if any(keyword in raw for keyword in keywords):
            sub = raw.title() or default_sub
            if category == "Traditional" and ("sari" in raw or "saree" in raw):
                sub = "Saree"
            elif category == "Accessories":
                if "watch" in raw:
                    sub = "Watch"
                elif "belt" in raw:
                    sub = "Belt"
                elif any(x in raw for x in ["sunglass", "eyewear", "glasses"]):
                    sub = "Eyewear"
            return (category, sub)
    return ("Item", "Item")


# ---------------------------------------------------------------------------
# Coarse 6-bucket mapping — used by chat for fast wardrobe-count queries.
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _tokens(value: str) -> List[str]:
    return _TOKEN_RE.sub(" ", str(value or "").lower()).strip().split()


def _has_any(tokens: Iterable[str], words: Iterable[str]) -> bool:
    token_set = set(tokens)
    return any(word in token_set for word in words)


CHAT_EXPLICIT_MAP: Dict[str, str] = {
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
    "jewelry": "Jewelry",
    "jewellery": "Jewelry",
    "outerwear": "Outerwear",
    "outer": "Outerwear",
    "dress": "Dresses",
    "dresses": "Dresses",
    "indian wear": "Dresses",
}

_CHAT_BUCKET_KEYWORDS: List[Tuple[str, List[str]]] = [
    (
        "Tops",
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
    ),
    (
        "Bottoms",
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
        ],
    ),
    (
        "Footwear",
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
        ],
    ),
    (
        "Jewelry",
        [
            "jewelry",
            "jewellery",
            "ring",
            "rings",
            "necklace",
            "necklaces",
            "bracelet",
            "bracelets",
            "bangle",
            "bangles",
            "earring",
            "earrings",
            "pendant",
            "pendants",
            "chain",
            "chains",
            "hoop",
            "hoops",
        ],
    ),
    (
        "Accessories",
        [
            "watch",
            "watches",
            "bag",
            "bags",
            "belt",
            "belts",
            "scarf",
            "scarves",
            "accessory",
            "accessories",
            "hat",
            "cap",
            "sunglass",
            "sunglasses",
        ],
    ),
    (
        "Outerwear",
        ["jacket", "coat", "blazer", "outerwear", "cardigan", "overshirt"],
    ),
    (
        "Dresses",
        ["dress", "dresses", "gown", "jumpsuit", "saree", "lehenga", "sherwani"],
    ),
]


def categorize_for_chat(item: dict) -> str:
    """Return one of: Tops/Bottoms/Footwear/Jewelry/Accessories/Outerwear/Dresses."""
    if not isinstance(item, dict):
        return "Accessories"
    explicit = (
        str(item.get("category") or item.get("cat") or item.get("type") or "")
        .strip()
        .lower()
    )
    if explicit in CHAT_EXPLICIT_MAP:
        return CHAT_EXPLICIT_MAP[explicit]
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
    tokens = _tokens(joined)
    for bucket, words in _CHAT_BUCKET_KEYWORDS:
        if _has_any(tokens, words):
            return bucket
    return "Accessories"


def infer_style_attributes(item: dict) -> Dict[str, object]:
    """Infer stable taste metadata for scoring without adding another service.

    These values are intentionally deterministic and conservative. Vision/LLM
    tagging can override them later, but the style engine always has a usable
    formality, aesthetic, visual weight, silhouette, and occasion-fitness base.
    """
    if not isinstance(item, dict):
        item = {}
    text = " ".join(
        str(item.get(k, "") or "").lower()
        for k in (
            "name",
            "label",
            "category",
            "sub_category",
            "subcategory",
            "type",
            "pattern",
            "style",
            "material",
            "fabric",
            "color",
            "color_name",
            "occasions",
        )
    )
    category = str(item.get("category") or "").lower()
    sub = str(item.get("sub_category") or item.get("subcategory") or item.get("type") or "").lower()
    pattern = str(item.get("pattern") or "").lower()
    color = str(item.get("color_name") or item.get("color") or item.get("color_code") or "").lower()

    formality = 3
    if any(x in text for x in ("blazer", "suit", "formal", "oxford", "derby", "loafer", "trouser", "tailored", "client", "boardroom", "business")):
        formality += 1
    if any(x in text for x in ("t-shirt", "tshirt", "tee", "shorts", "slides", "slider", "slipper", "sandal", "birkenstock", "crocs", "gym", "running")):
        formality -= 2
    if any(x in text for x in ("sneaker", "denim", "jeans", "polo", "chino")):
        formality -= 1
    if any(x in text for x in ("leather sneaker", "minimal sneaker", "white sneaker", "clean sneaker", "button-down", "button down", "shirt")):
        formality += 1
    formality = max(1, min(5, formality))

    visual_weight = 2
    if any(x in pattern for x in ("print", "pattern", "floral", "graphic", "check", "stripe")):
        visual_weight += 2
    if any(x in color for x in ("red", "orange", "yellow", "lime", "neon", "bright")):
        visual_weight += 1
    if any(x in text for x in ("tropical", "hawaiian", "statement", "chunky", "oversized")):
        visual_weight += 1
    if any(x in color for x in ("black", "white", "grey", "gray", "navy", "beige", "cream")) and pattern in {"", "plain", "solid"}:
        visual_weight -= 1
    visual_weight = max(1, min(5, visual_weight))

    if any(x in text for x in ("tropical", "hawaiian", "graphic", "statement", "printed", "pattern")):
        aesthetic_cluster = "expressive"
    elif any(x in text for x in ("loafer", "oxford", "trouser", "tailored", "blazer", "button-down", "button down")):
        aesthetic_cluster = "classic"
    elif any(x in text for x in ("sneaker", "denim", "jeans", "hoodie", "street")):
        aesthetic_cluster = "street-smart"
    elif any(x in text for x in ("slipper", "slides", "sandal", "birkenstock", "shorts")):
        aesthetic_cluster = "relaxed"
    elif visual_weight <= 2:
        aesthetic_cluster = "minimal"
    else:
        aesthetic_cluster = "polished"

    if any(x in text for x in ("blazer", "structured", "tailored", "trouser", "oxford")):
        silhouette_family = "structured"
    elif any(x in text for x in ("oversized", "loose", "wide", "relaxed")):
        silhouette_family = "relaxed"
    elif any(x in text for x in ("slim", "skinny", "fitted")):
        silhouette_family = "slim"
    elif "dress" in category or "dress" in sub:
        silhouette_family = "one-piece"
    else:
        silhouette_family = "clean"

    occasion_fitness = {
        "office": 0.5,
        "date": 0.5,
        "party": 0.5,
        "travel": 0.5,
        "casual": 0.6,
        "wedding": 0.3,
    }
    if formality >= 4:
        occasion_fitness.update({"office": 0.85, "date": 0.75, "wedding": 0.75})
    if aesthetic_cluster == "expressive":
        occasion_fitness.update({"party": 0.8, "date": 0.65, "office": 0.35})
    if aesthetic_cluster in {"relaxed", "street-smart"}:
        occasion_fitness.update({"casual": 0.85, "travel": 0.75, "office": min(occasion_fitness["office"], 0.45)})
    if any(x in text for x in ("sandal", "slipper", "slides", "birkenstock", "crocs")):
        occasion_fitness.update({"office": 0.1, "date": 0.2, "wedding": 0.1})
    if any(x in text for x in ("loafer", "oxford", "derby", "formal", "leather sneaker", "minimal sneaker")):
        occasion_fitness.update({"office": max(occasion_fitness["office"], 0.85), "date": max(occasion_fitness["date"], 0.75)})

    return {
        "formality": formality,
        "aesthetic_cluster": aesthetic_cluster,
        "visual_weight": visual_weight,
        "silhouette_family": silhouette_family,
        "occasion_fitness": occasion_fitness,
    }


__all__ = (
    "CANONICAL_CATEGORY_KEYWORDS",
    "CANONICAL_CATEGORIES",
    "CHAT_EXPLICIT_MAP",
    "categorize_for_chat",
    "infer_style_attributes",
    "normalize_category_from_label",
)
