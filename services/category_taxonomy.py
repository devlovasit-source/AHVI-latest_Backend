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
            "short",
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
            "crop",
            "sweater",
            "hoodie",
            "polo",
        ],
    ),
    ("Dresses", "Dress", ["dress", "gown", "jumpsuit"]),
    (
        "Outerwear",
        "Outerwear",
        ["jacket", "coat", "blazer", "outerwear", "cardigan"],
    ),
    ("Bags", "Bag", ["bag", "handbag", "backpack", "purse", "tote", "clutch"]),
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
            "hoop",
            "hoops",
        ],
    ),
    (
        "Accessories",
        "Accessory",
        ["watch", "watches", "belt", "scarf", "hat", "cap", "sunglass", "sunglasses"],
    ),
    (
        "Indian Wear",
        "Indian Wear",
        ["saree", "kurta", "lehenga", "dupatta", "sherwani"],
    ),
]

CANONICAL_CATEGORIES: set[str] = {row[0] for row in CANONICAL_CATEGORY_KEYWORDS}


def normalize_category_from_label(label: str) -> Tuple[str, str]:
    """Best-effort label -> (category, sub_category) mapping for capture."""
    raw = (label or "").strip().lower()
    if not raw:
        return ("Item", "Item")
    if any(x in raw for x in ["saree", "kurta", "lehenga", "dupatta", "sherwani"]):
        return ("Indian Wear", raw.title() or "Indian Wear")
    if any(
        x in raw
        for x in [
            "shirt",
            "tshirt",
            "t-shirt",
            "top",
            "blouse",
            "crop",
            "sweater",
            "hoodie",
            "tee",
        ]
    ):
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
    if any(
        x in raw
        for x in [
            "jewelry",
            "jewellery",
            "bracelet",
            "ring",
            "earring",
            "necklace",
            "bangle",
            "pendant",
            "chain",
            "hoop",
        ]
    ):
        return ("Jewelry", raw.title() or "Jewelry")
    if any(x in raw for x in ["watch", "belt", "scarf", "hat", "cap", "sunglass"]):
        return ("Accessories", raw.title() or "Accessory")
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


__all__ = (
    "CANONICAL_CATEGORY_KEYWORDS",
    "CANONICAL_CATEGORIES",
    "CHAT_EXPLICIT_MAP",
    "categorize_for_chat",
    "normalize_category_from_label",
)
