"""Canonical style-item contract shared by stylist CTAs and board shuffle.

Pure module: stdlib + typing only.  No pipeline/router/engine imports so it is
safe to import from anywhere (routers, services, tests) without pulling heavy
dependencies.

Provides one canonical answer for:
- item identity        (canonical_item_id)
- item provenance      (canonical_item_source / is_wardrobe_item / is_style_asset)
- item role            (canonical_item_role - ported from
                        brain.engines.wardrobe_selector.WardrobeSelector.normalize_item_role,
                        with the stylist lite-path non-fashion / sport-swim guards)
- accessory subtype    (canonical_accessory_type)
- display image        (canonical_image_url)
- normalized envelope  (normalize_style_item)
- the fixed-item invariant (assert_fixed_items_preserved / FixedItemLostError)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

_ID_KEYS = ("item_id", "id", "$id", "itemId", "image_id", "asset_id")


def canonical_item_id(item: Any) -> str:
    """Extract the canonical id of an item.

    Checked in order: item_id | id | $id | itemId | image_id | asset_id.
    NEVER uses name/label/index/python id().  Returns "" when missing.
    """
    if not isinstance(item, dict):
        return ""
    for key in _ID_KEYS:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


# ---------------------------------------------------------------------------
# Source / provenance
# ---------------------------------------------------------------------------

VALID_SOURCES = frozenset({"wardrobe", "style_asset", "catalog", "generated", "unknown"})

_SOURCE_MAP = {
    # wardrobe family
    "wardrobe": "wardrobe",
    "user_wardrobe": "wardrobe",
    "uploaded": "wardrobe",
    "closet": "wardrobe",
    # style asset family
    "style_asset": "style_asset",
    "asset": "style_asset",
    "asset_library": "style_asset",
    "curated": "style_asset",
    "style-library": "style_asset",
    "style_library": "style_asset",
    # catalog family
    "catalog": "catalog",
    "commerce": "catalog",
    # generated
    "generated": "generated",
}


def canonical_item_source(item: Any) -> str:
    """Normalize an item's source to {wardrobe, style_asset, catalog, generated, unknown}.

    Missing or unrecognized sources map to "unknown".  IMPORTANT: unknown is
    NOT wardrobe - callers that know provenance (e.g. items freshly listed
    from the user's own wardrobe collection) must stamp source explicitly.
    """
    if not isinstance(item, dict):
        return "unknown"
    raw = str(item.get("source") or "").strip().lower()
    if not raw:
        return "unknown"
    return _SOURCE_MAP.get(raw, "unknown")


def is_wardrobe_item(item: Any) -> bool:
    return canonical_item_source(item) == "wardrobe"


def is_style_asset(item: Any) -> bool:
    return canonical_item_source(item) == "style_asset"


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

VALID_ROLES = frozenset({"top", "bottom", "dress", "outerwear", "footwear", "accessory", "unknown"})

# Explicit-field alias map (TYPE_MAP-style, ported from WardrobeSelector).
_TYPE_MAP = {
    "tshirt": "top", "t-shirt": "top", "tshirts": "top", "tee": "top",
    "tees": "top", "shirt": "top", "shirts": "top", "top": "top",
    "tops": "top", "polo": "top", "polos": "top", "kurta": "top",
    "kurtas": "top", "hoodie": "top", "hoodies": "top", "blouse": "top",
    "sweater": "top", "tank": "top", "camisole": "top",

    "bottom": "bottom", "bottoms": "bottom", "pant": "bottom",
    "pants": "bottom", "trouser": "bottom", "trousers": "bottom",
    "jean": "bottom", "jeans": "bottom", "shorts": "bottom",
    "chino": "bottom", "chinos": "bottom", "skirt": "bottom",
    "leggings": "bottom", "joggers": "bottom",

    "shoe": "footwear", "shoes": "footwear", "sneaker": "footwear",
    "sneakers": "footwear", "loafer": "footwear", "loafers": "footwear",
    "heel": "footwear", "heels": "footwear", "boot": "footwear",
    "boots": "footwear", "sandal": "footwear", "sandals": "footwear",
    "slipper": "footwear", "slippers": "footwear", "footwear": "footwear",

    "dress": "dress", "dresses": "dress", "gown": "dress",
    "saree": "dress", "sari": "dress", "jumpsuit": "dress",
    "lehenga": "dress", "one_piece": "dress", "one-piece": "dress",
    "one piece": "dress",

    "outerwear": "outerwear", "jacket": "outerwear", "jackets": "outerwear",
    "coat": "outerwear", "coats": "outerwear", "blazer": "outerwear",
    "blazers": "outerwear", "cardigan": "outerwear", "shrug": "outerwear",

    "accessory": "accessory", "accessories": "accessory",
    "watch": "accessory", "watches": "accessory", "belt": "accessory",
    "belts": "accessory", "bag": "accessory", "bags": "accessory",
    "handbag": "accessory", "handbags": "accessory", "purse": "accessory",
    "clutch": "accessory", "tote": "accessory", "cap": "accessory",
    "caps": "accessory", "hat": "accessory", "hats": "accessory",
    "jewelry": "accessory", "jewellery": "accessory",
    "bracelet": "accessory", "necklace": "accessory", "ring": "accessory",
    "rings": "accessory", "earring": "accessory", "earrings": "accessory",
    "sunglasses": "accessory", "eyewear": "accessory", "scarf": "accessory",
}

_EXPLICIT_FIELDS = (
    "role", "slot", "type", "category", "cat", "category_group",
    "sub_category", "subcategory", "subCategory", "garment_type",
)

_TEXT_FIELDS = ("name", "label", "title", "description")

_TOP_TOKENS = {
    "shirt", "shirts", "tshirt", "tshirts", "tee", "tees", "top", "tops",
    "polo", "polos", "kurta", "kurtas", "hoodie", "hoodies", "blouse",
    "sweater", "tank", "camisole",
}
_BOTTOM_TOKENS = {
    "jeans", "jean", "pants", "pant", "trousers", "trouser", "chinos",
    "chino", "shorts", "skirt", "leggings", "joggers", "bottom", "bottoms",
}
_FOOTWEAR_TOKENS = {
    "footwear", "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
    "loafer", "loafers", "sandal", "sandals", "heel", "heels",
    "slipper", "slippers",
}
_ACCESSORY_TOKENS = {
    "watch", "watches", "belt", "belts", "bag", "bags", "handbag",
    "handbags", "purse", "clutch", "tote", "cap", "caps", "hat", "hats",
    "jewelry", "jewellery", "bracelet", "necklace", "ring", "rings",
    "earring", "earrings", "sunglasses", "eyewear", "scarf", "scarves",
}
_DRESS_TOKENS = {"dress", "dresses", "gown", "saree", "sari", "jumpsuit", "lehenga"}
_OUTERWEAR_TOKENS = {
    "jacket", "jackets", "coat", "coats", "outerwear", "blazer",
    "blazers", "cardigan", "shrug",
}

# Non-fashion junk that gets mis-saved into wardrobes - never a stylable role.
# (Same guard as the stylist lite path so chargers never land in a look.)
_NON_FASHION_TOKENS = (
    "charger", "cable", "adapter", "bottle", "phone", "remote", "mouse",
    "keyboard", "laptop", "earbud", "headphone", "airpod", "power bank",
    "powerbank", "plug", "wire", "battery", "speaker", "camera", "mug",
    "cup", "pen", "book", "box",
)
# Activity / swim / sport gear - real apparel but not stylable into looks.
_SPORT_SWIM_TOKENS = (
    "swim", "goggle", "wetsuit", "snorkel", "flipper", "cleat",
    "shin guard", "helmet", "ski ", "snowboard", "life jacket",
    "life vest", "boxing glove", "swimsuit", "swimwear",
)


def _full_blob(item: Dict[str, Any]) -> str:
    parts = []
    for key in _EXPLICIT_FIELDS + _TEXT_FIELDS:
        parts.append(str(item.get(key) or ""))
    return " ".join(parts).lower()


def canonical_item_role(item: Any) -> str:
    """Normalize an item into {top,bottom,dress,outerwear,footwear,accessory,unknown}.

    Logic ported from brain.engines.wardrobe_selector.WardrobeSelector
    .normalize_item_role (explicit-field alias pass first, then token-aware
    text fallback with phrase guards), plus the stylist lite-path non-fashion
    and sport/swim guards applied FIRST so junk never gets a stylable role.
    """
    if not isinstance(item, dict):
        return "unknown"

    blob_all = _full_blob(item)
    padded = f" {blob_all} "
    if any(tok in padded for tok in _NON_FASHION_TOKENS):
        return "unknown"
    if any(tok in padded for tok in _SPORT_SWIM_TOKENS):
        return "unknown"

    # 1. Trust explicit structured fields first (exact alias match).
    for key in _EXPLICIT_FIELDS:
        value = str(item.get(key) or "").strip().lower()
        normalized = _TYPE_MAP.get(value)
        if normalized:
            return normalized

    # 2. Token-aware text fallback.
    blob = " ".join(str(item.get(k) or "") for k in _TEXT_FIELDS).lower()
    tokens = set(re.sub(r"[^a-z0-9]+", " ", blob).split())

    # Phrase guards: "dress shirt" is a top; "shirt dress" / "maxi dress" /
    # "mini dress" are dresses.
    if "dress shirt" in blob:
        return "top"
    if "shirt dress" in blob or "maxi dress" in blob or "mini dress" in blob:
        return "dress"

    # Tops win over misleading words like "short" or "dress" in shirt names.
    if tokens & _TOP_TOKENS:
        return "top"
    if tokens & _BOTTOM_TOKENS:
        return "bottom"
    if tokens & _FOOTWEAR_TOKENS:
        return "footwear"
    if tokens & _ACCESSORY_TOKENS:
        return "accessory"
    if tokens & _DRESS_TOKENS:
        return "dress"
    if tokens & _OUTERWEAR_TOKENS:
        return "outerwear"
    return "unknown"


# ---------------------------------------------------------------------------
# Accessory subtype
# ---------------------------------------------------------------------------

VALID_ACCESSORY_TYPES = frozenset({
    "bag", "watch", "belt", "necklace", "bracelet", "ring", "earring",
    "eyewear", "headwear", "scarf", "jewelry", "other",
})

_ACCESSORY_TYPE_CHECKS = (
    ("watch", {"watch", "watches"}),
    ("eyewear", {"sunglass", "sunglasses", "eyewear", "glasses"}),
    ("bag", {"bag", "bags", "purse", "handbag", "handbags", "clutch", "tote"}),
    ("bracelet", {"bracelet", "bracelets"}),
    ("ring", {"ring", "rings"}),
    ("necklace", {"necklace", "necklaces"}),
    ("earring", {"earring", "earrings"}),
    ("belt", {"belt", "belts"}),
    ("scarf", {"scarf", "scarves"}),
    ("headwear", {"cap", "caps", "hat", "hats", "beanie", "beret"}),
    ("jewelry", {"jewelry", "jewellery"}),
)


def canonical_accessory_type(item: Any) -> str:
    """Token-based accessory subtype (see reference outfit_pipeline._accessory_type)."""
    if not isinstance(item, dict):
        return "other"
    blob = _full_blob(item)
    tokens = set(re.sub(r"[^a-z0-9]+", " ", blob).split())
    for subtype, wanted in _ACCESSORY_TYPE_CHECKS:
        if tokens & wanted:
            return subtype
    return "other"


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

_IMAGE_KEYS = (
    "normalized_url", "normalizedUrl", "masked_url", "maskedUrl",
    "image_url", "imageUrl", "url", "image",
)


def canonical_image_url(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    for key in _IMAGE_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ---------------------------------------------------------------------------
# Normalized envelope
# ---------------------------------------------------------------------------


def normalize_style_item(item: Any) -> Dict[str, Any]:
    """Normalize an item into the shared envelope.

    Preserves the original id and source values exactly (canonicalized forms
    are what downstream code keys on).  Position (board placement) passes
    through untouched when present.
    """
    src = item if isinstance(item, dict) else {}
    role = canonical_item_role(src)
    normalized: Dict[str, Any] = {
        "item_id": canonical_item_id(src),
        "id": canonical_item_id(src),
        "name": str(src.get("name") or src.get("label") or "").strip() or "Item",
        "role": role,
        "slot": role,
        "source": canonical_item_source(src),
        "image_url": str(src.get("image_url") or "").strip() or canonical_image_url(src),
        "category": str(src.get("category") or "").strip(),
        "sub_category": str(src.get("sub_category") or src.get("subcategory") or "").strip(),
        "color": str(src.get("color") or src.get("color_name") or src.get("color_code") or "").strip(),
    }
    normalized["owned"] = normalized["source"] == "wardrobe"
    if role == "accessory":
        normalized["accessory_type"] = canonical_accessory_type(src)
    for key in (
        "masked_url", "board_image_url", "normalized_url", "position",
        "x", "y", "width", "height", "scale", "z", "rotation",
    ):
        value = src.get(key)
        if value not in (None, ""):
            normalized[key] = value
    return normalized


# ---------------------------------------------------------------------------
# Fixed-item invariant
# ---------------------------------------------------------------------------


class FixedItemLostError(Exception):
    """Raised when a fixed (locked/anchor) item went missing from a candidate."""

    def __init__(
        self,
        missing_ids: List[str],
        stage: str = "unknown",
        mismatched_fields: Optional[Dict[str, List[str]]] = None,
    ):
        self.missing_ids = list(missing_ids)
        self.stage = str(stage)
        self.mismatched_fields = dict(mismatched_fields or {})
        super().__init__(
            f"fixed items lost at stage={self.stage}: {', '.join(self.missing_ids)}"
        )


def assert_fixed_items_preserved(
    fixed_items: Iterable[Any],
    candidate_items: Iterable[Any],
    stage: str = "unknown",
) -> None:
    """Raise FixedItemLostError when any fixed item's canonical id is absent
    from candidate_items.  Fixed items with empty ids are the caller's error
    and are reported as missing (empty id can never be verified as present).
    """
    candidates = {
        canonical_item_id(i): i
        for i in (candidate_items or [])
        if canonical_item_id(i)
    }
    missing: List[str] = []
    mismatched: Dict[str, List[str]] = {}
    for fixed in fixed_items or []:
        fid = canonical_item_id(fixed)
        if not fid or fid not in candidates:
            missing.append(fid or "<missing-id>")
            continue
        candidate = candidates[fid]
        fields: List[str] = []
        if canonical_item_source(fixed) != canonical_item_source(candidate):
            fields.append("source")
        fixed_image = str(fixed.get("image_url") or "").strip()
        candidate_image = str(candidate.get("image_url") or "").strip()
        if fixed_image and fixed_image != candidate_image:
            fields.append("image_url")
        elif not fixed_image and canonical_image_url(fixed) != canonical_image_url(candidate):
            fields.append("image_url")
        if canonical_item_role(fixed) != canonical_item_role(candidate):
            fields.append("slot")
        if "shuffle" in str(stage).lower():
            for placement in ("position", "x", "y", "width", "height", "scale", "z", "rotation"):
                if fixed.get(placement) != candidate.get(placement):
                    fields.append(placement)
        if fields:
            mismatched[fid] = fields
    if missing or mismatched:
        raise FixedItemLostError(missing or sorted(mismatched), stage=stage, mismatched_fields=mismatched)
