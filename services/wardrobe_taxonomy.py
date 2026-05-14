"""Canonical wardrobe taxonomy.

One place. Capture preview, save, and update-labels all flow through here.
Unknown / low-confidence items NEVER default to Accessories — they fall
through to "Needs Review" so the user can correct them manually.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


CATEGORIES = {
    "Tops",
    "Bottoms",
    "Dresses",
    "Outerwear",
    "Footwear",
    "Bags",
    "Jewelry",
    "Accessories",
    "Traditional",
    "Needs Review",
}


def _tokens(value: Any) -> List[str]:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip().split()


def _has_any(text: str, tokens: List[str], words: List[str]) -> bool:
    token_set = set(tokens)
    return any(word in token_set for word in words) or any(
        word in text for word in words
    )


def _blob(*parts: Any) -> Tuple[str, List[str]]:
    text = " ".join(str(p or "") for p in parts).lower()
    return text, _tokens(text)


def normalize(
    category: Any = "",
    name: Any = "",
    sub_category: Any = "",
    confidence: Any = None,
) -> Tuple[str, str]:
    """Return (canonical_category, canonical_subcategory).

    Strong garment/accessory signals win over weak explicit "Tops" /
    "Accessories" labels from older clients or low-confidence vision.
    """
    text, toks = _blob(category, name, sub_category)

    def has(words: List[str]) -> bool:
        return _has_any(text, toks, words)

    # Saree / lehenga first — these are Dresses, NOT Accessories or Traditional.
    if has(["saree", "sari"]):
        return "Dresses", "Saree"
    if has(["lehenga"]):
        return "Dresses", "Lehenga"
    if has(["gown"]):
        return "Dresses", "Gown"
    if has(["jumpsuit"]):
        return "Dresses", "Jumpsuit"

    # Other ethnic — keep in Traditional so styling rules can target it.
    if has(["sherwani"]):
        return "Traditional", "Sherwani"
    if has(["anarkali"]):
        return "Traditional", "Anarkali"
    if has(["kurta", "kurti"]):
        return "Tops", "Kurta"
    if has(["dupatta"]):
        return "Accessories", "Dupatta"
    if has(["salwar", "churidar"]):
        return "Bottoms", "Salwar"

    # Dresses
    if has(["one piece", "one-piece"]):
        return "Dresses", "One-Piece Dress"
    if has(["mini dress"]):
        return "Dresses", "Mini Dress"
    if has(["midi dress"]):
        return "Dresses", "Midi Dress"
    if has(["maxi dress"]):
        return "Dresses", "Maxi Dress"
    if has(["dress", "dresses"]):
        return "Dresses", "Dress"

    # Bags
    if has(["handbag"]):
        return "Bags", "Handbag"
    if has(["tote"]):
        return "Bags", "Tote Bag"
    if has(["shoulder bag", "sling", "crossbody", "cross body"]):
        return "Bags", "Shoulder Bag"
    if has(["clutch"]):
        return "Bags", "Clutch"
    if has(["backpack"]):
        return "Bags", "Backpack"
    if has(["bag", "bags", "purse"]):
        return "Bags", "Bag"

    # Jewelry
    if has(["ring", "rings"]):
        return "Jewelry", "Ring"
    if has(["bracelet", "bracelets", "bangle", "bangles"]):
        return "Jewelry", "Bracelet"
    if has(["necklace", "necklaces", "chain", "pendant"]):
        return "Jewelry", "Necklace"
    if has(["earring", "earrings"]):
        return "Jewelry", "Earrings"
    if has(["jewelry", "jewellery"]):
        return "Jewelry", "Jewelry"

    # Watches / belts / eyewear stay in Accessories.
    if has(["watch", "watches"]):
        return "Accessories", "Watch"
    if has(["belt", "belts"]):
        return "Accessories", "Belt"
    if has(["sunglass", "sunglasses", "eyewear", "glasses"]):
        return "Accessories", "Eyewear"
    if has(["cap", "hat", "beanie"]):
        return "Accessories", "Headwear"
    if has(["scarf", "stole"]):
        return "Accessories", "Scarf"

    # Footwear
    if has([
        "sneaker", "sneakers", "boot", "boots", "loafer", "loafers",
        "heel", "heels", "sandal", "sandals", "slipper", "slippers",
        "slide", "slides", "slider", "sliders", "espadrille", "espadrilles",
        "oxford", "derby", "moccasin", "shoe", "shoes", "footwear",
    ]):
        return "Footwear", "Footwear"

    # Tops
    if has([
        "polo", "polos", "shirt", "shirts", "tee", "tshirt", "tshirts",
        "blouse", "blouses", "camisole", "cami", "tank", "top", "tops",
        "hoodie", "hoodies", "sweater", "sweaters",
    ]):
        return "Tops", "Top"

    # Bottoms
    if has([
        "pants", "pant", "trousers", "trouser", "jeans", "jean",
        "shorts", "skirt", "skirts", "legging", "leggings",
        "chino", "chinos", "joggers", "jogger",
    ]):
        return "Bottoms", "Bottom"

    # Outerwear
    if has(["jacket", "coat", "blazer", "outerwear", "cardigan", "overshirt", "parka"]):
        return "Outerwear", "Outerwear"

    # Explicit category passthrough — only AFTER strong garment signals.
    explicit = str(category or "").strip().lower()
    explicit_map = {
        "tops": ("Tops", "Top"),
        "top": ("Tops", "Top"),
        "bottoms": ("Bottoms", "Bottom"),
        "bottom": ("Bottoms", "Bottom"),
        "footwear": ("Footwear", "Footwear"),
        "shoe": ("Footwear", "Footwear"),
        "shoes": ("Footwear", "Footwear"),
        "outerwear": ("Outerwear", "Outerwear"),
        "dresses": ("Dresses", "Dress"),
        "dress": ("Dresses", "Dress"),
        "indian wear": ("Traditional", "Traditional"),
        "traditional": ("Traditional", "Traditional"),
        "bags": ("Bags", "Bag"),
        "bag": ("Bags", "Bag"),
        "jewelry": ("Jewelry", "Jewelry"),
        "jewellery": ("Jewelry", "Jewelry"),
    }
    if explicit in explicit_map:
        return explicit_map[explicit]

    # Low confidence with no strong signal → Needs Review.
    try:
        conf = float(confidence) if confidence is not None else 1.0
    except Exception:
        conf = 1.0
    if conf < 0.45:
        return "Needs Review", "Needs Review"

    # Final fallback NEVER lands in Accessories silently.
    return "Needs Review", "Needs Review"


def display_name(name: Any, category: str, sub_category: str) -> str:
    raw = str(name or "").strip()
    raw = raw.replace("Sari", "Saree").replace("sari", "Saree")
    if raw:
        return raw
    if sub_category:
        return sub_category
    if category and category != "Needs Review":
        return category
    return "Item"


def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate-safe taxonomy normalization for a single detected/saved item."""
    if not isinstance(item, dict):
        return item

    result = dict(item)
    confidence = (
        result.get("confidence")
        or result.get("vision_confidence")
        or result.get("score")
    )
    category, sub_category = normalize(
        category=result.get("category"),
        name=result.get("name") or result.get("label") or result.get("title"),
        sub_category=result.get("sub_category") or result.get("subcategory"),
        confidence=confidence,
    )
    result["category"] = category
    result["sub_category"] = sub_category
    result["subcategory"] = sub_category
    result["name"] = display_name(
        result.get("name") or result.get("label") or result.get("title"),
        category,
        sub_category,
    )

    if category == "Needs Review":
        result["requires_manual_entry"] = True
        result["needs_review"] = True

    return result


def build_review_card(image_url: str = "", image_base64: str = "", reason: str = "") -> Dict[str, Any]:
    """Card returned when vision detection yields nothing usable but image exists."""
    return {
        "name": "Review item",
        "label": "Review item",
        "category": "Needs Review",
        "sub_category": "Needs Review",
        "subcategory": "Needs Review",
        "requires_manual_entry": True,
        "needs_review": True,
        "image_url": image_url,
        "masked_url": image_url,
        "raw_url": image_url,
        "masked_image_base64": image_base64,
        "raw_image_base64": image_base64,
        "review_reason": reason or "low_confidence",
    }


__all__ = [
    "CATEGORIES",
    "normalize",
    "normalize_item",
    "display_name",
    "build_review_card",
]
