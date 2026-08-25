"""Single source of truth for what counts as a *fashion* wardrobe item.

Every wardrobe-to-style transformation must pass through
``sanitize_fashion_wardrobe_items`` before items can reach style prompts,
ownership chips, missing-piece intelligence, editorial boards or any other
stylist-facing surface. A charger appearing inside "You Already Own" breaks
trust more than any styling mistake.

Matching uses category + name heuristics, never metadata alone — wardrobe
rows arrive from multiple clients with inconsistent category labels.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Categories that immediately mark an item as non-fashion regardless of name.
BLOCKED_CATEGORIES: frozenset[str] = frozenset(
    {
        "electronics",
        "electronic",
        "gadget",
        "gadgets",
        "charger",
        "misc",
        "unknown",
        "travel_accessory",
        "travel-accessory",
        "stationery",
        "medicine",
        "supplement",
        "food",
        "drink",
        "skincare",
        "grooming",
    }
)

# Name/category substring tokens that mark an item as non-fashion.
# Compared against a lowercase blob of name + category + tags.
BLOCKED_NAME_TOKENS: tuple[str, ...] = (
    "charger",
    "charging",
    "cable",
    "power bank",
    "powerbank",
    "adapter",
    "headphone",
    "earphone",
    "earbud",
    "airpod",
    "speaker",
    "tripod",
    "laptop stand",
    "mouse",
    "keyboard",
    "razor",
    "shaver",
    "trimmer",
    "comb",
    "toothbrush",
    "toothpaste",
    "deodorant",
    "perfume bottle",
    "bottle",
    "tumbler",
    "flask",
    "neck pillow",
    "travel pillow",
    "eye mask",
    "eyemask",
    "ear plug",
    "earplug",
    "first aid",
    "medicine",
    "supplement",
    "vitamin",
    "notebook",
    "diary",
    "pen ",
    " pen",
    "pencil",
    "umbrella",
    "weighing scale",
    "luggage tag",
    "packing cube",
    "ziploc",
    "sanitizer",
)

# Short or ambiguous tokens that only count as a match when they equal a whole word
# or when no fashion token is present — substring matching would false-positive
# ("bottle" in "Water Bottle Bag", "cap" in "escape", "comb" in "combat boots").
_EXACT_WORD_TOKENS: frozenset[str] = frozenset({"comb", "pen", "mask", "bottle", "sanitizer", "vitamin"})

# Positive fashion signals. An item whose blob hits one of these survives
# even when its category field is junk ("misc", empty, etc.).
FASHION_TOKENS: tuple[str, ...] = (
    # tops
    "shirt", "tshirt", "t-shirt", "tee", "polo", "top", "blouse", "hoodie",
    "sweatshirt", "sweater", "cardigan", "knit", "kurta", "tunic", "camisole",
    # bottoms
    "trouser", "pant", "chino", "jean", "denim", "short", "skirt", "jogger",
    "cargo", "legging", "palazzo",
    # dress / sets
    "dress", "gown", "jumpsuit", "saree", "lehenga", "sherwani", "bandhgala",
    "suit", "blazer", "waistcoat", "nehru",
    # outerwear
    "jacket", "coat", "overshirt", "parka", "windcheater", "shrug",
    # footwear
    "shoe", "sneaker", "loafer", "boot", "sandal", "heel", "flat", "mule",
    "espadrille", "slipper", "slide", "flip flop", "flipflop", "oxford",
    "derby", "brogue",
    # accessories
    "belt", "watch", "sunglass", "scarf", "muffler", "tie", "pocket square",
    "cufflink", "hat", "cap", "beanie", "glove",
    # bags
    "bag", "tote", "backpack", "sling", "clutch", "wallet", "cardholder",
    "briefcase", "duffle", "messenger",
    # jewellery
    "necklace", "bracelet", "earring", "ring", "bangle", "chain", "pendant",
    "brooch", "anklet", "jewellery", "jewelry",
    # skincare / grooming
    "skincare", "skin care", "moisturizer", "moisturiser", "serum", "sunscreen",
    "cleanser", "face wash", "facewash", "lotion", "toner", "grooming", "toiletries",
)

# Category labels that are explicitly fashion. Items with these categories
# pass unless a blocked name token overrides (e.g. category "accessory",
# name "Phone Charger").
FASHION_CATEGORIES: frozenset[str] = frozenset(
    {
        "top", "tops", "bottom", "bottoms", "dress", "dresses", "footwear",
        "shoes", "outerwear", "ethnicwear", "ethnic", "accessory",
        "accessories", "bag", "bags", "watch", "watches", "jewellery",
        "jewelry", "shirt", "tshirt", "t-shirt", "trouser", "trousers",
        "jeans", "jacket", "blazer", "kurta", "saree", "festive",
        "loungewear", "activewear", "innerwear", "skincare", "grooming", "toiletries",
    }
)


def _blob(item: Dict[str, Any]) -> str:
    parts = [
        str(item.get("name") or item.get("title") or ""),
        str(item.get("category") or item.get("type") or item.get("subcategory") or ""),
    ]
    tags = item.get("tags")
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags)
    return " ".join(parts).lower()


def _has_blocked_token(blob: str, category: str = "") -> bool:
    words = set(re.findall(r"[a-z]+", blob))
    # If the item has an explicit fashion category or fashion token (e.g. "bag", "tote", "shoe", "boot"),
    # ambiguous tokens like "bottle" should not false-positive block valid fashion accessories.
    has_fashion_signal = (
        category.strip().lower() in FASHION_CATEGORIES
        or any(ft in blob for ft in FASHION_TOKENS)
    )

    for token in BLOCKED_NAME_TOKENS:
        cleaned = token.strip()
        if cleaned in _EXACT_WORD_TOKENS:
            if cleaned in words and not has_fashion_signal:
                return True
        elif token in blob:
            # If the blob contains a strong fashion token (e.g., "bag", "shoe"), do not block
            # unless the category itself is non-fashion.
            if has_fashion_signal and any(ft in blob for ft in ("bag", "tote", "backpack", "clutch", "shoe", "boot", "sneaker", "sandal", "heel")):
                continue
            return True
    return False


def is_fashion_item(item: Any) -> bool:
    """True when an item is safe to surface in stylist outputs."""
    if not isinstance(item, dict):
        return False
    name = str(item.get("name") or item.get("title") or "").strip()
    if not name:
        return False
    category = str(item.get("category") or item.get("type") or "").strip().lower()
    blob = _blob(item)
    # Hard blocks first: blocked category or blocked token kills the item
    # even when the name sounds garment-ish.
    if category in BLOCKED_CATEGORIES:
        return False
    if _has_blocked_token(blob, category):
        return False
    # Positive signals: explicit fashion category, or a garment token in
    # the name/tags blob.
    if category in FASHION_CATEGORIES:
        return True
    if any(token in blob for token in FASHION_TOKENS):
        return True
    # Unknown category AND no garment token → not trustworthy for styling.
    return False


def sanitize_fashion_wardrobe_items(
    items: Any,
    *,
    source: str = "",
) -> List[Dict[str, Any]]:
    """Filter a raw wardrobe list down to fashion items only.

    Non-dict entries are dropped. Removals are logged at DEBUG with the
    item name/category and the calling source so future leaks are
    traceable without exposing data at INFO.
    """
    if not isinstance(items, list):
        return []
    kept: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if is_fashion_item(item):
            kept.append(item)
        else:
            logger.debug(
                "AHVI_NON_FASHION_ITEM_REMOVED name=%r category=%r source=%s",
                str(item.get("name") or item.get("title") or "")[:60],
                str(item.get("category") or item.get("type") or "")[:40],
                source or "unspecified",
            )
    removed = sum(1 for i in items if isinstance(i, dict)) - len(kept)
    if removed > 0:
        logger.info(
            "AHVI_WARDROBE_SANITIZED kept=%d removed=%d source=%s",
            len(kept),
            removed,
            source or "unspecified",
        )
    return kept
