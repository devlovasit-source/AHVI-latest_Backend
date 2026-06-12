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
    "Skincare",
    "Makeup",
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


def _skincare_subcategory(name: Any, sub_category: Any) -> str:
    """Pick a sensible subcategory for a Skincare item.

    Falls back to the user's own sub_category text, then to 'Skincare'
    so the column is never empty.
    """
    blob = " ".join([str(name or ""), str(sub_category or "")]).lower()
    if "sunscreen" in blob or "spf" in blob or "sunblock" in blob:
        return "Sunscreen"
    if "moisturizer" in blob or "moisturiser" in blob or "lotion" in blob:
        return "Moisturizer"
    if "serum" in blob:
        return "Serum"
    if "cleanser" in blob or "face wash" in blob or "facewash" in blob:
        return "Cleanser"
    if "toner" in blob:
        return "Toner"
    if "exfoliator" in blob or "scrub" in blob:
        return "Exfoliator"
    if "mask" in blob:
        return "Mask"
    raw_sub = str(sub_category or "").strip()
    return raw_sub or "Skincare"


def _makeup_subcategory(name: Any, sub_category: Any) -> str:
    blob = " ".join([str(name or ""), str(sub_category or "")]).lower()
    if "lipstick" in blob:
        return "Lipstick"
    if "lip gloss" in blob or "lipgloss" in blob:
        return "Lip Gloss"
    if "lip balm" in blob or "lipbalm" in blob:
        return "Lip Balm"
    if "mascara" in blob:
        return "Mascara"
    if "eyeliner" in blob or "kajal" in blob:
        return "Eyeliner"
    if "foundation" in blob:
        return "Foundation"
    if "concealer" in blob:
        return "Concealer"
    if "blush" in blob:
        return "Blush"
    if "bronzer" in blob:
        return "Bronzer"
    if "highlighter" in blob:
        return "Highlighter"
    if "eyeshadow" in blob:
        return "Eyeshadow"
    if "compact" in blob or "primer" in blob:
        return "Primer"
    raw_sub = str(sub_category or "").strip()
    return raw_sub or "Makeup"


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

    # Honour explicit user-supplied category for non-garment categories
    # FIRST. When a user opens the edit dialog and picks 'Skincare' or
    # 'Makeup' for a sunscreen / lipstick, we must keep that choice even
    # though the item's name doesn't contain a recognised garment token.
    explicit_raw = str(category or "").strip().lower()
    if explicit_raw in {"skincare", "skin care"}:
        return "Skincare", _skincare_subcategory(name, sub_category)
    if explicit_raw in {"makeup", "make up", "make-up", "cosmetics"}:
        return "Makeup", _makeup_subcategory(name, sub_category)
    # Strong product keyword detection — covers items whose name implies
    # the category even when the user didn't pick it explicitly.
    if has(["sunscreen", "spf", "moisturizer", "moisturiser", "serum",
            "cleanser", "toner", "facewash", "face wash", "lotion",
            "sunblock", "exfoliator", "retinol", "vitamin c"]):
        return "Skincare", _skincare_subcategory(name, sub_category)
    if has(["lipstick", "lip gloss", "lipgloss", "lip balm", "mascara",
            "eyeliner", "foundation", "concealer", "blush", "bronzer",
            "highlighter", "kajal", "eyeshadow", "compact", "primer"]):
        return "Makeup", _makeup_subcategory(name, sub_category)

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
    if explicit_raw in {"needs review", "needs_review", "review"}:
        return "Needs Review", "Needs Review"

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
        "accessories": ("Accessories", "Accessory"),
        "accessory": ("Accessories", "Accessory"),
        "skincare": ("Skincare", "Skincare"),
        "skin care": ("Skincare", "Skincare"),
        "makeup": ("Makeup", "Makeup"),
        "make up": ("Makeup", "Makeup"),
        "make-up": ("Makeup", "Makeup"),
        "cosmetics": ("Makeup", "Makeup"),
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


# ---------------------------------------------------------------------------
# Deterministic preview taxonomy guard
# ---------------------------------------------------------------------------
# Runs AFTER Ollama vision, AFTER Gemini preview validator merge, and AFTER
# _normalize_capture_preview_item so well-known garments never depend on
# Gemini being available to land in the correct category. Wins over the
# default_metadata() fallback ("Traditional"/"Accessories") that older
# heuristics produced.

_PREVIEW_TEXT_FIELDS = (
    "name", "label", "title",
    "sub_category", "subcategory", "subCategory",
    "category",
)


def _preview_blob(item: Dict[str, Any]) -> str:
    return " ".join(str(item.get(f) or "") for f in _PREVIEW_TEXT_FIELDS).lower()


def _has_token(blob: str, *tokens: str) -> bool:
    return any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", blob) for t in tokens)


def enforce_preview_taxonomy(item: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic taxonomy + suitability override for known garments.

    Saree/sari, lehenga, one-piece, boxers/briefs/underwear, and
    pajama/sleepwear are hard-coded so the preview is correct even if
    Ollama mislabels them and Gemini is unavailable.
    """
    if not isinstance(item, dict):
        return item
    blob = _preview_blob(item)
    out = dict(item)

    def _public(cat: str, sub: str) -> Dict[str, Any]:
        out["category"] = cat
        out["sub_category"] = sub
        out["subcategory"] = sub
        out["subCategory"] = sub
        out["privateWear"] = False
        out["publicWear"] = True
        out["styleEligible"] = True
        return out

    def _private(cat: str) -> Dict[str, Any]:
        out["category"] = cat
        out["privateWear"] = True
        out["publicWear"] = False
        out["styleEligible"] = False
        return out

    if _has_token(blob, "saree", "sari", "sarees", "saris"):
        return _public("Dresses", "Saree")
    if _has_token(blob, "lehenga", "lehengas"):
        return _public("Dresses", "Lehenga")
    if "one-piece" in blob or "one piece" in blob:
        return _public("Dresses", "One-piece Dress")
    if _has_token(
        blob, "boxer", "boxers", "brief", "briefs", "underwear", "innerwear"
    ):
        return _private("Innerwear")
    if _has_token(
        blob, "pajama", "pajamas", "pyjama", "pyjamas", "nightwear", "sleepwear"
    ):
        return _private("Sleepwear")
    return out


__all__ = [
    "CATEGORIES",
    "normalize",
    "normalize_item",
    "display_name",
    "build_review_card",
    "enforce_preview_taxonomy",
]
