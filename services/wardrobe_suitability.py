from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List

logger = logging.getLogger("ahvi.wardrobe_suitability")

PRIVATE_WEAR_ALIASES = {
    "boxer",
    # Plural forms — alias matching is whole-word, so "Cotton Boxers"
    # missed the singular alias and previewed as public shorts.
    "boxers",
    "boxer shorts",
    "briefs",
    "mens brief",
    "underwear",
    "undergarment",
    "innerwear",
    "trunks",
    "sports trunk",
    "compression shorts",
    "compression short",
    "base layer",
    "thermal inner",
    "lingerie",
    "sleep shorts",
    "pajama",
    "pajamas",
    "pyjama",
    "pyjamas",
    "nightwear",
    "sleepwear",
    "lounge shorts",
}

PRIVATE_NORMALIZED_CATEGORIES = {
    "innerwear",
    "underwear",
    "private_wear",
    "private wear",
    "sleepwear",
    "base_layer",
    "base layer",
}

EXCLUDED_PUBLIC_CONTEXTS = [
    "office",
    "work",
    "client_meeting",
    "formal",
    "presentation",
    "interview",
    "business_dinner",
    "networking",
    "date",
    "dinner",
    "public_outfit",
]

INVALID_PUBLIC_OCCASIONS = {
    "work",
    "office",
    "client meeting",
    "client_meeting",
    "dinner",
    "date",
    "formal",
    "party",
    "public travel",
    "public_travel",
    "travel",
}

ALLOWED_PRIVATE_OCCASIONS = ["Home", "Private", "Lounge", "Base Layer"]


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _iter_text(value: Any, depth: int = 0) -> Iterable[str]:
    if depth > 3:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"image", "image_url", "imageurl", "masked_url", "raw_image_base64"}:
                continue
            yield from _iter_text(child, depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_text(child, depth + 1)
    elif value is not None:
        text = str(value).strip()
        if text:
            yield text


def _contains_private_alias(text: str) -> bool:
    clean = f" {_norm(text)} "
    return any(f" {_norm(alias)} " in clean for alias in PRIVATE_WEAR_ALIASES)


def item_text_blob(item: Dict[str, Any]) -> str:
    return " ".join(_iter_text(item or {}))


def is_private_wear_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    # Only garment-identifying fields — NOT prose (story/explanation/why_it_works
    # /styling_tip). LLM board copy legitimately says "base layer" / "layer this",
    # and item_text_blob (which scans everything) was matching the private alias
    # "base layer" and false-flagging entire office boards as private wear.
    blob = _norm(
        " ".join(
            str(item.get(k) or "")
            for k in (
                "name", "title", "label", "garment_type", "type",
                "category", "sub_category", "subcategory",
                "normalizedCategory", "normalized_category",
                "normalizedSubCategory", "normalized_sub_category",
                "style_role",
            )
        )
    )
    if _contains_private_alias(blob):
        return True
    normalized = _norm(
        item.get("normalizedCategory")
        or item.get("normalized_category")
        or item.get("normalizedSubCategory")
        or item.get("normalized_sub_category")
        or item.get("category")
        or item.get("sub_category")
        or item.get("subcategory")
    )
    return normalized in PRIVATE_NORMALIZED_CATEGORIES


def _clean_private_occasions(values: Any) -> List[str]:
    raw = values if isinstance(values, list) else []
    kept: List[str] = []
    removed: List[str] = []
    allowed = {_norm(v) for v in ALLOWED_PRIVATE_OCCASIONS}
    for value in raw:
        text = str(value or "").strip()
        if not text:
            continue
        if _norm(text) in allowed:
            kept.append(text)
        else:
            removed.append(text)
    if removed:
        logger.info("wardrobe_invalid_occasion_removed removed=%s", removed)
    return kept or list(ALLOWED_PRIVATE_OCCASIONS)


def apply_metadata_guard(item: Dict[str, Any], *, source: str = "") -> Dict[str, Any]:
    guarded = dict(item or {})
    if not is_private_wear_item(guarded):
        return guarded

    logger.info(
        "wardrobe_private_wear_detected source=%s name=%s category=%s sub_category=%s",
        source,
        guarded.get("name") or guarded.get("label"),
        guarded.get("category"),
        guarded.get("sub_category") or guarded.get("subcategory"),
    )
    guarded.update(
        {
            "category": "Innerwear",
            "sub_category": "Private Wear",
            "subcategory": "Private Wear",
            "subCategory": "Private Wear",
            "normalizedCategory": "innerwear",
            "normalizedSubCategory": "private_wear",
            "publicWear": False,
            "styleEligible": False,
            "privateWear": True,
            "excludedContexts": list(EXCLUDED_PUBLIC_CONTEXTS),
        }
    )
    guarded["occasions"] = _clean_private_occasions(
        guarded.get("occasions") or guarded.get("occasion_tags") or guarded.get("tags")
    )
    guarded["occasion_tags"] = list(guarded["occasions"])
    guarded["tags"] = list(guarded["occasions"])
    suitability = {
        "publicWear": False,
        "styleEligible": False,
        "privateWear": True,
        "normalizedCategory": "innerwear",
        "normalizedSubCategory": "private_wear",
        "excludedContexts": list(EXCLUDED_PUBLIC_CONTEXTS),
    }
    meta = guarded.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    meta["suitability"] = suitability
    guarded["metadata"] = meta
    style_meta = guarded.get("style_metadata")
    if isinstance(style_meta, str):
        try:
            style_meta = json.loads(style_meta)
        except Exception:
            style_meta = {}
    if not isinstance(style_meta, dict):
        style_meta = {}
    style_meta["suitability"] = suitability
    guarded["style_metadata"] = style_meta
    logger.info("wardrobe_metadata_guard_applied source=%s", source)
    logger.info("wardrobe_style_eligible_false source=%s", source)
    return guarded


def is_style_eligible(item: Dict[str, Any], context: Any = None) -> bool:
    if not isinstance(item, dict):
        return False
    if is_private_wear_item(item):
        return False
    if item.get("publicWear") is False or item.get("styleEligible") is False or item.get("privateWear") is True:
        return False
    normalized = _norm(
        item.get("normalizedCategory")
        or item.get("normalized_category")
        or item.get("category")
        or item.get("sub_category")
        or item.get("subcategory")
    )
    if normalized in PRIVATE_NORMALIZED_CATEGORIES:
        return False
    ctx = _norm(context)
    excluded = {_norm(x) for x in (item.get("excludedContexts") or [])}
    if ctx and ctx in excluded:
        return False
    return True


def outfit_contains_private_wear(outfit: Dict[str, Any]) -> bool:
    if not isinstance(outfit, dict):
        return False
    if is_private_wear_item(outfit):
        return True
    for key in ("items", "wardrobe_items", "accessories", "layers"):
        values = outfit.get(key)
        if isinstance(values, list) and any(is_private_wear_item(v) for v in values if isinstance(v, dict)):
            return True
    return False


def filter_style_eligible_items(items: Any, context: Any = None) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    kept: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and is_style_eligible(item, context):
            kept.append(item)
        elif isinstance(item, dict):
            logger.info(
                "editorial_guard_rejected_private_wear context=%s item=%s",
                context,
                item.get("name") or item.get("label") or item.get("category"),
            )
    return kept
