import base64
import hashlib
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from services.category_taxonomy import infer_style_attributes

try:
    from brain.engines.occasion_interpreter import interpret_occasion_context
    from brain.engines.occasion_style_rules import (
        detect_wardrobe_gap,
        get_occasion_rule,
        reject_board_for_occasion,
        score_item_for_occasion,
    )
except Exception:  # pragma: no cover - keeps legacy style flow bootable
    interpret_occasion_context = None
    detect_wardrobe_gap = None
    get_occasion_rule = None
    reject_board_for_occasion = None
    score_item_for_occasion = None

try:
    from brain.engines.outfit_quality_guard import (
        reject_board_for_occasion as reject_quality_board_for_occasion,
    )
except Exception:  # pragma: no cover - optional during partial deploys
    reject_quality_board_for_occasion = None

try:
    from services.r2_storage import R2Storage, R2StorageError
except Exception:  # pragma: no cover - optional deploy dependency
    R2Storage = None
    R2StorageError = Exception

logger = logging.getLogger("ahvi.style_flow")


STYLE_ACTION_CHIPS = ["More looks", "Next best options", "Try different shoes"]
TRUST_LAYER_RUNTIME_INFERENCE = str(os.getenv("AHVI_TRUST_LAYER_RUNTIME_INFERENCE", "")).lower() in {"1", "true", "yes"}

_OCCASION_FORBIDDEN_OFFICE = [
    "boardroom",
    "professional",
    "office",
    "executive",
    "corporate",
    "friday",
    "client",
]

_OCCASION_CARD_LANGUAGE = {
    "date": {
        "badge": "DATE NIGHT",
        "titles": [
            "Soft Statement",
            "Evening Ease",
            "Dinner Polish",
            "After-Dark Edit",
            "Quiet Confidence",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE,
    },
    "date_night": {
        "badge": "DATE NIGHT",
        "titles": [
            "Soft Statement",
            "Evening Ease",
            "Dinner Polish",
            "After-Dark Edit",
            "Quiet Confidence",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE,
    },
    "beach": {
        "badge": "BEACH",
        "titles": [
            "Coastal Ease",
            "Resort Casual",
            "Light Vacation",
            "Beach Ready",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE + ["formal"],
    },
    "office": {
        "badge": "OFFICE",
        "titles": [
            "Boardroom Casual",
            "Creative Professional",
            "Clean Friday",
            "Executive Minimal",
        ],
        "forbidden_title_words": [],
    },
    "daily": {
        "badge": "DAILY",
        "titles": [
            "Polished Neutral",
            "Sharp Daily",
            "Smart Ease",
            "Clean Edit",
            "Refined Casual",
            "Easy Daily",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE,
    },
    "casual": {
        "badge": "CASUAL",
        "titles": [
            "Polished Neutral",
            "Sharp Daily",
            "Smart Ease",
            "Clean Edit",
            "Refined Casual",
            "Easy Daily",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE + ["formal"],
    },
    "temple_modest": {
        "badge": "TEMPLE",
        "titles": [
            "Modest Grace",
            "Soft Traditional",
            "Temple Ready",
            "Respectful Ease",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE + ["club", "party", "strapless", "crop"],
    },
}


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(value: Any) -> set[str]:
    import re

    return set(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


_CANONICAL_OCCASIONS = {
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
    "daily",
    "temple_modest",
}


def _normalize_occasion_value(value: Any, query: Any = "") -> str:
    text = " ".join([_safe_text(value), _safe_text(query)]).lower().replace("-", "_")
    if any(k in text for k in ("temple_modest", "temple", "mandir", "pooja", "puja", "religious", "shrine", "darshan")):
        return "temple_modest"
    if any(k in text for k in ("date_night", "date night", "date", "dinner", "tonight")):
        return "date_night"
    if any(k in text for k in ("beach", "pool", "seaside", "coastal", "resort")):
        return "beach"
    if any(k in text for k in ("office", "work", "meeting", "client", "business", "corporate", "boardroom")):
        return "office"
    if "brunch" in text:
        return "brunch"
    if any(k in text for k in ("rave", "club")):
        return "rave"
    if "cocktail" in text:
        return "cocktail"
    if any(k in text for k in ("party", "house_party", "after_hours", "night out")):
        return "house_party"
    if any(k in text for k in ("travel", "airport", "flight", "vacation", "trip")):
        return "travel"
    if any(k in text for k in ("workout", "gym", "fitness", "training", "yoga", "running")):
        return "workout"
    if any(k in text for k in ("wedding", "reception", "ceremony", "sangeet", "formal event", "event")):
        return "wedding"
    if any(k in text for k in ("daily", "today")):
        return "daily"
    if any(k in text for k in ("casual", "weekend", "errand", "coffee")):
        return "casual"
    return _safe_text(value).lower() if _safe_text(value).lower() in _CANONICAL_OCCASIONS else ""


def apply_occasion_card_language(cards: List[Dict[str, Any]], occasion: str) -> List[Dict[str, Any]]:
    normalized = _normalize_occasion_value(occasion)
    lang = _OCCASION_CARD_LANGUAGE.get(normalized)
    if not lang:
        for card in cards or []:
            if isinstance(card, dict) and normalized:
                card.setdefault("occasion", normalized)
        return cards

    titles = list(lang.get("titles") or [])
    badge = _safe_text(lang.get("badge"))
    forbidden = [_safe_text(w).lower() for w in (lang.get("forbidden_title_words") or [])]

    for idx, card in enumerate(cards or []):
        if not isinstance(card, dict):
            continue
        title = _safe_text(card.get("title")).lower()
        if titles and (not title or any(word and word in title for word in forbidden)):
            card["title"] = titles[idx % len(titles)]
            card["name"] = card["title"]
        if badge:
            card["badge"] = badge
            card["occasion_label"] = badge
        card.setdefault("occasion", normalized)
    return cards


def _filter_boards_for_occasion(cards: List[Dict[str, Any]], occasion: str) -> List[Dict[str, Any]]:
    if reject_quality_board_for_occasion is None:
        return cards
    normalized = _normalize_occasion_value(occasion)
    filtered: List[Dict[str, Any]] = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        try:
            rejected, reason = reject_quality_board_for_occasion(card, normalized)
        except Exception:
            rejected, reason = False, ""
        if rejected:
            logger.info(
                "ahvi.board_rejected occasion=%s reason=%s title=%s",
                normalized,
                reason,
                card.get("title"),
            )
            continue
        filtered.append(card)
    return filtered


def _ahvi_item_blob(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    parts = []
    for key in (
        "id",
        "name",
        "title",
        "category",
        "subcategory",
        "type",
        "color",
        "material",
        "pattern",
        "tags",
        "style_tags",
        "details",
    ):
        value = item.get(key)
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(v or "") for v in value)
        else:
            parts.append(str(value or ""))
    return " ".join(parts).lower()


def _ahvi_slot_counts(items: list[dict]) -> dict:
    counts = {
        "top": 0,
        "bottom": 0,
        "footwear": 0,
        "accessory": 0,
        "outerwear": 0,
        "total": 0,
    }
    for item in items or []:
        blob = _ahvi_item_blob(item)
        counts["total"] += 1
        if any(
            w in blob
            for w in [
                "shirt",
                "t-shirt",
                "tee",
                "top",
                "polo",
                "kurta",
                "blouse",
                "sweater",
                "hoodie",
            ]
        ):
            counts["top"] += 1
        elif any(w in blob for w in ["pant", "trouser", "jean", "shorts", "skirt", "bottom"]):
            counts["bottom"] += 1
        elif any(
            w in blob
            for w in [
                "shoe",
                "sneaker",
                "loafer",
                "boot",
                "sandal",
                "slide",
                "footwear",
                "slipper",
            ]
        ):
            counts["footwear"] += 1
        elif any(w in blob for w in ["blazer", "jacket", "coat", "overshirt"]):
            counts["outerwear"] += 1
        elif any(
            w in blob
            for w in [
                "watch",
                "belt",
                "bag",
                "jewelry",
                "jewellery",
                "ring",
                "bracelet",
                "cap",
                "eyewear",
                "sunglasses",
                "chain",
            ]
        ):
            counts["accessory"] += 1
    return counts


def _ahvi_has_core_slots(slot_counts: dict) -> bool:
    return (
        int(slot_counts.get("top", 0) or 0) > 0
        and int(slot_counts.get("bottom", 0) or 0) > 0
        and int(slot_counts.get("footwear", 0) or 0) > 0
    )


def _ahvi_missing_core_slots_response(slot_counts: dict) -> dict:
    missing = []
    if int(slot_counts.get("top", 0) or 0) <= 0:
        missing.append("top")
    if int(slot_counts.get("bottom", 0) or 0) <= 0:
        missing.append("bottom")
    if int(slot_counts.get("footwear", 0) or 0) <= 0:
        missing.append("footwear")
    return {
        "success": True,
        "type": "missing_core_wardrobe_slots",
        "message": (
            "I couldn't build a complete style board from your wardrobe yet. "
            "Please add at least one top, bottom, and footwear item."
        ),
        "cards": [],
        "style_boards": [],
        "data": {
            "missing_slots": missing,
            "slot_counts": slot_counts,
        },
        "chips": [
            "Add wardrobe item",
            "Upload outfit pieces",
            "Try again",
        ],
    }


def _ahvi_missing_occasion_response(
    occasion: str,
    slot_counts: dict,
    closest_board: dict | None = None,
) -> dict:
    normalized = str(occasion or "").lower()
    if normalized in {"date", "date_night"}:
        message = (
            "I don't see enough strong date-night options yet. "
            "I'd avoid forcing office styling into an evening brief."
        )
        missing_items = [
            {
                "label": "Evening shirt",
                "reason": "Adds date-night personality without looking corporate.",
                "cta": "Find this",
            },
            {
                "label": "Clean casual footwear",
                "reason": "Keeps the look polished without turning office-heavy.",
                "cta": "Find this",
            },
            {
                "label": "One intentional accessory",
                "reason": "Finishes the look without clutter.",
                "cta": "Find this",
            },
        ]
        chips = [
            "Show closest option",
            "Find evening shirt",
            "Find clean casual footwear",
        ]
    elif normalized == "beach":
        message = (
            "I don't see enough beach-ready pieces yet. "
            "I'd rather not force formal trousers or loafers into a beach brief."
        )
        missing_items = [
            {
                "label": "Linen shirt",
                "reason": "Adds breathable coastal texture.",
                "cta": "Find this",
            },
            {
                "label": "Sandals or slides",
                "reason": "Makes the look sand-friendly.",
                "cta": "Find this",
            },
            {
                "label": "Relaxed shorts",
                "reason": "Fixes the silhouette for beachwear.",
                "cta": "Find this",
            },
        ]
        chips = [
            "Show closest option",
            "Find linen shirt",
            "Find sandals",
        ]
    else:
        message = (
            "I don't see enough occasion-ready options yet. "
            "I'd rather not force a weak look."
        )
        missing_items = []
        chips = [
            "Show closest option",
            "Try another occasion",
        ]
    payload = {
        "success": True,
        "type": "missing_occasion_wardrobe",
        "message": message,
        "cards": [],
        "style_boards": [],
        "data": {
            "occasion": normalized,
            "slot_counts": slot_counts,
            "missing_items": missing_items,
            "closest_safe_brief": (
                "evening casual"
                if normalized in {"date", "date_night"}
                else "light casual"
            ),
        },
        "chips": chips,
    }
    if closest_board:
        payload["message"] = (
            "I don't see strong occasion-ready pieces yet, but I found one safe direction."
        )
        payload["cards"] = [closest_board]
        payload["style_boards"] = [closest_board]
        payload["data"]["closest_board"] = closest_board
    return payload


def _ahvi_pick_closest_safe_board(cards: list[dict], occasion: str) -> dict | None:
    normalized = str(occasion or "").lower()
    if not cards:
        return None
    blocked_by_occasion = {
        "date": [
            "boardroom",
            "professional",
            "office",
            "executive",
            "corporate",
            "clean friday",
            "workwear",
            "client",
        ],
        "date_night": [
            "boardroom",
            "professional",
            "office",
            "executive",
            "corporate",
            "clean friday",
            "workwear",
            "client",
        ],
        "beach": [
            "black trousers",
            "black pants",
            "loafers",
            "dress shoes",
            "blazer",
            "office",
            "professional",
            "formal",
        ],
    }
    blocked = blocked_by_occasion.get(normalized, [])
    for original in cards:
        if not isinstance(original, dict):
            continue
        card = dict(original)
        blob = _ahvi_item_blob(card)
        for item in card.get("items") or []:
            blob += " " + _ahvi_item_blob(item)
        if any(word in blob for word in blocked):
            continue
        try:
            from services.wardrobe_intelligence_service import board_has_occasion_conflict
            if board_has_occasion_conflict(card, normalized):
                continue
        except Exception:
            pass
        if normalized in {"date", "date_night"}:
            card["title"] = "Soft Statement"
            card["badge"] = "DATE NIGHT"
            card["occasion_label"] = "DATE NIGHT"
            card["occasion"] = "date_night"
            card["explanation"] = (
                "This is the safest evening direction from the current wardrobe. "
                "It keeps the read intentional without leaning into office polish."
            )
        elif normalized == "beach":
            card["title"] = "Closest Light Casual Option"
            card["badge"] = "COASTAL"
            card["occasion_label"] = "COASTAL"
            card["occasion"] = "beach"
            card["explanation"] = (
                "This is the closest relaxed direction from the current wardrobe, "
                "but it is not fully beach-ready yet."
            )
        return card
    return None


def _list_text(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [_safe_text(v) for v in value if _safe_text(v)]
    text = _safe_text(value)
    if not text:
        return []
    return [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]


_STYLE_PREF_MAP = {
    "clean minimal": {
        "style_energy": "minimal/monochrome",
        "palette": "minimal neutral",
        "accessory_policy": "restrained",
        "silhouette": "clean",
    },
    "minimalist": {
        "style_energy": "minimal/monochrome",
        "palette": "minimal neutral",
        "accessory_policy": "restrained",
        "silhouette": "clean",
    },
    "soft elegant": {
        "style_energy": "polished/social",
        "palette": "soft contrast",
        "accessory_policy": "refined",
        "silhouette": "polished",
    },
    "street cool": {
        "style_energy": "elevated/casual",
        "palette": "urban neutral",
        "accessory_policy": "edited",
        "silhouette": "street-smart",
    },
    "boho artisanal": {
        "style_energy": "relaxed/creative",
        "palette": "warm expressive",
        "accessory_policy": "textured",
        "silhouette": "creative",
    },
    "party glam": {
        "style_energy": "expressive/statement",
        "palette": "sharp contrast",
        "accessory_policy": "statement",
        "silhouette": "social",
    },
    "formal chic": {
        "style_energy": "safest/refined",
        "palette": "refined neutral",
        "accessory_policy": "polished",
        "silhouette": "tailored",
    },
    "casual": {
        "style_energy": "elevated/casual",
        "palette": "easy neutral",
        "accessory_policy": "edited",
        "silhouette": "relaxed",
    },
}


def normalize_style_identity(user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile = _dict(user_profile)
    prefs = (
        _list_text(profile.get("stylePreferences"))
        or _list_text(profile.get("styles"))
        or _list_text(profile.get("style_preferences"))
        or _list_text(profile.get("preferred_styles"))
    )
    normalized_prefs = [_safe_text(pref) for pref in prefs if _safe_text(pref)]
    mapped = [
        _STYLE_PREF_MAP.get(_safe_text(pref).lower())
        for pref in normalized_prefs
        if _STYLE_PREF_MAP.get(_safe_text(pref).lower())
    ]
    preferred_energies = []
    preferred_palettes = []
    preferred_silhouettes = []
    accessory_policies = []
    for row in mapped:
        preferred_energies.append(row["style_energy"])
        preferred_palettes.append(row["palette"])
        preferred_silhouettes.append(row["silhouette"])
        accessory_policies.append(row["accessory_policy"])

    identity = {
        "gender": _safe_text(profile.get("gender") or profile.get("style_gender")),
        "shop_prefs": _list_text(profile.get("shopPrefs") or profile.get("shop_prefs")),
        "skin_tone": _safe_text(profile.get("skinTone") or profile.get("skin_tone")),
        "body_shape": _safe_text(profile.get("bodyShape") or profile.get("body_shape")),
        "style_preferences": normalized_prefs,
        "location_label": _safe_text(profile.get("locationLabel") or profile.get("location_label")),
        "preferred_style_energies": list(dict.fromkeys(preferred_energies)),
        "preferred_palettes": list(dict.fromkeys(preferred_palettes)),
        "preferred_silhouettes": list(dict.fromkeys(preferred_silhouettes)),
        "accessory_policy": next((p for p in accessory_policies if p), ""),
        "profile_fields_used": [
            key
            for key, value in {
                "gender": profile.get("gender") or profile.get("style_gender"),
                "shopPrefs": profile.get("shopPrefs") or profile.get("shop_prefs"),
                "skinTone": profile.get("skinTone") or profile.get("skin_tone"),
                "bodyShape": profile.get("bodyShape") or profile.get("body_shape"),
                "stylePreferences": normalized_prefs,
                "locationLabel": profile.get("locationLabel") or profile.get("location_label"),
            }.items()
            if value
        ],
    }
    return identity


def item_image(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return _safe_text(
        item.get("masked_url")
        or item.get("maskedUrl")
        or item.get("image_url")
        or item.get("imageUrl")
        or item.get("raw_url")
        or item.get("rawUrl")
        or item.get("url")
        or item.get("image")
    )


def item_key(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return _safe_text(
        item.get("$id")
        or item.get("id")
        or item.get("item_id")
        or item.get("itemId")
        or item.get("image_id")
        or item.get("name")
        or item.get("label")
    ).lower()


def item_role(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return "unknown"
    blob = " ".join(
        _safe_text(item.get(k))
        for k in (
            "role",
            "slot",
            "type",
            "category",
            "sub_category",
            "subcategory",
            "main_category",
            "name",
            "label",
            "description",
        )
    )
    tokens = _tokens(blob)

    if tokens.intersection(
        {
            "shoe",
            "shoes",
            "sneaker",
            "sneakers",
            "loafer",
            "loafers",
            "boot",
            "boots",
            "sandal",
            "sandals",
            "footwear",
            "slipper",
            "slippers",
            "slider",
            "sliders",
        }
    ):
        return "footwear"
    if tokens.intersection(
        {
            "watch",
            "belt",
            "sunglass",
            "sunglasses",
            "eyewear",
            "bag",
            "bracelet",
            "ring",
            "necklace",
            "earring",
            "jewelry",
            "jewellery",
            "accessory",
            "accessories",
            "cap",
            "hat",
            "scarf",
        }
    ):
        return "accessory"
    if tokens.intersection(
        {
            "top",
            "shirt",
            "shirts",
            "tee",
            "tshirt",
            "polo",
            "jacket",
            "blazer",
            "sweater",
            "hoodie",
            "kurta",
            "blouse",
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
            "chino",
            "chinos",
            "shorts",
            "skirt",
        }
    ):
        return "bottom"
    if tokens.intersection({"dress", "dresses", "saree", "sari", "lehenga", "gown"}):
        return "dress"
    return "unknown"


def normalize_item(item: Dict[str, Any], role: Optional[str] = None) -> Dict[str, Any]:
    row = dict(item or {})
    resolved = role or item_role(row)
    if resolved in {"top", "bottom", "dress", "footwear", "accessory"}:
        row["role"] = resolved
        row["slot"] = resolved
    if resolved == "top":
        row.setdefault("category", "Tops")
    elif resolved == "bottom":
        row.setdefault("category", "Bottoms")
    elif resolved == "dress":
        row.setdefault("category", "Dresses")
    elif resolved == "footwear":
        row.setdefault("category", "Footwear")
    elif resolved == "accessory":
        row.setdefault("category", "Accessories")

    image = item_image(row)
    if image:
        row.setdefault("image_url", image)
        row.setdefault("imageUrl", image)
        row.setdefault("masked_url", image)
        row.setdefault("maskedUrl", image)
    if TRUST_LAYER_RUNTIME_INFERENCE and "formality" not in row:
        attrs = infer_style_attributes(row)
        for key, value in attrs.items():
            row.setdefault(key, value)
    return row


def card_signature(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    parts = []
    for item in _card_items(card, include_slots=True):
        key = item_key(item)
        if key:
            parts.append(key)
    if parts:
        return "|".join(sorted(set(parts)))
    return _safe_text(card.get("id") or card.get("title") or card.get("name")).lower()


def core_card_signature(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    by_role: Dict[str, str] = {}
    for item in _card_items(card, include_slots=True):
        role = item_role(item)
        if role not in {"top", "bottom", "dress", "footwear"}:
            continue
        key = item_key(item)
        if key and role not in by_role:
            by_role[role] = key
    if "dress" in by_role:
        parts = [by_role.get("dress", ""), by_role.get("footwear", "")]
    else:
        parts = [
            by_role.get("top", ""),
            by_role.get("bottom", ""),
            by_role.get("footwear", ""),
        ]
    clean = [p for p in parts if p]
    return "|".join(clean)


def _card_items(card: Dict[str, Any], *, include_slots: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key in ("items", "accessories"):
        value = card.get(key)
        if isinstance(value, list):
            out.extend([dict(x) for x in value if isinstance(x, dict)])
    if include_slots:
        for key in ("top", "bottom", "dress", "shoes", "footwear", "outerwear"):
            value = card.get(key)
            if isinstance(value, dict):
                out.append(dict(value))
    return out


def card_is_complete(card: Dict[str, Any]) -> bool:
    roles = {item_role(item) for item in _card_items(card, include_slots=True)}
    if "dress" in roles and "footwear" in roles:
        return True
    return {"top", "bottom", "footwear"}.issubset(roles)


def _accessory_type(item: Dict[str, Any]) -> str:
    blob = " ".join(_safe_text(item.get(k)) for k in ("name", "label", "category", "type", "sub_category", "subcategory"))
    tokens = _tokens(blob)
    for typ, words in (
        ("watch", {"watch", "watches"}),
        ("eyewear", {"sunglass", "sunglasses", "eyewear", "glasses"}),
        ("bag", {"bag", "bags", "purse", "tote", "clutch"}),
        ("belt", {"belt", "belts"}),
        ("headwear", {"cap", "caps", "hat", "hats", "beanie"}),
        ("scarf", {"scarf", "scarves"}),
        ("jewelry", {"ring", "rings", "necklace", "bracelet", "earring", "jewelry", "jewellery"}),
    ):
        if tokens.intersection(words):
            return typ
    return "accessory"


_SOCIAL_ACCESSORIES = {"watch", "belt", "jewelry", "eyewear", "bag", "scarf"}
_PROFESSIONAL_ACCESSORIES = {"watch", "belt", "jewelry", "eyewear", "bag", "scarf"}
_CASUAL_ACCESSORIES = {"watch", "belt", "jewelry", "eyewear", "bag", "scarf", "headwear"}


def _allows_headwear(query: str) -> bool:
    q = str(query or "").lower()
    return any(
        k in q
        for k in (
            "cap",
            "hat",
            "street",
            "sport",
            "outdoor",
            "travel",
            "airport",
            "flight",
            "vacation",
            "beach",
            "weekend",
            "rave",
            "festival",
        )
    )


def _accessory_allowed_for_query(item: Dict[str, Any], query: str) -> bool:
    typ = _accessory_type(item)
    kind = _occasion_kind(query)
    if typ == "headwear" and not _allows_headwear(query):
        return False
    if kind in {"office", "date", "wedding"}:
        return typ in _PROFESSIONAL_ACCESSORIES
    if kind == "party":
        return typ in _SOCIAL_ACCESSORIES or _allows_headwear(query)
    if kind in {"travel", "casual", "daily"}:
        return typ in _CASUAL_ACCESSORIES
    return typ != "headwear" or _allows_headwear(query)


def _accessory_priority(item: Dict[str, Any], query: str) -> int:
    typ = _accessory_type(item)
    kind = _occasion_kind(query)
    if kind in {"office", "date", "wedding"}:
        order = ["belt", "watch", "jewelry", "eyewear", "bag", "scarf", "accessory", "headwear"]
    elif kind == "party":
        order = ["jewelry", "watch", "belt", "eyewear", "bag", "scarf", "headwear", "accessory"]
    elif kind == "travel":
        order = ["bag", "eyewear", "watch", "belt", "scarf", "headwear", "jewelry", "accessory"]
    else:
        order = ["watch", "belt", "eyewear", "bag", "jewelry", "scarf", "headwear", "accessory"]
    try:
        return order.index(typ)
    except ValueError:
        return 99


def _curate_accessories_for_card(card: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Keep boards styled, not stuffed.

    The upstream outfit builder may attach any available accessory. This final
    pass removes occasion-breaking accents (for example a cap in office/date),
    dedupes accessory types, and caps the final board to six useful items.
    """
    if not isinstance(card, dict):
        return card
    core = [
        item
        for item in card.get("items", [])
        if isinstance(item, dict) and item_role(item) != "accessory"
    ]
    raw_accessories: List[Dict[str, Any]] = []
    for item in list(card.get("accessories") or []) + list(card.get("items") or []):
        if isinstance(item, dict) and item_role(item) == "accessory":
            raw_accessories.append(item)

    seen_keys: set[str] = set()
    seen_types: set[str] = set()
    accessories: List[Dict[str, Any]] = []
    for item in sorted(raw_accessories, key=lambda x: (_accessory_priority(x, query), item_key(x))):
        if not _accessory_allowed_for_query(item, query):
            continue
        key = item_key(item)
        typ = _accessory_type(item)
        if key and key in seen_keys:
            continue
        if typ in seen_types:
            continue
        normalized = normalize_item(item, "accessory")
        accessories.append(normalized)
        if key:
            seen_keys.add(key)
        seen_types.add(typ)

    # Premium boards do not need six items. They may carry six only when the
    # accessories are genuinely useful and distinct.
    accessory_budget = max(0, min(6 - len(core), 3))
    if _occasion_kind(query) in {"office", "date", "wedding"}:
        accessory_budget = min(accessory_budget, 2)
    accessories = accessories[:accessory_budget]

    fixed = dict(card)
    fixed["accessories"] = accessories
    fixed["items"] = core + accessories
    fixed["item_count"] = len(fixed["items"])
    fixed["accessory_policy_applied"] = {
        "max_items": 6,
        "accessory_budget": accessory_budget,
        "accessory_types": [_accessory_type(item) for item in accessories],
    }
    return fixed


def _canonicalize_card(card: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    raw_items = _card_items(card, include_slots=True)
    if not raw_items:
        return None

    chosen: Dict[str, Dict[str, Any]] = {}
    accessories: List[Dict[str, Any]] = []
    seen_item_keys: set[str] = set()
    seen_accessory_types: set[str] = set()

    for item in raw_items:
        role = item_role(item)
        if role not in {"top", "bottom", "dress", "footwear", "accessory"}:
            continue
        if not item_image(item):
            continue
        key = item_key(item)
        if key and key in seen_item_keys:
            continue
        if role == "accessory":
            typ = _accessory_type(item)
            if typ in seen_accessory_types:
                continue
            accessories.append(normalize_item(item, "accessory"))
            seen_accessory_types.add(typ)
            if key:
                seen_item_keys.add(key)
            continue
        if role not in chosen:
            chosen[role] = normalize_item(item, role)
            if key:
                seen_item_keys.add(key)

    if "dress" in chosen:
        chosen.pop("top", None)
        chosen.pop("bottom", None)

    final_items: List[Dict[str, Any]] = []
    for role in ("dress", "top", "bottom", "footwear"):
        if role in chosen:
            final_items.append(chosen[role])
    final_items.extend(accessories[:4])

    fixed = dict(card)
    fixed["items"] = final_items
    fixed["accessories"] = accessories[:4]
    fixed.setdefault("id", f"style_card_{index + 1}")

    if not card_is_complete(fixed):
        return None
    return fixed


def _occasion_flags(query: str) -> Dict[str, bool]:
    q = str(query or "").lower()
    return {
        "temple_modest": any(k in q for k in ("temple", "mandir", "pooja", "puja", "religious", "shrine", "darshan")),
        "beach": any(k in q for k in ("beach", "pool", "seaside", "coastal", "sand-friendly", "sand friendly")),
        "workout": any(k in q for k in ("workout", "gym", "fitness", "training", "yoga", "running")),
        "brunch": any(k in q for k in ("brunch",)),
        "office": any(k in q for k in ("office", "work", "meeting", "client", "business", "interview", "corporate")),
        "date": any(k in q for k in ("date", "dinner", "night")),
        "party": any(k in q for k in ("party", "club", "after-hours", "night out")),
        "travel": any(k in q for k in ("travel", "airport", "flight", "vacation", "trip")),
        "wedding": any(k in q for k in ("wedding", "reception", "ceremony", "sangeet", "formal event", "event")),
        "casual": any(k in q for k in ("casual", "weekend", "errand", "coffee")),
    }


def _item_by_role(card: Dict[str, Any], role: str) -> Dict[str, Any]:
    for item in card.get("items", []):
        if isinstance(item, dict) and item_role(item) == role:
            return item
    return {}


def _role_key(card: Dict[str, Any], role: str) -> str:
    return item_key(_item_by_role(card, role))


def _base_outfit_signature(card: Dict[str, Any]) -> str:
    if _role_key(card, "dress"):
        return _role_key(card, "dress")
    return "|".join(
        part
        for part in [
            _role_key(card, "top"),
            _role_key(card, "bottom"),
        ]
        if part
    )


def _item_blob(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return " ".join(
        _safe_text(item.get(k))
        for k in (
            "name",
            "label",
            "title",
            "category",
            "sub_category",
            "subcategory",
            "type",
            "style",
            "pattern",
            "color",
            "material",
            "fabric",
        )
    ).lower()


def _item_formality(item: Dict[str, Any]) -> float:
    try:
        return float(_dict(item).get("formality") or 3)
    except Exception:
        return 3.0


def _item_visual_weight(item: Dict[str, Any]) -> float:
    try:
        return float(_dict(item).get("visual_weight") or 2)
    except Exception:
        return 2.0


def _item_cluster(item: Dict[str, Any]) -> str:
    return _safe_text(_dict(item).get("aesthetic_cluster") or "polished")


def _item_occasion_fitness(item: Dict[str, Any], kind: str) -> float:
    data = _dict(_dict(item).get("occasion_fitness"))
    try:
        return float(data.get(kind) if data.get(kind) is not None else 0.5)
    except Exception:
        return 0.5


def _coherence_score(card: Dict[str, Any]) -> float:
    core_items = [
        item for item in card.get("items", [])
        if isinstance(item, dict) and item_role(item) in {"top", "bottom", "dress", "footwear"}
    ]
    if len(core_items) < 2:
        return 0.0
    formalities = [_item_formality(item) for item in core_items]
    weights = [_item_visual_weight(item) for item in core_items]
    clusters = [_item_cluster(item) for item in core_items]
    formality_spread = max(formalities) - min(formalities)
    weight_total = sum(weights)
    cluster_count = len(set(clusters))
    score = 2.0
    if formality_spread <= 1.5:
        score += 1.0
    elif formality_spread >= 3.0:
        score -= 2.0
    if weight_total <= 8:
        score += 0.8
    elif weight_total >= 12:
        score -= 1.5
    if cluster_count <= 2:
        score += 0.8
    elif cluster_count >= 4:
        score -= 1.2
    return score


def _occasion_kind(query: str) -> str:
    flags = _occasion_flags(query)
    for key in ("temple_modest", "beach", "workout", "brunch", "office", "date", "party", "travel", "wedding", "casual"):
        if flags.get(key):
            return key
    # Generic "today"/"daily"/no signal → daily, NOT office.
    return "daily"


def _style_direction(query: str) -> str:
    q = str(query or "").lower()
    if _occasion_kind(query) == "beach":
        return "coastal_casual"
    if _occasion_kind(query) == "workout":
        return "training_functional"
    if _occasion_kind(query) == "brunch":
        return "daytime_polish"
    if any(k in q for k in ("corporate", "boardroom", "formal", "client", "presentation")):
        return "corporate_office"
    if any(k in q for k in ("creative", "agency", "studio", "design")):
        return "creative_office"
    if any(k in q for k in ("startup", "start-up", "casual office")):
        return "startup_office"
    if any(k in q for k in ("friday", "relaxed office", "casual friday")):
        return "friday_office"
    if _occasion_kind(query) == "office":
        return "smart_casual_office"
    if _occasion_kind(query) == "daily":
        return "smart_casual_office"
    flags = _occasion_flags(query)
    if flags["beach"]:
        return "coastal_casual"
    if flags["workout"]:
        return "training_functional"
    if flags["brunch"]:
        return "daytime_polish"
    if flags["office"]:
        return "smart_casual_office"
    if flags["date"]:
        return "date_night"
    if flags["party"]:
        return "party"
    if flags["travel"]:
        return "comfort_polish"
    if flags["wedding"]:
        return "formal_event"
    if flags["casual"]:
        return "clean_daily"
    return "daily"


def interpret_occasion(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if interpret_occasion_context is not None:
        return interpret_occasion_context(query, context)

    ctx = _dict(context)
    q = _safe_text(query).lower()
    kind = _occasion_kind(query)
    weather = _safe_text(ctx.get("weather") or ctx.get("condition")).lower()
    time_of_day = _safe_text(ctx.get("time_of_day") or ctx.get("hour")).lower()
    module_context = _safe_text(ctx.get("module_context")).lower()

    chips: List[Dict[str, str]] = []
    ask_user = False
    reason = ""
    confidence = "high"

    if kind == "date":
        if any(k in q for k in ("dinner", "restaurant", "candle", "tonight", "evening")):
            brief = "polished dinner, low light, intentional but not overworked"
        elif any(k in q for k in ("coffee", "day", "afternoon", "walk")):
            brief = "daytime date, easy polish, movement friendly"
        elif any(k in q for k in ("rooftop", "drinks", "bar")):
            brief = "rooftop casual, warm evening, relaxed polish"
        else:
            brief = "polished date, social ease, clean footwear"
            ask_user = "date" in q and not any(k in q for k in ("today", "now", "quick", "suggest"))
            confidence = "medium" if ask_user else "high"
            reason = "date_venue_changes_formality"
            chips = [
                {"label": "Polished dinner", "value": "date_polished_dinner"},
                {"label": "Rooftop easy", "value": "date_rooftop_easy"},
                {"label": "Daytime quiet", "value": "date_daytime_quiet"},
            ]
    elif kind == "office":
        if any(k in q for k in ("client", "presentation", "meeting", "corporate", "interview")):
            brief = "client-facing office, polish required, smart footwear"
        elif any(k in q for k in ("creative", "studio", "agency")):
            brief = "creative office, expressive but controlled"
        elif any(k in q for k in ("startup", "start-up", "friday", "casual office")):
            brief = "relaxed office, intentional smart casual"
        else:
            brief = "smart casual office, clean structure, credible footwear"
    elif kind == "party":
        if any(k in q for k in ("rave", "club", "edm", "festival")):
            brief = "rave party, after-hours energy, movement friendly, expressive edge"
        elif any(k in q for k in ("cocktail", "lounge", "bar", "drinks")):
            brief = "cocktail party, sharp social polish, after-dark restraint"
        elif any(k in q for k in ("house", "birthday", "friends")):
            brief = "house party, relaxed social polish, memorable but easy"
        else:
            brief = "after-hours social, sharper contrast, memorable but edited"
            ask_user = True
            confidence = "medium"
            reason = "party_atmosphere_changes_formality"
            chips = [
                {"label": "Rave energy", "value": "party_rave_energy"},
                {"label": "Cocktail sharp", "value": "party_cocktail_sharp"},
                {"label": "House easy", "value": "party_house_easy"},
            ]
    elif kind == "travel":
        brief = "travel comfort polish, movement friendly, weather aware"
    elif kind == "wedding":
        brief = "formal event, respectful polish, ceremony-aware restraint"
        if not any(k in q for k in ("reception", "wedding", "formal", "traditional", "ceremony")):
            ask_user = True
            confidence = "medium"
            reason = "event_formality_or_cultural_context"
            chips = [
                {"label": "Ceremony refined", "value": "event_ceremony_refined"},
                {"label": "Reception polish", "value": "event_reception_polish"},
                {"label": "Statement occasion", "value": "event_statement_occasion"},
            ]
    elif kind == "casual":
        brief = "clean daily, comfortable but intentional"
    else:
        brief = "elevated daily, smart casual, repeat-aware"

    if "rain" in weather:
        brief = f"{brief}, rain-ready"
    elif any(k in weather for k in ("hot", "summer")):
        brief = f"{brief}, heat-conscious"
    elif any(k in weather for k in ("cold", "winter")):
        brief = f"{brief}, layered"
    if time_of_day and kind in {"date", "party"}:
        brief = f"{brief}, {time_of_day}"

    # Normal style flows should not become questionnaires. Only ask when a
    # single missing choice can prevent a visibly wrong board.
    if module_context in {"style", "wardrobe"} and ask_user:
        ask_user = True

    return {
        "resolved_brief": brief,
        "confidence": confidence,
        "ask_user": bool(ask_user),
        "question": (
            "Which brief is closer?"
            if ask_user
            else ""
        ),
        "chips": chips[:3],
        "board_generation_notes": {
            "occasion_kind": kind,
            "style_direction": _style_direction(query),
            "reason": reason or "context_sufficient",
            "max_questions": 3,
        },
    }


def _clarification_response(interpretation: Dict[str, Any]) -> Dict[str, Any]:
    chips = interpretation.get("chips") if isinstance(interpretation.get("chips"), list) else []
    return {
        "success": True,
        "message": interpretation.get("question") or "Which brief is closer?",
        "board": "style",
        "type": "clarification",
        "cards": [],
        "style_boards": [],
        "chips": chips[:3],
        "board_ids": "",
        "data": {"outfits": [], "rendered_boards": []},
        "meta": {
            "question_count": 1,
            "max_questions": 3,
            "reason": _dict(interpretation.get("board_generation_notes")).get("reason"),
            "occasion_interpretation": interpretation,
        },
        "audio_job_id": "offline",
    }


def _wardrobe_gap_response(
    *,
    query: str,
    wardrobe: Any,
    interpretation: Dict[str, Any],
    finalized: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    notes = _dict(interpretation.get("board_generation_notes"))
    occasion = _safe_text(interpretation.get("occasion") or notes.get("occasion_kind") or _occasion_kind(query))
    occasion = _normalize_occasion_value(occasion, query) or occasion
    wardrobe_items = wardrobe if isinstance(wardrobe, list) else []
    if detect_wardrobe_gap is not None and get_occasion_rule is not None:
        try:
            gap = detect_wardrobe_gap(wardrobe_items, occasion, get_occasion_rule(occasion))
        except Exception:
            gap = {}
    else:
        gap = {}
    rule = _dict(gap.get("rule"))
    missing_items = gap.get("missing_items") if isinstance(gap.get("missing_items"), list) else []
    missing_items = [item for item in missing_items if isinstance(item, dict)][:4]
    if not missing_items:
        missing_items = [
            {"label": "Occasion-ready hero", "reason": "Starts the outfit in the right atmosphere", "cta": "Find this"},
            {"label": "Right footwear", "reason": "Footwear controls the occasion register", "cta": "Find this"},
        ]
    find_chips = [
        {
            "label": f"Find {_safe_text(item.get('label')).lower()}",
            "value": f"find_this:{_safe_text(item.get('label'))}",
        }
        for item in missing_items[:2]
        if _safe_text(item.get("label"))
    ]

    chips = (
        [{"label": "Show closest option", "value": "show_closest_safe_option"}]
        + find_chips
    )[:3]

    normalized_occasion = (
        _safe_text(occasion).lower().replace("-", "_").replace(" ", "_")
    )

    if normalized_occasion in {"date", "date_night", "datenight"}:
        message = (
            "I don't see enough strong date-night options yet. "
            "I'd avoid forcing office styling into an evening brief."
        )
    elif normalized_occasion in {"beach", "beach_wear", "beachwear", "coastal"}:
        message = (
            "I don't see enough beach-ready pieces yet. "
            "I'd rather not force formal trousers or loafers into a beach brief."
        )
    elif normalized_occasion in {"temple_modest", "temple", "mandir", "pooja", "puja"}:
        message = (
            "I don't see enough temple-ready modest pieces yet. "
            "I'd rather not force a club or office silhouette into a temple brief."
        )
    else:
        brief = _safe_text(
            interpretation.get("resolved_brief")
            or rule.get("resolved_brief")
            or occasion
        )
        message = (
            "I don't see enough occasion-ready options yet. "
            f"For {brief}, I would rather protect the look than force a board that reads wrong."
        )

    data = _dict(_dict(finalized).get("data"))
    data.update(
        {
            "outfits": [],
            "rendered_boards": [],
            "missing_items": missing_items,
            "find_this_recommendations": missing_items,
            "closest_safe_brief": _safe_text(gap.get("closest_safe_brief")) or "clean daily",
            "occasion_interpretation": interpretation,
            "wardrobe_gap": {
                "occasion": occasion,
                "slot_scores": _dict(gap.get("slot_scores")),
                "has_enough": bool(gap.get("has_enough")),
            },
        }
    )

    logger.info(
        "ahvi.return_response type=missing_occasion_wardrobe occasion=%s message=%s chips=%s",
        occasion,
        message,
        chips,
    )

    return {
        "success": True,
        "message": message,
        "board": "style",
        "type": "missing_occasion_wardrobe",
        "cards": [],
        "style_boards": [],
        "chips": chips,
        "board_ids": "",
        "data": data,
        "meta": {
            **_dict(_dict(result).get("meta")),
            **_dict(_dict(finalized).get("meta")),
            "board_count": 0,
            "occasion_interpretation": interpretation,
            "wardrobe_limitation_reason": "missing_occasion_wardrobe",
            "wardrobe_gap": {
                "occasion": occasion,
                "missing_count": len(missing_items),
                "closest_safe_brief": _safe_text(gap.get("closest_safe_brief")) or "clean daily",
            },
        },
        "audio_job_id": "offline",
    }


def _office_direction(query: str) -> str:
    # Backward-compatible name for existing callers/tests. The returned value is
    # now the general style direction, not only an office subtype.
    return _style_direction(query)


def _footwear_mood(item: Dict[str, Any]) -> str:
    text = _item_blob(item)
    if any(k in text for k in ("birkenstock", "sandal", "slipper", "slider", "slides", "flip flop", "flip-flop", "crocs")):
        return "relaxed sandal"
    if any(k in text for k in ("chunky", "runner", "running", "trainer", "gym", "athletic", "sports", "hiking", "trail")):
        return "athletic"
    if any(k in text for k in ("loafer", "oxford", "derby", "formal", "monk strap")):
        return "formal polish"
    if any(k in text for k in ("slip-on", "slip on", "suede", "beige slip")):
        return "polished sneaker"
    if any(k in text for k in ("leather sneaker", "minimal sneaker", "white sneaker", "cream sneaker", "clean sneaker")):
        return "polished sneaker"
    if "sneaker" in text:
        return "casual sneaker"
    if any(k in text for k in ("boot", "chelsea")):
        return "structured boot"
    return "neutral footwear"


def _footwear_formality_score(item: Dict[str, Any], query: str) -> float:
    mood = _footwear_mood(item)
    direction = _office_direction(query)
    if direction == "corporate_office":
        return {
            "formal polish": 3.0,
            "polished sneaker": 1.3,
            "structured boot": 1.0,
            "casual sneaker": -1.0,
            "relaxed sandal": -8.0,
            "athletic": -7.0,
        }.get(mood, 0.0)
    if direction in {"smart_casual_office", "friday_office", "startup_office"}:
        return {
            "formal polish": 2.4,
            "polished sneaker": 2.2,
            "structured boot": 1.2,
            "casual sneaker": 0.4,
            "relaxed sandal": -7.0,
            "athletic": -5.0,
        }.get(mood, 0.0)
    if direction == "creative_office":
        return {
            "formal polish": 1.5,
            "polished sneaker": 2.0,
            "structured boot": 1.5,
            "casual sneaker": 0.8,
            "relaxed sandal": -3.5,
            "athletic": -3.0,
        }.get(mood, 0.0)
    if direction == "date_night":
        return {
            "formal polish": 2.2,
            "structured boot": 2.0,
            "polished sneaker": 1.2,
            "casual sneaker": -2.0,
            "relaxed sandal": -7.0,
            "athletic": -6.0,
        }.get(mood, 0.0)
    if direction in {"clean_daily", "daily"}:
        return {
            "formal polish": 1.4,
            "polished sneaker": 1.6,
            "structured boot": 1.0,
            "casual sneaker": 0.2,
            "relaxed sandal": -2.5,
            "athletic": -2.0,
        }.get(mood, 0.0)
    if direction in {"comfort_polish"}:
        return {
            "formal polish": 0.2,
            "polished sneaker": 1.7,
            "structured boot": 1.0,
            "casual sneaker": 1.2,
            "relaxed sandal": 0.1,
            "athletic": -0.5,
        }.get(mood, 0.0)
    return 0.0


def _top_office_score(item: Dict[str, Any], query: str) -> float:
    if not _occasion_flags(query)["office"]:
        return 0.0
    text = _item_blob(item)
    score = 0.0
    if any(k in text for k in ("button-down", "button down", "buttondown", "oxford", "shirt", "polo", "overshirt")):
        score += 2.0
    if any(k in text for k in ("structured", "tailored", "crisp", "clean")):
        score += 0.8
    expressive_ok = _office_direction(query) in {"creative_office", "startup_office", "friday_office"}
    if any(k in text for k in ("tropical", "hawaiian", "vacation", "beach", "resort", "loud")) and not expressive_ok:
        score -= 6.0
    return score


def _palette_direction(card: Dict[str, Any]) -> str:
    colors = [
        _safe_text(item.get("color") or item.get("color_code")).lower()
        for item in card.get("items", [])
        if isinstance(item, dict) and _safe_text(item.get("color") or item.get("color_code"))
    ]
    neutrals = {"black", "white", "off white", "grey", "gray", "navy", "cream", "beige", "brown"}
    unique = [c for c in dict.fromkeys(colors) if c]
    if not unique:
        return "neutral"
    if all(c in neutrals for c in unique):
        return "minimal neutral"
    if any(c in {"blue", "teal", "green", "mint", "navy"} for c in unique):
        return "cool contrast"
    if any(c in {"orange", "red", "yellow", "maroon"} for c in unique):
        return "warm statement"
    return "balanced color"


def _silhouette_category(card: Dict[str, Any]) -> str:
    top_text = _item_blob(_item_by_role(card, "top") or _item_by_role(card, "dress"))
    bottom_text = _item_blob(_item_by_role(card, "bottom"))
    footwear_text = _item_blob(_item_by_role(card, "footwear"))
    full_text = " ".join([top_text, bottom_text, footwear_text])
    if "black" in top_text and "black" in bottom_text:
        return "minimal"
    if any(k in full_text for k in ("blazer", "loafer", "oxford", "formal", "trouser", "tailored")):
        return "executive"
    if any(k in top_text for k in ("shirt", "button", "oxford")) and any(k in bottom_text for k in ("trouser", "pant", "chino")):
        return "tailored"
    if "jean" in bottom_text or "denim" in bottom_text:
        return "relaxed"
    if any(k in full_text for k in ("print", "pattern", "tropical", "statement", "graphic")):
        return "creative"
    if any(k in footwear_text for k in ("sneaker", "boot")):
        return "street-smart"
    return "tailored"


def _silhouette_mood(card: Dict[str, Any]) -> str:
    labels = {
        "executive": "polished column",
        "tailored": "clean tailoring",
        "minimal": "modern monochrome",
        "relaxed": "relaxed smart",
        "creative": "creative structure",
        "street-smart": "street-smart balance",
    }
    return labels.get(_silhouette_category(card), "easy structure")


def _footwear_energy(item: Dict[str, Any]) -> str:
    mood = _footwear_mood(item)
    if mood == "formal polish":
        return "polished"
    if mood == "polished sneaker":
        return "elevated casual"
    if mood == "structured boot":
        return "structured"
    if mood == "casual sneaker":
        return "casual"
    if mood == "relaxed sandal":
        return "relaxed"
    if mood == "athletic":
        return "athletic"
    return "neutral"


def _formality_energy(card: Dict[str, Any], query: str = "") -> str:
    footwear = _footwear_formality_score(_item_by_role(card, "footwear"), query)
    text = _card_blob(card)
    score = footwear
    if any(k in text for k in ("blazer", "formal", "loafer", "oxford", "trouser", "tailored")):
        score += 2.0
    if any(k in text for k in ("tee", "tshirt", "shorts", "sandal", "slipper", "birkenstock")):
        score -= 2.0
    if score >= 3.0:
        return "formal"
    if score >= 1.0:
        return "smart"
    if score <= -2.0:
        return "very casual"
    return "casual"


def _style_energy(card: Dict[str, Any], query: str) -> str:
    silhouette = _silhouette_category(card)
    palette = _palette_direction(card)
    footwear = _footwear_energy(_item_by_role(card, "footwear"))
    text = _card_blob(card)
    if silhouette in {"executive", "tailored"} and _formality_energy(card, query) in {"formal", "smart"}:
        return "safest/refined"
    if silhouette == "minimal" or palette == "minimal neutral":
        return "minimal/monochrome"
    if any(k in text for k in ("print", "pattern", "statement", "tropical", "graphic")):
        return "expressive/statement"
    if footwear in {"polished", "structured"} or _occasion_kind(query) in {"date", "party", "wedding"}:
        return "polished/social"
    if footwear == "elevated casual" or silhouette == "street-smart":
        return "elevated/casual"
    if silhouette in {"relaxed", "creative"}:
        return "relaxed/creative"
    return "elevated/casual"


def _accessory_mood(card: Dict[str, Any]) -> str:
    types = [_accessory_type(item) for item in card.get("accessories", []) if isinstance(item, dict)]
    if not types:
        return "clean minimal"
    if "watch" in types and len(types) == 1:
        return "minimal watch"
    if "watch" in types and "eyewear" in types:
        return "watch and eyewear"
    if "bag" in types:
        return "utility polish"
    return "subtle accessories"


def _diversity_profile(card: Dict[str, Any], query: str) -> Dict[str, Any]:
    return {
        "silhouette": _silhouette_mood(card),
        "silhouette_category": _silhouette_category(card),
        "palette": _palette_direction(card),
        "footwear_mood": _footwear_mood(_item_by_role(card, "footwear")),
        "footwear_energy": _footwear_energy(_item_by_role(card, "footwear")),
        "formality_energy": _formality_energy(card, query),
        "style_energy": _style_energy(card, query),
        "accessory_mood": _accessory_mood(card),
        "coherence_score": round(_coherence_score(card), 3),
        "hero_visual_weight": _item_visual_weight(_item_by_role(card, "top") or _item_by_role(card, "dress")),
    }


def _diversity_bonus(card: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
    if not selected:
        return 2.0
    profile = _diversity_profile(card, _safe_text(card.get("_style_query")))
    selected_profiles = [s.get("diversity_profile") or _diversity_profile(s, _safe_text(s.get("_style_query"))) for s in selected]
    bonus = 0.0
    for key in ("style_energy", "silhouette", "palette", "footwear_mood", "accessory_mood"):
        if profile.get(key) not in {p.get(key) for p in selected_profiles}:
            bonus += 0.7
    if _role_key(card, "top") and _role_key(card, "top") not in {_role_key(s, "top") for s in selected}:
        bonus += 1.0
    if _role_key(card, "bottom") and _role_key(card, "bottom") not in {_role_key(s, "bottom") for s in selected}:
        bonus += 1.0
    return bonus


def _redundancy_penalty(card: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
    penalty = 0.0
    bottom = _role_key(card, "bottom")
    top = _role_key(card, "top") or _role_key(card, "dress")
    top_bottom = _top_bottom_signature(card)
    selected_bottoms = [_role_key(s, "bottom") for s in selected]
    selected_tops = [_role_key(s, "top") or _role_key(s, "dress") for s in selected]
    selected_top_bottoms = {_top_bottom_signature(s) for s in selected}
    if top and top in selected_tops:
        penalty += selected_tops.count(top) * 3.0
    if bottom and bottom in selected_bottoms:
        penalty += selected_bottoms.count(bottom) * 4.5
    if top_bottom and top_bottom in selected_top_bottoms:
        # This is the main visible failure: same shirt + same pant with shoe/accessory swaps.
        penalty += 18.0
    if _footwear_mood(_item_by_role(card, "footwear")) in {
        _footwear_mood(_item_by_role(s, "footwear")) for s in selected
    }:
        penalty += 1.2
    if _role_key(card, "footwear") and _role_key(card, "footwear") in {_role_key(s, "footwear") for s in selected}:
        penalty += 2.5
    if _style_energy(card, _safe_text(card.get("_style_query"))) in {
        _style_energy(s, _safe_text(s.get("_style_query"))) for s in selected
    }:
        penalty += 1.5
    if set(_accessory_types(card)).intersection({typ for s in selected for typ in _accessory_types(s)}):
        penalty += 0.6
    return penalty


def _card_blob(card: Dict[str, Any]) -> str:
    return " ".join(
        [
            _safe_text(card.get("title")),
            _safe_text(card.get("occasion")),
            _safe_text(card.get("vibe")),
            *[
                " ".join(
                    _safe_text(item.get(k))
                    for k in ("name", "label", "category", "sub_category", "subcategory", "style", "pattern", "color")
                )
                for item in card.get("items", [])
                if isinstance(item, dict)
            ],
        ]
    ).lower()


def _quality_score(card: Dict[str, Any], query: str) -> float:
    text = _card_blob(card)
    flags = _occasion_flags(query)
    score = float(card.get("score") or 0.0) / 100.0
    roles = {item_role(item) for item in card.get("items", []) if isinstance(item, dict)}
    score += len(roles.intersection({"top", "bottom", "dress", "footwear"}))
    score += min(2, len([x for x in card.get("accessories", []) if isinstance(x, dict)])) * 0.35
    score += _coherence_score(card)

    if flags["office"]:
        if any(k in text for k in ("button-down", "button down", "shirt", "trouser", "black pants", "off white", "loafer")):
            score += 3.0
        if any(k in text for k in ("watch", "belt", "sneaker")):
            score += 1.0
        if any(k in text for k in ("tropical", "hawaiian", "vacation", "beach", "party", "loud")):
            score -= 5.0
        if any(k in text for k in ("shorts", "slipper", "slides", "slider", "sandal", "birkenstock", "crocs")):
            score -= 4.0
        score += _top_office_score(_item_by_role(card, "top"), query)
        score += _footwear_formality_score(_item_by_role(card, "footwear"), query)
    if flags["date"]:
        if any(k in text for k in ("watch", "black", "off white", "loafer")):
            score += 1.5
        score += _footwear_formality_score(_item_by_role(card, "footwear"), query)
    if flags["party"] and any(k in text for k in ("print", "pattern", "statement", "black")):
        score += 1.0
    if flags["travel"]:
        if any(k in text for k in ("sneaker", "boot", "jacket", "layer", "denim", "chino")):
            score += 1.4
        if any(k in text for k in ("formal", "heel", "slipper", "slides")):
            score -= 1.8
    if flags["wedding"]:
        if any(k in text for k in ("formal", "blazer", "loafer", "oxford", "kurta", "saree", "dress", "gown")):
            score += 2.0
        if any(k in text for k in ("tee", "tshirt", "running", "gym", "slipper", "slides", "shorts")):
            score -= 4.0
    if flags["casual"] and not any(flags[k] for k in ("office", "date", "party", "travel", "wedding")):
        if any(k in text for k in ("sneaker", "denim", "shirt", "tee", "polo", "chino")):
            score += 1.0
    return score


def _occasion_fit_score(card: Dict[str, Any], query: str) -> float:
    text = _card_blob(card)
    kind = _occasion_kind(query)
    footwear = _footwear_energy(_item_by_role(card, "footwear"))
    formality = _formality_energy(card, query)
    silhouette = _silhouette_category(card)
    score = 0.0
    core_items = [
        item for item in card.get("items", [])
        if isinstance(item, dict) and item_role(item) in {"top", "bottom", "dress", "footwear"}
    ]
    if core_items:
        score += sum(_item_occasion_fitness(item, kind) for item in core_items) / len(core_items)
        if get_occasion_rule is not None and score_item_for_occasion is not None:
            try:
                rule = get_occasion_rule(kind)
                score += sum(score_item_for_occasion(item, rule) for item in core_items) / len(core_items)
            except Exception:
                pass
    if kind == "beach":
        if footwear in {"relaxed", "casual", "elevated casual"}:
            score += 2.0
        if any(k in text for k in ("linen", "cotton", "shorts", "sandals", "slides", "espadrille", "tote", "sunglasses")):
            score += 1.5
        if any(k in text for k in ("black pants", "black trousers", "loafers", "dress shoes", "blazer", "suit", "charcoal")):
            score -= 8.0
    elif kind == "brunch":
        if footwear in {"casual", "elevated casual", "polished", "structured"}:
            score += 1.2
        if any(k in text for k in ("linen", "cotton", "dress", "shirt", "jeans", "chino")):
            score += 0.8
    elif kind == "workout":
        if any(k in text for k in ("training", "gym", "shorts", "leggings", "jogger", "running shoes", "sports")):
            score += 2.0
        if footwear in {"polished", "structured"} or any(k in text for k in ("loafer", "dress shoes", "blazer")):
            score -= 7.0
    if kind == "office":
        if formality in {"formal", "smart"}:
            score += 2.0
        if footwear in {"relaxed", "athletic"}:
            score -= 4.0
        if any(k in text for k in ("tropical", "vacation", "beach", "loud")) and _style_direction(query) not in {"creative_office", "startup_office", "friday_office"}:
            score -= 3.0
    elif kind == "date":
        if formality in {"smart", "formal"} or footwear in {"polished", "structured", "elevated casual"}:
            score += 1.8
        if footwear in {"relaxed", "athletic"}:
            score -= 3.0
    elif kind == "party":
        if _style_energy(card, query) in {"expressive/statement", "polished/social", "minimal/monochrome"}:
            score += 1.8
        if _style_energy(card, query) == "safest/refined" and "minimal" not in text:
            score -= 0.8
    elif kind == "travel":
        if footwear in {"casual", "elevated casual", "structured"}:
            score += 1.8
        if footwear in {"relaxed", "athletic"}:
            score -= 1.0 if "airport" in str(query).lower() else 0.4
        if silhouette in {"relaxed", "street-smart", "tailored"}:
            score += 0.8
    elif kind == "wedding":
        if formality in {"formal", "smart"}:
            score += 2.2
        if footwear in {"relaxed", "athletic"}:
            score -= 4.0
    else:
        if footwear in {"casual", "elevated casual", "polished"}:
            score += 0.8
    return score


def _allows_relaxed_footwear(query: str) -> bool:
    q = _safe_text(query).lower()
    return any(
        key in q
        for key in (
            "beach",
            "pool",
            "resort",
            "vacation",
            "holiday",
            "lounge",
            "home",
            "sunday",
            "errand",
            "very casual",
        )
    )


def _hard_rejection_reason(card: Dict[str, Any], query: str) -> str:
    kind = _occasion_kind(query)
    if get_occasion_rule is not None and reject_board_for_occasion is not None:
        try:
            reason = reject_board_for_occasion(card, kind, get_occasion_rule(kind))
            if reason:
                return reason
        except Exception:
            pass
    footwear_mood = _footwear_mood(_item_by_role(card, "footwear"))
    if (
        kind in {"office", "date", "wedding"}
        and footwear_mood == "relaxed sandal"
        and not _allows_relaxed_footwear(query)
    ):
        return "relaxed_footwear_blocked_for_occasion"
    if (
        kind == "daily"
        and _style_direction(query) == "smart_casual_office"
        and footwear_mood == "relaxed sandal"
        and not _allows_relaxed_footwear(query)
    ):
        return "relaxed_footwear_blocked_for_smart_daily"
    if kind in {"office", "date", "wedding"}:
        for item in card.get("accessories", []) or []:
            if isinstance(item, dict) and _accessory_type(item) == "headwear" and not _allows_headwear(query):
                return "headwear_blocked_for_polished_occasion"
    return ""


_STYLE_DNA_TARGETS = {
    "office": [
        {"style_energy": "safest/refined", "archetype": "Hero Look", "title": "Boardroom Casual"},
        {"style_energy": "relaxed/creative", "archetype": "Creative Professional", "title": "Creative Professional"},
        {"style_energy": "polished/social", "archetype": "Relaxed Sharp", "title": "Clean Friday"},
        {"style_energy": "minimal/monochrome", "archetype": "Elevated Option", "title": "Executive Minimal"},
        {"style_energy": "elevated/casual", "archetype": "Safest Option", "title": "Sharp Daily"},
        {"style_energy": "expressive/statement", "archetype": "Backup Option", "title": "Relaxed Sharp"},
    ],
    "date": [
        {"style_energy": "polished/social", "archetype": "Hero Look", "title": "Date Night Edit"},
        {"style_energy": "minimal/monochrome", "archetype": "Elevated Option", "title": "Evening Minimal"},
        {"style_energy": "expressive/statement", "archetype": "Creative Professional", "title": "Soft Statement"},
        {"style_energy": "elevated/casual", "archetype": "Relaxed Sharp", "title": "Confident Casual"},
        {"style_energy": "safest/refined", "archetype": "Safest Option", "title": "Polished Dinner"},
        {"style_energy": "relaxed/creative", "archetype": "Backup Option", "title": "After-Dark Smart"},
    ],
    "party": [
        {"style_energy": "expressive/statement", "archetype": "Hero Look", "title": "After-Hours Edit"},
        {"style_energy": "minimal/monochrome", "archetype": "Safest Option", "title": "Clean Contrast"},
        {"style_energy": "polished/social", "archetype": "Elevated Option", "title": "Polished Edge"},
        {"style_energy": "relaxed/creative", "archetype": "Relaxed Sharp", "title": "Statement Ease"},
        {"style_energy": "elevated/casual", "archetype": "Creative Professional", "title": "Night-Out Sharp"},
        {"style_energy": "safest/refined", "archetype": "Backup Option", "title": "Smart Presence"},
    ],
    "travel": [
        {"style_energy": "elevated/casual", "archetype": "Hero Look", "title": "Airport Clean"},
        {"style_energy": "relaxed/creative", "archetype": "Relaxed Sharp", "title": "Comfort Polish"},
        {"style_energy": "minimal/monochrome", "archetype": "Safest Option", "title": "Travel Minimal"},
        {"style_energy": "polished/social", "archetype": "Elevated Option", "title": "Arrival Ready"},
        {"style_energy": "safest/refined", "archetype": "Backup Option", "title": "Layered Practical"},
        {"style_energy": "expressive/statement", "archetype": "Creative Professional", "title": "Vacation Smart"},
    ],
    "wedding": [
        {"style_energy": "safest/refined", "archetype": "Hero Look", "title": "Formal Polish"},
        {"style_energy": "polished/social", "archetype": "Elevated Option", "title": "Event Ready"},
        {"style_energy": "expressive/statement", "archetype": "Creative Professional", "title": "Statement Occasion"},
        {"style_energy": "minimal/monochrome", "archetype": "Safest Option", "title": "Evening Minimal"},
        {"style_energy": "elevated/casual", "archetype": "Backup Option", "title": "Refined Traditional"},
        {"style_energy": "relaxed/creative", "archetype": "Relaxed Sharp", "title": "Soft Formal"},
    ],
    "beach": [
        {"style_energy": "relaxed/creative", "archetype": "Hero Look", "title": "Coastal Ease"},
        {"style_energy": "minimal/monochrome", "archetype": "Safest Option", "title": "Resort Minimal"},
        {"style_energy": "elevated/casual", "archetype": "Relaxed Sharp", "title": "Sunset Casual"},
        {"style_energy": "polished/social", "archetype": "Elevated Option", "title": "Beach-to-Dinner"},
        {"style_energy": "expressive/statement", "archetype": "Creative Professional", "title": "Resort Statement"},
        {"style_energy": "safest/refined", "archetype": "Backup Option", "title": "Clean Coastal"},
    ],
    "brunch": [
        {"style_energy": "polished/social", "archetype": "Hero Look", "title": "Daylight Polish"},
        {"style_energy": "elevated/casual", "archetype": "Relaxed Sharp", "title": "Garden Easy"},
        {"style_energy": "relaxed/creative", "archetype": "Creative Professional", "title": "Soft Day Edit"},
        {"style_energy": "minimal/monochrome", "archetype": "Safest Option", "title": "Quiet Brunch"},
        {"style_energy": "safest/refined", "archetype": "Elevated Option", "title": "Hotel Sharp"},
        {"style_energy": "expressive/statement", "archetype": "Backup Option", "title": "Friends-in-Town"},
    ],
    "workout": [
        {"style_energy": "elevated/casual", "archetype": "Hero Look", "title": "Training Clean"},
        {"style_energy": "minimal/monochrome", "archetype": "Safest Option", "title": "Gym Minimal"},
        {"style_energy": "relaxed/creative", "archetype": "Relaxed Sharp", "title": "Mobility Reset"},
        {"style_energy": "polished/social", "archetype": "Elevated Option", "title": "Studio Ready"},
        {"style_energy": "expressive/statement", "archetype": "Creative Professional", "title": "Active Edge"},
        {"style_energy": "safest/refined", "archetype": "Backup Option", "title": "Clean Movement"},
    ],
    "daily": [
        {"style_energy": "elevated/casual", "archetype": "Hero Look", "title": "Polished Neutral"},
        {"style_energy": "safest/refined", "archetype": "Safest Option", "title": "Sharp Daily"},
        {"style_energy": "relaxed/creative", "archetype": "Relaxed Sharp", "title": "Smart Ease"},
        {"style_energy": "minimal/monochrome", "archetype": "Elevated Option", "title": "Clean Edit"},
        {"style_energy": "expressive/statement", "archetype": "Creative Professional", "title": "Refined Casual"},
        {"style_energy": "polished/social", "archetype": "Backup Option", "title": "Signature Fit"},
    ],
}
_STYLE_DNA_TARGETS["casual"] = _STYLE_DNA_TARGETS["daily"]


def _style_targets_for_query(query: str) -> List[Dict[str, str]]:
    return _STYLE_DNA_TARGETS.get(_occasion_kind(query), _STYLE_DNA_TARGETS["daily"])


def _style_dna_match_score(card: Dict[str, Any], target: Dict[str, str], query: str) -> float:
    profile = _diversity_profile(card, query)
    score = 0.0
    if profile.get("style_energy") == target.get("style_energy"):
        score += 4.0
    if target.get("style_energy") == "minimal/monochrome" and profile.get("palette") == "minimal neutral":
        score += 1.0
    if target.get("style_energy") == "safest/refined" and profile.get("formality_energy") in {"formal", "smart"}:
        score += 1.0
    if target.get("style_energy") == "relaxed/creative" and profile.get("silhouette_category") in {"relaxed", "creative", "street-smart"}:
        score += 1.0
    if target.get("style_energy") == "polished/social" and profile.get("footwear_energy") in {"polished", "structured", "elevated casual"}:
        score += 1.0
    return score


def _identity_match_score(card: Dict[str, Any], style_identity: Dict[str, Any]) -> float:
    if not style_identity:
        return 0.0
    score = 0.0
    energy = _safe_text(card.get("style_energy") or _style_energy(card, _safe_text(card.get("_style_query"))))
    palette = _safe_text(card.get("palette_direction") or _palette_direction(card))
    silhouette = _safe_text(card.get("silhouette_category") or _silhouette_category(card))
    preferred_energies = set(style_identity.get("preferred_style_energies") or [])
    preferred_palettes = set(style_identity.get("preferred_palettes") or [])
    preferred_silhouettes = set(style_identity.get("preferred_silhouettes") or [])

    if energy and energy in preferred_energies:
        score += 2.5
    if any(palette and pref and pref.split()[0] in palette for pref in preferred_palettes):
        score += 0.8
    if silhouette and silhouette in preferred_silhouettes:
        score += 0.8

    accessory_policy = _safe_text(style_identity.get("accessory_policy")).lower()
    accessory_count = len([x for x in card.get("accessories", []) if isinstance(x, dict)])
    if accessory_policy == "restrained":
        score += 0.8 if accessory_count <= 1 else -1.2
    elif accessory_policy in {"statement", "textured"}:
        text = _card_blob(card)
        if any(k in text for k in ("print", "pattern", "texture", "ring", "bracelet", "watch")):
            score += 0.6
    elif accessory_policy in {"polished", "refined"} and accessory_count <= 2:
        score += 0.4
    return score


_ARCHETYPES = [
    "Hero Look",
    "Safest Option",
    "Elevated Option",
    "Relaxed Sharp",
    "Creative Professional",
    "Backup Option",
]


def _title_for(card: Dict[str, Any], query: str, index: int, archetype: str = "") -> str:
    existing = _safe_text(card.get("look_name") or card.get("title") or card.get("name"))
    lower = existing.lower()
    generic = (
        not existing
        or lower in {
            "styled look",
            "ahvi style board",
            "hero look",
            "easy win",
            "signature combo",
            "polished daily",
            "today's edit",
        }
        or lower.startswith("look ")
    )
    if not generic:
        return existing

    flags = _occasion_flags(query)
    if flags["office"]:
        title_by_archetype = {
            "Hero Look": "Boardroom Casual",
            "Safest Option": "Sharp Daily",
            "Elevated Option": "Executive Minimal",
            "Relaxed Sharp": "Clean Friday",
            "Creative Professional": "Creative Professional",
            "Backup Option": "Relaxed Sharp",
        }
        titles = ["Boardroom Casual", "Sharp Daily", "Creative Professional", "Clean Friday", "Executive Minimal", "Relaxed Sharp"]
    elif flags["date"]:
        title_by_archetype = {
            "Hero Look": "Date Night Edit",
            "Safest Option": "Polished Dinner",
            "Elevated Option": "After-Dark Smart",
            "Relaxed Sharp": "Confident Casual",
            "Creative Professional": "Soft Statement",
            "Backup Option": "Evening Minimal",
        }
        titles = ["Date Night Edit", "After-Dark Smart", "Polished Dinner", "Soft Statement", "Evening Minimal", "Confident Casual"]
    elif flags["party"]:
        title_by_archetype = {
            "Hero Look": "After-Hours Edit",
            "Safest Option": "Clean Contrast",
            "Elevated Option": "Polished Edge",
            "Relaxed Sharp": "Statement Ease",
            "Creative Professional": "Night-Out Sharp",
            "Backup Option": "Smart Presence",
        }
        titles = ["After-Hours Edit", "Statement Ease", "Night-Out Sharp", "Clean Contrast", "Smart Presence", "Polished Edge"]
    else:
        title_by_archetype = {
            "Hero Look": "Polished Neutral",
            "Safest Option": "Sharp Daily",
            "Elevated Option": "Clean Edit",
            "Relaxed Sharp": "Smart Ease",
            "Creative Professional": "Refined Casual",
            "Backup Option": "Signature Fit",
        }
        titles = ["Polished Neutral", "Sharp Daily", "Smart Ease", "Clean Edit", "Refined Casual", "Signature Fit"]
    if archetype in title_by_archetype:
        return title_by_archetype[archetype]
    return titles[index % len(titles)]


_EXPLANATION_MODES = [
    "color_harmony",
    "silhouette_balance",
    "texture_contrast",
    "occasion_alignment",
    "footwear_polish",
    "smart_contrast",
    "minimal_aesthetic",
    "relaxed_tailoring",
]


def _item_name(card: Dict[str, Any], role: str, fallback: str) -> str:
    item = _item_by_role(card, role)
    return _safe_text(item.get("name") or item.get("label") or item.get("title") or fallback)


def _explanation_for(card: Dict[str, Any], query: str, index: int) -> Dict[str, str]:
    mode = _EXPLANATION_MODES[index % len(_EXPLANATION_MODES)]
    top = _item_name(card, "top", _item_name(card, "dress", "the hero piece"))
    bottom = _item_name(card, "bottom", "the base")
    footwear = _item_name(card, "footwear", "the footwear")
    top_color = _safe_text(_item_by_role(card, "top").get("color") or _item_by_role(card, "dress").get("color")).title()
    bottom_color = _safe_text(_item_by_role(card, "bottom").get("color")).title()
    footwear_mood = _footwear_mood(_item_by_role(card, "footwear"))
    has_accessory = bool([x for x in card.get("accessories", []) if isinstance(x, dict)])

    hero_label = top if top != "the hero piece" else "The hero piece"
    base_label = "darker base" if "black" in bottom.lower() or bottom_color.lower() == "black" else "quieter base"
    accent_line = "The small accent breaks the severity." if has_accessory else "Nothing extra is fighting the line."

    copy = {
        "color_harmony": f"{hero_label} carries the personality. The {base_label} keeps it from turning loud.",
        "silhouette_balance": f"Structure leads above; {footwear} gives the line a cleaner finish.",
        "texture_contrast": f"The visual pressure stays controlled. {bottom} and {footwear} keep the finish grounded.",
        "occasion_alignment": f"The register is clean enough for the brief without feeling dressed-up for its own sake.",
        "footwear_polish": f"{footwear} sharpens the mood into {footwear_mood}. Intentional rather than easy.",
        "smart_contrast": f"Sharper above, quieter below. The point is the restraint.",
        "minimal_aesthetic": f"A clean column, then one controlled break. {accent_line}",
        "relaxed_tailoring": f"Ease with a backbone. The polish comes from what is left out.",
    }
    tips = [
        f"Roll the sleeves once on {top} if you want the line less stiff.",
        f"Keep {top} untucked only if the hem sits above mid-hip.",
        "Keep accessories minimal here; the clean line is doing the work.",
        f"Let {footwear} stay visible; it is carrying the polish.",
        "If this moves into evening, keep the collar open and skip extra accessories.",
        "Do not add a cap here unless the brief is explicitly weekend or street.",
    ]
    return {
        "explanation_mode": mode,
        "why_it_works": copy[mode],
        "styling_tip": tips[index % len(tips)],
    }


def _layout_metadata(card: Dict[str, Any], archetype: str) -> Dict[str, Any]:
    presets = {
        "Hero Look": "editorial_overlap_left",
        "Safest Option": "clean_catalog_stack",
        "Elevated Option": "magazine_depth_right",
        "Relaxed Sharp": "footwear_anchor_left",
        "Creative Professional": "asymmetric_editorial",
        "Backup Option": "compact_accessory_rail",
    }
    hierarchy = ["dress", "footwear", "accessory"] if _item_by_role(card, "dress") else ["top", "bottom", "footwear", "accessory"]
    return {
        "layout_preset": presets.get(archetype, "editorial_overlap_left"),
        "visual_hierarchy": hierarchy,
        "composition_notes": ["footwear_anchor", "visible_bottom", "accessory_rail"],
    }


def _item_id(item: Dict[str, Any]) -> str:
    return _safe_text(
        item.get("id")
        or item.get("$id")
        or item.get("item_id")
        or item.get("itemId")
        or item.get("image_id")
        or item.get("name")
    )


def _composition_metadata(card: Dict[str, Any]) -> Dict[str, Any]:
    items = [item for item in card.get("items", []) if isinstance(item, dict)]
    hero = _item_by_role(card, "dress") or _item_by_role(card, "top") or (items[0] if items else {})
    anchor = _item_by_role(card, "footwear") or (items[-1] if items else {})
    profile = _diversity_profile(card, _safe_text(card.get("_style_query")))
    mode = "grid" if (
        profile.get("style_energy") == "minimal/monochrome"
        and len(items) <= 4
        and profile.get("accessory_mood") in {"no accessories", "clean minimal", "minimal watch"}
    ) else "stack"

    # Runtime layout spec. x/y are center positions in board space.
    stack_positions = {
        "hero": (0.40, 0.46, 3, 0),
        "anchor": (0.27, 0.74, 5, -5),
        "support_0": (0.63, 0.53, 1, 0),
        "support_1": (0.62, 0.34, 1, 0),
        "accent_0": (0.80, 0.22, 6, 0),
        "accent_1": (0.86, 0.34, 6, 4),
        "accent_2": (0.78, 0.45, 6, -3),
    }
    grid_positions = {
        "hero": (0.38, 0.42, 3, 0),
        "anchor": (0.32, 0.74, 5, 0),
        "support_0": (0.68, 0.42, 1, 0),
        "support_1": (0.68, 0.62, 1, 0),
        "accent_0": (0.72, 0.76, 6, 0),
        "accent_1": (0.84, 0.76, 6, 0),
        "accent_2": (0.78, 0.88, 6, 0),
    }
    positions = grid_positions if mode == "grid" else stack_positions

    composition_items: List[Dict[str, Any]] = []
    support_index = 0
    accent_index = 0
    for item in items[:6]:
        role = item_role(item)
        ident = _item_id(item)
        if not ident:
            continue
        if ident == _item_id(hero):
            comp_role = "hero"
            size = 0.38
            key = "hero"
        elif ident == _item_id(anchor) or role == "footwear":
            comp_role = "anchor"
            size = 0.22
            key = "anchor"
        elif role == "accessory":
            comp_role = "accent"
            size = 0.08
            key = f"accent_{min(accent_index, 2)}"
            accent_index += 1
        else:
            comp_role = "support"
            size = 0.16
            key = f"support_{min(support_index, 1)}"
            support_index += 1
        x, y, z, rotation = positions.get(key, (0.70, 0.24, 1, 0))
        composition_items.append(
            {
                "id": ident,
                "role": comp_role,
                "relative_size": size,
                "x": x,
                "y": y,
                "z": z,
                "rotation": rotation if mode == "stack" else 0,
            }
        )
    return {
        "composition_mode": mode,
        "hero_item_id": _item_id(hero),
        "anchor_item_id": _item_id(anchor),
        "composition_items": composition_items,
    }


def _office_has_strong_footwear(cards: List[Dict[str, Any]], query: str) -> bool:
    if not _occasion_flags(query)["office"]:
        return False
    return any(_footwear_formality_score(_item_by_role(card, "footwear"), query) > 0.5 for card in cards)


MAX_TOP_REUSE = 2
MAX_BOTTOM_REUSE = 2
MAX_FOOTWEAR_REUSE = 2
MAX_ACCESSORY_REUSE = 2
MAX_TOP_BOTTOM_REUSE = 1
RELAXED_TOP_BOTTOM_REUSE = 2


def _accessory_keys(card: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for item in card.get("accessories", []):
        if isinstance(item, dict):
            key = item_key(item)
            if key:
                keys.append(key)
    return keys


def _accessory_types(card: Dict[str, Any]) -> List[str]:
    return [_accessory_type(item) for item in card.get("accessories", []) if isinstance(item, dict)]


def _top_bottom_signature(card: Dict[str, Any]) -> str:
    if _role_key(card, "dress"):
        return _role_key(card, "dress")
    top = _role_key(card, "top")
    bottom = _role_key(card, "bottom")
    return "|".join(part for part in (top, bottom) if part)


def _selected_count(selected: List[Dict[str, Any]], role: str, value: str) -> int:
    if not value:
        return 0
    return sum(1 for selected_card in selected if _role_key(selected_card, role) == value)


def _select_diverse_cards(cards: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
    if not cards:
        return []
    unique_tops = {k for k in ((_role_key(card, "top") or _role_key(card, "dress")) for card in cards) if k}
    unique_bottoms = {k for k in (_role_key(card, "bottom") for card in cards) if k}
    unique_footwear = {k for k in (_role_key(card, "footwear") for card in cards) if k}
    unique_accessories = {key for card in cards for key in _accessory_keys(card)}
    unique_energies = {_style_energy(card, query) for card in cards}
    unique_bases = {k for k in (_top_bottom_signature(card) for card in cards) if k}
    enforce_top_limit = len(unique_tops) > 1
    enforce_bottom_limit = len(unique_bottoms) > 1
    enforce_footwear_limit = len(unique_footwear) > 1
    enforce_accessory_limit = len(unique_accessories) > 1
    enforce_base_variation = len(unique_bases) > 1
    strong_office_footwear_exists = _office_has_strong_footwear(cards, query)
    selected: List[Dict[str, Any]] = []
    selected_sigs: set[str] = set()

    def _count_value(values: List[str], value: str) -> int:
        return sum(1 for x in values if x and x == value)

    def can_add(card: Dict[str, Any], *, strict: bool) -> bool:
        core = _safe_text(card.get("_style_core_signature"))
        if core in selected_sigs:
            return False
        top = _role_key(card, "top") or _role_key(card, "dress")
        bottom = _role_key(card, "bottom")
        footwear = _role_key(card, "footwear")
        base_sig = _top_bottom_signature(card)
        selected_tops = [_role_key(s, "top") or _role_key(s, "dress") for s in selected]
        selected_bases = [_top_bottom_signature(s) for s in selected]

        if enforce_base_variation and base_sig:
            # Strict mode treats a repeated top+bottom pair as the same outfit, even
            # when footwear/accessories change. Relaxed mode allows a second pass only
            # when the wardrobe is too small to fill the requested count.
            max_base = MAX_TOP_BOTTOM_REUSE if strict else RELAXED_TOP_BOTTOM_REUSE
            if _count_value(selected_bases, base_sig) >= max_base:
                return False
            if strict and base_sig in selected_bases:
                return False
        if enforce_top_limit and top and _count_value(selected_tops, top) >= MAX_TOP_REUSE:
            return False
        if enforce_bottom_limit and bottom:
            if _selected_count(selected, "bottom", bottom) >= MAX_BOTTOM_REUSE:
                return False
        if enforce_footwear_limit and footwear:
            if _selected_count(selected, "footwear", footwear) >= MAX_FOOTWEAR_REUSE:
                return False
        if enforce_accessory_limit:
            selected_accessories = [key for selected_card in selected for key in _accessory_keys(selected_card)]
            selected_types = [typ for selected_card in selected for typ in _accessory_types(selected_card)]
            if any(selected_accessories.count(key) >= MAX_ACCESSORY_REUSE for key in _accessory_keys(card)):
                return False
            if strict and any(selected_types.count(typ) >= MAX_ACCESSORY_REUSE for typ in _accessory_types(card)):
                return False
        if _occasion_flags(query)["office"] and strong_office_footwear_exists and len(selected) < 3:
            if _footwear_formality_score(_item_by_role(card, "footwear"), query) <= -3.0:
                return False
        if strict:
            if len(selected) < min(limit, len(unique_energies)):
                energy = _style_energy(card, query)
                if energy in {_style_energy(s, query) for s in selected}:
                    return False
            if len(selected) < min(3, limit):
                hero = top
                if hero and hero in selected_tops:
                    return False
        return True

    remaining = list(cards)
    targets = _style_targets_for_query(query)[:limit]
    for target in targets:
        if len(selected) >= limit:
            break
        choices = [card for card in remaining if can_add(card, strict=True)]
        if not choices:
            choices = [card for card in remaining if can_add(card, strict=False)]
        if not choices:
            continue
        choices.sort(
            key=lambda card: (
                float(card.get("_style_quality_score") or 0.0)
                + _occasion_fit_score(card, query)
                + _style_dna_match_score(card, target, query)
                + _diversity_bonus(card, selected)
                - _redundancy_penalty(card, selected)
            ),
            reverse=True,
        )
        picked = choices[0]
        picked["_target_style_energy"] = target.get("style_energy")
        picked["_target_archetype"] = target.get("archetype")
        picked["_target_title"] = target.get("title")
        selected.append(picked)
        selected_sigs.add(_safe_text(picked.get("_style_core_signature")))
        remaining = [card for card in remaining if card is not picked]

    for strict in (True, False):
        while len(selected) < limit:
            choices = [card for card in remaining if can_add(card, strict=strict)]
            if not choices:
                break
            choices.sort(
                key=lambda card: (
                    float(card.get("_style_quality_score") or 0.0)
                    + _occasion_fit_score(card, query)
                    + _diversity_bonus(card, selected)
                    - _redundancy_penalty(card, selected)
                ),
                reverse=True,
            )
            picked = choices[0]
            selected.append(picked)
            selected_sigs.add(_safe_text(picked.get("_style_core_signature")))
            remaining = [card for card in remaining if card is not picked]
    return selected[:limit]


def finalize_style_cards(
    cards: Any,
    *,
    query: str,
    style_identity: Optional[Dict[str, Any]] = None,
    occasion_interpretation: Optional[Dict[str, Any]] = None,
    exclude_signatures: Any = None,
    requested_count: Optional[int] = None,
    default_limit: int = 6,
) -> List[Dict[str, Any]]:
    excluded = {
        _safe_text(x).lower()
        for x in (exclude_signatures or [])
        if _safe_text(x)
    }
    limit = max(1, min(6, requested_count or default_limit))

    canonical: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, card in enumerate(cards or []):
        if not isinstance(card, dict):
            continue
        fixed = _canonicalize_card(card, idx)
        if not fixed:
            continue
        rejection_reason = _hard_rejection_reason(fixed, query)
        if rejection_reason:
            logger.info(
                "style_flow.card_rejected reason=%s query=%s items=%s",
                rejection_reason,
                query,
                [
                    _safe_text(item.get("name") or item.get("title"))
                    for item in fixed.get("items", [])
                    if isinstance(item, dict)
                ],
            )
            continue
        fixed = _curate_accessories_for_card(fixed, query)
        rejection_reason = _hard_rejection_reason(fixed, query)
        if rejection_reason:
            logger.info(
                "style_flow.card_rejected reason=%s query=%s items=%s",
                rejection_reason,
                query,
                [
                    _safe_text(item.get("name") or item.get("title"))
                    for item in fixed.get("items", [])
                    if isinstance(item, dict)
                ],
            )
            continue
        sig = card_signature(fixed)
        core_sig = core_card_signature(fixed) or sig
        if not sig or core_sig in seen or sig in excluded or core_sig in excluded:
            continue
        fixed["_style_signature"] = sig
        fixed["_style_core_signature"] = core_sig
        fixed["_style_quality_score"] = _quality_score(fixed, query) + _occasion_fit_score(fixed, query)
        fixed["_style_query"] = query
        fixed["_style_identity"] = dict(style_identity or {})
        fixed["_style_identity_score"] = _identity_match_score(fixed, style_identity or {})
        fixed["_style_quality_score"] += float(fixed.get("_style_identity_score") or 0.0)
        seen.add(core_sig)
        canonical.append(fixed)

    canonical.sort(key=lambda c: float(c.get("_style_quality_score") or 0.0), reverse=True)
    canonical = _select_diverse_cards(canonical, query, limit)
    for idx, card in enumerate(canonical):
        archetype = _safe_text(card.get("_target_archetype")) or _ARCHETYPES[idx % len(_ARCHETYPES)]
        title = _safe_text(card.get("_target_title")) or _title_for(card, query, idx, archetype)
        profile = _diversity_profile(card, query)
        explanation = _explanation_for(card, query, idx)
        layout = _layout_metadata(card, archetype)
        composition = _composition_metadata(card)
        card["title"] = title
        card["name"] = title
        card["style_archetype"] = archetype
        card["style_direction"] = _style_direction(query)
        card["style_energy"] = profile.get("style_energy")
        card["silhouette_category"] = profile.get("silhouette_category")
        card["palette_direction"] = profile.get("palette")
        card["footwear_energy"] = profile.get("footwear_energy")
        card["formality_energy"] = profile.get("formality_energy")
        card["occasion_fit"] = round(_occasion_fit_score(card, query), 3)
        card["diversity_profile"] = profile
        card["explanation_mode"] = explanation["explanation_mode"]
        card["why_it_works"] = explanation["why_it_works"]
        card["explanation"] = explanation["why_it_works"]
        card["reason"] = explanation["why_it_works"]
        card["style_reason"] = explanation["why_it_works"]
        card["styling_tip"] = explanation["styling_tip"]
        card["layout_preset"] = layout["layout_preset"]
        card["visual_hierarchy"] = layout["visual_hierarchy"]
        card["composition_notes"] = layout["composition_notes"]
        card["composition_mode"] = composition["composition_mode"]
        card["hero_item_id"] = composition["hero_item_id"]
        card["anchor_item_id"] = composition["anchor_item_id"]
        card["composition_items"] = composition["composition_items"]
        card["occasion_interpretation"] = occasion_interpretation or {}
        card["style_metadata"] = {
            "style_signature": card.get("_style_signature"),
            "core_style_signature": card.get("_style_core_signature"),
            "base_outfit_signature": _base_outfit_signature(card),
            "quality_score": round(float(card.get("_style_quality_score") or 0.0), 3),
            "coherence_score": round(_coherence_score(card), 3),
            "occasion_fit": card["occasion_fit"],
            "style_archetype": archetype,
            "style_direction": card["style_direction"],
            "style_energy": card["style_energy"],
            "silhouette_category": card["silhouette_category"],
            "palette_direction": card["palette_direction"],
            "footwear_energy": card["footwear_energy"],
            "formality_energy": card["formality_energy"],
            "explanation_mode": card["explanation_mode"],
            "layout_preset": card["layout_preset"],
            "composition_mode": card["composition_mode"],
            "hero_item_id": card["hero_item_id"],
            "anchor_item_id": card["anchor_item_id"],
            "identity_match_score": round(float(card.get("_style_identity_score") or 0.0), 3),
            "occasion_interpretation": occasion_interpretation or {},
        }
    return canonical[:limit]


def board_item_ids(cards: Any) -> List[str]:
    ids: List[str] = []
    seen: set[str] = set()
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        for item in card.get("items", []):
            if not isinstance(item, dict):
                continue
            ident = _safe_text(item.get("$id") or item.get("id") or item.get("item_id") or item.get("itemId") or item.get("image_id"))
            if ident and ident not in seen:
                seen.add(ident)
                ids.append(ident)
    return ids


def visual_intelligence_from_outfit(outfit: Dict[str, Any]) -> Dict[str, Any]:
    parts = [
        _dict(outfit.get("top")),
        _dict(outfit.get("bottom")),
        _dict(outfit.get("dress")),
        _dict(outfit.get("shoes")),
    ] + [x for x in (outfit.get("accessories") or []) if isinstance(x, dict)]
    colors = [_safe_text(p.get("color")).lower() for p in parts if _safe_text(p.get("color"))]
    patterns = [_safe_text(p.get("pattern")).lower() for p in parts if _safe_text(p.get("pattern"))]
    styles = [_safe_text(p.get("style")).lower() for p in parts if _safe_text(p.get("style"))]
    return {
        "dominant_palette": sorted(set(colors))[:4],
        "pattern_mix": sorted(set(patterns))[:4],
        "style_signals": sorted(set(styles))[:4],
        "composition_score": float(outfit.get("score") or 0.0),
        "story": _safe_text(_dict(outfit.get("story")).get("subtitle") or outfit.get("explanation")),
    }


def render_style_boards(
    cards: List[Dict[str, Any]],
    context: Dict[str, Any],
    *,
    user_id: str,
    include_base64: bool,
    upload_to_r2: bool = False,
) -> List[Dict[str, Any]]:
    """Render multiple style boards in parallel.

    Each card's `style_board_renderer.render(board)` is CPU+I/O heavy
    (Pillow drawing + R2 upload), and previously they ran serially in a
    for-loop. For 6 cards that meant 6-18s of cumulative blocking. We now
    fan out across a ThreadPoolExecutor so the total wall time becomes
    roughly max(card_render) rather than sum(card_render).
    """
    if not cards:
        return []
    if not include_base64 and not upload_to_r2:
        return []

    storage = None
    if upload_to_r2 and R2Storage is not None:
        try:
            storage = R2Storage()
        except Exception:
            storage = None

    style_dna = _dict(context.get("style_dna"))
    try:
        from brain.engines.style_board_engine import style_board_engine
        from brain.engines.style_board_renderer import style_board_renderer
    except Exception as exc:
        logger.warning("style board renderer unavailable: %s", exc)
        return []

    def _build_one_board(card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        items = card.get("items") if isinstance(card.get("items"), list) else []
        if not items:
            return None
        board: Dict[str, Any] = {}
        image_bytes: bytes = b""
        try:
            board = style_board_engine.build_board(
                {
                    "items": items,
                    "score": card.get("score"),
                    "style_archetype": card.get("style_archetype"),
                    "style_energy": card.get("style_energy"),
                    "layout_preset": card.get("layout_preset"),
                    "composition_mode": card.get("composition_mode"),
                    "hero_item_id": card.get("hero_item_id"),
                    "anchor_item_id": card.get("anchor_item_id"),
                    "composition_items": card.get("composition_items"),
                    "visual_hierarchy": card.get("visual_hierarchy"),
                    "composition_notes": card.get("composition_notes"),
                    "diversity_profile": card.get("diversity_profile"),
                },
                context,
            )
            board.update(
                {
                    "style_archetype": card.get("style_archetype"),
                    "style_energy": card.get("style_energy"),
                    "layout_preset": card.get("layout_preset"),
                    "composition_mode": card.get("composition_mode"),
                    "hero_item_id": card.get("hero_item_id"),
                    "anchor_item_id": card.get("anchor_item_id"),
                    "composition_items": card.get("composition_items"),
                    "visual_hierarchy": card.get("visual_hierarchy"),
                    "composition_notes": card.get("composition_notes"),
                    "diversity_profile": card.get("diversity_profile"),
                }
            )
            image_bytes = style_board_renderer.render(board)
        except Exception as exc:
            logger.warning(
                "style board render failed user=%s error=%s",
                user_id, exc,
            )
            image_bytes = b""

        return {"card": card, "board": board, "image_bytes": image_bytes, "items": items}

    # Render boards in parallel. Workers capped at min(len(cards), 4) so we
    # don't oversubscribe CPU on small Cloud Run instances.
    from concurrent.futures import ThreadPoolExecutor
    rendered_results: List[Optional[Dict[str, Any]]] = []
    worker_cap = max(1, min(len(cards), 4))
    render_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_cap) as pool:
        rendered_results = list(pool.map(_build_one_board, cards))
    render_ms = round((time.perf_counter() - render_started) * 1000, 1)
    logger.info(
        "ahvi.board_render.timing user=%s cards=%s workers=%s total_ms=%s",
        user_id, len(cards), worker_cap, render_ms,
    )

    rendered: List[Dict[str, Any]] = []
    for idx, prepared in enumerate(rendered_results):
        if prepared is None:
            continue
        card = prepared["card"]
        board = prepared["board"]
        image_bytes = prepared["image_bytes"]
        items = prepared["items"]

        image_base64 = base64.b64encode(image_bytes).decode("ascii") if include_base64 and image_bytes else None
        image_url = None
        upload_error = None
        if storage and image_bytes:
            try:
                uploaded = storage.upload_style_board_image(
                    user_id=str(user_id or "user"),
                    image_bytes=image_bytes,
                    extension="png",
                )
                image_url = uploaded.get("image_url")
            except (R2StorageError, Exception) as exc:
                upload_error = str(exc)

        rendered.append(
            {
                "board_id": str(uuid.uuid4()),
                "type": "style",
                "label": card.get("title") or "AHVI Style Board",
                "score": card.get("score"),
                "aesthetic": board.get("aesthetic"),
                "vibe": board.get("vibe"),
                "items": items,
                "image_base64": image_base64,
                "image_url": image_url,
                "upload_error": upload_error,
                "style_signature": card.get("_style_signature") or card_signature(card),
                "board_payload": {
                    "items": items,
                    "aesthetic": board.get("aesthetic"),
                    "vibe": board.get("vibe"),
                    "score": card.get("score"),
                    "style_dna": style_dna,
                    "card_id": card.get("id") or f"style_card_{idx + 1}",
                    "style_archetype": card.get("style_archetype"),
                    "style_direction": card.get("style_direction"),
                    "diversity_profile": card.get("diversity_profile"),
                    "layout_preset": card.get("layout_preset"),
                    "composition_mode": card.get("composition_mode"),
                    "hero_item_id": card.get("hero_item_id"),
                    "anchor_item_id": card.get("anchor_item_id"),
                    "composition_items": card.get("composition_items"),
                    "visual_hierarchy": card.get("visual_hierarchy"),
                    "composition_notes": card.get("composition_notes"),
                },
            }
        )
    return rendered


def _style_signature_hash(signatures: List[str]) -> str:
    return hashlib.sha1("|".join(signatures).encode("utf-8")).hexdigest() if signatures else ""


def _board_metadata_summary(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for idx, card in enumerate(cards or []):
        if not isinstance(card, dict):
            continue
        metadata = _dict(card.get("style_metadata"))
        summary.append(
            {
                "index": idx,
                "title": card.get("title"),
                "style_archetype": card.get("style_archetype"),
                "style_direction": card.get("style_direction"),
                "style_energy": card.get("style_energy"),
                "silhouette_category": card.get("silhouette_category"),
                "palette_direction": card.get("palette_direction"),
                "footwear_energy": card.get("footwear_energy"),
                "formality_energy": card.get("formality_energy"),
                "explanation_mode": card.get("explanation_mode"),
                "style_signature": metadata.get("style_signature") or card.get("_style_signature") or card_signature(card),
                "core_style_signature": metadata.get("core_style_signature") or card.get("_style_core_signature") or core_card_signature(card),
                "base_outfit_signature": metadata.get("base_outfit_signature") or _base_outfit_signature(card),
                "quality_score": metadata.get("quality_score"),
                "coherence_score": metadata.get("coherence_score"),
                "occasion_fit": card.get("occasion_fit"),
                "layout_preset": card.get("layout_preset"),
                "composition_mode": card.get("composition_mode"),
                "hero_item_id": card.get("hero_item_id"),
                "anchor_item_id": card.get("anchor_item_id"),
                "occasion_interpretation": card.get("occasion_interpretation"),
            }
        )
    return summary


def finalize_style_response_payload(
    result: Dict[str, Any],
    *,
    user_id: str,
    query: str,
    wardrobe: Any = None,
    context: Optional[Dict[str, Any]] = None,
    include_base64: bool = False,
    upload_to_r2: bool = False,
    style_action: str = "",
    exclude_style_signatures: Any = None,
    requested_board_count: Optional[int] = None,
    cache_bypass: bool = True,
) -> Dict[str, Any]:
    ctx = dict(context or {})
    style_identity = normalize_style_identity(_dict(ctx.get("user_profile")))
    ctx["style_identity"] = style_identity
    occasion_interpretation = _dict(ctx.get("occasion_interpretation")) or interpret_occasion(query, ctx)
    ctx["occasion_interpretation"] = occasion_interpretation
    resolved_brief = _safe_text(occasion_interpretation.get("resolved_brief"))
    finalizer_query = f"{query} {resolved_brief}".strip() if resolved_brief else query
    raw_cards = result.get("cards") if isinstance(result.get("cards"), list) else []
    raw_outfits = result.get("outfits") if isinstance(result.get("outfits"), list) else []
    candidates = list(raw_outfits or []) + list(raw_cards or [])

    finalize_started = time.perf_counter()
    cards = finalize_style_cards(
        candidates,
        query=finalizer_query,
        style_identity=style_identity,
        occasion_interpretation=occasion_interpretation,
        exclude_signatures=exclude_style_signatures,
        requested_count=requested_board_count if style_action in {"more_options", "more_looks", "next_best"} else None,
    )
    normalized_occasion = _normalize_occasion_value(
        occasion_interpretation.get("occasion")
        or _dict(occasion_interpretation.get("board_generation_notes")).get("occasion_kind")
        or ctx.get("occasion"),
        query,
    )
    cards = apply_occasion_card_language(cards, normalized_occasion)
    wardrobe_items = wardrobe if isinstance(wardrobe, list) else []
    if not wardrobe_items:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_items = candidate.get("items")
            if isinstance(candidate_items, list):
                wardrobe_items.extend(item for item in candidate_items if isinstance(item, dict))
            else:
                wardrobe_items.append(candidate)
    slot_counts = _ahvi_slot_counts(wardrobe_items)
    logger.info(
        "ahvi.wardrobe_slot_counts occasion=%s total=%s top=%s bottom=%s footwear=%s accessory=%s outerwear=%s",
        normalized_occasion,
        slot_counts.get("total"),
        slot_counts.get("top"),
        slot_counts.get("bottom"),
        slot_counts.get("footwear"),
        slot_counts.get("accessory"),
        slot_counts.get("outerwear"),
    )
    if not _ahvi_has_core_slots(slot_counts):
        logger.info(
            "ahvi.missing_core_slots occasion=%s slot_counts=%s",
            normalized_occasion,
            slot_counts,
        )
        return _ahvi_missing_core_slots_response(slot_counts)
    filtered_cards = []
    rejected_cards = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        if reject_quality_board_for_occasion is None:
            rejected, reason = False, ""
        else:
            try:
                rejected, reason = reject_quality_board_for_occasion(card, normalized_occasion)
            except Exception:
                rejected, reason = False, ""
        if rejected:
            rejected_cards.append((card, reason))
            logger.info(
                "ahvi.board_rejected occasion=%s reason=%s title=%s",
                normalized_occasion,
                reason,
                card.get("title"),
            )
            continue
        filtered_cards.append(card)
    if not filtered_cards:
        closest_board = _ahvi_pick_closest_safe_board(cards, normalized_occasion)
        logger.info(
            "ahvi.missing_occasion_wardrobe occasion=%s rejected=%s slot_counts=%s closest=%s",
            normalized_occasion,
            len(rejected_cards),
            slot_counts,
            bool(closest_board),
        )
        return _ahvi_missing_occasion_response(
            occasion=normalized_occasion,
            slot_counts=slot_counts,
            closest_board=closest_board,
        )
    cards = apply_occasion_card_language(filtered_cards, normalized_occasion)
    finalize_ms = round((time.perf_counter() - finalize_started) * 1000, 2)
    ids = board_item_ids(cards)
    render_started = time.perf_counter()
    rendered = render_style_boards(
        cards,
        ctx,
        user_id=user_id,
        include_base64=include_base64,
        upload_to_r2=upload_to_r2,
    )
    render_ms = round((time.perf_counter() - render_started) * 1000, 2)
    signatures = [card.get("_style_signature") or card_signature(card) for card in cards]
    core_signatures = [card.get("_style_core_signature") or core_card_signature(card) for card in cards]
    style_signature = _style_signature_hash([s for s in signatures if s])
    primary_board_id = ids[0] if ids else ""

    logger.info(
        "style_flow.final_response user=%s cards=%d core_signatures=%s signatures=%s style_action=%s cache_bypass=%s finalize_ms=%s render_ms=%s profile_fields=%s",
        user_id,
        len(cards),
        core_signatures,
        signatures,
        style_action or "",
        bool(cache_bypass),
        finalize_ms,
        render_ms,
        style_identity.get("profile_fields_used") or [],
    )
    logger.info(
        "ahvi.final_boards occasion=%s titles=%s badges=%s",
        normalized_occasion,
        [c.get("title") for c in cards[:6]],
        [c.get("badge") or c.get("occasion_label") for c in cards[:6]],
    )

    data = {
        "outfits": cards,
        "visual_intelligence": visual_intelligence_from_outfit(raw_outfits[0]) if raw_outfits and isinstance(raw_outfits[0], dict) else {},
        "pipeline": _dict(result.get("pipeline")),
        "rendered_boards": rendered or cards,
        "board_item_ids": ids,
        "board_metadata": _board_metadata_summary(cards),
    }
    return {
        "cards": cards,
        "style_boards": cards,
        "board_ids": primary_board_id,
        "data": data,
        "meta": {
            "style_action": style_action or None,
            "style_signature": style_signature or None,
            "core_style_signatures": core_signatures,
            "board_metadata": _board_metadata_summary(cards),
            "board_count": len(cards),
            "has_more_style_options": bool(cards),
            "cache_bypass": bool(cache_bypass),
            "style_cache_bypass": bool(cache_bypass),
            "style_identity": style_identity,
            "occasion_interpretation": occasion_interpretation,
            "profile_sources": {
                "appwrite_profile": bool(style_identity.get("profile_fields_used")),
                "request_profile": bool(_dict(ctx.get("user_profile"))),
            },
            "timing_ms": {
                "style_finalization": finalize_ms,
                "style_rendering": render_ms,
            },
        },
    }


def build_style_flow_response(
    *,
    user_id: str,
    query: str,
    wardrobe: Any,
    user_profile: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    include_base64: bool = False,
    upload_to_r2: bool = False,
    style_action: str = "",
    exclude_style_signatures: Any = None,
    requested_board_count: Optional[int] = None,
    cache_bypass: bool = True,
) -> Dict[str, Any]:
    from brain.outfit_pipeline import get_daily_outfits

    started = time.perf_counter()
    ctx = dict(context or {})
    ctx.setdefault("query", query)
    ctx.setdefault("user_id", user_id)
    if user_profile is not None:
        ctx.setdefault("user_profile", user_profile)
    ctx.setdefault("style_identity", normalize_style_identity(_dict(ctx.get("user_profile"))))
    occasion_interpretation = interpret_occasion(query, ctx)
    ctx["occasion_interpretation"] = occasion_interpretation
    normalized_occasion = _normalize_occasion_value(
        occasion_interpretation.get("occasion")
        or _dict(occasion_interpretation.get("board_generation_notes")).get("occasion_kind")
        or ctx.get("occasion"),
        query,
    )
    if normalized_occasion:
        ctx["occasion"] = normalized_occasion
    logger.info(
        "ahvi.occasion_context occasion=%s query=%s",
        normalized_occasion or _occasion_kind(query),
        query,
    )
    if (
        occasion_interpretation.get("ask_user")
        and not style_action
        and not requested_board_count
    ):
        return _clarification_response(occasion_interpretation)

    candidate_started = time.perf_counter()
    result = get_daily_outfits(
        {
            "user_id": user_id,
            "wardrobe": wardrobe,
            "context": ctx,
        }
    )
    candidate_ms = round((time.perf_counter() - candidate_started) * 1000, 2)
    if not isinstance(result, dict):
        result = {}

    finalized = finalize_style_response_payload(
        result,
        user_id=user_id,
        query=query,
        wardrobe=wardrobe,
        context=ctx,
        include_base64=include_base64,
        upload_to_r2=upload_to_r2,
        style_action=style_action,
        exclude_style_signatures=exclude_style_signatures,
        requested_board_count=requested_board_count,
        cache_bypass=cache_bypass,
    )
    if finalized.get("type") in {"missing_core_wardrobe_slots", "missing_occasion_wardrobe"}:
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "style_flow.timing user=%s candidates_ms=%s total_ms=%s wardrobe_count=%s cards=%s fallback=%s",
            user_id,
            candidate_ms,
            total_ms,
            len(wardrobe) if isinstance(wardrobe, list) else 0,
            len(finalized.get("cards") or []),
            finalized.get("type"),
        )
        return {
            **finalized,
            "board": "style",
            "board_ids": "",
            "audio_job_id": "offline",
            "meta": {
                "analysis_source": "style_flow_service",
                "board_count": len(finalized.get("cards") or []),
                "occasion_interpretation": occasion_interpretation,
                "timing_ms": {
                    "candidate_generation": candidate_ms,
                    "style_flow_total": total_ms,
                },
            },
        }
    cards = finalized["cards"]
    if not cards:
        gap_kind = _safe_text(
            occasion_interpretation.get("occasion")
            or _dict(occasion_interpretation.get("board_generation_notes")).get("occasion_kind")
            or _occasion_kind(query)
        )
        gap_kind = _normalize_occasion_value(gap_kind, query) or gap_kind
        if gap_kind and gap_kind != "daily":
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "style_flow.timing user=%s candidates_ms=%s total_ms=%s wardrobe_count=%s cards=0 gap=%s",
                user_id,
                candidate_ms,
                total_ms,
                len(wardrobe) if isinstance(wardrobe, list) else 0,
                gap_kind,
            )
            return _wardrobe_gap_response(
                query=query,
                wardrobe=wardrobe,
                interpretation=occasion_interpretation,
                finalized=finalized,
                result=result,
            )
    total_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "style_flow.timing user=%s candidates_ms=%s total_ms=%s wardrobe_count=%s cards=%s",
        user_id,
        candidate_ms,
        total_ms,
        len(wardrobe) if isinstance(wardrobe, list) else 0,
        len(cards),
    )
    return {
        "success": bool(cards),
        "message": result.get("context")
        or (
            "I pulled together wardrobe-based looks that match your request, occasion, and style profile."
            if cards
            else "I couldn't build a reliable style board from your wardrobe yet."
        ),
        "board": "style",
        "type": "cards" if cards else "missing_outfit_cards",
        "cards": cards,
        "style_boards": finalized["style_boards"],
        "chips": STYLE_ACTION_CHIPS if cards else [],
        "board_ids": finalized["board_ids"],
        "data": finalized["data"],
        "meta": {
            **_dict(result.get("meta")),
            **finalized["meta"],
            "analysis_source": "style_flow_service",
            "timing_ms": {
                **_dict(finalized.get("meta", {}).get("timing_ms")),
                "candidate_generation": candidate_ms,
                "style_flow_total": total_ms,
            },
        },
    }
