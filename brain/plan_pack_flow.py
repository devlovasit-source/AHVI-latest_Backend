import logging
import math
import re
from typing import Any, Dict, List, Optional

from brain.engines.packing.packing_engine import packing_engine

logger = logging.getLogger("ahvi.plan_pack")

# Private items never use real wardrobe imagery — icon/suggested only.
_PRIVATE_LABEL_TOKENS = (
    "innerwear", "underwear", "bra", "brief", "boxer", "sock",
    "sleepwear", "nightwear", "pajama", "pyjama", "lingerie",
)


def _is_private_label(label: str) -> bool:
    lowered = str(label or "").lower()
    return any(token in lowered for token in _PRIVATE_LABEL_TOKENS)


# Strict allowlist: ONLY these clothing / public-wearable groups may pull a
# real wardrobe image. Everything else (toiletries, tech, documents, health)
# stays icon-only. Gate runs BEFORE any wardrobe matching.
_CLOTHING_GROUP_TOKENS = (
    "top", "shirt", "t-shirt", "tshirt", "tee", "polo", "kurta", "blouse", "tank",
    "bottom", "jean", "trouser", "pant", "short", "skirt", "chino", "legging",
    "footwear", "shoe", "sneaker", "sandal", "boot", "flat", "heel", "loafer",
    "swim", "beach",
    "outer", "layer", "jacket", "coat", "blazer", "cardigan", "sweater",
    "hoodie", "overshirt", "shrug", "dress",
)


def _is_clothing_label(label: str) -> bool:
    """True only if the label is a clothing/wearable group eligible for
    wardrobe imagery. Private items are never eligible."""
    if _is_private_label(label):
        return False
    lowered = str(label or "").lower()
    return any(token in lowered for token in _CLOTHING_GROUP_TOKENS)


# Ordered most-specific-first; _icon_key_for returns the first key found in the
# label. Differentiated semantic keys so e.g. Passport and Tickets never share
# an icon.
_ICON_KEYS = {
    "sunscreen": "sunscreen",
    "sun screen": "sunscreen",
    "spf": "sunscreen",
    "sunglass": "sunglasses",
    "lip balm": "lip_balm",
    "lip-balm": "lip_balm",
    "wet wipe": "wet_wipes",
    "wipes": "wet_wipes",
    "moisturizer": "moisturizer",
    "moisturiser": "moisturizer",
    "toiletr": "toiletries",
    "sanitizer": "sanitizer",
    "sanitiser": "sanitizer",
    "face mask": "face_mask",
    "power bank": "power_bank",
    "powerbank": "power_bank",
    "earphone": "earphones",
    "earbud": "earphones",
    "headphone": "earphones",
    "charger": "charger",
    "adapter": "charger",
    "phone": "charger",
    "boarding": "tickets",
    "ticket": "tickets",
    "booking": "tickets",
    "passport": "passport",
    "wallet": "wallet",
    "water bottle": "water_bottle",
    "hydration": "water_bottle",
    "medicine": "medicine",
    "medication": "medicine",
    "first aid": "medicine",
    "first-aid": "medicine",
    "umbrella": "umbrella",
    "travel pillow": "travel_pillow",
    "neck pillow": "travel_pillow",
    "visa": "document",
    "invoice": "document",
    "itinerary": "document",
    "document": "document",
    "towel": "towel",
    "camera": "camera",
    "jacket": "jacket",
    "outer layer": "jacket",
    "coat": "jacket",
    "blazer": "jacket",
    "shoes": "shoes",
    "footwear": "shoes",
}


_VISUAL_SECTION_KEYWORDS = {
    "clothes": (
        "top",
        "tops",
        "shirt",
        "t-shirt",
        "tshirt",
        "tee",
        "polo",
        "kurta",
        "blouse",
        "bottom",
        "bottoms",
        "jeans",
        "trousers",
        "pants",
        "shorts",
        "skirt",
        "footwear",
        "shoes",
        "sneakers",
        "sandals",
        "flats",
        "heels",
        "jacket",
        "hoodie",
        "blazer",
        "coat",
        "cardigan",
        "overshirt",
        "swimwear",
        "beachwear",
    ),
    "essentials": (
        "toiletries",
        "sunscreen",
        "moisturizer",
        "medicine",
        "first-aid",
        "first aid",
        "water",
        "hydration",
        "towel",
    ),
    "tech": (
        "charger",
        "phone",
        "power bank",
        "camera",
        "laptop",
        "tablet",
        "headphones",
        "adapter",
    ),
    "documents": (
        "passport",
        "document",
        "documents",
        "id",
        "boarding",
        "ticket",
        "booking",
        "visa",
        "invoice",
    ),
    "weather": (
        "rain",
        "jacket",
        "waterproof",
        "warm",
        "thermal",
        "socks",
        "cotton",
        "linen",
        "cap",
        "hat",
        "hydration",
        "layer",
    ),
}


_SECTION_FALLBACK_ICON = {
    "clothes": "clothes",
    "essentials": "essentials",
    "tech": "tech",
    "documents": "document",
    "health": "health",
    "weather": "weather",
}


def _icon_key_for(label: str, *, section: str = "") -> Optional[str]:
    lowered = str(label or "").lower()
    for key, icon in _ICON_KEYS.items():
        if key in lowered:
            return icon
    return _SECTION_FALLBACK_ICON.get(section)


def _image_url_from_item(item: Dict[str, Any]) -> str:
    for key in (
        "display_image_url",
        "displayImageUrl",
        "normalized_url",
        "normalizedUrl",
        "imageUrl",
        "image_url",
        "masked_url",
        "maskedUrl",
        "thumbnail",
        "photoUrl",
        "image",
        "url",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


_PACKING_IMAGE_BASE = (
    "https://pub-43484c7ec0d741cabcac4df01e98344b.r2.dev/mens-assets"
)

# Curated product imagery for non-wardrobe packing essentials. These are kept
# separate from the user's wardrobe so toiletries, documents, and tech never
# accidentally inherit a clothing image.
_PACKING_IMAGE_ALIASES = {
    "sunscreen": "skincare/retinol.jpg",
    "sun screen": "skincare/retinol.jpg",
    "spf": "skincare/retinol.jpg",
    "sunglasses": "other_accessories/blacksunglasses.jpg",
    "lip balm": "skincare/lipbalm.jpg",
    "toiletries": "skincare/grooming kit.jpg",
    "toiletry": "skincare/grooming kit.jpg",
    "face mask": "skincare/facemask.jpg",
    "power bank": "travel/black powerbank.jpg",
    "powerbank": "travel/black powerbank.jpg",
    "charger": "travel/c type charger.jpg",
    "phone": "travel/c type charger.jpg",
    "earphone": "travel/wireless earphones.jpg",
    "headphone": "travel/wireless earphones.jpg",
    "neck pillow": "travel/blue neck pillow.jpg",
    "travel pillow": "travel/blue neck pillow.jpg",
    "water bottle": "travel/blacktumbler.jpg",
    "hydration": "travel/blacktumbler.jpg",
    "medicine": "travel/medicalkit.jpg",
    "first aid": "travel/medicalkit.jpg",
    "first-aid": "travel/medicalkit.jpg",
    "umbrella": "travel/foldable umberella.jpg",
    "document": "travel/documentholder.jpg",
    "passport": "travel/documentholder.jpg",
    "ticket": "travel/documentholder.jpg",
    "visa": "travel/documentholder.jpg",
    "tops": "tops/whitetshirt.jpg",
    "top": "tops/whitetshirt.jpg",
    "shirt": "tops/whiteshirt.jpg",
    "t-shirt": "tops/whitetshirt.jpg",
    "bottoms": "bottoms/beigechinos.jpg",
    "bottom": "bottoms/beigechinos.jpg",
    "footwear": "footwear/whitesneakers.jpg",
    "shoes": "footwear/whitesneakers.jpg",
    "outer layer": "outerwear/blackcoat.jpg",
    "jacket": "outerwear/blackcoat.jpg",
    "blazer": "outerwear/blackblazer.jpg",
    "beachwear": "bottoms/blueswimshorts.jpg",
    "swimwear": "bottoms/blueswimshorts.jpg",
}


def _packing_image_url(label: str) -> Optional[str]:
    lowered = str(label or "").lower()
    for alias, path in _PACKING_IMAGE_ALIASES.items():
        if alias in lowered:
            return f"{_PACKING_IMAGE_BASE}/{path}"
    return None


def _matches_wardrobe(label: str, item: Dict[str, Any]) -> bool:
    needle = str(label or "").lower()
    if not needle:
        return False
    searchable = " ".join(
        str(item.get(k) or "")
        for k in (
            "name",
            "category",
            "sub_category",
            "subcategory",
            "type",
            "tags",
            "color",
            "occasion",
        )
    ).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", needle) if len(t) >= 4]
    return any(token in searchable for token in tokens)


def _wardrobe_search_text(item: Dict[str, Any]) -> str:
    return " ".join(
        str(item.get(k) or "")
        for k in (
            "name",
            "category",
            "sub_category",
            "subcategory",
            "type",
            "family",
            "tags",
            "color",
            "occasion",
        )
    ).lower()


_OUTER_LAYER_ALLOW = (
    "jacket", "coat", "blazer", "cardigan", "sweater", "hoodie",
    "overshirt", "raincoat", "windbreaker", "outerwear",
)
_OUTER_LAYER_REJECT = (
    "cap", "hat", "beanie", "headwear", "helmet", "scarf",
    "turban", "bandana", "accessory",
)
_OUTER_LAYER_LABEL_TOKENS = (
    "outer", "layer", "jacket", "coat", "blazer", "cardigan",
    "sweater", "hoodie", "overshirt", "windbreaker", "raincoat",
)


def _is_outer_layer_label(label: str) -> bool:
    lowered = str(label or "").lower()
    return any(t in lowered for t in _OUTER_LAYER_LABEL_TOKENS)


def _matches_visual_section(label: str, section: str, item: Dict[str, Any]) -> bool:
    label_lower = str(label or "").lower()
    # Slot-specific override: an "Outer layer" / "Light/Warm/Rain layer" / Jacket
    # slot may ONLY take an actual outer garment — never headwear (swim cap,
    # hat, beanie) or accessories. Runs first so the broad clothing allowlist
    # and the loose token matcher cannot leak a cap into a layer slot.
    if _is_outer_layer_label(label_lower):
        text = _wardrobe_search_text(item)
        if any(bad in text for bad in _OUTER_LAYER_REJECT):
            return False
        allow = _OUTER_LAYER_ALLOW
        if "light" in label_lower:
            allow = allow + ("shirt", "overshirt", "top", "tee")
        return any(good in text for good in allow)
    if _matches_wardrobe(label, item):
        return True
    searchable = _wardrobe_search_text(item)
    keywords = _VISUAL_SECTION_KEYWORDS.get(section, ())
    if section == "clothes":
        if "tops" in label_lower or label_lower == "top":
            keywords = ("top", "tops", "shirt", "t-shirt", "tshirt", "tee", "polo", "kurta", "blouse", "tank", "hoodie")
        elif "bottom" in label_lower:
            keywords = ("bottom", "bottoms", "jeans", "trousers", "pants", "shorts", "skirt", "chino", "leggings")
        elif "footwear" in label_lower:
            keywords = ("footwear", "shoe", "shoes", "sneaker", "sneakers", "sandals", "boots", "flats", "heels", "loafers")
        elif "outer" in label_lower or "layer" in label_lower:
            keywords = ("outerwear", "jacket", "coat", "blazer", "cardigan", "shrug", "sweater", "hoodie", "overshirt")
        elif "swim" in label_lower or "beach" in label_lower:
            keywords = ("swimwear", "beachwear", "swimsuit", "shorts")
        else:
            return False
    return any(keyword in searchable for keyword in keywords)


def _quantity_from_label(label: str) -> tuple[str, int, str]:
    clean = str(label or "").strip()
    match = re.search(r"\s+x\s*(\d+)\s*$", clean, re.I)
    if not match:
        return clean, 1, clean
    quantity = max(1, int(match.group(1)))
    base = re.sub(r"\s+x\s*\d+\s*$", "", clean, flags=re.I).strip()
    return base or clean, quantity, clean


def _section_for_visual_label(label: str, default: str) -> str:
    lowered = str(label or "").lower()
    for section, keywords in _VISUAL_SECTION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return section
    return default


def _visual_section_item(
    label: str,
    *,
    section: str,
    wardrobe: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    base_label, quantity, display_label = _quantity_from_label(label)
    item_id = re.sub(r"[^a-z0-9]+", "_", f"{section}_{base_label}".lower()).strip("_")
    # Strict gate: only clothing/wearable groups may use wardrobe images.
    # Non-clothing (toiletries/tech/documents/health) + private -> icon only.
    private = _is_private_label(base_label)
    eligible = _is_clothing_label(base_label)
    matches: List[Dict[str, Any]] = []
    if eligible and not private:
        for item in wardrobe or []:
            if not isinstance(item, dict):
                continue
            if _matches_visual_section(base_label, section, item):
                image_url = _image_url_from_item(item)
                if image_url:
                    matches.append(item)
    # Limit images to the requested quantity (Tops x2 -> 2, Outer x1 -> 1).
    limit = max(1, min(int(quantity or 1), 4))
    if matches:
        capped = matches[:limit]
        image_urls = [_image_url_from_item(item) for item in capped]
        wardrobe_ids = [
            str(item.get("$id") or item.get("id") or item.get("item_id") or item.get("wardrobe_item_id") or "")
            for item in capped
            if str(item.get("$id") or item.get("id") or item.get("item_id") or item.get("wardrobe_item_id") or "").strip()
        ]
        try:
            logger.info(
                "plan_pack_visual_item label=%s category=%s wardrobe_count=%d matched=%d first=%s source=wardrobe",
                base_label, section, len(wardrobe or []), len(matches),
                str(capped[0].get("name") or "")[:40],
            )
        except Exception:
            pass
        return {
            "id": f"{item_id}_qty_{quantity}",
            "label": base_label,
            "quantity": quantity,
            "display_label": display_label,
            "section": section,
            "category": section,
            "source": "wardrobe",
            "image_url": image_urls[0] if image_urls else None,
            "image_urls": image_urls,
            "wardrobe_item_ids": wardrobe_ids,
            "assetIcon": None,
            "iconKey": None,
            "asset_key": None,
            "packed": False,
            "missing": False,
        }

    icon_key = _icon_key_for(base_label, section=section)
    try:
        logger.info(
            "plan_pack_visual_item label=%s category=%s wardrobe_count=%d matched=0 source=icon private=%s",
            base_label, section, len(wardrobe or []), private,
        )
    except Exception:
        pass
    return {
        "id": f"{item_id}_qty_{quantity}",
        "label": base_label,
        "quantity": quantity,
        "display_label": display_label,
        "section": section,
        "category": section,
        "source": "icon",
        "image_url": _packing_image_url(base_label),
        "image_urls": [],
        "wardrobe_item_ids": [],
        "assetIcon": None,
        "iconKey": icon_key or "generic",
        "asset_key": None,
        "packed": False,
        "missing": False,
    }


def build_visual_packing_sections(
    cards: List[Dict[str, Any]],
    *,
    wardrobe: Optional[List[Dict[str, Any]]] = None,
    destination: str = "",
    duration_label: str = "",
    weather: str = "",
    time_of_day: str = "",
) -> List[Dict[str, Any]]:
    sections: Dict[str, Dict[str, Any]] = {
        "clothes": {"id": "clothes", "title": "Clothes", "items": []},
        "essentials": {"id": "essentials", "title": "Essentials", "items": []},
        "tech": {"id": "tech", "title": "Tech", "items": []},
        "documents": {"id": "documents", "title": "Documents", "items": []},
        "weather": {"id": "weather", "title": "Weather", "items": []},
    }

    for card in cards:
        if not isinstance(card, dict):
            continue
        card_id = str(card.get("id") or "").lower()
        if card_id == "trip_plan":
            continue
        default_section = "essentials"
        if card_id == "packing_clothes":
            default_section = "clothes"
        elif card_id == "weather_time_adjustments":
            default_section = "weather"

        raw_items = card.get("items") if isinstance(card.get("items"), list) else []
        for raw in raw_items:
            label = ""
            if isinstance(raw, dict):
                label = str(raw.get("label") or raw.get("title") or raw.get("name") or "").strip()
            else:
                label = str(raw or "").strip()
            if not label:
                continue
            section = _section_for_visual_label(label, default_section)
            sections[section]["items"].append(
                _visual_section_item(label, section=section, wardrobe=wardrobe)
            )

    defaults = {
        "documents": ["Passport/ID", "Tickets/bookings"],
        "tech": ["Phone + charger", "Power bank"],
        "essentials": ["Toiletries kit"],
    }
    for section, labels in defaults.items():
        if not sections[section]["items"]:
            sections[section]["items"] = [
                _visual_section_item(label, section=section, wardrobe=wardrobe)
                for label in labels
            ]

    subtitle_parts = [
        part
        for part in (
            duration_label,
            f"{weather.title()} {time_of_day}" if weather else "",
            destination if destination not in {"Your Trip", "Carry-On Trip"} else "",
        )
        if str(part or "").strip()
    ]
    subtitle = " | ".join(subtitle_parts[:2])

    out: List[Dict[str, Any]] = []
    for section_id in ("clothes", "essentials", "tech", "documents", "weather"):
        section = sections[section_id]
        items = section["items"]
        if not items:
            continue
        out.append(
            {
                "id": section_id,
                "title": section["title"],
                "subtitle": subtitle,
                # Count = number of display item GROUPS (Tops x6 is one tile),
                # not the quantity sum. piece_count keeps the total for callers.
                "item_count": len(items),
                "piece_count": sum(int(item.get("quantity") or 1) for item in items),
                "items": items,
            }
        )
    return out


def _visual_item(
    label: str,
    *,
    category: str,
    wardrobe: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    clean = str(label or "").strip()
    for item in wardrobe or []:
        if not isinstance(item, dict):
            continue
        if _matches_wardrobe(clean, item):
            image_url = _image_url_from_item(item)
            if image_url:
                return {
                    "label": clean,
                    "category": category,
                    "checked": False,
                    "imageUrl": image_url,
                    "image_url": image_url,
                    "assetIcon": None,
                    "source": "wardrobe",
                    "wardrobeItemId": item.get("$id") or item.get("id"),
                }
    icon_key = _icon_key_for(clean, section=category)
    return {
        "label": clean,
        "category": category,
        "checked": False,
        "imageUrl": None,
        "image_url": _packing_image_url(clean),
        "assetIcon": None,
        "iconKey": icon_key or "generic",
        "source": "icon",
    }


def _parse_days(text: str) -> int:
    lowered = (text or "").lower()
    patterns = [
        r"(\d+)\s*[- ]?\s*day",
        r"(\d+)\s*days",
        r"for\s+(\d+)\s*days",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            try:
                return max(1, min(21, int(match.group(1))))
            except Exception:
                pass

    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "week": 7,
    }
    for word, value in words.items():
        if f"{word} day" in lowered or f"{word}-day" in lowered:
            return value
    if "carry-on" in lowered or "carry on" in lowered:
        return 1
    return 3


def _has_explicit_duration(text: str) -> bool:
    lowered = (text or "").lower()
    if re.search(r"\d+\s*[- ]?\s*days?", lowered):
        return True
    return any(
        token in lowered
        for token in (
            "one day",
            "two day",
            "three day",
            "four day",
            "five day",
            "six day",
            "seven day",
            "week",
        )
    )


def _detect_scenario(text: str) -> str:
    lowered = (text or "").lower()
    if any(k in lowered for k in ["camping", "camp", "trek", "hike", "hiking"]):
        return "camping"
    if any(k in lowered for k in ["birthday party", "birthday", "party plan"]):
        return "birthday"
    if any(k in lowered for k in ["wedding", "marriage", "bride", "groom"]):
        return "wedding"
    if any(
        k in lowered for k in ["business", "work trip", "conference", "client meeting"]
    ):
        return "business"
    if any(k in lowered for k in ["goa", "beach", "vacation", "holiday", "trip"]):
        return "travel"
    if any(k in lowered for k in ["pack", "packing"]):
        return "travel"
    return "general"


def _extract_destination(text: str) -> str:
    lowered = (text or "").lower()
    if "goa" in lowered:
        return "Goa"
    if "carry-on" in lowered or "carry on" in lowered:
        return "Carry-On Trip"
    if "camp" in lowered:
        return "Camping"
    if "birthday" in lowered:
        return "Birthday Party"
    if "business" in lowered:
        return "Business Travel"
    if "wedding" in lowered:
        return "Wedding Event"
    m = re.search(
        r"(?:to|for)\s+([a-z ]{2,30})(?:trip|travel|vacation|holiday|wedding)?", lowered
    )
    if m:
        candidate = re.sub(r"\b(a|an|the|\d+|day|days)\b", " ", m.group(1))
        candidate = re.sub(r"\s+", " ", candidate).strip(" -")
        if candidate:
            return candidate.title()
    return "Your Trip"


def _packing_clothes(days: int, scenario: str) -> List[str]:
    tops = days + 1
    bottoms = max(2, math.ceil(days * 0.6))
    innerwear = days + 1
    socks = days + 1
    sleepwear = max(1, math.ceil(days / 3))
    footwear = 2
    outerwear = 1

    if scenario == "business":
        footwear = 2
        outerwear = 1
    if scenario == "wedding":
        footwear = 3
        outerwear = 1
    if scenario == "travel":
        footwear = 2
        outerwear = 1

    return [
        f"Tops x{tops}",
        f"Bottoms x{bottoms}",
        f"Innerwear x{innerwear}",
        f"Socks x{socks}",
        f"Sleepwear x{sleepwear}",
        f"Footwear x{footwear}",
        f"Outer layer x{outerwear}",
    ]


def _scenario_addons(scenario: str) -> List[str]:
    if scenario == "camping":
        return [
            "Torch/headlamp",
            "Power bank",
            "Reusable water bottle",
            "Insect repellent",
            "First-aid kit",
            "Weather-safe outer layer",
        ]
    if scenario == "birthday":
        return [
            "Guest list",
            "Cake order",
            "Decor checklist",
            "Music/playlist",
            "Food and drinks plan",
            "Gift or return favors",
        ]
    if scenario == "business":
        return [
            "Formal shirt/blouse",
            "Blazer",
            "Laptop + charger",
            "Business cards",
            "Meeting-ready shoes",
        ]
    if scenario == "wedding":
        return [
            "Main wedding outfit",
            "Backup festive outfit",
            "Jewelry/accessories",
            "Ethnic footwear",
            "Gift envelope",
        ]
    if scenario == "travel":
        return [
            "Sunscreen",
            "Sunglasses",
            "Beachwear/swimwear",
            "Toiletries kit",
            "Power bank",
        ]
    return [
        "Toiletries kit",
        "Phone + charger",
        "Personal medicine",
    ]


def _normalize_weather(context: Dict[str, Any]) -> str:
    destination_weather = (
        context.get("destination_weather")
        or context.get("destinationWeather")
        or context.get("trip_weather")
        or context.get("tripWeather")
    )
    if not destination_weather and str(context.get("weather_scope") or "").lower() == "destination":
        destination_weather = context.get("weather") or context.get("weather_data")
    # Direct engine callers historically passed `weather`; retain that only
    # when no device-context provenance is present.
    if not destination_weather and not context.get("context_usage") and not context.get("location_context"):
        destination_weather = context.get("weather") or context.get("weather_data")
    if isinstance(destination_weather, dict):
        weather = str(
            destination_weather.get("condition")
            or destination_weather.get("weather_type")
            or destination_weather.get("summary")
            or ""
        ).lower()
    else:
        weather = str(destination_weather or "").lower()
    if any(k in weather for k in ["rain", "storm", "drizzle"]):
        return "rainy"
    if any(k in weather for k in ["cold", "chill", "winter"]):
        return "cold"
    if any(k in weather for k in ["hot", "heat", "humid", "warm", "summer"]):
        return "hot"
    if any(k in weather for k in ["mild", "clear", "cloud", "temperate"]):
        return "mild"
    return "unavailable"


def _time_of_day(context: Dict[str, Any]) -> str:
    value = str(context.get("time_of_day") or context.get("time") or "").lower()
    if value in ("morning", "afternoon", "evening", "night"):
        return value
    return "daytime"


def _weather_layer_items(weather: str) -> List[str]:
    if weather == "rainy":
        return ["Compact rain jacket", "Waterproof footwear", "Quick-dry bag cover"]
    if weather == "cold":
        return ["Warm jacket", "Thermal innerwear", "Socks x2 extra"]
    if weather == "hot":
        return ["Breathable cotton/linen", "Cap/hat", "Hydration bottle"]
    if weather == "mild":
        return ["Light layer for evenings"]
    return ["Destination weather unavailable - check forecast"]


def _time_based_tasks(time_of_day: str) -> List[str]:
    if time_of_day == "morning":
        return [
            "Pack documents and chargers the night before",
            "Keep a quick breakfast/snack ready",
        ]
    if time_of_day == "evening":
        return [
            "Keep one ready-to-wear outfit on top",
            "Add travel-size freshening kit",
        ]
    if time_of_day == "night":
        return ["Keep sleepwear and essentials accessible", "Add eye mask/comfort kit"]
    return ["Keep first-day essentials in carry-on"]


def _timeline_checklist(days: int, scenario: str) -> List[str]:
    if scenario == "birthday":
        return [
            "Confirm date and time",
            "Finalize guest list",
            "Set venue or home setup",
            "Order cake",
            "Plan decor",
            "Plan food and drinks",
            "Send invitations",
            "Prepare music/playlist",
            "Arrange return gifts/favors",
            "Add to calendar",
        ]

    base = [
        "Confirm travel/event dates",
        "Book transport and stay",
        "Prepare outfits by day",
        "Pack essentials the night before",
    ]
    if scenario == "business":
        base.extend(["Prepare meeting deck", "Keep IDs and booking invoices ready"])
    if scenario == "wedding":
        base.extend(["Confirm ceremony timeline", "Coordinate with family/group"])
    if scenario == "camping":
        base.extend(["Check campsite rules", "Confirm route and emergency contacts"])
    if days >= 5:
        base.append("Add laundry plan for longer stay")
    return base


def _ui_cards(
    days: int,
    destination: str,
    scenario: str,
    weather: str,
    time_of_day: str,
    *,
    explicit_duration: bool = True,
    wardrobe: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    clothes = _packing_clothes(days=days, scenario=scenario)
    smart_packing = packing_engine.build_packing(
        {
            "days": days,
            "purpose": scenario,
            "destination": destination,
            "weather": weather,
            "gender": "universal",
        }
    )
    addons: List[str] = []
    for card in smart_packing.get("cards", []):
        title = str((card or {}).get("title") or (card or {}).get("category") or "").lower()
        if title not in {"purpose", "weather", "activity"}:
            continue
        items = card.get("items") if isinstance(card, dict) else []
        if isinstance(items, list):
            addons.extend([str(item) for item in items if str(item).strip()])
    if not addons:
        addons = _scenario_addons(scenario=scenario)
    timeline = _timeline_checklist(days=days, scenario=scenario)
    weather_items = _weather_layer_items(weather=weather)
    if scenario != "birthday":
        timeline = timeline + _time_based_tasks(time_of_day=time_of_day)

    if scenario == "birthday":
        primary_title = "Birthday Party Plan"
    elif scenario == "camping":
        primary_title = "Camping Prep Checklist"
    elif scenario == "wedding":
        primary_title = "Wedding Prep Checklist"
    elif scenario == "business":
        primary_title = "Business Travel Plan"
    elif destination == "Goa":
        primary_title = f"{days}-Day Goa Trip"
    elif destination == "Carry-On Trip":
        primary_title = "Carry-on Packing Checklist"
    else:
        primary_title = f"{days}-Day Plan"

    duration_label = f"{days} days" if explicit_duration else "Short trip"
    primary_subtitle = (
        "Birthday event"
        if scenario == "birthday"
        else f"{destination} · {duration_label}"
        if destination not in {"Your Trip", "Carry-On Trip"}
        else "Short carry-on trip"
        if destination == "Carry-On Trip"
        else duration_label
    )
    calendar_action = {
        "type": "open_module",
        "module": "calendar",
        "intent": "open_calendar",
        "route": "/organize/calendar",
        "label": "Open calendar",
    }
    checklist_action = {
        "type": "plan_pack_action",
        "module": "plan_pack",
        "intent": "open_checklist",
        "route": "plan_pack_checklist",
        "label": "Open checklist",
    }

    cards = [
        {
            "id": "trip_plan",
            "title": primary_title,
            "kind": "checklist",
            "subtitle": primary_subtitle,
            "items": [
                _visual_item(item, category="plan", wardrobe=wardrobe)
                for item in timeline
            ],
            "action": calendar_action if scenario == "birthday" else {
                "type": "plan_pack_action",
                "module": "plan_pack",
                "intent": "view_plan",
                "route": "plan_pack",
                "label": "View plan",
            },
        },
    ]
    if scenario == "birthday":
        return cards

    cards.extend([
        {
            "id": "packing_clothes",
            "title": "Packing List - Clothes",
            "kind": "checklist",
            "subtitle": duration_label,
            "items": [
                _visual_item(item, category="clothes", wardrobe=wardrobe)
                for item in clothes
            ],
            "action": checklist_action,
        },
        {
            "id": "packing_essentials",
            "title": "Packing List - Essentials",
            "kind": "checklist",
            "subtitle": destination if destination != "Your Trip" else scenario.title(),
            "items": [
                _visual_item(item, category="essentials", wardrobe=wardrobe)
                for item in addons
            ],
            "action": checklist_action,
        },
        {
            "id": "weather_time_adjustments",
            "title": "Weather & Time Adjustments",
            "kind": "checklist",
            "subtitle": f"{weather.title()} | {time_of_day.title()}",
            "items": [
                _visual_item(item, category="weather", wardrobe=wardrobe)
                for item in weather_items
            ],
            "action": {
                "type": "plan_pack_action",
                "module": "plan_pack",
                "intent": "weather_prep",
                "route": "plan_pack_weather",
                "label": "Weather prep",
            },
        },
    ])
    return cards


def build_plan_pack_response(
    text: str, context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    context = context or {}
    wardrobe = context.get("wardrobe") if isinstance(context.get("wardrobe"), list) else []
    explicit_duration = _has_explicit_duration(text)
    days = _parse_days(text)
    scenario = _detect_scenario(text)
    destination = _extract_destination(text)
    weather = _normalize_weather(context=context)
    time_of_day = _time_of_day(context=context)

    cards = _ui_cards(
        days=days,
        destination=destination,
        scenario=scenario,
        weather=weather,
        time_of_day=time_of_day,
        explicit_duration=explicit_duration,
        wardrobe=wardrobe,
    )
    duration_label = f"{days} days" if explicit_duration else "Short trip"
    visual_sections = build_visual_packing_sections(
        cards,
        wardrobe=wardrobe,
        destination=destination,
        duration_label=duration_label,
        weather=weather,
        time_of_day=time_of_day,
    )
    if scenario == "birthday":
        message = "Built your birthday party plan."
    elif destination == "Carry-On Trip" and not explicit_duration:
        message = "Built your carry-on packing checklist for a short trip."
    else:
        message = f"Built your {scenario} plan and weather-aware packing checklist for {days} days."

    return {
        "intent": "plan_pack",
        "message": message,
        "board": "plan_pack",
        "type": "checklists",
        "visual_type": "visual_packing_checklist",
        "chips": ["Open checklist", "Plan outfits", "Weather prep", "Save trip plan"],
        "quick_actions": [
            {"label": "Open checklist", "module": "plan_pack", "intent": "open_checklist"},
            {"label": "Plan outfits", "module": "style", "intent": "plan_outfits"},
            {"label": "Weather prep", "module": "plan_pack", "intent": "weather_prep"},
            {"label": "Save trip plan", "module": "plan_pack", "intent": "save_plan"},
        ],
        "cards": cards,
        "visual_sections": visual_sections,
        "data": {
            "days": days,
            "duration_label": duration_label,
            "destination": destination,
            "scenario": scenario,
            "weather": weather,
            "weather_status": "available" if weather != "unavailable" else "unavailable",
            "weather_scope": "destination",
            "time_of_day": time_of_day,
            "can_save_to_life_board": True,
            "source_text": text,
            "visual_type": "visual_packing_checklist",
            "visual_sections": visual_sections,
        },
    }
