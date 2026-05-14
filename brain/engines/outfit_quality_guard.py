from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from brain.engines.style_rules_engine import style_engine

logger = logging.getLogger("ahvi.outfit_quality_guard")

LOUD_COLORS = {"yellow", "orange", "neon", "fluorescent", "bright yellow", "bright orange", "lime"}

SMART_OCCASIONS = {
    "smart casual", "date", "date night", "date_night", "office", "business casual",
    "evening", "evening casual", "dinner", "brunch",
}

MALE_BLOCKED_CATEGORIES = {
    "saree", "lehenga", "skirt", "gown", "dress", "heels", "heel",
    "heeled boots", "heeled_boots", "women sandals", "women_sandals",
    "sports bra", "sports_bra", "bra", "bralette", "bustier",
}

SMART_FOOTWEAR_GOOD = {
    "sneakers", "minimal sneakers", "minimal_sneakers", "white sneakers",
    "cream sneakers", "loafers", "chelsea boots", "chelsea_boots",
    "formal shoes", "leather sneakers", "boots", "leather boots",
}

ATHLETIC_FOOTWEAR = {
    "running shoes", "sports shoes", "athletic shoes", "trainers", "gym shoes",
}

DATE_NIGHT_FOOTWEAR_PREMIUM = {
    "chelsea boots", "chelsea_boots", "leather boots", "dress boots", "boots",
    "loafers", "formal shoes", "derby", "oxford", "monk strap",
}

DATE_NIGHT_FOOTWEAR_OK = {
    "minimal sneakers", "minimal_sneakers", "white sneakers", "cream sneakers",
    "leather sneakers", "clean sneakers",
}

RELAXED_FOOTWEAR = {
    "slider", "sliders", "slipper", "slippers", "flip flop", "flip-flop",
    "sandals", "sandal", "crocs", "birkenstock", "birkenstocks",
}

RAIN_BAD_FOOTWEAR = {
    "suede", "canvas", "white sneakers", "cream sneakers",
    "slipper", "slippers", "slider", "sliders",
}

RAIN_GOOD_FOOTWEAR = {
    "boots", "chelsea boots", "chelsea_boots", "leather boots",
    "waterproof", "dark sneakers",
}

HEAVY_FOOTWEAR = {"boots", "chunky boots", "combat boots", "heavy boots"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _item_text(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    parts = [
        item.get("name"), item.get("title"), item.get("label"),
        item.get("category"), item.get("sub_category"), item.get("subcategory"),
        item.get("type"), item.get("slot"), item.get("role"),
        item.get("category_group"), item.get("garment_type"),
        item.get("color"), item.get("dominant_color"),
        item.get("style"), item.get("material"), item.get("fabric"),
    ]
    return " ".join(_norm(p) for p in parts if p)


def _item_category(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return _norm(
        item.get("category")
        or item.get("sub_category")
        or item.get("subcategory")
        or item.get("type")
        or item.get("slot")
        or item.get("role")
    )


def _item_color(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return _norm(item.get("color") or item.get("dominant_color"))


def _has_any(text: str, terms: set[str]) -> bool:
    clean = _norm(text).replace("_", " ")
    return any(term.replace("_", " ") in clean for term in terms)


def _tokens(text: str) -> set[str]:
    import re

    return set(re.sub(r"[^a-z0-9]+", " ", _norm(text)).split())


def normalize_occasion(occasion: Any) -> str:
    text = _norm(occasion).replace("-", "_")
    if any(w in text for w in ["date_night", "date night", "date", "dinner", "tonight"]):
        return "date_night"
    if any(w in text for w in ["beach", "pool", "seaside", "coastal", "resort"]):
        return "beach"
    if any(w in text for w in ["office", "corporate_office", "smart_casual_office", "work", "meeting", "client", "boardroom"]):
        return "office"
    if "brunch" in text:
        return "brunch"
    if any(w in text for w in ["party", "club", "rave", "after_hours", "night out", "cocktail"]):
        return "party"
    if any(w in text for w in ["travel", "airport", "flight", "vacation", "trip"]):
        return "travel"
    if any(w in text for w in ["workout", "gym", "fitness", "training", "yoga", "running"]):
        return "workout"
    if any(w in text for w in ["temple_modest", "temple", "mandir", "pooja", "puja", "religious", "shrine", "darshan"]):
        return "temple_modest"
    if any(w in text for w in ["wedding", "reception", "ceremony", "event"]):
        return "wedding"
    if any(w in text for w in ["daily", "today"]):
        return "daily"
    if any(w in text for w in ["casual", "weekend"]):
        return "casual"
    return text


def _board_blob(board: dict) -> str:
    parts = [
        board.get("title"),
        board.get("badge"),
        board.get("occasion"),
        board.get("occasion_label"),
        board.get("explanation"),
        board.get("why"),
        board.get("why_it_works"),
        board.get("style_direction"),
    ]

    for item in board.get("items") or []:
        if not isinstance(item, dict):
            continue
        parts.extend(
            [
                item.get("name"),
                item.get("category"),
                item.get("subcategory"),
                item.get("sub_category"),
                item.get("color"),
                item.get("material"),
            ]
        )

    return " ".join(str(p or "") for p in parts).lower()


def reject_board_for_occasion(board: dict, occasion: str) -> Tuple[bool, str]:
    blob = _board_blob(board)
    occasion = normalize_occasion(occasion)

    if occasion == "date_night":
        forbidden = [
            "boardroom",
            "professional",
            "office",
            "executive",
            "corporate",
            "workwear",
            "clean friday",
            "client",
        ]
        for word in forbidden:
            if word in blob:
                return True, f"date_forbidden_{word}"

    if occasion == "beach":
        forbidden = [
            "black trousers",
            "black pants",
            "loafers",
            "dress shoes",
            "blazer",
            "office",
            "boardroom",
            "professional",
            "formal polish",
        ]
        for word in forbidden:
            if word in blob:
                return True, f"beach_forbidden_{word}"

    if occasion == "office":
        forbidden = [
            "slippers",
            "beach sandals",
            "gym shorts",
            "running shorts",
            "strapless",
            "camisole",
            " cami ",
            "corset",
            "bralette",
            "crop top",
            "mini skirt",
            "miniskirt",
            "club",
            "party dress",
            "embroidered",
            "sequin",
            "festive",
            "flip flop",
            "flip-flop",
            "sliders",
            "slides",
            "birkenstock",
            "espadrille",
        ]
        for word in forbidden:
            if word in blob:
                return True, f"office_forbidden_{word.strip()}"

    if occasion == "temple_modest":
        forbidden = [
            "strapless",
            "camisole",
            " cami ",
            "corset",
            "bralette",
            "crop top",
            " crop ",
            "shorts",
            "mini skirt",
            "miniskirt",
            " mini ",
            "club",
            "party dress",
            "deep neck",
            "low neck",
            "backless",
            "combat boots",
            "chunky boots",
        ]
        for word in forbidden:
            if word in blob:
                return True, f"temple_forbidden_{word.strip()}"

    return False, ""


def _role_from_item(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""

    explicit = (
        item.get("role")
        or item.get("slot")
        or item.get("type")
        or item.get("category_group")
        or item.get("category")
        or item.get("sub_category")
        or item.get("subcategory")
        or item.get("garment_type")
    )
    normalized = _norm(explicit).replace(" ", "_")
    role_map = {
        "tops": "top",
        "top": "top",
        "shirt": "top",
        "shirts": "top",
        "tshirt": "top",
        "t_shirt": "top",
        "tee": "top",
        "bottoms": "bottom",
        "bottom": "bottom",
        "pant": "bottom",
        "pants": "bottom",
        "trouser": "bottom",
        "trousers": "bottom",
        "jean": "bottom",
        "jeans": "bottom",
        "footwear": "footwear",
        "shoes": "footwear",
        "shoe": "footwear",
        "sneaker": "footwear",
        "sneakers": "footwear",
        "boot": "footwear",
        "boots": "footwear",
        "loafer": "footwear",
        "loafers": "footwear",
        "slipper": "footwear",
        "slippers": "footwear",
        "slider": "footwear",
        "sliders": "footwear",
        "sandal": "footwear",
        "sandals": "footwear",
        "dress": "dress",
        "dresses": "dress",
        "accessory": "accessory",
        "accessories": "accessory",
        "watch": "accessory",
        "bag": "accessory",
        "belt": "accessory",
        "jewelry": "accessory",
        "jewellery": "accessory",
    }
    if normalized in role_map:
        return role_map[normalized]

    blob = _item_text(item)
    toks = _tokens(blob)

    if "dress shirt" in blob:
        return "top"
    if toks & {"shirt", "shirts", "tshirt", "tee", "top", "tops", "polo", "kurta", "hoodie"}:
        return "top"
    if toks & {"pants", "pant", "trousers", "trouser", "jeans", "jean", "chinos", "chino", "shorts", "skirt"}:
        return "bottom"
    if toks & {"footwear", "shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "loafer", "loafers", "sandal", "sandals", "slipper", "slippers", "slider", "sliders"}:
        return "footwear"
    if toks & {"watch", "watches", "bag", "belt", "jewelry", "jewellery", "ring", "bracelet", "necklace", "sunglasses", "cap"}:
        return "accessory"
    if toks & {"dress", "gown", "jumpsuit", "saree", "lehenga"}:
        return "dress"

    return ""


def _extract_outfit_slots(outfit: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    top = outfit.get("top") or outfit.get("shirt") or outfit.get("upper") or {}
    bottom = outfit.get("bottom") or outfit.get("pants") or outfit.get("trouser") or {}
    footwear = outfit.get("footwear") or outfit.get("shoes") or outfit.get("shoe") or {}
    accessories = outfit.get("accessories") or []

    if not isinstance(accessories, list):
        accessories = []

    # Important: many AHVI outfits/cards are item-list based, not top/bottom fields.
    # Without this extraction, Tan Sliders in outfit["items"] will not be penalized.
    items = outfit.get("items") if isinstance(outfit.get("items"), list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = _role_from_item(item)
        if role == "top" and not top:
            top = item
        elif role == "bottom" and not bottom:
            bottom = item
        elif role == "footwear" and not footwear:
            footwear = item
        elif role == "accessory":
            key = str(item.get("id") or item.get("$id") or item.get("item_id") or item.get("name") or "")
            existing = {
                str(x.get("id") or x.get("$id") or x.get("item_id") or x.get("name") or "")
                for x in accessories
                if isinstance(x, dict)
            }
            if key not in existing:
                accessories.append(item)

    return (
        top if isinstance(top, dict) else {},
        bottom if isinstance(bottom, dict) else {},
        footwear if isinstance(footwear, dict) else {},
        accessories,
    )


def _is_male_user(user_profile: Dict[str, Any]) -> bool:
    text = " ".join(
        _norm(user_profile.get(k))
        for k in ["gender", "style_gender", "sex", "persona"]
        if user_profile.get(k)
    )
    return "male" in text or "man" in text


def _explicitly_requested_bold(intent: str, query: str) -> bool:
    text = f"{_norm(intent)} {_norm(query)}"
    return any(
        word in text
        for word in ["bold", "streetwear", "sporty", "sneakerhead", "statement", "athletic", "gym"]
    )


def _is_loud_footwear(item: Dict[str, Any]) -> bool:
    text = _item_text(item)
    color = _item_color(item)
    return any(c in color for c in LOUD_COLORS) or any(c in text for c in LOUD_COLORS)


def _is_athletic_footwear(item: Dict[str, Any]) -> bool:
    text = _item_text(item)
    return any(x in text for x in ATHLETIC_FOOTWEAR)


def _is_relaxed_footwear(item: Dict[str, Any]) -> bool:
    return _has_any(_item_text(item), RELAXED_FOOTWEAR)


def _is_male_blocked_item(item: Dict[str, Any]) -> bool:
    text = _item_text(item)
    return any(blocked in text for blocked in MALE_BLOCKED_CATEGORIES)


def _is_date_or_evening_context(context_text: str) -> bool:
    return any(x in context_text for x in ["date", "date night", "date_night", "dinner", "evening", "night out", "smart casual"])


def _is_office_context(context_text: str) -> bool:
    return any(x in context_text for x in ["office", "business", "meeting", "work", "formal"])


def _is_beach_or_relaxed_context(context_text: str) -> bool:
    return any(x in context_text for x in ["beach", "pool", "resort", "vacation", "holiday", "errand", "lounge"])


def _is_temple_context(context_text: str) -> bool:
    return any(x in context_text for x in ["temple", "mandir", "pooja", "puja", "religious", "shrine", "darshan", "temple_modest"])


def _is_rain_context(context_text: str) -> bool:
    return any(x in context_text for x in ["rain", "rainy", "storm", "wet", "monsoon"])


def _is_hot_context(context_text: str) -> bool:
    return any(x in context_text for x in ["hot", "humid", "summer", "heat", "sunny", "warm"])


def _contextual_occasion_weather_adjustment(
    *,
    top: Dict[str, Any],
    bottom: Dict[str, Any],
    footwear: Dict[str, Any],
    accessories: List[Dict[str, Any]],
    occasion_text: str,
    query: str,
    outfit: Dict[str, Any],
) -> Tuple[int, List[str]]:
    delta = 0
    reasons: List[str] = []

    footwear_text = _item_text(footwear)
    top_text = _item_text(top)
    bottom_text = _item_text(bottom)
    accessory_text = " ".join(_item_text(a) for a in accessories or [] if isinstance(a, dict))

    weather_bits = [
        outfit.get("weather"),
        outfit.get("weather_condition"),
        outfit.get("weather_mode"),
        outfit.get("temperature_mode"),
        outfit.get("season"),
    ]

    context_text = " ".join([
        _norm(occasion_text),
        _norm(query),
        _norm(outfit.get("intent")),
        _norm(outfit.get("use_case")),
        _norm(outfit.get("scenario")),
        " ".join(_norm(x) for x in weather_bits if x),
        footwear_text,
        top_text,
        bottom_text,
    ])

    is_date_evening = _is_date_or_evening_context(context_text)
    is_office = _is_office_context(context_text)
    is_beach_relaxed = _is_beach_or_relaxed_context(context_text)
    is_temple = _is_temple_context(context_text)
    is_rain = _is_rain_context(context_text)
    is_hot = _is_hot_context(context_text)

    if is_temple:
        temple_blob = " ".join([top_text, bottom_text, accessory_text])
        if any(t in temple_blob for t in ("saree", "sari", "kurta", "kurti", "salwar", "anarkali", "dupatta", "long skirt")):
            delta += 14
            reasons.append("Traditional silhouette fits temple etiquette")
        if any(t in temple_blob for t in ("modest", "covered", "long sleeve", "full sleeve")):
            delta += 6
            reasons.append("Modest coverage suits temple context")
        if any(t in temple_blob for t in ("strapless", "cami", "camisole", "crop", "mini", "shorts", "backless", "deep neck", "low neck", "club", "party")):
            delta -= 40
            reasons.append("Revealing/club silhouette weakens temple respect")
        if any(t in footwear_text for t in ("combat", "chunky boots", "heel ", "high heel")):
            delta -= 14
            reasons.append("Heavy/western footwear feels off for temple")
        elif any(t in footwear_text for t in ("sandal", "sandals", "flat", "kolhapuri", "juti", "mojari")):
            delta += 6
            reasons.append("Easy-off footwear suits temple etiquette")

    if footwear:
        if is_date_evening:
            if _has_any(footwear_text, DATE_NIGHT_FOOTWEAR_PREMIUM):
                delta += 18
                reasons.append("Premium footwear improves date/evening polish")
            elif _has_any(footwear_text, DATE_NIGHT_FOOTWEAR_OK):
                delta += 8
                reasons.append("Clean footwear keeps the date look wearable")
            elif _has_any(footwear_text, RELAXED_FOOTWEAR):
                delta -= 28
                reasons.append("Relaxed footwear weakens date/evening polish")

        if is_office:
            if _has_any(footwear_text, {"loafers", "formal shoes", "oxford", "derby", "leather sneakers"}):
                delta += 14
                reasons.append("Footwear matches office polish")
            elif _has_any(footwear_text, RELAXED_FOOTWEAR):
                delta -= 35
                reasons.append("Relaxed footwear is too casual for office styling")

        if is_beach_relaxed:
            if _has_any(footwear_text, {"sandals", "sandal", "sliders", "slider", "slippers", "slipper"}):
                delta += 14
                reasons.append("Relaxed footwear fits beach/resort context")
            elif _has_any(footwear_text, {"formal shoes", "oxford", "derby"}):
                delta -= 14
                reasons.append("Formal footwear feels heavy for relaxed context")

        if is_rain:
            if _has_any(footwear_text, RAIN_GOOD_FOOTWEAR):
                delta += 10
                reasons.append("Footwear handles rainy weather better")
            if _has_any(footwear_text, RAIN_BAD_FOOTWEAR):
                delta -= 14
                reasons.append("Footwear is less practical for rain")

        if is_hot and not is_date_evening and not is_rain:
            if _has_any(footwear_text, HEAVY_FOOTWEAR):
                delta -= 8
                reasons.append("Heavy footwear can feel warm for hot weather")
            if _has_any(footwear_text, {"sandals", "sandal", "minimal sneakers", "white sneakers"}):
                delta += 6
                reasons.append("Footwear is breathable enough for warm weather")

    if is_date_evening:
        if (
            "shirt" in top_text
            and ("jeans" in bottom_text or "denim" in bottom_text)
            and _has_any(footwear_text, {"boots", "chelsea boots", "leather boots"})
        ):
            delta += 10
            reasons.append("Shirt, denim and boots create a strong date-night silhouette")

        if "shirt" in top_text and _has_any(footwear_text, {"loafers", "formal shoes", "leather sneakers"}):
            delta += 7
            reasons.append("Shirt and polished footwear improve smart-casual balance")

        if _has_any(accessory_text, {"watch", "bracelet", "ring"}):
            delta += 4
            reasons.append("Accessory adds controlled polish")

    return delta, reasons


def _dedupe_accessories(accessories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    final = []
    for item in accessories or []:
        if not isinstance(item, dict):
            continue
        category = _item_category(item) or _norm(item.get("name") or item.get("title"))
        key = category or _item_text(item)
        if key in seen:
            continue
        seen.add(key)
        final.append(item)
    return final


def guard_outfit(
    outfit: Dict[str, Any],
    user_profile: Dict[str, Any] | None = None,
    intent: str = "",
    query: str = "",
) -> Tuple[bool, int, List[str], Dict[str, Any]]:
    user_profile = user_profile or {}
    fixed = dict(outfit)
    reasons: List[str] = []
    penalty = 0

    top, bottom, footwear, accessories = _extract_outfit_slots(outfit)

    occasion_text = _norm(
        intent
        or outfit.get("occasion")
        or outfit.get("use_case")
        or outfit.get("scenario")
    )
    board_rejected, board_reason = reject_board_for_occasion(outfit, occasion_text or query)
    if board_rejected:
        return False, -100, [board_reason], fixed

    outfit_text = " ".join([_item_text(top), _item_text(bottom), _item_text(footwear), occasion_text])
    if query:
        outfit_text = f"{outfit_text} {_norm(query)}"

    # AHVI STYLE FLOW:
    # Central deterministic rule gate from style_rules_engine.
    # This keeps scorer + final guard aligned.
    try:
        rules = style_engine.get_scoring_rules(
            {},
            {
                "user_profile": user_profile,
                "intent": intent,
                "occasion": occasion_text,
                "query": query,
            },
        )
        if hasattr(style_engine, "is_blocked_item"):
            for item in [top, bottom, footwear, *accessories]:
                if not item:
                    continue
                blocked, reason = style_engine.is_blocked_item(item, rules)
                if blocked:
                    return False, -100, [f"Blocked by style rules: {reason}"], fixed
    except Exception:
        logger.exception("style_rule_guard_failed")

    if _is_male_user(user_profile):
        for item in [top, bottom, footwear, *accessories]:
            if item and _is_male_blocked_item(item):
                return False, -100, ["Blocked item for male profile unless explicitly requested"], fixed

    is_smart_occasion = any(o in occasion_text or o in outfit_text for o in SMART_OCCASIONS)
    bold_requested = _explicitly_requested_bold(intent, query)

    if footwear:
        footwear_text = _item_text(footwear)
        relaxed_footwear = _is_relaxed_footwear(footwear)

        if (
            is_smart_occasion
            and relaxed_footwear
            and not bold_requested
            and not _is_beach_or_relaxed_context(outfit_text)
        ):
            return False, -100, ["Relaxed footwear blocked for smart/date/office styling"], fixed

        if is_smart_occasion and _is_loud_footwear(footwear) and not bold_requested:
            penalty -= 45
            reasons.append("Loud footwear weakens smart/elevated outfit")

        if is_smart_occasion and _is_athletic_footwear(footwear) and not bold_requested:
            penalty -= 35
            reasons.append("Athletic footwear does not match smart occasion")

        if is_smart_occasion and not any(x in footwear_text for x in SMART_FOOTWEAR_GOOD):
            penalty -= 15
            reasons.append("Footwear is not ideal for smart styling")

    contextual_delta, contextual_reasons = _contextual_occasion_weather_adjustment(
        top=top,
        bottom=bottom,
        footwear=footwear,
        accessories=accessories,
        occasion_text=occasion_text,
        query=query,
        outfit=outfit if isinstance(outfit, dict) else {},
    )

    penalty += contextual_delta
    reasons.extend(contextual_reasons)
    fixed["_editorial_rank_delta"] = contextual_delta
    fixed["_editorial_rank_reasons"] = contextual_reasons

    if accessories:
        fixed["accessories"] = _dedupe_accessories(accessories)

    allowed = penalty > -70
    return allowed, penalty, reasons, fixed


def _stable_item_key(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return _norm(
        item.get("id")
        or item.get("$id")
        or item.get("item_id")
        or item.get("itemId")
        or item.get("name")
        or item.get("title")
    )


def _enforce_outfit_diversity(outfits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reject look-alike boards before they reach the carousel.

    Rules:
      - same top+bottom pair → only the highest-scoring instance kept
      - same dress → only the highest-scoring instance kept
      - shoe-only / accessory-only swaps are not new looks
      - same bottom appears at most twice across the deck
    """
    if not outfits:
        return outfits

    sorted_outfits = sorted(
        outfits, key=lambda x: float(x.get("score") or 0), reverse=True
    )

    seen_pair: set[str] = set()
    seen_dress: set[str] = set()
    bottom_count: Dict[str, int] = {}
    surviving: List[Dict[str, Any]] = []

    for outfit in sorted_outfits:
        top, bottom, footwear, _ = _extract_outfit_slots(outfit)
        top_key = _stable_item_key(top)
        bottom_key = _stable_item_key(bottom)
        dress_key = _stable_item_key(outfit.get("dress") or {})
        if not dress_key:
            # Treat any item with role=dress inside items[] as a dress slot.
            for item in outfit.get("items") or []:
                if isinstance(item, dict) and _role_from_item(item) == "dress":
                    dress_key = _stable_item_key(item)
                    break

        if dress_key:
            if dress_key in seen_dress:
                continue
            seen_dress.add(dress_key)
            surviving.append(outfit)
            continue

        pair_key = f"{top_key}|{bottom_key}" if top_key and bottom_key else ""
        if pair_key and pair_key in seen_pair:
            # Same top+bottom already present — shoe/accessory variation only.
            continue
        if bottom_key:
            used = bottom_count.get(bottom_key, 0)
            if used >= 2:
                continue
            bottom_count[bottom_key] = used + 1
        if pair_key:
            seen_pair.add(pair_key)

        surviving.append(outfit)

    return surviving


def _apply_bottom_reuse_cooldown(outfits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Demote repeated bottoms before the pipeline diversifier builds cards.

    The finalizer also enforces diversity, but raw `outfit_pipeline.final_cards`
    is created before the service finalizer logs. This keeps the raw pipeline
    from looking like six versions of the same black-pants outfit.
    """
    keyed: List[Tuple[str, Dict[str, Any]]] = []
    for outfit in outfits or []:
        _, bottom, _, _ = _extract_outfit_slots(outfit)
        key = _stable_item_key(bottom)
        if key:
            keyed.append((key, outfit))

    unique_bottoms = {key for key, _ in keyed}
    if len(unique_bottoms) <= 1:
        return outfits

    seen: Dict[str, int] = {}
    adjusted: List[Dict[str, Any]] = []
    for outfit in sorted(outfits, key=lambda x: float(x.get("score") or 0), reverse=True):
        _, bottom, _, _ = _extract_outfit_slots(outfit)
        bottom_key = _stable_item_key(bottom)
        repeat_count = seen.get(bottom_key, 0) if bottom_key else 0
        fixed = dict(outfit)

        if bottom_key and repeat_count >= 2:
            penalty = min(75, 24 * (repeat_count - 1))
            try:
                fixed["score"] = float(fixed.get("score") or 0.0) - penalty
                fixed["rank_score"] = float(fixed.get("score") or 0.0)
            except Exception:
                pass
            fixed["_bottom_reuse_penalty"] = -penalty
            reasons = list(fixed.get("_quality_guard_reasons") or [])
            reasons.append("Repeated bottom demoted to preserve outfit variety")
            fixed["_quality_guard_reasons"] = reasons

        if bottom_key:
            seen[bottom_key] = repeat_count + 1
        adjusted.append(fixed)

    adjusted.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return adjusted


def filter_and_guard_outfits(
    outfits: List[Dict[str, Any]],
    user_profile: Dict[str, Any] | None = None,
    intent: str = "",
    query: str = "",
) -> List[Dict[str, Any]]:
    guarded: List[Dict[str, Any]] = []

    for outfit in outfits or []:
        allowed, penalty, reasons, fixed = guard_outfit(
            outfit=outfit,
            user_profile=user_profile,
            intent=intent,
            query=query,
        )

        if not allowed:
            continue

        fixed["_quality_guard_penalty"] = penalty
        fixed["_quality_guard_reasons"] = reasons

        if "score" in fixed:
            try:
                fixed["score"] = float(fixed.get("score") or 0) + penalty
            except Exception:
                pass

        # AHVI V2.1: keep rank_score aligned with editorial-adjusted score.
        # _build_cards() prefers rank_score over score, so stale rank_score can
        # otherwise let weak footwear looks appear first.
        try:
            fixed["rank_score"] = float(fixed.get("score") or fixed.get("rank_score") or 0.0)
        except Exception:
            fixed["rank_score"] = fixed.get("score", fixed.get("rank_score", 0.0))

        logger.info(
            "outfit_quality_editorial_rank score=%s delta=%s footwear=%s reasons=%s",
            fixed.get("score"),
            fixed.get("_editorial_rank_delta"),
            _item_text(_extract_outfit_slots(fixed)[2])[:80],
            fixed.get("_editorial_rank_reasons"),
        )

        guarded.append(fixed)

    guarded.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    guarded = _apply_bottom_reuse_cooldown(guarded)
    guarded = _enforce_outfit_diversity(guarded)
    return guarded

