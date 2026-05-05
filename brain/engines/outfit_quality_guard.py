from __future__ import annotations

from typing import Any, Dict, List, Tuple

LOUD_COLORS = {"yellow", "orange", "neon", "fluorescent", "bright yellow", "bright orange", "lime"}

SMART_OCCASIONS = {
    "smart casual", "date", "date night", "office", "business casual",
    "evening", "evening casual", "dinner", "brunch",
}

MALE_BLOCKED_CATEGORIES = {
    "saree", "lehenga", "skirt", "gown", "dress", "heels", "heel",
    "heeled boots", "heeled_boots", "women sandals", "women_sandals",
}

SMART_FOOTWEAR_GOOD = {
    "sneakers", "minimal sneakers", "minimal_sneakers", "white sneakers",
    "cream sneakers", "loafers", "chelsea boots", "chelsea_boots",
    "formal shoes", "leather sneakers", "boots",
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
    "sandals", "sandal", "crocs",
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
        for word in [
            "bold", "streetwear", "sporty", "sneakerhead",
            "statement", "athletic", "gym",
        ]
    )


def _is_loud_footwear(item: Dict[str, Any]) -> bool:
    text = _item_text(item)
    color = _item_color(item)
    return any(c in color for c in LOUD_COLORS) or any(c in text for c in LOUD_COLORS)


def _is_athletic_footwear(item: Dict[str, Any]) -> bool:
    text = _item_text(item)
    return any(x in text for x in ATHLETIC_FOOTWEAR)


def _is_male_blocked_item(item: Dict[str, Any]) -> bool:
    text = _item_text(item)
    return any(blocked in text for blocked in MALE_BLOCKED_CATEGORIES)


def _is_date_or_evening_context(context_text: str) -> bool:
    return any(x in context_text for x in ["date", "date night", "dinner", "evening", "night out", "smart casual"])


def _is_office_context(context_text: str) -> bool:
    return any(x in context_text for x in ["office", "business", "meeting", "work", "formal"])


def _is_beach_or_relaxed_context(context_text: str) -> bool:
    return any(x in context_text for x in ["beach", "pool", "resort", "vacation", "holiday", "errand", "lounge"])


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
    is_rain = _is_rain_context(context_text)
    is_hot = _is_hot_context(context_text)

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

    top = outfit.get("top") or outfit.get("shirt") or outfit.get("upper") or {}
    bottom = outfit.get("bottom") or outfit.get("pants") or outfit.get("trouser") or {}
    footwear = outfit.get("footwear") or outfit.get("shoes") or outfit.get("shoe") or {}
    accessories = outfit.get("accessories") or []

    occasion_text = _norm(intent or outfit.get("occasion") or outfit.get("use_case") or outfit.get("scenario"))

    outfit_text = " ".join([_item_text(top), _item_text(bottom), _item_text(footwear), occasion_text])

    if _is_male_user(user_profile):
        for item in [top, bottom, footwear, *accessories]:
            if item and _is_male_blocked_item(item):
                return False, -100, ["Blocked item for male profile unless explicitly requested"], fixed

    is_smart_occasion = any(o in occasion_text or o in outfit_text for o in SMART_OCCASIONS)
    bold_requested = _explicitly_requested_bold(intent, query)

    if footwear:
        footwear_text = _item_text(footwear)

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
        top=top if isinstance(top, dict) else {},
        bottom=bottom if isinstance(bottom, dict) else {},
        footwear=footwear if isinstance(footwear, dict) else {},
        accessories=accessories if isinstance(accessories, list) else [],
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

        guarded.append(fixed)

    guarded.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return guarded
