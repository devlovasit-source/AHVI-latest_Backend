from __future__ import annotations

from typing import Any, Dict, List, Tuple


LOUD_COLORS = {
    "yellow", "orange", "neon", "fluorescent",
    "bright yellow", "bright orange", "lime",
}

SMART_OCCASIONS = {
    "smart casual", "date", "date night", "office",
    "business casual", "evening", "evening casual",
    "dinner", "brunch",
}

MALE_BLOCKED_CATEGORIES = {
    "saree", "lehenga", "skirt", "gown", "dress",
    "heels", "heel", "heeled boots", "heeled_boots",
    "women sandals", "women_sandals",
}

SMART_FOOTWEAR_GOOD = {
    "sneakers", "minimal sneakers", "minimal_sneakers",
    "white sneakers", "cream sneakers", "loafers",
    "chelsea boots", "chelsea_boots", "formal shoes",
    "leather sneakers", "boots",
}

ATHLETIC_FOOTWEAR = {
    "running shoes", "sports shoes", "athletic shoes",
    "trainers", "gym shoes",
}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _item_text(item: Dict[str, Any]) -> str:
    parts = [
        item.get("name"),
        item.get("title"),
        item.get("category"),
        item.get("sub_category"),
        item.get("subcategory"),
        item.get("type"),
        item.get("color"),
        item.get("style"),
    ]
    return " ".join(_norm(p) for p in parts if p)


def _item_category(item: Dict[str, Any]) -> str:
    return _norm(
        item.get("category")
        or item.get("sub_category")
        or item.get("subcategory")
        or item.get("type")
    )


def _item_color(item: Dict[str, Any]) -> str:
    return _norm(item.get("color") or item.get("dominant_color"))


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


def _dedupe_accessories(accessories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    final = []

    for item in accessories or []:
        category = _item_category(item) or _norm(item.get("name") or item.get("title"))

        if category in seen:
            continue

        seen.add(category)
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

    occasion_text = _norm(
        intent
        or outfit.get("occasion")
        or outfit.get("use_case")
        or outfit.get("scenario")
    )

    outfit_text = " ".join([
        _item_text(top),
        _item_text(bottom),
        _item_text(footwear),
        occasion_text,
    ])

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
