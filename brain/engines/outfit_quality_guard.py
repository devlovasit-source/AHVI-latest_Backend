from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from brain.engines.style_rules_engine import style_engine
from services.wardrobe_intelligence_service import (
    _style_meta,
    board_has_occasion_conflict,
    normalize_occasion as normalize_style_occasion,
)
from services.wardrobe_suitability import outfit_contains_private_wear

logger = logging.getLogger("ahvi.outfit_quality_guard")

try:
    from services.agent_metadata_validator import item_metadata_v2_reject_reason
except Exception:  # pragma: no cover
    item_metadata_v2_reject_reason = None  # type: ignore

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
    return set(re.sub(r"[^a-z0-9]+", " ", _norm(text)).split())


def _is_t_shirt_text(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", _norm(text)).strip()
    return any(
        f" {term} " in f" {normalized} "
        for term in ("t shirt", "tshirt", "tee")
    )


def normalize_occasion(occasion: Any) -> str:
    text = _norm(occasion).replace("-", "_")
    readable = text.replace("_", " ")
    tokens = _tokens(readable)
    if (
        any(w in readable for w in ["workout", "gym", "fitness", "strength training"])
        or tokens.intersection({"training", "yoga", "running", "cardio", "pilates"})
    ):
        return "workout"
    if "casual dinner" in readable:
        return "casual_dinner"
    if any(w in readable for w in ("client dinner", "business dinner")):
        return "client_dinner"
    if "dinner" in readable and any(
        w in readable for w in ("formal", "refined", "elegant", "black tie", "dressy")
    ):
        return "client_dinner"
    if any(w in text for w in ["date_night", "date night", "date", "dinner", "tonight"]):
        return "date_night"
    if any(w in text for w in ["beach", "pool", "seaside", "coastal", "resort"]):
        return "beach"
    if (
        any(w in text for w in ["office", "corporate_office", "smart_casual_office", "meeting", "client", "boardroom"])
        or "work" in tokens
    ):
        return "office"
    if "brunch" in text:
        return "brunch"
    if any(w in text for w in ["rave", "club"]):
        return "rave"
    if "cocktail" in text:
        return "cocktail"
    if any(w in text for w in ["party", "house_party", "after_hours", "night out"]):
        return "party"
    if any(w in text for w in ["travel", "airport", "flight", "vacation", "trip"]):
        return "travel"
    if any(w in text for w in ["temple_modest", "temple", "mandir", "pooja", "puja", "religious", "shrine", "darshan"]):
        return "temple_modest"
    if any(
        w in text
        for w in [
            "wedding", "reception", "ceremony", "event", "engagement",
            "sangeet", "haldi", "mehendi", "mehndi", "baraat", "nikah",
            "shaadi", "roka", "varmala",
        ]
    ):
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
    if outfit_contains_private_wear(board):
        logger.info(
            "editorial_guard_rejected_private_wear occasion=%s title=%s",
            occasion,
            board.get("title") if isinstance(board, dict) else "",
        )
        return True, "private_wear_forbidden"
    if board_has_occasion_conflict(board, occasion):
        return True, f"metadata_forbidden_{normalize_style_occasion(occasion)}"
    blob = _board_blob(board)
    occasion = normalize_occasion(occasion)

    if occasion == "client_dinner":
        board_items = list(board.get("items") or [])
        for key in ("top", "bottom", "dress", "footwear", "shoes", "outerwear"):
            if isinstance(board.get(key), dict):
                board_items.append(board[key])
        for item in board_items:
            if not isinstance(item, dict):
                continue
            item_text = _item_text(item)
            if _is_t_shirt_text(item_text):
                return True, "formal_dinner_forbidden_t_shirt"
            if _role_from_item(item) == "bottom" and _tokens(item_text) & {"jeans", "denim"}:
                return True, "formal_dinner_forbidden_denim"

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
            "shorts",
            "running shorts",
            "gym shorts",
            "sliders",
            "slides",
            "slippers",
            "flip flop",
            "flip-flop",
        ]
        for word in forbidden:
            if word in blob:
                return True, f"date_forbidden_{word}"

    if occasion == "beach":
        forbidden = [
            "black trousers",
            "black pants",
            "loafers",
            "formal loafers",
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
            "shorts",
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
            "red loafers",
            "burgundy loafers",
            "red shoes",
            "party shirt",
            "loud print",
            "loud pattern",
            "tropical",
            "hawaiian",
            "floral",
            "ornate",
            "shiny",
            "satin",
            "glossy",
            # Always-on deterministic safety: never let casual/private/gym
            # wear reach an office / client-meeting / board-meeting board.
            "boxer",
            "boxers",
            "brief",
            "briefs",
            "underwear",
            "track pant",
            "track pants",
            "trackpant",
            "sweatpant",
            "sweatpants",
            "joggers",
            "gym",
            "gymwear",
            "activewear",
            "athleisure",
            "jersey",
            "sleepwear",
            "pajama",
            "pyjama",
            "nightwear",
            "crocs",
            "beachwear",
            "swim",
            "swimwear",
            "tank top",
            # Casual headwear has no place in office / client / board meetings.
            # "beanie" is a safe substring (matches "whitebeanie"); multi-word
            # terms avoid false hits like "capsule" / "escape".
            "beanie",
            "snapback",
            "baseball cap",
            "bucket hat",
            "skull cap",
        ]
        for word in forbidden:
            if word in blob:
                return True, f"office_forbidden_{word.strip()}"

    if occasion == "wedding":
        # Deterministic wedding/formal-event safety. Always on — independent of
        # any LLM or agent_orchestration payload. Blocks casual, private, gym,
        # beach, and sloppy footwear from wedding/engagement/sangeet/haldi
        # boards.
        forbidden = [
            "boxer",
            "boxers",
            "brief",
            "briefs",
            "underwear",
            "innerwear",
            "track pant",
            "track pants",
            "trackpant",
            "sweatpant",
            "sweatpants",
            "joggers",
            "gym",
            "gymwear",
            "activewear",
            "athleisure",
            "jersey",
            "sleepwear",
            "pajama",
            "pyjama",
            "nightwear",
            "shorts",
            "gym shorts",
            "running shorts",
            "beachwear",
            "swim",
            "swimwear",
            "crocs",
            "flip flop",
            "flip-flop",
            "slider",
            "sliders",
            "slipper",
            "slippers",
            "hoodie",
            "tank top",
            "beanie",
            "snapback",
            "baseball cap",
            "bucket hat",
            "skull cap",
        ]
        for word in forbidden:
            if word in blob:
                return True, f"wedding_forbidden_{word.strip().replace(' ', '_')}"

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


# ── Complete-outfit slot enforcement ─────────────────────────────────────────
# A "complete outfit" board must be one of:
#   (top + bottom + footwear)   OR   (one-piece/dress + footwear)
# where one-piece covers dress / gown / jumpsuit / saree / lehenga (all map to
# role "dress" via _role_from_item). Accessories are optional and NEVER satisfy a
# required clothing slot. Missing required slots -> repair from safe assets
# (respecting source policy) or reject; never emit an incomplete "complete" board.

def board_roles(items: Any) -> set:
    roles = set()
    for item in items or []:
        if isinstance(item, dict):
            role = _role_from_item(item)
            if role:
                roles.add(role)
    return roles


def is_complete_board(items: Any) -> bool:
    roles = board_roles(items)
    if "footwear" not in roles:
        return False
    if "dress" in roles:  # one-piece: no separate top/bottom required
        return True
    return {"top", "bottom"}.issubset(roles)


def missing_required_slots(items: Any) -> List[str]:
    roles = board_roles(items)
    if "dress" in roles:
        return [] if "footwear" in roles else ["footwear"]
    return [slot for slot in ("top", "bottom", "footwear") if slot not in roles]


def _source_ok_for_policy(item: Dict[str, Any], source_policy: str) -> bool:
    try:
        from services.style_item_contract import canonical_item_source
        src = _norm(canonical_item_source(dict(item)))
    except Exception:
        src = _norm(item.get("source") or item.get("provenance") or item.get("origin"))
    policy = _norm(source_policy)
    if policy in {"wardrobe_only", "wardrobe", "owned"}:
        # Wardrobe-only repair must never pull catalog / style assets.
        return src in {"wardrobe", "owned"}
    return True


def _pick_repair_candidate(
    pool: List[Dict[str, Any]], slot: str, source_policy: str, used_keys: set
) -> Dict[str, Any] | None:
    for cand in pool or []:
        if not isinstance(cand, dict):
            continue
        if _role_from_item(cand) != slot:
            continue
        if _stable_item_key(cand) in used_keys:
            continue
        if not _source_ok_for_policy(cand, source_policy):
            continue
        return cand
    return None


def repair_or_reject_board(
    card: Dict[str, Any],
    *,
    source_policy: str = "",
    candidate_pool: Any = None,
) -> Tuple[Dict[str, Any] | None, str]:
    """Return (card, status). Complete -> (card, "complete"). Otherwise fill
    missing required slots from candidate_pool honoring source_policy
    (wardrobe_only never uses catalog/style assets). If it cannot be completed
    -> (None, "rejected_incomplete"). Never fetches; pure over the given pool."""
    if not isinstance(card, dict):
        return None, "rejected_incomplete"
    items = [i for i in (card.get("items") or []) if isinstance(i, dict)]
    if is_complete_board(items):
        return card, "complete"
    pool = [c for c in (candidate_pool or []) if isinstance(c, dict)]
    used = {_stable_item_key(i) for i in items}
    repaired = list(items)
    for slot in missing_required_slots(items):
        cand = _pick_repair_candidate(pool, slot, source_policy, used)
        if cand is None:
            return None, "rejected_incomplete"
        repaired.append(cand)
        used.add(_stable_item_key(cand))
    if not is_complete_board(repaired):
        return None, "rejected_incomplete"
    out = dict(card)
    out["items"] = repaired
    return out, "repaired"


def filter_complete_boards(cards: Any) -> List[Dict[str, Any]]:
    """Drop cards that are not complete outfits (reject-only, no repair)."""
    out: List[Dict[str, Any]] = []
    for card in cards or []:
        if isinstance(card, dict) and is_complete_board(card.get("items")):
            out.append(card)
    return out


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
    toks = _tokens(text)
    for blocked in MALE_BLOCKED_CATEGORIES:
        clean = blocked.replace("_", " ")
        if len(clean) <= 3:
            if clean in toks:
                return True
            continue
        if clean in text:
            return True
    return False


def _is_date_or_evening_context(context_text: str) -> bool:
    return any(x in context_text for x in ["date", "date night", "date_night", "dinner", "evening", "night out", "smart casual"])


def _is_office_context(context_text: str) -> bool:
    return any(x in context_text for x in ["office", "business", "meeting", "work"])


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
            elif _has_any(footwear_text, {"formal shoes", "formal loafers", "loafers", "loafer", "oxford", "derby"}):
                delta -= 28
                reasons.append("Formal footwear feels heavy for beach/resort context")

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

        if (
            "shirt" in _tokens(top_text)
            and not _is_t_shirt_text(top_text)
            and _has_any(footwear_text, {"loafers", "formal shoes", "leather sneakers"})
        ):
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


_AHVI_PROFESSIONAL_OCCASIONS = {
    "office", "business", "business_casual", "client", "meeting",
    "interview", "boardroom", "corporate", "professional",
}
_AHVI_FORMAL_OCCASIONS = {
    "wedding", "formal", "ceremony", "reception", "gala", "cocktail",
}
_AHVI_LOUNGEWEAR_TERMS = {
    "boxer", "boxers", "pyjama", "pajama", "pj ", "lounge", "loungewear",
    "sleepwear", "nightwear", "night suit", "innerwear",
}
_AHVI_GYMWEAR_TERMS = {
    "gym", "gymwear", "activewear", "athleisure", "yoga pant",
    "sports bra", "track pant", "tracks", "compression",
}
_AHVI_OPEN_TOE_TERMS = {
    "slipper", "slide", "slider", "sandal", "flip flop", "flip-flop",
    "open toe", "open-toe",
}


def _ahvi_guard_blob(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return " ".join(
        str(item.get(k) or "").lower()
        for k in (
            "name",
            "title",
            "label",
            "category",
            "sub_category",
            "subcategory",
            "type",
            "slot",
            "role",
            "garment_type",
        )
    )


def _ahvi_accessory_type_for_guard(item: Dict[str, Any]) -> str:
    blob = _ahvi_guard_blob(item)
    for typ, terms in (
        ("watch", ("watch",)),
        ("belt", ("belt",)),
        ("bag", ("bag", "tote", "clutch", "purse")),
        ("eyewear", ("sunglass", "eyewear", "glasses")),
        ("headwear", ("cap", "hat", "beanie")),
        ("jewelry", ("ring", "necklace", "bracelet", "earring", "jewelry", "jewellery")),
        ("scarf", ("scarf",)),
    ):
        if any(t in blob for t in terms):
            return typ
    return ""


def _ahvi_agent_quality_reject(
    outfit: Dict[str, Any],
    top: Dict[str, Any],
    bottom: Dict[str, Any],
    footwear: Dict[str, Any],
    accessories: List[Dict[str, Any]],
    occasion_text: str,
    agent_payload: Dict[str, Any],
) -> str:
    """Return non-empty rejection reason if the agent's policy is violated."""
    occasion = (occasion_text or "").lower()
    if isinstance(agent_payload, dict):
        agent_occ = str(agent_payload.get("occasion") or "").lower()
        if agent_occ:
            occasion = f"{occasion} {agent_occ}".strip()
        avoid = [str(t).strip().lower() for t in (agent_payload.get("avoid_items") or []) if str(t).strip()]
    else:
        avoid = []
    is_professional = any(o in occasion for o in _AHVI_PROFESSIONAL_OCCASIONS)
    is_formal = any(o in occasion for o in _AHVI_FORMAL_OCCASIONS)
    allow_open_toe = bool(
        isinstance(agent_payload, dict)
        and (agent_payload.get("accessory_policy") or {}).get("allow_open_toe_footwear")
    )

    blobs = {
        "top": _ahvi_guard_blob(top),
        "bottom": _ahvi_guard_blob(bottom),
        "footwear": _ahvi_guard_blob(footwear),
    }

    # Lounge / sleepwear in professional setting.
    if is_professional or is_formal:
        for slot, blob in blobs.items():
            if any(term in blob for term in _AHVI_LOUNGEWEAR_TERMS):
                return f"loungewear/sleepwear in {slot} blocked for professional/formal context"

    # Gymwear for formal events.
    if is_formal or is_professional:
        for slot, blob in blobs.items():
            if any(term in blob for term in _AHVI_GYMWEAR_TERMS):
                return f"gymwear in {slot} blocked for formal/professional context"

    # Open-toe footwear in business meetings unless explicitly allowed.
    if (is_professional or is_formal) and blobs["footwear"]:
        if any(term in blobs["footwear"] for term in _AHVI_OPEN_TOE_TERMS) and not allow_open_toe:
            return "open-toe footwear blocked for professional/formal context"

    # avoid_items list (any slot or accessory).
    if avoid:
        for slot, blob in blobs.items():
            if any(term in blob for term in avoid):
                return f"agent avoid_items: blocked {slot}"
        for acc in accessories or []:
            blob = _ahvi_guard_blob(acc)
            if any(term in blob for term in avoid):
                return "agent avoid_items: blocked accessory"

    # Duplicate accessory type (watch+watch, belt+belt, etc.)
    seen_types: set[str] = set()
    for acc in accessories or []:
        typ = _ahvi_accessory_type_for_guard(acc)
        if not typ:
            continue
        if typ in seen_types:
            return f"duplicate accessory type: {typ}"
        seen_types.add(typ)

    return ""


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

    # ---- AHVI Metadata Validator: per-item style_metadata guard ----
    occasion_for_meta = (intent or str(outfit.get("occasion") or query) or "").strip().lower()
    if occasion_for_meta:
        for slot_item in [top, bottom, footwear, *accessories]:
            if not isinstance(slot_item, dict):
                continue
            if item_metadata_v2_reject_reason is not None:
                archetype = (
                    outfit.get("style_archetype")
                    or _style_meta(slot_item).get("style_archetype")
                    or ""
                )
                v2_reason = item_metadata_v2_reject_reason(
                    slot_item,
                    occasion=occasion_for_meta,
                    archetype=archetype,
                )
                if v2_reason:
                    logger.info(
                        "AHVI_ITEM_OCCASION_REJECTED occasion=%s item=%s reason=%s source=outfit_quality_guard",
                        occasion_for_meta,
                        slot_item.get("id") or slot_item.get("$id") or slot_item.get("name"),
                        v2_reason,
                    )
                    return False, -100, [v2_reason], fixed
                logger.info(
                    "AHVI_METADATA_V2_USED occasion=%s item=%s",
                    occasion_for_meta,
                    slot_item.get("id") or slot_item.get("$id") or slot_item.get("name"),
                )
            meta = slot_item.get("style_metadata")
            if not isinstance(meta, dict):
                continue
            blocked = [str(o).strip().lower() for o in (meta.get("blocked_occasions") or [])]
            if blocked and any(o and o in occasion_for_meta for o in blocked):
                reason = f"style_metadata.blocked_occasions matched {occasion_for_meta}"
                logger.info("ahvi.metadata.guard_reject item=%s reason=%s", slot_item.get("$id"), reason)
                return False, -100, [reason], fixed
            formality = str(meta.get("formality") or "").strip().lower()
            style_role = str(meta.get("style_role") or "").strip().lower()
            professional = any(
                o in occasion_for_meta
                for o in ("office", "client", "meeting", "business", "interview", "boardroom")
            )
            formal = any(
                o in occasion_for_meta
                for o in ("wedding", "formal", "ceremony", "reception", "gala")
            )
            if professional and formality in {"homewear", "lounge", "loungewear", "sleepwear"}:
                return False, -100, [f"style_metadata.formality={formality} blocked for professional"], fixed
            if (professional or formal) and style_role in {"activewear", "gymwear", "athleisure"}:
                return False, -100, [f"style_metadata.style_role={style_role} blocked for formal/professional"], fixed
            confidence = float(meta.get("confidence") or 0.0)
            if confidence and confidence < 0.4 and (professional or formal):
                penalty -= 10
                reasons.append("low-confidence style_metadata for high-stakes occasion")

            # ---- Metadata Validator richness signals ----
            def _meta_float(key: str) -> float:
                try:
                    return float(meta.get(key) or 0.0)
                except Exception:
                    return 0.0

            client_meeting_score = _meta_float("client_meeting_score")
            boardroom_score = _meta_float("boardroom_score")
            capsule_score = _meta_float("capsule_score")
            versatility_score = _meta_float("versatility_score")
            date_night_score = _meta_float("date_night_score")

            visual_noise = str(meta.get("visual_noise") or "").strip().lower()
            statement_level = str(meta.get("statement_level") or "").strip().lower()
            pattern_intensity = str(meta.get("pattern_intensity") or "").strip().lower()
            risk_flags = {
                str(flag).strip().lower()
                for flag in (meta.get("risk_flags") or [])
                if str(flag).strip()
            }

            if professional:
                if client_meeting_score and client_meeting_score < 0.50:
                    penalty -= 40
                    reasons.append("low client meeting suitability")

                if boardroom_score and boardroom_score < 0.45:
                    penalty -= 30
                    reasons.append("low boardroom suitability")

                if visual_noise == "high":
                    penalty -= 35
                    reasons.append("high visual noise for professional context")

                if pattern_intensity in {"loud", "moderate"}:
                    penalty -= 25
                    reasons.append("pattern intensity weakens professional polish")

                if statement_level in {"statement", "risky"}:
                    penalty -= 30
                    reasons.append("statement/risky garment in professional context")

                if "too_casual_for_professional" in risk_flags:
                    return False, -100, ["style_metadata risk: too casual for professional"], fixed

                if "high_visual_noise" in risk_flags and client_meeting_score < 0.55:
                    penalty -= 30
                    reasons.append("high visual noise risk")

            if formal:
                if boardroom_score and boardroom_score < 0.40:
                    penalty -= 35
                    reasons.append("low formal suitability")

                if statement_level == "risky":
                    return False, -100, ["style_metadata risk: risky garment for formal context"], fixed

            if "capsule" in occasion_for_meta or "capsule" in query.lower():
                if capsule_score:
                    if capsule_score < 0.45:
                        penalty -= 35
                        reasons.append("low capsule wardrobe suitability")
                    elif capsule_score >= 0.80:
                        penalty += 10
                        reasons.append("strong capsule wardrobe foundation")

                if versatility_score and versatility_score < 0.45:
                    penalty -= 20
                    reasons.append("low versatility for capsule wardrobe")

                if statement_level in {"statement", "risky"}:
                    penalty -= 30
                    reasons.append("statement garment weakens capsule wardrobe")

            if "date" in occasion_for_meta or "dinner" in occasion_for_meta:
                if date_night_score and date_night_score >= 0.75:
                    penalty += 8
                    reasons.append("strong date-night suitability")

    # ---- AHVI Style Orchestrator agent guard ----
    agent_payload = outfit.get("agent_orchestration") if isinstance(outfit, dict) else None
    if not isinstance(agent_payload, dict):
        agent_payload = (user_profile or {}).get("agent_orchestration") if isinstance(user_profile, dict) else None
    if isinstance(agent_payload, dict):
        agent_reason = _ahvi_agent_quality_reject(
            outfit, top, bottom, footwear, accessories,
            occasion_text=intent or str(outfit.get("occasion") or ""),
            agent_payload=agent_payload,
        )
        if agent_reason:
            logger.info("ahvi.agent.outfit_quality_reject reason=%s", agent_reason)
            return False, -100, [agent_reason], fixed

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

    if is_smart_occasion:
        bottom_text = _item_text(bottom)
        top_text = _item_text(top)
        if any(term in bottom_text for term in ("shorts", "short pants", "running shorts", "gym shorts")):
            return False, -100, ["Shorts blocked for office/date/client styling"], fixed
        if (
            "shirt" in top_text
            and "gold" in top_text
            and any(term in top_text for term in ("shiny", "metallic", "satin", "sequin", "sequins", "glossy"))
        ):
            return False, -100, ["Shiny gold shirt blocked for smart styling"], fixed

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


_OUTERWEAR_TERMS = {
    "blazer", "jacket", "coat", "overcoat", "trench", "overshirt",
    "cardigan", "sherwani", "bandhgala", "nehru",
}


def _outerwear_of(outfit: Dict[str, Any]) -> Dict[str, Any]:
    """The layer the UI shows as hero above the shirt (blazer/jacket/...)."""
    explicit = outfit.get("outerwear")
    if isinstance(explicit, dict) and explicit:
        return explicit
    for item in outfit.get("items") or []:
        if isinstance(item, dict) and _has_any(_item_text(item), _OUTERWEAR_TERMS):
            return item
    return {}


def _visible_hero(outfit: Dict[str, Any]) -> Dict[str, Any]:
    """Match the UI hero order exactly: dress > outerwear > top.

    The old diversity keyed on top+bottom, so two boards sharing the same
    blazer-as-hero but different shirts looked 'distinct' to the backend
    while the UI showed the same hero. This makes diversity see what the
    user sees.
    """
    dress = outfit.get("dress") if isinstance(outfit.get("dress"), dict) else {}
    if not dress:
        for item in outfit.get("items") or []:
            if isinstance(item, dict) and _role_from_item(item) == "dress":
                dress = item
                break
    if dress:
        return dress
    outer = _outerwear_of(outfit)
    if outer:
        return outer
    top, _b, _f, _a = _extract_outfit_slots(outfit)
    return top if isinstance(top, dict) else {}


def _item_has_image(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    for k in (
        "masked_url", "maskedUrl", "image_url", "imageUrl", "normalized_url",
        "normalizedUrl", "url", "raw_url", "rawUrl", "raw_image_base64",
        "masked_image_base64",
    ):
        if str(item.get(k) or "").strip():
            return True
    return False


def _enforce_outfit_diversity(
    outfits: List[Dict[str, Any]],
    *,
    min_keep: int = 6,
) -> List[Dict[str, Any]]:
    """Reject look-alike boards before they reach the carousel.

    Rules:
      - same VISIBLE hero (dress > outerwear > top) → only the best kept
      - same top+bottom pair → only the highest-scoring instance kept
      - a hero whose image is missing is demoted (no placeholder hero) when
        an image-backed alternative exists
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
    seen_hero: set[str] = set()
    bottom_count: Dict[str, int] = {}
    surviving: List[Dict[str, Any]] = []
    skipped: List[Tuple[Dict[str, Any], str]] = []
    hero_freq: Counter[str] = Counter()

    # Only demote image-less heroes when an image-backed alternative exists.
    # If the whole deck lacks images (bad data / tests), don't collapse it.
    any_image_hero = any(
        _item_has_image(_visible_hero(o)) for o in sorted_outfits
    )

    for outfit in sorted_outfits:
        top, bottom, footwear, _ = _extract_outfit_slots(outfit)
        top_key = _stable_item_key(top)
        bottom_key = _stable_item_key(bottom)

        hero = _visible_hero(outfit)
        hero_key = _stable_item_key(hero)
        hero_has_image = _item_has_image(hero)
        if hero_key:
            hero_freq[hero_key] += 1

        # Phase 2: never lead with a placeholder. Demote an image-less hero so
        # it only surfaces if nothing image-backed remains — but only when the
        # deck actually has an image-backed alternative.
        if hero_key and not hero_has_image and any_image_hero:
            skipped.append((outfit, "no_image_hero"))
            continue

        # Phase 1: the same visible hero must not repeat while alternatives
        # exist. Highest-scoring instance of each hero wins.
        if hero_key and hero_key in seen_hero:
            skipped.append((outfit, "duplicate_visible_hero"))
            continue

        dress_key = _stable_item_key(outfit.get("dress") or {})
        if not dress_key:
            # Treat any item with role=dress inside items[] as a dress slot.
            for item in outfit.get("items") or []:
                if isinstance(item, dict) and _role_from_item(item) == "dress":
                    dress_key = _stable_item_key(item)
                    break

        if dress_key:
            if dress_key in seen_dress:
                skipped.append((outfit, "duplicate_dress"))
                continue
            seen_dress.add(dress_key)
            if hero_key:
                seen_hero.add(hero_key)
            surviving.append(outfit)
            continue

        # Include the visible hero: a different blazer/jacket over the same
        # shirt+pants is a genuinely different look, so it must not be
        # collapsed by the top+bottom rule.
        pair_key = (
            f"{hero_key}|{top_key}|{bottom_key}" if top_key and bottom_key else ""
        )
        if pair_key and pair_key in seen_pair:
            # Same hero + top + bottom — shoe/accessory variation only.
            skipped.append((outfit, "duplicate_top_bottom"))
            continue
        if bottom_key:
            used = bottom_count.get(bottom_key, 0)
            if used >= 2:
                skipped.append((outfit, "bottom_reuse_cap"))
                continue
            bottom_count[bottom_key] = used + 1
        if pair_key:
            seen_pair.add(pair_key)
        if hero_key:
            seen_hero.add(hero_key)

        surviving.append(outfit)

    # Visibility for debugging the "same hero on every board" symptom.
    logger.info(
        "outfit_quality_guard.visible_hero kept=%d hero_frequency=%s",
        len(surviving),
        dict(hero_freq.most_common(8)),
    )

    strict_count = len(surviving)
    if len(surviving) < min_keep and skipped:
        # Keep quality strict, but don't let similarity-only rules starve
        # the carousel. Refill with image-backed, non-placeholder boards
        # first; a placeholder hero is the last resort.
        refill_priority = {
            "duplicate_visible_hero": 0,
            "duplicate_dress": 0,
            "duplicate_top_bottom": 1,
            "bottom_reuse_cap": 2,
            "no_image_hero": 9,  # last — avoid a placeholder hero if possible
        }
        ordered_skips = sorted(
            skipped, key=lambda pair: refill_priority.get(pair[1], 5)
        )
        selected_ids = {id(outfit) for outfit in surviving}
        for outfit, reason in ordered_skips:
            if len(surviving) >= min_keep:
                break
            if id(outfit) in selected_ids:
                continue
            fixed = dict(outfit)
            reasons = list(fixed.get("_quality_guard_reasons") or [])
            reasons.append(f"diversity_refill:{reason}")
            fixed["_quality_guard_reasons"] = reasons
            surviving.append(fixed)
            selected_ids.add(id(outfit))
        logger.info(
            "outfit_quality_guard.diversity_refill strict=%d refilled=%d target=%d skipped_reasons=%s",
            strict_count,
            len(surviving),
            min_keep,
            dict(Counter(reason for _, reason in skipped)),
        )

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
    reject_reasons: Counter[str] = Counter()

    # Lazy import to avoid circular: style_scorer also imports from this
    # package indirectly via wardrobe_intelligence_service.
    try:
        from brain.engines.style_scorer import score_occasion_compatibility
    except Exception:
        score_occasion_compatibility = None  # type: ignore

    occasion_context = {
        "occasion": intent,
        "interpreted_occasion": intent,
        "prompt": query,
        "resolved_prompt": query,
    }

    def _hero_name(o: Dict[str, Any]) -> str:
        if not isinstance(o, dict):
            return "?"
        t = o.get("top") or o.get("dress") or {}
        return str((t or {}).get("name") or (t or {}).get("label") or "?")[:30]

    for outfit in outfits or []:
        allowed, penalty, reasons, fixed = guard_outfit(
            outfit=outfit,
            user_profile=user_profile,
            intent=intent,
            query=query,
        )

        if not allowed:
            for reason in list(reasons or [])[:5]:
                reject_reasons[str(reason)] += 1
            if not reasons:
                reject_reasons["guard_penalty_threshold"] += 1
            logger.info(
                "outfit_quality_guard.rejected occ=%s hero=%r penalty=%d reasons=%s",
                intent or "",
                _hero_name(outfit),
                int(penalty or 0),
                list((reasons or []))[:5],
            )
            continue

        # Global occasion compatibility — applies to every occasion, not
        # just beach. Drops outfits that hard-reject on the matrix and
        # demotes outfits that fall below the min_required_score.
        if score_occasion_compatibility is not None:
            occ_result = score_occasion_compatibility(fixed, occasion_context)
            fixed.setdefault("score_meta", {})
            fixed["score_meta"].update({
                "occasion_compatibility_score": occ_result["score"],
                "occasion_boosts": occ_result["boosts"],
                "occasion_penalties": occ_result["penalties"],
                "occasion_reject": occ_result["reject"],
                "occasion_reason": occ_result["reason"],
                "occasion_resolved": occ_result["occasion"],
            })
            if occ_result["reject"]:
                reject_reasons[
                    "occasion_mismatch:"
                    + (",".join(occ_result.get("penalties") or [])[:80] or occ_result.get("reason") or "unknown")
                ] += 1
                logger.info(
                    "outfit_quality_editorial_rank reason='occasion_mismatch: %s for %s' score_meta=%s",
                    ", ".join(occ_result["penalties"][:3]) or occ_result["reason"],
                    occ_result["occasion"] or "?",
                    fixed["score_meta"],
                )
                logger.info(
                    "outfit_quality_guard.rejected occ=%s hero=%r reason=occasion_mismatch penalties=%s",
                    intent or "",
                    _hero_name(outfit),
                    list(occ_result.get("penalties") or [])[:3],
                )
                # Drop entirely.
                continue
            # Apply occasion-aware weighting:
            # +20 for a perfect fit, +0 at the midpoint, large negative
            # for poor fits so colour-only matches don't win.
            occ_delta = (occ_result["score"] - 0.5) * 40.0
            penalty += occ_delta
            if occ_result["penalties"]:
                reasons = list(reasons) + [
                    f"occasion_penalty:{p}" for p in occ_result["penalties"][:3]
                ]
            if occ_result["boosts"]:
                reasons = list(reasons) + [
                    f"occasion_boost:{b}" for b in occ_result["boosts"][:3]
                ]

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
    pre_diversity_count = len(guarded)
    guarded = _apply_bottom_reuse_cooldown(guarded)
    # min_keep=1: keep every genuinely-distinct look but NEVER pad the carousel
    # with near-duplicates (same top+bottom, swapped shoe/accessory). A thin
    # wardrobe now shows fewer strong looks instead of 6 repeated ones. Refill
    # only triggers to avoid an empty carousel.
    guarded = _enforce_outfit_diversity(guarded, min_keep=1)
    logger.info(
        "outfit_quality_guard.summary occ=%s accepted_before_diversity=%d after_diversity=%d rejected_reasons=%s",
        intent or "",
        pre_diversity_count,
        len(guarded),
        dict(reject_reasons.most_common(8)),
    )
    return guarded

