import base64
import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

try:
    from services.r2_storage import R2Storage, R2StorageError
except Exception:  # pragma: no cover - optional deploy dependency
    R2Storage = None
    R2StorageError = Exception

logger = logging.getLogger("ahvi.style_flow")


STYLE_ACTION_CHIPS = ["More looks", "Next best options", "Try different shoes"]


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _tokens(value: Any) -> set[str]:
    import re

    return set(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


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
    blob = " ".join(_safe_text(item.get(k)) for k in ("name", "label", "category", "type", "sub_category"))
    tokens = _tokens(blob)
    for typ, words in (
        ("watch", {"watch", "watches"}),
        ("eyewear", {"sunglass", "sunglasses", "eyewear", "glasses"}),
        ("bag", {"bag", "bags", "purse", "tote", "clutch"}),
        ("belt", {"belt", "belts"}),
        ("headwear", {"cap", "hat"}),
        ("scarf", {"scarf", "scarves"}),
        ("jewelry", {"ring", "necklace", "bracelet", "earring", "jewelry", "jewellery"}),
    ):
        if tokens.intersection(words):
            return typ
    return "accessory"


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
        "office": any(k in q for k in ("office", "work", "meeting", "client")),
        "date": any(k in q for k in ("date", "dinner", "night")),
        "party": any(k in q for k in ("party", "club", "after-hours", "night out")),
        "travel": any(k in q for k in ("travel", "airport", "flight", "vacation", "trip")),
        "wedding": any(k in q for k in ("wedding", "reception", "ceremony", "sangeet", "formal event")),
        "casual": any(k in q for k in ("casual", "daily", "today", "weekend", "errand", "coffee")),
    }


def _item_by_role(card: Dict[str, Any], role: str) -> Dict[str, Any]:
    for item in card.get("items", []):
        if isinstance(item, dict) and item_role(item) == role:
            return item
    return {}


def _role_key(card: Dict[str, Any], role: str) -> str:
    return item_key(_item_by_role(card, role))


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


def _occasion_kind(query: str) -> str:
    flags = _occasion_flags(query)
    for key in ("office", "date", "party", "travel", "wedding", "casual"):
        if flags.get(key):
            return key
    return "daily"


def _style_direction(query: str) -> str:
    q = str(query or "").lower()
    if any(k in q for k in ("corporate", "boardroom", "formal", "client", "presentation")):
        return "corporate_office"
    if any(k in q for k in ("creative", "agency", "studio", "design")):
        return "creative_office"
    if any(k in q for k in ("startup", "start-up", "casual office")):
        return "startup_office"
    if any(k in q for k in ("friday", "relaxed office", "casual friday")):
        return "friday_office"
    flags = _occasion_flags(query)
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


def _office_direction(query: str) -> str:
    # Backward-compatible name for existing callers/tests. The returned value is
    # now the general style direction, not only an office subtype.
    return _style_direction(query)


def _footwear_mood(item: Dict[str, Any]) -> str:
    text = _item_blob(item)
    if any(k in text for k in ("loafer", "oxford", "derby", "formal", "monk strap")):
        return "formal polish"
    if any(k in text for k in ("leather sneaker", "minimal sneaker", "white sneaker", "cream sneaker", "clean sneaker")):
        return "polished sneaker"
    if "sneaker" in text:
        return "casual sneaker"
    if any(k in text for k in ("birkenstock", "sandal", "slipper", "slider", "slides", "flip flop", "flip-flop", "crocs")):
        return "relaxed sandal"
    if any(k in text for k in ("boot", "chelsea")):
        return "structured boot"
    if any(k in text for k in ("running", "trainer", "gym", "athletic", "sports")):
        return "athletic"
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
            "relaxed sandal": -5.0,
            "athletic": -4.0,
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
    top_bottom = "|".join([top, bottom])
    selected_bottoms = [_role_key(s, "bottom") for s in selected]
    selected_top_bottoms = {
        "|".join([_role_key(s, "top") or _role_key(s, "dress"), _role_key(s, "bottom")])
        for s in selected
    }
    if bottom and bottom in selected_bottoms:
        penalty += selected_bottoms.count(bottom) * 2.5
    if top_bottom.strip("|") and top_bottom in selected_top_bottoms:
        penalty += 6.0
    if _footwear_mood(_item_by_role(card, "footwear")) in {
        _footwear_mood(_item_by_role(s, "footwear")) for s in selected
    }:
        penalty += 0.8
    if _role_key(card, "footwear") and _role_key(card, "footwear") in {_role_key(s, "footwear") for s in selected}:
        penalty += 1.6
    if _style_energy(card, _safe_text(card.get("_style_query"))) in {
        _style_energy(s, _safe_text(s.get("_style_query"))) for s in selected
    }:
        penalty += 1.0
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
    if flags["date"] and any(k in text for k in ("watch", "black", "off white", "loafer")):
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
    palette = _palette_direction(card)
    silhouette = _silhouette_mood(card)
    footwear_mood = _footwear_mood(_item_by_role(card, "footwear"))

    copy = {
        "color_harmony": f"The {top} sets a {palette} direction, while {bottom} keeps the palette grounded so the outfit reads intentional.",
        "silhouette_balance": f"The {top} and {bottom} create a {silhouette} shape, with {footwear} anchoring the proportions instead of competing with them.",
        "texture_contrast": f"The contrast between {top} and {bottom} gives the look depth, and {footwear} keeps the finish practical without flattening the styling.",
        "occasion_alignment": f"For this request, {top} feels appropriate because {bottom} keeps the base controlled and {footwear} supports the occasion.",
        "footwear_polish": f"The {footwear} changes the mood to {footwear_mood}, making the outfit feel styled rather than just matched.",
        "smart_contrast": f"The sharper read of {top} works against the easier base of {bottom}, giving the board a clean smart-casual tension.",
        "minimal_aesthetic": f"This keeps the outfit restrained: {top}, {bottom}, and {footwear} form a simple line with no unnecessary visual noise.",
        "relaxed_tailoring": f"The look stays relaxed but neat; {top} adds structure, {bottom} keeps it wearable, and {footwear} finishes it with ease.",
    }
    tips = [
        f"Roll the sleeves once on {top} if you want the office look to feel less stiff.",
        f"Keep {top} untucked only if the hem sits above mid-hip.",
        f"Swap to loafers after 6 PM if this needs to move from office to dinner.",
        "Keep accessories minimal here; the cleaner line is what makes the board feel premium.",
        f"Let {footwear} stay visible; it is doing the polish work in this outfit.",
        f"If {bottom} is slim, avoid oversized accessories so the silhouette stays sharp.",
        "This palette works best in daylight because the contrast stays clean without feeling heavy.",
        f"Use a single watch or eyewear piece; more than that will distract from {top}.",
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


def _office_has_strong_footwear(cards: List[Dict[str, Any]], query: str) -> bool:
    if not _occasion_flags(query)["office"]:
        return False
    return any(_footwear_formality_score(_item_by_role(card, "footwear"), query) > 0.5 for card in cards)


MAX_BOTTOM_REUSE = 2
MAX_FOOTWEAR_REUSE = 2
MAX_ACCESSORY_REUSE = 2


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


def _selected_count(selected: List[Dict[str, Any]], role: str, value: str) -> int:
    if not value:
        return 0
    return sum(1 for selected_card in selected if _role_key(selected_card, role) == value)


def _select_diverse_cards(cards: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
    if not cards:
        return []
    unique_bottoms = {k for k in (_role_key(card, "bottom") for card in cards) if k}
    unique_footwear = {k for k in (_role_key(card, "footwear") for card in cards) if k}
    unique_accessories = {key for card in cards for key in _accessory_keys(card)}
    unique_energies = {_style_energy(card, query) for card in cards}
    enforce_bottom_limit = len(unique_bottoms) > 1
    enforce_footwear_limit = len(unique_footwear) > 1
    enforce_accessory_limit = len(unique_accessories) > 1
    strong_office_footwear_exists = _office_has_strong_footwear(cards, query)
    selected: List[Dict[str, Any]] = []
    selected_sigs: set[str] = set()

    def can_add(card: Dict[str, Any], *, strict: bool) -> bool:
        core = _safe_text(card.get("_style_core_signature"))
        if core in selected_sigs:
            return False
        bottom = _role_key(card, "bottom")
        footwear = _role_key(card, "footwear")
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
            top_bottom = "|".join([_role_key(card, "top") or _role_key(card, "dress"), bottom])
            selected_top_bottoms = {
                "|".join([_role_key(s, "top") or _role_key(s, "dress"), _role_key(s, "bottom")])
                for s in selected
            }
            if top_bottom.strip("|") and top_bottom in selected_top_bottoms:
                return False
            if len(selected) < 3:
                hero = _role_key(card, "top") or _role_key(card, "dress")
                if hero and hero in {_role_key(s, "top") or _role_key(s, "dress") for s in selected}:
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
        sig = card_signature(fixed)
        core_sig = core_card_signature(fixed) or sig
        if not sig or core_sig in seen or sig in excluded or core_sig in excluded:
            continue
        fixed["_style_signature"] = sig
        fixed["_style_core_signature"] = core_sig
        fixed["_style_quality_score"] = _quality_score(fixed, query) + _occasion_fit_score(fixed, query)
        fixed["_style_query"] = query
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

    rendered: List[Dict[str, Any]] = []
    style_dna = _dict(context.get("style_dna"))
    try:
        from brain.engines.style_board_engine import style_board_engine
        from brain.engines.style_board_renderer import style_board_renderer
    except Exception as exc:
        logger.warning("style board renderer unavailable: %s", exc)
        return []

    for idx, card in enumerate(cards):
        items = card.get("items") if isinstance(card.get("items"), list) else []
        if not items:
            continue
        board = {}
        image_bytes = b""
        try:
            board = style_board_engine.build_board(
                {
                    "items": items,
                    "score": card.get("score"),
                    "style_archetype": card.get("style_archetype"),
                    "style_energy": card.get("style_energy"),
                    "layout_preset": card.get("layout_preset"),
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
                    "visual_hierarchy": card.get("visual_hierarchy"),
                    "composition_notes": card.get("composition_notes"),
                    "diversity_profile": card.get("diversity_profile"),
                }
            )
            image_bytes = style_board_renderer.render(board)
        except Exception as exc:
            logger.warning("style board render failed user=%s idx=%s error=%s", user_id, idx, exc)

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
                    "visual_hierarchy": card.get("visual_hierarchy"),
                    "composition_notes": card.get("composition_notes"),
                },
            }
        )
    return rendered


def _style_signature_hash(signatures: List[str]) -> str:
    return hashlib.sha1("|".join(signatures).encode("utf-8")).hexdigest() if signatures else ""


def finalize_style_response_payload(
    result: Dict[str, Any],
    *,
    user_id: str,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    include_base64: bool = False,
    upload_to_r2: bool = False,
    style_action: str = "",
    exclude_style_signatures: Any = None,
    requested_board_count: Optional[int] = None,
    cache_bypass: bool = True,
) -> Dict[str, Any]:
    ctx = dict(context or {})
    raw_cards = result.get("cards") if isinstance(result.get("cards"), list) else []
    raw_outfits = result.get("outfits") if isinstance(result.get("outfits"), list) else []
    candidates = list(raw_outfits or []) + list(raw_cards or [])

    cards = finalize_style_cards(
        candidates,
        query=query,
        exclude_signatures=exclude_style_signatures,
        requested_count=requested_board_count if style_action in {"more_options", "more_looks", "next_best"} else None,
    )
    ids = board_item_ids(cards)
    rendered = render_style_boards(
        cards,
        ctx,
        user_id=user_id,
        include_base64=include_base64,
        upload_to_r2=upload_to_r2,
    )
    signatures = [card.get("_style_signature") or card_signature(card) for card in cards]
    core_signatures = [card.get("_style_core_signature") or core_card_signature(card) for card in cards]
    style_signature = _style_signature_hash([s for s in signatures if s])
    primary_board_id = ids[0] if ids else ""

    logger.info(
        "style_flow.final_response user=%s cards=%d core_signatures=%s signatures=%s style_action=%s cache_bypass=%s",
        user_id,
        len(cards),
        core_signatures,
        signatures,
        style_action or "",
        bool(cache_bypass),
    )

    data = {
        "outfits": cards,
        "visual_intelligence": visual_intelligence_from_outfit(raw_outfits[0]) if raw_outfits and isinstance(raw_outfits[0], dict) else {},
        "pipeline": _dict(result.get("pipeline")),
        "rendered_boards": rendered or cards,
        "board_item_ids": ids,
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
            "board_count": len(cards),
            "has_more_style_options": bool(cards),
            "cache_bypass": bool(cache_bypass),
            "style_cache_bypass": bool(cache_bypass),
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

    ctx = dict(context or {})
    ctx.setdefault("query", query)
    ctx.setdefault("user_id", user_id)
    if user_profile is not None:
        ctx.setdefault("user_profile", user_profile)

    result = get_daily_outfits(
        {
            "user_id": user_id,
            "wardrobe": wardrobe,
            "context": ctx,
        }
    )
    if not isinstance(result, dict):
        result = {}

    finalized = finalize_style_response_payload(
        result,
        user_id=user_id,
        query=query,
        context=ctx,
        include_base64=include_base64,
        upload_to_r2=upload_to_r2,
        style_action=style_action,
        exclude_style_signatures=exclude_style_signatures,
        requested_board_count=requested_board_count,
        cache_bypass=cache_bypass,
    )
    cards = finalized["cards"]
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
        },
    }
