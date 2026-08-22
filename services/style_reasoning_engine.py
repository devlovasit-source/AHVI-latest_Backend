from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger("ahvi.style_reasoning")

from brain.tone.tone_engine import tone_engine
from prompts.core_prompts import AHVI_SYSTEM_PROMPT
from prompts.styling_prompts import OCCASION_INTERPRETER_PROMPT
from services.ai_gateway import generate_text, parse_json_object
from services.stylist_knowledge_service import (
    ARCHETYPE_LIBRARY,
    BODY_PROPORTION_ADVICE,
    COLOR_ADVICE,
    COLOR_BODY_ADVICE,
    OCCASION_ADVICE,
    SHOPPING_ASSIST,
    STYLE_ADVICE,
    STYLE_EDUCATION,
    STYLE_PAIRING,
    WARDROBE_STYLE,
    classify_style_mode,
)

GENERAL = "general"
VISUAL_INSPIRATION = "visual_inspiration"
_ADVICE_MODES = {BODY_PROPORTION_ADVICE, COLOR_ADVICE, OCCASION_ADVICE}

_STYLE_REASONING_MODES = {
    GENERAL,
    STYLE_ADVICE,
    VISUAL_INSPIRATION,
    WARDROBE_STYLE,
    SHOPPING_ASSIST,
    STYLE_EDUCATION,
    COLOR_BODY_ADVICE,
    STYLE_PAIRING,
    *_ADVICE_MODES,
}

_GEMINI_MODES = {
    STYLE_ADVICE,
    VISUAL_INSPIRATION,
    SHOPPING_ASSIST,
    STYLE_EDUCATION,
    COLOR_BODY_ADVICE,
    STYLE_PAIRING,
    *_ADVICE_MODES,
}


def _norm(value: Any) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


_RECURSIVE_PREFIXES = (
    # Long forms (chip values) first so they win over the bare suffix forms.
    "show visual inspiration for:",
    "show visual inspiration for",
    "use my wardrobe for:",
    "use my wardrobe for",
    "find missing pieces for:",
    "find missing pieces for",
    "show shopping ideas for:",
    "show shopping ideas for",
    # Bare forms — same intents without the leading verb. Cover the typed
    # variants that bypass chip wrapping (e.g. "visual inspiration for a
    # conference talk") so downstream extractors see the real tail.
    "visual inspiration for:",
    "visual inspiration for",
    "shopping ideas for:",
    "shopping ideas for",
    "missing pieces for:",
    "missing pieces for",
    "wardrobe for:",
    "wardrobe for",
)


def _clean_recursive_prompt(query: str) -> str:
    """Strip stacked action prefixes so the base occasion survives instead of
    polluting the prompt as "show visual inspiration for: show visual
    inspiration for: coffee date". Keeps only the trailing real intent."""
    text = str(query or "").strip()
    changed = True
    guard = 0
    while changed and guard < 6:
        changed = False
        guard += 1
        low = text.lower()
        for pref in _RECURSIVE_PREFIXES:
            if low.startswith(pref):
                text = text[len(pref):].strip(" :·-")
                changed = True
                break
        # Collapse an internal " · " chain to its last meaningful segment.
        if " · " in text:
            tail = text.split(" · ")[-1].strip()
            if tail:
                text = tail
                changed = True
    return text or str(query or "").strip()


def _intent_name(intent: dict | str | None) -> str:
    if isinstance(intent, dict):
        return _norm(intent.get("intent"))
    return _norm(intent)


def _confidence(intent: dict | str | None, fallback: float) -> float:
    if not isinstance(intent, dict):
        return fallback
    try:
        return max(0.0, min(1.0, float(intent.get("confidence", fallback))))
    except Exception:
        return fallback


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


_PAIRING_TRIGGERS = (
    "what to pair with",
    "what goes with",
    "how do i style",
    "how to style",
    "ways to wear",
    "ways to style",
    "how can i wear",
    "what matches",
    "style this",
    "pair this with",
    "pair with",
)

_ANCHOR_COLORS = (
    "white", "black", "grey", "gray", "navy", "blue", "brown", "tan",
    "beige", "cream", "olive", "green", "red", "burgundy", "pink",
    "yellow", "gold", "silver", "charcoal",
)

_ANCHOR_CATEGORIES = {
    "shirt": ("shirt", "shirts", "button down", "button-down", "oxford"),
    "footwear": ("loafers", "loafer", "sneakers", "sneaker", "shoes", "boots", "boot", "sandals", "sandal"),
    "bottom": ("trousers", "trouser", "pants", "pant", "jeans", "denim", "chinos", "chino", "shorts"),
    "outerwear": ("blazer", "jacket", "coat", "overshirt"),
    "top": ("tee", "t shirt", "tshirt", "polo", "knit", "sweater", "hoodie", "top"),
    "dress": ("dress", "gown", "jumpsuit"),
    "ethnic": ("kurta", "saree", "sherwani", "lehenga"),
}


def _extract_pairing_anchor(query: str) -> Dict[str, str]:
    q = _norm(_clean_recursive_prompt(query))
    anchor = q
    trigger_matched = False
    for trigger in _PAIRING_TRIGGERS:
        if trigger in q:
            trigger_matched = True
            tail = q.split(trigger, 1)[1].strip()
            if tail:
                anchor = tail
                break
    anchor = re.sub(r"^(a|an|the|my|this|these|those)\s+", "", anchor).strip()
    anchor = re.sub(r"\b(casually|formally|well|better|today|outfit|look)\b", " ", anchor)
    anchor = re.sub(r"\s+", " ", anchor).strip()
    color = next((c for c in _ANCHOR_COLORS if re.search(rf"\b{re.escape(c)}\b", anchor)), "")
    if color == "gray":
        color = "grey"
    category = ""
    for cat, terms in _ANCHOR_CATEGORIES.items():
        if any(re.search(rf"\b{re.escape(term)}\b", anchor) for term in terms):
            category = cat
            break
    name = anchor or "the item"
    # Guard: if no pairing trigger was found AND the surviving anchor text
    # doesn't read as a garment (no known family, no anchor color, no anchor
    # category), the query is occasion / context phrasing — not a real anchor.
    # Skip rather than poison downstream archetype selection with junk like
    # "visual inspiration for a conference talk".
    if not trigger_matched and not category and not color and not _target_family(name):
        logger.info(
            "AHVI_STYLE_ANCHOR_SKIPPED reason=no_garment_family text=%r",
            name,
        )
        return {}
    logger.info(
        "AHVI_STYLE_PAIRING_ANCHOR name=%r category=%s color=%s",
        name,
        category,
        color,
    )
    return {"name": name, "category": category, "color": color}


def _extract_anchor_piece(query: str) -> str:
    q = _norm(_clean_recursive_prompt(query))
    if not q:
        return ""
    anchor = q
    patterns = (
        r"\bhow\s+do\s+i\s+pair\s+(?:my\s+|a\s+|an\s+|the\s+)?(.+)",
        r"\bwhat\s+to\s+pair\s+with\s+(?:my\s+|a\s+|an\s+|the\s+)?(.+)",
        r"\bwhat\s+goes\s+with\s+(?:my\s+|a\s+|an\s+|the\s+)?(.+)",
        r"\bhow\s+do\s+i\s+style\s+(?:my\s+|a\s+|an\s+|the\s+)?(.+)",
        r"\bhow\s+can\s+i\s+wear\s+(?:my\s+|a\s+|an\s+|the\s+)?(.+)",
        r"\bways\s+to\s+style\s+(?:my\s+|a\s+|an\s+|the\s+)?(.+)",
        r"\bstyle\s+(?:my\s+|a\s+|an\s+|the\s+)?(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            anchor = match.group(1).strip()
            break
    anchor = re.sub(r"\b(outfit|look|looks|ideas|idea|please|today|casually|formally|well|better)\b", " ", anchor)
    anchor = re.sub(r"^(my|a|an|the|this|these|those)\s+", "", anchor).strip()
    words = anchor.split()
    # Keep the anchor compact: color/adjective + item noun is usually enough.
    if len(words) > 5:
        family_index = -1
        for idx, word in enumerate(words):
            if _target_family(word):
                family_index = idx
        if family_index >= 0:
            start = max(0, family_index - 3)
            anchor = " ".join(words[start : family_index + 1])
    anchor = re.sub(r"\s+", " ", anchor).strip()
    # An anchor must actually name a garment. A generic request ("suggest a
    # complete outfit for me today") names none, so it must NOT yield a false
    # anchor like "suggest a complete for me".
    if anchor and not any(_target_family(word) for word in anchor.split()):
        return ""
    if anchor and anchor != q:
        logger.info("AHVI_ANCHOR_EXTRACTED query=%r anchor=%r", query, anchor)
    return anchor


def _hero_too_sentence_like(hero_piece: Any, query: str = "") -> bool:
    hero = _norm(hero_piece)
    q = _norm(query)
    if not hero:
        return True
    if q and (hero == q or q in hero or hero in q and len(hero.split()) > 5):
        return True
    if len(hero.split()) > 6:
        return True
    return any(
        phrase in f" {hero} "
        for phrase in (
            " how do i ",
            " what to ",
            " what should ",
            " pair with ",
            " goes with ",
            " outfit for ",
            " show visual ",
        )
    )


def _apply_anchor_piece_to_visual_directions(
    visual_directions: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    anchor = _extract_anchor_piece(query)
    if not anchor or not _target_family(anchor):
        return visual_directions
    out: List[Dict[str, Any]] = []
    for direction in visual_directions or []:
        item = dict(direction)
        hero = _asset_text(item.get("hero_piece") or item.get("heroPiece"))
        if _hero_too_sentence_like(hero, query):
            item["hero_piece"] = anchor
            pieces = _safe_list(item.get("items") or item.get("pieces"), limit=6)
            if pieces and _hero_too_sentence_like(pieces[0], query):
                pieces[0] = anchor
                item["items"] = pieces
                item["pieces"] = pieces
        out.append(item)
    return out


def _safe_list(value: Any, *, limit: int = 8) -> List[str]:
    if isinstance(value, list):
        out = [str(x or "").strip() for x in value if str(x or "").strip()]
    elif str(value or "").strip():
        out = [str(value).strip()]
    else:
        out = []
    return out[:limit]


def _asset_text(value: Any) -> str:
    return str(value or "").strip()


def _asset_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_asset_text(v).lower() for v in value if _asset_text(v)]
    text = _asset_text(value)
    if not text:
        return []
    try:
        import json

        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [_asset_text(v).lower() for v in parsed if _asset_text(v)]
    except Exception:
        pass
    return [part.strip().lower() for part in re.split(r"[,|]", text) if part.strip()]


# --- Non-fashion exclusion (P0 catalog hygiene) -------------------------------
# The style-board catalog carries non-garment rows (airpods, chewing gum, hair
# dryer, travel gadgets, wallets, skincare). They must NEVER surface on a board.
# Single authority used by every retrieval path (hero / support / accessory /
# complete_the_look). Category-based + name-signal based.
_NONFASHION_CATEGORIES = {"grooming", "travel"}
_NONFASHION_NAME_SIGNALS = (
    "airpod", "earbud", "chewing gum", "chewinggum", "mouth freshener",
    "mouthfreshener", "breath", "medical kit", "medicalkit", "hair dryer",
    "hairdryer", "hair straightener", "hairstraightener", "drink flask",
    "drinkflask", "flask", "document holder", "documentholder",
    "weighing scale", "weighingscale", "travel container", "travelcontainer",
    "travel file", "umbrella", "umberella", "wallet", "card case", "cardcase",
    "skincare", "skin care", "hand sanitizer", "sanitizer",
)


def _is_nonfashion_asset(asset: Dict[str, Any]) -> bool:
    """True when an asset is non-garment junk that must not reach a style board."""
    if not isinstance(asset, dict):
        return False
    if _asset_text(asset.get("category")).strip().lower() in _NONFASHION_CATEGORIES:
        return True
    blob = " ".join(
        [
            _asset_text(asset.get("name")),
            _asset_text(asset.get("subcategory")),
            _asset_text(asset.get("category")),
        ]
    ).lower()
    return any(sig in blob for sig in _NONFASHION_NAME_SIGNALS)


def _asset_category_terms(asset: Dict[str, Any]) -> set[str]:
    text = " ".join(
        [
            _asset_text(asset.get("name")),
            _asset_text(asset.get("category")),
            _asset_text(asset.get("subcategory")),
            " ".join(_asset_list(asset.get("tags"))),
        ]
    ).lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    terms: set[str] = set(tokens)
    if tokens.intersection({"top", "tops", "shirt", "shirts", "oxford", "polo", "tee", "tshirt", "knit", "knitwear", "sweater", "hoodie", "blouse"}):
        terms.add("top")
    if tokens.intersection({"outerwear", "jacket", "jackets", "blazer", "blazers", "overshirt", "coat", "hoodie"}):
        terms.add("outerwear")
    if tokens.intersection({"bottom", "bottoms", "trouser", "trousers", "pant", "pants", "jean", "jeans", "denim", "chino", "chinos", "skirt"}):
        terms.add("bottom")
    if tokens.intersection({"shoe", "shoes", "footwear", "loafer", "loafers", "sneaker", "sneakers", "heel", "heels", "sandal", "sandals", "boot", "boots"}):
        terms.add("footwear")
    if tokens.intersection({"accessory", "accessories", "belt", "watch", "bag", "tote", "sling", "hat", "cap", "sunglasses", "bracelet", "necklace", "earrings", "jewelry", "jewellery"}):
        terms.add("accessory")
    if tokens.intersection({"loungewear", "shorts"}):
        terms.add("blocked_hero")
    return terms


_STYLE_ASSET_LOAD_LIMIT = max(300, int(os.getenv("AHVI_STYLE_ASSET_LIMIT", "700")))


def _style_asset_rows(limit: int = 0) -> List[Dict[str, Any]]:
    try:
        from services.appwrite_proxy import AppwriteProxy

        # Catalog grew past 300 once women assets were imported (~500+). The
        # old hard cap of 300 dropped later pages (the ethnic/festive ones),
        # so wedding boards never saw sarees/lehengas. Tunable via env.
        target = max(1, min(int(limit or _STYLE_ASSET_LOAD_LIMIT), _STYLE_ASSET_LOAD_LIMIT))
        proxy = AppwriteProxy()
        rows: List[Dict[str, Any]] = []
        offset = 0
        pages = 0
        while len(rows) < target:
            page_limit = min(100, target - len(rows))
            page = proxy.list_documents(
                "style_assets",
                limit=page_limit,
                offset=offset,
                return_meta=True,
            )
            if isinstance(page, dict):
                page_rows = page.get("documents") or []
                meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
            else:
                page_rows = page if isinstance(page, list) else []
                meta = {}
            page_rows = [row for row in page_rows if isinstance(row, dict)]
            if not page_rows:
                break
            rows.extend(page_rows)
            pages += 1
            offset += len(page_rows)
            if meta and not meta.get("has_more"):
                break
            if not meta and len(page_rows) < page_limit:
                break

        cleaned: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            asset = _normalize_style_asset(row)
            key = _asset_text(
                asset.get("asset_id")
                or asset.get("$id")
                or asset.get("id")
                or asset.get("name")
            ).lower()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            cleaned.append(asset)
            if len(cleaned) >= target:
                break
        _validate_style_assets(cleaned)
        logger.info(
            "AHVI_STYLE_ASSETS_LOADED rows=%s pages=%s requested=%s",
            len(cleaned),
            pages,
            target,
        )
        return cleaned
    except Exception as exc:  # noqa: BLE001
        logger.info("AHVI_STYLE_ASSETS_UNAVAILABLE err=%s", str(exc)[:120])
        return []


_ASSET_MALE_GENDERS = {"male", "man", "men", "mens", "masculine", "m"}
_ASSET_FEMALE_GENDERS = {"female", "woman", "women", "womens", "feminine", "f"}
_ASSET_UNISEX_GENDERS = {"all", "any", "unisex", "neutral", "genderless"}
_FEMININE_ACCESSORY_TERMS = {
    "earring",
    "earrings",
    "necklace",
    "necklaces",
    "bracelet",
    "bracelets",
    "jewelry",
    "jewellery",
    "bangle",
    "bangles",
}
_MALE_BLOCKED_STYLE_TERMS = {
    "dress",
    "dresses",
    "skirt",
    "skirts",
    "earring",
    "earrings",
    "necklace",
    "necklaces",
    "bracelet",
    "bracelets",
    "blouse",
    "blouses",
    "heel",
    "heels",
    "camisole",
    "camisoles",
    "crop",
    "bralette",
    "bralettes",
    "corset",
    "corsets",
    "bandeau",
    "gown",
    "gowns",
    "lehenga",
    "saree",
    "sari",
}
_SAFE_ACCESSORY_TERMS = {
    "watch",
    "belt",
    "loafer",
    "loafers",
    "sneaker",
    "sneakers",
    "bag",
    "tote",
    "sling",
    "messenger",
    "pouch",
    "wallet",
    "bottle",
    "overshirt",
}
# Male-only garments to strip from a FEMALE persona (symmetric to the
# feminine block above). Kept tight to genuinely male-coded ethnic pieces —
# "kurta"/"jutti" are unisex (kurti/jutti for women) and stay allowed.
# Matched as substrings so multi-word terms ("nehru jacket") are caught.
_FEMALE_BLOCKED_STYLE_TERMS = (
    "sherwani",
    "nehru jacket",
    "bandhgala",
    "bandi vest",
    "achkan",
    "pathani",
    "kurta pajama",
    "kurta pyjama",
    "kurta pajamas",
    "dhoti",
    "lungi",
    "mojari",
    "safari suit",
)


def _asset_gender(value: Any) -> str:
    raw = _norm(value)
    if not raw:
        return "missing"
    if raw in _ASSET_MALE_GENDERS:
        return "male"
    if raw in _ASSET_FEMALE_GENDERS:
        return "female"
    if raw in _ASSET_UNISEX_GENDERS:
        return "unisex"
    return raw


def _normalize_style_asset(asset: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(asset)
    out["asset_id"] = _asset_text(
        out.get("asset_id") or out.get("assetId") or out.get("id") or out.get("$id")
    )
    out["board_image_url"] = _asset_text(
        out.get("board_image_url")
        or out.get("boardImageUrl")
        or out.get("board_url")
        or out.get("boardUrl")
    )
    out["transparent_image_url"] = _asset_text(
        out.get("transparent_image_url")
        or out.get("transparentImageUrl")
        or out.get("transparent_url")
        or out.get("transparentUrl")
    )
    out["cutout_url"] = _asset_text(
        out.get("cutout_url")
        or out.get("cutoutUrl")
    )
    out["rmbg_url"] = _asset_text(
        out.get("rmbg_url")
        or out.get("rmbgUrl")
    )
    out["catalog_image_url"] = _asset_text(
        out.get("catalog_image_url")
        or out.get("catalogImageUrl")
    )
    out["cutout_status"] = _asset_text(
        out.get("cutout_status")
        or out.get("cutoutStatus")
    ).lower()
    out["board_r2_key"] = _asset_text(
        out.get("board_r2_key")
        or out.get("boardR2Key")
    )
    out["image_url"] = _asset_text(
        out.get("image_url")
        or out.get("imageUrl")
        or out.get("url")
        or out.get("asset_url")
        or out.get("asset_path")
    )
    out["subcategory"] = _asset_text(
        out.get("subcategory") or out.get("sub_category") or out.get("subCategory")
    )
    out["allowed_slots"] = out.get("allowed_slots") or out.get("allowedSlots") or out.get("slots") or []
    out["avoid_for"] = out.get("avoid_for") or out.get("avoidFor") or []
    out["style_tags"] = out.get("style_tags") or out.get("styleTags") or []
    gender = _asset_gender(out.get("gender"))
    if gender in {"male", "female", "unisex"}:
        out["gender"] = gender
    if not _asset_text(out.get("status")):
        out["status"] = "active"
    for key in ("colors", "archetypes", "occasions", "tags", "style_tags", "allowed_slots", "avoid_for"):
        if _asset_text(out.get(key)) and not isinstance(out.get(key), list):
            out[key] = _asset_list(out.get(key))
    return out


def _prompt_gender_override(query: Any) -> str:
    q = f" {_norm(query)} "
    female_markers = (
        " women ", " womens ", " woman ", " female ", " feminine ",
        " ladies ", " girl ", " girls ",
    )
    male_markers = (
        " men ", " mens ", " man ", " male ", " masculine ",
        " menswear ", " menswear inspired ",
    )
    neutral_markers = (
        " androgynous ", " gender neutral ", " genderless ", " unisex ",
    )
    if any(marker in q for marker in neutral_markers):
        return "unisex"
    if any(marker in q for marker in female_markers):
        return "female"
    if any(marker in q for marker in male_markers):
        return "male"
    return ""


def _prompt_allows_feminine_accessory(query: Any) -> bool:
    q = f" {_norm(query)} "
    return any(f" {term} " in q for term in _FEMININE_ACCESSORY_TERMS)


def _prompt_allows_gendered_feminine_style(query: Any) -> bool:
    return _prompt_gender_override(query) == "female" or _prompt_allows_feminine_accessory(query)


def _contains_male_blocked_style_term(value: Any) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", _norm(value)))
    return bool(tokens.intersection(_MALE_BLOCKED_STYLE_TERMS))


def _contains_female_blocked_style_term(value: Any) -> bool:
    blob = _norm(value)
    return any(term in blob for term in _FEMALE_BLOCKED_STYLE_TERMS)


def _style_text_allowed_for_gender(value: Any, target_gender: str, *, allow_feminine: bool = False) -> bool:
    if not _asset_text(value):
        return False
    if target_gender in {"male", "unknown", "unisex"} and not allow_feminine:
        if _contains_male_blocked_style_term(value):
            return False
    # Symmetric: strip male-only garments (sherwani, nehru jacket, ...) from a
    # female persona so gender-neutral archetypes carrying male-coded ethnic
    # items don't leak into women's directions.
    if target_gender == "female":
        if _contains_female_blocked_style_term(value):
            return False
    return True


def _filter_style_terms_for_gender(
    values: List[Any],
    *,
    target_gender: str,
    allow_feminine: bool = False,
    limit: int = 6,
) -> List[str]:
    out: List[str] = []
    for value in values or []:
        text = _asset_text(value)
        if _style_text_allowed_for_gender(text, target_gender, allow_feminine=allow_feminine):
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _safe_component_fallback(target_gender: str) -> List[str]:
    if target_gender == "female":
        return ["clean top", "tailored bottom", "polished footwear"]
    return ["clean shirt", "tailored trouser", "polished footwear"]


_GARMENT_CATEGORY_TERMS: Dict[str, set[str]] = {
    "top": {
        "shirt", "shirts", "tee", "tshirt", "t-shirt", "polo", "knit", "sweater",
        "jumper", "blouse", "top", "button", "button-down", "buttondown", "oxford",
        "linen", "crewneck", "crew-neck",
    },
    "bottom": {
        "trouser", "trousers", "pant", "pants", "jean", "jeans", "denim", "chino",
        "chinos", "skirt", "shorts",
    },
    "footwear": {
        "shoe", "shoes", "loafer", "loafers", "sneaker", "sneakers", "boot", "boots",
        "heel", "heels", "sandal", "sandals", "formal shoes",
    },
    "outerwear": {
        "blazer", "jacket", "overshirt", "coat", "cardigan", "shacket", "layer",
        "outerwear",
    },
    "accessory": {
        "watch", "belt", "bag", "tote", "sling", "wallet", "bracelet", "necklace",
        "earring", "earrings", "ring", "scarf", "cap", "sunglasses",
    },
    "dress": {"dress", "dresses", "gown", "gowns", "saree", "sari", "lehenga"},
}

_CATEGORY_CANONICAL = {
    "top": "top",
    "shirt": "top",
    "bottom": "bottom",
    "pant": "bottom",
    "pants": "bottom",
    "trouser": "bottom",
    "trousers": "bottom",
    "footwear": "footwear",
    "shoe": "footwear",
    "shoes": "footwear",
    "outerwear": "outerwear",
    "layer": "outerwear",
    "accessory": "accessory",
    "jewelry": "accessory",
    "jewellery": "accessory",
    "dress": "dress",
}

_COFFEE_DATE_MISSING_FALLBACKS = [
    "dark wash straight-leg jeans",
    "dark brown penny loafers",
    "olive cotton overshirt",
    "cream knit polo",
]

_GENERAL_MISSING_FALLBACKS = [
    "olive cotton overshirt",
    "brushed steel watch",
    "dark brown penny loafers",
    "dark brown leather belt",
]

_GENERIC_MISSING_WORDS = {"clean", "simple", "basic", "neutral", "minimal"}

_SPECIFIC_MISSING_NAMES: Dict[str, List[str]] = {
    "clean overshirt": ["Olive Cotton Overshirt", "Navy Twill Overshirt", "Charcoal Utility Overshirt"],
    "soft overshirt": ["Olive Cotton Overshirt", "Navy Twill Overshirt", "Charcoal Utility Overshirt"],
    "neutral blazer": ["Soft Camel Relaxed Blazer", "Charcoal Double-Breasted Blazer"],
    "structured blazer": ["Soft Camel Relaxed Blazer", "Charcoal Double-Breasted Blazer"],
    "neutral loafers": ["Dark Brown Penny Loafers", "Black Leather Loafers"],
    "simple watch": ["Brushed Steel Watch", "Leather-Strap Watch"],
    "minimal watch": ["Brushed Steel Watch", "Leather-Strap Watch"],
    "structured belt": ["Dark Brown Leather Belt", "Cognac Leather Belt"],
    "clean knit polo": ["Cream Knit Polo", "Navy Knit Polo"],
    "soft knit sweater": ["Dark Wash Straight-Leg Jeans", "Olive Cotton Overshirt"],
    "dark wash jeans": ["Dark Wash Straight-Leg Jeans"],
    "dark wash straight-leg jeans": ["Dark Wash Straight-Leg Jeans"],
}


def _style_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _norm(value)))


def _style_category(value: Any) -> str:
    text = _norm(value)
    tokens = _style_tokens(text)
    for raw, canonical in _CATEGORY_CANONICAL.items():
        if raw in tokens:
            return canonical
    for category, terms in _GARMENT_CATEGORY_TERMS.items():
        if tokens.intersection({t.replace("-", "") for t in terms} | terms):
            return category
    return ""


def _hero_expected_slot(hero_text: str) -> str | None:
    text = _norm(hero_text)
    tokens = _style_tokens(text)
    if not text:
        return None
    if tokens.intersection({"pants", "pant", "trouser", "trousers", "chino", "chinos", "jeans", "jean", "shorts", "short"}):
        return "bottom"
    if tokens.intersection({
        "loafer", "loafers", "sneaker", "sneakers", "boot", "boots",
        "shoe", "shoes", "footwear",
        "heel", "heels", "sandal", "sandals", "pump", "pumps",
        "flat", "flats", "mule", "mules", "wedge", "wedges",
    }) or (
        "oxford" in tokens and tokens.intersection({"shoe", "shoes"})
    ):
        return "footwear"
    if "blazer" in tokens:
        return "blazer"
    if tokens.intersection({"jacket", "overshirt", "coat"}):
        return "outerwear"
    if tokens.intersection({"shirt", "oxford", "button", "down", "buttondown", "polo", "tee", "tshirt", "t", "knit", "sweater", "hoodie", "sweatshirt"}):
        return "top"
    return None


_SHIRT_COLORS: tuple[str, ...] = (
    "white",
    "offwhite",
    "ivory",
    "cream",
    "beige",
    "tan",
    "khaki",
    "olive",
    "green",
    "blue",
    "navy",
    "lightblue",
    "skyblue",
    "black",
    "charcoal",
    "grey",
    "gray",
    "brown",
    "lightbrown",
    "darkbrown",
    "burgundy",
    "maroon",
    "red",
    "yellow",
    "pink",
    "purple",
)
_COMPATIBLE_COLOR_GROUPS: tuple[set[str], ...] = (
    {"blue", "navy", "lightblue", "skyblue"},
    {"white", "offwhite", "ivory", "cream"},
    {"beige", "tan", "khaki"},
    {"grey", "gray", "charcoal"},
    {"brown", "lightbrown", "darkbrown"},
    {"burgundy", "maroon", "red"},
)


def _compact_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(value))


def _compact_tokens(value: Any) -> list[str]:
    """Return per-word alphanum-only tokens.

    Splits on whitespace ONLY so hyphenated/embedded forms like ``T-Shirt``
    collapse into a single token ``tshirt``. This avoids cross-word
    substring collisions (e.g. ``shirt``+``shirt`` -> ``shirtshirt``
    falsely matching ``tshirt``)."""
    text = str(value or "").lower()
    if not text:
        return []
    out: list[str] = []
    for word in text.split():
        compact = re.sub(r"[^a-z0-9]+", "", word)
        if compact:
            out.append(compact)
    return out


def _full_compact(value: Any) -> str:
    """All-alphanum lowercase squash. Useful for low-collision compound
    markers like ``buttondown``/``dressshirt`` where cross-word adjacency
    rarely produces false positives."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


# Needles that must match a token exactly. Short common substrings like "tee"
# otherwise false-positive on unrelated words ("steel" contains "tee", "watch"
# contains "tch", etc.) and pollute family classification.
_TOKEN_EXACT_ONLY_MARKERS: set[str] = {"tee", "tie", "cap", "ring"}


def _tokens_contain(tokens: list[str], needles: tuple[str, ...]) -> bool:
    for tok in tokens:
        for needle in needles:
            if not needle:
                continue
            if needle in _TOKEN_EXACT_ONLY_MARKERS:
                if tok == needle:
                    return True
            elif needle in tok:
                return True
    return False


def _hero_shirt_intent(hero_text: str) -> str | None:
    """Classify the hero piece intent for top garments.

    Returns one of ``formal_shirt``, ``tshirt``, ``polo``, ``knit`` or
    ``None`` when no shirt-family intent is detected.
    """
    tokens = _compact_tokens(hero_text)
    if not tokens:
        return None
    full = _full_compact(hero_text)
    has_tshirt = _tokens_contain(tokens, ("tshirt", "tee"))
    has_polo = _tokens_contain(tokens, ("polo",))
    has_formal_marker = _tokens_contain(
        tokens, ("oxford", "buttondown", "dressshirt", "formalshirt", "oxfordshirt")
    ) or any(marker in full for marker in ("buttondown", "dressshirt", "formalshirt", "oxfordshirt"))
    has_knit = _tokens_contain(tokens, ("knit", "knitwear", "sweater", "cardigan"))
    if has_formal_marker:
        return "formal_shirt"
    if has_polo and not has_formal_marker:
        return "polo"
    if has_tshirt and not has_formal_marker:
        return "tshirt"
    if has_knit:
        return "knit"
    if _tokens_contain(tokens, ("shirt",)) and not has_tshirt and not has_polo:
        return "formal_shirt"
    return None


def _asset_shirt_intent(asset_blob: str) -> str | None:
    tokens = _compact_tokens(asset_blob)
    if not tokens:
        return None
    if _tokens_contain(tokens, ("tshirt", "tee")):
        return "tshirt"
    if _tokens_contain(tokens, ("polo",)):
        return "polo"
    if _tokens_contain(tokens, ("hoodie", "sweatshirt")):
        return "casual_pullover"
    if _tokens_contain(tokens, ("oxford", "buttondown", "dressshirt", "formalshirt")):
        return "formal_shirt"
    if _tokens_contain(tokens, ("knit", "knitwear", "sweater", "cardigan")):
        return "knit"
    if _tokens_contain(tokens, ("shirt",)):
        return "formal_shirt"
    return None


def _extract_simple_colors(text: Any) -> set[str]:
    compact = _compact_text(text)
    if not compact:
        return set()
    found: set[str] = set()
    for color in _SHIRT_COLORS:
        if color in compact:
            found.add(color)
    # Drop pure substrings overshadowed by a longer color match (e.g. "lightblue" wins over "blue").
    overshadowed: set[str] = set()
    for color in found:
        for other in found:
            if color != other and color in other and len(color) < len(other):
                overshadowed.add(color)
                break
    return found - overshadowed


def _colors_share_group(a: str, b: str) -> bool:
    if a == b:
        return True
    for group in _COMPATIBLE_COLOR_GROUPS:
        if a in group and b in group:
            return True
    return False


# ---------------------------------------------------------------------
# Style asset policy
# ---------------------------------------------------------------------
# Central decision layer for what assets are allowed in which placement
# (hero / complete_the_look / missing_piece) for a given occasion + target
# text. Every selection path below funnels through ``_asset_allowed_for_context``
# and ``_asset_context_score`` so we stop accumulating ad-hoc patches for
# beanie/cap/tshirt/hoodie/sandal mismatches.
# ---------------------------------------------------------------------

_OFFICE_POLICY_OCCASIONS: set[str] = {
    "office",
    "startup_office",
    "client_meeting",
    "client meeting",
    "work",
    "presentation",
    "business",
    "interview",
    "conference",
    "formal",
    "funeral",
    "religious",
    "wedding",
}
_COFFEE_DATE_POLICY_OCCASIONS: set[str] = {
    "coffee_date",
    "coffee date",
    "first_date",
    "casual_date",
}

# Broader casual/social set that should read relaxed, not office-casual.
# Reuses the same demote/boost rules as coffee date. Source of truth for
# "approachable, relaxed, confident" occasions on the visual-board path.
_CASUAL_SOCIAL_OCCASIONS: set[str] = {
    "coffee_date",
    "coffee date",
    "coffee",
    "cafe_date",
    "cafe date",
    "first_date",
    "first date",
    "date",
    "date_night",
    "date night",
    "brunch",
    "brunch_date",
    "brunch date",
    "casual_date",
    "casual_day",
    "social_occasion",
    "social occasion",
    "weekend",
}

# Family token rules: longer / compound tokens first so e.g. "buttondown" is
# matched before "shirt".
_FAMILY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("buttondown", "dressshirt", "formalshirt", "oxfordshirt"), "shirt"),
    (("oxford",), "shirt"),
    (("tshirt", "tee"), "tshirt"),
    (("polo",), "polo"),
    (("hoodie",), "hoodie"),
    (("sweatshirt",), "sweatshirt"),
    (("knitwear", "knit", "sweater", "cardigan"), "knit"),
    (("brooch", "pocketsquare", "stole", "dupatta"), "festive_accessory"),
    (("blazer",), "blazer"),
    (("overshirt",), "overshirt"),
    (("jacket",), "jacket"),
    (("coat",), "coat"),
    (("kurta", "sherwani", "bandhgala", "waistcoat", "nehrujacket"), "ethnic"),
    (("jeans", "denim"), "jeans"),
    (("chino",), "chino"),
    (("trouser", "pant"), "trouser"),
    (("cargopants",), "cargo_pants"),
    (("joggers",), "joggers"),
    (("swimshorts", "swimtrunk"), "swim_shorts"),
    (("gymshorts", "runningshorts"), "gym_shorts"),
    (("shorts",), "shorts"),
    (("formalshoe", "dressshoe"), "formal_shoe"),
    (("mojari", "jutti"), "formal_shoe"),
    (("loafer",), "loafer"),
    (("espadrille", "espadrilles"), "espadrille"),
    (("sneaker",), "sneaker"),
    (("flipflop", "flipflops"), "flip_flops"),
    (("slider", "slides"), "slide"),
    (("sandal",), "sandal"),
    (("boot",), "boot"),
    (("laptopbag",), "laptop_bag"),
    (("messenger",), "messenger_bag"),
    (("backpack",), "backpack"),
    (("duffle", "dufflebag"), "duffle_bag"),
    (("wallet", "cardcase", "cardholder"), "cardholder"),
    (("briefcase",), "briefcase"),
    (("bag", "tote", "crossbody", "sling"), "bag"),
    (("belt",), "belt"),
    (("watch",), "watch"),
    (("sunglass", "aviator"), "sunglasses"),
    (("beanie",), "beanie"),
    (("baseballcap", "trucker"), "cap"),
    (("cap",), "cap"),
    (("fedora", "sunhat"), "hat"),
    (("hat",), "hat"),
    (("scarf", "muffler"), "scarf"),
    (("tie",), "tie"),
    (("necklace", "bracelet", "earring", "ring", "chain", "jewellery", "jewelry"), "jewellery"),
    (
        (
            "powerbank",
            "neckpillow",
            "pillow",
            "suitcase",
            "luggage",
            "pouch",
            "bottle",
            "tumbler",
            "keychain",
            "eyemask",
            "adapter",
            "headphone",
            "earbud",
            "earbuds",
            "earphone",
            "earphones",
            "charger",
            "phone",
            "electronics",
        ),
        "travel",
    ),
    (
        (
            "skincare",
            "moisturizer",
            "serum",
            "sunscreen",
            "toiletry",
            "comb",
            "razor",
            "shaver",
            "straightener",
            "hairdryer",
            "dryer",
        ),
        "grooming",
    ),
    (("loungewear", "lounge", "pyjama", "pajama"), "loungewear"),
    # Generic fallback so e.g. "blackshirt" / "whiteshirt" resolve to shirt
    # once the more-specific tshirt/polo/oxford rules miss.
    (("shirt",), "shirt"),
)

_FAMILY_GROUP: dict[str, str] = {
    "shirt": "top",
    "tshirt": "top",
    "polo": "top",
    "hoodie": "top",
    "sweatshirt": "top",
    "knit": "top",
    "ethnic": "top",
    "overshirt": "outerwear",
    "blazer": "outerwear",
    "jacket": "outerwear",
    "coat": "outerwear",
    "jeans": "bottom",
    "chino": "bottom",
    "trouser": "bottom",
    "cargo_pants": "bottom",
    "joggers": "bottom",
    "shorts": "bottom",
    "swim_shorts": "bottom",
    "gym_shorts": "bottom",
    "formal_shoe": "footwear",
    "loafer": "footwear",
    "espadrille": "footwear",
    "sneaker": "footwear",
    "sandal": "footwear",
    "slide": "footwear",
    "flip_flops": "footwear",
    "boot": "footwear",
    "belt": "accessory",
    "watch": "accessory",
    "bag": "accessory",
    "laptop_bag": "accessory",
    "messenger_bag": "accessory",
    "backpack": "accessory",
    "duffle_bag": "accessory",
    "cardholder": "accessory",
    "briefcase": "accessory",
    "sunglasses": "accessory",
    "cap": "accessory",
    "hat": "accessory",
    "beanie": "accessory",
    "scarf": "accessory",
    "tie": "accessory",
    "jewellery": "accessory",
    "festive_accessory": "accessory",
    "travel": "travel",
    "grooming": "grooming",
    "electronics": "electronics",
    "loungewear": "loungewear",
}

# Hero target_family -> set of asset families that are acceptable.
_FAMILY_ALLOWED_FOR_TARGET: dict[str, set[str]] = {
    "shirt": {"shirt"},
    "tshirt": {"tshirt"},
    "polo": {"polo"},
    "hoodie": {"hoodie", "sweatshirt"},
    "sweatshirt": {"sweatshirt", "hoodie"},
    "knit": {"knit"},
    "blazer": {"blazer"},
    "jacket": {"jacket", "overshirt"},
    "overshirt": {"overshirt", "jacket"},
    "coat": {"coat", "jacket"},
    "jeans": {"jeans"},
    "trouser": {"trouser", "chino"},
    "chino": {"chino", "trouser"},
    "shorts": {"shorts"},
    "formal_shoe": {"formal_shoe"},
    "loafer": {"loafer"},
    "espadrille": {"espadrille"},
    "sneaker": {"sneaker"},
    "belt": {"belt"},
    "watch": {"watch"},
    "bag": {"bag", "laptop_bag", "messenger_bag", "backpack", "duffle_bag", "briefcase"},
    "laptop_bag": {"laptop_bag", "messenger_bag", "briefcase", "bag"},
    "sunglasses": {"sunglasses"},
}

_OFFICE_CTL_REJECT_FAMILIES: set[str] = {
    "beanie",
    "cap",
    "hat",
    "sunglasses",
    "sneaker",
    "slide",
    "flip_flops",
    "sandal",
    "grooming",
    "travel",
    "electronics",
    "swim_shorts",
    "gym_shorts",
    "loungewear",
}
_OFFICE_CTL_PREFER_FAMILIES: set[str] = {
    "belt",
    "watch",
    "laptop_bag",
    "messenger_bag",
    "cardholder",
    "briefcase",
    "formal_shoe",
    "loafer",
    "bag",
}
_HERO_FORBIDDEN_GROUPS: set[str] = {"accessory", "travel", "grooming", "electronics", "loungewear"}
_COFFEE_DATE_ALLOWED_ACC: set[str] = {"belt", "watch", "loafer", "sneaker", "sunglasses"}

# Casual/social demotions (coffee date, first date, brunch, date night).
# Business/formal pieces read as "office casual" and kill the relaxed mood;
# beanie/cap are too street for a date. Demoted unless the user explicitly
# asked for that piece (checked against the hero/target text).
_CASUAL_SOCIAL_DEMOTE_FAMILIES: set[str] = {
    "blazer",
    "coat",
    "tie",
    "beanie",
    "cap",
}
# Relaxed-but-considered pieces that should rise for casual/social looks.
_CASUAL_SOCIAL_BOOST_FAMILIES: set[str] = {
    "polo",
    "overshirt",
    "knit",
    "chino",
    "jeans",
    "loafer",
    "sneaker",
    "watch",
}


def _policy_occasion_key(occasion: str) -> str:
    return re.sub(r"\s+", "_", str(occasion or "").strip().lower())


def _is_office_occasion_policy(occasion: str) -> bool:
    key = _policy_occasion_key(occasion)
    return key in {_policy_occasion_key(o) for o in _OFFICE_POLICY_OCCASIONS}


def _is_coffee_date_policy(occasion: str) -> bool:
    key = _policy_occasion_key(occasion)
    return key in {_policy_occasion_key(o) for o in _COFFEE_DATE_POLICY_OCCASIONS}


def _is_casual_social_policy(occasion: str) -> bool:
    key = _policy_occasion_key(occasion)
    return key in {_policy_occasion_key(o) for o in _CASUAL_SOCIAL_OCCASIONS}


_WEDDING_OCCASION_TERMS = (
    "wedding", "reception", "sangeet", "haldi", "mehendi", "mehndi",
    "engagement", "baraat", "shaadi", "nikah", "festive", "varmala", "roka",
)


def _is_wedding_occasion_policy(occasion: str) -> bool:
    text = str(occasion or "").lower().replace("_", " ")
    return any(term in text for term in _WEDDING_OCCASION_TERMS)


def _visual_occasion_family(occasion: Any, target_text: Any = "") -> str:
    text = f"{occasion or ''} {target_text or ''}".lower().replace("_", " ")
    if _is_music_event_context(text):
        return "music_event"
    if any(
        term in text
        for term in (
            "christian wedding",
            "christian marriage",
            "church wedding",
            "church marriage",
            "cathedral wedding",
            "western formal wedding",
        )
    ):
        return "christian_wedding"
    if any(term in text for term in ("funeral", "memorial", "condolence", "wake service")):
        return "funeral"
    if any(
        term in text
        for term in (
            "conference",
            "presentation",
            "keynote",
            "client meeting",
            "business meeting",
            "networking",
            "panel talk",
        )
    ):
        return "conference"
    if any(term in text for term in ("airport", "air travel", "flight", "travel day", "in transit")):
        return "travel"
    if any(
        term in text
        for term in (
            "haldi",
            "mehendi",
            "mehndi",
            "sangeet",
            "shaadi",
            "nikah",
            "wedding",
            "marriage",
            "reception",
            "engagement",
            "festival",
            "festive",
            "traditional",
            "puja",
            "pooja",
            "ethnic event",
            "kurta",
            "sherwani",
            "bandhgala",
            "ethnic",
        )
    ):
        return "indian_festive"
    return ""


_MUSIC_EVENT_CONTEXT_TERMS: tuple[str, ...] = (
    "music festival",
    "concert",
    "rave",
    "live event",
    "live music",
    "gig",
    "festival outfit",
    "festival look",
)

_MUSIC_EVENT_FORBIDDEN_TERMS: tuple[str, ...] = (
    "wedding",
    "sangeet",
    "festive heritage",
    "refined traditional",
    "wedding day ease",
    "celebration kurta",
    "bandhgala",
    "nehru",
    "kurta",
    "churidar",
    "mojari",
    "jutti",
    "oxford shirt",
    "business shirt",
    "formal loafer",
    "leather belt",
    "cardholder",
    "grooming",
    "electronics",
)

_MUSIC_EVENT_DIRECTION_NAMES: tuple[str, ...] = (
    "Festival Street",
    "Creative Casual",
    "Indie Festival",
    "Summer Concert",
    "Boho Casual",
    "Street-Cool Festival",
    "Outdoor Concert",
)

_MUSIC_EVENT_FALLBACK_PIECES: tuple[tuple[str, ...], ...] = (
    ("Graphic T-Shirt", "Cargo Pants", "Lightweight Overshirt", "Comfortable Sneakers"),
    ("Printed Shirt", "Relaxed Trousers", "Clean Sneakers", "Crossbody Bag"),
    ("Linen Shirt", "Utility Pants", "Canvas Sneakers", "Canvas Tote"),
    ("Breezy Shirt", "Relaxed Shorts", "Comfortable Sneakers", "Lightweight Bag"),
    ("Relaxed Shirt", "Straight Denim", "Canvas Sneakers", "Festival Bracelet"),
    ("Graphic Tee", "Cargo Pants", "Lightweight Jacket", "Clean Sneakers"),
    ("Camp-Collar Shirt", "Chinos", "Comfortable Sneakers", "Compact Crossbody"),
)


def _is_music_event_context(text: Any) -> bool:
    normalized = _norm(str(text or "").replace("_", " "))
    if not normalized:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    for term in _MUSIC_EVENT_CONTEXT_TERMS:
        if " " in term:
            if term in normalized:
                return True
        elif term in tokens:
            return True
    return "festival" in tokens and any(
        term in tokens for term in ("music", "concert", "rave", "band", "live")
    )


def _music_event_text_blocked(text: Any) -> bool:
    normalized = _norm(str(text or "").replace("_", " "))
    if not normalized:
        return False
    return any(term in normalized for term in _MUSIC_EVENT_FORBIDDEN_TERMS)


def _music_event_safe_direction_name(index: int) -> str:
    return _MUSIC_EVENT_DIRECTION_NAMES[index % len(_MUSIC_EVENT_DIRECTION_NAMES)]


def _music_event_safe_pieces(index: int) -> List[str]:
    return list(_MUSIC_EVENT_FALLBACK_PIECES[index % len(_MUSIC_EVENT_FALLBACK_PIECES)])


def _music_event_visible_guard(
    directions: List[Dict[str, Any]],
    *,
    occasion: Any = "",
    query: Any = "",
) -> List[Dict[str, Any]]:
    """Final visible-output guard for music festival / concert boards.

    Upstream can mix LLM titles, selected archetypes, fallback directions, and
    enriched assets. This runs after editorial polish so wedding/ethnic/formal
    labels cannot escape into the UI for concert/festival prompts.
    """
    context = f"{occasion or ''} {query or ''}"
    if not _is_music_event_context(context):
        return directions
    repaired: List[Dict[str, Any]] = []
    blocked_seen: List[str] = []
    for index, direction in enumerate(directions or []):
        if not isinstance(direction, dict):
            continue
        out = dict(direction)
        safe_name = _music_event_safe_direction_name(index)
        for key in ("title", "archetype", "direction_name"):
            current = _asset_text(out.get(key))
            if _music_event_text_blocked(current) or current not in _MUSIC_EVENT_DIRECTION_NAMES:
                if current:
                    blocked_seen.append(current)
                out[key] = safe_name

        pieces = _safe_list(out.get("items") or out.get("pieces"), limit=8)
        kept_pieces = [piece for piece in pieces if not _music_event_text_blocked(piece)]
        if len(kept_pieces) != len(pieces):
            blocked_seen.extend(str(piece) for piece in pieces if _music_event_text_blocked(piece))
        if not kept_pieces:
            kept_pieces = _music_event_safe_pieces(index)
        out["items"] = kept_pieces
        out["pieces"] = kept_pieces

        hero = _asset_text(out.get("hero_piece"))
        if not hero or _music_event_text_blocked(hero):
            if hero:
                blocked_seen.append(hero)
            out["hero_piece"] = kept_pieces[0]

        safe_note = (
            f"{safe_name} keeps the look expressive, breathable, and practical for a live music crowd."
        )
        for key in (
            "subtitle",
            "description",
            "why_it_works",
            "why_this_works",
            "short_note",
            "style_note",
            "styling_tip",
            "complete_the_look_copy",
        ):
            if key in out and _music_event_text_blocked(out.get(key)):
                blocked_seen.append(_asset_text(out.get(key))[:80])
                out[key] = safe_note

        complete = out.get("complete_the_look")
        if isinstance(complete, list):
            safe_support: List[Any] = []
            for item in complete:
                item_blob = str(item)
                name = _asset_text(item.get("name") if isinstance(item, dict) else item)
                if _music_event_text_blocked(name) or _music_event_text_blocked(item_blob):
                    blocked_seen.append(name or item_blob[:80])
                    continue
                safe_support.append(item)
            out["complete_the_look"] = safe_support[:3]

        mp = out.get("missing_piece")
        if isinstance(mp, dict) and _music_event_text_blocked(mp.get("name")):
            blocked_seen.append(_asset_text(mp.get("name")))
            out["missing_piece"] = {
                "name": "Comfortable Festival Sneakers",
                "category": "Footwear",
                "reason": "Keeps the look practical for standing, walking, and dancing.",
            }

        repaired.append(out)

    if not repaired:
        safe_name = _music_event_safe_direction_name(0)
        pieces = _music_event_safe_pieces(0)
        repaired = [
            {
                "title": safe_name,
                "archetype": safe_name,
                "direction_name": safe_name,
                "hero_piece": pieces[0],
                "items": pieces,
                "pieces": pieces,
                "palette": ["black", "stone", "olive"],
                "description": "Creative casual pieces built for movement, heat, and a live-event crowd.",
                "why_it_works": "It keeps the outfit expressive without drifting into wedding or office formality.",
                "complete_the_look": [],
            }
        ]

    if blocked_seen:
        logger.info(
            "AHVI_MUSIC_EVENT_GUARD_APPLIED blocked=%s kept=%s context=%s",
            sorted(set(blocked_seen)),
            [
                _asset_text(d.get("direction_name") or d.get("title"))
                for d in repaired
                if isinstance(d, dict)
            ],
            str(context or "")[:160],
        )
    return repaired


def _asset_policy_blob(asset: Dict[str, Any]) -> str:
    return " ".join(
        [
            _asset_text(asset.get("name")),
            _asset_text(asset.get("category")),
            _asset_text(asset.get("subcategory")),
            " ".join(_asset_list(asset.get("tags"))),
            " ".join(_asset_list(asset.get("style_tags"))),
            " ".join(_asset_list(asset.get("archetypes"))),
            " ".join(_asset_list(asset.get("occasions"))),
            " ".join(_asset_list(asset.get("colors"))),
        ]
    ).lower().replace("_", " ")


def _occasion_asset_block_reason(
    asset: Dict[str, Any],
    *,
    occasion: str = "",
    placement: str = "hero",
    target_text: str = "",
) -> str:
    family = _asset_family(asset)
    blob = _asset_policy_blob(asset)
    context = f"{occasion or ''} {target_text or ''}".lower().replace("_", " ")
    occasion_family = _visual_occasion_family(occasion, target_text)
    outdoor = any(term in context for term in ("beach", "outdoor", "daytime", "sunny", "park"))
    cold = any(term in context for term in ("cold", "winter", "snow", "freezing", "chilly"))
    business_travel = any(
        term in context for term in ("business travel", "work trip", "client", "conference")
    )
    comfort_requested = any(
        term in context for term in ("comfort", "comfortable", "mobility", "orthopedic")
    )

    if occasion_family == "indian_festive":
        if family in {
            "beanie",
            "cap",
            "hat",
            "hoodie",
            "sweatshirt",
            "knit",
            "overshirt",
            "jacket",
            "blazer",
            "tshirt",
            "jeans",
            "gym_shorts",
            "shorts",
            "sneaker",
            "flip_flops",
            "slide",
            "boot",
            "polo",
            "loafer",
            "backpack",
            "laptop_bag",
            "messenger_bag",
            "briefcase",
            "cardholder",
            "belt",
        }:
            return family
        if family == "sunglasses" and not outdoor:
            return "sunglasses_without_outdoor_context"
        if family == "shirt":
            return "western_or_relaxed_shirt"
        if family == "formal_shoe" and not any(
            term in blob for term in ("mojari", "jutti", "ethnic", "kolhapuri")
        ):
            return "western_formal_shoe"
    if occasion_family == "music_event":
        if any(term in blob for term in _MUSIC_EVENT_FORBIDDEN_TERMS):
            return "music_event_forbidden"
        if placement == "hero" and family in {"shirt", "formal_shoe", "loafer", "belt", "ethnic"}:
            return "music_event_formal_or_ethnic"

    if occasion_family == "christian_wedding":
        if family in {
            "beanie",
            "cap",
            "hat",
            "hoodie",
            "sweatshirt",
            "gym_shorts",
            "shorts",
            "sneaker",
        }:
            return family
        if family == "sunglasses" and not outdoor:
            return "sunglasses_without_outdoor_context"

    if occasion_family == "funeral":
        if family in {"beanie", "cap", "hat", "hoodie", "sweatshirt", "gym_shorts", "shorts"}:
            return family
        if family == "sunglasses":
            return "sunglasses"
        if family == "sneaker" and not comfort_requested:
            return "sneaker_without_comfort_context"
        if any(
            term in blob
            for term in (
                "bright",
                "festive",
                "party",
                "neon",
                "yellow",
                "orange",
                "pink",
                "gold",
                "sequin",
            )
        ):
            return "bright_or_festive"

    if occasion_family == "travel":
        if family == "beanie" and not cold:
            return "beanie_without_cold_context"
        if family == "blazer" and not business_travel:
            return "blazer_without_business_context"

    if occasion_family == "conference":
        if family in {
            "beanie",
            "cap",
            "hat",
            "hoodie",
            "sweatshirt",
            "gym_shorts",
            "shorts",
            "sneaker",
            "sunglasses",
        }:
            return family
    return ""


_ETHNIC_SUBCATS = {
    "kurta", "kurti", "kurta_set", "kurtaset", "sherwani", "bandhgala", "bandhgalaset",
    "nehrujacket", "nehru_jacket", "waistcoat", "indowestern", "indo_western",
    "achkan", "ethnic",
}


def _is_ethnic_asset(asset: Dict[str, Any]) -> bool:
    if not isinstance(asset, dict):
        return False
    cat = _asset_text(asset.get("category")).lower()
    sub = _asset_text(asset.get("subcategory")).lower().replace(" ", "_")
    if cat == "ethnic":
        return True
    if sub in _ETHNIC_SUBCATS:
        return True
    blob = (
        _asset_text(asset.get("name")).lower()
        + " "
        + " ".join(_asset_list(asset.get("tags"))).lower()
    )
    return any(t in blob for t in ("kurta", "sherwani", "bandhgala", "festive", "nehru"))


def _ethnic_garment_subtype(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    if "nehru" in text:
        return "nehru_jacket"
    if "bandhgala" in text:
        return "bandhgala"
    if "sherwani" in text:
        return "sherwani"
    if "kurta set" in text or "kurtaset" in text:
        return "kurta_set"
    if "kurta" in text or "kurti" in text:
        return "kurta"
    return ""


def _ethnic_asset_subtype(asset: Dict[str, Any]) -> str:
    return _ethnic_garment_subtype(_asset_policy_blob(asset))


def _demo_safe_visuals_enabled() -> bool:
    return str(os.getenv("AHVI_DEMO_SAFE_VISUALS", "true")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


_MULTI_TOKEN_COMPOUNDS: tuple[tuple[tuple[str, ...], str], ...] = (
    # (required_token_set, family). All tokens must be present (substring-in-token).
    (("button", "down"), "shirt"),
    (("dress", "shirt"), "shirt"),
    (("formal", "shirt"), "shirt"),
    (("oxford", "shirt"), "shirt"),
    (("button", "up"), "shirt"),
    (("polo", "shirt"), "polo"),
    (("dress", "shoe"), "formal_shoe"),
    (("dress", "shoes"), "formal_shoe"),
    (("formal", "shoe"), "formal_shoe"),
    (("formal", "shoes"), "formal_shoe"),
    (("flip", "flop"), "flip_flops"),
    (("flip", "flops"), "flip_flops"),
    (("laptop", "bag"), "laptop_bag"),
    (("messenger", "bag"), "messenger_bag"),
    (("neck", "pillow"), "travel"),
    (("eye", "mask"), "travel"),
    (("water", "bottle"), "travel"),
    (("sun", "glasses"), "sunglasses"),
    (("baseball", "cap"), "cap"),
    (("knit", "wear"), "knit"),
    (("nehru", "jacket"), "ethnic"),
    (("pocket", "square"), "festive_accessory"),
)


def _detect_family(text: Any) -> str:
    if not text:
        return ""
    tokens = _compact_tokens(text)
    if not tokens:
        return ""
    # Multi-token compounds first ("button down", "dress shirt", ...).
    for needles, family in _MULTI_TOKEN_COMPOUNDS:
        if all(_tokens_contain(tokens, (n,)) for n in needles):
            return family
    # Per-token markers. Tokens are alphanum-only per word so "T-Shirt" is
    # one token ``tshirt`` while "shirt" + "shirt" never combine into
    # ``shirtshirt`` (which would otherwise match ``tshirt`` as substring).
    for markers, family in _FAMILY_RULES:
        if _tokens_contain(tokens, markers):
            return family
    return ""


def _asset_family(asset: Dict[str, Any]) -> str:
    blob = " ".join(
        [
            _asset_text(asset.get("name")),
            _asset_text(asset.get("subcategory")),
            _asset_text(asset.get("category")),
            " ".join(_asset_list(asset.get("tags"))),
        ]
    )
    return _detect_family(blob)


def _target_family(target_text: str) -> str:
    return _detect_family(target_text)


def _asset_allowed_for_context(
    asset: Dict[str, Any],
    *,
    occasion: str = "",
    placement: str = "hero",
    target_text: str = "",
) -> bool:
    family = _asset_family(asset)
    group = _FAMILY_GROUP.get(family, "")
    target_family = _target_family(target_text)
    if _occasion_asset_block_reason(
        asset,
        occasion=occasion,
        placement=placement,
        target_text=target_text,
    ):
        return False
    if placement == "hero":
        if group in _HERO_FORBIDDEN_GROUPS:
            return False
        if family in {"loungewear", "swim_shorts", "gym_shorts"}:
            return False
        if _demo_safe_visuals_enabled():
            if group and group not in {"top", "bottom", "outerwear", "footwear"}:
                return False
            if target_family and not family:
                return False
        if target_family:
            if target_family == "polo" and "knit" in _norm(target_text) and family == "knit":
                return True
            allowed = _FAMILY_ALLOWED_FOR_TARGET.get(target_family)
            if allowed is not None:
                if not family or family not in allowed:
                    return False
            elif _demo_safe_visuals_enabled():
                target_group = _FAMILY_GROUP.get(target_family, "")
                if target_group and group and target_group != group:
                    return False
            if target_family == "shirt" and family in {"tshirt", "polo", "hoodie", "sweatshirt"}:
                return False
            if target_family == "tshirt" and family in {"shirt", "polo", "hoodie", "sweatshirt"}:
                return False
            if target_family == "polo" and family in {"shirt", "tshirt", "hoodie", "sweatshirt"}:
                return False
            if target_family in {"hoodie", "sweatshirt"} and family in {"shirt", "tshirt", "polo"}:
                return False
        return True
    if placement == "complete":
        if family == "cardholder":
            return False
        if _is_office_occasion_policy(occasion) and family in _OFFICE_CTL_REJECT_FAMILIES:
            return False
        if group in {"travel", "grooming", "electronics"}:
            return False
        # Casual/social: beanie + cap read too street for a date / brunch and
        # break the "approachable, put-together" mood. Hard-reject them so
        # they can never enter complete_the_look for these occasions.
        if _is_casual_social_policy(occasion) and family in {"beanie", "cap", "cardholder"}:
            return False
        return True
    if placement == "missing":
        if family == "cardholder":
            return False
        # Casual/social missing pieces must stay on-occasion: never suggest a
        # beanie or cap to "complete" a coffee date / brunch look.
        if group in {"travel", "grooming", "electronics"}:
            return False
        if _is_casual_social_policy(occasion) and family in {"beanie", "cap", "cardholder"}:
            return False
        target_group = _FAMILY_GROUP.get(target_family, "")
        if target_group == "accessory":
            return True
        if group in _HERO_FORBIDDEN_GROUPS and target_group and target_group != group:
            return False
        if target_family:
            allowed = _FAMILY_ALLOWED_FOR_TARGET.get(target_family)
            if allowed is not None and family and family not in allowed:
                return False
        return True
    return True


def _asset_context_score(
    asset: Dict[str, Any],
    *,
    occasion: str = "",
    placement: str = "hero",
    target_text: str = "",
) -> int:
    family = _asset_family(asset)
    target_family = _target_family(target_text)
    occasion_family = _visual_occasion_family(occasion, target_text)
    blob = _asset_policy_blob(asset)
    bonus = 0
    # Wedding/festive intelligence: strongly prefer ethnic/festive heroes and
    # demote plain Western office formals so a wedding board stops looking like
    # an office board. Scoped to wedding occasions only.
    if occasion_family == "indian_festive":
        if _is_ethnic_asset(asset):
            bonus += 30
        elif placement == "hero" and family in {"blazer", "shirt", "polo"}:
            bonus -= 12
    if occasion_family == "indian_festive":
        if _is_ethnic_asset(asset):
            bonus += 20
        if any(term in blob for term in ("kurta", "sherwani", "bandhgala", "nehru jacket")):
            bonus += 16
        if any(term in blob for term in ("mojari", "jutti", "ethnic sandal", "kolhapuri")):
            bonus += 18
        if family == "festive_accessory" or any(
            term in blob for term in ("brooch", "pocket square", "festive watch", "festive ring")
        ):
            bonus += 14
        if any(
            color in blob
            for color in (
                "cream",
                "ivory",
                "gold",
                "mustard",
                "marigold",
                "yellow",
                "red",
                "maroon",
                "green",
                "beige",
                "navy",
            )
        ):
            bonus += 6
    elif occasion_family == "christian_wedding":
        if family in {"blazer", "shirt", "trouser", "formal_shoe", "loafer", "tie", "watch"}:
            bonus += 16
    elif occasion_family == "funeral":
        if family in {"blazer", "shirt", "trouser", "formal_shoe", "loafer", "watch"}:
            bonus += 14
        if any(color in blob for color in ("black", "charcoal", "navy")):
            bonus += 10
    elif occasion_family == "travel":
        if family in {
            "sneaker",
            "loafer",
            "trouser",
            "chino",
            "overshirt",
            "jacket",
            "backpack",
            "duffle_bag",
        }:
            bonus += 12
    elif occasion_family == "conference":
        if family in {"blazer", "shirt", "trouser", "formal_shoe", "loafer", "belt", "watch"}:
            bonus += 16
    # Casual/social occasion intelligence (coffee date, first date, brunch,
    # date night). Demote business/formal pieces + beanie/cap, boost relaxed
    # pieces — across hero AND complete placements — unless the user
    # explicitly asked for the demoted piece in their query/hero text.
    if _is_casual_social_policy(occasion):
        # User explicitly asked for the demoted piece (e.g. "blazer for a
        # coffee date") → respect it, skip the demotion.
        requested_family = _detect_family(f"{target_text} {occasion}")
        explicitly_requested = bool(family) and family == requested_family
        if family in _CASUAL_SOCIAL_DEMOTE_FAMILIES and not explicitly_requested:
            bonus -= 14
        elif family in _CASUAL_SOCIAL_BOOST_FAMILIES:
            bonus += 8
    if placement == "complete":
        if _is_office_occasion_policy(occasion):
            if family in _OFFICE_CTL_PREFER_FAMILIES:
                bonus += 12
            if family in _OFFICE_CTL_REJECT_FAMILIES:
                bonus -= 25
        if _is_coffee_date_policy(occasion):
            if family in _COFFEE_DATE_ALLOWED_ACC:
                bonus += 4
            if family == "sunglasses" and not any(
                cue in str(occasion or "").lower() for cue in ("outdoor", "daytime", "park", "beach")
            ):
                bonus -= 6
    if placement in {"hero", "missing"} and target_family:
        allowed = _FAMILY_ALLOWED_FOR_TARGET.get(target_family, {target_family})
        if family and family in allowed:
            bonus += 6
        elif family:
            bonus -= 4
    return bonus


# ---------------------------------------------------------------------
# End style asset policy
# ---------------------------------------------------------------------


def _asset_matches_hero_slot(asset: Dict[str, Any], expected_slot: str | None, hero_text: str) -> bool:
    if not expected_slot:
        return True
    asset_text = " ".join(
        [
            _asset_text(asset.get("name")),
            _asset_text(asset.get("category")),
            _asset_text(asset.get("subcategory")),
            " ".join(_asset_list(asset.get("tags"))),
        ]
    ).lower()
    tokens = _style_tokens(asset_text)
    hero_tokens = _style_tokens(hero_text)
    has = lambda *terms: bool(tokens.intersection(set(terms)))

    if expected_slot == "bottom":
        if has("jacket", "outerwear", "overshirt", "coat", "blazer", "hoodie", "sweatshirt"):
            return False
        return has("bottom", "bottoms", "trouser", "trousers", "pant", "pants", "chino", "chinos", "jean", "jeans", "denim", "short", "shorts")

    if expected_slot == "blazer":
        if has("hoodie", "sweatshirt", "overshirt"):
            return False
        return "blazer" in tokens

    if expected_slot == "outerwear":
        if has("hoodie", "sweatshirt") and not hero_tokens.intersection({"hoodie", "sweatshirt"}):
            return False
        return has("jacket", "overshirt", "coat", "blazer", "outerwear")

    if expected_slot == "top":
        if has("jacket", "blazer", "overshirt", "outerwear", "coat"):
            return False
        intent = _hero_shirt_intent(hero_text)
        asset_intent = _asset_shirt_intent(asset_text)
        if intent == "formal_shirt":
            # Block t-shirts/polos/hoodies/sweatshirts from formal shirt heroes.
            if asset_intent in {"tshirt", "polo", "casual_pullover"}:
                return False
            if asset_intent == "formal_shirt":
                return True
            # Fallback: accept only assets whose blob clearly reads as a shirt.
            asset_tokens_compact = _compact_tokens(asset_text)
            return _tokens_contain(asset_tokens_compact, ("shirt",)) and not _tokens_contain(
                asset_tokens_compact, ("tshirt", "tee")
            )
        if intent == "tshirt":
            if asset_intent in {"formal_shirt", "polo", "casual_pullover", "knit"}:
                return False
            return asset_intent == "tshirt"
        if intent == "polo":
            if asset_intent in {"formal_shirt", "tshirt", "casual_pullover"}:
                return False
            if "knit" in _norm(hero_text) and asset_intent == "knit":
                return True
            return asset_intent == "polo"
        if intent == "knit":
            if asset_intent in {"tshirt", "polo", "formal_shirt"}:
                return False
            return asset_intent in {"knit", "casual_pullover"}
        # No specific shirt intent → keep prior permissive behaviour but
        # still reject hoodies/sweatshirts unless hero explicitly asked for them.
        if has("hoodie", "sweatshirt") and not hero_tokens.intersection({"hoodie", "sweatshirt"}):
            return False
        return has("top", "shirt", "oxford", "button", "buttondown", "polo", "tee", "tshirt", "knit", "knitwear", "sweater", "hoodie", "sweatshirt")

    if expected_slot == "footwear":
        return has("shoe", "shoes", "loafer", "loafers", "sneaker", "sneakers", "boot", "boots", "footwear", "oxford")

    return True


def _hero_asset_allowed(
    asset: Dict[str, Any], direction: Dict[str, Any], occasion: str = ""
) -> bool:
    hero = _asset_text(direction.get("hero_piece")) or " ".join(
        _safe_list(direction.get("items") or direction.get("pieces"), limit=1)
    )
    # Wedding/festive: the LLM direction's hero is usually Western ("Gray
    # Blazer"), so the family gate below would reject in-pool ethnic pieces
    # (kurta/bandhgala/sherwani) before they can be scored. Let ethnic assets
    # through for wedding occasions so they compete. When the hero already
    # names an ethnic garment, keep the image paired to that exact subtype.
    if _visual_occasion_family(occasion) == "indian_festive" and _is_ethnic_asset(asset):
        hero_subtype = _ethnic_garment_subtype(hero)
        asset_subtype = _ethnic_asset_subtype(asset)
        if hero_subtype:
            if hero_subtype == "kurta_set":
                return asset_subtype in {"kurta_set", "kurta"}
            if hero_subtype == "kurta":
                return asset_subtype in {"kurta", "kurta_set"}
            return asset_subtype == hero_subtype
        return True
    expected_slot = _hero_expected_slot(hero)
    if expected_slot and not _asset_matches_hero_slot(asset, expected_slot, hero):
        return False
    hero_category = _style_category(hero)
    if not hero_category:
        return True
    asset_terms = _asset_category_terms(asset)
    if asset_terms.intersection({"blocked_hero"}):
        return False
    hero_tokens = _style_tokens(hero)
    # Shirt-family intent: route by intent so polo/tshirt/formal_shirt heroes
    # don't fall into the wrong branch when both "shirt" and "polo" are
    # mentioned (e.g. "Cream Polo Shirt").
    shirt_intent = _hero_shirt_intent(hero)
    if shirt_intent == "polo":
        asset_intent = _asset_shirt_intent(_asset_text(asset.get("name")) + " " + _asset_text(asset.get("subcategory")) + " " + " ".join(_asset_list(asset.get("tags"))))
        if "knit" in _norm(hero) and asset_intent == "knit":
            return True
        return asset_intent == "polo" or "polo" in asset_terms
    if shirt_intent == "tshirt":
        asset_intent = _asset_shirt_intent(_asset_text(asset.get("name")) + " " + _asset_text(asset.get("subcategory")) + " " + " ".join(_asset_list(asset.get("tags"))))
        return asset_intent == "tshirt"
    if shirt_intent == "knit" or hero_tokens.intersection({"sweater", "knit", "knitwear"}):
        return bool(asset_terms.intersection({"sweater", "knit", "knitwear"}))
    if shirt_intent == "formal_shirt" or hero_tokens.intersection({"shirt", "oxford", "linen", "button", "buttondown"}):
        asset_intent = _asset_shirt_intent(_asset_text(asset.get("name")) + " " + _asset_text(asset.get("subcategory")) + " " + " ".join(_asset_list(asset.get("tags"))))
        if asset_intent in {"tshirt", "polo", "casual_pullover"}:
            return False
        return asset_intent == "formal_shirt" or bool(asset_terms.intersection({"shirt", "oxford", "linen", "button", "buttondown"}))
    if hero_tokens.intersection({"overshirt"}):
        return "overshirt" in asset_terms or "outerwear" in asset_terms
    if hero_tokens.intersection({"jacket"}):
        return "jacket" in asset_terms or "outerwear" in asset_terms
    if hero_tokens.intersection({"hoodie", "sweatshirt"}):
        return bool(asset_terms.intersection({"hoodie", "sweatshirt"}))
    if hero_tokens.intersection({"blazer"}):
        return "blazer" in asset_terms or "outerwear" in asset_terms
    if hero_tokens.intersection({"trouser", "trousers", "pant", "pants", "chino", "chinos"}):
        return bool(asset_terms.intersection({"trouser", "trousers", "pant", "pants", "chino", "chinos", "bottom"}))
    if hero_tokens.intersection({"jean", "jeans", "denim"}):
        return bool(asset_terms.intersection({"jean", "jeans", "denim", "bottom"}))
    if hero_tokens.intersection({"loafer", "loafers"}):
        return bool(asset_terms.intersection({"loafer", "loafers", "footwear"}))
    if hero_tokens.intersection({"sneaker", "sneakers"}):
        return bool(asset_terms.intersection({"sneaker", "sneakers", "footwear"}))
    if hero_category == "top":
        return bool(asset_terms.intersection({"top", "outerwear"}))
    if hero_category == "outerwear":
        return bool(asset_terms.intersection({"outerwear", "top"}))
    if hero_category == "bottom":
        return "bottom" in asset_terms
    if hero_category == "footwear":
        return "footwear" in asset_terms
    if hero_category == "accessory":
        return "accessory" in asset_terms or "footwear" in asset_terms
    if hero_category == "dress":
        return "dress" in asset_terms or "top" in asset_terms
    return True


def _hero_asset_match_bonus(asset: Dict[str, Any], direction: Dict[str, Any]) -> int:
    hero = _asset_text(direction.get("hero_piece"))
    if not hero:
        return 0
    hero_tokens = {token for token in _style_tokens(hero) if len(token) > 3}
    asset_blob = " ".join(
        [
            _asset_text(asset.get("name")),
            _asset_text(asset.get("subcategory")),
            " ".join(_asset_list(asset.get("tags"))),
        ]
    ).lower()
    if not hero_tokens:
        return 0
    overlap = sum(1 for token in hero_tokens if token in asset_blob)
    if _norm(hero) and _norm(hero) in _norm(asset.get("name")):
        return 12
    if overlap >= 2:
        return 8
    if overlap == 1:
        return 4
    return 0


def _direction_component_categories(components: List[str]) -> set[str]:
    return {cat for cat in (_style_category(item) for item in components) if cat}


def _mentioned_garment_categories(text: Any) -> set[str]:
    tokens = _style_tokens(text)
    found: set[str] = set()
    for category, terms in _GARMENT_CATEGORY_TERMS.items():
        normalized_terms = {t.replace("-", "") for t in terms} | terms
        if tokens.intersection(normalized_terms):
            found.add(category)
    return found


def _style_similarity(a: Any, b: Any) -> float:
    at = _style_tokens(a)
    bt = _style_tokens(b)
    if not at or not bt:
        return 0.0
    overlap = len(at.intersection(bt))
    return overlap / max(1, min(len(at), len(bt)))


def _style_items_similar(a: Any, b: Any) -> bool:
    ac = _style_category(a)
    bc = _style_category(b)
    if ac and bc and ac == bc and _style_similarity(a, b) >= 0.45:
        return True
    return _style_similarity(a, b) >= 0.75


def _missing_name_is_generic(name: Any) -> bool:
    tokens = _style_tokens(name)
    if not tokens:
        return True
    return bool(tokens.intersection(_GENERIC_MISSING_WORDS)) and len(tokens) <= 2


def _specific_missing_piece_name(
    candidate: Any,
    direction: Dict[str, Any],
    *,
    occasion: str = "",
) -> str:
    raw = _asset_text(candidate)
    if not raw:
        return ""
    norm = _norm(raw)
    palette = _safe_list(direction.get("palette") or direction.get("colors"), limit=5)
    palette_blob = " ".join(palette).lower()
    archetype = _norm(direction.get("archetype"))
    options = list(_SPECIFIC_MISSING_NAMES.get(norm) or [])
    if not options:
        category = _style_category(raw)
        if category == "outerwear":
            options = ["Olive Cotton Overshirt", "Navy Twill Overshirt", "Soft Camel Relaxed Blazer"]
        elif category == "footwear":
            options = ["Dark Brown Penny Loafers", "Clean White Leather Sneakers"]
        elif category == "bottom":
            options = ["Dark Wash Straight-Leg Jeans", "Stone Tailored Chinos"]
        elif category == "accessory":
            options = ["Brushed Steel Watch", "Dark Brown Leather Belt"]
        elif category == "top":
            options = ["Cream Knit Polo", "White Oxford Shirt"]
    if not options:
        return raw.title()
    preferred = options[0]
    if any(color in palette_blob for color in ("navy", "blue")):
        preferred = next((item for item in options if "navy" in item.lower() or "dark wash" in item.lower()), preferred)
    elif any(color in palette_blob for color in ("camel", "tan", "brown", "tobacco")):
        preferred = next((item for item in options if any(x in item.lower() for x in ("camel", "brown", "cognac"))), preferred)
    elif any(color in palette_blob for color in ("charcoal", "black", "grey", "gray")):
        preferred = next((item for item in options if any(x in item.lower() for x in ("charcoal", "black", "steel"))), preferred)
    if "quiet luxury" in archetype:
        preferred = next((item for item in options if any(x in item.lower() for x in ("camel", "steel", "penny"))), preferred)
    if "coffee" in _norm(occasion):
        preferred = next(
            (
                item
                for item in options
                if any(x in item.lower() for x in ("dark wash", "penny", "cotton", "knit"))
            ),
            preferred,
        )
    return preferred


def _missing_piece_reason_for_direction(
    missing_name: Any,
    direction: Dict[str, Any],
    *,
    occasion: str = "",
) -> str:
    name = _asset_text(missing_name) or "This piece"
    if _visual_occasion_family(occasion, name) == "indian_festive":
        occasion_text = _norm(occasion)
        if "haldi" in occasion_text:
            return "Completes the festive kurta look while staying comfortable for rituals."
        if "sangeet" in occasion_text:
            return "Adds festive structure while keeping movement easy."
        return "Adds traditional polish without making the outfit feel heavy."
    components = _safe_list(direction.get("items") or direction.get("pieces"), limit=6)
    hero = _asset_text(direction.get("hero_piece")) or (components[0] if components else "the main piece")
    support = components[1] if len(components) > 1 else (components[0] if components else "the outfit")
    occ = _asset_text(occasion) or "this plan"
    formula = _direction_formula_signature(direction)
    if formula == ("shirt", "wide_leg"):
        return f"{name} completes the clean-shirt direction for {occ} and balances {support.lower()} already used in this look."
    if formula in {("blazer", "trouser"), ("blazer", "denim")}:
        return f"{name} finishes the tailored direction for {occ} without making {hero.lower()} feel too formal."
    if formula == ("knit", "tailored_trouser"):
        return f"{name} adds finish to the soft-texture direction and keeps {support.lower()} looking intentional for {occ}."
    return f"{name} strengthens this {occ} look by supporting {hero.lower()} and making the components feel complete."


def _festive_missing_piece_replacement(
    occasion: Any,
    direction: Dict[str, Any] | None = None,
    *,
    target_gender: str = "unknown",
    allow_feminine: bool = False,
) -> Dict[str, Any] | None:
    if _visual_occasion_family(occasion) != "indian_festive":
        return None
    del direction, target_gender, allow_feminine
    return {
        "name": "Ethnic Footwear",
        "category": "Footwear",
        "reason": "Completes the festive kurta look while staying comfortable for rituals.",
        "unlocks": ["Festive styling"],
    }


def _palette_terms(value: Any) -> set[str]:
    values = _safe_list(value, limit=8)
    terms: set[str] = set()
    for item in values:
        terms.update(_style_tokens(item))
    return terms


def _palette_overlap(a: Any, b: Any) -> float:
    at = _palette_terms(a)
    bt = _palette_terms(b)
    if not at or not bt:
        return 0.0
    return len(at.intersection(bt)) / max(1, min(len(at), len(bt)))


def _silhouette_tokens(direction: Dict[str, Any]) -> set[str]:
    components = _safe_list(direction.get("items") or direction.get("pieces"), limit=8)
    blob = " ".join(components).lower()
    tokens: set[str] = set()
    categories = _direction_component_categories(components)
    tokens.update(categories)
    if any(word in blob for word in ("blazer", "jacket", "overshirt", "coat", "cardigan")):
        tokens.add("structured_layer")
    if any(word in blob for word in ("knit", "sweater", "polo", "linen", "suede", "textured")):
        tokens.add("texture")
    if any(word in blob for word in ("tee", "t-shirt", "crew-neck", "crewneck")):
        tokens.add("minimal_base")
    if any(word in blob for word in ("cargo", "utility", "overshirt", "chore")):
        tokens.add("utility")
    if any(word in blob for word in ("loafer", "loafers")):
        tokens.add("loafer_footwear")
    if any(word in blob for word in ("sneaker", "sneakers")):
        tokens.add("sneaker_footwear")
    return tokens


def _direction_formula_signature(direction: Dict[str, Any]) -> tuple[str, ...]:
    components = _safe_list(direction.get("items") or direction.get("pieces"), limit=8)
    blob = " ".join([_asset_text(direction.get("hero_piece")), *components]).lower()
    categories = _direction_component_categories(components)
    has = lambda *terms: any(term in blob for term in terms)

    if has("matching set", "co-ord", "co ord", "coordinated set", "suit set"):
        return ("matching_set",)
    if has("dress") and categories.intersection({"outerwear", "accessory"}):
        return ("dress", "layer")
    if has("blazer", "jacket") and categories.intersection({"bottom"}):
        if has("jean", "denim"):
            return ("blazer", "denim")
        return ("blazer", "trouser")
    if has("knit", "sweater", "polo") and categories.intersection({"bottom"}):
        return ("knit", "tailored_trouser")
    if has("oxford", "button-down", "button down", "shirt") and categories.intersection({"bottom"}):
        if has("wide leg", "wide-leg", "wideleg"):
            return ("shirt", "wide_leg")
        if has("jean", "denim"):
            return ("shirt", "denim")
        return ("shirt", "trouser")
    if has("overshirt", "utility layer", "shacket") and has("jean", "denim"):
        return ("overshirt", "denim")
    if has("tee", "t-shirt", "crew-neck", "crewneck") and categories.intersection({"outerwear"}):
        return ("tee", "layer")
    hero_role = _style_category(direction.get("hero_piece")) or "hero"
    silhouette = _silhouette_tokens(direction)
    formula_bits = {hero_role}
    for bit in ("structured_layer", "texture", "minimal_base", "utility"):
        if bit in silhouette:
            formula_bits.add(bit)
    return tuple(sorted(formula_bits))


_STRUCTURED_LAYER_HERO_FAMILIES: set[str] = {"blazer", "jacket", "coat"}


def _directions_too_similar(candidate: Dict[str, Any], accepted: Dict[str, Any]) -> tuple[bool, str]:
    hero = _asset_text(candidate.get("hero_piece"))
    accepted_hero = _asset_text(accepted.get("hero_piece"))
    if hero and accepted_hero and _style_items_similar(hero, accepted_hero):
        return True, "same_hero"
    # Repeated structured-layer hero (blazer/jacket/coat) across directions
    # reads as "blazer, blazer, blazer" even when the formulas differ. Treat
    # a second structured-layer hero as a duplicate so the diversity guard
    # swaps in a distinct, more relaxed direction.
    hero_family = _detect_family(hero)
    accepted_hero_family = _detect_family(accepted_hero)
    if (
        hero_family in _STRUCTURED_LAYER_HERO_FAMILIES
        and accepted_hero_family in _STRUCTURED_LAYER_HERO_FAMILIES
    ):
        return True, "repeated_structured_layer_hero"
    formula = _direction_formula_signature(candidate)
    accepted_formula = _direction_formula_signature(accepted)
    palette_overlap = _palette_overlap(
        candidate.get("palette") or candidate.get("colors"),
        accepted.get("palette") or accepted.get("colors"),
    )
    if formula and formula == accepted_formula:
        return True, "same_formula"
    candidate_silhouette = _silhouette_tokens(candidate)
    accepted_silhouette = _silhouette_tokens(accepted)
    if candidate_silhouette and accepted_silhouette:
        overlap = len(candidate_silhouette.intersection(accepted_silhouette)) / max(
            1, min(len(candidate_silhouette), len(accepted_silhouette))
        )
        if overlap >= 0.75 and palette_overlap >= 0.6:
            return True, "same_silhouette_palette"
    if palette_overlap >= 0.85 and _style_category(hero) == _style_category(accepted_hero):
        return True, "same_palette_hero_role"
    return False, ""


_GENERIC_VISUAL_STRATEGIES: List[Dict[str, Any]] = [
    {
        "title": "Structured Layer",
        "subtitle": "Shape-led polish",
        "archetype": "Structured Ease",
        "hero_piece": "Unstructured Navy Blazer",
        "items": ["Unstructured Navy Blazer", "White Crew-Neck T-Shirt", "Tailored Stone Trouser", "Dark Brown Penny Loafers"],
        "palette": ["navy", "white", "tan"],
        "description": "A structured layer leads the outfit while the tee and tailored trouser keep it approachable.",
        "why_it_works": "The layer gives the look shape without making it stiff. The lighter base keeps the overall impression relaxed and intentional.",
        "styling_tip": "Keep the layer open for an easier finish.",
    },
    {
        "title": "Soft Texture",
        "subtitle": "Tactile and easy",
        "archetype": "Textural Ease",
        "hero_piece": "Fine-Gauge Knit Polo",
        "items": ["Fine-Gauge Knit Polo", "Tailored Khaki Trouser", "Clean White Leather Sneakers"],
        "palette": ["cream", "olive", "brown"],
        "description": "Texture becomes the focal point, with quiet separates keeping the outfit grounded.",
        "why_it_works": "A softer hero piece makes the look feel considered without feeling formal. The restrained base keeps the texture from becoming busy.",
        "styling_tip": "Keep the texture neat and let it be the focal point.",
    },
    {
        "title": "Clean Shirt Base",
        "subtitle": "Crisp and minimal",
        "archetype": "Clean Minimal",
        "hero_piece": "Crisp Oxford Shirt",
        "items": ["Crisp Oxford Shirt", "Stone Wide-Leg Trouser", "Black Leather Loafers"],
        "palette": ["white", "stone", "charcoal"],
        "description": "A crisp shirt anchors the look, with the wide-leg trouser keeping everything sharp but easy.",
        "why_it_works": "The shirt creates clarity, while the relaxed trouser shape keeps the outfit from feeling overworked.",
        "styling_tip": "Roll sleeves once if the setting feels relaxed.",
    },
    {
        "title": "Relaxed Utility",
        "subtitle": "Casual polish",
        "archetype": "Modern Utility",
        "hero_piece": "Relaxed Matching Set",
        "items": ["Relaxed Matching Set", "Plain Ecru Base Top", "Clean White Leather Sneakers"],
        "palette": ["olive", "ecru", "tobacco"],
        "description": "A utility layer adds ease and function while clean basics keep the outfit refined.",
        "why_it_works": "The utility detail makes the look feel practical and current. A simple base keeps it polished rather than rugged.",
        "styling_tip": "Keep pockets and layers neat so it reads intentional.",
    },
]


def _generic_diversity_replacement(index: int, category: str | None) -> Dict[str, Any]:
    replacement = dict(_GENERIC_VISUAL_STRATEGIES[index % len(_GENERIC_VISUAL_STRATEGIES)])
    if category:
        replacement["description"] = _direction_description_from_source(replacement, category)
        replacement["why_it_works"] = _direction_why_from_source(replacement, category)
    replacement["pieces"] = list(replacement.get("items") or [])
    replacement["colors"] = list(replacement.get("palette") or [])
    return replacement


def _apply_generic_visual_diversity(
    directions: List[Dict[str, Any]],
    *,
    category: str | None = None,
) -> List[Dict[str, Any]]:
    diversified: List[Dict[str, Any]] = []
    for idx, direction in enumerate(directions):
        candidate = dict(direction)
        reason = ""
        for accepted in diversified:
            duplicate, reason = _directions_too_similar(candidate, accepted)
            if duplicate:
                replacement = _generic_diversity_replacement(idx, category)
                # Preserve the selected archetype if one was already enforced;
                # the role/formula changes, not the registry source of truth.
                if candidate.get("archetype"):
                    replacement["archetype"] = candidate.get("archetype")
                candidate = _ensure_direction_logic(_normalize_direction(replacement, replacement))
                break
        diversified.append(candidate)
        if reason:
            logger.info(
                "AHVI_VISUAL_DIVERSITY_GUARD_APPLIED index=%d reason=%s formula=%s",
                idx,
                reason,
                _direction_formula_signature(candidate),
            )
    return diversified


_FESTIVE_BOARD_ALLOWED_TERMS: tuple[str, ...] = (
    "kurta",
    "kurta set",
    "nehru jacket",
    "bandhgala",
    "sherwani",
    "churidar",
    "ethnic trouser",
    "ethnic pant",
    "jutti",
    "mojari",
    "kolhapuri",
    "ethnic sandal",
    "festive sandal",
    "indo western",
    "indowestern",
)
_FESTIVE_BOARD_DISALLOWED_TERMS: tuple[str, ...] = (
    "oxford shirt",
    "oxford",
    "polo",
    "basic tee",
    "t shirt",
    "tshirt",
    "hoodie",
    "sweatshirt",
    "sneaker",
    "running shoe",
    "business loafer",
    "loafer",
    "jeans",
    "denim",
)


def _festive_direction_compatibility(direction: Dict[str, Any]) -> float:
    hero = _asset_text(direction.get("hero_piece"))
    components = _safe_list(direction.get("items") or direction.get("pieces"), limit=8)
    core = [hero, *components]
    core = list(dict.fromkeys(item for item in core if item))
    if not core:
        return 0.0
    safe_count = 0
    for item in core:
        blob = item.lower().replace("_", " ").replace("-", " ")
        if any(term in blob for term in _FESTIVE_BOARD_DISALLOWED_TERMS):
            return 0.0
        if any(term in blob for term in _FESTIVE_BOARD_ALLOWED_TERMS):
            safe_count += 1
        else:
            return 0.0
    hero_blob = hero.lower().replace("_", " ").replace("-", " ")
    if not any(term in hero_blob for term in _FESTIVE_BOARD_ALLOWED_TERMS):
        return 0.0
    return safe_count / max(1, len(core))


def _festive_visual_direction_replacements() -> List[Dict[str, Any]]:
    return [
        {
            "title": "Vibrant Celebration",
            "archetype": "Vibrant Celebration",
            "hero_piece": "Marigold Yellow Kurta",
            "items": ["Marigold Yellow Kurta", "Ivory Churidar", "Gold Juttis"],
            "pieces": ["Marigold Yellow Kurta", "Ivory Churidar", "Gold Juttis"],
            "palette": ["marigold yellow", "ivory", "gold"],
            "colors": ["marigold yellow", "ivory", "gold"],
            "description": "A bright kurta with traditional separates keeps the celebration festive and comfortable.",
            "why_it_works": "The kurta leads with color while the churidar and juttis keep the silhouette culturally grounded.",
            "style_note": "Keep the fabric breathable and the finishing details restrained.",
        },
        {
            "title": "Festive Heritage",
            "archetype": "Festive Heritage",
            "hero_piece": "Embroidered Nehru Jacket",
            "items": ["Embroidered Nehru Jacket", "Cream Kurta Set", "Brown Mojaris"],
            "pieces": ["Embroidered Nehru Jacket", "Cream Kurta Set", "Brown Mojaris"],
            "palette": ["cream", "maroon", "brown"],
            "colors": ["cream", "maroon", "brown"],
            "description": "A Nehru jacket adds festive structure over a comfortable traditional base.",
            "why_it_works": "The jacket creates ceremony-ready polish while the kurta set and mojaris preserve ease.",
            "style_note": "Let one embroidered layer carry the detail.",
        },
        {
            "title": "Refined Traditional",
            "archetype": "Refined Traditional",
            "hero_piece": "Navy Bandhgala",
            "items": ["Navy Bandhgala", "Ivory Ethnic Trousers", "Black Mojaris"],
            "pieces": ["Navy Bandhgala", "Ivory Ethnic Trousers", "Black Mojaris"],
            "palette": ["navy", "ivory", "black"],
            "colors": ["navy", "ivory", "black"],
            "description": "A clean bandhgala direction brings formal festive polish without western businesswear.",
            "why_it_works": "The bandhgala gives structure while ethnic trousers and mojaris keep the occasion language consistent.",
            "style_note": "Keep accessories minimal so the tailoring remains the focus.",
        },
    ]


def _enforce_festive_visual_directions(
    directions: List[Dict[str, Any]],
    *,
    occasion: Any,
) -> List[Dict[str, Any]]:
    if _visual_occasion_family(occasion) != "indian_festive":
        return directions
    replacements = _festive_visual_direction_replacements()
    guarded: List[Dict[str, Any]] = []
    for index, direction in enumerate(directions or []):
        score = _festive_direction_compatibility(direction)
        if score < 1.0:
            replacement = dict(replacements[index % len(replacements)])
            logger.info(
                "AHVI_FESTIVE_BOARD_REPLACED occasion=%s index=%d hero=%r compatibility=%.2f replacement=%r",
                _asset_text(occasion),
                index,
                _asset_text(direction.get("hero_piece")),
                score,
                replacement.get("hero_piece"),
            )
            guarded.append(replacement)
        else:
            guarded.append(direction)
    while len(guarded) < 3:
        guarded.append(dict(replacements[len(guarded) % len(replacements)]))
    return guarded[:3]


def _component_blob(components: List[str]) -> str:
    return ", ".join(str(item).strip() for item in components if str(item).strip())


def _direction_description_from_source(direction: Dict[str, Any], occasion: str = "") -> str:
    components = _safe_list(direction.get("items") or direction.get("pieces"), limit=6)
    hero = _asset_text(direction.get("hero_piece")) or (components[0] if components else "the hero piece")
    support = components[1] if len(components) > 1 else "clean supporting pieces"
    occ = _asset_text(occasion) or "this setting"
    return f"{hero} leads the look, with {support} keeping it balanced for {occ}."


def _direction_why_from_source(direction: Dict[str, Any], occasion: str = "") -> str:
    components = _safe_list(direction.get("items") or direction.get("pieces"), limit=6)
    hero = _asset_text(direction.get("hero_piece")) or (components[0] if components else "the main piece")
    support = components[1] if len(components) > 1 else "the rest of the outfit"
    occ = _asset_text(occasion) or "the moment"
    return (
        f"For {occ}, {hero.lower()} gives the outfit a clear point of view. "
        f"{support} keeps it wearable, so the final look feels intentional without becoming forced."
    )


def _direction_tip_from_source(direction: Dict[str, Any]) -> str:
    components = " ".join(_safe_list(direction.get("items") or direction.get("pieces"), limit=6)).lower()
    if "blazer" in components:
        return "Keep the blazer open for an easier finish."
    if "knit" in components or "sweater" in components:
        return "Keep the knit neat at the hem."
    if "tee" in components or "t-shirt" in components:
        return "Choose a clean neckline and simple layers."
    if "shirt" in components:
        return "Roll sleeves once for a relaxed finish."
    return "Keep one detail polished and the rest easy."


def _direction_text_contradicts_components(text: Any, components: List[str]) -> bool:
    if not _asset_text(text):
        return False
    mentioned = _mentioned_garment_categories(text)
    if not mentioned:
        return False
    allowed = _direction_component_categories(components)
    # Accessories are allowed in copy when Complete the Look exists later; core
    # garment contradictions are what break user trust.
    core_mentioned = mentioned - {"accessory"}
    core_allowed = allowed - {"accessory"}
    return bool(core_mentioned - core_allowed)


def _direction_text_disallowed_for_gender(
    text: Any,
    *,
    target_gender: str,
    allow_feminine: bool = False,
) -> bool:
    if not _asset_text(text):
        return False
    return not _style_text_allowed_for_gender(text, target_gender, allow_feminine=allow_feminine)


def _missing_piece_duplicate_reason(
    missing_name: Any,
    *,
    hero_piece: Any,
    components: List[str],
) -> str:
    name = _asset_text(missing_name)
    if not name:
        return "empty"
    if _style_items_similar(name, hero_piece):
        return "hero_duplicate"
    missing_category = _style_category(name)
    hero_category = _style_category(hero_piece)
    if missing_category and hero_category and missing_category == hero_category and _style_similarity(name, hero_piece) >= 0.35:
        return "hero_same_category"
    for component in components:
        if _style_items_similar(name, component):
            return "component_duplicate"
        comp_category = _style_category(component)
        if missing_category and comp_category and missing_category == comp_category and _style_similarity(name, component) >= 0.35:
            return "component_same_category"
    return ""


def _fallback_missing_piece_for_direction(
    direction: Dict[str, Any],
    *,
    occasion: str = "",
    target_gender: str = "unknown",
    allow_feminine: bool = False,
) -> str:
    festive = _festive_missing_piece_replacement(
        occasion,
        direction,
        target_gender=target_gender,
        allow_feminine=allow_feminine,
    )
    if festive:
        return _asset_text(festive.get("name"))
    components = _safe_list(direction.get("items") or direction.get("pieces"), limit=6)
    hero = _asset_text(direction.get("hero_piece"))
    if "coffee" in _norm(occasion):
        candidates = _COFFEE_DATE_MISSING_FALLBACKS
    else:
        candidates = _GENERAL_MISSING_FALLBACKS
    for candidate in candidates:
        specific = _specific_missing_piece_name(candidate, direction, occasion=occasion)
        if not _style_text_allowed_for_gender(specific, target_gender, allow_feminine=allow_feminine):
            continue
        if not _missing_piece_duplicate_reason(specific, hero_piece=hero, components=components):
            return specific
    return ""


def _sanitize_direction_for_gender(
    direction: Dict[str, Any],
    *,
    target_gender: str,
    allow_feminine: bool = False,
) -> Dict[str, Any]:
    out = dict(direction)
    pieces = _filter_style_terms_for_gender(
        _safe_list(out.get("items") or out.get("pieces"), limit=8),
        target_gender=target_gender,
        allow_feminine=allow_feminine,
        limit=6,
    )
    if not pieces:
        pieces = _safe_component_fallback(target_gender)
    hero = _asset_text(out.get("hero_piece") or out.get("heroPiece"))
    if not _style_text_allowed_for_gender(hero, target_gender, allow_feminine=allow_feminine):
        hero = pieces[0] if pieces else ""
    out["hero_piece"] = hero
    out["pieces"] = pieces
    out["items"] = pieces
    return out


def _missing_piece_allowed_for_gender(
    missing_piece: Dict[str, Any] | None,
    *,
    target_gender: str,
    allow_feminine: bool = False,
) -> bool:
    if not missing_piece:
        return False
    blob = " ".join(
        [
            _asset_text(missing_piece.get("name")),
            _asset_text(missing_piece.get("category")),
            " ".join(_safe_list(missing_piece.get("unlocks"), limit=8)),
        ]
    )
    return _style_text_allowed_for_gender(blob, target_gender, allow_feminine=allow_feminine)


def _resolve_asset_gender(*, query: Any, user_profile: Any) -> str:
    override = _prompt_gender_override(query)
    if override:
        logger.info("AHVI_ASSET_GENDER_CONTEXT source=prompt gender=%s", override)
        return override
    try:
        from services.style_context_service import _resolve_gender

        profile_gender = _resolve_gender(user_profile if isinstance(user_profile, dict) else {})
    except Exception:  # noqa: BLE001
        profile_gender = "unknown"
    if profile_gender in {"male", "female"}:
        logger.info("AHVI_ASSET_GENDER_CONTEXT source=profile gender=%s", profile_gender)
        return profile_gender
    logger.info("AHVI_ASSET_GENDER_CONTEXT source=neutral gender=unknown")
    return "unknown"


def _asset_allowed_for_gender(asset: Dict[str, Any], target_gender: str) -> bool:
    asset_genders = _asset_list(asset.get("gender"))
    if not asset_genders:
        asset_genders = ["missing"]
    normalized = {_asset_gender(g) for g in asset_genders}
    blob = " ".join(
        [
            _asset_text(asset.get("name")),
            _asset_text(asset.get("category")),
            _asset_text(asset.get("subcategory")),
            " ".join(_asset_list(asset.get("tags"))),
        ]
    ).lower()
    is_feminine_accessory = any(term in blob for term in _FEMININE_ACCESSORY_TERMS)
    if target_gender == "male":
        if is_feminine_accessory and not bool(asset.get("_allow_feminine_accessory")):
            return False
        return bool(normalized.intersection({"male", "unisex"}))
    if target_gender == "female":
        return bool(normalized.intersection({"female", "unisex"}))
    # Unknown/unisex users should stay neutral unless the prompt explicitly
    # asked for feminine accessories.
    if is_feminine_accessory and not bool(asset.get("_allow_feminine_accessory")):
        return False
    return bool(normalized.intersection({"unisex"}))


def _validate_style_assets(assets: List[Dict[str, Any]]) -> None:
    required = ("asset_id", "name", "category", "image_url", "gender", "status")
    quality_fields = ("subcategory", "colors", "archetypes", "occasions")
    for asset in assets:
        missing = [field for field in required if not _asset_text(asset.get(field))]
        weak = [field for field in quality_fields if not _asset_list(asset.get(field)) and not _asset_text(asset.get(field))]
        bad_gender = _asset_gender(asset.get("gender")) not in {"male", "female", "unisex"}
        if missing or bad_gender:
            logger.warning(
                "AHVI_STYLE_ASSET_INVALID asset_id=%s name=%s missing=%s gender=%s",
                _asset_text(asset.get("asset_id") or asset.get("$id")),
                _asset_text(asset.get("name")),
                missing,
                _asset_text(asset.get("gender")),
            )
        elif weak:
            logger.info(
                "AHVI_STYLE_ASSET_WEAK_METADATA asset_id=%s name=%s missing=%s",
                _asset_text(asset.get("asset_id") or asset.get("$id")),
                _asset_text(asset.get("name")),
                weak,
            )


# Lightweight keyword estimators for an asset's formality/movement/energy. Asset
# metadata carries no numeric axes, so we infer from the text blob. Used only by
# the canonical-brief scoring path (brief != None); legacy scoring is untouched.
_ASSET_FORMAL_SIGNALS = (
    "oxford", "loafer", "blazer", "suit", "tuxedo", "tie", "derby", "brogue",
    "dress shoe", "dress shoes", "formal trouser", "tailored trouser",
    "bandhgala", "sherwani", "nehru jacket", "achkan", "heel", "heels", "gown",
    "pump", "pumps",
)
_ASSET_CASUAL_SIGNALS = (
    "tee", "t-shirt", "tshirt", "graphic", "sneaker", "cargo", "hoodie",
    "jogger", "shorts", "overshirt", "cap", "denim", "jean", "sweatshirt",
    "slides", "sandal", "track", "jersey", "tank",
)
_ASSET_MOVEMENT_SIGNALS = (
    "sneaker", "cargo", "jogger", "track", "jersey", "stretch", "shorts",
    "tee", "hoodie", "sandal", "slides", "relaxed", "elastic",
)
_ASSET_LOWMOVE_SIGNALS = (
    "oxford", "loafer", "suit", "blazer", "heel", "heels", "pencil",
    "tailored", "dress shoe", "derby", "gown", "pump",
)
_ASSET_HIGHENERGY_SIGNALS = (
    "graphic", "print", "printed", "neon", "bright", "bold", "color block",
    "colorblock", "statement", "sequin", "metallic", "floral", "logo",
)


def _asset_axis_estimates(blob: str) -> tuple[int, int, int]:
    """Estimate (formality 1..5, movement 1..9, energy 1..9) from an item text
    blob. Heuristic — keyword counts clamped to the axis range."""
    b = (blob or "").lower()
    formality = 3
    for s in _ASSET_FORMAL_SIGNALS:
        if s in b:
            formality += 1
    for s in _ASSET_CASUAL_SIGNALS:
        if s in b:
            formality -= 1
    formality = max(1, min(5, formality))
    movement = 5
    for s in _ASSET_MOVEMENT_SIGNALS:
        if s in b:
            movement += 1
    for s in _ASSET_LOWMOVE_SIGNALS:
        if s in b:
            movement -= 1
    movement = max(1, min(9, movement))
    energy = 4
    for s in _ASSET_HIGHENERGY_SIGNALS:
        if s in b:
            energy += 1
    energy = max(1, min(9, energy))
    return formality, movement, energy


def _asset_score(
    asset: Dict[str, Any],
    *,
    direction: Dict[str, Any],
    occasion: str,
    target_gender: str = "unknown",
    brief: Dict[str, Any] | None = None,
) -> int:
    blob = " ".join(
        [
            _asset_text(asset.get("name")),
            _asset_text(asset.get("category")),
            _asset_text(asset.get("subcategory")),
            " ".join(_asset_list(asset.get("tags"))),
        ]
    ).lower()
    direction_terms = " ".join(
        [
            _asset_text(direction.get("hero_piece")),
            " ".join(_safe_list(direction.get("items") or direction.get("pieces"), limit=8)),
            " ".join(_safe_list(direction.get("colors") or direction.get("palette"), limit=6)),
        ]
    ).lower()
    score = 0
    if _asset_text(asset.get("status")).lower() not in {"", "active", "published"}:
        score -= 8
    if _asset_text(asset.get("image_url") or asset.get("imageUrl")):
        score += 3
    if _asset_has_board_cutout(asset):
        score += 12
    elif _asset_text(asset.get("board_image_url") or asset.get("boardImageUrl")):
        score += 4
    elif _asset_text(asset.get("cutout_status") or asset.get("cutoutStatus")):
        score -= 2
    asset_gender = _asset_gender(asset.get("gender"))
    if target_gender in {"male", "female"}:
        if asset_gender == target_gender:
            score += 6
        elif asset_gender == "unisex":
            score += 2
    archetype = _asset_text(direction.get("archetype")).lower()
    asset_archetypes = _asset_list(asset.get("archetypes"))
    asset_occasions = _asset_list(asset.get("occasions"))
    # Field-mismatch recovery: catalog stores `tags`, scorer wanted `style_tags`
    # (0% live coverage). Fall back to tags so the +3 style term works.
    asset_style_tags = _asset_list(asset.get("style_tags")) or _asset_list(asset.get("tags"))
    asset_allowed_slots = _asset_list(asset.get("allowed_slots"))
    asset_avoid_for = _asset_list(asset.get("avoid_for"))
    if archetype and archetype in asset_archetypes:
        score += 5
    elif archetype and asset_archetypes:
        score -= 2
    occasion_norm = _norm(occasion)
    if occasion_norm and occasion_norm in asset_occasions:
        score += 4
    elif occasion_norm and asset_occasions:
        if not any(term and (term in occasion_norm or occasion_norm in term) for term in asset_occasions):
            score -= 3
    if occasion_norm and any(term and (term in occasion_norm or occasion_norm in term) for term in asset_avoid_for):
        score -= 12
    for tag in asset_style_tags:
        if tag and (
            tag in archetype
            or tag in occasion_norm
            or tag in direction_terms
        ):
            score += 3
    slot = "accessory" if _style_category(direction.get("hero_piece")) == "accessory" else "hero"
    if asset_allowed_slots:
        if slot in asset_allowed_slots or _style_category(direction.get("hero_piece")) in asset_allowed_slots:
            score += 3
        elif "hero" in asset_allowed_slots:
            score += 1
        else:
            score -= 5
    # Catalog stores `colors`; support legacy singular `color` too.
    asset_color_values = _asset_list(asset.get("colors")) or _asset_list(asset.get("color"))
    for color in _safe_list(direction.get("colors") or direction.get("palette"), limit=6):
        if color.lower() in asset_color_values:
            score += 2
    # Color intent matching: prefer the exact hero color, penalise wildly
    # different colors. Colors are inferred from hero text + asset blob so
    # this works even when curated metadata is sparse.
    hero_colors = _extract_simple_colors(direction.get("hero_piece"))
    asset_color_blob = " ".join(
        [
            _asset_text(asset.get("name")),
            " ".join(_asset_list(asset.get("colors")) or _asset_list(asset.get("color"))),
            " ".join(_asset_list(asset.get("tags"))),
            _asset_text(asset.get("subcategory")),
        ]
    )
    asset_colors = _extract_simple_colors(asset_color_blob)
    if hero_colors and asset_colors:
        if hero_colors & asset_colors:
            score += 10
        elif any(_colors_share_group(h, a) for h in hero_colors for a in asset_colors):
            score += 3
        else:
            score -= 6
    elif hero_colors and not asset_colors:
        # Asset color unknown — small penalty so explicitly-colored peers win.
        score -= 1
    for token in re.findall(r"[a-z0-9]+", direction_terms):
        if len(token) > 3 and token in blob:
            score += 1
    score += _hero_asset_match_bonus(asset, direction)
    if any(term in blob for term in ("hat", "cap", "sunglass", "sunglasses")) and any(
        term in occasion_norm for term in ("coffee", "date")
    ):
        score -= 3
    # Canonical-brief authenticity scoring (only when a brief is threaded in;
    # legacy callers pass brief=None → identical behavior).
    if brief:
        # Hard veto: forbidden archetype or forbidden item signal. Return a
        # strongly-negative score so the candidate is dropped (score>0 gate).
        forbidden_arch = {str(a).strip().lower() for a in (brief.get("forbidden_archetypes") or []) if str(a).strip()}
        if forbidden_arch and any(a in forbidden_arch for a in asset_archetypes):
            return -100
        forbidden_items = [str(s).strip().lower() for s in (brief.get("forbidden_item_signals") or []) if str(s).strip()]
        if forbidden_items and any(s in blob for s in forbidden_items):
            return -100
        bf = brief.get("formality")
        if isinstance(bf, (int, float)):
            af, am, ae = _asset_axis_estimates(blob)
            # Formality distance: a high-formality oxford on a low-formality
            # festival is docked hard; an exact match costs nothing.
            score -= int(round(2 * abs(af - float(bf))))
            bm = brief.get("movement")
            if isinstance(bm, (int, float)) and float(bm) >= 7 and am <= 3:
                score -= 4  # occasion needs movement; asset is rigid/formal
            be = brief.get("energy")
            if isinstance(be, (int, float)) and float(be) >= 7 and ae <= 3:
                score -= 3  # occasion is high-energy; asset reads flat/formal
    return score


def _best_style_asset(
    assets: List[Dict[str, Any]],
    *,
    direction: Dict[str, Any],
    occasion: str,
    accessory_only: bool = False,
    target_gender: str = "unknown",
    allow_feminine_accessory: bool = False,
    placement: str | None = None,
    brief: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    matches = _best_style_assets(
        assets,
        direction=direction,
        occasion=occasion,
        accessory_only=accessory_only,
        target_gender=target_gender,
        allow_feminine_accessory=allow_feminine_accessory,
        limit=1,
        placement=placement,
        brief=brief,
    )
    return matches[0] if matches else None


def _best_style_assets(
    assets: List[Dict[str, Any]],
    *,
    direction: Dict[str, Any],
    occasion: str,
    accessory_only: bool = False,
    target_gender: str = "unknown",
    allow_feminine_accessory: bool = False,
    limit: int = 3,
    placement: str | None = None,
    brief: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    # Resolve placement so the central policy can gate + score correctly.
    if placement is None:
        placement = "complete" if accessory_only else "hero"
    target_text = _asset_text(direction.get("hero_piece")) or " ".join(
        _safe_list(direction.get("items") or direction.get("pieces"), limit=1)
    )
    candidates: List[tuple[int, Dict[str, Any]]] = []
    _validate_style_assets([asset for asset in assets if isinstance(asset, dict)])
    reject_log_cap = 5
    rejected_logged = 0
    rejected_total = 0
    occasion_blocked: List[str] = []
    for raw_asset in assets:
        asset = dict(raw_asset)
        asset["_allow_feminine_accessory"] = allow_feminine_accessory
        image_url = _asset_text(asset.get("image_url") or asset.get("imageUrl"))
        if not image_url:
            continue
        if _is_nonfashion_asset(asset):
            logger.info(
                "catalog_asset_excluded_nonfashion asset=%r category=%r",
                _asset_text(asset.get("name")),
                _asset_text(asset.get("category")),
            )
            continue
        if not _asset_allowed_for_gender(asset, target_gender):
            continue
        asset_terms = _asset_category_terms(asset)
        is_accessory = bool(asset_terms.intersection({"accessory", "footwear"}))
        if accessory_only != is_accessory:
            continue
        if not accessory_only and not _hero_asset_allowed(asset, direction, occasion):
            rejected_total += 1
            if rejected_logged < reject_log_cap:
                logger.info(
                    "AHVI_HERO_ASSET_REJECTED hero=%r asset=%r category=%r subcategory=%r",
                    direction.get("hero_piece"),
                    asset.get("name"),
                    asset.get("category"),
                    asset.get("subcategory"),
                )
                rejected_logged += 1
            continue
        block_reason = _occasion_asset_block_reason(
            asset,
            occasion=occasion,
            placement=placement,
            target_text=target_text,
        )
        if block_reason:
            occasion_blocked.append(
                f"{_asset_text(asset.get('name')) or 'unknown'}:{block_reason}"
            )
            continue
        # Central policy gate: blocks accessory/travel/grooming heroes,
        # enforces target-family match (oxford never picks t-shirt), and
        # filters office complete_the_look against beanie/cap/sandal etc.
        if not _asset_allowed_for_context(
            asset,
            occasion=occasion,
            placement=placement,
            target_text=target_text,
        ):
            if not accessory_only:
                rejected_total += 1
                if rejected_logged < reject_log_cap:
                    logger.info(
                        "AHVI_HERO_ASSET_REJECTED hero=%r asset=%r category=%r subcategory=%r reason=policy",
                        direction.get("hero_piece"),
                        asset.get("name"),
                        asset.get("category"),
                        asset.get("subcategory"),
                    )
                    rejected_logged += 1
            continue
        score = _asset_score(asset, direction=direction, occasion=occasion, target_gender=target_gender, brief=brief)
        score += _asset_context_score(
            asset,
            occasion=occasion,
            placement=placement,
            target_text=target_text,
        )
        if score > 0:
            candidates.append((score, asset))
    if not accessory_only and rejected_total > reject_log_cap:
        logger.info(
            "AHVI_HERO_ASSET_REJECTED_SUMMARY hero=%r rejected=%d suppressed=%d",
            _asset_text(direction.get("hero_piece")),
            rejected_total,
            rejected_total - rejected_logged,
        )
    if not candidates:
        logger.info(
            "AHVI_ASSET_GUARD occasion=%s family=%s blocked=%s selected=[]",
            _asset_text(occasion),
            _visual_occasion_family(occasion, target_text) or "general",
            occasion_blocked[:8],
        )
        if not accessory_only and _demo_safe_visuals_enabled():
            logger.info(
                "AHVI_HERO_ASSET_NO_SAFE_MATCH hero=%r",
                _asset_text(direction.get("hero_piece")),
            )
        return []
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    if not accessory_only:
        selected = candidates[: max(1, limit)]
        logger.info(
            "AHVI_ASSET_GUARD occasion=%s family=%s blocked=%s selected=%s",
            _asset_text(occasion),
            _visual_occasion_family(occasion, target_text) or "general",
            occasion_blocked[:8],
            [_asset_text(asset.get("name")) for _, asset in selected],
        )
        for score, asset in selected:
            logger.info(
                "AHVI_HERO_ASSET_SELECTED hero=%r asset=%r category=%r subcategory=%r score=%s",
                _asset_text(direction.get("hero_piece")),
                _asset_text(asset.get("name")),
                _asset_text(asset.get("category")),
                _asset_text(asset.get("subcategory")),
                score,
            )
        return [asset for _, asset in selected]

    selected: List[Dict[str, Any]] = []
    used_groups: set[str] = set()
    used_subcategories: set[str] = set()
    for _, asset in candidates:
        group = _complete_item_group(asset)
        subcategory = _complete_item_subcategory_key(asset)
        if group in used_groups:
            continue
        if subcategory and subcategory in used_subcategories:
            continue
        selected.append(asset)
        used_groups.add(group)
        if subcategory:
            used_subcategories.add(subcategory)
        if len(selected) >= max(1, limit):
            break
    if len(selected) < max(1, limit):
        for _, asset in candidates:
            key = _asset_text(asset.get("asset_id") or asset.get("$id") or asset.get("name"))
            if any(_asset_text(item.get("asset_id") or item.get("$id") or item.get("name")) == key for item in selected):
                continue
            group = _complete_item_group(asset)
            if group in used_groups:
                continue
            selected.append(asset)
            used_groups.add(group)
            if len(selected) >= max(1, limit):
                break
    logger.info(
        "AHVI_ASSET_GUARD occasion=%s family=%s blocked=%s selected=%s",
        _asset_text(occasion),
        _visual_occasion_family(occasion, target_text) or "general",
        occasion_blocked[:8],
        [_asset_text(asset.get("name")) for asset in selected[: max(1, limit)]],
    )
    return selected[: max(1, limit)]


def _accessory_asset_to_complete_item(asset: Dict[str, Any], direction: Dict[str, Any]) -> Dict[str, Any]:
    archetype = _asset_text(direction.get("archetype")) or "this direction"
    item = {
        "name": _asset_text(asset.get("name")) or "Accessory",
        "category": _asset_text(asset.get("category")) or "accessory",
        "asset_id": _asset_text(asset.get("asset_id") or asset.get("$id")),
        "reason": "Completes the look with the right level of finish.",
        "unlocks": _safe_list(asset.get("archetypes"), limit=4) or [archetype],
    }
    return _apply_board_image_fields(item, asset)


def _complete_item_group(item: Dict[str, Any]) -> str:
    blob = " ".join(
        [
            _asset_text(item.get("name")),
            _asset_text(item.get("category")),
            _asset_text(item.get("subcategory")),
            " ".join(_safe_list(item.get("tags"), limit=6)),
        ]
    ).lower()
    if any(term in blob for term in ("belt",)):
        return "belt"
    if any(term in blob for term in ("hat", "cap")):
        return "hat"
    if any(term in blob for term in ("sunglass", "sunglasses", "shade", "shades")):
        return "sunglasses"
    if any(term in blob for term in ("loafer", "sneaker", "shoe", "footwear", "heel", "sandal", "boot")):
        return "footwear"
    if any(term in blob for term in ("bracelet", "watch", "bangle", "necklace", "earring", "jewelry", "jewellery", "ring")):
        return "jewellery_watch"
    if any(term in blob for term in ("bag", "tote", "sling", "messenger", "pouch", "backpack", "crossbody")):
        return "bag"
    if any(term in blob for term in ("overshirt", "jacket", "scarf", "wrap", "layer")):
        return "layer"
    return _norm(item.get("subcategory") or item.get("category") or item.get("name")) or "accessory"


def _complete_item_subcategory_key(item: Dict[str, Any]) -> str:
    subcategory = _norm(item.get("subcategory"))
    if subcategory and subcategory not in {"accessory", "accessories", "fashion", "style"}:
        return subcategory
    return _complete_item_group(item)


def _complete_item_allowed_for_gender(
    item: Dict[str, Any],
    *,
    target_gender: str,
    allow_feminine: bool = False,
) -> bool:
    blob = " ".join(
        [
            _asset_text(item.get("name")),
            _asset_text(item.get("category")),
            _asset_text(item.get("reason")),
            " ".join(_safe_list(item.get("unlocks"), limit=6)),
        ]
    )
    return _style_text_allowed_for_gender(blob, target_gender, allow_feminine=allow_feminine)


def _sanitize_complete_the_look(
    items: List[Any],
    *,
    target_gender: str,
    allow_feminine: bool = False,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    seen_groups: set[str] = set()
    seen_subcategories: set[str] = set()
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        name = _asset_text(raw.get("name") or raw.get("title"))
        if not name:
            continue
        key = _norm(name)
        if key in seen:
            continue
        item = dict(raw)
        item["name"] = name
        group = _complete_item_group(item)
        subcategory = _complete_item_subcategory_key(item)
        if group in seen_groups:
            continue
        if subcategory and subcategory in seen_subcategories:
            continue
        if not _complete_item_allowed_for_gender(
            item,
            target_gender=target_gender,
            allow_feminine=allow_feminine,
        ):
            continue
        seen.add(key)
        seen_groups.add(group)
        if subcategory:
            seen_subcategories.add(subcategory)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _default_complete_the_look(
    direction: Dict[str, Any],
    occasion: str,
    *,
    target_gender: str = "unknown",
    index: int = 0,
) -> List[Dict[str, Any]]:
    archetype = _asset_text(direction.get("archetype")) or "this direction"
    # Last-resort copy only. Curated style_assets are still the source of truth;
    # these labels stay product-like so the UI never shows abstract placeholders.
    component_blob = " ".join(_safe_list(direction.get("items") or direction.get("pieces"), limit=8)).lower()
    occ = _norm(occasion)
    arch = _norm(archetype)
    formula = _direction_formula_signature(direction)

    if "funeral" in occ or "memorial" in occ:
        pools = [["Black Leather Belt", "Formal Black Shoes", "Brushed Steel Watch"]]
    elif target_gender == "female":
        archetype_pools = {
            "quiet luxury": [["Leather Top Handle Bag", "Gold Watch", "Cashmere Wrap"]],
            "modern authority": [["Structured Tote", "Statement Watch", "Pointed Pumps"]],
            "creative agency": [["Sculptural Earrings", "Fashion Sneaker", "Slouchy Tote"]],
            "startup founder": [["Tech Tote", "Minimal Watch", "White Leather Sneaker"]],
        }
        pools = next((items for key, items in archetype_pools.items() if key in arch), [])
        if not pools:
            pools = [
                ["Structured Handbag", "Delicate Earrings", "Block-Heel Sandals"],
                ["Silk Scarf", "Soft Shoulder Bag", "Neutral Flats"],
                ["Gold Watch", "Compact Handbag", "Polished Sandals"],
            ]
    elif target_gender == "male":
        archetype_pools = {
            "quiet luxury": [["Leather-Strap Watch", "Dark Brown Penny Loafers", "Cashmere Scarf"]],
            "modern authority": [["Structured Leather Tote", "Statement Watch", "Black Leather Belt"]],
            "creative agency": [["Fashion Sneaker", "Canvas Tote", "Textured Overshirt"]],
            "startup founder": [["Tech Backpack", "Brushed Steel Watch", "Clean White Leather Sneakers"]],
        }
        pools = next((items for key, items in archetype_pools.items() if key in arch), [])
        if not pools:
            if formula in {("blazer", "denim"), ("blazer", "trouser")}:
                pools = [
                    ["Brushed Steel Watch", "Dark Brown Leather Belt", "Dark Brown Penny Loafers"],
                    ["Leather-Strap Watch", "Cognac Leather Belt", "Structured Messenger Bag"],
                ]
            elif formula == ("knit", "tailored_trouser"):
                pools = [
                    ["Suede Belt", "Clean White Leather Sneakers", "Olive Cotton Overshirt"],
                    ["Brushed Steel Watch", "Dark Brown Penny Loafers", "Canvas Tote"],
                ]
            elif formula in {("tee", "layer"), ("overshirt", "denim")}:
                pools = [
                    ["Casual Field Watch", "Light Cotton Overshirt", "Canvas Sling Bag"],
                    ["Clean White Leather Sneakers", "Tech Backpack", "Dark Brown Leather Belt"],
                ]
            else:
                pools = [
                    ["Brushed Steel Watch", "Dark Brown Leather Belt", "Canvas Tote"],
                    ["Leather-Strap Watch", "Suede Belt", "Clean White Leather Sneakers"],
                    ["Casual Field Watch", "Structured Messenger Bag", "Olive Cotton Overshirt"],
                ]
    else:
        pools = [
            ["Classic Watch", "Canvas Tote", "Dark Brown Leather Belt"],
            ["Clean White Leather Sneakers", "Olive Cotton Overshirt", "Compact Crossbody Bag"],
            ["Brushed Steel Watch", "Dark Brown Penny Loafers", "Soft Layering Scarf"],
        ]
    names = pools[index % len(pools)]
    return [
        {
            "name": name,
            "category": "accessory",
            "image_url": "",
            "reason": f"Finishes {archetype} without crowding the look.",
            "unlocks": [archetype],
        }
        for name in names[:3]
    ]


def _complete_item_blob(item: Dict[str, Any]) -> str:
    return " ".join(
        [
            _asset_text(item.get("name") or item.get("title")),
            _asset_text(item.get("category")),
            _asset_text(item.get("subcategory") or item.get("sub_category")),
            " ".join(_safe_list(item.get("tags"), limit=8)),
        ]
    ).lower()


def _is_work_complete_look_occasion(occasion: Any) -> bool:
    text = _norm(occasion)
    return any(
        term in text
        for term in (
            "office",
            "client meeting",
            "client",
            "work",
            "business",
            "interview",
            "presentation",
            "conference",
            "formal",
            "funeral",
            "religious",
            "wedding",
        )
    )


def _work_complete_item_allowed(item: Dict[str, Any]) -> bool:
    blob = _complete_item_blob(item)
    reject_terms = (
        "beanie",
        "cap",
        "hat",
        "sunglass",
        "sunglasses",
        "sneaker",
        "sneakers",
        "sandal",
        "sandals",
        "slide",
        "slides",
        "flip flop",
        "flipflop",
        "slipper",
        "slippers",
        "travel",
        "grooming",
        "skincare",
        "electronics",
        "hoodie",
        "t-shirt",
        "tshirt",
    )
    if any(term in blob for term in reject_terms):
        return False
    allowed_terms = (
        "belt",
        "watch",
        "loafer",
        "loafers",
        "derby",
        "formal_shoe",
        "oxford_shoe",
        "monk strap",
        "monkstrap",
        "monk_strap",
        "laptop bag",
        "laptop_bag",
        "briefcase",
        "messenger bag",
        "messenger_bag",
        "messenger",
    )
    if any(term in blob for term in allowed_terms):
        return True
    if "formal" in blob and any(term in blob for term in ("shoe", "shoes", "footwear")):
        return True
    if "oxford" in blob and any(term in blob for term in ("shoe", "shoes", "footwear")):
        return True
    return False


_FESTIVE_SAFE_SUPPORT_TERMS: tuple[str, ...] = (
    "mojari",
    "jutti",
    "kolhapuri",
    "ethnic sandal",
    "formal ethnic footwear",
    "ethnic footwear",
    "sherwani",
    "bandhgala",
    "nehru jacket",
    "kurta set",
    "stole",
    "dupatta",
    "brooch",
    "pocket square",
    "festive watch",
    "subtle ring",
)
_FESTIVE_UNSAFE_SUPPORT_TERMS: tuple[str, ...] = (
    "cap",
    "beanie",
    "bucket hat",
    "baseball cap",
    "snapback",
    "sneaker",
    "trainer",
    "running",
    "runnig",
    "nike",
    "athletic",
    "boot",
    "duffle",
    "duffel",
    "backpack",
    "laptop bag",
    "messenger bag",
    "wallet",
    "sunglass",
    "aviator",
    "overshirt",
    "relaxed shirt",
    "oxford",
    "polo",
    "hoodie",
    "sweatshirt",
    "denim",
    "jeans",
    "flip flop",
    "slipper",
    "suede blue shoe",
    "casual shoe",
)


def _safe_visual_support_assets(
    items: Any,
    occasion_family: str,
    query: Any = "",
    direction: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    rows = [dict(item) for item in (items or []) if isinstance(item, dict)]
    if occasion_family != "indian_festive":
        occasion_text = _asset_text(query)
        target_text = ""
        if isinstance(direction, dict):
            target_text = _asset_text(
                direction.get("hero_piece")
                or direction.get("direction_name")
                or direction.get("title")
            )
        kept = [
            item
            for item in rows
            if _asset_allowed_for_context(
                item,
                occasion=occasion_text,
                placement="complete",
                target_text=target_text,
            )
        ]
        dropped = [
            _asset_text(item.get("name") or item.get("title")) or "unknown"
            for item in rows
            if item not in kept
        ]
        if dropped:
            logger.info(
                "AHVI_SUPPORT_ASSET_SUPPRESSED family=%s dropped=%s kept=%s",
                occasion_family or "general",
                dropped[:12],
                [_asset_text(item.get("name") or item.get("title")) for item in kept[:12]],
            )
        return kept
    kept: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for item in rows:
        name = _asset_text(item.get("name") or item.get("title")) or "unknown"
        blob = _asset_policy_blob(item)
        cap_like = bool(
            re.search(r"\bca\b", blob)
            and _FAMILY_GROUP.get(_asset_family(item)) == "accessory"
        )
        explicitly_unsafe = cap_like or any(term in blob for term in _FESTIVE_UNSAFE_SUPPORT_TERMS)
        clearly_safe = any(term in blob for term in _FESTIVE_SAFE_SUPPORT_TERMS)
        if "suede blue shoe" in blob and not any(
            term in blob for term in ("ethnic", "formal loafer", "mojari", "jutti")
        ):
            explicitly_unsafe = True
        if clearly_safe and not explicitly_unsafe:
            kept.append(item)
        else:
            dropped.append(name)
    logger.info(
        "AHVI_SUPPORT_ASSET_SUPPRESSED family=%s dropped=%s kept=%s",
        occasion_family,
        dropped[:12],
        [_asset_text(item.get("name") or item.get("title")) for item in kept[:12]],
    )
    return kept


def _filter_complete_the_look_for_occasion(
    complete: List[Dict[str, Any]],
    occasion: Any,
    target_gender: str = "unknown",
) -> List[Dict[str, Any]]:
    if not complete:
        return complete or []
    kept: List[Dict[str, Any]] = []
    removed: List[str] = []
    for item in complete:
        if not isinstance(item, dict):
            continue
        block_reason = _occasion_asset_block_reason(
            item,
            occasion=_asset_text(occasion),
            placement="complete",
            target_text=_asset_text(item.get("name") or item.get("title")),
        )
        if block_reason:
            removed.append(_asset_text(item.get("name") or item.get("title")) or "unknown")
            continue
        if (
            _visual_occasion_family(occasion) != "indian_festive"
            and _is_work_complete_look_occasion(occasion)
            and not _work_complete_item_allowed(item)
        ):
            removed.append(_asset_text(item.get("name") or item.get("title")) or "unknown")
            continue
        kept.append(item)
    if removed or len(kept) != len(complete):
        logger.info(
            "AHVI_COMPLETE_LOOK_FILTERED occasion=%s before=%d after=%d removed=%s",
            _asset_text(occasion),
            len(complete),
            len(kept),
            ",".join(removed[:8]),
        )
    return kept


# ── Itemized board_items contract for the frontend 85 flat-lay board ─────────
# Frontend (commit 0057706) renders AhviOutfitBoardCard -> EditorialBoardCanvas
# only when a direction carries real, role-tagged, image-bearing pieces. One
# hero image is not enough. Build board_items alongside the legacy fields.
_BOARD_ALLOWED_ROLES = {"top", "bottom", "footwear", "dress", "outerwear", "accessory"}


def _asset_has_board_cutout(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return bool(
        _asset_text(item.get("board_image_url") or item.get("boardImageUrl"))
        and _asset_text(item.get("cutout_status") or item.get("cutoutStatus")).lower() == "ready"
    )


def _board_image_resolution(item: Any) -> Dict[str, str]:
    """Resolve the image used by the 85 board without losing catalog provenance.

    Runtime priority for style assets:
    board_image_url -> transparent_image_url -> cutout_url -> rmbg_url ->
    image_url -> catalog_image_url. Wardrobe-specific normalized/masked URLs are
    kept before raw image_url for backward compatibility.
    """
    if not isinstance(item, dict):
        return {
            "image_url": "",
            "board_image_url": "",
            "catalog_image_url": "",
            "board_status": "catalog_fallback",
            "used": "catalog_fallback",
        }

    board_url = _asset_text(item.get("board_image_url") or item.get("boardImageUrl"))
    transparent_url = _asset_text(
        item.get("transparent_image_url")
        or item.get("transparentImageUrl")
        or item.get("transparent_url")
        or item.get("transparentUrl")
    )
    cutout_url = _asset_text(item.get("cutout_url") or item.get("cutoutUrl"))
    rmbg_url = _asset_text(item.get("rmbg_url") or item.get("rmbgUrl"))
    normalized_url = _asset_text(item.get("normalized_url") or item.get("normalizedUrl"))
    masked_url = _asset_text(item.get("masked_url") or item.get("maskedUrl"))
    raw_url = _asset_text(item.get("image_url") or item.get("imageUrl"))
    catalog_url = _asset_text(item.get("catalog_image_url") or item.get("catalogImageUrl"))

    ordered = (
        ("board_image_url", board_url),
        ("transparent_image_url", transparent_url),
        ("cutout_url", cutout_url),
        ("rmbg_url", rmbg_url),
        ("normalized_url", normalized_url),
        ("masked_url", masked_url),
        ("image_url", raw_url),
        ("catalog_image_url", catalog_url),
    )
    used = ""
    selected = ""
    for key, value in ordered:
        if value:
            used = key
            selected = value
            break

    if board_url:
        status = "cutout_ready"
        used_label = "board_image_url"
    else:
        status = "catalog_fallback" if selected else "missing_image"
        used_label = "catalog_fallback" if selected else "missing"

    return {
        "image_url": selected,
        "board_image_url": board_url,
        "catalog_image_url": catalog_url or raw_url,
        "board_status": status,
        "used": used_label,
        "source_field": used,
    }


def _board_image_url(item: Any) -> str:
    return _board_image_resolution(item).get("image_url", "")


def _apply_board_image_fields(target: Dict[str, Any], asset: Dict[str, Any]) -> Dict[str, Any]:
    resolved = _board_image_resolution(asset)
    if resolved.get("image_url"):
        target["image_url"] = resolved["image_url"]
        target["imageUrl"] = resolved["image_url"]
    if resolved.get("board_image_url"):
        target["board_image_url"] = resolved["board_image_url"]
        target["boardImageUrl"] = resolved["board_image_url"]
    if resolved.get("catalog_image_url"):
        target["catalog_image_url"] = resolved["catalog_image_url"]
        target["catalogImageUrl"] = resolved["catalog_image_url"]
    if _asset_text(asset.get("cutout_status") or asset.get("cutoutStatus")):
        target["cutout_status"] = _asset_text(asset.get("cutout_status") or asset.get("cutoutStatus")).lower()
    if _asset_text(asset.get("board_r2_key") or asset.get("boardR2Key")):
        target["board_r2_key"] = _asset_text(asset.get("board_r2_key") or asset.get("boardR2Key"))
    target["board_image_status"] = resolved.get("board_status") or "catalog_fallback"
    return target


def _board_item_role(name: Any, category: Any = "") -> str:
    blob = f"{_asset_text(name)} {_asset_text(category)}".strip()
    if not blob:
        return ""
    # Trust an explicit category field first. Catalog assets carry a clean
    # category ("bottom"/"top"/...), but their names are concatenated
    # ("Mens Greytrouser"), so the token/name parser below misses them and the
    # board lost its bottom. The category map is authoritative when present.
    _cat_role = {
        "bottom": "bottom", "bottoms": "bottom",
        "top": "top", "tops": "top",
        "footwear": "footwear", "shoe": "footwear", "shoes": "footwear",
        "outerwear": "outerwear",
        "dress": "dress", "dresses": "dress",
        "accessory": "accessory", "accessories": "accessory",
    }.get(_norm(_asset_text(category)))
    if _cat_role:
        return _cat_role
    tokens = _style_tokens(_norm(blob))
    if tokens.intersection(
        {"dress", "gown", "saree", "sari", "lehenga", "lehnga", "anarkali", "jumpsuit", "frock"}
    ):
        return "dress"
    if tokens.intersection(
        {
            "watch", "belt", "bag", "tote", "clutch", "backpack", "sling",
            "sunglasses", "sunglass", "necklace", "bracelet", "earring",
            "earrings", "ring", "scarf", "tie", "cap", "hat", "jewellery", "jewelry",
        }
    ):
        return "accessory"
    slot = _hero_expected_slot(blob)
    if slot == "blazer":
        return "outerwear"
    if slot in {"top", "bottom", "footwear", "outerwear"}:
        return slot
    return ""


def _build_board_items(
    direction: Dict[str, Any], *, wardrobe_intent: bool
) -> List[Dict[str, Any]]:
    """Itemized, role-tagged, image-bearing board pieces. No fake data: items
    without an image_url are excluded; one hero image is never reused as
    top/bottom/footwear. Wardrobe intent yields owned-only items."""
    items: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    def add(name: str, source_item: Dict[str, Any], role: str, source: str, owned: bool) -> None:
        resolved = _board_image_resolution(source_item)
        url = resolved.get("image_url", "")
        if not url or role not in _BOARD_ALLOWED_ROLES or url in seen_urls:
            return
        seen_urls.add(url)
        item = {
            "name": name,
            "role": role,
            "image_url": url,
            "source": source,
            "owned": owned,
            "board_status": resolved.get("board_status") or "catalog_fallback",
            "image_source": resolved.get("used") or "catalog_fallback",
        }
        if resolved.get("board_image_url"):
            item["board_image_url"] = resolved["board_image_url"]
        if resolved.get("catalog_image_url"):
            item["catalog_image_url"] = resolved["catalog_image_url"]
        logger.info(
            "ahvi.board_image_resolver role=%s used=%s",
            role,
            item["image_source"],
        )
        items.append(item)

    owned_rows = direction.get("owned_items")
    for it in owned_rows if isinstance(owned_rows, list) else []:
        if not isinstance(it, dict):
            continue
        name = _asset_text(it.get("name") or it.get("title") or it.get("label"))
        add(
            name,
            it,
            _board_item_role(name, it.get("category")),
            "wardrobe",
            True,
        )

    # Wardrobe intent: owned items only — never blend in generic assets.
    if wardrobe_intent:
        return items

    hero_name = _asset_text(direction.get("hero_piece") or direction.get("heroPiece"))
    add(hero_name, direction, _board_item_role(hero_name), "asset", False)
    complete = direction.get("complete_the_look")
    for it in complete if isinstance(complete, list) else []:
        if not isinstance(it, dict):
            continue
        name = _asset_text(it.get("name") or it.get("title") or it.get("label"))
        owned = bool(it.get("owned") or it.get("wardrobeItemId") or it.get("wardrobe_item_id"))
        add(
            name,
            it,
            _board_item_role(name, it.get("category")),
            "wardrobe" if owned else "asset",
            owned,
        )
    return items


def _board_items_viable(items: List[Dict[str, Any]]) -> bool:
    roles = {i.get("role") for i in items}
    classic = {"top", "bottom", "footwear"}.issubset(roles)
    dress = {"dress", "footwear"}.issubset(roles)
    known = sum(1 for i in items if i.get("role") in _BOARD_ALLOWED_ROLES)
    return classic or dress or known >= 3


# ── Visual board sanitizer ──────────────────────────────────────────────────
# Single entry point that repairs a RAW wardrobe dump into a viable, slot-based
# visual board. The frontend must never receive an un-deduped wardrobe list
# (two pants, two belts, duplicate images). Reuses the production board-role
# detector (_board_item_role) and family detector (_wardrobe_item_family).
_BOARD_ACCESSORY_CAP = 2  # belt/watch/bag/sunglasses combined, max 2 on a board.
_BAG_FAMILIES: set[str] = {
    "bag",
    "laptop_bag",
    "messenger_bag",
    "backpack",
    "duffle_bag",
    "briefcase",
}


def _count_summary(values: Any) -> str:
    """Compact "key:n,key:n" tally for aggregate logs (sorted, empties dropped)."""
    counts: Dict[str, int] = {}
    for v in values:
        key = str(v or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return ",".join(f"{k}:{counts[k]}" for k in sorted(counts)) or "-"


def sanitize_board_items_for_visual_board(
    items: Any,
    occasion: str = "",
    style_direction: str = "",
) -> Dict[str, Any] | None:
    """Repair a raw wardrobe list into a viable, slot-based visual board.

    - normalises each item into a board role/family
    - removes duplicate item_id / image_url / normalized name
    - enforces family caps: max 1 bottom, 1 belt, 1 footwear, 1 bag,
      max 2 accessories total
    - applies the dress/full-body exclusion (a dress drops a separate
      top + bottom)
    - validates outfit viability

    Returns structured visual slots::

        {"primary", "bottom", "footwear", "outerwear", "accessories", "items"}

    or ``None`` when no viable outfit can be assembled (e.g. an
    accessory-only or footwear-only dump). Callers MUST treat ``None`` as
    "do not return a visual board" and fall back to text/cards.
    """
    rows = _wardrobe_normalised(items)
    input_n = len(rows)

    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    seen_names: set[str] = set()

    tops: List[Dict[str, Any]] = []
    bottoms: List[Dict[str, Any]] = []
    dresses: List[Dict[str, Any]] = []
    outer: List[Dict[str, Any]] = []
    footwear: List[Dict[str, Any]] = []
    accessories: List[Dict[str, Any]] = []
    belt_used = False
    bag_used = False
    # Aggregate drop counters (logged once, never per-item).
    dropped_duplicate = 0
    dropped_family_cap = 0
    dropped_incompatible = 0

    for row in rows:
        if _wardrobe_item_blocked(row):
            dropped_incompatible += 1
            continue
        name = _asset_text(row.get("name"))
        if not name:
            dropped_incompatible += 1
            continue
        role = _board_item_role(name, row.get("category"))
        if role not in _BOARD_ALLOWED_ROLES:
            dropped_incompatible += 1
            continue
        ident = _asset_text(row.get("id")) or name.lower()
        norm_name = name.strip().lower()
        image = _asset_text(row.get("image_url"))
        if ident in seen_ids or norm_name in seen_names:
            dropped_duplicate += 1
            continue
        if image and image in seen_images:
            dropped_duplicate += 1
            continue
        family = _wardrobe_item_family(row)
        # Hard per-family caps that the accessory bucket alone can't express.
        if family == "belt":
            if belt_used:
                dropped_family_cap += 1
                continue
            belt_used = True
        elif family in _BAG_FAMILIES:
            if bag_used:
                dropped_family_cap += 1
                continue
            bag_used = True
        seen_ids.add(ident)
        seen_names.add(norm_name)
        if image:
            seen_images.add(image)
        slot_item = {
            "id": row.get("id", ""),
            "name": name,
            "role": role,
            "family": family,
            "category": row.get("category", ""),
            "image_url": image,
            "owned": True,
        }
        if role == "dress":
            dresses.append(slot_item)
        elif role == "top":
            tops.append(slot_item)
        elif role == "bottom":
            bottoms.append(slot_item)
        elif role == "outerwear":
            outer.append(slot_item)
        elif role == "footwear":
            footwear.append(slot_item)
        elif role == "accessory":
            accessories.append(slot_item)

    is_dress = bool(dresses)
    if is_dress:
        # Full-body item excludes a separate top + bottom.
        primary = dresses[0]
        bottom = None
        # Extra dresses -> family cap; discarded top/bottom -> incompatible.
        dropped_family_cap += max(0, len(dresses) - 1)
        dropped_incompatible += len(tops) + len(bottoms)
    else:
        primary = tops[0] if tops else None
        bottom = bottoms[0] if bottoms else None
        dropped_family_cap += max(0, len(tops) - 1) + max(0, len(bottoms) - 1)

    kept_accessories = accessories[:_BOARD_ACCESSORY_CAP]
    dropped_family_cap += (
        max(0, len(footwear) - 1)
        + max(0, len(outer) - 1)
        + max(0, len(accessories) - len(kept_accessories))
    )

    slots: Dict[str, Any] = {
        "primary": primary,
        "bottom": bottom,
        "footwear": footwear[0] if footwear else None,
        "outerwear": outer[0] if outer else None,
        "accessories": kept_accessories,
    }

    ordered: List[Dict[str, Any]] = []
    for key in ("primary", "bottom", "outerwear", "footwear"):
        if slots[key]:
            ordered.append(slots[key])
    ordered.extend(slots["accessories"])
    slots["items"] = ordered

    roles_summary = _count_summary(i.get("role") for i in ordered)
    families_summary = _count_summary(i.get("family") for i in ordered)

    # Viability: need a primary (top or dress) + footwear, and for a top-led
    # board a bottom too. Accessory-only / footwear-only dumps are rejected.
    viable = (
        primary is not None
        and slots["footwear"] is not None
        and (is_dress or bottom is not None)
    )
    if not viable:
        logger.info(
            "AHVI_BOARD_SANITIZER_RESULT status=rejected reason=no_viable_board "
            "occasion=%r input=%d roles=%s families=%s",
            occasion,
            input_n,
            roles_summary,
            families_summary,
        )
        return None
    logger.info(
        "AHVI_BOARD_SANITIZER_RESULT status=ok occasion=%r input=%d output=%d "
        "roles=%s families=%s accessories=%d dropped_duplicate=%d "
        "dropped_family_cap=%d dropped_incompatible=%d",
        occasion,
        input_n,
        len(ordered),
        roles_summary,
        families_summary,
        len(slots["accessories"]),
        dropped_duplicate,
        dropped_family_cap,
        dropped_incompatible,
    )
    return slots


def _enrich_visual_directions_with_assets(
    visual_directions: List[Dict[str, Any]],
    *,
    occasion: str | None,
    target_gender: str = "unknown",
    allow_feminine_accessory: bool = False,
    brief: Dict[str, Any] | None = None,
    wardrobe_intent: bool = False,
) -> List[Dict[str, Any]]:
    if not visual_directions:
        return visual_directions
    assets = _style_asset_rows()
    occasion_text = _asset_text(occasion)
    enriched: List[Dict[str, Any]] = []
    for idx, direction in enumerate(visual_directions):
        out = _sanitize_direction_for_gender(
            dict(direction),
            target_gender=target_gender,
            allow_feminine=allow_feminine_accessory,
        )
        image_url = _asset_text(out.get("image_url") or out.get("imageUrl"))
        if image_url and assets:
            attached_asset_id = _asset_text(out.get("asset_id"))
            attached_asset = next(
                (
                    asset
                    for asset in assets
                    if (
                        attached_asset_id
                        and _asset_text(asset.get("asset_id") or asset.get("$id"))
                        == attached_asset_id
                    )
                    or _asset_text(asset.get("image_url") or asset.get("imageUrl"))
                    == image_url
                ),
                None,
            )
            if attached_asset and (
                _is_nonfashion_asset(attached_asset)
                or not _asset_allowed_for_context(
                    attached_asset,
                    occasion=occasion_text,
                    placement="hero",
                    target_text=_asset_text(out.get("hero_piece")),
                )
                or not _hero_asset_allowed(attached_asset, out, occasion_text)
            ):
                logger.info(
                    "AHVI_ASSET_GUARD occasion=%s family=%s blocked=%s selected=[]",
                    occasion_text,
                    _visual_occasion_family(occasion_text, out.get("hero_piece")) or "general",
                    [_asset_text(attached_asset.get("name"))],
                )
                out.pop("image_url", None)
                out.pop("imageUrl", None)
                out.pop("asset_id", None)
                image_url = ""
        if not image_url and assets:
            asset = _best_style_asset(
                assets,
                direction=out,
                occasion=occasion_text,
                target_gender=target_gender,
                allow_feminine_accessory=allow_feminine_accessory,
                brief=brief,
            )
            if asset:
                _apply_board_image_fields(out, asset)
                out["asset_id"] = _asset_text(asset.get("asset_id") or asset.get("$id"))
        complete = out.get("complete_the_look")
        if not isinstance(complete, list) or not complete:
            complete = []
        complete = _sanitize_complete_the_look(
            complete,
            target_gender=target_gender,
            allow_feminine=allow_feminine_accessory,
            limit=3,
        )
        if assets:
            accessory_assets = _best_style_assets(
                assets,
                direction=out,
                occasion=occasion_text,
                accessory_only=True,
                target_gender=target_gender,
                allow_feminine_accessory=allow_feminine_accessory,
                limit=3,
                brief=brief,
            )
            if accessory_assets:
                # Keep essential outfit pieces (bottom/footwear/top/outerwear)
                # AHEAD of accessories so the limit=3 cap can't evict the pant.
                # Previously accessories were prepended, so 3 accessories (bag +
                # hat + belt) crowded out the bottom and shipped a pant-less board.
                existing = [item for item in complete if isinstance(item, dict)]

                def _is_essential_piece(item: Dict[str, Any]) -> bool:
                    return _board_item_role(
                        _asset_text(item.get("name")), item.get("category")
                    ) in {"top", "bottom", "outerwear", "dress", "footwear"}

                essential_items = [it for it in existing if _is_essential_piece(it)]
                extra_items = [it for it in existing if not _is_essential_piece(it)]
                accessory_items = [
                    _accessory_asset_to_complete_item(asset, out)
                    for asset in accessory_assets
                ]
                complete = _sanitize_complete_the_look(
                    [*essential_items, *accessory_items, *extra_items],
                    target_gender=target_gender,
                    allow_feminine=allow_feminine_accessory,
                    limit=3,
                )
        if not complete:
            complete = _default_complete_the_look(
                out,
                occasion_text,
                target_gender=target_gender,
                index=idx,
            )
        complete = _filter_complete_the_look_for_occasion(
            complete,
            occasion_text,
            target_gender,
        )
        complete = _safe_visual_support_assets(
            complete,
            _visual_occasion_family(occasion_text, out.get("hero_piece")),
            occasion_text,
            out,
        )
        # Guarantee a bottom on top-led catalog boards. If neither the hero nor
        # the completed pieces carry a bottom (e.g. accessories filled the slots),
        # retrieve the best bottom asset so the board never ships pant-less.
        if not wardrobe_intent and assets:
            hero_role = _board_item_role(_asset_text(out.get("hero_piece")))
            has_bottom = hero_role in {"bottom", "dress"} or any(
                _board_item_role(_asset_text(it.get("name")), it.get("category"))
                in {"bottom", "dress"}
                for it in complete
                if isinstance(it, dict)
            )
            if not has_bottom:
                # Score against a BOTTOM hero so the retrieval surfaces bottoms
                # (scoring against the real top-hero only returns tops).
                bottom_dir = dict(out)
                bottom_dir["hero_piece"] = "tailored trousers"
                bottom_dir.pop("complete_the_look", None)
                bottom_candidates = _best_style_assets(
                    assets,
                    direction=bottom_dir,
                    occasion=occasion_text,
                    accessory_only=False,
                    target_gender=target_gender,
                    allow_feminine_accessory=allow_feminine_accessory,
                    limit=25,
                    brief=brief,
                )
                bottom_asset = next(
                    (
                        a
                        for a in bottom_candidates
                        if _board_item_role(_asset_text(a.get("name")), a.get("category"))
                        == "bottom"
                    ),
                    None,
                )
                if bottom_asset:
                    bottom_item = _accessory_asset_to_complete_item(bottom_asset, out)
                    # Bottom leads; keep essentials, then accessories — cap at 3.
                    complete = [bottom_item] + [
                        it for it in complete if isinstance(it, dict)
                    ]
                    logger.info(
                        "AHVI_BOARD_BOTTOM_BACKFILL occasion=%s asset=%s",
                        occasion_text,
                        _asset_text(bottom_asset.get("name")),
                    )
        out["complete_the_look"] = complete[:3]
        out = _validate_visual_direction_consistency(
            out,
            occasion=occasion_text,
            target_gender=target_gender,
            allow_feminine=allow_feminine_accessory,
        )
        # Itemized board contract for the 85 board (additive — legacy fields kept).
        board_items = _build_board_items(out, wardrobe_intent=wardrobe_intent)
        viable = _board_items_viable(board_items)
        if wardrobe_intent and not viable:
            out["board_status"] = "insufficient_wardrobe_items"
            out["fallback_reason"] = (
                "Need top, bottom and footwear from wardrobe to build this board."
            )
        else:
            has_catalog_fallback = any(
                item.get("board_status") == "catalog_fallback"
                for item in board_items
                if item.get("source") == "asset"
            )
            if has_catalog_fallback:
                out["board_status"] = "catalog_fallback"
            else:
                out["board_status"] = "viable" if viable else "partial"
        out["board_items"] = board_items
        out["boardItems"] = board_items
        enriched.append(out)
    total_board_items = sum(len(item.get("board_items") or []) for item in enriched)
    cutout_ready_count = sum(
        1
        for item in enriched
        for bi in (item.get("board_items") or [])
        if bi.get("board_image_url") and bi.get("board_status") != "catalog_fallback"
    )
    roles_present = sorted(
        {
            bi.get("role")
            for item in enriched
            for bi in (item.get("board_items") or [])
            if bi.get("role")
        }
    )
    status = (
        "viable"
        if any(_board_items_viable(item.get("board_items") or []) for item in enriched)
        else "insufficient"
    )
    logger.info(
        "AHVI_VISUAL_ASSETS_ENRICHED directions=%d assets=%d with_images=%d gender=%s",
        len(enriched),
        len(assets),
        sum(1 for item in enriched if item.get("image_url")),
        target_gender,
    )
    logger.info(
        "ahvi.style_board_items_contract direction_count=%d board_items_count=%d "
        "roles_present=%s cutout_ready_count=%d wardrobe_intent=%s status=%s",
        len(enriched),
        total_board_items,
        ",".join(roles_present) or "none",
        cutout_ready_count,
        wardrobe_intent,
        status,
    )
    return enriched


def _occasion_category(query: str) -> tuple[str | None, str | None, str | None, str | None]:
    q = _norm(query)

    sensitive = (
        "funeral",
        "memorial",
        "condolence",
        "wake",
        "church",
        "temple",
        "mosque",
        "prayer",
        "religious",
        "ceremony",
        "traditional",
    )
    work = (
        "office",
        "work",
        "client",
        "pitch",
        "meeting",
        "interview",
        "conference",
        "presentation",
        "business",
    )
    social = (
        "date",
        "coffee",
        "dinner",
        "drinks",
        "party",
        "wedding",
        "brunch",
        "lunch",
        "birthday",
        "reception",
        "festival",
    )

    has_sensitive = _has_any(q, sensitive)
    has_work = _has_any(q, work)
    has_social = _has_any(q, social)

    if has_work and has_social:
        return ("hybrid_occasion", "polished", "smart-to-social", "hybrid occasion")
    if has_sensitive:
        return ("sensitive_occasion", "respectful", "polished", "sensitive occasion")
    if has_work:
        return ("work_occasion", "competent", "smart", "work occasion")
    if has_social:
        return ("social_occasion", "warm", "smart casual", "social occasion")
    if _has_any(q, ("beach", "travel", "airport", "vacation", "trip")):
        return ("travel_occasion", "practical", "casual", "travel occasion")
    return ("custom_occasion", "considered", "context-aware", "custom occasion")


def _fallback_cta(query: str) -> List[Dict[str, str]]:
    return [
        {"label": "Use my wardrobe", "value": f"Use my wardrobe for: {query}"},
        {"label": "Show visual inspiration", "value": f"Show visual inspiration for: {query}"},
        {"label": "Find missing pieces", "value": f"Show shopping ideas for: {query}"},
    ]


def _reason_for_mode(mode: str, category: str | None) -> str:
    if mode == WARDROBE_STYLE:
        return "wardrobe_request"
    if mode == VISUAL_INSPIRATION:
        return "visual_inspiration_request"
    if mode == SHOPPING_ASSIST:
        return "shopping_request"
    if mode == COLOR_BODY_ADVICE:
        return "body_color_advice"
    if mode == STYLE_EDUCATION:
        return "style_education"
    return category or "style_advice"


def _fallback_goal(mode: str, category: str | None) -> str:
    if mode == VISUAL_INSPIRATION:
        return "Turn the occasion into three clear visual directions."
    if mode == COLOR_BODY_ADVICE:
        return "Find colors and proportions that flatter without feeling overworked."
    if mode == STYLE_EDUCATION:
        return "Explain the style idea in a practical, usable way."
    if mode == SHOPPING_ASSIST:
        return "Identify the missing piece without pushing a full new outfit."
    if category == "sensitive_occasion":
        return "Look respectful, calm, and quietly put together."
    if category == "hybrid_occasion":
        return "Look professional first, with an easy shift into a social setting."
    if category == "work_occasion":
        return "Look credible first, then add just enough personality."
    if category == "social_occasion":
        return "Look approachable, confident, and comfortable in motion."
    return "Create a context-aware outfit direction that feels natural and intentional."


def _fallback_atmosphere(category: str | None) -> str:
    return {
        "sensitive_occasion": "respectful and understated",
        "hybrid_occasion": "professional first, social second",
        "work_occasion": "polished and precise",
        "social_occasion": "easy, warm, and lightly styled",
        "travel_occasion": "practical, relaxed, and prepared",
    }.get(category or "", "considered and context-aware")


def _fallback_impression(category: str | None) -> str:
    return {
        "sensitive_occasion": "understated and considerate",
        "hybrid_occasion": "competent but not stiff",
        "work_occasion": "credible and composed",
        "social_occasion": "intentional but relaxed",
        "travel_occasion": "easy and prepared",
    }.get(category or "", "considered and self-assured")


def _fallback_missing_piece(query: str, category: str | None) -> str:
    if category == "sensitive_occasion":
        return (
            "A pair of clean, closed leather shoes would anchor this and quietly "
            "carry across other formal or respectful settings."
        )
    if category in {"work_occasion", "hybrid_occasion"}:
        return (
            "A well-cut neutral blazer would do the most work here — it sharpens "
            "the look for the room and still relaxes for drinks afterward."
        )
    if category == "social_occasion":
        return (
            "A brown suede loafer would elevate this and earn its place across "
            "coffee dates, weekend dinners, and smart-casual office days."
        )
    if category == "travel_occasion":
        return (
            "One light structured layer would lift the outfit from purely "
            "practical to put-together without adding bulk."
        )
    return (
        "One refined pair of shoes is the piece that would shift this from fine "
        "to intentional, and it would carry across several other settings too."
    )


def _fallback_emotion(category: str | None) -> str:
    if category in {"work_occasion", "hybrid_occasion"}:
        return "professional"
    if category == "social_occasion":
        return "social"
    if category == "sensitive_occasion":
        return "vulnerable"
    return "neutral"


def _fallback_advice(query: str, mode: str, category: str | None) -> str:
    context = query.strip() or "this"
    if mode == VISUAL_INSPIRATION:
        return "I would frame this as three different directions so you can pick the mood before choosing pieces."
    if mode == COLOR_BODY_ADVICE:
        return (
            "For warm olive skin, stay close to colors with warmth and depth: olive, tobacco, cream, rust, "
            "warm navy, and deep teal. Very icy pastels or flat greys can work, but they need a warmer layer "
            "near the face."
        )
    if mode == STYLE_EDUCATION:
        return (
            "Think of the dress code as context, not a costume. Match the room first, then use fit, texture, "
            "and one deliberate detail to make it feel like you."
        )
    if mode == SHOPPING_ASSIST:
        return (
            "I would solve the missing piece first, not rebuild the whole outfit. Start with the shoe or layer "
            "that changes the formality, then choose a color that already works with what you own."
        )
    if category == "sensitive_occasion":
        return (
            "For this, keep the outfit quiet, respectful, and easy to sit or stand in for a while. Choose muted "
            "or deeper colors, clean lines, closed footwear, and minimal accessories so the clothes support the "
            "moment instead of asking for attention."
        )
    if category == "hybrid_occasion":
        return (
            "Dress for the professional room first, then build in one small switch for later. A sharp base, clean "
            "shoes, and a layer you can remove or soften after work will carry you from presentation to drinks "
            "without looking like two different outfits."
        )
    if category == "work_occasion":
        return (
            "Lead with polish: a sharper base, clean footwear, and one controlled detail. If the day moves into "
            "something social, use a layer or accessory you can soften after work."
        )
    if category == "social_occasion":
        return (
            "Keep it relaxed but intentional: one clean hero piece, comfortable footwear, and a palette that feels "
            "warm rather than loud. You want to look considered, not over-planned."
        )
    return (
        f"For {context}, read the setting first: formality, venue, culture, weather, and how much movement the "
        "day needs. Then build a clean base, choose footwear that fits the room, and keep accessories restrained."
    )


def _fallback_visual_directions(mode: str, category: str | None) -> List[Dict[str, Any]]:
    if category == "sensitive_occasion":
        return [
            {
                "title": "Quiet Formal",
                "description": "A dark, clean base with closed shoes and almost no shine.",
                "palette": ["black", "charcoal", "deep navy"],
                "pieces": ["plain shirt or kurta", "tailored trousers", "closed shoes"],
                "style_note": "Respectful, restrained, and appropriate for a serious setting.",
            },
            {
                "title": "Soft Traditional",
                "description": "Modest coverage, muted color, and a gentle fabric texture.",
                "palette": ["deep grey", "ink blue", "soft white"],
                "pieces": ["modest top", "straight trousers", "simple layer"],
                "style_note": "Keeps cultural sensitivity and comfort in balance.",
            },
            {
                "title": "Minimal Polished",
                "description": "A tonal outfit with structure but no loud details.",
                "palette": ["navy", "stone", "black"],
                "pieces": ["structured shirt", "clean bottom", "low-profile footwear"],
                "style_note": "Polished without looking dressed for attention.",
            },
        ]
    if category == "hybrid_occasion":
        return [
            {
                "title": "Presentation Base",
                "description": "Sharp shirt, tailored bottom, and serious shoes for the first room.",
                "palette": ["navy", "white", "charcoal"],
                "pieces": ["crisp shirt", "tailored trousers", "loafers"],
                "style_note": "Keep the authority in the base layer.",
            },
            {
                "title": "After-Work Softening",
                "description": "A removable layer or open collar that relaxes the same outfit.",
                "palette": ["ink", "taupe", "cream"],
                "pieces": ["light blazer", "soft knit", "sleek belt"],
                "style_note": "The transition should feel effortless, not like a costume change.",
            },
            {
                "title": "One Social Detail",
                "description": "A textured accessory or richer color that reads well in evening light.",
                "palette": ["slate", "burgundy", "black"],
                "pieces": ["tonal shirt", "structured trousers", "subtle accessory"],
                "style_note": "One relaxed detail is enough after a client-facing day.",
            },
        ]
    if category == "work_occasion":
        return [
            {
                "title": "Sharp Day Base",
                "description": "A crisp top, tailored bottom, and professional footwear.",
                "palette": ["navy", "white", "charcoal"],
                "pieces": ["crisp shirt", "tailored trousers", "loafers"],
                "style_note": "Credible first, with a clean line from meeting to commute.",
            },
            {
                "title": "Desk To Drinks",
                "description": "Work polish with one softer evening detail.",
                "palette": ["ink", "taupe", "cream"],
                "pieces": ["lightweight blazer", "knit or shirt", "sleek shoes"],
                "style_note": "Remove the layer or open the collar after work to relax it.",
            },
            {
                "title": "Quiet Authority",
                "description": "Tonal dressing with minimal accessories and stronger shoes.",
                "palette": ["slate", "black", "steel blue"],
                "pieces": ["tonal top", "structured bottom", "watch or belt"],
                "style_note": "Precise, not flashy.",
            },
        ]
    if mode == COLOR_BODY_ADVICE:
        return [
            {
                "title": "Warm Depth",
                "description": "Warm, earthy colors placed near the face.",
                "palette": ["olive", "cream", "rust"],
                "pieces": ["olive shirt", "cream layer", "brown footwear"],
                "style_note": "Adds warmth without washing out olive undertones.",
            },
            {
                "title": "Grounded Contrast",
                "description": "A deeper neutral base with one warm accent.",
                "palette": ["warm navy", "camel", "ivory"],
                "pieces": ["navy base", "camel layer", "ivory top"],
                "style_note": "Keeps contrast clean but not harsh.",
            },
            {
                "title": "Muted Statement",
                "description": "A controlled color moment with soft neutrals around it.",
                "palette": ["teal", "stone", "tobacco"],
                "pieces": ["teal top", "stone trouser", "tobacco shoe"],
                "style_note": "Color reads intentional rather than loud.",
            },
        ]
    return [
        {
            "title": "Relaxed Oxford",
            "description": "Oxford shirt, clean denim, and easy footwear.",
            "palette": ["navy", "white", "tan"],
            "pieces": ["Oxford shirt", "dark denim", "clean sneakers"],
            "style_note": "Approachable and tidy without feeling formal.",
        },
        {
            "title": "Knit Polo Polish",
            "description": "A knit top with sharper trousers and refined shoes.",
            "palette": ["cream", "olive", "brown"],
            "pieces": ["knit polo", "straight trousers", "loafers"],
            "style_note": "Soft texture makes the polish feel relaxed.",
        },
        {
            "title": "Soft Layered Casual",
            "description": "Light layer over a simple base with grounded footwear.",
            "palette": ["stone", "blue", "charcoal"],
            "pieces": ["light jacket", "plain tee", "chinos"],
            "style_note": "Useful when the setting might shift.",
        },
    ]


def _coerce_mode(query: str, intent: dict | str | None, context: dict | None) -> str:
    """Resolve the style mode with a HARD precedence so a wardrobe action
    never loses to a visual-inspiration phrase that happens to sit in the
    same string (e.g. chip value
    "Use my wardrobe for: show visual inspiration for coffee date").

        use_wardrobe > find_missing_pieces > visual_inspiration > style_advice
    """
    ctx = context if isinstance(context, dict) else {}
    style_action = _norm(ctx.get("style_action"))
    next_action = _norm(ctx.get("next_action"))
    module_context = str(ctx.get("module_context") or ctx.get("module") or "")
    # _norm strips underscores ("style_advice" -> "style advice"); restore them
    # so the intent name matches the mode constants.
    intent_value = _intent_name(intent).replace(" ", "_")
    q = _norm(query)
    action_blob = f"{q} {style_action} {next_action}"

    # 1) use_wardrobe — highest precedence.
    if (
        module_context.lower() in {"wardrobe", "closet"}
        or _has_any(action_blob, ("use_wardrobe", "use my wardrobe", "use wardrobe", "from my wardrobe", "with my wardrobe"))
    ):
        logger.info("AHVI_STYLE_ROUTE_FORCED mode=wardrobe_style reason=use_wardrobe")
        return WARDROBE_STYLE

    # 2) find_missing_pieces.
    if _has_any(action_blob, ("find_missing_pieces", "find missing pieces", "missing piece", "missing pieces", "what should i buy", "shopping ideas", "complete the look", "complete this look", "find this", "find similar", "shop this", "buy similar")):
        logger.info("AHVI_STYLE_ROUTE_FORCED mode=missing_pieces reason=find_missing_pieces")
        return SHOPPING_ASSIST

    # 3) visual_inspiration.
    if _has_any(action_blob, ("show visual inspiration", "visual inspiration", "generate moodboard", "show moodboard", "moodboard for")):
        return VISUAL_INSPIRATION

    # 4) explicit intent, else classify (style_advice default).
    if intent_value in _STYLE_REASONING_MODES:
        return intent_value
    return (
        classify_style_mode(
            query,
            module_context=module_context,
            style_action=str(ctx.get("style_action") or ""),
        )
        or GENERAL
    )


def _build_reasoning_prompt(
    *,
    query: str,
    mode: str,
    category: str | None,
    user_profile: dict,
    context: dict,
    policy: dict | None = None,
    style_ctx: dict | None = None,
    persona: dict | None = None,
    archetypes: list | None = None,
) -> str:
    policy = policy or {}
    style_ctx = style_ctx or {}
    persona = persona or {}
    archetypes = archetypes or []
    # Resolve a single gender context for every generation branch so archetype
    # and advice prompts enforce it consistently (prompt override > profile).
    target_gender = _resolve_asset_gender(query=query, user_profile=user_profile)
    anchor = _extract_pairing_anchor(query) if mode == STYLE_PAIRING else {}
    if mode == STYLE_PAIRING:
        import json as _json

        gender = str(persona.get("gender_profile") or target_gender or "unknown")
        archetype_names = [a.get("name") for a in archetypes if isinstance(a, dict)]
        # Gender-filter the archetype item hints fed to the model so a female
        # persona never receives male-only ethnic pieces (sherwani, nehru
        # jacket, ...) — and vice-versa for feminine-only items on a male.
        _allow_fem = target_gender == "female" or _prompt_allows_gendered_feminine_style(query)

        def _gender_items(items):
            return [
                _asset_text(it)
                for it in (items or [])
                if _style_text_allowed_for_gender(it, target_gender, allow_feminine=_allow_fem)
            ]

        _arch_compact = [
            {
                "name": a.get("name"),
                "impression": a.get("impression"),
                "preferred_items": _gender_items(a.get("preferred_items")),
                "avoid_items": a.get("avoid_items"),
                "palette": a.get("palette"),
            }
            for a in archetypes
            if isinstance(a, dict)
        ]
        _arch_json = _json.dumps(_arch_compact, ensure_ascii=False)
        _persona_json = _json.dumps(persona, ensure_ascii=False)
        return f"""
{AHVI_SYSTEM_PROMPT}

You are AHVI's PERSONAL stylist (not a generic fashion encyclopedia). The user
is asking an open-ended pairing question. Ground every route in their persona +
the selected archetypes below.

Persona context (obey — never assume beyond it):
{_persona_json}

Selected archetypes (build routes ONLY from these — do not invent others):
{_arch_json}

Return ONLY valid JSON matching this schema:
{{
  "mode": "style_pairing",
  "anchor_item": {{"name": string, "category": string, "color": string}},
  "stylist_reasoning": string,
  "pairing_routes": [
    {{
      "title": string,
      "archetype": string,
      "impression_created": string,
      "use_case": string,
      "strategy": string,
      "items": [string],
      "palette": [string],
      "why_it_works": string,
      "avoid": [string],
      "styling_tip": string,
      "persona_fit_reason": string,
      "archetype_reasoning": string,
      "dna_alignment": string,
      "wardrobe_alignment": string
    }}
  ],
  "what_to_avoid": [string],
  "next_actions": ["Use my wardrobe", "Show visual inspiration", "Find missing pieces"],
  "follow_up_question": string|null,
  "confidence": float
}}

Rules:
- Return 4-5 pairing_routes, each mapped to a DIFFERENT selected archetype
  (set route.archetype to that archetype's exact name). Use evocative titles,
  never "Option 1" / "Casual Look".
- Persona gender = {gender}. If male: NEVER suggest skirts, dresses, camisoles,
  heels, sarees, lehengas, or feminine-only silhouettes (unless the user
  explicitly asked). If female: feminine routes allowed, but NEVER suggest
  male-only garments (sherwani, nehru jacket, bandhgala, dhoti, kurta-pajama,
  mojari); use female ethnic equivalents (lehenga, saree, anarkali, kurti,
  salwar) instead, and still respect style DNA. If unknown: keep items
  gender-neutral (trousers, denim, chinos, shirts, polos, knitwear,
  overshirts, jackets, sneakers, loafers, boots).
- Do not mention gender unless relevant. Never assume beyond persona context.
- persona_fit_reason: one line on why this route suits THIS user.
- archetype is the controlled source of truth; title is only a secondary
  generated label. Never invent archetypes outside Allowed archetype names.
- archetype_reasoning, dna_alignment, and wardrobe_alignment should explain
  why the selected archetype fits the anchor, persona, and wardrobe reality.
- Include avoid guidance per route. Do not generate wardrobe boards, images, or
  shopping links. Do not sound like a textbook.
- Allowed archetype names: {archetype_names}

Known deterministic mode: style_pairing
Detected anchor_item: {anchor}
User query: {_clean_recursive_prompt(query)}
"""
    if mode in _ADVICE_MODES:
        if mode == BODY_PROPORTION_ADVICE:
            _shape = (
                '"principles": [string],   // proportion rules (e.g. "vertical lines elongate")\n'
                '  "do": [string],          // concrete styling moves\n'
                '  "avoid": [string],       // what shortens / unbalances\n'
                '  "outfit_examples": [string]'
            )
            _label = "body_proportion_advice"
        elif mode == COLOR_ADVICE:
            _shape = (
                '"recommended_colors": [string],\n'
                '  "avoid_colors": [string],\n'
                '  "why": [string],         // why these work for them\n'
                '  "outfit_palettes": [string]'
            )
            _label = "color_advice"
        else:  # OCCASION_ADVICE
            _shape = (
                '"do": [string],\n'
                '  "avoid": [string],\n'
                '  "better_alternatives": [string],\n'
                '  "styling_routes": [string]'
            )
            _label = "occasion_advice"
        return f"""
{AHVI_SYSTEM_PROMPT}

You are AHVI's senior stylist answering an open-ended {_label} question — not
building an outfit board. Be specific and practical, like a stylist, never a
textbook. No "styling principles" headings.

Return ONLY valid JSON:
{{
  "mode": "{_label}",
  "stylist_reasoning": string,   // 1-2 sentence human summary
  {_shape},
  "what_to_avoid": [string],
  "confidence": float
}}

Rules:
- 3-5 items per list, concrete and wearable.
- Ground in the persona/style context if present; never invent personal data.
- No images, no wardrobe board, no shopping links.

User query: {_clean_recursive_prompt(query)}
"""
    return f"""
{AHVI_SYSTEM_PROMPT}

{OCCASION_INTERPRETER_PROMPT}

You are AHVI's senior stylist — a real human stylist thinking out loud, not a
fashion database listing templates.

Before recommending any clothing, decide privately in this order:
1. What impression should the user create in this exact moment?
2. What level of ease, polish, or restraint does the occasion need?
3. What styling approach best fits the moment?
4. What styling risk should be avoided, and why?
5. What atmosphere should the outfit communicate?
6. What single missing piece would most improve this direction?

Lead with the opinion and the reasoning. The outfit directions only support
that reasoning — they never replace it.

Return ONLY valid JSON matching this schema:
{{
  "mode": "style_advice | visual_inspiration | color_body_advice | style_education | shopping_assist",
  "occasion": string|null,
  "goal": string,
  "impression": string,
  "atmosphere": string,
  "confidence_strategy": string,
  "emotion_state": "neutral | excited | frustrated | vulnerable | professional | social",
  "stylist_reasoning": string,
  "what_to_avoid": [string],
  "missing_piece_reasoning": string,
  "visual_directions": [
    {{
      "title": string,
      "archetype": string,
      "impression": string,
      "strategy": string,
      "description": string,
      "palette": [string],
      "pieces": [string],
      "why_it_works": string,
      "board_brief": {{"hero": string, "support": string, "footwear": string, "accent": string}},
      "style_note": string
    }}
  ],
  "missing_piece": {{"name": string, "category": string, "reason": string, "unlocks": [string]}},
  "visual_inspiration_board": {{"title": string, "aesthetic": string, "mood": string, "palette": [string], "hero_piece": string, "silhouette": string, "styling_notes": string}},
  "follow_up_question": string|null,
  "confidence": float
}}

confidence_strategy = one line on how to make the user feel confident in this
specific moment (what to lean into, what to reassure).
board_brief = a compact brief the board renderer can visualize: which piece is
the hero, what supports it, the footwear, and the one accent.

Writing rules for stylist_reasoning:
- Speak like a stylist explaining a decision. Use phrasing such as
  "This works because...", "I would avoid...", "The priority here is...".
- Explain the outfit feeling first, then the clothing logic.
- Avoid internal planning labels; write only the final stylist recommendation.
- KEEP IT SHORT: 35-60 words for a normal occasion, max 70 for multi-event.
  No markdown, no headings, no "**Core:**" / "**Presentation Layer:**" labels,
  no bullet lists. One tight paragraph.

Gender context (obey for every piece, palette, and missing_piece): persona
gender = {target_gender}. If male: NEVER suggest skirts, dresses, blouses,
heels, sarees, lehengas, or feminine-only silhouettes unless the user
explicitly asked. If female: feminine pieces allowed, but NEVER suggest
male-only garments (sherwani, nehru jacket, bandhgala, dhoti, kurta-pajama,
mojari) — use female ethnic equivalents (lehenga, saree, anarkali, kurti,
salwar). If unknown: keep every piece gender-neutral. Do not mention gender
unless relevant.

Wardrobe grounding: if the style context shows wardrobe_available with items,
do NOT describe a garment as owned ("wear your suit") unless that garment
appears in the wardrobe list. Any garment the user does not own goes ONLY in
missing_piece, clearly marked as a suggestion to acquire — never implied owned.

Obey the policy.outfit_validation_principles: if the occasion and an item
clash (shiny/formal for coffee, office polish forced into date night, formal
shirt for a game), do NOT overconfidently recommend it — name the mismatch
softly in what_to_avoid and stylist_reasoning, then offer one correction.

Memory: if style_ctx.memory.recently_worn is non-empty, you MAY add ONE short
clause about rotating in a fresher option (e.g. "since you wore the navy shirt
recently, I'm rotating in something fresher"). Do NOT over-explain. If
style_ctx.memory is null/absent, NEVER invent or mention wear history.

Personalization: if style_ctx.style_dna is present, ground the reasoning in it
naturally ("your wardrobe leans relaxed-modern", lean into preferred_colors /
preferred_silhouettes, respect avoided_colors + avoid_style_keywords). If
style_dna is absent or empty, do NOT invent personal taste — stay occasion-led.

Obey the policy.wardrobe_management_principles for weak/empty wardrobes: say
what the wardrobe leans toward first (e.g. office/casual), acknowledge what CAN
be built, then frame gaps as one or two occasion-specific anchors to ADD
(e.g. "a relaxed evening shirt and a softer shoe") — never a long missing list,
never a blunt "I don't see options".

When visual_inspiration: shape visual_inspiration_board from
policy.mood_board_contract (pick aesthetic from its aesthetic_taxonomy, mood +
keywords from its emotion_mapping) and end with a clear next action +
missing_piece per policy.inspiration_board_contract.

For visual_inspiration mode also fill visual_inspiration_board:
{{"title","aesthetic","mood","palette":[],"hero_piece","silhouette","styling_notes"}}.

If Selected visual archetypes are provided, every visual_direction.archetype
must be one of those exact names. The archetype is the visible source of truth;
title is only a secondary edition label.

Each visual_direction.why_it_works must explain the STYLING LOGIC, not just
restate the pieces. Bad: "Oxford shirt with denim." Good: "The shirt creates
structure while the denim keeps it approachable."

missing_piece_reasoning must justify ONE piece and where else it earns its
place. Bad: "Brown loafers." Good: "A brown suede loafer would elevate this
and carry across coffee dates, weekend dinners, and smart-casual office days."

Ban this generic filler unless genuinely unavoidable:
"balanced silhouette", "color harmony", "approachable and tidy",
"elevated aesthetic", "perfect for". Replace with a real reason.

Hard rules:
- Do not generate image prompts or real images.
- For style_advice and visual_inspiration, return exactly 3 visual_directions.
- Each direction must differ by mood, silhouette, palette, or formality.
- For "Show visual inspiration", make the cards the main response.
- Do not open with "Here are styling principles". Do not sound like a textbook.
- Different occasions MUST produce clearly different goal/impression/avoid.
  Christian funeral: respectful presence, understated, no bright/flashy.
  Coffee date: approachable confidence, intentional not corporate.
  Client presentation + drinks: credibility first then social ease,
  avoid full formal that feels awkward later (transitional dressing).
  Wedding guest: celebratory restraint, festive without competing, no bridal.
  Beach dinner: relaxed evening polish, no swimwear/flip-flops after sunset.
- Wardrobe styling is not allowed in this response.
- MULTI-EVENT / TRANSITION: if style context has sub_occasions or a
  style_strategy, the user is dressing for MORE THAN ONE event in sequence.
  Do NOT collapse it to a single occasion (never "date night" for a game +
  dinner). Reason about the transition: dress for the first event's needs
  (e.g. comfort + movement for a game), then make the later event feel
  intentional without a full outfit change. goal/impression must reflect the
  transition, and what_to_avoid should flag anything that fails either event.
  ALSO include a "transition_plan" object: {{"keep": [items that carry across],
  "swap": [items to change], "add": [pieces that upgrade for the 2nd event],
  "avoid": [what fails either event], "dinner_ready": "one line on the upgraded
  look"}}. Do NOT force one rigid outfit — give a keep/swap/add path.

Style policy (compact — obey, do not echo verbatim):
{policy}

Style context (compact — the user's real situation):
{style_ctx}

Selected visual archetypes:
{[a.get("name") for a in archetypes if isinstance(a, dict)]}

Known deterministic mode: {mode}
Detected category: {category or "unknown"}
User query: {query}
"""


def _clean_direction_title(title: Any) -> str:
    text = re.sub(r"\s+", " ", str(title or "").replace("_", " ")).strip(" -:|")
    if not text:
        return ""
    text = re.sub(r"\bCelebn\b", "Celebration", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcustom occasion\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsocial occasion\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -:|")
    if re.search(r"\bwedding\b.*\bcousin wedding\b", text, flags=re.IGNORECASE):
        return "Cousin Wedding"
    if re.search(r"\bwedding\s+haldi ceremony\b", text, flags=re.IGNORECASE):
        return "Haldi Ceremony"
    if re.fullmatch(r"haldi ceremony(?:\s+haldi)?", text, flags=re.IGNORECASE):
        return "Haldi Ceremony"
    return text


def _normalize_direction(value: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(value) if isinstance(value, dict) else {}
    pieces = _safe_list(item.get("pieces") or item.get("items") or fallback.get("pieces"), limit=6)
    palette = _safe_list(item.get("palette") or item.get("colors") or fallback.get("palette"), limit=5)
    style_note = str(
        item.get("styling_tip")
        or item.get("style_note")
        or fallback.get("styling_tip")
        or fallback.get("style_note")
        or ""
    ).strip()
    hero_piece = str(
        item.get("hero_piece")
        or item.get("heroPiece")
        or (pieces[0] if pieces else "")
    ).strip()
    return {
        "title": _clean_direction_title(
            item.get("title") or fallback.get("title") or "Style Direction"
        ),
        "subtitle": str(item.get("subtitle") or item.get("style_direction") or item.get("styleDirection") or "").strip(),
        "archetype": _clean_direction_title(
            item.get("archetype") or fallback.get("archetype") or ""
        ),
        "impression": str(item.get("impression") or fallback.get("impression") or "").strip(),
        "strategy": str(item.get("strategy") or fallback.get("strategy") or "").strip(),
        "description": str(item.get("description") or fallback.get("description") or "").strip(),
        "hero_piece": hero_piece,
        "hero_piece_reasoning": str(
            item.get("hero_piece_reasoning")
            or item.get("heroPieceReasoning")
            or ""
        ).strip(),
        "palette": palette,
        "colors": palette,
        "pieces": pieces,
        "items": pieces,
        "why_it_works": str(
            item.get("why_it_works") or fallback.get("why_it_works") or ""
        ).strip(),
        "why_this_works": str(
            item.get("why_this_works") or item.get("whyThisWorks") or item.get("why_it_works") or fallback.get("why_this_works") or ""
        ).strip(),
        "board_brief": item.get("board_brief") if isinstance(item.get("board_brief"), dict) else {},
        "style_note": style_note,
        "styling_tip": style_note[:80],
        "missing_piece": item.get("missing_piece") if isinstance(item.get("missing_piece"), dict) else {},
        "complete_the_look": item.get("complete_the_look") if isinstance(item.get("complete_the_look"), list) else [],
        "archetype_reasoning": str(item.get("archetype_reasoning") or "").strip(),
        "dna_alignment": str(item.get("dna_alignment") or "").strip(),
        "wardrobe_alignment": str(item.get("wardrobe_alignment") or "").strip(),
    }


def _ensure_direction_logic(direction: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee why_it_works + strategy so even fallback cards read like a
    stylist explaining logic, not a database listing pieces."""
    pieces = direction.get("pieces") or []
    title = str(direction.get("title") or "this direction").strip()
    if not direction.get("why_it_works"):
        if len(pieces) >= 2:
            direction["why_it_works"] = (
                f"The {str(pieces[0]).lower()} sets the structure while the "
                f"{str(pieces[1]).lower()} keeps it easy — so the look reads "
                "intentional without trying too hard."
            )
        else:
            direction["why_it_works"] = (
                f"{title} keeps one clear focal point so the outfit feels "
                "deliberate rather than busy."
            )
    if not direction.get("strategy"):
        direction["strategy"] = str(direction.get("style_note") or "").strip()
    return direction


def _validate_visual_direction_consistency(
    direction: Dict[str, Any],
    *,
    occasion: str = "",
    target_gender: str = "unknown",
    allow_feminine: bool = False,
) -> Dict[str, Any]:
    out = dict(direction)
    components = _safe_list(out.get("items") or out.get("pieces"), limit=6)
    if not components:
        components = _safe_component_fallback(target_gender)
    hero = _asset_text(out.get("hero_piece"))
    if not hero or not any(_style_items_similar(hero, item) for item in components):
        # The visual card has one source object. If Gemini supplied a hero that
        # is not actually represented in the component list, promote the first
        # component instead of letting the card contradict itself.
        hero = components[0]
    out["hero_piece"] = hero
    out["items"] = components
    out["pieces"] = components

    rewritten_fields: List[str] = []
    if _direction_text_contradicts_components(out.get("description"), components) or _direction_text_disallowed_for_gender(
        out.get("description"), target_gender=target_gender, allow_feminine=allow_feminine
    ):
        out["description"] = _direction_description_from_source(out, occasion)
        rewritten_fields.append("description")
    if _direction_text_contradicts_components(out.get("why_it_works"), components) or _direction_text_disallowed_for_gender(
        out.get("why_it_works"), target_gender=target_gender, allow_feminine=allow_feminine
    ):
        out["why_it_works"] = _direction_why_from_source(out, occasion)
        rewritten_fields.append("why_it_works")
    if _direction_text_contradicts_components(out.get("why_this_works"), components) or _direction_text_disallowed_for_gender(
        out.get("why_this_works"), target_gender=target_gender, allow_feminine=allow_feminine
    ):
        out["why_this_works"] = out.get("why_it_works") or _direction_why_from_source(out, occasion)
        rewritten_fields.append("why_this_works")
    if not _asset_text(out.get("description")):
        out["description"] = _direction_description_from_source(out, occasion)
    if not _asset_text(out.get("why_it_works")):
        out["why_it_works"] = _direction_why_from_source(out, occasion)
    if not _asset_text(out.get("why_this_works")):
        out["why_this_works"] = out.get("why_it_works") or _direction_why_from_source(out, occasion)

    tip = _asset_text(out.get("styling_tip") or out.get("style_note"))
    if (
        _direction_text_contradicts_components(tip, components)
        or _direction_text_disallowed_for_gender(tip, target_gender=target_gender, allow_feminine=allow_feminine)
        or not tip
    ):
        tip = _direction_tip_from_source(out)
    out["styling_tip"] = tip[:80]
    out["style_note"] = out["styling_tip"]

    mp = out.get("missing_piece")
    if isinstance(mp, dict) and _asset_text(mp.get("name")):
        name = _asset_text(mp.get("name"))
        reason = _missing_piece_duplicate_reason(name, hero_piece=hero, components=components)
        generic_name = _missing_name_is_generic(name) or _norm(name) in _SPECIFIC_MISSING_NAMES
        occasion_block = _occasion_asset_block_reason(
            mp,
            occasion=occasion,
            placement="missing",
            target_text=name,
        )
        if (
            reason
            or generic_name
            or occasion_block
            or not _style_text_allowed_for_gender(
                name,
                target_gender,
                allow_feminine=allow_feminine,
            )
        ):
            replacement = ""
            festive_replacement = _festive_missing_piece_replacement(
                occasion,
                out,
                target_gender=target_gender,
                allow_feminine=allow_feminine,
            )
            if festive_replacement:
                out["missing_piece"] = festive_replacement
                replacement = _asset_text(festive_replacement.get("name"))
            elif generic_name:
                specific_name = _specific_missing_piece_name(name, out, occasion=occasion)
                if (
                    specific_name
                    and _style_text_allowed_for_gender(specific_name, target_gender, allow_feminine=allow_feminine)
                    and not _missing_piece_duplicate_reason(specific_name, hero_piece=hero, components=components)
                ):
                    replacement = specific_name
            if not replacement:
                replacement = _fallback_missing_piece_for_direction(
                    out,
                    occasion=occasion,
                    target_gender=target_gender,
                    allow_feminine=allow_feminine,
                )
            if replacement and not festive_replacement:
                out["missing_piece"] = {
                    "name": replacement,
                    "category": _style_category(replacement) or "style piece",
                    "reason": _missing_piece_reason_for_direction(replacement, out, occasion=occasion),
                    "unlocks": [out.get("archetype") or out.get("title") or "Style direction"],
                }
            elif not replacement:
                out.pop("missing_piece", None)
            rewritten_fields.append(
                f"missing_piece:{occasion_block or reason or ('generic' if generic_name else 'gender')}"
            )
        else:
            mp = dict(mp)
            mp["reason"] = _missing_piece_reason_for_direction(name, out, occasion=occasion)
            out["missing_piece"] = mp

    if rewritten_fields:
        logger.info(
            "AHVI_VISUAL_DIRECTION_CONSISTENCY_REWRITTEN title=%r fields=%s components=%s",
            out.get("title"),
            rewritten_fields,
            components,
        )
    return out


# ---------------------------------------------------------------------
# Editorial UX polish
# ---------------------------------------------------------------------
# Backend payload contract for the premium personal-stylist frontend:
# every visual direction gets a short stylist note, a named direction,
# a 3-word adjective triad, a wardrobe match %, a recommendation badge
# and a "complete the look" line. The response also surfaces an
# editorial_cover summary so the client can render the magazine-style
# top card without re-deriving anything.
#
# Everything here is additive except `_two_sentences`, which caps
# existing stylist note fields server-side. The underlying AI / styling
# engine is untouched.
# ---------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _two_sentences(text: Any, *, max_chars: int = 240) -> str:
    """Trim AI-generated note text to at most two sentences and ~240 chars.

    Returns ``""`` for empty input. Preserves trailing punctuation.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(raw) if s.strip()]
    if not sentences:
        return raw[:max_chars].rstrip()
    short = " ".join(sentences[:2]).strip()
    if len(short) > max_chars:
        short = short[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-") + "…"
    return short


_DIRECTION_ADJECTIVES: dict[str, list[str]] = {
    # archetype name (lower) -> 3 adjectives. Falls back to a generic triad.
    "modern professional": ["Confident", "Structured", "Approachable"],
    "modern authority": ["Confident", "Structured", "Approachable"],
    "executive minimalist": ["Clean", "Refined", "Understated"],
    "creative speaker": ["Modern", "Relaxed", "Tech-Forward"],
    "smart casual edge": ["Sharp", "Easy", "Versatile"],
    "refined weekend": ["Relaxed", "Considered", "Polished"],
    "polished casual": ["Polished", "Easy", "Considered"],
    "urban minimalist": ["Clean", "Sharp", "Quiet"],
    "modern utility": ["Practical", "Sharp", "Current"],
    "quiet luxury": ["Refined", "Understated", "Elevated"],
    "contemporary classic": ["Timeless", "Tailored", "Composed"],
    "off-duty tailoring": ["Tailored", "Easy", "Polished"],
    "italian summer": ["Breezy", "Sunlit", "Polished"],
    "resort sophisticate": ["Breezy", "Refined", "Sunlit"],
    "festive modern": ["Celebratory", "Tailored", "Modern"],
    "modern desi": ["Festive", "Modern", "Rooted"],
    "boho artisanal": ["Soft", "Crafted", "Earthy"],
    "structured ease": ["Structured", "Easy", "Considered"],
    "textural ease": ["Textural", "Relaxed", "Warm"],
    "clean minimal": ["Clean", "Quiet", "Considered"],
}
_DEFAULT_DIRECTION_ADJECTIVES: list[str] = ["Considered", "Modern", "Confident"]


# Human stylist voice per occasion. Leads each card's short_note so the copy
# reads like a person, not an LLM. Keyed by normalized occasion; falls back
# to a neutral confident line. Reuses the same voice intent as
# ahvi_personality/02_tone/tone_rules.json (relaxed warmth, concise).
_OCCASION_VOICE: dict[str, str] = {
    "coffee_date": "Relaxed, approachable, and easy to wear — polished without feeling overdressed.",
    "coffee": "Relaxed, approachable, and easy to wear — polished without feeling overdressed.",
    "cafe_date": "Relaxed, approachable, and easy to wear — polished without feeling overdressed.",
    "first_date": "Warm and approachable, with just enough polish to feel intentional.",
    "date": "Easy confidence — put-together but never trying too hard.",
    "date_night": "Quietly magnetic — refined enough for evening, relaxed enough to enjoy it.",
    "brunch": "Light, easy, and put-together for a slow weekend table.",
    "brunch_date": "Light, easy, and put-together for a slow weekend table.",
    "casual_day": "Effortless and comfortable, with a considered finish.",
    "weekend": "Easy weekend polish — comfortable, considered, unforced.",
    "client_meeting": "Quiet authority — sharp first impression, comfortable all day.",
    "office": "Composed and professional, with room to move.",
    "startup_office": "Modern and sharp without the suit-and-tie stiffness.",
    "conference": "Confident and credible — sharp enough for the stage, comfortable enough for a full day.",
    "conference_talk": "Confident and credible — sharp enough for the stage, comfortable enough for a full day.",
    "presentation": "Composed and camera-ready, with steady presence.",
    "keynote": "Confident and credible — built to hold the room.",
    "interview": "Sharp, sincere, and quietly confident.",
    "wedding": "Celebratory and refined, dressed for the moment without stealing it.",
    "wedding_guest": "Celebratory and refined, dressed for the moment without stealing it.",
    "funeral": "Respectful and understated — keeps attention where it belongs.",
    "vacation": "Sunlit ease — relaxed, breathable, and ready to wander.",
    "travel": "Comfortable, layered, and ready for a long day in transit.",
    "airport_travel": "Comfortable, layered, and ready for a long day in transit.",
    "workout": "Built to move — focused, breathable, recovery-ready.",
    "gym": "Built to move — focused, breathable, recovery-ready.",
    "beach": "Cool, breathable, and ready for sand and sun.",
}
_DEFAULT_OCCASION_VOICE: str = "Considered and confident, styled to feel like you."


def _occasion_voice_note(occasion: Any) -> str:
    key = _occasion_key_for_editorial(occasion)
    return _OCCASION_VOICE.get(key, _DEFAULT_OCCASION_VOICE)


def _limit_words(text: Any, limit: int = 18) -> str:
    words = str(text or "").strip().split()
    return " ".join(words[:limit]).strip()


def _direction_short_note(
    direction: Dict[str, Any],
    *,
    occasion: Any,
    context_text: Any = "",
    index: int = 0,
) -> str:
    family = _visual_occasion_family(occasion, context_text)
    hero = _asset_text(direction.get("hero_piece"))
    archetype = _asset_text(direction.get("archetype") or direction.get("title"))
    templates: dict[str, tuple[str, ...]] = {
        "indian_festive": (
            f"Bright, breathable, and festive around {hero or 'traditional tailoring'}, with enough ease for rituals.",
            "Traditional polish with celebratory color, kept light enough for a long ceremony.",
            "Festive texture and confident color create presence without overpowering the celebration.",
        ),
        "christian_wedding": (
            "Ceremony-ready tailoring that feels celebratory, respectful, and comfortable through the reception.",
            "Classic wedding polish with a lighter touch for photographs, greetings, and a long celebration.",
            "Refined formal layers keep the mood special without competing with the wedding party.",
        ),
        "funeral": (
            "Respectful and understated, with quiet structure and minimal detail.",
            "Dark, composed layers keep the focus on the occasion and remain comfortable.",
            "Simple formal pieces create a dignified look without drawing unnecessary attention.",
        ),
        "travel": (
            "Comfort-first layers that still look sharp when you arrive.",
            "Easy movement, practical layers, and polished footwear carry this through a long travel day.",
            "Relaxed structure keeps the outfit comfortable in transit and composed at arrival.",
        ),
        "conference": (
            "Confident structure for the room, relaxed enough for long networking hours.",
            "Professional lines and comfortable finishing details keep the focus on your presentation.",
            "Confident, credible tailoring that stays comfortable from the first session through networking.",
        ),
    }
    if family in templates:
        return _limit_words(templates[family][index % len(templates[family])])
    occasion_key = _occasion_key_for_editorial(occasion)
    if occasion_key in {"coffee_date", "coffee", "cafe_date", "first_date", "date"}:
        coffee_notes = (
            "Easy, polished, and approachable without looking over-planned.",
            "Relaxed proportions and considered details make this feel natural, warm, and put-together.",
            "Quiet polish keeps the look confident while leaving room for an easy conversation.",
        )
        return _limit_words(coffee_notes[index % len(coffee_notes)])
    known_note = _occasion_voice_note(occasion)
    if known_note != _DEFAULT_OCCASION_VOICE:
        variants = (
            known_note,
            f"{archetype or 'This direction'} keeps the look purposeful, comfortable, and right for the occasion.",
            f"{hero or 'The hero piece'} gives this direction a distinct, occasion-ready point of view.",
        )
        return _limit_words(variants[index % len(variants)])
    if occasion_key or str(context_text or "").strip():
        return _limit_words(
            f"{archetype or 'This direction'} uses {hero or 'considered pieces'} for a clear, occasion-aware point of view."
        )
    return _limit_words(_DEFAULT_OCCASION_VOICE)


_OCCASION_CURATED_FOR: dict[str, list[str]] = {
    "conference": ["Stage Presence", "Networking", "All-Day Comfort"],
    "conference_talk": ["Stage Presence", "Networking", "All-Day Comfort"],
    "presentation": ["Stage Presence", "Composed Energy", "Camera-Ready"],
    "keynote": ["Stage Presence", "Composed Energy", "All-Day Comfort"],
    "seminar": ["Quiet Authority", "Networking", "All-Day Comfort"],
    "panel": ["Quiet Authority", "Networking", "Camera-Ready"],
    "client_meeting": ["Quiet Authority", "Sharp First Impression", "All-Day Comfort"],
    "client meeting": ["Quiet Authority", "Sharp First Impression", "All-Day Comfort"],
    "office": ["Polished Day", "Networking", "Easy Movement"],
    "startup_office": ["Modern Sharpness", "Camera-Ready", "Comfort"],
    "interview": ["Sharp First Impression", "Quiet Authority", "Composed Energy"],
    "coffee_date": ["Effortless Charm", "Considered Touches", "Easy Movement"],
    "coffee date": ["Effortless Charm", "Considered Touches", "Easy Movement"],
    "date_night": ["Magnetic Polish", "Considered Touches", "Warmth"],
    "first_date": ["Effortless Charm", "Considered Touches", "Warmth"],
    "wedding": ["Celebratory Polish", "Camera-Ready", "All-Day Comfort"],
    "wedding_guest": ["Celebratory Polish", "Camera-Ready", "All-Day Comfort"],
    "birthday_party": ["Magnetic Energy", "Camera-Ready", "Movement"],
    "vacation": ["Sunlit Ease", "Travel-Ready", "Considered Touches"],
    "airport_travel": ["Travel-Ready", "Easy Layers", "Composed Energy"],
    "travel": ["Travel-Ready", "Easy Layers", "Composed Energy"],
    "workout": ["Performance", "Movement", "Recovery-Ready"],
    "gym": ["Performance", "Movement", "Recovery-Ready"],
    "beach": ["Sunlit Ease", "Movement", "Cool Comfort"],
    "casual_day": ["Effortless Energy", "Considered Touches", "Movement"],
    "weekend": ["Easy Movement", "Considered Touches", "Warmth"],
}
_DEFAULT_CURATED_FOR: list[str] = ["Considered Energy", "Comfort", "Confident Presence"]


def _occasion_key_for_editorial(occasion: Any) -> str:
    return re.sub(r"\s+", "_", str(occasion or "").strip().lower())


def _occasion_label_for_editorial(occasion: Any) -> str:
    text = str(occasion or "").strip()
    if not text:
        return "Curated Look"
    cleaned = text.replace("_", " ").strip()
    return cleaned.upper()


def _direction_adjectives_from_archetype(archetype: Any) -> list[str]:
    key = str(archetype or "").strip().lower()
    if key in _DIRECTION_ADJECTIVES:
        return list(_DIRECTION_ADJECTIVES[key])
    return list(_DEFAULT_DIRECTION_ADJECTIVES)


def _curated_for_for_occasion(occasion: Any) -> list[str]:
    key = _occasion_key_for_editorial(occasion)
    return list(_OCCASION_CURATED_FOR.get(key, _DEFAULT_CURATED_FOR))


def _wardrobe_match_pct(
    direction: Dict[str, Any],
    wardrobe_items: Any,
) -> int | None:
    """Heuristic ownership %.

    Tokenises each direction item and counts how many appear inside any
    wardrobe item's name/category/tags. Returns ``None`` when no wardrobe
    signal is available so the client can choose to hide the badge.
    """
    if not isinstance(wardrobe_items, list) or not wardrobe_items:
        return None
    items = _safe_list(direction.get("items") or direction.get("pieces"), limit=6)
    if not items:
        return None
    wardrobe_blob = " ".join(
        " ".join(
            [
                _asset_text(w.get("name")) if isinstance(w, dict) else _asset_text(w),
                _asset_text((w or {}).get("category")) if isinstance(w, dict) else "",
                " ".join(_asset_list((w or {}).get("tags"))) if isinstance(w, dict) else "",
            ]
        )
        for w in wardrobe_items
        if w
    ).lower()
    if not wardrobe_blob.strip():
        return None
    matched = 0
    for item in items:
        family = _target_family(item)
        item_tokens = [t for t in re.findall(r"[a-z0-9]+", str(item).lower()) if len(t) >= 4]
        if any(tok in wardrobe_blob for tok in item_tokens):
            matched += 1
        elif family and family in wardrobe_blob:
            matched += 1
    pct = int(round((matched / len(items)) * 100))
    return max(0, min(100, pct))


def _occasion_fit_label(pct: int | None) -> str:
    if pct is None:
        return "Strong"
    if pct >= 85:
        return "Excellent"
    if pct >= 60:
        return "Strong"
    if pct >= 30:
        return "Good"
    return "Inspiring"


def _editorial_badge(pct: int | None) -> dict[str, Any]:
    return {
        "stars": 5,
        "label": "Recommended",
        "occasion_fit": _occasion_fit_label(pct),
        "wardrobe_match_pct": pct,
    }


def _complete_the_look_copy(
    missing_piece: Dict[str, Any] | None,
    occasion: Any,
) -> str:
    if not isinstance(missing_piece, dict):
        return ""
    name = _asset_text(missing_piece.get("name"))
    if not name:
        return ""
    label = str(occasion or "").replace("_", " ").strip() or "the moment"
    return f"One piece away from a polished {label}-ready look."


# ---- Ownership truth -------------------------------------------------
# Per-direction owned_items list. Strict fashion-family allowlist so the
# UI never renders chargers / electronics / misc wardrobe rows as
# stylist recommendations.

_OWNERSHIP_ALLOWED_FAMILIES: dict[str, str] = {
    # asset family -> public ownership bucket label.
    "shirt": "top",
    "tshirt": "top",
    "polo": "top",
    "hoodie": "top",
    "sweatshirt": "top",
    "knit": "top",
    "ethnic": "ethnicwear",
    "blazer": "outerwear",
    "jacket": "outerwear",
    "coat": "outerwear",
    "overshirt": "outerwear",
    "jeans": "bottom",
    "chino": "bottom",
    "trouser": "bottom",
    "cargo_pants": "bottom",
    "joggers": "bottom",
    "shorts": "bottom",
    "formal_shoe": "footwear",
    "loafer": "footwear",
    "sneaker": "footwear",
    "sandal": "footwear",
    "slide": "footwear",
    "boot": "footwear",
    "belt": "accessory",
    "watch": "watch",
    "bag": "bag",
    "laptop_bag": "bag",
    "messenger_bag": "bag",
    "backpack": "bag",
    "duffle_bag": "bag",
    "cardholder": "accessory",
    "sunglasses": "accessory",
    "tie": "accessory",
    "scarf": "accessory",
    "jewellery": "jewellery",
}

# Per-bucket caps for owned-item lists. Keeps a styled look from collecting
# two bottoms / two bags / three accessories out of a noisy wardrobe match.
_OWNED_BUCKET_CAPS: dict[str, int] = {
    "bottom": 1,
    "footwear": 1,
    "bag": 1,
    "outerwear": 1,
    "accessory": 2,
}

_OWNERSHIP_BLOCKED_NAME_TOKENS: tuple[str, ...] = (
    "charger",
    "cable",
    "power bank",
    "powerbank",
    "headphone",
    "earbud",
    "comb",
    "razor",
    "toothbrush",
    "skincare",
    "moisturizer",
    "serum",
    "sunscreen",
    "toiletry",
    "adapter",
    "pillow",
    "bottle",
    "tumbler",
    "eye mask",
    "eyemask",
    "first aid",
    "medicine",
    "supplement",
    "pen",
    "notebook",
    "book",
    "stationery",
    "electronics",
)


def _wardrobe_normalised(wardrobe_items: Any) -> List[Dict[str, Any]]:
    """Normalise wardrobe rows so per-item ownership checks stay cheap."""
    if not isinstance(wardrobe_items, list):
        return []
    # Shared fashion gate first: keeps this path consistent with
    # build_style_context / format_wardrobe_for_llm. String-only rows pass
    # through untouched (handled below) since the sanitizer drops non-dicts.
    try:
        from services.wardrobe_sanitizer import is_fashion_item

        wardrobe_items = [
            raw
            for raw in wardrobe_items
            if not isinstance(raw, dict) or is_fashion_item(raw)
        ]
    except Exception:  # noqa: BLE001 - never break ownership on import issues
        pass
    out: List[Dict[str, Any]] = []
    for raw in wardrobe_items:
        if isinstance(raw, dict):
            name = _asset_text(raw.get("name") or raw.get("title"))
            if not name:
                continue
            out.append(
                {
                    "id": _asset_text(
                        raw.get("id") or raw.get("$id") or raw.get("asset_id")
                    ),
                    "name": name,
                    "category": _asset_text(
                        raw.get("category") or raw.get("subcategory")
                    ),
                    "tags": [str(t).lower() for t in _asset_list(raw.get("tags"))],
                    "image_url": _asset_text(
                        raw.get("image_url") or raw.get("imageUrl")
                    ),
                }
            )
        elif isinstance(raw, str) and raw.strip():
            out.append(
                {
                    "id": "",
                    "name": raw.strip(),
                    "category": "",
                    "tags": [],
                    "image_url": "",
                }
            )
    return out


def _wardrobe_item_blocked(item: Dict[str, Any]) -> bool:
    blob = " ".join(
        [
            str(item.get("name") or "").lower(),
            str(item.get("category") or "").lower(),
            " ".join(item.get("tags") or []),
        ]
    )
    return any(token in blob for token in _OWNERSHIP_BLOCKED_NAME_TOKENS)


def _wardrobe_item_family(item: Dict[str, Any]) -> str:
    family = _detect_family(item.get("name"))
    if not family:
        family = _detect_family(item.get("category"))
    if not family and item.get("tags"):
        family = _detect_family(" ".join(item.get("tags") or []))
    return family


def _ownership_match(
    piece_name: str,
    wardrobe: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """Return the wardrobe row that best matches a styled piece, or None."""
    piece_lower = piece_name.lower().strip()
    if not piece_lower:
        return None
    piece_family = _detect_family(piece_name)
    piece_tokens = {
        t for t in re.findall(r"[a-z0-9]+", piece_lower) if len(t) >= 4
    }
    high_specificity = {
        "formal_shoe",
        "loafer",
        "blazer",
        "watch",
        "laptop_bag",
        "messenger_bag",
        "tie",
    }
    best_family_match: Dict[str, Any] | None = None
    for row in wardrobe:
        if _wardrobe_item_blocked(row):
            continue
        row_family = _wardrobe_item_family(row)
        if row_family and row_family not in _OWNERSHIP_ALLOWED_FAMILIES:
            continue
        row_name = row["name"].lower()
        if piece_lower in row_name or row_name in piece_lower:
            return row
        if piece_family and row_family == piece_family:
            row_tokens = {
                t for t in re.findall(r"[a-z0-9]+", row_name) if len(t) >= 4
            }
            if piece_tokens & row_tokens:
                return row
            if row_family in high_specificity and best_family_match is None:
                best_family_match = row
    return best_family_match


def _build_owned_items(
    direction: Dict[str, Any],
    wardrobe: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int, int]:
    """Return (owned_items, owned_count, total_items)."""
    pieces = _safe_list(direction.get("items") or direction.get("pieces"), limit=8)
    if not pieces:
        return [], 0, 0
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    seen_names: set[str] = set()
    bucket_counts: Dict[str, int] = {}
    owned: List[Dict[str, Any]] = []
    has_full_body = False
    # Aggregate guard counters (logged once, never per-item).
    matched = 0
    skipped_duplicate = 0
    skipped_family_cap = 0
    skipped_dress_conflict = 0
    for piece in pieces:
        match = _ownership_match(piece, wardrobe)
        if not match:
            continue
        if _wardrobe_item_blocked(match):
            continue
        family = _wardrobe_item_family(match)
        bucket = _OWNERSHIP_ALLOWED_FAMILIES.get(family)
        if not bucket:
            continue
        matched += 1
        identifier = match.get("id") or match["name"].lower()
        norm_name = str(match["name"]).strip().lower()
        image = str(match.get("image_url") or "").strip()
        # De-dupe by id, normalized name, and image_url.
        if identifier in seen_ids or norm_name in seen_names:
            skipped_duplicate += 1
            continue
        if image and image in seen_images:
            skipped_duplicate += 1
            continue
        # Dress / full-body item excludes a separate top + bottom (and vice
        # versa): never let a one-piece coexist with a top/bottom.
        if bucket in {"dress", "ethnicwear"}:
            if any(b in bucket_counts for b in ("top", "bottom")):
                skipped_dress_conflict += 1
                continue
            has_full_body = True
        elif bucket in {"top", "bottom"} and has_full_body:
            skipped_dress_conflict += 1
            continue
        # Family caps: max 1 bottom, 1 footwear, 1 bag, 1 belt; max 2
        # accessories total. ``belt`` lives in the accessory bucket so the
        # accessory cap covers a second belt, but we cap belt by family too.
        cap = _OWNED_BUCKET_CAPS.get(bucket)
        if cap is not None and bucket_counts.get(bucket, 0) >= cap:
            skipped_family_cap += 1
            continue
        if family == "belt" and bucket_counts.get("__belt", 0) >= 1:
            skipped_family_cap += 1
            continue
        seen_ids.add(identifier)
        seen_names.add(norm_name)
        if image:
            seen_images.add(image)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if family == "belt":
            bucket_counts["__belt"] = bucket_counts.get("__belt", 0) + 1
        owned.append(
            {
                "id": match.get("id", ""),
                "name": match["name"],
                "family": bucket,
                "category": match.get("category", ""),
                "image_url": match.get("image_url", ""),
                "owned": True,
            }
        )
    logger.info(
        "AHVI_OWNED_ITEMS_GUARD status=ok requested=%d matched=%d output=%d "
        "skipped_duplicate=%d skipped_family_cap=%d skipped_dress_conflict=%d "
        "accessories=%d",
        len(pieces),
        matched,
        len(owned),
        skipped_duplicate,
        skipped_family_cap,
        skipped_dress_conflict,
        bucket_counts.get("accessory", 0),
    )
    return owned, len(owned), len(pieces)


# ---------------------------------------------------------------------------
# Shared Style Brain — post-LLM visual guard (Phase C).
# Gated by STYLE_SHARED_BRAIN (default OFF). Reuses the wardrobe path's pure
# occasion guard + the existing gender sanitizer, and adds a small color-clash
# strip and a weather-aware headwear strip. Never returns empty directions.
# ---------------------------------------------------------------------------

_WINTER_HEADWEAR: tuple[str, ...] = (
    "beanie",
    "wool hat",
    "woolen hat",
    "knit cap",
    "knit hat",
    "knitted hat",
    "skull cap",
    "earmuff",
    "ear muff",
    "balaclava",
    "ushanka",
)

_COLD_TOKENS: tuple[str, ...] = (
    "cold",
    "snow",
    "winter",
    "freezing",
    "chilly",
    "frost",
    "sleet",
)


def _shared_brain_enabled() -> bool:
    """Single feature-flag gate. Default OFF so production is unchanged until
    ops opt in via ``STYLE_SHARED_BRAIN=true``."""
    return os.getenv("STYLE_SHARED_BRAIN", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _weather_is_cold(weather: Any) -> bool:
    """True only when we have positive evidence of cold weather. Absent or
    ambiguous weather is treated as NOT cold (so winter headwear is stripped)."""
    if not weather:
        return False
    if isinstance(weather, str):
        return any(t in weather.lower() for t in _COLD_TOKENS)
    if isinstance(weather, dict):
        blob = " ".join(
            str(weather.get(k) or "")
            for k in ("condition", "summary", "description", "weather")
        ).lower()
        if any(t in blob for t in _COLD_TOKENS):
            return True
        for k in ("temp_c", "temperature", "temp", "feels_like_c"):
            v = weather.get(k)
            if v is None:
                continue
            try:
                if float(v) <= 12.0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _direction_dominant_color(direction: Dict[str, Any]) -> str:
    palette = _safe_list(direction.get("palette") or direction.get("colors"), limit=6)
    for entry in palette:
        c = str(entry or "").strip().lower()
        if c:
            return c
    hero_colors = _extract_simple_colors(direction.get("hero_piece"))
    return next(iter(hero_colors), "")


def _strip_color_clashes(direction: Dict[str, Any], stats: Dict[str, int]) -> Dict[str, Any]:
    """Drop palette entries and supporting assets that clash with the
    direction's dominant color. Assets with no color are skipped (logged) so
    the guard degrades safely on uncolored metadata."""
    try:
        from brain.engines.styling.color_compatibility import colors_clash
    except Exception:  # noqa: BLE001
        return direction

    out = dict(direction)
    dominant = _direction_dominant_color(out)
    if not dominant:
        logger.info(
            "AHVI_VISUAL_GUARD_COLOR_SKIP_NOCOLOR scope=direction title=%r",
            out.get("title"),
        )
        return out

    def _clean_color_list(values: List[Any]) -> List[Any]:
        kept: List[Any] = []
        for entry in values:
            c = str(entry or "").strip().lower()
            if c and c != dominant and colors_clash(dominant, c):
                stats["color_drop"] = stats.get("color_drop", 0) + 1
                logger.info(
                    "AHVI_VISUAL_GUARD_COLOR_CLASH scope=palette dominant=%s color=%s title=%r",
                    dominant,
                    c,
                    out.get("title"),
                )
                continue
            kept.append(entry)
        return kept

    if isinstance(out.get("palette"), list):
        out["palette"] = _clean_color_list(out["palette"])
    if isinstance(out.get("colors"), list):
        out["colors"] = _clean_color_list(out["colors"])

    complete = out.get("complete_the_look")
    if isinstance(complete, list):
        kept_assets: List[Any] = []
        for item in complete:
            if not isinstance(item, dict):
                kept_assets.append(item)
                continue
            color = str(item.get("color") or "").strip().lower()
            if not color:
                color = next(iter(_extract_simple_colors(item.get("name"))), "")
            if not color:
                logger.info(
                    "AHVI_VISUAL_GUARD_COLOR_SKIP_NOCOLOR scope=support name=%r",
                    item.get("name"),
                )
                kept_assets.append(item)
                continue
            if color != dominant and colors_clash(dominant, color):
                stats["color_drop"] = stats.get("color_drop", 0) + 1
                logger.info(
                    "AHVI_VISUAL_GUARD_COLOR_CLASH scope=support dominant=%s color=%s name=%r",
                    dominant,
                    color,
                    item.get("name"),
                )
                continue
            kept_assets.append(item)
        out["complete_the_look"] = kept_assets
    return out


def _strip_weather_inappropriate_headwear(
    direction: Dict[str, Any], ctx: Dict[str, Any], stats: Dict[str, int]
) -> Dict[str, Any]:
    """Remove winter headwear (beanie etc.) unless the weather is actually
    cold. Keeps everything else untouched."""
    if _weather_is_cold((ctx or {}).get("weather")):
        return direction

    def _is_winter(text: Any) -> bool:
        t = str(text or "").lower()
        return any(term in t for term in _WINTER_HEADWEAR)

    out = dict(direction)
    for key in ("items", "pieces"):
        vals = out.get(key)
        if isinstance(vals, list):
            new_vals = [v for v in vals if not _is_winter(v)]
            if len(new_vals) != len(vals):
                stats["weather_drop"] = stats.get("weather_drop", 0) + (len(vals) - len(new_vals))
                logger.info(
                    "AHVI_VISUAL_GUARD_WEATHER_DROP scope=%s title=%r",
                    key,
                    out.get("title"),
                )
            out[key] = new_vals

    if _is_winter(out.get("hero_piece")):
        fallback_items = _safe_list(out.get("items") or out.get("pieces"), limit=6)
        out["hero_piece"] = fallback_items[0] if fallback_items else ""
        stats["weather_drop"] = stats.get("weather_drop", 0) + 1
        logger.info(
            "AHVI_VISUAL_GUARD_WEATHER_DROP scope=hero title=%r",
            out.get("title"),
        )

    complete = out.get("complete_the_look")
    if isinstance(complete, list):
        new_complete: List[Any] = []
        for item in complete:
            name = item.get("name") if isinstance(item, dict) else item
            if _is_winter(name):
                stats["weather_drop"] = stats.get("weather_drop", 0) + 1
                logger.info(
                    "AHVI_VISUAL_GUARD_WEATHER_DROP scope=support name=%r", name
                )
                continue
            new_complete.append(item)
        out["complete_the_look"] = new_complete
    return out


def _direction_to_guard_board(direction: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a visual direction (string pieces + hero + dict supporting assets)
    into the board shape ``reject_board_for_occasion`` inspects, so the occasion
    guard sees the direction's actual garments — not just its title text."""
    pieces = _safe_list(direction.get("items") or direction.get("pieces"), limit=8)
    items: List[Dict[str, Any]] = [{"name": str(p)} for p in pieces]
    hero = _asset_text(direction.get("hero_piece"))
    if hero:
        items.append({"name": hero})
    for c in direction.get("complete_the_look") or []:
        if isinstance(c, dict):
            items.append(
                {
                    "name": _asset_text(c.get("name")),
                    "color": _asset_text(c.get("color")),
                    "category": _asset_text(c.get("category")),
                }
            )
    return {
        "title": direction.get("title"),
        "badge": direction.get("badge"),
        "occasion": direction.get("occasion"),
        "explanation": direction.get("description") or direction.get("short_note"),
        "why": direction.get("why"),
        "why_it_works": direction.get("why_it_works"),
        "style_direction": direction.get("style_direction"),
        "items": items,
    }


def _repair_direction_for_occasion(
    direction: Dict[str, Any], occ: str
) -> Dict[str, Any] | None:
    """Try to repair an occasion-rejected direction by stripping the offending
    piece(s). Returns the repaired direction, or None if it still rejects."""
    try:
        from brain.engines.outfit_quality_guard import reject_board_for_occasion
    except Exception:  # noqa: BLE001
        return direction

    out = dict(direction)
    reject, reason = reject_board_for_occasion(_direction_to_guard_board(out), occ)
    if not reject:
        return out

    term = ""
    if "_forbidden_" in reason:
        term = reason.split("_forbidden_", 1)[1].replace("_", " ").strip()
    if not term:
        # Reason not tied to a specific item (e.g. metadata/private-wear) —
        # unrepairable, drop the whole direction.
        return None

    def _hit(text: Any) -> bool:
        return bool(term) and term in str(text or "").lower()

    for key in ("items", "pieces"):
        vals = out.get(key)
        if isinstance(vals, list):
            out[key] = [v for v in vals if not _hit(v)]

    complete = out.get("complete_the_look")
    if isinstance(complete, list):
        out["complete_the_look"] = [
            i for i in complete if not _hit(i.get("name") if isinstance(i, dict) else i)
        ]

    if _hit(out.get("hero_piece")):
        items = _safe_list(out.get("items") or out.get("pieces"), limit=6)
        out["hero_piece"] = items[0] if items else ""

    reject_again, _ = reject_board_for_occasion(_direction_to_guard_board(out), occ)
    if reject_again:
        return None
    return out


def _strip_forbidden_item_signals(
    direction: Dict[str, Any], signals: List[str], stats: Dict[str, int]
) -> Dict[str, Any]:
    """Remove pieces/hero/supporting assets whose text hits a forbidden item
    signal (e.g. kurta/bandhgala/mojari for a non-ethnic occasion)."""
    if not signals:
        return direction

    def _hit(text: Any) -> bool:
        t = str(text or "").lower()
        return any(s in t for s in signals)

    out = dict(direction)
    for key in ("items", "pieces"):
        vals = out.get(key)
        if isinstance(vals, list):
            kept = [v for v in vals if not _hit(v)]
            if len(kept) != len(vals):
                stats["item_veto"] = stats.get("item_veto", 0) + (len(vals) - len(kept))
                logger.info(
                    "visual_guard.repair reason=forbidden_item scope=%s title=%r", key, out.get("title")
                )
            out[key] = kept
    if _hit(out.get("hero_piece")):
        items = _safe_list(out.get("items") or out.get("pieces"), limit=6)
        out["hero_piece"] = items[0] if items else ""
        stats["item_veto"] = stats.get("item_veto", 0) + 1
    complete = out.get("complete_the_look")
    if isinstance(complete, list):
        out["complete_the_look"] = [
            i for i in complete if not _hit(i.get("name") if isinstance(i, dict) else i)
        ]
    return out


def _estimate_text_formality(text: Any) -> int:
    """Formality 1..5 estimate for a single garment text (brief scoring path)."""
    af, _, _ = _asset_axis_estimates(str(text or "").lower())
    return af


def _repair_direction_for_formality(
    direction: Dict[str, Any], brief_formality: float, stats: Dict[str, int]
) -> Dict[str, Any] | None:
    """Pre-render authenticity gate. Any piece whose formality exceeds
    ``brief.formality + 2`` is over-formal for the occasion (e.g. an oxford /
    formal blazer [5] on a music_festival [2]) and is stripped. If nothing
    wearable remains the board is dropped so the guard's fallback keeps the
    originals (never blanks the screen). Off-by-construction for high-formality
    occasions: threshold>=max-estimate (5) means no piece can exceed it."""
    threshold = brief_formality + 2
    pieces = _safe_list(direction.get("items") or direction.get("pieces"), limit=10)
    hero = _asset_text(direction.get("hero_piece"))
    blobs = [str(p) for p in pieces if str(p).strip()]
    if hero:
        blobs.append(hero)
    if not blobs:
        return direction
    over = [b for b in blobs if _estimate_text_formality(b) > threshold]
    if not over:
        return direction

    out = dict(direction)
    for key in ("items", "pieces"):
        vals = out.get(key)
        if isinstance(vals, list):
            kept = [v for v in vals if _estimate_text_formality(v) <= threshold]
            if len(kept) != len(vals):
                stats["formality_repair"] = stats.get("formality_repair", 0) + (len(vals) - len(kept))
            out[key] = kept
    if _estimate_text_formality(out.get("hero_piece")) > threshold:
        items = _safe_list(out.get("items") or out.get("pieces"), limit=6)
        out["hero_piece"] = items[0] if items else ""
    remaining = _safe_list(out.get("items") or out.get("pieces"), limit=10)
    if not remaining and not _asset_text(out.get("hero_piece")):
        logger.info(
            "visual_guard.reject reason=formality title=%r brief_formality=%s",
            direction.get("title"), brief_formality,
        )
        return None
    logger.info(
        "visual_guard.repair reason=formality title=%r brief_formality=%s stripped=%d",
        out.get("title"), brief_formality, len(over),
    )
    return out


def _apply_style_guard(
    directions: List[Dict[str, Any]], ctx: Dict[str, Any] | None
) -> List[Dict[str, Any]]:
    """Post-generation visual guard: occasion -> color -> weather -> gender.
    No-op unless STYLE_SHARED_BRAIN is enabled. Never returns empty (falls back
    to the original directions if every direction is rejected)."""
    if not _shared_brain_enabled():
        return directions
    if not isinstance(directions, list) or not directions:
        return directions
    try:
        from brain.engines.outfit_quality_guard import (
            normalize_occasion,
            reject_board_for_occasion,
        )
    except Exception:  # noqa: BLE001
        return directions

    ctx = ctx or {}
    occ = normalize_occasion(str(ctx.get("canonical_occasion") or ""))
    gender = str(ctx.get("gender") or "unknown")
    query = str(ctx.get("_query") or "")
    allow_fem = gender == "female" or _prompt_allows_gendered_feminine_style(query)

    out: List[Dict[str, Any]] = []
    stats = {
        "occ_reject": 0,
        "color_drop": 0,
        "weather_drop": 0,
        "gender_drop": 0,
        "kept": 0,
    }
    for direction in directions:
        if not isinstance(direction, dict):
            out.append(direction)
            continue
        d = direction
        # 0. Canonical-brief veto: forbidden archetypes / item signals
        # (e.g. ethnic kurta/bandhgala/mojari on a music festival). Drop a whole
        # direction whose archetype is forbidden; strip forbidden item signals.
        forbidden_arch = {str(a).strip().lower() for a in (ctx.get("forbidden_archetypes") or []) if str(a).strip()}
        forbidden_items = [str(s).strip().lower() for s in (ctx.get("forbidden_item_signals") or []) if str(s).strip()]
        if forbidden_arch:
            arch = str(d.get("archetype") or d.get("direction_name") or d.get("title") or "").strip().lower()
            if arch and arch in forbidden_arch:
                stats["arch_veto"] = stats.get("arch_veto", 0) + 1
                logger.info(
                    "visual_guard.reject reason=forbidden_archetype title=%r archetype=%s",
                    d.get("title"), arch,
                )
                continue
        if forbidden_items:
            d = _strip_forbidden_item_signals(d, forbidden_items, stats)
        # 0b. Formality authenticity gate — board mean formality must not exceed
        # brief.formality + 2 (kills oxford/loafer/belt drift on a festival).
        brief_formality = ctx.get("formality")
        if isinstance(brief_formality, (int, float)):
            repaired_f = _repair_direction_for_formality(d, float(brief_formality), stats)
            if repaired_f is None:
                continue
            d = repaired_f
        # 1. Occasion compatibility (whole-direction) — reuse the wardrobe guard.
        reject, reason = reject_board_for_occasion(_direction_to_guard_board(d), occ)
        if reject:
            stats["occ_reject"] += 1
            logger.info(
                "visual_guard.reject AHVI_VISUAL_GUARD_OCC_REJECT title=%r occ=%s reason=%s",
                d.get("title"),
                occ,
                reason,
            )
            repaired = _repair_direction_for_occasion(d, occ)
            if repaired is None:
                continue
            logger.info(
                "visual_guard.repair title=%r occ=%s reason=%s",
                d.get("title"),
                occ,
                reason,
            )
            d = repaired
        # 2. Color clash.
        d = _strip_color_clashes(d, stats)
        # 3. Weather-aware headwear.
        d = _strip_weather_inappropriate_headwear(d, ctx, stats)
        # 4. Gender — reuse the existing sanitizer already used during enrichment.
        before = len(_safe_list(d.get("items") or d.get("pieces"), limit=8))
        d = _sanitize_direction_for_gender(
            d, target_gender=gender, allow_feminine=allow_fem
        )
        after = len(_safe_list(d.get("items") or d.get("pieces"), limit=8))
        if after < before:
            stats["gender_drop"] += before - after
            logger.info(
                "AHVI_VISUAL_GUARD_GENDER_DROP title=%r gender=%s dropped=%d",
                d.get("title"),
                gender,
                before - after,
            )
        out.append(d)
        stats["kept"] += 1

    logger.info(
        "visual_guard.summary AHVI_VISUAL_GUARD_SUMMARY occ=%s in=%d kept=%d stats=%s",
        occ,
        len(directions),
        len(out),
        stats,
    )
    # Never blank the screen: if every direction was rejected, keep the originals.
    return out or directions


def _recognized_archetype_names() -> set[str]:
    """Authoritative set of archetype names we treat as already-curated.

    Any archetype on a direction whose (case/space-normalised) name is in this
    set is considered a genuine, intentional archetype and is PRESERVED as-is.
    Anything outside the set is free-text the LLM invented (e.g. "Polished
    Daily") and gets remapped to the best-fit curated-library name.

    The set is the union of:
      * the persona / visual-inspiration registry (`ARCHETYPE_LIBRARY` — the
        source `select_archetypes` draws from; includes "Creative Executive",
        "Approachable Executive", "Resort Sophisticate", and every festive /
        ceremony persona),
      * the generic curated visual strategies (`_GENERIC_VISUAL_STRATEGIES`),
      * the occasion-specific festive replacement archetypes
        (`_festive_visual_direction_replacements()`).
    """
    global _RECOGNIZED_ARCHETYPE_NAMES_CACHE
    cached = globals().get("_RECOGNIZED_ARCHETYPE_NAMES_CACHE")
    if cached is not None:
        return cached
    names: set[str] = set()

    def _add(raw: Any) -> None:
        norm = _normalize_archetype_key(raw)
        if norm:
            names.add(norm)

    try:
        for entry in ARCHETYPE_LIBRARY or []:
            if isinstance(entry, dict):
                _add(entry.get("name"))
    except Exception:  # noqa: BLE001
        pass
    for entry in _GENERIC_VISUAL_STRATEGIES:
        _add(entry.get("archetype"))
    try:
        for entry in _festive_visual_direction_replacements():
            _add(entry.get("archetype"))
    except Exception:  # noqa: BLE001
        pass
    _RECOGNIZED_ARCHETYPE_NAMES_CACHE = names
    return names


def _normalize_archetype_key(value: Any) -> str:
    """Lower-case, whitespace-collapsed key for archetype-name comparison."""
    text = _clean_direction_title(value)
    if not text:
        return ""
    return " ".join(text.lower().split())


_RECOGNIZED_ARCHETYPE_NAMES_CACHE: set[str] | None = None


def _is_recognized_archetype(value: Any) -> bool:
    key = _normalize_archetype_key(value)
    if not key:
        return False
    return key in _recognized_archetype_names()


def _assign_curated_archetype(direction: Dict[str, Any], occasion: Any) -> str:
    """Deterministically pick a curated-library archetype for a direction.

    Rule-based (NO LLM). Scores the direction's hero piece / items / palette /
    title against the curated library entries and returns the best-fit
    archetype name. For festive/traditional occasions the festive archetype set
    is preferred. ALWAYS returns a non-empty name.
    """
    if not isinstance(direction, dict):
        return _GENERIC_VISUAL_STRATEGIES[0]["archetype"]

    is_festive = _visual_occasion_family(occasion) == "indian_festive"

    # Build the candidate library: festive set first when the occasion calls for
    # it, otherwise the generic set. We score every candidate and pick the best.
    if is_festive:
        candidates = _festive_visual_direction_replacements()
        fallback = candidates[0]["archetype"]
    else:
        candidates = _GENERIC_VISUAL_STRATEGIES
        fallback = candidates[0]["archetype"]

    # Tokenise the incoming direction's descriptive signal.
    dir_blob = " ".join(
        [
            _asset_text(direction.get("hero_piece") or direction.get("heroPiece")),
            " ".join(_safe_list(direction.get("items") or direction.get("pieces"), limit=8)),
            _asset_text(direction.get("title")),
            _asset_text(direction.get("subtitle")),
        ]
    )
    dir_tokens = _style_tokens(dir_blob)
    dir_palette = set(_style_tokens(" ".join(_safe_list(direction.get("palette") or direction.get("colors"), limit=6))))
    dir_hero_cat = _style_category(direction.get("hero_piece") or direction.get("heroPiece"))

    best_name = fallback
    best_score = -1.0
    for cand in candidates:
        cand_blob = " ".join(
            [
                _asset_text(cand.get("hero_piece")),
                " ".join(_safe_list(cand.get("items"), limit=8)),
                _asset_text(cand.get("title")),
            ]
        )
        cand_tokens = _style_tokens(cand_blob)
        cand_palette = set(_style_tokens(" ".join(_safe_list(cand.get("palette"), limit=6))))
        score = 0.0
        # Garment/term overlap is the primary signal.
        if dir_tokens and cand_tokens:
            score += 2.0 * len(dir_tokens & cand_tokens) / max(1, len(cand_tokens))
        # Palette overlap is a secondary signal.
        if dir_palette and cand_palette:
            score += 1.0 * len(dir_palette & cand_palette) / max(1, len(cand_palette))
        # Hero category match is a strong tie-breaker.
        if dir_hero_cat and dir_hero_cat == _style_category(cand.get("hero_piece")):
            score += 1.5
        if score > best_score:
            best_score = score
            best_name = str(cand.get("archetype") or fallback)

    return _clean_direction_title(best_name) or fallback


def _apply_editorial_polish(
    directions: List[Dict[str, Any]],
    *,
    occasion: Any,
    wardrobe_items: Any,
    context_text: Any = "",
) -> List[Dict[str, Any]]:
    """Decorate each direction with editorial UX fields. Additive only."""
    normalised_wardrobe = _wardrobe_normalised(wardrobe_items)
    has_wardrobe_signal = bool(normalised_wardrobe)
    out: List[Dict[str, Any]] = []
    used_notes: set[str] = set()
    for index, direction in enumerate(directions or []):
        if not isinstance(direction, dict):
            out.append(direction)
            continue
        polished = dict(direction)
        polished["title"] = _clean_direction_title(polished.get("title"))
        polished["archetype"] = _clean_direction_title(polished.get("archetype"))
        # Curated-archetype assignment. Two cases get the best-fit library
        # archetype assigned deterministically:
        #   1. the direction arrived WITHOUT an archetype (wardrobe /
        #      visual-inspiration paths often omit it), and
        #   2. the direction carries a FREE-TEXT archetype the LLM invented
        #      that is not in any recognized registry (e.g. "Polished Daily").
        # Genuine persona / curated / occasion archetypes (anything in
        # `_recognized_archetype_names()` — Creative Executive, Approachable
        # Executive, Resort Sophisticate, Festive Heritage, etc.) are
        # AUTHORITATIVE and preserved exactly as-is.
        if not polished["archetype"] or not _is_recognized_archetype(polished["archetype"]):
            polished["archetype"] = _assign_curated_archetype(polished, occasion)
        archetype = polished["archetype"] or polished["title"]
        direction_name = _clean_direction_title(
            archetype or polished["title"] or "Curated Direction"
        )
        polished["direction_name"] = direction_name
        polished["adjectives"] = _direction_adjectives_from_archetype(archetype)
        # Cap stylist notes server-side so the client never renders walls of text.
        polished["why_it_works"] = _two_sentences(polished.get("why_it_works"))
        polished["why_this_works"] = _two_sentences(
            polished.get("why_this_works") or polished.get("why_it_works")
        )
        # short_note leads with the human stylist voice for this occasion so
        # the card never reads like generic LLM prose. The model's reasoning
        # stays available (capped) in why_it_works for detail.
        short_note = _direction_short_note(
            polished,
            occasion=occasion,
            context_text=context_text,
            index=index,
        )
        if short_note in used_notes:
            short_note = _limit_words(
                f"{direction_name} gives this board its own practical, occasion-ready character."
            )
        used_notes.add(short_note)
        polished["short_note"] = short_note
        if _asset_text(polished.get("description")) == _DEFAULT_OCCASION_VOICE:
            polished["description"] = short_note
        if polished.get("style_note"):
            polished["style_note"] = _two_sentences(polished.get("style_note"), max_chars=120)
        # Ownership truth: only real wardrobe matches count.
        owned_items, owned_count, total_items = _build_owned_items(
            polished, normalised_wardrobe
        )
        polished["owned_items"] = owned_items
        polished["owned_count"] = owned_count
        polished["total_items"] = total_items
        if has_wardrobe_signal and total_items > 0:
            match_pct = int(round((owned_count / total_items) * 100))
        else:
            match_pct = _wardrobe_match_pct(polished, wardrobe_items)
        polished["wardrobe_match_pct"] = match_pct
        polished["badge"] = _editorial_badge(match_pct)
        polished["curated_for"] = _curated_for_for_occasion(occasion)
        polished["complete_the_look_copy"] = _complete_the_look_copy(
            polished.get("missing_piece"), occasion
        )
        out.append(polished)
    return out


def _build_editorial_cover(
    directions: List[Dict[str, Any]],
    *,
    occasion: Any,
) -> Dict[str, Any]:
    """Top-of-response magazine cover summary."""
    label = _occasion_label_for_editorial(occasion)
    top = next((d for d in directions or [] if isinstance(d, dict)), None) or {}
    direction_name = _clean_direction_title(
        top.get("direction_name")
        or top.get("archetype")
        or top.get("title")
        or "Curated Look"
    )
    match_pcts = [d.get("wardrobe_match_pct") for d in directions or [] if isinstance(d, dict)]
    match_pcts = [p for p in match_pcts if isinstance(p, int)]
    match_pct = max(match_pcts) if match_pcts else None
    return {
        "occasion_label": label,
        "direction_name": direction_name,
        "wardrobe_match_pct": match_pct,
        "curated_for": _curated_for_for_occasion(occasion),
        "badge": _editorial_badge(match_pct),
    }


# ---------------------------------------------------------------------
# End editorial UX polish
# ---------------------------------------------------------------------


def _normalize_visual_directions(
    value: Any,
    mode: str,
    category: str | None,
    archetypes: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    fallbacks = _fallback_visual_directions(mode, category)
    rows = value if isinstance(value, list) else []
    selected = [a for a in (archetypes or []) if isinstance(a, dict)]
    out: List[Dict[str, Any]] = []
    for idx in range(3):
        source = rows[idx] if idx < len(rows) else {}
        direction = _ensure_direction_logic(_normalize_direction(source, fallbacks[idx]))
        if selected:
            arch = selected[idx % len(selected)]
            arch_name = str(arch.get("name") or "").strip()
            if arch_name:
                raw_archetype = str((source or {}).get("archetype") or "").strip() if isinstance(source, dict) else ""
                # Preserve a meaningful generated archetype (e.g. Gemini returns
                # "Church Formal"). Only fall back to the library archetype when
                # the model gave us nothing usable. Blindly overwriting collapsed
                # varied directions into the same handful of library labels.
                keep_generated = bool(raw_archetype) and _norm(raw_archetype) not in {
                    "general", "default", "none", "unknown", "n/a",
                }
                applied = _clean_direction_title(
                    raw_archetype if keep_generated else arch_name
                )
                direction["archetype"] = applied
                # Enrich from the library archetype only where the generated
                # direction is missing fields — don't clobber good model output.
                if not direction.get("impression"):
                    direction["impression"] = ", ".join(str(x) for x in (arch.get("impression") or []) if str(x).strip())
                if not direction.get("style_keywords"):
                    direction["style_keywords"] = [str(x) for x in (arch.get("style_keywords") or []) if str(x).strip()][:5]
                if not direction.get("palette"):
                    direction["palette"] = [str(x) for x in (arch.get("palette") or []) if str(x).strip()][:5]
                direction["why_this_works"] = direction.get("why_it_works") or direction.get("style_note") or ""
                logger.info(
                    "AHVI_VISUAL_ARCHETYPE_APPLIED index=%d requested=%r applied=%r kept_generated=%s title=%r",
                    idx,
                    raw_archetype,
                    applied,
                    keep_generated,
                    direction.get("title"),
                )
        out.append(direction)
    return _apply_generic_visual_diversity(out, category=category)


def _fallback_pairing_routes(anchor: Dict[str, str]) -> List[Dict[str, Any]]:
    name = str(anchor.get("name") or "the item").strip()
    category = str(anchor.get("category") or "").strip()
    color = str(anchor.get("color") or "").strip()
    base = name if name != "the item" else "this piece"
    if category == "footwear" or any(x in name for x in ("loafer", "sneaker", "shoe", "boot")):
        return [
            {
                "title": "Smart Casual",
                "use_case": "office-adjacent days",
                "strategy": "Use the shoes to sharpen relaxed separates.",
                "items": [base, "button-down shirt", "chinos", "simple watch"],
                "palette": [color or "black", "white", "navy", "tan"],
                "why_it_works": "The footwear adds polish, while chinos keep it from becoming fully formal.",
                "avoid": ["Overly shiny belts", "matching everything in the same dark tone"],
                "styling_tip": "Let the trouser hem sit cleanly on the shoe.",
            },
            {
                "title": "Office Clean",
                "use_case": "meetings and workdays",
                "strategy": "Lean into structure without going stiff.",
                "items": [base, "crisp shirt", "tailored trousers", "belt"],
                "palette": [color or "black", "blue", "grey"],
                "why_it_works": "A structured base makes the footwear feel intentional and credible.",
                "avoid": ["Shorts", "athletic socks", "loud patterned trousers"],
                "styling_tip": "Match the belt mood, not necessarily the exact color.",
            },
            {
                "title": "Evening Minimal",
                "use_case": "dinner or drinks",
                "strategy": "Keep the outfit tonal and let texture do the work.",
                "items": [base, "dark shirt", "straight trousers"],
                "palette": [color or "black", "charcoal", "cream"],
                "why_it_works": "The darker palette gives evening polish without needing a statement piece.",
                "avoid": ["Corporate blazer unless the setting is formal"],
                "styling_tip": "Open the collar slightly to soften the polish.",
            },
            {
                "title": "Weekend Neat",
                "use_case": "casual plans",
                "strategy": "Use one clean top so the shoes do not feel overdressed.",
                "items": [base, "plain tee or polo", "denim", "light overshirt"],
                "palette": [color or "black", "stone", "blue"],
                "why_it_works": "Casual layers make polished footwear feel relaxed and wearable.",
                "avoid": ["Gym shorts", "wrinkled oversized tees"],
                "styling_tip": "Choose denim with a clean wash rather than heavy distressing.",
            },
        ]
    if "blazer" in name or category == "outerwear":
        return [
            {
                "title": "T-Shirt Tailoring",
                "use_case": "casual dinner",
                "strategy": "Break the blazer's corporate signal with a clean tee.",
                "items": [base, "plain tee", "straight denim", "minimal sneakers"],
                "palette": [color or "navy", "white", "blue"],
                "why_it_works": "The tee and sneakers relax the blazer while the jacket keeps shape.",
                "avoid": ["Dress shirt plus formal trousers if you want casual"],
                "styling_tip": "Keep the tee neckline clean and the blazer unbuttoned.",
            },
            {
                "title": "Knit Softness",
                "use_case": "smart casual weekends",
                "strategy": "Swap office shirting for soft texture.",
                "items": [base, "fine knit", "chinos", "suede loafers"],
                "palette": [color or "navy", "cream", "brown"],
                "why_it_works": "Knitwear makes the blazer feel warm and easy instead of corporate.",
                "avoid": ["Tie", "stiff dress shoes"],
                "styling_tip": "Use a thinner knit so the shoulder line stays smooth.",
            },
            {
                "title": "Denim Contrast",
                "use_case": "creative casual",
                "strategy": "Let denim pull the blazer down a notch.",
                "items": [base, "casual shirt", "dark denim", "clean sneakers"],
                "palette": [color or "charcoal", "blue", "white"],
                "why_it_works": "Denim adds ease while the blazer keeps the outfit intentional.",
                "avoid": ["Ripped denim with a formal blazer"],
                "styling_tip": "Keep the denim straight or slim, not baggy.",
            },
            {
                "title": "Summer Relaxed",
                "use_case": "warm evenings",
                "strategy": "Pair the blazer with breathable pieces.",
                "items": [base, "linen shirt", "light chinos", "loafers"],
                "palette": [color or "navy", "ecru", "tan"],
                "why_it_works": "Light fabrics stop the blazer from feeling too serious.",
                "avoid": ["Heavy wool trousers"],
                "styling_tip": "Roll sleeves only if the blazer fabric is relaxed enough.",
            },
        ]
    return [
        {
            "title": "Smart Casual",
            "use_case": "office-adjacent or polished daily",
            "strategy": "Use clean structure around the anchor.",
            "items": [base, "chinos or tailored trousers", "loafers or minimal sneakers", "simple watch"],
            "palette": [color or "white", "navy", "tan"],
            "why_it_works": "The base feels intentional without pushing the outfit into full formal.",
            "avoid": ["Too many statement accessories", "overly shiny shoes"],
            "styling_tip": "Keep one piece relaxed so the outfit stays modern.",
        },
        {
            "title": "Business Casual",
            "use_case": "meetings and workdays",
            "strategy": "Add sharper bottoms and restrained footwear.",
            "items": [base, "structured trousers", "belt", "loafers"],
            "palette": [color or "white", "grey", "brown"],
            "why_it_works": "The structured pieces make the anchor read professional rather than plain.",
            "avoid": ["Loud prints near the anchor"],
            "styling_tip": "Tuck only if the trouser waistband looks clean.",
        },
        {
            "title": "Weekend Clean",
            "use_case": "coffee, errands, casual lunch",
            "strategy": "Relax the anchor with denim and easy footwear.",
            "items": [base, "straight denim", "clean sneakers", "light overshirt"],
            "palette": [color or "white", "blue", "stone"],
            "why_it_works": "Denim makes the anchor feel approachable while the clean shoe keeps polish.",
            "avoid": ["Distressed denim if the anchor is already crisp"],
            "styling_tip": "Leave a little ease in the fit.",
        },
        {
            "title": "Evening Minimal",
            "use_case": "dinner or drinks",
            "strategy": "Use contrast and a darker base.",
            "items": [base, "dark trousers", "sleek shoes", "minimal accessory"],
            "palette": [color or "white", "black", "charcoal"],
            "why_it_works": "The darker pieces make the anchor feel deliberate and evening-ready.",
            "avoid": ["Office-heavy layering"],
            "styling_tip": "Keep accessories quiet so the contrast does the work.",
        },
        {
            "title": "Summer Relaxed",
            "use_case": "warm days or vacations",
            "strategy": "Pair the anchor with breathable textures.",
            "items": [base, "linen or cotton bottom", "sandals or canvas sneakers"],
            "palette": [color or "white", "ecru", "olive"],
            "why_it_works": "Lighter texture keeps the anchor fresh and relaxed.",
            "avoid": ["Heavy formal trousers in heat"],
            "styling_tip": "Use softer colors if the fabric is crisp.",
        },
    ]


def _pairing_ctas(anchor: Dict[str, Any], routes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """CTAs that carry the selected route + anchor INTO the follow-up query, so
    a stateless follow-up still builds the right thing without server session."""
    name = str((anchor or {}).get("name") or "this piece").strip()
    top = str((routes[0].get("title") if routes else "") or "").strip()
    suffix = f"{top} with {name}".strip() if top else name
    return [
        {"label": "Use my wardrobe", "value": f"Use my wardrobe to build {suffix}"},
        {"label": "Show visual inspiration", "value": f"Show visual inspiration for {suffix}"},
        {"label": "Find missing pieces", "value": f"Find missing pieces for {suffix}"},
    ]


def _pairing_last_style_context(anchor: Dict[str, Any], routes: List[Dict[str, Any]], context: dict) -> Dict[str, Any]:
    import time as _t

    ctx = {
        "last_style_mode": "style_pairing",
        "anchor_item": {
            "name": str((anchor or {}).get("name") or "").strip(),
            "category": str((anchor or {}).get("category") or "").strip(),
        },
        "selected_route": str(routes[0].get("title") if routes else "").strip(),
        "selected_archetypes": [str(r.get("archetype") or r.get("title") or "").strip() for r in routes][:5],
        "persona_context": (context or {}).get("persona") or {},
        "timestamp": int(_t.time()),
    }
    logger.info(
        "AHVI_PAIRING_CONTEXT_SAVED anchor=%r route=%s archetypes=%s",
        ctx["anchor_item"]["name"], ctx["selected_route"], ctx["selected_archetypes"],
    )
    return ctx


def _normalize_pairing_routes(
    value: Any,
    anchor: Dict[str, str],
    gender: str = "unknown",
    wardrobe: Any = None,
    selected_archetypes: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    fallbacks = _fallback_pairing_routes(anchor)
    selected = [a for a in (selected_archetypes or []) if isinstance(a, dict) and str(a.get("name") or "").strip()]
    allowed_names = [str(a.get("name") or "").strip() for a in selected]
    allowed_lookup = {name.lower(): a for name, a in zip(allowed_names, selected)}
    normalized: List[Dict[str, Any]] = []
    total_removed = 0
    _wardrobe = wardrobe if isinstance(wardrobe, list) else []
    for idx in range(min(5, max(4, len(rows), len(fallbacks)))):
        source = rows[idx] if idx < len(rows) and isinstance(rows[idx], dict) else {}
        fb = fallbacks[idx % len(fallbacks)]
        requested_archetype = str(source.get("archetype") or "").strip()
        applied_arch = selected[idx % len(selected)] if selected else {}
        if requested_archetype and requested_archetype.lower() in allowed_lookup:
            applied_arch = allowed_lookup[requested_archetype.lower()]
        applied_archetype = str(applied_arch.get("name") or requested_archetype or "").strip()
        if allowed_names:
            logger.info(
                "AHVI_ARCHETYPE_ENFORCED index=%d requested=%r applied=%r allowed=%s",
                idx,
                requested_archetype,
                applied_archetype,
                allowed_names,
            )
        items = _safe_list(source.get("items") or fb.get("items"), limit=8)
        avoid = _safe_list(source.get("avoid") or fb.get("avoid"), limit=5)
        # Deterministic persona safety net: strip feminine-only items for a male
        # persona even if Gemini slipped one in.
        try:
            from services.stylist_knowledge_service import filter_items_for_persona

            items, removed = filter_items_for_persona(items, gender)
            if removed:
                total_removed += len(removed)
                avoid = (avoid + removed)[:6]
        except Exception:  # noqa: BLE001
            pass
        route = {
            "title": str(source.get("title") or fb.get("title") or "Pairing Route").strip(),
            "archetype": applied_archetype,
            "impression_created": str(
                source.get("impression_created")
                or ", ".join(str(x) for x in (applied_arch.get("impression") or []) if str(x).strip())
            ).strip(),
            "use_case": str(source.get("use_case") or source.get("useCase") or fb.get("use_case") or "").strip(),
            "strategy": str(source.get("strategy") or fb.get("strategy") or "").strip(),
            "items": items,
            "palette": _safe_list(source.get("palette") or applied_arch.get("palette") or fb.get("palette"), limit=6),
            "why_it_works": str(source.get("why_it_works") or source.get("whyItWorks") or fb.get("why_it_works") or "").strip(),
            "avoid": avoid,
            "styling_tip": str(source.get("styling_tip") or source.get("style_note") or source.get("styleNote") or fb.get("styling_tip") or "").strip(),
            "persona_fit_reason": str(source.get("persona_fit_reason") or "").strip(),
            "archetype_reasoning": str(source.get("archetype_reasoning") or source.get("persona_fit_reason") or "").strip(),
        }
        # Wardrobe reality scoring — how buildable is this route from what they own.
        if _wardrobe:
            try:
                from services.stylist_knowledge_service import score_route_against_wardrobe

                route["wardrobe_reality"] = score_route_against_wardrobe(route, _wardrobe)
            except Exception:  # noqa: BLE001
                pass
        normalized.append(route)
    if total_removed:
        logger.info("AHVI_PAIRING_PERSONA_FILTER_APPLIED gender=%s removed=%d", gender, total_removed)
    logger.info("AHVI_PAIRING_ROUTES_BUILT count=%d titles=%s", len(normalized), [r["title"] for r in normalized])
    return normalized[:5]


def _pairing_routes_as_visual_directions(routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "archetype": route.get("archetype"),
            "title": route.get("title"),
            "subtitle": route.get("strategy"),
            "impression": route.get("impression_created"),
            "strategy": route.get("strategy"),
            "description": route.get("use_case"),
            "hero_piece": (route.get("items") or [""])[0] if isinstance(route.get("items"), list) else "",
            "hero_piece_reasoning": route.get("strategy"),
            "palette": route.get("palette") if isinstance(route.get("palette"), list) else [],
            "colors": route.get("palette") if isinstance(route.get("palette"), list) else [],
            "pieces": route.get("items") if isinstance(route.get("items"), list) else [],
            "items": route.get("items") if isinstance(route.get("items"), list) else [],
            "why_it_works": route.get("why_it_works"),
            "why_this_works": route.get("why_it_works"),
            "style_note": route.get("styling_tip"),
            "styling_tip": str(route.get("styling_tip") or "")[:80],
            "use_case": route.get("use_case"),
            "avoid": route.get("avoid") if isinstance(route.get("avoid"), list) else [],
            "archetype_reasoning": route.get("archetype_reasoning"),
            "dna_alignment": route.get("persona_fit_reason"),
            "wardrobe_alignment": route.get("wardrobe_reality") if isinstance(route.get("wardrobe_reality"), dict) else None,
        }
        for route in routes
    ]


def _gemini_reasoning(
    *,
    query: str,
    mode: str,
    category: str | None,
    user_profile: dict,
    context: dict,
) -> Dict[str, Any]:
    # Compact policy + style context (never the full rule libraries).
    policy: Dict[str, Any] = {}
    style_ctx: Dict[str, Any] = {}
    try:
        from brain.config_loader import get_style_policy_context

        policy = get_style_policy_context(
            intent=mode, occasion=str(context.get("occasion") or category or ""), mode=mode
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ahvi.style.policy_failed err=%s", str(exc)[:140])
    try:
        from services.style_context_service import build_style_context, compact_context_for_prompt

        full_ctx = build_style_context(
            query=query,
            occasion=str(context.get("occasion") or category or "") or None,
            mode=mode if mode in {"style_advice", "visual_inspiration", "wardrobe_style", "missing_pieces"} else "style_advice",
            wardrobe_items=context.get("wardrobe") or context.get("wardrobe_items"),
            weather=context.get("weather") or context.get("weather_context"),
            event_context=context.get("event_context"),
            user_profile=user_profile,
            last_style_context=context.get("last_style_context"),
            user_id=str(context.get("user_id") or ""),
        )
        style_ctx = compact_context_for_prompt(full_ctx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ahvi.style.context_failed err=%s", str(exc)[:140])

    # Wire the 4 intelligence configs as compact principle slices into the
    # policy block (outfit_validation + wardrobe_management always; mood +
    # inspiration contracts only for visual_inspiration).
    try:
        from brain import config_loader as _cfg

        policy["outfit_validation_principles"] = _cfg.get_outfit_validation_principles()
        policy["wardrobe_management_principles"] = _cfg.get_wardrobe_management_principles()
        policy["personalization_principles"] = _cfg.get_personalization_principles()
        policy["visual_response_principles"] = _cfg.get_visual_response_principles()
        policy["decision_principles"] = _cfg.get_decision_principles()
        if mode == VISUAL_INSPIRATION:
            policy["mood_board_contract"] = _cfg.get_mood_board_contract()
            policy["inspiration_board_contract"] = _cfg.get_inspiration_board_contract()
        _slices = [k for k in (
            "outfit_validation_principles", "wardrobe_management_principles",
            "personalization_principles", "visual_response_principles",
            "decision_principles", "mood_board_contract", "inspiration_board_contract",
        ) if k in policy]
        logger.info("AHVI_CONFIG_USAGE_AUDIT mode=%s slices=%s", mode, _slices)
        logger.info("AHVI_CONFIG_SLICES_USED count=%d slices=%s", len(_slices), _slices)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ahvi.style.config_slices_failed err=%s", str(exc)[:140])

    # Persona + archetype selection for visual style paths.
    persona = {}
    selected_archetypes = []
    if mode in {STYLE_PAIRING, VISUAL_INSPIRATION}:
        try:
            from services.style_context_service import build_pairing_persona
            from services.stylist_knowledge_service import select_archetypes

            _uprof = user_profile if isinstance(user_profile, dict) else {}
            persona = build_pairing_persona(
                user_profile=_uprof,
                style_dna=_uprof.get("style_dna") or _uprof.get("styleDNA"),
                wardrobe_summary=(style_ctx or {}).get("wardrobe_summary"),
            )
            # Only extract a pairing anchor for actual STYLE_PAIRING queries.
            # For VISUAL_INSPIRATION the query is occasion / mood phrasing
            # (e.g. "visual inspiration for a conference talk"), not a
            # garment anchor — passing it as an anchor poisons archetype
            # selection and emits a misleading AHVI_STYLE_PAIRING_ANCHOR log.
            _anchor = _extract_pairing_anchor(query) if mode == STYLE_PAIRING else {}
            _dna_raw = _uprof.get("style_dna") or _uprof.get("styleDNA") or {}
            _gender = _resolve_asset_gender(query=query, user_profile=_uprof)
            selected_archetypes = select_archetypes(
                anchor=_anchor,
                occasion=str(context.get("occasion") or category or ""),
                style_keywords=persona.get("style_dna") or [],
                style_dna=_dna_raw if isinstance(_dna_raw, dict) else {},
                gender=_gender,
            )
            logger.info(
                "AHVI_PERSONAL_STYLIST_CONTEXT_BUILT gender=%s archetypes=%s",
                _gender, [a.get("name") for a in selected_archetypes],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ahvi.pairing_persona_failed err=%s", str(exc)[:140])

    prompt = _build_reasoning_prompt(
        query=_clean_recursive_prompt(query),
        mode=mode,
        category=category,
        user_profile=user_profile,
        context=context,
        policy=policy,
        style_ctx=style_ctx,
        persona=persona,
        archetypes=selected_archetypes,
    )
    raw = generate_text(
        prompt,
        options={"temperature": 0.45, "max_output_tokens": 1600},
        user_profile=user_profile,
        signals={"context_mode": "style_reasoning", "style_mode": mode},
        usecase="style_reasoning",
    )
    logger.info("AHVI_STYLE_GEMINI_RAW_LEN usecase=style_reasoning len=%d", len(str(raw or "")))
    parsed = parse_json_object(raw)
    if isinstance(parsed, dict):
        parsed["_selected_archetypes"] = selected_archetypes
        parsed["_persona_context"] = persona
        logger.info(
            "AHVI_STYLE_GEMINI_PARSED_KEYS keys=%s",
            ",".join(sorted(parsed.keys()))[:240],
        )
    return parsed


_ADVICE_BLOCK_FIELDS = {
    "body_proportion_advice": ("principles", "do", "avoid", "outfit_examples"),
    "color_advice": ("recommended_colors", "avoid_colors", "why", "outfit_palettes"),
    "occasion_advice": ("do", "avoid", "better_alternatives", "styling_routes"),
}


def _build_advice_block(mode: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """Structured open-ended advice block (body_proportion / color / occasion)."""
    fields = _ADVICE_BLOCK_FIELDS.get(mode)
    if not fields:
        return None
    block: Dict[str, Any] = {"type": mode}
    has_any = False
    for f in fields:
        vals = _safe_list(payload.get(f), limit=6)
        block[f] = vals
        if vals:
            has_any = True
    block["summary"] = str(payload.get("stylist_reasoning") or "").strip()
    if not has_any:
        return None
    logger.info("AHVI_ADVICE_BLOCK_BUILT mode=%s fields=%s", mode, [f for f in fields if block.get(f)])
    return block


def _build_missing_piece(payload: Dict[str, Any], reasoning_text: str) -> Dict[str, Any] | None:
    mp = payload.get("missing_piece")
    if isinstance(mp, dict) and str(mp.get("name") or "").strip():
        return {
            "name": str(mp.get("name") or "").strip(),
            "category": str(mp.get("category") or "").strip(),
            "reason": str(mp.get("reason") or reasoning_text or "").strip(),
            "unlocks": [str(u).strip() for u in (mp.get("unlocks") or []) if str(u).strip()][:6],
        }
    return None


def _sanitize_missing_piece_for_gender(
    missing_piece: Dict[str, Any] | None,
    *,
    target_gender: str,
    allow_feminine: bool = False,
) -> Dict[str, Any] | None:
    if not missing_piece:
        return None
    if not _missing_piece_allowed_for_gender(
        missing_piece,
        target_gender=target_gender,
        allow_feminine=allow_feminine,
    ):
        logger.info(
            "AHVI_MISSING_PIECE_GENDER_FILTERED gender=%s name=%s category=%s",
            target_gender,
            _asset_text(missing_piece.get("name")),
            _asset_text(missing_piece.get("category")),
        )
        return None
    return missing_piece


def _enrich_missing_piece_with_asset(
    missing_piece: Dict[str, Any] | None,
    *,
    assets: List[Dict[str, Any]] | None = None,
    occasion: str | None = None,
    target_gender: str = "unknown",
    allow_feminine: bool = False,
) -> Dict[str, Any] | None:
    if not missing_piece:
        return missing_piece
    missing_piece = _sanitize_missing_piece_for_gender(
        missing_piece,
        target_gender=target_gender,
        allow_feminine=allow_feminine,
    )
    if not missing_piece:
        return None
    out = dict(missing_piece)
    occasion_family = _visual_occasion_family(occasion, out.get("name"))
    if occasion_family == "indian_festive" and not _safe_visual_support_assets(
        [out],
        occasion_family,
        occasion,
        {"hero_piece": out.get("name")},
    ):
        replacement = _festive_missing_piece_replacement(occasion)
        logger.info(
            "AHVI_ASSET_GUARD occasion=%s family=%s blocked=%s selected=%s",
            _asset_text(occasion),
            occasion_family,
            [_asset_text(out.get("name")) or "unknown"],
            [_asset_text(replacement.get("name"))] if replacement else [],
        )
        logger.info(
            "AHVI_MISSING_PIECE_REPLACED family=%s original=%s replacement=Ethnic Footwear",
            occasion_family,
            _asset_text(out.get("name")) or "unknown",
        )
        return replacement
    block_reason = _occasion_asset_block_reason(
        out,
        occasion=_asset_text(occasion),
        placement="missing",
        target_text=_asset_text(out.get("name") or out.get("category")),
    )
    if block_reason:
        replacement = _festive_missing_piece_replacement(
            occasion,
            target_gender=target_gender,
            allow_feminine=allow_feminine,
        )
        logger.info(
            "AHVI_ASSET_GUARD occasion=%s family=%s blocked=%s selected=%s",
            _asset_text(occasion),
            _visual_occasion_family(occasion, out.get("name")) or "general",
            [f"{_asset_text(out.get('name')) or 'unknown'}:{block_reason}"],
            [_asset_text(replacement.get("name"))] if replacement else [],
        )
        if not replacement:
            return None
        logger.info(
            "AHVI_MISSING_PIECE_REPLACED family=%s original=%s replacement=Ethnic Footwear",
            _visual_occasion_family(occasion, out.get("name")) or "general",
            _asset_text(out.get("name")) or "unknown",
        )
        return replacement
    if _asset_text(out.get("image_url") or out.get("imageUrl")):
        return out
    rows = assets if isinstance(assets, list) else _style_asset_rows()
    if not rows:
        return out
    direction = {
        "hero_piece": out.get("name"),
        "items": [out.get("name"), out.get("category")],
        "colors": [],
        "archetype": "",
    }
    asset = _best_style_asset(
        rows,
        direction=direction,
        occasion=_asset_text(occasion),
        target_gender=target_gender,
        allow_feminine_accessory=allow_feminine,
        placement="missing",
    )
    if asset:
        out["image_url"] = _asset_text(asset.get("image_url") or asset.get("imageUrl"))
        out["asset_id"] = _asset_text(asset.get("asset_id") or asset.get("$id"))
    return out


def _dedupe_missing_piece_against_directions(
    missing_piece: Dict[str, Any] | None,
    visual_directions: List[Dict[str, Any]],
    *,
    occasion: str = "",
    target_gender: str = "unknown",
    allow_feminine: bool = False,
) -> Dict[str, Any] | None:
    if not missing_piece:
        return None
    name = _asset_text(missing_piece.get("name"))
    if not name:
        return None
    for direction in visual_directions or []:
        components = _safe_list(direction.get("items") or direction.get("pieces"), limit=8)
        reason = _missing_piece_duplicate_reason(
            name,
            hero_piece=direction.get("hero_piece"),
            components=components,
        )
        if reason:
            replacement = _fallback_missing_piece_for_direction(
                direction,
                occasion=occasion,
                target_gender=target_gender,
                allow_feminine=allow_feminine,
            )
            if replacement:
                out = dict(missing_piece)
                out["name"] = replacement
                out["category"] = _style_category(replacement) or out.get("category") or "style piece"
                out["reason"] = _missing_piece_reason_for_direction(replacement, direction, occasion=occasion)
                out.pop("image_url", None)
                out.pop("imageUrl", None)
                out.pop("asset_id", None)
                logger.info(
                    "AHVI_MISSING_PIECE_DEDUPED old=%r new=%r reason=%s",
                    name,
                    replacement,
                    reason,
                )
                return out
            logger.info("AHVI_MISSING_PIECE_DROPPED_DUPLICATE name=%r reason=%s", name, reason)
            return None
    return missing_piece


def _build_visual_inspiration_board(
    payload: Dict[str, Any],
    visual_directions: List[Dict[str, Any]],
    goal: str,
    impression: str,
    missing_piece: Dict[str, Any] | None,
    query: str,
) -> Dict[str, Any]:
    """Premium visual-inspiration metadata block. Builds an image_prompt for a
    future generation step, but does NOT generate images yet."""
    direct = payload.get("visual_inspiration_board")
    direct = direct if isinstance(direct, dict) else {}
    first = visual_directions[0] if visual_directions else {}
    palette = first.get("palette") if isinstance(first.get("palette"), list) else []
    pieces = first.get("pieces") if isinstance(first.get("pieces"), list) else []
    board = {
        "type": "visual_inspiration_board",
        "archetype": str(first.get("archetype") or direct.get("archetype") or "").strip(),
        "title": str(direct.get("title") or first.get("title") or "Style Inspiration").strip(),
        "aesthetic": str(direct.get("aesthetic") or first.get("strategy") or "").strip(),
        "mood": str(direct.get("mood") or impression or "").strip(),
        "palette": [str(p).strip() for p in (direct.get("palette") or palette) if str(p).strip()][:6],
        "hero_piece": str(direct.get("hero_piece") or (pieces[0] if pieces else "")).strip(),
        "silhouette": str(direct.get("silhouette") or "").strip(),
        "styling_notes": str(
            direct.get("styling_notes") or first.get("why_it_works") or first.get("style_note") or goal
        ).strip(),
        "missing_piece": missing_piece,
    }
    board["image_prompt"] = _build_inspiration_image_prompt(board, query)
    board["inspiration_image_url"] = ""
    board["image_status"] = "not_generated"
    return board


def _build_inspiration_image_prompt(board: Dict[str, Any], query: str) -> str:
    """Editorial moodboard image prompt from the inspiration metadata.
    Generation is wired later (Imagen/Flux) — this only builds the prompt."""
    occ = _clean_recursive_prompt(query).strip() or "this occasion"
    parts = [f"Editorial fashion moodboard for {occ}."]
    if board.get("aesthetic"):
        parts.append(f"Aesthetic: {board['aesthetic']}.")
    if board.get("mood"):
        parts.append(f"Mood: {board['mood']}.")
    if board.get("palette"):
        parts.append(f"Palette: {', '.join(board['palette'])}.")
    if board.get("hero_piece"):
        parts.append(f"Hero piece: {board['hero_piece']}.")
    if board.get("silhouette"):
        parts.append(f"Silhouette: {board['silhouette']}.")
    parts.append("No faces. No text. Pinterest-style board.")
    return " ".join(parts)


def _compact_reasoning(text: str, *, multi_event: bool = False) -> str:
    """Trim stylist_reasoning for the UI: strip markdown headings/bold, collapse
    whitespace, cap at ~60 words (70 for multi-event) on a sentence boundary."""
    raw = str(text or "").strip()
    if not raw:
        return raw
    # Strip markdown headings, bold labels like "**Core:**", bullets.
    raw = re.sub(r"\*\*[^*]+\*\*:?", "", raw)
    raw = re.sub(r"(?m)^#{1,6}\s*", "", raw)
    raw = re.sub(r"(?m)^\s*[-*•]\s*", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" :-•")
    raw = _scrub_internal_style_language(raw)
    cap = 70 if multi_event else 60
    words = raw.split()
    if len(words) <= cap:
        compacted = raw
    else:
        clipped = " ".join(words[:cap])
        # End on the last sentence boundary if one exists in range.
        m = list(re.finditer(r"[.!?]", clipped))
        compacted = clipped[: m[-1].end()] if m else clipped.rstrip(",;: ") + "."
    logger.info(
        "AHVI_REASONING_COMPACTED words_in=%d words_out=%d multi_event=%s",
        len(words), len(compacted.split()), multi_event,
    )
    return compacted


def _scrub_internal_style_language(text: str) -> str:
    out = str(text or "")
    replacements = [
        (r"\bthe social outcome is connection\b", "relaxed polish works best"),
        (r"\bsocial outcome is connection\b", "relaxed polish works best"),
        (r"\bthe styling strategy is to convey\b", "the look should feel"),
        (r"\bstyling strategy is to convey\b", "the look should feel"),
        (r"\bthe styling strategy is\b", "the styling approach is"),
        (r"\bstyling strategy\b", "styling approach"),
        (r"\bsocial outcome\b", "impression"),
        (r"\bwe want to signal\b", "the outfit should show"),
        (r"\bsignal\b", "show"),
        (r"\barchetype selection\b", "style direction"),
    ]
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


_VISIBLE_PLACEHOLDER_REPLACEMENTS = [
    (r"\bcustom_occasion\b", "casual outing"),
    (r"\bhybrid_occasion\b", "work-to-social occasion"),
    (r"\bwork_occasion\b", "work setting"),
    (r"\btravel_occasion\b", "travel day"),
    (r"\bclean shirt\b", "crisp shirt"),
    (r"\bsimple footwear\b", "polished shoes"),
    (r"\bminimal straight bottom\b", "streamlined trouser"),
    (r"\btailored bottom\b", "tailored trouser"),
    (r"\brelaxed tailored bottom\b", "relaxed tailored trouser"),
    (r"\bcasual tailored bottom\b", "casual tailored trouser"),
    (r"\blow-profile footwear\b", "low-profile shoes"),
    (r"\bclean casual footwear\b", "clean casual shoes"),
    (r"\bclean supporting pieces\b", "well-chosen supporting pieces"),
    (r"\bsensitive_occasion\b", "respectful occasion"),
    (r"\bcrisp shirt\s+base\b", "Classic Tailoring"),
    (r"\bshirt\s+base\b", "Classic Tailoring"),
    (r"\bstyle piece\b", "wardrobe piece"),
]


def _scrub_visible_style_text(text: Any, *, query: str = "") -> str:
    out = str(text or "")
    if not out:
        return out
    out = _scrub_internal_style_language(out)
    festive_context = _visual_occasion_family(query) == "indian_festive"
    if festive_context and out.strip() == _DEFAULT_OCCASION_VOICE:
        return "Completes the festive kurta look while staying comfortable for rituals."
    social_replacement = "coffee date" if any(term in _norm(query) for term in ("coffee", "date")) else "social outing"
    out = re.sub(r"\bsocial_occasion\b", social_replacement, out, flags=re.IGNORECASE)
    out = re.sub(
        r"\bcustom_occasion\b",
        "festive ceremony" if festive_context else "casual outing",
        out,
        flags=re.IGNORECASE,
    )
    for pattern, replacement in _VISIBLE_PLACEHOLDER_REPLACEMENTS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    if festive_context and re.search(
        r"\b(?:wedding\s+)?haldi ceremony\s+(?:festive ceremony\s+)?(?:casual outing\s+)?haldi look\b",
        out,
        flags=re.IGNORECASE,
    ):
        out = "Completes the festive kurta look while staying comfortable for rituals."
    return re.sub(r"\s+", " ", out).strip()


def _scrub_visible_style_payload(value: Any, *, query: str = "") -> Any:
    if isinstance(value, dict):
        return {key: _scrub_visible_style_payload(val, query=query) for key, val in value.items()}
    if isinstance(value, list):
        return [_scrub_visible_style_payload(item, query=query) for item in value]
    if isinstance(value, str):
        return _scrub_visible_style_text(value, query=query)
    return value


_PERSONALITY_TEXT_FIELDS: set[str] = {
    "advice",
    "stylist_reasoning",
    "confidence_strategy",
    "missing_piece_reasoning",
    "description",
    "explanation",
    "reason",
    "short_note",
    "style_note",
    "styling_tip",
    "subtitle",
    "summary",
    "tip",
    "why",
    "why_it_works",
    "why_this_works",
    "complete_the_look_copy",
    "styling_notes",
}

_PERSONALITY_SKIP_KEYS: set[str] = {
    "$id",
    "asset_id",
    "assetId",
    "board_id",
    "card_id",
    "category",
    "subcategory",
    "sub_category",
    "id",
    "image_base64",
    "image_id",
    "image_status",
    "image_url",
    "imageUrl",
    "inspiration_image_url",
    "item_id",
    "itemId",
    "name",
    "title",
    "label",
    "value",
    "url",
}

_PERSONALITY_ROBOTIC_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (
        r"\bthis look by supporting ([^.,;]+?) and making (?:the )?components complete\b",
        r"\1 keeps the outfit feeling finished.",
    ),
    (
        r"\bby supporting ([^.,;]+?) and making (?:the )?components complete\b",
        r"with \1 as the finishing detail",
    ),
    (r"\bthe components complete\b", "the outfit feel finished"),
    (r"\bcomponents complete\b", "outfit finished"),
    (r"\bthis direction by\b", "this direction with"),
    (r"\bthis look is designed to\b", "This keeps the look"),
    (r"\bin order to\b", "to"),
    (r"\butilize\b", "use"),
    (r"\bleverage\b", "use"),
    (r"\bsynergy\b", "balance"),
    (r"\boptimal\b", "strong"),
    (r"\bcurated ensemble\b", "look"),
    (r"\bensemble\b", "look"),
    (r"\bgarment components\b", "pieces"),
)


def _personality_rules_available() -> bool:
    try:
        from services.ahvi_personality_rules import load_personality_rules

        rules = load_personality_rules()
        return bool(isinstance(rules, dict) and rules.get("loaded"))
    except Exception as exc:  # noqa: BLE001 - fail open, never block styling.
        logger.warning("ahvi.personality_rules.failed stage=style_polish error=%s", str(exc)[:180])
        return False


def _personality_sentence_cap(text: str, *, max_chars: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return cleaned
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    if sentences:
        cleaned = " ".join(sentences[:2]).strip()
    words = cleaned.split()
    if len(words) > 34:
        cleaned = " ".join(words[:34]).rstrip(" ,;:-") + "."
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-") + "."
    return cleaned


def _personality_polish_text(text: Any, *, field: str = "", query: str = "") -> str:
    out = str(text or "").strip()
    if not out:
        return out
    out = _scrub_visible_style_text(out, query=query)
    for pattern, replacement in _PERSONALITY_ROBOTIC_REPLACEMENTS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    out = re.sub(r"\b(Considered and confident, styled to feel like you\.)\s*\1", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\.{2,}", ".", out)
    out = re.sub(r"\s+", " ", out).strip()
    field_key = str(field or "").lower()
    if field_key in {"reason", "missing_piece_reasoning"}:
        if re.search(r"\bmissing\b", field_key) or "missing" in _norm(out):
            out = out.replace("It completes the outfit.", "It finishes the look without overcomplicating it.")
        return _personality_sentence_cap(out, max_chars=170)
    if field_key in {"style_note", "styling_tip", "short_note", "subtitle", "summary", "tip"}:
        return _personality_sentence_cap(out, max_chars=140)
    return _personality_sentence_cap(out, max_chars=220)


def _apply_personality_text_polish(value: Any, *, query: str = "", field: str = "") -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, val in value.items():
            if key in _PERSONALITY_SKIP_KEYS:
                out[key] = val
            elif key in _PERSONALITY_TEXT_FIELDS:
                out[key] = _apply_personality_text_polish(val, query=query, field=key)
            elif isinstance(val, (dict, list)):
                out[key] = _apply_personality_text_polish(val, query=query, field=key)
            else:
                out[key] = val
        return out
    if isinstance(value, list):
        return [_apply_personality_text_polish(item, query=query, field=field) for item in value]
    if isinstance(value, str) and field in _PERSONALITY_TEXT_FIELDS:
        return _personality_polish_text(value, field=field, query=query)
    return value


def apply_personality_text_polish_to_final_payload(value: Any, *, query: str = "") -> Any:
    """Final P0 personality-pack wording guard.

    The normalized pack is used only as permission to apply deterministic copy
    constraints: concise, practical, visual-first wording. It never changes
    garment names, IDs, categories, assets, images, chips, or payload shape.
    """
    if not _personality_rules_available():
        return value
    return _apply_personality_text_polish(value, query=query)


_MALE_FINAL_TEXT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # Multiword first.
    (r"\bpointed[\s-]+toe\s+flats\b", "formal shoes"),
    (r"\bmidi\s+dress(?:es)?\b", "tailored trousers"),
    (r"\bsoft\s+waves\b", "clean grooming"),
    (r"\bdewy\s+makeup\b", "fresh grooming"),
    (r"\bmaang[\s-]?tikka\b", "a watch"),
    (r"\b(?:dress|dresses|gown|gowns|skirt|skirts|saree|sari|lehenga)s?\b", "tailored trousers"),
    (r"\bblouse(?:s)?\b", "crisp shirt"),
    (r"\b(?:heel|heels|pump|pumps|stiletto|stilettos|wedges)\b", "formal shoes"),
    (r"\bnecklace(?:s)?\b", "watch"),
    (r"\bearrings?\b", "belt"),
    (r"\bbangles?\b", "a watch"),
    (r"\b(?:jhumkas?|mangalsutra|payal|anklets?)\b", "a watch"),
    (r"\bdupatta\b", "a pocket square"),
    (r"\bclutch(?:es)?\b", "a sleek bag"),
    (r"\b(?:kurti|anarkali|choli|sharara|gharara|salwar)s?\b", "a kurta"),
    (r"\b(?:makeup|lipstick|mascara|eyeliner|kajal|kohl|bindi)\b", "clean grooming"),
)

_MALE_FINAL_BLOCKED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("pointed-toe flats", r"\bpointed[\s-]+toe\s+flats\b"),
    ("midi dress", r"\bmidi\s+dress(?:es)?\b"),
    ("dress", r"\bdress(?:es)?\b"),
    ("gown", r"\bgown(?:s)?\b"),
    ("skirt", r"\bskirt(?:s)?\b"),
    ("blouse", r"\bblouse(?:s)?\b"),
    ("saree", r"\bsaree\b"),
    ("lehenga", r"\blehenga\b"),
    ("heels", r"\bheels?\b"),
    ("pumps", r"\bpumps?\b"),
    ("necklace", r"\bnecklace(?:s)?\b"),
    ("earrings", r"\bearrings?\b"),
    ("bangles", r"\bbangles?\b"),
    ("makeup", r"\b(?:makeup|lipstick|mascara|eyeliner|kajal|kohl|bindi)\b"),
    ("soft waves", r"\bsoft\s+waves\b"),
    ("dewy makeup", r"\bdewy\s+makeup\b"),
    ("dupatta", r"\bdupatta\b"),
    ("clutch", r"\bclutch(?:es)?\b"),
    ("kurti", r"\b(?:kurti|anarkali|choli|sharara|gharara|salwar)s?\b"),
    ("jhumka", r"\b(?:jhumkas?|mangalsutra|payal|anklets?|maang[\s-]?tikka)\b"),
)

_MALE_FORMAL_CONTEXT_TERMS = {
    "business",
    "client",
    "conference",
    "formal",
    "funeral",
    "meeting",
    "office",
    "presentation",
    "talk",
    "wedding",
}

# Festive/ethnic contexts get a men's ethnic template instead of a blazer one.
_MALE_FESTIVE_CONTEXT_TERMS = {
    "haldi",
    "sangeet",
    "mehendi",
    "mehndi",
    "festive",
    "reception",
    "engagement",
    "diwali",
    "baraat",
    "wedding",
    "shaadi",
}

_GENDER_GUARD_SKIP_KEYS = {
    "$id",
    "asset_id",
    "board_id",
    "card_id",
    "id",
    "image_base64",
    "image_id",
    "image_url",
    "imageUrl",
    "item_id",
    "itemId",
    "url",
}


def _male_final_guard_removed_terms(value: Any) -> List[str]:
    text = str(value or "")
    if not text:
        return []
    removed: List[str] = []
    for label, pattern in _MALE_FINAL_BLOCKED_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            removed.append(label)
    return removed


def _male_formal_context_replacement(context: str = "") -> str:
    normalized = _norm(context)
    # Festive takes precedence over generic formal (a haldi is not a boardroom).
    if any(term in normalized for term in _MALE_FESTIVE_CONTEXT_TERMS):
        return (
            "a marigold or ivory kurta with a Nehru jacket, churidar or "
            "tailored trousers, juttis, finished with a watch and pocket square"
        )
    if any(term in normalized for term in _MALE_FORMAL_CONTEXT_TERMS):
        return "blazer, crisp shirt, tailored trousers, formal shoes, belt/watch, optional laptop bag"
    return "crisp shirt, tailored trousers, loafers, belt/watch"


def _sanitize_male_final_text(text: Any, *, context: str = "") -> tuple[str, List[str]]:
    out = str(text or "")
    if not out:
        return out, []
    removed = _male_final_guard_removed_terms(out)
    if not removed:
        return out, []
    formal_context = any(
        term in _norm(context)
        for term in (_MALE_FORMAL_CONTEXT_TERMS | _MALE_FESTIVE_CONTEXT_TERMS)
    )
    if formal_context and (len(removed) >= 2 or "," in out):
        return _male_formal_context_replacement(context), removed
    for pattern, replacement in _MALE_FINAL_TEXT_REPLACEMENTS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    if not out or _male_final_guard_removed_terms(out):
        out = _male_formal_context_replacement(context)
    return out, removed


def _male_sanitize_item_dict(item: Dict[str, Any], *, context: str) -> tuple[Dict[str, Any], List[str]]:
    out = dict(item)
    removed: List[str] = []
    original_name = out.get("name") or out.get("title") or out.get("label")
    item_removed = _male_final_guard_removed_terms(original_name)
    for key, value in list(out.items()):
        if key in _GENDER_GUARD_SKIP_KEYS:
            continue
        sanitized, key_removed = apply_gender_guard_to_final_payload(
            value,
            target_gender="male",
            context=context,
            log=False,
        )
        out[key] = sanitized
        removed.extend(key_removed)
    if item_removed:
        removed.extend(item_removed)
        lower_name = _norm(original_name)
        if any(term in lower_name for term in ("necklace", "earring")):
            replacement_name, role = "belt/watch", "accessory"
        elif any(term in lower_name for term in ("heel", "pump", "pointed toe flat", "pointed toe flats")):
            replacement_name, role = "formal shoes", "footwear"
        elif "blouse" in lower_name:
            replacement_name, role = "crisp shirt", "top"
        else:
            replacement_name, role = "tailored trousers", "bottom"
        for key in ("name", "title", "label"):
            if key in out:
                out[key] = replacement_name
        for key in ("role", "category", "subcategory", "sub_category", "type"):
            if key in out:
                out[key] = role
    return out, sorted(set(removed))


def apply_gender_guard_to_final_payload(
    value: Any,
    *,
    target_gender: str,
    context: str = "",
    log: bool = True,
) -> tuple[Any, List[str]]:
    """Final visible-output guard for gendered fashion language.

    Asset filters run earlier, but model text and final cards can still carry
    generated labels. This function is intentionally recursive so callers can
    apply it to plain text, card lists, or full response payloads.
    """
    gender = _asset_gender(target_gender)
    if gender != "male":
        return value, []
    removed: List[str] = []
    if isinstance(value, dict):
        if any(_male_final_guard_removed_terms(value.get(k)) for k in ("name", "title", "label")):
            sanitized, item_removed = _male_sanitize_item_dict(value, context=context)
            removed.extend(item_removed)
            out = sanitized
        else:
            out = {}
            for key, val in value.items():
                if key in _GENDER_GUARD_SKIP_KEYS:
                    out[key] = val
                    continue
                sanitized, key_removed = apply_gender_guard_to_final_payload(
                    val,
                    target_gender=gender,
                    context=context,
                    log=False,
                )
                out[key] = sanitized
                removed.extend(key_removed)
    elif isinstance(value, list):
        out = []
        for item in value:
            sanitized, item_removed = apply_gender_guard_to_final_payload(
                item,
                target_gender=gender,
                context=context,
                log=False,
            )
            out.append(sanitized)
            removed.extend(item_removed)
    elif isinstance(value, str):
        out, removed = _sanitize_male_final_text(value, context=context)
    else:
        out = value
    removed = sorted(set(removed))
    if removed and log:
        logger.info(
            "AHVI_GENDER_GUARD_APPLIED gender=male removed=%s context=%s",
            removed,
            str(context or "")[:160],
        )
    return out, removed


def _coerce_emotion(value: Any, category: str | None) -> str:
    emotion = _norm(value)
    if emotion in {"neutral", "excited", "frustrated", "vulnerable", "professional", "social"}:
        return emotion
    return _fallback_emotion(category)


def _coerce_ai_mode(value: Any, fallback: str) -> str:
    mode = _norm(value).replace(" ", "_")
    if mode in _GEMINI_MODES:
        return mode
    return fallback if fallback in _GEMINI_MODES else STYLE_ADVICE


def _build_response(
    *,
    query: str,
    mode: str,
    category: str | None,
    tone: str | None,
    formality: str | None,
    occasion: str | None,
    confidence: float,
    ai_payload: Dict[str, Any] | None,
    user_profile: dict,
    context: dict,
) -> Dict[str, Any]:
    payload = ai_payload if isinstance(ai_payload, dict) else {}
    final_mode = mode if mode in _GEMINI_MODES else _coerce_ai_mode(payload.get("mode"), mode)
    selected_archetypes = payload.get("_selected_archetypes") if isinstance(payload.get("_selected_archetypes"), list) else []
    persona_context = payload.get("_persona_context") if isinstance(payload.get("_persona_context"), dict) else {}
    pairing_anchor = _extract_pairing_anchor(query) if final_mode == STYLE_PAIRING else {}
    pairing_gender = "unknown"
    if final_mode == STYLE_PAIRING:
        raw_anchor = payload.get("anchor_item") if isinstance(payload.get("anchor_item"), dict) else {}
        pairing_anchor = {
            "name": str(raw_anchor.get("name") or pairing_anchor.get("name") or "").strip(),
            "category": str(raw_anchor.get("category") or pairing_anchor.get("category") or "").strip(),
            "color": str(raw_anchor.get("color") or pairing_anchor.get("color") or "").strip(),
        }
        try:
            from services.style_context_service import _resolve_gender

            pairing_gender = _resolve_gender(user_profile if isinstance(user_profile, dict) else {})
        except Exception:  # noqa: BLE001
            pairing_gender = "unknown"
    asset_gender = _resolve_asset_gender(query=query, user_profile=user_profile)
    allow_feminine_style = _prompt_allows_gendered_feminine_style(query)
    asset_occasion = " ".join(
        dict.fromkeys(
            str(value).strip()
            for value in (
                (context or {}).get("occasion"),
                payload.get("occasion"),
                occasion,
                category,
                query,
            )
            if str(value or "").strip()
        )
    )
    # Shared Style Brain (Phase A/B): build ONE canonical context and route the
    # visual path's occasion through it. Gated by STYLE_SHARED_BRAIN (default
    # OFF) so production behavior is identical until ops opt in.
    canonical_ctx: Dict[str, Any] | None = None
    if _shared_brain_enabled():
        try:
            from services.style_context_service import build_canonical_style_context

            canonical_ctx = build_canonical_style_context(
                query=query,
                user_profile=user_profile,
                intent=payload,
                router_occasion=(
                    str((context or {}).get("occasion") or occasion or category or "").strip()
                    or None
                ),
                weather=(context or {}).get("weather") or (context or {}).get("weather_context"),
                event_context=(context or {}).get("event_context"),
                style_dna=(context or {}).get("style_dna") or (user_profile or {}).get("style_dna"),
            )
            # Keep the prompt-override-aware gender for the guard, and stash the
            # query so the gender sanitizer can honor prompt overrides.
            canonical_ctx["gender"] = asset_gender
            canonical_ctx["_query"] = query
            canon_occ = str(canonical_ctx.get("canonical_occasion") or "").strip()
            if canon_occ:
                logger.info(
                    "AHVI_VISUAL_OCCASION_CANON raw=%r canonical=%s",
                    asset_occasion,
                    canon_occ,
                )
                asset_occasion = canon_occ
        except Exception as exc:  # noqa: BLE001 — fail open, never break the visual path.
            logger.warning("AHVI_CANONICAL_CTX_WIRE_FAILED err=%s", repr(exc)[:160])
            canonical_ctx = None

    goal = str(payload.get("goal") or _fallback_goal(final_mode, category)).strip()
    impression = str(payload.get("impression") or _fallback_impression(category)).strip()
    atmosphere = str(payload.get("atmosphere") or _fallback_atmosphere(category)).strip()
    emotion_state = _coerce_emotion(payload.get("emotion_state"), category)
    # stylist_reasoning leads. Accept legacy stylist_advice as alias so older
    # Gemini responses and tests keep working.
    raw_advice = str(
        payload.get("stylist_reasoning")
        or payload.get("stylist_advice")
        or _fallback_advice(query, final_mode, category)
    ).strip()
    missing_piece_reasoning = str(
        payload.get("missing_piece_reasoning") or _fallback_missing_piece(query, category)
    ).strip()
    confidence_strategy = str(
        payload.get("confidence_strategy")
        or "Lean into what already fits well and keep one deliberate detail — "
        "confidence reads as ease, not effort."
    ).strip()
    polished_advice = tone_engine.apply(
        raw_advice,
        user_profile=user_profile,
        signals={"mode": final_mode, "emotion_state": emotion_state},
        context=context,
    )
    is_multi_event = bool((context or {}).get("multi_event")) or (context or {}).get("occasion") == "multi_event"
    polished_advice = _compact_reasoning(polished_advice, multi_event=is_multi_event)
    transition_plan = None
    if is_multi_event:
        _tp = payload.get("transition_plan")
        if isinstance(_tp, dict):
            transition_plan = {
                "keep": _safe_list(_tp.get("keep"), limit=6),
                "swap": _safe_list(_tp.get("swap"), limit=6),
                "add": _safe_list(_tp.get("add"), limit=6),
                "avoid": _safe_list(_tp.get("avoid"), limit=6),
                "dinner_ready": str(_tp.get("dinner_ready") or "").strip(),
            }
    follow_up = str(payload.get("follow_up_question") or "").strip() or None
    pairing_routes: List[Dict[str, Any]] = []
    if final_mode == STYLE_PAIRING:
        pairing_routes = _normalize_pairing_routes(
            payload.get("pairing_routes"), pairing_anchor, pairing_gender,
            wardrobe=context.get("wardrobe") or context.get("wardrobe_items"),
            selected_archetypes=selected_archetypes,
        )
        if pairing_routes and any(r.get("wardrobe_reality") for r in pairing_routes):
            logger.info(
                "AHVI_WARDROBE_REALITY_BUILT routes=%d scores=%s",
                len(pairing_routes),
                [r.get("wardrobe_reality", {}).get("match_score") for r in pairing_routes],
            )
        visual_directions = _pairing_routes_as_visual_directions(pairing_routes)
    else:
        visual_directions = _normalize_visual_directions(
            payload.get("visual_directions"),
            final_mode,
            category,
            selected_archetypes if final_mode == VISUAL_INSPIRATION else None,
        )
    visual_directions = _apply_anchor_piece_to_visual_directions(visual_directions, query)
    visual_directions = _enforce_festive_visual_directions(
        visual_directions,
        occasion=asset_occasion,
    )
    visual_directions = _enrich_visual_directions_with_assets(
        visual_directions,
        occasion=asset_occasion,
        target_gender=asset_gender,
        allow_feminine_accessory=allow_feminine_style,
        # Canonical brief drives formality/movement/energy scoring + hard veto.
        # None unless STYLE_SHARED_BRAIN is on (canonical_ctx built only then).
        brief=canonical_ctx,
        # Wardrobe intent -> board_items must be owned-only (no generic assets).
        wardrobe_intent=(final_mode == WARDROBE_STYLE),
    )
    # Shared Style Brain (Phase C): post-generation visual guard. No-op unless
    # STYLE_SHARED_BRAIN is enabled. Same list-of-dicts shape in/out.
    if canonical_ctx is not None:
        visual_directions = _apply_style_guard(visual_directions, canonical_ctx)
    try:
        final_confidence = max(0.0, min(1.0, float(payload.get("confidence", confidence))))
    except Exception:
        final_confidence = confidence

    what_to_avoid = _safe_list(payload.get("what_to_avoid"), limit=6)
    missing_piece = _enrich_missing_piece_with_asset(
        _build_missing_piece(payload, missing_piece_reasoning),
        occasion=asset_occasion,
        target_gender=asset_gender,
        allow_feminine=allow_feminine_style,
    )
    missing_piece = _dedupe_missing_piece_against_directions(
        missing_piece,
        visual_directions,
        occasion=asset_occasion,
        target_gender=asset_gender,
        allow_feminine=allow_feminine_style,
    )
    if missing_piece and not _asset_text(missing_piece.get("image_url") or missing_piece.get("imageUrl")):
        missing_piece = _enrich_missing_piece_with_asset(
            missing_piece,
            occasion=asset_occasion,
            target_gender=asset_gender,
            allow_feminine=allow_feminine_style,
        )
    visual_inspiration_board = None
    if final_mode == VISUAL_INSPIRATION:
        visual_inspiration_board = _build_visual_inspiration_board(
            payload, visual_directions, goal, impression, missing_piece, query
        )
    if final_mode in {STYLE_PAIRING, VISUAL_INSPIRATION}:
        logger.info(
            "AHVI_ARCHETYPE_REASONING_APPLIED mode=%s archetypes=%s dna=%s wardrobe=%s",
            final_mode,
            [str(a.get("name") or "").strip() for a in selected_archetypes if isinstance(a, dict)],
            bool(persona_context.get("style_dna")),
            bool(context.get("wardrobe") or context.get("wardrobe_items")),
        )

    polished_advice = _scrub_visible_style_text(polished_advice, query=query)
    confidence_strategy = _scrub_visible_style_text(confidence_strategy, query=query)
    missing_piece_reasoning = _scrub_visible_style_text(missing_piece_reasoning, query=query)
    # Cap long stylist text fields server-side so the editorial UI never
    # has to render walls of LLM prose.
    polished_advice = _two_sentences(polished_advice, max_chars=320)
    confidence_strategy = _two_sentences(confidence_strategy, max_chars=240)
    missing_piece_reasoning = _two_sentences(missing_piece_reasoning, max_chars=200)
    visual_directions = _scrub_visible_style_payload(visual_directions, query=query)
    # Editorial polish: add direction_name, adjectives, badge, curated_for,
    # short_note, complete_the_look_copy. Additive — keeps legacy fields.
    _wardrobe_for_polish = context.get("wardrobe") or context.get("wardrobe_items")
    visual_directions = _apply_editorial_polish(
        visual_directions,
        occasion=payload.get("occasion") or occasion or category or "",
        wardrobe_items=_wardrobe_for_polish,
        context_text=query,
    )
    final_occasion_text = _asset_text(payload.get("occasion") or occasion or category or "")
    visual_directions = _music_event_visible_guard(
        visual_directions,
        occasion=final_occasion_text,
        query=query,
    )
    for direction in visual_directions:
        if isinstance(direction, dict) and isinstance(direction.get("complete_the_look"), list):
            direction["complete_the_look"] = _safe_visual_support_assets(
                direction.get("complete_the_look"),
                _visual_occasion_family(final_occasion_text, f"{query} {direction.get('hero_piece') or ''}"),
                f"{final_occasion_text} {query}",
                direction,
            )[:3]
    if final_mode == VISUAL_INSPIRATION and _is_music_event_context(f"{final_occasion_text} {query}"):
        visual_inspiration_board = _build_visual_inspiration_board(
            payload, visual_directions, goal, impression, missing_piece, query
        )
    editorial_cover = _build_editorial_cover(
        visual_directions,
        occasion=payload.get("occasion") or occasion or category or "",
    )
    missing_piece = _scrub_visible_style_payload(missing_piece, query=query) if missing_piece else None
    visual_inspiration_board = (
        _scrub_visible_style_payload(visual_inspiration_board, query=query)
        if visual_inspiration_board
        else None
    )
    what_to_avoid = _scrub_visible_style_payload(what_to_avoid, query=query)

    response = {
        "mode": final_mode,
        "occasion": str(payload.get("occasion") or occasion or "").strip() or None,
        "tone": tone,
        "formality": formality,
        "should_use_wardrobe": False,
        "should_generate_board": False,
        "advice": polished_advice,
        "stylist_reasoning": polished_advice,
        "goal": goal,
        "impression": impression,
        "atmosphere": atmosphere,
        "confidence_strategy": confidence_strategy,
        "missing_piece_reasoning": missing_piece_reasoning,
        "missing_piece": missing_piece,
        "visual_inspiration_board": visual_inspiration_board,
        "advice_block": _build_advice_block(final_mode, payload),
        "transition_plan": transition_plan,
        "is_transition": bool(is_multi_event),
        "anchor_item": pairing_anchor or None,
        "pairing_routes": pairing_routes,
        "last_style_context": (_pairing_last_style_context(pairing_anchor, pairing_routes, context) if final_mode == STYLE_PAIRING else None),
        "archetype_reasoning": str(payload.get("archetype_reasoning") or "").strip(),
        "dna_alignment": str(payload.get("dna_alignment") or persona_context.get("style_dna") or "").strip(),
        "wardrobe_alignment": str(payload.get("wardrobe_alignment") or "").strip(),
        "follow_up_question": follow_up,
        "cta": (
            _pairing_ctas(pairing_anchor, pairing_routes)
            if final_mode == STYLE_PAIRING
            else _fallback_cta(query)
        ),
        "visual_directions": visual_directions,
        "editorial_cover": editorial_cover,
        "what_to_avoid": what_to_avoid,
        "meta": {
            "source": "style_reasoning_engine",
            "reason": _reason_for_mode(final_mode, category),
            "goal": goal,
            "impression": impression,
            "atmosphere": atmosphere,
            "missing_piece_reasoning": missing_piece_reasoning,
            "emotion_state": emotion_state,
            "confidence": final_confidence,
            "asset_gender": asset_gender,
            "target_gender": asset_gender,
            "anchor_item": pairing_anchor or None,
            "selected_archetypes": [str(a.get("name") or "").strip() for a in selected_archetypes if isinstance(a, dict)],
            "archetype_reasoning": str(payload.get("archetype_reasoning") or "").strip(),
            "dna_alignment": str(payload.get("dna_alignment") or persona_context.get("style_dna") or "").strip(),
            "wardrobe_alignment": str(payload.get("wardrobe_alignment") or "").strip(),
        },
    }
    response = apply_personality_text_polish_to_final_payload(response, query=query)
    guarded, removed = apply_gender_guard_to_final_payload(
        response,
        target_gender=asset_gender,
        context=str(payload.get("occasion") or occasion or category or query),
    )
    return guarded


def reason(
    query: str,
    intent: dict | str | None = None,
    user_profile: dict | None = None,
    context: dict | None = None,
    wardrobe_summary: dict | None = None,
    history: list | None = None,
) -> Dict[str, Any]:
    del wardrobe_summary, history
    safe_query = str(query or "").strip()
    safe_profile = user_profile if isinstance(user_profile, dict) else {}
    safe_context = context if isinstance(context, dict) else {}
    mode = _coerce_mode(safe_query, intent, safe_context)
    category, tone, formality, occasion = _occasion_category(safe_query)
    confidence = _confidence(intent, 0.9 if mode != GENERAL else 0.55)
    if mode == STYLE_PAIRING:
        logger.info("AHVI_STYLE_PAIRING_ROUTE query=%r", safe_query[:120])
        logger.info(
            "AHVI_PAIRING_FLOW_ORDER step=general_suggestions auto_wardrobe=False auto_visual=False ctas=%s",
            ["Show visual inspiration", "Use my wardrobe", "Find missing pieces"],
        )

    if mode == WARDROBE_STYLE:
        return {
            "mode": WARDROBE_STYLE,
            "occasion": occasion,
            "tone": tone,
            "formality": formality,
            "should_use_wardrobe": True,
            "should_generate_board": True,
            "advice": "",
            "follow_up_question": None,
            "cta": _fallback_cta(safe_query),
            "visual_directions": [],
            "meta": {
                "source": "style_reasoning_engine",
                "reason": _reason_for_mode(mode, category),
                "goal": "Build the look from the user's wardrobe.",
                "atmosphere": _fallback_atmosphere(category),
                "emotion_state": _fallback_emotion(category),
                "confidence": confidence,
            },
        }

    if mode == GENERAL:
        return {
            "mode": GENERAL,
            "occasion": None,
            "tone": None,
            "formality": None,
            "should_use_wardrobe": False,
            "should_generate_board": False,
            "advice": "",
            "follow_up_question": None,
            "cta": [],
            "visual_directions": [],
            "meta": {
                "source": "style_reasoning_engine",
                "reason": "not_style_request",
                "goal": "",
                "atmosphere": "",
                "emotion_state": "neutral",
                "confidence": confidence,
            },
        }

    ai_payload: Dict[str, Any] | None = None
    try:
        ai_payload = _gemini_reasoning(
            query=safe_query,
            mode=mode,
            category=category,
            user_profile=safe_profile,
            context=safe_context,
        )
    except Exception as exc:
        logger.warning(
            "ahvi.style_reasoning_gemini_failed mode=%s err=%s", mode, repr(exc)[:200]
        )
        ai_payload = None

    built = _build_response(
        query=safe_query,
        mode=mode,
        category=category,
        tone=tone,
        formality=formality,
        occasion=occasion,
        confidence=confidence,
        ai_payload=ai_payload,
        user_profile=safe_profile,
        context=safe_context,
    )
    # Wardrobe grounding: when a wardrobe is available, any suggested garment
    # that is not owned must live in missing_piece (the prompt enforces this);
    # log that grounding was active so we can audit owned-vs-suggested.
    _wardrobe = safe_context.get("wardrobe") or safe_context.get("wardrobe_items")
    if isinstance(_wardrobe, list) and _wardrobe:
        logger.info(
            "AHVI_WARDROBE_GROUNDING_APPLIED wardrobe_items=%d has_missing_piece=%s",
            len(_wardrobe),
            bool(built.get("missing_piece")),
        )
    return built


class _StyleReasoningEngine:
    def reason(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return reason(*args, **kwargs)


style_reasoning_engine = _StyleReasoningEngine()
