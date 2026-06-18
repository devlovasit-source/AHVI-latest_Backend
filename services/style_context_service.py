"""Unified Style Context builder.

Assembles ONE style-context object from the sources AHVI already has
(query, wardrobe, profile/style-DNA, weather, calendar/event, last style
context). It does NOT create new databases or fetch on its own — callers
pass in whatever live data they hold (request.wardrobe, weather_data,
effective_user_profile, current_memory). Missing sources degrade to {}.

This is the single brief the Gemini Stylist Brain reasons over.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ahvi.style_context")

_STYLE_MODES = {"style_advice", "visual_inspiration", "wardrobe_style", "missing_pieces"}

# Lightweight category buckets for a wardrobe summary without a DB.
_CATEGORY_KEYS = (
    "top",
    "bottom",
    "footwear",
    "outerwear",
    "dress",
    "accessory",
    "bag",
    "ethnic",
)


def _norm(value: Any) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Multi-event / transition-outfit detection. Runs BEFORE generic occasion
# mapping so "basketball game at 6pm then team dinner at 10pm" is not flattened
# to "date night".
# --------------------------------------------------------------------------
_MULTI_EVENT_CONNECTORS = (
    " then ",
    " and then ",
    " followed by ",
    " after ",
    " before ",
    " later ",
    " into ",
    " and ",
    " to ",          # "basketball game to dinner", "office to drinks"
    " transition ",  # "transition from a game to dinner"
)

# generic_canon -> specific canons that suppress it (so "basketball game" is one
# event, not basketball_game + sports_game).
_EVENT_SUPPRESSORS = {
    "sports_game": {"basketball_game", "football_game", "soccer_game", "cricket_match", "tennis_match", "sports_practice"},
    "dinner": {"team_dinner"},
    "party": {"birthday_party", "house_party"},
    "office": {"office_meeting", "client_meeting", "conference"},
    "office_meeting": {"client_meeting", "client_presentation"},
    "work": {"office", "office_meeting", "client_meeting", "client_presentation", "interview", "conference"},
}

# keyword -> (canonical_sub_occasion, category)
_EVENT_LEXICON = (
    ("basketball", ("basketball_game", "active")),
    ("football", ("football_game", "active")),
    ("soccer", ("soccer_game", "active")),
    ("cricket", ("cricket_match", "active")),
    ("tennis", ("tennis_match", "active")),
    ("match", ("sports_game", "active")),
    ("game", ("sports_game", "active")),
    ("practice", ("sports_practice", "active")),
    ("gym", ("workout", "active")),
    ("workout", ("workout", "active")),
    ("training", ("workout", "active")),
    ("yoga", ("yoga", "active")),
    ("run", ("run", "active")),
    ("client presentation", ("client_presentation", "work")),
    ("presentation", ("client_presentation", "work")),
    ("client meeting", ("client_meeting", "work")),
    ("interview", ("interview", "work")),
    ("office meeting", ("office_meeting", "work")),
    ("meeting", ("office_meeting", "work")),
    ("office", ("office", "work")),
    ("conference", ("conference", "work")),
    ("work", ("work", "work")),
    ("drinks", ("drinks", "drinks")),
    ("cocktails", ("drinks", "drinks")),
    ("happy hour", ("drinks", "drinks")),
    ("team dinner", ("team_dinner", "social")),
    ("dinner", ("dinner", "social")),
    ("lunch", ("lunch", "social")),
    ("brunch", ("brunch", "social")),
    ("birthday party", ("birthday_party", "social")),
    ("birthday", ("birthday_party", "social")),
    ("house party", ("house_party", "social")),
    ("party", ("party", "social")),
    ("reception", ("reception", "social")),
    ("wedding", ("wedding", "social")),
    ("ceremony", ("ceremony", "ceremony")),
    ("date", ("date", "date")),
    ("movie", ("movie", "social")),
    ("concert", ("concert", "social")),
    ("travel", ("travel", "travel")),
    ("flight", ("travel", "travel")),
    ("airport", ("travel", "travel")),
)

# Higher = higher social/formality risk → leads the styling for a compound
# prompt. The style_reasoning prompt reads sub_occasions in order, so the
# dominant event must come first. gym/sports only lead when they're the only
# event (single-occasion prompts never reach detect_multi_event).
_EVENT_FORMALITY: Dict[str, int] = {
    "funeral": 100,
    "wedding": 92, "reception": 92, "ceremony": 90,
    "client_presentation": 82, "client_meeting": 80, "interview": 80,
    "conference": 80, "office_meeting": 72, "office": 70, "work": 70,
    "date": 66, "drinks": 64,
    "team_dinner": 60, "dinner": 60, "brunch": 56, "lunch": 56,
    "concert": 52, "birthday_party": 50, "house_party": 50, "party": 50,
    "movie": 46,
    "travel": 30,
    "basketball_game": 14, "football_game": 14, "soccer_game": 14,
    "cricket_match": 14, "tennis_match": 14, "sports_game": 14,
    "workout": 10, "yoga": 10, "run": 10, "sports_practice": 10,
}


def _dominant_occasion(sub_occasions: List[str]) -> str:
    if not sub_occasions:
        return ""
    return max(sub_occasions, key=lambda o: _EVENT_FORMALITY.get(o, 40))


_TIME_RE = re.compile(r"\b(\d{1,2})\s*(?::\s*(\d{2}))?\s*([ap])\.?m\.?\b", re.IGNORECASE)


def _extract_times(text: str) -> List[str]:
    out: List[str] = []
    for m in _TIME_RE.finditer(text):
        hour = int(m.group(1)) % 12
        minute = int(m.group(2) or 0)
        if m.group(3).lower() == "p":
            hour += 12
        out.append(f"{hour:02d}:{minute:02d}")
    return out


def _strategy_for(categories: List[str]) -> str:
    cats = set(categories)
    if "work" in cats and "drinks" in cats:
        return "day_to_night_transition"
    if "work" in cats and ("social" in cats or "date" in cats):
        return "work_to_social_transition"
    if "active" in cats and ("social" in cats or "drinks" in cats):
        return "transition_outfit"
    if "travel" in cats and "social" in cats:
        return "travel_to_dinner_transition"
    if "ceremony" in cats and "social" in cats:
        return "ceremony_to_reception_transition"
    return "transition_outfit"


def detect_multi_event(query: Any) -> Optional[Dict[str, Any]]:
    """Detect a multi-event / transition-outfit prompt. Returns structured
    context or None. Requires a sequencing connector AND >=2 distinct events
    so single-occasion prompts are never misclassified."""
    raw = str(query or "")
    q = " " + _norm(raw) + " "
    if not any(conn in q for conn in _MULTI_EVENT_CONNECTORS):
        return None

    # Collect events in order of appearance, de-duplicated by canonical name.
    found: List[tuple] = []
    seen: set = set()
    for kw, (canon, cat) in _EVENT_LEXICON:
        pos = q.find(" " + kw)
        if pos >= 0 and canon not in seen:
            seen.add(canon)
            found.append((pos, canon, cat))
    found.sort(key=lambda t: t[0])
    canon_set = {c for _, c, _ in found}
    # Drop generic events when a more specific sibling was also matched.
    kept = [
        (pos, canon, cat)
        for (pos, canon, cat) in found
        if not (_EVENT_SUPPRESSORS.get(canon) and _EVENT_SUPPRESSORS[canon] & canon_set)
    ]
    sub_occasions = [c for _, c, _ in kept]
    categories = [cat for _, _, cat in kept]
    if len(sub_occasions) < 2:
        return None
    # time_sequence keeps the real chronological order (as written).
    times = _extract_times(raw)
    time_sequence = [
        {"event": sub_occasions[i], "time": times[i] if i < len(times) else None}
        for i in range(len(sub_occasions))
    ]
    # Lead the styling with the higher-formality event. The reasoning prompt
    # reads sub_occasions in order, so a gym/sports event no longer steers a
    # "gym then brunch" look toward athleisure. Chronology preserved in
    # time_sequence above.
    dominant = _dominant_occasion(sub_occasions)
    ordered = [dominant] + [o for o in sub_occasions if o != dominant]
    result = {
        "occasion": "multi_event",
        "sub_occasions": ordered,
        "dominant_occasion": dominant,
        "style_strategy": _strategy_for(categories),
        "time_sequence": time_sequence,
    }
    logger.info(
        "AHVI_MULTI_EVENT_DETECTED sub_occasions=%s dominant=%s style_strategy=%s",
        ordered,
        dominant,
        result["style_strategy"],
    )
    return result


def _safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _category_of(item: Dict[str, Any]) -> str:
    raw = _norm(item.get("category") or item.get("type") or item.get("name"))
    for key in _CATEGORY_KEYS:
        if key in raw:
            return key
    if any(w in raw for w in ("shirt", "tee", "kurta", "blouse", "polo")):
        return "top"
    if any(w in raw for w in ("jean", "trouser", "pant", "chino", "short", "skirt")):
        return "bottom"
    if any(w in raw for w in ("shoe", "sneaker", "loafer", "heel", "boot", "sandal")):
        return "footwear"
    return "other"


def _wardrobe_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cat: Dict[str, int] = {}
    for it in items:
        if isinstance(it, dict):
            by_cat[_category_of(it)] = by_cat.get(_category_of(it), 0) + 1
    return {
        "total_items": len(items),
        "by_category": by_cat,
        "has_top": by_cat.get("top", 0) > 0,
        "has_bottom": by_cat.get("bottom", 0) > 0,
        "has_footwear": by_cat.get("footwear", 0) > 0,
    }


def _image_url(item: Dict[str, Any]) -> str:
    for key in (
        "normalized_url",
        "normalizedUrl",
        "masked_url",
        "maskedUrl",
        "image_url",
        "imageUrl",
        "image",
        "url",
        "thumbnail",
        "photo",
    ):
        val = str(item.get(key) or "").strip()
        if val:
            return val
    return ""


def _normalize_wardrobe_items(raw_items: Any) -> List[Dict[str, Any]]:
    # Central choke point: every style flow builds its wardrobe view from
    # here, so non-fashion rows (chargers, skincare, travel gear) are removed
    # once and can never leak into prompts, ownership or missing-piece UI.
    from services.wardrobe_sanitizer import sanitize_fashion_wardrobe_items

    fashion_only = sanitize_fashion_wardrobe_items(
        _safe_list(raw_items), source="build_style_context"
    )
    out: List[Dict[str, Any]] = []
    for it in fashion_only:
        out.append(
            {
                "id": str(it.get("id") or it.get("$id") or it.get("item_id") or "").strip(),
                "name": str(it.get("name") or it.get("title") or "").strip(),
                "category": _category_of(it),
                "image_url": _image_url(it),
                "color": str(it.get("color") or it.get("colour") or "").strip(),
                "_raw": it,
            }
        )
    return out


def build_style_context(
    *,
    query: str,
    occasion: Optional[str] = None,
    mode: str = "style_advice",
    wardrobe_items: Any = None,
    weather: Any = None,
    event_context: Any = None,
    user_profile: Any = None,
    last_style_context: Any = None,
    user_id: str = "",
    memory: Any = None,
) -> Dict[str, Any]:
    """Assemble the unified style context. All inputs optional; absent
    sources become empty structures so the prompt stays well-formed."""
    safe_mode = str(mode or "style_advice").strip().lower()
    if safe_mode not in _STYLE_MODES:
        safe_mode = "style_advice"

    items = _normalize_wardrobe_items(wardrobe_items)
    profile = _safe_dict(user_profile)

    # Multi-event / transition-outfit detection BEFORE generic occasion use.
    multi_event = detect_multi_event(query)
    # Style DNA / preferences are read from the profile if present; we do not
    # fabricate them.
    style_dna = _safe_dict(profile.get("style_dna") or profile.get("styleDNA"))
    preferences = _safe_dict(
        profile.get("style_preferences")
        or profile.get("preferences")
        or profile.get("style")
    )

    resolved_occasion = (str(occasion or "").strip().lower() or None)
    if multi_event:
        resolved_occasion = "multi_event"

    # Style memory (wear + saved-board). Use caller-provided memory if given,
    # else load from producers. Always present (neutral when no data).
    if isinstance(memory, dict):
        style_memory = memory
    else:
        try:
            from services.style_memory_service import build_style_memory_context

            style_memory = build_style_memory_context(user_id, items)
        except Exception:  # noqa: BLE001
            style_memory = {}

    context = {
        "query": str(query or "").strip(),
        "occasion": resolved_occasion,
        "mode": safe_mode,
        "wardrobe_available": len(items) > 0,
        "wardrobe_summary": _wardrobe_summary(items),
        "wardrobe_items": items,
        "weather_context": _safe_dict(weather),
        "event_context": _safe_dict(event_context),
        "style_dna": style_dna,
        "preferences": preferences,
        "last_style_context": _safe_dict(last_style_context),
        # Multi-event / transition context (None for single-occasion prompts).
        "multi_event": multi_event,
        "sub_occasions": multi_event["sub_occasions"] if multi_event else [],
        "style_strategy": multi_event["style_strategy"] if multi_event else None,
        "time_sequence": multi_event["time_sequence"] if multi_event else [],
        # Style memory signals (neutral when no data).
        "recently_worn_ids": (style_memory or {}).get("recently_worn_ids", []),
        "underworn_ids": (style_memory or {}).get("underworn_ids", []),
        "saved_item_ids": (style_memory or {}).get("saved_item_ids", []),
        "disliked_item_ids": (style_memory or {}).get("disliked_item_ids", []),
        "favorite_colors": (style_memory or {}).get("favorite_colors", []),
        "favorite_categories": (style_memory or {}).get("favorite_categories", []),
        "saved_board_patterns": (style_memory or {}).get("saved_board_patterns", []),
    }

    logger.info(
        "AHVI_STYLE_CONTEXT_BUILT mode=%s occasion=%s wardrobe_items=%d "
        "wardrobe_available=%s weather=%s event=%s",
        context["mode"],
        context["occasion"],
        len(items),
        context["wardrobe_available"],
        bool(context["weather_context"]),
        bool(context["event_context"]),
    )
    return context


_ROLE_BY_CATEGORY = {
    "top": "hero",
    "dress": "hero",
    "outerwear": "support",
    "bottom": "support",
    "footwear": "footwear",
    "accessory": "accessory",
    "bag": "accessory",
}


def build_editorial_wardrobe_board(
    *,
    title: str,
    goal: str,
    impression: str,
    stylist_reasoning: str,
    wardrobe_items: List[Dict[str, Any]],
    palette: Optional[List[str]] = None,
    why_it_works: str = "",
    missing_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Editorial wardrobe board built from the user's ACTUAL items (with real
    images). Not a generic module-card list. Returns {} if no usable items."""
    used: List[Dict[str, Any]] = []
    seen_roles_hero = False
    for it in wardrobe_items or []:
        if not isinstance(it, dict):
            continue
        cat = it.get("category") or "other"
        role = _ROLE_BY_CATEGORY.get(cat, "support")
        if role == "hero" and seen_roles_hero:
            role = "support"
        if role == "hero":
            seen_roles_hero = True
        used.append(
            {
                "id": it.get("id", ""),
                "name": it.get("name", ""),
                "category": cat,
                "image_url": it.get("image_url", ""),
                "role": role,
            }
        )
    if not used:
        return {}
    board = {
        "type": "editorial_wardrobe_board",
        "title": title,
        "goal": goal,
        "impression": impression,
        "stylist_reasoning": stylist_reasoning,
        "used_wardrobe_items": used[:8],
        "missing_items": missing_items or [],
        "palette": (palette or [])[:5],
        "why_it_works": why_it_works,
    }
    logger.info(
        "AHVI_EDITORIAL_BOARD_BUILT items=%d hero=%s title=%r",
        len(board["used_wardrobe_items"]),
        seen_roles_hero,
        title[:50],
    )
    return board


def build_missing_piece_intelligence(
    *,
    wardrobe_summary: Dict[str, Any],
    missing_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """missing_piece_intelligence block. owned_percentage is a rough coverage
    signal from the wardrobe summary; missing_items carry reason + unlocks."""
    summary = _safe_dict(wardrobe_summary)
    total = int(summary.get("total_items") or 0)
    # Coverage heuristic: how many of the core slots are present.
    core_present = sum(
        1 for k in ("has_top", "has_bottom", "has_footwear") if summary.get(k)
    )
    owned_pct = int(round((core_present / 3) * 100)) if total else 0
    from services.wardrobe_sanitizer import is_fashion_item

    cleaned: List[Dict[str, Any]] = []
    for m in missing_items or []:
        if not isinstance(m, dict):
            continue
        # Missing pieces are stylist recommendations — a charger or skincare
        # bottle here is as trust-breaking as in the ownership chips.
        if not is_fashion_item(m):
            logger.debug(
                "AHVI_NON_FASHION_ITEM_REMOVED name=%r category=%r source=missing_piece_intelligence",
                str(m.get("name") or "")[:60],
                str(m.get("category") or "")[:40],
            )
            continue
        cleaned.append(
            {
                "name": str(m.get("name") or "").strip(),
                "category": str(m.get("category") or "").strip(),
                "reason": str(m.get("reason") or "").strip(),
                "image_url": str(m.get("image_url") or m.get("imageUrl") or "").strip(),
                "asset_id": str(m.get("asset_id") or "").strip(),
                "unlocks": [str(u).strip() for u in _safe_list(m.get("unlocks")) if str(u).strip()][:6],
            }
        )
    return {
        "type": "missing_piece_intelligence",
        "owned_percentage": owned_pct,
        "missing_items": cleaned,
    }


def compact_style_dna(style_dna: Any, preferences: Any) -> Dict[str, Any]:
    """Build a compact, prompt-safe Style DNA contract from the user's stored
    DNA + preferences. Returns {} when nothing meaningful exists, so Gemini
    never hallucinates personalization for a blank profile."""
    dna = _safe_dict(style_dna)
    prefs = _safe_dict(preferences)

    def _top_archetypes(value: Any, limit: int = 3) -> List[str]:
        d = _safe_dict(value)
        ranked = sorted(
            ((k, v) for k, v in d.items() if isinstance(v, (int, float)) and v > 0),
            key=lambda kv: kv[1],
            reverse=True,
        )
        out = [k for k, _ in ranked][:limit]
        if out:
            return out
        # plain list form
        return [str(x) for x in _safe_list(value)][:limit]

    color = _safe_dict(dna.get("color_dna"))
    sil = _safe_dict(dna.get("silhouette_dna"))

    contract = {
        "style_archetypes": _top_archetypes(dna.get("style_archetypes"))
        or [str(x) for x in _safe_list(prefs.get("archetypes"))][:3],
        "preferred_colors": [str(x) for x in (color.get("core_colors") or color.get("power_colors") or prefs.get("colors") or [])][:6],
        "avoided_colors": [str(x) for x in (color.get("avoided_colors") or prefs.get("avoided_colors") or [])][:6],
        "preferred_silhouettes": [str(x) for x in (sil.get("preferred_fits") or sil.get("preferred_shapes") or prefs.get("silhouettes") or [])][:5],
        "preferred_formality": str(
            dna.get("preferred_formality")
            or _safe_dict(dna.get("style_identity")).get("formality")
            or prefs.get("formality")
            or ""
        ).strip(),
        "preferred_style_keywords": [str(x) for x in (prefs.get("style_keywords") or _safe_list(dna.get("style_keywords")))][:6],
        "avoid_style_keywords": [str(x) for x in (prefs.get("avoid_keywords") or _safe_list(dna.get("avoid_style_keywords")))][:6],
    }
    # Drop empty fields; if nothing populated, return {}.
    contract = {k: v for k, v in contract.items() if v}
    if contract:
        logger.info(
            "AHVI_STYLE_DNA_CONTEXT_USED keys=%s archetypes=%s",
            sorted(contract.keys()),
            contract.get("style_archetypes"),
        )
    return contract


_MALE_GENDER_TOKENS = {"male", "man", "men", "mens", "masculine", "m"}
_FEMALE_GENDER_TOKENS = {"female", "woman", "women", "womens", "feminine", "f"}


def _resolve_gender(profile: Dict[str, Any]) -> str:
    p = _safe_dict(profile)
    candidates = [
        p.get("style_gender"), p.get("gender"), p.get("preferred_gender"),
        p.get("target_gender"),
    ]
    for key in ("preferences", "style_preferences", "stylePreference"):
        nested = _safe_dict(p.get(key))
        candidates += [nested.get("style_gender"), nested.get("gender")]
    for c in candidates:
        t = _norm(c)
        if t in _MALE_GENDER_TOKENS:
            return "male"
        if t in _FEMALE_GENDER_TOKENS:
            return "female"
    return "unknown"


def build_pairing_persona(
    *,
    user_profile: Any = None,
    style_dna: Any = None,
    wardrobe_summary: Any = None,
) -> Dict[str, Any]:
    """Compact persona context for persona-aware pairing. Neutral when the
    profile is empty — never assume gender beyond available signals."""
    profile = _safe_dict(user_profile)
    gender = _resolve_gender(profile)
    dna = compact_style_dna(style_dna, profile.get("style_preferences") or profile.get("preferences"))
    summary = _safe_dict(wardrobe_summary)
    by_cat = _safe_dict(summary.get("by_category"))

    avoid_cats: List[str] = []
    if gender == "male":
        avoid_cats = ["skirt", "dress", "camisole", "heels"]

    confidence = 0.0
    if gender != "unknown":
        confidence += 0.6
    if dna:
        confidence += 0.25
    if by_cat:
        confidence += 0.15

    persona = {
        "gender_profile": gender,
        "preferred_fit": str(profile.get("preferred_fit") or profile.get("fit") or "").strip(),
        "style_dna": dna.get("style_archetypes", []) if isinstance(dna, dict) else [],
        "preferred_categories": [k for k, v in by_cat.items() if v] if by_cat else [],
        "avoid_categories": avoid_cats,
        "wardrobe_gender_signal": gender if by_cat else "unknown",
        "persona_confidence": round(min(confidence, 1.0), 2),
    }
    logger.info(
        "AHVI_PAIRING_PERSONA_CONTEXT gender=%s dna=%d cats=%d confidence=%.2f",
        gender, len(persona["style_dna"]), len(persona["preferred_categories"]),
        persona["persona_confidence"],
    )
    return persona


# Ethnic occasion families + item signals. When the occasion is NOT one of these
# (and the user did not ask for desi/fusion), Indian ethnic archetypes/items are
# forbidden — so a music festival can never produce kurta/bandhgala/mojari.
_ETHNIC_FAMILIES = {"festive_general", "festive_daytime", "festive_evening", "temple_modest"}
_ETHNIC_ARCHETYPES = (
    "Festive Heritage", "Refined Traditional", "Celebration Kurta", "Sangeet Statement",
    "Sunlit Traditional", "Wedding Day Ease", "Temple Modest",
)
_ETHNIC_ITEM_SIGNALS = (
    "kurta", "kurta pajama", "kurta pyjama", "sherwani", "bandhgala", "bandi vest",
    "nehru jacket", "mojari", "jutti", "juti", "churidar", "dhoti", "kolhapuri",
    "lehenga", "saree", "sari", "dupatta", "achkan", "pathani", "anarkali",
)
_DESI_CUES = (
    "desi", "ethnic", "fusion", "indo", "indian", "kurta", "sherwani", "traditional",
    "sangeet", "mehendi", "haldi", "diwali", "eid", "navratri", "saree", "lehenga",
)


# Numeric occasion axes keyed by occasion_family. ONE table = single source for
# formality (1..5), energy (1..9), movement (1..9) + required_traits. Lets the
# asset scorer judge whether a piece *feels* right for the occasion (festival =
# low formality, high energy + movement) instead of only matching keywords. Keys
# match _FAMILY_ARCHETYPE_POOL families; "concert_social" is an alias of
# social_party for the music-festival/concert path (occasion_style_rules).
OCCASION_FAMILY_PROFILE: Dict[str, Dict[str, Any]] = {
    "concert_social": {
        "formality": 2, "energy": 9, "movement": 9,
        "required_traits": ["comfortable", "expressive", "movement_ready"],
    },
    "social_party": {
        "formality": 2, "energy": 8, "movement": 7,
        "required_traits": ["expressive", "comfortable", "movement_ready"],
    },
    "professional": {
        "formality": 5, "energy": 3, "movement": 3,
        "required_traits": ["polished", "structured"],
    },
    "travel_easy": {
        "formality": 2, "energy": 4, "movement": 9,
        "required_traits": ["comfortable", "movement_ready"],
    },
    "relaxed_casual": {
        "formality": 2, "energy": 4, "movement": 6,
        "required_traits": ["comfortable"],
    },
    "evening_date": {
        "formality": 4, "energy": 5, "movement": 4,
        "required_traits": ["polished"],
    },
    "festive_general": {
        "formality": 4, "energy": 7, "movement": 4,
        "required_traits": ["expressive", "polished"],
    },
    "festive_daytime": {
        "formality": 3, "energy": 7, "movement": 5,
        "required_traits": ["expressive", "comfortable"],
    },
    "festive_evening": {
        "formality": 4, "energy": 8, "movement": 4,
        "required_traits": ["expressive"],
    },
    "christian_ceremony": {
        "formality": 5, "energy": 3, "movement": 3,
        "required_traits": ["polished", "structured"],
    },
    "resort_summer": {
        "formality": 2, "energy": 5, "movement": 6,
        "required_traits": ["comfortable"],
    },
    "somber_formal": {
        "formality": 5, "energy": 1, "movement": 2,
        "required_traits": ["polished", "structured"],
    },
    "temple_modest": {
        "formality": 3, "energy": 3, "movement": 4,
        "required_traits": ["modest", "comfortable"],
    },
}

# Neutral mid profile when no family resolves — never penalises (distance small).
_DEFAULT_FAMILY_PROFILE: Dict[str, Any] = {
    "formality": 3, "energy": 5, "movement": 5, "required_traits": [],
}


def occasion_family_profile(family: Any) -> Dict[str, Any]:
    """Return the numeric axis profile for an occasion family, defaulting to a
    neutral mid profile so an unknown family never vetoes by distance."""
    return dict(OCCASION_FAMILY_PROFILE.get(str(family or "").strip(), _DEFAULT_FAMILY_PROFILE))


def _resolve_brief_archetypes(canonical_occasion: Any, query: Any):
    """Return (occasion_family, cultural_context, allowed, forbidden_arch,
    forbidden_items). Reuses the visual path's own family resolver + pool so the
    brief speaks the same language as select_archetypes. Fails open to neutral."""
    try:
        from services.stylist_knowledge_service import (
            _FAMILY_ARCHETYPE_POOL,
            _resolve_occasion_family,
        )
    except Exception:  # noqa: BLE001
        return "", "neutral", [], [], []
    fam = _resolve_occasion_family(str(canonical_occasion or "")) or _resolve_occasion_family(_norm(query))
    allowed = list(_FAMILY_ARCHETYPE_POOL.get(fam, ()))
    desi = any(cue in _norm(query) for cue in _DESI_CUES)
    if fam in _ETHNIC_FAMILIES or desi:
        return fam, "indian_ethnic", allowed, [], []
    return fam, "western", allowed, list(_ETHNIC_ARCHETYPES), list(_ETHNIC_ITEM_SIGNALS)


def build_canonical_style_context(
    *,
    query: str,
    user_profile: Any = None,
    intent: Optional[Dict[str, Any]] = None,
    router_occasion: Optional[str] = None,
    weather: Any = None,
    event_context: Optional[Dict[str, Any]] = None,
    style_dna: Any = None,
) -> Dict[str, Any]:
    """Single canonical style context shared by every visual/style flow.

    Pure assembly over functions the wardrobe path already uses:
    - occasion via ``build_brief`` (same engine the wardrobe path resolves with)
    - gender via ``_resolve_gender``
    - DNA via ``compact_style_dna``

    Does NOT mutate the caller's context. Fails open: any resolver error
    degrades to safe defaults so the visual path never breaks.
    """
    profile = _safe_dict(user_profile)

    # 1. canonical occasion via the SAME engine the wardrobe path uses.
    brief: Dict[str, Any] = {}
    canonical_occasion = "daily"
    try:
        from brain.engines.style_brief import build_brief

        brief = build_brief(
            query,
            router_occasion=router_occasion,
            agent_payload=intent or {},
            weather=weather,
        )
        canonical_occasion = str(brief.get("occasion") or "daily") or "daily"
    except Exception:  # noqa: BLE001 — fail open to a safe default.
        logger.warning("AHVI_CANONICAL_CTX brief_failed query=%r", str(query or "")[:80])
        brief = {}
        canonical_occasion = str(router_occasion or "daily").strip().lower() or "daily"

    # 2. gender via the single resolver (prompt-override handled by callers).
    try:
        gender = _resolve_gender(profile)
    except Exception:  # noqa: BLE001
        gender = "unknown"

    # 3. DNA compacted (empty dict when nothing meaningful exists).
    try:
        dna = compact_style_dna(
            style_dna if style_dna is not None else profile.get("style_dna"),
            profile.get("preferences") or profile.get("style_preferences"),
        )
    except Exception:  # noqa: BLE001
        dna = {}

    # 4. Occasion family + cultural gating (single authority for the board path).
    try:
        family, cultural, allowed_arch, forbidden_arch, forbidden_items = _resolve_brief_archetypes(
            canonical_occasion, query
        )
    except Exception:  # noqa: BLE001
        family, cultural, allowed_arch, forbidden_arch, forbidden_items = "", "neutral", [], [], []

    # 5. Numeric axes (formality/energy/movement + required_traits) from the ONE
    #    family-profile table. This is what lets the asset scorer + board guard
    #    judge authenticity by *feel*, not just keyword/archetype match.
    try:
        axes = occasion_family_profile(family)
    except Exception:  # noqa: BLE001
        axes = dict(_DEFAULT_FAMILY_PROFILE)

    ctx = {
        "canonical_occasion": canonical_occasion,
        "occasion_brief": brief,
        "gender": gender,
        "style_dna": dna,
        "profile": profile,
        "weather": weather,
        "event_context": _safe_dict(event_context),
        # Canonical Style Brain fields — one source of truth for selection/guard.
        "occasion_family": family,
        "cultural_context": cultural,
        "allowed_archetypes": allowed_arch,
        "forbidden_archetypes": forbidden_arch,
        "forbidden_item_signals": forbidden_items,
        # Numeric axes — formality 1..5, energy/movement 1..9, required_traits.
        "formality": axes.get("formality"),
        "energy": axes.get("energy"),
        "movement": axes.get("movement"),
        "required_traits": list(axes.get("required_traits") or []),
    }
    logger.info(
        "style_context.built canonical_occasion=%s family=%s cultural=%s gender=%s "
        "formality=%s energy=%s movement=%s forbidden_arch=%d forbidden_items=%d",
        canonical_occasion,
        family,
        cultural,
        gender,
        axes.get("formality"),
        axes.get("energy"),
        axes.get("movement"),
        len(forbidden_arch),
        len(forbidden_items),
    )
    return ctx


def compact_context_for_prompt(context: Dict[str, Any]) -> Dict[str, Any]:
    """Trim the style context to a small, prompt-safe slice. Avoids shipping
    every wardrobe item / raw payloads into the Gemini prompt."""
    ctx = _safe_dict(context)
    items = _safe_list(ctx.get("wardrobe_items"))[:18]
    style_dna_compact = compact_style_dna(ctx.get("style_dna"), ctx.get("preferences"))

    # Compact memory slice for the prompt — map worn IDs to names so Gemini can
    # reference real pieces. Present only when memory exists (else None).
    _recent_ids = {str(x) for x in (ctx.get("recently_worn_ids") or [])}
    _id_to_name = {}
    for it in _safe_list(ctx.get("wardrobe_items")):
        if isinstance(it, dict):
            _id_to_name[str(it.get("id") or "")] = it.get("name") or ""
    recently_worn_names = [
        _id_to_name[i] for i in _recent_ids if _id_to_name.get(i)
    ][:5]
    memory_slice = None
    if recently_worn_names or ctx.get("favorite_colors"):
        memory_slice = {
            "recently_worn": recently_worn_names,
            "favorite_colors": (ctx.get("favorite_colors") or [])[:5],
        }
        logger.info(
            "AHVI_STYLE_MEMORY_CONTEXT_USED recent_named=%d fav_colors=%d",
            len(recently_worn_names), len(ctx.get("favorite_colors") or []),
        )

    return {
        "query": ctx.get("query", ""),
        "occasion": ctx.get("occasion"),
        "mode": ctx.get("mode", "style_advice"),
        "wardrobe_available": ctx.get("wardrobe_available", False),
        "wardrobe_summary": ctx.get("wardrobe_summary", {}),
        "wardrobe_items": [
            {"name": it.get("name"), "category": it.get("category"), "color": it.get("color")}
            for it in items
            if isinstance(it, dict)
        ],
        "weather": {
            k: ctx.get("weather_context", {}).get(k)
            for k in ("condition", "temp_c", "temperature", "summary")
            if isinstance(ctx.get("weather_context"), dict)
            and ctx.get("weather_context", {}).get(k) is not None
        },
        "preferences": ctx.get("preferences", {}),
        "style_dna": style_dna_compact or None,
        "style_dna_archetypes": style_dna_compact.get("style_archetypes"),
        "last_style_mode": _safe_dict(ctx.get("last_style_context")).get("last_style_mode"),
        "base_occasion": _safe_dict(ctx.get("last_style_context")).get("base_occasion"),
        "sub_occasions": ctx.get("sub_occasions") or [],
        "style_strategy": ctx.get("style_strategy"),
        "time_sequence": ctx.get("time_sequence") or [],
        "memory": memory_slice,
    }
