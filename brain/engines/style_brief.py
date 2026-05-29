"""AHVI Style Brief — turn user intent into a structured stylist contract,
then validate boards against it.

Core principle: a wrong-occasion board is worse than no board.

This module replaces the legacy substring keyword matching with token-aware
detection. "workout" must never resolve to "office" via a "work" substring.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ahvi.style_brief")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_PHRASE_TOKENS: Dict[str, str] = {
    "client meeting": "client_meeting",
    "client-meeting": "client_meeting",
    "date night": "date_night",
    "date-night": "date_night",
    "night out": "night_out",
    "house party": "house_party",
    "happy hour": "happy_hour",
    "after hours": "after_hours",
    "after-hours": "after_hours",
    "coffee run": "coffee_run",
    "weekend coffee": "coffee_run",
    "smart casual": "smart_casual",
    "business casual": "business_casual",
    "open mic": "open_mic",
    "rooftop bar": "rooftop_bar",
    "office today": "office",
    "today office": "office",
}


def _normalize_phrases(text: str) -> str:
    out = " " + text.lower().strip() + " "
    for phrase, replacement in _PHRASE_TOKENS.items():
        out = out.replace(" " + phrase + " ", " " + replacement + " ")
    return out.strip()


def tokenize(text: Any) -> List[str]:
    """Return whole-word tokens from a query, after replacing known phrases."""
    raw = _normalize_phrases(str(text or "").lower())
    return [t for t in re.split(r"[^a-z0-9_]+", raw) if t]


# ---------------------------------------------------------------------------
# Occasion vocabulary — token-driven, NOT substring
# ---------------------------------------------------------------------------

_OCCASION_TOKENS: Dict[str, set] = {
    # ORDER OF EVALUATION MATTERS: more specific buckets first.
    "workout": {
        "workout",
        "gym",
        "fitness",
        "training",
        "yoga",
        "running",
        "lift",
        "cardio",
        "session",
        "strength",
    },
    "client_meeting": {
        "client_meeting",
        "client",
        "boardroom",
        "presentation",
        "pitch",
        "interview",
        "investor",
    },
    "office": {
        "office",
        "work",
        "meeting",
        "business",
        "corporate",
        "deskwork",
        "wfh",
    },
    "date_night": {
        "date_night",
        "date",
        "dinner",
        "tonight",
        "rooftop_bar",
        "after_hours",
    },
    "beach": {"beach", "pool", "seaside", "coastal", "resort", "shore"},
    "brunch": {"brunch"},
    "rave": {"rave", "club", "edm", "festival"},
    "cocktail": {"cocktail"},
    "wedding": {"wedding", "reception", "ceremony", "sangeet"},
    "party": {"party", "house_party", "night_out", "happy_hour", "celebration"},
    "travel": {"travel", "airport", "flight", "vacation", "trip", "transit"},
    "temple_modest": {"temple", "mandir", "pooja", "puja", "shrine", "darshan", "religious"},
    "daily": {"daily", "today"},
    "casual": {"casual", "weekend", "errand", "coffee_run", "coffee"},
}

# When BOTH client and office tokens fire, prefer client_meeting.
_OCCASION_PRIORITY: List[str] = [
    "workout",          # workout MUST beat office to fix the "work in workout" bug.
    "client_meeting",
    "wedding",
    "beach",
    "rave",
    "cocktail",
    "date_night",
    "party",
    "travel",
    "temple_modest",
    "brunch",
    "office",
    "casual",
    "daily",
]


def detect_occasion_from_tokens(query: Any) -> Tuple[str, List[str]]:
    """Return (occasion, matched_tokens). Empty occasion means unknown."""
    tokens = set(tokenize(query))
    if not tokens:
        return "", []
    for occ in _OCCASION_PRIORITY:
        wanted = _OCCASION_TOKENS.get(occ, set())
        hit = tokens & wanted
        if hit:
            return occ, sorted(hit)
    return "", []


# ---------------------------------------------------------------------------
# Brief schema + builder
# ---------------------------------------------------------------------------

# Per-occasion stylist contract.
# allowed/forbidden refer to tokenized item text (name + category + sub_cat etc.).
_OCCASION_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "workout": {
        "sub_intent": "gym",
        "formality": "low",
        "movement_requirement": "high",
        "polish_requirement": "low",
        "required_slots": ["active_top", "active_bottom", "athletic_footwear"],
        "preferred_item_signals": [
            "tee", "tank", "training", "track", "shorts", "joggers",
            "sneakers", "running", "sports bra", "compression",
        ],
        "forbidden_item_signals": [
            "loafer", "loafers", "trouser", "trousers", "button down",
            "button-down", "blazer", "suit", "belt", "office", "sequined",
            "slides", "sandal", "sandals", "saree", "lehenga", "kurta",
        ],
        "board_mood": ["clean", "performance", "functional"],
        "allowed_badges": ["GYM", "WORKOUT", "TRAINING"],
        "allowed_titles": ["Gym Training", "Strength Session", "Cardio Ready"],
    },
    "office": {
        "sub_intent": "office",
        "formality": "mid",
        "movement_requirement": "low",
        "polish_requirement": "high",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": [
            "shirt", "button down", "polo", "trouser", "trousers", "chino",
            "chinos", "loafer", "loafers", "derby", "oxford", "belt",
            "watch", "blazer",
        ],
        "forbidden_item_signals": [
            "shorts", "slides", "sandal", "sandals", "tank", "boxer",
            "boxers", "gym", "track", "training", "sequined", "rave",
            "tropical",
        ],
        "board_mood": ["composed", "polished", "intentional"],
        "allowed_badges": ["OFFICE", "BOARDROOM"],
        "allowed_titles": ["Boardroom Casual", "Creative Professional", "Clean Friday"],
    },
    "client_meeting": {
        "sub_intent": "client_meeting",
        "formality": "high",
        "movement_requirement": "low",
        "polish_requirement": "high",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": [
            "button down", "shirt", "trouser", "trousers", "loafer",
            "loafers", "belt", "watch", "blazer", "derby", "oxford",
        ],
        "forbidden_item_signals": [
            "shorts", "slides", "sandal", "sandals", "tank", "tee",
            "t-shirt", "boxer", "boxers", "gym", "track", "training",
            "sequined", "tropical", "saree", "lehenga",
        ],
        "board_mood": ["composed", "polished", "client-ready"],
        "allowed_badges": ["CLIENT READY", "BOARDROOM", "OFFICE"],
        "allowed_titles": ["Client Polish", "Minimal Executive", "Boardroom Casual"],
    },
    "date_night": {
        "sub_intent": "dinner_date",
        "formality": "mid",
        "movement_requirement": "low",
        "polish_requirement": "high",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": [
            "black shirt", "shirt", "loafer", "loafers", "watch",
            "chelsea boots", "dark", "intentional",
        ],
        "forbidden_item_signals": [
            "shorts", "slides", "sandal", "sandals", "gym", "track",
            "training", "office", "boardroom", "saree", "lehenga",
            "sequined", "tropical",
        ],
        "board_mood": ["soft", "evening", "intentional"],
        "allowed_badges": ["DATE NIGHT", "DINNER", "EVENING"],
        "allowed_titles": ["Dinner Polish", "After-Dark Edit", "Soft Statement"],
    },
    "beach": {
        "sub_intent": "beach",
        "formality": "low",
        "movement_requirement": "medium",
        "polish_requirement": "low",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": [
            "linen", "cotton", "shorts", "sandal", "sandals", "tee",
            "shirt", "slides",
        ],
        "forbidden_item_signals": [
            "blazer", "suit", "trouser", "trousers", "loafer", "loafers",
            "boots", "office", "client",
        ],
        "board_mood": ["coastal", "easy", "light"],
        "allowed_badges": ["BEACH", "COASTAL"],
        "allowed_titles": ["Coastal Ease", "Resort Casual", "Light Vacation"],
    },
    "travel": {
        "sub_intent": "travel",
        "formality": "mid",
        "movement_requirement": "high",
        "polish_requirement": "mid",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": [
            "sneaker", "sneakers", "joggers", "hoodie", "shirt", "tee",
            "track", "comfort", "layer",
        ],
        "forbidden_item_signals": [
            "saree", "lehenga", "stiletto", "heels", "boardroom",
        ],
        "board_mood": ["comfort", "transit", "layered"],
        "allowed_badges": ["TRAVEL", "TRANSIT"],
        "allowed_titles": ["Long-Day Comfort", "Polished Transit", "Layered Travel"],
    },
    "party": {
        "sub_intent": "house_party",
        "formality": "mid",
        "movement_requirement": "medium",
        "polish_requirement": "mid",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": ["statement", "jacket", "boots", "denim", "shirt"],
        "forbidden_item_signals": [
            "client", "office", "boardroom", "saree", "gym", "track",
        ],
        "board_mood": ["after-hours", "social", "edited"],
        "allowed_badges": ["PARTY", "AFTER HOURS"],
        "allowed_titles": ["After-Hours Edit", "Cocktail Sharp", "House Easy"],
    },
    "wedding": {
        "sub_intent": "ceremony",
        "formality": "high",
        "movement_requirement": "low",
        "polish_requirement": "high",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": [
            "kurta", "sherwani", "suit", "blazer", "trouser", "trousers",
            "loafer", "loafers", "watch",
        ],
        "forbidden_item_signals": [
            "shorts", "slides", "tee", "t-shirt", "gym", "track", "tank",
        ],
        "board_mood": ["respectful", "composed", "ceremony"],
        "allowed_badges": ["WEDDING", "CEREMONY", "FORMAL"],
        "allowed_titles": ["Composed Formal", "Reception Polish", "Ceremony Refined"],
    },
    "casual": {
        "sub_intent": "casual",
        "formality": "low",
        "movement_requirement": "medium",
        "polish_requirement": "low",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": ["tee", "shirt", "denim", "sneaker", "sneakers", "hoodie"],
        "forbidden_item_signals": ["client", "boardroom", "saree", "sequined"],
        "board_mood": ["easy", "clean", "daily"],
        "allowed_badges": ["CASUAL", "DAILY"],
        "allowed_titles": ["Clean Daily", "Refined Casual", "Easy Daily"],
    },
    "daily": {
        "sub_intent": "daily",
        "formality": "mid",
        "movement_requirement": "medium",
        "polish_requirement": "mid",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": ["shirt", "trouser", "trousers", "sneaker", "sneakers"],
        "forbidden_item_signals": ["sequined", "ceremony"],
        "board_mood": ["clean", "smart", "daily"],
        "allowed_badges": ["DAILY", "TODAY"],
        "allowed_titles": ["Polished Neutral", "Sharp Daily", "Smart Ease"],
    },
}


_DEFAULT_CONTRACT: Dict[str, Any] = {
    "sub_intent": "outfit_generation",
    "formality": "mid",
    "movement_requirement": "medium",
    "polish_requirement": "mid",
    "required_slots": ["top", "bottom", "footwear"],
    "preferred_item_signals": [],
    "forbidden_item_signals": [],
    "board_mood": ["clean"],
    "allowed_badges": ["DAILY"],
    "allowed_titles": ["Considered Look"],
}


def build_brief(
    query: Any = "",
    *,
    router_occasion: Optional[str] = None,
    agent_payload: Optional[Dict[str, Any]] = None,
    weather: Any = None,
) -> Dict[str, Any]:
    """Resolve a structured style brief.

    Precedence:
    1. Router-supplied occasion wins when present, UNLESS the agent payload
       has confidence >= 0.8 with a different occasion (explicit conflict).
    2. Otherwise, token-detected occasion from the query.
    3. Otherwise, falls back to "daily".
    """
    agent_payload = agent_payload if isinstance(agent_payload, dict) else {}
    try:
        agent_conf = float(agent_payload.get("confidence") or 0.0)
    except Exception:
        agent_conf = 0.0
    agent_occ_raw = str(agent_payload.get("occasion") or "").strip().lower()

    router_occ = str(router_occasion or "").strip().lower() or None
    token_occ, matched_tokens = detect_occasion_from_tokens(query)

    occasion = ""
    chosen_from = ""
    if router_occ:
        # Router was explicit; agent may only override on high confidence.
        if (
            agent_occ_raw
            and agent_occ_raw != router_occ
            and agent_conf >= 0.8
        ):
            occasion, chosen_from = agent_occ_raw, "agent_high_confidence"
        else:
            occasion, chosen_from = router_occ, "router"
    elif token_occ:
        occasion, chosen_from = token_occ, "tokens"
    elif agent_occ_raw:
        occasion, chosen_from = agent_occ_raw, "agent"
    else:
        occasion, chosen_from = "daily", "default"

    contract = dict(_OCCASION_CONTRACTS.get(occasion, _DEFAULT_CONTRACT))

    # Layer agent enrichment that does NOT override occasion-level fields.
    avoid = list(agent_payload.get("avoid_items") or [])
    if avoid:
        contract["forbidden_item_signals"] = list(
            dict.fromkeys(
                [s.lower() for s in contract.get("forbidden_item_signals", [])]
                + [str(s).lower() for s in avoid]
            )
        )
    if agent_payload.get("required_slots"):
        contract["required_slots"] = list(agent_payload["required_slots"]) or contract[
            "required_slots"
        ]

    brief = {
        "occasion": occasion,
        "sub_intent": str(agent_payload.get("sub_intent") or contract.get("sub_intent") or ""),
        "formality": str(agent_payload.get("formality") or contract.get("formality") or "mid"),
        "movement_requirement": contract.get("movement_requirement", "medium"),
        "polish_requirement": contract.get("polish_requirement", "mid"),
        "required_slots": list(contract.get("required_slots", [])),
        "allowed_roles": list(contract.get("required_slots", [])),
        "forbidden_roles": [],
        "preferred_item_signals": list(contract.get("preferred_item_signals", [])),
        "forbidden_item_signals": list(contract.get("forbidden_item_signals", [])),
        "board_mood": list(contract.get("board_mood", [])),
        "allowed_badges": list(contract.get("allowed_badges", [])),
        "allowed_titles": list(contract.get("allowed_titles", [])),
        "weather": str(weather or "").lower(),
        "_provenance": {
            "chosen_from": chosen_from,
            "router_occasion": router_occ or "",
            "token_occasion": token_occ or "",
            "agent_occasion": agent_occ_raw,
            "agent_confidence": agent_conf,
            "matched_tokens": matched_tokens,
        },
    }

    logger.info(
        "style_brief.created occasion=%s sub_intent=%s chosen_from=%s tokens=%s "
        "agent_confidence=%.2f",
        brief["occasion"],
        brief["sub_intent"],
        chosen_from,
        matched_tokens,
        agent_conf,
    )
    return brief


def safe_badge_for(brief: Dict[str, Any]) -> str:
    badges = (brief or {}).get("allowed_badges") or []
    return badges[0] if badges else "DAILY"


def safe_title_for(brief: Dict[str, Any], index: int = 0) -> str:
    titles = (brief or {}).get("allowed_titles") or ["Considered Look"]
    return titles[index % len(titles)]


# ---------------------------------------------------------------------------
# Board validation
# ---------------------------------------------------------------------------

def _item_blob(item: Dict[str, Any]) -> str:
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
            "material",
            "style",
        )
    )


def _board_blob(board: Dict[str, Any]) -> str:
    parts = [str(board.get("title") or ""), str(board.get("badge") or "")]
    for item in board.get("items") or []:
        parts.append(_item_blob(item if isinstance(item, dict) else {}))
    return " ".join(parts).lower()


def validate_board(
    board: Dict[str, Any], brief: Dict[str, Any]
) -> Tuple[bool, List[str], Dict[str, float]]:
    """Return (passes, reasons, scores)."""
    reasons: List[str] = []
    scores = {
        "occasion_fit_score": 0.5,
        "movement_fit_score": 0.5,
        "polish_fit_score": 0.5,
        "role_fit": 0.5,
        "forbidden_signal_hits": 0.0,
    }
    if not isinstance(board, dict) or not isinstance(brief, dict):
        return False, ["invalid_input"], scores

    blob = _board_blob(board)
    forbidden = [str(s).lower() for s in brief.get("forbidden_item_signals") or []]
    preferred = [str(s).lower() for s in brief.get("preferred_item_signals") or []]

    hits = [f for f in forbidden if f and f in blob]
    if hits:
        scores["forbidden_signal_hits"] = float(len(hits))
        reasons.append(f"forbidden_signal:{hits[:3]}")
        return False, reasons, scores

    matches = sum(1 for p in preferred if p and p in blob)
    if preferred:
        scores["occasion_fit_score"] = min(1.0, 0.5 + 0.1 * matches)
        scores["polish_fit_score"] = scores["occasion_fit_score"]

    # Required slots check.
    slots_present = {
        str(it.get("role") or it.get("slot") or "").lower()
        for it in (board.get("items") or [])
        if isinstance(it, dict)
    }
    required = {str(s).lower() for s in brief.get("required_slots") or []}
    canonical = {"top", "bottom", "footwear"}
    legacy_required = required & canonical
    if legacy_required and not legacy_required.issubset(slots_present):
        scores["role_fit"] = 0.3
        # Don't reject on legacy slot mismatch alone — the pipeline already
        # enforces core slots upstream. Soft penalty only.
    else:
        scores["role_fit"] = 0.8

    stylist_reason = (
        f"Brief={brief.get('occasion')}, matches={matches}, "
        f"forbidden_hits={len(hits)}"
    )
    reasons.append(stylist_reason)
    return True, reasons, scores


# ---------------------------------------------------------------------------
# Set-level board selection
# ---------------------------------------------------------------------------

def _hero(board: Dict[str, Any]) -> str:
    for item in board.get("items") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("slot") or "").lower()
        if role == "top":
            return str(item.get("id") or item.get("$id") or item.get("name") or "").lower()
    return ""


def _bottom_key(board: Dict[str, Any]) -> str:
    for item in board.get("items") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("slot") or "").lower()
        if role == "bottom":
            return str(item.get("id") or item.get("$id") or item.get("name") or "").lower()
    return ""


def _palette(board: Dict[str, Any]) -> str:
    colors = []
    for item in board.get("items") or []:
        if not isinstance(item, dict):
            continue
        c = str(item.get("color") or item.get("color_code") or "").lower()
        if c:
            colors.append(c)
    return "|".join(sorted(set(colors)))


def select_board_set(
    candidates: List[Dict[str, Any]],
    brief: Dict[str, Any],
    *,
    max_n: int = 3,
) -> List[Dict[str, Any]]:
    """Pick the best diverse subset of validated boards.

    Greedy selection optimizing for:
    - higher occasion_fit_score
    - hero diversity
    - top+bottom pair diversity
    - palette diversity
    """
    if not isinstance(candidates, list) or not candidates:
        return []

    validated: List[Tuple[Dict[str, Any], float]] = []
    for card in candidates:
        if not isinstance(card, dict):
            continue
        passes, reasons, scores = validate_board(card, brief)
        card.setdefault("_brief_scores", {}).update(scores)
        card.setdefault("_brief_reasons", []).extend(reasons)
        if passes:
            existing_score = float(card.get("score") or card.get("ml_score") or 0.0)
            quality = (
                scores["occasion_fit_score"] * 5.0
                + scores["role_fit"] * 2.0
                + existing_score / 10.0
            )
            validated.append((card, quality))

    if not validated:
        return []

    validated.sort(key=lambda x: x[1], reverse=True)

    chosen: List[Dict[str, Any]] = []
    seen_heros: set = set()
    seen_pairs: set = set()
    seen_palettes: set = set()

    def maybe_add(card: Dict[str, Any], strict: bool) -> bool:
        if len(chosen) >= max_n:
            return False
        hero = _hero(card)
        bottom = _bottom_key(card)
        pair = f"{hero}|{bottom}"
        palette = _palette(card)
        if strict:
            if hero and hero in seen_heros:
                return False
            if pair and pair in seen_pairs:
                return False
        chosen.append(card)
        if hero:
            seen_heros.add(hero)
        if pair:
            seen_pairs.add(pair)
        if palette:
            seen_palettes.add(palette)
        return True

    # First pass: strict diversity.
    for card, _q in validated:
        maybe_add(card, strict=True)
    # Second pass: fill remaining when wardrobe is sparse.
    for card, _q in validated:
        if len(chosen) >= max_n:
            break
        if card in chosen:
            continue
        maybe_add(card, strict=False)

    # Annotate with roles for the first 3.
    role_labels = ["primary", "alternate", "expressive"]
    for idx, card in enumerate(chosen):
        card["set_role"] = role_labels[idx] if idx < len(role_labels) else "extra"

    logger.info(
        "style_set.selected occasion=%s chosen=%d from=%d",
        brief.get("occasion"),
        len(chosen),
        len(candidates),
    )
    return chosen


__all__ = [
    "tokenize",
    "detect_occasion_from_tokens",
    "build_brief",
    "validate_board",
    "select_board_set",
    "safe_badge_for",
    "safe_title_for",
]
