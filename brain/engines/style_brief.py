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
    "basketball game": "basketball_game",
    "beach dinner": "beach_dinner",
    "business dinner": "client_dinner",
    "casual dinner": "casual_dinner",
    "christian funeral": "funeral",
    "client dinner": "client_dinner",
    "client meeting": "client_meeting",
    "client-meeting": "client_meeting",
    "client pitch": "client_presentation",
    "client presentation": "client_presentation",
    "coffee date": "coffee_date",
    "date night": "date_night",
    "date-night": "date_night",
    "first date": "first_date",
    "office meeting": "office_meeting",
    "team dinner": "team_dinner",
    "wedding guest": "wedding_guest",
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
    "basketball_game": {"basketball_game", "basketball"},
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
    "client_presentation": {
        "client_presentation",
        "presentation",
        "pitch",
        "investor",
        "boardroom",
        "conference",
    },
    "client_dinner": {"client_dinner"},
    "office_meeting": {"office_meeting"},
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
        "networking",
    },
    "date_night": {
        "date_night",
        "date",
        "dinner",
        "tonight",
        "rooftop_bar",
        "after_hours",
    },
    "coffee_date": {"coffee_date"},
    "first_date": {"first_date"},
    "casual_dinner": {"casual_dinner"},
    "beach_dinner": {"beach_dinner"},
    "team_dinner": {"team_dinner"},
    "beach": {"beach", "pool", "seaside", "coastal", "resort", "shore"},
    "brunch": {"brunch"},
    "rave": {"rave", "club", "edm", "festival"},
    "cocktail": {"cocktail"},
    "wedding_guest": {"wedding_guest"},
    "wedding": {"wedding", "reception", "ceremony", "sangeet"},
    "funeral": {"funeral", "memorial", "condolence", "wake"},
    "party": {"party", "house_party", "night_out", "happy_hour", "celebration"},
    "travel": {"travel", "airport", "flight", "vacation", "trip", "transit"},
    "temple_modest": {"temple", "mandir", "pooja", "puja", "shrine", "darshan", "religious"},
    "swimming": {"swim", "swimming", "swimwear", "swimsuit", "pool_day"},
    "capsule": {"capsule", "essentials", "core_wardrobe", "minimal_wardrobe", "wardrobe_essentials"},
    "daily": {"daily", "today"},
    "casual": {"casual", "weekend", "errand", "coffee_run", "coffee"},
}

# When BOTH client and office tokens fire, prefer client_meeting.
_OCCASION_PRIORITY: List[str] = [
    "capsule",          # capsule beats everything — different engine.
    "swimming",         # swimming beats beach because swim items are stricter.
    "basketball_game",
    "workout",          # workout MUST beat office to fix the "work in workout" bug.
    "client_presentation",
    "client_dinner",
    "client_meeting",
    "office_meeting",
    "coffee_date",
    "first_date",
    "beach_dinner",
    "casual_dinner",
    "team_dinner",
    "funeral",
    "wedding_guest",
    "wedding",
    "beach",
    "rave",
    "cocktail",
    "party",            # party should beat date_night when both fire
                        # (e.g. "party look tonight" — `tonight` alone
                        # is ambiguous, `party` is the strong signal).
    "date_night",
    "travel",
    "temple_modest",
    "brunch",
    "office",
    "casual",
    "daily",
]


# ---------------------------------------------------------------------------
# Compound occasion pre-resolver
# ---------------------------------------------------------------------------
# Compound prompts ("X after Y", "X then Y") used to collapse to the single
# highest-PRIORITY family — which could pick the wrong, less-formal half
# (e.g. "wedding reception after work" -> office). This deterministic
# pre-resolver detects a connector + 2+ occasion families and instead picks
# the highest social/formality-risk event. Single-occasion prompts (no
# connector) are never touched, so existing behavior is preserved exactly.

_CONNECTOR_RE = re.compile(
    r"\b(after|afterwards|afterward|before|then|later|plus|post|followed\s+by|and\s+then)\b"
)

# Higher = wins a compound. Funeral/wedding/professional outrank
# beach/pool/gym/travel. gym/swimming/beach only win when sole occasion
# (no connector -> never reach here).
_OCCASION_FORMALITY: Dict[str, int] = {
    "funeral": 100,
    "wedding": 92,
    "wedding_guest": 92,
    "temple_modest": 88,
    "client_presentation": 82,
    "client_meeting": 80,
    "client_dinner": 80,
    "office_meeting": 72,
    "office": 70,
    "cocktail": 66,
    "date_night": 60,
    "beach_dinner": 60,
    "casual_dinner": 58,
    "team_dinner": 58,
    "first_date": 58,
    "party": 50,
    "rave": 48,
    "brunch": 46,
    "coffee_date": 42,
    "casual": 40,
    "daily": 38,
    "travel": 30,
    "beach": 25,
    "swimming": 20,
    "basketball_game": 12,
    "workout": 10,
    "capsule": -1,  # special engine — never compound-override
}


def _has_connector(query: Any) -> bool:
    return bool(_CONNECTOR_RE.search(str(query or "").lower()))


def _detect_families(tokens: set) -> List[str]:
    """All occasion families whose tokens fire, in priority order, deduped."""
    families: List[str] = []
    for occ in _OCCASION_PRIORITY:
        if tokens & _OCCASION_TOKENS.get(occ, set()) and occ not in families:
            families.append(occ)
    return families


def detect_compound_context(query: Any) -> Optional[Dict[str, Any]]:
    """Return compound metadata when the prompt spans 2+ events via a
    connector, else None. Higher-formality event is primary."""
    tokens = set(tokenize(query))
    if not tokens or not _has_connector(query):
        return None
    families = [f for f in _detect_families(tokens) if f != "capsule"]
    if len(families) < 2:
        return None
    ranked = sorted(
        families, key=lambda f: _OCCASION_FORMALITY.get(f, 0), reverse=True
    )
    return {
        "is_compound": True,
        "primary_occasion": ranked[0],
        "secondary_occasion": ranked[1],
        "transition_required": True,
        "compound_reason": "Selected the higher-formality destination event.",
        "compound_note": (
            "Since this needs to work across both events, AHVI is "
            "prioritizing the more polished setting."
        ),
    }


def detect_occasion_from_tokens(query: Any) -> Tuple[str, List[str]]:
    """Return (occasion, matched_tokens). Empty occasion means unknown.

    Compound prompts (connector + 2+ families) resolve to the highest
    formality-risk family. All other prompts keep the original
    first-match-by-priority behavior unchanged.
    """
    tokens = set(tokenize(query))
    if not tokens:
        return "", []

    first = next(
        (occ for occ in _OCCASION_PRIORITY if tokens & _OCCASION_TOKENS.get(occ, set())),
        None,
    )
    if first is None:
        return "", []

    if first != "capsule" and _has_connector(query):
        families = [f for f in _detect_families(tokens) if f != "capsule"]
        if len(families) >= 2:
            winner = max(families, key=lambda f: _OCCASION_FORMALITY.get(f, 0))
            return winner, sorted(tokens & _OCCASION_TOKENS.get(winner, set()))

    return first, sorted(tokens & _OCCASION_TOKENS.get(first, set()))


_ARCHETYPE_ALIASES: Dict[str, str] = {
    "date": "date_night",
    "dinner_date": "date_night",
    "coffee": "coffee_date",
    "coffee_run": "casual",
    "office": "office_meeting",
    "business": "office_meeting",
    "client_meeting": "client_presentation",
    "beach": "beach",
    "wedding": "wedding_guest",
    "sensitive": "funeral",
}


def resolve_occasion_archetype(value: Any = "", query: Any = "") -> str:
    """Resolve the most specific wardrobe-board occasion archetype."""
    combined = f"{value or ''} {query or ''}".strip()
    token_occ, _hits = detect_occasion_from_tokens(combined)
    if token_occ:
        return token_occ
    raw = str(value or "").strip().lower().replace("-", "_")
    readable = raw.replace("_", " ")
    for alias, target in _ARCHETYPE_ALIASES.items():
        if alias == raw or alias == readable or alias in readable:
            return target
    return raw


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
            "short sleeve", "short-sleeve", "sequined", "tropical",
            "saree", "lehenga",
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
            "tee", "t-shirt", "navy blue sneaker", "blue sneaker",
            "running sneaker", "sports sneaker", "athletic sneaker",
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
    "swimming": {
        "sub_intent": "swimming",
        "formality": "low",
        "movement_requirement": "high",
        "polish_requirement": "low",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": [
            "swimwear", "swim trunks", "swimsuit", "rash guard", "slides",
            "sandals", "towel", "swim cap",
        ],
        "forbidden_item_signals": [
            "blazer", "suit", "trouser", "trousers", "loafer", "loafers",
            "boots", "office", "boardroom", "tie", "shirt",
        ],
        "board_mood": ["coastal", "easy", "swim-ready"],
        "allowed_badges": ["SWIM", "POOL"],
        "allowed_titles": ["Pool Ready", "Swim Casual", "Coastal Swim"],
    },
    "capsule": {
        "sub_intent": "capsule_wardrobe",
        "formality": "mid",
        "movement_requirement": "medium",
        "polish_requirement": "mid",
        "required_slots": ["top", "bottom", "footwear"],
        "preferred_item_signals": [
            "white", "black", "navy", "neutral", "shirt", "trouser",
            "trousers", "chino", "chinos", "denim", "tee", "sneaker",
            "sneakers", "loafer", "loafers", "blazer",
        ],
        "forbidden_item_signals": [
            "sequined", "neon", "embroidered", "printed", "novelty",
            "shiny", "metallic", "statement",
        ],
        "board_mood": ["timeless", "versatile", "neutral", "repeatable"],
        "allowed_badges": ["CAPSULE", "ESSENTIALS"],
        "allowed_titles": ["Capsule Foundation", "Versatile Core", "Essentials Edit"],
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

_OCCASION_CONTRACTS.update(
    {
        "coffee_date": {
            "sub_intent": "coffee_date",
            "formality": "low",
            "movement_requirement": "medium",
            "polish_requirement": "mid",
            "required_slots": ["top", "bottom", "footwear"],
            "preferred_item_signals": [
                "oxford", "linen", "overshirt", "polo", "button down",
                "button-down", "relaxed", "chino", "chinos", "denim",
                "jeans", "clean sneaker", "sneakers", "suede loafer",
                "watch",
            ],
            "forbidden_item_signals": [
                "formal trouser", "formal trousers", "black formal",
                "wedding", "wedding shirt", "tuxedo", "shiny", "satin",
                "sequin", "sequined", "festive", "embroidered",
                "embroidery", "corporate", "boardroom", "office", "client",
                "gold ring", "statement gold",
            ],
            "board_mood": ["relaxed", "conversational", "approachable", "intentional"],
            "allowed_badges": ["COFFEE DATE", "RELAXED SOCIAL"],
            "allowed_titles": ["Relaxed Oxford", "Soft Coffee Polish", "Easy Conversation"],
        },
        "first_date": {
            "sub_intent": "first_date",
            "formality": "mid",
            "movement_requirement": "medium",
            "polish_requirement": "mid",
            "required_slots": ["top", "bottom", "footwear"],
            "preferred_item_signals": ["shirt", "polo", "knit", "chino", "denim", "sneaker", "loafer", "watch"],
            "forbidden_item_signals": ["gym", "track", "slipper", "slides", "tuxedo", "corporate", "boardroom"],
            "board_mood": ["warm", "approachable", "intentional"],
            "allowed_badges": ["FIRST DATE", "SOCIAL"],
            "allowed_titles": ["Easy First Impression", "Soft Confidence", "Relaxed Polish"],
        },
        "casual_dinner": {
            "sub_intent": "casual_dinner",
            "formality": "mid",
            "movement_requirement": "low",
            "polish_requirement": "mid",
            "required_slots": ["top", "bottom", "footwear"],
            "preferred_item_signals": ["shirt", "knit", "polo", "chino", "denim", "loafer", "clean sneaker", "watch"],
            "forbidden_item_signals": ["gym", "track", "slides", "slipper", "tuxedo", "corporate"],
            "board_mood": ["easy", "evening-aware", "social"],
            "allowed_badges": ["DINNER", "SMART CASUAL"],
            "allowed_titles": ["Dinner Ease", "Soft Evening Casual", "Clean Social"],
        },
        "client_dinner": {
            "sub_intent": "client_dinner",
            "formality": "mid_high",
            "movement_requirement": "low",
            "polish_requirement": "high",
            "required_slots": ["top", "bottom", "footwear"],
            "preferred_item_signals": ["button down", "shirt", "trouser", "chino", "loafer", "derby", "watch", "belt", "blazer"],
            "forbidden_item_signals": ["shorts", "slides", "slipper", "gym", "track", "loud print", "sequin", "neon", "beach"],
            "board_mood": ["professional-first", "social-second", "composed"],
            "allowed_badges": ["CLIENT DINNER", "POLISHED SOCIAL"],
            "allowed_titles": ["Client Dinner Polish", "Professional Social", "Composed Evening"],
        },
        "beach_dinner": {
            "sub_intent": "beach_dinner",
            "formality": "mid",
            "movement_requirement": "medium",
            "polish_requirement": "mid",
            "required_slots": ["top", "bottom", "footwear"],
            "preferred_item_signals": ["linen", "cotton", "camp collar", "resort", "chino", "sandal", "espadrille", "slides", "lightweight"],
            "forbidden_item_signals": ["office trouser", "black trousers", "black pants", "oxford", "derby", "formal leather", "heavy blazer", "corporate"],
            "board_mood": ["breathable", "sunset", "relaxed polish"],
            "allowed_badges": ["BEACH DINNER", "COASTAL EVENING"],
            "allowed_titles": ["Coastal Dinner", "Sunset Polish", "Breathable Evening"],
        },
        "wedding_guest": {
            **_OCCASION_CONTRACTS["wedding"],
            "sub_intent": "wedding_guest",
            "allowed_badges": ["WEDDING GUEST", "CEREMONY", "FORMAL"],
            "allowed_titles": ["Guest Polish", "Ceremony Refined", "Reception Ready"],
        },
        "funeral": {
            "sub_intent": "funeral",
            "formality": "high",
            "movement_requirement": "low",
            "polish_requirement": "high",
            "required_slots": ["top", "bottom", "footwear"],
            "preferred_item_signals": ["dark", "black", "navy", "charcoal", "plain", "shirt", "trouser", "closed shoe", "loafer"],
            "forbidden_item_signals": ["bright", "neon", "shiny", "satin", "sequin", "loud print", "floral print", "party", "shorts", "slides"],
            "board_mood": ["respectful", "quiet", "understated"],
            "allowed_badges": ["RESPECTFUL", "UNDERSTATED"],
            "allowed_titles": ["Quiet Formal", "Respectful Minimal", "Understated Polish"],
        },
        "office_meeting": {
            **_OCCASION_CONTRACTS["office"],
            "sub_intent": "office_meeting",
            "allowed_badges": ["OFFICE MEETING", "OFFICE"],
            "allowed_titles": ["Meeting Ready", "Office Polish", "Clean Professional"],
        },
        "client_presentation": {
            **_OCCASION_CONTRACTS["client_meeting"],
            "sub_intent": "client_presentation",
            "allowed_badges": ["CLIENT READY", "PRESENTATION"],
            "allowed_titles": ["Presentation Polish", "Client Authority", "Composed Pitch"],
        },
        "basketball_game": {
            "sub_intent": "basketball_game",
            "formality": "low",
            "movement_requirement": "high",
            "polish_requirement": "low",
            "required_slots": ["top", "bottom", "footwear"],
            "preferred_item_signals": ["tee", "polo", "jersey", "denim", "chino", "sneaker", "jacket", "cap"],
            "forbidden_item_signals": ["formal shirt", "button down", "button-down", "embroidered", "loafer", "blazer", "oxford", "tuxedo"],
            "board_mood": ["active", "casual", "team-ready"],
            "allowed_badges": ["GAME", "CASUAL"],
            "allowed_titles": ["Game Casual", "Courtside Easy", "Team Dinner Transition"],
        },
        "team_dinner": {
            "sub_intent": "team_dinner",
            "formality": "mid",
            "movement_requirement": "medium",
            "polish_requirement": "mid",
            "required_slots": ["top", "bottom", "footwear"],
            "preferred_item_signals": ["shirt", "polo", "chino", "denim", "clean sneaker", "loafer", "jacket"],
            "forbidden_item_signals": ["tuxedo", "wedding", "shiny", "sequin", "slides", "slipper"],
            "board_mood": ["social", "easy", "team-friendly"],
            "allowed_badges": ["TEAM DINNER", "SOCIAL"],
            "allowed_titles": ["Team Dinner Ease", "Clean Social", "Relaxed Table Ready"],
        },
    }
)


# Accessory-occasion rules. Cleaner separation from item-level signals
# because accessories can be removed in isolation (a board can still be
# valid if a forbidden accessory is scrubbed but the core top/bottom/
# footwear pieces remain occasion-correct).
_ACCESSORY_OCCASION_RULES: Dict[str, Dict[str, List[str]]] = {
    "office": {
        "allowed": ["watch", "belt", "ring", "bag", "leather bag", "briefcase", "tie"],
        "forbidden": [
            "swim cap", "swimming cap", "swim goggles", "swimming goggles",
            "goggles", "water bottle", "gym bottle", "shaker", "cap", "hat",
            "beanie", "headband", "sweatband", "wristband", "fanny pack",
            "beach", "swim", "snorkel", "flip flop", "athletic", "sports",
            "headphones", "earbuds case",
        ],
    },
    "client_meeting": {
        "allowed": ["watch", "belt", "ring", "bag", "briefcase", "tie"],
        "forbidden": [
            "swim cap", "swimming cap", "swim goggles", "swimming goggles",
            "goggles", "water bottle", "gym bottle", "shaker", "cap", "hat",
            "beanie", "headband", "sweatband", "wristband", "fanny pack",
            "beach", "swim", "snorkel", "athletic", "sports", "tropical",
            "neon", "sequined", "headphones",
        ],
    },
    "wedding": {
        "allowed": ["watch", "belt", "ring", "tie", "pocket square", "bag", "clutch"],
        "forbidden": [
            "swim cap", "swim goggles", "goggles", "water bottle", "gym bottle",
            "shaker", "cap", "beanie", "headband", "fanny pack", "athletic",
            "sports", "beach", "swim", "snorkel",
        ],
    },
    "date_night": {
        "allowed": ["watch", "belt", "ring", "bag", "clutch", "chain", "scarf"],
        "forbidden": [
            "swim cap", "swimming cap", "swim goggles", "swimming goggles",
            "goggles", "water bottle", "gym bottle", "shaker", "cap", "hat",
            "beanie", "headband", "sweatband", "wristband", "fanny pack",
            "snorkel", "athletic", "sports bottle",
        ],
    },
    "workout": {
        "allowed": [
            "water bottle", "gym bottle", "shaker", "cap", "headband",
            "sweatband", "wristband", "towel", "earbuds case", "headphones",
            "sports watch", "fitness tracker",
        ],
        "forbidden": [
            "tie", "pocket square", "clutch", "briefcase", "dress watch",
            "leather bag",
        ],
    },
    "beach": {
        "allowed": [
            "sunglasses", "straw hat", "swim cap", "swim goggles", "goggles",
            "snorkel", "tote", "beach bag", "sandals",
        ],
        "forbidden": ["tie", "pocket square", "briefcase"],
    },
    "travel": {
        "allowed": [
            "backpack", "duffel", "watch", "sunglasses", "tote", "neck pillow",
            "earbuds case", "passport holder",
        ],
        "forbidden": ["swim cap", "swim goggles"],
    },
    "party": {
        "allowed": ["chain", "ring", "watch", "bag", "clutch", "sunglasses"],
        "forbidden": ["swim cap", "swim goggles", "water bottle", "gym bottle"],
    },
    "casual": {
        "allowed": ["watch", "cap", "bag", "sunglasses", "chain", "ring"],
        "forbidden": ["swim cap", "swim goggles"],
    },
    "daily": {
        "allowed": ["watch", "cap", "bag", "sunglasses", "chain", "ring"],
        "forbidden": ["swim cap", "swim goggles", "snorkel"],
    },
    "swimming": {
        "allowed": [
            "swim cap", "swim goggles", "goggles", "snorkel", "towel",
            "sunglasses", "tote", "beach bag", "slides", "sandals",
            "water bottle",
        ],
        "forbidden": [
            "tie", "pocket square", "briefcase", "watch", "belt", "loafer",
        ],
    },
    "capsule": {
        "allowed": ["watch", "belt", "ring", "bag", "sunglasses"],
        "forbidden": [
            "swim cap", "swim goggles", "snorkel", "fanny pack", "neon",
            "statement", "sequined", "novelty",
        ],
    },
}

_ACCESSORY_OCCASION_RULES.update(
    {
        "coffee_date": _ACCESSORY_OCCASION_RULES["date_night"],
        "first_date": _ACCESSORY_OCCASION_RULES["date_night"],
        "casual_dinner": _ACCESSORY_OCCASION_RULES["date_night"],
        "client_dinner": _ACCESSORY_OCCASION_RULES["client_meeting"],
        "beach_dinner": _ACCESSORY_OCCASION_RULES["beach"],
        "wedding_guest": _ACCESSORY_OCCASION_RULES["wedding"],
        "funeral": _ACCESSORY_OCCASION_RULES["wedding"],
        "office_meeting": _ACCESSORY_OCCASION_RULES["office"],
        "client_presentation": _ACCESSORY_OCCASION_RULES["client_meeting"],
        "basketball_game": _ACCESSORY_OCCASION_RULES["casual"],
        "team_dinner": _ACCESSORY_OCCASION_RULES["date_night"],
    }
)


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

    router_occ_raw = str(router_occasion or "").strip().lower() or None
    router_occ = resolve_occasion_archetype(router_occ_raw or "", "") if router_occ_raw else None
    token_occ, matched_tokens = detect_occasion_from_tokens(query)
    if token_occ:
        token_occ = resolve_occasion_archetype(token_occ, query)
    agent_occ = resolve_occasion_archetype(agent_occ_raw, "") if agent_occ_raw else ""

    occasion = ""
    chosen_from = ""
    if router_occ:
        # "Client meeting" is stricter than generic office. Some upstream
        # routers collapse it to office, so keep the user's explicit client
        # wording authoritative when token detection finds it.
        if router_occ in {"office", "office_meeting"} and token_occ in {"client_meeting", "client_presentation"}:
            occasion, chosen_from = token_occ, "tokens_over_office_router"
        # Router was explicit; agent may only override on high confidence.
        elif (
            agent_occ
            and agent_occ != router_occ
            and agent_conf >= 0.8
        ):
            occasion, chosen_from = agent_occ, "agent_high_confidence"
        else:
            occasion, chosen_from = router_occ, "router"
    elif token_occ:
        occasion, chosen_from = token_occ, "tokens"
    elif agent_occ:
        occasion, chosen_from = agent_occ, "agent"
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

    # Compound occasion metadata (deterministic). Frontend/copy may read this;
    # no frontend change required. occasion above already resolved to the
    # higher-formality event via detect_occasion_from_tokens.
    compound = detect_compound_context(query)
    if compound:
        brief["compound"] = compound
        brief["is_compound"] = True
        logger.info(
            "style_brief.compound primary=%s secondary=%s occasion=%s",
            compound["primary_occasion"],
            compound["secondary_occasion"],
            brief["occasion"],
        )

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


def _is_accessory_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    role = str(item.get("role") or item.get("slot") or "").lower()
    if role == "accessory":
        return True
    blob = " ".join(
        str(item.get(k) or "").lower()
        for k in ("category", "sub_category", "subcategory", "type", "garment_type")
    )
    return any(
        kw in blob
        for kw in (
            "accessor", "watch", "belt", "bag", "ring", "necklace", "bracelet",
            "earring", "cap", "hat", "scarf", "sunglass", "goggle", "bottle",
        )
    )


def _accessory_token_blob(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return " ".join(
        str(item.get(k) or "").lower()
        for k in (
            "name", "title", "label", "category", "sub_category", "subcategory",
            "type", "garment_type",
        )
    )


def is_accessory_forbidden(
    item: Dict[str, Any], brief: Dict[str, Any]
) -> Tuple[bool, str]:
    """Return (forbidden, reason). Reason names which rule matched.

    An accessory is forbidden when it textually matches the occasion's
    forbidden_accessories list and does NOT match the occasion's allowed list.
    """
    if not isinstance(item, dict) or not isinstance(brief, dict):
        return False, ""
    occ = str(brief.get("occasion") or "").lower()
    rules = _ACCESSORY_OCCASION_RULES.get(occ)
    if not rules:
        return False, ""
    blob = _accessory_token_blob(item)
    if not blob:
        return False, ""

    forbidden = [t.lower() for t in rules.get("forbidden") or []]
    allowed = [t.lower() for t in rules.get("allowed") or []]

    # Allow wins when explicit (e.g. workout allows "water bottle" even
    # though it lives on the office forbidden list).
    if any(a and a in blob for a in allowed):
        return False, ""

    for tok in forbidden:
        if tok and tok in blob:
            return True, f"occasion_incompatible:{tok}"
    return False, ""


def strip_forbidden_accessories(
    board: Dict[str, Any], brief: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Remove occasion-incompatible accessories from a board in-place.

    Returns (board, removed_items). Mutates `board['items']` and
    `board['accessories']`. Never touches non-accessory roles (top, bottom,
    footwear, dress, outerwear) — those go through validate_board.
    """
    removed: List[Dict[str, Any]] = []
    if not isinstance(board, dict) or not isinstance(brief, dict):
        return board, removed

    items = board.get("items") or []
    if not isinstance(items, list):
        return board, removed

    new_items: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            new_items.append(item)
            continue
        if _is_accessory_item(item):
            forbidden, reason = is_accessory_forbidden(item, brief)
            if forbidden:
                removed.append(dict(item))
                try:
                    logger.info(
                        "outfit_accessory_removed occasion=%s item=%s reason=%s",
                        brief.get("occasion"),
                        str(item.get("name") or item.get("title") or item.get("label") or "?"),
                        reason,
                    )
                except Exception:
                    pass
                continue
        new_items.append(item)
    board["items"] = new_items

    if isinstance(board.get("accessories"), list):
        kept_accs: List[Dict[str, Any]] = []
        for acc in board.get("accessories") or []:
            if not isinstance(acc, dict):
                kept_accs.append(acc)
                continue
            forbidden, _ = is_accessory_forbidden(acc, brief)
            if not forbidden:
                kept_accs.append(acc)
        board["accessories"] = kept_accs

    if removed:
        board.setdefault("removed_accessories", []).extend(removed)
    return board, removed


def core_outfit_complete(board: Dict[str, Any]) -> bool:
    """Did the board still have top + bottom + footwear (or dress + footwear)
    after accessory scrub?"""
    if not isinstance(board, dict):
        return False
    roles = {
        str(it.get("role") or it.get("slot") or "").lower()
        for it in (board.get("items") or [])
        if isinstance(it, dict)
    }
    if "dress" in roles and "footwear" in roles:
        return True
    return {"top", "bottom", "footwear"}.issubset(roles)


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


def _accessory_keys(board: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for item in board.get("items") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("slot") or "").lower()
        if role != "accessory":
            continue
        keys.append(
            str(item.get("id") or item.get("$id") or item.get("name") or "").lower()
        )
    return [k for k in keys if k]


def _rotate_accessories_across_boards(
    boards: List[Dict[str, Any]],
) -> None:
    """In-place dedup so the same accessory doesn't appear in every board.

    Per board (after the first), drop any accessory item whose id/name
    already appeared on a previously-chosen board. Top/bottom/footwear
    are never touched.
    """
    if not isinstance(boards, list) or len(boards) <= 1:
        return
    used: set = set()
    for idx, board in enumerate(boards):
        if not isinstance(board, dict):
            continue
        items = board.get("items") or []
        new_items: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                new_items.append(item)
                continue
            role = str(item.get("role") or item.get("slot") or "").lower()
            if role != "accessory":
                new_items.append(item)
                continue
            key = str(item.get("id") or item.get("$id") or item.get("name") or "").lower()
            if idx > 0 and key and key in used:
                # Drop the duplicate accessory; another board can keep it.
                continue
            if key:
                used.add(key)
            new_items.append(item)
        board["items"] = new_items
        if isinstance(board.get("accessories"), list):
            board["accessories"] = [
                acc for acc in board["accessories"]
                if not isinstance(acc, dict)
                or str(acc.get("id") or acc.get("$id") or acc.get("name") or "").lower() not in used
                or (
                    # Keep accessory if THIS board's primary accessory.
                    idx == 0
                )
            ]


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
    strict_hero_occasion = str(brief.get("occasion") or "").lower() in {
        "client_meeting",
        "wedding",
    }

    def maybe_add(card: Dict[str, Any], strict: bool) -> bool:
        if len(chosen) >= max_n:
            return False
        hero = _hero(card)
        bottom = _bottom_key(card)
        pair = f"{hero}|{bottom}"
        palette = _palette(card)
        if strict_hero_occasion and hero and hero in seen_heros:
            return False
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

    # Accessory rotation: drop accessories already worn by an earlier
    # board so the same gold ring / watch doesn't appear three times.
    _rotate_accessories_across_boards(chosen)

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
    "resolve_occasion_archetype",
    "build_brief",
    "validate_board",
    "select_board_set",
    "safe_badge_for",
    "safe_title_for",
]
