import base64
import hashlib
import logging
import math
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from services.category_taxonomy import infer_style_attributes
from services.wardrobe_suitability import outfit_contains_private_wear

try:
    from brain.engines.style_brief import resolve_occasion_archetype
except Exception:  # pragma: no cover
    resolve_occasion_archetype = None

try:
    from brain.engines.occasion_interpreter import interpret_occasion_context
    from brain.engines.occasion_style_rules import (
        detect_wardrobe_gap,
        get_occasion_rule,
        reject_board_for_occasion,
        score_item_for_occasion,
    )
except Exception:  # pragma: no cover - keeps legacy style flow bootable
    interpret_occasion_context = None
    detect_wardrobe_gap = None
    get_occasion_rule = None
    reject_board_for_occasion = None
    score_item_for_occasion = None

try:
    from brain.engines.outfit_quality_guard import (
        reject_board_for_occasion as reject_quality_board_for_occasion,
    )
except Exception:  # pragma: no cover - optional during partial deploys
    reject_quality_board_for_occasion = None

try:
    from services.agent_metadata_validator import item_metadata_v2_reject_reason
except Exception:  # pragma: no cover
    item_metadata_v2_reject_reason = None  # type: ignore

try:
    from services.agent_style_orchestrator import (
        is_enabled as _agent_style_enabled,
        merge_agent_payload_into_context as _agent_merge_into_context,
        orchestrate_style_request_sync as _agent_orchestrate_sync,
        start_style_orchestration as _agent_start_orchestration,
    )
except Exception:  # pragma: no cover - safe if agent service is missing
    _agent_style_enabled = lambda: False  # noqa: E731
    _agent_merge_into_context = lambda ctx, _payload: ctx  # noqa: E731
    _agent_orchestrate_sync = None
    _agent_start_orchestration = None

# Wardrobe items that are not apparel must never appear in a styled board, even
# when the agent (which normally filters them) is skipped for latency.
_NON_APPAREL_KEYWORDS = (
    "charger", "passport", "water bottle", "bottle", "swim cap", "goggles",
    "umbrella", "wallet", "earphone", "headphone", "power bank", "powerbank",
    "key", "phone", "laptop", "notebook", "book", "medicine", "toothbrush",
)


def _agent_overlap_wait_seconds() -> float:
    try:
        return max(0.0, min(float(os.getenv("AGENT_STYLE_OVERLAP_WAIT_SECONDS", "8")), 30.0))
    except Exception:
        return 8.0


def _strip_non_apparel(wardrobe: Any) -> List[Dict[str, Any]]:
    """Deterministically drop obvious non-apparel items so boards stay clean
    regardless of whether the agent's avoid_items arrives in time."""
    if not isinstance(wardrobe, list):
        return wardrobe
    kept = []
    for it in wardrobe:
        if not isinstance(it, dict):
            continue
        blob = " ".join(
            str(it.get(k) or "").lower()
            for k in ("name", "label", "category", "type", "sub_category")
        )
        if any(kw in blob for kw in _NON_APPAREL_KEYWORDS):
            continue
        kept.append(it)
    return kept

try:
    from services.r2_storage import R2Storage, R2StorageError
except Exception:  # pragma: no cover - optional deploy dependency
    R2Storage = None
    R2StorageError = Exception

logger = logging.getLogger("ahvi.style_flow")


STYLE_ACTION_CHIPS = ["More looks", "Next best options", "Try different shoes"]
TRUST_LAYER_RUNTIME_INFERENCE = str(os.getenv("AHVI_TRUST_LAYER_RUNTIME_INFERENCE", "")).lower() in {"1", "true", "yes"}

_OCCASION_FORBIDDEN_OFFICE = [
    "boardroom",
    "professional",
    "office",
    "executive",
    "corporate",
    "friday",
    "client",
]

_OCCASION_CARD_LANGUAGE = {
    "date": {
        "badge": "DATE NIGHT",
        "titles": [
            "Soft Statement",
            "Evening Ease",
            "Dinner Polish",
            "After-Dark Edit",
            "Quiet Confidence",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE,
    },
    "date_night": {
        "badge": "DATE NIGHT",
        "titles": [
            "Soft Statement",
            "Evening Ease",
            "Dinner Polish",
            "After-Dark Edit",
            "Quiet Confidence",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE,
    },
    "coffee_date": {
        "badge": "COFFEE DATE",
        "titles": [
            "Relaxed Oxford",
            "Soft Coffee Polish",
            "Easy Conversation",
            "Approachable Edit",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE + ["evening", "after-dark", "boardroom", "formal"],
    },
    "first_date": {
        "badge": "FIRST DATE",
        "titles": ["Easy First Impression", "Soft Confidence", "Relaxed Polish"],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE + ["boardroom"],
    },
    "casual_dinner": {
        "badge": "DINNER",
        "titles": ["Dinner Ease", "Soft Evening Casual", "Clean Social"],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE,
    },
    "client_dinner": {
        "badge": "CLIENT DINNER",
        "titles": ["Client Dinner Polish", "Professional Social", "Composed Evening"],
        "forbidden_title_words": ["party", "rave", "beach"],
    },
    "beach_dinner": {
        "badge": "BEACH DINNER",
        "titles": ["Coastal Dinner", "Sunset Polish", "Breathable Evening"],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE + ["formal"],
    },
    "wedding_guest": {
        "badge": "WEDDING GUEST",
        "titles": ["Guest Polish", "Ceremony Refined", "Reception Ready"],
        "forbidden_title_words": ["office", "boardroom", "gym"],
    },
    "funeral": {
        "badge": "RESPECTFUL",
        "titles": ["Quiet Formal", "Respectful Minimal", "Understated Polish"],
        "forbidden_title_words": ["party", "statement", "bright"],
    },
    "office_meeting": {
        "badge": "OFFICE MEETING",
        "titles": ["Meeting Ready", "Office Polish", "Clean Professional"],
        "forbidden_title_words": [],
    },
    "client_presentation": {
        "badge": "PRESENTATION",
        "titles": ["Presentation Polish", "Client Authority", "Composed Pitch"],
        "forbidden_title_words": [],
    },
    "basketball_game": {
        "badge": "GAME",
        "titles": ["Game Casual", "Courtside Easy", "Team Dinner Transition"],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE + ["formal"],
    },
    "team_dinner": {
        "badge": "TEAM DINNER",
        "titles": ["Team Dinner Ease", "Clean Social", "Relaxed Table Ready"],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE,
    },
    "beach": {
        "badge": "BEACH",
        "titles": [
            "Coastal Ease",
            "Resort Casual",
            "Light Vacation",
            "Beach Ready",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE + ["formal"],
    },
    "office": {
        "badge": "OFFICE",
        "titles": [
            "Boardroom Casual",
            "Creative Professional",
            "Clean Friday",
            "Executive Minimal",
        ],
        "forbidden_title_words": [],
    },
    "daily": {
        "badge": "DAILY",
        "titles": [
            "Polished Neutral",
            "Sharp Daily",
            "Smart Ease",
            "Clean Edit",
            "Refined Casual",
            "Easy Daily",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE,
    },
    "casual": {
        "badge": "CASUAL",
        "titles": [
            "Polished Neutral",
            "Sharp Daily",
            "Smart Ease",
            "Clean Edit",
            "Refined Casual",
            "Easy Daily",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE + ["formal"],
    },
    "temple_modest": {
        "badge": "TEMPLE",
        "titles": [
            "Modest Grace",
            "Soft Traditional",
            "Temple Ready",
            "Respectful Ease",
        ],
        "forbidden_title_words": _OCCASION_FORBIDDEN_OFFICE + ["club", "party", "strapless", "crop"],
    },
}


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_editorial_copy(value: Any, fallback: str) -> str:
    text = _safe_text(value)
    if not text:
        return fallback
    internal_markers = ("graph_", "occasion_fit:", "occasion_penalty:", "occasion_reject:")
    if not any(marker in text for marker in internal_markers):
        return text
    cleaned = re.sub(r"\bgraph_[a-z0-9_]+\b", "the pieces work together", text)
    cleaned = re.sub(r"\boccasion_fit:[a-z0-9_\- ]+\b", "the footwear and base read intentional", cleaned)
    cleaned = re.sub(r"\boccasion_penalty:[a-z0-9_\- ]+\b", "one detail needs refinement", cleaned)
    cleaned = re.sub(r"\boccasion_reject:[a-z0-9_\- ]+\b", "this needs a closer occasion fit", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    return cleaned.strip() or fallback


def _tokens(value: Any) -> set[str]:
    import re

    return set(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


_CANONICAL_OCCASIONS = {
    "coffee_date",
    "first_date",
    "date_night",
    "casual_dinner",
    "client_dinner",
    "beach_dinner",
    "wedding_guest",
    "funeral",
    "office_meeting",
    "client_presentation",
    "basketball_game",
    "team_dinner",
    "date_night",
    "swimming",
    "beach",
    "office",
    "client_meeting",
    "brunch",
    "party",
    "house_party",
    "rave",
    "cocktail",
    "travel",
    "workout",
    "wedding",
    "casual",
    "daily",
    "capsule",
    "capsule_wardrobe",
    "style_advice",
    "daily_wear",
    "temple_modest",
}


def _normalize_occasion_value(value: Any, query: Any = "") -> str:
    """Token-aware occasion normalization.

    The old substring-based check returned "office" for "workout outfit" via
    the "work" prefix match. style_brief.detect_occasion_from_tokens does
    whole-word matching with a priority list (workout > office) so this
    bug is gone. The explicit value still wins when it is already a
    canonical occasion key.
    """
    explicit = _safe_text(value).lower().replace("-", "_")
    combined_text = f"{_safe_text(value)} {_safe_text(query)}".strip().lower()
    # Beach / coastal / resort guard. A vacation brief must never resolve to
    # office or coffee-date styling (which surfaces Oxford shirts + loafers /
    # formal shoes). Coastal intent wins unless the user explicitly set a
    # formal occasion. "dinner" routes to beach_dinner so evening polish is
    # still allowed. Placed before generic detection so it is authoritative.
    if any(
        token in combined_text
        for token in ("beach", "coastal", "seaside", "poolside", "resort", "tropical")
    ) and explicit not in {
        "office",
        "office_meeting",
        "client_meeting",
        "client_presentation",
        "client_dinner",
        "funeral",
        "wedding",
        "wedding_guest",
    }:
        resolved = "beach_dinner" if "dinner" in combined_text else "beach"
        logger.info("AHVI_OCCASION_ARCHETYPE=%s source=%s", resolved, "coastal_guard")
        return resolved
    if resolve_occasion_archetype is not None:
        try:
            archetype = resolve_occasion_archetype(explicit, combined_text)
            if archetype in _CANONICAL_OCCASIONS:
                logger.info("AHVI_OCCASION_ARCHETYPE=%s source=%s", archetype, "style_flow")
                return archetype
        except Exception:
            pass
    if "capsule" in combined_text and any(
        token in combined_text
        for token in ("wardrobe", "essentials", "core", "minimalist", "foundation")
    ):
        return "capsule"
    if "rave" in combined_text:
        return "party"
    if any(token in combined_text for token in ("style tips", "style_tip", "give me style")):
        if "current outfit:" in combined_text or "weather:" in combined_text or "daily wear" in combined_text:
            return "style_advice"
    if explicit in _CANONICAL_OCCASIONS or explicit == "client_meeting":
        return explicit
    try:
        from brain.engines.style_brief import detect_occasion_from_tokens

        combined = f"{_safe_text(value)} {_safe_text(query)}".strip()
        occ, _ = detect_occasion_from_tokens(combined)
        if occ:
            return occ
    except Exception:
        pass
    return explicit if explicit in _CANONICAL_OCCASIONS else ""


def apply_occasion_card_language(cards: List[Dict[str, Any]], occasion: str) -> List[Dict[str, Any]]:
    normalized = _normalize_occasion_value(occasion)
    lang = _OCCASION_CARD_LANGUAGE.get(normalized)
    if not lang:
        for card in cards or []:
            if isinstance(card, dict) and normalized:
                card.setdefault("occasion", normalized)
        return cards

    titles = list(lang.get("titles") or [])
    badge = _safe_text(lang.get("badge"))
    forbidden = [_safe_text(w).lower() for w in (lang.get("forbidden_title_words") or [])]

    for idx, card in enumerate(cards or []):
        if not isinstance(card, dict):
            continue
        title = _safe_text(card.get("title")).lower()
        if titles and (not title or any(word and word in title for word in forbidden)):
            card["title"] = titles[idx % len(titles)]
            card["name"] = card["title"]
        if badge:
            card["badge"] = badge
            card["occasion_label"] = badge
        card.setdefault("occasion", normalized)
    return cards


def _filter_boards_for_occasion(cards: List[Dict[str, Any]], occasion: str) -> List[Dict[str, Any]]:
    if reject_quality_board_for_occasion is None:
        return cards
    normalized = _normalize_occasion_value(occasion)
    filtered: List[Dict[str, Any]] = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        try:
            rejected, reason = reject_quality_board_for_occasion(card, normalized)
        except Exception:
            rejected, reason = False, ""
        if rejected:
            logger.info(
                "ahvi.board_rejected occasion=%s reason=%s title=%s",
                normalized,
                reason,
                card.get("title"),
            )
            continue
        filtered.append(card)
    return filtered


def _ahvi_item_blob(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    parts = []
    for key in (
        "id",
        "name",
        "title",
        "category",
        "subcategory",
        "type",
        "color",
        "material",
        "pattern",
        "tags",
        "style_tags",
        "details",
    ):
        value = item.get(key)
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(v or "") for v in value)
        else:
            parts.append(str(value or ""))
    return " ".join(parts).lower()


def _ahvi_slot_counts(items: list[dict]) -> dict:
    counts = {
        "top": 0,
        "bottom": 0,
        "footwear": 0,
        "accessory": 0,
        "outerwear": 0,
        "total": 0,
    }
    for item in items or []:
        blob = _ahvi_item_blob(item)
        counts["total"] += 1
        if any(
            w in blob
            for w in [
                "shirt",
                "t-shirt",
                "tee",
                "top",
                "polo",
                "kurta",
                "blouse",
                "sweater",
                "hoodie",
            ]
        ):
            counts["top"] += 1
        elif any(w in blob for w in ["pant", "trouser", "jean", "shorts", "skirt", "bottom"]):
            counts["bottom"] += 1
        elif any(
            w in blob
            for w in [
                "shoe",
                "sneaker",
                "loafer",
                "boot",
                "sandal",
                "slide",
                "footwear",
                "slipper",
            ]
        ):
            counts["footwear"] += 1
        elif any(w in blob for w in ["blazer", "jacket", "coat", "overshirt"]):
            counts["outerwear"] += 1
        elif any(
            w in blob
            for w in [
                "watch",
                "belt",
                "bag",
                "jewelry",
                "jewellery",
                "ring",
                "bracelet",
                "cap",
                "eyewear",
                "sunglasses",
                "chain",
            ]
        ):
            counts["accessory"] += 1
    return counts


def _ahvi_has_core_slots(slot_counts: dict) -> bool:
    return (
        int(slot_counts.get("top", 0) or 0) > 0
        and int(slot_counts.get("bottom", 0) or 0) > 0
        and int(slot_counts.get("footwear", 0) or 0) > 0
    )


# --- Office/client-meeting board sanitization ------------------------------
# A single underwear/loungewear/casual item should not torpedo an otherwise
# usable office outfit. Strip the offending piece before quality rejection.

_OFFICE_OCCASIONS = {
    "office",
    "startup_office",
    "client_meeting",
    "client meeting",
    "presentation",
    "business",
    "interview",
    "conference",
}

_OFFICE_BAD_TOKENS = (
    "boxer",
    "underwear",
    "sleepwear",
    "loungewear",
    "lounge",
    "pyjama",
    "pajama",
    "swim shorts",
    "swim_shorts",
    "swimshorts",
    "swim trunk",
    "gym shorts",
    "gym_shorts",
    "gymshorts",
    "running shorts",
    "running_shorts",
    "runningshorts",
    "flip flop",
    "flip-flop",
    "flipflop",
    "slider",
    "slides",
    "slipper",
    "beach",
    "pool",
    " cap",  # leading space avoids false-positive on "captain", "capri"
    "cap ",
    "baseballcap",
    "hat ",
    " hat",
    "fedora",
    "sunhat",
    "sun hat",
    "sunglass",  # covers sunglass / sunglasses
)


def _office_occasion_key(occasion: str) -> str:
    return str(occasion or "").strip().lower().replace(" ", "_")


def _is_office_occasion(occasion: str) -> bool:
    key = _office_occasion_key(occasion)
    return key in {_office_occasion_key(o) for o in _OFFICE_OCCASIONS}


def _is_office_bad_item(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    blob = _ahvi_item_blob(item)
    extra_parts: list[str] = []
    for key in (
        "sub_category",
        "subCategory",
        "use_case",
        "useCase",
        "occasion",
        "occasions",
        "tags",
        "style_tags",
    ):
        value = item.get(key)
        if isinstance(value, (list, tuple, set)):
            extra_parts.extend(str(v or "") for v in value)
        else:
            extra_parts.append(str(value or ""))
    blob = (blob + " " + " ".join(extra_parts)).lower()
    # Pad with spaces so leading/trailing space tokens (e.g. " cap") match
    # edge positions instead of being skipped.
    padded = f" {blob} "
    return any(token in padded for token in _OFFICE_BAD_TOKENS)


def _sanitize_office_board(
    card: dict, occasion: str
) -> tuple[dict, list[str]]:
    """Return (sanitized_card, removed_item_names).

    For office-style occasions, drop items that read as underwear/lounge/
    swim/beach/cap/sunglasses. Leave non-office occasions untouched.
    """
    if not isinstance(card, dict):
        return card, []
    if not _is_office_occasion(occasion):
        return card, []
    items = card.get("items")
    if not isinstance(items, list):
        return card, []
    kept: list = []
    removed: list[str] = []
    for item in items:
        if isinstance(item, dict) and _is_office_bad_item(item):
            removed.append(
                str(item.get("name") or item.get("title") or item.get("id") or "")
            )
            continue
        kept.append(item)
    if not removed:
        return card, []
    sanitized = dict(card)
    sanitized["items"] = kept
    return sanitized, removed


def _has_minimum_board_slots(card: dict) -> bool:
    if not isinstance(card, dict):
        return False
    items = card.get("items")
    if not isinstance(items, list):
        return False
    counts = _ahvi_slot_counts([i for i in items if isinstance(i, dict)])
    top_or_outer = int(counts.get("top", 0) or 0) + int(
        counts.get("outerwear", 0) or 0
    )
    return (
        top_or_outer > 0
        and int(counts.get("bottom", 0) or 0) > 0
        and int(counts.get("footwear", 0) or 0) > 0
    )


def _ahvi_missing_core_slots_response(slot_counts: dict) -> dict:
    missing = []
    if int(slot_counts.get("top", 0) or 0) <= 0:
        missing.append("top")
    if int(slot_counts.get("bottom", 0) or 0) <= 0:
        missing.append("bottom")
    if int(slot_counts.get("footwear", 0) or 0) <= 0:
        missing.append("footwear")
    present = []
    if int(slot_counts.get("top", 0) or 0) > 0:
        present.append("top")
    if int(slot_counts.get("bottom", 0) or 0) > 0:
        present.append("bottom")
    if int(slot_counts.get("footwear", 0) or 0) > 0:
        present.append("footwear")
    if present:
        message = (
            f"I found your {', '.join(present)} slot"
            f"{'' if len(present) == 1 else 's'}, but I still need "
            f"{' and '.join(missing)} to complete this look. "
            "A complete board needs top, bottom, and footwear."
        )
    else:
        message = (
            "I need at least one top, one bottom, and one footwear item before I can build a real style board."
        )
    return {
        "success": True,
        "type": "missing_core_wardrobe_slots",
        "message": message,
        "cards": [],
        "style_boards": [],
        "data": {
            "missing_slots": missing,
            "slot_counts": slot_counts,
        },
        "chips": [
            "Add wardrobe item",
            "Upload outfit pieces",
            "Try again",
        ],
    }


def _ahvi_missing_occasion_response(
    occasion: str,
    slot_counts: dict,
    closest_board: dict | None = None,
) -> dict:
    normalized = str(occasion or "").lower()
    if normalized in {"coffee_date", "coffee"}:
        message = (
            "I don't see enough relaxed coffee-date pieces yet. "
            "Your current options lean too formal or too statement-heavy for that mood."
        )
        missing_items = [
            {
                "label": "Relaxed Oxford Shirt",
                "reason": "Unlocks coffee-date looks without feeling corporate or festive.",
                "cta": "Find this",
            },
            {
                "label": "Soft chinos",
                "reason": "Breaks up formal trouser repetition and keeps the outfit conversational.",
                "cta": "Find this",
            },
            {
                "label": "Clean casual sneakers",
                "reason": "Keeps the look approachable instead of evening-formal.",
                "cta": "Find this",
            },
        ]
        chips = [
            "Show closest option",
            "Find relaxed oxford shirt",
            "Find clean casual sneakers",
        ]
    elif normalized in {"date", "date_night"}:
        message = (
            "I don't see enough strong date-night options yet. "
            "I'd avoid forcing office styling into an evening brief."
        )
        missing_items = [
            {
                "label": "Evening shirt",
                "reason": "Adds date-night personality without looking corporate.",
                "cta": "Find this",
            },
            {
                "label": "Clean casual footwear",
                "reason": "Keeps the look polished without turning office-heavy.",
                "cta": "Find this",
            },
            {
                "label": "One intentional accessory",
                "reason": "Finishes the look without clutter.",
                "cta": "Find this",
            },
        ]
        chips = [
            "Show closest option",
            "Find evening shirt",
            "Find clean casual footwear",
        ]
    elif normalized == "beach":
        message = (
            "I don't see enough beach-ready pieces yet. "
            "I'd rather not force formal trousers or loafers into a beach brief."
        )
        missing_items = [
            {
                "label": "Linen shirt",
                "reason": "Adds breathable coastal texture.",
                "cta": "Find this",
            },
            {
                "label": "Sandals or slides",
                "reason": "Makes the look sand-friendly.",
                "cta": "Find this",
            },
            {
                "label": "Relaxed shorts",
                "reason": "Fixes the silhouette for beachwear.",
                "cta": "Find this",
            },
        ]
        chips = [
            "Show closest option",
            "Find linen shirt",
            "Find sandals",
        ]
    else:
        message = (
            "I don't see enough occasion-ready options yet. "
            "I'd rather not force a weak look."
        )
        missing_items = []
        chips = [
            "Show closest option",
            "Try another occasion",
        ]
    payload = {
        "success": True,
        "type": (
            "weak_occasion_match"
            if _ahvi_has_core_slots(slot_counts)
            else "missing_occasion_wardrobe"
        ),
        "message": message,
        "cards": [],
        "style_boards": [],
        "data": {
            "occasion": normalized,
            "slot_counts": slot_counts,
            "missing_items": missing_items,
            "missing_piece_intelligence": [
                {
                    "missing_item": item.get("label"),
                    "reason": item.get("reason"),
                    "unlocks": (
                        ["Refined Weekend", "Smart Casual Edge", "Polished Casual"]
                        if normalized in {"coffee_date", "coffee", "casual"}
                        else ["Better occasion fit", "Cleaner styling route"]
                    ),
                    "owned": False,
                }
                for item in missing_items
                if isinstance(item, dict)
            ],
            "owned_percentage": 0 if not _ahvi_has_core_slots(slot_counts) else 50,
            "weak_occasion_match": _ahvi_has_core_slots(slot_counts),
            "closest_safe_brief": (
                "evening casual"
                if normalized in {"date", "date_night"}
                else "light casual"
            ),
        },
        "chips": chips,
    }
    if closest_board:
        board_metadata = _board_metadata_summary([closest_board])
        payload["message"] = (
            "I don't see strong occasion-ready pieces yet, but I found one safe direction."
        )
        payload["cards"] = [closest_board]
        payload["style_boards"] = [closest_board]
        payload["data"]["outfits"] = [closest_board]
        payload["data"]["rendered_boards"] = [closest_board]
        payload["data"]["board_metadata"] = board_metadata
        payload["data"]["closest_board"] = closest_board
        payload["meta"] = {
            "board_count": 1,
            "board_metadata": board_metadata,
            "style_signature": closest_board.get("_style_signature"),
            "core_style_signatures": [closest_board.get("_style_core_signature")],
            "weak_occasion_match": True,
            "closest_option": True,
        }
    return payload


def _ahvi_pick_closest_safe_board(cards: list[dict], occasion: str) -> dict | None:
    normalized = str(occasion or "").lower()
    if not cards:
        return None
    blocked_by_occasion = {
        "date": [
            "boardroom",
            "professional",
            "office",
            "executive",
            "corporate",
            "clean friday",
            "workwear",
            "client",
        ],
        "date_night": [
            "boardroom",
            "professional",
            "office",
            "executive",
            "corporate",
            "clean friday",
            "workwear",
            "client",
        ],
        "beach": [
            "black trousers",
            "black pants",
            "loafers",
            "dress shoes",
            "blazer",
            "office",
            "professional",
            "formal",
        ],
    }
    blocked = blocked_by_occasion.get(normalized, [])
    for original in cards:
        if not isinstance(original, dict):
            continue
        card = dict(original)
        blob = _ahvi_item_blob(card)
        for item in card.get("items") or []:
            blob += " " + _ahvi_item_blob(item)
        if any(word in blob for word in blocked):
            continue
        try:
            from services.wardrobe_intelligence_service import board_has_occasion_conflict
            if board_has_occasion_conflict(card, normalized):
                continue
        except Exception:
            pass
        if normalized in {"date", "date_night"}:
            card["title"] = "Soft Statement"
            card["badge"] = "DATE NIGHT"
            card["occasion_label"] = "DATE NIGHT"
            card["occasion"] = "date_night"
            card["explanation"] = (
                "This is the safest evening direction from the current wardrobe. "
                "It keeps the read intentional without leaning into office polish."
            )
        elif normalized == "beach":
            card["title"] = "Closest Light Casual Option"
            card["badge"] = "COASTAL"
            card["occasion_label"] = "COASTAL"
            card["occasion"] = "beach"
            card["explanation"] = (
                "This is the closest relaxed direction from the current wardrobe, "
                "but it is not fully beach-ready yet."
            )
        return card
    return None


def _list_text(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [_safe_text(v) for v in value if _safe_text(v)]
    text = _safe_text(value)
    if not text:
        return []
    return [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]


_STYLE_PREF_MAP = {
    "clean minimal": {
        "style_energy": "minimal/monochrome",
        "palette": "minimal neutral",
        "accessory_policy": "restrained",
        "silhouette": "clean",
    },
    "minimalist": {
        "style_energy": "minimal/monochrome",
        "palette": "minimal neutral",
        "accessory_policy": "restrained",
        "silhouette": "clean",
    },
    "soft elegant": {
        "style_energy": "polished/social",
        "palette": "soft contrast",
        "accessory_policy": "refined",
        "silhouette": "polished",
    },
    "street cool": {
        "style_energy": "elevated/casual",
        "palette": "urban neutral",
        "accessory_policy": "edited",
        "silhouette": "street-smart",
    },
    "boho artisanal": {
        "style_energy": "relaxed/creative",
        "palette": "warm expressive",
        "accessory_policy": "textured",
        "silhouette": "creative",
    },
    "party glam": {
        "style_energy": "expressive/statement",
        "palette": "sharp contrast",
        "accessory_policy": "statement",
        "silhouette": "social",
    },
    "formal chic": {
        "style_energy": "safest/refined",
        "palette": "refined neutral",
        "accessory_policy": "polished",
        "silhouette": "tailored",
    },
    "casual": {
        "style_energy": "elevated/casual",
        "palette": "easy neutral",
        "accessory_policy": "edited",
        "silhouette": "relaxed",
    },
}


def normalize_style_identity(user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile = _dict(user_profile)
    prefs = (
        _list_text(profile.get("stylePreferences"))
        or _list_text(profile.get("styles"))
        or _list_text(profile.get("style_preferences"))
        or _list_text(profile.get("preferred_styles"))
    )
    normalized_prefs = [_safe_text(pref) for pref in prefs if _safe_text(pref)]
    mapped = [
        _STYLE_PREF_MAP.get(_safe_text(pref).lower())
        for pref in normalized_prefs
        if _STYLE_PREF_MAP.get(_safe_text(pref).lower())
    ]
    preferred_energies = []
    preferred_palettes = []
    preferred_silhouettes = []
    accessory_policies = []
    for row in mapped:
        preferred_energies.append(row["style_energy"])
        preferred_palettes.append(row["palette"])
        preferred_silhouettes.append(row["silhouette"])
        accessory_policies.append(row["accessory_policy"])

    identity = {
        "gender": _safe_text(profile.get("gender") or profile.get("style_gender")),
        "shop_prefs": _list_text(profile.get("shopPrefs") or profile.get("shop_prefs")),
        "skin_tone": _safe_text(profile.get("skinTone") or profile.get("skin_tone")),
        "body_shape": _safe_text(profile.get("bodyShape") or profile.get("body_shape")),
        "style_preferences": normalized_prefs,
        "location_label": _safe_text(profile.get("locationLabel") or profile.get("location_label")),
        "preferred_style_energies": list(dict.fromkeys(preferred_energies)),
        "preferred_palettes": list(dict.fromkeys(preferred_palettes)),
        "preferred_silhouettes": list(dict.fromkeys(preferred_silhouettes)),
        "accessory_policy": next((p for p in accessory_policies if p), ""),
        "profile_fields_used": [
            key
            for key, value in {
                "gender": profile.get("gender") or profile.get("style_gender"),
                "shopPrefs": profile.get("shopPrefs") or profile.get("shop_prefs"),
                "skinTone": profile.get("skinTone") or profile.get("skin_tone"),
                "bodyShape": profile.get("bodyShape") or profile.get("body_shape"),
                "stylePreferences": normalized_prefs,
                "locationLabel": profile.get("locationLabel") or profile.get("location_label"),
            }.items()
            if value
        ],
    }
    return identity


def item_image(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return _safe_text(
        item.get("normalized_url")
        or item.get("normalizedUrl")
        or item.get("masked_url")
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
            "outerwear",
            "overshirt",
            "overshirts",
            "jacket",
            "jackets",
            "blazer",
            "blazers",
            "cardigan",
            "cardigans",
            "coat",
            "coats",
            "shacket",
            "layer",
            "layers",
        }
    ):
        return "outerwear"
    if tokens.intersection(
        {
            "top",
            "shirt",
            "shirts",
            "tee",
            "tshirt",
            "polo",
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
    if resolved in {"top", "bottom", "dress", "footwear", "outerwear", "accessory"}:
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
    elif resolved == "outerwear":
        row.setdefault("category", "Outerwear")
    elif resolved == "accessory":
        row.setdefault("category", "Accessories")

    image = item_image(row)
    if image:
        row.setdefault("image_url", image)
        row.setdefault("imageUrl", image)
        row.setdefault("masked_url", image)
        row.setdefault("maskedUrl", image)
    if TRUST_LAYER_RUNTIME_INFERENCE and "formality" not in row:
        attrs = infer_style_attributes(row)
        for key, value in attrs.items():
            row.setdefault(key, value)
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


def _variant_card_signature(card: Any) -> str:
    """Broad-pool exact signature.

    The normal signature is intentionally conservative for small/default calls.
    For 5-6 board generation, keep card variants distinct when AHVI changes the
    hero, visible pieces, accessory direction, or target archetype.
    """
    if not isinstance(card, dict):
        return ""
    metadata = card.get("style_metadata") if isinstance(card.get("style_metadata"), dict) else {}
    source_sig = _safe_text(
        card.get("pipeline_style_signature")
        or metadata.get("pipeline_style_signature")
    )
    role_parts = []
    for role in ("dress", "top", "bottom", "footwear", "outerwear"):
        key = _role_key(card, role)
        if key:
            role_parts.append(f"{role}:{key}")
    accessory_parts = [f"accessory:{key}" for key in sorted(set(_accessory_keys(card)))]
    target_parts = [
        f"hero:{_safe_text(card.get('hero_item_id'))}",
        f"target:{_safe_text(card.get('_target_archetype') or card.get('style_archetype'))}",
        f"energy:{_safe_text(card.get('_target_style_energy') or card.get('style_energy'))}",
        f"pipeline:{source_sig}",
    ]
    clean = [part.lower() for part in role_parts + accessory_parts + target_parts if part and not part.endswith(":")]
    if clean:
        return "|".join(clean)
    return card_signature(card)


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
    blob = " ".join(_safe_text(item.get(k)) for k in ("name", "label", "category", "type", "sub_category", "subcategory"))
    tokens = _tokens(blob)
    for typ, words in (
        ("watch", {"watch", "watches"}),
        ("eyewear", {"sunglass", "sunglasses", "eyewear", "glasses"}),
        ("bottle", {"bottle", "flask", "hydration"}),
        ("towel", {"towel", "towels"}),
        ("bag", {"bag", "bags", "purse", "tote", "clutch"}),
        ("belt", {"belt", "belts"}),
        ("headwear", {"cap", "caps", "hat", "hats", "beanie"}),
        ("scarf", {"scarf", "scarves"}),
        ("jewelry", {"ring", "rings", "necklace", "bracelet", "earring", "jewelry", "jewellery"}),
    ):
        if tokens.intersection(words):
            return typ
    return "accessory"


_SOCIAL_ACCESSORIES = {"watch", "belt", "jewelry", "eyewear", "bag", "scarf"}
_PROFESSIONAL_ACCESSORIES = {"watch", "belt", "jewelry", "bag", "scarf"}
_DATE_ACCESSORIES = {"watch", "jewelry", "scarf", "belt"}
_BEACH_ACCESSORIES = {"eyewear", "bag", "towel", "headwear", "bottle"}
_WORKOUT_ACCESSORIES = {"bag", "bottle", "watch", "headwear", "towel"}
_CASUAL_ACCESSORIES = {"watch", "belt", "jewelry", "eyewear", "bag", "scarf", "headwear", "bottle"}


def _allows_headwear(query: str) -> bool:
    q = str(query or "").lower()
    return any(
        k in q
        for k in (
            "cap",
            "hat",
            "street",
            "sport",
            "outdoor",
            "travel",
            "airport",
            "flight",
            "vacation",
            "beach",
            "weekend",
            "rave",
            "festival",
        )
    )


def _explicit_accessory_types(query: str) -> set:
    """Accessory types the user named outright ("... and a bag"). These are hard
    constraints: the occasion accessory whitelist must not drop them (a bag was
    previously discarded for date/dinner because it is not in _DATE_ACCESSORIES)."""
    try:
        from services.style_explicit_roles import extract_requested_roles

        roles = set(extract_requested_roles(query))
    except Exception:  # noqa: BLE001 - never break curation
        return set()
    types: set = set()
    if "bag" in roles:
        types.add("bag")
    return types


def _accessory_allowed_for_query(item: Dict[str, Any], query: str) -> bool:
    typ = _accessory_type(item)
    kind = _occasion_kind(query)
    if typ in _explicit_accessory_types(query):
        return True
    if typ == "headwear" and not _allows_headwear(query):
        return False
    if kind == "office" or kind == "wedding":
        return typ in _PROFESSIONAL_ACCESSORIES
    if kind == "swimming":
        return typ in _BEACH_ACCESSORIES
    if kind == "date":
        return typ in _DATE_ACCESSORIES
    if kind == "party":
        return typ in _SOCIAL_ACCESSORIES or _allows_headwear(query)
    if kind == "beach":
        return typ in _BEACH_ACCESSORIES
    if kind == "workout":
        return typ in _WORKOUT_ACCESSORIES
    if kind in {"travel", "casual", "daily"}:
        return typ in _CASUAL_ACCESSORIES
    return typ != "headwear" or _allows_headwear(query)


def _accessory_priority(item: Dict[str, Any], query: str) -> int:
    typ = _accessory_type(item)
    kind = _occasion_kind(query)
    # Explicitly requested types outrank every occasion default so the budget
    # cap can never trim them away.
    if typ in _explicit_accessory_types(query):
        return -1
    if kind == "office":
        order = ["watch", "belt", "bag", "scarf", "jewelry", "accessory", "eyewear", "headwear"]
    elif kind == "date":
        order = ["watch", "jewelry", "scarf", "belt", "bag", "accessory", "eyewear", "headwear"]
    elif kind == "wedding":
        order = ["watch", "belt", "jewelry", "scarf", "bag", "accessory", "eyewear", "headwear"]
    elif kind == "party":
        order = ["jewelry", "watch", "belt", "eyewear", "bag", "scarf", "headwear", "accessory"]
    elif kind == "beach":
        order = ["eyewear", "bag", "towel", "headwear", "bottle", "watch", "accessory", "jewelry"]
    elif kind == "workout":
        order = ["bottle", "bag", "watch", "headwear", "towel", "accessory", "jewelry"]
    elif kind == "travel":
        order = ["bag", "eyewear", "watch", "belt", "scarf", "headwear", "jewelry", "accessory"]
    else:
        order = ["watch", "belt", "eyewear", "bag", "jewelry", "scarf", "headwear", "accessory"]
    try:
        return order.index(typ)
    except ValueError:
        return 99


def _curate_accessories_for_card(card: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Keep boards styled, not stuffed.

    The upstream outfit builder may attach any available accessory. This final
    pass removes occasion-breaking accents (for example a cap in office/date),
    dedupes accessory types, and caps the final board to six useful items.
    """
    if not isinstance(card, dict):
        return card
    core = [
        item
        for item in card.get("items", [])
        if isinstance(item, dict) and item_role(item) != "accessory"
    ]
    raw_accessories: List[Dict[str, Any]] = []
    for item in list(card.get("accessories") or []) + list(card.get("items") or []):
        if isinstance(item, dict) and item_role(item) == "accessory":
            raw_accessories.append(item)

    seen_keys: set[str] = set()
    seen_types: set[str] = set()
    accessories: List[Dict[str, Any]] = []
    for item in sorted(raw_accessories, key=lambda x: (_accessory_priority(x, query), item_key(x))):
        if not _accessory_allowed_for_query(item, query):
            continue
        key = item_key(item)
        typ = _accessory_type(item)
        if key and key in seen_keys:
            continue
        if typ in seen_types:
            continue
        normalized = normalize_item(item, "accessory")
        accessories.append(normalized)
        if key:
            seen_keys.add(key)
        seen_types.add(typ)

    # Premium boards do not need six items. They may carry six only when the
    # accessories are genuinely useful and distinct.
    accessory_budget = max(0, min(6 - len(core), 3))
    if _occasion_kind(query) in {"office", "date", "wedding"}:
        accessory_budget = min(accessory_budget, 2)
    explicit_types = _explicit_accessory_types(query)
    if explicit_types:
        # Never let the occasion cap evict a type the user asked for by name.
        accessory_budget = max(
            accessory_budget,
            sum(1 for a in accessories if _accessory_type(a) in explicit_types),
        )
    accessories = accessories[:accessory_budget]

    fixed = dict(card)
    fixed["accessories"] = accessories
    fixed["items"] = core + accessories
    fixed["item_count"] = len(fixed["items"])
    fixed["accessory_policy_applied"] = {
        "max_items": 6,
        "accessory_budget": accessory_budget,
        "accessory_types": [_accessory_type(item) for item in accessories],
    }
    return fixed


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
        if role not in {"top", "bottom", "dress", "footwear", "outerwear", "accessory"}:
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
    for role in ("dress", "outerwear", "top", "bottom", "footwear"):
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
    normalized = _normalize_occasion_value("", q)
    return {
        "temple_modest": any(k in q for k in ("temple", "mandir", "pooja", "puja", "religious", "shrine", "darshan")),
        "swimming": any(k in q for k in ("swim", "swimming", "swimwear", "swimsuit", "pool")),
        "beach": normalized in {"beach", "beach_dinner"} or any(k in q for k in ("beach", "seaside", "coastal", "sand-friendly", "sand friendly")),
        "workout": any(k in q for k in ("workout", "gym", "fitness", "training", "yoga", "running")),
        "brunch": any(k in q for k in ("brunch",)),
        "office": normalized in {"office_meeting", "client_presentation", "client_dinner"} or any(k in q for k in ("office", "work", "meeting", "client", "business", "interview", "corporate")),
        "date": normalized in {"date_night", "coffee_date", "first_date", "casual_dinner", "team_dinner"} or any(k in q for k in ("date", "dinner", "night")),
        "party": any(k in q for k in ("party", "club", "after-hours", "night out")),
        "travel": any(k in q for k in ("travel", "airport", "flight", "vacation", "trip")),
        "wedding": normalized in {"wedding_guest", "funeral"} or any(k in q for k in ("wedding", "reception", "ceremony", "sangeet", "formal event", "event", "funeral", "memorial")),
        "casual": any(k in q for k in ("casual", "weekend", "errand", "coffee")),
    }


def _item_by_role(card: Dict[str, Any], role: str) -> Dict[str, Any]:
    for item in card.get("items", []):
        if isinstance(item, dict) and item_role(item) == role:
            return item
    return {}


def _role_key(card: Dict[str, Any], role: str) -> str:
    return item_key(_item_by_role(card, role))


def _base_outfit_signature(card: Dict[str, Any]) -> str:
    if _role_key(card, "dress"):
        return _role_key(card, "dress")
    return "|".join(
        part
        for part in [
            _role_key(card, "top"),
            _role_key(card, "bottom"),
        ]
        if part
    )


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


def _item_formality(item: Dict[str, Any]) -> float:
    try:
        return float(_dict(item).get("formality") or 3)
    except Exception:
        return 3.0


def _item_visual_weight(item: Dict[str, Any]) -> float:
    try:
        return float(_dict(item).get("visual_weight") or 2)
    except Exception:
        return 2.0


def _item_cluster(item: Dict[str, Any]) -> str:
    return _safe_text(_dict(item).get("aesthetic_cluster") or "polished")


def _item_occasion_fitness(item: Dict[str, Any], kind: str) -> float:
    data = _dict(_dict(item).get("occasion_fitness"))
    try:
        return float(data.get(kind) if data.get(kind) is not None else 0.5)
    except Exception:
        return 0.5


def _coherence_score(card: Dict[str, Any]) -> float:
    core_items = [
        item for item in card.get("items", [])
        if isinstance(item, dict) and item_role(item) in {"top", "bottom", "dress", "footwear"}
    ]
    if len(core_items) < 2:
        return 0.0
    formalities = [_item_formality(item) for item in core_items]
    weights = [_item_visual_weight(item) for item in core_items]
    clusters = [_item_cluster(item) for item in core_items]
    formality_spread = max(formalities) - min(formalities)
    weight_total = sum(weights)
    cluster_count = len(set(clusters))
    score = 2.0
    if formality_spread <= 1.5:
        score += 1.0
    elif formality_spread >= 3.0:
        score -= 2.0
    if weight_total <= 8:
        score += 0.8
    elif weight_total >= 12:
        score -= 1.5
    if cluster_count <= 2:
        score += 0.8
    elif cluster_count >= 4:
        score -= 1.2
    return score


def _occasion_kind(query: str) -> str:
    normalized = _normalize_occasion_value("", query)
    if normalized in {
        "coffee_date", "first_date", "date_night", "casual_dinner",
        "client_dinner", "beach_dinner", "wedding_guest", "funeral",
        "office_meeting", "client_presentation", "basketball_game",
        "team_dinner",
    }:
        return normalized
    flags = _occasion_flags(query)
    for key in ("temple_modest", "swimming", "beach", "workout", "brunch", "office", "date", "party", "travel", "wedding", "casual"):
        if flags.get(key):
            return key
    # Generic "today"/"daily"/no signal → daily, NOT office.
    return "daily"


def _style_direction(query: str) -> str:
    q = str(query or "").lower()
    kind = _occasion_kind(query)
    if kind == "coffee_date":
        return "relaxed_social_coffee"
    if kind == "first_date":
        return "approachable_first_date"
    if kind == "casual_dinner":
        return "casual_evening"
    if kind == "client_dinner":
        return "professional_social_transition"
    if kind == "beach_dinner":
        return "coastal_evening"
    if kind == "wedding_guest":
        return "wedding_guest"
    if kind == "funeral":
        return "respectful_understated"
    if kind == "office_meeting":
        return "smart_casual_office"
    if kind == "client_presentation":
        return "corporate_office"
    if kind == "basketball_game":
        return "sports_casual"
    if kind == "team_dinner":
        return "team_social"
    if kind == "swimming":
        return "swim_functional"
    if kind == "beach":
        return "coastal_casual"
    if kind == "workout":
        return "training_functional"
    if kind == "brunch":
        return "daytime_polish"
    if any(k in q for k in ("corporate", "boardroom", "formal", "client", "presentation")):
        return "corporate_office"
    if any(k in q for k in ("creative", "agency", "studio", "design")):
        return "creative_office"
    if any(k in q for k in ("startup", "start-up", "casual office")):
        return "startup_office"
    if any(k in q for k in ("friday", "relaxed office", "casual friday")):
        return "friday_office"
    if kind == "office":
        return "smart_casual_office"
    if kind == "daily":
        return "smart_casual_office"
    flags = _occasion_flags(query)
    if flags["swimming"]:
        return "swim_functional"
    if flags["beach"]:
        return "coastal_casual"
    if flags["workout"]:
        return "training_functional"
    if flags["brunch"]:
        return "daytime_polish"
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


def interpret_occasion(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if interpret_occasion_context is not None:
        return interpret_occasion_context(query, context)

    ctx = _dict(context)
    q = _safe_text(query).lower()
    kind = _occasion_kind(query)
    weather = _safe_text(ctx.get("weather") or ctx.get("condition")).lower()
    time_of_day = _safe_text(ctx.get("time_of_day") or ctx.get("hour")).lower()
    module_context = _safe_text(ctx.get("module_context")).lower()

    chips: List[Dict[str, str]] = []
    ask_user = False
    reason = ""
    confidence = "high"

    if kind == "swimming":
        brief = "swim-ready, pool appropriate, quick-dry, easy layers"
    elif kind == "date":
        if any(k in q for k in ("dinner", "restaurant", "candle", "tonight", "evening")):
            brief = "polished dinner, low light, intentional but not overworked"
        elif any(k in q for k in ("coffee", "day", "afternoon", "walk")):
            brief = "daytime date, easy polish, movement friendly"
        elif any(k in q for k in ("rooftop", "drinks", "bar")):
            brief = "rooftop casual, warm evening, relaxed polish"
        else:
            brief = "polished date, social ease, clean footwear"
            ask_user = "date" in q and not any(k in q for k in ("today", "now", "quick", "suggest"))
            confidence = "medium" if ask_user else "high"
            reason = "date_venue_changes_formality"
            chips = [
                {"label": "Polished dinner", "value": "date_polished_dinner"},
                {"label": "Rooftop easy", "value": "date_rooftop_easy"},
                {"label": "Daytime quiet", "value": "date_daytime_quiet"},
            ]
    elif kind == "office":
        if any(k in q for k in ("client", "presentation", "meeting", "corporate", "interview")):
            brief = "client-facing office, polish required, smart footwear"
        elif any(k in q for k in ("creative", "studio", "agency")):
            brief = "creative office, expressive but controlled"
        elif any(k in q for k in ("startup", "start-up", "friday", "casual office")):
            brief = "relaxed office, intentional smart casual"
        else:
            brief = "smart casual office, clean structure, credible footwear"
    elif kind == "party":
        if any(k in q for k in ("rave", "club", "edm", "festival")):
            brief = "rave party, after-hours energy, movement friendly, expressive edge"
        elif any(k in q for k in ("cocktail", "lounge", "bar", "drinks")):
            brief = "cocktail party, sharp social polish, after-dark restraint"
        elif any(k in q for k in ("house", "birthday", "friends")):
            brief = "house party, relaxed social polish, memorable but easy"
        else:
            brief = "after-hours social, sharper contrast, memorable but edited"
            ask_user = True
            confidence = "medium"
            reason = "party_atmosphere_changes_formality"
            chips = [
                {"label": "Rave energy", "value": "party_rave_energy"},
                {"label": "Cocktail sharp", "value": "party_cocktail_sharp"},
                {"label": "House easy", "value": "party_house_easy"},
            ]
    elif kind == "travel":
        brief = "travel comfort polish, movement friendly, weather aware"
    elif kind == "wedding":
        brief = "formal event, respectful polish, ceremony-aware restraint"
        if not any(k in q for k in ("reception", "wedding", "formal", "traditional", "ceremony")):
            ask_user = True
            confidence = "medium"
            reason = "event_formality_or_cultural_context"
            chips = [
                {"label": "Ceremony refined", "value": "event_ceremony_refined"},
                {"label": "Reception polish", "value": "event_reception_polish"},
                {"label": "Statement occasion", "value": "event_statement_occasion"},
            ]
    elif kind == "casual":
        brief = "clean daily, comfortable but intentional"
    else:
        brief = "elevated daily, smart casual, repeat-aware"

    if "rain" in weather:
        brief = f"{brief}, rain-ready"
    elif any(k in weather for k in ("hot", "summer")):
        brief = f"{brief}, heat-conscious"
    elif any(k in weather for k in ("cold", "winter")):
        brief = f"{brief}, layered"
    if time_of_day and kind in {"date", "party"}:
        brief = f"{brief}, {time_of_day}"

    # Normal style flows should not become questionnaires. Only ask when a
    # single missing choice can prevent a visibly wrong board.
    if module_context in {"style", "wardrobe"} and ask_user:
        ask_user = True

    return {
        "resolved_brief": brief,
        "confidence": confidence,
        "ask_user": bool(ask_user),
        "question": (
            "Which brief is closer?"
            if ask_user
            else ""
        ),
        "chips": chips[:3],
        "board_generation_notes": {
            "occasion_kind": kind,
            "style_direction": _style_direction(query),
            "reason": reason or "context_sufficient",
            "max_questions": 3,
        },
    }


def _clarification_response(interpretation: Dict[str, Any]) -> Dict[str, Any]:
    chips = interpretation.get("chips") if isinstance(interpretation.get("chips"), list) else []
    return {
        "success": True,
        "message": interpretation.get("question") or "Which brief is closer?",
        "board": "style",
        "type": "clarification",
        "cards": [],
        "style_boards": [],
        "chips": chips[:3],
        "board_ids": "",
        "data": {"outfits": [], "rendered_boards": []},
        "meta": {
            "question_count": 1,
            "max_questions": 3,
            "reason": _dict(interpretation.get("board_generation_notes")).get("reason"),
            "occasion_interpretation": interpretation,
        },
        "audio_job_id": "offline",
    }


def _wardrobe_gap_response(
    *,
    query: str,
    wardrobe: Any,
    interpretation: Dict[str, Any],
    finalized: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    notes = _dict(interpretation.get("board_generation_notes"))
    occasion = _safe_text(interpretation.get("occasion") or notes.get("occasion_kind") or _occasion_kind(query))
    occasion = _normalize_occasion_value(occasion, query) or occasion
    wardrobe_items = wardrobe if isinstance(wardrobe, list) else []
    slot_counts = _ahvi_slot_counts(wardrobe_items)
    has_core_slots = _ahvi_has_core_slots(slot_counts)
    response_type = "weak_occasion_match" if has_core_slots else "missing_occasion_wardrobe"
    if detect_wardrobe_gap is not None and get_occasion_rule is not None:
        try:
            gap = detect_wardrobe_gap(wardrobe_items, occasion, get_occasion_rule(occasion))
        except Exception:
            gap = {}
    else:
        gap = {}
    rule = _dict(gap.get("rule"))
    missing_items = gap.get("missing_items") if isinstance(gap.get("missing_items"), list) else []
    missing_items = [item for item in missing_items if isinstance(item, dict)][:4]
    if not missing_items:
        missing_items = [
            {"label": "Occasion-ready hero", "reason": "Starts the outfit in the right atmosphere", "cta": "Find this"},
            {"label": "Right footwear", "reason": "Footwear controls the occasion register", "cta": "Find this"},
        ]
    find_chips = [
        {
            "label": f"Find {_safe_text(item.get('label')).lower()}",
            "value": f"find_this:{_safe_text(item.get('label'))}",
        }
        for item in missing_items[:2]
        if _safe_text(item.get("label"))
    ]

    chips = (
        [{"label": "Show closest option", "value": "show_closest_option"}]
        + find_chips
    )[:3]

    normalized_occasion = (
        _safe_text(occasion).lower().replace("-", "_").replace(" ", "_")
    )

    if normalized_occasion in {"date", "date_night", "datenight"}:
        message = (
            "I don't see enough strong date-night options yet. "
            "I'd avoid forcing office styling into an evening brief."
        )
    elif normalized_occasion in {"beach", "beach_wear", "beachwear", "coastal"}:
        message = (
            "I don't see enough beach-ready pieces yet. "
            "I'd rather not force formal trousers or loafers into a beach brief."
        )
    elif normalized_occasion in {"temple_modest", "temple", "mandir", "pooja", "puja"}:
        message = (
            "I don't see enough temple-ready modest pieces yet. "
            "I'd rather not force a club or office silhouette into a temple brief."
        )
    else:
        brief = _safe_text(
            interpretation.get("resolved_brief")
            or rule.get("resolved_brief")
            or occasion
        )
        message = (
            "I don't see enough occasion-ready options yet. "
            f"For {brief}, I would rather protect the look than force a board that reads wrong."
        )

    data = _dict(_dict(finalized).get("data"))
    data.update(
        {
            "outfits": [],
            "rendered_boards": [],
            "missing_items": missing_items,
            "missing_piece_intelligence": [
                {
                    "missing_item": item.get("label"),
                    "reason": item.get("reason"),
                    "unlocks": (
                        ["Refined Weekend", "Smart Casual Edge", "Polished Casual"]
                        if normalized_occasion in {"coffee_date", "coffee", "casual"}
                        else ["Better occasion fit", "Cleaner styling route"]
                    ),
                    "owned": False,
                }
                for item in missing_items
                if isinstance(item, dict)
            ],
            "owned_percentage": 0 if not has_core_slots else 50,
            "find_this_recommendations": missing_items,
            "closest_safe_brief": _safe_text(gap.get("closest_safe_brief")) or "clean daily",
            "occasion_interpretation": interpretation,
            "wardrobe_gap": {
                "occasion": occasion,
                "slot_counts": slot_counts,
                "slot_scores": _dict(gap.get("slot_scores")),
                "has_enough": bool(gap.get("has_enough")),
                "weak_occasion_match": has_core_slots,
                "reason": response_type,
            },
        }
    )

    logger.info(
        "ahvi.return_response type=%s occasion=%s message=%s chips=%s",
        response_type,
        occasion,
        message,
        chips,
    )

    return {
        "success": True,
        "message": message,
        "board": "style",
        "type": response_type,
        "cards": [],
        "style_boards": [],
        "chips": chips,
        "board_ids": "",
        "data": data,
        "meta": {
            **_dict(_dict(result).get("meta")),
            **_dict(_dict(finalized).get("meta")),
            "board_count": 0,
            "occasion_interpretation": interpretation,
            "wardrobe_limitation_reason": response_type,
            "wardrobe_gap": {
                "occasion": occasion,
                "missing_count": len(missing_items),
                "slot_counts": slot_counts,
                "weak_occasion_match": has_core_slots,
                "closest_safe_brief": _safe_text(gap.get("closest_safe_brief")) or "clean daily",
            },
        },
        "audio_job_id": "offline",
    }


def _office_direction(query: str) -> str:
    # Backward-compatible name for existing callers/tests. The returned value is
    # now the general style direction, not only an office subtype.
    return _style_direction(query)


def _footwear_mood(item: Dict[str, Any]) -> str:
    text = _item_blob(item)
    if any(k in text for k in ("birkenstock", "sandal", "slipper", "slider", "slides", "flip flop", "flip-flop", "crocs")):
        return "relaxed sandal"
    if any(k in text for k in ("chunky", "runner", "running", "trainer", "gym", "athletic", "sports", "hiking", "trail")):
        return "athletic"
    if any(k in text for k in ("loafer", "oxford", "derby", "formal", "monk strap")):
        return "formal polish"
    if any(k in text for k in ("slip-on", "slip on", "suede", "beige slip")):
        return "polished sneaker"
    if any(k in text for k in ("leather sneaker", "minimal sneaker", "white sneaker", "cream sneaker", "clean sneaker")):
        return "polished sneaker"
    if "sneaker" in text:
        return "casual sneaker"
    if any(k in text for k in ("boot", "chelsea")):
        return "structured boot"
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
            "casual sneaker": -2.0,
            "relaxed sandal": -7.0,
            "athletic": -6.0,
        }.get(mood, 0.0)
    if direction in {"clean_daily", "daily"}:
        return {
            "formal polish": 1.4,
            "polished sneaker": 1.6,
            "structured boot": 1.0,
            "casual sneaker": 0.2,
            "relaxed sandal": -2.5,
            "athletic": -2.0,
        }.get(mood, 0.0)
    if direction in {"comfort_polish"}:
        return {
            "formal polish": 0.2,
            "polished sneaker": 1.7,
            "structured boot": 1.0,
            "casual sneaker": 1.2,
            "relaxed sandal": 0.1,
            "athletic": -0.5,
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
        "coherence_score": round(_coherence_score(card), 3),
        "hero_visual_weight": _item_visual_weight(_item_by_role(card, "top") or _item_by_role(card, "dress")),
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
    top_bottom = _top_bottom_signature(card)
    selected_bottoms = [_role_key(s, "bottom") for s in selected]
    selected_tops = [_role_key(s, "top") or _role_key(s, "dress") for s in selected]
    selected_top_bottoms = {_top_bottom_signature(s) for s in selected}
    if top and top in selected_tops:
        penalty += selected_tops.count(top) * 3.0
    if bottom and bottom in selected_bottoms:
        penalty += selected_bottoms.count(bottom) * 4.5
    if top_bottom and top_bottom in selected_top_bottoms:
        # This is the main visible failure: same shirt + same pant with shoe/accessory swaps.
        penalty += 18.0
    if _footwear_mood(_item_by_role(card, "footwear")) in {
        _footwear_mood(_item_by_role(s, "footwear")) for s in selected
    }:
        penalty += 1.2
    if _role_key(card, "footwear") and _role_key(card, "footwear") in {_role_key(s, "footwear") for s in selected}:
        penalty += 2.5
    if _style_energy(card, _safe_text(card.get("_style_query"))) in {
        _style_energy(s, _safe_text(s.get("_style_query"))) for s in selected
    }:
        penalty += 1.5
    if set(_accessory_types(card)).intersection({typ for s in selected for typ in _accessory_types(s)}):
        penalty += 0.6
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


def _daily_composition_score(card: Dict[str, Any], query: str) -> float:
    kind = _occasion_kind(query)
    direction = _style_direction(query)
    if kind not in {"daily", "casual"} and direction not in {"daily", "clean_daily", "smart_casual_office"}:
        return 0.0
    text = _card_blob(card)
    score = 0.0
    if _item_by_role(card, "outerwear"):
        score += 1.4
    if any(k in text for k in ("overshirt", "jacket", "blazer", "cardigan", "layer", "shacket")):
        score += 1.0
    if any(k in text for k in ("linen", "knit", "textured", "ribbed", "cotton", "suede", "denim", "canvas")):
        score += 0.9
    if any(k in text for k in ("watch", "belt", "bag", "sling", "bracelet", "ring", "necklace")):
        score += 0.45
    if any(k in text for k in ("graphic tee", "plain tee", "t-shirt", "tee")) and not any(k in text for k in ("layer", "jacket", "overshirt", "blazer")):
        score -= 0.5
    return score


def _daily_composition_notes(card: Dict[str, Any], query: str) -> List[str]:
    if _daily_composition_score(card, query) <= 0:
        return []
    text = _card_blob(card)
    notes: List[str] = []
    if _item_by_role(card, "outerwear") or any(k in text for k in ("overshirt", "jacket", "blazer", "cardigan", "layer")):
        notes.append("layering")
    if any(k in text for k in ("linen", "knit", "textured", "ribbed", "cotton", "suede", "denim", "canvas")):
        notes.append("texture")
    if _hero_item(card):
        notes.append("hero_piece")
    if any(k in text for k in ("watch", "belt", "bag", "sling", "bracelet", "ring", "necklace")):
        notes.append("accessory_balance")
    return list(dict.fromkeys(notes))[:4]


def _style_dna_alignment_text(style_identity: Dict[str, Any], controlled_archetype: str) -> str:
    prefs = style_identity if isinstance(style_identity, dict) else {}
    style_values: List[str] = []
    for key in ("stylePreferences", "style_preferences", "preferred_styles", "preferredStyle"):
        value = prefs.get(key)
        if isinstance(value, list):
            style_values.extend(str(v).strip() for v in value if str(v).strip())
        elif str(value or "").strip():
            style_values.append(str(value).strip())
    if not style_values and controlled_archetype:
        style_values.append(controlled_archetype)
    style_values = list(dict.fromkeys(style_values))[:3]
    if not style_values:
        return ""
    return f"Your Style DNA leans {', '.join(style_values)}, so this direction should feel natural rather than forced."


def _quality_score(card: Dict[str, Any], query: str) -> float:
    text = _card_blob(card)
    kind = _occasion_kind(query)
    flags = _occasion_flags(query)
    score = float(card.get("score") or 0.0) / 100.0
    roles = {item_role(item) for item in card.get("items", []) if isinstance(item, dict)}
    score += len(roles.intersection({"top", "bottom", "dress", "footwear"}))
    score += min(2, len([x for x in card.get("accessories", []) if isinstance(x, dict)])) * 0.35
    score += _coherence_score(card)

    if flags["office"]:
        if any(k in text for k in ("button-down", "button down", "shirt", "trouser", "off white", "loafer")):
            score += 3.0
        if any(k in text for k in ("black pants", "black trousers")):
            score += 0.7
        if any(k in text for k in ("watch", "belt", "sneaker")):
            score += 1.0
        if any(k in text for k in ("tropical", "hawaiian", "vacation", "beach", "party", "loud")):
            score -= 5.0
        if any(k in text for k in ("shorts", "slipper", "slides", "slider", "sandal", "birkenstock", "crocs")):
            score -= 4.0
        score += _top_office_score(_item_by_role(card, "top"), query)
        score += _footwear_formality_score(_item_by_role(card, "footwear"), query)
    if flags["date"]:
        if any(k in text for k in ("watch", "off white", "loafer")):
            score += 1.5
        if any(k in text for k in ("black pants", "black trousers", "black shirt")):
            score += 0.6
        score += _footwear_formality_score(_item_by_role(card, "footwear"), query)
    if kind == "coffee_date":
        if any(k in text for k in ("oxford", "linen", "polo", "chino", "denim", "clean sneaker", "suede loafer", "cotton")):
            score += 2.4
        if any(k in text for k in ("formal trouser", "formal trousers", "black trousers", "black pants", "corporate", "boardroom", "office")):
            score -= 3.2
        if any(k in text for k in ("wedding", "embroidered", "embroidery", "festive", "shiny", "satin", "sequin", "gold ring")):
            score -= 5.0
    if kind == "client_dinner":
        if any(k in text for k in ("button", "shirt", "trouser", "chino", "loafer", "watch", "belt", "blazer")):
            score += 2.0
        if any(k in text for k in ("loud print", "party shirt", "neon", "slides", "slipper", "gym", "track")):
            score -= 4.0
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
    daily_boost = _daily_composition_score(card, query)
    if daily_boost:
        score += daily_boost
        logger.info(
            "AHVI_DAILY_COMPOSITION_SCORE_APPLIED boost=%.2f notes=%s",
            daily_boost,
            _daily_composition_notes(card, query),
        )
    return score


def _occasion_fit_score(card: Dict[str, Any], query: str) -> float:
    text = _card_blob(card)
    kind = _occasion_kind(query)
    footwear = _footwear_energy(_item_by_role(card, "footwear"))
    formality = _formality_energy(card, query)
    silhouette = _silhouette_category(card)
    score = 0.0
    core_items = [
        item for item in card.get("items", [])
        if isinstance(item, dict) and item_role(item) in {"top", "bottom", "dress", "footwear"}
    ]
    if core_items:
        score += sum(_item_occasion_fitness(item, kind) for item in core_items) / len(core_items)
        if get_occasion_rule is not None and score_item_for_occasion is not None:
            try:
                rule = get_occasion_rule(kind)
                score += sum(score_item_for_occasion(item, rule) for item in core_items) / len(core_items)
            except Exception:
                pass
    if kind == "beach":
        if footwear in {"relaxed", "casual", "elevated casual"}:
            score += 2.0
        if any(
            k in text
            for k in (
                "linen", "cotton", "shorts", "sandals", "slides",
                "espadrille", "tote", "sunglasses", "tropical", "hawaiian",
                "floral", "printed", "patterned", "resort print",
                "vacation print", "camp collar", "open collar", "open shirt",
            )
        ):
            score += 2.2
        if any(k in text for k in ("black pants", "black trousers", "loafers", "dress shoes", "blazer", "suit", "charcoal")):
            score -= 8.0
    elif kind == "brunch":
        if footwear in {"casual", "elevated casual", "polished", "structured"}:
            score += 1.2
        if any(k in text for k in ("linen", "cotton", "dress", "shirt", "jeans", "chino")):
            score += 0.8
    elif kind == "workout":
        if any(k in text for k in ("training", "gym", "shorts", "leggings", "jogger", "running shoes", "sports")):
            score += 2.0
        if footwear in {"polished", "structured"} or any(k in text for k in ("loafer", "dress shoes", "blazer")):
            score -= 7.0
    if kind == "office":
        if formality in {"formal", "smart"}:
            score += 2.0
        if footwear in {"relaxed", "athletic"}:
            score -= 4.0
        if any(
            k in text
            for k in (
                "tropical", "hawaiian", "vacation", "beach", "loud",
                "red loafers", "burgundy loafers", "red shoes", "party shirt",
                "embroidered", "festive", "satin", "shiny", "glossy",
            )
        ) and _style_direction(query) not in {"creative_office", "startup_office", "friday_office"}:
            score -= 5.0
    elif kind == "coffee_date":
        if formality in {"casual", "smart"} or footwear in {"casual", "elevated casual", "polished"}:
            score += 1.5
        if any(k in text for k in ("oxford", "linen", "polo", "chino", "denim", "clean sneaker", "suede loafer", "cotton")):
            score += 2.5
        if any(k in text for k in ("formal trouser", "formal trousers", "black trousers", "black pants", "corporate", "boardroom", "office")):
            score -= 4.0
        if any(k in text for k in ("wedding", "embroidered", "embroidery", "festive", "shiny", "satin", "sequin", "gold ring")):
            score -= 7.0
    elif kind in {"first_date", "casual_dinner", "team_dinner"}:
        if formality in {"smart", "casual"} or footwear in {"polished", "structured", "elevated casual", "casual"}:
            score += 1.5
        if any(k in text for k in ("gym", "track", "slides", "slippers", "flip flop", "tuxedo", "boardroom")):
            score -= 4.0
    elif kind == "client_dinner":
        if formality in {"smart", "formal"} or footwear in {"polished", "structured"}:
            score += 2.0
        if any(k in text for k in ("shirt", "trouser", "chino", "loafer", "watch", "belt", "blazer")):
            score += 1.5
        if any(k in text for k in ("shorts", "slides", "slippers", "gym", "track", "loud print", "neon", "beach")):
            score -= 6.0
    elif kind == "beach_dinner":
        if any(k in text for k in ("linen", "cotton", "camp collar", "resort", "lightweight", "chino", "sandal", "espadrille", "slides")):
            score += 2.4
        if any(k in text for k in ("office trouser", "black trousers", "black pants", "oxford", "derby", "formal leather", "heavy blazer", "corporate")):
            score -= 8.0
    elif kind == "funeral":
        if formality in {"smart", "formal"}:
            score += 2.0
        if any(k in text for k in ("black", "navy", "charcoal", "plain", "closed", "minimal")):
            score += 1.5
        if any(k in text for k in ("bright", "neon", "shiny", "sequin", "loud print", "party", "shorts", "slides", "gold")):
            score -= 8.0
    elif kind in {"office_meeting", "client_presentation"}:
        if formality in {"formal", "smart"}:
            score += 2.2
        if footwear in {"relaxed", "athletic"}:
            score -= 4.5
        if any(k in text for k in ("tropical", "vacation", "beach", "loud", "embroidered", "festive", "satin", "shiny", "glossy", "shorts")):
            score -= 5.0
    elif kind == "basketball_game":
        if any(k in text for k in ("tee", "jersey", "denim", "chino", "sneaker", "jacket", "cap")):
            score += 2.0
        if any(k in text for k in ("formal shirt", "button-down", "button down", "embroidered", "loafer", "blazer", "oxford", "tuxedo")):
            score -= 7.0
    elif kind in {"date", "date_night"}:
        if formality in {"smart", "formal"} or footwear in {"polished", "structured", "elevated casual"}:
            score += 1.8
        if footwear in {"relaxed", "athletic"}:
            score -= 3.0
        if any(k in text for k in ("shorts", "gym shorts", "running shorts", "board shorts", "sliders", "slippers", "flip flop", "flip-flop")):
            score -= 6.0
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


def _allows_relaxed_footwear(query: str) -> bool:
    q = _safe_text(query).lower()
    return any(
        key in q
        for key in (
            "beach",
            "pool",
            "resort",
            "vacation",
            "holiday",
            "lounge",
            "home",
            "sunday",
            "errand",
            "very casual",
        )
    )


def _hard_rejection_reason(card: Dict[str, Any], query: str) -> str:
    kind = _occasion_kind(query)
    if get_occasion_rule is not None and reject_board_for_occasion is not None:
        try:
            reason = reject_board_for_occasion(card, kind, get_occasion_rule(kind))
            if reason:
                return reason
        except Exception:
            pass
    footwear_mood = _footwear_mood(_item_by_role(card, "footwear"))
    smart_occasions = {
        "office", "office_meeting", "client_presentation", "client_dinner",
        "date", "date_night", "coffee_date", "first_date", "casual_dinner",
        "team_dinner", "wedding", "wedding_guest", "funeral",
    }
    if (
        kind in smart_occasions
        and footwear_mood == "relaxed sandal"
        and not _allows_relaxed_footwear(query)
    ):
        return "relaxed_footwear_blocked_for_occasion"
    if (
        kind == "daily"
        and _style_direction(query) == "smart_casual_office"
        and footwear_mood == "relaxed sandal"
        and not _allows_relaxed_footwear(query)
    ):
        return "relaxed_footwear_blocked_for_smart_daily"
    if kind in smart_occasions:
        for item in card.get("accessories", []) or []:
            if isinstance(item, dict) and _accessory_type(item) == "headwear" and not _allows_headwear(query):
                return "headwear_blocked_for_polished_occasion"
    text = _card_blob(card)
    if kind == "coffee_date":
        if any(k in text for k in ("wedding shirt", "tuxedo", "shiny gold", "sequin")):
            return "coffee_date_formal_or_festive_blocked"
        if "embroidered" in text and any(k in text for k in ("formal trouser", "black trousers", "black pants")):
            return "coffee_date_wedding_energy_blocked"
    if kind == "client_dinner" and any(k in text for k in ("slides", "slipper", "gym", "shorts", "loud print", "neon")):
        return "client_dinner_casual_or_loud_blocked"
    if kind == "beach_dinner" and any(
        k in text
        for k in (
            "office trouser", "black trousers", "black pants", "oxford",
            "derby", "formal leather", "heavy blazer", "corporate",
        )
    ):
        return "beach_dinner_office_weight_blocked"
    return ""


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
    "beach": [
        {"style_energy": "relaxed/creative", "archetype": "Hero Look", "title": "Coastal Ease"},
        {"style_energy": "minimal/monochrome", "archetype": "Safest Option", "title": "Resort Minimal"},
        {"style_energy": "elevated/casual", "archetype": "Relaxed Sharp", "title": "Sunset Casual"},
        {"style_energy": "polished/social", "archetype": "Elevated Option", "title": "Beach-to-Dinner"},
        {"style_energy": "expressive/statement", "archetype": "Creative Professional", "title": "Resort Statement"},
        {"style_energy": "safest/refined", "archetype": "Backup Option", "title": "Clean Coastal"},
    ],
    "brunch": [
        {"style_energy": "polished/social", "archetype": "Hero Look", "title": "Daylight Polish"},
        {"style_energy": "elevated/casual", "archetype": "Relaxed Sharp", "title": "Garden Easy"},
        {"style_energy": "relaxed/creative", "archetype": "Creative Professional", "title": "Soft Day Edit"},
        {"style_energy": "minimal/monochrome", "archetype": "Safest Option", "title": "Quiet Brunch"},
        {"style_energy": "safest/refined", "archetype": "Elevated Option", "title": "Hotel Sharp"},
        {"style_energy": "expressive/statement", "archetype": "Backup Option", "title": "Friends-in-Town"},
    ],
    "workout": [
        {"style_energy": "elevated/casual", "archetype": "Hero Look", "title": "Training Clean"},
        {"style_energy": "minimal/monochrome", "archetype": "Safest Option", "title": "Gym Minimal"},
        {"style_energy": "relaxed/creative", "archetype": "Relaxed Sharp", "title": "Mobility Reset"},
        {"style_energy": "polished/social", "archetype": "Elevated Option", "title": "Studio Ready"},
        {"style_energy": "expressive/statement", "archetype": "Creative Professional", "title": "Active Edge"},
        {"style_energy": "safest/refined", "archetype": "Backup Option", "title": "Clean Movement"},
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
_STYLE_DNA_TARGETS["office_meeting"] = _STYLE_DNA_TARGETS["office"]
_STYLE_DNA_TARGETS["client_presentation"] = _STYLE_DNA_TARGETS["office"]
_STYLE_DNA_TARGETS["client_dinner"] = _STYLE_DNA_TARGETS["office"]
_STYLE_DNA_TARGETS["date_night"] = _STYLE_DNA_TARGETS["date"]
_STYLE_DNA_TARGETS["first_date"] = _STYLE_DNA_TARGETS["date"]
_STYLE_DNA_TARGETS["casual_dinner"] = _STYLE_DNA_TARGETS["date"]
_STYLE_DNA_TARGETS["team_dinner"] = _STYLE_DNA_TARGETS["date"]
_STYLE_DNA_TARGETS["coffee_date"] = [
    {"style_energy": "elevated/casual", "archetype": "Hero Look", "title": "Relaxed Oxford"},
    {"style_energy": "relaxed/creative", "archetype": "Relaxed Sharp", "title": "Easy Conversation"},
    {"style_energy": "safest/refined", "archetype": "Elevated Option", "title": "Soft Coffee Polish"},
    {"style_energy": "minimal/monochrome", "archetype": "Safest Option", "title": "Approachable Edit"},
]
_STYLE_DNA_TARGETS["beach_dinner"] = _STYLE_DNA_TARGETS["beach"]
_STYLE_DNA_TARGETS["wedding_guest"] = _STYLE_DNA_TARGETS["wedding"]
_STYLE_DNA_TARGETS["funeral"] = _STYLE_DNA_TARGETS["wedding"]
_STYLE_DNA_TARGETS["basketball_game"] = _STYLE_DNA_TARGETS["casual"]


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


def _controlled_style_archetype_for_card(card: Dict[str, Any], query: str) -> str:
    """Map a wardrobe board onto the controlled fashion archetype library.

    Existing board roles still drive layout/diversity; this value is the
    user-visible style archetype.
    """
    try:
        from services.stylist_knowledge_service import select_archetypes

        names = [
            _safe_text(item.get("name") or item.get("label") or item.get("title"))
            for item in (card.get("items") or [])
            if isinstance(item, dict)
        ]
        anchor = {
            "name": " ".join(names[:4]),
            "category": _safe_text(card.get("style_energy") or ""),
            "color": _safe_text(card.get("palette_direction") or ""),
        }
        selected = select_archetypes(
            anchor=anchor,
            occasion=_occasion_kind(query),
            style_keywords=[
                _safe_text(card.get("style_energy")),
                _safe_text(card.get("silhouette_category")),
                _safe_text(card.get("palette_direction")),
            ],
            style_dna={},
            limit=1,
        )
        name = _safe_text((selected[0] if selected else {}).get("name"))
        return name or "Elevated Essentials"
    except Exception as exc:  # noqa: BLE001
        logger.warning("ahvi.board_archetype_map_failed err=%s", str(exc)[:120])
        return "Elevated Essentials"


def _identity_match_score(card: Dict[str, Any], style_identity: Dict[str, Any]) -> float:
    if not style_identity:
        return 0.0
    score = 0.0
    energy = _safe_text(card.get("style_energy") or _style_energy(card, _safe_text(card.get("_style_query"))))
    palette = _safe_text(card.get("palette_direction") or _palette_direction(card))
    silhouette = _safe_text(card.get("silhouette_category") or _silhouette_category(card))
    preferred_energies = set(style_identity.get("preferred_style_energies") or [])
    preferred_palettes = set(style_identity.get("preferred_palettes") or [])
    preferred_silhouettes = set(style_identity.get("preferred_silhouettes") or [])

    if energy and energy in preferred_energies:
        score += 2.5
    if any(palette and pref and pref.split()[0] in palette for pref in preferred_palettes):
        score += 0.8
    if silhouette and silhouette in preferred_silhouettes:
        score += 0.8

    accessory_policy = _safe_text(style_identity.get("accessory_policy")).lower()
    accessory_count = len([x for x in card.get("accessories", []) if isinstance(x, dict)])
    if accessory_policy == "restrained":
        score += 0.8 if accessory_count <= 1 else -1.2
    elif accessory_policy in {"statement", "textured"}:
        text = _card_blob(card)
        if any(k in text for k in ("print", "pattern", "texture", "ring", "bracelet", "watch")):
            score += 0.6
    elif accessory_policy in {"polished", "refined"} and accessory_count <= 2:
        score += 0.4
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

    kind = _occasion_kind(query)
    flags = _occasion_flags(query)
    if kind == "coffee_date":
        title_by_archetype = {
            "Hero Look": "Relaxed Oxford",
            "Safest Option": "Approachable Edit",
            "Elevated Option": "Soft Coffee Polish",
            "Relaxed Sharp": "Easy Conversation",
            "Creative Professional": "Soft Personality",
            "Backup Option": "Quiet Coffee Fit",
        }
        titles = ["Relaxed Oxford", "Easy Conversation", "Soft Coffee Polish", "Approachable Edit"]
    elif kind == "client_dinner":
        title_by_archetype = {
            "Hero Look": "Client Dinner Polish",
            "Safest Option": "Composed Evening",
            "Elevated Option": "Professional Social",
            "Relaxed Sharp": "After-Work Ease",
            "Creative Professional": "Measured Personality",
            "Backup Option": "Quiet Authority",
        }
        titles = ["Client Dinner Polish", "Professional Social", "Composed Evening"]
    elif flags["office"]:
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
    footwear_mood = _footwear_mood(_item_by_role(card, "footwear"))
    has_accessory = bool([x for x in card.get("accessories", []) if isinstance(x, dict)])

    kind = _occasion_kind(query)
    accessory_line = (
        "The accessory stays restrained so the outfit still reads intentional."
        if has_accessory
        else "The clean line works without extra accessories."
    )

    by_occasion = {
        "coffee_date": (
            "For a coffee date, this stays relaxed first and polished second. "
            f"{top} feels intentional without looking dressed up, while {footwear} keeps the mood approachable rather than formal."
        ),
        "client_dinner": (
            "For a client dinner, the priority is professional first and social second. "
            f"{top} keeps the room credible, while {footwear} and the cleaner base let the look relax after work."
        ),
        "first_date": (
            "This keeps the first impression easy and considered. The outfit has enough polish to feel intentional without turning the moment into a formal event."
        ),
        "casual_dinner": (
            "This reads dinner-aware without becoming date-night heavy. The pieces keep the shape clean while leaving the mood relaxed enough for a casual table."
        ),
        "beach_dinner": (
            "For a beach dinner, this keeps breathability in front and adds just enough evening polish. It avoids office weight while still looking intentional after sunset."
        ),
        "funeral": (
            "This keeps the attention on the moment, not the clothes. The darker, quieter choices feel respectful, and the closed footwear keeps the outfit composed."
        ),
        "office_meeting": (
            f"{top}, {bottom}, and {footwear} keep the meeting impression clear: polished, credible, and free of casual noise."
        ),
        "client_presentation": (
            "For a client presentation, this leads with credibility. The structure keeps attention on what you are saying, while the styling stays controlled rather than flashy."
        ),
        "basketball_game": (
            "This respects the game first: comfortable, casual, and easy to move in. It can still carry into a team dinner without feeling like officewear."
        ),
        "team_dinner": (
            "This keeps the social energy easy and team-friendly. It looks considered at the table without feeling dressed for a formal date."
        ),
        "office": (
            (
                f"{top}, {bottom}, and {footwear} keep the look client-ready, "
                "structured, and free of casual noise."
            )
            if "client" in str(query or "").lower()
            else (
                f"The {top}-and-{footwear} pairing keeps the look office-facing and polished, "
                f"while {bottom} gives it a cleaner business line."
            )
        ),
        "date": (
            f"{footwear} adds evening polish while {top} keeps the outfit relaxed enough for dinner."
        ),
        "party": (
            f"The statement energy comes through without overwhelming the base; {footwear} keeps it night-out ready."
        ),
        "beach": (
            "This keeps the outfit practical for water or heat while separating swim essentials from everyday casualwear."
        ),
        "workout": (
            "The active pieces prioritize movement, breathability, and easy transitions before and after training."
        ),
        "wedding": (
            f"The polished core keeps the outfit event-appropriate, and {footwear} gives it a formal finish."
        ),
        "travel": (
            "The pieces stay comfortable for movement while still looking considered on arrival."
        ),
    }
    capsule_copy = (
        "These pieces form a repeatable wardrobe foundation because they can be recombined across work, casual, and smart-casual settings."
        if "capsule" in str(query or "").lower()
        else ""
    )
    copy = {
        "color_harmony": capsule_copy or by_occasion.get(kind, f"{top} and {bottom} keep the palette balanced for the request."),
        "silhouette_balance": by_occasion.get(kind, f"{top} sets the proportion while {footwear} finishes the line as {footwear_mood}."),
        "texture_contrast": by_occasion.get(kind, f"{bottom} and {footwear} keep the texture mix grounded and wearable."),
        "occasion_alignment": by_occasion.get(kind, f"The outfit matches the brief without over-styling it. {accessory_line}"),
        "footwear_polish": by_occasion.get(kind, f"{footwear} sharpens the outfit into {footwear_mood}."),
        "smart_contrast": by_occasion.get(kind, f"The upper and base pieces create contrast while staying wearable."),
        "minimal_aesthetic": capsule_copy or by_occasion.get(kind, f"The outfit stays clean and focused. {accessory_line}"),
        "relaxed_tailoring": by_occasion.get(kind, f"The shape feels easy but still put-together for the occasion."),
    }
    tips = [
        f"Roll the sleeves once on {top} if you want the line less stiff.",
        f"Keep {top} untucked only if the hem sits above mid-hip.",
        "Keep accessories minimal here; the clean line is doing the work.",
        f"Let {footwear} stay visible; it is carrying the polish.",
        "If this moves into evening, keep the collar open and skip extra accessories.",
        "Do not add a cap here unless the brief is explicitly weekend or street.",
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


def _item_id(item: Dict[str, Any]) -> str:
    return _safe_text(
        item.get("id")
        or item.get("$id")
        or item.get("item_id")
        or item.get("itemId")
        or item.get("image_id")
        or item.get("name")
    )


def _display_item_name(item: Dict[str, Any]) -> str:
    return _safe_text(
        item.get("name")
        or item.get("title")
        or item.get("label")
        or item.get("subcategory")
        or item.get("sub_category")
        or item.get("category")
    )


def _item_color(item: Dict[str, Any]) -> str:
    raw = _safe_text(
        item.get("color")
        or item.get("color_name")
        or item.get("colorName")
        or item.get("primary_color")
        or item.get("primaryColor")
        or item.get("color_code")
        or item.get("colorCode")
    )
    if not raw:
        return ""
    text = raw.strip().lstrip("#")
    named = {
        "000000": "Black",
        "ffffff": "White",
        "808080": "Grey",
        "000080": "Navy",
        "8b4513": "Brown",
        "d2b48c": "Tan",
    }
    return named.get(text.lower(), raw.replace("_", " ").title())


def _hero_item(card: Dict[str, Any]) -> Dict[str, Any]:
    hero = (
        _item_by_role(card, "dress")
        or _item_by_role(card, "outerwear")
        or _item_by_role(card, "top")
    )
    if hero:
        return hero
    items = card.get("items")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return items[0]
    return {}


def _card_component_names(card: Dict[str, Any], *, limit: int = 6) -> List[str]:
    names: List[str] = []
    for item in card.get("items", []):
        if not isinstance(item, dict):
            continue
        name = _display_item_name(item)
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _card_colors(card: Dict[str, Any], *, limit: int = 5) -> List[str]:
    colors: List[str] = []
    for item in card.get("items", []):
        if not isinstance(item, dict):
            continue
        color = _item_color(item)
        if color and color.lower() not in {c.lower() for c in colors}:
            colors.append(color)
        if len(colors) >= limit:
            break
    return colors


def _hero_piece_reasoning(card: Dict[str, Any], query: str) -> str:
    hero = _hero_item(card)
    hero_name = _display_item_name(hero)
    if not hero_name:
        return ""
    style_direction = _style_direction(query).replace("_", " ")
    kind = _occasion_kind(query).replace("_", " ")
    return (
        f"{hero_name} anchors the {kind} brief with {style_direction} energy."
    )[:180]


def _composition_metadata(card: Dict[str, Any]) -> Dict[str, Any]:
    items = [item for item in card.get("items", []) if isinstance(item, dict)]
    hero = _item_by_role(card, "dress") or _item_by_role(card, "top") or (items[0] if items else {})
    anchor = _item_by_role(card, "footwear") or (items[-1] if items else {})
    profile = _diversity_profile(card, _safe_text(card.get("_style_query")))
    mode = "grid" if (
        profile.get("style_energy") == "minimal/monochrome"
        and len(items) <= 4
        and profile.get("accessory_mood") in {"no accessories", "clean minimal", "minimal watch"}
    ) else "stack"

    # Runtime layout spec. x/y are center positions in board space.
    stack_positions = {
        "hero": (0.40, 0.46, 3, 0),
        "anchor": (0.27, 0.74, 5, -5),
        "support_0": (0.63, 0.53, 1, 0),
        "support_1": (0.62, 0.34, 1, 0),
        "accent_0": (0.80, 0.22, 6, 0),
        "accent_1": (0.86, 0.34, 6, 4),
        "accent_2": (0.78, 0.45, 6, -3),
    }
    grid_positions = {
        "hero": (0.38, 0.42, 3, 0),
        "anchor": (0.32, 0.74, 5, 0),
        "support_0": (0.68, 0.42, 1, 0),
        "support_1": (0.68, 0.62, 1, 0),
        "accent_0": (0.72, 0.76, 6, 0),
        "accent_1": (0.84, 0.76, 6, 0),
        "accent_2": (0.78, 0.88, 6, 0),
    }
    positions = grid_positions if mode == "grid" else stack_positions

    composition_items: List[Dict[str, Any]] = []
    support_index = 0
    accent_index = 0
    for item in items[:6]:
        role = item_role(item)
        ident = _item_id(item)
        if not ident:
            continue
        if ident == _item_id(hero):
            comp_role = "hero"
            size = 0.38
            key = "hero"
        elif ident == _item_id(anchor) or role == "footwear":
            comp_role = "anchor"
            size = 0.22
            key = "anchor"
        elif role == "accessory":
            comp_role = "accent"
            size = 0.08
            key = f"accent_{min(accent_index, 2)}"
            accent_index += 1
        else:
            comp_role = "support"
            size = 0.16
            key = f"support_{min(support_index, 1)}"
            support_index += 1
        x, y, z, rotation = positions.get(key, (0.70, 0.24, 1, 0))
        composition_items.append(
            {
                "id": ident,
                "role": comp_role,
                "relative_size": size,
                "x": x,
                "y": y,
                "z": z,
                "rotation": rotation if mode == "stack" else 0,
            }
        )
    return {
        "composition_mode": mode,
        "hero_item_id": _item_id(hero),
        "anchor_item_id": _item_id(anchor),
        "composition_items": composition_items,
    }


def _office_has_strong_footwear(cards: List[Dict[str, Any]], query: str) -> bool:
    if not _occasion_flags(query)["office"]:
        return False
    return any(_footwear_formality_score(_item_by_role(card, "footwear"), query) > 0.5 for card in cards)


MAX_TOP_REUSE = 2
MAX_BOTTOM_REUSE = 1
MAX_FOOTWEAR_REUSE = 2
MAX_ACCESSORY_REUSE = 1
MAX_TOP_BOTTOM_REUSE = 1
RELAXED_TOP_BOTTOM_REUSE = 2


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


def _top_bottom_signature(card: Dict[str, Any]) -> str:
    if _role_key(card, "dress"):
        return _role_key(card, "dress")
    top = _role_key(card, "top")
    bottom = _role_key(card, "bottom")
    return "|".join(part for part in (top, bottom) if part)


def _selected_count(selected: List[Dict[str, Any]], role: str, value: str) -> int:
    if not value:
        return 0
    return sum(1 for selected_card in selected if _role_key(selected_card, role) == value)


def _select_diverse_cards(
    cards: List[Dict[str, Any]],
    query: str,
    limit: int,
    *,
    allow_core_variants: bool = False,
) -> List[Dict[str, Any]]:
    if not cards:
        return []
    unique_tops = {k for k in ((_role_key(card, "top") or _role_key(card, "dress")) for card in cards) if k}
    unique_bottoms = {k for k in (_role_key(card, "bottom") for card in cards) if k}
    unique_footwear = {k for k in (_role_key(card, "footwear") for card in cards) if k}
    unique_accessories = {key for card in cards for key in _accessory_keys(card)}
    unique_energies = {_style_energy(card, query) for card in cards}
    unique_bases = {k for k in (_top_bottom_signature(card) for card in cards) if k}
    enforce_top_limit = len(unique_tops) > 1
    enforce_bottom_limit = len(unique_bottoms) > 1
    enforce_footwear_limit = len(unique_footwear) > 1
    enforce_accessory_limit = len(unique_accessories) > 1
    enforce_base_variation = len(unique_bases) > 1
    strong_office_footwear_exists = _office_has_strong_footwear(cards, query)
    # Occasion Intelligence V2: selection may request a broad candidate pool,
    # but visible 3-board responses must not let one bottom or shoe dominate.
    max_top_reuse = MAX_TOP_REUSE
    max_bottom_reuse = 2
    max_footwear_reuse = MAX_FOOTWEAR_REUSE
    max_accessory_reuse = 2
    relaxed_base_reuse = RELAXED_TOP_BOTTOM_REUSE
    logger.info(
        "style_flow.finalizer_diversity_caps limit=%s pool=%s unique_tops=%s unique_bottoms=%s unique_footwear=%s unique_accessories=%s max_top_reuse=%s max_bottom_reuse=%s max_footwear_reuse=%s max_accessory_reuse=%s relaxed_base_reuse=%s",
        limit,
        len(cards),
        len(unique_tops),
        len(unique_bottoms),
        len(unique_footwear),
        len(unique_accessories),
        max_top_reuse,
        max_bottom_reuse,
        max_footwear_reuse,
        max_accessory_reuse,
        relaxed_base_reuse,
    )
    logger.info(
        "AHVI_DIVERSITY_CAP_APPLIED limit=%s pool=%s max_top_reuse=%s max_bottom_reuse=%s max_footwear_reuse=%s max_accessory_reuse=%s sparse=%s",
        limit,
        len(cards),
        max_top_reuse,
        max_bottom_reuse,
        max_footwear_reuse,
        max_accessory_reuse,
        {
            "top": len(unique_tops) <= 1,
            "bottom": len(unique_bottoms) <= 1,
            "footwear": len(unique_footwear) <= 1,
            "accessory": len(unique_accessories) <= 1,
        },
    )
    selected: List[Dict[str, Any]] = []
    selected_sigs: set[str] = set()
    selected_exact_sigs: set[str] = set()

    def _count_value(values: List[str], value: str) -> int:
        return sum(1 for x in values if x and x == value)

    def can_add(card: Dict[str, Any], *, strict: bool) -> bool:
        core = _safe_text(card.get("_style_core_signature"))
        exact = _safe_text(card.get("_style_signature"))
        if exact and exact in selected_exact_sigs:
            return False
        if core in selected_sigs and (strict or not allow_core_variants):
            return False
        top = _role_key(card, "top") or _role_key(card, "dress")
        bottom = _role_key(card, "bottom")
        footwear = _role_key(card, "footwear")
        base_sig = _top_bottom_signature(card)
        selected_tops = [_role_key(s, "top") or _role_key(s, "dress") for s in selected]
        selected_bases = [_top_bottom_signature(s) for s in selected]

        if enforce_base_variation and base_sig:
            # Strict mode treats a repeated top+bottom pair as the same outfit, even
            # when footwear/accessories change. Relaxed mode allows a second pass only
            # when the wardrobe is too small to fill the requested count.
            max_base = MAX_TOP_BOTTOM_REUSE if strict else relaxed_base_reuse
            if _count_value(selected_bases, base_sig) >= max_base:
                return False
            if strict and not allow_core_variants and base_sig in selected_bases:
                return False
        if enforce_top_limit and top and _count_value(selected_tops, top) >= max_top_reuse:
            return False
        if enforce_bottom_limit and bottom:
            max_bottom = MAX_BOTTOM_REUSE if strict and not allow_core_variants else max_bottom_reuse
            if _selected_count(selected, "bottom", bottom) >= max_bottom:
                return False
        if enforce_footwear_limit and footwear:
            if _selected_count(selected, "footwear", footwear) >= max_footwear_reuse:
                return False
        if enforce_accessory_limit:
            selected_accessories = [key for selected_card in selected for key in _accessory_keys(selected_card)]
            selected_types = [typ for selected_card in selected for typ in _accessory_types(selected_card)]
            if any(selected_accessories.count(key) >= max_accessory_reuse for key in _accessory_keys(card)):
                return False
            if strict and not allow_core_variants and any(selected_types.count(typ) >= MAX_ACCESSORY_REUSE for typ in _accessory_types(card)):
                return False
        if _occasion_flags(query)["office"] and strong_office_footwear_exists and len(selected) < 3:
            if _footwear_formality_score(_item_by_role(card, "footwear"), query) <= -3.0:
                return False
        if strict:
            if len(selected) < min(limit, len(unique_energies)):
                energy = _style_energy(card, query)
                if energy in {_style_energy(s, query) for s in selected}:
                    return False
            if len(selected) < min(3, limit):
                hero = top
                if hero and hero in selected_tops:
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
        selected_exact_sigs.add(_safe_text(picked.get("_style_signature")))
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
            selected_exact_sigs.add(_safe_text(picked.get("_style_signature")))
            remaining = [card for card in remaining if card is not picked]
    logger.info(
        "accessory_diversity.applied selected=%d requested=%d unique_accessories=%d max_reuse=%d",
        len(selected),
        limit,
        len(unique_accessories),
        max_accessory_reuse,
    )
    return selected[:limit]


def finalize_style_cards(
    cards: Any,
    *,
    query: str,
    style_identity: Optional[Dict[str, Any]] = None,
    occasion_interpretation: Optional[Dict[str, Any]] = None,
    exclude_signatures: Any = None,
    requested_count: Optional[int] = None,
    default_limit: int = 6,
    candidate_pool_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    excluded = {
        _safe_text(x).lower()
        for x in (exclude_signatures or [])
        if _safe_text(x)
    }
    # When the caller wants a broader candidate pool for downstream
    # validation/selection (e.g. style_brief.select_board_set needs diverse
    # heroes), `candidate_pool_size` lifts the standard 6-card cap.
    hard_cap = max(6, int(candidate_pool_size or 6))
    limit = max(1, min(hard_cap, requested_count or candidate_pool_size or default_limit))

    allow_core_variants = bool(
        (candidate_pool_size and int(candidate_pool_size or 0) > 6)
        or (requested_count and int(requested_count or 0) > 3)
    )
    canonical: List[Dict[str, Any]] = []
    seen: set[str] = set()
    seen_exact: set[str] = set()
    skipped_counts: Dict[str, int] = {}
    for idx, card in enumerate(cards or []):
        if not isinstance(card, dict):
            continue
        fixed = _canonicalize_card(card, idx)
        if not fixed:
            continue
        rejection_reason = _hard_rejection_reason(fixed, query)
        if rejection_reason:
            logger.info(
                "style_flow.card_rejected reason=%s query=%s items=%s",
                rejection_reason,
                query,
                [
                    _safe_text(item.get("name") or item.get("title"))
                    for item in fixed.get("items", [])
                    if isinstance(item, dict)
                ],
            )
            continue
        fixed = _curate_accessories_for_card(fixed, query)
        rejection_reason = _hard_rejection_reason(fixed, query)
        if rejection_reason:
            logger.info(
                "style_flow.card_rejected reason=%s query=%s items=%s",
                rejection_reason,
                query,
                [
                    _safe_text(item.get("name") or item.get("title"))
                    for item in fixed.get("items", [])
                    if isinstance(item, dict)
                ],
            )
            continue
        sig = _variant_card_signature(fixed) if allow_core_variants else card_signature(fixed)
        core_sig = core_card_signature(fixed) or sig
        logger.info(
            "style_flow.signature_variant hero=%s top=%s bottom=%s footwear=%s sig=%s",
            _safe_text(fixed.get("hero_item_id")),
            _role_key(fixed, "top") or _role_key(fixed, "dress"),
            _role_key(fixed, "bottom"),
            _role_key(fixed, "footwear"),
            sig,
        )
        logger.info(
            "style_flow.signature_debug sig=%s core_sig=%s title=%s items=%s",
            sig,
            core_sig,
            fixed.get("title"),
            [
                item.get("name") or item.get("title") or item.get("label")
                for item in fixed.get("items", [])
                if isinstance(item, dict)
            ],
        )
        if not sig:
            skipped_counts["missing_signature"] = skipped_counts.get("missing_signature", 0) + 1
            continue
        if sig in seen_exact:
            skipped_counts["duplicate_exact_signature"] = skipped_counts.get("duplicate_exact_signature", 0) + 1
            continue
        if not allow_core_variants and core_sig in seen:
            skipped_counts["duplicate_core_signature"] = skipped_counts.get("duplicate_core_signature", 0) + 1
            continue
        if sig in excluded or core_sig in excluded:
            skipped_counts["excluded_signature"] = skipped_counts.get("excluded_signature", 0) + 1
            continue
        fixed["_style_signature"] = sig
        fixed["_style_core_signature"] = core_sig
        fixed["_style_quality_score"] = _quality_score(fixed, query) + _occasion_fit_score(fixed, query)
        fixed["_style_query"] = query
        fixed["_style_identity"] = dict(style_identity or {})
        fixed["_style_identity_score"] = _identity_match_score(fixed, style_identity or {})
        fixed["_style_quality_score"] += float(fixed.get("_style_identity_score") or 0.0)
        seen.add(core_sig)
        seen_exact.add(sig)
        canonical.append(fixed)

    canonical.sort(key=lambda c: float(c.get("_style_quality_score") or 0.0), reverse=True)
    pre_diverse_count = len(canonical)
    canonical = _select_diverse_cards(
        canonical,
        query,
        limit,
        allow_core_variants=allow_core_variants,
    )
    logger.info(
        "style_flow.finalizer_funnel input=%d canonical=%d selected=%d limit=%d skipped=%s",
        len(cards or []),
        pre_diverse_count,
        len(canonical),
        limit,
        skipped_counts,
    )
    for idx, card in enumerate(canonical):
        board_role = _safe_text(card.get("_target_archetype")) or _ARCHETYPES[idx % len(_ARCHETYPES)]
        title = _safe_text(card.get("_target_title")) or _title_for(card, query, idx, board_role)
        profile = _diversity_profile(card, query)
        explanation = _explanation_for(card, query, idx)
        layout = _layout_metadata(card, board_role)
        composition = _composition_metadata(card)
        controlled_archetype = _controlled_style_archetype_for_card(card, query)
        hero_piece = _display_item_name(_hero_item(card))
        colors = _card_colors(card)
        components = _card_component_names(card)
        daily_notes = _daily_composition_notes(card, query)
        style_dna_alignment = _style_dna_alignment_text(style_identity or {}, controlled_archetype)
        card["title"] = title
        card["name"] = title
        card["subtitle"] = _STRATEGY_LABELS.get(card.get("look_strategy"), "") or board_role
        card["board_role"] = board_role
        card["style_archetype"] = controlled_archetype
        card["style_direction"] = _style_direction(query)
        card["hero_piece"] = hero_piece
        card["hero_piece_reasoning"] = _hero_piece_reasoning(card, query)
        card["colors"] = colors
        card["color_story"] = " \u2022 ".join(colors)
        card["outfit_components"] = components
        card["daily_composition_notes"] = daily_notes
        card["style_dna_alignment"] = style_dna_alignment
        card["persona_fit_reason"] = style_dna_alignment
        card["style_energy"] = profile.get("style_energy")
        card["silhouette_category"] = profile.get("silhouette_category")
        card["palette_direction"] = profile.get("palette")
        card["footwear_energy"] = profile.get("footwear_energy")
        card["formality_energy"] = profile.get("formality_energy")
        card["occasion_fit"] = round(_occasion_fit_score(card, query), 3)
        logger.info(
            "AHVI_OCCASION_SCORE_APPLIED archetype=%s title=%r occasion_fit=%.3f quality_score=%.3f",
            _occasion_kind(query),
            title,
            card["occasion_fit"],
            float(card.get("_style_quality_score") or 0.0),
        )
        logger.info(
            "AHVI_BOARD_ARCHETYPE_APPLIED style_archetype=%s board_role=%s title=%r",
            controlled_archetype,
            board_role,
            title,
        )
        card["diversity_profile"] = profile
        card["explanation_mode"] = explanation["explanation_mode"]
        card["why_it_works"] = explanation["why_it_works"]
        card["explanation"] = explanation["why_it_works"]
        card["reason"] = explanation["why_it_works"]
        card["style_reason"] = explanation["why_it_works"]
        card["styling_tip"] = _safe_text(explanation["styling_tip"])[:80]
        logger.info(
            "AHVI_EDITORIAL_REASONING_MODE archetype=%s mode=%s title=%r",
            _occasion_kind(query),
            card["explanation_mode"],
            title,
        )
        card["layout_preset"] = layout["layout_preset"]
        card["visual_hierarchy"] = layout["visual_hierarchy"]
        card["composition_notes"] = layout["composition_notes"]
        card["composition_mode"] = composition["composition_mode"]
        card["hero_item_id"] = composition["hero_item_id"]
        card["anchor_item_id"] = composition["anchor_item_id"]
        card["composition_items"] = composition["composition_items"]
        card["occasion_interpretation"] = occasion_interpretation or {}
        card["style_metadata"] = {
            "style_signature": card.get("_style_signature"),
            "core_style_signature": card.get("_style_core_signature"),
            "base_outfit_signature": _base_outfit_signature(card),
            "quality_score": round(float(card.get("_style_quality_score") or 0.0), 3),
            "coherence_score": round(_coherence_score(card), 3),
            "occasion_fit": card["occasion_fit"],
            "style_archetype": controlled_archetype,
            "board_role": board_role,
            "style_direction": card["style_direction"],
            "hero_piece": hero_piece,
            "colors": colors,
            "outfit_components": components,
            "daily_composition_notes": daily_notes,
            "style_dna_alignment": style_dna_alignment,
            "style_energy": card["style_energy"],
            "silhouette_category": card["silhouette_category"],
            "palette_direction": card["palette_direction"],
            "footwear_energy": card["footwear_energy"],
            "formality_energy": card["formality_energy"],
            "explanation_mode": card["explanation_mode"],
            "layout_preset": card["layout_preset"],
            "composition_mode": card["composition_mode"],
            "hero_item_id": card["hero_item_id"],
            "anchor_item_id": card["anchor_item_id"],
            "identity_match_score": round(float(card.get("_style_identity_score") or 0.0), 3),
            "occasion_interpretation": occasion_interpretation or {},
        }
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
    """Render multiple style boards in parallel.

    Each card's `style_board_renderer.render(board)` is CPU+I/O heavy
    (Pillow drawing + R2 upload), and previously they ran serially in a
    for-loop. For 6 cards that meant 6-18s of cumulative blocking. We now
    fan out across a ThreadPoolExecutor so the total wall time becomes
    roughly max(card_render) rather than sum(card_render).
    """
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

    style_dna = _dict(context.get("style_dna"))
    try:
        from brain.engines.style_board_engine import style_board_engine
        from brain.engines.style_board_renderer import style_board_renderer
    except Exception as exc:
        logger.warning("style board renderer unavailable: %s", exc)
        return []

    def _build_one_board(card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        items = card.get("items") if isinstance(card.get("items"), list) else []
        if not items:
            return None
        board: Dict[str, Any] = {}
        image_bytes: bytes = b""
        try:
            board = style_board_engine.build_board(
                {
                    "items": items,
                    "score": card.get("score"),
                    "style_archetype": card.get("style_archetype"),
                    "style_energy": card.get("style_energy"),
                    "layout_preset": card.get("layout_preset"),
                    "composition_mode": card.get("composition_mode"),
                    "hero_item_id": card.get("hero_item_id"),
                    "anchor_item_id": card.get("anchor_item_id"),
                    "composition_items": card.get("composition_items"),
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
                    "composition_mode": card.get("composition_mode"),
                    "hero_item_id": card.get("hero_item_id"),
                    "anchor_item_id": card.get("anchor_item_id"),
                    "composition_items": card.get("composition_items"),
                    "visual_hierarchy": card.get("visual_hierarchy"),
                    "composition_notes": card.get("composition_notes"),
                    "diversity_profile": card.get("diversity_profile"),
                }
            )
            image_bytes = style_board_renderer.render(board)
        except Exception as exc:
            logger.warning(
                "style board render failed user=%s error=%s",
                user_id, exc,
            )
            image_bytes = b""

        return {"card": card, "board": board, "image_bytes": image_bytes, "items": items}

    # Render boards in parallel. Workers capped at min(len(cards), 4) so we
    # don't oversubscribe CPU on small Cloud Run instances.
    from concurrent.futures import ThreadPoolExecutor
    rendered_results: List[Optional[Dict[str, Any]]] = []
    worker_cap = max(1, min(len(cards), 4))
    render_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=worker_cap) as pool:
        rendered_results = list(pool.map(_build_one_board, cards))
    render_ms = round((time.perf_counter() - render_started) * 1000, 1)
    logger.info(
        "ahvi.board_render.timing user=%s cards=%s workers=%s total_ms=%s",
        user_id, len(cards), worker_cap, render_ms,
    )

    rendered: List[Dict[str, Any]] = []
    for idx, prepared in enumerate(rendered_results):
        if prepared is None:
            continue
        card = prepared["card"]
        board = prepared["board"]
        image_bytes = prepared["image_bytes"]
        items = prepared["items"]

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
        _bp = board.get("board_payload") if isinstance(board.get("board_payload"), dict) else board
        logger.info(
            "AHVI_BOARD_RENDER_IMAGE_URL has_url=%s upload_error=%s",
            bool(image_url), bool(upload_error),
        )
        logger.info(
            "AHVI_BOARD_RENDER_LAYOUT_USED layout_preset=%s composition_mode=%s",
            _safe_text((_bp or {}).get("layout_preset")) or "default",
            _safe_text((_bp or {}).get("composition_mode")) or "default",
        )

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
                    "composition_mode": card.get("composition_mode"),
                    "hero_item_id": card.get("hero_item_id"),
                    "anchor_item_id": card.get("anchor_item_id"),
                    "composition_items": card.get("composition_items"),
                    "visual_hierarchy": card.get("visual_hierarchy"),
                    "composition_notes": card.get("composition_notes"),
                },
            }
        )
    return rendered


def _style_signature_hash(signatures: List[str]) -> str:
    return hashlib.sha1("|".join(signatures).encode("utf-8")).hexdigest() if signatures else ""


_STRUCTURED_OCCASION_TOKENS = (
    "office",
    "work",
    "formal",
    "interview",
    "business",
    "corporate",
    "professional",
    "meeting",
    "boardroom",
)

_COMPOSITION_PROPORTION_DEFAULTS = {
    "top": 1.0,
    "bottom": 0.95,
    "footwear": 0.55,
    "outerwear": 0.9,
    "accessory": 0.35,
    "dress": 1.0,
}


def _normalize_brief_role(raw_role: Any, name: str = "") -> str:
    """Map a free-form item role/name onto a canonical composition role."""
    role = str(raw_role or "").strip().lower()
    blob = f"{role} {str(name or '').lower()}"
    if any(t in blob for t in ("dress", "gown", "jumpsuit", "saree", "sari", "kurta set")):
        if "dress" in role or "gown" in blob or "jumpsuit" in blob:
            return "dress"
    if any(t in blob for t in ("jacket", "blazer", "coat", "outer", "overcoat", "cardigan", "shrug")):
        return "outerwear"
    if any(t in blob for t in ("shoe", "footwear", "boot", "sneaker", "heel", "sandal", "loafer", "flat")):
        return "footwear"
    if any(t in blob for t in ("bag", "belt", "watch", "jewel", "accessor", "scarf", "hat", "sunglass", "tie", "clutch", "purse")):
        return "accessory"
    if any(t in blob for t in ("pant", "trouser", "jean", "skirt", "short", "bottom", "chino", "legging")):
        return "bottom"
    if any(t in blob for t in ("top", "shirt", "tee", "t-shirt", "blouse", "sweater", "knit", "hoodie", "kurta", "kurti")):
        return "top"
    # Fall back to whatever canonical bucket the raw role already names.
    if role in _COMPOSITION_PROPORTION_DEFAULTS or role == "dress":
        return role
    return "top"


def _build_composition_brief(
    board_items: Optional[List[Dict[str, Any]]],
    occasion: Any = "",
) -> Dict[str, Any]:
    """v1 rule-based styling-intent brief for the frontend board renderer.

    Purely additive and defensive: never raises on missing/odd fields. Encodes
    hero role, layout mode, per-role size weights, layering order, mood, accent
    placement, and a best-effort palette anchor derived from item color fields.
    """
    try:
        items = [it for it in (board_items or []) if isinstance(it, dict)]

        roles: List[str] = []
        for it in items:
            role = _normalize_brief_role(
                it.get("role"),
                str(it.get("name") or it.get("title") or it.get("label") or ""),
            )
            roles.append(role)
        role_set = set(roles)

        has_dress = "dress" in role_set
        has_outerwear = "outerwear" in role_set
        has_top = "top" in role_set

        # hero_role
        if has_dress:
            hero_role = "dress"
        elif has_outerwear:
            hero_role = "outerwear"
        else:
            hero_role = "top"

        # layout_mode
        if has_dress:
            layout_mode = "dress_focused"
        elif has_outerwear:
            layout_mode = "layered"
        elif role_set and role_set.issubset({"accessory", "footwear"}):
            layout_mode = "accessory_heavy"
        else:
            layout_mode = "flat"

        # proportions: only for roles actually present (fallback to all defaults).
        if role_set:
            proportions = {
                r: _COMPOSITION_PROPORTION_DEFAULTS.get(r, 1.0)
                for r in role_set
                if r in _COMPOSITION_PROPORTION_DEFAULTS
            } or dict(_COMPOSITION_PROPORTION_DEFAULTS)
        else:
            proportions = dict(_COMPOSITION_PROPORTION_DEFAULTS)

        # layering: top sits in front of outerwear when both present.
        layering: List[Dict[str, str]] = []
        if has_outerwear and has_top:
            layering.append({"front": "top", "behind": "outerwear"})

        # mood
        occ = str(occasion or "").lower()
        mood = "structured" if any(t in occ for t in _STRUCTURED_OCCASION_TOKENS) else "relaxed"

        # accents
        accents = {"placement": "negative_space", "roles": ["accessory"]}

        # palette: best-effort from first item carrying a color/color_code field.
        anchor_color = None
        warmth = None
        for it in items:
            color = it.get("color") or it.get("color_name") or it.get("dominant_color")
            color_code = it.get("color_code") or it.get("hex") or it.get("color_hex")
            if color or color_code:
                anchor_color = str(color or color_code).strip() or None
                warmth = _color_warmth(str(color or color_code))
                break
        palette = (
            {"anchor_color": anchor_color, "warmth": warmth}
            if anchor_color is not None
            else None
        )

        return {
            "hero_role": hero_role,
            "layout_mode": layout_mode,
            "proportions": proportions,
            "layering": layering,
            "mood": mood,
            "accents": accents,
            "palette": palette,
        }
    except Exception:  # noqa: BLE001 - brief is purely additive, never break the board
        return {
            "hero_role": "top",
            "layout_mode": "flat",
            "proportions": dict(_COMPOSITION_PROPORTION_DEFAULTS),
            "layering": [],
            "mood": "relaxed",
            "accents": {"placement": "negative_space", "roles": ["accessory"]},
            "palette": None,
        }


def _color_warmth(value: str) -> Optional[str]:
    """Crude warm/cool/neutral classification from a color name or hex code."""
    v = str(value or "").strip().lower()
    if not v:
        return None
    warm_names = ("red", "orange", "yellow", "gold", "brown", "tan", "beige", "coral", "rust", "amber", "peach", "maroon")
    cool_names = ("blue", "green", "teal", "navy", "purple", "violet", "indigo", "mint", "cyan", "aqua")
    neutral_names = ("black", "white", "grey", "gray", "charcoal", "silver", "cream", "ivory")
    if any(n in v for n in warm_names):
        return "warm"
    if any(n in v for n in cool_names):
        return "cool"
    if any(n in v for n in neutral_names):
        return "neutral"
    # Hex path: #RRGGBB -> compare red vs blue channel.
    hexv = v.lstrip("#")
    if len(hexv) == 6:
        try:
            r = int(hexv[0:2], 16)
            g = int(hexv[2:4], 16)
            b = int(hexv[4:6], 16)
            if abs(r - b) <= 24 and abs(r - g) <= 24:
                return "neutral"
            return "warm" if r >= b else "cool"
        except ValueError:
            return None
    return None


def _board_metadata_summary(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for idx, card in enumerate(cards or []):
        if not isinstance(card, dict):
            continue
        metadata = _dict(card.get("style_metadata"))
        summary.append(
            {
                "index": idx,
                "title": card.get("title"),
                "style_archetype": card.get("style_archetype"),
                "style_direction": card.get("style_direction"),
                "style_energy": card.get("style_energy"),
                "silhouette_category": card.get("silhouette_category"),
                "palette_direction": card.get("palette_direction"),
                "footwear_energy": card.get("footwear_energy"),
                "formality_energy": card.get("formality_energy"),
                "explanation_mode": card.get("explanation_mode"),
                "style_signature": metadata.get("style_signature") or card.get("_style_signature") or card_signature(card),
                "core_style_signature": metadata.get("core_style_signature") or card.get("_style_core_signature") or core_card_signature(card),
                "base_outfit_signature": metadata.get("base_outfit_signature") or _base_outfit_signature(card),
                "quality_score": metadata.get("quality_score"),
                "coherence_score": metadata.get("coherence_score"),
                "occasion_fit": card.get("occasion_fit"),
                "layout_preset": card.get("layout_preset"),
                "composition_mode": card.get("composition_mode"),
                "hero_item_id": card.get("hero_item_id"),
                "anchor_item_id": card.get("anchor_item_id"),
                "occasion_interpretation": card.get("occasion_interpretation"),
            }
        )
    return summary


def finalize_style_response_payload(
    result: Dict[str, Any],
    *,
    user_id: str,
    query: str,
    wardrobe: Any = None,
    context: Optional[Dict[str, Any]] = None,
    include_base64: bool = False,
    upload_to_r2: bool = False,
    style_action: str = "",
    show_closest_option: bool = False,
    allow_closest_option: bool = False,
    closest: bool = False,
    exclude_style_signatures: Any = None,
    requested_board_count: Optional[int] = None,
    cache_bypass: bool = True,
    candidate_pool_size: Optional[int] = None,
) -> Dict[str, Any]:
    ctx = dict(context or {})
    style_identity = normalize_style_identity(_dict(ctx.get("user_profile")))
    ctx["style_identity"] = style_identity
    occasion_interpretation = _dict(ctx.get("occasion_interpretation")) or interpret_occasion(query, ctx)
    ctx["occasion_interpretation"] = occasion_interpretation
    resolved_brief = _safe_text(occasion_interpretation.get("resolved_brief"))
    finalizer_query = f"{query} {resolved_brief}".strip() if resolved_brief else query
    raw_cards = result.get("cards") if isinstance(result.get("cards"), list) else []
    raw_outfits = result.get("outfits") if isinstance(result.get("outfits"), list) else []
    # Pipeline returns both outfit dicts and their rendered card dicts. Feeding
    # both into the finalizer doubles the same looks and makes signature dedupe
    # collapse otherwise valid 5-6 board sets. Prefer cards because they carry
    # display metadata; fall back to outfits only when cards are absent.
    source_candidates = list(raw_cards or []) if raw_cards else list(raw_outfits or [])
    candidates: List[Dict[str, Any]] = []
    seen_candidate_sigs: set[str] = set()
    for candidate in source_candidates:
        if not isinstance(candidate, dict):
            continue
        metadata = candidate.get("style_metadata") if isinstance(candidate.get("style_metadata"), dict) else {}
        sig = _safe_text(
            candidate.get("_style_signature")
            or candidate.get("style_signature")
            or candidate.get("pipeline_style_signature")
            or metadata.get("style_signature")
            or metadata.get("pipeline_style_signature")
            or card_signature(candidate)
        )
        if sig and sig in seen_candidate_sigs:
            continue
        if sig:
            seen_candidate_sigs.add(sig)
        candidates.append(candidate)
    logger.info(
        "style_flow.candidate_merge raw_cards=%s raw_outfits=%s merged_unique=%s",
        len(raw_cards),
        len(raw_outfits),
        len(candidates),
    )

    finalize_started = time.perf_counter()
    cards = finalize_style_cards(
        candidates,
        query=finalizer_query,
        style_identity=style_identity,
        occasion_interpretation=occasion_interpretation,
        exclude_signatures=exclude_style_signatures,
        requested_count=requested_board_count if style_action in {"more_options", "more_looks", "next_best"} else None,
        candidate_pool_size=candidate_pool_size,
    )
    raw_candidate_count = len(candidates)
    pool_count = len(cards)
    normalized_occasion = _normalize_occasion_value(
        occasion_interpretation.get("occasion")
        or _dict(occasion_interpretation.get("board_generation_notes")).get("occasion_kind")
        or ctx.get("occasion"),
        query,
    )
    safety_action = (style_action or ctx.get("style_action") or "").strip().lower()
    closest_option_requested = (
        safety_action in {"show_closest_option", "closest_option", "show_closest"}
        or bool(show_closest_option)
        or bool(allow_closest_option)
        or bool(closest)
        or bool(ctx.get("show_closest_option"))
        or bool(ctx.get("allow_closest_option"))
        or bool(ctx.get("closest"))
    )
    logger.info(
        "style_flow.closest_requested user=%s occasion=%s closest_requested=%s style_action=%s",
        user_id,
        normalized_occasion,
        closest_option_requested,
        safety_action,
    )
    cards = apply_occasion_card_language(cards, normalized_occasion)
    wardrobe_items = wardrobe if isinstance(wardrobe, list) else []
    if not wardrobe_items:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_items = candidate.get("items")
            if isinstance(candidate_items, list):
                wardrobe_items.extend(item for item in candidate_items if isinstance(item, dict))
            else:
                wardrobe_items.append(candidate)
    slot_counts = _ahvi_slot_counts(wardrobe_items)
    logger.info(
        "style_board.slot_audit user_id=%s occasion=%s raw_wardrobe_count=%s top_count=%s bottom_count=%s footwear_count=%s accessory_count=%s outerwear_count=%s normalized_categories=%s",
        user_id,
        normalized_occasion,
        slot_counts.get("total"),
        slot_counts.get("top"),
        slot_counts.get("bottom"),
        slot_counts.get("footwear"),
        slot_counts.get("accessory"),
        slot_counts.get("outerwear"),
        {
            "top": slot_counts.get("top"),
            "bottom": slot_counts.get("bottom"),
            "footwear": slot_counts.get("footwear"),
            "accessory": slot_counts.get("accessory"),
            "outerwear": slot_counts.get("outerwear"),
        },
    )
    if not _ahvi_has_core_slots(slot_counts):
        logger.info(
            "ahvi.missing_core_slots occasion=%s slot_counts=%s",
            normalized_occasion,
            slot_counts,
        )
        return _ahvi_missing_core_slots_response(slot_counts)
    filtered_cards = []
    rejected_cards = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        sanitized_card, removed_items = _sanitize_office_board(card, normalized_occasion)
        if removed_items:
            kept_count = len(sanitized_card.get("items") or [])
            logger.info(
                "AHVI_BOARD_SANITIZED occasion=%s title=%s removed=%s kept=%d",
                normalized_occasion,
                card.get("title"),
                removed_items,
                kept_count,
            )
            if _has_minimum_board_slots(sanitized_card):
                card = sanitized_card
            else:
                rejected_cards.append((card, "missing_required_slots_after_sanitize"))
                logger.info(
                    "ahvi.board_rejected occasion=%s reason=%s title=%s",
                    normalized_occasion,
                    "missing_required_slots_after_sanitize",
                    card.get("title"),
                )
                logger.info(
                    "AHVI_OUTFIT_DROPPED_WEAK_MATCH occasion=%s reason=%s title=%s",
                    normalized_occasion,
                    "missing_required_slots_after_sanitize",
                    card.get("title"),
                )
                continue
        v2_reason = _metadata_v2_board_reject(card, normalized_occasion)
        if v2_reason:
            rejected, reason = True, v2_reason
        elif reject_quality_board_for_occasion is None:
            rejected, reason = False, ""
        else:
            try:
                rejected, reason = reject_quality_board_for_occasion(card, normalized_occasion)
            except Exception:
                rejected, reason = False, ""
        if rejected:
            rejected_cards.append((card, reason))
            logger.info(
                "ahvi.board_rejected occasion=%s reason=%s title=%s",
                normalized_occasion,
                reason,
                card.get("title"),
            )
            logger.info(
                "AHVI_OUTFIT_DROPPED_WEAK_MATCH occasion=%s reason=%s title=%s",
                normalized_occasion,
                reason,
                card.get("title"),
            )
            continue
        filtered_cards.append(card)
    logger.info(
        "AHVI_OUTFIT_VALIDATION_APPLIED occasion=%s input=%d accepted=%d rejected=%d",
        normalized_occasion,
        len(cards or []),
        len(filtered_cards),
        len(rejected_cards),
    )
    if not filtered_cards:
        closest_board = _ahvi_pick_closest_safe_board(cards, normalized_occasion)
        closest_rejected_reason = ""
        if closest_option_requested and not closest_board:
            closest_source = None
            if rejected_cards:
                closest_source, closest_rejected_reason = rejected_cards[0]
            else:
                for candidate in candidates:
                    if isinstance(candidate, dict) and isinstance(candidate.get("items"), list):
                        closest_source = candidate
                        closest_rejected_reason = "finalizer_filtered_all"
                        break
            if closest_source:
                closest_board = dict(closest_source)
        if closest_option_requested and closest_board:
            closest_board["title"] = "Closest wardrobe option"
            closest_board["badge"] = "CLOSEST OPTION"
            closest_board["occasion_label"] = "CLOSEST OPTION"
            closest_board["occasion"] = normalized_occasion
            closest_board["why_it_works"] = (
                "This is the closest wardrobe-based option I found, but it still "
                "needs refinement for this occasion. I would improve it with a "
                "linen/cotton shirt and sandals."
            )
            closest_board["explanation"] = closest_board["why_it_works"]
            closest_board.setdefault("score_meta", {})
            closest_board["score_meta"].update(
                {
                    "occasion_reject": True,
                    "closest_option_override": True,
                    "closest_option_reason": closest_rejected_reason,
                }
            )
            logger.info(
                "style_closest_option_from_rejected user_id=%s occasion=%s reason=%s",
                user_id,
                normalized_occasion,
                closest_rejected_reason,
            )
        if closest_option_requested and closest_board:
            filtered_cards = [closest_board]
        else:
            logger.info(
                "ahvi.missing_occasion_wardrobe occasion=%s rejected=%s slot_counts=%s closest=%s",
                normalized_occasion,
                len(rejected_cards),
                slot_counts,
                bool(closest_board),
            )
            return _ahvi_missing_occasion_response(
                occasion=normalized_occasion,
                slot_counts=slot_counts,
                closest_board=closest_board,
            )
    if filtered_cards:
        cards = apply_occasion_card_language(filtered_cards, normalized_occasion)
    public_cards = []
    for card in cards or []:
        if outfit_contains_private_wear(card):
            logger.info(
                "editorial_guard_rejected_private_wear user_id=%s occasion=%s title=%s",
                user_id,
                normalized_occasion,
                card.get("title") if isinstance(card, dict) else "",
            )
            logger.info(
                "editorial_guard_rejected_non_public_garment user_id=%s occasion=%s",
                user_id,
                normalized_occasion,
            )
            continue
        public_cards.append(card)
    if cards and not public_cards:
        msg = "I couldn't find enough appropriate public pieces for this occasion."
        logger.info(
            "editorial_guard_rejected_occasion_mismatch user_id=%s occasion=%s reason=non_public_only",
            user_id,
            normalized_occasion,
        )
        return {
            "success": True,
            "ok": True,
            "type": "missing_public_outfit",
            "intent": "style",
            "message": {"role": "assistant", "content": msg},
            "message_text": msg,
            "response": msg,
            "cards": [],
            "style_boards": [],
            "chips": [
                {"label": "Try a different occasion", "value": "Try a different occasion"},
                {"label": "Add public pieces", "value": "Add public pieces"},
            ],
            "data": {"outfits": [], "rendered_boards": [], "occasion": normalized_occasion},
            "meta": {"mode": "private_wear_guard", "occasion_interpretation": occasion_interpretation},
            "audio_job_id": "offline",
        }
    cards = public_cards
    finalize_ms = round((time.perf_counter() - finalize_started) * 1000, 2)
    ids = board_item_ids(cards)
    render_started = time.perf_counter()
    rendered = render_style_boards(
        cards,
        ctx,
        user_id=user_id,
        include_base64=include_base64,
        upload_to_r2=upload_to_r2,
    )
    render_ms = round((time.perf_counter() - render_started) * 1000, 2)
    signatures = [card.get("_style_signature") or card_signature(card) for card in cards]
    core_signatures = [card.get("_style_core_signature") or core_card_signature(card) for card in cards]
    style_signature = _style_signature_hash([s for s in signatures if s])
    primary_board_id = ids[0] if ids else ""

    logger.info(
        "style_flow.final_response user=%s cards=%d core_signatures=%s signatures=%s style_action=%s cache_bypass=%s finalize_ms=%s render_ms=%s profile_fields=%s",
        user_id,
        len(cards),
        core_signatures,
        signatures,
        style_action or "",
        bool(cache_bypass),
        finalize_ms,
        render_ms,
        style_identity.get("profile_fields_used") or [],
    )
    logger.info(
        "style_board.generate request_id=%s user_id=%s source=%s prompt=%s interpreted_occasion=%s wardrobe_item_count=%s selected_item_ids=%s accessory_count=%s cache_hit=%s fallback_used=%s fallback_reason=%s",
        _safe_text(ctx.get("request_id") or ctx.get("requestId")),
        user_id,
        _safe_text(ctx.get("source") or _dict(ctx.get("signals")).get("source") or "style_flow"),
        query,
        normalized_occasion,
        len(wardrobe_items),
        ids,
        sum(
            len([x for x in (card.get("accessories") or []) if isinstance(x, dict)])
            for card in cards
            if isinstance(card, dict)
        ),
        False,
        False,
        "",
    )
    logger.info(
        "ahvi.final_boards occasion=%s titles=%s badges=%s",
        normalized_occasion,
        [c.get("title") for c in cards[:6]],
        [c.get("badge") or c.get("occasion_label") for c in cards[:6]],
    )

    # ─────────────────────────────────────────────────────────────
    # FINAL STYLE SAFETY GATE
    # ─────────────────────────────────────────────────────────────
    # Suppress confident boards when the best card's occasion
    # compatibility falls below the per-occasion threshold. Two
    # exceptions:
    #   1. user explicitly asked to see the closest option
    #   2. user is asking for more variants of an already-confident set
    #      (style_action ∈ more_options/more_looks/next_best/show_closest)
    try:
        from brain.engines.style_scorer import occasion_confidence_threshold
    except Exception:
        occasion_confidence_threshold = None  # type: ignore

    occasion_scores = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        sm = c.get("score_meta") if isinstance(c.get("score_meta"), dict) else {}
        s = sm.get("occasion_compatibility_score")
        if isinstance(s, (int, float)):
            occasion_scores.append(float(s))
    best_occasion_score = max(occasion_scores) if occasion_scores else None
    threshold = (
        occasion_confidence_threshold(normalized_occasion)
        if occasion_confidence_threshold is not None
        else 0.68
    )

    weak_match = (
        bool(cards)
        and best_occasion_score is not None
        and best_occasion_score < threshold
        and not closest_option_requested
    )

    if weak_match:
        logger.info(
            "AHVI_OUTFIT_DROPPED_WEAK_MATCH user_id=%s occasion=%s best_score=%.2f threshold=%.2f cards=%s",
            user_id, normalized_occasion, best_occasion_score, threshold, len(cards),
        )
        logger.info(
            "AHVI_MISSING_PIECE_FROM_VALIDATION occasion=%s reason=below_confidence_threshold",
            normalized_occasion,
        )
        return _ahvi_missing_occasion_response(
            normalized_occasion,
            _ahvi_slot_counts(wardrobe_items),
        )

    if closest_option_requested:
        logger.info(
            "style_closest_option_requested user_id=%s occasion=%s best_score=%s",
            user_id, normalized_occasion,
            f"{best_occasion_score:.2f}" if best_occasion_score is not None else "n/a",
        )
        # Re-label cards so we never falsely advertise an ideal fit.
        for c in cards[:1]:
            if isinstance(c, dict):
                c["title"] = "Closest wardrobe option"
                c["why_it_works"] = (
                    "This is the closest match from your wardrobe, but it still "
                    "needs refinement for the occasion."
                )
                c["badge"] = (c.get("badge") or "CLOSEST OPTION")

    logger.info(
        "occasion_score_applied user_id=%s occasion=%s best_score=%s threshold=%.2f closest=%s",
        user_id, normalized_occasion,
        f"{best_occasion_score:.2f}" if best_occasion_score is not None else "n/a",
        threshold, closest_option_requested,
    )

    # Itemized board contract for the frontend renderer. Wardrobe cards carry
    # their pieces under `items` (name/role/image_url = transparent cutout) but
    # NOT `board_items`; the renderer reads `board_items` and only trusts an
    # image when it sees board_image_url / cutout_ready, so without this it
    # dropped every piece and rendered a checklist instead of a board.
    for _card in cards:
        if not isinstance(_card, dict):
            continue
        if _card.get("board_items"):
            # Pieces already present; still attach the additive composition brief.
            _card["composition_brief"] = _build_composition_brief(
                _card.get("board_items"),
                normalized_occasion,
            )
            continue
        _board_items: List[Dict[str, Any]] = []
        for _it in _card.get("items") or []:
            if not isinstance(_it, dict):
                continue
            _url = str(
                _it.get("image_url")
                or _it.get("board_image_url")
                or _it.get("normalized_url")
                or _it.get("masked_url")
                or ""
            ).strip()
            _name = str(_it.get("name") or _it.get("title") or _it.get("label") or "").strip()
            if not _url or not _name:
                continue
            _board_items.append(
                {
                    **_it,
                    "name": _name,
                    "role": _it.get("role") or "",
                    "image_url": _url,
                    "board_image_url": _url,
                    "board_status": "cutout_ready",
                }
            )
        if _board_items:
            _card["board_items"] = _board_items
        # Additive styling-intent brief for the frontend board renderer.
        _card["composition_brief"] = _build_composition_brief(
            _card.get("board_items") or _board_items,
            normalized_occasion,
        )

    data = {
        "outfits": cards,
        "visual_intelligence": visual_intelligence_from_outfit(raw_outfits[0]) if raw_outfits and isinstance(raw_outfits[0], dict) else {},
        "pipeline": _dict(result.get("pipeline")),
        "rendered_boards": rendered or cards,
        "board_item_ids": ids,
        "board_metadata": _board_metadata_summary(cards),
        "occasion_score": best_occasion_score,
        "occasion": normalized_occasion,
    }
    response_payload = {
        "cards": cards,
        "style_boards": cards,
        "board_ids": primary_board_id,
        "data": data,
        "meta": {
            "style_action": style_action or None,
            "style_signature": style_signature or None,
            "core_style_signatures": core_signatures,
            "board_metadata": _board_metadata_summary(cards),
            "board_count": len(cards),
            "has_more_style_options": bool(cards),
            "cache_bypass": bool(cache_bypass),
            "style_cache_bypass": bool(cache_bypass),
            "style_identity": style_identity,
            "occasion_interpretation": occasion_interpretation,
            "occasion_compatibility_score": best_occasion_score,
            "occasion_compatibility_threshold": threshold,
            "closest_option": closest_option_requested,
            "profile_sources": {
                "appwrite_profile": bool(style_identity.get("profile_fields_used")),
                "request_profile": bool(_dict(ctx.get("user_profile"))),
            },
            "timing_ms": {
                "style_finalization": finalize_ms,
                "style_rendering": render_ms,
            },
        },
    }
    try:
        from services.style_reasoning_engine import (
            _resolve_asset_gender,
            apply_gender_guard_to_final_payload,
        )

        profile_for_gender = _dict(ctx.get("user_profile"))
        profile_for_gender.update(
            {
                k: v
                for k, v in {
                    "gender": ctx.get("target_gender") or ctx.get("gender") or _dict(ctx.get("style_identity")).get("gender"),
                    "style_gender": ctx.get("style_gender"),
                    "target_gender": ctx.get("target_gender"),
                }.items()
                if v
            }
        )
        target_gender = _resolve_asset_gender(query=query, user_profile=profile_for_gender)
        guarded_payload, removed = apply_gender_guard_to_final_payload(
            response_payload,
            target_gender=target_gender,
            context=normalized_occasion or query,
        )
        return guarded_payload
    except Exception:
        logger.warning("style_flow.gender_guard_failed", exc_info=True)
        return response_payload


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
    show_closest_option: bool = False,
    allow_closest_option: bool = False,
    closest: bool = False,
    exclude_style_signatures: Any = None,
    requested_board_count: Optional[int] = None,
    cache_bypass: bool = True,
) -> Dict[str, Any]:
    from brain.outfit_pipeline import get_daily_outfits

    started = time.perf_counter()
    ctx = dict(context or {})
    ctx.setdefault("query", query)
    ctx.setdefault("user_id", user_id)
    if user_profile is not None:
        ctx.setdefault("user_profile", user_profile)
    ctx.setdefault("style_identity", normalize_style_identity(_dict(ctx.get("user_profile"))))

    # Deterministic non-apparel strip so boards never show chargers/passports/
    # water bottles even when the (slow) agent's avoid_items is skipped.
    wardrobe = _strip_non_apparel(wardrobe)

    # Style memory (wear + saved boards) -> scorer context. Neutral when no
    # data; never blocks styling.
    try:
        from services.style_memory_service import build_style_memory_context

        _mem = build_style_memory_context(user_id, wardrobe)
        for _k in (
            "recently_worn_ids", "underworn_ids", "saved_item_ids",
            "disliked_item_ids", "wear_counts",
        ):
            ctx.setdefault(_k, _mem.get(_k, []))
    except Exception:  # noqa: BLE001
        logger.debug("ahvi.style_memory_load_failed", exc_info=True)

    # Kick off the slow Vertex agent orchestration in the background NOW so its
    # ~20-40s overlaps with occasion interpretation + combo generation instead
    # of blocking in series. We wait only a short budget for it later; if it
    # isn't ready we proceed without it (the background run still caches its
    # result for the next request).
    _agent_future = None
    if _agent_start_orchestration is not None and _agent_style_enabled():
        try:
            _agent_future = _agent_start_orchestration(
                message=query,
                user_id=user_id,
                wardrobe_items=wardrobe if isinstance(wardrobe, list) else None,
                chips=list(_dict(context).get("chips") or []),
                weather=_dict(_dict(context).get("weather")),
                profile=_dict(_dict(context).get("user_profile") or user_profile),
                context=ctx,
            )
        except Exception:  # noqa: BLE001
            _agent_future = None

    occasion_interpretation = interpret_occasion(query, ctx)
    ctx["occasion_interpretation"] = occasion_interpretation
    normalized_occasion = _normalize_occasion_value(
        occasion_interpretation.get("occasion")
        or _dict(occasion_interpretation.get("board_generation_notes")).get("occasion_kind")
        or ctx.get("occasion"),
        query,
    )
    if normalized_occasion:
        ctx["occasion"] = normalized_occasion
    logger.info(
        "occasion_profile.selected occasion=%s query=%r wardrobe_count=%s",
        normalized_occasion or _occasion_kind(query),
        query,
        len(wardrobe) if isinstance(wardrobe, list) else 0,
    )

    # Capsule wardrobe short-circuit. Capsule requests don't want a
    # single styled board — they want a foundation + sample looks +
    # missing-slot guidance. Detect early and bypass the standard
    # outfit pipeline entirely.
    try:
        from brain.engines.capsule_engine import (
            build_capsule_response,
            looks_like_capsule_request,
        )

        if looks_like_capsule_request(query) or normalized_occasion == "capsule":
            logger.info(
                "ahvi.style_flow.capsule_request user=%s query=%r", user_id, query
            )
            logger.info("capsule_flow.selected user=%s query=%r", user_id, query)
            return build_capsule_response(
                user_id=user_id,
                wardrobe=wardrobe if isinstance(wardrobe, list) else [],
                query=query,
            )
    except Exception:
        logger.warning("ahvi.style_flow.capsule_route_failed", exc_info=True)

    # AHVI Style Orchestrator agent layer — produces structured intent from
    # the AHVI Style Orchestrator Agent / Gemini. Fully gated by the
    # ENABLE_AGENT_STYLE_ORCHESTRATOR env flag; safe defaults are merged when
    # the flag is off so downstream engines can still read the keys uniformly.
    try:
        agent_payload = None
        if _agent_future is not None:
            # The agent has been running in the background since the top of this
            # function (overlapping occasion interpretation). Wait only a short
            # budget; if it isn't ready we proceed without it (boards are still
            # clean via the deterministic non-apparel strip + occasion
            # guardrails) and the background run caches its result for next time.
            wait_budget = _agent_overlap_wait_seconds()
            try:
                agent_payload = _agent_future.result(timeout=wait_budget)
            except Exception:  # FuturesTimeoutError or run error
                agent_payload = None
                logger.info(
                    "ahvi.agent.overlap_proceed_without_agent waited=%.1fs", wait_budget
                )
        elif _agent_orchestrate_sync is not None and _agent_style_enabled():
            # Fallback: no background future (shouldn't normally happen).
            agent_payload = _agent_orchestrate_sync(
                message=query,
                user_id=user_id,
                wardrobe_items=wardrobe if isinstance(wardrobe, list) else None,
                chips=list(ctx.get("chips") or []),
                weather=_dict(ctx.get("weather")),
                profile=_dict(ctx.get("user_profile") or user_profile),
                context=ctx,
            )

        if agent_payload:
            _agent_merge_into_context(ctx, agent_payload)
            logger.info(
                "ahvi.agent.style_orchestration occasion=%s sub_intent=%s "
                "formality=%s style_direction=%s clarification_needed=%s confidence=%.2f",
                agent_payload.get("occasion"),
                agent_payload.get("sub_intent"),
                agent_payload.get("formality"),
                agent_payload.get("style_direction"),
                agent_payload.get("clarification_needed"),
                float(agent_payload.get("confidence") or 0.0),
            )
            if agent_payload.get("avoid_items"):
                logger.info(
                    "ahvi.agent.avoid_items_applied count=%d items=%s",
                    len(agent_payload.get("avoid_items") or []),
                    list(agent_payload.get("avoid_items") or [])[:10],
                )
            if agent_payload.get("required_slots"):
                logger.info(
                    "ahvi.agent.required_slots_applied slots=%s",
                    list(agent_payload.get("required_slots") or []),
                )
    except Exception:
        # Agent layer is best-effort; never let it break the legacy flow.
        logger.warning("ahvi.agent.style_orchestrator_merge_failed", exc_info=True)

    closest_requested = (
        str(style_action or ctx.get("style_action") or "").strip().lower()
        in {"show_closest_option", "closest_option", "show_closest"}
        or bool(show_closest_option)
        or bool(allow_closest_option)
        or bool(closest)
        or bool(ctx.get("show_closest_option"))
        or bool(ctx.get("allow_closest_option"))
        or bool(ctx.get("closest"))
    )
    if closest_requested:
        style_action = "show_closest_option"
        ctx["style_action"] = "show_closest_option"
        ctx["show_closest_option"] = True
        ctx["allow_closest_option"] = True
        ctx["closest"] = True
    logger.info(
        "ahvi.occasion_context occasion=%s query=%s",
        normalized_occasion or _occasion_kind(query),
        query,
    )
    logger.info(
        "style_flow.closest_requested user=%s occasion=%s closest_requested=%s",
        user_id,
        normalized_occasion,
        closest_requested,
    )
    _agent_payload = _dict(ctx.get("agent_orchestration"))
    _agent_blocks_clarification = bool(_agent_payload) and not _agent_payload.get(
        "clarification_needed", False
    )
    if (
        occasion_interpretation.get("ask_user")
        and not style_action
        and not requested_board_count
        and not _agent_blocks_clarification
    ):
        return _clarification_response(occasion_interpretation)

    # Broader candidate pool so style_brief.select_board_set has real
    # diversity to choose from. Final response is sliced to requested N
    # after brief validation/selection. This must be set before
    # get_daily_outfits(), otherwise the raw combo funnel has already
    # collapsed to the old 3-card/24-pool path.
    try:
        _final_target = int(requested_board_count or 6)
    except Exception:
        _final_target = 6
    _final_target = max(1, min(_final_target, 6))
    _pool_size = max(_final_target * 10, 48)
    ctx["requested_board_count"] = _final_target
    ctx["candidate_pool_size"] = _pool_size
    ctx["raw_candidate_target"] = _pool_size
    logger.info(
        "candidate_pool.expanded requested=%d pool=%d wardrobe_count=%s",
        _final_target,
        _pool_size,
        len(wardrobe) if isinstance(wardrobe, list) else 0,
    )

    candidate_started = time.perf_counter()
    result = get_daily_outfits(
        {
            "user_id": user_id,
            "wardrobe": wardrobe,
            "context": ctx,
        }
    )
    candidate_ms = round((time.perf_counter() - candidate_started) * 1000, 2)
    if not isinstance(result, dict):
        result = {}

    finalized = finalize_style_response_payload(
        result,
        user_id=user_id,
        query=query,
        wardrobe=wardrobe,
        context=ctx,
        include_base64=include_base64,
        upload_to_r2=upload_to_r2,
        style_action=style_action,
        show_closest_option=closest_requested,
        allow_closest_option=closest_requested,
        closest=closest_requested,
        exclude_style_signatures=exclude_style_signatures,
        requested_board_count=_final_target,
        cache_bypass=cache_bypass,
        candidate_pool_size=_pool_size,
    )
    logger.info(
        "style_candidates.generated raw_candidates=%d candidate_cards=%d requested=%d pool=%d",
        len(result.get("cards") or []) + len(result.get("outfits") or []),
        len(finalized.get("cards") or []),
        _final_target,
        _pool_size,
    )
    logger.info(
        "raw_candidates.generated raw_candidates=%d candidate_cards=%d requested=%d pool=%d",
        len(result.get("cards") or []) + len(result.get("outfits") or []),
        len(finalized.get("cards") or []),
        _final_target,
        _pool_size,
    )
    if finalized.get("type") in {
        "missing_core_wardrobe_slots",
        "missing_occasion_wardrobe",
        "weak_occasion_match",
    }:
        logger.info(
            "missing_slots.generated type=%s slots=%s shopping_gaps=%s",
            finalized.get("type"),
            finalized.get("missing_slots") or [],
            finalized.get("shopping_gaps") or [],
        )
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "style_flow.timing user=%s candidates_ms=%s total_ms=%s wardrobe_count=%s cards=%s fallback=%s",
            user_id,
            candidate_ms,
            total_ms,
            len(wardrobe) if isinstance(wardrobe, list) else 0,
            len(finalized.get("cards") or []),
            finalized.get("type"),
        )
        return {
            **finalized,
            "board": "style",
            "board_ids": "",
            "audio_job_id": "offline",
            "meta": {
                "analysis_source": "style_flow_service",
                "board_count": len(finalized.get("cards") or []),
                "occasion_interpretation": occasion_interpretation,
                "timing_ms": {
                    "candidate_generation": candidate_ms,
                    "style_flow_total": total_ms,
                },
            },
        }
    cards = finalized["cards"]

    # AHVI Style Brief enforcement: turn user intent into a brief, validate
    # every card against it, pick a diverse set, and reject wrong-occasion
    # boards rather than re-labelling them. Wrong-occasion board is worse
    # than no board.
    try:
        from brain.engines.style_brief import (
            build_brief,
            core_outfit_complete,
            safe_badge_for,
            safe_title_for,
            select_board_set,
            strip_forbidden_accessories,
        )

        _agent_payload = _dict(ctx.get("agent_orchestration"))
        brief = build_brief(
            query=query,
            router_occasion=normalized_occasion or ctx.get("occasion"),
            agent_payload=_agent_payload,
            weather=ctx.get("weather"),
        )
        ctx["style_brief"] = brief

        if cards:
            _candidate_pool_count = len(cards)

            # Accessory scrub BEFORE validation — a board with a great core
            # outfit shouldn't die because of one wrong accessory. Run on
            # the full pool so select_board_set has clean candidates.
            for card in cards:
                if not isinstance(card, dict):
                    continue
                _, _removed_accs = strip_forbidden_accessories(card, brief)
                # Drop card only if the scrub destroyed the core outfit
                # (very rare — accessory removal doesn't touch top/bottom/
                # footwear).
                if not core_outfit_complete(card):
                    card["_brief_drop_reason"] = "core_incomplete_after_accessory_scrub"
            cards = [c for c in cards if not (isinstance(c, dict) and c.get("_brief_drop_reason"))]

            chosen = select_board_set(cards, brief, max_n=_final_target)
            rejected = _candidate_pool_count - len(chosen)
            if rejected > 0:
                logger.info(
                    "style_board.rejected occasion=%s rejected=%d from=%d",
                    brief.get("occasion"), rejected, _candidate_pool_count,
                )
            logger.info(
                "style_candidates.generated candidate_cards=%d selected=%d requested=%d",
                _candidate_pool_count, len(chosen), _final_target,
            )
            logger.info(
                "valid_cards_after_guard candidate_cards=%d selected=%d requested=%d occasion=%s",
                _candidate_pool_count,
                len(chosen),
                _final_target,
                brief.get("occasion"),
            )
            for card in chosen:
                # Enforce occasion-safe badge + title only when current ones
                # conflict with the brief's allowed set.
                allowed_badges = set(brief.get("allowed_badges") or [])
                if allowed_badges and str(card.get("badge") or "").upper() not in allowed_badges:
                    card["badge"] = safe_badge_for(brief)
                allowed_titles = brief.get("allowed_titles") or []
                if allowed_titles and _safe_text(card.get("title")) not in set(allowed_titles):
                    card["title"] = safe_title_for(brief, chosen.index(card))
                    card["name"] = card["title"]
                logger.info(
                    "style_board.validated occasion=%s title=%r badge=%s set_role=%s",
                    brief.get("occasion"),
                    card.get("title"),
                    card.get("badge"),
                    card.get("set_role"),
                )
            cards = chosen
            finalized["cards"] = chosen
            finalized["style_boards"] = chosen

        if not cards and (brief.get("occasion") not in {"", "daily"}):
            logger.info(
                "style_fallback.intent_protected occasion=%s reason=no_board_passed_brief",
                brief.get("occasion"),
            )
    except Exception:
        logger.warning("style_brief.enforcement_failed", exc_info=True)

    if not cards:
        gap_kind = _safe_text(
            occasion_interpretation.get("occasion")
            or _dict(occasion_interpretation.get("board_generation_notes")).get("occasion_kind")
            or _occasion_kind(query)
        )
        gap_kind = _normalize_occasion_value(gap_kind, query) or gap_kind
        if gap_kind and gap_kind != "daily":
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "style_flow.timing user=%s candidates_ms=%s total_ms=%s wardrobe_count=%s cards=0 gap=%s",
                user_id,
                candidate_ms,
                total_ms,
                len(wardrobe) if isinstance(wardrobe, list) else 0,
                gap_kind,
            )
            logger.info("missing_slots.generated gap=%s cards=0", gap_kind)
            return _wardrobe_gap_response(
                query=query,
                wardrobe=wardrobe,
                interpretation=occasion_interpretation,
                finalized=finalized,
                result=result,
            )
    total_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
        "style_flow.timing user=%s candidates_ms=%s total_ms=%s wardrobe_count=%s cards=%s",
        user_id,
        candidate_ms,
        total_ms,
        len(wardrobe) if isinstance(wardrobe, list) else 0,
        len(cards),
    )
    occasion_label = str(query or normalized_occasion or "this").replace(" Â· ", " ").strip()
    kind = _occasion_kind(query or normalized_occasion)
    if kind == "coffee_date":
        style_intro = (
            "I’d keep this relaxed first and polished second. Here are three directions "
            "that feel intentional without getting too formal."
        )
    elif kind in {"office_meeting", "client_presentation", "client_dinner"}:
        style_intro = (
            "I’d lead with credibility, then keep the finish wearable. These options stay "
            "polished without turning stiff."
        )
    elif kind in {"beach_dinner", "travel"}:
        style_intro = (
            "I’d keep the base breathable and the finish evening-aware. These options avoid "
            "forcing formal pieces into a relaxed setting."
        )
    else:
        style_intro = (
            f"I built these around {occasion_label}. Each option has a clear hero piece, "
            "color story, and styling reason."
        )
    raw_response_message = (
        "This is the closest wardrobe-based option I found, but it still needs "
        f"refinement for {str(query or normalized_occasion).replace(' · ', ' ').strip()}. "
        "I would improve it with a linen/cotton shirt and sandals."
        if cards and closest_requested
        else result.get("context")
        or (
            style_intro if cards else "I couldn't build a reliable style board from your wardrobe yet."
        )
    )
    fallback_message = (
        style_intro if cards else "I couldn't build a reliable style board from your wardrobe yet."
    )
    response_message = _clean_editorial_copy(raw_response_message, fallback_message)
    response_payload = {
        "success": bool(cards),
        "message": response_message,
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
            "timing_ms": {
                **_dict(finalized.get("meta", {}).get("timing_ms")),
                "candidate_generation": candidate_ms,
                "style_flow_total": total_ms,
            },
        },
    }
    try:
        from services.style_reasoning_engine import (
            _resolve_asset_gender,
            apply_gender_guard_to_final_payload,
        )

        profile_for_gender = _dict(ctx.get("user_profile"))
        profile_for_gender.update(
            {
                k: v
                for k, v in {
                    "gender": ctx.get("target_gender") or ctx.get("gender") or _dict(ctx.get("style_identity")).get("gender"),
                    "style_gender": ctx.get("style_gender"),
                    "target_gender": ctx.get("target_gender"),
                }.items()
                if v
            }
        )
        target_gender = _resolve_asset_gender(query=query, user_profile=profile_for_gender)
        guarded_payload, removed = apply_gender_guard_to_final_payload(
            response_payload,
            target_gender=target_gender,
            context=normalized_occasion or query,
        )
        return guarded_payload
    except Exception:
        logger.warning("style_flow.gender_guard_failed", exc_info=True)
        return response_payload


# ============================================================================
# AHVI WARDROBE BOARD CURATION (Gemini-assisted, after deterministic candidates)
# Deterministic generation stays source of truth. Gemini only RANKS and
# RE-TITLES already-built candidate cards; it only ever sees / returns
# candidate IDs, so it can never invent wardrobe items. On failure we fall back
# to deterministic selection + diversity.
# ============================================================================

_LOOK_STRATEGY_ORDER = (
    "best_overall",
    "relaxed_alternative",
    "elevated_alternative",
    "personality_alternative",
)
_STRATEGY_LABELS = {
    "best_overall": "Best Overall",
    "relaxed": "Relaxed Alternative",
    "polished": "Elevated Alternative",
    "personality": "Personality Alternative",
    "relaxed_alternative": "Relaxed Alternative",
    "elevated_alternative": "Elevated Alternative",
    "personality_alternative": "Personality Alternative",
}
_FALLBACK_TITLES = (
    "Quietly Intentional",
    "Soft Polished Casual",
    "Relaxed Evening Smart",
    "Effortless Ease",
)


# Occasion guardrails: keyword patterns that disqualify an item for an
# occasion. Applied BEFORE Gemini curation so bad candidates never surface.
_OCCASION_REJECT_KEYWORDS = {
    "date": (("shiny", "gold formal"), ("wedding shirt",), ("sequin", "loud party"), ("tuxedo",)),
    "coffee": (("shiny", "gold formal"), ("wedding shirt",), ("sequin", "loud party"), ("tuxedo",)),
    "coffee_date": (("shiny",), ("wedding shirt",), ("sequin",), ("tuxedo",), ("embroidered", "formal trouser"), ("gold ring",)),
    "first_date": (("tuxedo",), ("gym",), ("slides",)),
    "casual_dinner": (("tuxedo",), ("gym",), ("slides",)),
    "client_dinner": (("gym",), ("slides",), ("shorts",), ("loud print",), ("neon",)),
    "beach_dinner": (("office trouser",), ("black trousers", "loafer"), ("oxford shoe",), ("heavy blazer",)),
    "wedding_guest": (("gym",), ("shorts",), ("slides",)),
    "casual outing": (("shiny", "gold formal"), ("tuxedo",), ("sequin",)),
    "basketball_game": (("formal shirt",), ("button-down",), ("button down",), ("embroidered",), ("loafer",), ("blazer",), ("oxford",)),
    "sports_game": (("formal shirt",), ("embroidered",), ("loafer",), ("blazer",)),
    "workout": (("formal shirt",), ("loafer",), ("blazer",), ("jeans",)),
    "client_presentation": (("athletic short",), ("gym short",), ("flip flop",), ("sequin", "loud party")),
    "office_meeting": (("athletic short",), ("flip flop",), ("sequin",)),
    "funeral": (("bright",), ("shiny",), ("sequin",), ("neon",), ("loud print",), ("floral print",)),
    "sensitive": (("bright",), ("shiny",), ("sequin",), ("neon",)),
    "beach_dinner": (("formal leather",), ("heavy blazer",), ("office trouser",), ("oxford shoe",)),
    "beach": (("heavy blazer",), ("office trouser",), ("formal leather",)),
}


def _guard_norm(value: Any) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _occasion_reject_keys(occasion: str) -> tuple:
    occ = _guard_norm(occasion).replace(" ", "_")
    if occ in _OCCASION_REJECT_KEYWORDS:
        return _OCCASION_REJECT_KEYWORDS[occ]
    # loose alias matching
    for key, rules in _OCCASION_REJECT_KEYWORDS.items():
        if key in occ or occ in key:
            return rules
    return tuple()


def _occasion_guardrail_reject(card: Dict[str, Any], occasion: str) -> str:
    """Return a reject reason if any item in the card violates the occasion
    guardrail, else empty string. Single source of truth: the existing
    occasion_style_rules engine; the keyword map is a fallback only when the
    engine has no rule for the occasion."""
    # Primary: occasion_style_rules engine.
    try:
        from brain.engines.occasion_style_rules import (
            get_occasion_rule,
            reject_board_for_occasion,
        )

        occ_norm = _guard_norm(occasion).replace(" ", "_")
        rule = get_occasion_rule(occ_norm)
        if rule:
            reason = reject_board_for_occasion(card, occ_norm, rule)
            if reason:
                return f"rules:{reason}"
    except Exception:  # noqa: BLE001
        pass

    # Safety net: keyword map. Runs when the engine has no specific rule for the
    # occasion (e.g. coffee) or didn't catch a known-bad pairing.
    rules = _occasion_reject_keys(occasion)
    if not rules:
        return ""
    blob = " ".join(
        _guard_norm(it.get("name")) + " " + _guard_norm(it.get("category")) + " " + _guard_norm(it.get("color")) + " " + _guard_norm(it.get("material"))
        for it in _card_items(card, include_slots=True)
    )
    for group in rules:
        if all(token in blob for token in group):
            return "keyword:" + "+".join(group)
    return ""


def _metadata_v2_board_reject(card: Dict[str, Any], occasion: str) -> str:
    if item_metadata_v2_reject_reason is None:
        return ""
    archetype = _safe_text(
        card.get("style_archetype")
        or _dict(card.get("style_metadata")).get("style_archetype")
    )
    for item in _card_items(card, include_slots=True):
        if not isinstance(item, dict):
            continue
        reason = item_metadata_v2_reject_reason(
            item,
            occasion=occasion,
            archetype=archetype,
        )
        if reason:
            logger.info(
                "AHVI_ITEM_OCCASION_REJECTED occasion=%s item=%s reason=%s source=style_flow_final",
                occasion,
                item.get("id") or item.get("$id") or item.get("name"),
                reason,
            )
            if archetype:
                logger.info(
                    "AHVI_ITEM_ARCHETYPE_REJECTED archetype=%s item=%s reason=%s",
                    archetype,
                    item.get("id") or item.get("$id") or item.get("name"),
                    reason,
                )
            return reason
    return ""


def _curation_item_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    tags = item.get("style_tags") or item.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    return {
        "id": item_key(item),
        "name": _safe_text(item.get("name") or item.get("label")),
        "category": item_role(item),
        "color": _safe_text(item.get("color") or item.get("colour")),
        "style_tags": [_safe_text(t) for t in tags][:4],
    }


def _candidate_summary(card: Dict[str, Any], idx: int) -> Dict[str, Any]:
    items = [_curation_item_summary(it) for it in _card_items(card, include_slots=True)]
    palette = card.get("palette") if isinstance(card.get("palette"), list) else []
    meta = card.get("meta") if isinstance(card.get("meta"), dict) else {}
    return {
        "candidate_id": "look_%02d" % (idx + 1),
        "items": items,
        "palette": [_safe_text(p) for p in palette][:5],
        "occasion_score": float(meta.get("occasion_score") or card.get("score") or 0.0),
        "comfort_score": float(meta.get("comfort_score") or 0.0),
        "formality_score": float(meta.get("formality_score") or 0.0),
        "notes": _safe_text(card.get("why_it_works") or card.get("explanation"))[:160],
    }


def _curation_prompt(summaries: List[Dict[str, Any]], reasoning: Dict[str, Any], occasion: str) -> str:
    import json as _json

    brief = {
        "occasion": occasion,
        "goal": _safe_text(reasoning.get("goal")),
        "impression": _safe_text(reasoning.get("impression")),
        "atmosphere": _safe_text(reasoning.get("atmosphere")),
        "confidence_strategy": _safe_text(reasoning.get("confidence_strategy")),
        "what_to_avoid": reasoning.get("what_to_avoid") or [],
    }
    schema = (
        '{"selected_candidates":[{"candidate_id":"look_01","rank":1,'
        '"title":"distinct evocative title, never \'Considered Look\'",'
        '"look_strategy":"best_overall | relaxed_alternative | elevated_alternative | personality_alternative",'
        '"why_it_works":"stylist reasoning about the social/contextual strategy, not item matching",'
        '"styling_tip":"one concrete wearable tip","what_to_avoid":["string"],'
        '"missing_piece":{"name":"","category":"","reason":"","unlocks":[]}}]}'
    )
    return (
        "You are AHVI's senior stylist curating a wardrobe board.\n\n"
        "The candidate outfits below were already built from the user's REAL "
        "wardrobe. You may ONLY choose from these candidate_id values. You may "
        "NOT invent items.\n\n"
        "Stylist brief:\n" + _json.dumps(brief, ensure_ascii=False) + "\n\n"
        "Candidates:\n" + _json.dumps(summaries, ensure_ascii=False) + "\n\n"
        "Select the best 3-4 candidates. Make them feel hand-picked, not "
        "mechanical. Assign one distinct look_strategy each in this priority: "
        "Look 1 = best_overall (safest, strongest), Look 2 = relaxed_alternative, "
        "Look 3 = elevated_alternative, Look 4 = personality_alternative.\n\n"
        "Return ONLY valid JSON:\n" + schema + "\n\n"
        "Rules:\n"
        "- candidate_id must come from the list above; invalid ids are ignored.\n"
        "- why_it_works explains the strategy, e.g. 'The embroidered shirt adds "
        "personality without becoming loud; the loafers give enough polish for a "
        "date while the relaxed shirt keeps it from feeling like office wear.'\n"
        "- Ban filler: 'balanced silhouette', 'color harmony', 'elevated "
        "aesthetic', 'perfect for'.\n"
        "- missing_piece is OPTIONAL; include only if it meaningfully improves "
        "the look, and it is a MISSING (not owned) item.\n"
    )


def _gemini_curate(summaries: List[Dict[str, Any]], reasoning: Dict[str, Any], occasion: str) -> List[Dict[str, Any]]:
    try:
        from services.llm_service import generate_text
        from services.ai_gateway import parse_json_object
    except Exception:  # noqa: BLE001
        return []
    try:
        raw = generate_text(
            _curation_prompt(summaries, reasoning, occasion),
            options={"temperature": 0.5, "max_output_tokens": 1400},
            signals={"context_mode": "board_curation"},
            usecase="style_reasoning",
        )
        parsed = parse_json_object(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ahvi.board_curation_gemini_failed err=%s", str(exc)[:140])
        return []
    rows = parsed.get("selected_candidates") if isinstance(parsed, dict) else None
    return rows if isinstance(rows, list) else []


def _signature_parts(card: Dict[str, Any]) -> Dict[str, str]:
    by_role: Dict[str, str] = {}
    for item in _card_items(card, include_slots=True):
        role = item_role(item)
        if role in {"top", "bottom", "dress", "footwear"} and role not in by_role:
            by_role[role] = item_key(item)
    return by_role


def _requested_explicit_roles(query: Any) -> List[str]:
    """Explicit garment roles named in the prompt; [] when nothing explicit."""
    try:
        from services.style_explicit_roles import extract_requested_roles

        return list(extract_requested_roles(query))
    except Exception:  # noqa: BLE001 - never break curation
        return []


def _explicit_role_pool(
    cards: List[Dict[str, Any]], candidate_pool: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Deterministic repair pool: caller-supplied wardrobe pool first, then every
    item already present on a sibling candidate board."""
    pool: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(item: Any) -> None:
        if not isinstance(item, dict):
            return
        key = item_key(item) or _safe_text(item.get("name"))
        if key and key in seen:
            return
        if key:
            seen.add(key)
        pool.append(item)

    for item in candidate_pool or []:
        _add(item)
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        for item in list(card.get("items") or []) + list(card.get("accessories") or []):
            _add(item)
    return pool


def _enforce_explicit_roles_on_cards(
    cards: List[Dict[str, Any]],
    *,
    query: str,
    occasion: str,
    candidate_pool: Optional[List[Dict[str, Any]]] = None,
    enforcement: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Repair-or-reject each card against the explicitly requested roles.

    Returns the surviving cards. When nothing survives, [enforcement] carries the
    typed failure detail so the router can answer missing_explicit_roles instead
    of silently shipping an incomplete board.
    """
    required = _requested_explicit_roles(query)
    if isinstance(enforcement, dict):
        enforcement.setdefault("requested_roles", list(required))
        enforcement.setdefault("required_explicit_roles", list(required))
        enforcement.setdefault("repair_attempted", False)
        enforcement.setdefault("status", "satisfied")
    if not required:
        return cards

    try:
        from services.style_explicit_roles import (
            board_explicit_roles,
            enforce_explicit_roles,
        )
    except Exception:  # noqa: BLE001 - never break curation
        return cards

    pool = _explicit_role_pool(cards, candidate_pool)
    available = sorted(set(board_explicit_roles(pool)))
    policy = ""
    for card in cards or []:
        if isinstance(card, dict) and card.get("source_policy"):
            policy = _safe_text(card.get("source_policy"))
            break

    kept: List[Dict[str, Any]] = []
    all_missing: set = set()
    for card in cards or []:
        fixed, status, missing = enforce_explicit_roles(
            card,
            required,
            candidate_pool=pool,
            source_policy=policy,
            occasion=occasion,
        )
        if fixed is None:
            all_missing.update(missing)
            logger.warning(
                "AHVI_EXPLICIT_ROLES_REJECTED missing=%s requested=%s title=%s",
                sorted(missing), required, _safe_text(card.get("title")) if isinstance(card, dict) else "",
            )
            continue
        if status == "repaired":
            logger.info(
                "AHVI_EXPLICIT_ROLES_REPAIRED requested=%s title=%s",
                required, _safe_text(fixed.get("title")),
            )
        kept.append(fixed)

    if isinstance(enforcement, dict):
        enforcement["repair_attempted"] = True
        enforcement["available_roles"] = available
        enforcement["missing_roles"] = sorted(all_missing) if not kept else []
        enforcement["status"] = "missing_explicit_roles" if not kept else "satisfied"

    logger.info(
        "AHVI_EXPLICIT_ROLES_ENFORCED requested=%s kept=%d dropped=%d available=%s",
        required, len(kept), len(cards or []) - len(kept), available,
    )
    return kept


def curate_wardrobe_boards(
    cards: List[Dict[str, Any]],
    *,
    query: str,
    occasion: str,
    reasoning: Optional[Dict[str, Any]] = None,
    wardrobe_count: int = 0,
    target: int = 4,
    candidate_pool: Optional[List[Dict[str, Any]]] = None,
    enforcement: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Rank + re-title deterministic candidate cards with Gemini, enforce
    diversity, attach stylist curation metadata. Never invents items."""
    valid = [c for c in cards if isinstance(c, dict)]
    if not valid:
        return cards
    reasoning = reasoning if isinstance(reasoning, dict) else {}

    # Occasion guardrails: drop bad candidates (shiny gold formal shirt for a
    # coffee date, loafers for a basketball game, etc.) before curation. Keep at
    # least one card so we never return an empty board.
    guard_reasons: List[str] = []
    kept_guard = []
    for c in valid:
        reason = _occasion_guardrail_reject(c, occasion)
        if reason:
            guard_reasons.append(reason)
        else:
            kept_guard.append(c)
    if kept_guard:
        valid = kept_guard

    # Explicit requested-role contract. Roles the user named ("... with a dress,
    # shoes, and a bag") are HARD constraints. Enforced BEFORE Gemini curation so
    # every title / why / styling note is written against the FINAL item set.
    valid = _enforce_explicit_roles_on_cards(
        valid,
        query=query,
        occasion=occasion,
        candidate_pool=candidate_pool,
        enforcement=enforcement,
    )
    if not valid:
        return []

    _rule_source = (
        "occasion_style_rules" if any(r.startswith("rules:") for r in guard_reasons)
        else ("keyword_fallback" if guard_reasons else "none")
    )
    logger.info(
        "AHVI_OCCASION_GUARDRAILS_APPLIED occasion=%s rejected_count=%d reasons=%s",
        occasion, len(guard_reasons), guard_reasons[:6],
    )
    logger.info("AHVI_OCCASION_RULES_SOURCE occasion=%s source=%s", occasion, _rule_source)

    target = max(3, min(target, len(valid)))
    wardrobe_limited = bool(wardrobe_count) and wardrobe_count < 6

    summaries = [_candidate_summary(c, i) for i, c in enumerate(valid)]
    id_to_card = {"look_%02d" % (i + 1): valid[i] for i in range(len(valid))}
    logger.info(
        "AHVI_BOARD_CANDIDATES_GENERATED candidate_count=%d occasion=%s wardrobe_item_count=%d",
        len(valid), occasion, wardrobe_count,
    )

    gemini_rows = _gemini_curate(summaries, reasoning, occasion)
    ordered: List[Any] = []
    seen_ids: set = set()
    for row in sorted(gemini_rows, key=lambda r: int(r.get("rank") or 99)):
        cid = _safe_text(row.get("candidate_id"))
        if cid in id_to_card and cid not in seen_ids:
            seen_ids.add(cid)
            ordered.append((id_to_card[cid], row))
    for cid, card in id_to_card.items():
        if len(ordered) >= target:
            break
        if cid not in seen_ids:
            seen_ids.add(cid)
            ordered.append((card, {}))

    logger.info(
        "AHVI_BOARD_GEMINI_RANKED selected_ids=%s titles=%s strategies=%s",
        [r.get("candidate_id") for _, r in ordered if r],
        [_safe_text(r.get("title")) for _, r in ordered if r],
        [_safe_text(r.get("look_strategy")) for _, r in ordered if r],
    )

    top_counts: Dict[str, int] = {}
    foot_counts: Dict[str, int] = {}
    seen_core: set = set()
    rejected = 0
    curated: List[Dict[str, Any]] = []
    for card, row in ordered:
        sig = core_card_signature(card)
        parts = _signature_parts(card)
        top_k, foot_k = parts.get("top", ""), parts.get("footwear", "")
        dup_combo = bool(sig) and sig in seen_core
        over_top = bool(top_k) and top_counts.get(top_k, 0) >= 2
        over_foot = bool(foot_k) and foot_counts.get(foot_k, 0) >= 2
        if (dup_combo or over_top or over_foot) and not wardrobe_limited:
            rejected += 1
            continue
        seen_core.add(sig)
        if top_k:
            top_counts[top_k] = top_counts.get(top_k, 0) + 1
        if foot_k:
            foot_counts[foot_k] = foot_counts.get(foot_k, 0) + 1
        curated.append(_apply_curation(card, row, len(curated), occasion=occasion))
        if len(curated) >= target:
            break

    if len(curated) < min(3, len(valid)):
        for card, row in ordered:
            if any(cc is card for cc in curated):
                continue
            curated.append(_apply_curation(card, row, len(curated), occasion=occasion))
            if len(curated) >= min(3, len(valid)):
                break

    repeated_tops = sum(1 for v in top_counts.values() if v > 1)
    repeated_foot = sum(1 for v in foot_counts.values() if v > 1)
    diversity_meta = {
        "repeated_tops": repeated_tops,
        "repeated_footwear": repeated_foot,
        "wardrobe_limited": bool(wardrobe_limited),
        "rejected_duplicates": rejected,
    }
    logger.info(
        "AHVI_BOARD_DIVERSITY_APPLIED repeated_tops=%d repeated_footwear=%d wardrobe_limited=%s rejected_duplicates=%d",
        repeated_tops, repeated_foot, bool(wardrobe_limited), rejected,
    )
    for c in curated:
        c["diversity_meta"] = diversity_meta
    # Final completeness gate: never emit a board labelled "complete outfit"
    # that is missing a required slot (top+bottom+footwear, or one-piece+footwear).
    try:
        from brain.engines.outfit_quality_guard import is_complete_board as _is_complete
        complete = [c for c in curated if _is_complete(c.get("items"))]
        if len(complete) != len(curated):
            logger.warning(
                "AHVI_BOARD_INCOMPLETE_DROPPED count=%d", len(curated) - len(complete)
            )
            curated = complete
    except Exception:  # noqa: BLE001 - enforcement must never break curation
        pass
    # Explicit roles must still be present after curation / accessory trimming.
    required_roles = _requested_explicit_roles(query)
    if required_roles:
        try:
            from services.style_explicit_roles import missing_explicit_roles

            kept = []
            for card in curated:
                gap = missing_explicit_roles(card.get("items"), required_roles)
                if gap:
                    logger.warning(
                        "AHVI_BOARD_EXPLICIT_ROLE_DROPPED missing=%s title=%s",
                        gap, _safe_text(card.get("title")),
                    )
                    continue
                kept.append(card)
            curated = kept
            if isinstance(enforcement, dict) and not curated:
                enforcement["status"] = "missing_explicit_roles"
                enforcement["repair_attempted"] = True
        except Exception:  # noqa: BLE001
            pass
    logger.info(
        "AHVI_BOARD_FINAL_LOOKS selected_count=%d titles=%s strategies=%s",
        len(curated),
        [_safe_text(c.get("title")) for c in curated],
        [_safe_text(c.get("look_strategy")) for c in curated],
    )
    return curated


def _is_weak_why(text: str) -> bool:
    """Generic / templated curation copy we want to replace with the
    storyteller's occasion-aware editorial line."""
    t = _guard_norm(text)
    if not t or len(t.split()) < 5:
        return True
    weak_markers = (
        "sets the structure while",
        "keeps it easy",
        "finishes the line",
        "sets the proportion",
        "carries the personality",
        "clean and intentional",
        "keeps one clear focal point",
    )
    return any(m in t for m in weak_markers)


def _apply_curation(
    card: Dict[str, Any], row: Dict[str, Any], index: int, occasion: str = ""
) -> Dict[str, Any]:
    out = dict(card)
    strategy = _safe_text(row.get("look_strategy")).lower()
    strategy = {
        "relaxed": "relaxed_alternative",
        "polished": "elevated_alternative",
        "personality": "personality_alternative",
    }.get(strategy, strategy)
    if strategy not in _LOOK_STRATEGY_ORDER:
        strategy = _LOOK_STRATEGY_ORDER[min(index, len(_LOOK_STRATEGY_ORDER) - 1)]
    title = _clean_editorial_copy(row.get("title"), "")
    why = _clean_editorial_copy(row.get("why_it_works"), _safe_text(out.get("why_it_works")))

    # P3: board_storyteller fallback when Gemini title/why is missing or weak.
    if (not title or title.lower() == "considered look") or _is_weak_why(why):
        try:
            from brain.response.board_storyteller import fallback_title_and_why

            st_title, st_why = fallback_title_and_why(occasion, index)
            if (not title or title.lower() == "considered look") and st_title:
                title = st_title
            if _is_weak_why(why) and st_why:
                why = st_why
            logger.info(
                "AHVI_BOARD_STORYTELLER_FALLBACK_USED occasion=%s index=%d title=%r",
                occasion, index, title,
            )
        except Exception:  # noqa: BLE001
            pass
    if not title or title.lower() == "considered look":
        title = _FALLBACK_TITLES[min(index, len(_FALLBACK_TITLES) - 1)]
    tip = _clean_editorial_copy(row.get("styling_tip"), _safe_text(out.get("styling_tip")))

    out["title"] = title
    out["look_strategy"] = strategy
    out["strategy_label"] = _STRATEGY_LABELS.get(strategy, strategy.title())
    logger.info(
        "AHVI_BOARD_STRATEGY_ASSIGNED occasion=%s index=%d strategy=%s title=%r",
        occasion,
        index,
        strategy,
        title,
    )
    if why:
        out["why_it_works"] = why
    if tip:
        out["styling_tip"] = tip
    avoid = row.get("what_to_avoid")
    if isinstance(avoid, list) and avoid:
        out["what_to_avoid"] = [_safe_text(a) for a in avoid][:4]
    mp = row.get("missing_piece")
    if isinstance(mp, dict) and _safe_text(mp.get("name")):
        out["missing_piece"] = {
            "name": _safe_text(mp.get("name")),
            "category": _safe_text(mp.get("category")),
            "reason": _safe_text(mp.get("reason")),
            "unlocks": [_safe_text(u) for u in (mp.get("unlocks") or []) if _safe_text(u)][:6],
        }
    out["novelty_score"] = round(1.0 - (index * 0.18), 2)
    out["repeat_penalty"] = 0.0
    if isinstance(out.get("story"), dict):
        out["story"] = {**out["story"], "headline": title}
    return out
