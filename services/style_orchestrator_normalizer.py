"""Normalize raw AHVI Style Orchestrator agent JSON into a canonical brief.

`normalize_style_orchestrator_brief(raw, *, has_wardrobe_context)` turns the
agent's loose JSON into the backend-safe `CanonicalStyleBrief` dict.
`validate_canonical_brief(brief)` decides whether the brief is trustworthy or
the caller should fall back to the legacy/deterministic pipeline.

Never raises on malformed input — bad payloads yield a low-confidence brief that
validation rejects, so the user flow always continues.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from services.style_orchestrator_schema import (
    ACCESSORY_SLOTS,
    ALLOWED_REQUIRED_SLOTS,
    OPTIONAL_SLOT_VOCAB,
    WARDROBE_USAGE_VALUES,
    CanonicalStyleBrief,
    Formality,
    PaletteDirection,
    StyleDirection,
)


def _snake(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [value]
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        return []
    return [str(p).strip() for p in parts if str(p).strip()]


# ---- formality --------------------------------------------------------------
_FORMALITY_SCORE = {
    "lounge": 1, "sleep": 1, "sleepwear": 1, "home": 1, "loungewear": 1,
    "casual": 2, "relaxed": 2, "everyday": 2,
    "smart_casual": 3, "elevated_casual": 3, "smart": 3,
    "business_casual": 4, "business_professional": 4, "professional": 4, "business": 4,
    "formal": 5, "black_tie": 5, "wedding_formal": 5, "cocktail": 4,
}


def _formality(value: Any) -> Formality:
    if isinstance(value, dict):
        label = str(value.get("label") or "").strip() or "casual"
        try:
            score = int(value.get("score"))
        except Exception:
            score = _FORMALITY_SCORE.get(_snake(label), 2)
        score = max(1, min(5, score))
        return Formality(label=label, score=score)
    label = str(value or "").strip() or "casual"
    score = _FORMALITY_SCORE.get(_snake(label), 2)
    return Formality(label=label, score=score)


# ---- style / palette (primary + alternates) ---------------------------------
def _primary_alternates(value: Any) -> Tuple[str, List[str]]:
    if isinstance(value, dict):
        primary = str(value.get("primary") or "").strip()
        alternates = _str_list(value.get("alternates"))
        return primary, alternates
    items = _str_list(value)
    if not items:
        return "", []
    return items[0], items[1:]


def _palette(value: Any) -> PaletteDirection:
    primary, alternates = _primary_alternates(value)
    return PaletteDirection(
        primary=_snake(primary) if primary else "",
        alternates=[_snake(a) for a in alternates],
    )


# ---- wardrobe usage ---------------------------------------------------------
def _wardrobe_usage(value: Any, *, has_wardrobe_context: bool) -> str:
    if isinstance(value, bool):
        return "owned_first" if value else "inspiration_only"
    text = _snake(value)
    if text in WARDROBE_USAGE_VALUES:
        return text
    if text in {"owned", "preferred", "wardrobe", "owned_preferred"}:
        return "owned_first"
    if text in {"inspiration", "suggested", "new"}:
        return "inspiration_only"
    if text in {"mixed", "both"}:
        return "mixed_owned_and_suggested"
    return "owned_first" if has_wardrobe_context else "inspiration_only"


# ---- avoid items ------------------------------------------------------------
_AVOID_MAP = {
    "athletic_sneakers": "athletic_sneakers",
    "sneakers": "athletic_sneakers",
    "distressed_denim": "distressed_denim",
    "ripped_jeans": "distressed_denim",
    "graphic_t_shirts": "graphic_tshirts",
    "graphic_tshirts": "graphic_tshirts",
    "graphic_tees": "graphic_tshirts",
    "overtly_casual_sandals": "casual_sandals",
    "casual_sandals": "casual_sandals",
    "flip_flops": "casual_sandals",
    "excessive_transparency": "transparent_items",
    "transparent_items": "transparent_items",
    "sheer": "transparent_items",
    "excessive_or_distracting_jewelry": "distracting_jewelry",
    "distracting_jewelry": "distracting_jewelry",
    "excessive_jewelry": "distracting_jewelry",
}


def _avoid_items(value: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw_item in _str_list(value):
        key = _snake(raw_item)
        norm = _AVOID_MAP.get(key, key)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# ---- slots ------------------------------------------------------------------
_PRIMARY_SLOT_ALIASES = {
    "top": "top", "shirt": "top", "tops": "top",
    "bottom": "bottom", "bottoms": "bottom", "pant": "bottom", "pants": "bottom",
    "footwear": "footwear", "shoes": "footwear", "shoe": "footwear",
    "dress": "dress", "gown": "dress",
    "one_piece": "one_piece", "jumpsuit": "one_piece",
}
_OPTIONAL_SLOT_ALIASES = {
    "outerwear": "outerwear", "jacket": "outerwear", "blazer": "outerwear", "coat": "outerwear",
    "watch": "watch", "belt": "belt", "bag": "bag",
    "jewelry": "minimal_jewelry", "minimal_jewelry": "minimal_jewelry",
    "minimal_jewellery": "minimal_jewelry", "jewellery": "minimal_jewelry",
}

# Occasions that must stay top+bottom+footwear (outerwear optional) unless the
# user explicitly asked for a layer.
_CORE_THREE_OCCASIONS = {
    "client_meeting", "office_meeting", "office", "work", "business_meeting",
    "interview", "coffee_date", "casual", "dinner", "date",
}


def _split_slots(
    raw_required: Any,
    *,
    occasion: str,
    layer_requested: bool,
) -> Tuple[List[str], List[str]]:
    required: List[str] = []
    optional: List[str] = []

    def add(dst: List[str], slot: str) -> None:
        if slot and slot not in dst:
            dst.append(slot)

    for raw_slot in _str_list(raw_required):
        key = _snake(raw_slot)
        if key in _PRIMARY_SLOT_ALIASES:
            add(required, _PRIMARY_SLOT_ALIASES[key])
        elif key in _OPTIONAL_SLOT_ALIASES:
            add(optional, _OPTIONAL_SLOT_ALIASES[key])
        elif key in ACCESSORY_SLOTS:
            add(optional, key)
        # silently drop unknown slot tokens

    # Demote outerwear out of required unless the user explicitly asked for a
    # layer, or the occasion list permits an outerwear-required look.
    if "outerwear" in required and (occasion in _CORE_THREE_OCCASIONS) and not layer_requested:
        required = [s for s in required if s != "outerwear"]
        add(optional, "outerwear")

    # Accessory slots never stay required.
    for acc in list(required):
        if acc in ACCESSORY_SLOTS:
            required.remove(acc)
            add(optional, acc)

    # Dress/one-piece replaces top+bottom.
    has_one_piece = any(s in {"dress", "one_piece"} for s in required)
    if not required or (not has_one_piece and not {"top", "bottom", "footwear"}.issubset(set(required))):
        if has_one_piece:
            for s in ("footwear",):
                add(required, s)
        else:
            for s in ("top", "bottom", "footwear"):
                add(required, s)

    required = [s for s in required if s in ALLOWED_REQUIRED_SLOTS]
    optional = [s for s in optional if s in OPTIONAL_SLOT_VOCAB]
    return required, optional


# ---- confidence -------------------------------------------------------------
def _confidence(raw: Dict[str, Any], *, required_present: bool) -> float:
    if "confidence" in raw:
        try:
            return max(0.0, min(1.0, float(raw["confidence"])))
        except Exception:
            pass
    core = ["occasion", "style_direction", "required_slots"]
    optional = ["sub_intent", "formality", "avoid_items", "palette_direction", "accessory_policy"]
    missing_optional = sum(1 for k in optional if not raw.get(k))
    if required_present and missing_optional <= 1:
        return 0.82
    if missing_optional >= 3:
        return 0.6
    return 0.7


# ---- public API -------------------------------------------------------------
def normalize_style_orchestrator_brief(
    raw: Any, *, has_wardrobe_context: bool = True, layer_requested: bool = False
) -> Dict[str, Any]:
    """Convert raw agent JSON to a canonical brief dict. Never raises."""
    if not isinstance(raw, dict):
        return CanonicalStyleBrief(confidence=0.0, raw_agent_brief={}).to_dict()

    occasion = _snake(raw.get("occasion"))
    sub_intent = _snake(raw.get("sub_intent")) or "outfit_generation"
    formality = _formality(raw.get("formality"))
    sd_primary, sd_alternates = _primary_alternates(raw.get("style_direction"))
    palette = _palette(raw.get("palette_direction"))
    wardrobe_usage = _wardrobe_usage(
        raw.get("wardrobe_usage"), has_wardrobe_context=has_wardrobe_context
    )
    avoid_items = _avoid_items(raw.get("avoid_items"))
    required_slots, optional_slots = _split_slots(
        raw.get("required_slots"), occasion=occasion, layer_requested=layer_requested
    )
    accessory_policy = _snake(raw.get("accessory_policy"))
    clarification = bool(raw.get("clarification_needed", False))
    if not clarification and occasion and has_wardrobe_context:
        clarification = False
    required_present = bool(occasion) and bool(required_slots) and bool(sd_primary or sd_alternates)
    confidence = _confidence(raw, required_present=required_present)

    brief = CanonicalStyleBrief(
        occasion=occasion,
        sub_intent=sub_intent,
        formality=formality,
        style_direction=StyleDirection(primary=sd_primary, alternates=sd_alternates),
        wardrobe_usage=wardrobe_usage,
        avoid_items=avoid_items,
        required_slots=required_slots,
        optional_slots=optional_slots,
        palette_direction=palette,
        accessory_policy=accessory_policy,
        clarification_needed=clarification,
        confidence=confidence,
        raw_agent_brief=dict(raw),
    )
    return brief.to_dict()


_MIN_CONFIDENCE = 0.55


def validate_canonical_brief(brief: Any) -> Tuple[bool, str]:
    """Return (ok, reason). ok=False means the caller should fall back to the
    legacy/deterministic pipeline. Never raises."""
    if not isinstance(brief, dict):
        return False, "not_a_dict"
    if not brief.get("occasion"):
        return False, "missing_occasion"
    if not brief.get("required_slots"):
        return False, "missing_required_slots"
    sd = brief.get("style_direction") or {}
    if not (sd.get("primary") or sd.get("alternates")):
        return False, "missing_style_direction"
    if brief.get("clarification_needed"):
        return False, "clarification_needed"
    try:
        if float(brief.get("confidence", 0.0)) < _MIN_CONFIDENCE:
            return False, "low_confidence"
    except Exception:
        return False, "bad_confidence"
    return True, "ok"
