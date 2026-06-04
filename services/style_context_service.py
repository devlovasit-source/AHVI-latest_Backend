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
    for key in ("image_url", "imageUrl", "image", "url", "thumbnail", "photo"):
        val = str(item.get(key) or "").strip()
        if val:
            return val
    return ""


def _normalize_wardrobe_items(raw_items: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in _safe_list(raw_items):
        if not isinstance(it, dict):
            continue
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
) -> Dict[str, Any]:
    """Assemble the unified style context. All inputs optional; absent
    sources become empty structures so the prompt stays well-formed."""
    safe_mode = str(mode or "style_advice").strip().lower()
    if safe_mode not in _STYLE_MODES:
        safe_mode = "style_advice"

    items = _normalize_wardrobe_items(wardrobe_items)
    profile = _safe_dict(user_profile)
    # Style DNA / preferences are read from the profile if present; we do not
    # fabricate them.
    style_dna = _safe_dict(profile.get("style_dna") or profile.get("styleDNA"))
    preferences = _safe_dict(
        profile.get("style_preferences")
        or profile.get("preferences")
        or profile.get("style")
    )

    context = {
        "query": str(query or "").strip(),
        "occasion": (str(occasion or "").strip().lower() or None),
        "mode": safe_mode,
        "wardrobe_available": len(items) > 0,
        "wardrobe_summary": _wardrobe_summary(items),
        "wardrobe_items": items,
        "weather_context": _safe_dict(weather),
        "event_context": _safe_dict(event_context),
        "style_dna": style_dna,
        "preferences": preferences,
        "last_style_context": _safe_dict(last_style_context),
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
    cleaned: List[Dict[str, Any]] = []
    for m in missing_items or []:
        if not isinstance(m, dict):
            continue
        cleaned.append(
            {
                "name": str(m.get("name") or "").strip(),
                "category": str(m.get("category") or "").strip(),
                "reason": str(m.get("reason") or "").strip(),
                "unlocks": [str(u).strip() for u in _safe_list(m.get("unlocks")) if str(u).strip()][:6],
            }
        )
    return {
        "type": "missing_piece_intelligence",
        "owned_percentage": owned_pct,
        "missing_items": cleaned,
    }


def compact_context_for_prompt(context: Dict[str, Any]) -> Dict[str, Any]:
    """Trim the style context to a small, prompt-safe slice. Avoids shipping
    every wardrobe item / raw payloads into the Gemini prompt."""
    ctx = _safe_dict(context)
    items = _safe_list(ctx.get("wardrobe_items"))[:18]
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
        "style_dna_archetypes": (
            _safe_dict(ctx.get("style_dna")).get("style_archetypes")
            if isinstance(ctx.get("style_dna"), dict)
            else None
        ),
        "last_style_mode": _safe_dict(ctx.get("last_style_context")).get("last_style_mode"),
        "base_occasion": _safe_dict(ctx.get("last_style_context")).get("base_occasion"),
    }
