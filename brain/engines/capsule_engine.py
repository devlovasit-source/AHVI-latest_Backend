"""AHVI Capsule Wardrobe engine.

A capsule wardrobe is NOT a single outfit. It is a curated foundation of
versatile, neutral, repeatable garments that recombine across multiple
occasions. The normal outfit pipeline picks ONE styled board per request;
the capsule engine picks a FOUNDATION (one of each core role) plus
sample looks + missing slots + shopping gaps.

Detection: occasion == "capsule" or tokens contain capsule / essentials
/ core_wardrobe / minimal_wardrobe.

Output shape:
{
    "type": "capsule_wardrobe",
    "capsule_foundation": [<core wardrobe item dicts>],
    "sample_looks": [<2-3 ways to combine foundation>],
    "missing_slots": ["outerwear", ...],
    "shopping_gaps": [{"label": "Navy chinos", "reason": "..."}],
    "styling_note": "...",
}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ahvi.capsule_engine")


_CORE_ROLES = ("top", "bottom", "footwear", "outerwear")

_FOUNDATION_TARGETS = {
    "top": 3,
    "bottom": 2,
    "footwear": 2,
    "outerwear": 1,
}


def _item_text(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    return " ".join(
        str(item.get(k) or "").lower()
        for k in (
            "name",
            "title",
            "category",
            "sub_category",
            "subcategory",
            "type",
            "role",
            "slot",
            "material",
            "color",
        )
    )


def _item_role(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return "unknown"
    role = str(item.get("role") or item.get("slot") or "").lower()
    if role in {"top", "bottom", "footwear", "outerwear", "accessory", "dress"}:
        return role
    blob = _item_text(item)
    if any(k in blob for k in ("shoe", "sneaker", "loafer", "boot", "sandal")):
        return "footwear"
    if any(k in blob for k in ("blazer", "jacket", "coat", "overshirt")):
        return "outerwear"
    if any(k in blob for k in ("trouser", "chino", "jean", "shorts", "skirt", "bottom")):
        return "bottom"
    if any(k in blob for k in ("shirt", "tee", "polo", "blouse", "kurta", "sweater", "hoodie")):
        return "top"
    if any(k in blob for k in ("watch", "belt", "ring", "bag", "necklace", "bracelet")):
        return "accessory"
    return "unknown"


_PREFERRED_TOKENS = {
    "neutral", "white", "black", "navy", "grey", "gray", "beige", "cream",
    "tan", "olive",
    "shirt", "trouser", "trousers", "chino", "chinos", "denim", "tee",
    "sneaker", "sneakers", "loafer", "loafers", "blazer",
    "merino", "linen", "cotton", "wool",
    "minimal", "essentials", "classic", "timeless", "versatile",
}

_FORBIDDEN_TOKENS = {
    "sequined", "neon", "embroidered", "printed", "novelty", "shiny",
    "metallic", "statement", "loud", "rave", "festival", "swim",
}


def _capsule_item_score(item: Dict[str, Any]) -> float:
    """Score a single wardrobe item for capsule suitability.

    Combines Metadata Validator signals (capsule_score / versatility_score
    / visual_noise / statement_level) with name-token heuristics so the
    engine still works when richness fields are missing.
    """
    if not isinstance(item, dict):
        return 0.0
    meta = item.get("style_metadata") if isinstance(item.get("style_metadata"), dict) else {}
    score = 0.5  # neutral baseline so every item is comparable
    try:
        cap = float(meta.get("capsule_score") or 0.0)
        ver = float(meta.get("versatility_score") or 0.0)
        if cap:
            score += (cap - 0.5) * 1.5
        if ver:
            score += (ver - 0.5) * 1.0
    except Exception:
        pass
    noise = str(meta.get("visual_noise") or "").strip().lower()
    stmt = str(meta.get("statement_level") or "").strip().lower()
    if noise == "high":
        score -= 0.5
    if stmt in {"statement", "risky"}:
        score -= 0.6
    blob = _item_text(item)
    for tok in _PREFERRED_TOKENS:
        if tok in blob:
            score += 0.08
    for tok in _FORBIDDEN_TOKENS:
        if tok in blob:
            score -= 0.3
    if max(score, -2.0) < score:
        pass
    return round(max(-2.0, min(score, 3.0)), 3)


_METADATA_SUBSET_KEYS = (
    "capsule_score",
    "versatility_score",
    "visual_noise",
    "statement_level",
    "formality",
    "occasions",
)


def _item_image(item: Dict[str, Any]) -> str:
    return str(
        item.get("normalized_url") or item.get("normalizedUrl")
        or item.get("masked_url") or item.get("maskedUrl")
        or item.get("image_url") or item.get("imageUrl")
        or ""
    ).strip()


def _foundation_entry(item: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Canonical structured foundation entry. Nothing is fabricated: ids and
    images come from the wardrobe record (empty when absent) and ownership is
    true by contract — the input list IS the user's wardrobe."""
    meta = item.get("style_metadata") if isinstance(item.get("style_metadata"), dict) else {}
    compact_meta = {k: meta[k] for k in _METADATA_SUBSET_KEYS if k in meta}
    entry: Dict[str, Any] = {
        "item_id": str(item.get("item_id") or item.get("id") or item.get("$id") or "").strip(),
        "name": str(item.get("name") or item.get("title") or "Item").strip(),
        "role": role,
        "category": str(item.get("category") or ""),
        "sub_category": str(item.get("sub_category") or item.get("subcategory") or ""),
        "source": str(item.get("source") or "wardrobe"),
        "owned": True,
        "capsule_score": _capsule_item_score(item),
        "style_metadata": compact_meta,
    }
    image_url = _item_image(item)
    if image_url:
        entry["image_url"] = image_url
    return entry


def _group_by_role(wardrobe: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {r: [] for r in _CORE_ROLES + ("accessory",)}
    for item in wardrobe or []:
        if not isinstance(item, dict):
            continue
        role = _item_role(item)
        if role in out:
            out[role].append(item)
    return out


def _build_sample_looks(
    foundation: Dict[str, List[Dict[str, Any]]],
    *,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """Compose 2–3 capsule sample looks from the foundation."""
    tops = foundation.get("top") or []
    bottoms = foundation.get("bottom") or []
    footwears = foundation.get("footwear") or []
    outerwears = foundation.get("outerwear") or []
    accessories = foundation.get("accessory") or []
    if not tops or not bottoms or not footwears:
        return []
    titles = [
        "Foundation Daily",
        "Smart Layer",
        "Off-Duty Edit",
        "Workday Capsule",
        "Dinner Repeat",
        "Travel Uniform",
    ]
    roles = ["primary", "alternate", "expressive", "work", "evening", "travel"]
    available_variation = max(
        len(tops),
        len(bottoms),
        len(footwears),
        len(accessories) or 1,
    )
    look_count = min(max(3, int(limit or 6)), 6)
    look_count = min(look_count, max(3, available_variation + (1 if outerwears else 0)))
    looks: List[Dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for idx in range(look_count):
        t = tops[idx % len(tops)]
        b = bottoms[(idx + (idx // max(1, len(tops)))) % len(bottoms)]
        f = footwears[(idx + (idx // 2)) % len(footwears)]
        signature = "|".join(
            str(
                (item or {}).get("id")
                or (item or {}).get("$id")
                or (item or {}).get("name")
                or ""
            )
            for item in (t, b, f)
        )
        if signature in seen_signatures and idx >= 3:
            continue
        seen_signatures.add(signature)
        items = [
            {**t, "role": "top"},
            {**b, "role": "bottom"},
            {**f, "role": "footwear"},
        ]
        if outerwears and idx in {0, 1, 3, 5}:
            items.append({**outerwears[idx % len(outerwears)], "role": "outerwear"})
        if accessories:
            items.append({**accessories[idx % len(accessories)], "role": "accessory"})
        looks.append({
            "id": f"capsule_look_{idx + 1}",
            "title": titles[idx % len(titles)],
            "badge": "CAPSULE",
            "occasion": "capsule_wardrobe",
            "occasion_label": "CAPSULE",
            "items": items,
            "set_role": roles[idx % len(roles)],
            "stylist_note": (
                "These pieces form a repeatable wardrobe foundation because "
                "they can be recombined across work, casual, and smart-casual settings."
            ),
        })
    return looks


def _missing_slots(
    foundation: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    return [
        role for role in _CORE_ROLES
        if len(foundation.get(role) or []) < _FOUNDATION_TARGETS.get(role, 1)
    ]


_GAP_SUGGESTIONS = {
    "top": {"label": "Crisp white shirt", "reason": "Versatile across office, date, dinner."},
    "bottom": {"label": "Navy or charcoal trousers", "reason": "Pairs with every top in the foundation."},
    "footwear": {"label": "Brown leather loafers", "reason": "Bridges casual + smart in one shoe."},
    "outerwear": {"label": "Neutral overshirt or blazer", "reason": "Adds layer for office / evening."},
}


def _shopping_gaps(missing: List[str]) -> List[Dict[str, str]]:
    return [_GAP_SUGGESTIONS[role] for role in missing if role in _GAP_SUGGESTIONS]


def build_capsule_response(
    user_id: str,
    wardrobe: List[Dict[str, Any]],
    query: str = "",
) -> Dict[str, Any]:
    """Entry point: route capsule-intent requests here instead of through
    the normal outfit pipeline.

    Returns a dict shaped for style_flow_service.build_style_flow_response
    callers — `cards` populated with the foundation + sample looks so the
    existing UI renders them, plus capsule-specific metadata under `data`.
    """
    grouped = _group_by_role(wardrobe or [])

    # Rank every role's items by capsule suitability + take the top N.
    foundation: Dict[str, List[Dict[str, Any]]] = {}
    for role in _CORE_ROLES + ("accessory",):
        items = grouped.get(role) or []
        items_sorted = sorted(
            items, key=lambda it: _capsule_item_score(it), reverse=True
        )
        cap = _FOUNDATION_TARGETS.get(role, 2 if role == "accessory" else 1)
        foundation[role] = items_sorted[:cap]

    sample_looks = _build_sample_looks(foundation, limit=6)
    missing = _missing_slots(foundation)
    gaps = _shopping_gaps(missing)

    # Canonical foundation contract: a FLAT list of structured item objects
    # (role-ordered, rank-ordered within each role). Display-only strings are
    # exposed separately under capsule_foundation_labels — never mixed into
    # the structured field.
    foundation_entries: List[Dict[str, Any]] = [
        _foundation_entry(item, role)
        for role in _CORE_ROLES + ("accessory",)
        for item in (foundation.get(role) or [])
    ]
    foundation_labels = [entry["name"] for entry in foundation_entries]

    if sample_looks:
        styling_note = (
            "These pieces form a versatile wardrobe foundation because "
            "they can be recombined across multiple occasions without "
            "leaning into novelty or noise."
        )
        message = (
            "Here's a capsule foundation pulled from your wardrobe. "
            "Each piece earns its place by being neutral, versatile, "
            "and repeatable across briefs you ask about most."
        )
    elif any(foundation.get(r) for r in ("top", "bottom", "footwear")):
        styling_note = (
            "I started a capsule foundation but you'll need a few more "
            "core pieces before I can build sample looks from it."
        )
        message = styling_note
    else:
        styling_note = (
            "I couldn't find enough neutral, versatile pieces in your "
            "wardrobe to seed a capsule yet."
        )
        message = styling_note

    logger.info(
        "ahvi.capsule_engine.built user=%s foundation_counts=%s missing=%s sample_looks=%d",
        user_id,
        {r: len(foundation.get(r) or []) for r in _CORE_ROLES + ("accessory",)},
        missing,
        len(sample_looks),
    )

    response: Dict[str, Any] = {
        "success": bool(sample_looks),
        "type": "capsule_wardrobe",
        "intent": "capsule",
        "board": "style",
        "message": message,
        "capsule_foundation": foundation_entries,
        "capsule_foundation_labels": foundation_labels,
        "sample_looks": sample_looks,
        "missing_slots": missing,
        "shopping_gaps": gaps,
        "styling_note": styling_note,
        "cards": sample_looks,
        "style_boards": sample_looks,
        "board_ids": sample_looks[0]["id"] if sample_looks else "",
        "chips": (
            ["Save capsule", "Find missing piece", "Try another look"]
            if sample_looks
            else ["Find missing piece", "Add wardrobe item"]
        ),
        "data": {
            "capsule_foundation": foundation_entries,
            "capsule_foundation_labels": foundation_labels,
            "sample_looks": sample_looks,
            "missing_slots": missing,
            "shopping_gaps": gaps,
            "styling_note": styling_note,
        },
        "meta": {
            "mode": "capsule_wardrobe",
            "foundation_counts": {
                r: len(foundation.get(r) or []) for r in _CORE_ROLES
            },
        },
        "audio_job_id": "offline",
    }
    return response


def looks_like_capsule_request(query: Any) -> bool:
    """Cheap token check used by style_flow_service to route early."""
    if not query:
        return False
    text = str(query).lower()
    for trigger in (
        "capsule wardrobe",
        "capsule",
        "wardrobe essentials",
        "build essentials",
        "core wardrobe",
        "minimal wardrobe",
    ):
        if trigger in text:
            return True
    return False


__all__ = [
    "build_capsule_response",
    "looks_like_capsule_request",
]
