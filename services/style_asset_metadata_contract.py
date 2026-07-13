"""Deterministic canonical metadata contract for curated Style assets.

This module is deliberately pure: it performs no I/O, makes no network or LLM
calls, and never guesses semantic metadata that is not present in the source.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping


METADATA_VERSION = "1.0"
METADATA_STATUSES = {"rejected", "limited", "ready"}

_ROLE_ALIASES = {
    "tops": "top",
    "shirts": "top",
    "bottoms": "bottom",
    "pants": "bottom",
    "trousers": "bottom",
    "shoes": "footwear",
    "accessories": "accessory",
    "jackets": "outerwear",
    "dresses": "dress",
    "onepiece": "one_piece",
    "one piece": "one_piece",
}
VALID_ROLES = {
    "top", "bottom", "footwear", "accessory", "outerwear", "dress",
    "one_piece", "ethnic",
}
_GENDER_ALIASES = {
    "m": "male", "man": "male", "men": "male", "mens": "male",
    "masculine": "male", "f": "female", "woman": "female",
    "women": "female", "womens": "female", "feminine": "female",
    "all": "unisex", "any": "unisex", "neutral": "unisex",
    "genderless": "unisex",
}
_INCOMPATIBLE_SOURCES = {
    "wardrobe", "user_wardrobe", "closet", "saved_board", "wear_history",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and (not isinstance(value, str) or value.strip()):
            return value
    return None


def _list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        text = _text(value)
        if not text:
            return []
        try:
            parsed = json.loads(text)
            values = parsed if isinstance(parsed, list) else re.split(r"[,|]", text)
        except (TypeError, ValueError, json.JSONDecodeError):
            values = re.split(r"[,|]", text)
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        normalized = _text(item).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _number(value: Any, *, low: float, high: float) -> float | int | None:
    if isinstance(value, bool) or value is None or _text(value) == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < low or number > high:
        return None
    return int(number) if number.is_integer() else round(number, 2)


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = _text(value).lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _gender(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    normalized = _text(value).lower()
    normalized = _GENDER_ALIASES.get(normalized, normalized)
    return normalized if normalized in {"male", "female", "unisex"} else "unknown"


def _role(value: Any) -> str:
    normalized = re.sub(r"[_-]+", " ", _text(value).lower()).strip()
    normalized = _ROLE_ALIASES.get(normalized, normalized.replace(" ", "_"))
    return normalized if normalized in VALID_ROLES else ""


def _usable_image(value: Any) -> bool:
    image = _text(value)
    return bool(image and not image.lower().startswith(("data:", "javascript:")))


def normalize_style_asset_metadata(
    raw: Mapping[str, Any] | None,
    *,
    trusted_style_asset_source: bool = False,
) -> Dict[str, Any]:
    """Return one canonical internal shape and deterministic readiness fields."""
    row = dict(raw or {})
    out = dict(row)

    asset_id = _text(_first(row, "asset_id", "assetId", "id", "$id"))
    name = _text(_first(row, "name", "title", "label"))
    raw_role = _first(row, "role", "category", "main_category", "category_group", "type")
    role = _role(raw_role) or _role(_first(row, "category", "main_category", "category_group", "type"))
    category = _text(_first(row, "category", "main_category", "category_group", "role")).lower()
    if role:
        category = category or role
    sub_category = _text(_first(row, "sub_category", "subcategory", "subCategory")).lower()

    board_image_url = _text(_first(
        row, "board_image_url", "boardImageUrl", "board_url", "boardUrl",
        "normalized_url", "normalizedUrl", "cutout_url", "cutoutUrl",
        "transparent_image_url", "transparentImageUrl", "rmbg_url", "rmbgUrl",
        "image_url", "imageUrl", "url", "asset_url", "asset_path",
    ))
    image_url = _text(_first(row, "image_url", "imageUrl", "url", "asset_url", "asset_path"))
    image_url = image_url or board_image_url
    normalized_url = _text(_first(row, "normalized_url", "normalizedUrl"))
    cutout_url = _text(_first(row, "cutout_url", "cutoutUrl", "transparent_image_url", "transparentImageUrl"))

    gender_fit = _gender(_first(row, "gender_fit", "genderFit", "gender", "gender_support", "genderSupport"))
    colors = _list(_first(row, "colors", "color_palette", "colorPalette", "palette", "color"))
    archetypes = _list(_first(row, "archetypes", "style_archetypes", "styleArchetypes"))
    occasions = _list(_first(row, "occasion_families", "occasionFamilies", "occasions", "occasion_tags", "occasion"))
    traits = _list(_first(row, "traits", "style_traits", "styleTraits", "required_traits"))
    weather_tags = _list(_first(row, "weather_tags", "weatherTags", "weather"))

    out.update({
        "asset_id": asset_id,
        "name": name,
        "source": "style_asset",
        "role": role,
        "category": category,
        "sub_category": sub_category,
        "subcategory": sub_category,
        "gender_fit": gender_fit,
        "gender": gender_fit,
        "board_image_url": board_image_url,
        "image_url": image_url,
        "normalized_url": normalized_url,
        "cutout_url": cutout_url,
        "colors": colors,
        "pattern": _text(_first(row, "pattern", "print")),
        "material": _text(_first(row, "material", "fabric")),
        "finish": _text(row.get("finish")),
        "visual_noise": _number(_first(row, "visual_noise", "visualNoise"), low=1, high=9),
        "statement_level": _number(_first(row, "statement_level", "statementLevel"), low=1, high=9),
        "archetypes": archetypes,
        "occasion_families": occasions,
        "occasions": occasions,
        "formality": _number(_first(row, "formality", "formality_level", "formalityLevel"), low=1, high=9),
        "energy": _number(_first(row, "energy", "energy_level", "energyLevel"), low=1, high=9),
        "movement": _number(_first(row, "movement", "movement_level", "movementLevel"), low=1, high=9),
        "traits": traits,
        "weather_tags": weather_tags,
        "cultural_context": _list(_first(row, "cultural_context", "culturalContext")),
        "metadata_version": METADATA_VERSION,
        "metadata_updated_at": _text(_first(row, "metadata_updated_at", "metadataUpdatedAt")),
    })

    source_value = _text(row.get("source")).lower()
    if source_value and source_value != "style_asset":
        out["source_origin"] = _text(row.get("source"))
    provenance_invalid = source_value in _INCOMPATIBLE_SOURCES
    explicitly_unsafe = (
        _bool(_first(row, "unsafe", "is_unsafe", "isUnsafe")) is True
        or _bool(_first(row, "wearable", "is_wearable", "isWearable")) is False
        or _text(_first(row, "safety_status", "safetyStatus")).lower()
        in {"unsafe", "rejected", "nonwearable", "non_wearable"}
    )

    professional_safety = {
        key: row[key]
        for key in (
            "professional_safe", "professionalism_score", "client_meeting_score",
            "boardroom_score", "safety_tags",
        )
        if key in row and row[key] not in (None, "", [])
    }
    if professional_safety:
        out["professional_safety"] = professional_safety

    missing: List[str] = []
    if not asset_id:
        missing.append("asset_id")
    if not name:
        missing.append("name")
    if not _usable_image(board_image_url):
        missing.append("board_image_url")
    if not role:
        missing.append("role")
    if gender_fit == "unknown":
        missing.append("gender_fit")
    if not colors:
        missing.append("colors")
    if not (archetypes or occasions):
        missing.append("occasion_or_archetype")

    detail_signals = sum((
        out["formality"] is not None,
        bool(traits),
        out["visual_noise"] is not None or out["statement_level"] is not None,
        bool(weather_tags),
        bool(professional_safety),
    ))
    if detail_signals < 2:
        missing.append("style_or_safety_detail")

    score = 0.0
    score += 0.15 if asset_id else 0
    score += 0.05 if name else 0
    score += 0.20 if _usable_image(board_image_url) else 0
    score += 0.15 if role else 0
    score += 0.10 if gender_fit != "unknown" else 0
    score += 0.10 if colors else 0
    score += 0.10 if (archetypes or occasions) else 0
    score += 0.04 if out["formality"] is not None else 0
    score += 0.03 if traits else 0
    score += 0.03 if (out["visual_noise"] is not None or out["statement_level"] is not None) else 0
    score += 0.02 if weather_tags else 0
    score += 0.03 if professional_safety else 0
    score = round(min(1.0, score), 2)

    hard_reject = (
        not asset_id or not _usable_image(board_image_url) or not role
        or provenance_invalid or explicitly_unsafe
    )
    if hard_reject:
        metadata_status = "rejected"
    elif gender_fit != "unknown" and colors and (archetypes or occasions) and detail_signals >= 2 and score >= 0.75:
        metadata_status = "ready"
    else:
        metadata_status = "limited"

    out["metadata_status"] = metadata_status
    out["metadata_score"] = score
    out["missing_metadata_fields"] = sorted(set(missing))
    # Semantic unknowns stay absent; only structural diagnostics use explicit
    # unknown values (notably gender_fit) so callers cannot mistake blanks for
    # asserted evidence.
    for key in (
        "pattern", "material", "finish", "visual_noise", "statement_level",
        "formality", "energy", "movement", "normalized_url", "cutout_url",
        "cultural_context",
    ):
        if out.get(key) in (None, "", [], {}):
            out.pop(key, None)
    return out


def canonical_metadata_update(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the idempotent, evidence-backed fields suitable for backfill."""
    normalized = normalize_style_asset_metadata(raw, trusted_style_asset_source=True)
    fields = (
        "asset_id", "name", "source", "source_origin", "role", "category",
        "sub_category", "gender_fit", "board_image_url", "normalized_url",
        "cutout_url", "colors", "pattern", "material", "finish", "visual_noise",
        "statement_level", "archetypes", "occasion_families", "formality",
        "energy", "movement", "traits", "weather_tags", "cultural_context",
        "metadata_version", "metadata_status",
        "metadata_score", "missing_metadata_fields",
    )
    return {key: normalized[key] for key in fields if normalized.get(key) not in (None, "", [], {})}


def summarize_style_asset_metadata(assets: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    statuses: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    genders: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    invalid_images = 0
    total = 0
    for raw in assets:
        asset = normalize_style_asset_metadata(raw, trusted_style_asset_source=True)
        total += 1
        statuses[asset["metadata_status"]] += 1
        roles[asset.get("role") or "unknown"] += 1
        genders[asset.get("gender_fit") or "unknown"] += 1
        missing.update(asset.get("missing_metadata_fields") or [])
        invalid_images += int("board_image_url" in (asset.get("missing_metadata_fields") or []))
    return {
        "total": total,
        "status_counts": dict(sorted(statuses.items())),
        "role_counts": dict(sorted(roles.items())),
        "gender_counts": dict(sorted(genders.items())),
        "missing_field_counts": dict(sorted(missing.items())),
        "invalid_image_count": invalid_images,
    }
