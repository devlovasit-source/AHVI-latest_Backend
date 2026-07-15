"""Pure compatibility gate for fixed Style-asset anchors.

The gate consumes only canonical structured evidence.  It never inspects an
asset name, URL, filename, or id to invent gender, occasion, or weather facts.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from services.professional_safety import evaluate_professional_safety
from services.style_asset_metadata_contract import normalize_style_asset_metadata
from services.style_item_contract import canonical_item_role, canonical_item_source
from services.style_reasoning_engine import asset_weather_compatibility_score


_PROFESSIONAL = {
    "client_dinner", "client_meeting", "client_presentation", "office",
    "office_meeting", "smart_casual", "team_dinner",
}
_CASUAL = {
    "basketball_game", "brunch", "capsule", "casual", "casual_dinner",
    "coffee_date", "daily", "date_night", "first_date", "party", "travel",
}
_FESTIVE = {
    "cultural", "festive", "temple_modest", "wedding", "wedding_guest",
}
_ACTIVE = {"beach", "resort", "swimming", "workout"}
_VALID_FIXED_ROLES = {"top", "bottom", "dress", "outerwear", "footwear", "accessory"}


def _strings(value: Any) -> list[str]:
    values: Iterable[Any] = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item).strip().lower() for item in values if str(item or "").strip()]


def _occasion_group(value: str) -> str:
    if value in _PROFESSIONAL:
        return "professional"
    if value in _CASUAL:
        return "casual"
    if value in _FESTIVE:
        return "festive"
    if value in _ACTIVE:
        return "active"
    return value


def _decision(allowed: bool, reason_code: str, evidence_source: str, **extra: Any) -> Dict[str, Any]:
    return {
        "allowed": bool(allowed),
        "reason_code": reason_code,
        "evidence_source": evidence_source,
        **extra,
    }


def evaluate_style_asset_anchor(
    raw_item: Mapping[str, Any], canonical_context: Mapping[str, Any] | None
) -> Dict[str, Any]:
    """Normalize and evaluate one trusted server Style asset.

    ``allowed`` is false for any hard source, readiness, role, professional,
    weather, gender, or occasion conflict. Missing compatibility evidence is
    neutral so Batch 10 ``limited`` assets remain eligible.
    """
    raw = dict(raw_item or {})
    context = dict(canonical_context or {})
    # A caller-provided source is evidence, not authority.  Reject an
    # explicitly non-style-asset source before normalization can apply the
    # trusted repository default.
    if "source" in raw and canonical_item_source(raw) != "style_asset":
        return {
            "allowed": False,
            "reason_code": "source_policy_violation",
            "failed_gate": "source",
            "item": dict(raw),
            "decisions": {
                "source": _decision(False, "source_policy_violation", "caller_metadata")
            },
        }
    item = normalize_style_asset_metadata(raw, trusted_style_asset_source=True)

    source = _decision(
        canonical_item_source(item) == "style_asset",
        "style_asset_source" if canonical_item_source(item) == "style_asset" else "source_policy_violation",
        "server_repository",
    )
    persisted_rejected = str(raw.get("metadata_status") or "").strip().lower() == "rejected"
    rejected = persisted_rejected or item.get("metadata_status") == "rejected"
    readiness = _decision(
        not rejected,
        "metadata_rejected" if rejected else f"metadata_{item.get('metadata_status') or 'limited'}",
        "canonical_metadata_contract",
        status=item.get("metadata_status") or "limited",
    )

    role_value = canonical_item_role(item)
    role = _decision(
        role_value in _VALID_FIXED_ROLES,
        "wearable_role" if role_value in _VALID_FIXED_ROLES else "invalid_wearable_role",
        "canonical_item_contract",
        role=role_value,
    )

    occasion = context.get("canonical_occasion") or context.get("occasion")
    professional = evaluate_professional_safety(item, occasion)

    weather_score = asset_weather_compatibility_score(item, context)
    weather_present = bool(context.get("weather_context") or context.get("weather"))
    weather = _decision(
        weather_score > -2,
        "weather_incompatible" if weather_score <= -2 else (
            "weather_compatible" if weather_score > 0 else "weather_neutral"
        ),
        "structured_metadata" if weather_present else "none",
        score=weather_score,
    )

    target_gender = str(context.get("style_gender") or context.get("gender") or "unknown").strip().lower()
    asset_gender = str(item.get("gender_fit") or item.get("gender") or "unknown").strip().lower()
    gender_conflict = (
        target_gender in {"male", "female"}
        and asset_gender in {"male", "female"}
        and target_gender != asset_gender
    )
    gender = _decision(
        not gender_conflict,
        "gender_mismatch" if gender_conflict else (
            "gender_match" if target_gender == asset_gender and target_gender != "unknown" else "gender_neutral"
        ),
        "structured_metadata" if asset_gender != "unknown" else "none",
        target_gender=target_gender,
        asset_gender=asset_gender,
    )

    current = str(occasion or "").strip().lower()
    families = set(_strings(item.get("occasion_families")))
    safety_tags = set(_strings(item.get("safety_tags")))
    current_group = _occasion_group(current)
    family_groups = {_occasion_group(value) for value in families}
    cultural_professional = (
        current_group == "professional"
        and "cultural_professional" in safety_tags
        and bool(families & _FESTIVE)
    )
    occasion_conflict = bool(
        current and families and current not in families
        and current_group not in family_groups and not cultural_professional
    )
    occasion_decision = _decision(
        not occasion_conflict,
        "occasion_mismatch" if occasion_conflict else (
            "occasion_match" if families else "occasion_neutral"
        ),
        "structured_metadata" if families else "none",
        canonical_occasion=current or None,
    )

    decisions = {
        "source": source,
        "readiness": readiness,
        "role": role,
        "professional": dict(professional),
        "weather": weather,
        "gender": gender,
        "occasion": occasion_decision,
    }
    first_failure = next(
        (name for name in ("source", "readiness", "role", "professional", "weather", "gender", "occasion")
         if not decisions[name].get("allowed")),
        None,
    )
    return {
        "allowed": first_failure is None,
        "reason_code": (
            decisions[first_failure].get("reason_code") if first_failure else "anchor_compatible"
        ),
        "failed_gate": first_failure,
        "item": item,
        "decisions": decisions,
    }


__all__ = ["evaluate_style_asset_anchor"]
