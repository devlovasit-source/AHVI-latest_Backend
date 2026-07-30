"""Canonical professional-item safety evaluation shared by every Style path."""

from __future__ import annotations

from typing import Any, Mapping


# Existing production thresholds: outfit_quality_guard used 0.50 for client
# meetings and 0.45 for boardrooms; style_scorer used 0.50 as the
# professionalism midpoint. Keep these centralized and documented.
CLIENT_MEETING_MIN_SCORE = 0.50
BOARDROOM_MIN_SCORE = 0.45
PROFESSIONAL_MIN_SCORE = 0.50

PROFESSIONAL_OCCASIONS = frozenset({
    "office", "client_meeting", "corporate_office", "interview",
    "business_formal", "business_casual", "presentation", "startup_office",
    "boardroom", "executive_meeting", "business", "conference",
})
CLIENT_MEETING_OCCASIONS = frozenset({"client_meeting", "executive_meeting"})
BOARDROOM_OCCASIONS = frozenset({"boardroom", "executive_meeting"})

_CANONICAL_FIELDS = (
    "professional_safe", "professionalism_score", "client_meeting_score",
    "boardroom_score", "safety_tags",
)
_LEGACY_FIELDS = (
    "business_safe", "blocked_occasions", "style_tags", "occasion_tags", "tags",
    "fabric_finish", "finish", "pattern", "material", "fabric",
)
_POSITIVE_TAGS = {"office", "client_meeting", "boardroom", "cultural_professional"}
_UNSAFE_LEGACY_TAGS = (
    "party", "festive", "evening wear", "clubwear", "nightclub", "shiny",
    "metallic", "sequin", "sequins", "satin", "sleepwear", "loungewear",
    "swimwear", "beachwear",
)
_OFFICE_BAD_TOP_TOKENS = ("shiny", "satin", "metallic", "sequin", "sequins")
_OFFICE_BAD_TOKENS = (
    "boxer", "underwear", "sleepwear", "loungewear", "lounge", "pyjama",
    "pajama", "swim shorts", "swim_shorts", "swimshorts", "swim trunk",
    "gym shorts", "gym_shorts", "gymshorts", "running shorts",
    "running_shorts", "runningshorts", "flip flop", "flip-flop", "flipflop",
    "slider", "slides", "slipper", "beach", "pool", " cap", "cap ",
    "baseballcap", "hat ", " hat", "fedora", "sunhat", "sun hat",
    "sunglass", "camo", "camouflage", "cargo", "party", "festive",
    "distressed", "graphic tee", "graphic-tee", "streetwear", " shorts",
    "sandals", " sandal", "slippers",
)


def normalize_professional_occasion(occasion: Any) -> str:
    return str(occasion or "").strip().lower().replace("-", "_").replace(" ", "_")


def is_professional_occasion(occasion: Any) -> bool:
    return normalize_professional_occasion(occasion) in PROFESSIONAL_OCCASIONS


def _list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = value.replace("|", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = []
    return [normalize_professional_occasion(v) for v in values if str(v or "").strip()]


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merged_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for key in ("style_metadata", "professional_safety"):
        nested = item.get(key)
        if isinstance(nested, Mapping):
            evidence.update(nested)
    evidence.update(item)
    return evidence


def _role_blob(item: Mapping[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("role", "category", "main_category", "sub_category", "subcategory", "type")
    ).lower()


def _name_blob(item: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "name", "label", "category", "sub_category", "subcategory", "use_case",
        "useCase", "occasion", "occasions", "tags", "style_tags",
    ):
        value = item.get(key)
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(v or "") for v in value)
        else:
            parts.append(str(value or ""))
    return f" {' '.join(parts).lower()} "


def is_office_bad_item(item: Mapping[str, Any]) -> bool:
    return _office_bad_token(item) is not None


def _office_bad_token(item: Mapping[str, Any]) -> str | None:
    blob = _name_blob(item)
    return next((token.strip().replace(" ", "_") for token in _OFFICE_BAD_TOKENS if token in blob), None)


def is_hard_professional_block(decision: Mapping[str, Any]) -> bool:
    """True for explicit/legacy/fallback vetoes, false for score cutoffs."""
    reason = str(decision.get("reason_code") or "")
    return not bool(decision.get("allowed")) and reason not in {
        "client_meeting_score_below_threshold",
        "boardroom_score_below_threshold",
        "professionalism_score_below_threshold",
    }


def _result(allowed: bool, reason: str, score: float, source: str) -> dict[str, Any]:
    return {
        "allowed": bool(allowed),
        "reason_code": reason,
        "score": round(max(0.0, min(1.0, float(score))), 3),
        "evidence_source": source,
    }


def evaluate_professional_safety(item: Mapping[str, Any], occasion: Any) -> dict[str, Any]:
    """Evaluate one item with canonical hard-negative-first precedence."""
    if not isinstance(item, Mapping) or not is_professional_occasion(occasion):
        return _result(True, "not_professional_context", 0.5, "none")

    occ = normalize_professional_occasion(occasion)
    evidence = _merged_evidence(item)
    safety_tags = set(_list(evidence.get("safety_tags")))
    canonical_present = any(
        evidence.get(key) is not None and evidence.get(key) != []
        for key in _CANONICAL_FIELDS
    )

    # A. Explicit canonical hard negatives always win.
    if evidence.get("professional_safe") is False:
        return _result(False, "professional_safe_false", 0.0, "canonical")
    if "not_professional" in safety_tags:
        return _result(False, "not_professional", 0.0, "canonical")
    if occ in CLIENT_MEETING_OCCASIONS and "not_client_meeting" in safety_tags:
        return _result(False, "not_client_meeting", 0.0, "canonical")
    if occ in BOARDROOM_OCCASIONS and "not_boardroom" in safety_tags:
        return _result(False, "not_boardroom", 0.0, "canonical")

    # B. Context-specific evidence uses the pre-existing production cutoffs.
    professional_score = _number(evidence.get("professionalism_score"))
    client_score = _number(evidence.get("client_meeting_score"))
    boardroom_score = _number(evidence.get("boardroom_score"))
    if occ in CLIENT_MEETING_OCCASIONS and client_score is not None and client_score < CLIENT_MEETING_MIN_SCORE:
        return _result(False, "client_meeting_score_below_threshold", client_score, "canonical")
    if occ in BOARDROOM_OCCASIONS and boardroom_score is not None and boardroom_score < BOARDROOM_MIN_SCORE:
        return _result(False, "boardroom_score_below_threshold", boardroom_score, "canonical")
    if professional_score is not None and professional_score < PROFESSIONAL_MIN_SCORE:
        return _result(False, "professionalism_score_below_threshold", professional_score, "canonical")

    # C. Explicit positives are decisive only after relevant negatives/scores.
    relevant_positive = bool(
        evidence.get("professional_safe") is True
        or "office" in safety_tags
        or "cultural_professional" in safety_tags
        or (occ in CLIENT_MEETING_OCCASIONS and "client_meeting" in safety_tags)
        or (occ in BOARDROOM_OCCASIONS and "boardroom" in safety_tags)
    )
    if relevant_positive:
        score = (
            boardroom_score if occ in BOARDROOM_OCCASIONS and boardroom_score is not None
            else client_score if occ in CLIENT_MEETING_OCCASIONS and client_score is not None
            else professional_score if professional_score is not None
            else 1.0
        )
        return _result(True, "canonical_positive", score, "canonical")
    if canonical_present:
        score = professional_score if professional_score is not None else 0.5
        return _result(True, "canonical_neutral", score, "canonical")

    # D. Legacy compatibility.
    blocked = set(_list(evidence.get("blocked_occasions")))
    if occ in blocked:
        return _result(False, "blocked_occasions", 0.0, "legacy")
    if evidence.get("business_safe") is False:
        return _result(False, "business_safe_false", 0.0, "legacy")
    legacy_tags = " ".join(
        " ".join(_list(evidence.get(key))) for key in ("style_tags", "occasion_tags", "tags")
    )
    for tag in _UNSAFE_LEGACY_TAGS:
        if normalize_professional_occasion(tag) in legacy_tags:
            return _result(False, f"legacy_tag_{normalize_professional_occasion(tag)}", 0.0, "legacy")

    role_blob = _role_blob(evidence)
    is_top_or_dress = any(token in role_blob for token in ("top", "shirt", "blouse", "tee", "kurta", "sweater", "hoodie", "polo", "dress"))
    if is_top_or_dress:
        finish_blob = " ".join(
            str(evidence.get(key) or "").lower()
            for key in ("fabric_finish", "finish", "pattern", "material", "fabric")
        )
        for token in _OFFICE_BAD_TOP_TOKENS:
            if token in finish_blob:
                return _result(False, f"fabric_finish_{token}", 0.0, "legacy")

    legacy_present = any(
        evidence.get(key) is not None and evidence.get(key) != []
        for key in _LEGACY_FIELDS
    )
    if evidence.get("business_safe") is True:
        return _result(True, "business_safe_true", 1.0, "legacy")
    if legacy_present:
        return _result(True, "legacy_neutral", 0.5, "legacy")

    # E. Tokens are a last resort only when no structured/legacy evidence exists.
    bad_token = _office_bad_token(evidence)
    if bad_token:
        return _result(False, f"name_token_{bad_token}", 0.0, "fallback")
    if is_top_or_dress:
        name_color = f"{evidence.get('name') or evidence.get('label') or ''} {evidence.get('color_name') or evidence.get('color') or ''}".lower()
        for token in _OFFICE_BAD_TOP_TOKENS:
            if token in name_color:
                return _result(False, f"name_token_{token}_top", 0.0, "fallback")
        if "gold" in name_color:
            return _result(False, "name_token_gold_top", 0.0, "fallback")
    return _result(True, "fallback_clear", 0.5, "fallback")
