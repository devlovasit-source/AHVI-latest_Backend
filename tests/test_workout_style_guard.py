"""Workout occasion regression tests.

The "workout" query historically resolved to "office" via the "work"
substring. These tests pin token-aware behavior across the layers.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from brain.engines.style_brief import build_brief, validate_board
from brain.engines.outfit_quality_guard import guard_outfit


def _outfit_with_meta(items_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "top": items_meta.get("top", {}),
        "bottom": items_meta.get("bottom", {}),
        "footwear": items_meta.get("footwear", {}),
    }


def test_workout_query_does_not_normalize_to_office():
    from services.style_flow_service import _normalize_occasion_value
    assert _normalize_occasion_value("", "workout outfit suggestion for gym") == "workout"
    # Even when the query says "office" inside a wider sentence, "workout"
    # token wins per priority.
    assert _normalize_occasion_value("", "workout, not for the office") == "workout"


def test_office_query_still_normalizes_to_office():
    from services.style_flow_service import _normalize_occasion_value
    assert _normalize_occasion_value("", "office today") == "office"
    assert _normalize_occasion_value("", "work outfit for monday") == "office"


def test_brief_for_workout_rejects_loafers_board():
    brief = build_brief(query="gym workout session")
    board = {
        "id": "b1",
        "title": "Boardroom Casual",
        "items": [
            {"id": "1", "name": "Shirt", "category": "button down", "role": "top"},
            {"id": "2", "name": "Trousers", "category": "trousers", "role": "bottom"},
            {"id": "3", "name": "Loafers", "category": "loafers", "role": "footwear"},
        ],
    }
    passes, reasons, _ = validate_board(board, brief)
    assert not passes
    assert any("forbidden_signal" in r for r in reasons)


def test_brief_for_workout_passes_activewear():
    brief = build_brief(query="gym workout")
    board = {
        "id": "b1",
        "title": "Training Session",
        "items": [
            {"id": "1", "name": "Training Tee", "category": "tee", "role": "top"},
            {"id": "2", "name": "Track Pants", "category": "track", "role": "bottom"},
            {"id": "3", "name": "Running Sneakers", "category": "sneakers", "role": "footwear"},
        ],
    }
    passes, _, _ = validate_board(board, brief)
    assert passes


def test_workout_outfit_blocked_by_legacy_quality_guard_when_formal_for_event():
    """Cross-check that the legacy quality guard still rejects formal pieces
    on a workout brief at the agent_orchestration level."""
    outfit = _outfit_with_meta({
        "top": {"name": "Black Blazer", "category": "blazer"},
        "bottom": {"name": "Wool Trousers", "category": "trousers"},
        "footwear": {"name": "Leather Loafers", "category": "loafers"},
    })
    outfit["agent_orchestration"] = {
        "occasion": "workout",
        "formality": "low",
        "avoid_items": ["blazer", "trousers", "loafers"],
        "required_slots": ["top", "bottom", "footwear"],
        "confidence": 0.9,
    }
    allowed, _, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"}, intent="workout", query="gym workout"
    )
    assert allowed is False
    assert reasons


def test_workout_badge_never_office_via_style_brief():
    from brain.engines.style_brief import safe_badge_for
    brief = build_brief(query="gym workout")
    assert safe_badge_for(brief) in {"GYM", "WORKOUT", "TRAINING"}
