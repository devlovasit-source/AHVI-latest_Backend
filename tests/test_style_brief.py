"""Style Brief — token-aware intent detection + brief building."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from brain.engines.style_brief import (
    build_brief,
    detect_occasion_from_tokens,
    safe_badge_for,
    select_board_set,
    tokenize,
    validate_board,
)


# ---------------------------------------------------------------------------
# Tokenization + occasion detection — the "workout / work" bug regression
# ---------------------------------------------------------------------------

def test_workout_query_creates_workout_brief():
    """The classic bug: 'workout' must never match 'work' inside 'office'."""
    occ, hits = detect_occasion_from_tokens("workout outfit suggestion for gym")
    assert occ == "workout"
    assert "workout" in hits or "gym" in hits


def test_work_outfit_still_normalizes_to_office():
    occ, _ = detect_occasion_from_tokens("work outfit for monday")
    assert occ == "office"


def test_client_meeting_phrase_beats_office():
    occ, hits = detect_occasion_from_tokens("outfit for client meeting tomorrow")
    assert occ == "client_meeting"
    assert "client_meeting" in hits or "client" in hits


def test_date_night_phrase_normalizes_correctly():
    occ, _ = detect_occasion_from_tokens("dinner date tonight")
    assert occ == "date_night"


def test_dating_app_does_not_match_date_token():
    # Whole-word matching: 'dating' should not collide with 'date'.
    occ, _ = detect_occasion_from_tokens("dating app profile")
    assert occ != "date_night"


def test_unknown_query_returns_empty():
    occ, hits = detect_occasion_from_tokens("xyzzy nonsense")
    assert occ == ""
    assert hits == []


def test_tokenize_collapses_known_phrases():
    tokens = tokenize("after-hours rooftop bar")
    assert "after_hours" in tokens or "rooftop_bar" in tokens


# ---------------------------------------------------------------------------
# Brief build — router precedence + agent override gates
# ---------------------------------------------------------------------------

def test_router_occasion_beats_low_confidence_agent():
    brief = build_brief(
        query="outfit for client meeting",
        router_occasion="client_meeting",
        agent_payload={"occasion": "casual", "confidence": 0.0},
    )
    assert brief["occasion"] == "client_meeting"
    assert brief["_provenance"]["chosen_from"] == "router"


def test_high_confidence_agent_overrides_router():
    brief = build_brief(
        query="outfit for tonight",
        router_occasion="office",
        agent_payload={"occasion": "date_night", "confidence": 0.85},
    )
    assert brief["occasion"] == "date_night"
    assert brief["_provenance"]["chosen_from"] == "agent_high_confidence"


def test_brief_falls_back_to_tokens_when_router_silent():
    brief = build_brief(query="gym training session")
    assert brief["occasion"] == "workout"


def test_workout_brief_forbids_office_signals():
    brief = build_brief(query="gym workout", router_occasion=None)
    forbidden = " ".join(brief["forbidden_item_signals"]).lower()
    for tok in ("loafer", "trouser", "button down", "blazer", "belt"):
        assert tok in forbidden


def test_client_meeting_brief_requires_high_polish():
    brief = build_brief(router_occasion="client_meeting", query="client meeting")
    assert brief["polish_requirement"] == "high"
    assert "shirt" in " ".join(brief["preferred_item_signals"])


def test_agent_avoid_items_extend_forbidden_signals():
    brief = build_brief(
        router_occasion="office",
        agent_payload={"avoid_items": ["sequined", "neon"], "confidence": 0.4},
    )
    forbidden = brief["forbidden_item_signals"]
    assert "sequined" in [s.lower() for s in forbidden]
    assert "neon" in [s.lower() for s in forbidden]


# ---------------------------------------------------------------------------
# Board validation
# ---------------------------------------------------------------------------

def _board(items: Any, **kw: Any) -> Dict[str, Any]:
    out = {"id": "b1", "title": "Test", "items": list(items)}
    out.update(kw)
    return out


def test_workout_brief_rejects_office_outfit():
    brief = build_brief(query="gym workout")
    board = _board(
        [
            {"id": "1", "name": "Crisp Shirt", "category": "button down", "role": "top"},
            {"id": "2", "name": "Wool Trousers", "category": "trousers", "role": "bottom"},
            {"id": "3", "name": "Leather Loafers", "category": "loafers", "role": "footwear"},
        ]
    )
    passes, reasons, _ = validate_board(board, brief)
    assert passes is False
    assert any("forbidden_signal" in r for r in reasons)


def test_client_meeting_rejects_shorts_and_slides():
    brief = build_brief(router_occasion="client_meeting", query="client meeting")
    board = _board(
        [
            {"id": "1", "name": "Cotton Shirt", "category": "shirt", "role": "top"},
            {"id": "2", "name": "Beach Shorts", "category": "shorts", "role": "bottom"},
            {"id": "3", "name": "Slides", "category": "slides", "role": "footwear"},
        ]
    )
    passes, reasons, _ = validate_board(board, brief)
    assert passes is False
    assert any("forbidden_signal" in r for r in reasons)


def test_date_night_rejects_office_heavy_combo():
    brief = build_brief(router_occasion="date_night", query="dinner date")
    board = _board(
        [
            {"id": "1", "name": "Office Crisp Shirt", "category": "shirt", "role": "top"},
            {"id": "2", "name": "Boardroom Trouser", "category": "trousers", "role": "bottom"},
            {"id": "3", "name": "Loafer", "category": "loafer", "role": "footwear"},
        ]
    )
    passes, _, _ = validate_board(board, brief)
    # Office/boardroom tokens are in date_night.forbidden_item_signals.
    assert passes is False


def test_office_board_passes_office_brief():
    brief = build_brief(router_occasion="office", query="office today")
    board = _board(
        [
            {"id": "1", "name": "Cotton Shirt", "category": "button down", "role": "top"},
            {"id": "2", "name": "Wool Trousers", "category": "trousers", "role": "bottom"},
            {"id": "3", "name": "Leather Loafers", "category": "loafers", "role": "footwear"},
        ]
    )
    passes, _, scores = validate_board(board, brief)
    assert passes is True
    assert scores["occasion_fit_score"] >= 0.6


# ---------------------------------------------------------------------------
# Set selection
# ---------------------------------------------------------------------------

def test_select_board_set_avoids_repeated_hero_when_alternatives_exist():
    brief = build_brief(router_occasion="office", query="office today")
    same_top = {"id": "shirt_a", "name": "Shirt A", "category": "button down", "role": "top"}
    other_top = {"id": "shirt_b", "name": "Shirt B", "category": "button down", "role": "top"}
    trouser_a = {"id": "trs_a", "name": "Trouser A", "category": "trousers", "role": "bottom"}
    trouser_b = {"id": "trs_b", "name": "Trouser B", "category": "trousers", "role": "bottom"}
    loafer = {"id": "shoe_a", "name": "Loafers", "category": "loafers", "role": "footwear"}

    cards = [
        {"id": "c1", "title": "A", "items": [same_top, trouser_a, loafer]},
        {"id": "c2", "title": "B", "items": [same_top, trouser_b, loafer]},
        {"id": "c3", "title": "C", "items": [other_top, trouser_b, loafer]},
    ]
    chosen = select_board_set(cards, brief, max_n=3)
    heroes = [c["items"][0]["id"] for c in chosen]
    assert len(set(heroes)) >= 2, "hero should diversify when alternatives exist"


def test_select_board_set_preserves_count_when_only_one_top():
    brief = build_brief(router_occasion="office", query="office today")
    top = {"id": "only_top", "name": "Shirt", "category": "button down", "role": "top"}
    trouser_a = {"id": "trs_a", "name": "Trouser A", "category": "trousers", "role": "bottom"}
    trouser_b = {"id": "trs_b", "name": "Trouser B", "category": "trousers", "role": "bottom"}
    loafer = {"id": "shoe_a", "name": "Loafers", "category": "loafers", "role": "footwear"}

    cards = [
        {"id": "c1", "title": "A", "items": [top, trouser_a, loafer]},
        {"id": "c2", "title": "B", "items": [top, trouser_b, loafer]},
    ]
    chosen = select_board_set(cards, brief, max_n=3)
    assert len(chosen) == 2  # don't drop boards when wardrobe is sparse


# ---------------------------------------------------------------------------
# Badge integrity
# ---------------------------------------------------------------------------

def test_safe_badge_is_brief_consistent():
    for occ in ("workout", "client_meeting", "date_night"):
        brief = build_brief(router_occasion=occ)
        badge = safe_badge_for(brief)
        assert badge in brief["allowed_badges"]


def test_workout_badge_never_office():
    brief = build_brief(query="gym workout")
    badge = safe_badge_for(brief)
    assert badge in {"GYM", "WORKOUT", "TRAINING"}
    assert badge != "OFFICE"
