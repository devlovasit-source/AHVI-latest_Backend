"""Accessory occasion-fit scrub.

Wrong accessory must not survive on an office/client_meeting board.
Workout boards keep their water bottle. Beach keeps swim cap.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from brain.engines.style_brief import (
    build_brief,
    core_outfit_complete,
    is_accessory_forbidden,
    strip_forbidden_accessories,
)


def _accessory(name: str, category: str = "accessory") -> Dict[str, Any]:
    return {"id": name, "name": name, "category": category, "role": "accessory"}


def _core(top: str = "White Shirt", bottom: str = "Grey Pants", footwear: str = "Black Loafers") -> List[Dict[str, Any]]:
    return [
        {"id": "t1", "name": top, "category": "button down", "role": "top"},
        {"id": "b1", "name": bottom, "category": "trousers", "role": "bottom"},
        {"id": "f1", "name": footwear, "category": "loafers", "role": "footwear"},
    ]


def _board(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"id": "board_x", "title": "Test", "items": list(items)}


# ---------------------------------------------------------------------------
# is_accessory_forbidden
# ---------------------------------------------------------------------------

def test_office_brief_flags_swim_cap_water_bottle():
    brief = build_brief(router_occasion="office", query="office today")
    for name in ("Swim Cap", "Swimming Goggles", "Water Bottle", "Gym Bottle"):
        forbidden, reason = is_accessory_forbidden(_accessory(name), brief)
        assert forbidden, f"{name} should be forbidden for office"
        assert "occasion_incompatible" in reason


def test_office_brief_keeps_watch_belt_ring_bag():
    brief = build_brief(router_occasion="office", query="office today")
    for name in ("Leather Watch", "Brown Belt", "Simple Ring", "Leather Bag"):
        forbidden, _ = is_accessory_forbidden(_accessory(name), brief)
        assert not forbidden, f"{name} should be allowed for office"


def test_workout_brief_keeps_water_bottle():
    brief = build_brief(router_occasion="workout", query="gym workout")
    forbidden, _ = is_accessory_forbidden(_accessory("Water Bottle"), brief)
    assert not forbidden


def test_workout_brief_rejects_tie_and_briefcase():
    brief = build_brief(router_occasion="workout", query="gym workout")
    for name in ("Silk Tie", "Leather Briefcase"):
        forbidden, _ = is_accessory_forbidden(_accessory(name), brief)
        assert forbidden


def test_beach_brief_keeps_swim_cap_and_goggles():
    brief = build_brief(router_occasion="beach", query="beach today")
    for name in ("Swim Cap", "Swim Goggles", "Snorkel"):
        forbidden, _ = is_accessory_forbidden(_accessory(name), brief)
        assert not forbidden


def test_client_meeting_brief_flags_swim_kit():
    brief = build_brief(router_occasion="client_meeting", query="client meeting")
    for name in ("Swim Cap", "Swimming Goggles", "Water Bottle"):
        forbidden, _ = is_accessory_forbidden(_accessory(name), brief)
        assert forbidden


# ---------------------------------------------------------------------------
# strip_forbidden_accessories — core stays intact
# ---------------------------------------------------------------------------

def test_office_board_strips_swim_cap_and_water_bottle_keeps_core():
    brief = build_brief(router_occasion="office", query="office today")
    board = _board(
        _core()
        + [
            _accessory("Swim Cap"),
            _accessory("Water Bottle"),
            _accessory("Leather Watch"),
        ]
    )
    cleaned, removed = strip_forbidden_accessories(board, brief)
    removed_names = sorted(r["name"] for r in removed)
    kept_names = sorted(i["name"] for i in cleaned["items"])
    assert removed_names == ["Swim Cap", "Water Bottle"]
    assert "Leather Watch" in kept_names
    assert "White Shirt" in kept_names
    assert "Grey Pants" in kept_names
    assert "Black Loafers" in kept_names
    assert core_outfit_complete(cleaned) is True


def test_client_meeting_board_strips_gold_ring_only_if_in_forbidden_set():
    """Gold Ring is NOT in the office/client forbidden list — must be kept."""
    brief = build_brief(router_occasion="client_meeting", query="client meeting")
    board = _board(_core() + [_accessory("Gold Ring")])
    cleaned, removed = strip_forbidden_accessories(board, brief)
    assert not removed
    assert any(i["name"] == "Gold Ring" for i in cleaned["items"])


def test_workout_board_keeps_water_bottle_after_scrub():
    brief = build_brief(router_occasion="workout", query="gym workout")
    board = _board(
        [
            {"id": "t1", "name": "Training Tee", "category": "tee", "role": "top"},
            {"id": "b1", "name": "Track Pants", "category": "track", "role": "bottom"},
            {"id": "f1", "name": "Running Sneakers", "category": "sneakers", "role": "footwear"},
            _accessory("Water Bottle"),
            _accessory("Silk Tie"),  # this is forbidden for workout
        ]
    )
    cleaned, removed = strip_forbidden_accessories(board, brief)
    kept_names = [i["name"] for i in cleaned["items"]]
    assert "Water Bottle" in kept_names
    assert "Silk Tie" not in kept_names
    assert removed and removed[0]["name"] == "Silk Tie"
    assert core_outfit_complete(cleaned) is True


def test_beach_board_keeps_swim_cap():
    brief = build_brief(router_occasion="beach", query="beach today")
    board = _board(
        [
            {"id": "t1", "name": "Linen Shirt", "category": "shirt", "role": "top"},
            {"id": "b1", "name": "Linen Shorts", "category": "shorts", "role": "bottom"},
            {"id": "f1", "name": "Sandals", "category": "sandals", "role": "footwear"},
            _accessory("Swim Cap"),
        ]
    )
    cleaned, removed = strip_forbidden_accessories(board, brief)
    assert not removed
    assert any(i["name"] == "Swim Cap" for i in cleaned["items"])


def test_scrub_preserves_existing_non_accessory_signals():
    """Scrub must not touch top/bottom/footwear even if their names contain
    forbidden tokens — that's the item-level guard's job, not the
    accessory scrub's."""
    brief = build_brief(router_occasion="office", query="office today")
    board = _board(
        [
            {"id": "t1", "name": "Beach Print Shirt", "category": "shirt", "role": "top"},
            {"id": "b1", "name": "Grey Pants", "category": "trousers", "role": "bottom"},
            {"id": "f1", "name": "Loafers", "category": "loafers", "role": "footwear"},
        ]
    )
    cleaned, removed = strip_forbidden_accessories(board, brief)
    # No accessories present — nothing removed.
    assert not removed
    assert any(i["name"] == "Beach Print Shirt" for i in cleaned["items"])
