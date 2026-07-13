"""Style Flow acceptance — 10 canonical prompts.

Verifies each high-level user intent flows through:
- correct brief occasion
- correct badge family
- no wrong-occasion items leaking through
- capsule short-circuits the normal outfit pipeline
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from brain.engines.style_brief import (
    build_brief,
    detect_occasion_from_tokens,
    safe_badge_for,
    validate_board,
)
from brain.engines.capsule_engine import looks_like_capsule_request, build_capsule_response


# ---------------------------------------------------------------------------
# Occasion detection for each canonical prompt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("outfit for client meeting", "client_meeting"),
        ("office outfit today", "office"),
        ("date night dinner", "date_night"),
        ("party look tonight", "party"),
        ("build me a capsule wardrobe", "capsule"),
        ("swimming costume for the pool", "swimming"),
        ("beach outfit for vacation", "beach"),
        ("workout outfit for gym", "workout"),
        ("wedding outfit for ceremony", "wedding"),
        ("casual look for weekend coffee", "casual"),
    ],
)
def test_canonical_prompts_detect_correct_occasion(prompt: str, expected: str):
    occ, _ = detect_occasion_from_tokens(prompt)
    assert occ == expected, f"{prompt!r} resolved to {occ}, expected {expected}"


# ---------------------------------------------------------------------------
# Each occasion produces a brief with the right badge family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "occasion,badge_family",
    [
        ("client_meeting", {"CLIENT READY", "BOARDROOM", "OFFICE"}),
        ("office", {"OFFICE", "BOARDROOM"}),
        ("date_night", {"DATE NIGHT", "DINNER", "EVENING"}),
        ("party", {"PARTY", "AFTER HOURS"}),
        ("workout", {"GYM", "WORKOUT", "TRAINING"}),
        ("travel", {"TRAVEL", "TRANSIT"}),
        ("beach", {"BEACH", "COASTAL"}),
        ("swimming", {"SWIM", "POOL"}),
        ("wedding", {"WEDDING", "CEREMONY", "FORMAL"}),
        ("capsule", {"CAPSULE", "ESSENTIALS"}),
    ],
)
def test_each_occasion_has_safe_badge(occasion: str, badge_family: set):
    brief = build_brief(router_occasion=occasion)
    badge = safe_badge_for(brief)
    assert badge in badge_family, f"{occasion} → {badge} not in {badge_family}"


# ---------------------------------------------------------------------------
# Wrong-occasion items must be rejected by validate_board
# ---------------------------------------------------------------------------

def _board(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"id": "b1", "title": "Test", "items": items}


def _item(role: str, name: str, category: str) -> Dict[str, Any]:
    return {"id": name, "name": name, "category": category, "role": role}


def test_client_meeting_rejects_shorts_and_slides():
    brief = build_brief(router_occasion="client_meeting")
    board = _board([
        _item("top", "Cotton Shirt", "shirt"),
        _item("bottom", "Beach Shorts", "shorts"),
        _item("footwear", "Slides", "slides"),
    ])
    passes, reasons, _ = validate_board(board, brief)
    assert not passes
    assert any("forbidden_signal" in r for r in reasons)


def test_workout_rejects_office_outfit():
    brief = build_brief(router_occasion="workout")
    board = _board([
        _item("top", "Shirt", "button down"),
        _item("bottom", "Wool Trousers", "trousers"),
        _item("footwear", "Loafers", "loafers"),
    ])
    passes, reasons, _ = validate_board(board, brief)
    assert not passes


def test_swimming_rejects_blazer_and_trousers():
    brief = build_brief(router_occasion="swimming")
    board = _board([
        _item("top", "Blazer", "blazer"),
        _item("bottom", "Wool Trousers", "trousers"),
        _item("footwear", "Loafers", "loafers"),
    ])
    passes, _, _ = validate_board(board, brief)
    assert not passes


def test_date_night_rejects_office_heavy():
    brief = build_brief(router_occasion="date_night")
    board = _board([
        _item("top", "Office Shirt", "shirt"),
        _item("bottom", "Boardroom Trouser", "trousers"),
        _item("footwear", "Loafer", "loafer"),
    ])
    passes, _, _ = validate_board(board, brief)
    assert not passes


def test_capsule_rejects_sequined_statement_pieces():
    brief = build_brief(router_occasion="capsule")
    board = _board([
        _item("top", "Sequined Top", "top"),
        _item("bottom", "Trousers", "trousers"),
        _item("footwear", "Loafers", "loafers"),
    ])
    passes, _, _ = validate_board(board, brief)
    assert not passes


# ---------------------------------------------------------------------------
# Capsule short-circuit detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "build me a capsule wardrobe",
        "capsule wardrobe for the week",
        "wardrobe essentials please",
        "minimal wardrobe edit",
        "core wardrobe foundation",
    ],
)
def test_capsule_trigger_phrases_detected(prompt: str):
    assert looks_like_capsule_request(prompt) is True


def test_normal_phrase_does_not_trigger_capsule():
    assert looks_like_capsule_request("outfit for client meeting") is False
    assert looks_like_capsule_request("dinner date") is False


# ---------------------------------------------------------------------------
# Capsule engine output shape — even on a tiny wardrobe
# ---------------------------------------------------------------------------

def _tiny_wardrobe() -> List[Dict[str, Any]]:
    return [
        {"id": "t1", "name": "White Shirt", "category": "shirt"},
        {"id": "t2", "name": "Navy Tee", "category": "tee"},
        {"id": "t3", "name": "Black Polo", "category": "polo"},
        {"id": "b1", "name": "Grey Trousers", "category": "trousers"},
        {"id": "b2", "name": "Navy Chinos", "category": "chinos"},
        {"id": "f1", "name": "Brown Loafers", "category": "loafers"},
        {"id": "f2", "name": "White Sneakers", "category": "sneakers"},
        {"id": "o1", "name": "Navy Blazer", "category": "blazer"},
        {"id": "a1", "name": "Leather Watch", "category": "watch"},
    ]


def test_capsule_response_returns_foundation_and_sample_looks():
    resp = build_capsule_response(user_id="u1", wardrobe=_tiny_wardrobe())
    assert resp["type"] == "capsule_wardrobe"
    assert resp["success"] is True
    assert resp["cards"], "capsule should return sample_looks as cards"
    data = resp["data"]
    assert data["capsule_foundation"], "foundation list populated"
    assert isinstance(data["sample_looks"], list)
    assert data["styling_note"]


def test_capsule_empty_wardrobe_does_not_crash():
    resp = build_capsule_response(user_id="u1", wardrobe=[])
    assert resp["type"] == "capsule_wardrobe"
    assert resp["success"] is False
    assert resp["cards"] == []
    assert resp["data"]["missing_slots"]


def test_capsule_metadata_signals_prefer_versatile_items():
    """Items with high capsule_score should rise in the foundation."""
    wardrobe = [
        {
            "id": "neutral_shirt",
            "name": "White Shirt",
            "category": "shirt",
            "style_metadata": {
                "capsule_score": 0.95,
                "versatility_score": 0.9,
                "visual_noise": "low",
            },
        },
        {
            "id": "novelty_shirt",
            "name": "Sequined Shirt",
            "category": "shirt",
            "style_metadata": {
                "capsule_score": 0.10,
                "visual_noise": "high",
                "statement_level": "statement",
            },
        },
        {"id": "b1", "name": "Trousers", "category": "trousers"},
        {"id": "f1", "name": "Loafers", "category": "loafers"},
    ]
    resp = build_capsule_response(user_id="u1", wardrobe=wardrobe)
    foundation_tops = [
        f["name"] for f in resp["data"]["capsule_foundation"]
        if f.get("role") == "top"
    ]
    # The high-capsule-score shirt should appear; novelty may also appear
    # if we need to fill the quota but must not outrank.
    assert "White Shirt" in foundation_tops
    if "Sequined Shirt" in foundation_tops:
        assert foundation_tops.index("White Shirt") < foundation_tops.index("Sequined Shirt")


# ---------------------------------------------------------------------------
# Capsule foundation canonical schema + ranking signals (Batch 5 contract)
# ---------------------------------------------------------------------------

def test_capsule_foundation_entries_are_structured_objects():
    resp = build_capsule_response(user_id="u1", wardrobe=_tiny_wardrobe())
    foundation = resp["data"]["capsule_foundation"]
    assert foundation and isinstance(foundation, list)
    for entry in foundation:
        assert isinstance(entry, dict)
        for key in ("item_id", "name", "role", "category", "source", "owned", "style_metadata"):
            assert key in entry, f"missing {key} in {entry}"
        assert entry["role"] in {"top", "bottom", "footwear", "outerwear", "accessory", "dress"}
    # Display-only strings live in a separate field, one per entry.
    labels = resp["data"]["capsule_foundation_labels"]
    assert labels == [e["name"] for e in foundation]
    assert all(isinstance(l, str) for l in labels)


def test_capsule_ids_and_images_survive_normalization():
    wardrobe = [
        {"id": "t1", "name": "White Shirt", "category": "shirt",
         "image_url": "https://cdn.test/t1.png"},
        {"id": "b1", "name": "Grey Trousers", "category": "trousers",
         "normalized_url": "https://cdn.test/b1_norm.png"},
        {"id": "f1", "name": "Loafers", "category": "loafers"},
    ]
    resp = build_capsule_response(user_id="u1", wardrobe=wardrobe)
    by_id = {e["item_id"]: e for e in resp["data"]["capsule_foundation"]}
    assert by_id["t1"]["image_url"] == "https://cdn.test/t1.png"
    assert by_id["b1"]["image_url"] == "https://cdn.test/b1_norm.png"
    assert by_id["b1"]["role"] == "bottom"  # trousers never classified as top
    assert by_id["b1"]["category"] == "trousers"  # original category preserved
    assert "image_url" not in by_id["f1"]  # never fabricated


def test_capsule_visual_noise_and_statement_are_penalized():
    from brain.engines.capsule_engine import _capsule_item_score

    quiet = {"name": "Shirt", "category": "shirt",
             "style_metadata": {"visual_noise": "low"}}
    loud = {"name": "Shirt", "category": "shirt",
            "style_metadata": {"visual_noise": "high"}}
    statement = {"name": "Shirt", "category": "shirt",
                 "style_metadata": {"statement_level": "statement"}}
    assert _capsule_item_score(loud) < _capsule_item_score(quiet)
    assert _capsule_item_score(statement) < _capsule_item_score(quiet)


def test_capsule_missing_metadata_is_neutral_not_dominant():
    from brain.engines.capsule_engine import _capsule_item_score

    explicit_high = {"name": "Shirt", "category": "shirt",
                     "style_metadata": {"capsule_score": 0.95, "versatility_score": 0.9}}
    no_metadata = {"name": "Shirt", "category": "shirt"}
    explicit_low = {"name": "Shirt", "category": "shirt",
                    "style_metadata": {"capsule_score": 0.1, "visual_noise": "high"}}
    assert _capsule_item_score(explicit_low) < _capsule_item_score(no_metadata)
    assert _capsule_item_score(no_metadata) < _capsule_item_score(explicit_high)


def test_capsule_sample_looks_reference_foundation_items():
    resp = build_capsule_response(user_id="u1", wardrobe=_tiny_wardrobe())
    foundation_ids = {e["item_id"] for e in resp["data"]["capsule_foundation"]}
    assert resp["data"]["sample_looks"]
    for look in resp["data"]["sample_looks"]:
        for item in look["items"]:
            item_id = str(item.get("item_id") or item.get("id") or "")
            assert item_id in foundation_ids, (
                f"sample look references non-foundation item {item!r}"
            )


def test_capsule_insufficient_wardrobe_returns_missing_slot_guidance():
    resp = build_capsule_response(
        user_id="u1", wardrobe=[{"id": "t1", "name": "White Shirt", "category": "shirt"}]
    )
    assert resp["success"] is False
    assert "bottom" in resp["data"]["missing_slots"]
    assert "footwear" in resp["data"]["missing_slots"]
    assert resp["data"]["shopping_gaps"]
