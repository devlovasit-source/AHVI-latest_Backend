"""Tests for style-board P0 fixes.
"""

from __future__ import annotations

from typing import Any, Dict, List
import pytest

from brain.outfit_pipeline import _ahvi_pipe_item_allowed
from services.style_flow_service import _select_diverse_cards, build_style_flow_response

# ---------------------------------------------------------------------------
# P0 Fix 1: Global inappropriate garment guard
# ---------------------------------------------------------------------------

def test_boxers_sleepwear_slippers_never_appear_in_office_or_casual_boards():
    # Contexts to test: office and casual
    contexts = [
        {"query": "something nice to wear to the office"},
        {"query": "casual outfit for the weekend"},
    ]
    
    # Inappropriate items
    items = [
        {"name": "silk boxers", "category": "underwear", "type": "bottom"},
        {"name": "comfy pajamas", "category": "sleepwear", "type": "bottom"},
        {"name": "fluffy slippers", "category": "footwear", "type": "footwear"},
        {"name": "loungewear pants", "category": "lounge", "type": "bottom"},
    ]
    
    for context in contexts:
        for item in items:
            # Should be blocked in both office and casual contexts
            assert _ahvi_pipe_item_allowed(item, context) is False


def test_lounge_sleepwear_allowed_only_when_explicitly_requested():
    # PJ party context for a male user
    pajama_context = {"query": "what to wear to a pajama party", "user_profile": {"gender": "male"}}
    lounge_context = {"query": "comfortable loungewear to lounge around at home", "user_profile": {"gender": "male"}}
    
    pajama_item = {"name": "comfy pajamas", "category": "sleepwear", "type": "bottom"}
    lounge_item = {"name": "loungewear pants", "category": "lounge", "type": "bottom"}
    
    # In a sleep/lounge context, these shouldn't be blocked by the global guard for a male user
    assert _ahvi_pipe_item_allowed(pajama_item, pajama_context) is True
    assert _ahvi_pipe_item_allowed(lounge_item, lounge_context) is True


@pytest.mark.parametrize(
    "item",
    [
        {"name": "boxer", "category": "bottom", "type": "bottom"},
        {"name": "underwear", "category": "underwear", "type": "bottom"},
        {"name": "slipper", "category": "footwear", "type": "footwear"},
        {"name": "boxer-briefs", "category": "underwear", "type": "bottom"},
    ],
)
def test_inappropriate_garment_guard_blocks_complete_tokens(item):
    assert _ahvi_pipe_item_allowed(item, {"query": "office outfit"}) is False


def test_boxer_fit_shirt_not_blocked_by_garment_guard():
    item = {"name": "boxer fit shirt", "category": "shirt", "type": "top"}
    assert _ahvi_pipe_item_allowed(item, {"query": "office outfit"}) is True


# ---------------------------------------------------------------------------
# P0 Fix 2: First 3 cards must be visually distinct
# ---------------------------------------------------------------------------

def test_first_3_cards_do_not_repeat_top_bottom_footwear_when_enough_items_exist():
    # Create 5 cards where there are duplicates but enough alternative unique choices exist
    cards = [
        # Card 1: Top T1, Bottom B1, Footwear F1
        {
            "title": "Look 1",
            "items": [
                {"id": "T1", "category": "shirt", "role": "top"},
                {"id": "B1", "category": "pants", "role": "bottom"},
                {"id": "F1", "category": "shoes", "role": "footwear"},
            ],
            "_style_core_signature": "T1|B1|F1",
            "_style_signature": "T1|B1|F1",
            "_style_quality_score": 90.0,
        },
        # Card 2: Reuses T1 (should be deferred if strict and alternative top exists)
        {
            "title": "Look 2 (Repeated top)",
            "items": [
                {"id": "T1", "category": "shirt", "role": "top"},
                {"id": "B2", "category": "jeans", "role": "bottom"},
                {"id": "F2", "category": "boots", "role": "footwear"},
            ],
            "_style_core_signature": "T1|B2|F2",
            "_style_signature": "T1|B2|F2",
            "_style_quality_score": 85.0,
        },
        # Card 3: Unique Top T2, Bottom B2, Footwear F2
        {
            "title": "Look 3",
            "items": [
                {"id": "T2", "category": "shirt", "role": "top"},
                {"id": "B2", "category": "jeans", "role": "bottom"},
                {"id": "F2", "category": "boots", "role": "footwear"},
            ],
            "_style_core_signature": "T2|B2|F2",
            "_style_signature": "T2|B2|F2",
            "_style_quality_score": 80.0,
        },
        # Card 4: Unique Top T3, Bottom B3, Footwear F3
        {
            "title": "Look 4",
            "items": [
                {"id": "T3", "category": "tee", "role": "top"},
                {"id": "B3", "category": "shorts", "role": "bottom"},
                {"id": "F3", "category": "sneakers", "role": "footwear"},
            ],
            "_style_core_signature": "T3|B3|F3",
            "_style_signature": "T3|B3|F3",
            "_style_quality_score": 75.0,
        },
    ]
    
    # Run diversification selection with a limit of 3
    selected = _select_diverse_cards(cards, "office today", limit=3)
    
    assert len(selected) == 3
    
    # Assert that all 3 selected cards have completely unique top, bottom, and footwear items
    selected_tops = [next(i["id"] for i in s["items"] if i["role"] == "top") for s in selected]
    selected_bottoms = [next(i["id"] for i in s["items"] if i["role"] == "bottom") for s in selected]
    selected_shoes = [next(i["id"] for i in s["items"] if i["role"] == "footwear") for s in selected]
    
    assert len(set(selected_tops)) == 3, f"Tops not unique: {selected_tops}"
    assert len(set(selected_bottoms)) == 3, f"Bottoms not unique: {selected_bottoms}"
    assert len(set(selected_shoes)) == 3, f"Footwear not unique: {selected_shoes}"


# ---------------------------------------------------------------------------
# P0 Fix 3: Remove blocking sync agent fallback
# ---------------------------------------------------------------------------

def test_style_response_does_not_block_on_sync_agent_fallback(monkeypatch):
    # Mock _agent_start_orchestration to always return None, ensuring the sync fallback branch is triggered.
    import services.style_flow_service as sfs
    monkeypatch.setattr(sfs, "_agent_start_orchestration", lambda *args, **kwargs: None)

    # Mock _agent_orchestrate_sync to detect if it's called (it should NOT be).
    called_sync_agent = False
    def mock_sync_agent(*args, **kwargs):
        nonlocal called_sync_agent
        called_sync_agent = True
        return {"occasion": "mocked_sync"}

    monkeypatch.setattr(sfs, "_agent_orchestrate_sync", mock_sync_agent)
    monkeypatch.setattr(sfs, "_agent_style_enabled", lambda: True)

    # Run build_style_flow_response. It should proceed without calling the sync agent.
    response = build_style_flow_response(
        user_id="test_user",
        query="pajama party",
        wardrobe=[],
        context={"chips": [], "weather": {}},
    )

    assert called_sync_agent is False, "Synchronous fallback agent was called when it should have been skipped."
    # Assert that no exception occurred (test passes if no assert fails before this point)

# ---------------------------------------------------------------------------
# P0 Fix 4: Safe brief bypass only
# ---------------------------------------------------------------------------

def test_brief_rejection_does_not_cause_fallback_when_safe_cards_exist(monkeypatch):
    # Mock select_board_set to return empty (simulating strict weather/aesthetic/brief rejection)
    import brain.engines.style_brief as sb
    monkeypatch.setattr(sb, "select_board_set", lambda cards, brief, max_n: [])
    
    # Mock wardrobe with core slots and finalized cards exist, including one unsafe card
    mock_cards = [
        {
            "id": "c1",
            "title": "Safe Look 1",
            "look_strategy": "safest_option",
            "items": [
                {"id": "t1", "category": "shirt", "role": "top", "image_url": "t1.jpg"},
                {"id": "b1", "category": "pants", "role": "bottom", "image_url": "b1.jpg"},
                {"id": "f1", "category": "shoes", "role": "footwear", "image_url": "f1.jpg"},
            ],
            "style_metadata": {"style_signature": "look_1"}
        },
        {
            "id": "c2_unsafe",
            "title": "Unsafe Look (Boxers)",
            "look_strategy": "unsafe",
            "items": [
                {"id": "t2", "category": "shirt", "role": "top", "image_url": "t2.jpg"},
                {"id": "b2", "category": "boxer", "role": "bottom", "image_url": "b2.jpg"},
                {"id": "f2", "category": "shoes", "role": "footwear", "image_url": "f2.jpg"},
            ],
            "style_metadata": {"style_signature": "look_2_unsafe"}
        }
    ]
    
    import services.style_flow_service as sfs
    # Mock finalized card generation to return our safe finalized cards
    monkeypatch.setattr(sfs, "finalize_style_cards", lambda *args, **kwargs: list(mock_cards))
    monkeypatch.setattr(sfs, "_ahvi_has_core_slots", lambda *args: True)
    import brain.outfit_pipeline as op
    monkeypatch.setattr(op, "get_daily_outfits", lambda *args: {"cards": list(mock_cards), "outfits": []})
    
    # Run build_style_flow_response and expect it to return only the safe cards
    # because of the brief bypass, and reject the unsafe one
    res = build_style_flow_response(
        user_id="test_user",
        query="office travel today", # Office query to trigger sanitize
        wardrobe=mock_cards[0]["items"], # Provide some items for context
        context={"chips": [], "weather": {}},
    )
    
    assert res["success"] is True
    assert len(res["cards"]) == 1
    assert res["cards"][0]["id"] == "c1"
    assert res["cards"][0]["id"] != "c2_unsafe", "Unsafe card should have been filtered out."
