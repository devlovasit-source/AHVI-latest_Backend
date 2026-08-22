"""Tests for the AHVI Board Storyteller enrichment layer."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from brain.response.board_storyteller import board_storyteller, enrich_boards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _board(idx: int, **extras: Any) -> Dict[str, Any]:
    base = {"id": f"board_{idx}", "title": f"Look {idx + 1}", "items": []}
    base.update(extras)
    return base


def _outfit(idx: int, **extras: Any) -> Dict[str, Any]:
    base = {
        "id": f"board_{idx}",
        "score_meta": {"reasons": ["palette aligned"]},
        "score_breakdown": {"palette": 1.2, "style_graph": 0.8},
        "items": [],
    }
    base.update(extras)
    return base


# ---------------------------------------------------------------------------
# Core schema guarantees
# ---------------------------------------------------------------------------

def test_story_object_exists_on_every_board():
    boards = [_board(i) for i in range(3)]
    outfits = [_outfit(i) for i in range(3)]
    enriched = enrich_boards(boards, outfits, {"occasion": "office"})
    for b in enriched:
        assert isinstance(b.get("story"), dict)
        story = b["story"]
        for key in (
            "headline",
            "summary",
            "why",
            "personal_note",
            "occasion_fit",
            "tip",
            "role",
        ):
            assert key in story


def test_summary_is_short_one_sentence():
    enriched = enrich_boards(
        [_board(0)], [_outfit(0)], {"occasion": "office"}
    )
    summary = enriched[0]["story"]["summary"]
    assert summary
    assert len(summary) <= 140
    # Single sentence — at most one terminal punct.
    assert summary.count(".") + summary.count("!") + summary.count("?") <= 2


def test_why_and_tip_not_empty():
    enriched = enrich_boards(
        [_board(0)], [_outfit(0)], {"occasion": "office"}
    )
    story = enriched[0]["story"]
    assert story["why"]
    assert story["tip"]


def test_headline_is_short():
    enriched = enrich_boards(
        [_board(0)], [_outfit(0)], {"occasion": "client_meeting"}
    )
    headline = enriched[0]["story"]["headline"]
    assert headline
    assert len(headline.split()) <= 4


def test_old_compat_fields_present():
    enriched = enrich_boards(
        [_board(0)], [_outfit(0)], {"occasion": "office"}
    )
    b = enriched[0]
    assert b["why_it_works"] == b["story"]["why"]
    assert b["explanation"] == b["story"]["summary"]
    assert b["styling_tip"] == b["story"]["tip"]


# ---------------------------------------------------------------------------
# Role table
# ---------------------------------------------------------------------------

def test_role_table_office():
    boards = [_board(i) for i in range(3)]
    outfits = [_outfit(i) for i in range(3)]
    enriched = enrich_boards(boards, outfits, {"occasion": "office"})
    roles = [b["story"]["role"] for b in enriched]
    assert roles[0] == "Safest polished option"
    assert roles[1] == "More relaxed office option"
    assert roles[2] == "Sharper authority option"


def test_role_table_date():
    boards = [_board(i) for i in range(3)]
    outfits = [_outfit(i) for i in range(3)]
    enriched = enrich_boards(boards, outfits, {"occasion": "date_night"})
    roles = [b["story"]["role"] for b in enriched]
    assert "effortless" in roles[0].lower()
    assert "date-night" in roles[1].lower()
    assert "evening" in roles[2].lower()


def test_role_table_workout():
    boards = [_board(i) for i in range(3)]
    outfits = [_outfit(i) for i in range(3)]
    enriched = enrich_boards(boards, outfits, {"occasion": "workout"})
    roles = [b["story"]["role"] for b in enriched]
    assert roles[0].startswith("Clean performance")
    assert roles[1].startswith("Comfort-first")
    assert roles[2].startswith("Outdoor-ready")


def test_role_table_default_fallback_for_unknown_occasion():
    boards = [_board(i) for i in range(3)]
    outfits = [_outfit(i) for i in range(3)]
    enriched = enrich_boards(boards, outfits, {"occasion": "house_party"})
    roles = [b["story"]["role"] for b in enriched]
    assert roles[0] == "Safest polished option"
    assert roles[1] == "Softer alternate"
    assert roles[2] == "Bolder style move"


# ---------------------------------------------------------------------------
# Tone differs by occasion
# ---------------------------------------------------------------------------

def test_office_summary_reads_professional():
    enriched = enrich_boards(
        [_board(0)], [_outfit(0)], {"occasion": "client_meeting"}
    )
    summary = enriched[0]["story"]["summary"].lower()
    # Premium, composed register — no influencer hype.
    assert "sharp" in summary or "composed" in summary or "ready" in summary
    assert "!" not in summary


def test_date_story_softer_register():
    enriched = enrich_boards(
        [_board(0)], [_outfit(0)], {"occasion": "date_night"}
    )
    summary = enriched[0]["story"]["summary"].lower()
    assert "soft" in summary or "intentional" in summary or "easy" in summary


def test_workout_story_practical():
    enriched = enrich_boards(
        [_board(0)], [_outfit(0)], {"occasion": "workout"}
    )
    summary = enriched[0]["story"]["summary"].lower()
    assert "move" in summary or "performance" in summary or "built" in summary
    tip = enriched[0]["story"]["tip"].lower()
    assert tip


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------

def test_missing_metadata_still_returns_story():
    # Minimum-viable inputs: no score_meta, no breakdown, no occasion.
    enriched = enrich_boards(
        [{"id": "x"}], [{"id": "x"}], {}
    )
    assert enriched
    story = enriched[0]["story"]
    assert story["headline"]
    assert story["summary"]
    assert story["why"]
    assert story["role"]


def test_malformed_boards_do_not_crash():
    enriched = enrich_boards(
        [None, {"id": "ok"}, 42], [], {"occasion": "office"}
    )
    # First/last are passed through, middle gets enriched.
    assert enriched[0] is None
    assert isinstance(enriched[1], dict) and isinstance(enriched[1].get("story"), dict)
    assert enriched[2] == 42


def test_story_strings_polished_no_forbidden_starters():
    enriched = enrich_boards(
        [_board(0)], [_outfit(0)], {"occasion": "office"}
    )
    story = enriched[0]["story"]
    for key in ("summary", "why", "personal_note", "occasion_fit", "tip"):
        val = story.get(key, "")
        assert not val.lower().startswith("sure!")
        assert "here are some ideas" not in val.lower()


def test_agent_orchestration_occasion_used_when_board_missing():
    enriched = enrich_boards(
        [_board(0)],
        [_outfit(0)],
        {"agent_orchestration": {"occasion": "client_meeting"}},
    )
    assert enriched[0]["story"]["role"] == "Safest polished option"


def test_palette_direction_influences_fallback_tip():
    enriched = enrich_boards(
        [_board(0)],
        [_outfit(0, score_meta={}, score_breakdown={})],
        {"occasion": "house_party", "palette_direction": ["navy"]},
    )
    tip = enriched[0]["story"]["tip"].lower()
    assert "navy" in tip or "palette" in tip or "accent" in tip


# ---------------------------------------------------------------------------
# Premium board layout planner
# ---------------------------------------------------------------------------

from brain.response.board_storyteller import build_premium_board_layout  # noqa: E402


def _it(item_id, name, category):
    return {"id": item_id, "name": name, "category": category}


def test_dress_board_gets_premium_editorial_dress_preset():
    outfit = _outfit(0, items=[
        _it("dress-1", "Red Polka Dot Dress", "Dresses"),
        _it("sneak-1", "White Sneakers", "Footwear"),
        _it("watch-1", "Gold Watch", "Accessories"),
    ])
    layout = build_premium_board_layout(_board(0), outfit, {})
    assert layout["layout_preset"] == "premium_editorial_dress"


def test_dress_board_has_dress_hero_and_no_bottom_role():
    outfit = _outfit(0, items=[
        _it("dress-1", "Red Polka Dot Dress", "Dresses"),
        _it("jeans-1", "Blue Jeans", "Bottoms"),  # must NOT be placed with a dress
        _it("sneak-1", "White Sneakers", "Footwear"),
    ])
    layout = build_premium_board_layout(_board(0), outfit, {})
    roles = [c["role"] for c in layout["composition_items"]]
    ids = [c["id"] for c in layout["composition_items"]]
    assert "hero" in roles
    assert layout["hero_item_id"] == "dress-1"
    assert "jeans-1" not in ids, "bottoms must not be placed with a dress"


def test_top_bottom_board_gets_stack_preset():
    outfit = _outfit(0, items=[
        _it("shirt-1", "White Shirt", "Tops"),
        _it("trouser-1", "Stone Trousers", "Bottoms"),
        _it("shoe-1", "White Sneakers", "Footwear"),
    ])
    layout = build_premium_board_layout(_board(0), outfit, {})
    assert layout["layout_preset"] == "premium_top_bottom_stack"
    assert layout["hero_item_id"] == "shirt-1"


def test_single_item_board_gets_minimal_preset():
    outfit = _outfit(0, items=[_it("dress-1", "Red Polka Dot Dress", "Dresses")])
    layout = build_premium_board_layout(_board(0), outfit, {})
    assert layout["layout_preset"] == "premium_minimal_single_item"
    assert len(layout["composition_items"]) == 1
    assert layout["composition_items"][0]["role"] == "hero"


def test_composition_items_present_for_every_enriched_board():
    boards = [_board(i, items=[_it(f"d{i}", "Red Dress", "Dresses"),
                               _it(f"s{i}", "White Sneakers", "Footwear")]) for i in range(3)]
    outfits = [_outfit(i, items=boards[i]["items"]) for i in range(3)]
    enriched = enrich_boards(boards, outfits, {"occasion": "date"})
    for b in enriched:
        assert isinstance(b.get("composition_items"), list)
        assert b["composition_items"], "every rendered board must have composition_items"
        assert b.get("layout_preset", "").startswith("premium_")


def test_red_polka_dot_dress_does_not_place_brown_leather_shoes():
    outfit = _outfit(0, items=[
        _it("dress-1", "Red Polka Dot Dress", "Dresses"),
        _it("leather-1", "Brown Leather Shoes", "Footwear"),
    ])
    layout = build_premium_board_layout(_board(0), outfit, {})
    ids = [c["id"] for c in layout["composition_items"]]
    assert "leather-1" not in ids, "men's leather shoes must not support a dress"
    # A good missing suggestion is offered instead of the bad pairing.
    labels = " ".join(m.get("label", "").lower() for m in layout["missing_items"])
    assert any(g in labels for g in ("sneaker", "sandal", "flat"))
