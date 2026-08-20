"""Deterministic reasoning helper shared by initial Style This build and
post-shuffle regeneration (fix/style-this-shuffle-reasoning)."""

from services.style_board_reasoning import FALLBACK_STYLING_NOTE, build_styling_note


def _item(item_id, name):
    return {"item_id": item_id, "name": name}


def test_no_strategy_produces_deterministic_fallback_text():
    anchor = _item("anchor-1", "Gold Bracelet")
    items = [anchor, _item("shirt-1", "White Shirt")]

    assert build_styling_note(anchor, items, None) == FALLBACK_STYLING_NOTE
    assert build_styling_note(anchor, items, {}) == FALLBACK_STYLING_NOTE


def test_reason_references_support_item_and_strategy_fields():
    anchor = _item("anchor-1", "Gold Bracelet")
    items = [anchor, _item("shirt-1", "Green Flannel Shirt"), _item("shoe-1", "White Sneakers")]
    strategy = {"direction_title": "Refined Weekend", "reasoning_intent": "easy, intentional"}

    note = build_styling_note(anchor, items, strategy)

    assert "Green Flannel Shirt" in note
    assert "Gold Bracelet" in note
    assert "Refined Weekend" in note
    assert "easy and intentional" in note


def test_reason_skips_the_anchor_itself_when_choosing_the_support_item():
    anchor = _item("anchor-1", "Gold Bracelet")
    # Anchor listed first, as _lite_build_outfit/out_items both do.
    items = [anchor, _item("shirt-1", "Blue Oxford Shirt")]
    strategy = {"direction_title": "Polished Casual", "reasoning_intent": "confident"}

    note = build_styling_note(anchor, items, strategy)

    assert "Blue Oxford Shirt" in note
    assert note.startswith("Blue Oxford Shirt complements Gold Bracelet")


def test_reason_falls_back_to_generic_support_text_when_no_named_support_item():
    anchor = _item("anchor-1", "Gold Bracelet")
    items = [anchor]  # nothing else on the board
    strategy = {"direction_title": "Refined Weekend", "reasoning_intent": "easy"}

    note = build_styling_note(anchor, items, strategy)

    assert "the supporting pieces" in note
