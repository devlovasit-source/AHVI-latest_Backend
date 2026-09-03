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


def test_changed_footwear_references_the_actual_changed_item():
    anchor = {"item_id": "shirt-1", "name": "Blue Shirt", "slot": "top"}
    items = [
        anchor,
        {"item_id": "bottom-1", "name": "Black Trousers", "slot": "bottom"},
        {"item_id": "shoe-2", "name": "White Sneakers", "slot": "footwear"},
    ]
    strategy = {"direction_title": "Fresh Edit", "reasoning_intent": "easy"}

    note = build_styling_note(anchor, items, strategy, changed_slots=["footwear"])

    assert "White Sneakers" in note
    assert "Black Trousers" not in note


def test_changed_bottom_references_only_the_changed_slot():
    anchor = {"item_id": "shirt-1", "name": "Blue Shirt", "slot": "top"}
    items = [
        anchor,
        {"item_id": "bottom-2", "name": "Grey Jeans", "slot": "bottom"},
        {"item_id": "shoe-1", "name": "Brown Loafers", "slot": "footwear"},
    ]
    strategy = {"direction_title": "Grounded Edit", "reasoning_intent": "balanced"}

    note = build_styling_note(anchor, items, strategy, changed_slots=["bottom"])

    assert "Grey Jeans" in note
    assert "Brown Loafers" not in note


def test_multiple_changed_slots_use_changed_slot_order_deterministically():
    anchor = {"item_id": "shirt-1", "name": "Blue Shirt", "slot": "top"}
    items = [
        anchor,
        {"item_id": "accessory-1", "name": "Leather Belt", "slot": "accessory"},
        {"item_id": "shoe-2", "name": "White Sneakers", "slot": "footwear"},
        {"item_id": "bottom-2", "name": "Grey Jeans", "slot": "bottom"},
    ]
    strategy = {"direction_title": "Two-Part Edit", "reasoning_intent": "intentional"}

    first = build_styling_note(anchor, items, strategy, changed_slots=["bottom", "footwear"])
    second = build_styling_note(anchor, items, strategy, changed_slots=["bottom", "footwear"])

    assert first == second
    assert first.startswith("Grey Jeans complements Blue Shirt")


def test_no_changed_slots_preserves_existing_fallback_behavior():
    anchor = {"item_id": "shirt-1", "name": "Blue Shirt", "slot": "top"}
    items = [
        anchor,
        {"item_id": "bottom-1", "name": "Black Trousers", "slot": "bottom"},
    ]
    strategy = {"direction_title": "Fallback Edit", "reasoning_intent": "easy"}

    note = build_styling_note(anchor, items, strategy)

    assert note.startswith("Black Trousers complements Blue Shirt")


def test_anchor_is_never_selected_as_changed_item():
    anchor = {"item_id": "shirt-1", "name": "Blue Shirt", "slot": "top"}
    items = [
        anchor,
        {"item_id": "shoe-2", "name": "White Sneakers", "slot": "footwear"},
    ]
    strategy = {"direction_title": "Protected Edit", "reasoning_intent": "safe"}

    note = build_styling_note(anchor, items, strategy, changed_slots=["top", "footwear"])

    assert note.startswith("White Sneakers complements Blue Shirt")


def test_locked_unchanged_item_is_never_presented_as_the_mutation():
    anchor = {"item_id": "shirt-1", "name": "Blue Shirt", "slot": "top"}
    items = [
        anchor,
        {"item_id": "bottom-1", "name": "Locked Trousers", "slot": "bottom", "locked": True},
        {"item_id": "shoe-2", "name": "White Sneakers", "slot": "footwear"},
    ]
    strategy = {"direction_title": "Unlocked Edit", "reasoning_intent": "focused"}

    note = build_styling_note(anchor, items, strategy, changed_slots=["footwear"])

    assert note.startswith("White Sneakers complements Blue Shirt")
    assert "Locked Trousers" not in note
