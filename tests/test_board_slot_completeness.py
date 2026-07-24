"""Complete-outfit slot enforcement.

Rules: complete = (top + bottom + footwear) OR (one-piece/dress + footwear).
Accessories never satisfy a required slot. Missing slots -> repair from safe
assets (respecting source policy) or reject. Wardrobe-only repair must never
pull catalog/style assets.
"""
from brain.engines.outfit_quality_guard import (
    is_complete_board,
    missing_required_slots,
    repair_or_reject_board,
)


def _i(role, source="wardrobe", name=None):
    return {"item_id": f"{role}-{source}", "role": role, "source": source, "name": name or role}


def test_top_bottom_shoes_is_complete():
    items = [_i("top"), _i("bottom"), _i("footwear")]
    assert is_complete_board(items) is True
    assert missing_required_slots(items) == []


def test_dress_plus_shoes_is_complete():
    items = [_i("dress"), _i("footwear")]
    assert is_complete_board(items) is True
    assert missing_required_slots(items) == []


def test_ethnic_one_piece_does_not_require_top_or_bottom():
    # saree/lehenga/gown/jumpsuit all classify as role "dress".
    for one_piece in ("saree", "lehenga", "gown", "jumpsuit"):
        items = [
            {"item_id": f"{one_piece}-1", "category": one_piece, "source": "wardrobe"},
            _i("footwear"),
        ]
        assert is_complete_board(items) is True, one_piece
        assert missing_required_slots(items) == []


def test_missing_top_is_incomplete():
    items = [_i("bottom"), _i("footwear")]
    assert is_complete_board(items) is False
    assert "top" in missing_required_slots(items)


def test_missing_bottom_is_incomplete():
    items = [_i("top"), _i("footwear")]
    assert is_complete_board(items) is False
    assert "bottom" in missing_required_slots(items)


def test_missing_footwear_is_incomplete():
    items = [_i("top"), _i("bottom")]
    assert is_complete_board(items) is False
    assert "footwear" in missing_required_slots(items)


def test_two_accessories_do_not_replace_top():
    items = [_i("bottom"), _i("footwear"), _i("accessory", name="watch"), _i("accessory", name="belt")]
    assert is_complete_board(items) is False
    assert "top" in missing_required_slots(items)


def test_repair_fills_missing_top_from_pool():
    card = {"items": [_i("bottom"), _i("footwear")]}
    pool = [_i("top")]
    fixed, status = repair_or_reject_board(card, source_policy="wardrobe_only", candidate_pool=pool)
    assert status == "repaired"
    assert is_complete_board(fixed["items"]) is True


def test_reject_when_no_safe_replacement():
    card = {"items": [_i("bottom"), _i("footwear")]}
    fixed, status = repair_or_reject_board(card, source_policy="wardrobe_only", candidate_pool=[])
    assert fixed is None
    assert status == "rejected_incomplete"


def test_wardrobe_only_repair_never_pulls_catalog_assets():
    card = {"items": [_i("bottom"), _i("footwear")]}
    # Only a CATALOG/style-asset top is available -> wardrobe-only must not use it.
    pool = [_i("top", source="style_asset")]
    fixed, status = repair_or_reject_board(card, source_policy="wardrobe_only", candidate_pool=pool)
    assert fixed is None
    assert status == "rejected_incomplete"
    # But an inspiration/mixed board MAY use it.
    fixed2, status2 = repair_or_reject_board(card, source_policy="style_asset", candidate_pool=pool)
    assert status2 == "repaired"
    assert is_complete_board(fixed2["items"]) is True
