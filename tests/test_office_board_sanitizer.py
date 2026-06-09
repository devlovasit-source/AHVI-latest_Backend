"""Office/client-meeting board sanitization tests.

Covers ``_sanitize_office_board``, ``_is_office_bad_item``, and
``_has_minimum_board_slots`` in services.style_flow_service.
"""

from __future__ import annotations

from services import style_flow_service as sfs


def _item(name: str, **extra):
    base = {"name": name, "category": "", "subcategory": ""}
    base.update(extra)
    return base


def test_client_meeting_keeps_board_after_removing_boxer():
    card = {
        "title": "Client Meeting Look",
        "items": [
            _item("White Oxford Shirt", category="top", subcategory="shirt"),
            _item("Grey Trouser", category="bottom", subcategory="trouser"),
            _item("Black Loafers", category="footwear", subcategory="loafer"),
            _item("Cotton Boxer", category="loungewear", subcategory="boxer"),
        ],
    }
    sanitized, removed = sfs._sanitize_office_board(card, "client_meeting")
    assert removed == ["Cotton Boxer"]
    assert len(sanitized["items"]) == 3
    assert sfs._has_minimum_board_slots(sanitized)


def test_client_meeting_rejects_board_when_no_core_slots_remain():
    card = {
        "title": "Bad Board",
        "items": [
            _item("Cotton Boxer", category="loungewear", subcategory="boxer"),
            _item("Pool Slides", category="footwear", subcategory="slides"),
        ],
    }
    sanitized, removed = sfs._sanitize_office_board(card, "client_meeting")
    # Both items flagged.
    assert set(removed) == {"Cotton Boxer", "Pool Slides"}
    assert not sfs._has_minimum_board_slots(sanitized)


def test_office_removes_swim_shorts_and_cap_and_sunglasses():
    card = {
        "title": "Office",
        "items": [
            _item("White Shirt", subcategory="shirt"),
            _item("Navy Trouser", subcategory="trouser"),
            _item("Brown Loafers", subcategory="loafer"),
            _item("Swim Shorts", subcategory="swim_shorts"),
            _item("Baseball Cap", subcategory="cap"),
            _item("Aviator Sunglasses", subcategory="sunglasses"),
        ],
    }
    sanitized, removed = sfs._sanitize_office_board(card, "office")
    assert {"Swim Shorts", "Baseball Cap", "Aviator Sunglasses"} <= set(removed)
    kept_names = {i["name"] for i in sanitized["items"]}
    assert {"White Shirt", "Navy Trouser", "Brown Loafers"} <= kept_names


def test_office_keeps_blazer_belt_watch_bag_shirt_pants_loafers():
    items = [
        _item("Navy Blazer", subcategory="blazer"),
        _item("Leather Belt", subcategory="belt"),
        _item("Steel Watch", subcategory="watch"),
        _item("Laptop Bag", subcategory="laptop_bag"),
        _item("White Shirt", subcategory="shirt"),
        _item("Grey Pants", subcategory="trouser"),
        _item("Black Formal Shoes", subcategory="formal_shoe"),
    ]
    card = {"title": "Pure Office", "items": items}
    sanitized, removed = sfs._sanitize_office_board(card, "presentation")
    assert removed == []
    assert sanitized is card  # untouched when nothing to strip


def test_beach_occasion_untouched_by_sanitizer():
    card = {
        "title": "Beach Day",
        "items": [
            _item("Linen Shirt", subcategory="shirt"),
            _item("Swim Shorts", subcategory="swim_shorts"),
            _item("Flip Flops", subcategory="flip_flops"),
            _item("Sunglasses", subcategory="sunglasses"),
        ],
    }
    sanitized, removed = sfs._sanitize_office_board(card, "beach")
    assert removed == []
    assert sanitized is card


def test_date_night_untouched_by_sanitizer():
    card = {
        "title": "Date Night",
        "items": [
            _item("Black Shirt", subcategory="shirt"),
            _item("Black Jeans", subcategory="jeans"),
            _item("Black Loafers", subcategory="loafer"),
        ],
    }
    sanitized, removed = sfs._sanitize_office_board(card, "date_night")
    assert removed == []


def test_is_office_occasion_matches_aliases():
    for occ in ("office", "Office", "CLIENT_MEETING", "client meeting", "interview"):
        assert sfs._is_office_occasion(occ)
    for occ in ("beach", "vacation", "date_night", ""):
        assert not sfs._is_office_occasion(occ)


def test_has_minimum_board_slots_requires_top_or_outer_plus_bottom_and_footwear():
    assert sfs._has_minimum_board_slots(
        {
            "items": [
                _item("Shirt", subcategory="shirt"),
                _item("Trouser", subcategory="trouser"),
                _item("Loafer", subcategory="loafer"),
            ]
        }
    )
    assert sfs._has_minimum_board_slots(
        {
            "items": [
                _item("Blazer", subcategory="blazer"),
                _item("Pants", subcategory="trouser"),
                _item("Loafer", subcategory="loafer"),
            ]
        }
    )
    assert not sfs._has_minimum_board_slots(
        {
            "items": [
                _item("Shirt", subcategory="shirt"),
                _item("Loafer", subcategory="loafer"),
            ]
        }
    )
