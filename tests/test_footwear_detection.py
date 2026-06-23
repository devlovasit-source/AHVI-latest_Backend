"""Footwear detection widening (follow-up patch).

Name-only footwear (heels, sandals, pumps, ...) must resolve to the footwear
slot so the visual board sanitizer doesn't drop valid shoes and skip the
board. Bottom detection still wins first, so "flat front trousers" stays a
bottom (no false positive).
"""

from __future__ import annotations

import pytest

from services.style_reasoning_engine import (
    _hero_expected_slot,
    sanitize_board_items_for_visual_board,
)


def _item(name: str, category: str = ""):
    return {
        "name": name,
        "image_url": f"https://img/{name.replace(' ', '_').lower()}.png",
        "category": category,
        "id": name.lower(),
    }


@pytest.mark.parametrize(
    "name",
    [
        "Black Heels", "Strappy Heel", "Brown Sandals", "Leather Sandal",
        "Nude Pumps", "Suede Pump", "Ballet Flats", "Pointed Flat",
        "Tan Mules", "Heeled Mule", "Cork Wedges", "Espadrille Wedge",
        "Black Loafers", "White Sneaker", "Chelsea Boots", "Oxford Shoes",
    ],
)
def test_name_only_footwear_detected(name):
    assert _hero_expected_slot(name) == "footwear"


@pytest.mark.parametrize(
    "name, slot",
    [
        ("Grey Flat Front Trousers", "bottom"),  # "flat" present, bottom wins
        ("Flat Front Chinos", "bottom"),
        ("White Oxford Shirt", "top"),
        ("Navy Blazer", "blazer"),
        ("Steel Watch", None),  # unrelated, no footwear false positive
    ],
)
def test_no_footwear_false_positive(name, slot):
    assert _hero_expected_slot(name) == slot


def test_name_only_heels_make_dress_board_viable():
    slots = sanitize_board_items_for_visual_board(
        [
            _item("Floral Maxi Dress"),
            _item("Black Heels"),  # name-only, no category
        ],
        occasion="brunch",
    )
    assert slots is not None
    assert any(i["role"] == "footwear" for i in slots["items"])


def test_name_only_sandals_make_office_board_viable():
    slots = sanitize_board_items_for_visual_board(
        [
            _item("White Oxford Shirt"),
            _item("Grey Trousers"),
            _item("Brown Sandals"),  # name-only footwear
        ],
        occasion="office",
    )
    assert slots is not None
    assert {"top", "bottom", "footwear"} <= {i["role"] for i in slots["items"]}
