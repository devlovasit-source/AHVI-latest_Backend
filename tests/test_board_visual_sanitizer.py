"""Visual board sanitizer acceptance tests.

Covers ``sanitize_board_items_for_visual_board`` (P0 fix): a raw wardrobe
dump must be repaired into a deduped, family-capped, slot-based board before
it can reach the frontend. No raw wardrobe list with duplicate
bottoms/belts/images/names may produce a visual board.
"""

from __future__ import annotations

from services.style_reasoning_engine import sanitize_board_items_for_visual_board


def _item(name: str, image_url: str = "", category: str = "", _id: str = ""):
    return {
        "name": name,
        "image_url": image_url or f"https://img/{name.replace(' ', '_').lower()}.png",
        "category": category,
        "id": _id or name.lower(),
    }


def _roles(slots):
    return [i["role"] for i in slots["items"]]


def _count_role(slots, role):
    return sum(1 for i in slots["items"] if i["role"] == role)


def _count_family(slots, family):
    return sum(1 for i in slots["items"] if i.get("family") == family)


def test_board_rejects_two_bottomwear():
    slots = sanitize_board_items_for_visual_board(
        [
            _item("White Oxford Shirt"),
            _item("Grey Trousers"),
            _item("Navy Chinos"),
            _item("Black Loafers"),
        ],
        occasion="office",
    )
    assert slots is not None
    assert _count_role(slots, "bottom") == 1


def test_board_rejects_two_belts():
    slots = sanitize_board_items_for_visual_board(
        [
            _item("White Oxford Shirt"),
            _item("Grey Trousers"),
            _item("Black Loafers"),
            _item("Brown Leather Belt"),
            _item("Black Leather Belt"),
        ],
        occasion="office",
    )
    assert slots is not None
    assert _count_family(slots, "belt") == 1


def test_board_dedupes_duplicate_image_url():
    dup = "https://img/same.png"
    slots = sanitize_board_items_for_visual_board(
        [
            _item("White Oxford Shirt", image_url=dup, _id="a"),
            _item("Grey Trousers"),
            _item("Black Loafers"),
            _item("Blue Shirt", image_url=dup, _id="b"),
        ],
        occasion="office",
    )
    assert slots is not None
    urls = [i["image_url"] for i in slots["items"]]
    assert len(urls) == len(set(urls))


def test_board_dedupes_duplicate_normalized_name():
    slots = sanitize_board_items_for_visual_board(
        [
            _item("White Shirt", _id="a"),
            _item("white shirt", _id="b"),  # same normalized name
            _item("Grey Trousers"),
            _item("Black Loafers"),
        ],
        occasion="office",
    )
    assert slots is not None
    names = [i["name"].strip().lower() for i in slots["items"]]
    assert len(names) == len(set(names))


def test_office_board_includes_top_bottom_footwear():
    slots = sanitize_board_items_for_visual_board(
        [
            _item("White Oxford Shirt"),
            _item("Grey Trousers"),
            _item("Black Loafers"),
        ],
        occasion="office",
    )
    assert slots is not None
    assert {"top", "bottom", "footwear"} <= set(_roles(slots))


def test_dress_board_includes_dress_and_footwear_excludes_top_bottom():
    slots = sanitize_board_items_for_visual_board(
        [
            _item("Floral Maxi Dress"),
            _item("Strappy Heels", category="footwear"),
            _item("White Oxford Shirt"),
            _item("Grey Trousers"),
        ],
        occasion="brunch",
    )
    assert slots is not None
    roles = set(_roles(slots))
    assert "dress" in roles
    assert _count_role(slots, "footwear") == 1
    assert "top" not in roles
    assert "bottom" not in roles


def test_accessory_only_board_is_rejected():
    slots = sanitize_board_items_for_visual_board(
        [
            _item("Brown Leather Belt"),
            _item("Steel Watch"),
            _item("Aviator Sunglasses"),
        ],
        occasion="office",
    )
    assert slots is None


def test_raw_dump_with_three_bottoms_two_belts_has_no_duplicates():
    slots = sanitize_board_items_for_visual_board(
        [
            _item("White Oxford Shirt"),
            _item("Grey Trousers"),
            _item("Navy Chinos"),
            _item("Blue Jeans"),
            _item("Brown Leather Belt"),
            _item("Black Leather Belt"),
            _item("Black Loafers"),
        ],
        occasion="office",
    )
    # Either rejected, or repaired with at most one bottom and one belt.
    if slots is not None:
        assert _count_role(slots, "bottom") == 1
        assert _count_family(slots, "belt") == 1
        assert len(slots["accessories"]) <= 2


def test_accessories_capped_at_two():
    slots = sanitize_board_items_for_visual_board(
        [
            _item("White Oxford Shirt"),
            _item("Grey Trousers"),
            _item("Black Loafers"),
            _item("Brown Leather Belt"),
            _item("Steel Watch"),
            _item("Aviator Sunglasses"),
            _item("Leather Tie"),
        ],
        occasion="office",
    )
    assert slots is not None
    assert len(slots["accessories"]) <= 2
