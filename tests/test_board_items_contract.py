"""Itemized board_items contract for the frontend 85 flat-lay board.

Covers the additive contract built in
services.style_reasoning_engine._build_board_items:
  - non-wardrobe directions expose top/bottom/footwear with image URLs
  - wardrobe intent yields owned-only items (no generic 'asset' source)
  - items without an image_url are excluded (no fake data)
"""

from services.style_reasoning_engine import (
    _build_board_items,
    _board_items_viable,
)


def test_coffee_date_board_items_have_top_bottom_footwear():
    direction = {
        "hero_piece": "Soft Oxford Shirt",
        "image_url": "https://cdn/shirt.png",
        "complete_the_look": [
            {"name": "Relaxed Chinos", "image_url": "https://cdn/chino.png"},
            {"name": "Clean Sneakers", "image_url": "https://cdn/sneakers.png"},
            {"name": "Minimal Watch", "image_url": "https://cdn/watch.png"},
        ],
    }
    items = _build_board_items(direction, wardrobe_intent=False)
    roles = {i["role"] for i in items}
    assert {"top", "bottom", "footwear"}.issubset(roles)
    assert _board_items_viable(items)
    assert all(i["image_url"] for i in items)


def test_wardrobe_intent_board_items_are_owned_only():
    direction = {
        "hero_piece": "Soft Oxford Shirt",
        "image_url": "https://cdn/shirt.png",
        "owned_items": [
            {"name": "My Tee", "image_url": "https://cdn/wardrobe/tee.png"},
        ],
        "complete_the_look": [
            {"name": "Clean Sneakers", "image_url": "https://cdn/sneakers.png"},
        ],
    }
    items = _build_board_items(direction, wardrobe_intent=True)
    # Only owned wardrobe items — no generic asset leaks in.
    assert items, "expected the owned item to be included"
    assert all(i["source"] == "wardrobe" and i["owned"] for i in items)
    assert all("sneakers" not in i["image_url"] for i in items)


def test_items_without_image_url_excluded():
    direction = {
        "hero_piece": "Shirt",
        "complete_the_look": [{"name": "Pants"}],
    }
    assert _build_board_items(direction, wardrobe_intent=False) == []
