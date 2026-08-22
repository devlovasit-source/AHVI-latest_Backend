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
    _enrich_visual_directions_with_assets,
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


def test_board_items_use_board_image_url_before_catalog_image():
    direction = {
        "hero_piece": "Soft Oxford Shirt",
        "image_url": "https://cdn/catalog-shirt.jpg",
        "board_image_url": "https://cdn/cutout-shirt.png",
        "cutout_status": "ready",
        "complete_the_look": [
            {
                "name": "Relaxed Chinos",
                "image_url": "https://cdn/catalog-chino.jpg",
                "board_image_url": "https://cdn/cutout-chino.png",
                "cutout_status": "ready",
            },
            {
                "name": "Clean Sneakers",
                "image_url": "https://cdn/catalog-sneakers.jpg",
                "board_image_url": "https://cdn/cutout-sneakers.png",
                "cutout_status": "ready",
            },
        ],
    }

    items = _build_board_items(direction, wardrobe_intent=False)

    assert items[0]["image_url"] == "https://cdn/cutout-shirt.png"
    assert items[0]["board_image_url"] == "https://cdn/cutout-shirt.png"
    assert items[0]["catalog_image_url"] == "https://cdn/catalog-shirt.jpg"
    assert all(item["image_source"] == "board_image_url" for item in items)


def test_missing_cutout_marks_catalog_fallback():
    direction = {
        "hero_piece": "Soft Oxford Shirt",
        "image_url": "https://cdn/catalog-shirt.jpg",
        "complete_the_look": [
            {"name": "Relaxed Chinos", "image_url": "https://cdn/catalog-chino.jpg"},
            {"name": "Clean Sneakers", "image_url": "https://cdn/catalog-sneakers.jpg"},
        ],
    }

    items = _build_board_items(direction, wardrobe_intent=False)

    assert items
    assert all(item["board_status"] == "catalog_fallback" for item in items)


def test_visual_enrichment_prefers_cutout_ready_assets(monkeypatch):
    import services.style_reasoning_engine as engine

    monkeypatch.setattr(
        engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "catalog-shirt",
                "name": "Soft Oxford Shirt",
                "category": "top",
                "image_url": "https://cdn/catalog-shirt.jpg",
                "archetypes": ["Modern Professional"],
                "occasions": ["coffee date"],
                "gender": "unisex",
                "status": "active",
            },
            {
                "asset_id": "cutout-shirt",
                "name": "Soft Oxford Shirt",
                "category": "top",
                "image_url": "https://cdn/catalog-shirt-2.jpg",
                "board_image_url": "https://cdn/cutout-shirt.png",
                "cutout_status": "ready",
                "archetypes": ["Modern Professional"],
                "occasions": ["coffee date"],
                "gender": "unisex",
                "status": "active",
            },
        ],
    )

    directions = _enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Modern Professional",
                "hero_piece": "Soft Oxford Shirt",
                "items": ["Soft Oxford Shirt", "Chinos"],
            }
        ],
        occasion="coffee date",
    )

    assert directions[0]["image_url"] == "https://cdn/cutout-shirt.png"
    assert directions[0]["board_image_url"] == "https://cdn/cutout-shirt.png"
    assert directions[0]["catalog_image_url"] == "https://cdn/catalog-shirt-2.jpg"
