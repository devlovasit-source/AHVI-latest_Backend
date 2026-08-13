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


def test_bare_board_url_falls_back_without_cutout_metadata():
    direction = {
        "hero_piece": "Soft Oxford Shirt",
        "image_url": "https://cdn/catalog-shirt.jpg",
        "board_image_url": "https://cdn/unknown-shirt.png",
        "complete_the_look": [
            {"name": "Relaxed Chinos", "image_url": "https://cdn/catalog-chino.jpg"},
            {"name": "Clean Sneakers", "image_url": "https://cdn/catalog-sneakers.jpg"},
        ],
    }

    items = _build_board_items(direction, wardrobe_intent=False)

    assert items[0]["image_url"] == "https://cdn/catalog-shirt.jpg"
    assert "board_image_url" not in items[0]
    assert items[0]["board_status"] == "catalog_fallback"
    assert items[0]["image_source"] == "image_url"


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

    # AHVI_COMPLETE_VISUAL_FIXTURE
    # This test validates image selection/propagation. The production path now
    # also enforces a complete core outfit, so provide deterministic supporting
    # assets rather than weakening the completeness validator.
    repair_bottom = {
        "asset_id": "fixture-bottom-1",
        "name": "Tailored Navy Trouser",
        "category": "bottom",
        "role": "bottom",
        "image_url": "https://cdn.example/fixture-bottom.jpg",
        "board_image_url": "https://cdn.example/fixture-bottom.png",
        "catalog_image_url": "https://cdn.example/fixture-bottom.jpg",
        "cutout_status": "ready",
        "archetypes": ["Modern Professional"],
        "occasions": ["office", "coffee date", "today"],
        "gender": "unisex",
        "status": "active",
    }

    repair_footwear = {
        "asset_id": "fixture-footwear-1",
        "name": "Minimal Navy Sneaker",
        "category": "footwear",
        "role": "footwear",
        "image_url": "https://cdn.example/fixture-footwear.jpg",
        "board_image_url": "https://cdn.example/fixture-footwear.png",
        "catalog_image_url": "https://cdn.example/fixture-footwear.jpg",
        "cutout_status": "ready",
        "archetypes": ["Modern Professional"],
        "occasions": ["office", "coffee date", "today"],
        "gender": "unisex",
        "status": "active",
    }

    original_best_style_assets = engine._best_style_assets

    def _fixture_best_style_assets(
        _assets,
        *,
        direction,
        **_kwargs,
    ):
        probe = str(
            direction.get("hero_piece")
            or direction.get("heroPiece")
            or ""
        ).lower()

        if "trouser" in probe:
            return [dict(repair_bottom)]

        if "shoe" in probe:
            return [dict(repair_footwear)]

        # Preserve the real asset-ranking behaviour for the hero-selection
        # call. This test specifically proves that the cutout-ready shirt wins
        # even though the catalogue-only shirt appears first in inventory.
        return original_best_style_assets(
            _assets,
            direction=direction,
            **_kwargs,
        )

    monkeypatch.setattr(
        engine,
        "_best_style_assets",
        _fixture_best_style_assets,
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
