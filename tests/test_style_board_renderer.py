from brain.engines.style_board_renderer import StyleBoardRenderer


def _item(item_id, name, category):
    return {
        "id": item_id,
        "name": name,
        "category": category,
        "image_url": f"https://x/{item_id}.png",
    }


def test_composition_layout_matches_hero_piece_before_first_item_fallback():
    renderer = StyleBoardRenderer()
    items = [
        _item("shirt-1", "White Oxford Shirt", "top"),
        _item("blazer-1", "Gray Blazer", "outerwear"),
        _item("shoe-1", "Brown Loafers", "footwear"),
    ]

    layout = renderer._build_composition_layout(
        items,
        {
            "hero_piece": "Gray Blazer",
            "composition_items": [{"id": "shirt-1"}, {"id": "blazer-1"}, {"id": "shoe-1"}],
            "composition_mode": "stack",
        },
    )

    midground_ids = [item["id"] for item in layout["layers"]["midground"]]
    assert midground_ids[0] == "blazer-1"


def test_composition_layout_does_not_use_incompatible_first_item_as_hero():
    renderer = StyleBoardRenderer()
    items = [
        _item("shoe-1", "Brown Loafers", "footwear"),
        _item("knit-1", "Fine Gauge Knit Sweater", "top"),
        _item("jeans-1", "Dark Jeans", "bottom"),
    ]

    layout = renderer._build_composition_layout(
        items,
        {
            "hero_piece": "Fine Gauge Knit Sweater",
            "composition_items": [{"id": "shoe-1"}, {"id": "knit-1"}, {"id": "jeans-1"}],
            "composition_mode": "stack",
        },
    )

    midground_ids = [item["id"] for item in layout["layers"]["midground"]]
    assert midground_ids[0] == "knit-1"


def test_composition_layout_footwear_hero_only_uses_footwear():
    renderer = StyleBoardRenderer()
    items = [
        _item("shirt-1", "White Shirt", "top"),
        _item("bottom-1", "Stone Trouser", "bottom"),
        _item("shoe-1", "Black Oxford Shoes", "footwear"),
    ]

    layout = renderer._build_composition_layout(
        items,
        {
            "hero_piece": "Black Oxford Shoes",
            "composition_items": [{"id": "shirt-1"}, {"id": "bottom-1"}, {"id": "shoe-1"}],
            "composition_mode": "grid",
        },
    )

    midground_ids = [item["id"] for item in layout["layers"]["midground"]]
    assert midground_ids[0] == "shoe-1"


# ---------------------------------------------------------------------------
# Layout / shadow / bounds upgrades
# ---------------------------------------------------------------------------

from PIL import Image  # noqa: E402


def test_renderer_uses_explicit_composition_items_not_grid_fallback():
    renderer = StyleBoardRenderer()
    items = [
        _item("dress-1", "Red Polka Dot Dress", "Dresses"),
        _item("sneak-1", "White Sneakers", "Footwear"),
    ]
    board = {
        "composition_mode": "stack",
        "hero_item_id": "dress-1",
        "composition_items": [
            {"id": "dress-1", "role": "hero", "x": 0.46, "y": 0.42, "z": 4, "relative_size": 0.36},
            {"id": "sneak-1", "role": "support", "x": 0.72, "y": 0.78, "z": 5, "relative_size": 0.16},
        ],
    }
    layout = renderer._build_composition_layout(items, board)
    assert layout["composition"] != "grid"
    # Explicit x/y honored (spec layout), not the corner-grid fallback.
    assert abs(layout["placements"]["dress-1"]["x"] - 0.46) < 0.001
    assert abs(layout["placements"]["dress-1"]["y"] - 0.42) < 0.001


def test_fallback_layout_is_editorial_not_grid():
    renderer = StyleBoardRenderer()
    items = [
        _item("dress-1", "Maxi Dress", "Dresses"),
        _item("shoe-1", "Sandals", "Footwear"),
        _item("bag-1", "Clutch", "Accessories"),
    ]
    layout = renderer._build_fallback_layout(items)
    assert layout.get("composition") == "editorial_fallback"
    # Dress hero centered-ish, not in a corner grid cell.
    hero = layout["placements"]["dress-1"]
    assert 0.40 <= hero["x"] <= 0.55
    assert 0.35 <= hero["y"] <= 0.50


def _alpha_circle_png(size=200):
    # Opaque disc on a fully transparent canvas -> transparent corners.
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    from PIL import ImageDraw
    ImageDraw.Draw(img).ellipse([size * 0.2, size * 0.2, size * 0.8, size * 0.8], fill=(200, 40, 40, 255))
    return img


def test_alpha_shadow_is_not_a_rectangular_block():
    renderer = StyleBoardRenderer()
    out = renderer._add_shadow(_alpha_circle_png(200))
    # A rectangular shadow would darken the top-left corner; the alpha-mask
    # shadow must leave the corner transparent.
    corner_alpha = out.getpixel((1, 1))[3]
    assert corner_alpha == 0, "shadow must follow the silhouette, not a rectangle"


def test_oversized_item_is_clamped_within_canvas():
    renderer = StyleBoardRenderer()

    class _BigRenderer(StyleBoardRenderer):
        def _load_image(self, item):
            return Image.new("RGBA", (4000, 5000), (10, 10, 10, 255))

    big = _BigRenderer()
    img = big._prepare_image(
        {"id": "x", "name": "Huge Dress", "category": "Dresses"},
        {"scale": 5.0},
    )
    cw, ch = renderer.CANVAS_SIZE
    assert img.size[0] <= cw and img.size[1] <= ch, "hero must be clamped inside canvas"
