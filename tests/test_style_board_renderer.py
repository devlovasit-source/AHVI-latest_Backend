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
