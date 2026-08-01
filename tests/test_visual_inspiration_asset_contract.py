from services import style_reasoning_engine as engine


def test_visual_direction_restores_attached_asset_board_png(monkeypatch):
    asset = {
        "$id": "appwrite-doc-1",
        "asset_id": "asset-shirt-1",
        "name": "Soft Oxford Shirt",
        "category": "top",
        "subcategory": "shirt",
        "role": "top",
        "image_url": "https://cdn.example/catalog-shirt.jpg",
        "catalog_image_url": "https://cdn.example/catalog-shirt.jpg",
        "board_image_url": "https://cdn.example/board-shirt-cutout.png",
        "board_r2_key": "style-assets/board-shirt-cutout.png",
        "cutout_status": "ready",
        "gender": "male",
        "status": "active",
        "archetypes": ["Modern Professional"],
        "occasions": ["office"],
        "colors": ["blue"],
        "tags": ["shirt", "office"],
    }

    monkeypatch.setattr(
        engine,
        "_style_asset_rows",
        lambda limit=0: [dict(asset)],
    )
    monkeypatch.setattr(
        engine,
        "_is_nonfashion_asset",
        lambda _asset: False,
    )
    monkeypatch.setattr(
        engine,
        "_asset_allowed_for_context",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        engine,
        "_hero_asset_allowed",
        lambda *_args, **_kwargs: True,
    )

    directions = engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Modern Professional",
                "title": "Modern Professional",
                "hero_piece": "Soft Oxford Shirt",
                "image_url": asset["image_url"],
                "asset_id": asset["asset_id"],
                "items": ["Soft Oxford Shirt"],
                "complete_the_look": [],
            }
        ],
        occasion="office",
        target_gender="male",
        wardrobe_intent=False,
    )

    assert len(directions) == 1

    direction = directions[0]

    assert direction["asset_id"] == "asset-shirt-1"
    assert (
        direction["board_image_url"]
        == "https://cdn.example/board-shirt-cutout.png"
    )
    assert (
        direction["catalog_image_url"]
        == "https://cdn.example/catalog-shirt.jpg"
    )

    board_items = direction["board_items"]

    matching = [
        item
        for item in board_items
        if item.get("asset_id") == "asset-shirt-1"
    ]

    assert matching
    assert (
        matching[0]["board_image_url"]
        == "https://cdn.example/board-shirt-cutout.png"
    )
    assert (
        matching[0]["catalog_image_url"]
        == "https://cdn.example/catalog-shirt.jpg"
    )
    assert matching[0]["source"] == "style_asset"
    assert matching[0]["selected_field"] == "board_image_url"
    assert matching[0]["source_kind"] == "style_asset_cutout"
    assert matching[0]["expected_transparent"] is True
    assert matching[0]["requires_frame"] is False
