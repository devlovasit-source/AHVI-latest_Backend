from services.style_asset_contract import (
    adapt_style_asset,
    enrich_style_asset_rows,
)


def test_board_image_url_is_primary_board_cutout():
    row = {
        "$id": "doc-1",
        "asset_id": "asset-1",
        "name": "Black leather top",
        "category": "top",
        "source": "style_asset",
        "board_image_url": "https://r2.example/top_cutout.png",
        "catalog_image_url": "https://r2.example/top.jpg",
        "image_url": "https://source.example/original.jpg",
    }

    result = adapt_style_asset(row)

    assert result["item_id"] == "asset-1"
    assert result["asset_id"] == "asset-1"
    assert result["source"] == "style_asset"
    assert result["board_image_url"].endswith("_cutout.png")
    assert result["catalog_image_url"].endswith(".jpg")
    assert result["image_url"].endswith("original.jpg")
    assert result["selected_field"] == "board_image_url"
    assert result["source_kind"] == "style_asset_cutout"
    assert result["expected_transparent"] is True
    assert result["requires_frame"] is False


def test_catalog_image_is_framed_fallback():
    result = adapt_style_asset({
        "asset_id": "asset-2",
        "catalog_image_url": "https://r2.example/catalog.jpg",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "catalog_image_url"
    assert result["source_kind"] == "catalog_fallback"
    assert result["expected_transparent"] is False
    assert result["requires_frame"] is True


def test_generated_synthetic_item_recovers_full_asset_record_by_url():
    inventory = [{
        "$id": "appwrite-doc-3",
        "asset_id": "asset-3",
        "name": "Grey loafers",
        "category": "footwear",
        "image_url": "https://r2.example/grey-loafers.jpg",
        "board_image_url": "https://r2.example/grey-loafers-cutout.png",
        "catalog_image_url": "https://r2.example/grey-loafers-catalog.jpg",
    }]

    selected = [{
        "item_id": "grey loafers::https://r2.example/grey-loafers.jpg",
        "name": "Grey loafers",
        "role": "footwear",
        "image_url": "https://r2.example/grey-loafers.jpg",
    }]

    result = enrich_style_asset_rows(selected, inventory=inventory)[0]

    assert result["item_id"] == "asset-3"
    assert result["asset_id"] == "asset-3"
    assert result["board_image_url"].endswith("-cutout.png")
    assert result["catalog_image_url"].endswith("-catalog.jpg")
    assert result["source"] == "style_asset"
    assert result["selected_field"] == "board_image_url"
