from services.style_asset_contract import (
    adapt_style_asset,
    enrich_style_asset_rows,
    resolve_style_asset_image,
)


def test_board_image_url_is_primary_board_cutout():
    row = {
        "$id": "doc-1",
        "asset_id": "asset-1",
        "name": "Black leather top",
        "category": "top",
        "source": "style_asset",
        "board_image_url": "https://r2.example/top_cutout.png",
        "cutout_status": "ready",
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


def test_bare_board_image_url_is_not_transparent_provenance():
    result = adapt_style_asset({
        "asset_id": "asset-bare-board",
        "board_image_url": "https://r2.example/unknown.png",
        "catalog_image_url": "https://r2.example/catalog.jpg",
        "image_url": "https://source.example/original.jpg",
    })

    assert "board_image_url" not in result
    assert result["selected_field"] == "catalog_image_url"
    assert result["source_kind"] == "catalog_fallback"
    assert result["expected_transparent"] is False
    assert result["requires_frame"] is True


def test_validated_cutout_beats_processed_and_raw_candidates():
    result = adapt_style_asset({
        "asset_id": "asset-cutout",
        "board_image_url": "https://r2.example/cutout.png",
        "cutout_status": "ready",
        "normalized_url": "https://r2.example/processed.png",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "board_image_url"
    assert result["source_kind"] == "style_asset_cutout"
    assert result["expected_transparent"] is True


def test_explicit_cutout_url_beats_ready_board_and_fallbacks():
    result = resolve_style_asset_image({
        "source": "style_asset",
        "cutout_url": "https://r2.example/cutout.png",
        "board_image_url": "https://r2.example/board.png",
        "cutout_status": "ready",
        "normalized_url": "https://r2.example/processed.png",
        "catalog_image_url": "https://r2.example/catalog.jpg",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "cutout_url"
    assert result["source_kind"] == "style_asset_cutout"
    assert result["expected_transparent"] is True


def test_ready_board_image_url_beats_normalized_catalog_and_raw():
    result = resolve_style_asset_image({
        "source": "style_asset",
        "board_image_url": "https://r2.example/board.png",
        "cutout_status": "ready",
        "normalized_url": "https://r2.example/processed.png",
        "catalog_image_url": "https://r2.example/catalog.jpg",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "board_image_url"
    assert result["source_kind"] == "style_asset_cutout"
    assert result["expected_transparent"] is True


def test_failed_board_image_url_falls_back_to_normalized():
    result = resolve_style_asset_image({
        "source": "style_asset",
        "board_image_url": "https://r2.example/failed-board.png",
        "cutout_status": "failed",
        "normalized_url": "https://r2.example/processed.png",
        "catalog_image_url": "https://r2.example/catalog.jpg",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "normalized_url"
    assert result["source_kind"] == "style_asset_processed"
    assert result["expected_transparent"] is False


def test_null_status_board_image_url_falls_back_to_catalog():
    result = resolve_style_asset_image({
        "source": "style_asset",
        "board_image_url": "https://r2.example/unknown-board.png",
        "catalog_image_url": "https://r2.example/catalog.jpg",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "catalog_image_url"
    assert result["source_kind"] == "catalog_fallback"
    assert result["expected_transparent"] is False


def test_normalized_beats_catalog_and_raw_without_validated_cutout():
    result = resolve_style_asset_image({
        "source": "style_asset",
        "normalized_url": "https://r2.example/processed.png",
        "catalog_image_url": "https://r2.example/catalog.jpg",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "normalized_url"
    assert result["expected_transparent"] is False


def test_catalog_beats_raw_without_normalized_or_cutout():
    result = resolve_style_asset_image({
        "source": "style_asset",
        "catalog_image_url": "https://r2.example/catalog.jpg",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "catalog_image_url"
    assert result["expected_transparent"] is False


def test_raw_fallback_reports_original_opaque_provenance():
    result = resolve_style_asset_image({
        "source": "style_asset",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "image_url"
    assert result["source_kind"] == "style_asset_original"
    assert result["expected_transparent"] is False


def test_wardrobe_validated_masked_url_beats_normalized_and_raw():
    result = resolve_style_asset_image({
        "source": "wardrobe",
        "masked_url": "https://r2.example/masked.png",
        "normalized_url": "https://r2.example/processed.png",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "masked_url"
    assert result["source_kind"] == "wardrobe_cutout"
    assert result["expected_transparent"] is True


def test_wardrobe_normalized_url_beats_raw():
    result = resolve_style_asset_image({
        "source": "wardrobe",
        "normalized_url": "https://r2.example/processed.png",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "normalized_url"
    assert result["source_kind"] == "wardrobe_processed"
    assert result["expected_transparent"] is False


def test_masked_alias_equal_to_raw_is_not_transparent():
    result = adapt_style_asset({
        "asset_id": "asset-fabricated-mask",
        "masked_url": "https://source.example/original.jpg",
        "normalized_url": "https://r2.example/processed.png",
        "image_url": "https://source.example/original.jpg",
    })

    assert result["selected_field"] == "normalized_url"
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
        "cutout_status": "ready",
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
