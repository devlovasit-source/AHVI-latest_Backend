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


# ---------------------------------------------------------------------------
# Visual Inspiration board_image_url release-blocker regression.
#
# The live style_assets data has image_url == catalog_image_url (raw) and a
# distinct board_image_url (the real cutout PNG) with cutout_status=ready.
# A visual-inspiration direction is resolved through this contract more than
# once on its way to the final board_items payload:
#   1. _apply_board_image_fields (per-direction hero/complete_the_look pass)
#   2. _build_board_items -> _board_image_resolution (itemized board pass)
#   3. enrich_style_asset_rows -> adapt_style_asset (Appwrite record rejoin)
# By pass 2/3, image_url already holds the pass-1 SELECTED url (the cutout),
# not the true raw source. Checking board_image_url/cutout_url against a
# raw-alias set built from image_url then falsely flags the genuine cutout
# as "the same as raw" and drops it in favor of the actual raw
# catalog_image_url -- reintroducing the raw photo on the board.
# ---------------------------------------------------------------------------


def test_resolution_is_stable_across_repeated_passes():
    """Re-running resolve_style_asset_image on its own prior output (as
    happens across _apply_board_image_fields -> _board_image_resolution ->
    adapt_style_asset) must not regress the selection from the board cutout
    back to the raw/catalog image."""
    raw = "https://source.example/raw.jpg"
    board = "https://r2.example/cutout.png"

    asset = {
        "asset_id": "asset-stable",
        "image_url": raw,
        "board_image_url": board,
        "catalog_image_url": raw,
        "cutout_status": "ready",
        "source": "style_asset",
    }

    pass1 = resolve_style_asset_image(asset)
    assert pass1["selected_field"] == "board_image_url"
    assert pass1["selected_url"] == board

    # Simulate what _apply_board_image_fields writes back onto the direction:
    # image_url is overwritten with the pass-1 selected url.
    resolved_once = dict(asset)
    resolved_once["image_url"] = pass1["selected_url"]
    resolved_once["board_image_url"] = pass1["board_image_url"]
    resolved_once["catalog_image_url"] = pass1["catalog_image_url"] or raw

    pass2 = resolve_style_asset_image(resolved_once)
    assert pass2["selected_field"] == "board_image_url"
    assert pass2["selected_url"] == board
    assert pass2["selected_url"] != raw


def test_enrich_style_asset_rows_preserves_board_cutout_after_reresolution():
    """board_items built by _build_board_items already carry the resolved
    cutout as image_url before they reach enrich_style_asset_rows. Rejoining
    them with the full Appwrite inventory row must not overwrite that cutout
    with the raw image_url/catalog_image_url."""
    raw = "https://source.example/raw.jpg"
    board = "https://r2.example/cutout.png"

    inventory_row = {
        "asset_id": "asset-vi-1",
        "$id": "asset-vi-1",
        "name": "Essential Tee",
        "image_url": raw,
        "board_image_url": board,
        "catalog_image_url": raw,
        "cutout_status": "ready",
        "source": "style_asset",
    }

    # Already resolved by _build_board_items -- image_url is the cutout.
    already_resolved_board_item = {
        "name": "Essential Tee",
        "role": "top",
        "image_url": board,
        "source": "asset",
        "owned": False,
        "board_image_url": board,
        "catalog_image_url": raw,
    }

    result = enrich_style_asset_rows(
        [already_resolved_board_item], inventory=[inventory_row]
    )[0]

    assert result["image_url"] == board
    assert result["image_url"] != raw
    assert result["board_image_url"] == board


def test_genuine_raw_alias_board_url_is_still_rejected():
    """Safety must be unchanged: when board_image_url really is the same
    object as the raw upload (cutout never actually happened despite a
    ready status), it must still lose to catalog_image_url on the very
    first pass -- this is not a case the repeated-pass fix should touch."""
    raw = "https://source.example/raw.jpg"

    broken_asset = {
        "asset_id": "asset-broken",
        "image_url": raw,
        "board_image_url": raw,  # never actually cut out
        "catalog_image_url": raw,
        "cutout_status": "ready",
        "source": "style_asset",
    }

    result = resolve_style_asset_image(broken_asset)

    assert result["selected_field"] != "board_image_url"
    assert result["selected_field"] == "catalog_image_url"
    assert result["selected_url"] == raw
