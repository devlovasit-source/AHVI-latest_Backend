"""Initial Style This board-image provenance (pre-Shuffle).

Covers _lite_group -> _lite_pick -> _lite_build_outfit -> _lite_item ->
_register_style_this_direction - the pipeline efc78aa's normalize_style_item
fix does NOT reach, because that fix only protects the separate
ConstrainedOutfitBuilder/Shuffle path. An item admitted by
is_board_renderable() at _lite_group time must still be is_board_renderable()
in the final registered/returned board item: Flutter's resolver looks for the
actual winning field (cutout_url, rmbg_url, ...), not just a non-empty
image_url, so collapsing to image_url-only silently produces an empty
hanger even though the item was genuinely board-safe pre-selection.
"""

import pytest

from routers import stylist
from services.style_board_image_readiness import is_board_renderable

_ANCHOR = {
    "id": "trousers-1",
    "name": "Black Night Trousers",
    "category": "Bottoms",
    "normalized_url": "https://images.test/trousers-1-normalized.png",
}


def _footwear():
    return {
        "id": "sneak-1",
        "name": "White Sneakers",
        "category": "Footwear",
        "normalized_url": "https://images.test/sneak-1-normalized.png",
    }


def _accessory():
    return {
        "id": "watch-1",
        "name": "Gold Watch",
        "category": "Accessories",
        "normalized_url": "https://images.test/watch-1-normalized.png",
    }


def _cutout_only_top():
    return {
        "id": "shirt-cutout",
        "name": "Checkered Shirt",
        "category": "Tops",
        "cutout_url": "https://images.test/shirt-cutout-cutout.png",
        "cutout_status": "ready",
    }


def _rmbg_only_top():
    return {
        "id": "shirt-rmbg",
        "name": "Checkered Shirt",
        "category": "Tops",
        "rmbg_url": "https://images.test/shirt-rmbg-rmbg.png",
        "image_status": "rmbg_complete",
    }


def _processed_only_top():
    return {
        "id": "shirt-processed",
        "name": "Checkered Shirt",
        "category": "Tops",
        "processed_url": "https://images.test/shirt-processed-processed.png",
        "image_status": "rmbg_complete",
    }


def _transparent_only_top():
    return {
        "id": "shirt-transparent",
        "name": "Checkered Shirt",
        "category": "Tops",
        "transparent_url": "https://images.test/shirt-transparent-transparent.png",
    }


def _board_image_only_top():
    return {
        "id": "shirt-board",
        "name": "Checkered Shirt",
        "category": "Tops",
        "board_image_url": "https://images.test/shirt-board-board.png",
        "board_status": "cutout_ready",
    }


def _req(wardrobe):
    return stylist.ItemStyleRequest(
        user_id="u1",
        mode="style_this",
        anchor_item=_ANCHOR,
        wardrobe=wardrobe,
        occasion=None,
    )


def _selected_top(direction):
    matches = [item for item in direction["items"] if item["item_id"].startswith("shirt-")]
    assert matches, f"no top item selected in direction {direction['title']!r}"
    return matches[0]


@pytest.mark.parametrize(
    "top_factory",
    [
        _cutout_only_top,
        _rmbg_only_top,
        _processed_only_top,
        _transparent_only_top,
        _board_image_only_top,
    ],
    ids=["cutout_url", "rmbg_url", "processed_url", "transparent_url", "board_image_url"],
)
def test_initial_style_this_preserves_board_image_provenance(top_factory):
    top = top_factory()
    assert is_board_renderable(top) is True, "fixture must be genuinely board-renderable pre-selection"

    wardrobe = [_ANCHOR, top, _footwear(), _accessory()]
    result = stylist.style_wardrobe_item("trousers-1", _req(wardrobe))

    assert result["success"] is True
    for direction in result["style_directions"]:
        selected_top = _selected_top(direction)
        assert is_board_renderable(selected_top) is True, (
            f"{direction['title']!r} lost board-safe provenance for the "
            f"selected top item ({top['id']!r})"
        )


def test_initial_style_this_normalized_url_regression():
    """normalized_url-only top must keep working (pre-existing green path)."""
    top = {
        "id": "shirt-normalized",
        "name": "Checkered Shirt",
        "category": "Tops",
        "normalized_url": "https://images.test/shirt-normalized-normalized.png",
    }
    wardrobe = [_ANCHOR, top, _footwear(), _accessory()]
    result = stylist.style_wardrobe_item("trousers-1", _req(wardrobe))

    assert result["success"] is True
    for direction in result["style_directions"]:
        assert is_board_renderable(_selected_top(direction)) is True


@pytest.mark.parametrize(
    "top_factory",
    [
        _cutout_only_top,
        _rmbg_only_top,
        _processed_only_top,
        _transparent_only_top,
        _board_image_only_top,
        lambda: {
            "id": "shirt-masked",
            "name": "Checkered Shirt",
            "category": "Tops",
            "masked_url": "https://images.test/shirt-masked-masked.png",
        },
    ],
    ids=["cutout_url", "rmbg_url", "processed_url", "transparent_url", "board_image_url", "masked_url"],
)
def test_lite_item_alone_preserves_renderability(top_factory):
    """_lite_item() must not depend on the registration-boundary merge to
    stay renderable - the function itself must satisfy the invariant."""
    top = top_factory()
    assert is_board_renderable(top) is True

    output = stylist._lite_item(top, "support")

    assert is_board_renderable(output) is True


def test_lite_item_transparent_url_projects_to_masked_url_field():
    """is_board_renderable() passing on the projected item isn't sufficient
    proof Flutter will show it - lib/util/wardrobe_image_resolver.dart's
    wardrobe (non-style-asset) branch never reads a bare transparent_url
    field, only masked_url. _lite_item must emit the field name Flutter
    actually looks for, not just a name the backend's own readiness gate
    happens to also treat as unconditional.
    """
    top = _transparent_only_top()
    output = stylist._lite_item(top, "support")

    assert is_board_renderable(output) is True
    assert "masked_url" in output, (
        "transparent_url must be projected as masked_url - the only field "
        "name the Flutter wardrobe resolver's non-asset branch honors "
        "unconditionally - or the item silently stops rendering client-side "
        "despite is_board_renderable() reporting True"
    )
    assert "transparent_url" not in output


def test_lite_item_does_not_fabricate_masked_url_alias():
    """A genuinely non-renderable raw-only item stays non-renderable through
    _lite_item - the fix must not accidentally start admitting forged
    provenance."""
    raw_only = {
        "id": "shirt-raw",
        "name": "Checkered Shirt",
        "category": "Tops",
        "image_url": "https://images.test/shirt-raw-raw.png",
    }
    assert is_board_renderable(raw_only) is False

    output = stylist._lite_item(raw_only, "support")

    assert is_board_renderable(output) is False


def test_initial_style_this_masked_url_regression():
    """masked_url-only top must keep working (pre-existing green path)."""
    top = {
        "id": "shirt-masked",
        "name": "Checkered Shirt",
        "category": "Tops",
        "masked_url": "https://images.test/shirt-masked-masked.png",
    }
    wardrobe = [_ANCHOR, top, _footwear(), _accessory()]
    result = stylist.style_wardrobe_item("trousers-1", _req(wardrobe))

    assert result["success"] is True
    for direction in result["style_directions"]:
        assert is_board_renderable(_selected_top(direction)) is True
