"""Style-asset cutout validation.

The observed bug: catalog_<id>.png is RGBA 960x1096 with EVERY alpha byte at
255 — fully opaque with a baked white background — yet it was exposed as a
masked/catalog image, so Wardrobe and Style Boards rendered white rectangles.
bg_service.remove_background fails OPEN (returns the original bytes when
RMBG_SERVICE_URL is unset or the call errors), so nothing downstream may trust
a result it has not inspected pixel-by-pixel. No network here.
"""
from io import BytesIO

import pytest
from PIL import Image

from scripts.batch_rmbg_style_assets import (
    CUTOUT_VERSION,
    _target_key,
    is_effectively_opaque,
    png_alpha_stats,
    validate_alpha_png,
    validate_cutout_png,
)


def _png(mode="RGBA", size=(64, 64), alpha=255, color=(255, 255, 255)):
    img = Image.new(mode, size, color + (alpha,) if mode == "RGBA" else color)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _cutout_png(size=(64, 64), fg_box=(16, 16, 48, 48)):
    """Transparent background with an opaque garment region — a real cutout."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    for x in range(fg_box[0], fg_box[2]):
        for y in range(fg_box[1], fg_box[3]):
            img.putpixel((x, y), (250, 250, 250, 255))  # light garment
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def test_fully_opaque_rgba_png_is_detected_as_opaque():
    # The exact shape of the reported R2 object: RGBA, alpha == 255 everywhere.
    data = _png(alpha=255)
    has_alpha, ratio, reason = png_alpha_stats(data)
    assert has_alpha is True  # it HAS an alpha channel...
    assert ratio == 0.0       # ...but no transparency at all
    assert reason == "ok"
    assert is_effectively_opaque(data) is True


def test_transparent_png_is_skipped():
    assert is_effectively_opaque(_cutout_png()) is False


def test_valid_processed_png_passes_validation():
    ok, reason, ratio = validate_cutout_png(_cutout_png())
    assert ok is True
    assert reason == "ok"
    assert 0.0 < ratio < 1.0


def test_fully_transparent_output_is_rejected():
    ok, reason, _ratio = validate_cutout_png(_png(alpha=0))
    assert ok is False
    assert reason in {"no_opaque_pixels", "almost_fully_transparent"}


def test_still_opaque_output_is_rejected():
    # bg_service failing open returns the original opaque image; must not pass.
    ok, reason, _ratio = validate_cutout_png(_png(alpha=255))
    assert ok is False
    assert reason in {"no_transparent_pixels", "still_opaque"}


def test_corrupt_output_is_rejected():
    ok, reason, _ratio = validate_cutout_png(b"not-a-png")
    assert ok is False
    assert reason.startswith("invalid_image") or reason == "not_png"


def test_non_png_output_is_rejected():
    img = Image.new("RGB", (32, 32), (255, 255, 255))
    out = BytesIO()
    img.save(out, format="JPEG")
    ok, reason = validate_alpha_png(out.getvalue())
    assert ok is False
    assert reason == "not_png"


def test_light_garment_on_transparent_background_is_accepted():
    # Near-white garment pixels must survive: acceptance is alpha-based
    # segmentation output, never naive white-pixel deletion.
    ok, reason, ratio = validate_cutout_png(_cutout_png())
    assert ok is True and reason == "ok"
    assert ratio < 0.9  # the garment is still there


def test_cutout_key_is_versioned_and_never_the_source_key():
    asset = {"image_url": "https://cdn.test/style-assets/catalog_abc.png"}
    key = _target_key(asset)
    assert key.endswith(f"_cutout_v{CUTOUT_VERSION}.png")
    assert "catalog_abc.png" not in key  # source object is never overwritten
    assert key == f"style-assets/catalog_abc_cutout_v{CUTOUT_VERSION}.png"


def test_key_is_stable_for_the_same_asset_so_reruns_are_idempotent():
    asset = {"image_url": "https://cdn.test/style-assets/catalog_abc.png"}
    assert _target_key(asset) == _target_key(dict(asset))


@pytest.mark.parametrize("alpha,expect_opaque", [(255, True), (0, False), (8, False)])
def test_alpha_threshold_boundaries(alpha, expect_opaque):
    assert is_effectively_opaque(_png(alpha=alpha)) is expect_opaque
