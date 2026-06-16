"""Tests for the deterministic catalog image pipeline (RMBG + rotate + center).

Pure CPU/PIL. No network, no R2, no Imagen/Flux.
"""

import io

import pytest
from PIL import Image, ImageDraw

from services import catalog_image_service as c
from services.image_normalizer import get_image_size


def _garment_png(w, h, color=(30, 90, 200, 255), margin=0.2):
    """Transparent PNG with a centered solid rectangle 'garment'."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([w * margin, h * margin, w * (1 - margin), h * (1 - margin)], fill=color)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _empty_png(w=500, h=500):
    b = io.BytesIO()
    Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(b, "PNG")
    return b.getvalue()


# 1. transparent PNG centered on 1600 canvas
def test_centered_on_1600_canvas():
    out = c.generate_catalog_image(_garment_png(800, 800), {"category": "top"})
    assert out["success"] is True
    assert get_image_size(out["catalog_image_bytes"]) == (1600, 1600)


# 2. white/off-white background output generated (JPEG, opaque)
def test_offwhite_background_output():
    out = c.normalize_catalog_image(_garment_png(800, 800), "top", {})
    assert out["success"] is True
    img = Image.open(io.BytesIO(out["image_bytes"]))
    assert img.mode in ("RGB", "RGBA")
    # corner pixel should be the off-white background
    corner = img.convert("RGB").getpixel((5, 5))
    assert all(abs(a - b) <= 6 for a, b in zip(corner, c.OFF_WHITE_RGB))


# 3. small artifacts removed / trimmed -> garment fills frame, not tiny input size
def test_small_input_is_upscaled_and_filled():
    out = c.generate_catalog_image(_garment_png(300, 300), {"category": "dress"})
    assert out["success"] is True
    assert out["validation"]["visible_ratio"] > 0.20


# P0: sideways top rotates upright
def test_sideways_top_rotates():
    out = c.normalize_catalog_image(_garment_png(900, 300), "top", {})
    assert out["rotation_applied"] == -90


# P0: upright dress does not rotate
def test_upright_dress_no_rotation():
    out = c.normalize_catalog_image(_garment_png(300, 900), "dress", {})
    assert out["rotation_applied"] == 0


# P0: footwear does not rotate even when wide
def test_footwear_not_rotated():
    out = c.normalize_catalog_image(_garment_png(900, 300), "footwear", {})
    assert out["rotation_applied"] == 0


# 5. invalid category skipped
def test_invalid_category_skipped():
    out = c.generate_catalog_image(_garment_png(800, 800), {"category": "skincare"})
    assert out["success"] is False
    assert out["reason"] == "category_not_eligible"
    assert c.category_allowed("skincare") is False
    assert c.category_allowed("top") is True


# 7. validation rejects empty mask
def test_empty_mask_fails_gracefully():
    assert c.normalize_catalog_image(_empty_png(), "top", {})["success"] is False
    g = c.generate_catalog_image(_empty_png(), {"category": "top"})
    assert g["success"] is False
    assert g["reason"] == "empty_foreground"


# Phase 8 hook is designed but not implemented
def test_ai_hook_not_implemented():
    with pytest.raises(NotImplementedError):
        c.generate_catalog_image_ai(b"x", provider="imagen")


# --- Router integration: catalog never blocks save, flag gating ---
def test_router_helper_noop_when_flag_off(monkeypatch):
    monkeypatch.setenv("ENABLE_CATALOG_IMAGE_GENERATION", "false")
    monkeypatch.setenv("ENABLE_CATALOG_NORMALIZATION", "false")
    from routers import wardrobe_capture as wc

    item = {"category": "top"}
    wc._maybe_generate_catalog_image(item, _garment_png(800, 800), "id1")
    assert "catalogUrl" not in item
    assert "catalogStatus" not in item


def test_router_helper_never_raises_on_upload_failure(monkeypatch):
    monkeypatch.setenv("ENABLE_CATALOG_IMAGE_GENERATION", "true")
    from routers import wardrobe_capture as wc

    # Force R2 upload to blow up — helper must swallow and mark catalog_failed.
    class _BoomR2:
        def upload_catalog_image(self, **kw):
            raise RuntimeError("r2 down")

    monkeypatch.setattr(wc, "R2Storage", lambda: _BoomR2())
    item = {"category": "top"}
    wc._maybe_generate_catalog_image(item, _garment_png(800, 800), "id2")
    assert item["catalogStatus"] == "catalog_failed"  # save still proceeds


def test_router_helper_skips_invalid_category(monkeypatch):
    monkeypatch.setenv("ENABLE_CATALOG_IMAGE_GENERATION", "true")
    from routers import wardrobe_capture as wc

    item = {"category": "skincare"}
    wc._maybe_generate_catalog_image(item, _garment_png(800, 800), "id3")
    assert item["catalogStatus"] == "catalog_skipped_category"
    assert "catalogUrl" not in item


def test_router_helper_success_sets_aliases(monkeypatch):
    monkeypatch.setenv("ENABLE_CATALOG_IMAGE_GENERATION", "true")
    from routers import wardrobe_capture as wc

    class _OkR2:
        def upload_catalog_image(self, *, file_id, image_bytes, extension="jpg"):
            return {
                "catalog_file_name": f"catalog_{file_id}.{extension}",
                "catalog_url": f"https://cdn.test/catalog_{file_id}.{extension}",
            }

    monkeypatch.setattr(wc, "R2Storage", lambda: _OkR2())
    item = {"category": "top"}
    wc._maybe_generate_catalog_image(item, _garment_png(900, 900), "id4")
    assert item["catalogStatus"] == "catalog_ready"
    assert item["catalogUrl"] == item["catalog_url"]
    assert item["catalogUrl"].endswith("catalog_id4.jpg")
    assert item["catalogMethod"] == "rmbg_center_normalize"
