"""Orientation gate helpers: category relevance + clearly-sideways detection.

Covers the safety-critical rule: only portrait-expected apparel is gated, and
only a CLEARLY landscape foreground (>= threshold) is flagged, so wide tops,
skirts, bags and footwear are never falsely rejected.
"""
import io

from PIL import Image

from services.catalog_png_generation_service import (
    _apparel_looks_sideways,
    _classify_orientation,
    _orientation_confidence_min,
    _orientation_regen_max,
    _orientation_relevant_category,
)


def _rgba_with_foreground(w: int, h: int, fg_w: int, fg_h: int) -> bytes:
    """A transparent canvas with an opaque foreground rectangle of fg_w x fg_h,
    centered. The alpha bbox aspect = fg_w / fg_h."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x0 = (w - fg_w) // 2
    y0 = (h - fg_h) // 2
    for x in range(x0, x0 + fg_w):
        for y in range(y0, y0 + fg_h):
            img.putpixel((x, y), (10, 10, 10, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_relevant_categories():
    assert _orientation_relevant_category("top")
    assert _orientation_relevant_category("dress")
    assert _orientation_relevant_category("outerwear")
    # non-portrait categories are never gated
    assert not _orientation_relevant_category("footwear")
    assert not _orientation_relevant_category("bag")
    assert not _orientation_relevant_category("accessory")
    assert not _orientation_relevant_category("jewellery")


def test_bottom_gated_only_when_full_length():
    assert _orientation_relevant_category("bottom", {"name": "blue jeans"})
    assert _orientation_relevant_category("bottom", {"sub_category": "trousers"})
    # skirts / shorts are legitimately wide -> not gated
    assert not _orientation_relevant_category("bottom", {"name": "denim skirt"})
    assert not _orientation_relevant_category("bottom", {})


def test_upright_top_not_flagged():
    # portrait garment (taller than wide)
    up = _rgba_with_foreground(1000, 1000, 500, 800)
    sideways, aspect = _apparel_looks_sideways(up, "top")
    assert not sideways
    assert aspect < 1.5


def test_sideways_top_flagged():
    # clearly landscape (rotated 90 deg) -> width >> height
    rot = _rgba_with_foreground(1000, 1000, 800, 460)
    sideways, aspect = _apparel_looks_sideways(rot, "top")
    assert sideways
    assert aspect >= 1.5


def test_wide_oversized_top_not_flagged():
    # oversized/batwing top: wider than a slim tee but NOT clearly sideways
    wide = _rgba_with_foreground(1000, 1000, 720, 640)  # aspect ~1.12
    sideways, _ = _apparel_looks_sideways(wide, "top")
    assert not sideways


def test_landscape_footwear_not_flagged():
    # footwear is legitimately landscape and must never be flagged
    shoe = _rgba_with_foreground(1000, 1000, 800, 400)
    sideways, _ = _apparel_looks_sideways(shoe, "footwear")
    assert not sideways


def test_flag_defaults():
    assert _orientation_confidence_min() == 0.85
    assert _orientation_regen_max() == 1


def test_classifier_fail_safe_never_false_upright():
    # No vertex creds in test env -> the classifier must fail SAFE (uncertain,
    # confidence 0), never a false 'upright' that would let a bad image through.
    up = _rgba_with_foreground(1000, 1000, 500, 800)
    res = _classify_orientation(up, "top")
    assert res["orientation"] != "upright"
    assert res["confidence"] == 0.0
    assert "orientation" in res and "evidence" in res
