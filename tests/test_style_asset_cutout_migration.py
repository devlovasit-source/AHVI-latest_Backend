from __future__ import annotations

from io import BytesIO

from PIL import Image

from scripts import batch_rmbg_style_assets as mig


def _png(mode: str = "RGBA", *, transparent: bool = True) -> bytes:
    if mode == "RGBA":
        bg = (255, 255, 255, 0) if transparent else (255, 255, 255, 255)
        img = Image.new(mode, (64, 64), bg)
    else:
        img = Image.new(mode, (64, 64), 255)
    if mode == "RGBA":
        for x in range(16, 48):
            for y in range(16, 48):
                img.putpixel((x, y), (20, 80, 180, 255))
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def test_validate_alpha_png_requires_transparency():
    ok, reason = mig.validate_alpha_png(_png(transparent=True))
    assert ok, reason

    ok, reason = mig.validate_alpha_png(_png(transparent=False))
    assert not ok
    assert reason == "no_transparent_pixels"


def test_select_p0_assets_skips_existing_board_image_and_caps_accessories():
    rows = [
        {"$id": "skip", "name": "White Shirt", "category": "top", "image_url": "https://cdn/shirt.jpg", "board_image_url": "https://cdn/shirt.png"},
        {"$id": "top", "name": "Blue Shirt", "category": "top", "image_url": "https://cdn/top.jpg"},
        *[
            {"$id": f"acc-{i}", "name": f"Watch {i}", "category": "accessory", "image_url": f"https://cdn/watch-{i}.jpg"}
            for i in range(15)
        ],
    ]

    selected = mig._select_p0_assets(rows)
    ids = [row["$id"] for row in selected]

    assert "skip" not in ids
    assert "top" in ids
    assert sum(1 for row in selected if mig._asset_role(row) == "accessory") == 10


def test_select_p0_assets_can_filter_category():
    rows = [
        {"$id": "top", "name": "Blue Shirt", "category": "top", "image_url": "https://cdn/top.jpg"},
        {"$id": "bottom", "name": "Relaxed Chinos", "category": "bottom", "image_url": "https://cdn/bottom.jpg"},
        {"$id": "shoe", "name": "Clean Sneakers", "category": "footwear", "image_url": "https://cdn/shoe.jpg"},
    ]

    selected = mig._select_p0_assets(rows, category="footwear", limit=2)

    assert [row["$id"] for row in selected] == ["shoe"]
