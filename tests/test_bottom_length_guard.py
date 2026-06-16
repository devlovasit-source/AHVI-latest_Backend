"""Bottoms pants/shorts sanity guard tests (deterministic, image-based)."""

import io

from PIL import Image

from services import wardrobe_taxonomy as wt


def _cutout(w, h, color=(120, 160, 220, 255)):
    """Transparent PNG with an opaque garment rect filling most of the frame."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    # rect with small margin -> foreground aspect ~ h/w
    from PIL import ImageDraw

    d = ImageDraw.Draw(im)
    d.rectangle([int(w * 0.1), int(h * 0.05), int(w * 0.9), int(h * 0.95)], fill=color)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _tall():   # trousers-shaped cutout (h/w high)
    return _cutout(300, 900)


def _wide():   # shorts-shaped cutout (h/w ~1)
    return _cutout(700, 600)


def _ambiguous():  # in the uncertain band
    return _cutout(500, 650)  # aspect ~1.3


# 1. full-length trousers misdetected as shorts -> corrected to trousers
def test_trousers_misdetected_as_shorts_corrected():
    item = {"category": "Bottoms", "sub_category": "Shorts", "name": "Light Blue Shorts"}
    out = wt.apply_bottom_length_guard(item, _tall())
    assert out["sub_category"] == "Trousers"
    assert "short" not in out["name"].lower()
    assert "trousers" in out["name"].lower()  # "Light Blue Trousers"
    assert out["_bottom_length_corrected"] == "detector_shorts_but_crop_trousers"


# 2. actual shorts remain shorts (detector right, heuristic agrees -> no change)
def test_actual_shorts_unchanged():
    item = {"category": "Bottoms", "sub_category": "Shorts", "name": "Beige Shorts"}
    out = wt.apply_bottom_length_guard(item, _wide())
    assert out["sub_category"] == "Shorts"
    assert out["name"] == "Beige Shorts"
    assert "_bottom_length_corrected" not in out


# pants mislabeled as... actually a wide cutout + trouser label -> corrected to shorts
def test_trouser_label_but_wide_crop_becomes_shorts():
    item = {"category": "Bottoms", "sub_category": "Trousers", "name": "Khaki Trousers"}
    out = wt.apply_bottom_length_guard(item, _wide())
    assert out["sub_category"] == "Shorts"
    assert "shorts" in out["name"].lower()
    assert out["_bottom_length_corrected"] == "detector_trousers_but_crop_shorts"


# 3. ambiguous bottom remains unchanged
def test_ambiguous_unchanged():
    item = {"category": "Bottoms", "sub_category": "Shorts", "name": "Blue Shorts"}
    out = wt.apply_bottom_length_guard(item, _ambiguous())
    assert out["sub_category"] == "Shorts"
    assert "_bottom_length_corrected" not in out


# 4. non-bottom item unchanged
def test_non_bottom_unchanged():
    item = {"category": "Tops", "sub_category": "Shirt", "name": "White Shirt"}
    out = wt.apply_bottom_length_guard(item, _tall())
    assert out == item
    assert wt.infer_bottom_length_from_crop(_tall(), "Tops") == "unknown"


# 5. correction updates BOTH name and sub_category
def test_correction_updates_name_and_sub():
    item = {"category": "Bottoms", "sub_category": "Shorts", "name": "Navy Cotton Shorts"}
    out = wt.apply_bottom_length_guard(item, _tall())
    assert out["sub_category"] == "Trousers"
    assert out["subcategory"] == "Trousers"
    assert out["name"] == "Navy Cotton Trousers"


# 6. no change when uncertain (no bytes / undecodable)
def test_no_change_when_no_bytes():
    item = {"category": "Bottoms", "sub_category": "Shorts", "name": "X Shorts"}
    assert wt.apply_bottom_length_guard(item, b"") == item
    assert wt.infer_bottom_length_from_crop(b"", "Bottoms") == "unknown"


# heuristic direction sanity
def test_infer_directions():
    assert wt.infer_bottom_length_from_crop(_tall(), "Bottoms") == "trousers"
    assert wt.infer_bottom_length_from_crop(_wide(), "Bottoms") == "shorts"
    assert wt.infer_bottom_length_from_crop(_ambiguous(), "Bottoms") == "unknown"


# --- normalize(): Bottoms tokens beat bare "dress" substring ---
def test_dress_pants_routes_to_bottoms_not_dresses():
    cat, sub = wt.normalize(category="Bottoms", name="Navy Blue Dress Pants", sub_category="Dress Pants")
    assert cat == "Bottoms"
    assert sub == "Trousers"


def test_dress_trousers_routes_to_bottoms():
    cat, _ = wt.normalize(category="Bottoms", name="Charcoal Dress Trousers", sub_category="")
    assert cat == "Bottoms"


def test_plain_dress_still_dresses():
    cat, sub = wt.normalize(category="Dresses", name="Red Summer Dress", sub_category="Dress")
    assert cat == "Dresses"


def test_jeans_route_to_bottoms_jeans():
    cat, sub = wt.normalize(category="Bottoms", name="Blue Jeans", sub_category="")
    assert (cat, sub) == ("Bottoms", "Jeans")


def test_denim_jacket_not_bottoms():
    cat, _ = wt.normalize(category="Outerwear", name="Denim Jacket", sub_category="Jacket")
    assert cat == "Outerwear"


# guard does not invent a correction when detector + heuristic agree (trousers/trousers)
def test_agreement_no_correction():
    item = {"category": "Bottoms", "sub_category": "Trousers", "name": "Grey Trousers"}
    out = wt.apply_bottom_length_guard(item, _tall())
    assert "_bottom_length_corrected" not in out
    assert out["sub_category"] == "Trousers"
