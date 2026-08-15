"""P0-2 focused regression: a Style This anchor must never surface a
masked_url that is merely a fabricated alias of the raw upload (RMBG never
completed) as a board-safe/transparent image. The anchor must use the
existing safe-failure path (return None) instead of silently rendering an
original person photo."""

from __future__ import annotations

from services.style_this_anchor import canonical_style_this_anchor


def test_anchor_rejects_masked_url_that_aliases_the_raw_upload_with_no_other_image():
    """Matches the live P0-2 evidence exactly: item carries only a fabricated
    masked_url (no inline image_url at all). That alias must not count as a
    safe image, so the anchor resolves to the existing safe-failure (None)."""
    item = {
        "item_id": "acc-1",
        "id": "acc-1",
        "name": "Necklace",
        "category": "accessory",
        "source": "wardrobe",
        "raw_url": "https://raw/person.png",
        "masked_url": "https://raw/person.png",
    }
    assert canonical_style_this_anchor(item, expected_item_id="acc-1") is None


def test_anchor_falls_back_to_honest_raw_image_when_masked_url_aliases_it():
    """When a real (non-fabricated) image_url also exists, the anchor may
    still resolve using it -- just as an honest, non-transparent original,
    never mislabeled as a masked cutout."""
    item = {
        "item_id": "acc-1b",
        "id": "acc-1b",
        "name": "Necklace",
        "category": "accessory",
        "source": "wardrobe",
        "image_url": "https://raw/person.png",
        "masked_url": "https://raw/person.png",
    }
    anchor = canonical_style_this_anchor(item, expected_item_id="acc-1b")
    assert anchor is not None
    assert anchor["image_url"] == "https://raw/person.png"
    assert anchor["expected_transparent"] is False


def test_anchor_accepts_a_genuine_masked_cutout():
    item = {
        "item_id": "acc-2",
        "id": "acc-2",
        "name": "Necklace",
        "category": "accessory",
        "source": "wardrobe",
        "image_url": "https://raw/person.png",
        "masked_url": "https://cdn/cutout.png",
    }
    anchor = canonical_style_this_anchor(item, expected_item_id="acc-2")
    assert anchor is not None
    assert anchor["image_url"] == "https://cdn/cutout.png"
    assert anchor["expected_transparent"] is True
