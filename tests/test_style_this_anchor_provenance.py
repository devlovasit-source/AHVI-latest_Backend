"""canonical_style_this_anchor() must preserve source provenance, not
collapse the winning presentation field into image_url. That collapse was a
second, independent source of the normalized_url false-positive regression:
once image_url is alias-checked (see services.style_board_image_readiness),
manufacturing image_url == normalized_url makes a genuinely-processed item
look like an unprocessed raw upload and get rejected as not board-safe.
"""
from __future__ import annotations

from services.style_this_anchor import canonical_style_this_anchor
from services.style_board_image_readiness import _board_url_identity, resolve_board_image_candidate

RAW = "https://images.test/raw-upload.png"
PROCESSED = "https://images.test/catalog-normalized.png"


def _item(item_id="item-1", **extra):
    base = {"id": item_id, "name": "Test Item", "category": "Tops", "source": "wardrobe"}
    base.update(extra)
    return base


# ---------- A. raw + normalized: both must survive distinct ----------

def test_raw_and_normalized_both_preserved_distinct():
    anchor = canonical_style_this_anchor(_item(image_url=RAW, normalized_url=PROCESSED))

    assert anchor["image_url"] == RAW
    assert anchor["normalized_url"] == PROCESSED
    assert anchor["safe_image_url"] == PROCESSED
    assert _board_url_identity(anchor["image_url"]) != _board_url_identity(anchor["normalized_url"])


def test_raw_and_normalized_anchor_is_board_renderable():
    anchor = canonical_style_this_anchor(_item(image_url=RAW, normalized_url=PROCESSED))
    result = resolve_board_image_candidate(anchor)
    assert result["renderable"] is True
    assert result["selected_url"] == PROCESSED


# ---------- B. processed-only: must not manufacture image_url ----------

def test_processed_only_does_not_manufacture_image_url():
    anchor = canonical_style_this_anchor(_item(normalized_url=PROCESSED))

    assert anchor["normalized_url"] == PROCESSED
    assert anchor["safe_image_url"] == PROCESSED
    assert not anchor.get("image_url")


def test_processed_only_anchor_is_board_renderable():
    anchor = canonical_style_this_anchor(_item(normalized_url=PROCESSED))
    result = resolve_board_image_candidate(anchor)
    assert result["renderable"] is True
    assert result["selected_url"] == PROCESSED


# ---------- C. catalog/cutout priority preserved for safe_image_url ----------

def test_cutout_outranks_normalized_for_safe_image_url():
    anchor = canonical_style_this_anchor(
        _item(cutout_url=PROCESSED + "-cutout", normalized_url=PROCESSED, image_url=RAW)
    )
    assert anchor["safe_image_url"] == PROCESSED + "-cutout"
    # provenance fields untouched regardless of which one won the ranking
    assert anchor["image_url"] == RAW
    assert anchor["normalized_url"] == PROCESSED


def test_catalog_image_url_outranks_normalized_for_safe_image_url():
    anchor = canonical_style_this_anchor(
        _item(catalog_image_url=PROCESSED + "-catalog", normalized_url=PROCESSED, image_url=RAW)
    )
    assert anchor["safe_image_url"] == PROCESSED + "-catalog"
    assert anchor["image_url"] == RAW


# ---------- D. raw-only: must stay unsafe, not board-safe merely because image_url exists ----------

def test_raw_only_image_url_preserved_as_is():
    """image_url is the lowest-priority fallback in _SAFE_IMAGE_FIELDS (pre-existing,
    unrelated to this fix) -- safe_image_url legitimately equals it when nothing
    else is available. What must NOT happen is resolve_board_image_candidate
    treating that as board-safe merely because a URL is present in image_url."""
    anchor = canonical_style_this_anchor(_item(image_url=RAW))
    assert anchor["image_url"] == RAW
    assert anchor["safe_image_url"] == RAW


def test_raw_only_anchor_is_not_board_renderable():
    anchor = canonical_style_this_anchor(_item(image_url=RAW))
    result = resolve_board_image_candidate(anchor)
    assert result["renderable"] is False


# ---------- essential regression: the exact false-positive chain that broke shuffle ----------

def test_raw_item_through_anchor_to_readiness_never_manufactures_alias():
    """raw item (image_url=RAW, normalized_url=PROCESSED) -> canonical_style_this_anchor()
    -> resolve_board_image_candidate(anchor). The anchor must retain distinct
    image_url/normalized_url, and readiness must select the processed URL as
    renderable -- this is the exact chain that regressed test_style_this_persisted_shuffle.py
    when canonical_style_this_anchor manufactured image_url = normalized_url."""
    raw_item = _item(image_url=RAW, normalized_url=PROCESSED)

    anchor = canonical_style_this_anchor(raw_item)
    assert anchor["image_url"] == RAW
    assert anchor["normalized_url"] == PROCESSED
    assert anchor["safe_image_url"] == PROCESSED
    assert anchor["image_url"] != anchor["normalized_url"]

    result = resolve_board_image_candidate(anchor)
    assert result["renderable"] is True
    assert result["selected_url"] == PROCESSED
