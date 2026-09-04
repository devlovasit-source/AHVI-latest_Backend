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
        _item(
            cutout_url=PROCESSED + "-cutout",
            cutout_status="ready",
            normalized_url=PROCESSED,
            image_url=RAW,
        )
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

def test_raw_only_returns_none_without_allow_missing_image():
    """image_url is never a resolve_board_image_candidate candidate field --
    a raw-only item has no board-safe image at all, so canonical_style_this_anchor
    now correctly returns None (delegating entirely to the canonical resolver)
    rather than the old behavior of fabricating a "safe" image_url that was
    never actually safe."""
    anchor = canonical_style_this_anchor(_item(image_url=RAW))
    assert anchor is None


def test_raw_only_with_allow_missing_image_has_no_renderable_candidate():
    anchor = canonical_style_this_anchor(_item(image_url=RAW), allow_missing_image=True)
    assert anchor["image_url"] == RAW
    assert not anchor.get("safe_image_url")
    result = resolve_board_image_candidate(anchor)
    assert result["renderable"] is False


# ---------- d014 live shape: aliased normalized_url with a genuinely distinct masked_url ----------

def test_aliased_normalized_with_distinct_masked_selects_masked():
    """Live regression: services/style_this_anchor.py's own _SAFE_IMAGE_FIELDS
    ranked normalized_url before masked_url and never alias-checked its pick,
    so an item with image_url == normalized_url (legacy alias) but a genuinely
    distinct masked_url got the ALIASED field selected as safe_image_url --
    confirmed live on wardrobe item d0140d0c-69fc-459b-a1c9-1f2fe799e249.
    Delegating to resolve_board_image_candidate (masked_url checked before
    normalized_url, with alias-checking) fixes this."""
    X, Y = "https://images.test/X-aliased.png", "https://images.test/Y-masked.png"
    anchor = canonical_style_this_anchor(_item(image_url=X, normalized_url=X, masked_url=Y))

    assert anchor["image_url"] == X
    assert anchor["normalized_url"] == X
    assert anchor["masked_url"] == Y
    assert anchor["safe_image_url"] == Y
    assert anchor["source_kind"] == "wardrobe_masked"

    result = resolve_board_image_candidate(anchor)
    assert result["renderable"] is True
    assert result["selected_url"] == Y


def test_pure_normalized_raw_alias_with_no_other_field_returns_none():
    X = "https://images.test/X-only-alias.png"
    anchor = canonical_style_this_anchor(_item(image_url=X, normalized_url=X))
    assert anchor is None


def test_masked_only_selects_masked_source_kind():
    Y = "https://images.test/masked-only.png"
    anchor = canonical_style_this_anchor(_item(masked_url=Y))
    assert anchor["safe_image_url"] == Y
    assert anchor["source_kind"] == "wardrobe_masked"
    assert not anchor.get("image_url")


# ---------- support-item admission chain: aliased item must only be admitted via its genuine masked asset ----------

def test_d014_shape_support_item_admitted_only_via_masked_never_via_aliased_normalized():
    """The serialized item (as it would appear in a board's items/board_items)
    must carry the genuine masked_url distinctly from the aliased image_url/
    normalized_url pair, so a downstream consumer (Flutter's own resolver, or
    resolve_board_image_candidate again) can only ever resolve the genuine
    masked asset -- never the aliased one -- for this item."""
    X, Y = "https://images.test/X-aliased.png", "https://images.test/Y-masked.png"
    source_item = _item(item_id="d014-like", image_url=X, normalized_url=X, masked_url=Y)

    anchor = canonical_style_this_anchor(source_item)
    assert anchor["image_url"] == X
    assert anchor["normalized_url"] == X
    assert anchor["masked_url"] == Y

    result = resolve_board_image_candidate(anchor)
    assert result["selected_field"] in {"masked_url", "maskedUrl"}
    assert result["selected_url"] == Y
    assert result["selected_url"] != anchor["normalized_url"]


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
