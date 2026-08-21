"""Board-item image provenance guard (style_flow_service._adapt_board_item).

A raw image_url (often a selfie/mirror photo) must never be promoted to
board_image_url + cutout_ready. Only real transparent cutouts earn cutout_ready.
"""
from services.style_flow_service import (
    _adapt_board_item,
    _enrich_board_piece_from_wardrobe,
)

_SELFIE = "https://cdn/user/selfie.jpg"
_MASK = "https://cdn/masked/shirt.png"
_CAT = "https://cdn/catalog_jeans.png"


def test_raw_selfie_not_promoted_to_board_cutout():
    # Device failure: image_url is a selfie, no cutout/normalized.
    e = _adapt_board_item({"name": "Blue shirt", "role": "top", "image_url": _SELFIE})
    assert e is not None  # role kept
    assert e.get("board_image_url") is None  # NOT copied
    assert e.get("board_status") != "cutout_ready"  # NOT stamped ready
    assert e["image_url"] == _SELFIE  # raw preserved (frontend guard omits it)


def test_masked_cutout_wins_over_raw():
    e = _adapt_board_item(
        {"name": "Shirt", "role": "top", "image_url": _SELFIE, "masked_url": _MASK}
    )
    assert e["board_image_url"] == _MASK
    assert e["board_status"] == "cutout_ready"


def test_raw_plus_catalog_keeps_catalog_not_cutout():
    e = _adapt_board_item(
        {"name": "Jeans", "role": "bottom", "image_url": _SELFIE, "normalized_url": _CAT}
    )
    assert e.get("board_image_url") is None  # no forged cutout
    assert e.get("board_status") != "cutout_ready"
    assert e["normalized_url"] == _CAT  # frontend frames the catalog jeans


def test_bare_board_image_url_without_provenance_not_trusted():
    e = _adapt_board_item(
        {"name": "X", "role": "top", "board_image_url": _SELFIE, "image_url": _SELFIE}
    )
    assert e.get("board_image_url") is None  # bare board_image_url stripped
    assert e.get("board_status") != "cutout_ready"


def test_transparent_field_earns_cutout_ready():
    e = _adapt_board_item(
        {"name": "Tee", "role": "top", "transparent_url": _MASK}
    )
    assert e["board_image_url"] == _MASK
    assert e["board_status"] == "cutout_ready"


def test_no_name_skipped():
    assert _adapt_board_item({"role": "top", "masked_url": _MASK}) is None


def test_no_image_at_all_skipped():
    assert _adapt_board_item({"name": "Empty", "role": "top"}) is None


def test_forged_cutout_ready_is_scrubbed_for_raw_only():
    # Upstream falsely stamped cutout_ready on a raw item -> must be removed.
    e = _adapt_board_item(
        {"name": "Y", "role": "top", "image_url": _SELFIE, "board_status": "cutout_ready"}
    )
    assert e.get("board_status") != "cutout_ready"


def test_masked_url_aliasing_raw_is_not_a_cutout():
    # Device blocker: row.setdefault("masked_url", image) copies the raw selfie
    # into masked_url. That fabricated cutout must NOT be trusted as board-ready.
    e = _adapt_board_item(
        {"name": "Black T-Shirt", "role": "top", "image_url": _SELFIE, "masked_url": _SELFIE}
    )
    assert e.get("board_image_url") is None
    assert e.get("board_status") != "cutout_ready"


def test_enrich_pulls_catalog_from_wardrobe_record():
    # Board piece has only a fabricated masked=raw; the wardrobe record has the
    # real catalog image. Join by id -> piece resolves to framed catalog, not raw.
    by_id = {"tee-1": {"item_id": "tee-1", "normalized_url": _CAT}}
    piece = {"item_id": "tee-1", "name": "Black T-Shirt", "role": "top",
             "image_url": _SELFIE, "masked_url": _SELFIE}
    e = _adapt_board_item(_enrich_board_piece_from_wardrobe(piece, by_id))
    assert e.get("board_status") != "cutout_ready"
    assert e["normalized_url"] == _CAT


def test_enrich_pulls_real_cutout_from_wardrobe_record():
    # cutout_url requires cutout_status="ready" to earn cutout_ready - aligned
    # with the Flutter resolver, which gates cutout_url the same way. A bare
    # cutout_url with no status is covered by
    # test_bare_cutout_url_without_status_not_trusted below.
    by_id = {
        "tee-1": {
            "item_id": "tee-1",
            "cutout_url": _MASK,
            "cutout_status": "ready",
        }
    }
    piece = {"item_id": "tee-1", "name": "Black T-Shirt", "role": "top",
             "image_url": _SELFIE, "masked_url": _SELFIE}
    e = _adapt_board_item(_enrich_board_piece_from_wardrobe(piece, by_id))
    assert e["board_image_url"] == _MASK
    assert e["board_status"] == "cutout_ready"


def test_bare_cutout_url_without_status_not_trusted():
    # Device blocker (readiness-gate forensic): the backend previously trusted
    # a bare cutout_url unconditionally while Flutter required cutout_status
    # == "ready" for the same field - a real contract mismatch. Reconciled:
    # both sides now require the status.
    by_id = {"tee-1": {"item_id": "tee-1", "cutout_url": _MASK}}
    piece = {"item_id": "tee-1", "name": "Black T-Shirt", "role": "top",
             "image_url": _SELFIE, "masked_url": _SELFIE}
    e = _adapt_board_item(_enrich_board_piece_from_wardrobe(piece, by_id))
    assert e.get("board_image_url") is None
    assert e.get("board_status") != "cutout_ready"


def test_enrich_no_matching_record_is_unchanged():
    piece = {"item_id": "x", "name": "Y", "role": "top", "image_url": _SELFIE}
    assert _enrich_board_piece_from_wardrobe(piece, {}) == piece


def test_masked_alias_falls_back_to_catalog():
    # Same fabricated masked, but a real catalog image exists -> keep catalog,
    # not cutout_ready (opaque product image, frontend frames it).
    e = _adapt_board_item(
        {
            "name": "Black T-Shirt",
            "role": "top",
            "image_url": _SELFIE,
            "masked_url": _SELFIE,
            "normalized_url": _CAT,
        }
    )
    assert e.get("board_status") != "cutout_ready"
    assert e["normalized_url"] == _CAT
