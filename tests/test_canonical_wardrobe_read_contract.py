"""Canonical wardrobe image READ contract.

The write gate (test_canonical_wardrobe_image_contract) stops NEW records
from ever aliasing a processed field to the raw upload. It cannot fix rows
written before it existed, and every visual-board surface used to serialize
a wardrobe record's persisted `image_url` verbatim.

Live case this pins (candidate revision ahvi-backend-01041-qam, read-only):

    Black Loafers  820c0030-9249-4b33-a656-78059f8131f5
      image_url      -> RAW bucket object
      masked_url     -> genuine cutout (PNG colour-type 6)
      normalized_url -> genuine catalog cutout

Style Board, Style This and Daily Wear all emitted that raw image_url in
board_items[n].image_url. The canonical read contract resolves the same row
to the masked cutout, and the board must serialize THAT.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.style_board_image_readiness import (  # noqa: E402
    canonicalize_wardrobe_image_contract,
    serialize_wardrobe_board_item,
)
from services.style_flow_service import (  # noqa: E402
    _adapt_board_item,
    _canonicalize_card_items,
)
from services.board_service import (  # noqa: E402
    _decode_outfit_items,
    reenrich_saved_board_items,
)

RAW_HOST = "https://pub-9ca6234baa424e56882e953c97ffbe14.r2.dev"
WARDROBE_HOST = "https://pub-d4d02883ddda4a1bba452bfe6d1be814.r2.dev"

LOAFERS_ID = "820c0030-9249-4b33-a656-78059f8131f5"
LOAFERS_RAW = f"{RAW_HOST}/raw_{LOAFERS_ID}.png"
LOAFERS_MASK = f"{WARDROBE_HOST}/wardrobe_{LOAFERS_ID}.png"
LOAFERS_CATALOG = f"{WARDROBE_HOST}/catalog_{LOAFERS_ID}.png"


def _loafers() -> dict:
    """The persisted shape, field for field, as read from Appwrite."""
    return {
        "item_id": LOAFERS_ID,
        "name": "Black Loafers",
        "category": "Footwear",
        "role": "footwear",
        "image_url": LOAFERS_RAW,
        "masked_url": LOAFERS_MASK,
        "normalized_url": LOAFERS_CATALOG,
    }


# --------------------------------------------------------------------------
# PHASE 10 -- the Black Loafers raw leak
# --------------------------------------------------------------------------

def test_black_loafers_canonical_resolution_is_not_raw():
    contract = canonicalize_wardrobe_image_contract(_loafers())
    assert contract["safe_image_url"] == LOAFERS_MASK
    assert contract["safe_image_source"] == "masked_url"
    assert contract["board_ready"] is True
    assert contract["expected_transparent"] is True
    assert contract["safe_image_url"] != LOAFERS_RAW
    # The persisted legacy value is reported untouched, not rewritten.
    assert contract["image_url"] == LOAFERS_RAW


def test_black_loafers_board_item_serializes_safe_image_not_raw():
    """The regression: the serialized board item must not carry the raw URL
    in image_url, and must carry the safe asset there instead."""
    item = _adapt_board_item(_loafers())
    assert item is not None
    assert item["image_url"] == LOAFERS_MASK
    assert item["image_url"] != LOAFERS_RAW
    assert item["safe_image_url"] == LOAFERS_MASK
    assert item["board_ready"] is True
    # Raw provenance is preserved, just not as the presentation image.
    assert item["original_image_url"] == LOAFERS_RAW
    # ...and the frozen-snapshot triple must travel with the rewritten
    # image_url, or lib/util/wardrobe_image_resolver.dart rejects the mask as
    # a self-alias and renders an empty hanger (the ce4ade1 regression).
    assert item["selected_field"]
    assert item["source_kind"] == "processed_cutout"
    assert item["expected_transparent"] is True


def test_no_raw_host_anywhere_in_board_item_presentation_fields():
    item = _adapt_board_item(_loafers())
    assert item is not None
    for field in ("image_url", "safe_image_url", "board_image_url", "masked_url"):
        assert RAW_HOST not in str(item.get(field) or ""), field


# --------------------------------------------------------------------------
# PHASE 11 -- raw-only record
# --------------------------------------------------------------------------

def _raw_only() -> dict:
    return {
        "item_id": "raw-only-1",
        "name": "Unprocessed Shirt",
        "category": "Tops",
        "role": "top",
        "image_url": f"{RAW_HOST}/raw_rawonly1.png",
        "masked_url": "",
        "normalized_url": "",
    }


def test_raw_only_record_is_not_board_ready():
    contract = canonicalize_wardrobe_image_contract(_raw_only())
    assert contract["safe_image_url"] == ""
    assert contract["safe_image_source"] == "none"
    assert contract["board_ready"] is False
    assert contract["expected_transparent"] is False


def test_raw_only_record_is_excluded_from_visual_boards():
    assert serialize_wardrobe_board_item(_raw_only()) is None
    # And the board adapter skips it rather than emitting the raw upload.
    assert _adapt_board_item(_raw_only()) is None


def test_raw_only_masked_alias_is_still_excluded():
    """masked_url that merely aliases the raw upload is fabricated provenance."""
    aliased = _raw_only()
    aliased["masked_url"] = aliased["image_url"] + "?sig=cachebust"
    assert canonicalize_wardrobe_image_contract(aliased)["board_ready"] is False
    assert _adapt_board_item(aliased) is None


# --------------------------------------------------------------------------
# PHASE 12 -- mixed legacy record (raw image_url + real processed assets)
# --------------------------------------------------------------------------

def test_mixed_legacy_record_prefers_mask_over_catalog():
    contract = canonicalize_wardrobe_image_contract(_loafers())
    assert contract["safe_image_source"] == "masked_url"
    assert contract["safe_image_url"] == LOAFERS_MASK
    assert contract["masked_url"] == LOAFERS_MASK
    assert contract["normalized_url"] == LOAFERS_CATALOG


def test_mixed_legacy_record_falls_to_catalog_when_mask_missing():
    row = _loafers()
    row["masked_url"] = ""
    contract = canonicalize_wardrobe_image_contract(row)
    assert contract["safe_image_url"] == LOAFERS_CATALOG
    assert contract["safe_image_source"] == "normalized_url"
    assert contract["board_ready"] is True
    # A framed catalog shot is not a transparent cutout.
    assert contract["expected_transparent"] is False
    item = _adapt_board_item(row)
    assert item is not None
    assert item["image_url"] == LOAFERS_CATALOG
    assert item["original_image_url"] == LOAFERS_RAW


# --------------------------------------------------------------------------
# PHASE 13 -- privacy / catalog-only record
# --------------------------------------------------------------------------

def _catalog_only() -> dict:
    """The shape the write contract now produces under
    WARDROBE_PRIVACY_CATALOG_ONLY: no raw, no mask, catalog in both fields.
    Verified live on 3d7ded9b-dc20-4278-b1fc-0e385b27becb."""
    url = f"{WARDROBE_HOST}/catalog_3d7ded9b.png"
    return {
        "item_id": "3d7ded9b",
        "name": "Pink T-Shirt",
        "category": "Tops",
        "role": "top",
        "image_url": url,
        "masked_url": "",
        "normalized_url": url,
    }


def test_privacy_catalog_only_is_board_ready():
    contract = canonicalize_wardrobe_image_contract(_catalog_only())
    assert contract["board_ready"] is True
    assert contract["safe_image_source"] == "catalog"
    assert contract["safe_image_url"] == _catalog_only()["normalized_url"]
    # image_url == normalized_url must NOT be read as a raw alias here: the
    # record has no raw provenance at all.
    assert contract["expected_transparent"] is False


def test_privacy_catalog_only_board_item_has_no_self_aliased_image_url():
    """No distinct upload exists, so image_url must be dropped rather than
    republished as a copy of the winning field - a self-alias is exactly what
    the Flutter resolver rejects."""
    item = _adapt_board_item(_catalog_only())
    assert item is not None
    assert item["safe_image_url"] == _catalog_only()["normalized_url"]
    assert item["board_ready"] is True
    assert "original_image_url" not in item
    assert item.get("image_url", "") == ""


# --------------------------------------------------------------------------
# Masked-only record (no raw upload) - must not regress to an empty hanger
# --------------------------------------------------------------------------

def test_serializer_is_idempotent():
    """Re-serializing an already-serialized item must not lose the rewrite.

    original_image_url is stronger provenance than the generic image_url; if
    image_url were consulted first, a second pass would see image_url == safe,
    conclude there is no distinct upload, and drop both fields - silently
    un-doing the fix wherever two serialization passes overlap.
    """
    once = serialize_wardrobe_board_item(_loafers())
    twice = serialize_wardrobe_board_item(once)
    assert twice is not None
    assert twice["image_url"] == LOAFERS_MASK
    assert twice["original_image_url"] == LOAFERS_RAW
    assert twice["safe_image_url"] == LOAFERS_MASK
    assert twice["source_kind"] == "processed_cutout"


def test_frozen_snapshot_exemption_does_not_admit_a_real_selfie():
    """The frozen-snapshot carve-out exempts image_url from the raw-alias veto.

    It must NOT exempt the actual upload: a snapshot whose "mask" is genuinely
    the raw photo still has to be rejected, or the carve-out becomes a way to
    dress a selfie up as a cutout by attaching two metadata fields.
    """
    selfie = f"{RAW_HOST}/raw_selfie.png"
    forged = {
        "item_id": "forged-1",
        "name": "Forged",
        "selected_field": "masked_url",
        "source_kind": "processed_cutout",
        "image_url": selfie,
        "masked_url": selfie,
        "original_image_url": selfie,
    }
    contract = canonicalize_wardrobe_image_contract(forged)
    assert contract["safe_image_url"] == ""
    assert contract["board_ready"] is False
    assert serialize_wardrobe_board_item(forged) is None


def test_card_items_list_is_canonicalized_not_only_board_items():
    """Live leak: cards[n].board_items was fixed but cards[n].items still
    carried the raw upload in image_url - both lists reach the client."""
    card = {"title": "Look", "items": [_loafers()]}
    _canonicalize_card_items(card, {})
    item = card["items"][0]
    assert item["image_url"] == LOAFERS_MASK
    assert item["image_url"] != LOAFERS_RAW
    assert item["safe_image_url"] == LOAFERS_MASK
    assert item["original_image_url"] == LOAFERS_RAW


def test_card_items_keep_membership_but_lose_images_when_unready():
    """Membership drives roles/slots and the "n of m locked" counts, so an
    unready item stays in the list - it just carries no image at all."""
    card = {"title": "Look", "items": [_raw_only()]}
    _canonicalize_card_items(card, {})
    item = card["items"][0]
    assert item["name"] == "Unprocessed Shirt"
    assert item["board_ready"] is False
    assert item.get("image_url", "") == ""
    assert RAW_HOST not in json_dumps_safe(item)


def json_dumps_safe(obj) -> str:
    import json

    return json.dumps(obj, default=str)


def test_saved_board_reenriches_stale_raw_item_from_current_wardrobe():
    """PHASE 9: a board saved before processing finished froze the raw upload
    as its selected asset. Re-serving that is a raw photo on a visual board."""
    saved = [
        {
            "id": LOAFERS_ID,
            "name": "Black Loafers",
            "role": "footwear",
            "image_url": LOAFERS_RAW,
            "selected_field": "image_url",
            "source_kind": "original",
            "expected_transparent": False,
        }
    ]
    out = reenrich_saved_board_items(saved, {LOAFERS_ID: _loafers()})
    assert out[0]["image_url"] == LOAFERS_MASK
    assert out[0]["image_url"] != LOAFERS_RAW
    assert out[0]["safe_image_url"] == LOAFERS_MASK
    assert out[0]["source_kind"] == "processed_cutout"
    assert out[0]["board_ready"] is True
    assert out[0]["original_image_url"] == LOAFERS_RAW


def test_saved_board_strips_image_when_item_no_longer_board_ready():
    saved = [{"id": "raw-only-1", "name": "Unprocessed Shirt", "image_url": f"{RAW_HOST}/raw_rawonly1.png"}]
    out = reenrich_saved_board_items(saved, {"raw-only-1": _raw_only()})
    assert out[0].get("image_url", "") == ""
    assert out[0]["board_ready"] is False
    assert RAW_HOST not in str(out[0])


def test_saved_board_leaves_unknown_items_untouched():
    saved = [{"id": "not-in-wardrobe", "name": "Gone", "image_url": "https://x/y.png"}]
    out = reenrich_saved_board_items(saved, {LOAFERS_ID: _loafers()})
    assert out[0]["image_url"] == "https://x/y.png"


def test_saved_board_outfit_items_decode_from_exploded_string():
    """Appwrite stores the JSON payload exploded one character per element."""
    payload = '[{"id":"a","image_url":"https://x/y.png"}]'
    assert _decode_outfit_items(list(payload)) == [{"id": "a", "image_url": "https://x/y.png"}]
    assert _decode_outfit_items(payload) == [{"id": "a", "image_url": "https://x/y.png"}]
    assert _decode_outfit_items(None) is None
    assert _decode_outfit_items(["not json"]) is None


def test_masked_only_record_keeps_winning_field_and_drops_image_url():
    row = {
        "item_id": "masked-only-1",
        "name": "Cutout Only Top",
        "category": "Tops",
        "role": "top",
        "masked_url": f"{WARDROBE_HOST}/wardrobe_maskedonly1.png",
    }
    item = _adapt_board_item(row)
    assert item is not None
    assert item["masked_url"] == row["masked_url"]
    assert item["safe_image_url"] == row["masked_url"]
    assert item.get("image_url", "") == ""
    assert "original_image_url" not in item
