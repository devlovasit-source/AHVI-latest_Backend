"""services.style_board_image_readiness - known-unsafe-masked-item denylist.

P0 forensic follow-up to the byte-identity guard (test_style_board_image_
readiness_byte_identity.py). "Black T-Shirt" (item id
6a96d6cf-9916-4594-a141-a3eba776700c) has a masked_url that is NOT byte-
identical to its raw upload (RMBG ran and produced distinct bytes) but is
still unsafe: direct visual inspection shows the wearer's chin/beard/arm,
not an isolated garment. Byte-identity cannot see this - only the item-ID
denylist can. A safe normalized_url catalog cutout exists for this item, so
the fix is precedence (fall through to it), not global masked_url rejection.

Network calls are mocked - unit tests only.
"""

import services.style_board_image_readiness as readiness
from services.style_board_image_readiness import resolve_board_image_candidate

_TSHIRT_ID = "6a96d6cf-9916-4594-a141-a3eba776700c"
_LOAFERS_ID = "45aa2af8-2f6a-47ec-9055-bcb99d9f5d18"
_GOOD_ITEM_ID = "aaaaaaaa-0000-0000-0000-000000000000"


def setup_function(_):
    readiness._byte_identity_cache.clear()


def _no_fetch(monkeypatch):
    # These tests never rely on the byte-identity network path; force it to
    # report "distinct" (never identical) so denylist behavior is isolated.
    monkeypatch.setattr(readiness, "_fetch_bounded_hash", lambda url: url)


# 1. Black Loafers raw-byte alias must still be rejected (byte-identity
# guard unchanged by the denylist addition).
def test_black_loafers_byte_identity_still_rejected(monkeypatch):
    shared_hash = "same-hash"
    monkeypatch.setattr(
        readiness,
        "_fetch_bounded_hash",
        lambda url: shared_hash,
    )
    item = {
        "id": _LOAFERS_ID,
        "image_url": "https://cdn.test/raw_loafers.png",
        "masked_url": "https://other.test/masked_loafers.png",
        "normalized_url": "https://cdn.test/catalog_loafers.png",
    }
    result = resolve_board_image_candidate(item)
    assert result["selected_field"] == "normalized_url"


# 2 & 3. Black T-Shirt's exact known state: bad masked_url not selected,
# safe normalized_url selected instead.
def test_black_tshirt_bad_mask_rejected_safe_alternative_selected(monkeypatch):
    _no_fetch(monkeypatch)
    item = {
        "id": _TSHIRT_ID,
        "image_url": "https://cdn.test/raw_tshirt.png",
        "masked_url": "https://cdn.test/wardrobe_tshirt.png",
        "normalized_url": "https://cdn.test/catalog_tshirt.png",
    }
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is True
    assert result["selected_field"] == "normalized_url"
    assert result["selected_url"] != item["masked_url"]


# Black T-Shirt with no safe alternative: must not be admitted (never falls
# back to the unsafe masked_url or the raw upload).
def test_black_tshirt_without_safe_alternative_not_admitted(monkeypatch):
    _no_fetch(monkeypatch)
    item = {
        "id": _TSHIRT_ID,
        "image_url": "https://cdn.test/raw_tshirt.png",
        "masked_url": "https://cdn.test/wardrobe_tshirt.png",
    }
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is False
    assert result["reason"] == "no_board_safe_image"


# 4 & 5. A different item with a genuinely good masked asset (distinct
# bytes, not on the denylist) remains admitted - the denylist is scoped to
# the one proven-bad ID, not a population-wide rule.
def test_good_masked_asset_on_other_item_remains_admitted(monkeypatch):
    _no_fetch(monkeypatch)
    item = {
        "id": _GOOD_ITEM_ID,
        "image_url": "https://cdn.test/raw_other.png",
        "masked_url": "https://cdn.test/masked_other.png",
    }
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is True
    assert result["selected_field"] == "masked_url"


# 6. Repeated normalization is idempotent - the denylist is a pure lookup,
# no state mutation.
def test_black_tshirt_resolution_idempotent(monkeypatch):
    _no_fetch(monkeypatch)
    item = {
        "id": _TSHIRT_ID,
        "image_url": "https://cdn.test/raw_tshirt.png",
        "masked_url": "https://cdn.test/wardrobe_tshirt.png",
        "normalized_url": "https://cdn.test/catalog_tshirt.png",
    }
    first = resolve_board_image_candidate(item)
    second = resolve_board_image_candidate(item)
    assert first == second


# 7. Raw-only Black T-Shirt (no masked_url, no normalized_url) is never
# admitted.
def test_black_tshirt_raw_only_never_admitted(monkeypatch):
    _no_fetch(monkeypatch)
    item = {"id": _TSHIRT_ID, "image_url": "https://cdn.test/raw_tshirt.png"}
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is False
