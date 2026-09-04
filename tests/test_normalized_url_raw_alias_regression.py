"""Phase 2C integration regression: prove the persistence boundary
(services.wardrobe_persistence_service) and the Style Board readiness
boundary (services.style_board_image_readiness) agree on the same
normalized_url raw-alias invariant, without any Appwrite/network IO.

Live device bug: a wardrobe item's normalized_url was byte-identical to its
raw image_url. The old backend persisted it unguarded, and the old Flutter
resolver (lib/util/wardrobe_image_resolver.dart, fixed in e8a53d6) trusted
any normalized_url as board-safe. This test drives one item through both
backend defense layers and asserts neither one ever selects the raw upload
as a processed/catalog source.

Expected to FAIL until Phase 3 fixes both boundaries.
"""

from services import wardrobe_persistence_service as persistence
from services.style_board_image_readiness import resolve_board_image_candidate

_RAW = "https://cdn.test/user/raw_photo.jpg"


def test_raw_alias_rejected_by_both_persistence_and_readiness_boundaries():
    # 1. Persistence boundary: build the document that would be sent to
    #    Appwrite for an item whose normalized_url aliases its raw upload.
    doc = persistence._build_appwrite_doc(
        user_id="user-1",
        file_id="item-1",
        item={"name": "Blue Shirt", "category": "Tops"},
        raw_url=_RAW,
        masked_url="",
        normalized_url=_RAW,
    )
    assert doc["image_url"] == _RAW
    assert doc["normalized_url"] == ""

    # 2. Readiness boundary: even if a legacy record somehow still has the
    #    alias stored (pre-fix data), the read-side gate must not classify
    #    it as a board-safe catalog source.
    legacy_stored_item = {"image_url": _RAW, "normalized_url": _RAW}
    result = resolve_board_image_candidate(legacy_stored_item)
    assert result["selected_field"] != "normalized_url"
    assert result["renderable"] is False

    # 3. The sanitized persistence output, fed back through the readiness
    #    gate, must also never select the raw upload as processed/catalog.
    sanitized_as_stored = {"image_url": doc["image_url"], "normalized_url": doc["normalized_url"]}
    result2 = resolve_board_image_candidate(sanitized_as_stored)
    assert result2["selected_field"] != "normalized_url"
