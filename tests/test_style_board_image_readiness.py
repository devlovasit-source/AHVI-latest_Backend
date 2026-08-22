"""services.style_board_image_readiness - the one shared authority for whether
an item has a genuine, board-safe processed image. Aligned with the Flutter
wardrobe_image_resolver contract; see the readiness-gate forensic for the
exact matrix this implements.
"""

from services.style_board_image_readiness import (
    is_board_renderable,
    project_board_image_fields,
    resolve_board_image_candidate,
)

_RAW = "https://cdn/user/raw_photo.jpg"
_MASK = "https://cdn/masked/shirt.png"
_CAT = "https://cdn/normalized/shirt.png"
_RMBG = "https://cdn/rmbg/shirt.png"
_PROCESSED = "https://cdn/processed/shirt.png"


# A. masked_url == image_url -> NOT renderable
def test_masked_equals_image_not_renderable():
    item = {"image_url": _RAW, "masked_url": _RAW}
    assert is_board_renderable(item) is False


# B. genuine masked_url -> renderable
def test_genuine_masked_url_renderable():
    item = {"image_url": _RAW, "masked_url": _MASK}
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is True
    assert result["selected_field"] == "masked_url"


# C. normalized_url -> renderable
def test_normalized_url_renderable():
    item = {"image_url": _RAW, "normalized_url": _CAT}
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is True
    assert result["selected_field"] == "normalized_url"
    assert result["reason"] == "catalog_normalized"


def test_normalized_url_equal_to_image_url_still_renderable():
    # Device blocker (readiness-gate final device gate): unlike masked_url,
    # normalized_url is never alias-checked against image_url - it matches
    # the Flutter wardrobe_image_resolver contract, which admits its
    # unconditional normalized_url candidates without an alias check
    # (normalized_url represents a regenerated catalog shot, not a raw
    # upload copy, by contract - even when a given item's value happens to
    # be identical to its image_url).
    item = {"image_url": _RAW, "normalized_url": _RAW}
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is True
    assert result["selected_field"] == "normalized_url"


# D. valid RMBG processed_url + completed status -> renderable
def test_processed_url_with_complete_status_renderable():
    item = {"image_url": _RAW, "processed_url": _PROCESSED, "image_status": "rmbg_complete"}
    assert is_board_renderable(item) is True


def test_processed_url_without_complete_status_not_renderable():
    item = {"image_url": _RAW, "processed_url": _PROCESSED}
    assert is_board_renderable(item) is False


# E. valid rmbg_url + completed status -> renderable
def test_rmbg_url_with_complete_status_renderable():
    item = {"image_url": _RAW, "rmbg_url": _RMBG, "image_status": "rmbg_complete"}
    assert is_board_renderable(item) is True


def test_rmbg_url_without_complete_status_not_renderable():
    item = {"image_url": _RAW, "rmbg_url": _RMBG, "image_status": "rmbg_pending"}
    assert is_board_renderable(item) is False


# F. forged board_status without actual processed candidate -> NOT renderable
def test_forged_board_status_without_candidate_not_renderable():
    item = {"image_url": _RAW, "board_status": "cutout_ready"}
    assert is_board_renderable(item) is False


def test_board_image_url_with_correct_status_renderable():
    item = {
        "image_url": _RAW,
        "board_image_url": _MASK,
        "board_status": "cutout_ready",
    }
    assert is_board_renderable(item) is True


def test_board_image_url_aliasing_raw_not_renderable_even_with_status():
    # Forged provenance: board_image_url == raw AND a (wrongly) stamped
    # "cutout_ready" status. Must not be trusted.
    item = {
        "image_url": _RAW,
        "board_image_url": _RAW,
        "board_status": "cutout_ready",
    }
    assert is_board_renderable(item) is False


# G. raw image only -> NOT renderable
def test_raw_image_only_not_renderable():
    item = {"image_url": _RAW}
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is False
    assert result["reason"] == "no_board_safe_image"


def test_empty_item_not_renderable():
    assert is_board_renderable({}) is False


def test_non_dict_not_renderable():
    assert is_board_renderable(None) is False
    assert is_board_renderable("not a dict") is False


def test_cutout_url_requires_ready_status():
    item = {"image_url": _RAW, "cutout_url": _MASK}
    assert is_board_renderable(item) is False
    item["cutout_status"] = "ready"
    assert is_board_renderable(item) is True


def test_transparent_url_unconditional_like_masked_url():
    item = {"image_url": _RAW, "transparent_url": _MASK}
    assert is_board_renderable(item) is True


# ---------------------------------------------- project_board_image_fields
#
# Board serializers (initial Style This's _lite_item, registration) need the
# winning candidate carried through under its own canonical field + status,
# not flattened into a single image_url - collapsing it is exactly how a
# renderable item becomes an empty hanger downstream.


def test_project_not_renderable_returns_empty_dict():
    assert project_board_image_fields({"image_url": _RAW}) == {}
    assert project_board_image_fields({}) == {}
    assert project_board_image_fields(None) == {}


def test_project_masked_url():
    item = {"image_url": _RAW, "masked_url": _MASK}
    assert project_board_image_fields(item) == {"masked_url": _MASK}


def test_project_normalized_url():
    item = {"image_url": _RAW, "normalized_url": _CAT}
    assert project_board_image_fields(item) == {"normalized_url": _CAT}


def test_project_cutout_url_carries_status():
    item = {"image_url": _RAW, "cutout_url": _MASK, "cutout_status": "ready"}
    assert project_board_image_fields(item) == {
        "cutout_url": _MASK,
        "cutout_status": "ready",
    }


def test_project_rmbg_url_carries_status():
    item = {"image_url": _RAW, "rmbg_url": _RMBG, "image_status": "rmbg_complete"}
    assert project_board_image_fields(item) == {
        "rmbg_url": _RMBG,
        "image_status": "rmbg_complete",
    }


def test_project_processed_url_carries_status():
    item = {"image_url": _RAW, "processed_url": _PROCESSED, "image_status": "rmbg_complete"}
    assert project_board_image_fields(item) == {
        "processed_url": _PROCESSED,
        "image_status": "rmbg_complete",
    }


def test_project_board_image_url_carries_status():
    item = {"image_url": _RAW, "board_image_url": _MASK, "board_status": "cutout_ready"}
    assert project_board_image_fields(item) == {
        "board_image_url": _MASK,
        "board_status": "cutout_ready",
    }


def test_project_transparent_url_maps_to_masked_url():
    # The Flutter wardrobe_image_resolver's non-asset branch never reads a
    # bare transparent_url field - only masked_url (which is also what
    # wardrobe_persistence_service folds transparent_url into on write).
    # Projecting it under its own name would silently stop rendering once
    # the item reaches Flutter, even though is_board_renderable() says True.
    item = {"image_url": _RAW, "transparent_url": _MASK}
    assert project_board_image_fields(item) == {"masked_url": _MASK}


def test_project_transparent_url_camel_case_maps_to_masked_url():
    item = {"image_url": _RAW, "transparentUrl": _MASK}
    assert project_board_image_fields(item) == {"masked_url": _MASK}


def test_transparent_url_camel_case_unconditional():
    item = {"image_url": _RAW, "transparentUrl": _MASK}
    assert is_board_renderable(item) is True


def test_rmbg_url_camel_case_with_complete_status_renderable():
    item = {"image_url": _RAW, "rmbgUrl": _RMBG, "image_status": "rmbg_complete"}
    assert is_board_renderable(item) is True
    assert project_board_image_fields(item) == {
        "rmbg_url": _RMBG,
        "image_status": "rmbg_complete",
    }


def test_processed_url_camel_case_with_complete_status_renderable():
    item = {
        "image_url": _RAW,
        "processedUrl": _PROCESSED,
        "image_status": "rmbg_complete",
    }
    assert is_board_renderable(item) is True
    assert project_board_image_fields(item) == {
        "processed_url": _PROCESSED,
        "image_status": "rmbg_complete",
    }


def test_transparent_image_url_camel_case_normalized_to_snake_case():
    item = {"image_url": _RAW, "transparentImageUrl": _MASK}
    assert is_board_renderable(item) is True
    assert project_board_image_fields(item) == {"transparent_image_url": _MASK}


def test_project_camel_case_field_normalized_to_snake_case():
    item = {"image_url": _RAW, "cutoutUrl": _MASK, "cutout_status": "ready"}
    assert project_board_image_fields(item) == {
        "cutout_url": _MASK,
        "cutout_status": "ready",
    }


# ---------------------------------------------- URL-identity alias parity
#
# lib/util/wardrobe_image_resolver.dart's _urlIdentity() compares scheme
# (lower) + host (lower) + explicit port + path only - query and fragment are
# always dropped. A masked_url that is a signed/cache-busted copy of
# image_url (same host+path, different query token) is exact-string-
# different but Flutter's _urlIdentity() calls it the SAME resource and
# rejects it as a fabricated alias. The live device bug: backend's old
# alias check was plain string equality, so it missed exactly this case and
# admitted an item Flutter would - and did - render as an empty hanger.


def test_query_only_difference_is_alias():
    item = {
        "image_url": "https://cdn.test/item.png?token=abc",
        "masked_url": "https://cdn.test/item.png?token=xyz",
    }
    assert is_board_renderable(item) is False


def test_fragment_only_difference_is_alias():
    item = {
        "image_url": "https://cdn.test/item.png#one",
        "masked_url": "https://cdn.test/item.png#two",
    }
    assert is_board_renderable(item) is False


def test_scheme_and_host_casing_difference_is_alias():
    item = {
        "image_url": "https://CDN.test/item.png",
        "masked_url": "HTTPS://cdn.TEST/item.png",
    }
    assert is_board_renderable(item) is False


def test_query_and_fragment_together_is_alias():
    item = {
        "image_url": "https://cdn.test/item.png",
        "masked_url": "https://cdn.test/item.png?v=2#frag",
    }
    assert is_board_renderable(item) is False


def test_different_object_key_stays_distinct_and_renderable():
    """H: genuinely different object keys must never be treated as aliases."""
    item = {
        "image_url": "https://cdn.test/raw/item.png",
        "masked_url": "https://cdn.test/masked/item.png",
    }
    assert is_board_renderable(item) is True


def test_same_filename_different_path_stays_distinct_and_renderable():
    """I: same filename, different directory - must stay distinct."""
    item = {
        "image_url": "https://cdn.test/raw/item.png",
        "masked_url": "https://cdn.test/processed/item.png",
    }
    assert is_board_renderable(item) is True


def test_trailing_slash_difference_stays_distinct():
    """F: _urlIdentity() does not normalize trailing slashes - paths that
    differ only by a trailing slash are NOT the same identity in Flutter,
    so the backend must not treat them as aliases either."""
    item = {
        "image_url": "https://cdn.test/item.png/",
        "masked_url": "https://cdn.test/item.png",
    }
    assert is_board_renderable(item) is True


def test_normalized_url_still_never_alias_checked_against_raw():
    """Guard against over-normalizing: normalized_url keeps its existing
    unconditional catalog contract - it must not start being rejected via
    the new identity check just because its host/path happens to match the
    raw image_url (that guarantee predates this fix and Flutter doesn't
    alias-check normalized_url against image_url either)."""
    item = {
        "image_url": "https://cdn.test/item.png?sig=abc",
        "normalized_url": "https://cdn.test/item.png?sig=xyz",
    }
    result = resolve_board_image_candidate(item)
    assert result["renderable"] is True
    assert result["selected_field"] == "normalized_url"


def test_genuine_masked_image_with_query_difference_from_unrelated_raw():
    """G/genuine-distinct control: a real cutout hosted on a different path
    must remain renderable even though it happens to carry its own query
    string - the identity check must not become overzealous."""
    item = {
        "image_url": "https://cdn.test/raw/item.png?x=1",
        "masked_url": "https://cdn.test/masked/item.png?x=2",
    }
    assert is_board_renderable(item) is True
