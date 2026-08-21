"""services.style_board_image_readiness - the one shared authority for whether
an item has a genuine, board-safe processed image. Aligned with the Flutter
wardrobe_image_resolver contract; see the readiness-gate forensic for the
exact matrix this implements.
"""

from services.style_board_image_readiness import (
    is_board_renderable,
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
