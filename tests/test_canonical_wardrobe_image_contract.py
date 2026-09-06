"""Canonical wardrobe image provenance contract.

A board-safe asset must never originate from the raw upload. Two paths used to
break that:

1. r2_storage.upload_wardrobe_images normalized `masked_image_bytes or
   raw_image_bytes` and, on normalization failure, fell back to the same
   expression -- so a missing or failed mask published RAW PIXELS under
   wardrobe_{id}_normalized.png, a name every downstream consumer reads as
   processed.

2. wardrobe_capture promoted the generic legacy `image_url` (and then
   masked_image_url) into `normalized_url`, so a record could claim a
   normalized asset it never had.

Physically observed downstream: wardrobe records whose masked_url was an
opaque photo with no alpha channel, which the Style Board then rendered.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.r2_storage import R2Storage  # noqa: E402
from services.wardrobe_persistence_service import (  # noqa: E402
    canonicalize_wardrobe_image_write,
)

RAW = b"RAW-ORIGINAL-PHOTO-BYTES"
MASK = b"MASKED-RMBG-CUTOUT-BYTES"


class _FakeClient:
    """Records every object written so a test can assert on actual bytes."""

    def __init__(self) -> None:
        self.puts: dict[str, bytes] = {}

    def put_object(self, bucket, name, data, length=None, content_type=None):  # noqa: D102
        self.puts[name] = data.read()


@pytest.fixture()
def storage(monkeypatch):
    s = R2Storage()
    s.raw_bucket = "raw-bucket"
    s.raw_public_url = "https://raw.test"
    s.wardrobe_bucket = "wardrobe-bucket"
    s.wardrobe_public_url = "https://wardrobe.test"
    client = _FakeClient()
    monkeypatch.setattr(s, "_client", lambda: client)
    s._test_client = client  # type: ignore[attr-defined]
    return s


def _written(storage) -> dict[str, bytes]:
    return storage._test_client.puts  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# CASE 1 / 2 -- normalization source
# --------------------------------------------------------------------------

def test_normalized_is_derived_from_mask_not_raw(storage, monkeypatch):
    """CASE 1: masked available, normalize succeeds -> normalized from mask."""
    seen = {}

    def _fake_normalize(data, **kwargs):
        seen["input"] = data
        return b"NORMALIZED-FROM:" + data

    monkeypatch.setattr(
        "services.r2_storage.normalize_wardrobe_image_bytes", _fake_normalize
    )
    out = storage.upload_wardrobe_images(
        file_id="item1", raw_image_bytes=RAW, masked_image_bytes=MASK
    )
    assert seen["input"] == MASK, "normalization must never be fed raw bytes"
    assert out["normalized_image_url"]
    assert RAW not in _written(storage)["wardrobe_item1_normalized.png"]


def test_normalization_failure_falls_back_to_mask_never_raw(storage, monkeypatch):
    """CASE 2: normalize raises -> fall back to the mask, never the raw upload."""

    def _boom(data, **kwargs):
        raise RuntimeError("normalizer unavailable")

    monkeypatch.setattr("services.r2_storage.normalize_wardrobe_image_bytes", _boom)
    storage.upload_wardrobe_images(
        file_id="item2", raw_image_bytes=RAW, masked_image_bytes=MASK
    )
    assert _written(storage)["wardrobe_item2_normalized.png"] == MASK
    assert _written(storage)["wardrobe_item2_normalized.png"] != RAW


def test_missing_mask_yields_no_normalized_asset(storage, monkeypatch):
    """CASE 3 / 9: no mask -> no normalized asset at all; raw kept separately.

    This is the regression that mattered: previously the raw bytes were
    normalized and published under a wardrobe_*_normalized.png name.
    """
    monkeypatch.setattr(
        "services.r2_storage.normalize_wardrobe_image_bytes",
        lambda data, **kwargs: b"NORMALIZED-FROM:" + data,
    )
    out = storage.upload_wardrobe_images(
        file_id="item3", raw_image_bytes=RAW, masked_image_bytes=b""
    )
    written = _written(storage)
    assert "wardrobe_item3_normalized.png" not in written
    assert "wardrobe_item3.png" not in written
    assert out["normalized_image_url"] == ""
    assert out["masked_image_url"] == ""
    # Raw provenance is still preserved, in the raw bucket only.
    assert out["raw_image_url"] == "https://raw.test/raw_item3.png"
    assert written["raw_item3.png"] == RAW
    # And the legacy compatibility field must not become the raw upload.
    assert out["image_url"] == ""


# --------------------------------------------------------------------------
# CASE 4/5/6 -- persistence write gate
# --------------------------------------------------------------------------

def test_processed_field_aliasing_raw_is_rejected():
    """CASE 4 + 5: a processed field pointing at the raw object is not safe."""
    aliased = canonicalize_wardrobe_image_write(
        raw_url="https://raw.test/raw_x.png",
        masked_url="https://raw.test/raw_x.png?sig=1",
        normalized_url="https://raw.test/raw_x.png",
    )
    assert aliased["masked_url"] == ""
    assert aliased["normalized_url"] == ""
    assert aliased["safe_image_url"] == ""
    assert aliased["safe_image_source"] == "none"
    assert aliased["board_ready"] is False


def test_safe_image_precedence_and_board_ready():
    """CASE 10: masked wins; a valid normalized alone is still board-ready."""
    both = canonicalize_wardrobe_image_write(
        raw_url="https://raw.test/raw_y.png",
        masked_url="https://wardrobe.test/wardrobe_y.png",
        normalized_url="https://wardrobe.test/wardrobe_y_normalized.png",
    )
    assert both["safe_image_source"] == "masked_url"
    assert both["safe_image_url"] == "https://wardrobe.test/wardrobe_y.png"
    assert both["board_ready"] is True

    normalized_only = canonicalize_wardrobe_image_write(
        raw_url="https://raw.test/raw_z.png",
        masked_url="",
        normalized_url="https://wardrobe.test/wardrobe_z_normalized.png",
    )
    assert normalized_only["safe_image_source"] == "normalized_url"
    assert normalized_only["board_ready"] is True

    raw_only = canonicalize_wardrobe_image_write(
        raw_url="https://raw.test/raw_w.png", masked_url="", normalized_url=""
    )
    assert raw_only["safe_image_url"] == ""
    assert raw_only["board_ready"] is False
    assert raw_only["raw_url"] == "https://raw.test/raw_w.png"


def test_raw_is_never_the_safe_image():
    """The whole point: no input shape may make raw the safe asset."""
    for masked, normalized in [
        ("", ""),
        ("https://raw.test/raw_q.png", ""),
        ("", "https://raw.test/raw_q.png"),
        ("https://raw.test/raw_q.png?x=1", "https://raw.test/raw_q.png#f"),
    ]:
        result = canonicalize_wardrobe_image_write(
            raw_url="https://raw.test/raw_q.png",
            masked_url=masked,
            normalized_url=normalized,
        )
        assert result["safe_image_url"] != "https://raw.test/raw_q.png"
        assert result["board_ready"] is False


def test_privacy_catalog_only_is_board_ready_without_raw_or_mask():
    """CASE 7: face-risk items persist only the regenerated catalog image."""
    result = canonicalize_wardrobe_image_write(
        raw_url="",
        masked_url="",
        normalized_url="https://wardrobe.test/catalog_p.png",
        privacy_catalog_only=True,
    )
    assert result["raw_url"] == ""
    assert result["masked_url"] == ""
    assert result["normalized_url"] == "https://wardrobe.test/catalog_p.png"
    assert result["safe_image_source"] == "catalog"
    assert result["board_ready"] is True


def test_privacy_catalog_only_without_catalog_is_not_board_ready():
    result = canonicalize_wardrobe_image_write(
        raw_url="", masked_url="", normalized_url="", privacy_catalog_only=True
    )
    assert result["safe_image_url"] == ""
    assert result["safe_image_source"] == "none"
    assert result["board_ready"] is False
