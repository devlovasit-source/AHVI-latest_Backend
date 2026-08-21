"""routers.data generic outfits write path must never fabricate masked_url
by copying image_url - that falsely claims a raw/selfie photo is a
processed cutout (see services.style_board_image_readiness). A genuinely
missing masked_url is a rejected write, not a forged one.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routers import data  # noqa: E402


def _request_for_user(user_id="owner-1"):
    return SimpleNamespace(state=SimpleNamespace(user={"user_id": user_id}))


# P. missing masked_url no longer aliases image_url
def test_missing_masked_url_is_rejected_not_fabricated():
    with patch.object(data.qdrant_service, "enabled", return_value=False):
        with pytest.raises(HTTPException) as exc_info:
            data.create_document(
                _request_for_user("owner-1"),
                data.CreateRequest(
                    resource="outfits",
                    data={
                        "name": "Test Shirt",
                        "category": "Tops",
                        "image_url": "https://cdn/raw/shirt.jpg",
                    },
                ),
            )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["missing"]["masked_url"] is True


def test_genuine_masked_url_is_preserved_unchanged():
    captured = {}

    def fake_create(*, resource, payload, document_id):
        captured["payload"] = payload
        return {"$id": "doc_1", **payload}

    with patch.object(data.qdrant_service, "enabled", return_value=False), patch.object(
        data, "_create_document_with_schema_retries", side_effect=fake_create
    ):
        data.create_document(
            _request_for_user("owner-1"),
            data.CreateRequest(
                resource="outfits",
                data={
                    "name": "Test Shirt",
                    "category": "Tops",
                    "image_url": "https://cdn/raw/shirt.jpg",
                    "masked_url": "https://cdn/masked/shirt.png",
                },
            ),
        )

    assert captured["payload"]["image_url"] == "https://cdn/raw/shirt.jpg"
    assert captured["payload"]["masked_url"] == "https://cdn/masked/shirt.png"
    assert captured["payload"]["masked_url"] != captured["payload"]["image_url"]


def test_missing_image_url_falls_back_to_genuine_masked_url():
    # The one direction of fallback that IS safe: a real masked_url can
    # stand in for a missing image_url (it's a real processed image, just
    # promoted to the display field).
    captured = {}

    def fake_create(*, resource, payload, document_id):
        captured["payload"] = payload
        return {"$id": "doc_1", **payload}

    with patch.object(data.qdrant_service, "enabled", return_value=False), patch.object(
        data, "_create_document_with_schema_retries", side_effect=fake_create
    ):
        data.create_document(
            _request_for_user("owner-1"),
            data.CreateRequest(
                resource="outfits",
                data={
                    "name": "Test Shirt",
                    "category": "Tops",
                    "masked_url": "https://cdn/masked/shirt.png",
                },
            ),
        )

    assert captured["payload"]["image_url"] == "https://cdn/masked/shirt.png"
    assert captured["payload"]["masked_url"] == "https://cdn/masked/shirt.png"


def test_caller_supplied_identical_urls_accepted_as_raw_write_not_rejected():
    # Case C: the caller (not this endpoint) supplies image_url == masked_url
    # directly. This endpoint's contract was always "both fields present",
    # never "masked_url must be a genuine distinct cutout" - board-readiness
    # is enforced at Style This/Shuffle SELECTION time
    # (services.style_board_image_readiness), not at generic write time. A
    # write like this must still be accepted (it's a legitimate raw wardrobe
    # record) - it will simply never be chosen for a Style Board until it
    # has a real processed image.
    captured = {}

    def fake_create(*, resource, payload, document_id):
        captured["payload"] = payload
        return {"$id": "doc_1", **payload}

    same_url = "https://cdn/raw/shirt.jpg"
    with patch.object(data.qdrant_service, "enabled", return_value=False), patch.object(
        data, "_create_document_with_schema_retries", side_effect=fake_create
    ):
        data.create_document(
            _request_for_user("owner-1"),
            data.CreateRequest(
                resource="outfits",
                data={
                    "name": "Test Shirt",
                    "category": "Tops",
                    "image_url": same_url,
                    "masked_url": same_url,
                },
            ),
        )

    assert captured["payload"]["image_url"] == same_url
    assert captured["payload"]["masked_url"] == same_url

    # Confirm the downstream readiness gate correctly rejects this exact
    # persisted shape for board selection - the write/selection boundary is
    # where this case is actually handled, not at write time.
    from services.style_board_image_readiness import is_board_renderable

    assert is_board_renderable(captured["payload"]) is False
