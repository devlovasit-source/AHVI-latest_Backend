from services import wardrobe_persistence_service as persistence


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_create_document_updates_existing_item_on_retry(monkeypatch):
    calls = {"post": 0, "patch": 0}
    monkeypatch.setattr(persistence, "_appwrite_ready", lambda: True)

    def _post(*_args, **_kwargs):
        calls["post"] += 1
        return _Response(409, {"type": "document_already_exists"})

    def _patch(url, *, json, headers, timeout):
        calls["patch"] += 1
        assert url.endswith("/documents/item-1")
        assert json["data"]["name"] == "Blue Shirt"
        return _Response(200, {"$id": "item-1", "name": "Blue Shirt"})

    monkeypatch.setattr(persistence.requests, "post", _post)
    monkeypatch.setattr(persistence.requests, "patch", _patch)

    result = persistence._create_document("item-1", {"name": "Blue Shirt"})

    assert result["$id"] == "item-1"
    assert calls == {"post": 1, "patch": 1}


def test_create_document_update_retry_strips_pixel_hash(monkeypatch):
    calls = {"post": 0, "patch": 0}
    monkeypatch.setattr(persistence, "_appwrite_ready", lambda: True)

    def _post(*_args, **_kwargs):
        calls["post"] += 1
        return _Response(409, {"type": "document_already_exists"})

    def _patch(url, *, json, headers, timeout):
        calls["patch"] += 1
        assert url.endswith("/documents/item-1")
        assert "pixel_hash" not in json["data"]
        assert "display_image_url" not in json["data"]
        return _Response(200, {"$id": "item-1", "name": "Blue Shirt"})

    monkeypatch.setattr(persistence.requests, "post", _post)
    monkeypatch.setattr(persistence.requests, "patch", _patch)

    result = persistence._create_document(
        "item-1",
        {
            "name": "Blue Shirt",
            "pixel_hash": "abc",
            "display_image_url": "https://display.test/item.png",
        },
    )

    assert result["$id"] == "item-1"
    assert calls == {"post": 1, "patch": 1}


def test_final_persistence_normalizes_jeans_shorts_label():
    item = {
        "name": "Distressed Jeans Shorts",
        "category": "Bottoms",
        "sub_category": "Shorts",
    }

    normalized = persistence.normalize_final_wardrobe_item_name_and_taxonomy(item)

    assert normalized["name"] == "Distressed Jeans"
    assert normalized["category"] == "Bottoms"
    assert normalized["sub_category"] == "Jeans"


def _image_doc(*, raw_url="", masked_url="", normalized_url=""):
    return persistence._build_appwrite_doc(
        user_id="user-1",
        file_id="item-1",
        item={"name": "Blue Shirt", "category": "Tops"},
        raw_url=raw_url,
        masked_url=masked_url,
        normalized_url=normalized_url,
    )


def test_normalized_url_is_not_published_as_masked():
    doc = _image_doc(normalized_url="https://catalog.test/item-1.png")

    assert doc["normalized_url"] == "https://catalog.test/item-1.png"
    assert doc["masked_url"] == ""


def test_original_image_url_is_not_published_as_masked():
    doc = _image_doc(raw_url="https://raw.test/item-1.png")

    assert doc["image_url"] == "https://raw.test/item-1.png"
    assert doc["masked_url"] == ""
    assert doc["normalized_url"] == ""


def test_existing_masked_url_is_preserved_without_aliasing_other_fields():
    doc = _image_doc(
        raw_url="https://raw.test/item-1.png",
        masked_url="https://masked.test/item-1.png",
        normalized_url="https://catalog.test/item-1.png",
    )

    assert doc["image_url"] == "https://raw.test/item-1.png"
    assert doc["masked_url"] == "https://masked.test/item-1.png"
    assert doc["normalized_url"] == "https://catalog.test/item-1.png"


# ---------------------------------------------------------------------------
# normalized_url raw-alias regression (Phase 2A) - a wardrobe item whose
# backend record never got a genuinely distinct processed/catalog asset must
# never persist normalized_url == a known raw/original URL. Live-reproduced
# via the "Refined Ease" Style Board device bug: normalized_url and image_url
# resolved to the identical raw upload, and the (now-fixed) Flutter resolver
# had trusted it as a safe catalog_fallback. See
# lib/util/wardrobe_image_resolver.dart commit e8a53d6 (frontend fix) and
# services/style_board_image_readiness.py (this repo's read-boundary half).
#
# These tests characterize CURRENT (pre-fix) behavior and are expected to
# FAIL against _build_appwrite_doc until Phase 3 adds the raw-alias guard.


# A1 - raw alias must not persist.
def test_normalized_url_aliasing_raw_image_is_rejected():
    doc = _image_doc(
        raw_url="https://raw.test/item-1.png",
        normalized_url="https://raw.test/item-1.png",
    )

    assert doc["image_url"] == "https://raw.test/item-1.png"
    assert doc["normalized_url"] == ""


# A2 - distinct normalized URL must persist (legitimate processed asset).
def test_distinct_normalized_url_is_preserved():
    doc = _image_doc(
        raw_url="https://raw.test/item-1.png",
        normalized_url="https://catalog.test/item-1.png",
    )

    assert doc["image_url"] == "https://raw.test/item-1.png"
    assert doc["normalized_url"] == "https://catalog.test/item-1.png"


# A3 - raw-only input stays raw-only; no normalized_url is manufactured.
def test_raw_only_item_does_not_manufacture_normalized_url():
    doc = _image_doc(raw_url="https://raw.test/item-1.png")

    assert doc["image_url"] == "https://raw.test/item-1.png"
    assert doc["normalized_url"] == ""


# A4 - the invariant is "never aliases ANY known raw/original representation",
# not merely "!= image_url". Here raw_url (not item['image_url']) is the
# aliased field, and item['image_url'] is a different placeholder value -
# normalized_url must still be rejected because it aliases raw_url.
def test_normalized_url_aliasing_raw_url_param_is_rejected_even_when_image_url_differs():
    doc = persistence._build_appwrite_doc(
        user_id="user-1",
        file_id="item-1",
        item={
            "name": "Blue Shirt",
            "category": "Tops",
            "image_url": "https://placeholder.test/not-the-raw-url.png",
        },
        raw_url="https://raw.test/item-1.png",
        masked_url="",
        normalized_url="https://raw.test/item-1.png",
    )

    assert doc["normalized_url"] == ""


# A4b - canonical-equivalent alias (differs only by query token/fragment,
# same _board_url_identity()) must also be rejected, not just exact-string
# matches. Mirrors the identity semantics already covered in
# tests/test_style_board_image_readiness.py's URL-identity alias parity
# section.
def test_normalized_url_aliasing_raw_via_canonical_identity_is_rejected():
    doc = _image_doc(
        raw_url="https://raw.test/item-1.png?token=abc",
        normalized_url="https://raw.test/item-1.png?token=xyz",
    )

    assert doc["normalized_url"] == ""
