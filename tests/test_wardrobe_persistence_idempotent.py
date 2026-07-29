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
