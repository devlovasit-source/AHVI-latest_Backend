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
