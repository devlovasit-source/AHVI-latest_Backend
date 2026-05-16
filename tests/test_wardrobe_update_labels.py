import json

import services.wardrobe_persistence_service as persistence


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def test_update_labels_drops_missing_style_metadata_and_saves_labels(monkeypatch):
    calls = []

    def fake_patch(url, headers, json, timeout):
        calls.append(dict(json["data"]))
        if len(calls) == 1:
            return _FakeResponse(
                400,
                text='{"message":"Invalid document structure: Unknown attribute: \\"style_metadata\\"","code":400}',
            )
        return _FakeResponse(
            200,
            {
                "$id": "item_1",
                "userId": "user_1",
                "name": calls[-1].get("name"),
                "category": calls[-1].get("category"),
                "sub_category": calls[-1].get("sub_category"),
            },
        )

    monkeypatch.setattr(persistence.requests, "patch", fake_patch)

    result = persistence.update_item_labels(
        user_id="user_1",
        item_id="item_1",
        name="Gold Ring",
        category="Jewelry",
        subcategory="Ring",
        override_collection_id="outfits",
        override_database_id="db",
    )

    assert result["success"] is True
    assert result["partial"] is True
    assert result["dropped_keys"] == ["style_metadata"]
    assert calls[0]["style_metadata"]
    assert "style_metadata" not in calls[1]
    assert calls[1]["category"] == "Jewelry"
