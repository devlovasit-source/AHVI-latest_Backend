import json

import services.wardrobe_persistence_service as persistence


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def test_update_labels_writes_metadata_separately(monkeypatch):
    calls = []
    metadata_calls = []

    def fake_patch(url, headers, json, timeout):
        calls.append(dict(json["data"]))
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
    monkeypatch.setattr(
        persistence,
        "_persist_style_metadata_nonfatal",
        lambda **kwargs: metadata_calls.append(kwargs) or "updated",
    )

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
    assert result["partial"] is False
    assert result["dropped_keys"] == []
    assert "style_metadata" not in calls[0]
    assert calls[0]["category"] == "Jewelry"
    assert metadata_calls
    assert metadata_calls[0]["item_id"] == "item_1"
    assert metadata_calls[0]["user_id"] == "user_1"
    assert metadata_calls[0]["source"] == "update_labels"


def test_style_metadata_upsert_update_first_create_fallback(monkeypatch):
    calls = []

    class FakeProxy:
        def update_document(self, resource, document_id, data):
            calls.append(("update", resource, document_id, data))
            raise persistence.AppwriteProxyError("Appwrite request failed (404): not found")

        def create_document(self, resource, data, document_id="unique()"):
            calls.append(("create", resource, document_id, data))
            return {"$id": document_id, **data}

    monkeypatch.setattr(persistence, "AppwriteProxy", lambda: FakeProxy())

    result = persistence._upsert_style_metadata(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Black Loafers", "category": "Footwear"},
    )

    assert result == "created"
    assert calls[0][0] == "update"
    assert calls[0][1] == "wardrobe_style_metadata"
    assert calls[1][0] == "create"
    payload = calls[1][3]
    assert payload["item_id"] == "item_1"
    assert payload["userId"] == "user_1"
    assert json.loads(payload["style_metadata"])["subcategory"] == "Loafers"


def test_save_selected_writes_outfit_without_style_metadata_and_upserts_metadata(monkeypatch):
    created_docs = []
    metadata_calls = []

    monkeypatch.setattr(persistence, "_appwrite_ready", lambda: True)
    monkeypatch.setattr(
        persistence,
        "_create_document",
        lambda document_id, data: created_docs.append(dict(data)) or {"$id": document_id, **data},
    )
    monkeypatch.setattr(persistence.embedding_service, "encode_text", lambda text: [0.1, 0.2])
    monkeypatch.setattr(persistence.qdrant_service, "upsert_wardrobe_item", lambda payload: None)
    monkeypatch.setattr(
        persistence,
        "_persist_style_metadata_nonfatal",
        lambda **kwargs: metadata_calls.append(kwargs) or "created",
    )

    result = persistence.persist_selected_items(
        "user_1",
        ["item_1"],
        [
            {
                "item_id": "item_1",
                "name": "Black Loafers",
                "category": "Footwear",
                "sub_category": "Loafers",
                "masked_url": "https://example.test/item.png",
            }
        ],
    )

    assert result["success"] is True
    assert created_docs
    assert "style_metadata" not in created_docs[0]
    assert metadata_calls[0]["item_id"] == "item_1"
    assert metadata_calls[0]["source"] == "save_selected"


def test_build_appwrite_doc_preserves_gym_label():
    from services.wardrobe_persistence_service import _build_appwrite_doc

    doc = _build_appwrite_doc(
        user_id="user_1",
        file_id="item_gym_top",
        item={
            "name": "White Plain Top",
            "category": "Tops",
            "sub_category": "Tops",
            "occasions": ["Gym", "Casual", "Travel", "Sport"],
        },
        raw_url="https://example.test/raw.png",
        masked_url="",
        normalized_url="https://example.test/catalog.png",
    )

    assert doc["occasions"] == ["Gym", "Casual", "Travel", "Sport"]


def test_metadata_failure_does_not_raise(monkeypatch):
    class FakeProxy:
        def update_document(self, resource, document_id, data):
            raise RuntimeError("table missing")

    monkeypatch.setattr(persistence, "AppwriteProxy", lambda: FakeProxy())

    result = persistence._persist_style_metadata_nonfatal(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Leather Belt"},
        source="test",
    )

    assert result == "failed"


def test_delete_wardrobe_item_deletes_main_row_and_metadata(monkeypatch):
    deleted_urls = []
    metadata_deleted = []

    monkeypatch.setattr(
        persistence,
        "_fetch_document",
        lambda document_id: (
            {
                "$id": document_id,
                "userId": "user_1",
                "image_url": "https://example.test/wardrobe_item.png",
            },
            "outfits",
            "db",
        ),
    )

    def fake_delete(url, headers, timeout):
        deleted_urls.append(url)
        return _FakeResponse(204)

    monkeypatch.setattr(persistence.requests, "delete", fake_delete)
    monkeypatch.setattr(
        persistence,
        "_delete_wardrobe_style_metadata",
        lambda **kwargs: metadata_deleted.append(kwargs) or True,
    )

    result = persistence.delete_wardrobe_item(user_id="user_1", item_id="item-1")

    assert result["success"] is True
    assert result["deleted_item_id"] == "item-1"
    assert result["metadata_deleted"] is True
    assert result["item"]["$id"] == "item-1"
    assert deleted_urls == [
        f"{persistence.APPWRITE_ENDPOINT}/databases/db/collections/outfits/documents/item-1"
    ]
    assert metadata_deleted == [{"user_id": "user_1", "item_id": "item-1"}]


def test_delete_wardrobe_item_rejects_other_user(monkeypatch):
    monkeypatch.setattr(
        persistence,
        "_fetch_document",
        lambda document_id: (
            {"$id": document_id, "userId": "other_user"},
            "outfits",
            "db",
        ),
    )

    try:
        persistence.delete_wardrobe_item(user_id="user_1", item_id="item-1")
    except PermissionError as exc:
        assert "belong" in str(exc)
    else:
        raise AssertionError("Expected PermissionError")


def test_delete_style_metadata_tries_hyphenless_and_query_matches(monkeypatch):
    calls = []

    class FakeProxy:
        def delete_document(self, resource, document_id):
            calls.append(("delete", resource, document_id))
            if document_id == "item-1":
                raise persistence.AppwriteProxyError("Appwrite request failed (404): not found")

        def list_documents(self, resource, user_id=None, limit=100):
            calls.append(("list", resource, user_id, limit))
            return [
                {
                    "$id": "metadata_doc",
                    "item_id": "item-1",
                    "userId": "user_1",
                },
                {
                    "$id": "other_metadata_doc",
                    "item_id": "item-1",
                    "userId": "other_user",
                },
            ]

    monkeypatch.setattr(persistence, "AppwriteProxy", lambda: FakeProxy())

    assert persistence._delete_wardrobe_style_metadata(
        user_id="user_1",
        item_id="item-1",
    ) is True
    assert ("delete", "wardrobe_style_metadata", "item-1") in calls
    assert ("delete", "wardrobe_style_metadata", "item1") in calls
    assert ("delete", "wardrobe_style_metadata", "metadata_doc") in calls
    assert ("delete", "wardrobe_style_metadata", "other_metadata_doc") not in calls


def test_backfill_targets_metadata_table_and_dry_run_does_not_write(monkeypatch):
    from scripts import backfill_style_metadata

    writes = []

    class FakeProxy:
        def list_documents(self, resource, limit=100, offset=0, return_meta=False, **kwargs):
            assert resource == "outfits"
            if offset:
                return {"documents": [], "meta": {"has_more": False}}
            return {
                "documents": [
                    {
                        "$id": "item_1",
                        "userId": "user_1",
                        "name": "Leather Belt",
                        "category": "Accessories",
                    }
                ],
                "meta": {"has_more": False},
            }

    monkeypatch.setattr(backfill_style_metadata, "AppwriteProxy", lambda: FakeProxy())
    monkeypatch.setattr(
        backfill_style_metadata,
        "upsert_wardrobe_style_metadata_sync",
        lambda user_id, doc_id, style_meta: writes.append(
            {"user_id": user_id, "doc_id": doc_id, "style_meta": style_meta}
        )
        or {"status": "updated"},
    )

    dry = backfill_style_metadata.run(dry_run=True, all_users=True)
    assert {
        key: dry[key]
        for key in ("updated", "created", "skipped", "failed", "scanned", "unchanged")
    } == {"updated": 1, "created": 0, "skipped": 0, "failed": 0, "scanned": 1, "unchanged": 0}
    assert dry["total_items"] == 1
    assert dry["metadata_v2"] == 1
    assert writes == []

    real = backfill_style_metadata.run(dry_run=False, all_users=True)
    assert {
        key: real[key]
        for key in ("updated", "created", "skipped", "failed", "scanned", "unchanged")
    } == {"updated": 1, "created": 0, "skipped": 0, "failed": 0, "scanned": 1, "unchanged": 0}
    assert real["total_items"] == 1
    assert real["metadata_v2"] == 1
    assert writes[0]["doc_id"] == "item_1"
    assert writes[0]["user_id"] == "user_1"
