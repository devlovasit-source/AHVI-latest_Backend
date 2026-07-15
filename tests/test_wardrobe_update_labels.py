import json

import services.wardrobe_persistence_service as persistence


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def _style_meta_from_payload(monkeypatch, *, existing=None, validator=None, enabled=True):
    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: enabled)
    if isinstance(validator, BaseException):
        def _raise_validator(**kwargs):
            raise validator

        monkeypatch.setattr(persistence, "_agent_validate_metadata_sync", _raise_validator)
    else:
        monkeypatch.setattr(
            persistence,
            "_agent_validate_metadata_sync",
            lambda **kwargs: validator,
        )
    item = {
        "name": "Black Loafers",
        "category": "Footwear",
        "sub_category": "Loafers",
    }
    if existing is not None:
        item["style_metadata"] = dict(existing)
    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload=item,
    )
    return json.loads(payload["style_metadata"])


def _validator_result(**fields):
    return {
        "_validator_status": "validated",
        "_validator_authoritative_fields": list(fields),
        **fields,
    }


def test_validator_empty_null_and_unknown_preserve_existing_subcategory(monkeypatch):
    existing = {
        "category": "Footwear",
        "subcategory": "Loafers",
        "formality": "smart_casual",
    }
    for validator in ({}, None, {"subcategory": "unknown"}):
        meta = _style_meta_from_payload(
            monkeypatch, existing=existing, validator=validator
        )
        assert meta["subcategory"] == "Loafers"
        assert meta["formality"] == "smart_casual"


def test_partial_validator_updates_only_meaningful_field(monkeypatch):
    existing = {
        "category": "Footwear",
        "subcategory": "Loafers",
        "formality": "smart_casual",
        "style_role": "businesswear",
    }
    meta = _style_meta_from_payload(
        monkeypatch,
        existing=existing,
        validator=_validator_result(formality="formal", subcategory="unknown"),
    )
    assert meta["formality"] == "formal"
    assert meta["subcategory"] == "Loafers"
    assert meta["style_role"] == "businesswear"


def test_valid_validator_correction_updates_mutable_field(monkeypatch):
    meta = _style_meta_from_payload(
        monkeypatch,
        existing={"category": "Footwear", "subcategory": "Dress Shoes"},
        validator=_validator_result(subcategory="Loafers", confidence=0.94),
    )
    assert meta["subcategory"] == "Loafers"
    assert meta["agent_validated"] is True
    assert meta["agent_confidence"] == 0.94


def test_malformed_validator_preserves_complete_existing_metadata(monkeypatch):
    existing = {
        "category": "Footwear",
        "subcategory": "Loafers",
        "style_role": "businesswear",
        "metadata_updated_at": "2026-07-01T00:00:00Z",
    }
    meta = _style_meta_from_payload(
        monkeypatch, existing=existing, validator="not-json"
    )
    for key, value in existing.items():
        assert meta[key] == value
    assert meta["agent_validation_status"] == "degraded"
    assert meta["agent_validation_degraded_reason"] == "malformed_response"


def test_validator_exception_or_timeout_preserves_existing_metadata(monkeypatch):
    existing = {
        "category": "Footwear",
        "subcategory": "Loafers",
        "formality": "smart_casual",
    }
    for failure in (RuntimeError("unavailable"), TimeoutError("timeout")):
        meta = _style_meta_from_payload(
            monkeypatch, existing=existing, validator=failure
        )
        assert meta["subcategory"] == "Loafers"
        assert meta["formality"] == "smart_casual"
        assert meta["agent_validation_status"] == "degraded"


def test_validator_disabled_preserves_deterministic_behavior(monkeypatch):
    meta = _style_meta_from_payload(
        monkeypatch,
        validator=AssertionError("validator must not run"),
        enabled=False,
    )
    assert meta["category"] == "Footwear"
    assert meta["subcategory"] == "Loafers"
    assert "agent_validation_status" not in meta


def test_new_record_without_trustworthy_taxonomy_stays_neutral(monkeypatch):
    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: True)
    monkeypatch.setattr(persistence, "_agent_validate_metadata_sync", lambda **kwargs: {})
    payload = persistence._style_metadata_payload(
        item_id="item_new",
        user_id="user_1",
        item_payload={"name": "Unclassified item"},
    )
    meta = json.loads(payload["style_metadata"])
    assert meta["category"] == "unknown"
    assert meta["subcategory"] == "unknown"


def test_validator_fallback_processing_is_idempotent(monkeypatch):
    existing = {
        "category": "Footwear",
        "subcategory": "Loafers",
        "metadata_updated_at": "2026-07-01T00:00:00Z",
    }
    first = _style_meta_from_payload(monkeypatch, existing=existing, validator={})
    second = _style_meta_from_payload(monkeypatch, existing=first, validator={})
    assert second == first
    assert second["metadata_updated_at"] == existing["metadata_updated_at"]


def test_validator_cannot_change_identity_source_or_media_fields(monkeypatch):
    existing = {
        "category": "Footwear",
        "subcategory": "Loafers",
        "source": "wardrobe",
        "owner_id": "user_1",
        "image_url": "private-original-media",
        "storage_key": "private/original.png",
    }
    proposed = {
        "source": "catalog",
        "owner_id": "other_user",
        "image_url": "private-replacement-media",
        "storage_key": "other/replacement.png",
        "subcategory": "Oxfords",
    }
    meta = _style_meta_from_payload(
        monkeypatch,
        existing=existing,
        validator={
            "_validator_status": "validated",
            "_validator_authoritative_fields": list(proposed),
            **proposed,
        },
    )
    assert meta["subcategory"] == "Oxfords"
    for key in ("source", "owner_id", "image_url", "storage_key"):
        assert meta[key] == existing[key]


def test_upsert_reads_and_preserves_persisted_metadata_on_validator_fallback(monkeypatch):
    calls = []
    existing_meta = {
        "category": "Footwear",
        "subcategory": "Loafers",
        "style_role": "businesswear",
    }

    class FakeProxy:
        def get_document(self, resource, document_id):
            calls.append(("get", resource, document_id))
            return {
                "$id": document_id,
                "userId": "user_1",
                "style_metadata": json.dumps(existing_meta),
            }

        def update_document(self, resource, document_id, data):
            calls.append(("update", resource, document_id, data))
            return {"$id": document_id}

    monkeypatch.setattr(persistence, "AppwriteProxy", lambda: FakeProxy())
    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: True)
    monkeypatch.setattr(
        persistence,
        "_agent_validate_metadata_sync",
        lambda **kwargs: {
            "_validator_status": "degraded",
            "_validator_authoritative_fields": [],
            "_validator_degraded_reason": "empty_response",
        },
    )

    result = persistence._upsert_style_metadata(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Black Loafers", "category": "Footwear"},
    )

    assert result == "updated"
    written = json.loads(calls[-1][3]["style_metadata"])
    assert written["subcategory"] == "Loafers"
    assert written["style_role"] == "businesswear"


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
