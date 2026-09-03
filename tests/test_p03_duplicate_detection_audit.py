"""Focused P0.3 duplicate detection regression tests.

These tests cover the three lookup signals and the production failure mode where
Qdrant is reachable but the optional embedding models are unavailable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import routers.wardrobe_capture as wardrobe_capture
import services.wardrobe_persistence_service as wardrobe_persistence
from services.upload_batch_orchestrator import UploadBatchOrchestrator, deterministic_appwrite_id
from services.qdrant_service import QdrantService


def _item(**overrides):
    item = {
        "category": "Tops",
        "sub_category": "Shirt",
        "color_code": "#0000FF",
        "pattern": "plain",
        "occasions": ["casual"],
    }
    item.update(overrides)
    return item


def _result(*, checked=True, duplicate=False, reason=None, matched_item_id=None, **extra):
    return {
        "checked": checked,
        "is_duplicate": duplicate,
        "reason": reason,
        "confidence": 1.0 if duplicate else 0.0,
        "matched_item_id": matched_item_id,
        **extra,
    }


def test_exact_pixel_hash_blocks_same_user(monkeypatch):
    monkeypatch.setattr(
        wardrobe_capture.qdrant_service,
        "find_pixel_duplicate",
        lambda user_id, pixel_hash, max_distance: _result(
            reason="pixel_hash", duplicate=True, matched_item_id="existing-1", distance=0
        ),
    )

    result = wardrobe_capture._find_upload_duplicate(
        user_id="user-1", item=_item(), pixel_hash="a" * 16, image_embedding=[]
    )

    assert result["is_duplicate"] is True
    assert result["reason"] == "pixel_hash"
    assert result["matched_item_id"] == "existing-1"


def test_near_pixel_hash_within_threshold_blocks(monkeypatch):
    captured = {}

    def find_pixel(user_id, pixel_hash, max_distance):
        captured.update(user_id=user_id, pixel_hash=pixel_hash, max_distance=max_distance)
        return _result(reason="pixel_hash", duplicate=True, matched_item_id="existing-2", distance=6)

    monkeypatch.setattr(wardrobe_capture.qdrant_service, "find_pixel_duplicate", find_pixel)

    result = wardrobe_capture._find_upload_duplicate(
        user_id="user-2", item=_item(), pixel_hash="b" * 16, image_embedding=[]
    )

    assert result["is_duplicate"] is True
    assert captured == {"user_id": "user-2", "pixel_hash": "b" * 16, "max_distance": 6}


def test_distinct_pixel_hash_is_allowed(monkeypatch):
    monkeypatch.setattr(
        wardrobe_capture.qdrant_service,
        "find_pixel_duplicate",
        lambda *_args, **_kwargs: _result(checked=True),
    )

    result = wardrobe_capture._find_upload_duplicate(
        user_id="user-3", item=_item(), pixel_hash="c" * 16, image_embedding=[]
    )

    assert result["checked"] is True
    assert result["is_duplicate"] is False


def test_pixel_lookup_is_scoped_to_authenticated_user():
    captured = {}

    class FakeClient:
        def scroll(self, **kwargs):
            captured.update(kwargs)
            return ([], None)

    service = QdrantService.__new__(QdrantService)
    service.client = FakeClient()
    service._initialized = True
    service.collection = "wardrobe"
    service.image_collection = "wardrobe_images"
    service.vector_size = 512

    result = service.find_pixel_duplicate("user-4", "d" * 16)

    assert result["checked"] is True
    condition = captured["scroll_filter"].must[0]
    assert condition.key == "userId"
    assert condition.match.value == "user-4"


def test_metadata_duplicate_requires_same_family(monkeypatch):
    monkeypatch.setattr(wardrobe_capture, "encode_metadata", lambda _data: [0.1])
    monkeypatch.setattr(
        wardrobe_capture.qdrant_service,
        "find_duplicate",
        lambda *_args, **_kwargs: _result(
            reason="metadata",
            duplicate=True,
            matched_item_id="existing-5",
            score=0.998,
            payload={"category": "Tops", "type": "Shirt", "color": "#0000FF"},
        ),
    )

    result = wardrobe_capture._find_upload_duplicate(
        user_id="user-5", item=_item(), pixel_hash="", image_embedding=[]
    )

    assert result["is_duplicate"] is True
    assert result["reason"] == "metadata"


def test_metadata_match_from_different_family_is_allowed(monkeypatch):
    monkeypatch.setattr(wardrobe_capture, "encode_metadata", lambda _data: [0.1])
    monkeypatch.setattr(
        wardrobe_capture.qdrant_service,
        "find_duplicate",
        lambda *_args, **_kwargs: _result(
            reason="metadata",
            duplicate=True,
            matched_item_id="existing-6",
            score=0.998,
            payload={"category": "Bottoms", "type": "Trousers", "color": "#0000FF"},
        ),
    )

    result = wardrobe_capture._find_upload_duplicate(
        user_id="user-6", item=_item(), pixel_hash="", image_embedding=[]
    )

    assert result["is_duplicate"] is False


def test_image_vector_duplicate_is_reported(monkeypatch):
    monkeypatch.setattr(
        wardrobe_capture.qdrant_service,
        "find_image_duplicate",
        lambda *_args, **_kwargs: _result(
            reason="image_vector", duplicate=True, matched_item_id="existing-7", score=0.99
        ),
    )

    result = wardrobe_capture._find_upload_duplicate(
        user_id="user-7", item=_item(), pixel_hash="", image_embedding=[0.1, 0.2]
    )

    assert result["is_duplicate"] is True
    assert result["reason"] == "image_vector"
    assert result["matched_item_id"] == "existing-7"


def test_image_lookup_exception_is_not_reported_as_checked_negative():
    class FailingClient:
        def search(self, **_kwargs):
            raise RuntimeError("qdrant unavailable")

    service = QdrantService.__new__(QdrantService)
    service.client = FailingClient()
    service._initialized = True
    service.collection = "wardrobe"
    service.image_collection = "wardrobe_images"
    service.vector_size = 512

    result = service.find_image_duplicate([0.1], "user-8")

    assert result["checked"] is False
    assert result["is_duplicate"] is False


def test_metadata_lookup_exception_is_not_reported_as_checked_negative():
    class FailingClient:
        def search(self, **_kwargs):
            raise RuntimeError("qdrant unavailable")

    service = QdrantService.__new__(QdrantService)
    service.client = FailingClient()
    service._initialized = True
    service.collection = "wardrobe"
    service.image_collection = "wardrobe_images"
    service.vector_size = 512

    result = service.find_duplicate([0.1], "user-8")

    assert result["checked"] is False
    assert result["is_duplicate"] is False


def test_pixel_lookup_exception_is_not_reported_as_checked_negative():
    class FailingClient:
        def scroll(self, **_kwargs):
            raise RuntimeError("qdrant unavailable")

    service = QdrantService.__new__(QdrantService)
    service.client = FailingClient()
    service._initialized = True
    service.collection = "wardrobe"
    service.image_collection = "wardrobe_images"
    service.vector_size = 512

    result = service.find_pixel_duplicate("user-8", "e" * 16)

    assert result["checked"] is False
    assert result["is_duplicate"] is False


def test_semantic_lookup_excludes_pixel_only_points():
    captured = []

    class FakeClient:
        def search(self, **kwargs):
            captured.append(kwargs)
            return []

    service = QdrantService.__new__(QdrantService)
    service.client = FakeClient()
    service._initialized = True
    service.collection = "wardrobe"
    service.image_collection = "wardrobe_images"
    service.vector_size = 512

    service.find_duplicate([0.1], "user-9")
    service.search_similar([0.1], "user-9")

    assert len(captured) == 2
    for call in captured:
        query_filter = call["query_filter"]
        assert any(
            condition.key == "vector_kind"
            and condition.match.value == "pixel_only"
            for condition in query_filter.must_not
        )


def test_pixel_only_wardrobe_item_is_persisted_without_embedding():
    calls = []

    class FakeClient:
        def upsert(self, **kwargs):
            calls.append(kwargs)

    service = QdrantService.__new__(QdrantService)
    service.client = FakeClient()
    service._initialized = True
    service.collection = "wardrobe"
    service.image_collection = "wardrobe_images"
    service.vector_size = 512

    stored = service.upsert_wardrobe_item(
        {
            "id": "existing-9",
            "userId": "user-9",
            "category": "Tops",
            "type": "shirt",
            "color": "#0000FF",
            "pixel_hash": "f" * 16,
            "embedding": [],
        }
    )

    assert stored is True
    assert len(calls) == 1
    point = calls[0]["points"][0]
    assert calls[0]["wait"] is True
    assert len(point.vector) == 512
    assert any(value != 0.0 for value in point.vector)
    assert point.payload["pixel_hash"] == "f" * 16
    assert point.payload["vector_kind"] == "pixel_only"

    service.upsert_wardrobe_item(
        {
            "id": "existing-9",
            "userId": "user-9",
            "category": "Tops",
            "type": "shirt",
            "color": "#0000FF",
            "pixel_hash": "f" * 16,
            "embedding": [0.2] * 512,
        }
    )
    assert len(calls) == 2
    assert calls[0]["points"][0].id == calls[1]["points"][0].id == "existing-9"
    assert "vector_kind" not in calls[1]["points"][0].payload


def test_delete_uses_same_pixel_only_point_id_for_both_collections():
    calls = []

    class FakeClient:
        def delete(self, **kwargs):
            calls.append(kwargs)

    service = QdrantService.__new__(QdrantService)
    service.client = FakeClient()
    service._initialized = True
    service.collection = "wardrobe"
    service.image_collection = "wardrobe_images"
    service.vector_size = 512

    assert service.delete_item("existing-9") is True
    assert [call["collection_name"] for call in calls] == ["wardrobe", "wardrobe_images"]
    assert all(call["points_selector"] == ["existing-9"] for call in calls)
    assert all(call["wait"] is True for call in calls)


def test_delete_wardrobe_item_calls_qdrant_with_item_id(monkeypatch):
    monkeypatch.setattr(
        wardrobe_persistence,
        "_fetch_document",
        lambda _item_id: ({"$id": "existing-10", "userId": "user-10"}, "outfits", "db"),
    )
    monkeypatch.setattr(wardrobe_persistence, "_delete_document_at_location", lambda *args, **kwargs: None)
    monkeypatch.setattr(wardrobe_persistence, "_delete_wardrobe_style_metadata", lambda **kwargs: False)
    deleted = []
    monkeypatch.setattr(
        wardrobe_persistence.qdrant_service,
        "delete_item",
        lambda item_id: deleted.append(item_id) or True,
    )

    result = wardrobe_persistence.delete_wardrobe_item(user_id="user-10", item_id="existing-10")

    assert deleted == ["existing-10"]
    assert result["qdrant_deleted"] is True


@pytest.mark.parametrize("qdrant_result", [True, False])
def test_qdrant_indexed_reflects_upsert_result(monkeypatch, qdrant_result):
    monkeypatch.setattr(wardrobe_persistence, "_appwrite_ready", lambda: True)
    monkeypatch.setattr(
        wardrobe_persistence,
        "_create_document",
        lambda document_id, data: {"$id": document_id, **data},
    )
    monkeypatch.setattr(wardrobe_persistence, "_persist_style_metadata_nonfatal", lambda **kwargs: "skipped")
    monkeypatch.setattr(wardrobe_persistence.embedding_service, "encode_text", lambda _text: [])
    monkeypatch.setattr(
        wardrobe_persistence.qdrant_service,
        "upsert_wardrobe_item",
        lambda _item: qdrant_result,
    )

    result = wardrobe_persistence.persist_selected_items(
        "user-11",
        ["item-11"],
        [
            {
                "item_id": "item-11",
                "name": "Blue Shirt",
                "category": "Tops",
                "sub_category": "Shirt",
                "color_code": "#0000FF",
                "pattern": "plain",
                "occasions": ["casual"],
                "raw_url": "https://raw.test/item-11.png",
                "masked_url": "https://masked.test/item-11.png",
                "pixel_hash": "1" * 16,
            }
        ],
    )

    assert result["items"][0]["qdrant_indexed"] is qdrant_result


def _valid_orchestrator_item(item_id):
    return {
        "item_id": item_id,
        "name": "Blue Shirt",
        "category": "Tops",
        "validation_status": "ok",
        "image_url": f"https://cdn.test/{item_id}.jpg",
        "masked_url": f"https://cdn.test/{item_id}.png",
        "normalized_url": f"https://cdn.test/{item_id}-normalized.png",
    }


def _duplicate_orchestrator_item(item_id):
    item = _valid_orchestrator_item(item_id)
    item["duplicate"] = {
        "checked": True,
        "is_duplicate": True,
        "reason": "pixel_hash",
        "confidence": 1.0,
        "matched_item_id": "existing-12",
    }
    return item


@pytest.fixture
def orchestrator(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    instance = UploadBatchOrchestrator()
    instance._memory_batches = {}
    instance._memory_items = {}
    return instance


def _wire_orchestrator(monkeypatch, item, calls):
    async def analyze(_request, _capture_request):
        return {"success": True, "count": 1, "items": [dict(item)]}

    def persist(*, user_id, selected_item_ids, detected_items):
        calls.append((user_id, list(selected_item_ids), [dict(i) for i in detected_items]))
        return {
            "success": True,
            "saved_count": len(detected_items),
            "items": [{"$id": "wardrobe-12", **detected_items[0]}],
        }

    monkeypatch.setattr(wardrobe_capture, "analyze_capture", analyze)
    monkeypatch.setattr(wardrobe_capture, "_try_upload_inline_images", lambda item, **kwargs: item)
    monkeypatch.setattr(wardrobe_capture, "_maybe_generate_catalog_image", lambda item: None)
    monkeypatch.setattr(wardrobe_persistence, "persist_selected_items", persist)


@pytest.mark.anyio
async def test_orchestrator_duplicate_needs_review_without_persistence(orchestrator, monkeypatch):
    orchestrator.create_or_resume_batch(
        user_id="user-12", client_batch_request_id="batch-12", total_items=1
    )
    calls = []
    _wire_orchestrator(monkeypatch, _duplicate_orchestrator_item("item-12"), calls)

    result = await orchestrator.process_single_batch_item(
        http_request=None,
        user_id="user-12",
        batch_id="batch-12",
        client_upload_item_id="upload-12",
        image_base64="x" * 30,
    )

    assert result["status"] == "NEEDS_REVIEW"
    assert result["reason"] == "duplicate_wardrobe_item"
    stored = orchestrator._get_doc(
        orchestrator.items_collection,
        deterministic_appwrite_id("user-12", "upload-12"),
    )
    assert stored["error_code"] == "DUPLICATE_WARDROBE_ITEM"
    assert calls == []


@pytest.mark.anyio
async def test_orchestrator_add_anyway_persists_exactly_once(orchestrator, monkeypatch):
    orchestrator.create_or_resume_batch(
        user_id="user-13", client_batch_request_id="batch-13", total_items=1
    )
    calls = []
    _wire_orchestrator(monkeypatch, _duplicate_orchestrator_item("item-13"), calls)

    blocked = await orchestrator.process_single_batch_item(
        http_request=None,
        user_id="user-13",
        batch_id="batch-13",
        client_upload_item_id="upload-13",
        image_base64="x" * 30,
    )
    added = await orchestrator.process_single_batch_item(
        http_request=None,
        user_id="user-13",
        batch_id="batch-13",
        client_upload_item_id="upload-13",
        image_base64="x" * 30,
        override_duplicate=True,
    )
    retried = await orchestrator.process_single_batch_item(
        http_request=None,
        user_id="user-13",
        batch_id="batch-13",
        client_upload_item_id="upload-13",
        image_base64="x" * 30,
        override_duplicate=True,
    )

    assert blocked["status"] == "NEEDS_REVIEW"
    assert added["status"] == "ADDED_TO_WARDROBE"
    assert retried["idempotent"] is True
    assert len(calls) == 1


def _delete_request(user_id="delete-user"):
    return SimpleNamespace(state=SimpleNamespace(user={"user_id": user_id}))


def _wire_bulk_delete(monkeypatch, appwrite_results, qdrant_calls, *, qdrant_result=True):
    docs = {
        item_id: {"$id": item_id, "userId": "delete-user"}
        for item_id in appwrite_results
    }

    monkeypatch.setattr(wardrobe_capture, "_ahvi_fetch_outfit_doc", lambda item_id: docs[item_id])
    monkeypatch.setattr(wardrobe_capture, "_ahvi_delete_r2_images_for_item", lambda _item: {"status": "skipped"})

    def delete_appwrite(item_id):
        return {"ok": appwrite_results[item_id]}

    monkeypatch.setattr(wardrobe_capture, "_ahvi_delete_outfit_doc", delete_appwrite)

    def delete_qdrant(item_id):
        qdrant_calls.append(item_id)
        return qdrant_result

    monkeypatch.setattr(wardrobe_capture.qdrant_service, "delete_item", delete_qdrant)


def test_bulk_delete_three_successful_items_removes_all_qdrant_points(monkeypatch):
    qdrant_calls = []
    _wire_bulk_delete(
        monkeypatch,
        {"bulk-1": True, "bulk-2": True, "bulk-3": True},
        qdrant_calls,
    )

    result = wardrobe_capture.delete_selected(
        _delete_request(),
        wardrobe_capture.DeleteSelectedRequest(
            user_id="delete-user",
            item_ids=["bulk-1", "bulk-2", "bulk-3"],
            delete_r2=False,
        ),
    )

    assert result["success"] is True
    assert result["deleted_count"] == 3
    assert qdrant_calls == ["bulk-1", "bulk-2", "bulk-3"]


def test_bulk_delete_appwrite_failure_keeps_failed_qdrant_point(monkeypatch):
    qdrant_calls = []
    _wire_bulk_delete(
        monkeypatch,
        {"partial-1": True, "partial-2": False, "partial-3": True},
        qdrant_calls,
    )

    result = wardrobe_capture.delete_selected(
        _delete_request(),
        wardrobe_capture.DeleteSelectedRequest(
            user_id="delete-user",
            item_ids=["partial-1", "partial-2", "partial-3"],
            delete_r2=False,
        ),
    )

    assert result["success"] is False
    assert result["deleted_count"] == 2
    assert result["error_count"] == 1
    assert qdrant_calls == ["partial-1", "partial-3"]


def test_bulk_delete_is_idempotent_when_qdrant_point_is_absent(monkeypatch):
    qdrant_calls = []
    _wire_bulk_delete(monkeypatch, {"absent-1": True}, qdrant_calls, qdrant_result=True)

    result = wardrobe_capture.delete_selected(
        _delete_request(),
        wardrobe_capture.DeleteSelectedRequest(
            user_id="delete-user",
            item_ids=["absent-1"],
            delete_r2=False,
        ),
    )

    assert result["success"] is True
    assert result["deleted_count"] == 1
    assert qdrant_calls == ["absent-1"]


def test_delete_selected_removes_point_and_allows_same_image_reupload(monkeypatch):
    points = {"lifecycle-1": {"userId": "delete-user", "pixel_hash": "a" * 16}}

    def find_pixel(user_id, pixel_hash, max_distance):
        for point_id, point in points.items():
            if point["userId"] == user_id and point["pixel_hash"] == pixel_hash:
                return _result(
                    reason="pixel_hash",
                    duplicate=True,
                    matched_item_id=point_id,
                    distance=0,
                )
        return _result(checked=True)

    monkeypatch.setattr(wardrobe_capture.qdrant_service, "find_pixel_duplicate", find_pixel)
    monkeypatch.setattr(wardrobe_capture.qdrant_service, "delete_item", lambda item_id: points.pop(item_id, None) is not None)
    monkeypatch.setattr(wardrobe_capture, "_ahvi_fetch_outfit_doc", lambda _item_id: {"$id": "lifecycle-1", "userId": "delete-user"})
    monkeypatch.setattr(wardrobe_capture, "_ahvi_delete_r2_images_for_item", lambda _item: {"status": "skipped"})
    monkeypatch.setattr(wardrobe_capture, "_ahvi_delete_outfit_doc", lambda _item_id: {"ok": True})

    before = wardrobe_capture._find_upload_duplicate(
        user_id="delete-user",
        item=_item(),
        pixel_hash="a" * 16,
        image_embedding=[],
    )
    delete_result = wardrobe_capture.delete_selected(
        _delete_request(),
        wardrobe_capture.DeleteSelectedRequest(
            user_id="delete-user",
            item_ids=["lifecycle-1"],
            delete_r2=False,
        ),
    )
    after = wardrobe_capture._find_upload_duplicate(
        user_id="delete-user",
        item=_item(),
        pixel_hash="a" * 16,
        image_embedding=[],
    )

    assert before["is_duplicate"] is True
    assert delete_result["success"] is True
    assert after["is_duplicate"] is False
