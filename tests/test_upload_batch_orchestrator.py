"""Sequential wardrobe upload batch orchestrator (AHVI P0 upload MVP).

Runs entirely against the ENVIRONMENT=test in-memory fallback (network is
blocked for the whole suite - see tests/conftest.py); analyze_capture() and
persist_selected_items() are mocked per test so this suite verifies
orchestration (idempotency, sequencing, state machine, counters), not the
real vision/persistence pipelines those already have their own tests for.
"""

from __future__ import annotations

import pytest

import routers.wardrobe_capture as wardrobe_capture
import services.wardrobe_persistence_service as wardrobe_persistence_service
from services.upload_batch_orchestrator import (
    UploadBatchInfraError,
    UploadBatchOrchestrator,
    deterministic_appwrite_id,
    validate_status_transition,
)


@pytest.fixture(autouse=True)
def _test_environment(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")


@pytest.fixture
def orch():
    o = UploadBatchOrchestrator()
    # Fresh, isolated in-memory store per test - the class-level dicts are
    # otherwise shared across the whole test session.
    o._memory_batches = {}
    o._memory_items = {}
    return o


def _valid_item(item_id: str, category: str = "Tops") -> dict:
    return {
        "item_id": item_id,
        "category": category,
        "name": "Blue Shirt",
        "validation_status": "ok",
        "image_url": f"https://cdn.test/{item_id}-raw.jpg",
        "masked_url": f"https://cdn.test/{item_id}-masked.png",
        "normalized_url": f"https://cdn.test/{item_id}-normalized.png",
    }


def _needs_review_item(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "category": "Tops",
        "name": "Ambiguous Item",
        "validation_status": "needs_review",
    }


def _duplicate_item(item_id: str, *, checked: bool = True, confidence: float = 0.97, reason: str = "pixel_hash", matched_item_id: str = "existing-wardrobe-item-1") -> dict:
    item = _valid_item(item_id)
    item["duplicate"] = {
        "checked": checked,
        "is_duplicate": True,
        "reason": reason,
        "confidence": confidence,
        "matched_item_id": matched_item_id,
    }
    return item


def _not_checked_duplicate_item(item_id: str) -> dict:
    item = _valid_item(item_id)
    item["duplicate"] = {
        "checked": False,
        "is_duplicate": False,
        "reason": None,
        "confidence": 0.0,
        "matched_item_id": None,
    }
    return item


def _mock_analyze(monkeypatch, items_by_call):
    """items_by_call: list of item-lists, one per call to analyze_capture (or
    a single list reused for every call)."""
    calls = {"n": 0}

    async def _fake_analyze_capture(http_request, request):
        idx = calls["n"]
        calls["n"] += 1
        if isinstance(items_by_call, list) and items_by_call and isinstance(items_by_call[0], list):
            items = items_by_call[idx] if idx < len(items_by_call) else items_by_call[-1]
        else:
            items = items_by_call
        return {"success": bool(items), "count": len(items), "items": [dict(i) for i in items]}

    monkeypatch.setattr(wardrobe_capture, "analyze_capture", _fake_analyze_capture)
    return calls


def _mock_persist(monkeypatch, *, fail: bool = False):
    calls = {"n": 0, "args": []}

    def _fake_persist(*, user_id, selected_item_ids, detected_items):
        calls["n"] += 1
        calls["args"].append(
            {"user_id": user_id, "selected_item_ids": list(selected_item_ids), "detected_items": [dict(i) for i in detected_items]}
        )
        if fail:
            raise RuntimeError("PERSISTENCE_FAILED: simulated Appwrite failure")
        return {
            "success": True,
            "saved_count": len(detected_items),
            "items": [
                {"$id": f"w_{i.get('item_id')}", **i}
                for i in detected_items
            ],
            "skipped": 0,
            "errors": [],
        }

    monkeypatch.setattr(wardrobe_persistence_service, "persist_selected_items", _fake_persist)
    return calls


# --------------------------------------------------------------------- #
# 1. deterministic ID exactly 36 chars
# --------------------------------------------------------------------- #

def test_deterministic_id_is_exactly_36_hex_chars():
    doc_id = deterministic_appwrite_id("user-1", "batch-1")
    assert len(doc_id) == 36
    assert all(c in "0123456789abcdef" for c in doc_id)
    assert deterministic_appwrite_id("user-1", "batch-1") == doc_id


def test_status_transition_illegal_raises():
    with pytest.raises(ValueError):
        validate_status_transition("ADDED_TO_WARDROBE", "PROCESSING")


# --------------------------------------------------------------------- #
# 2. same batch request twice -> one logical batch
# --------------------------------------------------------------------- #

def test_same_batch_request_twice_is_one_logical_batch(orch):
    r1 = orch.create_or_resume_batch(user_id="u1", client_batch_request_id="batch-A", total_items=3)
    r2 = orch.create_or_resume_batch(user_id="u1", client_batch_request_id="batch-A", total_items=3)
    assert r1["success"] and r2["success"]
    assert r1["batch_id"] == r2["batch_id"] == "batch-A"
    assert r1["resumed"] is False
    assert r2["resumed"] is True


# --------------------------------------------------------------------- #
# 3 / 10. same upload item twice -> one wardrobe persistence, same wardrobe_item_id
# --------------------------------------------------------------------- #

@pytest.mark.anyio
async def test_retry_added_item_returns_same_id_zero_extra_persistence_calls(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="u2", client_batch_request_id="batch-B", total_items=1)
    _mock_analyze(monkeypatch, [_valid_item("item-1")])
    persist_calls = _mock_persist(monkeypatch)

    r1 = await orch.process_single_batch_item(
        http_request=None, user_id="u2", batch_id="batch-B",
        client_upload_item_id="upload-1", image_base64="x" * 30,
    )
    assert r1["status"] == "ADDED_TO_WARDROBE"
    assert persist_calls["n"] == 1

    r2 = await orch.process_single_batch_item(
        http_request=None, user_id="u2", batch_id="batch-B",
        client_upload_item_id="upload-1", image_base64="x" * 30,
    )
    assert r2["status"] == "ADDED_TO_WARDROBE"
    assert r2["wardrobe_item_id"] == r1["wardrobe_item_id"]
    assert r2["idempotent"] is True
    assert persist_calls["n"] == 1, "retry must not call persistence again"


# --------------------------------------------------------------------- #
# 4. one valid image -> ADDED_TO_WARDROBE
# --------------------------------------------------------------------- #

@pytest.mark.anyio
async def test_one_valid_image_added_to_wardrobe(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="u3", client_batch_request_id="batch-C", total_items=1)
    _mock_analyze(monkeypatch, [_valid_item("item-c1")])
    _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="u3", batch_id="batch-C",
        client_upload_item_id="upload-c1", image_base64="x" * 30,
    )
    assert res["success"] is True
    assert res["status"] == "ADDED_TO_WARDROBE"
    assert res["wardrobe_item_id"]

    status = orch.get_batch_status("u3", "batch-C")
    assert status["added_count"] == 1
    assert status["status"] == "COMPLETED"


# --------------------------------------------------------------------- #
# 5. three valid images -> sequential, all terminal, added_count=3
# --------------------------------------------------------------------- #

@pytest.mark.anyio
async def test_three_valid_images_sequential_all_added(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="u4", client_batch_request_id="batch-D", total_items=3)
    _mock_analyze(monkeypatch, [[_valid_item("item-d1")], [_valid_item("item-d2")], [_valid_item("item-d3")]])
    _mock_persist(monkeypatch)

    for i in (1, 2, 3):
        res = await orch.process_single_batch_item(
            http_request=None, user_id="u4", batch_id="batch-D",
            client_upload_item_id=f"upload-d{i}", image_base64="x" * 30,
        )
        assert res["status"] == "ADDED_TO_WARDROBE", f"item {i}: {res}"

    status = orch.get_batch_status("u4", "batch-D")
    assert status["added_count"] == 3
    assert status["status"] == "COMPLETED"


# --------------------------------------------------------------------- #
# 6. valid + invalid + valid -> COMPLETED_WITH_ISSUES
# --------------------------------------------------------------------- #

@pytest.mark.anyio
async def test_mixed_batch_completes_with_issues(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="u5", client_batch_request_id="batch-E", total_items=3)
    _mock_analyze(
        monkeypatch,
        [[_valid_item("item-e1")], [_needs_review_item("item-e2")], [_valid_item("item-e3")]],
    )
    _mock_persist(monkeypatch)

    r1 = await orch.process_single_batch_item(
        http_request=None, user_id="u5", batch_id="batch-E",
        client_upload_item_id="upload-e1", image_base64="x" * 30,
    )
    r2 = await orch.process_single_batch_item(
        http_request=None, user_id="u5", batch_id="batch-E",
        client_upload_item_id="upload-e2", image_base64="x" * 30,
    )
    r3 = await orch.process_single_batch_item(
        http_request=None, user_id="u5", batch_id="batch-E",
        client_upload_item_id="upload-e3", image_base64="x" * 30,
    )

    assert r1["status"] == "ADDED_TO_WARDROBE"
    assert r2["status"] == "NEEDS_REVIEW"
    assert r3["status"] == "ADDED_TO_WARDROBE", "item 3 must still run after item 2's failure"

    status = orch.get_batch_status("u5", "batch-E")
    assert status["added_count"] == 2
    assert status["needs_review_count"] == 1
    assert status["status"] == "COMPLETED_WITH_ISSUES"


# --------------------------------------------------------------------- #
# 7. persistence failure -> item FAILED, never a fake wardrobe ID
# --------------------------------------------------------------------- #

@pytest.mark.anyio
async def test_persistence_failure_sets_failed_never_fake_id(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="u6", client_batch_request_id="batch-F", total_items=1)
    _mock_analyze(monkeypatch, [_valid_item("item-f1")])
    _mock_persist(monkeypatch, fail=True)

    with pytest.raises(UploadBatchInfraError, match="PERSISTENCE_FAILED"):
        await orch.process_single_batch_item(
            http_request=None, user_id="u6", batch_id="batch-F",
            client_upload_item_id="upload-f1", image_base64="x" * 30,
        )

    status = orch.get_batch_status("u6", "batch-F")
    assert status["failed_count"] == 1

    doc = orch._get_doc(orch.items_collection, deterministic_appwrite_id("u6", "upload-f1"))
    assert doc["status"] == "FAILED"
    assert "wardrobe_item_id" not in doc


# --------------------------------------------------------------------- #
# 8. production Appwrite missing -> typed failure, no memory durability claim
# --------------------------------------------------------------------- #

def test_production_appwrite_missing_raises_typed_infra_error(orch, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(UploadBatchInfraError, match="UPLOAD_BATCH_STORE_UNAVAILABLE"):
        orch.create_or_resume_batch(user_id="u7", client_batch_request_id="batch-G", total_items=1)


# --------------------------------------------------------------------- #
# 9. cross-user batch access -> denied
# --------------------------------------------------------------------- #

def test_cross_user_batch_access_denied(orch):
    orch.create_or_resume_batch(user_id="owner", client_batch_request_id="private-batch", total_items=1)
    res = orch.get_batch_status("intruder", "private-batch")
    assert res["success"] is False
    # The batch doc id is derived from (user_id, client_batch_request_id), so
    # an intruder's lookup hashes to a different id entirely - not_found,
    # never leaking that a batch exists under someone else's id. Either
    # denial shape satisfies the canonical auth contract.
    assert res["reason"] in {"unauthorized", "batch_not_found"}


# --------------------------------------------------------------------- #
# 11. current persistence contract passthrough - image fields untouched
# --------------------------------------------------------------------- #

@pytest.mark.anyio
async def test_persistence_receives_item_with_current_image_fields_intact(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="u8", client_batch_request_id="batch-H", total_items=1)
    item = _valid_item("item-h1")
    _mock_analyze(monkeypatch, [item])
    persist_calls = _mock_persist(monkeypatch)

    await orch.process_single_batch_item(
        http_request=None, user_id="u8", batch_id="batch-H",
        client_upload_item_id="upload-h1", image_base64="x" * 30,
    )

    assert persist_calls["n"] == 1
    passed_item = persist_calls["args"][0]["detected_items"][0]
    assert passed_item["image_url"] == item["image_url"]
    assert passed_item["masked_url"] == item["masked_url"]
    assert passed_item["normalized_url"] == item["normalized_url"]
    assert passed_item["category"] == item["category"]
    assert passed_item["name"] == item["name"]


# --------------------------------------------------------------------- #
# Duplicate contract audit: canonical signal is analyze_capture()'s own
# item["duplicate"] object - no second detector, no routers.data engine.
# --------------------------------------------------------------------- #

@pytest.mark.anyio
async def test_normal_new_garment_persists(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="d1", client_batch_request_id="batch-dup1", total_items=1)
    _mock_analyze(monkeypatch, [_valid_item("item-new")])
    persist_calls = _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="d1", batch_id="batch-dup1",
        client_upload_item_id="upload-new", image_base64="x" * 30,
    )
    assert res["status"] == "ADDED_TO_WARDROBE"
    assert persist_calls["n"] == 1


@pytest.mark.anyio
async def test_duplicate_needs_review_zero_persistence(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="d2", client_batch_request_id="batch-dup2", total_items=1)
    _mock_analyze(monkeypatch, [_duplicate_item("item-dup")])
    persist_calls = _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="d2", batch_id="batch-dup2",
        client_upload_item_id="upload-dup", image_base64="x" * 30,
    )
    assert res["success"] is False
    assert res["status"] == "NEEDS_REVIEW"
    assert res["reason"] == "duplicate_wardrobe_item"
    assert persist_calls["n"] == 0

    status = orch.get_batch_status("d2", "batch-dup2")
    assert status["needs_review_count"] == 1
    assert status["added_count"] == 0


@pytest.mark.anyio
async def test_duplicate_matched_id_reason_confidence_preserved(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="d3", client_batch_request_id="batch-dup3", total_items=1)
    _mock_analyze(
        monkeypatch,
        [_duplicate_item("item-dup3", confidence=0.91, reason="image_vector", matched_item_id="wardrobe-xyz")],
    )
    _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="d3", batch_id="batch-dup3",
        client_upload_item_id="upload-dup3", image_base64="x" * 30,
    )
    assert res["matched_item_id"] == "wardrobe-xyz"
    assert res["duplicate_reason"] == "image_vector"
    assert res["duplicate_confidence"] == 0.91

    doc = orch._get_doc(orch.items_collection, deterministic_appwrite_id("d3", "upload-dup3"))
    assert doc["matched_item_id"] == "wardrobe-xyz"
    assert doc["duplicate_reason"] == "image_vector"
    assert doc["duplicate_confidence"] == 0.91
    assert doc["error_code"] == "DUPLICATE_WARDROBE_ITEM"

    # Retry (no override) must keep returning the SAME preserved fields.
    res2 = await orch.process_single_batch_item(
        http_request=None, user_id="d3", batch_id="batch-dup3",
        client_upload_item_id="upload-dup3", image_base64="x" * 30,
    )
    assert res2["matched_item_id"] == "wardrobe-xyz"
    assert res2["duplicate_reason"] == "image_vector"
    assert res2["duplicate_confidence"] == 0.91
    assert res2["idempotent"] is True


@pytest.mark.anyio
async def test_duplicate_detector_unavailable_is_normal_behavior(orch, monkeypatch):
    """checked=false must never block a save - the detector being
    unavailable is not evidence of a duplicate."""
    orch.create_or_resume_batch(user_id="d4", client_batch_request_id="batch-dup4", total_items=1)
    _mock_analyze(monkeypatch, [_not_checked_duplicate_item("item-unchecked")])
    persist_calls = _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="d4", batch_id="batch-dup4",
        client_upload_item_id="upload-unchecked", image_base64="x" * 30,
    )
    assert res["status"] == "ADDED_TO_WARDROBE"
    assert persist_calls["n"] == 1


@pytest.mark.anyio
async def test_explicit_add_anyway_persists_duplicate_once(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="d5", client_batch_request_id="batch-dup5", total_items=1)
    _mock_analyze(monkeypatch, [_duplicate_item("item-dup5")])
    persist_calls = _mock_persist(monkeypatch)

    # First call: blocked, needs review, zero persistence.
    blocked = await orch.process_single_batch_item(
        http_request=None, user_id="d5", batch_id="batch-dup5",
        client_upload_item_id="upload-dup5", image_base64="x" * 30,
    )
    assert blocked["status"] == "NEEDS_REVIEW"
    assert persist_calls["n"] == 0

    # Explicit "Add anyway" on the SAME item: persists exactly once.
    added = await orch.process_single_batch_item(
        http_request=None, user_id="d5", batch_id="batch-dup5",
        client_upload_item_id="upload-dup5", image_base64="x" * 30,
        override_duplicate=True,
    )
    assert added["status"] == "ADDED_TO_WARDROBE"
    assert added["success"] is True
    assert persist_calls["n"] == 1

    # A further retry (still override=True) must not persist again.
    retried = await orch.process_single_batch_item(
        http_request=None, user_id="d5", batch_id="batch-dup5",
        client_upload_item_id="upload-dup5", image_base64="x" * 30,
        override_duplicate=True,
    )
    assert retried["status"] == "ADDED_TO_WARDROBE"
    assert retried["idempotent"] is True
    assert persist_calls["n"] == 1

    # Batch counters must not double-count this one item across two states.
    status = orch.get_batch_status("d5", "batch-dup5")
    assert status["added_count"] == 1
    assert status["needs_review_count"] == 0


@pytest.mark.anyio
async def test_add_anyway_cannot_bypass_unrelated_rejection(orch, monkeypatch):
    """override_duplicate must only relax the DUPLICATE check - an item that
    is also needs_review for an unrelated reason must stay blocked."""
    item = _duplicate_item("item-dup-and-bad")
    item["validation_status"] = "needs_review"  # unrelated reason too
    orch.create_or_resume_batch(user_id="d6", client_batch_request_id="batch-dup6", total_items=1)
    _mock_analyze(monkeypatch, [item])
    persist_calls = _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="d6", batch_id="batch-dup6",
        client_upload_item_id="upload-dup-bad", image_base64="x" * 30,
        override_duplicate=True,
    )
    assert res["status"] == "NEEDS_REVIEW"
    assert res["reason"] == "not_auto_approved"
    assert persist_calls["n"] == 0


@pytest.mark.anyio
async def test_valid_duplicate_valid_first_and_third_continue(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="d7", client_batch_request_id="batch-dup7", total_items=3)
    _mock_analyze(
        monkeypatch,
        [[_valid_item("item-d7-1")], [_duplicate_item("item-d7-2")], [_valid_item("item-d7-3")]],
    )
    _mock_persist(monkeypatch)

    r1 = await orch.process_single_batch_item(
        http_request=None, user_id="d7", batch_id="batch-dup7",
        client_upload_item_id="upload-d7-1", image_base64="x" * 30,
    )
    r2 = await orch.process_single_batch_item(
        http_request=None, user_id="d7", batch_id="batch-dup7",
        client_upload_item_id="upload-d7-2", image_base64="x" * 30,
    )
    r3 = await orch.process_single_batch_item(
        http_request=None, user_id="d7", batch_id="batch-dup7",
        client_upload_item_id="upload-d7-3", image_base64="x" * 30,
    )

    assert r1["status"] == "ADDED_TO_WARDROBE"
    assert r2["status"] == "NEEDS_REVIEW"
    assert r2["reason"] == "duplicate_wardrobe_item"
    assert r3["status"] == "ADDED_TO_WARDROBE", "item 3 must still run after item 2 was flagged a duplicate"

    status = orch.get_batch_status("d7", "batch-dup7")
    assert status["added_count"] == 2
    assert status["needs_review_count"] == 1
    assert status["status"] == "COMPLETED_WITH_ISSUES"


@pytest.mark.anyio
async def test_retry_already_added_item_no_second_persistence(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="d8", client_batch_request_id="batch-dup8", total_items=1)
    _mock_analyze(monkeypatch, [_valid_item("item-d8")])
    persist_calls = _mock_persist(monkeypatch)

    r1 = await orch.process_single_batch_item(
        http_request=None, user_id="d8", batch_id="batch-dup8",
        client_upload_item_id="upload-d8", image_base64="x" * 30,
    )
    r2 = await orch.process_single_batch_item(
        http_request=None, user_id="d8", batch_id="batch-dup8",
        client_upload_item_id="upload-d8", image_base64="x" * 30,
    )
    assert r1["status"] == r2["status"] == "ADDED_TO_WARDROBE"
    assert r1["wardrobe_item_id"] == r2["wardrobe_item_id"]
    assert persist_calls["n"] == 1
