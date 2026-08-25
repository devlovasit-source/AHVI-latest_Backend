"""Sequential wardrobe upload batch orchestrator (AHVI P0 upload MVP).

Runs entirely against the ENVIRONMENT=test in-memory fallback (network is
blocked for the whole suite - see tests/conftest.py); analyze_capture() and
persist_selected_items() are mocked per test so this suite verifies
orchestration (idempotency, sequencing, state machine, counters), not the
real vision/persistence pipelines those already have their own tests for.
"""

from __future__ import annotations

import asyncio
import base64
import threading

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


# --------------------------------------------------------------------- #
# Reviewed-item save contract: persist the exact garment the user already
# reviewed/approved in preview, instead of re-running analyze_capture() on
# the source image bytes every save. See AHVI P0 upload save-contract fix.
# --------------------------------------------------------------------- #


def _mock_persist_empty(monkeypatch):
    """Faithfully models persist_selected_items() silently dropping an item
    with no raw_url/masked_url/normalized_url - a real success response
    (no exception) that persisted nothing."""
    calls = {"n": 0}

    def _fake_persist(*, user_id, selected_item_ids, detected_items):
        calls["n"] += 1
        return {"success": True, "saved_count": 0, "items": [], "skipped": len(detected_items), "errors": ["missing url"]}

    monkeypatch.setattr(wardrobe_persistence_service, "persist_selected_items", _fake_persist)
    return calls


@pytest.mark.anyio
async def test_reviewed_item_skips_analyze_capture_and_persists(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="r1", client_batch_request_id="batch-r1", total_items=1)
    analyze_calls = _mock_analyze(monkeypatch, [_valid_item("item-r1")])
    persist_calls = _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="r1", batch_id="batch-r1",
        client_upload_item_id="upload-r1", image_base64="x" * 30,
        reviewed_item=_valid_item("item-r1"),
    )

    assert analyze_calls["n"] == 0, "reviewed_item must never trigger a fresh analyze_capture() call"
    assert persist_calls["n"] == 1
    assert res["status"] == "ADDED_TO_WARDROBE"
    assert res["success"] is True


@pytest.mark.anyio
async def test_reviewed_item_multiple_items_from_one_source_persist_independently(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="r2", client_batch_request_id="batch-r2", total_items=2)
    analyze_calls = _mock_analyze(monkeypatch, [])
    persist_calls = _mock_persist(monkeypatch)

    r_top = await orch.process_single_batch_item(
        http_request=None, user_id="r2", batch_id="batch-r2",
        client_upload_item_id="upload-r2-top", image_base64="x" * 30,
        reviewed_item=_valid_item("item-r2-top", category="Tops"),
    )
    r_bottom = await orch.process_single_batch_item(
        http_request=None, user_id="r2", batch_id="batch-r2",
        client_upload_item_id="upload-r2-bottom", image_base64="x" * 30,
        reviewed_item=_valid_item("item-r2-bottom", category="Bottoms"),
    )

    assert analyze_calls["n"] == 0
    assert r_top["status"] == r_bottom["status"] == "ADDED_TO_WARDROBE"
    assert persist_calls["n"] == 2
    assert persist_calls["args"][0]["detected_items"][0]["item_id"] == "item-r2-top"
    assert persist_calls["args"][1]["detected_items"][0]["item_id"] == "item-r2-bottom"


@pytest.mark.anyio
async def test_reviewed_item_duplicate_without_override_needs_review(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="r3", client_batch_request_id="batch-r3", total_items=1)
    analyze_calls = _mock_analyze(monkeypatch, [])
    persist_calls = _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="r3", batch_id="batch-r3",
        client_upload_item_id="upload-r3", image_base64="x" * 30,
        reviewed_item=_duplicate_item("item-r3"),
    )

    assert analyze_calls["n"] == 0
    assert res["status"] == "NEEDS_REVIEW"
    assert res["reason"] == "duplicate_wardrobe_item"
    assert persist_calls["n"] == 0


@pytest.mark.anyio
async def test_reviewed_item_duplicate_with_override_persists_once(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="r4", client_batch_request_id="batch-r4", total_items=1)
    analyze_calls = _mock_analyze(monkeypatch, [])
    persist_calls = _mock_persist(monkeypatch)

    blocked = await orch.process_single_batch_item(
        http_request=None, user_id="r4", batch_id="batch-r4",
        client_upload_item_id="upload-r4", image_base64="x" * 30,
        reviewed_item=_duplicate_item("item-r4"),
    )
    assert blocked["status"] == "NEEDS_REVIEW"
    assert persist_calls["n"] == 0

    added = await orch.process_single_batch_item(
        http_request=None, user_id="r4", batch_id="batch-r4",
        client_upload_item_id="upload-r4", image_base64="x" * 30,
        reviewed_item=_duplicate_item("item-r4"),
        override_duplicate=True,
    )
    assert added["status"] == "ADDED_TO_WARDROBE"
    assert persist_calls["n"] == 1
    assert analyze_calls["n"] == 0


@pytest.mark.anyio
async def test_reviewed_item_retry_already_added_no_second_persistence(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="r5", client_batch_request_id="batch-r5", total_items=1)
    _mock_analyze(monkeypatch, [])
    persist_calls = _mock_persist(monkeypatch)

    r1 = await orch.process_single_batch_item(
        http_request=None, user_id="r5", batch_id="batch-r5",
        client_upload_item_id="upload-r5", image_base64="x" * 30,
        reviewed_item=_valid_item("item-r5"),
    )
    r2 = await orch.process_single_batch_item(
        http_request=None, user_id="r5", batch_id="batch-r5",
        client_upload_item_id="upload-r5", image_base64="x" * 30,
        reviewed_item=_valid_item("item-r5"),
    )

    assert r1["status"] == r2["status"] == "ADDED_TO_WARDROBE"
    assert r1["wardrobe_item_id"] == r2["wardrobe_item_id"]
    assert persist_calls["n"] == 1


@pytest.mark.anyio
async def test_reviewed_item_persistence_returns_zero_is_explicit_failure(orch, monkeypatch, caplog):
    orch.create_or_resume_batch(user_id="r6", client_batch_request_id="batch-r6", total_items=1)
    analyze_calls = _mock_analyze(monkeypatch, [])
    _mock_persist_empty(monkeypatch)

    with caplog.at_level("WARNING"):
        res = await orch.process_single_batch_item(
            http_request=None, user_id="r6", batch_id="batch-r6",
            client_upload_item_id="upload-r6", image_base64="x" * 30,
            reviewed_item=_valid_item("item-r6"),
        )

    assert analyze_calls["n"] == 0
    assert res["success"] is False
    assert res["status"] == "FAILED"
    assert res["error_code"] == "UPLOAD_ITEM_PERSISTENCE_FAILED"
    assert "ahvi.upload_batch.persistence_returned_no_items" in caplog.text

    doc = orch._get_doc(orch.items_collection, deterministic_appwrite_id("r6", "upload-r6"))
    assert doc["status"] == "FAILED"
    assert doc["error_code"] == "UPLOAD_ITEM_PERSISTENCE_FAILED"


@pytest.mark.anyio
async def test_no_reviewed_item_falls_back_to_analyze_capture(orch, monkeypatch):
    """Backwards compatibility: older clients that don't send reviewed_item
    still get the original re-analysis behavior."""
    orch.create_or_resume_batch(user_id="r7", client_batch_request_id="batch-r7", total_items=1)
    analyze_calls = _mock_analyze(monkeypatch, [_valid_item("item-r7")])
    persist_calls = _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="r7", batch_id="batch-r7",
        client_upload_item_id="upload-r7", image_base64="x" * 30,
    )

    assert analyze_calls["n"] == 1, "no reviewed_item -> must still exercise the analyze_capture fallback"
    assert persist_calls["n"] == 1
    assert res["status"] == "ADDED_TO_WARDROBE"


# --------------------------------------------------------------------- #
# FAILED-item retry semantics: FAILED is the one terminal status that IS
# retryable - same batch_id/client_upload_item_id, attempt_count increments,
# the normal process path actually re-runs. Every other terminal status
# (ADDED_TO_WARDROBE, REJECTED, NEEDS_REVIEW without override) stays exactly
# as before.
# --------------------------------------------------------------------- #


def _mock_persist_fail_then_succeed(monkeypatch, *, fail_times=1):
    """First `fail_times` calls raise (simulating a persistence hiccup);
    every call after that succeeds normally."""
    calls = {"n": 0}

    def _fake_persist(*, user_id, selected_item_ids, detected_items):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError(f"simulated persistence failure #{calls['n']}")
        return {
            "success": True,
            "saved_count": len(detected_items),
            "items": [{"$id": f"w_{i.get('item_id')}", **i} for i in detected_items],
            "skipped": 0,
            "errors": [],
        }

    monkeypatch.setattr(wardrobe_persistence_service, "persist_selected_items", _fake_persist)
    return calls


@pytest.mark.anyio
async def test_failed_item_retry_reprocesses_and_can_succeed(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="rf1", client_batch_request_id="batch-rf1", total_items=1)
    persist_calls = _mock_persist_fail_then_succeed(monkeypatch, fail_times=1)
    doc_id = deterministic_appwrite_id("rf1", "upload-rf1")

    with pytest.raises(UploadBatchInfraError, match="PERSISTENCE_FAILED"):
        await orch.process_single_batch_item(
            http_request=None, user_id="rf1", batch_id="batch-rf1",
            client_upload_item_id="upload-rf1", image_base64="x" * 30,
            reviewed_item=_valid_item("item-rf1"),
        )
    doc1 = orch._get_doc(orch.items_collection, doc_id)
    assert doc1["status"] == "FAILED"
    assert doc1["attempt_count"] == 1

    res = await orch.process_single_batch_item(
        http_request=None, user_id="rf1", batch_id="batch-rf1",
        client_upload_item_id="upload-rf1", image_base64="x" * 30,
        reviewed_item=_valid_item("item-rf1"),
    )

    assert persist_calls["n"] == 2, "retry must actually re-run persistence, not echo the stale failure"
    assert res["status"] == "ADDED_TO_WARDROBE"
    assert res["success"] is True

    doc2 = orch._get_doc(orch.items_collection, doc_id)
    assert doc2["status"] == "ADDED_TO_WARDROBE"
    assert doc2["attempt_count"] == 2
    assert not doc2.get("error_code"), "a stale error_code must not survive into the successful doc"

    status = orch.get_batch_status("rf1", "batch-rf1")
    assert status["added_count"] == 1
    assert status["failed_count"] == 0, "must not be double-counted in both failed_count and added_count"


@pytest.mark.anyio
async def test_failed_item_retry_fails_again_increments_attempt_and_stores_newest_error(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="rf2", client_batch_request_id="batch-rf2", total_items=1)
    persist_calls = _mock_persist_fail_then_succeed(monkeypatch, fail_times=2)
    doc_id = deterministic_appwrite_id("rf2", "upload-rf2")

    with pytest.raises(UploadBatchInfraError):
        await orch.process_single_batch_item(
            http_request=None, user_id="rf2", batch_id="batch-rf2",
            client_upload_item_id="upload-rf2", image_base64="x" * 30,
            reviewed_item=_valid_item("item-rf2"),
        )
    doc1 = orch._get_doc(orch.items_collection, doc_id)
    assert doc1["attempt_count"] == 1

    with pytest.raises(UploadBatchInfraError):
        await orch.process_single_batch_item(
            http_request=None, user_id="rf2", batch_id="batch-rf2",
            client_upload_item_id="upload-rf2", image_base64="x" * 30,
            reviewed_item=_valid_item("item-rf2"),
        )

    assert persist_calls["n"] == 2
    doc2 = orch._get_doc(orch.items_collection, doc_id)
    assert doc2["status"] == "FAILED"
    assert doc2["attempt_count"] == 2, "attempt_count must increment on every retry, even a repeat failure"
    assert doc2["error_code"] == "PERSISTENCE_FAILED"

    status = orch.get_batch_status("rf2", "batch-rf2")
    assert status["failed_count"] == 1, "re-affirming FAILED must never double-count the same item"


@pytest.mark.anyio
async def test_added_item_retry_stays_idempotent_after_the_failed_retry_patch(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="rf3", client_batch_request_id="batch-rf3", total_items=1)
    persist_calls = _mock_persist(monkeypatch)

    r1 = await orch.process_single_batch_item(
        http_request=None, user_id="rf3", batch_id="batch-rf3",
        client_upload_item_id="upload-rf3", image_base64="x" * 30,
        reviewed_item=_valid_item("item-rf3"),
    )
    r2 = await orch.process_single_batch_item(
        http_request=None, user_id="rf3", batch_id="batch-rf3",
        client_upload_item_id="upload-rf3", image_base64="x" * 30,
        reviewed_item=_valid_item("item-rf3"),
    )

    assert r1["status"] == r2["status"] == "ADDED_TO_WARDROBE"
    assert r1["wardrobe_item_id"] == r2["wardrobe_item_id"]
    assert r2["idempotent"] is True
    assert persist_calls["n"] == 1, "ADDED_TO_WARDROBE must stay terminal - zero extra persistence calls"


@pytest.mark.anyio
async def test_ordinary_needs_review_retry_stays_terminal_no_reprocessing(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="rf4", client_batch_request_id="batch-rf4", total_items=1)
    analyze_calls = _mock_analyze(monkeypatch, [])
    persist_calls = _mock_persist(monkeypatch)

    item = _valid_item("item-rf4")
    item["validation_status"] = "needs_review"
    first = await orch.process_single_batch_item(
        http_request=None, user_id="rf4", batch_id="batch-rf4",
        client_upload_item_id="upload-rf4", image_base64="x" * 30,
        reviewed_item=item,
    )
    assert first["status"] == "NEEDS_REVIEW"
    assert first["reason"] == "not_auto_approved"

    retry = await orch.process_single_batch_item(
        http_request=None, user_id="rf4", batch_id="batch-rf4",
        client_upload_item_id="upload-rf4", image_base64="x" * 30,
        reviewed_item=item,
    )

    assert retry["status"] == "NEEDS_REVIEW"
    assert retry["idempotent"] is True
    assert analyze_calls["n"] == 0
    assert persist_calls["n"] == 0, "ordinary NEEDS_REVIEW must never re-run processing on a bare retry"


@pytest.mark.anyio
async def test_duplicate_needs_review_retry_without_override_stays_needs_review(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="rf5", client_batch_request_id="batch-rf5", total_items=1)
    persist_calls = _mock_persist(monkeypatch)

    item = _duplicate_item("item-rf5")
    first = await orch.process_single_batch_item(
        http_request=None, user_id="rf5", batch_id="batch-rf5",
        client_upload_item_id="upload-rf5", image_base64="x" * 30,
        reviewed_item=item,
    )
    assert first["status"] == "NEEDS_REVIEW"
    assert first["reason"] == "duplicate_wardrobe_item"

    retry = await orch.process_single_batch_item(
        http_request=None, user_id="rf5", batch_id="batch-rf5",
        client_upload_item_id="upload-rf5", image_base64="x" * 30,
        reviewed_item=item,
        override_duplicate=False,
    )

    assert retry["status"] == "NEEDS_REVIEW"
    assert retry["idempotent"] is True
    assert persist_calls["n"] == 0


@pytest.mark.anyio
async def test_duplicate_needs_review_retry_with_override_reenters_and_persists_once(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="rf6", client_batch_request_id="batch-rf6", total_items=1)
    persist_calls = _mock_persist(monkeypatch)

    item = _duplicate_item("item-rf6")
    blocked = await orch.process_single_batch_item(
        http_request=None, user_id="rf6", batch_id="batch-rf6",
        client_upload_item_id="upload-rf6", image_base64="x" * 30,
        reviewed_item=item,
    )
    assert blocked["status"] == "NEEDS_REVIEW"
    assert persist_calls["n"] == 0

    added = await orch.process_single_batch_item(
        http_request=None, user_id="rf6", batch_id="batch-rf6",
        client_upload_item_id="upload-rf6", image_base64="x" * 30,
        reviewed_item=item,
        override_duplicate=True,
    )

    assert added["status"] == "ADDED_TO_WARDROBE"
    assert persist_calls["n"] == 1

    status = orch.get_batch_status("rf6", "batch-rf6")
    assert status["added_count"] == 1
    assert status["needs_review_count"] == 0, "the duplicate's needs_review_count must migrate to added_count, not double-count"


@pytest.mark.anyio
async def test_rejected_item_retry_stays_terminal(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="rf7", client_batch_request_id="batch-rf7", total_items=1)
    analyze_calls = _mock_analyze(monkeypatch, [])  # no items detected -> REJECTED
    persist_calls = _mock_persist(monkeypatch)

    first = await orch.process_single_batch_item(
        http_request=None, user_id="rf7", batch_id="batch-rf7",
        client_upload_item_id="upload-rf7", image_base64="x" * 30,
    )
    assert first["status"] == "REJECTED"

    retry = await orch.process_single_batch_item(
        http_request=None, user_id="rf7", batch_id="batch-rf7",
        client_upload_item_id="upload-rf7", image_base64="x" * 30,
    )

    assert retry["status"] == "REJECTED"
    assert retry["idempotent"] is True
    assert analyze_calls["n"] == 1, "REJECTED must stay terminal - retry must not re-analyze"
    assert persist_calls["n"] == 0

    status = orch.get_batch_status("rf7", "batch-rf7")
    assert status["rejected_count"] == 1, "must not be double-counted across the retry"


@pytest.mark.anyio
async def test_stale_processing_lease_is_reclaimed_and_retried(orch, monkeypatch):
    """A crashed/abandoned attempt leaves a PROCESSING doc with no terminal
    status - claim_item()'s existing last-write-wins reclaim must still
    apply unchanged after the FAILED-retry patch."""
    orch.create_or_resume_batch(user_id="rf8", client_batch_request_id="batch-rf8", total_items=1)
    doc_id = deterministic_appwrite_id("rf8", "upload-rf8")
    orch._memory_items[doc_id] = {
        "user_id": "rf8",
        "batch_id": "batch-rf8",
        "client_upload_item_id": "upload-rf8",
        "status": "PROCESSING",
        "attempt_count": 1,
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-01T00:00:00+00:00",
    }
    persist_calls = _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="rf8", batch_id="batch-rf8",
        client_upload_item_id="upload-rf8", image_base64="x" * 30,
        reviewed_item=_valid_item("item-rf8"),
    )

    assert res["status"] == "ADDED_TO_WARDROBE"
    assert persist_calls["n"] == 1
    doc = orch._get_doc(orch.items_collection, doc_id)
    assert doc["attempt_count"] == 2, "stale lease reclaim must still increment attempt_count"


# --------------------------------------------------------------------- #
# Catalog-generation mirror: WARDROBE_PRIVACY_CATALOG_ONLY=true makes
# _try_upload_inline_images() intentionally refuse the raw/masked crop for
# face-risk garments - normalized_url can then ONLY come from
# _maybe_generate_catalog_image(), the same canonical step save-selected's
# own Phase 2 already runs. These tests verify the orchestrator now mirrors
# that step instead of leaving the item structurally unable to persist.
# --------------------------------------------------------------------- #


def _face_risk_item_with_bytes(item_id: str, category: str = "Tops") -> dict:
    """A reviewed item carrying real inline bytes but NO pre-existing URL
    fields - the shape a fresh gemini-detected garment actually has, so
    _try_upload_inline_images (the REAL function, not mocked) exercises its
    genuine privacy/face-risk gate."""
    raw_b64 = "data:image/png;base64," + base64.b64encode(b"fake-raw-bytes").decode()
    masked_b64 = "data:image/png;base64," + base64.b64encode(b"fake-masked-bytes").decode()
    return {
        "item_id": item_id,
        "category": category,
        "name": "Blue Shirt",
        "validation_status": "ok",
        "raw_image_base64": raw_b64,
        "masked_image_base64": masked_b64,
    }


def _mock_catalog_image_success(monkeypatch, *, normalized_url="https://cdn.test/catalog.png"):
    calls = {"n": 0, "items": []}

    def _fake(item):
        calls["n"] += 1
        calls["items"].append(dict(item))
        item["catalogStatus"] = "catalog_ready"
        item["catalog_status"] = "catalog_ready"
        item["normalized_url"] = normalized_url
        item["normalizedUrl"] = normalized_url
        item["_catalog_done"] = True

    monkeypatch.setattr(wardrobe_capture, "_maybe_generate_catalog_image", _fake)
    return calls


def _mock_catalog_image_failure(monkeypatch):
    calls = {"n": 0}

    def _fake(item):
        calls["n"] += 1
        item["catalogStatus"] = "catalog_failed"
        item["catalog_status"] = "catalog_failed"
        item["_catalog_done"] = True
        # Deliberately leaves normalized_url/raw_url/masked_url unset - a
        # real catalog-generation failure (provider error, quality gate,
        # etc). _maybe_generate_catalog_image() never raises for this.

    monkeypatch.setattr(wardrobe_capture, "_maybe_generate_catalog_image", _fake)
    return calls


def _mock_persist_realistic(monkeypatch):
    """Models persist_selected_items()'s REAL missing-url skip (unlike
    _mock_persist(), which echoes every item back unconditionally) so these
    tests prove catalog generation is what actually bridges the gap, not an
    artifact of a too-permissive fake."""
    calls = {"n": 0, "args": []}

    def _fake_persist(*, user_id, selected_item_ids, detected_items):
        calls["n"] += 1
        calls["args"].append(
            {"user_id": user_id, "selected_item_ids": list(selected_item_ids), "detected_items": [dict(i) for i in detected_items]}
        )
        saved = [
            {"$id": f"w_{i.get('item_id')}", **i}
            for i in detected_items
            if i.get("raw_url") or i.get("masked_url") or i.get("normalized_url")
        ]
        return {"success": True, "saved_count": len(saved), "items": saved, "skipped": len(detected_items) - len(saved), "errors": []}

    monkeypatch.setattr(wardrobe_persistence_service, "persist_selected_items", _fake_persist)
    return calls


@pytest.mark.anyio
async def test_privacy_catalog_only_face_risk_uses_catalog_generation(orch, monkeypatch):
    monkeypatch.setenv("WARDROBE_PRIVACY_CATALOG_ONLY", "true")
    orch.create_or_resume_batch(user_id="cat1", client_batch_request_id="batch-cat1", total_items=1)
    catalog_calls = _mock_catalog_image_success(monkeypatch)
    persist_calls = _mock_persist_realistic(monkeypatch)

    item = _face_risk_item_with_bytes("item-cat1", category="Tops")
    res = await orch.process_single_batch_item(
        http_request=None, user_id="cat1", batch_id="batch-cat1",
        client_upload_item_id="upload-cat1", image_base64="x" * 30,
        reviewed_item=item,
    )

    assert catalog_calls["n"] == 1, "_maybe_generate_catalog_image must be invoked"
    passed_item = persist_calls["args"][0]["detected_items"][0]
    assert not passed_item.get("raw_url"), "the original face-risk crop must never be persisted as raw_url"
    assert not passed_item.get("masked_url"), "the original face-risk crop must never be persisted as masked_url"
    assert passed_item.get("normalized_url") == "https://cdn.test/catalog.png"
    assert persist_calls["n"] == 1
    assert res["success"] is True
    assert res["status"] == "ADDED_TO_WARDROBE"


@pytest.mark.anyio
async def test_privacy_catalog_only_catalog_failure_is_explicit_failed(orch, monkeypatch, caplog):
    monkeypatch.setenv("WARDROBE_PRIVACY_CATALOG_ONLY", "true")
    orch.create_or_resume_batch(user_id="cat2", client_batch_request_id="batch-cat2", total_items=1)
    catalog_calls = _mock_catalog_image_failure(monkeypatch)
    persist_calls = _mock_persist_realistic(monkeypatch)

    item = _face_risk_item_with_bytes("item-cat2", category="Tops")
    with caplog.at_level("WARNING"):
        res = await orch.process_single_batch_item(
            http_request=None, user_id="cat2", batch_id="batch-cat2",
            client_upload_item_id="upload-cat2", image_base64="x" * 30,
            reviewed_item=item,
        )

    assert catalog_calls["n"] == 1
    assert persist_calls["n"] == 1, "persist is still attempted - it is the one that catches the missing url"
    assert res["success"] is False
    assert res["status"] == "FAILED"
    assert res["error_code"] == "UPLOAD_ITEM_PERSISTENCE_FAILED"
    assert "ahvi.upload_batch.persistence_returned_no_items" in caplog.text


@pytest.mark.anyio
async def test_non_face_risk_item_unaffected_by_catalog_mirror(orch, monkeypatch):
    monkeypatch.setenv("WARDROBE_PRIVACY_CATALOG_ONLY", "true")
    orch.create_or_resume_batch(user_id="cat3", client_batch_request_id="batch-cat3", total_items=1)
    catalog_calls = _mock_catalog_image_success(monkeypatch, normalized_url="https://cdn.test/unused.png")
    persist_calls = _mock_persist(monkeypatch)

    item = _valid_item("item-cat3", category="Footwear")  # not face-risk - already has usable URLs
    res = await orch.process_single_batch_item(
        http_request=None, user_id="cat3", batch_id="batch-cat3",
        client_upload_item_id="upload-cat3", image_base64="x" * 30,
        reviewed_item=item,
    )

    # _maybe_generate_catalog_image is still called - it's an unconditional
    # step in canonical save-selected too - but a footwear item already
    # carrying valid inline URLs persists on those, matching canonical's
    # own never-blocking, idempotent design (no forced regression).
    assert catalog_calls["n"] == 1
    assert res["status"] == "ADDED_TO_WARDROBE"
    passed_item = persist_calls["args"][0]["detected_items"][0]
    assert passed_item["image_url"] == item["image_url"]


@pytest.mark.anyio
async def test_catalog_mirror_reviewed_item_still_skips_analyze_capture(orch, monkeypatch):
    monkeypatch.setenv("WARDROBE_PRIVACY_CATALOG_ONLY", "true")
    orch.create_or_resume_batch(user_id="cat4", client_batch_request_id="batch-cat4", total_items=1)
    analyze_calls = _mock_analyze(monkeypatch, [])
    _mock_catalog_image_success(monkeypatch)
    persist_calls = _mock_persist_realistic(monkeypatch)

    item = _face_risk_item_with_bytes("item-cat4", category="Tops")
    res = await orch.process_single_batch_item(
        http_request=None, user_id="cat4", batch_id="batch-cat4",
        client_upload_item_id="upload-cat4", image_base64="x" * 30,
        reviewed_item=item,
    )

    assert analyze_calls["n"] == 0, "catalog generation must never trigger a fresh analyze_capture() call"
    assert persist_calls["n"] == 1
    assert res["status"] == "ADDED_TO_WARDROBE"


@pytest.mark.anyio
async def test_duplicate_without_override_skips_catalog_and_persistence(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="cat5", client_batch_request_id="batch-cat5", total_items=1)
    catalog_calls = _mock_catalog_image_success(monkeypatch)
    persist_calls = _mock_persist(monkeypatch)

    res = await orch.process_single_batch_item(
        http_request=None, user_id="cat5", batch_id="batch-cat5",
        client_upload_item_id="upload-cat5", image_base64="x" * 30,
        reviewed_item=_duplicate_item("item-cat5"),
    )

    assert res["status"] == "NEEDS_REVIEW"
    assert catalog_calls["n"] == 0, "a duplicate blocked without override must never reach catalog generation"
    assert persist_calls["n"] == 0


@pytest.mark.anyio
async def test_duplicate_add_anyway_runs_catalog_then_persists_once(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="cat6", client_batch_request_id="batch-cat6", total_items=1)
    catalog_calls = _mock_catalog_image_success(monkeypatch)
    persist_calls = _mock_persist(monkeypatch)

    item = _duplicate_item("item-cat6")
    blocked = await orch.process_single_batch_item(
        http_request=None, user_id="cat6", batch_id="batch-cat6",
        client_upload_item_id="upload-cat6", image_base64="x" * 30,
        reviewed_item=item,
    )
    assert blocked["status"] == "NEEDS_REVIEW"
    assert catalog_calls["n"] == 0

    added = await orch.process_single_batch_item(
        http_request=None, user_id="cat6", batch_id="batch-cat6",
        client_upload_item_id="upload-cat6", image_base64="x" * 30,
        reviewed_item=item,
        override_duplicate=True,
    )
    assert added["status"] == "ADDED_TO_WARDROBE"
    assert catalog_calls["n"] == 1
    assert persist_calls["n"] == 1


@pytest.mark.anyio
async def test_failed_retry_reruns_catalog_and_can_succeed(orch, monkeypatch):
    monkeypatch.setenv("WARDROBE_PRIVACY_CATALOG_ONLY", "true")
    orch.create_or_resume_batch(user_id="cat7", client_batch_request_id="batch-cat7", total_items=1)
    catalog_state = {"n": 0}

    def _fake_catalog(item):
        catalog_state["n"] += 1
        if catalog_state["n"] == 1:
            item["catalogStatus"] = "catalog_failed"
            item["_catalog_done"] = True
        else:
            item["normalized_url"] = "https://cdn.test/catalog-retry.png"
            item["catalogStatus"] = "catalog_ready"
            item["_catalog_done"] = True

    monkeypatch.setattr(wardrobe_capture, "_maybe_generate_catalog_image", _fake_catalog)
    persist_calls = _mock_persist_realistic(monkeypatch)

    item = _face_risk_item_with_bytes("item-cat7", category="Tops")
    first = await orch.process_single_batch_item(
        http_request=None, user_id="cat7", batch_id="batch-cat7",
        client_upload_item_id="upload-cat7", image_base64="x" * 30,
        reviewed_item=item,
    )
    assert first["status"] == "FAILED"
    assert catalog_state["n"] == 1

    second = await orch.process_single_batch_item(
        http_request=None, user_id="cat7", batch_id="batch-cat7",
        client_upload_item_id="upload-cat7", image_base64="x" * 30,
        reviewed_item=item,
    )
    assert catalog_state["n"] == 2, "retry must actually re-run catalog generation, not echo the stale failure"
    assert second["status"] == "ADDED_TO_WARDROBE"
    assert persist_calls["n"] == 2


@pytest.mark.anyio
async def test_added_retry_remains_idempotent_with_catalog_mirror(orch, monkeypatch):
    monkeypatch.setenv("WARDROBE_PRIVACY_CATALOG_ONLY", "true")
    orch.create_or_resume_batch(user_id="cat8", client_batch_request_id="batch-cat8", total_items=1)
    catalog_calls = _mock_catalog_image_success(monkeypatch)
    persist_calls = _mock_persist_realistic(monkeypatch)

    item = _face_risk_item_with_bytes("item-cat8", category="Tops")
    r1 = await orch.process_single_batch_item(
        http_request=None, user_id="cat8", batch_id="batch-cat8",
        client_upload_item_id="upload-cat8", image_base64="x" * 30,
        reviewed_item=item,
    )
    r2 = await orch.process_single_batch_item(
        http_request=None, user_id="cat8", batch_id="batch-cat8",
        client_upload_item_id="upload-cat8", image_base64="x" * 30,
        reviewed_item=item,
    )

    assert r1["status"] == r2["status"] == "ADDED_TO_WARDROBE"
    assert r1["wardrobe_item_id"] == r2["wardrobe_item_id"]
    assert catalog_calls["n"] == 1, "idempotent ADDED retry must not re-run catalog generation"
    assert persist_calls["n"] == 1


# --------------------------------------------------------------------- #
# Execution context: _maybe_generate_catalog_image() eventually reaches a
# synchronous helper (services/bg_service.py) that calls asyncio.run(...)
# internally - legal only when NO event loop is already running on the
# calling thread. process_single_batch_item is async def, already running
# ON FastAPI's event loop, so a direct call crashes with "asyncio.run()
# cannot be called from a running event loop". The orchestrator must run
# catalog generation via asyncio.to_thread() to give it a plain thread with
# no event loop - physically proven on-device: the Nano Banana call itself
# succeeded (HTTP 200) but the transparency step crashed until this fix.
# --------------------------------------------------------------------- #


def _mock_catalog_image_thread_check(monkeypatch, *, normalized_url="https://cdn.test/catalog-thread.png"):
    """A fake catalog step that reproduces the REAL bug's exact trigger: a
    synchronous helper calling asyncio.run() internally. Records which
    thread it ran on and whether that asyncio.run() call actually succeeded
    - the same call that crashes when made directly from the event-loop
    thread and succeeds when run via asyncio.to_thread()."""
    results = {"n": 0, "thread_name": None, "is_main_thread": None, "asyncio_run_succeeded": None}

    def _fake(item):
        results["n"] += 1
        results["thread_name"] = threading.current_thread().name
        results["is_main_thread"] = threading.current_thread() is threading.main_thread()
        try:
            asyncio.run(asyncio.sleep(0))
            results["asyncio_run_succeeded"] = True
        except RuntimeError:
            results["asyncio_run_succeeded"] = False
        item["normalized_url"] = normalized_url
        item["normalizedUrl"] = normalized_url
        item["catalogStatus"] = "catalog_ready"
        item["_catalog_done"] = True

    monkeypatch.setattr(wardrobe_capture, "_maybe_generate_catalog_image", _fake)
    return results


@pytest.mark.anyio
async def test_catalog_generation_runs_off_the_event_loop_thread(orch, monkeypatch):
    orch.create_or_resume_batch(user_id="thr1", client_batch_request_id="batch-thr1", total_items=1)
    results = _mock_catalog_image_thread_check(monkeypatch)
    persist_calls = _mock_persist_realistic(monkeypatch)

    item = _face_risk_item_with_bytes("item-thr1", category="Tops")
    res = await orch.process_single_batch_item(
        http_request=None, user_id="thr1", batch_id="batch-thr1",
        client_upload_item_id="upload-thr1", image_base64="x" * 30,
        reviewed_item=item,
    )

    assert results["n"] == 1
    assert results["is_main_thread"] is False, "catalog generation must not run on the event-loop thread"
    assert results["asyncio_run_succeeded"] is True, (
        "the synchronous catalog helper's internal asyncio.run() must succeed - "
        "this is the exact call that raised 'asyncio.run() cannot be called "
        "from a running event loop' when invoked directly from async def "
        "process_single_batch_item"
    )
    assert persist_calls["n"] == 1
    passed_item = persist_calls["args"][0]["detected_items"][0]
    assert passed_item.get("normalized_url") == "https://cdn.test/catalog-thread.png"
    assert res["status"] == "ADDED_TO_WARDROBE"
