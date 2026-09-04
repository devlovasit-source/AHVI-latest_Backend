from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.account_deletion_service import (
    GRACE_PERIOD_DAYS,
    ACCOUNT_STATUS_ACTIVE,
    ACCOUNT_STATUS_PENDING_DELETION,
    request_account_deletion,
    cancel_account_deletion,
    get_account_deletion_status,
    execute_hard_purge,
    _parse_iso,
)


def test_iso_parsing_and_45_days_schedule():
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    parsed = _parse_iso(now_iso)
    assert parsed is not None
    assert abs((parsed - now).total_seconds()) < 1.0


@patch("services.appwrite_proxy.AppwriteProxy.update_document")
@patch("services.appwrite_service.is_appwrite_configured", return_value=True)
@patch("appwrite.services.users.Users.delete_sessions")
def test_request_account_deletion(mock_delete_sessions, mock_is_configured, mock_update_doc):
    user_id = "test_user_45"
    result = request_account_deletion(user_id=user_id, reason="No longer needed")

    assert result["success"] is True
    assert result["user_id"] == user_id
    assert result["account_status"] == ACCOUNT_STATUS_PENDING_DELETION
    assert result["grace_period_days"] == 45

    # Verify 45-day calculation
    req_dt = _parse_iso(result["deletion_requested_at"])
    sched_dt = _parse_iso(result["deletion_scheduled_at"])
    assert req_dt is not None and sched_dt is not None
    diff = sched_dt - req_dt
    assert diff.days == 45

    # Verify proxy was called
    mock_update_doc.assert_called_once()
    args, kwargs = mock_update_doc.call_args
    assert args[0] == "users"
    assert args[1] == user_id
    assert args[2]["account_status"] == "pending_deletion"
    assert args[2]["deletion_reason"] == "No longer needed"

    # Verify session revocation was invoked
    mock_delete_sessions.assert_called_once_with(user_id)


@patch("services.appwrite_proxy.AppwriteProxy.update_document")
def test_cancel_account_deletion(mock_update_doc):
    user_id = "test_user_cancel"
    result = cancel_account_deletion(user_id=user_id)

    assert result["success"] is True
    assert result["user_id"] == user_id
    assert result["account_status"] == ACCOUNT_STATUS_ACTIVE

    mock_update_doc.assert_called_once()
    args, kwargs = mock_update_doc.call_args
    assert args[0] == "users"
    assert args[1] == user_id
    assert args[2]["account_status"] == "active"
    assert args[2]["deletion_scheduled_at"] == ""


@patch("services.data_access_service.get_user_profile")
def test_get_account_deletion_status_pending(mock_get_profile):
    user_id = "test_user_status"
    sched_dt = datetime.now(timezone.utc) + timedelta(days=20, hours=5)
    mock_get_profile.return_value = {
        "userId": user_id,
        "account_status": "pending_deletion",
        "deletion_requested_at": (datetime.now(timezone.utc) - timedelta(days=25)).isoformat(),
        "deletion_scheduled_at": sched_dt.isoformat(),
        "deletion_reason": "Moving platforms",
    }

    status = get_account_deletion_status(user_id)
    assert status["account_status"] == "pending_deletion"
    assert status["days_remaining"] == 20
    assert status["is_expired"] is False
    assert status["grace_period_days"] == 45


@patch("services.r2_storage.R2Storage")
@patch("services.qdrant_service.qdrant_service")
@patch("services.appwrite_proxy.AppwriteProxy")
@patch("services.appwrite_service.is_appwrite_configured", return_value=True)
@patch("appwrite.services.users.Users.delete")
def test_execute_hard_purge(mock_auth_delete, mock_is_conf, mock_proxy_cls, mock_qdrant, mock_r2_cls):
    user_id = "user_to_purge"

    # Mock R2 client
    mock_r2_inst = MagicMock()
    mock_r2_inst.raw_bucket = "raw-bucket"
    mock_r2_inst.style_boards_bucket = "boards-bucket"
    mock_r2_inst.wardrobe_bucket = "wardrobe-bucket"
    mock_r2_client = MagicMock()
    mock_r2_client.list_objects.return_value = []
    mock_r2_inst._client.return_value = mock_r2_client
    mock_r2_cls.return_value = mock_r2_inst

    # Mock Qdrant
    mock_qdrant.client = MagicMock()
    mock_qdrant.collection = "wardrobe"
    mock_qdrant.image_collection = "wardrobe_images"
    mock_qdrant.memory_collection = "outfit_memory"
    mock_qdrant.user_memory_collection = "user_memory"

    # Mock AppwriteProxy
    mock_proxy_inst = MagicMock()
    mock_proxy_inst.list_documents.return_value = {"documents": []}
    mock_proxy_cls.return_value = mock_proxy_inst

    report = execute_hard_purge(user_id)

    assert report["success"] is True
    assert report["user_id"] == user_id
    assert report["targets"]["r2_storage"]["status"] == "completed"
    assert report["targets"]["qdrant_vectors"]["status"] == "completed"
    assert report["targets"]["appwrite_database"]["status"] == "completed"
    assert report["targets"]["appwrite_auth"]["status"] == "deleted"

    mock_auth_delete.assert_called_once_with(user_id)


def test_router_endpoints_with_test_client():
    from main import app

    client = TestClient(app, raise_server_exceptions=False)

    # 1. Status without auth -> 401
    resp = client.get("/api/account/status")
    assert resp.status_code in (401, 403)

    # 2. Delete without auth -> 401
    resp = client.post("/api/account/delete", json={"confirmation": "DELETE"})
    assert resp.status_code in (401, 403)

    # 3. Cancel-delete without auth -> 401
    resp = client.post("/api/account/cancel-delete")
    assert resp.status_code in (401, 403)
