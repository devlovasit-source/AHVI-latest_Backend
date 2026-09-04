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


def test_appwrite_proxy_user_field_map_safety():
    from services.appwrite_proxy import AppwriteProxy

    proxy = AppwriteProxy()
    # Check that style_board_states and notification_preferences are strictly mapped to userId
    assert proxy.user_field_map.get("style_board_states") == "userId"
    assert proxy.user_field_map.get("notification_preferences") == "userId"
    # style_assets is a global catalog and must be None
    assert proxy.user_field_map.get("style_assets") is None


@patch("services.r2_storage.R2Storage")
@patch("services.qdrant_service.qdrant_service")
@patch("services.appwrite_proxy.AppwriteProxy")
@patch("services.appwrite_service.is_appwrite_configured", return_value=True)
@patch("appwrite.services.users.Users.delete")
def test_execute_hard_purge_scoped_queries_only(mock_auth_delete, mock_is_conf, mock_proxy_cls, mock_qdrant, mock_r2_cls):
    user_id = "user_scoped_check"

    mock_r2_inst = MagicMock()
    mock_r2_inst.raw_bucket = None
    mock_r2_inst.style_boards_bucket = None
    mock_r2_inst.wardrobe_bucket = None
    mock_r2_cls.return_value = mock_r2_inst

    mock_qdrant.client = None

    mock_proxy_inst = MagicMock()
    from services.appwrite_proxy import AppwriteProxy
    real_proxy = AppwriteProxy()
    mock_proxy_inst.user_field_map = real_proxy.user_field_map
    mock_proxy_inst.list_documents.return_value = {"documents": []}
    mock_proxy_cls.return_value = mock_proxy_inst

    report = execute_hard_purge(user_id)
    assert report["success"] is True

    # Check all calls to list_documents
    list_calls = mock_proxy_inst.list_documents.call_args_list
    assert len(list_calls) > 0

    collections_queried = []
    for call in list_calls:
        args, kwargs = call
        resource = args[0]
        collections_queried.append(resource)
        # Every query MUST be scoped to user_id
        assert kwargs.get("user_id") == user_id
        # Every queried collection must have a mapped owner field
        assert real_proxy.user_field_map.get(resource) is not None

    # Verify style_assets was NOT queried
    assert "style_assets" not in collections_queried
    # Verify style_board_states and notification_preferences WERE queried
    assert "style_board_states" in collections_queried
    assert "notification_preferences" in collections_queried


@patch("services.r2_storage.R2Storage")
@patch("services.qdrant_service.qdrant_service")
@patch("services.appwrite_proxy.AppwriteProxy")
@patch("services.appwrite_service.is_appwrite_configured", return_value=True)
@patch("appwrite.services.users.Users.delete")
def test_execute_hard_purge_cross_tenant_document_safety(mock_auth_delete, mock_is_conf, mock_proxy_cls, mock_qdrant, mock_r2_cls):
    user_id = "target_tenant_user"

    mock_r2_inst = MagicMock()
    mock_r2_inst.raw_bucket = None
    mock_r2_inst.style_boards_bucket = None
    mock_r2_inst.wardrobe_bucket = None
    mock_r2_cls.return_value = mock_r2_inst
    mock_qdrant.client = None

    from services.appwrite_proxy import AppwriteProxy
    real_proxy = AppwriteProxy()

    mock_proxy_inst = MagicMock()
    mock_proxy_inst.user_field_map = real_proxy.user_field_map

    # Return documents for chat_threads, some belonging to other users or unowned
    def fake_list_documents(res, user_id=None, limit=100):
        if res == "chat_threads":
            return {
                "documents": [
                    {"$id": "doc_mine", "userId": "target_tenant_user"},
                    {"$id": "doc_other", "userId": "another_tenant_user"},
                    {"$id": "doc_unowned"},
                ]
            }
        return {"documents": []}

    mock_proxy_inst.list_documents.side_effect = fake_list_documents
    mock_proxy_cls.return_value = mock_proxy_inst

    report = execute_hard_purge(user_id)
    assert report["success"] is True

    # Check delete_document calls
    deleted_docs = [call[0] for call in mock_proxy_inst.delete_document.call_args_list]

    # Only doc_mine (and the user document itself from "users") should be deleted
    assert ("chat_threads", "doc_mine") in deleted_docs
    assert ("users", user_id) in deleted_docs

    # Other tenant's doc and unowned doc MUST NOT be deleted
    assert ("chat_threads", "doc_other") not in deleted_docs
    assert ("chat_threads", "doc_unowned") not in deleted_docs


def test_get_user_profile_fallback_never_returns_mismatched_user():
    from services.data_access_service import get_user_profile

    with patch("services.appwrite_proxy.AppwriteProxy.get_document", side_effect=Exception("Document not found")):
        # Scenario 1: Fallback returns an arbitrary user document (unscoped query)
        with patch("services.appwrite_proxy.AppwriteProxy.list_documents", return_value={"documents": [{"$id": "bob", "userId": "bob", "name": "Bob"}]}):
            profile = get_user_profile(user_id="alice")
            # Must NOT return Bob's record!
            assert profile == {}

        # Scenario 2: Fallback returns the matching user document
        with patch("services.appwrite_proxy.AppwriteProxy.list_documents", return_value={"documents": [{"$id": "alice", "userId": "alice", "name": "Alice"}]}):
            profile = get_user_profile(user_id="alice")
            assert profile.get("name") == "Alice"


def test_account_delete_endpoint_confirmation_validation():
    from main import app

    client = TestClient(app, raise_server_exceptions=False)
    headers = {"Authorization": "Bearer test_jwt_token"}

    with patch("main.get_current_user", return_value={"user_id": "auth_test_user"}):
        with patch("services.auth_helpers.require_user", return_value="auth_test_user"):
            with patch("routers.account.request_account_deletion", return_value={"success": True}):
                # Both "DELETE" and "CONFIRM" should be accepted
                res1 = client.post("/api/account/delete", json={"confirmation": "DELETE"}, headers=headers)
                assert res1.status_code == 200

                res2 = client.post("/api/account/delete", json={"confirmation": "CONFIRM"}, headers=headers)
                assert res2.status_code == 200

                res3 = client.post("/api/account/delete", json={"confirmation": "delete"}, headers=headers)
                assert res3.status_code == 200

                # Invalid confirmation token -> 400
                res_invalid = client.post("/api/account/delete", json={"confirmation": "INVALID"}, headers=headers)
                assert res_invalid.status_code == 400
                res_data = res_invalid.json()
                err_text = str(res_data.get("detail") or res_data.get("error", {}).get("message") or "")
                assert "DELETE' or 'CONFIRM'" in err_text


def test_middleware_auto_cancels_pending_deletion_on_user_login():
    from middleware.auth_middleware import _restore_account_if_pending_deletion

    mock_request = MagicMock()
    mock_request.url.path = "/api/wardrobe/items"

    with patch("services.account_deletion_service.get_account_deletion_status", return_value={"account_status": "pending_deletion", "is_expired": False}):
        with patch("services.account_deletion_service.cancel_account_deletion") as mock_cancel:
            _restore_account_if_pending_deletion("returning_user", mock_request)
            mock_cancel.assert_called_once_with("returning_user")

    # On status or delete paths, auto-cancellation must be skipped
    mock_request.url.path = "/api/account/status"
    with patch("services.account_deletion_service.get_account_deletion_status", return_value={"account_status": "pending_deletion", "is_expired": False}):
        with patch("services.account_deletion_service.cancel_account_deletion") as mock_cancel:
            _restore_account_if_pending_deletion("returning_user", mock_request)
            mock_cancel.assert_not_called()


def test_find_expired_accounts_pipeline():
    from scripts.purge_expired_accounts import find_expired_accounts

    # Test Attempt 1: Indexed query returning expired accounts
    mock_proxy = MagicMock()
    mock_proxy._collection_id.return_value = "users_col"
    mock_proxy._list_documents_page.return_value = {
        "documents": [
            {
                "userId": "exp_user_1",
                "account_status": "pending_deletion",
                "deletion_scheduled_at": "2020-01-01T00:00:00+00:00",
                "deletion_requested_at": "2019-11-17T00:00:00+00:00",
            }
        ],
        "used_query_syntax": True,
    }

    with patch("services.appwrite_proxy.AppwriteProxy", return_value=mock_proxy):
        expired = find_expired_accounts(limit=10)
        assert len(expired) == 1
        assert expired[0]["user_id"] == "exp_user_1"
