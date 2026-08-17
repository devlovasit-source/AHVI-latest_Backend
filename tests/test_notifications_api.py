"""
Integration-level tests for routers/notifications.py.

Unlike tests/test_notifications.py (which exercises NotificationStore against
a FakeProxy), this file hits the actual FastAPI routes through TestClient,
with notification_store / firebase_push_service / get_current_user mocked.
No real Appwrite or Firebase connection is required or used.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routers.notifications as notifications_router
from middleware.auth_middleware import get_current_user


def _build_app():
    app = FastAPI()
    app.include_router(notifications_router.router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "user_1"}
    return app


class NotificationPreferencesApiTests(unittest.TestCase):
    def setUp(self):
        self.app = _build_app()
        self.client = TestClient(self.app)

    def test_get_preferences_returns_all_three_categories(self):
        with patch.object(
            notifications_router.notification_store,
            "get_notification_preference",
            return_value=True,
        ) as mocked:
            resp = self.client.get("/api/notifications/preferences")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertEqual(
            body["preferences"], {"medi": True, "calendar": True, "style": True}
        )
        self.assertEqual(mocked.call_count, 3)

    def test_get_preferences_requires_auth(self):
        app = FastAPI()
        app.include_router(notifications_router.router)
        app.dependency_overrides[get_current_user] = lambda: {"user_id": ""}
        client = TestClient(app)

        resp = client.get("/api/notifications/preferences")
        self.assertEqual(resp.status_code, 401)

    def test_put_preference_valid_category(self):
        with patch.object(
            notifications_router.notification_store,
            "set_notification_preference",
            return_value=True,
        ) as mocked:
            resp = self.client.put(
                "/api/notifications/preferences/style", json={"enabled": False}
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(), {"success": True, "category": "style", "enabled": False}
        )
        mocked.assert_called_once_with(
            user_id="user_1", category="style", enabled=False
        )

    def test_put_preference_invalid_category_returns_400(self):
        resp = self.client.put(
            "/api/notifications/preferences/not_a_category", json={"enabled": True}
        )
        self.assertEqual(resp.status_code, 400)

    def test_put_preference_store_failure_returns_500(self):
        with patch.object(
            notifications_router.notification_store,
            "set_notification_preference",
            return_value=False,
        ):
            resp = self.client.put(
                "/api/notifications/preferences/medi", json={"enabled": True}
            )
        self.assertEqual(resp.status_code, 500)


class DispatchDueSuppressionApiTests(unittest.TestCase):
    """
    Exercises POST /api/notifications/dispatch-due end to end (router layer),
    confirming: disabled preference -> no FCM call + suppressed status,
    enabled preference -> FCM called + sent status.
    """

    def setUp(self):
        self.app = _build_app()
        self.client = TestClient(self.app)
        # dispatch-due checks a secret only if NOTIFICATIONS_DISPATCH_SECRET is set;
        # ensure it's unset so the (dev-only) open path is used in this test.
        os.environ.pop("NOTIFICATIONS_DISPATCH_SECRET", None)
        os.environ.pop("ENV", None)
        os.environ.pop("APP_ENV", None)
        os.environ.pop("ENVIRONMENT", None)

    def _due_reminder(self, source="style", doc_id="rem_1"):
        return {
            "$id": doc_id,
            "userId": "user_1",
            "message": "Time for your reminder",
            "source": source,
        }

    def test_disabled_category_is_suppressed_and_skips_fcm(self):
        with patch.object(
            notifications_router.notification_store,
            "list_due_reminders",
            return_value=[self._due_reminder(source="style")],
        ), patch.object(
            notifications_router.notification_store,
            "try_claim_reminder",
            return_value={"claimed": True, "reason": "new_claim"},
        ), patch.object(
            notifications_router.notification_store,
            "get_notification_preference_for_dispatch",
            return_value=(False, "disabled"),
        ), patch.object(
            notifications_router.notification_store, "list_devices"
        ) as mocked_list_devices, patch.object(
            notifications_router.firebase_push_service, "send_to_tokens"
        ) as mocked_send, patch.object(
            notifications_router.notification_store, "finalize_claim"
        ) as mocked_finalize, patch.object(
            notifications_router.notification_store, "mark_reminder"
        ) as mocked_mark:
            resp = self.client.post("/api/notifications/dispatch-due")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["processed"], 1)
        self.assertEqual(body["sent"], 0)
        self.assertEqual(body["suppressed"], 1)

        mocked_send.assert_not_called()
        mocked_list_devices.assert_not_called()
        mocked_finalize.assert_called_once_with(
            reminder_doc_id="rem_1", status="suppressed"
        )
        mocked_mark.assert_called_once_with(
            reminder_doc_id="rem_1", status="suppressed", error="user_preference"
        )

    def test_enabled_category_calls_fcm_and_marks_sent(self):
        with patch.object(
            notifications_router.notification_store,
            "list_due_reminders",
            return_value=[self._due_reminder(source="calendar", doc_id="rem_2")],
        ), patch.object(
            notifications_router.notification_store,
            "try_claim_reminder",
            return_value={"claimed": True, "reason": "new_claim"},
        ), patch.object(
            notifications_router.notification_store,
            "get_notification_preference_for_dispatch",
            return_value=(True, "enabled"),
        ), patch.object(
            notifications_router.notification_store,
            "list_devices",
            return_value=[{"token": "tok_abc"}],
        ), patch.object(
            notifications_router.firebase_push_service,
            "send_to_tokens",
            return_value={"success": True, "sent": 1, "failed": 0},
        ) as mocked_send, patch.object(
            notifications_router.notification_store, "finalize_claim"
        ) as mocked_finalize, patch.object(
            notifications_router.notification_store, "mark_reminder"
        ) as mocked_mark:
            resp = self.client.post("/api/notifications/dispatch-due")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["processed"], 1)
        self.assertEqual(body["sent"], 1)
        self.assertEqual(body["suppressed"], 0)

        mocked_send.assert_called_once()
        mocked_finalize.assert_called_once_with(reminder_doc_id="rem_2", status="sent")

    def test_preference_lookup_failure_releases_claim_and_keeps_scheduled(self):
        """
        The new boss-requested behaviour: a lookup failure must NOT be
        treated as enabled. The claim is released (not left dangling) and
        the reminder stays retryable.
        """
        with patch.object(
            notifications_router.notification_store,
            "list_due_reminders",
            return_value=[self._due_reminder(source="medi", doc_id="rem_3")],
        ), patch.object(
            notifications_router.notification_store,
            "try_claim_reminder",
            return_value={"claimed": True, "reason": "new_claim"},
        ), patch.object(
            notifications_router.notification_store,
            "get_notification_preference_for_dispatch",
            return_value=(False, "lookup_failed"),
        ), patch.object(
            notifications_router.firebase_push_service, "send_to_tokens"
        ) as mocked_send, patch.object(
            notifications_router.notification_store, "release_claim"
        ) as mocked_release, patch.object(
            notifications_router.notification_store, "mark_reminder"
        ) as mocked_mark:
            resp = self.client.post("/api/notifications/dispatch-due")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["sent"], 0)
        self.assertEqual(body["suppressed"], 0)

        mocked_send.assert_not_called()
        mocked_release.assert_called_once_with(reminder_doc_id="rem_3")
        mocked_mark.assert_called_once_with(
            reminder_doc_id="rem_3",
            status="scheduled",
            error="preference_lookup_failed",
        )

    def test_double_dispatch_only_sends_once(self):
        """
        Simulates two overlapping dispatch runs racing over the same due
        reminder: the second call's try_claim_reminder loses the race
        (already_processing), so FCM must only ever be called once total.
        """
        due_reminder = self._due_reminder(source="calendar", doc_id="rem_4")

        # First call wins the claim; second call (racing / duplicate
        # invocation) finds it already claimed by the first.
        claim_results = [
            {"claimed": True, "reason": "new_claim"},
            {"claimed": False, "reason": "already_processing"},
        ]

        with patch.object(
            notifications_router.notification_store,
            "list_due_reminders",
            return_value=[due_reminder],
        ), patch.object(
            notifications_router.notification_store,
            "try_claim_reminder",
            side_effect=claim_results,
        ), patch.object(
            notifications_router.notification_store,
            "get_notification_preference_for_dispatch",
            return_value=(True, "enabled"),
        ), patch.object(
            notifications_router.notification_store,
            "list_devices",
            return_value=[{"token": "tok_abc"}],
        ), patch.object(
            notifications_router.firebase_push_service,
            "send_to_tokens",
            return_value={"success": True, "sent": 1, "failed": 0},
        ) as mocked_send, patch.object(
            notifications_router.notification_store, "finalize_claim"
        ), patch.object(notifications_router.notification_store, "mark_reminder"):
            first_resp = self.client.post("/api/notifications/dispatch-due")
            second_resp = self.client.post("/api/notifications/dispatch-due")

        self.assertEqual(first_resp.json()["sent"], 1)
        self.assertEqual(second_resp.json()["sent"], 0)
        self.assertEqual(second_resp.json()["duplicate_prevented"], 1)
        # The key assertion: across BOTH dispatch calls, FCM was only
        # actually invoked once for this one reminder occurrence.
        mocked_send.assert_called_once()

    def test_already_sent_reminder_is_never_resent(self):
        """
        A reminder whose claim is already "sent" (e.g. list_due_reminders
        picked up a stale row, or a retry hit an already-completed
        occurrence) must not trigger a second FCM call.
        """
        with patch.object(
            notifications_router.notification_store,
            "list_due_reminders",
            return_value=[self._due_reminder(source="medi", doc_id="rem_5")],
        ), patch.object(
            notifications_router.notification_store,
            "try_claim_reminder",
            return_value={"claimed": False, "reason": "already_sent"},
        ), patch.object(
            notifications_router.firebase_push_service, "send_to_tokens"
        ) as mocked_send:
            resp = self.client.post("/api/notifications/dispatch-due")

        body = resp.json()
        self.assertEqual(body["sent"], 0)
        self.assertEqual(body["duplicate_prevented"], 1)
        mocked_send.assert_not_called()


if __name__ == "__main__":
    unittest.main()