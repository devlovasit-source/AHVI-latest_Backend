import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeProxy:
    def __init__(self):
        self.created = []
        self.updated = []
        self.deleted = []
        self.rows = []

    def update_document(self, resource, document_id, data):
        self.updated.append((resource, document_id, data))
        return {"$id": document_id, **data}

    def create_document(self, resource, data, document_id=None):
        self.created.append((resource, document_id, data))
        return {"$id": document_id or "doc", **data}

    def delete_document(self, resource, document_id):
        self.deleted.append((resource, document_id))
        return {"success": True}

    def list_documents(self, resource, **kwargs):
        return list(self.rows)


class FakeReminderStore:
    def __init__(self, *, due=None, devices=None):
        self.due = list(due or [])
        self.devices = list(devices or [])
        self.sent_keys = set()
        self.marked = []

    def list_due_reminders(self, **kwargs):
        return []

    def list_due_medicine_reminders(self, **kwargs):
        return list(self.due)

    def was_notification_sent(self, *, notification_key):
        return notification_key in self.sent_keys

    def list_devices(self, *, user_id):
        return list(self.devices)

    def mark_medicine_reminder(self, **kwargs):
        self.marked.append(kwargs)
        if kwargs.get("status") == "sent":
            self.sent_keys.add(kwargs.get("notification_key"))

    def mark_reminder(self, **kwargs):
        self.marked.append(kwargs)


class FakeFirebase:
    def __init__(self):
        self.calls = []

    def send_to_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": True, "sent": len(kwargs.get("tokens") or []), "failed": 0}


def _request(secret="test-secret", dry_run=""):
    return SimpleNamespace(
        headers={"x-dispatch-secret": secret},
        query_params={"dry_run": dry_run} if dry_run else {},
    )


class NotificationStoreTests(unittest.TestCase):
    def test_notification_resources_use_collection_env_names(self):
        with patch.dict(
            os.environ,
            {
                "APPWRITE_COLLECTION_NOTIFICATION_DEVICES": "devices_collection",
                "APPWRITE_COLLECTION_NOTIFICATION_REMINDERS": "reminders_collection",
            },
            clear=False,
        ):
            with patch("services.notification_store.AppwriteProxy", return_value=FakeProxy()):
                from services.notification_store import NotificationStore

                store = NotificationStore()

        self.assertEqual(store.devices_resource, "devices_collection")
        self.assertEqual(store.reminders_resource, "reminders_collection")

    def test_device_registration_upserts_user_token(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            doc_id = store.upsert_device(
                user_id="user_1",
                platform="android",
                token="token_12345678901234567890",
            )

        self.assertTrue(doc_id.startswith("dev_"))
        self.assertEqual(proxy.updated[0][0], "notification_devices")
        self.assertEqual(proxy.updated[0][2]["userId"], "user_1")
        self.assertEqual(proxy.updated[0][2]["platform"], "android")

    def test_schedule_reminders_persists_scheduled_rows(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            result = store.schedule_reminders(
                user_id="user_1",
                event_id="med_1",
                source="medi",
                reminders=[
                    {
                        "sendAtISO": "2026-05-01T12:00:00+00:00",
                        "message": "Take vitamin D",
                        "priority": "high",
                        "offsetMinutes": 0,
                    }
                ],
            )

        self.assertEqual(result["scheduled"], 1)
        self.assertEqual(proxy.updated[0][0], "notification_reminders")
        # Default is now "scheduled" so dispatch_due_reminders_task (which only
        # picks status=="scheduled") can actually send them.
        self.assertEqual(proxy.updated[0][2]["status"], "scheduled")
        self.assertEqual(proxy.updated[0][2]["message"], "Take vitamin D")
        self.assertEqual(proxy.updated[0][2]["lastError"], "")
        self.assertEqual(proxy.updated[0][2]["source"], "medi")

    def test_medi_pending_reminder_is_persisted_as_scheduled(self):
        proxy = FakeProxy()
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            store = NotificationStore()
            result = store.schedule_reminders(
                user_id="user_1",
                event_id="medlog_1",
                source="medicine",
                reminders=[
                    {
                        "status": "pending",
                        "sendAtISO": "2026-05-01T12:00:00+00:00",
                        "message": "Take Dolo",
                    }
                ],
            )

        self.assertEqual(result["scheduled"], 1)
        self.assertEqual(proxy.updated[0][2]["status"], "scheduled")

    def test_list_due_reminders_includes_existing_pending_medi_rows(self):
        proxy = FakeProxy()
        proxy.rows = [
            {
                "$id": "rem_1",
                "userId": "user_1",
                "eventId": "medlog_1",
                "source": "medicine",
                "status": "pending",
                "message": "Take Dolo",
                "sendAtISO": "2026-05-01T12:00:00+00:00",
            }
        ]
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            due = NotificationStore().list_due_reminders(
                now=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
                window_seconds=120,
            )

        self.assertEqual(len(due), 1)

    def test_list_due_medicine_reminders_scans_direct_appwrite_meds(self):
        now = datetime(2026, 5, 1, 6, 0, tzinfo=timezone.utc)
        proxy = FakeProxy()
        proxy.rows = [
            {
                "$id": "med_1",
                "userId": "user_1",
                "name": "Dolo",
                "time": now.astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%I:%M %p"),
                "reminder": True,
            }
        ]
        with patch("services.notification_store.AppwriteProxy", return_value=proxy):
            from services.notification_store import NotificationStore

            due = NotificationStore().list_due_medicine_reminders(now=now, window_seconds=180)

        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["medId"], "med_1")
        self.assertEqual(due[0]["source"], "medicine")
        self.assertTrue(due[0]["notificationKey"].startswith("med:user_1:med_1:"))

    def test_dispatch_due_dry_run_returns_due_medi_reminder(self):
        from routers import notifications

        due = [
            {
                "$id": "rem_1",
                "userId": "user_1",
                "source": "medicine",
                "message": "Time to take Dolo.",
                "sendAtISO": "2026-05-01T12:00:00+00:00",
                "medId": "med_1",
                "medName": "Dolo",
                "notificationKey": "med:user_1:med_1:2026-05-01T12:00:00+00:00",
            }
        ]
        store = FakeReminderStore(due=due)
        firebase = FakeFirebase()
        with patch.dict(os.environ, {"NOTIFICATIONS_DISPATCH_SECRET": "test-secret"}, clear=False):
            with patch.object(notifications, "notification_store", store), patch.object(notifications, "firebase_push_service", firebase):
                result = notifications.dispatch_due(_request(dry_run="true"), window_minutes=3)

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["window_seconds"], 180)
        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["due"][0]["medId"], "med_1")
        self.assertEqual(firebase.calls, [])

    def test_dispatch_due_sends_medi_payload_once(self):
        from routers import notifications

        key = "med:user_1:med_1:2026-05-01T12:00:00+00:00"
        due = [
            {
                "$id": "rem_1",
                "userId": "user_1",
                "source": "medicine",
                "message": "Time to take Dolo.",
                "sendAtISO": "2026-05-01T12:00:00+00:00",
                "medId": "med_1",
                "medName": "Dolo",
                "notificationKey": key,
            }
        ]
        store = FakeReminderStore(due=due, devices=[{"token": "token_12345678901234567890"}])
        firebase = FakeFirebase()
        with patch.dict(os.environ, {"NOTIFICATIONS_DISPATCH_SECRET": "test-secret"}, clear=False):
            with patch.object(notifications, "notification_store", store), patch.object(notifications, "firebase_push_service", firebase):
                first = notifications.dispatch_due(_request(), window_minutes=3)
                second = notifications.dispatch_due(_request(), window_minutes=3)

        self.assertEqual(first["sent"], 1)
        self.assertEqual(second["skipped"], 1)
        self.assertEqual(len(firebase.calls), 1)
        payload = firebase.calls[0]["data"]
        self.assertEqual(payload["action"], "med_reminder")
        self.assertEqual(payload["medId"], "med_1")
        self.assertEqual(payload["screen"], "medi")
        self.assertIn("ahvi://medi/reminder/rem_1", payload["deepLink"])

    def test_dispatch_due_no_device_token_skips_cleanly(self):
        from routers import notifications

        due = [
            {
                "$id": "rem_1",
                "userId": "user_1",
                "source": "medicine",
                "message": "Time to take Dolo.",
                "sendAtISO": "2026-05-01T12:00:00+00:00",
                "medId": "med_1",
                "medName": "Dolo",
                "notificationKey": "med:user_1:med_1:2026-05-01T12:00:00+00:00",
            }
        ]
        store = FakeReminderStore(due=due, devices=[])
        firebase = FakeFirebase()
        with patch.dict(os.environ, {"NOTIFICATIONS_DISPATCH_SECRET": "test-secret"}, clear=False):
            with patch.object(notifications, "notification_store", store), patch.object(notifications, "firebase_push_service", firebase):
                result = notifications.dispatch_due(_request(), window_minutes=3)

        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(firebase.calls, [])

    def test_dispatch_secret_is_enforced(self):
        from fastapi import HTTPException
        from routers import notifications

        with patch.dict(os.environ, {"NOTIFICATIONS_DISPATCH_SECRET": "test-secret"}, clear=False):
            with self.assertRaises(HTTPException) as ctx:
                notifications.dispatch_due(_request(secret="wrong"), window_minutes=3)

        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
