import os
import sys
import unittest
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
        self.assertEqual(proxy.updated[0][2]["status"], "scheduled")
        self.assertEqual(proxy.updated[0][2]["source"], "medi")


if __name__ == "__main__":
    unittest.main()