import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeProxy:
    def __init__(self):
        self.created = []
        self.updated = []
        self.rows = []
        self.doc = {}

    def create_document(self, resource, data, document_id="unique()"):
        self.created.append((resource, data, document_id))
        return {"$id": "doc_1", **data}

    def list_documents(self, resource, **kwargs):
        self.list_kwargs = kwargs
        return list(self.rows)

    def get_document(self, resource, document_id):
        return dict(self.doc)

    def update_document(self, resource, document_id, data):
        self.updated.append((resource, document_id, data))
        return {"$id": document_id, **self.doc, **data}


class CalendarMedLogTests(unittest.TestCase):
    def test_calendar_create_includes_user_id(self):
        from services import calendar_service

        proxy = FakeProxy()
        with patch.object(calendar_service, "AppwriteProxy", return_value=proxy):
            event = calendar_service.create_calendar_event(
                "user_1",
                {"title": "Office", "start_time": "2026-05-19T10:00:00+05:30"},
            )

        self.assertEqual(proxy.created[0][0], "calendar_events")
        self.assertEqual(proxy.created[0][1]["userId"], "user_1")
        self.assertEqual(proxy.created[0][1]["occasion"], "plan")
        self.assertNotIn("user_id", proxy.created[0][1])
        self.assertEqual(event["user_id"], "user_1")

    def test_calendar_text_parser_handles_tomorrow_10_am_kolkata(self):
        from services.calendar_service import parse_plan_text_to_payload

        payload = parse_plan_text_to_payload(
            "office tomorrow 10 AM",
            category="Office",
            now=datetime(2026, 5, 18, 2, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(payload["type"], "Office")
        self.assertEqual(payload["occasion"], "Office")
        self.assertEqual(payload["timezone"], "Asia/Kolkata")
        self.assertIn("2026-05-19T10:00:00+05:30", payload["start_time"])

    def test_med_log_create_includes_user_id(self):
        from services import med_log_service

        proxy = FakeProxy()
        with patch.object(med_log_service, "AppwriteProxy", return_value=proxy):
            log = med_log_service.create_med_log(
                "user_1",
                {
                    "medId": "med_1",
                    "medName": "Vitamin D",
                    "dose": "1 pill",
                    "time": "2026-05-18T08:00:00+05:30",
                    "status": "taken",
                },
            )

        self.assertEqual(proxy.created[0][0], "med_logs")
        self.assertEqual(proxy.created[0][1]["userId"], "user_1")
        self.assertEqual(log["status"], "taken")

    def test_med_log_fetch_filters_by_user_id(self):
        from services import med_log_service

        proxy = FakeProxy()
        proxy.rows = [
            {"$id": "mine", "userId": "user_1", "medId": "a", "medName": "A", "dose": "1", "time": "2026-05-18T08:00:00+00:00", "status": "taken"},
            {"$id": "other", "userId": "user_2", "medId": "b", "medName": "B", "dose": "1", "time": "2026-05-18T08:00:00+00:00", "status": "taken"},
        ]
        with patch.object(med_log_service, "AppwriteProxy", return_value=proxy):
            logs = med_log_service.list_med_logs("user_1")

        self.assertEqual([log["id"] for log in logs], ["mine"])
        self.assertEqual(proxy.list_kwargs["user_id"], "user_1")

    def test_med_log_update_rejects_other_owner(self):
        from services import med_log_service

        proxy = FakeProxy()
        proxy.doc = {"$id": "log_1", "userId": "other", "status": "taken"}
        with patch.object(med_log_service, "AppwriteProxy", return_value=proxy):
            with self.assertRaises(PermissionError):
                med_log_service.update_med_log_status("user_1", "log_1", "skipped")


if __name__ == "__main__":
    unittest.main()
