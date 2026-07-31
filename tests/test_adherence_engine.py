from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch


class FakeProxy:
    def __init__(self, rows=None):
        self.rows = rows or {}
        self.created = []
        self.updated = []

    def list_documents(self, resource, **kwargs):
        user_id = kwargs.get("user_id")
        rows = list(self.rows.get(resource, []))
        if user_id:
            rows = [r for r in rows if str(r.get("userId") or r.get("user_id")) == str(user_id)]
        return rows

    def get_document(self, resource, document_id):
        for row in self.rows.get(resource, []):
            if row.get("$id") == document_id or row.get("id") == document_id:
                return dict(row)
        raise Exception("not found")

    def update_document(self, resource, document_id, data):
        rows = self.rows.setdefault(resource, [])
        for idx, row in enumerate(rows):
            if row.get("$id") == document_id or row.get("id") == document_id:
                rows[idx] = {**row, **data, "$id": document_id}
                self.updated.append((resource, document_id, data))
                return dict(rows[idx])
        raise Exception("not found")

    def create_document(self, resource, data, document_id="unique()"):
        doc = {"$id": document_id if document_id != "unique()" else f"{resource}_{len(self.rows.get(resource, []))}", **data}
        self.rows.setdefault(resource, []).append(doc)
        self.created.append((resource, data, document_id))
        return dict(doc)


class FakeNotifications:
    def __init__(self):
        self.calls = []

    def schedule_reminders(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": True, "scheduled": len(kwargs.get("reminders") or [])}


def test_med_daily_generation_creates_pending_logs_and_reminders(monkeypatch):
    from brain.engines import adherence_engine
    from services.notification_store import NotificationStore

    proxy = FakeProxy(
        {
            "meds": [{"$id": "med_1", "userId": "user_1", "name": "Dolo", "dose": "650", "time": "09:00"}],
            "med_logs": [],
        }
    )
    notifications = NotificationStore()
    notifications._appwrite = proxy
    monkeypatch.setattr(adherence_engine, "notification_store", notifications)

    with patch.object(adherence_engine, "AppwriteProxy", return_value=proxy):
        data = adherence_engine.generate_daily_med_logs(
            "user_1",
            now=datetime(2026, 5, 28, 8, 0, tzinfo=timezone.utc),
        )

    assert len(data["logs"]) == 1
    assert data["logs"][0]["status"] == "pending"
    assert proxy.created[0][0] == "med_logs"
    reminders = proxy.rows[notifications.reminders_resource]
    assert len(reminders) == 2
    assert all(reminder["status"] == "scheduled" for reminder in reminders)
    due = notifications.list_due_reminders(
        now=datetime(2026, 5, 29, 0, 0, tzinfo=timezone.utc),
        window_seconds=60,
    )
    assert {reminder["$id"] for reminder in due} == {
        reminder["$id"] for reminder in reminders
    }


def test_med_auto_missed_and_manual_taken(monkeypatch):
    from brain.engines import adherence_engine

    proxy = FakeProxy(
        {
            "meds": [{"$id": "med_1", "userId": "user_1", "name": "Dolo", "dose": "650", "time": "09:00", "left": 2}],
            "med_logs": [
                {
                    "$id": "log_1",
                    "userId": "user_1",
                    "medId": "med_1",
                    "medName": "Dolo",
                    "dose": "650",
                    "time": "2026-05-28T09:00:00+05:30",
                    "status": "pending",
                }
            ],
        }
    )
    monkeypatch.setattr(adherence_engine, "notification_store", FakeNotifications())
    with patch.object(adherence_engine, "AppwriteProxy", return_value=proxy):
        missed = adherence_engine.auto_reconcile_overdue(
            "user_1",
            now=datetime(2026, 5, 28, 23, 59, tzinfo=adherence_engine._tz()),
        )
        taken = adherence_engine.mark_taken(
            "user_1",
            med_id="med_1",
            now=datetime(2026, 5, 28, 10, 0, tzinfo=adherence_engine._tz()),
        )

    assert missed["updated"] == 1
    assert taken["status"] == "taken"
    assert any(call[0] == "meds" and call[2].get("left") == 1 for call in proxy.updated)


def test_med_user_scoping_isolation(monkeypatch):
    from brain.engines import adherence_engine

    proxy = FakeProxy(
        {
            "meds": [
                {"$id": "mine", "userId": "user_1", "name": "Mine", "dose": "1"},
                {"$id": "other", "userId": "user_2", "name": "Other", "dose": "1"},
            ],
            "med_logs": [],
        }
    )
    monkeypatch.setattr(adherence_engine, "notification_store", FakeNotifications())
    with patch.object(adherence_engine, "AppwriteProxy", return_value=proxy):
        data = adherence_engine.generate_daily_med_logs("user_1")

    assert len(data["logs"]) == 1
    assert data["logs"][0]["medId"] == "mine"


def test_skincare_daily_generation_and_manual_status(monkeypatch):
    from brain.engines import adherence_engine

    proxy = FakeProxy(
        {
            "skincare_profiles": [{"$id": "profile_1", "userId": "user_1", "daySteps": [0, 1], "nightSteps": [2, 3]}],
            "skincare_logs": [],
        }
    )
    monkeypatch.setattr(adherence_engine, "notification_store", FakeNotifications())
    with patch.object(adherence_engine, "AppwriteProxy", return_value=proxy):
        data = adherence_engine.generate_daily_skincare_logs(
            "user_1",
            now=datetime(2026, 5, 28, 7, 0, tzinfo=adherence_engine._tz()),
        )
        completed = adherence_engine.mark_skincare_completed(
            "user_1",
            routine="morning",
            now=datetime(2026, 5, 28, 8, 0, tzinfo=adherence_engine._tz()),
        )

    assert {log["routine"] for log in data["logs"]} == {"morning", "night"}
    assert completed["status"] == "completed"
    assert completed["completedSteps"] == [0, 1]


def test_skincare_auto_missed(monkeypatch):
    from brain.engines import adherence_engine

    proxy = FakeProxy(
        {
            "skincare_profiles": [{"$id": "profile_1", "userId": "user_1"}],
            "skincare_logs": [
                {
                    "$id": "morning_log",
                    "userId": "user_1",
                    "date": "2026-05-28",
                    "routine": "morning",
                    "status": "pending",
                    "steps": [0, 1, 2],
                    "scheduledAtISO": "2026-05-28T06:00:00+05:30",
                }
            ],
        }
    )
    monkeypatch.setattr(adherence_engine, "notification_store", FakeNotifications())
    with patch.object(adherence_engine, "AppwriteProxy", return_value=proxy):
        data = adherence_engine.auto_reconcile_skincare_overdue(
            "user_1",
            now=datetime(2026, 5, 28, 12, 0, tzinfo=adherence_engine._tz()),
        )

    assert any(log.get("routine") == "morning" and log.get("status") == "missed" for log in data["logs"])
