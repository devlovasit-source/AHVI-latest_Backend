from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from routers import notifications
from services.notification_store import ReminderSchemaError, ReminderStoreError


def request(secret="secret"):
    return SimpleNamespace(headers={"x-dispatch-secret": secret}, state=SimpleNamespace(request_id=""))


class Store:
    def __init__(self, due=None, occurrences=None, error=None):
        self.due = due or []
        self.occurrences = occurrences or []
        self.error = error
        self.marked = []

    def list_due_reminders(self, **kwargs):
        return list(self.due)

    def list_medicine_occurrences_to_seed(self, **kwargs):
        if self.error:
            raise self.error
        return list(self.occurrences)

    def list_devices(self, **kwargs):
        return [{"token": "generic-token"}]

    def mark_reminder(self, **kwargs):
        self.marked.append(kwargs)


class Firebase:
    def __init__(self, ready=True):
        self.is_ready = ready
        self.calls = []

    def ready(self):
        return self.is_ready

    def send_to_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": True, "sent": 1, "failed": 0}


class Durable:
    instances = []

    def __init__(self, *, store, firebase):
        self.store = store
        self.firebase = firebase
        self.recovery_minutes = 15
        self.scheduled = []
        self.runs = []
        Durable.instances.append(self)

    def schedule_occurrence(self, **kwargs):
        self.scheduled.append(kwargs)

    def run(self, **kwargs):
        self.runs.append(kwargs)
        return {"due": 2, "claimed": 1, "dispatched": 1, "duplicate": 0, "cancelled_taken": 0, "cancelled_skipped": 0, "no_token": 0, "failed": 0}


def call(store, firebase, *, enabled=False, dry_run=False):
    Durable.instances = []
    env = {"NOTIFICATIONS_DISPATCH_SECRET": "secret", "ENABLE_DURABLE_MED_REMINDERS": str(enabled).lower()}
    with patch.dict("os.environ", env, clear=False), patch.object(notifications, "notification_store", store), patch.object(notifications, "firebase_push_service", firebase), patch.object(notifications, "MedicineReminderDispatcher", Durable):
        return notifications.dispatch_due(request(), dry_run=dry_run)


def test_feature_flag_off_preserves_legacy_medicine_dispatch():
    store = Store(due=[{"$id": "legacy", "userId": "u", "source": "medicine", "message": "legacy"}])
    firebase = Firebase()
    result = call(store, firebase, enabled=False)
    assert result == {"success": True, "processed": 1, "sent": 1, "failed": 0}
    assert len(firebase.calls) == 1


def test_feature_flag_on_excludes_legacy_medicine_and_uses_durable_dispatcher():
    store = Store(
        due=[
            {"$id": "legacy-med", "userId": "u", "source": "medicine", "message": "must-not-send"},
            {"$id": "generic", "userId": "u", "source": "calendar", "message": "generic"},
        ],
        occurrences=[{"userId": "u", "medId": "m", "time": "2026-05-01T20:00:00+00:00"}],
    )
    result = call(store, Firebase(), enabled=True)
    assert result["medicine"]["dispatched"] == 1
    assert result["generic"]["sent"] == 1
    assert len(Durable.instances[0].scheduled) == 1


def test_durable_dry_run_has_no_writes_or_sends():
    store = Store(occurrences=[{"userId": "u", "medId": "m", "time": "2026-05-01T20:00:00+00:00"}])
    firebase = Firebase()
    result = call(store, firebase, enabled=True, dry_run=True)
    instance = Durable.instances[0]
    assert result["dry_run"] is True
    assert instance.scheduled[0]["dry_run"] is True
    assert instance.runs[0]["dry_run"] is True
    assert not firebase.calls


def test_schema_and_store_failures_are_typed_and_do_not_send_generic():
    generic = {"$id": "generic", "userId": "u", "source": "calendar", "message": "generic"}
    for error, code in ((ReminderSchemaError("schema"), "MED_REMINDER_SCHEMA_UNAVAILABLE"), (ReminderStoreError("outage"), "MED_REMINDER_STORE_UNAVAILABLE")):
        firebase = Firebase()
        with pytest.raises(HTTPException) as exc:
            call(Store(due=[generic], error=error), firebase, enabled=True)
        assert exc.value.status_code == 503
        assert exc.value.detail == code
        assert not firebase.calls


def test_firebase_unavailable_is_typed_without_legacy_fallback():
    with pytest.raises(HTTPException) as exc:
        call(Store(), Firebase(ready=False), enabled=True)
    assert exc.value.detail == "MED_REMINDER_FIREBASE_UNAVAILABLE"


def test_invalid_dispatch_secret_is_rejected():
    with patch.dict("os.environ", {"NOTIFICATIONS_DISPATCH_SECRET": "secret"}, clear=False):
        with pytest.raises(HTTPException) as exc:
            notifications.dispatch_due(request("wrong"))
    assert exc.value.status_code == 401


def test_async_explicitly_rejects_durable_mode():
    with patch.dict("os.environ", {"NOTIFICATIONS_DISPATCH_SECRET": "secret", "ENABLE_DURABLE_MED_REMINDERS": "true"}, clear=False):
        with pytest.raises(HTTPException) as exc:
            notifications.dispatch_due_async(request())
    assert exc.value.detail == "MED_REMINDER_ASYNC_UNAVAILABLE"


def test_worker_explicitly_rejects_durable_mode(monkeypatch):
    import worker

    monkeypatch.setenv("ENABLE_DURABLE_MED_REMINDERS", "true")
    with pytest.raises(RuntimeError, match="MED_REMINDER_ASYNC_UNAVAILABLE"):
        worker.dispatch_due_reminders_task.run(window_seconds=60)
