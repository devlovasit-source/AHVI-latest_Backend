"""End-to-end reconciliation: wear_event_service.record_wear must complete
only the exact user+item's wear reminders, only on a newly-committed event,
and must never fail the wear itself if reminder completion errors."""

from services.appwrite_proxy import AppwriteProxyError
from services.wear_event_service import record_wear
from services.wardrobe_reminder_service import create_wear_reminder, list_wear_reminders


class FakeWearEventsProxy:
    def __init__(self):
        self.docs = {}

    def create_document(self, resource, data, document_id="unique()"):
        if document_id in self.docs:
            raise AppwriteProxyError("already exists", status_code=409)
        doc = {"$id": document_id, **data}
        self.docs[document_id] = doc
        return doc

    def get_document(self, resource, document_id):
        return self.docs[document_id]

    def list_documents(self, resource, user_id=None, limit=500, **kwargs):
        return [d for d in self.docs.values() if d.get("userId") == user_id]


class FakeOutfitHistoryProxy:
    def list_documents(self, *a, **k):
        return []

    def create_document(self, *a, **k):
        return {}

    def update_document(self, *a, **k):
        return {}


class FakeReminderProxy:
    def __init__(self):
        self.docs = {}

    def update_document(self, resource, document_id, patch):
        if document_id not in self.docs:
            raise RuntimeError("not found")
        self.docs[document_id].update(patch)
        return self.docs[document_id]

    def create_document(self, resource, data, document_id=None):
        doc_id = document_id or f"doc{len(self.docs) + 1}"
        doc = {"$id": doc_id, **data}
        self.docs[doc_id] = doc
        return doc

    def get_document(self, resource, document_id):
        if document_id not in self.docs:
            raise RuntimeError("(404) not found")
        return self.docs[document_id]

    def list_documents(self, resource, user_id=None, limit=500, **kwargs):
        return [d for d in self.docs.values() if d.get("userId") == user_id]


def _wire(monkeypatch):
    import services.wear_event_service as wear_svc
    import services.style_memory_service as memory_svc
    import services.notification_store as ns

    fake_events = FakeWearEventsProxy()
    fake_history = FakeOutfitHistoryProxy()
    monkeypatch.setattr(wear_svc, "_proxy", lambda: fake_events)
    monkeypatch.setattr(memory_svc, "_proxy", lambda: fake_history)
    fake_reminders = FakeReminderProxy()
    monkeypatch.setattr(ns.notification_store, "_appwrite", fake_reminders)
    return fake_reminders


def test_wearing_item_completes_its_own_reminder_only(monkeypatch):
    _wire(monkeypatch)
    create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")
    create_wear_reminder(user_id="u1", item_id="jacket1", send_at_iso="2026-09-02T09:00:00+00:00")

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-30T09:00:00+00:00")

    assert list_wear_reminders(user_id="u1", item_id="shirt1") == []
    remaining = list_wear_reminders(user_id="u1", item_id="jacket1")
    assert len(remaining) == 1


def test_cross_user_wear_does_not_touch_other_users_reminder(monkeypatch):
    _wire(monkeypatch)
    create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")
    create_wear_reminder(user_id="u2", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-30T09:00:00+00:00")

    assert list_wear_reminders(user_id="u1", item_id="shirt1") == []
    remaining_u2 = list_wear_reminders(user_id="u2", item_id="shirt1")
    assert len(remaining_u2) == 1


def test_duplicate_wear_does_not_re_run_reminder_completion(monkeypatch):
    fake_reminders = _wire(monkeypatch)
    create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")

    reminder_id = list_wear_reminders(user_id="u1", item_id="shirt1")[0]["reminder_id"]

    record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-30T09:00:00+00:00")
    assert fake_reminders.docs[reminder_id]["status"] == "completed"

    # Retry the same logical wear (same local day) — newly_created is False,
    # so completion must not run again (nothing to observe changing, but the
    # call path itself must be skipped rather than harmlessly re-executed).
    completed_before = fake_reminders.docs[reminder_id]["updatedAtISO"]
    result = record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-30T18:00:00+00:00")

    assert result["newly_created"] is False
    assert fake_reminders.docs[reminder_id]["status"] == "completed"
    assert fake_reminders.docs[reminder_id]["updatedAtISO"] == completed_before


def test_reminder_completion_failure_does_not_break_the_wear(monkeypatch):
    import services.wear_event_service as wear_svc

    monkeypatch.setattr(wear_svc, "_proxy", lambda: FakeWearEventsProxy())
    import services.style_memory_service as memory_svc

    monkeypatch.setattr(memory_svc, "_proxy", lambda: FakeOutfitHistoryProxy())

    import services.wardrobe_reminder_service as reminder_svc

    def _boom(**kwargs):
        raise RuntimeError("appwrite reminders collection unreachable")

    monkeypatch.setattr(reminder_svc, "complete_reminders_for_item", _boom)

    result = record_wear(user_id="u1", item_id="shirt1", occurred_at_iso="2026-08-30T09:00:00+00:00")

    assert result["newly_created"] is True
