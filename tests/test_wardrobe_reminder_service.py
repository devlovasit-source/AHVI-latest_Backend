import pytest

from services.wardrobe_reminder_service import (
    cancel_wear_reminder,
    complete_reminders_for_item,
    create_wear_reminder,
    list_wear_reminders,
)


class FakeReminderProxy:
    """In-memory stand-in matching AppwriteProxy's contract as used by
    NotificationStore: update_document raises for a missing doc (so
    schedule_reminders falls back to create_document), create_document
    honors an explicit document_id, list_documents filters by user_id."""

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


@pytest.fixture
def fake_store(monkeypatch):
    import services.notification_store as ns

    fake = FakeReminderProxy()
    monkeypatch.setattr(ns.notification_store, "_appwrite", fake)
    return fake


def test_create_wear_reminder(fake_store):
    result = create_wear_reminder(
        user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00"
    )
    assert result["success"] is True
    reminders = list_wear_reminders(user_id="u1", item_id="shirt1")
    assert len(reminders) == 1
    assert reminders[0]["status"] == "scheduled"
    assert reminders[0]["item_id"] == "shirt1"


def test_create_wear_reminder_requires_send_at(fake_store):
    with pytest.raises(ValueError):
        create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="")


def test_list_only_returns_scheduled(fake_store):
    create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")
    reminders = list_wear_reminders(user_id="u1", item_id="shirt1")
    rid = reminders[0]["reminder_id"]
    cancel_wear_reminder(user_id="u1", item_id="shirt1", reminder_id=rid)
    assert list_wear_reminders(user_id="u1", item_id="shirt1") == []


def test_cancel_is_idempotent(fake_store):
    create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")
    rid = list_wear_reminders(user_id="u1", item_id="shirt1")[0]["reminder_id"]
    assert cancel_wear_reminder(user_id="u1", item_id="shirt1", reminder_id=rid) is True
    assert cancel_wear_reminder(user_id="u1", item_id="shirt1", reminder_id=rid) is True


def test_cancel_missing_reminder_is_idempotent_true(fake_store):
    assert cancel_wear_reminder(user_id="u1", item_id="shirt1", reminder_id="nope") is True


def test_cancel_wrong_user_raises(fake_store):
    create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")
    rid = list_wear_reminders(user_id="u1", item_id="shirt1")[0]["reminder_id"]
    with pytest.raises(PermissionError):
        cancel_wear_reminder(user_id="u2", item_id="shirt1", reminder_id=rid)


def test_cancel_wrong_item_raises(fake_store):
    create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")
    rid = list_wear_reminders(user_id="u1", item_id="shirt1")[0]["reminder_id"]
    with pytest.raises(PermissionError):
        cancel_wear_reminder(user_id="u1", item_id="pants1", reminder_id=rid)


def test_item_a_wear_completes_only_item_a_reminder(fake_store):
    create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")
    create_wear_reminder(user_id="u1", item_id="jacket1", send_at_iso="2026-09-02T09:00:00+00:00")

    completed = complete_reminders_for_item(user_id="u1", item_id="shirt1")

    assert completed == 1
    assert list_wear_reminders(user_id="u1", item_id="shirt1") == []
    remaining = list_wear_reminders(user_id="u1", item_id="jacket1")
    assert len(remaining) == 1
    assert remaining[0]["status"] == "scheduled"


def test_cross_user_reminders_isolated_on_completion(fake_store):
    create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")
    create_wear_reminder(user_id="u2", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")

    completed = complete_reminders_for_item(user_id="u1", item_id="shirt1")

    assert completed == 1
    assert list_wear_reminders(user_id="u1", item_id="shirt1") == []
    remaining_u2 = list_wear_reminders(user_id="u2", item_id="shirt1")
    assert len(remaining_u2) == 1
    assert remaining_u2[0]["status"] == "scheduled"


def test_duplicate_completion_call_is_a_noop_second_time(fake_store):
    create_wear_reminder(user_id="u1", item_id="shirt1", send_at_iso="2026-09-01T09:00:00+00:00")

    first = complete_reminders_for_item(user_id="u1", item_id="shirt1")
    second = complete_reminders_for_item(user_id="u1", item_id="shirt1")

    assert first == 1
    assert second == 0  # already completed — nothing left in "scheduled" state
