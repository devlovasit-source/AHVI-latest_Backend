from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeProxy:
    def __init__(self, meds):
        self.meds = meds

    def list_documents(self, resource, **kwargs):
        assert resource == "meds"
        return list(self.meds)


class FakeNotificationStore:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def schedule_reminders(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": self.ok, "scheduled": 1 if self.ok else 0}


async def _chat_async(monkeypatch, meds, message):
    from services import module_chat_service

    store = FakeNotificationStore()
    monkeypatch.setattr(module_chat_service, "AppwriteProxy", lambda: FakeProxy(meds))
    monkeypatch.setattr(module_chat_service, "notification_store", store)
    res = await module_chat_service.handle_module_chat(
        {"domain": "medi", "module": "medi", "message": message},
        user_id="user_1",
    )
    return res, store


def _chat(monkeypatch, meds, message):
    return asyncio.run(_chat_async(monkeypatch, meds, message))


def test_medicine_reminder_named_med_schedules(monkeypatch):
    res, store = _chat(
        monkeypatch,
        [{"$id": "med_1", "userId": "user_1", "name": "Dolo", "dose": "650"}],
        "remind me to take Dolo at 9 PM",
    )

    assert res["message"].startswith("Done - I will remind you to take Dolo at ")
    assert res["quick_actions"] == ["Open Medicines", "Set another reminder"]
    assert res["open_module"] == {"module": "medicines", "route": "/organize/medicines"}
    assert len(store.calls) == 1
    call = store.calls[0]
    reminder = call["reminders"][0]
    assert call["source"] == "medi"
    assert reminder["medId"] == "med_1"
    assert reminder["medName"] == "Dolo"
    assert reminder["dose"] == "650"
    assert reminder["title"] == "Medicine reminder"
    assert reminder["body"] == "Time to take Dolo"
    assert reminder["sendAtISO"] == reminder["scheduledFor"]
    assert reminder["notificationKey"].startswith("med:med_1:")


def test_medicine_reminder_single_med_without_name_schedules(monkeypatch):
    res, store = _chat(
        monkeypatch,
        [{"$id": "med_1", "userId": "user_1", "name": "Paracetamol"}],
        "remind me at 9 PM",
    )

    assert "Paracetamol" in res["message"]
    assert len(store.calls) == 1
    assert store.calls[0]["reminders"][0]["medId"] == "med_1"


def test_medicine_reminder_multiple_meds_without_name_asks_clarification(monkeypatch):
    res, store = _chat(
        monkeypatch,
        [
            {"$id": "med_1", "userId": "user_1", "name": "Dolo"},
            {"$id": "med_2", "userId": "user_1", "name": "Vitamin D"},
        ],
        "medicine reminder at 10 PM",
    )

    assert res["message"] == "Which medicine should I set the reminder for?"
    assert store.calls == []


def test_medicine_reminder_without_time_asks_for_time(monkeypatch):
    res, store = _chat(
        monkeypatch,
        [{"$id": "med_1", "userId": "user_1", "name": "Dolo"}],
        "remind me to take Dolo",
    )

    assert res["message"] == "What time should I remind you?"
    assert store.calls == []


def test_medicine_reminder_without_meds_asks_to_add_first(monkeypatch):
    res, store = _chat(monkeypatch, [], "remind me to take my tablet tomorrow morning")

    assert res["message"] == "Add the medicine in Medi Tracker first, then I can set a reminder for it."
    assert store.calls == []
