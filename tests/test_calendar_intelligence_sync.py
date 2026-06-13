"""Sync calendar intelligence wiring (no Celery/Redis).

Covers:
- event creation survives when runtime fails (enrich returns {})
- meeting -> prep checklist + reminders scheduled
- flight -> packing list + buffer plan
- wedding -> outfit prompt
- reminders persisted with status=scheduled (no Celery/Redis)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from services import calendar_service as cs
from services import notification_store as ns


def _future_iso(hours: int = 6) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _event(title: str, **extra) -> dict:
    base = {
        "id": "evt-1",
        "title": title,
        "start_time": _future_iso(),
        "metadata": "{}",
    }
    base.update(extra)
    return base


class _CaptureProxy:
    """Records update_document calls; no network."""

    calls: list = []

    def update_document(self, resource, doc_id, data):
        _CaptureProxy.calls.append((resource, doc_id, data))
        return {"$id": doc_id, **data}


def _wire(monkeypatch, *, capture_reminders=None):
    monkeypatch.setenv("ENABLE_CALENDAR_INTELLIGENCE_SYNC", "true")
    _CaptureProxy.calls = []
    monkeypatch.setattr(cs, "AppwriteProxy", _CaptureProxy)

    def _fake_schedule(*, user_id, event_id, reminders, source="calendar"):
        if capture_reminders is not None:
            capture_reminders.extend(reminders)
        return {"success": True, "scheduled": len(reminders)}

    monkeypatch.setattr(ns.notification_store, "schedule_reminders", _fake_schedule)


# ---------- never breaks event creation ----------

def test_enrich_returns_empty_when_flag_off(monkeypatch):
    monkeypatch.delenv("ENABLE_CALENDAR_INTELLIGENCE_SYNC", raising=False)
    assert cs.enrich_event_with_intelligence("u1", _event("Client Meeting")) == {}


def test_enrich_never_raises_when_runtime_fails(monkeypatch):
    _wire(monkeypatch)

    import brain.engines.calendar_runtime as cr

    def _boom(*_a, **_k):
        raise RuntimeError("runtime exploded")

    monkeypatch.setattr(cr, "run_calendar_runtime", _boom)

    # Must swallow the error and return {} — event creation stays intact.
    assert cs.enrich_event_with_intelligence("u1", _event("Client Meeting")) == {}


# ---------- meeting: prep checklist + reminders ----------

def test_meeting_creates_checklist_and_schedules_reminders(monkeypatch):
    captured: list = []
    _wire(monkeypatch, capture_reminders=captured)

    extras = cs.enrich_event_with_intelligence("u1", _event("Client Meeting"))

    assert extras["event_group"] == "work"
    checklist = extras["checklist_bundle"]
    assert checklist and checklist.get("prepTonight")
    assert extras["reminders"], "meeting must produce reminders"
    # All scheduled reminders carry status=scheduled (no Celery needed).
    assert captured and all(r["status"] == "scheduled" for r in captured)


# ---------- flight: packing list + buffer plan ----------

def test_flight_creates_packing_and_buffer(monkeypatch):
    _wire(monkeypatch)

    extras = cs.enrich_event_with_intelligence("u1", _event("Flight to Delhi"))

    assert extras["event_group"] == "travel"
    assert "tickets" in extras["packing_list"]
    assert extras["buffer_plan"] is not None
    assert extras["buffer_plan"].get("leaveByISO")


# ---------- wedding: outfit prompt ----------

def test_wedding_creates_outfit_prompt(monkeypatch):
    _wire(monkeypatch)

    extras = cs.enrich_event_with_intelligence("u1", _event("Cousin's Wedding"))

    assert extras["outfit_prompt"] is not None
    assert extras["plan_outfit"] is not None
    assert extras["plan_outfit"]["status"] == "pending"
    assert extras["plan_outfit"]["occasion"] == "wedding"


# ---------- store: default status scheduled ----------

def test_schedule_reminders_defaults_status_scheduled(monkeypatch):
    captured: list = []

    class _Recorder:
        def update_document(self, resource, doc_id, data):
            raise Exception("force create path")

        def create_document(self, resource, data, document_id=None):
            captured.append(data)
            return {"$id": document_id, **data}

    store = ns.NotificationStore()
    monkeypatch.setattr(store, "_appwrite", _Recorder())

    store.schedule_reminders(
        user_id="u1",
        event_id="evt-1",
        reminders=[{"sendAtISO": _future_iso(), "message": "Lay out outfit tonight"}],
    )

    assert captured and captured[0]["status"] == "scheduled"
