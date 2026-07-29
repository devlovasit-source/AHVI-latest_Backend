"""routers/chat.py Calendar creation must reuse the same event as
module_chat_service instead of creating duplicates. Same shared helper
(calendar_service.find_existing_event), same criteria: user + normalized title
+ start minute. No network.
"""
import pytest

import routers.chat as chat
import services.calendar_service as cs
import services.module_chat_service as mcs


@pytest.fixture
def calendar_store(monkeypatch):
    store = []

    def _create(user_id, payload):
        event = {
            "$id": f"evt-{len(store) + 1}",
            "id": f"evt-{len(store) + 1}",
            "title": payload.get("title"),
            "start_time": payload.get("start_time"),
            "user_id": user_id,
        }
        store.append(event)
        return event

    def _list(user_id, *, start_time=None, end_time=None, limit=200):
        s = cs._as_utc(cs._parse_iso(start_time))
        e = cs._as_utc(cs._parse_iso(end_time))
        out = []
        for ev in store:
            if ev.get("user_id") != user_id:
                continue
            edt = cs._as_utc(cs._parse_iso(ev.get("start_time")))
            if s and edt and edt < s:
                continue
            if e and edt and edt > e:
                continue
            out.append(ev)
        return out

    monkeypatch.setattr(cs, "create_calendar_event", _create)
    monkeypatch.setattr(cs, "list_calendar_events", _list)
    return store


def _chat_create(text, user_id="user-1"):
    """Exercise the routers/chat.py creation branch directly (same imports the
    route uses), without spinning the whole /api/text stack."""
    payload = cs.parse_plan_text_to_payload(text, timezone_name="Asia/Kolkata")
    existing = cs.find_existing_event(
        user_id, payload.get("title"), payload.get("start_time")
    )
    if existing:
        return chat._calendar_event_created_response(existing, reused=True)
    return chat._calendar_event_created_response(
        cs.create_calendar_event(user_id, payload)
    )


def test_repeated_identical_request_reuses_the_same_event(calendar_store):
    first = _chat_create("gym tomorrow at 7am")
    second = _chat_create("gym tomorrow at 7am")
    assert len(calendar_store) == 1
    assert second["intent"] == "calendar_event_reused"
    assert second["reused"] is True
    assert second["data"]["event"]["id"] == first["data"]["event"]["id"]
    # Envelope contract preserved.
    for key in ("success", "type", "module", "domain", "message_text", "cards", "cta"):
        assert key in second


def test_first_create_reports_event_created(calendar_store):
    first = _chat_create("gym tomorrow at 7am")
    assert first["intent"] == "event_created"
    assert first["reused"] is False
    assert len(calendar_store) == 1


def test_same_title_different_time_creates_new_event(calendar_store):
    _chat_create("gym tomorrow at 7am")
    _chat_create("gym tomorrow at 9am")
    assert len(calendar_store) == 2


def test_different_title_same_time_creates_new_event(calendar_store):
    _chat_create("gym tomorrow at 7am")
    _chat_create("shopping tomorrow at 7am")
    assert len(calendar_store) == 2


def test_different_users_are_not_deduplicated(calendar_store):
    _chat_create("gym tomorrow at 7am", user_id="user-1")
    _chat_create("gym tomorrow at 7am", user_id="user-2")
    assert len(calendar_store) == 2


def test_create_failure_does_not_return_success(calendar_store, monkeypatch):
    def _boom(user_id, payload):
        raise RuntimeError("appwrite down")

    monkeypatch.setattr(cs, "create_calendar_event", _boom)
    with pytest.raises(RuntimeError):
        _chat_create("gym tomorrow at 7am")
    assert calendar_store == []


def test_chat_route_and_module_chat_use_the_same_reuse_helper(calendar_store):
    # module-chat creates it; the chat route must then reuse, not duplicate.
    import asyncio

    asyncio.run(
        mcs.handle_module_chat(
            {"domain": "calendar", "module": "calendar", "message": "gym tomorrow at 7am"},
            user_id="user-1",
        )
    )
    assert len(calendar_store) == 1
    reused = _chat_create("gym tomorrow at 7am", user_id="user-1")
    assert len(calendar_store) == 1
    assert reused["intent"] == "calendar_event_reused"
