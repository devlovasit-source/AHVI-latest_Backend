import asyncio

import pytest

import services.module_chat_service as mcs
import services.calendar_service as cs


@pytest.fixture
def fake_create(monkeypatch):
    created = []

    def _create(user_id, payload):
        event = {
            "$id": "evt-1",
            "id": "evt-1",
            "title": payload.get("title"),
            "start_time": payload.get("start_time"),
            "user_id": user_id,
        }
        created.append(event)
        return event

    monkeypatch.setattr(cs, "create_calendar_event", _create)
    return created


def test_detection_direct_one_turn_reminder_wording():
    assert mcs._looks_like_event_create(
        "set a reminder for tomorrow as I need to go for shopping at 5pm"
    ) is True


@pytest.mark.parametrize(
    "phrase",
    [
        "add shopping tomorrow at 5pm",
        "schedule shopping at 5pm tomorrow",
        "remind me to go shopping tomorrow at 5pm",
        "shopping at 5pm tomorrow",
    ],
)
def test_detection_creation_phrases(phrase):
    assert mcs._looks_like_event_create(phrase) is True


def test_detection_bare_time_alone_is_not_create():
    assert mcs._looks_like_event_create("shopping at 5pm") is False


def test_strip_occasion_prefix_removes_frontend_decoration():
    assert (
        mcs._strip_occasion_prefix("Occasion: Shopping\n\nshopping at 5pm")
        == "shopping at 5pm"
    )


def test_assemble_recovers_date_from_history():
    ctx = {"history": [{"role": "user", "content": "set a reminder for tomorrow"}]}
    assert "tomorrow" in mcs._assemble_event_text("shopping at 5pm", ctx)


def test_create_one_turn(fake_create):
    resp = asyncio.run(
        mcs.handle_calendar_chat(
            "set a reminder for tomorrow as I need to go for shopping at 5pm",
            {},
            "u1",
        )
    )
    assert resp.get("intent") == "calendar_event_created"
    assert len(fake_create) == 1


def test_create_multiturn_retains_date_from_history(fake_create):
    ctx = {"history": [{"role": "user", "content": "set a reminder for tomorrow"}]}
    resp = asyncio.run(mcs.handle_calendar_chat("shopping at 5pm", ctx, "u1"))
    assert resp.get("intent") == "calendar_event_created"
    assert len(fake_create) == 1


def test_create_ignores_occasion_prefix(fake_create):
    resp = asyncio.run(
        mcs.handle_calendar_chat(
            "Occasion: Shopping\n\nadd shopping tomorrow at 5pm", {}, "u1"
        )
    )
    assert resp.get("intent") == "calendar_event_created"


def test_incomplete_request_does_not_create(fake_create):
    resp = asyncio.run(mcs.handle_calendar_chat("i want to plan something", {}, "u1"))
    assert resp.get("intent") != "calendar_event_created"
    assert fake_create == []


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


@pytest.mark.parametrize(
    "text,expected",
    [
        ("set a reminder for tomorrow as I need to go to shopping at 5pm", "Shopping"),
        ("gym at 6am tomorrow", "Gym"),
        ("client meeting at 9pm", "Client Meeting"),
        ("birthday dinner tomorrow at 8pm", "Birthday Dinner"),
        ("add shopping tomorrow at 5pm", "Shopping"),
    ],
)
def test_clean_title_extraction(text, expected):
    payload = cs.parse_plan_text_to_payload(text)
    assert payload["title"] == expected


def test_doctor_appointment_title_preserved():
    payload = cs.parse_plan_text_to_payload("doctor appointment tomorrow at 9am")
    assert payload["title"].lower() == "doctor appointment"


def test_same_request_twice_creates_one_event(calendar_store):
    msg = "set a reminder for tomorrow as I need to go to shopping at 5pm"
    r1 = asyncio.run(mcs.handle_calendar_chat(msg, {}, "u1"))
    r2 = asyncio.run(mcs.handle_calendar_chat(msg, {}, "u1"))
    assert r1.get("intent") == "calendar_event_created"
    assert r2.get("intent") == "calendar_event_reused"
    assert len(calendar_store) == 1


def test_equivalent_phrasing_is_deduplicated(calendar_store):
    asyncio.run(mcs.handle_calendar_chat("add shopping tomorrow at 5pm", {}, "u1"))
    r2 = asyncio.run(mcs.handle_calendar_chat("schedule shopping at 5pm tomorrow", {}, "u1"))
    assert r2.get("intent") == "calendar_event_reused"
    assert len(calendar_store) == 1


def test_different_times_create_separate_events(calendar_store):
    asyncio.run(mcs.handle_calendar_chat("shopping tomorrow at 5pm", {}, "u1"))
    asyncio.run(mcs.handle_calendar_chat("shopping tomorrow at 7pm", {}, "u1"))
    assert len(calendar_store) == 2


def test_different_users_are_not_deduplicated(calendar_store):
    asyncio.run(mcs.handle_calendar_chat("shopping tomorrow at 5pm", {}, "u1"))
    asyncio.run(mcs.handle_calendar_chat("shopping tomorrow at 5pm", {}, "u2"))
    assert len(calendar_store) == 2


def test_multiturn_creation_is_idempotent(calendar_store):
    ctx = {"history": [{"role": "user", "content": "set a reminder for tomorrow"}]}
    asyncio.run(mcs.handle_calendar_chat("shopping at 5pm", ctx, "u1"))
    r2 = asyncio.run(mcs.handle_calendar_chat("shopping at 5pm", ctx, "u1"))
    assert r2.get("intent") == "calendar_event_reused"
    assert len(calendar_store) == 1
