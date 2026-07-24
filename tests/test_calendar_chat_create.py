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
