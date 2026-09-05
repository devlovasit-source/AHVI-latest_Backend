"""Unit and contract tests for CalendarTimelineAdapter."""

from datetime import datetime, timezone
from brain.temporal.adapters.calendar_adapter import CalendarTimelineAdapter
from brain.temporal.models import Flexibility, TimelineItemStatus, TimelineSourceType


def test_calendar_adapter_validation_and_normalization() -> None:
    adapter = CalendarTimelineAdapter()
    assert adapter.source_type == TimelineSourceType.CALENDAR

    raw_event = {
        "eventId": "evt_board_meeting",
        "title": "Client Board Meeting & Pitch",
        "startAtISO": "2026-09-05T14:00:00Z",
        "endAtISO": "2026-09-05T15:30:00Z",
        "location": "Boardroom A",
    }

    assert adapter.validate(raw_event) is True

    item = adapter.normalize(raw_event, user_id="usr_test_123")
    assert item is not None
    assert item.id == "cal_evt_board_meeting"
    assert item.user_id == "usr_test_123"
    assert item.source == TimelineSourceType.CALENDAR
    assert item.source_id == "evt_board_meeting"
    assert item.type == "work"
    assert item.title == "Client Board Meeting & Pitch"
    assert item.flexibility == Flexibility.FIXED
    assert item.preparation_required is True
    assert item.preparation_start is not None
    assert abs(item.duration_minutes - 90.0) < 0.1
