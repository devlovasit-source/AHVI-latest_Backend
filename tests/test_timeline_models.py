"""Unit tests for TimelineItem data model and enums."""

from datetime import datetime, timedelta, timezone
from brain.temporal.models import Flexibility, TimelineItem, TimelineItemStatus, TimelineSourceType


def test_timeline_item_schema_and_properties() -> None:
    now = datetime.now(timezone.utc)
    prep = now - timedelta(hours=2)
    end = now + timedelta(hours=1)

    item = TimelineItem(
        id="cal_123",
        user_id="usr_abc",
        source=TimelineSourceType.CALENDAR,
        source_id="evt_123",
        type="meeting",
        subtype="presentation",
        title="Client Pitch Meeting",
        start_time=now,
        end_time=end,
        priority=4,
        status=TimelineItemStatus.SCHEDULED,
        flexibility=Flexibility.FIXED,
        preparation_required=True,
        preparation_start=prep,
        metadata={"location": "Conference Room 1"},
    )

    assert item.id == "cal_123"
    assert item.user_id == "usr_abc"
    assert item.source == TimelineSourceType.CALENDAR
    assert item.source_id == "evt_123"
    assert item.flexibility == Flexibility.FIXED
    assert item.preparation_required is True
    assert abs(item.duration_minutes - 60.0) < 0.1
    assert abs(item.preparation_lead_minutes - 120.0) < 0.1
