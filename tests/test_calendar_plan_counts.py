from datetime import datetime, timezone

from services.calendar_service import calendar_plan_counts


def test_calendar_plan_counts_separates_total_today_upcoming_and_non_outfit():
    now = datetime(2026, 8, 9, 8, 0, tzinfo=timezone.utc)
    events = [
        {
            "type": "plan",
            "start_time": "2026-08-09T18:00:00+05:30",
            "metadata": '{"plan_kind":"outfit"}',
        },
        {
            "type": "plan",
            "start_time": "2026-08-10T12:00:00+05:30",
            "metadata": {"plan_kind": "outfit"},
        },
        {
            "type": "meeting",
            "start_time": "2026-08-09T14:00:00+05:30",
            "metadata": {"plan_kind": "non_outfit"},
        },
        {
            "type": "plan",
            "start_time": "2026-08-01T12:00:00+05:30",
            "metadata": {"plan_kind": "outfit"},
        },
    ]

    assert calendar_plan_counts(events, now=now) == {
        "total_outfit_plan_count": 3,
        "today_outfit_plan_count": 1,
        "upcoming_outfit_plan_count": 2,
    }
