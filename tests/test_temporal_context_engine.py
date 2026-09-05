"""Unit tests for TemporalContextEngine and ShadowModeEvaluator (Phase 2)."""

from datetime import datetime, timedelta, timezone
from brain.temporal.context_engine import context_engine
from brain.temporal.models import Flexibility, TimelineItem, TimelineItemStatus, TimelineSourceType
from brain.temporal.shadow_mode import is_shadow_mode_enabled, shadow_evaluator


def test_get_current_activity() -> None:
    now = datetime.now(timezone.utc)
    item_past = TimelineItem(
        id="item_past",
        user_id="u1",
        source=TimelineSourceType.CALENDAR,
        source_id="c_past",
        title="Past Meeting",
        start_time=now - timedelta(hours=2),
        end_time=now - timedelta(hours=1),
        priority=3,
    )
    item_active = TimelineItem(
        id="item_active",
        user_id="u1",
        source=TimelineSourceType.CALENDAR,
        source_id="c_active",
        title="Current Workshop",
        start_time=now - timedelta(minutes=15),
        end_time=now + timedelta(minutes=45),
        priority=4,
    )

    items = [item_past, item_active]
    current = context_engine.get_current_activity("u1", timestamp=now, items=items)
    assert current is not None
    assert current.id == "item_active"
    assert current.title == "Current Workshop"


def test_get_upcoming_activities() -> None:
    now = datetime.now(timezone.utc)
    item1 = TimelineItem(
        id="it1",
        user_id="u1",
        source=TimelineSourceType.CALENDAR,
        source_id="c1",
        title="Event 1",
        start_time=now + timedelta(hours=2),
        end_time=now + timedelta(hours=3),
        priority=3,
    )
    item2 = TimelineItem(
        id="it2",
        user_id="u1",
        source=TimelineSourceType.WORKOUT,
        source_id="w1",
        title="Workout Session",
        start_time=now + timedelta(hours=5),
        end_time=now + timedelta(hours=6),
        priority=2,
    )

    upcoming = context_engine.get_upcoming_activities("u1", window_hours=12.0, start_time=now, items=[item1, item2])
    assert len(upcoming) == 2
    assert upcoming[0].id == "it1"
    assert upcoming[1].id == "it2"


def test_calculate_free_busy_windows() -> None:
    now = datetime.now(timezone.utc)
    s_bound = now
    e_bound = now + timedelta(hours=4)

    # Busy block from +1h to +2h
    busy_item = TimelineItem(
        id="it_busy",
        user_id="u1",
        source=TimelineSourceType.CALENDAR,
        source_id="cb1",
        title="Busy Meeting",
        start_time=now + timedelta(hours=1),
        end_time=now + timedelta(hours=2),
        priority=3,
    )

    result = context_engine.calculate_free_busy_windows("u1", s_bound, e_bound, items=[busy_item])
    assert len(result["busy_blocks"]) == 1
    assert len(result["free_blocks"]) == 2
    assert result["total_free_minutes"] == 180.0  # 3 hours free out of 4


def test_detect_schedule_conflicts() -> None:
    now = datetime.now(timezone.utc)
    item_a = TimelineItem(
        id="it_a",
        user_id="u1",
        source=TimelineSourceType.CALENDAR,
        source_id="ca",
        title="Team Standup",
        start_time=now + timedelta(hours=1),
        end_time=now + timedelta(hours=2),
        priority=3,
    )
    item_b = TimelineItem(
        id="it_b",
        user_id="u1",
        source=TimelineSourceType.WORKOUT,
        source_id="wb",
        title="Gym Session",
        start_time=now + timedelta(hours=1, minutes=30),
        end_time=now + timedelta(hours=2, minutes=30),
        priority=2,
    )

    conflicts = context_engine.detect_schedule_conflicts([item_a, item_b])
    assert len(conflicts) == 1
    assert conflicts[0]["item_a_id"] == "it_a"
    assert conflicts[0]["item_b_id"] == "it_b"
    assert conflicts[0]["overlap_minutes"] == 30.0


def test_calculate_preparation_lead_time() -> None:
    now = datetime.now(timezone.utc)
    start = now + timedelta(hours=2)
    prep_start = now + timedelta(hours=1)

    item = TimelineItem(
        id="it_prep",
        user_id="u1",
        source=TimelineSourceType.CALENDAR,
        source_id="cp",
        title="Client Presentation",
        start_time=start,
        end_time=start + timedelta(hours=1),
        preparation_required=True,
        preparation_start=prep_start,
    )

    lead_info = context_engine.calculate_preparation_lead_time(item, current_time=now + timedelta(minutes=75))
    assert lead_info["preparation_required"] is True
    assert lead_info["preparation_lead_minutes"] == 60.0
    assert lead_info["is_in_prep_window"] is True
    assert lead_info["prep_window_open"] is True


def test_shadow_mode_evaluator_parity_gate() -> None:
    raw_events = [
        {
            "eventId": "evt_sm_1",
            "title": "Quarterly Business Review presentation",
            "startAtISO": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "endAtISO": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat(),
        }
    ]

    res = shadow_evaluator.evaluate_events(raw_events, user_id="usr_shadow_test")
    assert res["total_events"] == 1
    assert res["parity_passed"] is True
    assert res["diff_count"] == 0
