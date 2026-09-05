"""Unit tests for Phase 4 Multi-Module Adapters and Fault Isolation."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from brain.temporal.adapters.base_adapter import TimelineSourceAdapter
from brain.temporal.adapters.bill_adapter import bill_adapter
from brain.temporal.adapters.calendar_adapter import calendar_adapter
from brain.temporal.adapters.meal_adapter import meal_adapter
from brain.temporal.adapters.medication_adapter import medication_adapter
from brain.temporal.adapters.skincare_adapter import skincare_adapter
from brain.temporal.adapters.workout_adapter import workout_adapter
from brain.temporal.aggregation_service import TimelineAggregationService, aggregation_service
from brain.temporal.models import TimelineItem, TimelineSourceType


class CrashingAdapter(TimelineSourceAdapter):
    """Failing adapter stub for testing fault isolation."""

    @property
    def source_type(self) -> TimelineSourceType:
        return TimelineSourceType.CALENDAR

    def fetch_raw(self, user_id: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        raise RuntimeError("Simulated database failure in adapter")

    def validate(self, raw_item: Dict[str, Any]) -> bool:
        return False

    def normalize(self, raw_item: Dict[str, Any], user_id: str) -> Optional[TimelineItem]:
        return None


def test_individual_adapters_normalization() -> None:
    now = datetime.now(timezone.utc)
    user_id = "usr_adapter_test"

    # Workout
    w_item = workout_adapter.normalize({"id": "w1", "title": "HIIT Session", "scheduled_at": now.isoformat()}, user_id)
    assert w_item is not None
    assert w_item.source == TimelineSourceType.WORKOUT
    assert w_item.type == "fitness"

    # Meal
    m_item = meal_adapter.normalize({"id": "m1", "title": "Protein Lunch", "scheduled_at": now.isoformat()}, user_id)
    assert m_item is not None
    assert m_item.source == TimelineSourceType.MEAL

    # Medication
    med_item = medication_adapter.normalize({"id": "med1", "name": "Vitamin D", "scheduled_at": now.isoformat()}, user_id)
    assert med_item is not None
    assert med_item.source == TimelineSourceType.MEDICATION
    assert med_item.priority == 4

    # Bill
    b_item = bill_adapter.normalize({"id": "b1", "title": "Electric Bill", "due_date": now.isoformat()}, user_id)
    assert b_item is not None
    assert b_item.source == TimelineSourceType.BILL

    # Skincare
    s_item = skincare_adapter.normalize({"id": "s1", "routine_name": "Evening Routine", "scheduled_at": now.isoformat()}, user_id)
    assert s_item is not None
    assert s_item.source == TimelineSourceType.SKINCARE


def test_aggregation_service_fault_isolation() -> None:
    user_id = "usr_fault_test"
    crashing = CrashingAdapter()
    
    service = TimelineAggregationService(adapters=[crashing, workout_adapter])

    # Should not throw exception even though crashing adapter raises RuntimeError
    items = service.fetch_unified_timeline(user_id)
    assert isinstance(items, list)
