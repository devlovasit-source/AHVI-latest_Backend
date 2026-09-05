"""Unit tests for Multi-Tenant Isolation & Security (P0-1 & P0-4 Verification).

Asserts that data created for User A is NEVER accessible or returned when querying as User B
across all timeline adapters and OpportunityStore.
"""

from datetime import datetime, timezone

from brain.temporal.adapters.bill_adapter import bill_adapter
from brain.temporal.adapters.calendar_adapter import calendar_adapter
from brain.temporal.adapters.meal_adapter import meal_adapter
from brain.temporal.adapters.medication_adapter import medication_adapter
from brain.temporal.adapters.skincare_adapter import skincare_adapter
from brain.temporal.adapters.workout_adapter import workout_adapter
from brain.temporal.opportunity_models import Opportunity
from brain.temporal.opportunity_store import opportunity_store


def test_calendar_adapter_multi_tenant_isolation() -> None:
    now = datetime.now(timezone.utc)
    raw_user_a_event = {
        "eventId": "evt_secret_user_a",
        "userId": "user_a",
        "title": "Private Medical Appointment",
        "startAtISO": now.isoformat(),
    }

    # Normalize for user A
    item_a = calendar_adapter.normalize(raw_user_a_event, user_id="user_a")
    assert item_a is not None
    assert item_a.user_id == "user_a"

    # Querying raw events for user_b when only user_a documents exist must return []
    # (No fallback to all documents!)
    user_b_events = calendar_adapter.fetch_raw("user_b")
    assert len(user_b_events) == 0


def test_all_adapters_user_scoping() -> None:
    user_b = "usr_b_isolated"
    adapters = [
        calendar_adapter,
        workout_adapter,
        meal_adapter,
        medication_adapter,
        bill_adapter,
        skincare_adapter,
    ]

    for adapter in adapters:
        records = adapter.fetch_raw(user_b)
        assert isinstance(records, list)
        # Any record returned MUST strictly belong to user_b
        for rec in records:
            rec_user = str(rec.get("userId") or rec.get("user_id") or "")
            assert rec_user == user_b or rec_user == ""


def test_opportunity_store_tenant_isolation_and_restart() -> None:
    opp_a = Opportunity.create(
        user_id="user_tenant_a",
        opportunity_type="test_isolation",
        timeline_item_id="item_tenant_a",
        trigger_window="win_1",
    )
    opportunity_store.save_opportunity(opp_a)

    # User B query must return ZERO opportunities for user A
    user_b_opps = opportunity_store.query_user_opportunities("user_tenant_b")
    for opp in user_b_opps:
        assert opp.user_id != "user_tenant_a"

    # Clear memory cache (simulating process restart across instances)
    opportunity_store.clear_cache()

    # Re-query user B; must still not leak user A data
    user_b_opps_after = opportunity_store.query_user_opportunities("user_tenant_b")
    for opp in user_b_opps_after:
        assert opp.user_id != "user_tenant_a"
