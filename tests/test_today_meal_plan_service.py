from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch

import pytest

from services.today_meal_plan_service import get_today_meal_plan
from services.diet_service import get_diet_recommendation
from services.home_summary_service import generate_home_summary

KOLKATA = ZoneInfo("Asia/Kolkata")


def _now(hour: int = 12) -> datetime:
    return datetime(2026, 7, 29, hour, 0, tzinfo=KOLKATA)


def _doc(*, doc_id, created, plan_type="daily", meals=None, user_id="user_abc", name="Plan"):
    return {
        "$id": doc_id,
        "$createdAt": created,
        "planType": plan_type,
        "name": name,
        "userId": user_id,
        "meals": meals if meals is not None else [],
    }


def _proxy(docs):
    proxy = MagicMock()
    proxy.list_documents.return_value = docs
    return proxy


# 1. Same-day daily plan returns ready.
def test_same_day_daily_returns_ready():
    proxy = _proxy([
        _doc(doc_id="plan_1", created="2026-07-29T06:00:00Z",
             meals=[{"name": "Dal Rice", "type": "lunch", "cal": 400}]),
    ])
    res = get_today_meal_plan(user_id="user_abc", local_now=_now(12), proxy_client=proxy)
    assert res["status"] == "ready"
    assert res["plan_id"] == "plan_1"
    assert res["meal"]["name"] == "Dal Rice"
    assert res["meal"]["type"] == "lunch"


# 2. Same-day JSON-string meal is parsed.
def test_same_day_json_string_meal_parsed():
    proxy = _proxy([
        _doc(doc_id="plan_1", created="2026-07-29T06:00:00Z",
             meals=[json.dumps({"name": "Poha", "type": "breakfast", "cal": 250})]),
    ])
    res = get_today_meal_plan(user_id="user_abc", local_now=_now(8), proxy_client=proxy)
    assert res["status"] == "ready"
    assert res["meal"]["name"] == "Poha"
    assert res["meal"]["type"] == "breakfast"


# 3. Latest same-day daily plan is selected.
def test_latest_same_day_daily_selected():
    proxy = _proxy([
        _doc(doc_id="old", created="2026-07-29T04:00:00Z",
             meals=[{"name": "Old Lunch", "type": "lunch"}]),
        _doc(doc_id="new", created="2026-07-29T09:00:00Z",
             meals=[{"name": "New Lunch", "type": "lunch"}]),
    ])
    res = get_today_meal_plan(user_id="user_abc", local_now=_now(12), proxy_client=proxy)
    assert res["plan_id"] == "new"
    assert res["meal"]["name"] == "New Lunch"


# 4. Same-day non-daily plan works when no daily plan exists.
def test_same_day_non_daily_used_when_no_daily():
    proxy = _proxy([
        _doc(doc_id="custom_1", created="2026-07-29T06:00:00Z", plan_type="custom",
             meals=[{"name": "Custom Lunch", "type": "lunch"}]),
    ])
    res = get_today_meal_plan(user_id="user_abc", local_now=_now(12), proxy_client=proxy)
    assert res["status"] == "ready"
    assert res["plan_id"] == "custom_1"


# 5. Previous-day plan is not shown.
def test_previous_day_plan_not_shown():
    proxy = _proxy([
        _doc(doc_id="yesterday", created="2026-07-28T06:00:00Z",
             meals=[{"name": "Stale", "type": "lunch"}]),
    ])
    res = get_today_meal_plan(user_id="user_abc", local_now=_now(12), proxy_client=proxy)
    assert res["status"] == "unavailable"
    assert res["reason"] == "today_meal_plan_missing"


# 6. Future-dated plan is not shown.
def test_future_dated_plan_not_shown():
    proxy = _proxy([
        _doc(doc_id="tomorrow", created="2026-07-30T18:00:00Z",
             meals=[{"name": "Future", "type": "lunch"}]),
    ])
    res = get_today_meal_plan(user_id="user_abc", local_now=_now(12), proxy_client=proxy)
    assert res["status"] == "unavailable"


# 7. $createdAt UTC is converted to Asia/Kolkata before date comparison.
def test_utc_createdat_converted_to_kolkata_before_compare():
    # 2026-07-28T20:00Z -> 2026-07-29 01:30 IST (counts as today).
    # 2026-07-29T19:00Z -> 2026-07-30 00:30 IST (excluded).
    proxy = _proxy([
        _doc(doc_id="ist_today", created="2026-07-28T20:00:00Z",
             meals=[{"name": "IST Lunch", "type": "lunch"}]),
        _doc(doc_id="ist_tomorrow", created="2026-07-29T19:00:00Z",
             meals=[{"name": "Tomorrow", "type": "lunch"}]),
    ])
    res = get_today_meal_plan(user_id="user_abc", local_now=_now(12), proxy_client=proxy)
    assert res["status"] == "ready"
    assert res["plan_id"] == "ist_today"


# 8. Malformed meal data fails safely.
def test_malformed_meal_data_fails_safely():
    proxy = _proxy([
        _doc(doc_id="bad", created="2026-07-29T06:00:00Z",
             meals=[{"no_name": 1}, None, "", {"name": ""}, {}]),
    ])
    res = get_today_meal_plan(user_id="user_abc", local_now=_now(12), proxy_client=proxy)
    # All entries nameless/empty -> ignored -> unavailable, no crash.
    assert res["status"] == "unavailable"


# 8b. Meals field of wrong type fails safely.
def test_meals_wrong_type_fails_safely():
    proxy = _proxy([
        _doc(doc_id="bad2", created="2026-07-29T06:00:00Z", meals="not-a-list"),
    ])
    res = get_today_meal_plan(user_id="user_abc", local_now=_now(12), proxy_client=proxy)
    assert res["status"] == "unavailable"


# 9. Another user's plan is never returned.
def test_other_users_plan_never_returned():
    proxy = _proxy([
        _doc(doc_id="foreign", created="2026-07-29T06:00:00Z", user_id="user_other",
             meals=[{"name": "Not Yours", "type": "lunch"}]),
    ])
    res = get_today_meal_plan(user_id="user_abc", local_now=_now(12), proxy_client=proxy)
    assert res["status"] == "unavailable"


# 10. No Appwrite write method is called.
def test_no_write_methods_called():
    proxy = _proxy([
        _doc(doc_id="plan_1", created="2026-07-29T06:00:00Z",
             meals=[{"name": "Dal Rice", "type": "lunch"}]),
    ])
    get_today_meal_plan(user_id="user_abc", local_now=_now(12), proxy_client=proxy)
    assert proxy.create_document.call_count == 0
    assert proxy.update_document.call_count == 0
    assert proxy.delete_document.call_count == 0
    assert proxy.create_document_async.call_count == 0
    assert proxy.update_document_async.call_count == 0
    assert proxy.delete_document_async.call_count == 0


# 11. Existing diet recommendation still works when a valid profile is supplied.
def test_existing_diet_recommendation_still_works():
    from datetime import date
    res = get_diet_recommendation(
        user_id="user_abc", local_date=date(2026, 7, 29), local_hour=12,
        meal_type="lunch", profile={"diet_type": "veg"},
    )
    assert res["status"] == "ready"
    assert res["recommendation"]["meal_type"] == "lunch"


# 12. Home summary returns source=meal_plan and entity_id=plan $id.
@pytest.mark.anyio
async def test_home_eat_card_uses_meal_plan_source_and_entity_id():
    plan_res = {
        "status": "ready",
        "reason": None,
        "plan_id": "plan_xyz",
        "plan_name": "Today's Plan",
        "meal": {"name": "Grilled Paneer Bowl", "type": "lunch", "cal": 450},
    }
    with patch("services.home_summary_service.get_today_meal_plan", return_value=plan_res), \
         patch("services.home_summary_service.get_today_workout_card", return_value=None), \
         patch("services.home_summary_service.resolve_location_weather_context",
               return_value={"weather": {"status": "unavailable"}, "location": {}}):
        summary = await generate_home_summary("user_abc", {}, timezone_override="Asia/Kolkata")

    eat = summary["cards"]["eat"]
    assert eat["source"] == "meal_plan"
    assert eat["entity_id"] == "plan_xyz"
    assert eat["status"] == "ready"
    assert eat["available"] is True
    assert eat["context"] == "Grilled Paneer Bowl"
    assert eat["headline"] == "Lunch from today's plan"
    assert eat["action"] == "open_diet"
