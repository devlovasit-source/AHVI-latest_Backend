from __future__ import annotations

import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from main import app
from services.home_summary_service import generate_home_summary
from services.adherence_read_service import (
    get_today_skincare_state_readonly,
    get_today_medicine_state_readonly,
)

client = TestClient(app)

def test_unauthenticated_home_summary():
    """Unauthenticated requests to Home Summary are rejected."""
    response = client.get("/api/home/today-summary")
    assert response.status_code in {401, 403}

@pytest.mark.anyio
async def test_home_summary_returns_all_five_keys():
    """Authenticated Home Summary returns all five stable card keys with correct shape."""
    with patch("services.home_summary_service.get_today_medicine_state_readonly") as mock_med, \
         patch("services.home_summary_service.get_today_skincare_state_readonly") as mock_skin, \
         patch("services.home_summary_service.get_today_workout_card") as mock_workout, \
         patch("services.home_summary_service.resolve_location_weather_context") as mock_weather:

        mock_med.return_value = {
            "status": "ready",
            "source": "med_logs",
            "entity_id": "med_log_123",
            "projected": False,
            "overdue_count": 0,
            "due_now_count": 1,
            "due_soon_count": 0,
            "later_count": 0,
            "completed_count": 0,
            "next_due_at": "2026-07-29T10:00:00"
        }
        mock_skin.return_value = {
            "status": "ready",
            "source": "skincare_logs",
            "entity_id": "skin_log_456",
            "projected": False,
            "routine": "morning",
            "scheduled_at": "2026-07-29T08:00:00",
            "completed_steps": [],
            "total_steps": 3
        }
        mock_workout.return_value = {
            "id": "workout_abc",
            "title": "12-minute mobility session",
            "why_this": "A focused session."
        }
        mock_weather.return_value = {
            "weather": {"status": "available", "temperature": 25, "condition": "sunny"},
            "location": {"status": "available", "source": "profile"},
            "profile": {}
        }

        summary = await generate_home_summary(
            user_id="user_abc",
            user_profile={},
            timezone_override="Asia/Kolkata"
        )

        assert "wear" in summary["cards"]
        assert "move" in summary["cards"]
        assert "eat" in summary["cards"]
        assert "care" in summary["cards"]
        assert "medicine" in summary["cards"]

        assert summary["cards"]["wear"]["action"] == "open_daily_wear"
        assert summary["cards"]["move"]["action"] == "open_fitness"
        assert summary["cards"]["eat"]["action"] == "open_diet"
        assert summary["cards"]["care"]["action"] == "open_skincare"
        assert summary["cards"]["medicine"]["action"] == "open_medi"

@pytest.mark.anyio
async def test_card_isolation_and_partial_failure():
    """One card failure preserves the other four, capturing typed reason."""
    with patch("services.home_summary_service.get_today_medicine_state_readonly") as mock_med, \
         patch("services.home_summary_service.get_today_skincare_state_readonly") as mock_skin, \
         patch("services.home_summary_service.get_today_workout_card") as mock_workout, \
         patch("services.home_summary_service.get_diet_recommendation") as mock_diet, \
         patch("services.home_summary_service.resolve_location_weather_context") as mock_weather:

        mock_med.return_value = {"status": "ready", "logs": []}
        mock_skin.return_value = {"status": "ready", "logs": []}
        mock_workout.side_effect = Exception("Workout Service Crash!")
        mock_diet.return_value = {"status": "ready", "recommendation": {"id": "diet_1", "title": "Oats", "meal_type": "lunch"}}
        mock_weather.return_value = {"weather": {}, "location": {}}

        summary = await generate_home_summary("user_abc", {})

        assert summary["cards"]["move"]["status"] == "unavailable"
        assert summary["cards"]["move"]["reason"] == "workout_service_error"
        assert summary["cards"]["eat"]["status"] == "ready"
        assert summary["cards"]["eat"]["entity_id"] == "diet_1"

@pytest.mark.anyio
async def test_all_five_cards_independent_unavailable():
    """All five cards can independently become unavailable."""
    with patch("services.home_summary_service.get_today_medicine_state_readonly") as mock_med, \
         patch("services.home_summary_service.get_today_skincare_state_readonly") as mock_skin, \
         patch("services.home_summary_service.get_today_workout_card") as mock_workout, \
         patch("services.home_summary_service.get_diet_recommendation") as mock_diet, \
         patch("services.home_summary_service.resolve_location_weather_context") as mock_weather:

        mock_med.return_value = {"status": "unavailable", "reason": "medicine_schedule_unavailable"}
        mock_skin.return_value = {"status": "unavailable", "reason": "skincare_routine_unavailable"}
        mock_workout.return_value = None
        mock_diet.return_value = {"status": "unavailable", "reason": "diet_constraints_unresolved"}
        mock_weather.return_value = {"weather": {"status": "unavailable"}}

        summary = await generate_home_summary("user_abc", {})

        assert summary["cards"]["wear"]["status"] == "unavailable"
        assert summary["cards"]["move"]["status"] == "unavailable"
        assert summary["cards"]["eat"]["status"] == "unavailable"
        assert summary["cards"]["care"]["status"] == "unavailable"
        assert summary["cards"]["medicine"]["status"] == "unavailable"

@pytest.mark.anyio
async def test_read_failures_never_produce_pending_projections():
    """Adherence read failures never produce pending projections."""
    mock_proxy = MagicMock()
    mock_proxy.list_documents.side_effect = Exception("Appwrite Query Failed!")

    local_now = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))

    med_state = get_today_medicine_state_readonly("user_abc", local_now, proxy_client=mock_proxy)
    assert med_state["status"] == "error"
    assert med_state["reason"] == "medicine_logs_read_failed"
    assert med_state["projected"] is False

    skin_state = get_today_skincare_state_readonly("user_abc", local_now, proxy_client=mock_proxy)
    assert skin_state["status"] == "error"
    assert skin_state["reason"] == "skincare_logs_read_failed"
    assert skin_state["projected"] is False

@pytest.mark.anyio
async def test_projections_have_null_entity_id():
    """Projections must always use entity_id = null."""
    mock_proxy = MagicMock()
    mock_proxy.list_documents.side_effect = lambda coll, **kwargs: (
        [] if coll == "med_logs" else [{"$id": "med_active_123", "userId": "user_abc", "active": True, "time": "12:00"}]
    )

    local_now = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))

    med_state = get_today_medicine_state_readonly("user_abc", local_now, proxy_client=mock_proxy)
    assert med_state["status"] == "later"
    assert med_state["projected"] is True
    assert med_state["entity_id"] is None

@pytest.mark.anyio
async def test_no_medicine_details_exposed():
    """Wording is privacy-safe: no medicine name, dosage, or sensitive details appear in the returned state."""
    mock_proxy = MagicMock()
    mock_proxy.list_documents.side_effect = lambda coll, **kwargs: (
        [] if coll == "med_logs" else [{
            "$id": "med_active_123",
            "userId": "user_abc",
            "active": True,
            "name": "5mg Amlodipine",
            "dose": "Take 1 pill daily for high blood pressure",
            "time": "12:00"
        }]
    )

    local_now = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))

    with patch("services.adherence_read_service.AppwriteProxy", return_value=mock_proxy):
        summary = await generate_home_summary("user_abc", {}, timezone_override="Asia/Kolkata")
        card_json = str(summary["cards"]["medicine"])
        assert "Amlodipine" not in card_json
        assert "blood pressure" not in card_json
        assert "pill" not in card_json

@pytest.mark.anyio
async def test_no_fabricated_times_or_steps():
    """Active meds with invalid or missing times, or skincare with empty/missing steps return unavailable rather than fabricated defaults."""
    mock_proxy = MagicMock()
    mock_proxy.list_documents.side_effect = lambda coll, **kwargs: (
        [] if coll in {"med_logs", "skincare_logs"} else (
            [{"$id": "med_1", "userId": "user_abc", "active": True}] if coll == "meds" else
            [{"$id": "skin_profile", "userId": "user_abc"}]
        )
    )

    local_now = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))

    med_state = get_today_medicine_state_readonly("user_abc", local_now, proxy_client=mock_proxy)
    assert med_state["status"] == "unavailable"
    assert med_state["reason"] == "medicine_schedule_unavailable"

    skin_state = get_today_skincare_state_readonly("user_abc", local_now, proxy_client=mock_proxy)
    assert skin_state["status"] == "unavailable"
    assert skin_state["reason"] == "skincare_routine_unavailable"

@pytest.mark.anyio
async def test_utc_timestamps_converted_to_local():
    """UTC timestamps are converted to canonical local date correctly."""
    mock_proxy = MagicMock()
    mock_proxy.list_documents.return_value = [
        {"$id": "med_log_1", "userId": "user_abc", "time": "2026-07-29T10:00:00Z", "status": "pending"}
    ]

    local_now = datetime(2026, 7, 29, 20, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

    med_state = get_today_medicine_state_readonly("user_abc", local_now, proxy_client=mock_proxy)
    assert med_state["status"] == "overdue"

@pytest.mark.anyio
async def test_invalid_times_return_unavailable():
    """Invalid schedule times return unavailable rather than 09:00 fallback."""
    mock_proxy = MagicMock()
    mock_proxy.list_documents.side_effect = lambda coll, **kwargs: (
        [] if coll == "med_logs" else [
            {"$id": "med_1", "userId": "user_abc", "active": True, "time": "invalid_time_string"}
        ]
    )

    local_now = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))

    med_state = get_today_medicine_state_readonly("user_abc", local_now, proxy_client=mock_proxy)
    assert med_state["status"] == "unavailable"
    assert med_state["reason"] == "medicine_schedule_unavailable"

@pytest.mark.anyio
async def test_zero_celsius_is_preserved():
    """0°C remains valid weather and doesn't fall back to general style suggestion."""
    with patch("services.home_summary_service.resolve_location_weather_context") as mock_weather:
        mock_weather.return_value = {
            "weather": {"status": "available", "temperature": 0.0, "condition": "snowy"},
            "location": {"status": "available", "source": "profile"},
            "profile": {}
        }

        summary = await generate_home_summary("user_abc", {})
        assert summary["cards"]["wear"]["status"] == "ready"
        assert "Cold 0.0°C" in summary["cards"]["wear"]["context"]

@pytest.mark.anyio
async def test_home_summary_performs_no_mutations_strengthened():
    """GET /api/home/today-summary performs absolutely no persistence, log writes, or reconciliation calls."""
    from services.appwrite_proxy import AppwriteProxy

    proxy_mock = MagicMock()
    proxy_mock.list_documents.return_value = []

    with patch("services.adherence_read_service.AppwriteProxy", return_value=proxy_mock), \
         patch("routers.workouts._persist_cards") as mock_persist, \
         patch("routers.workouts._persist_workout_outfit") as mock_outfit, \
         patch("brain.engines.adherence_engine.generate_daily_med_logs") as mock_med_gen, \
         patch("brain.engines.adherence_engine.generate_daily_skincare_logs") as mock_skin_gen, \
         patch("brain.engines.adherence_engine.auto_reconcile_overdue") as mock_reconcile_med, \
         patch("brain.engines.adherence_engine.auto_reconcile_skincare_overdue") as mock_reconcile_skin:

        summary = await generate_home_summary("user_abc", {})

        # Verify no database mutations
        assert proxy_mock.create_document.call_count == 0
        assert proxy_mock.update_document.call_count == 0
        assert proxy_mock.delete_document.call_count == 0

        # Verify no persistence or reconciliation calls
        assert mock_persist.call_count == 0
        assert mock_outfit.call_count == 0
        assert mock_med_gen.call_count == 0
        assert mock_skin_gen.call_count == 0
        assert mock_reconcile_med.call_count == 0
        assert mock_reconcile_skin.call_count == 0
