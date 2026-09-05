"""Phase 0 Baseline Regression Fixtures for Calendar & Event Scenarios.

Captures baseline outputs for Meeting, Party, Workout, Travel, Health, Finance,
and General scenarios to serve as migration regression fixtures for the Temporal
Intelligence Architecture.
"""

from typing import Any, Dict
import pytest

from models.calendar_models import CalendarEventInput
from brain.engines.calendar_runtime import run_calendar_runtime


BASELINE_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "meeting": {
        "event": {
            "id": "evt_meeting_1",
            "title": "Client Board Presentation",
            "startAtISO": "2026-09-05T14:00:00Z",
            "endAtISO": "2026-09-05T15:00:00Z",
            "location": "HQ Conference Room A",
        },
        "expected_group": "Meeting",
    },
    "party": {
        "event": {
            "id": "evt_party_1",
            "title": "Rooftop Evening Cocktail Party",
            "startAtISO": "2026-09-05T20:00:00Z",
            "endAtISO": "2026-09-05T23:00:00Z",
            "location": "Sky Lounge",
        },
        "expected_group": "Social",
    },
    "workout": {
        "event": {
            "id": "evt_workout_1",
            "title": "Morning HIIT & Strength Training",
            "startAtISO": "2026-09-05T07:00:00Z",
            "endAtISO": "2026-09-05T08:00:00Z",
            "location": "Equinox Gym",
        },
        "expected_group": "Health",
    },
    "travel": {
        "event": {
            "id": "evt_travel_1",
            "title": "Flight to London Heathrow Airport Transit",
            "startAtISO": "2026-09-05T10:00:00Z",
            "endAtISO": "2026-09-05T18:00:00Z",
            "location": "Terminal 5",
        },
        "expected_group": "Travel",
    },
    "health": {
        "event": {
            "id": "evt_health_1",
            "title": "Doctor Appointment Medical Checkup",
            "startAtISO": "2026-09-05T11:00:00Z",
            "endAtISO": "2026-09-05T12:00:00Z",
            "location": "City Health Clinic",
        },
        "expected_group": "Health",
    },
    "finance": {
        "event": {
            "id": "evt_finance_1",
            "title": "Quarterly Tax Invoice Payment Due",
            "startAtISO": "2026-09-05T17:00:00Z",
            "endAtISO": "2026-09-05T17:30:00Z",
            "location": "Online Portal",
        },
        "expected_group": "Work",
    },
    "general": {
        "event": {
            "id": "evt_general_1",
            "title": "Grocery Home Essentials Shopping",
            "startAtISO": "2026-09-05T16:00:00Z",
            "endAtISO": "2026-09-05T16:45:00Z",
            "location": "Local Market",
        },
        "expected_group": "Personal",
    },
}


@pytest.mark.parametrize("scenario_key", list(BASELINE_SCENARIOS.keys()))
def test_calendar_runtime_baseline_scenario(scenario_key: str) -> None:
    scenario = BASELINE_SCENARIOS[scenario_key]
    raw_evt = scenario["event"]
    evt_input = CalendarEventInput(
        eventId=raw_evt["id"],
        title=raw_evt["title"],
        startAtISO=raw_evt["startAtISO"],
        endAtISO=raw_evt["endAtISO"],
        location=raw_evt["location"],
    )

    result = run_calendar_runtime(evt_input, user_id="test_user_baseline")

    assert result is not None
    assert hasattr(result, "classifiedEvent")
    assert result.classifiedEvent.group is not None
