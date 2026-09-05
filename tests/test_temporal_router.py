"""Unit tests for FastAPI temporal router endpoints."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers.temporal import router as temporal_router

app = FastAPI()
app.include_router(temporal_router)

client = TestClient(app)


def test_get_timeline_endpoint() -> None:
    response = client.get("/api/temporal/timeline?user_id=usr_api_test")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user_id"] == "usr_api_test"
    assert "timeline_items" in data


def test_get_context_endpoint() -> None:
    response = client.get("/api/temporal/context?user_id=usr_api_test&window_hours=12")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "upcoming_activities" in data
    assert "schedule_conflicts" in data


def test_get_opportunities_endpoint() -> None:
    response = client.get("/api/temporal/opportunities?user_id=usr_api_test")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "opportunities" in data


def test_evaluate_shadow_mode_endpoint() -> None:
    payload = {
        "user_id": "usr_sm_api",
        "raw_events": [
            {
                "eventId": "evt_api_1",
                "title": "Strategy Presentation",
                "startAtISO": "2026-09-05T14:00:00+00:00",
                "endAtISO": "2026-09-05T15:00:00+00:00",
            }
        ],
    }
    response = client.post("/api/temporal/evaluate-shadow-mode", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "evaluation" in data
    assert data["evaluation"]["parity_passed"] is True


def test_arbitrate_endpoint() -> None:
    payload = {
        "user_id": "usr_arb_api",
        "candidate_actions": [
            {
                "id": "act_api_1",
                "user_id": "usr_arb_api",
                "source_opportunity_id": "opp_api_1",
                "source_module": "style",
                "action_type": "OUTFIT_SUGGESTION",
                "priority": 4,
                "urgency": 0.8,
                "attention_cost": 0.4,
            }
        ],
    }
    response = client.post("/api/temporal/arbitrate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["output_count"] == 1
