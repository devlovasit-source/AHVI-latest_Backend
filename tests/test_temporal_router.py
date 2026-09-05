"""Unit tests for FastAPI temporal router endpoints with authentication dependency overrides."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from middleware.auth_middleware import get_current_user
from routers.temporal import router as temporal_router

app = FastAPI()
app.include_router(temporal_router)

# Override get_current_user to simulate authenticated identity
mock_user = {"user_id": "usr_api_test", "$id": "usr_api_test"}
app.dependency_overrides[get_current_user] = lambda: mock_user

client = TestClient(app)


def test_get_timeline_endpoint() -> None:
    response = client.get("/api/temporal/timeline")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user_id"] == "usr_api_test"
    assert "timeline_items" in data


def test_get_context_endpoint() -> None:
    response = client.get("/api/temporal/context?window_hours=12")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user_id"] == "usr_api_test"
    assert "upcoming_activities" in data
    assert "schedule_conflicts" in data


def test_get_opportunities_endpoint() -> None:
    response = client.get("/api/temporal/opportunities")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user_id"] == "usr_api_test"
    assert "opportunities" in data


def test_evaluate_shadow_mode_endpoint() -> None:
    payload = {
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
        "candidate_actions": [
            {
                "id": "act_api_1",
                "user_id": "usr_api_test",
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
    assert data["user_id"] == "usr_api_test"
    assert data["output_count"] == 1


def test_user_enumeration_and_all_users_cron_sweep(monkeypatch) -> None:
    """Test list_active_user_ids and system-wide temporal_all_users_background_sweep_task."""
    from services.data_access_service import list_active_user_ids
    from worker import temporal_all_users_background_sweep_task

    def mock_list_documents(self, resource, **kwargs):
        if resource == "users":
            return [
                {"$id": "usr_enum_1"},
                {"$id": "usr_enum_2"},
            ]
        return []


    monkeypatch.setattr("services.appwrite_proxy.AppwriteProxy.list_documents", mock_list_documents)

    uids = list_active_user_ids()
    assert uids == ["usr_enum_1", "usr_enum_2"]

    delays = []

    def mock_delay(user_id="", request_id=""):
        delays.append(user_id)
        return True

    monkeypatch.setattr("worker.temporal_background_sweep_task.delay", mock_delay)

    res = temporal_all_users_background_sweep_task()
    assert res["status"] == "success"
    assert res["users_enumerated"] == 2
    assert res["tasks_dispatched"] == 2
    assert delays == ["usr_enum_1", "usr_enum_2"]

