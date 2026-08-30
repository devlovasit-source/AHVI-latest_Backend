import inspect

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import home, workouts
from services import home_summary_service
from services.workout_card_service import (
    _display_title,
    get_today_workout_card,
    get_workout_recommendations,
)


def _resolved_context():
    return {
        "profile": {},
        "weather": {"status": "available", "condition": "clear", "temperature": 24},
        "location": {"status": "available", "source": "profile"},
        "context_usage": {"weather": {"source": "profile"}},
    }


def test_home_router_imports():
    assert home.router.prefix == "/home"


def test_home_today_summary_returns_all_five_cards(monkeypatch):
    monkeypatch.setattr(home, "get_user_profile", lambda **_: {"timezone": "Asia/Kolkata"})
    monkeypatch.setattr(home_summary_service, "resolve_location_weather_context", lambda **_: _resolved_context())
    monkeypatch.setattr(
        home_summary_service,
        "get_today_workout_card",
        lambda **_: {"id": "move-1", "title": "12-Min Mobility", "why_this": "Fits today."},
    )
    monkeypatch.setattr(
        home_summary_service,
        "get_today_meal_plan",
        lambda **_: {"status": "ready", "plan_id": "plan-1", "meal": {"type": "lunch", "name": "Salad"}},
    )
    monkeypatch.setattr(
        home_summary_service,
        "get_today_skincare_state_readonly",
        lambda *_: {"status": "ready", "routine": "morning", "entity_id": "skin-1", "source": "skincare"},
    )
    monkeypatch.setattr(
        home_summary_service,
        "get_today_medicine_state_readonly",
        lambda *_: {"status": "due_now", "due_now_count": 1, "entity_id": "med-1", "source": "medicine"},
    )
    app = FastAPI()
    app.include_router(home.router)
    app.dependency_overrides[home.get_current_user] = lambda: {"user_id": "authenticated-user"}

    response = TestClient(app).get("/home/today-summary?timezone=Asia/Kolkata")

    assert response.status_code == 200
    body = response.json()
    assert set(body["cards"]) == {"wear", "move", "eat", "care", "medicine"}
    assert body["timezone"] == "Asia/Kolkata"
    assert "context_usage" in body


def test_workout_routes_keep_contracts_and_authenticated_identity(monkeypatch):
    seen = {}

    def build_context(user_id, payload):
        seen.setdefault("context_user_ids", []).append(user_id)
        return {"user_id": user_id, **payload}

    def recommend(**kwargs):
        seen["recommend_user_id"] = kwargs["user_id"]
        return [{"id": "workout-1", "outfit_pairing": {"top": "tee"}, "reminders": ["water"]}]

    def today(**kwargs):
        seen["today_user_id"] = kwargs["user_id"]
        return {"id": "workout-2", "outfit_pairing": {}, "reminders": []}

    monkeypatch.setattr(workouts, "resolve_location_weather_context", lambda **_: _resolved_context())
    monkeypatch.setattr(workouts, "build_workout_context", build_context)
    monkeypatch.setattr(workouts, "get_workout_recommendations", recommend)
    monkeypatch.setattr(workouts, "get_today_workout_card", today)
    monkeypatch.setattr(workouts, "_persist_cards", lambda *_: None)
    app = FastAPI()
    app.include_router(workouts.router)
    app.dependency_overrides[workouts.get_current_user] = lambda: {"user_id": "authenticated-user"}
    client = TestClient(app)

    recommend_response = client.post("/workouts/recommend", json={"user_id": "forged-user", "duration": 20})
    today_response = client.get("/workouts/today")

    assert recommend_response.status_code == 200
    assert set(recommend_response.json()) == {"type", "recommendations", "meta", "context_usage"}
    assert today_response.status_code == 200
    assert set(today_response.json()) == {"type", "today_workout", "outfit_pairing", "reminders", "meta", "context_usage"}
    assert seen["recommend_user_id"] == "authenticated-user"
    assert seen["today_user_id"] == "authenticated-user"
    assert seen["context_user_ids"] == ["authenticated-user", "authenticated-user"]


def test_workout_service_signatures_and_title_normalization():
    recommendations = inspect.signature(get_workout_recommendations)
    today = inspect.signature(get_today_workout_card)
    assert list(recommendations.parameters) == ["user_id", "context", "limit"]
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in recommendations.parameters.values())
    assert recommendations.parameters["limit"].default == 3
    assert list(today.parameters) == ["user_id", "profile", "context"]
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in today.parameters.values())

    assert _display_title({"title": "Women — 10 Min Mobility"}) == "10-Min Mobility"
    assert _display_title({"title": "Men - 20 Min Strength"}) == "20-Min Strength"
    assert _display_title({"title": "Universal — 12 Min Mobility"}) == "12-Min Mobility"
    assert _display_title({"title": "30 Min Cardio"}) == "30-Min Cardio"
