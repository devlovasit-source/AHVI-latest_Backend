from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain.plan_pack_flow import build_plan_pack_response
from brain.daily_dependency_engine import build_daily_dependency_response
from routers import chat, workouts
from routers.stylist import ItemStyleRequest, style_wardrobe_item


def test_plan_pack_does_not_use_device_weather_as_destination_weather():
    response = build_plan_pack_response(
        "pack for a two day trip",
        {
            "weather_context": {"status": "available", "condition": "rain"},
            "location_context": {"status": "available", "source": "device"},
            "context_usage": {"weather": {"status": "available", "source": "provider"}},
        },
    )
    assert response["data"]["weather"] == "unavailable"
    assert response["data"]["weather_status"] == "unavailable"
    assert "mild" not in str(response).lower()
    assert "check forecast" in str(response).lower()


def test_plan_pack_uses_explicit_destination_weather():
    response = build_plan_pack_response(
        "pack for a two day trip",
        {"destination_weather": {"weather_type": "storm"}},
    )
    assert response["data"]["weather"] == "rainy"
    assert "Compact rain jacket" in str(response)


def test_module_chat_resolves_context_before_plan_pack_fast_path(monkeypatch):
    monkeypatch.setattr("services.location_weather_context.get_user_profile", lambda **_: {})
    monkeypatch.setattr(
        "services.location_weather_context.get_hourly_weather",
        lambda **_: {"condition": "rain", "temperature": 20, "time_of_day": "afternoon"},
    )
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "u1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    body = TestClient(app).post(
        "/api/chat/module-chat",
        json={
            "module": "planner",
            "message": "pack for a two day trip",
            "coordinates": {"lat": 12, "lng": 77},
        },
    ).json()
    assert body["context_usage"]["weather"]["status"] == "available"
    assert body["data"]["weather"] == "unavailable"


def test_text_route_resolves_direct_weather_before_plan_pack_fast_path(monkeypatch):
    monkeypatch.setattr("services.location_weather_context.get_user_profile", lambda **_: {})

    def fail_lookup(**kwargs):
        raise AssertionError("direct weather must override coordinate lookup")

    monkeypatch.setattr("services.location_weather_context.get_hourly_weather", fail_lookup)
    monkeypatch.setattr(chat, "_fetch_wardrobe_for_style", lambda *args, **kwargs: [])
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "u1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    response = TestClient(app).post(
        "/api/text",
        json={
            "messages": [{"role": "user", "content": "pack for a two day trip"}],
            "coordinates": {"lat": 12, "lng": 77},
            "weather": {"condition": "clear", "temperature": 26},
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["context_usage"]["weather"]["source"] == "direct_weather"
    assert body["data"]["weather"] == "unavailable"


def test_daily_dependency_reports_weather_unavailable_without_mild_default():
    class Appwrite:
        def list_documents(self, *args, **kwargs):
            return []

    response = build_daily_dependency_response(
        user_id="u1",
        context={"weather_data": {"status": "unavailable"}},
        appwrite=Appwrite(),
    )
    assert response["data"]["weather_status"] == "unavailable"
    assert response["data"]["weather"] == ""
    assert "mild" not in str(response).lower()


def test_workout_route_prefers_direct_weather_and_returns_provenance(monkeypatch):
    monkeypatch.setattr("services.location_weather_context.get_user_profile", lambda **_: {})

    def fail_lookup(**kwargs):
        raise AssertionError("direct weather should prevent lookup")

    monkeypatch.setattr("services.location_weather_context.get_hourly_weather", fail_lookup)
    monkeypatch.setattr(workouts, "_persist_cards", lambda *args, **kwargs: None)
    app = FastAPI()
    app.include_router(workouts.router, prefix="/api")
    app.dependency_overrides[workouts.get_current_user] = lambda: {"user_id": "u1"}
    response = TestClient(app).post(
        "/api/workouts/recommend",
        json={
            "weather": {"weather_type": "storm", "temp_c": 18},
            "coordinates": {"lat": 12, "lng": 77},
            "location": "home",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["context_usage"]["weather"]["source"] == "direct_weather"
    assert body["meta"]["context"]["weather_context"]["condition"] == "storm"


def test_item_style_consumes_canonical_weather(monkeypatch):
    monkeypatch.setattr("services.location_weather_context.get_user_profile", lambda **_: {})
    response = style_wardrobe_item(
        "shirt-1",
        ItemStyleRequest(
            user_id="u1",
            mode="style_this",
            anchor_item={"id": "shirt-1", "name": "White shirt", "category": "top"},
            wardrobe=[
                {"id": "pants-1", "name": "Black trousers", "category": "bottom"},
                {"id": "shoe-1", "name": "Leather shoes", "category": "footwear"},
            ],
            weather={"condition": "rain", "temperature_c": 20},
        ),
    )
    assert response["context_usage"]["weather_used"] is True
    assert "Rain-safe adjustment" in response["style_directions"][0]["styling_note"]
