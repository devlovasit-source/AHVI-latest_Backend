from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import stylist


class _Proxy:
    rows = [
        {
            "id": "shirt-1",
            "userId": "owner-1",
            "name": "Blue Shirt",
            "category": "Tops",
            "source": "wardrobe",
            "image_url": "https://images.test/shirt-1.png",
        }
    ]

    def list_documents(self, resource, **kwargs):
        if resource != "outfits":
            return {"documents": [], "meta": {"has_more": False}}
        user_id = kwargs.get("user_id")
        rows = [row for row in self.rows if row.get("userId") == user_id]
        if kwargs.get("return_meta"):
            return {"documents": rows, "meta": {"has_more": False}}
        return rows


def _client(monkeypatch):
    monkeypatch.setattr(stylist, "AppwriteProxy", _Proxy)
    app = FastAPI()

    @app.middleware("http")
    async def authenticate(request, call_next):
        request.state.user = {"user_id": "owner-1"}
        return await call_next(request)

    app.include_router(stylist.router, prefix="/api/stylist")
    return TestClient(app)


def test_legacy_build_outfit_cta_is_controlled_before_generation(monkeypatch):
    def generation_must_not_run(*args, **kwargs):
        raise AssertionError("legacy CTA must not generate an outfit")

    monkeypatch.setattr(stylist, "_lite_build_outfit", generation_must_not_run)
    monkeypatch.setattr(
        stylist,
        "resolve_location_weather_context",
        generation_must_not_run,
    )
    client = _client(monkeypatch)

    response = client.post(
        "/api/stylist/items/shirt-1/style",
        json={
            "user_id": "owner-1",
            "mode": "build_outfit",
            "scenario": "build_outfit",
            "anchor_item_id": "shirt-1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["intent"] == "try_on_coming_soon"
    assert body["action"] == "try_on_coming_soon"
    assert body["response_mode"] == "text_only"
    assert body["message"] == "Try-On is coming soon."
    assert body["error"]["code"] == "TRY_ON_COMING_SOON"
    assert body["style_directions"] == []
    assert "outfit" not in body


def test_legacy_scenario_without_mode_is_also_controlled(monkeypatch):
    client = _client(monkeypatch)

    response = client.post(
        "/api/stylist/items/shirt-1/style",
        json={
            "user_id": "owner-1",
            "scenario": "build_outfit",
            "anchor_item_id": "shirt-1",
        },
    )

    assert response.status_code == 200
    assert response.json()["error"]["code"] == "TRY_ON_COMING_SOON"
