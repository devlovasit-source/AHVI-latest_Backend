from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain.plan_pack_flow import build_plan_pack_response
from routers import chat


def _walk_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, str):
        yield value


def _visual_item(response, section_id, label):
    section = next(section for section in response["visual_sections"] if section["id"] == section_id)
    return next(item for item in section["items"] if item["label"] == label)


def test_plan_pack_returns_visual_sections_and_preserves_cards():
    response = build_plan_pack_response("Pack for a carry-on trip")

    assert response["type"] == "checklists"
    assert response["visual_type"] == "visual_packing_checklist"
    assert response["cards"]
    assert response["visual_sections"]
    assert {section["id"] for section in response["visual_sections"]}.issuperset(
        {"clothes", "essentials", "tech", "documents", "weather"}
    )


def test_plan_pack_wardrobe_items_map_to_visual_section_images():
    wardrobe = [
        {
            "$id": "shirt-1",
            "name": "White Cotton Shirt",
            "category": "Tops",
            "display_image_url": "https://example.com/shirt.png",
        }
    ]
    response = build_plan_pack_response("Pack for a carry-on trip", {"wardrobe": wardrobe})

    clothes = next(section for section in response["visual_sections"] if section["id"] == "clothes")
    tops = next(item for item in clothes["items"] if item["label"] == "Tops")

    assert tops["source"] == "wardrobe"
    assert tops["image_urls"] == ["https://example.com/shirt.png"]
    assert tops["wardrobe_item_ids"] == ["shirt-1"]


def test_plan_pack_empty_wardrobe_uses_icon_keys_without_broken_asset_paths():
    response = build_plan_pack_response("Pack for a carry-on trip", {"wardrobe": []})

    documents = next(section for section in response["visual_sections"] if section["id"] == "documents")
    tech = next(section for section in response["visual_sections"] if section["id"] == "tech")

    assert documents["items"]
    assert tech["items"]
    assert all(item["source"] == "icon" for item in documents["items"])
    assert all(item.get("assetIcon") is None for item in documents["items"])
    assert all(item.get("asset_key") is None for item in documents["items"])
    assert all("assets/icons/" not in value for value in _walk_strings(response))


def test_known_packing_items_return_semantic_icon_keys():
    travel_response = build_plan_pack_response("Pack for a carry-on trip", {"wardrobe": []})
    default_response = build_plan_pack_response("prepare my bag", {"wardrobe": []})

    sunscreen = _visual_item(travel_response, "essentials", "Sunscreen")
    charger = _visual_item(default_response, "tech", "Phone + charger")
    passport = _visual_item(default_response, "documents", "Passport/ID")

    assert sunscreen["source"] == "icon"
    assert sunscreen["iconKey"] == "sunscreen"
    assert sunscreen["assetIcon"] is None
    assert charger["iconKey"] == "charger"
    assert charger["assetIcon"] is None
    assert passport["iconKey"] == "documents"
    assert passport["assetIcon"] is None


def test_plan_pack_module_fetches_wardrobe_and_fails_open(monkeypatch):
    app = FastAPI()

    @app.middleware("http")
    async def user_middleware(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    calls = []

    def fake_fetch(user_id, request_wardrobe):
        calls.append((user_id, request_wardrobe))
        raise RuntimeError("appwrite unavailable")

    monkeypatch.setattr(chat, "_fetch_wardrobe_for_style", fake_fetch)

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "planner",
            "message": "pack for a carry-on trip",
            "history": [],
            "context_data": {},
            "user_profile": {"user_id": "user-1"},
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert calls
    assert body["type"] == "checklists"
    assert body["visual_type"] == "visual_packing_checklist"
    assert body["visual_sections"]
