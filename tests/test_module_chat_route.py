from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat


def test_module_chat_legacy_nested_route_exists(monkeypatch):
    def fake_response(*, module, user_message, history, context_data, user_profile):
        return {
            "success": True,
            "type": "module_chat",
            "module": module,
            "response": "ok",
            "message_text": "ok",
            "message": {"role": "assistant", "content": "ok"},
            "cards": [],
            "style_boards": [],
            "chips": [],
            "data": {"module": module, "rendered_boards": [], "outfits": []},
            "meta": {"mode": module, "board_count": 0},
        }

    monkeypatch.setattr(chat, "_module_llm_response", fake_response)
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={"module": "style", "message": "hello", "history": [], "context_data": {}, "user_profile": {}},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "module_chat"


def test_skincare_module_chat_replaces_truncated_spf_answer(monkeypatch):
    monkeypatch.setattr(
        chat,
        "chat_completion",
        lambda *args, **kwargs: "To recommend the best SPF, I need a bit more detail about",
    )

    result = chat._module_llm_response(
        module="skincare",
        user_message="Best SPF for my skin",
        history=[],
        context_data={},
        user_profile={},
    )

    answer = result["message"]["content"]
    assert result["type"] == "module_chat"
    assert "skin type" in answer
    assert "sun exposure" in answer
    assert answer.endswith(".")


def test_style_module_chat_routes_chip_to_style_flow(monkeypatch):
    captured = {}

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile):
        captured["query"] = query_text
        return {
            "success": True,
            "type": "cards",
            "message": "Travel look ready.",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "chips": ["More looks"],
            "board_ids": "look-1",
            "data": {"outfits": [{"id": "look-1"}], "rendered_boards": []},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={"module": "style", "message": "Travel", "history": [], "context_data": {}, "user_profile": {}},
    )

    body = response.json()
    assert response.status_code == 200
    assert captured["query"] == "Travel"
    assert body["message"]["content"] == "Travel look ready."
    assert body["message_text"] == "Travel look ready."
    assert body["response"] == "Travel look ready."
    assert body["cards"]
    assert body["style_boards"]


def test_style_module_chat_routes_beach_wear_without_empty_llm(monkeypatch):
    def fail_llm(*args, **kwargs):
        raise AssertionError("style prompts should not hit module LLM")

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile):
        assert query_text == "beach wear"
        return {
            "success": False,
            "type": "missing_outfit_cards",
            "message": "I need a top, bottom, and footwear before I can build a beach look.",
            "cards": [],
            "style_boards": [],
            "chips": [{"label": "Add wardrobe", "value": "Use my wardrobe"}],
            "data": {"outfits": [], "rendered_boards": []},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    monkeypatch.setattr(chat, "_module_llm_response", fail_llm)
    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={"module": "style", "message": "beach wear", "history": [], "context": {}, "user_profile": {}},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["message"]["content"]
    assert "beach" in body["message_text"]
    assert body["chips"]


def test_text_chat_bare_style_action_requires_context_without_crash():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/text",
        json={"messages": [{"role": "user", "content": "Show closest option"}]},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["type"] == "context_required"
    assert body["data"]["requires_context"] is True
    assert body["data"]["missing_context_for_action"] == "show closest option"
