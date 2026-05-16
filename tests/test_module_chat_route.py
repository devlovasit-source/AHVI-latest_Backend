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
