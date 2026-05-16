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
