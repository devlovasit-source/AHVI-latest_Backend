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


def test_text_chat_bare_style_action_recovers_previous_prompt_from_history():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/text",
        json={
            "messages": [
                {"role": "user", "content": "Beach wear · Casual beach walk"},
                {
                    "role": "assistant",
                    "content": "I checked your wardrobe against the occasion. I found a few close matches.",
                },
                {"role": "user", "content": "Show closest option"},
            ]
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Authenticated user is required"


def test_style_fallback_forwards_style_action(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {"user_id": user_id},
    )
    monkeypatch.setattr(
        chat,
        "_fetch_wardrobe_for_style",
        lambda user_id, request_wardrobe: [{"id": "top-1", "category": "top"}],
    )
    monkeypatch.setattr(chat, "_ahvi_item_allowed_for_user_profile", lambda *args, **kwargs: True)

    def fake_build_style_flow_response(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "message": "Closest option ready.",
            "type": "cards",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "data": {"outfits": [{"id": "look-1"}]},
            "meta": {},
        }

    monkeypatch.setattr(chat, "build_style_flow_response", fake_build_style_flow_response)

    response = chat._demo_style_board_payload(
        "user-1",
        "beach wear · Casual beach walk",
        request_wardrobe=[],
        user_profile={},
        style_action="show_closest_option",
    )

    assert response["cards"]
    assert captured["style_action"] == "show_closest_option"
    assert captured["show_closest_option"] is True
    assert captured["allow_closest_option"] is True
    assert captured["closest"] is True


def test_planner_module_routes_plan_pack_to_checklists(monkeypatch):
    def fail_module_chat(*args, **kwargs):
        raise AssertionError("plan-pack prompts should not use generic planner fallback")

    monkeypatch.setattr(chat, "handle_module_chat", fail_module_chat)
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "planner",
            "message": "plan and pack for a 2 day beach trip",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["type"] == "checklists"
    assert body["meta"]["intent"] == "plan_pack"
    assert len(body["cards"]) >= 3


def test_lifestyle_intent_engine_routes_known_prompts():
    from brain.intent_engine import detect_intent

    expected = {
        "Today's meals": ("organize_hub", "meal_planner"),
        "Today's workout": ("organize_hub", "workout"),
        "Morning skincare": ("organize_hub", "skincare"),
        "Pending bills": ("organize_hub", "bills"),
        "My medicines": ("organize_hub", "medicines"),
        "Today's events": ("organize_hub", "calendar"),
        "Upcoming events": ("organize_hub", "calendar"),
    }

    for prompt, (intent, module) in expected.items():
        row = detect_intent(prompt)
        assert row["intent"] == intent
        assert row["slots"]["module"] == module
        assert row["confidence"] >= 0.75


def test_plan_pack_prompts_route_without_generic_fallback():
    prompts = [
        "Help me prep for camping",
        "Plan for a 3 day Goa trip",
        "Plan a birthday party",
        "Pack for a carry-on trip",
    ]
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    for prompt in prompts:
        response = client.post(
            "/api/chat/module-chat",
            json={
                "module": "planner",
                "message": prompt,
                "history": [],
                "context_data": {},
                "user_profile": {},
            },
        )

        body = response.json()
        assert response.status_code == 200
        assert body["intent"] == "plan_pack"
        assert body["meta"]["intent"] == "plan_pack"
        assert body["cards"]
        assert body["quick_actions"] == ["Packing checklist", "Plan outfits", "Weather prep", "Save trip plan"]
        assert "I can help with style, planning, and wardrobe advice" not in body["message"]["content"]


def test_module_summary_prompts_return_cards_and_actions(monkeypatch):
    from services import module_summary_service

    monkeypatch.setattr(chat, "_state_user_id", lambda request: "user-1")
    monkeypatch.setattr(module_summary_service, "_docs", lambda *args, **kwargs: [])

    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    expected = {
        "Today's meals": "meals",
        "Today's workout": "workout",
        "Morning skincare": "skincare",
        "Pending bills": "bills",
        "My medicines": "medicines",
        "Today's events": "events",
        "Upcoming events": "events",
    }

    for prompt, module in expected.items():
        response = client.post(
            "/api/chat/module-chat",
            json={
                "module": "chat",
                "message": prompt,
                "history": [],
                "context_data": {},
                "user_profile": {},
            },
        )

        body = response.json()
        assert response.status_code == 200
        assert body["type"] == "module_card"
        assert body["module"] == module
        assert body["card"]
        assert body["quick_actions"]
        assert "I can help with style, planning, and wardrobe advice" not in body["message"]
