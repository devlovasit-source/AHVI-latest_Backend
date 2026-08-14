from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat
from services.style_conversation_context import resolve_style_conversation_context


def _board_payload():
    return {
        "board_id": "board-current",
        "revision": 3,
        "interaction_mode": "recommendation",
        "board_items": [
            {"item_id": "top-1", "role": "top", "name": "White Shirt"},
        ],
    }


def _module_client(monkeypatch, board_calls):
    monkeypatch.setattr(
        chat,
        "resolve_location_weather_context",
        lambda **kwargs: {
            "profile": dict(kwargs.get("profile") or {}),
            "location": {},
            "weather": {},
            "context_usage": {},
        },
    )
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, request_profile=None: {
            **(request_profile or {}),
            "user_id": user_id,
        },
    )
    monkeypatch.setattr(
        chat,
        "_module_llm_response",
        lambda **kwargs: {
            "message_text": "Text advice without a board.",
            "response": "Text advice without a board.",
        },
    )

    def fake_semantic(**kwargs):
        deterministic = kwargs.get("deterministic")
        if deterministic is None:
            return None
        return {**deterministic, "decision_source": "deterministic_fast_path"}

    monkeypatch.setattr(chat, "resolve_semantic_intent", fake_semantic)
    monkeypatch.setattr(chat, "_apply_style_compliance_gate", lambda payload, **kwargs: payload)

    def board_payload(*args, **kwargs):
        board_calls.append(kwargs)
        return {
            "success": True,
            "message": "Here is the look.",
            "cards": [{"id": "board-generated", "items": []}],
            "style_boards": [{"id": "board-generated", "items": []}],
            "data": {"outfits": [{"id": "board-generated"}]},
            "meta": {"mode": "wardrobe_style"},
        }

    monkeypatch.setattr(chat, "_demo_style_board_payload", board_payload)
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-vc8-continuity"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def test_dinner_context_reaches_new_module_chat_board_generation(monkeypatch):
    board_calls = []
    client = _module_client(monkeypatch, board_calls)

    response = client.post(
        "/api/module-chat",
        json={
            "domain": "style",
            "message": "What should I wear?",
            "history": [{"role": "user", "content": "I have dinner tomorrow."}],
        },
    )

    assert response.status_code == 200
    assert response.json()["style_boards"]
    assert len(board_calls) == 1
    resolved = board_calls[0]["resolved_context"]
    assert resolved["occasion"] == "dinner"
    assert resolved["date_context"] == "tomorrow"


def test_garment_referent_resolves_text_without_item_ids():
    cases = (
        (
            "What should I wear with it?",
            "I'm thinking of my blue shirt.",
            "blue shirt",
        ),
        (
            "What shoes go with it?",
            "I wore my black blazer yesterday.",
            "black blazer",
        ),
        (
            "What shoes work with the trousers?",
            "I have a blue shirt and black trousers.",
            "black trousers",
        ),
    )

    for message, prior_message, expected in cases:
        context, _ = resolve_style_conversation_context(
            current_message=message,
            recent_history=[{"role": "user", "content": prior_message}],
        )
        assert context.referent["type"] == "garment"
        assert context.referent["text"] == expected
        assert context.referent["resolved_to"] == expected
        assert "item_id" not in context.referent
        assert "anchor_id" not in context.referent


def test_current_board_does_not_override_clarification_or_advice(monkeypatch):
    board_calls = []
    client = _module_client(monkeypatch, board_calls)

    for message, expected_mode in (
        ("I'm confused about what to wear", "clarification"),
        ("Give me style tips", "text_only"),
    ):
        response = client.post(
            "/api/module-chat",
            json={
                "domain": "style",
                "message": message,
                "style_state": _board_payload(),
            },
        )
        body = response.json()
        assert response.status_code == 200
        assert body["response_mode"] == expected_mode
        assert body["style_boards"] == []
        assert body["cards"] == []

    assert board_calls == []


def test_explicit_current_turn_generation_can_follow_current_board(monkeypatch):
    board_calls = []
    client = _module_client(monkeypatch, board_calls)

    response = client.post(
        "/api/module-chat",
        json={
            "domain": "style",
            "message": "Build me another outfit for dinner tomorrow",
            "style_state": _board_payload(),
        },
    )

    assert response.status_code == 200
    assert response.json()["style_boards"]
    assert len(board_calls) == 1
    assert board_calls[0]["resolved_context"]["occasion"] == "dinner"
    assert board_calls[0]["resolved_context"]["date_context"] == "tomorrow"


def test_api_text_current_board_clarification_beats_legacy_board_generation(monkeypatch):
    board_calls = []
    client = _module_client(monkeypatch, board_calls)

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [
                {"role": "user", "content": "I'm confused about what to wear"}
            ],
            "style_state": _board_payload(),
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["response_mode"] == "clarification"
    assert body["cards"] == []
    assert body["style_boards"] == []
    assert board_calls == []
