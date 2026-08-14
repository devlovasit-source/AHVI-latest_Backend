import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain.intent_engine import STYLE_PAIRING, detect_intent
from routers import chat
from services.pre_classifier import classify_message


_VC8_PROMPTS = (
    ("hi", "text_only"),
    ("how are you doing?", "text_only"),
    ("give me style tips", "text_only"),
    ("what suits my skin tone", "text_only"),
    ("i meant what color suits my skin tone", "text_only"),
    ("im little emotionally lost today", "text_only"),
    ("im not sure what to do", "text_only"),
    ("who are you?", "text_only"),
    ("what should I wear today?", "wardrobe_recommendation"),
    ("style me for a party", "wardrobe_recommendation"),
    ("build me an outfit for dinner tomorrow", "wardrobe_recommendation"),
    ("how can I style my blue shirt?", "text_only"),
)


def _client(monkeypatch, *, persisted_profile=None):
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
            **(persisted_profile or {}),
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
    monkeypatch.setattr(
        chat,
        "resolve_semantic_intent",
        lambda **kwargs: (
            {**kwargs["deterministic"], "decision_source": "deterministic_fast_path"}
            if kwargs.get("deterministic") is not None
            else None
        ),
    )
    monkeypatch.setattr(chat, "_apply_style_compliance_gate", lambda payload, **kwargs: payload)
    monkeypatch.setattr(chat, "_should_default_visual_inspiration", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *args, **kwargs: {
            "success": True,
            "message": "Here is the look.",
            "cards": [{"id": "board-1", "items": []}],
            "style_boards": [{"id": "board-1", "items": []}],
            "data": {"outfits": [{"id": "board-1"}]},
            "meta": {"mode": "wardrobe_style"},
        },
    )
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-vc8"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def _semantic_recommendation():
    return {
        "domain": "style",
        "intent": "recommendation",
        "action": "recommend_wardrobe",
        "response_mode": "wardrobe_recommendation",
        "confidence": 0.99,
        "requires_clarification": False,
        "resolved_context": {},
        "constraints": {"required": [], "avoid": []},
        "referent": None,
        "reason_codes": ["forced_regression"],
        "missing_information": [],
    }


def test_vc8_exact_twelve_prompt_classification():
    for prompt, expected_mode in _VC8_PROMPTS:
        decision = classify_message(prompt)
        if expected_mode == "text_only":
            assert decision is not None, prompt
            assert decision["response_mode"] == expected_mode, prompt
        else:
            assert decision is None, prompt


def test_contraction_variants_are_supportive_text_only():
    prompts = (
        "don't know what to do today",
        "don’t know what to do today",
        "dont know what to do today",
        "I'm not sure what to do",
        "I’m not sure what to do",
        "Im not sure what to do",
    )

    for prompt in prompts:
        decision = classify_message(prompt)
        assert decision is not None, prompt
        assert decision["intent"] == "supportive_conversation", prompt
        assert decision["response_mode"] == "text_only", prompt


def test_general_help_uses_exact_full_message_boundary():
    for prompt in ("Can you help me?", "Could you help me?"):
        decision = classify_message(prompt)
        assert decision is not None, prompt
        assert decision["intent"] == "help_identity", prompt
        assert decision["response_mode"] == "text_only", prompt

    assert classify_message("Can you help me pick an outfit?") is None


def test_style_indecision_clarifies_without_stealing_explicit_generation():
    for prompt in (
        "I'm confused about what to wear",
        "I'm not sure what to wear",
        "I don't know what to wear",
    ):
        decision = classify_message(prompt)
        assert decision is not None, prompt
        assert decision["intent"] == "clarification", prompt
        assert decision["response_mode"] == "clarification", prompt

    assert classify_message("Build me an outfit because I'm confused what to wear") is None
    assert classify_message("Style me for dinner, I'm not sure what to wear") is None


def test_pairing_precedes_broad_generation_classification():
    pairing = classify_message("What should I wear with this shirt?")
    assert pairing is not None
    assert pairing["intent"] == "advice"
    assert pairing["response_mode"] == "text_only"

    assert classify_message("What should I wear today?") is None
    assert classify_message("Build a complete outfit around this shirt") is None
    assert detect_intent("What should I wear with this shirt?")["intent"] == STYLE_PAIRING


def test_vc8_exact_twelve_module_chat_contract(monkeypatch):
    client = _client(monkeypatch)

    for prompt, expected_mode in _VC8_PROMPTS:
        response = client.post(
            "/api/module-chat",
            json={"domain": "style", "message": prompt, "request_id": "vc8-" + str(len(prompt))},
        )
        body = response.json()
        assert response.status_code == 200, prompt
        assert body["response_mode"] == expected_mode, (prompt, body)
        if expected_mode == "text_only":
            assert body["style_boards"] == [], prompt
            assert body["cards"] == [], prompt
        else:
            assert body["style_boards"], prompt


def test_commit_one_module_chat_boundaries(monkeypatch):
    client = _client(monkeypatch)
    cases = (
        ("hi", "text_only", False),
        ("don't know what to do today", "text_only", False),
        ("don’t know what to do today", "text_only", False),
        ("dont know what to do today", "text_only", False),
        ("I'm not sure what to do", "text_only", False),
        ("I’m not sure what to do", "text_only", False),
        ("Im not sure what to do", "text_only", False),
        ("I'm little emotionally lost today", "text_only", False),
        ("I'm not sure what to do with my blue shirt", "text_only", False),
        ("What colors suit my skin tone?", "text_only", False),
        ("Can you help me?", "text_only", False),
        ("I'm confused about what to wear", "clarification", False),
        ("I'm confused", "text_only", False),
        ("What should I wear with this shirt?", "clarification", False),
        ("Can you help me pick an outfit?", "wardrobe_recommendation", True),
        ("What should I wear today?", "wardrobe_recommendation", True),
        ("Build a complete outfit around this shirt", "wardrobe_recommendation", True),
        ("Build me an outfit that suits my skin tone", "wardrobe_recommendation", True),
        ("Build me an outfit for dinner tomorrow", "wardrobe_recommendation", True),
        ("Style me using colors that suit me", "wardrobe_recommendation", True),
    )

    for prompt, expected_mode, expects_board in cases:
        response = client.post(
            "/api/module-chat",
            json={"domain": "style", "message": prompt, "request_id": "commit-one"},
        )
        body = response.json()
        assert response.status_code == 200, prompt
        assert body["response_mode"] == expected_mode, (prompt, body)
        assert bool(body.get("style_boards")) is expects_board, (prompt, body)


def test_color_advice_uses_persisted_profile_provenance(monkeypatch):
    client = _client(monkeypatch, persisted_profile={"skin_tone": "warm"})

    response = client.post(
        "/api/module-chat",
        json={"domain": "style", "message": "what suits my skin tone"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["response_mode"] == "text_only"
    assert body["context_provenance"] == {"source": "persisted_profile", "available": True}
    assert body["data"]["context_provenance"] == body["context_provenance"]
    assert body["style_boards"] == []


def test_color_advice_reports_absent_profile_without_inventing_context(monkeypatch):
    client = _client(monkeypatch)

    response = client.post(
        "/api/module-chat",
        json={"domain": "style", "message": "i meant what color suits my skin tone"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["response_mode"] == "text_only"
    assert body["context_provenance"] == {"source": "none", "available": False}
    assert body["style_boards"] == []


def test_supportive_prompt_with_garment_routes_to_pairing_text(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/api/module-chat",
        json={"domain": "style", "message": "I'm not sure what to do with my blue shirt"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["intent"] == "advice"
    assert body["response_mode"] == "text_only"
    assert body["style_boards"] == []


def test_supportive_prompt_with_wear_for_dinner_stays_style_path(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/api/module-chat",
        json={"domain": "style", "message": "I'm not sure what to wear for dinner"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["response_mode"] == "wardrobe_recommendation"
    assert body["style_boards"]


def test_plain_color_question_is_text_only(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/api/module-chat",
        json={"domain": "style", "message": "What colors suit my skin tone?"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["intent"] == "color_advice"
    assert body["response_mode"] == "text_only"
    assert body["style_boards"] == []


def test_plain_undertone_question_is_text_only(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/api/module-chat",
        json={"domain": "style", "message": "What is my undertone?"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["intent"] == "color_advice"
    assert body["response_mode"] == "text_only"
    assert body["style_boards"] == []


def test_outfit_generation_overrides_color_advice(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/api/module-chat",
        json={
            "domain": "style",
            "message": "Build me an outfit that suits my skin tone",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["response_mode"] == "wardrobe_recommendation"
    assert body["style_boards"]


def test_style_generation_with_color_context_is_allowed(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/api/module-chat",
        json={
            "domain": "style",
            "message": "Style me for a party using colors that suit my skin tone",
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["response_mode"] == "wardrobe_recommendation"
    assert body["style_boards"]


def test_semantic_recommendation_cannot_authorize_emotional_board(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(chat, "resolve_semantic_intent", lambda **kwargs: _semantic_recommendation())
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("semantic recommendation authorized a board")),
    )

    response = client.post(
        "/api/module-chat",
        json={"domain": "style", "message": "im not sure what to do"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["response_mode"] == "text_only"
    assert body["style_boards"] == []
    assert chat._has_positive_style_board_intent("im not sure what to do", "style") is False


def test_approved_mutation_returns_before_new_board_gate(monkeypatch):
    client = _client(monkeypatch)
    mutation = {
        "success": True,
        "type": "board_mutation",
        "message": "Updated the current look.",
        "message_text": "Updated the current look.",
        "response": "Updated the current look.",
        "cards": [{"id": "mutated-board"}],
        "style_boards": [{"id": "mutated-board"}],
        "response_mode": "wardrobe_recommendation",
        "meta": {"mode": "style_board_mutation", "board_mutated": True},
    }
    monkeypatch.setattr(
        chat,
        "resolve_semantic_intent",
        lambda **kwargs: {
            "domain": "style",
            "intent": "modify_current_look",
            "action": "modify_current_look",
            "response_mode": "wardrobe_recommendation",
            "confidence": 0.99,
            "requires_clarification": False,
            "resolved_context": {},
            "constraints": {"required": [], "avoid": []},
            "referent": None,
            "reason_codes": ["approved_mutation_regression"],
            "missing_information": [],
            "operation": {
                "type": "modify",
                "replace_roles": ["top"],
                "preserve_roles": [],
                "remove_roles": [],
                "constraints": [],
                "style_adjustments": {},
                "alternative_scope": None,
                "explanation_target": None,
            },
        },
    )
    monkeypatch.setattr(chat, "handle_board_operation", lambda *args, **kwargs: mutation)
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("new board path was used")),
    )

    response = client.post(
        "/api/module-chat",
        json={"domain": "style", "message": "make the top more casual"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["meta"]["board_mutated"] is True
    assert body["style_boards"]


def test_context_alone_cannot_authorize_style_board(monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(chat, "resolve_semantic_intent", lambda **kwargs: None)

    response = client.post(
        "/api/module-chat",
        json={
            "domain": "style",
            "message": "something for later",
            "context": {"occasion": "party"},
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["response_mode"] == "text_only"
    assert body["style_boards"] == []


def test_board_gate_can_be_called_for_direct_module_impl(monkeypatch):
    request = chat.ModuleChatRequest(
        domain="style",
        message="something for later",
        context={"occasion": "party"},
    )
    scope = {"type": "http", "method": "POST", "path": "/api/module-chat", "headers": []}
    http_request = chat.Request(scope)
    monkeypatch.setattr(
        chat,
        "resolve_location_weather_context",
        lambda **kwargs: {"profile": {}, "location": {}, "weather": {}, "context_usage": {}},
    )
    monkeypatch.setattr(
        chat,
        "_module_llm_response",
        lambda **kwargs: {"message_text": "I can help when you are ready."},
    )
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("board was not authorized")),
    )

    envelope = asyncio.run(
        chat._module_chat_impl(request, http_request, board_authorized=False)
    )

    assert envelope.get("style_boards", []) == []
    assert envelope["message_text"] == "I can help when you are ready."
