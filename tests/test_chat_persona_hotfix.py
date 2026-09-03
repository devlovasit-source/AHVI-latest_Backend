from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain.tone.tone_engine import tone_engine
from routers import chat
from services.pre_classifier import classify_message
from services import llm_service


def test_chat_completion_propagates_system_instruction_without_task_duplication(monkeypatch):
    captured = {}

    def fake_generate_text(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return "Natural reply."

    monkeypatch.setattr(llm_service, "generate_text", fake_generate_text)

    result = llm_service.chat_completion(
        [{"role": "user", "content": "Hi"}],
        system_instruction="Use a warm conversational voice.",
    )

    assert result == "Natural reply."
    assert captured["system_instruction"] == "Use a warm conversational voice."
    assert "System:\nUse a warm conversational voice." not in captured["prompt"]


def test_generate_text_reaches_gemini_config_system_instruction(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class FakeTypes:
        GenerateContentConfig = FakeConfig

    class FakeModels:
        @staticmethod
        def generate_content(*, model, contents, config):
            captured["contents"] = contents
            return SimpleNamespace(text="A natural answer.")

    monkeypatch.setattr(llm_service, "_gemini_enabled", lambda: True)
    monkeypatch.setattr(llm_service, "types", FakeTypes)
    monkeypatch.setattr(
        llm_service,
        "_get_gemini_client",
        lambda timeout_seconds=None: SimpleNamespace(models=FakeModels()),
    )
    monkeypatch.setattr(llm_service, "_thinking_config_disabled", lambda: None)

    result = llm_service.generate_text(
        "Say hello.",
        system_instruction="Use a warm conversational voice.",
    )

    assert result == "A natural answer."
    assert captured["config"]["system_instruction"] == "Use a warm conversational voice."
    assert "Use a warm conversational voice." not in captured["contents"]


def test_tone_context_aliases_use_canonical_conversation_and_styling():
    conversation = tone_engine.build_prompt_tone(signals={"context_mode": "chat"})
    assert conversation["context_mode"] == "conversation"
    assert "styling" not in conversation["tone_instruction"].lower()

    style = tone_engine.build_prompt_tone(signals={"context_mode": "style"})
    assert style["context_mode"] == "styling"
    assert tone_engine.config["context_modes"]["styling"]


def test_identity_capability_and_ownership_variants_are_deterministic():
    expected = {
        "who are you": "assistant_identity",
        "what are you": "assistant_identity",
        "what is AHVI": "product_identity",
        "tell me about AHVI": "product_identity",
        "what else can you assist with?": "product_capabilities",
        "what else can AHVI do?": "product_capabilities",
        "who are the owners of AHVI?": "product_ownership",
        "who founded AHVI?": "product_ownership",
        "who created AHVI?": "product_ownership",
    }
    for prompt, help_type in expected.items():
        decision = classify_message(prompt)
        assert decision is not None, prompt
        assert decision["intent"] == "help_identity", prompt
        assert decision["help_type"] == help_type, prompt
        assert chat._is_help_identity_request(prompt)


def test_ownership_is_grounded_and_capabilities_are_broad():
    ownership = chat._ahvi_help_identity_response("Who are the owners of AHVI?")
    assert "verified ownership information" in ownership["message"].lower()
    assert "don't have owners" not in ownership["message"].lower()

    capabilities = chat._ahvi_help_identity_response("What else can you assist with?")
    message = capabilities["message"].lower()
    for area in ("style", "planning", "meals", "workouts", "routines"):
        assert area in message


def test_revenge_request_is_safe_and_natural_without_provider_call(monkeypatch):
    def fail_provider(*args, **kwargs):
        raise AssertionError("revenge safety path must not call the provider")

    monkeypatch.setattr(chat, "chat_completion", fail_provider)
    response = chat._llm_chat_response(
        messages=[],
        english_input="That bitch cheated on me how should I take revenge?",
        user_id="user-1",
        user_profile={},
        user_message_style={},
    )
    message = response["message"].lower()
    assert "revenge" in message
    assert "hurt" in message
    assert "helpful and harmless" not in message


def test_style_release_gate_still_recognizes_office_outfit_request():
    prompt = "What should I wear to office tomorrow?"
    assert chat._is_explicit_style_request(prompt, "style")
    assert classify_message(prompt) is None


def _text_client(monkeypatch):
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
        "resolve_location_weather_context",
        lambda **kwargs: {
            "profile": dict(kwargs.get("profile") or {}),
            "location": {},
            "weather": {},
            "context_usage": {},
        },
    )
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "persona-test-user"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    (
        ("What else can you assist with?", "routines"),
        ("Who are the owners of AHVI?", "verified ownership information"),
    ),
)
def test_api_text_persona_regressions(monkeypatch, prompt, expected):
    response = _text_client(monkeypatch).post(
        "/api/text",
        json={"messages": [{"role": "user", "content": prompt}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "text"
    assert expected in str(body["message"]).lower()


@pytest.mark.parametrize(
    ("prompt", "route"),
    (
        ("Hi", "greeting"),
        ("How are you?", "small_talk"),
        ("What can you do?", "help"),
        ("What else can you assist with?", "help"),
        ("What can AHVI help me with?", "help"),
        ("Who are you?", "help"),
        ("What is AHVI?", "help"),
        ("Who owns AHVI?", "help"),
        ("Who are the owners of AHVI?", "help"),
        ("Who founded AHVI?", "help"),
        ("That bitch cheated on me how should I take revenge?", "safety"),
        ("I feel terrible after my breakup.", "conversation"),
        ("Plan my day.", "planning"),
        ("What should I eat today?", "meals"),
        ("What should I wear to office tomorrow?", "style"),
    ),
)
def test_persona_regression_matrix_routes_without_style_cross_talk(prompt, route):
    if route == "greeting":
        assert chat._is_greeting(prompt)
    elif route == "small_talk":
        assert chat._is_small_talk(prompt)
    elif route == "help":
        assert chat._is_help_identity_request(prompt)
    elif route == "safety":
        assert chat._is_revenge_request(prompt)
    elif route == "conversation":
        assert chat._detect_mode(prompt) == "casual"
        assert not chat._is_explicit_style_request(prompt)
    elif route == "planning":
        assert chat._detect_visual_board_type(prompt) == "trip_prep"
    elif route == "meals":
        assert chat._detect_visual_board_type(prompt) == "diet_plan"
    elif route == "style":
        assert chat._is_explicit_style_request(prompt, "style")
        assert classify_message(prompt) is None
