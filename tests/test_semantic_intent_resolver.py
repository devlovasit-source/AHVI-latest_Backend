import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat
from services import semantic_intent_resolver as resolver


def _decision(**overrides):
    value = {
        "domain": "style",
        "intent": "advice",
        "action": "provide_style_advice",
        "response_mode": "text_only",
        "confidence": 0.94,
        "requires_clarification": False,
        "resolved_context": {"occasion": "later"},
        "constraints": {"required": ["more casual"], "avoid": ["formal"]},
        "referent": {"kind": "look", "ordinal": 2},
        "reason_codes": ["follow_up_context"],
        "missing_information": [],
    }
    value.update(overrides)
    return value


def test_deterministic_decision_bypasses_model(monkeypatch):
    def fail_model(*args, **kwargs):
        raise AssertionError("deterministic requests must not call the model")

    monkeypatch.setattr(resolver, "_generate_text", fail_model)
    got = resolver.resolve_semantic_intent(
        current_message="Give me style tips",
        module_hint="style",
        deterministic={
            "domain": "style",
            "intent": "advice",
            "action": "provide_style_advice",
            "response_mode": "text_only",
        },
    )
    assert got["response_mode"] == "text_only"
    assert got["decision_source"] == "deterministic_fast_path"


def test_contextual_request_uses_validated_model_decision(monkeypatch):
    captured = {}

    def fake_model(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return json.dumps(_decision())

    monkeypatch.setattr(resolver, "_generate_text", fake_model)
    got = resolver.resolve_semantic_intent(
        current_message="something like that, but more casual",
        recent_history=[{"role": "assistant", "content": "Look two"}],
        module_hint="style",
        conversation_context={"last_style_context": "look two"},
        request_id="req_semantic_1",
    )
    assert got["decision_source"] == "llm_semantic_resolver"
    assert got["referent"] == {"kind": "look", "ordinal": 2}
    assert captured["kwargs"]["request_id"] == "req_semantic_1"
    assert "something like that" in captured["prompt"]


def test_low_confidence_becomes_clarification():
    got = resolver.validate_semantic_decision(
        _decision(confidence=0.2, resolved_context={})
    )
    assert got["response_mode"] == "clarification"
    assert got["requires_clarification"] is True
    assert got["intent"] == "clarification"


def test_invalid_json_context_falls_back_to_clarification(monkeypatch):
    monkeypatch.setattr(resolver, "_generate_text", lambda *args, **kwargs: "not json")
    got = resolver.resolve_semantic_intent(
        current_message="something for later",
        module_hint="style",
    )
    assert got["response_mode"] == "clarification"
    assert got["requires_clarification"] is True


def test_unsupported_mode_is_rejected():
    assert resolver.validate_semantic_decision(
        _decision(response_mode="style_this")
    ) is None


def test_model_cannot_change_anchor_or_locked_item_truth():
    assert resolver.validate_semantic_decision(
        _decision(anchor_item_id="belt123")
    ) is None
    assert resolver.validate_semantic_decision(
        _decision(resolved_context={"anchor_item_id": "dress456"})
    ) is None


def test_module_chat_uses_semantic_decision_and_echoes_request_id(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_generate_text",
        lambda *args, **kwargs: json.dumps(_decision()),
    )
    monkeypatch.setattr(
        chat,
        "_module_llm_response",
        lambda **kwargs: {
            "message_text": "Try a relaxed layer and clean sneakers.",
            "response": "Try a relaxed layer and clean sneakers.",
        },
    )
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/module-chat",
        json={
            "domain": "style",
            "message": "something like that, but more casual",
            "history": [{"role": "assistant", "content": "Look two"}],
            "context_data": {"last_style_context": "look two"},
            "request_id": "req_route_1",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["response_mode"] == "text_only"
    assert body["request_id"] == "req_route_1"
    assert body["style_boards"] == []


def test_obvious_style_request_uses_deterministic_fast_path(monkeypatch):
    def fail_model(*args, **kwargs):
        raise AssertionError("obvious Style requests must not call the model")

    monkeypatch.setattr(resolver, "_generate_text", fail_model)
    monkeypatch.setattr(
        chat,
        "_module_llm_response",
        lambda **kwargs: {"message_text": "Style advice."},
    )
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/module-chat",
        json={
            "domain": "style",
            "message": "Give me style tips",
            "request_id": "req_fast_1",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["domain"] == "style"
    assert body["intent"] == "advice"
    assert body["response_mode"] == "text_only"
    assert body["request_id"] == "req_fast_1"
    assert body["style_boards"] == []


def test_calendar_fast_path_returns_navigation_contract():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/module-chat",
        json={"domain": "style", "message": "calendar", "request_id": "req_cal_1"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["domain"] == "calendar"
    assert body["intent"] == "navigate"
    assert body["action"] == "open_calendar"
    assert body["response_mode"] == "calendar_navigation"
    assert body["open_module"]["route"] == "calendar"
    assert body["request_id"] == "req_cal_1"


def test_module_chat_invalid_semantics_returns_clarification(monkeypatch):
    monkeypatch.setattr(resolver, "_generate_text", lambda *args, **kwargs: "{")
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/module-chat",
        json={
            "domain": "style",
            "message": "something for later",
            "request_id": "req_route_2",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["response_mode"] == "clarification"
    assert body["request_id"] == "req_route_2"
    assert body["cards"] == []
    assert body["style_boards"] == []
