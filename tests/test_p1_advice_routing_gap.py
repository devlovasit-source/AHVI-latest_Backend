"""P1 (RC3): the routing gap behind height/body-type advice missing bullets.

Physical evidence: the height query hit AHVI_SEMANTIC_DECISION
decision_source=legacy_special_flow (the LLM-based semantic pre-classifier
returned nothing usable for this phrasing), which falls through to
_module_chat_impl's final generic _module_llm_response(...) call --
previously called WITHOUT is_advice=True, so STYLE_ADVICE_FORMAT_CONTRACT was
never sent to the model at all for this exact phrasing. The pre-existing
test_style_advice_bullet_format.py suite never caught this because it calls
_module_llm_response directly with is_advice already set by hand, bypassing
_module_chat_impl's routing decision entirely.

These tests exercise the real /api/chat/module-chat dispatch with no
semantic pre-classification available (the environment has no live LLM),
matching the exact production failure mode.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat
import services.llm_service as llm_service
import services.ai_gateway as ai_gateway
import services.semantic_intent_resolver as semantic_intent_resolver
import brain.intent_engine as intent_engine


@pytest.fixture(autouse=True)
def _fast_fail_llm(monkeypatch):
    """Same rationale as tests/test_unified_chat_context.py: this environment
    has no live Ollama/Gemini, so an unmocked call pays a real connect-timeout
    wait per call. Fail fast instead -- every call site already treats a
    generate_text exception as "no confident LLM answer" and falls back
    deterministically, so this doesn't change behavior, only speed."""

    def _raise(*args, **kwargs):
        raise RuntimeError("llm disabled in test")

    monkeypatch.setattr(llm_service, "generate_text", _raise)
    monkeypatch.setattr(ai_gateway, "generate_text", _raise)
    monkeypatch.setattr(semantic_intent_resolver, "_generate_text", _raise)
    monkeypatch.setattr(intent_engine, "generate_text", _raise)


def _client():
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def _post_module_chat(client, message: str):
    return client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": message,
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )


def _capture_system_instruction(monkeypatch, captured):
    def _fake_chat_completion(messages, *, system_instruction="", **kwargs):
        captured["system_instruction"] = system_instruction
        return "placeholder answer"

    monkeypatch.setattr(chat, "chat_completion", _fake_chat_completion)


def test_height_query_gets_advice_format_contract_via_real_dispatch(monkeypatch):
    captured = {}
    _capture_system_instruction(monkeypatch, captured)
    client = _client()

    r = _post_module_chat(client, "I am 5feet 8inches how will I look taller")

    assert r.status_code == 200
    assert chat.STYLE_ADVICE_FORMAT_CONTRACT in captured.get("system_instruction", ""), (
        "height advice never received the bullet-format instruction -- the "
        "legacy_special_flow fallback did not infer is_advice=True"
    )


def test_body_type_query_gets_advice_format_contract_via_real_dispatch(monkeypatch):
    captured = {}
    _capture_system_instruction(monkeypatch, captured)
    client = _client()

    r = _post_module_chat(client, "What will suit my body type?")

    assert r.status_code == 200
    assert chat.STYLE_ADVICE_FORMAT_CONTRACT in captured.get("system_instruction", "")


def test_dress_to_look_taller_gets_advice_format_contract_via_real_dispatch(monkeypatch):
    captured = {}
    _capture_system_instruction(monkeypatch, captured)
    client = _client()

    r = _post_module_chat(client, "How can I dress to look taller?")

    assert r.status_code == 200
    assert chat.STYLE_ADVICE_FORMAT_CONTRACT in captured.get("system_instruction", "")


def test_style_me_to_look_taller_stays_execution_not_advice_text(monkeypatch):
    """Regression guard: the fallback-routing fix must not swallow the
    execution phrasing that already worked (it never reaches the fallback
    at all -- it's claimed earlier by the board-authorized branch)."""
    called = {"board_payload": False}

    def _fake_board_payload(*args, **kwargs):
        called["board_payload"] = True
        return {"success": True, "cards": [], "message": "styled"}

    monkeypatch.setattr(chat, "_demo_style_board_payload", _fake_board_payload)
    monkeypatch.setattr(
        chat, "_apply_style_compliance_gate", lambda payload, **kwargs: payload
    )
    client = _client()

    r = _post_module_chat(client, "Style me to look taller")

    assert r.status_code == 200
    assert called["board_payload"] is True, (
        "execution phrasing must still reach board generation, not the "
        "advice-text fallback"
    )


def test_how_many_jackets_do_i_own_stays_non_advice(monkeypatch):
    """Regression guard: a factual/count question must not accidentally pick
    up advice formatting just because it also falls through to the generic
    fallback."""
    captured = {}
    _capture_system_instruction(monkeypatch, captured)
    client = _client()

    r = _post_module_chat(client, "How many jackets do I own?")

    assert r.status_code == 200
    assert chat.STYLE_ADVICE_FORMAT_CONTRACT not in captured.get("system_instruction", "")
