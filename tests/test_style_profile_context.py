"""Focused tests for saved-profile personalization in casual Style advice.

Covers the bounded personal-style-profile helper (services/style_context_service.py
:compact_personal_style_profile) and its wiring into the two casual-advice LLM
paths: routers/chat.py's module-chat text_only responder (_module_llm_response)
and services/style_reasoning_engine.py's advice-mode prompt builder.

Verified frontend contract under test:
- skinTone is a 1-based swatch index (NOT an undertone) mapped through the
  exact verified kSkinTones hex palette.
- skinTone/bodyShape are only trusted when onboarding1 == true.
- stylePreferences is only trusted when onboarding2 == true.
"""
import json
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _collapsed(text):
    """Collapse whitespace/newlines so multi-line prompt prose can be matched
    with a plain substring check regardless of line-wrap position."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()

from routers import chat
from services.style_context_service import compact_personal_style_profile


@pytest.fixture(autouse=True)
def _clear_chat_response_cache():
    # /api/text caches responses by (message, user_id, ...); several tests here
    # intentionally reuse the same reproduction phrase, so a stale hit would
    # skip the LLM call and falsely appear to pass/fail.
    chat._CHAT_CACHE.clear()
    yield
    chat._CHAT_CACHE.clear()

_SWATCH_HEX = {
    1: "#FDDBB4",
    2: "#F5C6A0",
    3: "#E8A87C",
    4: "#C68642",
    5: "#8D5524",
    6: "#4A2912",
    7: "#2C1A0E",
    8: "#1A0D07",
}


# ---------------------------------------------------------------------------
# Unit-level: compact_personal_style_profile() acceptance rules
# ---------------------------------------------------------------------------


def test_skin_tone_hidden_without_onboarding1_flag():
    assert "skin_tone" not in compact_personal_style_profile({"skinTone": 3})


def test_skin_tone_hidden_when_onboarding1_explicitly_false():
    out = compact_personal_style_profile({"skinTone": 3, "onboarding1": False})
    assert "skin_tone" not in out


@pytest.mark.parametrize("index,hex_value", sorted(_SWATCH_HEX.items()))
def test_skin_tone_translates_through_verified_palette(index, hex_value):
    out = compact_personal_style_profile({"skinTone": index, "onboarding1": True})
    assert out["skin_tone"] == {"swatch_hex": hex_value, "undertone": "unknown"}


@pytest.mark.parametrize("bad_value", [0, 9, -1, "not-a-number", None, ""])
def test_skin_tone_invalid_or_default_collision_index_is_unknown(bad_value):
    out = compact_personal_style_profile({"skinTone": bad_value, "onboarding1": True})
    assert "skin_tone" not in out


def test_skin_tone_raw_index_never_appears_as_the_semantic_value():
    out = compact_personal_style_profile({"skinTone": 3, "onboarding1": True})
    # The default-collision index (3) must never surface bare anywhere in the
    # bounded context -- only the translated hex swatch is prompt-safe.
    assert "3" not in json.dumps(out)
    assert out["skin_tone"]["swatch_hex"] == "#E8A87C"


def test_undertone_is_always_unknown_never_inferred_from_swatch():
    for index in _SWATCH_HEX:
        out = compact_personal_style_profile({"skinTone": index, "onboarding1": True})
        assert out["skin_tone"]["undertone"] == "unknown"


def test_body_shape_hidden_without_onboarding1_flag():
    assert "body_shape" not in compact_personal_style_profile({"bodyShape": "Hourglass"})


def test_body_shape_visible_with_onboarding1_true():
    out = compact_personal_style_profile({"bodyShape": "Hourglass", "onboarding1": True})
    assert out["body_shape"] == "Hourglass"


def test_body_shape_empty_value_is_unknown_even_with_onboarding1():
    out = compact_personal_style_profile({"bodyShape": "", "onboarding1": True})
    assert "body_shape" not in out


def test_body_shape_snake_case_alias_accepted():
    out = compact_personal_style_profile({"body_shape": "Pear", "onboarding1": True})
    assert out["body_shape"] == "Pear"


def test_style_preferences_camel_case_survives_with_onboarding2():
    out = compact_personal_style_profile(
        {"stylePreferences": ["Clean Minimal", "Soft Elegant"], "onboarding2": True}
    )
    assert out["style_preferences"] == ["Clean Minimal", "Soft Elegant"]


def test_style_preferences_snake_case_survives_with_onboarding2():
    out = compact_personal_style_profile(
        {"style_preferences": ["Streetwear", "Minimal"], "onboarding2": True}
    )
    assert out["style_preferences"] == ["Streetwear", "Minimal"]


def test_style_preferences_hidden_without_onboarding2_flag():
    out = compact_personal_style_profile({"stylePreferences": ["Clean Minimal"]})
    assert "style_preferences" not in out


def test_style_preferences_hidden_when_onboarding2_explicitly_false():
    out = compact_personal_style_profile(
        {"stylePreferences": ["Clean Minimal"], "onboarding2": False}
    )
    assert "style_preferences" not in out


def test_bounded_profile_never_includes_unrelated_account_fields():
    out = compact_personal_style_profile(
        {
            "skinTone": 3,
            "bodyShape": "Hourglass",
            "stylePreferences": ["Minimal"],
            "onboarding1": True,
            "onboarding2": True,
            "phone": "+1-555-0100",
            "dob": "1990-01-01",
            "email": "user@example.com",
            "$id": "doc-1",
            "$createdAt": "2024-01-01",
        }
    )
    dumped = json.dumps(out)
    for leaked in ("555-0100", "1990-01-01", "user@example.com", "doc-1"):
        assert leaked not in dumped
    assert set(out.keys()) <= {"skin_tone", "body_shape", "style_preferences", "style_dna"}


# ---------------------------------------------------------------------------
# Integration: module-chat text_only path (_module_llm_response)
# ---------------------------------------------------------------------------

_ADVICE_REPLY = "Try grounding the look in one clean anchor piece and a simple palette."


def _authed_module_chat_client(user_id="user-1"):
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": user_id}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def _capture_chat_completion(monkeypatch, captured, reply=_ADVICE_REPLY):
    def _fake(messages, *, system_instruction="", **kwargs):
        captured["system_instruction"] = system_instruction
        captured["user_profile"] = kwargs.get("user_profile")
        return reply

    monkeypatch.setattr(chat, "chat_completion", _fake)


def test_module_chat_color_advice_uses_saved_skin_tone_swatch(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {
            "user_id": user_id,
            "skinTone": 3,
            "onboarding1": True,
        },
    )
    _capture_chat_completion(monkeypatch, captured)
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "What colour suits my skin tone?",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert response.json()["style_boards"] == []
    prompt = captured["system_instruction"]
    # This is the ACTUAL boundary Gemini sees for this route: module-chat's
    # _handle_preclassified -> _module_llm_response. It does not go through
    # style_reasoning_engine, so the semantic guard must live here too.
    assert "#E8A87C" in prompt
    assert "saved shade information only" in _collapsed(prompt)
    for banned in ("warm undertone", "cool undertone", "neutral undertone"):
        assert banned not in prompt.lower()


def test_module_chat_color_advice_without_onboarding1_hides_skin_tone(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {"user_id": user_id, "skinTone": 3},
    )
    _capture_chat_completion(monkeypatch, captured)
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "What colour suits my skin tone?",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert '"personal_style_profile"' not in captured["system_instruction"]
    assert "#E8A87C" not in captured["system_instruction"]


def test_module_chat_advice_intent_carries_saved_style_preferences(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {
            "user_id": user_id,
            "stylePreferences": ["Clean Minimal", "Soft Elegant"],
            "onboarding2": True,
        },
    )
    _capture_chat_completion(monkeypatch, captured)
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "Give me style tips",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert response.json()["style_boards"] == []
    assert "Clean Minimal" in captured["system_instruction"]
    assert "Soft Elegant" in captured["system_instruction"]


def test_module_chat_style_preferences_without_onboarding2_not_claimed(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {
            "user_id": user_id,
            "stylePreferences": ["Clean Minimal"],
        },
    )
    _capture_chat_completion(monkeypatch, captured)
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "Give me style tips",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert "Clean Minimal" not in captured["system_instruction"]


def test_module_chat_greeting_never_triggers_persisted_profile_fetch(monkeypatch):
    def _fail_fetch(user_id, user_profile=None):
        raise AssertionError("greeting must not resolve the persisted Style profile")

    monkeypatch.setattr(chat, "_ahvi_resolve_effective_user_profile", _fail_fetch)
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "hi",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )
    assert response.status_code == 200


def test_module_chat_information_intent_never_triggers_persisted_profile_fetch(monkeypatch):
    def _fail_fetch(user_id, user_profile=None):
        raise AssertionError("generic information must not resolve the persisted Style profile")

    monkeypatch.setattr(chat, "_ahvi_resolve_effective_user_profile", _fail_fetch)
    captured = {}
    _capture_chat_completion(monkeypatch, captured)
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "What is a capsule wardrobe?",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )
    assert response.status_code == 200


def test_module_chat_profile_never_leaks_phone_dob_email(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {
            "user_id": user_id,
            "skinTone": 4,
            "bodyShape": "Rectangle",
            "onboarding1": True,
            "phone": "+1-555-0199",
            "dob": "1990-01-01",
            "email": "user@example.com",
        },
    )
    _capture_chat_completion(monkeypatch, captured)
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "What colour suits my skin tone?",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )
    assert response.status_code == 200
    for leaked in ("555-0199", "1990-01-01", "user@example.com"):
        assert leaked not in captured["system_instruction"]


def test_module_chat_cross_user_skin_tone_isolation(monkeypatch):
    import services.data_access_service as data_access_service

    profiles = {
        "user-a": {"skinTone": 1, "onboarding1": True},
        "user-b": {"skinTone": 8, "onboarding1": True},
    }
    monkeypatch.setattr(
        data_access_service,
        "get_user_profile",
        lambda *, user_id: dict(profiles.get(user_id, {})),
    )
    calls = []

    def _fake(messages, *, system_instruction="", **kwargs):
        calls.append(system_instruction)
        return _ADVICE_REPLY

    monkeypatch.setattr(chat, "chat_completion", _fake)

    for user_id in ("user-a", "user-b"):
        client = _authed_module_chat_client(user_id=user_id)
        response = client.post(
            "/api/chat/module-chat",
            json={
                "module": "style",
                "message": "What colour suits my skin tone?",
                "history": [],
                "context_data": {},
                "user_profile": {},
            },
        )
        assert response.status_code == 200

    assert "#FDDBB4" in calls[0] and "#1A0D07" not in calls[0]
    assert "#1A0D07" in calls[1] and "#FDDBB4" not in calls[1]


# ---------------------------------------------------------------------------
# Integration: /api/text style_reasoning_engine advice-mode prompt path
# ---------------------------------------------------------------------------


def _authed_text_client():
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def _fake_body_proportion_reasoning(prompt, **kwargs):
    return json.dumps(
        {
            "mode": "body_proportion_advice",
            "stylist_reasoning": "Grounded in your saved profile.",
            "principles": ["Vertical lines elongate."],
            "do": ["Wear a defined waist."],
            "avoid": ["Boxy silhouettes."],
            "outfit_examples": ["Wrap top with straight trousers."],
            "what_to_avoid": ["Overly busy prints."],
            "confidence": 0.9,
        }
    )


def test_text_chat_body_shape_reaches_advice_prompt_with_onboarding1(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {
            "user_id": user_id,
            "bodyShape": "Hourglass",
            "onboarding1": True,
        },
    )

    def _fake_generate_text(prompt, **kwargs):
        captured["prompt"] = prompt
        return _fake_body_proportion_reasoning(prompt, **kwargs)

    monkeypatch.setattr("services.style_reasoning_engine.generate_text", _fake_generate_text)
    client = _authed_text_client()

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "What will suit my body type?"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["style_boards"] == []
    assert "Hourglass" in captured["prompt"]
    assert "saved profile setting" in _collapsed(captured["prompt"])
    assert "not as a measured physical fact" in _collapsed(captured["prompt"])
    # Treated as a saved setting, never asserted as physical fact.
    assert "the user is definitely hourglass" not in _collapsed(captured["prompt"])


def test_text_chat_body_shape_absent_without_onboarding1(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {"user_id": user_id, "bodyShape": "Hourglass"},
    )

    def _fake_generate_text(prompt, **kwargs):
        captured["prompt"] = prompt
        return _fake_body_proportion_reasoning(prompt, **kwargs)

    monkeypatch.setattr("services.style_reasoning_engine.generate_text", _fake_generate_text)
    client = _authed_text_client()

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "What will suit my body type?"}],
        },
    )

    assert response.status_code == 200
    assert "Hourglass" not in captured["prompt"]


def test_text_chat_color_advice_prompt_carries_swatch_not_raw_index(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {
            "user_id": user_id,
            "skinTone": 5,
            "onboarding1": True,
        },
    )

    def _fake_generate_text(prompt, **kwargs):
        captured["prompt"] = prompt
        return json.dumps(
            {
                "mode": "color_advice",
                "stylist_reasoning": "Grounded in your saved shade.",
                "recommended_colors": ["olive", "rust"],
                "avoid_colors": ["neon"],
                "why": ["Complements the saved shade."],
                "outfit_palettes": ["olive + cream"],
                "what_to_avoid": ["Overly busy prints."],
                "confidence": 0.9,
            }
        )

    monkeypatch.setattr("services.style_reasoning_engine.generate_text", _fake_generate_text)
    client = _authed_text_client()

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "What colours suit my skin tone?"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["style_boards"] == []
    assert "#8D5524" in captured["prompt"]
    assert "never state or imply a warm/cool/neutral undertone" in _collapsed(captured["prompt"])
    for banned in ("you have warm undertones", "you have cool undertones", "you have neutral undertones"):
        assert banned not in captured["prompt"].lower()


# ---------------------------------------------------------------------------
# Privacy / auth
# ---------------------------------------------------------------------------


def test_unauthenticated_text_chat_never_resolves_profile(monkeypatch):
    def _fail_fetch(user_id, user_profile=None):
        raise AssertionError("unauthenticated request must not resolve any Style profile")

    monkeypatch.setattr(chat, "_ahvi_resolve_effective_user_profile", _fail_fetch)

    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    client = TestClient(app)

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "What colours suit my skin tone?"}],
        },
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Routing regression: casual advice must stay text-only, never a board
# ---------------------------------------------------------------------------


def test_skin_tone_question_stays_text_only_no_board(monkeypatch):
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {"user_id": user_id},
    )
    captured = {}
    _capture_chat_completion(monkeypatch, captured)
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "What colour suits my skin tone?",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["style_boards"] == []
    assert body["cards"] == []


def test_body_type_question_stays_text_only_no_board(monkeypatch):
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {"user_id": user_id},
    )
    monkeypatch.setattr(
        "services.style_reasoning_engine.generate_text", _fake_body_proportion_reasoning
    )
    client = _authed_text_client()

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": "What will suit my body type?"}],
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["style_boards"] == []
