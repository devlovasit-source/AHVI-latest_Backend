"""Regression coverage for the "I am 5feet 8inches how will I look taller"
device failure: a body-proportion / style-ADVICE question was misrouted into
wardrobe Style BOARD generation.

Root cause (proved by direct trace, not guesswork):
routers.chat._is_explicit_style_request() authorized a board from the bare
substring "look" (and "outfit") appearing anywhere in the message whenever
module_context was "style"/"wardrobe" -- with no requirement for an actual
creation/action verb. That is "TALKING ABOUT STYLE" (mentioning the verb "to
look") being misread as "REQUESTING STYLE EXECUTION". It is the fallback
signal module-chat's board-authorization gate (_has_positive_style_board_intent,
used by _module_chat_impl) consults whenever the upstream semantic-intent
layer fails to explicitly veto (LLM misclassification, or provider failure --
resolve_semantic_intent returning None). _COMPLETE_OUTFIT_CTA_PHRASES had the
same bare "what should i wear" defect via _is_generate_style_board_request.

The fix defers the *ambiguous* vocabulary ("outfit", "look", "what should i
wear"/"what to wear"/"what do i wear") to the EXISTING deterministic
classify_style_mode() classifier: only when it resolves to a narrow
body/color-advice topic (body_proportion_advice / color_advice /
color_body_advice -- never the generic catch-all "style_advice", which is
still too broad to safely veto on) does bare vocabulary stop being enough to
authorize a board. Unambiguous imperative phrasing ("style me", "style
this", "build/create/make an outfit", "use my wardrobe", "another look") is
checked first and always wins, so explicit execution requests are unaffected.
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat

PHYSICAL_QUERY = "I am 5feet 8inches how will I look taller"

ADVICE_QUERIES = [
    PHYSICAL_QUERY,
    "How can I dress to look taller?",
    "What should I wear to look taller?",
    "How do I look taller in clothes?",
    "I'm 5'8, what proportions will make me look taller?",
    "Do wide-leg trousers make me look shorter?",
]

EXECUTION_QUERIES = [
    "Style me to look taller",
    "Create an outfit that makes me look taller",
    "Use my wardrobe to make me look taller",
]


# ---------------------------------------------------------------------------
# Unit-level: the actual board-authorization gate, across every surface
# module-chat consults it from (style / wardrobe / default-home context).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ADVICE_QUERIES)
@pytest.mark.parametrize("module", ["style", "wardrobe", "", "home", "chat"])
def test_body_proportion_advice_never_authorizes_board(query, module):
    assert chat._has_positive_style_board_intent(query, module or "style") is False


@pytest.mark.parametrize("query", EXECUTION_QUERIES)
def test_explicit_execution_still_authorizes_board(query):
    assert chat._has_positive_style_board_intent(query, "style") is True


# ---------------------------------------------------------------------------
# Existing regression-protected wedding/casual/brunch board triggers must be
# completely unaffected (proves rule #2: explicit execution still wins, and
# the fix did not regress ordinary occasion-board generation).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "I am attending an Indian wedding",
        "Give me something casual but polished",
        "What should I wear to a wedding?",
        "Style me for a wedding",
        "Give me a casual outfit",
        "Style me casually",
        "What should I wear for brunch?",
        "I have a client meeting",
    ],
)
def test_existing_occasion_board_triggers_unaffected(query):
    assert chat._is_explicit_style_request(query, "style") is True


# ---------------------------------------------------------------------------
# Route-level: the actual physical reproduction, module-chat surface.
# resolve_semantic_intent is mocked to return None -- the documented
# "provider failure" fallback case (routers/chat.py's own comment: "_text_
# semantic unavailable/provider failure -> _text_board_veto stays False") --
# so this proves the fix holds even when the upstream semantic layer cannot
# veto at all, not just when it happens to classify correctly.
# ---------------------------------------------------------------------------


def _module_chat_client():
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def _fail_if_board_requested(*args, **kwargs):
    raise AssertionError(
        "body-proportion advice query must never reach wardrobe board construction"
    )


@pytest.mark.parametrize("module", ["style", "wardrobe"])
def test_physical_query_module_chat_no_board_when_semantic_provider_unavailable(
    monkeypatch, module
):
    monkeypatch.setattr(chat, "resolve_semantic_intent", lambda **kwargs: None)
    monkeypatch.setattr(chat, "_demo_style_board_payload", _fail_if_board_requested)
    monkeypatch.setattr(
        chat, "chat_completion", lambda messages, **kwargs: "Focus on vertical lines."
    )
    client = _module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": module,
            "message": PHYSICAL_QUERY,
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body.get("style_boards") in ([], None) or body.get("style_boards") == []
    assert body.get("cards") in ([], None) or body.get("cards") == []


def test_physical_query_module_chat_no_board_when_semantic_provider_misclassifies(
    monkeypatch,
):
    # The other documented failure mode: the semantic LLM does answer, but
    # gets it wrong (classifies an advice question as a wardrobe
    # recommendation instead of failing outright).
    monkeypatch.setattr(
        chat,
        "resolve_semantic_intent",
        lambda **kwargs: {
            "domain": "style",
            "intent": "recommendation",
            "action": "recommend_wardrobe",
            "response_mode": "wardrobe_recommendation",
        },
    )
    monkeypatch.setattr(chat, "_demo_style_board_payload", _fail_if_board_requested)
    monkeypatch.setattr(
        chat, "chat_completion", lambda messages, **kwargs: "Focus on vertical lines."
    )
    client = _module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": PHYSICAL_QUERY,
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body.get("style_boards") in ([], None) or body.get("style_boards") == []


def test_explicit_style_me_execution_still_reaches_board_construction(monkeypatch):
    captured = {}

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile, **kwargs):
        captured["query"] = query_text
        return {
            "success": True,
            "type": "cards",
            "message": "Look ready.",
            "cards": [{"id": "look-1", "items": []}],
            "style_boards": [{"id": "look-1", "items": []}],
            "chips": [],
            "board_ids": "look-1",
            "data": {"outfits": [{"id": "look-1"}], "rendered_boards": []},
            "meta": {"mode": "style_flow_service_adapter_v1"},
        }

    monkeypatch.setattr(chat, "resolve_semantic_intent", lambda **kwargs: None)
    monkeypatch.setattr(chat, "_demo_style_board_payload", fake_style_payload)
    client = _module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "Style me to look taller",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert captured.get("query") == "Style me to look taller"


# ---------------------------------------------------------------------------
# Route-level: /api/text surface for the same physical query. Deterministic
# classify_style_mode() already resolves this to body_proportion_advice
# (confirmed independent of any LLM), which routes to the text-advice
# response inside _text_chat_impl before any board-authorization check is
# reachable -- this proves the /api/text surface never needed the fix to
# begin with, only module-chat's fallback path did.
# ---------------------------------------------------------------------------


def _text_chat_client():
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
            "stylist_reasoning": "Vertical lines elongate the frame.",
            "principles": ["Vertical lines elongate."],
            "do": ["Wear a defined waist with a longer trouser break."],
            "avoid": ["Boxy silhouettes."],
            "outfit_examples": ["Wrap top with straight trousers."],
            "what_to_avoid": ["Overly busy prints."],
            "confidence": 0.9,
        }
    )


def test_physical_query_text_chat_no_board(monkeypatch):
    monkeypatch.setattr(chat, "resolve_semantic_intent", lambda **kwargs: None)
    monkeypatch.setattr(
        "services.style_reasoning_engine.generate_text", _fake_body_proportion_reasoning
    )
    monkeypatch.setattr(chat, "_demo_style_board_payload", _fail_if_board_requested)
    client = _text_chat_client()

    response = client.post(
        "/api/text",
        json={
            "module_context": "style",
            "messages": [{"role": "user", "content": PHYSICAL_QUERY}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body.get("style_boards") == []
