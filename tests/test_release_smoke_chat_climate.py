"""Release-candidate smoke gate for the chat-personalization + climate-
metadata integration (091d551). Not a substitute for the full focused suites
(tests/test_climate_metadata.py, tests/test_climate_chat_integration.py,
Session A's own suites) -- this is the small, high-signal set an RC audit
checks before proposing a release branch: auth boundary, advice-vs-execution
routing, named-item anchoring, and climate metadata basics all still work
together, end to end, from a fresh worktree.
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat
from services.style_item_contract import resolve_owned_item_mentions
from services.wardrobe_intelligence_service import (
    build_climate_profile,
    get_climate_property,
    merge_climate_profile,
    user_confirmed_material_tuple,
)


@pytest.fixture(autouse=True)
def _clear_chat_cache():
    chat._CHAT_CACHE.clear()
    yield
    chat._CHAT_CACHE.clear()


def _authed_client_router_only():
    """Isolated-router client, matching the pattern the existing chat test
    suites already use for functional (non-auth) coverage."""
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


# ---------------------------------------------------------------------------
# AUTH -- the real, whole-app auth_guard_middleware boundary (main.py), not
# the router-only test fixture. This is the one thing the isolated-router
# pattern used everywhere else in this suite cannot prove.
# ---------------------------------------------------------------------------


def test_auth_required_rejects_unauthenticated_text_request(monkeypatch):
    import main as app_main

    # Force the setting rather than trust the ambient AUTH_REQUIRED env var --
    # otherwise this test is only hermetic on machines/CI that happen to have
    # it unset (default True) or explicitly "true".
    monkeypatch.setattr(app_main.settings, "auth_required", True)
    client = TestClient(app_main.app, raise_server_exceptions=False)
    response = client.post(
        "/api/text",
        json={"module_context": "style", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401


def test_auth_guard_passes_through_when_get_current_user_succeeds(monkeypatch):
    import main as app_main

    monkeypatch.setattr(app_main.settings, "auth_required", True)

    async def fake_get_current_user(request):
        return {"user_id": "user-1"}

    monkeypatch.setattr(app_main, "get_current_user", fake_get_current_user)
    client = TestClient(app_main.app, raise_server_exceptions=False)
    response = client.post(
        "/api/text",
        json={"module_context": "style", "messages": [{"role": "user", "content": "hi"}]},
    )
    # The auth gate itself must not be what blocks this request once identity
    # resolves -- whatever happens downstream (no live Appwrite/Gemini creds
    # in this environment) is out of scope for an auth smoke check.
    assert response.status_code != 401


# ---------------------------------------------------------------------------
# STYLE ADVICE -- text, never an unsolicited board.
# ---------------------------------------------------------------------------


def test_skin_tone_advice_text_no_board(monkeypatch):
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {"user_id": user_id, "skinTone": 3, "onboarding1": True},
    )
    monkeypatch.setattr(chat, "chat_completion", lambda messages, **kwargs: "Warm earthy tones suit you.")
    client = _authed_client_router_only()

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


def test_body_proportion_advice_text_no_board(monkeypatch):
    monkeypatch.setattr(chat, "resolve_semantic_intent", lambda **kwargs: None)
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not build a board for advice")),
    )
    monkeypatch.setattr(chat, "chat_completion", lambda messages, **kwargs: "Focus on vertical lines.")
    client = _authed_client_router_only()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "I am 5feet 8inches how will I look taller",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )
    assert response.status_code == 200
    assert response.json().get("style_boards") in ([], None)


# ---------------------------------------------------------------------------
# STYLE EXECUTION -- explicit ask authorizes a board.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    ["Style me to look taller", "Use my wardrobe to make me look taller"],
)
def test_explicit_execution_authorized(query):
    assert chat._has_positive_style_board_intent(query, "wardrobe") is True


def test_explicit_style_me_reaches_board_construction(monkeypatch):
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
    client = _authed_client_router_only()

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
# WARDROBE QUERY -- a purely factual/informational question must not
# authorize a board even though bare "wardrobe" vocabulary is present. The
# veto lives in _blocks_new_style_board once the semantic layer classifies
# the query as informational (the same mechanism Session A's own advice
# tests exercise for color/body-proportion advice).
# ---------------------------------------------------------------------------


def test_factual_wardrobe_query_does_not_authorize_board_once_classified_informational():
    decision = {"intent": "information", "response_mode": "text_only"}
    assert chat._blocks_new_style_board(decision) is True


def test_factual_wardrobe_query_module_chat_no_board(monkeypatch):
    monkeypatch.setattr(
        chat,
        "resolve_semantic_intent",
        lambda **kwargs: {"domain": "wardrobe", "intent": "information", "response_mode": "text_only"},
    )
    monkeypatch.setattr(
        chat,
        "_demo_style_board_payload",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("factual query must not build a board")),
    )
    monkeypatch.setattr(chat, "chat_completion", lambda messages, **kwargs: "You have 4 tops in your wardrobe.")
    client = _authed_client_router_only()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "wardrobe",
            "message": "How many tops do I have?",
            "history": [],
            "context_data": {},
            "user_profile": {},
        },
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# NAMED ITEM -- exact owned-item resolution; fixed anchor / complete outfit
# preserved (climate_profile presence does not change any of this).
# ---------------------------------------------------------------------------

_WARDROBE = [
    {"id": "red-top", "item_id": "red-top", "name": "Red Top", "role": "top", "source": "wardrobe"},
    {
        "id": "blue-jacket",
        "item_id": "blue-jacket",
        "name": "Blue Puffer Jacket",
        "role": "top",
        "source": "wardrobe",
        "style_metadata": {
            "climate_profile": build_climate_profile({"name": "Blue Puffer Jacket", "category": "Outerwear"}),
            "climate_profile_version": "v1",
        },
    },
]


def test_named_item_exact_resolution_with_and_without_climate_profile():
    result = resolve_owned_item_mentions("Create an outfit using my Red Top", _WARDROBE)
    assert [r["item_id"] for r in result["resolved"]] == ["red-top"]

    result2 = resolve_owned_item_mentions("Create an outfit using my Blue Puffer Jacket", _WARDROBE)
    assert [r["item_id"] for r in result2["resolved"]] == ["blue-jacket"]


def test_generic_core_missing_treats_complete_outfit_as_complete():
    complete_items = [
        {"role": "top", "name": "Red Top"},
        {"role": "bottom", "name": "Blue Jeans"},
        {"role": "footwear", "name": "White Sneakers"},
    ]
    assert chat._generic_core_missing(complete_items) == []


def test_generic_core_missing_flags_incomplete_outfit():
    incomplete_items = [{"role": "top", "name": "Red Top"}]
    missing = chat._generic_core_missing(incomplete_items)
    assert "footwear" in missing
    assert "bottom" in missing


# ---------------------------------------------------------------------------
# CLIMATE METADATA -- release smoke slice (full depth lives in
# tests/test_climate_metadata.py, 63 tests).
# ---------------------------------------------------------------------------


def test_climate_legacy_item_reads_as_unknown_not_missing():
    # A pre-Session-B record has no climate_profile key at all.
    assert get_climate_property(None, "material") == ["unknown", 0, "x"]
    assert get_climate_property({}, "material") == ["unknown", 0, "x"]


def test_climate_new_item_profile_generated():
    profile = build_climate_profile({"name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"})
    assert profile["insulation"] == ["likely_insulated", 1, "d"]


def test_climate_explicit_apparel_material_persists(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})
    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Linen Shirt", "category": "Tops"},
        explicit_material="linen",
    )
    meta = json.loads(payload["style_metadata"])
    assert meta["climate_profile"]["material"] == ["linen", 3, "u"]


def test_climate_explicit_footwear_material_persists(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})
    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Suede Loafers", "category": "Footwear", "sub_category": "Loafers"},
        explicit_material="suede",
    )
    meta = json.loads(payload["style_metadata"])
    assert meta["climate_profile"]["material"] == ["suede", 3, "u"]


def test_climate_user_material_survives_enrichment():
    existing = {"material": ["linen", 3, "u"]}
    profile = build_climate_profile(
        {"name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"},
        existing_profile=existing,
    )
    assert profile["material"] == ["linen", 3, "u"]
    assert profile["insulation"] == ["likely_insulated", 1, "d"]


def test_climate_equal_authority_automated_evidence_is_stable():
    existing = {"fabric_weight": ["light", 2, "v"]}
    incoming = {"fabric_weight": ["medium", 2, "v"]}
    merged = merge_climate_profile(existing, incoming)
    assert merged["fabric_weight"] == ["light", 2, "v"]


# ---------------------------------------------------------------------------
# LEGACY -- climate_profile absence never fails existing wardrobe/chat flows.
# ---------------------------------------------------------------------------


def test_legacy_item_no_style_metadata_flows_through_chat_unbroken(monkeypatch):
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {"user_id": user_id, "skinTone": 3, "onboarding1": True},
    )
    monkeypatch.setattr(chat, "chat_completion", lambda messages, **kwargs: "Warm earthy tones suit you.")
    client = _authed_client_router_only()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "What colour suits my skin tone?",
            "history": [],
            "context_data": {"wardrobe": [{"id": "red-top", "name": "Red Top", "category": "Tops"}]},
            "user_profile": {},
        },
    )
    assert response.status_code == 200
