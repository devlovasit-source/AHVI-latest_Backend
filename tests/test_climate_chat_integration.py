"""Cross-feature integration tests: Session A (chat / profile personalization
/ owned-item anchoring) coexisting with Session B (Climate Metadata V1).

Climate metadata is additive garment description only. These tests prove:
- a wardrobe item's `style_metadata.climate_profile` never breaks or changes
  chat/style/anchor behavior that has nothing to do with climate
- explicit execution and advice-vs-execution routing are unaffected
- named-item anchor resolution is unaffected
- climate_profile has not been wired into any ranking/consumption path yet

No production code is changed here — these are read-only integration checks.
"""
import os
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chat
from services.style_item_contract import resolve_owned_item_mentions
from services.wardrobe_intelligence_service import build_climate_profile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _climate_wardrobe_item(item_id, name, category):
    return {
        "$id": item_id,
        "id": item_id,
        "item_id": item_id,
        "name": name,
        "category": category,
        "role": category.lower(),
        "source": "wardrobe",
        "style_metadata": {
            "climate_profile": build_climate_profile({"name": name, "category": category}),
            "climate_profile_version": "v1",
        },
    }


def _legacy_wardrobe_item(item_id, name, category):
    # No style_metadata at all -- the pre-Session-B shape every existing
    # wardrobe row has until backfilled.
    return {
        "$id": item_id,
        "id": item_id,
        "item_id": item_id,
        "name": name,
        "category": category,
        "role": category.lower(),
        "source": "wardrobe",
    }


LEGACY_TOP = _legacy_wardrobe_item("red-top", "Red Top", "Tops")
CLIMATE_JACKET = _climate_wardrobe_item("blue-jacket", "Blue Puffer Jacket", "Outerwear")


def _collapsed(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


@pytest.fixture(autouse=True)
def _clear_chat_cache():
    chat._CHAT_CACHE.clear()
    yield
    chat._CHAT_CACHE.clear()


def _authed_module_chat_client():
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def _capture_chat_completion(monkeypatch, captured, reply="Warm earthy tones will suit you."):
    def _fake(messages, **kwargs):
        captured["system_instruction"] = kwargs.get("system_instruction") or (
            messages[0]["content"] if messages else ""
        )
        captured["user_profile"] = kwargs.get("user_profile")
        return reply

    monkeypatch.setattr(chat, "chat_completion", _fake)


def _fail_if_board_requested(*args, **kwargs):
    raise AssertionError("advice query must never reach wardrobe board construction")


# ---------------------------------------------------------------------------
# 1. Legacy item (no climate_profile) + style advice query -> advice still works.
# ---------------------------------------------------------------------------


def test_legacy_wardrobe_item_style_advice_still_works(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {"user_id": user_id, "skinTone": 3, "onboarding1": True},
    )
    _capture_chat_completion(monkeypatch, captured)
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "What colour suits my skin tone?",
            "history": [],
            "context_data": {"wardrobe": [LEGACY_TOP]},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert response.json()["style_boards"] == []
    assert "#E8A87C" in captured["system_instruction"]


# ---------------------------------------------------------------------------
# 2. Item WITH climate_profile + normal Style Me execution -> board behavior
#    (routing, payload shape, wardrobe passed through) is unchanged. Climate
#    metadata must not become a consumer here.
# ---------------------------------------------------------------------------


def test_climate_enabled_item_style_me_execution_unchanged(monkeypatch):
    captured = {}

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile, **kwargs):
        captured["query"] = query_text
        captured["request_wardrobe"] = request_wardrobe
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
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "Style me to look taller",
            "history": [],
            "context_data": {"wardrobe": [CLIMATE_JACKET]},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert captured.get("query") == "Style me to look taller"
    # The climate-bearing item reached the board-construction boundary intact
    # -- not stripped, not crashed on, not specially branched.
    assert isinstance(captured.get("request_wardrobe"), list)
    assert any(item.get("$id") == "blue-jacket" for item in captured["request_wardrobe"])
    passed_item = next(i for i in captured["request_wardrobe"] if i.get("$id") == "blue-jacket")
    assert passed_item["style_metadata"]["climate_profile"]["insulation"] == ["likely_insulated", 1, "d"]


# ---------------------------------------------------------------------------
# 3. Named-item resolution against an item containing climate_profile ->
#    same item ID resolves, fixed-anchor behavior unchanged.
# ---------------------------------------------------------------------------


def test_named_item_resolution_unaffected_by_climate_profile():
    wardrobe = [CLIMATE_JACKET, LEGACY_TOP]
    result = resolve_owned_item_mentions("Create an outfit using my Blue Puffer Jacket", wardrobe)
    assert result["ambiguous"] == []
    assert result["unresolved"] == []
    assert len(result["resolved"]) == 1
    hit = result["resolved"][0]
    assert hit["item_id"] == "blue-jacket"
    assert hit["match_type"] == "exact"


# ---------------------------------------------------------------------------
# 4. Skin-tone advice question while wardrobe records contain climate_profile
#    -> text advice, no unsolicited board.
# ---------------------------------------------------------------------------


def test_skin_tone_advice_no_board_with_climate_enabled_wardrobe(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        chat,
        "_ahvi_resolve_effective_user_profile",
        lambda user_id, user_profile=None: {"user_id": user_id, "skinTone": 3, "onboarding1": True},
    )
    _capture_chat_completion(monkeypatch, captured)
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "style",
            "message": "What colour suits my skin tone?",
            "history": [],
            "context_data": {"wardrobe": [CLIMATE_JACKET]},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert response.json()["style_boards"] == []
    prompt = captured["system_instruction"]
    assert "#E8A87C" in prompt
    assert "saved shade information only" in _collapsed(prompt)


# ---------------------------------------------------------------------------
# 5. "I am 5feet 8inches how will I look taller" while climate metadata
#    exists -> text advice, no board.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["style", "wardrobe"])
def test_body_proportion_advice_no_board_with_climate_metadata_present(monkeypatch, module):
    monkeypatch.setattr(chat, "resolve_semantic_intent", lambda **kwargs: None)
    monkeypatch.setattr(chat, "_demo_style_board_payload", _fail_if_board_requested)
    monkeypatch.setattr(chat, "chat_completion", lambda messages, **kwargs: "Focus on vertical lines.")
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": module,
            "message": "I am 5feet 8inches how will I look taller",
            "history": [],
            "context_data": {"wardrobe": [CLIMATE_JACKET]},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body.get("style_boards") in ([], None)


# ---------------------------------------------------------------------------
# 6. Explicit "Use my wardrobe to make me look taller" with a climate-enabled
#    wardrobe -> execution remains authorized.
# ---------------------------------------------------------------------------


def test_explicit_use_my_wardrobe_execution_authorized_with_climate_metadata():
    assert (
        chat._has_positive_style_board_intent(
            "Use my wardrobe to make me look taller", "wardrobe"
        )
        is True
    )


def test_explicit_use_my_wardrobe_execution_reaches_board_construction(monkeypatch):
    captured = {}

    def fake_style_payload(*, user_id, query_text, request_wardrobe, user_profile, **kwargs):
        captured["query"] = query_text
        captured["request_wardrobe"] = request_wardrobe
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
    client = _authed_module_chat_client()

    response = client.post(
        "/api/chat/module-chat",
        json={
            "module": "wardrobe",
            "message": "Use my wardrobe to make me look taller",
            "history": [],
            "context_data": {"wardrobe": [CLIMATE_JACKET]},
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    assert captured.get("query") == "Use my wardrobe to make me look taller"


# ---------------------------------------------------------------------------
# 7. Saving/re-enriching a wardrobe item with climate_profile must not alter
#    any chat/style conversation state.
# ---------------------------------------------------------------------------


def test_climate_persistence_does_not_touch_chat_conversation_state(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    chat._CHAT_CACHE.clear()
    chat._CHAT_CACHE.set("some-conversation-key", {"turn": 1})
    snapshot_before = dict(chat._CHAT_CACHE._data)

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})

    persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Blue Puffer Jacket", "category": "Outerwear"},
        explicit_material="polyester",
    )

    snapshot_after = dict(chat._CHAT_CACHE._data)
    assert snapshot_after == snapshot_before
    assert chat._CHAT_CACHE.get("some-conversation-key") == {"turn": 1}
    chat._CHAT_CACHE.clear()


# ---------------------------------------------------------------------------
# 8. climate_profile remains unused by Daily Wear / Style ranking / Pack &
#    Plan / Qdrant in this integration (static scope audit, automated).
# ---------------------------------------------------------------------------

_ALLOWED_CLIMATE_PROFILE_FILES = {
    os.path.join("services", "wardrobe_intelligence_service.py"),
    os.path.join("services", "agent_metadata_validator.py"),
    os.path.join("services", "wardrobe_persistence_service.py"),
    os.path.join("scripts", "backfill_style_metadata.py"),
    os.path.join("tests", "test_climate_metadata.py"),
    os.path.join("tests", "test_climate_chat_integration.py"),
    os.path.join("tests", "test_release_smoke_chat_climate.py"),
}

_FORBIDDEN_CONSUMER_FILES = (
    os.path.join("brain", "ml", "outfit_ranker.py"),
    os.path.join("brain", "engines", "style_scorer.py"),
    os.path.join("services", "qdrant_service.py"),
)


def test_climate_profile_string_only_appears_in_the_wardrobe_metadata_boundary():
    hits = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules", ".venv")]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, REPO_ROOT)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except OSError:
                continue
            if "climate_profile" in text:
                hits.append(rel)

    unexpected = sorted(set(hits) - _ALLOWED_CLIMATE_PROFILE_FILES)
    assert unexpected == []


def test_climate_profile_absent_from_known_consumer_files():
    for rel in _FORBIDDEN_CONSUMER_FILES:
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        assert "climate_profile" not in text, rel
