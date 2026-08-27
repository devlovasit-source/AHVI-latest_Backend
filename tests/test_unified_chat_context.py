"""Characterization tests for the unified chat intent + owned-item anchor fix.

Root cause under test (see routers/chat.py):
  - `module_context == "wardrobe"` alone currently sets `visual_context`/
    `style_intent_candidate` True regardless of message content, and the one
    veto mechanism (`_text_board_veto`, via `semantic_intent_resolver`) is
    structurally disabled for module_hint="wardrobe" on any non-deterministic
    turn -- so a wardrobe-surface message with no positive execution signal
    can still be forced into a board.
  - There is no first-turn resolver that matches a named owned item ("Red
    Top") against the authenticated wardrobe before generic candidate
    ranking runs, so a named item can be silently substituted or a valid
    two-item request can be refused.

These tests characterize CURRENT behavior before any production code change.
Some are expected to FAIL today by design (that failure is the bug); they
are re-run after the fix lands and must then pass. Only the LLM/network
boundary and the Appwrite wardrobe-count fetch are mocked -- routing logic
itself runs for real.
"""
from __future__ import annotations

from typing import Any, Dict, List

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
    """This environment has no live Ollama/Gemini. Fail fast instead of paying
    real connect-timeout retries per call -- every call site already treats a
    generate_text exception as "no confident LLM answer" and falls back
    deterministically (see detect_intent's except-then-fallback and
    resolve_semantic_intent's except-then-None), so this does not change what
    is being characterized, only how long it takes."""

    def _raise(*args, **kwargs):
        raise RuntimeError("llm disabled in test")

    monkeypatch.setattr(llm_service, "generate_text", _raise)
    monkeypatch.setattr(ai_gateway, "generate_text", _raise)
    monkeypatch.setattr(semantic_intent_resolver, "_generate_text", _raise)
    monkeypatch.setattr(intent_engine, "generate_text", _raise)


def _client():
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def _it(name: str, role: str, source: str = "wardrobe") -> Dict[str, Any]:
    iid = name.lower().replace(" ", "-")
    return {
        "id": iid,
        "item_id": iid,
        "name": name,
        "category": role,
        "role": role,
        "source": source,
        "image_url": f"https://x/{iid}.png",
    }


class _FakeAppwriteProxy:
    """Stub for the server-side wardrobe fetch used by `_fast_wardrobe_count_response`.
    Deliberately ignores any client-sent `wardrobe` -- that path always hits
    Appwrite directly (see routers/chat.py:_fast_wardrobe_count_response)."""

    _DOCS = [
        {"id": "jacket-1", "name": "Blue Jacket", "category": "jacket"},
        {"id": "jacket-2", "name": "Black Bomber Jacket", "category": "jacket"},
        {"id": "shoes-1", "name": "Black Sneakers", "category": "shoes"},
    ]

    def list_documents(self, collection, user_id=None, limit=100, offset=0):
        if offset:
            return []
        return list(self._DOCS)


# A complete 3-slot wardrobe so that IF the routing bug forces a request into
# the board pipeline, it can actually produce cards (a clear, unambiguous
# signal) rather than a "missing slots" response that could be mistaken for
# "no board was attempted."
_ROUTING_TEST_WARDROBE = [_it("Blue Shirt", "top"), _it("Black Trousers", "bottom"), _it("Brown Loafers", "footwear")]


def _entered_board_pipeline(body: Dict[str, Any]) -> bool:
    """True if the request was routed into the style-board-generation pipeline
    at all -- whether it ultimately produced cards or failed for missing
    items. This is the real, unmockable signal (`build_style_flow_response`
    stamps `meta.analysis_source=style_flow_service`); it does NOT depend on
    which internal function reached that pipeline, so it catches both the
    `_demo_style_board_payload` fast-route and the `style_reasoning_engine`
    orchestrator route."""
    meta = body.get("meta") or {}
    if meta.get("analysis_source") == "style_flow_service":
        return True
    if body.get("cards"):
        return True
    if str(body.get("type") or "") in {"cards", "missing_core_wardrobe_slots", "missing_outfit_cards"}:
        return True
    return False


def _post_text(client, *, module_context: str, message: str, wardrobe=None):
    payload = {
        "module_context": module_context,
        "include_base64": False,
        "messages": [{"role": "user", "content": message}],
    }
    if wardrobe is not None:
        payload["wardrobe"] = wardrobe
    return client.post("/api/text", json=payload)


# ─────────────────────────────────────────────────────────────────────────
# WARDROBE QUERY
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_context", ["wardrobe", "style"])
def test_how_many_jackets_do_i_own_is_text_not_board(monkeypatch, module_context):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context=module_context,
        message="How many jackets do I own?",
        wardrobe=_ROUTING_TEST_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    assert not _entered_board_pipeline(body), (
        f"module_context={module_context}: wardrobe count query was routed into "
        f"the style-board pipeline. body={body}"
    )


def test_how_many_black_shoes_do_i_own_is_text_not_board_control_case(monkeypatch):
    """Control case: known-working today (uses the word 'shoes', which IS in
    the fast-path vocabulary). Both modules must stay text."""
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    for module_context in ("wardrobe", "style"):
        r = _post_text(
            client,
            module_context=module_context,
            message="How many black shoes do I own?",
            wardrobe=_ROUTING_TEST_WARDROBE,
        )
        assert r.status_code == 200
        assert not _entered_board_pipeline(r.json())


# ─────────────────────────────────────────────────────────────────────────
# GENERAL STYLE ADVICE
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_context", ["wardrobe", "style"])
def test_body_type_advice_is_not_a_board(monkeypatch, module_context):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context=module_context,
        message="What will suit my body type?",
        wardrobe=_ROUTING_TEST_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    assert not _entered_board_pipeline(body), f"module_context={module_context} body={body}"


@pytest.mark.parametrize("module_context", ["wardrobe", "style"])
def test_skin_tone_advice_is_not_a_board(monkeypatch, module_context):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context=module_context,
        message="What will suit my skin tone?",
        wardrobe=_ROUTING_TEST_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    assert not _entered_board_pipeline(body), f"module_context={module_context} body={body}"


@pytest.mark.parametrize("module_context", ["wardrobe", "style"])
def test_pooja_occasion_advice_is_not_a_board(monkeypatch, module_context):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context=module_context,
        message="What kind of outfit is appropriate to wear to a Pooja?",
        wardrobe=_ROUTING_TEST_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    assert not _entered_board_pipeline(body), f"module_context={module_context} body={body}"


# ─────────────────────────────────────────────────────────────────────────
# STYLE EXECUTION
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_context", ["wardrobe", "style"])
def test_style_me_for_a_pooja_is_board_authorized(monkeypatch, module_context):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context=module_context,
        message="Style me for a Pooja",
        wardrobe=_ROUTING_TEST_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    assert _entered_board_pipeline(body), f"module_context={module_context} body={body}"


# ─────────────────────────────────────────────────────────────────────────
# OWNED ITEM EXECUTION / NO-MATCH / AMBIGUOUS
#
# These exercise the REAL board-generation pipeline (no _demo_style_board_payload
# stub) with a small controlled wardrobe, since we need to inspect which items
# actually land in the board -- not just whether a board was produced.
# ─────────────────────────────────────────────────────────────────────────

RED_TOP = _it("Red Top", "top")
WHITE_SHIRT = _it("White Shirt", "top")
LEOPARD_SKIRT = _it("Leopard Print Skirt", "bottom")
BLUE_JEANS = _it("Blue Jeans", "bottom")
LOAFERS = _it("Brown Loafers", "footwear")
BLACK_SHIRT_1 = _it("Black Shirt", "top", source="wardrobe")
BLACK_SHIRT_1["id"] = BLACK_SHIRT_1["item_id"] = "black-shirt-1"
BLACK_SHIRT_2 = _it("Black Shirt", "top", source="wardrobe")
BLACK_SHIRT_2["id"] = BLACK_SHIRT_2["item_id"] = "black-shirt-2"

_OWNED_ITEM_WARDROBE = [RED_TOP, WHITE_SHIRT, LEOPARD_SKIRT, BLUE_JEANS, LOAFERS]


def _card_item_ids(body: Dict[str, Any]) -> set:
    ids = set()
    for card in body.get("cards") or []:
        for item in card.get("items") or []:
            if isinstance(item, dict):
                iid = item.get("item_id") or item.get("id")
                if iid:
                    ids.add(iid)
    return ids


def test_create_outfit_using_red_top_preserves_exact_item(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context="wardrobe",
        message="Create an outfit using my Red Top",
        wardrobe=_OWNED_ITEM_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    ids = _card_item_ids(body)
    assert RED_TOP["item_id"] in ids, (
        f"Red Top was not preserved in the generated board (Case C characterization). "
        f"card item ids={ids} body={body}"
    )


def test_style_me_using_red_top_and_leopard_skirt_preserves_both(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context="style",
        message="Style me using my Red Top and Leopard Print Skirt",
        wardrobe=_OWNED_ITEM_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    ids = _card_item_ids(body)
    assert RED_TOP["item_id"] in ids and LEOPARD_SKIRT["item_id"] in ids, (
        f"Both named items were not preserved together (Case D characterization). "
        f"card item ids={ids} body={body}"
    )


def test_no_match_owned_item_is_explicit_not_silent_substitution(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context="wardrobe",
        message="Style me using my Purple Spacesuit",
        wardrobe=_OWNED_ITEM_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    ids = _card_item_ids(body)
    # None of the *other* owned items should be silently substituted in as if
    # "Purple Spacesuit" had been satisfied.
    msg = str(body.get("message") or "").lower()
    mentions_not_found = "purple spacesuit" in msg and (
        "can't find" in msg or "cannot find" in msg or "don't have" in msg or "no purple" in msg
    )
    assert mentions_not_found, (
        f"expected an explicit not-found response for an item the user does not own; "
        f"got message={body.get('message')!r} ids={ids}"
    )


def test_ambiguous_owned_item_name_asks_for_clarification(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()
    wardrobe = [BLACK_SHIRT_1, BLACK_SHIRT_2, BLUE_JEANS, LOAFERS]

    r = _post_text(
        client,
        module_context="wardrobe",
        message="Create an outfit using my Black Shirt",
        wardrobe=wardrobe,
    )
    assert r.status_code == 200
    body = r.json()
    ids = _card_item_ids(body)
    picked_one_arbitrarily = bool(ids & {"black-shirt-1", "black-shirt-2"})
    assert not picked_one_arbitrarily, (
        f"an ambiguous same-name item was silently resolved instead of asking for "
        f"clarification. ids={ids} body={body}"
    )


# ─────────────────────────────────────────────────────────────────────────
# CROSS-SURFACE PARITY (resolved intent, not just board/no-board)
# ─────────────────────────────────────────────────────────────────────────

def test_red_top_resolves_to_same_item_id_both_modules(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    ids_by_module = {}
    for module_context in ("style", "wardrobe"):
        r = _post_text(
            client,
            module_context=module_context,
            message="Create an outfit using my Red Top",
            wardrobe=_OWNED_ITEM_WARDROBE,
        )
        assert r.status_code == 200
        ids_by_module[module_context] = _card_item_ids(r.json())

    assert RED_TOP["item_id"] in ids_by_module["style"]
    assert RED_TOP["item_id"] in ids_by_module["wardrobe"]
