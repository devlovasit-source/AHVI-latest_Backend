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
        # Board-safety candidate ConstrainedOutfitBuilder requires for a
        # fixed/locked item (services.style_board_image_readiness).
        "masked_url": f"https://x/{iid}-masked.png",
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


# ── Positive-execution-only authorization: semantic resolver fail-open ─────
#
# request-changes review round 2, item 2: interpreted_occasion must never
# authorize a board by itself, even if the semantic veto is unavailable
# (resolve_semantic_intent returns None -- provider failure, timeout, etc).
# visual_context is the sole authority; occasion only supplies context AFTER
# execution is already authorized.

@pytest.mark.parametrize("module_context", ["wardrobe", "style"])
def test_pooja_advice_stays_no_board_when_semantic_intent_unavailable(monkeypatch, module_context):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    monkeypatch.setattr(chat, "resolve_semantic_intent", lambda *a, **k: None)
    client = _client()

    r = _post_text(
        client,
        module_context=module_context,
        message="What kind of outfit is appropriate to wear to a Pooja?",
        wardrobe=_ROUTING_TEST_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    assert not _entered_board_pipeline(body), (
        f"occasion alone authorized a board under semantic-resolver failure: "
        f"module_context={module_context} body={body}"
    )


@pytest.mark.parametrize("module_context", ["wardrobe", "style"])
def test_pooja_execution_still_authorized_when_semantic_intent_unavailable(monkeypatch, module_context):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    monkeypatch.setattr(chat, "resolve_semantic_intent", lambda *a, **k: None)
    client = _client()

    r = _post_text(
        client,
        module_context=module_context,
        message="Style me for a Pooja",
        wardrobe=_ROUTING_TEST_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    assert _entered_board_pipeline(body), (
        f"a genuine execution request was blocked by semantic-resolver failure: "
        f"module_context={module_context} body={body}"
    )


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


# ─────────────────────────────────────────────────────────────────────────
# FIXED-ANCHOR COMPLETENESS
#
# P1 follow-up: a fixed anchor with no fillable supporting slots must never
# ship as a "complete" partial board (e.g. a lone top with no bottom/
# footwear). See routers/chat.py:_ahvi_construct_board_around_fixed_items.
# ─────────────────────────────────────────────────────────────────────────

WHITE_SNEAKERS = _it("White Sneakers", "footwear")
BLACK_TROUSERS = _it("Black Trousers", "bottom")

_RED_TOP_ONLY_WARDROBE = [RED_TOP]
_RED_TOP_COMPLETE_WARDROBE = [RED_TOP, BLACK_TROUSERS, WHITE_SNEAKERS]
_TWO_ANCHOR_COMPLETE_WARDROBE = [RED_TOP, LEOPARD_SKIRT, WHITE_SNEAKERS]


def test_fixed_anchor_with_no_supporting_items_is_rejected_not_partial_board(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context="wardrobe",
        message="Create an outfit using my Red Top",
        wardrobe=_RED_TOP_ONLY_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("success") is not True, (
        f"a lone fixed anchor with no eligible bottom/footwear was returned as a "
        f"successful board. body={body}"
    )
    assert not body.get("cards"), f"expected no cards for an incomplete outfit. body={body}"
    # The fixed anchor must still be acknowledged (found), not silently dropped.
    fixed_ids = (body.get("data") or {}).get("fixed_item_ids") or []
    assert RED_TOP["item_id"] in fixed_ids, (
        f"Red Top should be reported as the honored-but-insufficient anchor. body={body}"
    )


def test_fixed_anchor_with_full_supporting_wardrobe_returns_complete_board(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context="wardrobe",
        message="Create an outfit using my Red Top",
        wardrobe=_RED_TOP_COMPLETE_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    ids = _card_item_ids(body)
    roles = {
        item.get("role")
        for card in body.get("cards") or []
        for item in card.get("items") or []
        if isinstance(item, dict)
    }
    assert RED_TOP["item_id"] in ids, f"Red Top was not preserved. body={body}"
    assert "bottom" in roles and "footwear" in roles, (
        f"expected a complete outfit (top+bottom+footwear). roles={roles} body={body}"
    )


def test_two_fixed_anchors_with_full_supporting_wardrobe_returns_complete_board(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context="style",
        message="Style me using my Red Top and Leopard Print Skirt",
        wardrobe=_TWO_ANCHOR_COMPLETE_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    ids = _card_item_ids(body)
    roles = {
        item.get("role")
        for card in body.get("cards") or []
        for item in card.get("items") or []
        if isinstance(item, dict)
    }
    assert RED_TOP["item_id"] in ids and LEOPARD_SKIRT["item_id"] in ids, (
        f"both fixed anchors must be preserved. ids={ids} body={body}"
    )
    assert "footwear" in roles, f"expected footwear to complete the look. roles={roles} body={body}"


def test_invalid_final_outfit_around_fixed_item_is_rejected_not_returned(monkeypatch):
    """request-changes review round 2, item 1: if the outfit constructed
    around a fixed anchor would be occasion/safety-invalid, it must be
    rejected outright -- never silently returned as a valid board."""
    monkeypatch.setattr(chat, "reject_board_for_occasion", lambda card, occasion: (True, "test_forced_rejection"))
    result = chat._ahvi_construct_board_around_fixed_items(
        [RED_TOP], _OWNED_ITEM_WARDROBE, "daily", "Create an outfit using my Red Top", "user-1",
    )
    assert result["success"] is False
    assert result["cards"] == []
    assert result["reason"] == "occasion_incompatible"
    assert "red-top" not in _card_item_ids(result)


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


def test_mixed_owned_and_unowned_multi_item_no_silent_bottom_substitution(monkeypatch):
    """"my Red Top and Purple Skirt" -- Red Top is owned, Purple Skirt is not.
    Must not silently swap in Leopard Print Skirt / Blue Jeans for the unowned
    "Purple Skirt" and ship a board as if the request had been satisfied."""
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context="wardrobe",
        message="Style me using my Red Top and Purple Skirt",
        wardrobe=_OWNED_ITEM_WARDROBE,
    )
    assert r.status_code == 200
    body = r.json()
    ids = _card_item_ids(body)
    assert "leopard-print-skirt" not in ids
    assert "blue-jeans" not in ids
    msg = str(body.get("message") or "").lower()
    assert "purple skirt" in msg
    assert not ids or ids == {"red-top"}, f"unexpected silent substitution: ids={ids} body={body}"


def test_ampersand_conjunction_resolves_both_named_items(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    client = _client()

    r = _post_text(
        client,
        module_context="style",
        message="Style me using my Red Top & Leopard Print Skirt",
        wardrobe=_OWNED_ITEM_WARDROBE,
    )
    assert r.status_code == 200
    ids = _card_item_ids(r.json())
    assert RED_TOP["item_id"] in ids and LEOPARD_SKIRT["item_id"] in ids


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


# ─────────────────────────────────────────────────────────────────────────
# LEGACY WARDROBE OWNERSHIP PROVENANCE (P0: named-anchor "trusted source")
#
# Regression: a real Appwrite wardrobe document never carries a top-level
# `source` field (routers/wardrobe_capture.py never writes one), so
# canonical_item_source() reported "unknown" for every item fetched through
# the authenticated wardrobe path, and ConstrainedOutfitBuilder rejects an
# "unknown"-source fixed anchor outright ("A fixed item must include a
# trusted source."). Fix: _fetch_wardrobe_for_style stamps ownership
# provenance on rows it fetches from the user's own Appwrite collection
# (services.style_item_contract.stamp_wardrobe_ownership_source), mirroring
# the same trust boundary routers.stylist._resolve_style_this_anchor already
# established. A client-supplied `wardrobe` payload is deliberately NOT
# stamped -- ownership must come from the authenticated fetch, never from
# caller-supplied JSON.
# ─────────────────────────────────────────────────────────────────────────


def _legacy_appwrite_doc(name: str, role: str, item_id: str, **extra) -> Dict[str, Any]:
    """Shape of a real Appwrite wardrobe document: no top-level `source`
    field at all -- the exact physical shape that reproduced the P0
    trusted-source regression for legacy items like "Black Trousers"."""
    doc = {
        "id": item_id,
        "item_id": item_id,
        "name": name,
        "category": role,
        "image_url": f"https://x/{item_id}.png",
        "masked_url": f"https://x/{item_id}-masked.png",
    }
    doc.update(extra)
    return doc


def _fake_appwrite_with_docs(docs):
    class _Proxy:
        def list_documents(self, collection, user_id=None, limit=100, offset=0):
            return [] if offset else list(docs)
    return _Proxy


LEGACY_WHITE_TEE = _legacy_appwrite_doc("White Tee", "top", "legacy-white-tee")
LEGACY_BLUE_JEANS = _legacy_appwrite_doc("Blue Jeans", "bottom", "legacy-blue-jeans")
LEGACY_BLACK_TROUSERS = _legacy_appwrite_doc("Black Trousers", "bottom", "legacy-black-trousers")
LEGACY_BLACK_FLIP_FLOPS = _legacy_appwrite_doc("Black Flip-Flops", "footwear", "legacy-black-flip-flops")
LEGACY_WHITE_SNEAKERS = _legacy_appwrite_doc("White Sneakers", "footwear", "legacy-white-sneakers")

_LEGACY_COMPLETE_WARDROBE = [
    LEGACY_WHITE_TEE, LEGACY_BLUE_JEANS, LEGACY_BLACK_TROUSERS,
    LEGACY_BLACK_FLIP_FLOPS, LEGACY_WHITE_SNEAKERS,
]


def test_legacy_black_trousers_anchor_resolves_and_builds_board(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _fake_appwrite_with_docs(_LEGACY_COMPLETE_WARDROBE))

    result = chat._demo_style_board_payload(
        "user-1", "Create an outfit using my Black Trousers", None, resolved_occasion="daily",
    )
    ids = _card_item_ids(result)
    assert result.get("success") is True, f"legacy item anchor should build a board. result={result}"
    assert LEGACY_BLACK_TROUSERS["item_id"] in ids, f"anchor item not preserved. ids={ids} result={result}"


def test_legacy_black_flip_flops_anchor_resolves_and_builds_board(monkeypatch):
    monkeypatch.setattr(chat, "AppwriteProxy", _fake_appwrite_with_docs(_LEGACY_COMPLETE_WARDROBE))

    result = chat._demo_style_board_payload(
        "user-1", "Create an outfit using my Black Flip-Flops", None, resolved_occasion="daily",
    )
    ids = _card_item_ids(result)
    assert result.get("success") is True, f"legacy item anchor should build a board. result={result}"
    assert LEGACY_BLACK_FLIP_FLOPS["item_id"] in ids, f"anchor item not preserved. ids={ids} result={result}"


def test_legacy_anchor_with_climate_profile_is_unaffected_by_evidence_source(monkeypatch):
    """Climate Metadata V1 evidence-authority source codes (u/v/d/m/x) live
    under item['climate_profile'][field]['source'] -- a different namespace
    from the top-level item['source'] ownership field. A climate_profile
    carrying an evidence source must never be read as, or interfere with,
    wardrobe ownership trust."""
    item_with_climate = _legacy_appwrite_doc(
        "Black Trousers", "bottom", "legacy-black-trousers-climate",
        climate_profile={"material": {"value": "cotton", "confidence": 0.9, "source": "v"}},
    )
    wardrobe = [LEGACY_WHITE_TEE, item_with_climate, LEGACY_WHITE_SNEAKERS]
    monkeypatch.setattr(chat, "AppwriteProxy", _fake_appwrite_with_docs(wardrobe))

    result = chat._demo_style_board_payload(
        "user-1", "Create an outfit using my Black Trousers", None, resolved_occasion="daily",
    )
    ids = _card_item_ids(result)
    assert result.get("success") is True, f"climate_profile must not block anchor trust. result={result}"
    assert item_with_climate["item_id"] in ids, f"anchor item not preserved. ids={ids} result={result}"


def test_client_supplied_wardrobe_item_without_source_remains_untrusted(monkeypatch):
    """Security: ownership trust must come ONLY from the authenticated
    server-side wardrobe fetch, never from a caller-supplied `wardrobe`
    payload. An item with no `source` in a client-supplied list must still
    be rejected as an untrusted fixed anchor -- and the rejection message
    must be product-safe, never the raw internal validator string."""
    monkeypatch.setattr(chat, "AppwriteProxy", _FakeAppwriteProxy)
    untrusted_item = {
        "id": "injected-item", "item_id": "injected-item", "name": "Injected Blazer",
        "category": "outerwear", "image_url": "https://x/injected.png",
        "masked_url": "https://x/injected-masked.png",
    }
    client = _client()

    r = _post_text(
        client,
        module_context="wardrobe",
        message="Create an outfit using my Injected Blazer",
        wardrobe=[untrusted_item],
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("success") is not True, (
        f"a client-supplied unknown-source item became a trusted anchor. body={body}"
    )
    msg = str(body.get("message") or "")
    assert "trusted source" not in msg.lower(), f"internal validator string leaked to chat. body={body}"
    assert body.get("reason") == "unknown_item_source", f"expected typed reason code. body={body}"
