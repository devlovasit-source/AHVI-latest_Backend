"""Not for me (outfit-level rejection) + Change it (localized replacement)
feedback-memory tests. Covers: durable signature persistence, exact-outfit
penalty vs different-combination eligibility, per-user isolation, malformed
identity fail-safety, duplicate-rejection idempotency, optional reason,
change-item from/to/role recording via the existing mutation engine, and the
Shuffle-never-writes-dislike contract. Backend wear-memory idempotency and
recommendation-repeat penalty are covered by their own existing test files
and are not duplicated here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from brain.engines.style_scorer import (
    CHANGED_FROM_ITEM_PENALTY,
    CHANGED_TO_ITEM_BOOST,
    EXACT_REJECTED_OUTFIT_PENALTY,
    _memory_breakdown,
)
from brain.outfit_pipeline import _exclude_active_rejected_signatures
from services import style_board_shuffle_service as shuffle_service
from services.style_board_state_store import InMemoryBoardStateStore
from services.style_memory_service import (
    REJECTION_ACTIVE_DAYS,
    _outfit_signature,
    build_style_memory_context,
    load_active_rejected_outfit_signatures,
    load_outfit_change_memory,
    load_rejected_outfit_memory,
)

# Captured before any test monkeypatches it, so the wardrobe-source test can
# restore the real implementation for just itself.
from routers.style_memory import _canonical_wardrobe_candidates as _REAL_CANONICAL_WARDROBE_CANDIDATES


# ---------------------------------------------------------------------------
# Fake Qdrant — in-memory, same read/write contract as QdrantService's
# user_memory methods (list_user_memory / upsert_user_memory).
# ---------------------------------------------------------------------------
class FakeQdrant:
    """Mirrors the real QdrantService.upsert_user_memory contract exactly:
    returns True only on confirmed persistence, False on skip/failure —
    never a truthy-but-meaningless id. self.fail_upsert simulates a genuine
    Qdrant-side exception (client.upsert() raising) for tests that need to
    prove the API doesn't report false success."""

    def __init__(self):
        self.points = {}  # point_id -> payload
        self._counter = 0
        self.fail_upsert = False

    def upsert_user_memory(self, user_id, vector, payload, point_id=None):
        if not vector or not user_id or self.fail_upsert:
            return False
        self._counter += 1
        pid = point_id or f"auto-{self._counter}"
        self.points[pid] = dict(payload)
        return True

    def list_user_memory(self, user_id, memory_type, limit=50):
        rows = [
            p for p in self.points.values()
            if p.get("user_id") == user_id and p.get("memory_type") == memory_type
        ]
        rows.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
        return rows[:limit]


@pytest.fixture()
def fake_qdrant(monkeypatch):
    fake = FakeQdrant()
    # user_memory writes now use a fixed, always-non-empty sentinel vector
    # (services.qdrant_service.user_memory_sentinel_vector) instead of
    # encode_metadata() — routers/feedback.py and routers/style_memory.py no
    # longer import or call encode_metadata at all, so there is nothing left
    # to patch around the old "embeddings unavailable in test" workaround.
    # style_memory_service.py and routers/style_memory.py's memory writer
    # import qdrant_service lazily inside the function body, so patching the
    # source module also covers them.
    monkeypatch.setattr("services.qdrant_service.qdrant_service", fake)
    monkeypatch.setattr("routers.feedback.qdrant_service", fake)
    return fake


def _client(monkeypatch):
    from routers import feedback as feedback_router_module

    app = FastAPI()

    @app.middleware("http")
    async def authenticate(request, call_next):
        test_user = request.headers.get("x-test-user")
        if test_user:
            request.state.user = {"user_id": test_user}
        return await call_next(request)

    app.include_router(feedback_router_module.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# NOT FOR ME
# ---------------------------------------------------------------------------
def test_valid_rejection_persists_with_outfit_signature(fake_qdrant):
    client = _client(None)
    resp = client.post(
        "/api/feedback/board",
        json={
            "user_id": "u1",
            "action": "dislike",
            "board_payload": {"board": "b1", "type": "daily_wear"},
            "item_ids": ["shirt1", "pant1", "shoe1"],
        },
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    stored = list(fake_qdrant.points.values())
    assert len(stored) == 1
    assert stored[0]["memory_type"] == "disliked"
    assert stored[0]["outfit_signature"] == "pant1|shirt1|shoe1"
    assert set(stored[0]["item_ids"]) == {"shirt1", "pant1", "shoe1"}


def test_outfit_signature_persists_and_loads_via_style_memory(fake_qdrant):
    client = _client(None)
    client.post(
        "/api/feedback/board",
        json={
            "user_id": "u1",
            "action": "dislike",
            "board_payload": {},
            "item_ids": ["shirt1", "pant1", "shoe1"],
        },
        headers={"x-test-user": "u1"},
    )
    mem = load_rejected_outfit_memory("u1")
    assert mem["rejected_outfit_signatures"] == ["pant1|shirt1|shoe1"]


def test_exact_rejected_outfit_receives_strong_penalty():
    items = [{"id": "shirt1"}, {"id": "pant1"}, {"id": "shoe1"}]
    context = {"rejected_outfit_signatures": ["pant1|shirt1|shoe1"]}
    fields, reasons = _memory_breakdown(items, context)
    assert fields["exact_rejected_outfit_penalty"] == EXACT_REJECTED_OUTFIT_PENALTY
    assert any("passed on" in r for r in reasons)


def test_different_combination_with_shared_item_remains_eligible():
    # Rejected: shirt1 + pant1 + shoe1. Different: shirt1 + pant2 + shoe2.
    context = {"rejected_outfit_signatures": ["pant1|shirt1|shoe1"]}
    different = [{"id": "shirt1"}, {"id": "pant2"}, {"id": "shoe2"}]
    fields, _ = _memory_breakdown(different, context)
    assert fields["exact_rejected_outfit_penalty"] == 0.0


def test_user_a_rejection_does_not_affect_user_b(fake_qdrant):
    client = _client(None)
    client.post(
        "/api/feedback/board",
        json={"user_id": "u1", "action": "dislike", "board_payload": {}, "item_ids": ["shirt1", "pant1"]},
        headers={"x-test-user": "u1"},
    )
    mem_a = load_rejected_outfit_memory("u1")
    mem_b = load_rejected_outfit_memory("u2")
    assert mem_a["rejected_outfit_signatures"] == ["pant1|shirt1"]
    assert mem_b["rejected_outfit_signatures"] == []


def test_malformed_identity_fails_safe_no_signature(fake_qdrant):
    client = _client(None)
    resp = client.post(
        "/api/feedback/board",
        json={"user_id": "u1", "action": "dislike", "board_payload": {"board": "b1"}, "item_ids": []},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    stored = list(fake_qdrant.points.values())
    assert stored[0]["outfit_signature"] == ""
    # No signature means it can never contribute an exact-rejection penalty.
    mem = load_rejected_outfit_memory("u1")
    assert mem["rejected_outfit_signatures"] == []


def test_auth_rejects_cross_user_body_id(fake_qdrant):
    client = _client(None)
    resp = client.post(
        "/api/feedback/board",
        json={"user_id": "someone-else", "action": "dislike", "board_payload": {}, "item_ids": ["shirt1"]},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 403


def test_duplicate_rejection_does_not_multiply_influence(fake_qdrant):
    client = _client(None)
    payload = {
        "user_id": "u1",
        "action": "dislike",
        "board_payload": {},
        "item_ids": ["shirt1", "pant1"],
    }
    client.post("/api/feedback/board", json=payload, headers={"x-test-user": "u1"})
    client.post("/api/feedback/board", json=payload, headers={"x-test-user": "u1"})
    client.post("/api/feedback/board", json=payload, headers={"x-test-user": "u1"})
    # Deterministic point id -> same signature always overwrites, never duplicates.
    assert len(fake_qdrant.points) == 1
    mem = load_rejected_outfit_memory("u1")
    assert mem["rejected_outfit_signatures"] == ["pant1|shirt1"]


def test_optional_reason_is_persisted(fake_qdrant):
    client = _client(None)
    client.post(
        "/api/feedback/board",
        json={
            "user_id": "u1", "action": "dislike", "board_payload": {},
            "item_ids": ["shirt1"], "reason": "too_formal",
        },
        headers={"x-test-user": "u1"},
    )
    stored = list(fake_qdrant.points.values())
    assert stored[0]["reason"] == "too_formal"


def test_omitted_reason_works(fake_qdrant):
    client = _client(None)
    resp = client.post(
        "/api/feedback/board",
        json={"user_id": "u1", "action": "dislike", "board_payload": {}, "item_ids": ["shirt1"]},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    stored = list(fake_qdrant.points.values())
    assert stored[0]["reason"] is None


def test_not_for_me_never_populates_disliked_item_ids(fake_qdrant):
    """Rejecting an outfit must never blacklist its individual garments."""
    from services.style_memory_service import build_style_memory_context

    client = _client(None)
    client.post(
        "/api/feedback/board",
        json={"user_id": "u1", "action": "dislike", "board_payload": {}, "item_ids": ["shirt1", "pant1"]},
        headers={"x-test-user": "u1"},
    )
    ctx = build_style_memory_context("u1", wardrobe=[])
    assert ctx["disliked_item_ids"] == []


def test_like_action_is_unaffected_by_item_ids_extension(fake_qdrant):
    client = _client(None)
    resp = client.post(
        "/api/feedback/board",
        json={"user_id": "u1", "action": "like", "board_payload": {"board": "b1"}, "item_ids": ["shirt1"]},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    stored = list(fake_qdrant.points.values())
    assert stored[0]["memory_type"] == "liked"


def test_dislike_write_reaches_upsert_even_with_embeddings_disabled(fake_qdrant, monkeypatch):
    """Regression for the traced production defect: routers/feedback.py used
    to call encode_metadata(board_payload), which always returned [] because
    sentence-transformers is deliberately absent from requirements.txt —
    upsert_user_memory's `not vector` guard then silently skipped the write
    every time, with zero logging and an HTTP 200 response. This test proves
    the write no longer depends on the embedding model being available at
    all: even with embeddings explicitly disabled, a dislike must still
    reach (fake) Qdrant."""
    import services.embedding_service as embedding_service

    monkeypatch.setattr(embedding_service, "embeddings_enabled", lambda: False)

    client = _client(None)
    resp = client.post(
        "/api/feedback/board",
        json={"user_id": "u1", "action": "dislike", "board_payload": {}, "item_ids": ["shirt1", "pant1"]},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(fake_qdrant.points) == 1


def test_user_memory_sentinel_vector_is_512d_deterministic_and_nonzero():
    from services.qdrant_service import USER_MEMORY_VECTOR_DIMENSION, user_memory_sentinel_vector

    v1 = user_memory_sentinel_vector()
    v2 = user_memory_sentinel_vector()

    assert len(v1) == 512
    assert len(v1) == USER_MEMORY_VECTOR_DIMENSION
    assert v1 == v2  # deterministic
    assert any(component != 0 for component in v1)  # never all-zero


def test_dislike_upsert_success_returns_api_success(fake_qdrant):
    client = _client(None)
    resp = client.post(
        "/api/feedback/board",
        json={"user_id": "u1", "action": "dislike", "board_payload": {}, "item_ids": ["shirt1"]},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(fake_qdrant.points) == 1


def test_dislike_upsert_failure_does_not_return_false_success(fake_qdrant):
    """Durable persistence IS the purpose of Not-for-me. A Qdrant upsert
    failure must never be reported to the client as success=true — and must
    not leave a memory point behind either."""
    fake_qdrant.fail_upsert = True
    client = _client(None)
    resp = client.post(
        "/api/feedback/board",
        json={"user_id": "u1", "action": "dislike", "board_payload": {}, "item_ids": ["shirt1"]},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code != 200
    assert resp.status_code >= 500
    body = resp.json()
    assert "success" not in body or body["success"] is not True
    assert body["detail"]["code"] == "MEMORY_PERSISTENCE_FAILED"
    assert len(fake_qdrant.points) == 0


# ---------------------------------------------------------------------------
# SHUFFLE — must never write explicit rejection memory
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("action", ["shown", "saved", "dismissed", "regenerated", "clicked", "shared"])
def test_shuffle_and_passive_actions_never_write_dislike_memory(fake_qdrant, action):
    client = _client(None)
    resp = client.post(
        "/api/feedback/board",
        json={"user_id": "u1", "action": action, "board_payload": {"board": "b1"}, "item_ids": ["shirt1"]},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    assert fake_qdrant.points == {}
    mem = load_rejected_outfit_memory("u1")
    assert mem["rejected_outfit_signatures"] == []


# ---------------------------------------------------------------------------
# CHANGE IT
# ---------------------------------------------------------------------------
def _outfit_item(item_id, name, role):
    return {
        "item_id": item_id,
        "id": item_id,
        "name": name,
        "category": role,
        "role": role,
        "slot": role,
        "source": "wardrobe",
        "image_url": f"https://img/{item_id}.png",
    }


@pytest.fixture(autouse=True)
def _state_store():
    shuffle_service.set_state_store(InMemoryBoardStateStore())
    yield
    shuffle_service.set_state_store(None)


@pytest.fixture(autouse=True)
def default_wardrobe_candidates(monkeypatch):
    # Default candidate pool for tests that don't care about wardrobe
    # sourcing specifics. Tests proving the real Appwrite->sanitizer chain,
    # or an empty wardrobe, override this within their own body.
    monkeypatch.setattr(
        "routers.style_memory._canonical_wardrobe_candidates",
        lambda user_id: [
            _outfit_item("shoe-1", "Black Loafers", "footwear"),
            _outfit_item("shoe-3", "White Sneakers", "footwear"),
        ],
    )


def _call_change_item(**overrides):
    from routers.style_memory import ChangeItemRequest, change_item

    user_id = overrides.get("user_id", "u1")

    class _FakeRequest:
        state = type("S", (), {"user": {"user_id": user_id}})()

        class url:
            path = "/api/style/change-item"

    body = {
        "user_id": user_id,
        "board_id": "board-1",
        "revision": 1,
        "items": [
            _outfit_item("top-1", "White Shirt", "top"),
            _outfit_item("bottom-1", "Blue Jeans", "bottom"),
            _outfit_item("shoe-1", "Black Loafers", "footwear"),
        ],
        "old_item_id": "shoe-1",
        "occasion": "casual",
    }
    body.update(overrides)
    req = _FakeRequest()
    return change_item(req, ChangeItemRequest(**body))


def test_change_item_records_from_to_role(fake_qdrant):
    result = _call_change_item()
    assert result["success"] is True
    changed = result["data"]["changed_item_ids"]
    assert changed == ["shoe-3"]

    mem = load_outfit_change_memory("u1")
    assert mem["recent_changed_items"][0]["from_item_id"] == "shoe-1"
    assert mem["recent_changed_items"][0]["to_item_id"] == "shoe-3"
    assert mem["recent_changed_items"][0]["role"] == "footwear"
    assert mem["recent_changed_items"][0]["occasion"] == "casual"

    stored = list(fake_qdrant.points.values())
    assert stored[0]["role"] == "footwear"
    # Client-visible board id is restored — the internal namespaced key
    # never leaks into the persisted memory or the response.
    assert stored[0]["board_id"] == "board-1"
    assert "dailywear:" not in str(result)
    assert result["learning_persisted"] is True


def test_change_item_mutation_succeeds_even_when_memory_persistence_fails(fake_qdrant):
    """Board mutation success must never be transactional with Qdrant memory
    persistence (explicit product requirement) — a failed learning write is
    surfaced via learning_persisted=false, never a rolled-back mutation or a
    failed change-item response."""
    fake_qdrant.fail_upsert = True

    result = _call_change_item()

    assert result["success"] is True
    assert result["data"]["changed_item_ids"] == ["shoe-3"]
    assert result["learning_persisted"] is False
    # Nothing was actually persisted despite the mutation succeeding.
    assert len(fake_qdrant.points) == 0


def test_change_item_route_response_exposes_updated_board_to_frontend(fake_qdrant):
    """Route-level contract the Flutter client actually reads (daily_wear.dart
    _changeIt): result['data']['changed_item_ids'] for the id, and
    result['cards'][0]['items'] (mirrored in result['style_boards'][0]) for
    the renderable board — not just the internal mutation service's return
    value. Pins the exact shape so a future refactor can't silently break the
    adapter without a matching frontend change."""
    result = _call_change_item()
    assert result["success"] is True

    for key in ("cards", "style_boards"):
        boards = result[key]
        assert len(boards) == 1
        items = boards[0]["items"]
        item_ids = {item["id"] for item in items}
        assert "shoe-3" in item_ids
        assert "shoe-1" not in item_ids
        assert {"top-1", "bottom-1"} <= item_ids


def test_change_item_no_valid_replacement_is_typed_failure_not_silent_success(fake_qdrant, monkeypatch):
    """When the canonical wardrobe has no OTHER candidate for the role being
    changed, _choose_candidate (style_board_mutation_service.py) excludes the
    current item id from its own candidate pool and returns None, which the
    engine turns into a typed NO_VALID_REPLACEMENT failure — never a
    success=True with an empty changed_item_ids. This is what lets the
    frontend's `if (!success) { show "couldn't change" toast; return; }`
    branch be trusted: a no-op can never masquerade as success."""
    monkeypatch.setattr(
        "routers.style_memory._canonical_wardrobe_candidates",
        lambda user_id: [_outfit_item("shoe-1", "Black Loafers", "footwear")],
    )
    result = _call_change_item()

    assert result["success"] is False
    assert result["error"]["code"] == "NO_VALID_REPLACEMENT"
    assert "data" not in result
    assert "cards" not in result


def test_changed_from_signal_is_weak_and_contextual_role_and_occasion_match():
    items = [{"id": "shoe-1", "name": "shoe", "category": "footwear"}]
    context = {
        "occasion": "dinner",
        "recent_changed_items": [
            {"from_item_id": "shoe-1", "to_item_id": "shoe-3", "role": "footwear", "occasion": "dinner"}
        ],
    }
    fields, _ = _memory_breakdown(items, context)
    assert fields["changed_from_item_penalty"] == CHANGED_FROM_ITEM_PENALTY
    assert fields["changed_from_item_penalty"] > EXACT_REJECTED_OUTFIT_PENALTY  # much weaker


def test_changed_from_signal_does_not_apply_in_a_different_occasion():
    items = [{"id": "shoe-1", "name": "shoe", "category": "footwear"}]
    context = {
        "occasion": "work",
        "recent_changed_items": [
            {"from_item_id": "shoe-1", "to_item_id": "shoe-3", "role": "footwear", "occasion": "dinner"}
        ],
    }
    fields, _ = _memory_breakdown(items, context)
    assert fields["changed_from_item_penalty"] == 0.0


def test_changed_from_signal_fails_neutral_when_event_occasion_unknown():
    items = [{"id": "shoe-1", "name": "shoe", "category": "footwear"}]
    context = {
        "occasion": "dinner",
        "recent_changed_items": [
            {"from_item_id": "shoe-1", "to_item_id": "shoe-3", "role": "footwear", "occasion": ""}
        ],
    }
    fields, _ = _memory_breakdown(items, context)
    # Missing occasion on the OLD event must fail neutral, not globally penalize.
    assert fields["changed_from_item_penalty"] == 0.0


def test_selected_replacement_is_weak_positive_when_context_matches():
    items = [{"id": "shoe-3", "name": "sneaker", "category": "footwear"}]
    context = {
        "occasion": "dinner",
        "recent_changed_items": [
            {"from_item_id": "shoe-1", "to_item_id": "shoe-3", "role": "footwear", "occasion": "dinner"}
        ],
    }
    fields, _ = _memory_breakdown(items, context)
    assert fields["changed_to_item_boost"] == CHANGED_TO_ITEM_BOOST


def test_change_it_does_not_globally_blacklist_from_item(fake_qdrant):
    _call_change_item()
    # shoe-1 (the replaced item) must remain eligible elsewhere — only a
    # small, contextual per-item penalty, never removed from candidate
    # pools, and no disliked_item_ids entry is ever written by this path.
    from services.style_memory_service import build_style_memory_context

    ctx = build_style_memory_context("u1", wardrobe=[])
    assert ctx["disliked_item_ids"] == []
    assert ctx["recent_changed_items"][0]["from_item_id"] == "shoe-1"


def test_replacement_failure_does_not_create_feedback(fake_qdrant, monkeypatch):
    # old_item_id itself resolves fine (still in "wardrobe"), but there is no
    # OTHER candidate to swap in -> engine returns NO_REPLACEMENT_CANDIDATES.
    monkeypatch.setattr(
        "routers.style_memory._canonical_wardrobe_candidates",
        lambda user_id: [_outfit_item("shoe-1", "Black Loafers", "footwear")],
    )
    result = _call_change_item()
    assert result["success"] is False
    assert fake_qdrant.points == {}
    mem = load_outfit_change_memory("u1")
    assert mem["recent_changed_items"] == []


def test_change_item_requires_resolvable_role(fake_qdrant):
    with pytest.raises(HTTPException) as exc_info:
        _call_change_item(
            items=[{"id": "mystery-1", "name": "???"}],
            old_item_id="mystery-1",
        )
    assert exc_info.value.status_code == 422
    assert fake_qdrant.points == {}


def test_change_item_rejects_unknown_old_item_id(fake_qdrant):
    with pytest.raises(HTTPException) as exc_info:
        _call_change_item(old_item_id="not-in-outfit")
    assert exc_info.value.status_code == 400
    assert fake_qdrant.points == {}


# ---------------------------------------------------------------------------
# CANONICAL WARDROBE CANDIDATE SOURCE (not restricted to what's on-screen)
# ---------------------------------------------------------------------------
def test_change_item_sources_candidates_from_canonical_wardrobe_not_rendered_board(fake_qdrant, monkeypatch):
    """Board renders top1+bottom1+shoe1 with NO alternate footwear anywhere
    in the request. The user's canonical wardrobe (Appwrite `outfits`
    collection) contains shoe1 + shoe2. Changing shoe1 must still succeed,
    proving candidates come from the real wardrobe loader + sanitizer, not
    from whatever happens to be rendered on screen."""

    class FakeAppwriteProxy:
        def list_documents(self, resource, **kwargs):
            assert resource == "outfits"
            return [
                _outfit_item("shoe-1", "Black Loafers", "footwear"),
                _outfit_item("shoe-2", "Brown Boots", "footwear"),
            ]

    monkeypatch.setattr("services.appwrite_proxy.AppwriteProxy", FakeAppwriteProxy)
    # Re-point at the REAL implementation (undoing the autouse default-pool
    # stub for just this test) so this proves the actual Appwrite ->
    # sanitizer chain, not a shortcut.
    monkeypatch.setattr(
        "routers.style_memory._canonical_wardrobe_candidates",
        _REAL_CANONICAL_WARDROBE_CANDIDATES,
    )

    result = _call_change_item(
        items=[
            _outfit_item("top1", "White Shirt", "top"),
            _outfit_item("bottom1", "Blue Jeans", "bottom"),
            {"id": "shoe-1", "name": "Black Loafers", "category": "footwear"},
        ],
        old_item_id="shoe-1",
    )
    assert result["success"] is True
    assert result["data"]["changed_item_ids"] == ["shoe-2"]


# ---------------------------------------------------------------------------
# NAMESPACED MUTATION STATE + AUTHORITY
# ---------------------------------------------------------------------------
def test_cross_user_same_board_id_produces_independent_state(fake_qdrant):
    result_a = _call_change_item(user_id="user-a", board_id="board-1")
    assert result_a["success"] is True

    # user-b's request for the SAME client-visible board_id must not be
    # treated as a stale/forbidden mutation of user-a's board — it starts
    # its own independent revision history.
    result_b = _call_change_item(user_id="user-b", board_id="board-1")
    assert result_b["success"] is True
    assert result_b.get("error", {}).get("code") != "BOARD_FORBIDDEN"
    assert result_b.get("error", {}).get("code") != "STALE_BOARD_REVISION"


def _style_memory_client():
    from routers import style_memory as style_memory_router

    app = FastAPI()

    @app.middleware("http")
    async def authenticate(request, call_next):
        test_user = request.headers.get("x-test-user")
        if test_user:
            request.state.user = {"user_id": test_user}
        return await call_next(request)

    app.include_router(style_memory_router.router)
    return TestClient(app)


def test_change_item_unauthenticated_is_rejected(fake_qdrant):
    client = _style_memory_client()
    resp = client.post(
        "/api/style/change-item",
        json={
            "user_id": "u1",
            "board_id": "board-1",
            "items": [_outfit_item("shoe-1", "Black Loafers", "footwear")],
            "old_item_id": "shoe-1",
        },
    )
    assert resp.status_code == 401


def test_change_item_body_identity_cannot_override_authenticated_user(fake_qdrant):
    client = _style_memory_client()
    resp = client.post(
        "/api/style/change-item",
        json={
            "user_id": "someone-else",
            "board_id": "board-1",
            "items": [_outfit_item("shoe-1", "Black Loafers", "footwear")],
            "old_item_id": "shoe-1",
        },
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# PHASE 13 — deterministic end-to-end learning scenario
# ---------------------------------------------------------------------------
def test_deterministic_learning_scenario(fake_qdrant, monkeypatch):
    monkeypatch.setattr(
        "routers.style_memory._canonical_wardrobe_candidates",
        lambda user_id: [
            _outfit_item("shoe1", "Black Loafers", "footwear"),
            _outfit_item("shoe3", "White Sneakers", "footwear"),
        ],
    )
    client = _client(None)
    user = "learn-u1"

    # Recommendation A: shirt1 + pant1 + shoe1 -> Not for me.
    resp = client.post(
        "/api/feedback/board",
        json={
            "user_id": user, "action": "dislike", "board_payload": {},
            "item_ids": ["shirt1", "pant1", "shoe1"],
        },
        headers={"x-test-user": user},
    )
    assert resp.status_code == 200

    # Scoring A again shows a strong exact-rejection penalty.
    outfit_a = [{"id": "shirt1"}, {"id": "pant1"}, {"id": "shoe1"}]
    mem = load_rejected_outfit_memory(user)
    fields_a, _ = _memory_breakdown(outfit_a, {"rejected_outfit_signatures": mem["rejected_outfit_signatures"]})
    assert fields_a["exact_rejected_outfit_penalty"] == EXACT_REJECTED_OUTFIT_PENALTY

    # Recommendation B: shirt1 + pant2 + shoe2 -> different outfit, remains eligible.
    outfit_b = [{"id": "shirt1"}, {"id": "pant2"}, {"id": "shoe2"}]
    fields_b, _ = _memory_breakdown(outfit_b, {"rejected_outfit_signatures": mem["rejected_outfit_signatures"]})
    assert fields_b["exact_rejected_outfit_penalty"] == 0.0

    # User then sees shirt1+pant1+shoe1 again and taps Change it -> footwear,
    # replacing shoe1 with shoe3.
    result = _call_change_item(
        user_id=user,
        items=[
            _outfit_item("shirt1", "White Shirt", "top"),
            _outfit_item("pant1", "Blue Jeans", "bottom"),
            _outfit_item("shoe1", "Black Loafers", "footwear"),
        ],
        old_item_id="shoe1",
        occasion="dinner",
    )
    assert result["success"] is True
    assert result["data"]["changed_item_ids"] == ["shoe3"]

    change_mem = load_outfit_change_memory(user)
    events = change_mem["recent_changed_items"]
    assert events[0]["from_item_id"] == "shoe1"
    assert events[0]["to_item_id"] == "shoe3"
    assert events[0]["role"] == "footwear"
    assert events[0]["occasion"] == "dinner"

    # shoe1 = mild/local rejection in the SAME context (dinner, footwear);
    # shoe3 = weak selected affinity in that same context.
    fields_shoe1, _ = _memory_breakdown(
        [{"id": "shoe1", "name": "shoe", "category": "footwear"}],
        {"occasion": "dinner", "recent_changed_items": events},
    )
    fields_shoe3, _ = _memory_breakdown(
        [{"id": "shoe3", "name": "shoe", "category": "footwear"}],
        {"occasion": "dinner", "recent_changed_items": events},
    )
    assert fields_shoe1["changed_from_item_penalty"] == CHANGED_FROM_ITEM_PENALTY
    assert fields_shoe3["changed_to_item_boost"] == CHANGED_TO_ITEM_BOOST
    assert fields_shoe1["changed_from_item_penalty"] > EXACT_REJECTED_OUTFIT_PENALTY  # much milder

    # A DIFFERENT occasion (Work) must NOT inherit the changed-from penalty —
    # the signal is scoped to its own context, not a blanket opinion.
    fields_work, _ = _memory_breakdown(
        [{"id": "shoe1", "name": "shoe", "category": "footwear"}],
        {"occasion": "work", "recent_changed_items": events},
    )
    assert fields_work["changed_from_item_penalty"] == 0.0

    # shoe1 must NOT be globally unavailable: an outfit containing it (in a
    # different combination, same occasion) gets no exact-rejection penalty,
    # only the mild contextual change-from nudge — never a blackout.
    other_outfit_with_shoe1 = [
        {"id": "shirt2"}, {"id": "pant2"},
        {"id": "shoe1", "name": "shoe", "category": "footwear"},
    ]
    fields_other, _ = _memory_breakdown(
        other_outfit_with_shoe1,
        {
            "occasion": "dinner",
            "rejected_outfit_signatures": mem["rejected_outfit_signatures"],
            "recent_changed_items": events,
        },
    )
    assert fields_other["exact_rejected_outfit_penalty"] == 0.0
    assert fields_other["changed_from_item_penalty"] == CHANGED_FROM_ITEM_PENALTY


# ---------------------------------------------------------------------------
# REVISION AUTHORITY — server-resolved retry, not client-trusted revision
# ---------------------------------------------------------------------------
def test_stale_client_revision_is_auto_resolved_and_retried(fake_qdrant):
    # First change: durable revision goes 1 -> 2.
    first = _call_change_item(board_id="board-r1")
    assert first["success"] is True
    assert first["revision"] == 2

    # Simulate an app restart: the client's in-memory revision tracker is
    # gone, so it sends revision=1 again (its only known default) even
    # though the durable store is now at revision 2. The item being
    # changed this time is the NEW footwear item from the first mutation.
    to_item_from_first = first["data"]["changed_item_ids"][0]
    second = _call_change_item(
        board_id="board-r1",
        revision=1,  # stale — durable store is actually at revision 2
        items=[
            _outfit_item("top-1", "White Shirt", "top"),
            _outfit_item("bottom-1", "Blue Jeans", "bottom"),
            _outfit_item(to_item_from_first, "White Sneakers", "footwear"),
        ],
        old_item_id=to_item_from_first,
    )
    # Must succeed via the one bounded server-side retry, not surface a raw
    # STALE_BOARD_REVISION the client has no way to self-heal from, and must
    # not discard/reset the durable state built up by the first mutation.
    assert second["success"] is True
    assert second["revision"] == 3


# ---------------------------------------------------------------------------
# CLIENT METADATA IS NEVER TRUSTED — rehydrated from canonical wardrobe
# ---------------------------------------------------------------------------
def test_tampered_client_metadata_does_not_control_role_resolution(fake_qdrant, monkeypatch):
    # The user genuinely owns "shoe-1", but the client claims it's a "top"
    # named "Not Actually Shoes" — an attempt to control role resolution
    # via metadata rather than the server's own wardrobe record.
    monkeypatch.setattr(
        "routers.style_memory._canonical_wardrobe_candidates",
        lambda user_id: [
            _outfit_item("shoe-1", "Black Loafers", "footwear"),
            _outfit_item("shoe-3", "White Sneakers", "footwear"),
        ],
    )
    tampered_shoe = {
        "id": "shoe-1",
        "item_id": "shoe-1",
        "name": "Not Actually Shoes",
        "category": "top",
        "role": "top",
        "slot": "top",
        "source": "wardrobe",
        "image_url": "https://img/shoe-1.png",
    }
    result = _call_change_item(
        items=[
            _outfit_item("top-1", "White Shirt", "top"),
            _outfit_item("bottom-1", "Blue Jeans", "bottom"),
            tampered_shoe,
        ],
        old_item_id="shoe-1",
    )
    # Rehydrated from the canonical wardrobe record, this resolves as
    # footwear (its real role) and the swap succeeds against the real
    # footwear candidate — the tampered "top" claim is never used.
    assert result["success"] is True
    assert result["data"]["changed_item_ids"] == ["shoe-3"]


def test_unowned_item_id_cannot_smuggle_fabricated_metadata(fake_qdrant, monkeypatch):
    # "ghost-1" is not in the user's wardrobe at all. Even though the
    # client attaches full, well-formed metadata, an unmatched id is
    # stripped down to bare id before role resolution — it fails safe
    # (422, unresolvable role) rather than trusting fabricated metadata.
    monkeypatch.setattr(
        "routers.style_memory._canonical_wardrobe_candidates",
        lambda user_id: [_outfit_item("shoe-3", "White Sneakers", "footwear")],
    )
    fabricated = {
        "id": "ghost-1",
        "item_id": "ghost-1",
        "name": "Fabricated Loafers",
        "category": "footwear",
        "role": "footwear",
        "slot": "footwear",
        "source": "wardrobe",
        "image_url": "https://img/ghost-1.png",
    }
    with pytest.raises(HTTPException) as exc_info:
        _call_change_item(
            items=[
                _outfit_item("top-1", "White Shirt", "top"),
                _outfit_item("bottom-1", "Blue Jeans", "bottom"),
                fabricated,
            ],
            old_item_id="ghost-1",
        )
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# QDRANT PERSISTENCE LAYER — payload-index provisioning, list_user_memory
# request shape. Exercises services.qdrant_service.QdrantService directly
# with a fake low-level client (not FakeQdrant, which replaces the whole
# service and would bypass the real Filter/index-provisioning code these
# tests exist to prove).
# ---------------------------------------------------------------------------
class _FakeLowLevelQdrantClient:
    """Fakes just enough of qdrant_client.QdrantClient's surface for
    QdrantService.init()'s collection/index provisioning."""

    def __init__(self, existing_collections=(), raise_already_exists=False):
        self.created_indexes = []  # (collection, field, schema)
        self._existing = list(existing_collections)
        self.raise_already_exists = raise_already_exists

    def get_collections(self):
        Row = type("Row", (), {})
        result = Row()
        result.collections = [type("C", (), {"name": n})() for n in self._existing]
        return result

    def create_collection(self, **kwargs):
        pass

    def create_payload_index(self, collection_name, field_name, field_schema):
        if self.raise_already_exists:
            raise Exception(f"Index for {field_name} already exists")
        self.created_indexes.append((collection_name, field_name, field_schema))


def _bare_qdrant_service(client):
    """QdrantService with __init__ bypassed (no real QDRANT_URL/network) —
    just the attributes init()/list_user_memory() actually read."""
    from services.qdrant_service import QdrantService

    svc = QdrantService.__new__(QdrantService)
    svc.client = client
    svc._initialized = False
    svc.collection = "wardrobe"
    svc.image_collection = "wardrobe_images"
    svc.memory_collection = "outfit_memory"
    svc.user_memory_collection = "user_memory"
    svc.vector_size = 512
    svc.memory_vector_size = 8
    svc.user_memory_vector_size = 512
    return svc


def test_payload_index_provisioning_covers_every_field_list_user_memory_filters_on():
    from qdrant_client.models import PayloadSchemaType

    svc = _bare_qdrant_service(_FakeLowLevelQdrantClient())
    svc.init()

    user_memory_indexes = [
        (field, schema) for (collection, field, schema) in svc.client.created_indexes
        if collection == "user_memory"
    ]
    # list_user_memory() filters on exactly these two fields (services/
    # qdrant_service.py) — if a future change adds a third filtered field
    # without provisioning its index, this test must be updated deliberately,
    # not silently pass.
    assert {field for field, _ in user_memory_indexes} == {"user_id", "memory_type"}
    assert all(schema == PayloadSchemaType.KEYWORD for _, schema in user_memory_indexes)
    assert svc._initialized is True


def test_payload_index_provisioning_is_idempotent_when_indexes_already_exist():
    svc = _bare_qdrant_service(
        _FakeLowLevelQdrantClient(existing_collections=["user_memory"], raise_already_exists=True)
    )
    # Must not raise even though create_payload_index always raises
    # "already exists" — init() must complete and mark itself initialized.
    svc.init()
    assert svc._initialized is True


class _FakeScrollClient:
    def __init__(self, points):
        self.calls = []
        self._points = points

    def scroll(self, **kwargs):
        self.calls.append(kwargs)
        return self._points, None


def test_list_user_memory_builds_production_compatible_filter_and_returns_matches():
    from qdrant_client.models import Filter

    class _Point:
        def __init__(self, payload):
            self.payload = payload

    matching = _Point({"user_id": "u1", "memory_type": "disliked", "timestamp": "2026-01-01T00:00:00Z"})

    svc = _bare_qdrant_service(_FakeScrollClient([matching]))
    svc._initialized = True  # skip collection/index provisioning for this focused test

    result = svc.list_user_memory("u1", "disliked", limit=10)

    assert result == [matching.payload]
    call = svc.client.calls[0]
    assert call["collection_name"] == "user_memory"
    assert call["limit"] == 10
    assert call["with_payload"] is True
    assert call["with_vectors"] is False
    scroll_filter = call["scroll_filter"]
    assert isinstance(scroll_filter, Filter)
    conditions = {c.key: c.match.value for c in scroll_filter.must}
    assert conditions == {"user_id": "u1", "memory_type": "disliked"}


# ---------------------------------------------------------------------------
# ACTIVE-REJECTION HARD EXCLUSION (30-day cooldown)
#
# Live-proven defect: the exact_rejected_outfit_penalty (-4.0) computed by
# _memory_breakdown is attached only to unified_style/score_meta, a field
# OutfitRanker.rank() never reads (it sorts on the legacy item["score"]) —
# so a rejected outfit could still resurface as the #1 recommendation.
# services.style_memory_service.load_active_rejected_outfit_signatures +
# brain.outfit_pipeline._exclude_active_rejected_signatures fix this with a
# real, authoritative exclusion applied before OutfitRanker.rank() runs,
# reusing the existing _outfit_signature identity and live Qdrant rejection
# memory — no new persistence, no penalty-magnitude dependence.
# ---------------------------------------------------------------------------
def _seed_rejection(fake_qdrant, user_id, item_ids, *, days_ago=0.0, point_id=None):
    sig = _outfit_signature([{"id": i} for i in item_ids])
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    pid = point_id or f"seed-{len(fake_qdrant.points)}-{sig}"
    fake_qdrant.points[pid] = {
        "memory_type": "disliked",
        "user_id": user_id,
        "item_ids": item_ids,
        "outfit_signature": sig,
        "timestamp": ts.isoformat(),
    }
    return sig


def _combo(item_ids, score=5.0):
    return {"items": [{"id": i} for i in item_ids], "score": score}


def test_active_rejection_within_window_is_returned_as_active(fake_qdrant):
    sig = _seed_rejection(fake_qdrant, "u1", ["shirt1", "pant1", "shoe1"], days_ago=1)
    mem = load_active_rejected_outfit_signatures("u1")
    assert mem["active_rejected_outfit_signatures"] == [sig]


def test_expired_rejection_beyond_window_is_not_active(fake_qdrant):
    _seed_rejection(
        fake_qdrant, "u1", ["shirt1", "pant1", "shoe1"],
        days_ago=REJECTION_ACTIVE_DAYS + 1,
    )
    mem = load_active_rejected_outfit_signatures("u1")
    assert mem["active_rejected_outfit_signatures"] == []


def test_boundary_just_inside_30_days_is_active_just_outside_is_not(fake_qdrant):
    # Documented rule: age_days <= REJECTION_ACTIVE_DAYS is active. Seeding
    # at exactly "N days ago" would race real elapsed wall-clock time by the
    # moment the loader actually runs, so the boundary is proven on either
    # side of it instead (30d minus 1 minute = inside, plus 1 minute = outside).
    inside_sig = _seed_rejection(
        fake_qdrant, "u1", ["shirt1", "pant1"],
        days_ago=REJECTION_ACTIVE_DAYS - (1 / 1440), point_id="inside",
    )
    outside_sig = _seed_rejection(
        fake_qdrant, "u1", ["shirt2", "pant2"],
        days_ago=REJECTION_ACTIVE_DAYS + (1 / 1440), point_id="outside",
    )
    mem = load_active_rejected_outfit_signatures("u1")
    assert inside_sig in mem["active_rejected_outfit_signatures"]
    assert outside_sig not in mem["active_rejected_outfit_signatures"]


def test_multiple_active_rejected_signatures_all_returned(fake_qdrant):
    sig_a = _seed_rejection(fake_qdrant, "u1", ["shirt1", "pant1"], days_ago=1)
    sig_b = _seed_rejection(fake_qdrant, "u1", ["shirt2", "pant2"], days_ago=5)
    mem = load_active_rejected_outfit_signatures("u1")
    assert set(mem["active_rejected_outfit_signatures"]) == {sig_a, sig_b}


def test_recommended_memory_type_does_not_feed_hard_exclusion(fake_qdrant):
    fake_qdrant.points["rec-1"] = {
        "memory_type": "recommended",  # not "disliked" -> different Qdrant filter
        "user_id": "u1",
        "outfit_signature": "pant1|shirt1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    mem = load_active_rejected_outfit_signatures("u1")
    assert mem["active_rejected_outfit_signatures"] == []


def test_change_item_memory_type_does_not_feed_hard_exclusion(fake_qdrant):
    fake_qdrant.points["chg-1"] = {
        "memory_type": "changed_item",  # not "disliked" -> different Qdrant filter
        "user_id": "u1",
        "outfit_signature": "pant1|shirt1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    mem = load_active_rejected_outfit_signatures("u1")
    assert mem["active_rejected_outfit_signatures"] == []


def test_active_rejection_end_to_end_via_feedback_endpoint(fake_qdrant):
    """Proves the real write path's timestamp (routers/feedback.py's
    datetime.now(timezone.utc).isoformat()) is read back correctly as
    active — not just a hand-seeded payload."""
    client = _client(None)
    resp = client.post(
        "/api/feedback/board",
        json={
            "user_id": "u1", "action": "dislike", "board_payload": {},
            "item_ids": ["shirt1", "pant1", "shoe1"],
        },
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    mem = load_active_rejected_outfit_signatures("u1")
    assert mem["active_rejected_outfit_signatures"] == ["pant1|shirt1|shoe1"]


def test_exclusion_is_a_noop_when_no_active_rejections():
    scored = [_combo(["shirt1", "pant1"])]
    kept = _exclude_active_rejected_signatures(scored, set(), {}, "u1")
    assert kept is scored  # same object back -> genuine early-return no-op


def test_exact_match_excluded_shared_item_combination_remains_eligible():
    rejected_sig = _outfit_signature([{"id": "shirt1"}, {"id": "pant1"}, {"id": "shoe1"}])
    exact = _combo(["shirt1", "pant1", "shoe1"])
    different_combo_shares_one_item = _combo(["shirt1", "pant2", "shoe2"])

    kept = _exclude_active_rejected_signatures(
        [exact, different_combo_shares_one_item], {rejected_sig}, {}, "u1"
    )

    assert exact not in kept
    assert different_combo_shares_one_item in kept


def test_all_candidates_rejected_returns_empty_not_reintroduced():
    rejected_sig = _outfit_signature([{"id": "shirt1"}, {"id": "pant1"}])
    scored = [_combo(["shirt1", "pant1"])]

    kept = _exclude_active_rejected_signatures(scored, {rejected_sig}, {}, "u1")

    assert kept == []


def test_non_rejected_alternatives_survive_with_scores_untouched():
    rejected_sig = _outfit_signature([{"id": "shirt1"}, {"id": "pant1"}])
    combo_rejected = _combo(["shirt1", "pant1"])
    combo_a = _combo(["shirt2", "pant2"], score=3.0)
    combo_b = _combo(["shirt3", "pant3"], score=7.0)

    kept = _exclude_active_rejected_signatures(
        [combo_rejected, combo_a, combo_b], {rejected_sig}, {}, "u1"
    )

    assert combo_rejected not in kept
    assert combo_a in kept and combo_b in kept
    assert max(kept, key=lambda c: c["score"]) is combo_b


def test_exclusion_ignores_incidental_display_metadata():
    """Same item combination, different title/board/strategy/image — the
    canonical signature (services.style_memory_service._outfit_signature,
    the SAME function the write path and _memory_breakdown already use)
    reads only item identity, never display metadata."""
    rejected_sig = _outfit_signature(
        [{"id": "shirt1"}, {"id": "pant1"}, {"id": "shoe1"}]
    )
    combo = {
        "items": [
            {"id": "shirt1", "name": "White Shirt", "image_url": "https://img/a.png"},
            {"id": "pant1", "name": "Blue Jeans", "image_url": "https://img/b.png"},
            {"id": "shoe1", "name": "Red Loafers", "image_url": "https://img/c.png"},
        ],
        "title": "Refined Simplicity",
        "board_id": "board-xyz",
        "strategy": "best_overall",
        "score": 9.0,
    }

    kept = _exclude_active_rejected_signatures([combo], {rejected_sig}, {}, "u1")

    assert kept == []


def test_exclusion_logs_safe_diagnostics_not_raw_signature(caplog):
    import logging

    rejected_sig = _outfit_signature([{"id": "shirt1"}, {"id": "pant1"}])
    scored = [_combo(["shirt1", "pant1"])]

    with caplog.at_level(logging.INFO, logger="ahvi.outfit_pipeline"):
        _exclude_active_rejected_signatures(
            scored, {rejected_sig}, {rejected_sig: 3}, "u1"
        )

    assert "AHVI_REJECTED_SIGNATURE_EXCLUDED" in caplog.text
    assert rejected_sig not in caplog.text  # hash only, never the raw signature
    assert "3d" in caplog.text


# ---------------------------------------------------------------------------
# SCORER/EXCLUSION CONTRACT ALIGNMENT — an expired (> REJECTION_ACTIVE_DAYS)
# rejection must carry neither the hard exclusion NOR the diagnostic
# exact_rejected_outfit_penalty; both now derive from the same age-filtered
# load_active_rejected_outfit_signatures via build_style_memory_context, so
# they can never drift apart when scorer unification eventually happens.
# ---------------------------------------------------------------------------
def test_active_rejection_still_receives_scorer_penalty_via_context(fake_qdrant):
    _seed_rejection(fake_qdrant, "u1", ["shirt1", "pant1", "shoe1"], days_ago=1)

    ctx = build_style_memory_context("u1", wardrobe=[])
    outfit = [{"id": "shirt1"}, {"id": "pant1"}, {"id": "shoe1"}]
    fields, _ = _memory_breakdown(outfit, ctx)

    assert fields["exact_rejected_outfit_penalty"] == EXACT_REJECTED_OUTFIT_PENALTY


def test_expired_rejection_no_longer_receives_scorer_penalty_via_context(fake_qdrant):
    _seed_rejection(
        fake_qdrant, "u1", ["shirt1", "pant1", "shoe1"],
        days_ago=REJECTION_ACTIVE_DAYS + 1,
    )

    ctx = build_style_memory_context("u1", wardrobe=[])
    outfit = [{"id": "shirt1"}, {"id": "pant1"}, {"id": "shoe1"}]
    fields, _ = _memory_breakdown(outfit, ctx)

    assert fields["exact_rejected_outfit_penalty"] == 0.0
