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

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from brain.engines.style_scorer import (
    CHANGED_FROM_ITEM_PENALTY,
    CHANGED_TO_ITEM_BOOST,
    EXACT_REJECTED_OUTFIT_PENALTY,
    _memory_breakdown,
)
from services import style_board_shuffle_service as shuffle_service
from services.style_board_state_store import InMemoryBoardStateStore
from services.style_memory_service import (
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
    def __init__(self):
        self.points = {}  # point_id -> payload
        self._counter = 0

    def upsert_user_memory(self, user_id, vector, payload, point_id=None):
        if not vector:
            return None
        self._counter += 1
        pid = point_id or f"auto-{self._counter}"
        self.points[pid] = dict(payload)
        return pid

    def list_user_memory(self, user_id, memory_type, limit=50):
        rows = [
            p for p in self.points.values()
            if p.get("user_id") == user_id and p.get("memory_type") == memory_type
        ]
        rows.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
        return rows[:limit]


def _fixed_embedding(*_args, **_kwargs):
    # Deterministic non-empty vector so upsert_user_memory's real
    # "no vector -> no-op" guard never fires just because the test
    # environment has no live embedding backend configured.
    return [0.1, 0.2, 0.3]


@pytest.fixture()
def fake_qdrant(monkeypatch):
    fake = FakeQdrant()
    # routers/feedback.py imports `qdrant_service`/`encode_metadata` at
    # module load time (plain reference, not re-read per call), so the
    # module-level names must be patched directly. style_memory_service.py
    # and routers/style_memory.py's memory writer import qdrant_service
    # lazily inside the function body, so patching the source module also
    # covers them.
    monkeypatch.setattr("services.qdrant_service.qdrant_service", fake)
    monkeypatch.setattr("routers.feedback.qdrant_service", fake)
    monkeypatch.setattr("routers.feedback.encode_metadata", _fixed_embedding)
    monkeypatch.setattr("services.embedding_service.encode_metadata", _fixed_embedding)
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
