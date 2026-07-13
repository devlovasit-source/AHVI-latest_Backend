import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from brain.engines.style_scorer import _memory_breakdown, style_scorer
from routers import feedback as route
from services import style_context_service as scs
from services import style_memory_service as sms
from services.style_feedback_store import (
    AppwriteStyleFeedbackStore,
    FeedbackStoreError,
    FeedbackValidationError,
    aggregate_feedback_events,
    canonical_event,
    feedback_document_id,
    load_feedback_memory,
)


def _request(user_id="user-1"):
    return SimpleNamespace(
        state=SimpleNamespace(user={"user_id": user_id}),
        url=SimpleNamespace(path="/api/feedback/board"),
    )


class Conflict(Exception):
    status_code = 409


class MemoryProxy:
    def __init__(self):
        self.docs = {}

    def create_document(self, resource, data, document_id=None):
        assert resource == "style_feedback_events"
        if document_id in self.docs:
            raise Conflict()
        self.docs[document_id] = dict(data)
        return {"$id": document_id, **data}

    def list_documents(self, resource, *, user_id, limit):
        return list(self.docs.values())[:limit]


def _board_req(**overrides):
    data = {
        "event_id": "evt-1",
        "action": "like",
        "board_payload": {
            "board_id": "board-1",
            "source_policy": "wardrobe",
            "occasion": "office",
            "style_direction": "minimal",
            "hero_image": "data:image/png;base64,AAAA",
            "items": [{"id": "top-1", "color": "Navy", "category": "top", "image_url": "https://x"}],
        },
    }
    data.update(overrides)
    return route.BoardFeedbackRequest(**data)


def test_authenticated_like_persists_correct_owner_and_no_media(monkeypatch):
    proxy = MemoryProxy()
    monkeypatch.setattr(route, "AppwriteStyleFeedbackStore", lambda: AppwriteStyleFeedbackStore(proxy))
    monkeypatch.setattr(route, "_mirror_board_best_effort", lambda *_: None)

    result = route.feedback_board(_board_req(user_id="user-1"), _request("user-1"))

    assert result["success"] is True
    event = next(iter(proxy.docs.values()))
    assert event["userId"] == "user-1"
    assert event["boardId"] == "board-1"
    assert json.loads(event["itemIds"]) == ["top-1"]
    persisted = json.dumps(event).lower()
    assert "image" not in persisted and "base64" not in persisted and "https://" not in persisted


def test_cross_user_body_identity_is_rejected(monkeypatch):
    monkeypatch.setattr(route, "AppwriteStyleFeedbackStore", lambda: pytest.fail("must not write"))
    with pytest.raises(HTTPException) as exc:
        route.feedback_board(_board_req(user_id="victim"), _request("attacker"))
    assert exc.value.status_code == 403


def test_camel_case_cross_user_identity_is_rejected(monkeypatch):
    monkeypatch.setattr(route, "AppwriteStyleFeedbackStore", lambda: pytest.fail("must not write"))
    body = route.BoardFeedbackRequest.model_validate({
        "eventId": "evt-camel", "action": "like", "userId": "victim",
        "boardPayload": {"boardId": "board-1", "items": [{"id": "top-1"}]},
    })
    with pytest.raises(HTTPException) as exc:
        route.feedback_board(body, _request("attacker"))
    assert exc.value.status_code == 403


def test_authenticated_item_like_persists_for_authenticated_owner(monkeypatch):
    proxy = MemoryProxy()
    monkeypatch.setattr(route, "AppwriteStyleFeedbackStore", lambda: AppwriteStyleFeedbackStore(proxy))
    request = route.ItemFeedbackRequest.model_validate({
        "eventId": "item-event-1", "itemId": "Item-ABC", "feedback": "up",
        "userId": "user-1",
    })
    result = route.feedback_item(request, _request("user-1"))
    event = next(iter(proxy.docs.values()))
    assert result["success"] is True
    assert event["userId"] == "user-1"
    assert json.loads(event["itemIds"]) == ["Item-ABC"]


def test_same_event_retry_is_idempotent_and_has_one_document():
    proxy = MemoryProxy()
    store = AppwriteStyleFeedbackStore(proxy)
    event = canonical_event(user_id="user-1", event_id="evt-1", action="like", item_ids=["top-1"])
    first, second = store.append(event), store.append(event)
    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert len(proxy.docs) == 1
    assert feedback_document_id("user-1", "evt-1") in proxy.docs


def test_legacy_board_request_without_event_id_is_durably_idempotent(monkeypatch):
    proxy = MemoryProxy()
    monkeypatch.setattr(route, "AppwriteStyleFeedbackStore", lambda: AppwriteStyleFeedbackStore(proxy))
    monkeypatch.setattr(route, "_mirror_board_best_effort", lambda *_: None)
    request = route.BoardFeedbackRequest(
        action="like", board_payload={"board_id": "board-legacy", "items": [{"id": "top-1"}]}
    )
    first = route.feedback_board(request, _request())
    second = route.feedback_board(request, _request())
    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert len(proxy.docs) == 1
    assert next(iter(proxy.docs.values()))["eventId"].startswith("compat-")


def test_latest_dislike_supersedes_like_and_users_stay_isolated():
    events = [
        {"userId": "user-1", "eventId": "2", "action": "dislike", "boardId": "b1",
         "itemIds": '["top-1"]', "payload": '{"style_direction":"minimal"}', "createdAtISO": "2026-01-02"},
        {"userId": "user-1", "eventId": "1", "action": "like", "boardId": "b1",
         "itemIds": '["top-1"]', "payload": '{"style_direction":"classic"}', "createdAtISO": "2026-01-01"},
    ]
    memory = aggregate_feedback_events(events)
    other = aggregate_feedback_events([])
    assert memory["liked_item_ids"] == []
    assert memory["disliked_item_ids"] == ["top-1"]
    assert memory["liked_board_patterns"] == []
    assert memory["disliked_board_patterns"] == ["minimal"]
    assert other["liked_item_ids"] == other["disliked_item_ids"] == []


def test_store_read_filters_owner_before_aggregation():
    proxy = MemoryProxy()
    store = AppwriteStyleFeedbackStore(proxy)
    store.append(canonical_event(
        user_id="user-1", event_id="evt-owner", action="like", item_ids=["top-1"]
    ))
    assert load_feedback_memory("user-1", store=store)["liked_item_ids"] == ["top-1"]
    assert load_feedback_memory("user-2", store=store)["liked_item_ids"] == []


def test_saved_patterns_stay_separate_and_canonical_id_case_is_preserved():
    events = [
        {"action": "saved", "boardId": "Board-Saved", "itemIds": '["Item-ABC"]',
         "payload": '{"style_direction":"tailored"}', "createdAtISO": "2026-01-03"},
        {"action": "like", "boardId": "Board-Liked", "itemIds": '["Item-ABC"]',
         "payload": '{"style_direction":"minimal"}', "createdAtISO": "2026-01-02"},
    ]
    memory = aggregate_feedback_events(events)
    assert memory["liked_item_ids"] == ["Item-ABC"]
    assert memory["feedback_saved_item_ids"] == ["Item-ABC"]
    assert memory["feedback_saved_board_patterns"] == ["tailored"]
    assert memory["liked_board_patterns"] == ["minimal"]


def test_bounded_like_and_dislike_scoring_and_disliked_masks_other_boosts():
    items = [{"id": f"i-{n}"} for n in range(5)]
    liked, _ = _memory_breakdown(items, {"liked_item_ids": [x["id"] for x in items]})
    disliked, _ = _memory_breakdown(
        items,
        {"disliked_item_ids": [x["id"] for x in items],
         "underworn_ids": [x["id"] for x in items], "saved_item_ids": [x["id"] for x in items]},
    )
    assert liked["liked_item_affinity"] == 2.4
    assert disliked["disliked_item_penalty"] == -4.0
    assert disliked["underworn_boost"] == 0
    assert disliked["saved_board_affinity"] == 0


def test_occasion_incompatible_like_remains_rejected():
    items = [
        {"id": "gold", "name": "Shiny Gold Formal Shirt", "category": "top", "color": "gold"},
        {"id": "pants", "name": "Chinos", "category": "bottom"},
        {"id": "shoes", "name": "Clean Sneakers", "category": "footwear"},
    ]
    result = style_scorer.score_outfit(items, {"occasion": "coffee_date", "liked_item_ids": ["gold"]}, {})
    assert result["occasion_reject"] is True
    assert result["breakdown"]["liked_item_affinity"] == 0


def test_empty_feedback_is_neutral():
    fields, _ = _memory_breakdown([{"id": "top-1"}], {})
    assert all(value == 0 for value in fields.values())


def test_appwrite_outage_is_typed_and_qdrant_outage_is_best_effort(monkeypatch):
    class BrokenStore:
        def append(self, _event):
            raise FeedbackStoreError("offline")
    monkeypatch.setattr(route, "AppwriteStyleFeedbackStore", BrokenStore)
    with pytest.raises(HTTPException) as exc:
        route.feedback_board(_board_req(), _request())
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "FEEDBACK_STORE_UNAVAILABLE"

    proxy = MemoryProxy()
    monkeypatch.setattr(route, "AppwriteStyleFeedbackStore", lambda: AppwriteStyleFeedbackStore(proxy))
    monkeypatch.setattr(route, "encode_metadata", lambda *_: [0.1])
    monkeypatch.setattr(route.qdrant_service, "upsert_user_memory", lambda **_: (_ for _ in ()).throw(RuntimeError("down")))
    assert route.feedback_board(_board_req(event_id="evt-2"), _request())["success"] is True
    assert len(proxy.docs) == 1


def test_excessive_items_and_oversized_payload_are_rejected():
    with pytest.raises(FeedbackValidationError):
        canonical_event(user_id="u", event_id="e", action="like", item_ids=[str(i) for i in range(26)])
    with pytest.raises(FeedbackValidationError):
        canonical_event(user_id="u", event_id="e", action="like", board_payload={"blob": "x" * 100_001})


def test_media_like_values_are_removed_even_from_learning_fields():
    event = canonical_event(
        user_id="u", event_id="e-media", action="like",
        board_payload={
            "style_direction": "data:image/png;base64,AAAA",
            "items": [{"id": "top-1", "color": "https://example.test/image.png"}],
        },
    )
    persisted = json.dumps(event).lower()
    assert "data:image" not in persisted
    assert "base64" not in persisted
    assert "https://" not in persisted


def test_canonical_context_loads_aggregated_feedback_once(monkeypatch):
    calls = []
    monkeypatch.setattr(sms, "load_wear_memory", lambda *_: {
        "recently_worn_ids": [], "underworn_ids": [], "wear_counts": {}, "last_worn_at": {}})
    monkeypatch.setattr(sms, "load_saved_board_memory", lambda *_: {
        "saved_item_ids": [], "saved_board_patterns": [], "favorite_colors": [], "favorite_categories": []})
    monkeypatch.setattr("services.style_feedback_store.load_feedback_memory", lambda uid: calls.append(uid) or {
        "liked_item_ids": ["top-1"], "disliked_item_ids": [], "feedback_saved_item_ids": []})
    context = scs.build_canonical_style_context(
        query="daily outfit", user_id="user-1", wardrobe_items=[{"id": "top-1"}],
        user_profile={}, profile_is_authenticated=True,
    )
    assert calls == ["user-1"]
    assert context["liked_item_ids"] == ["top-1"]
