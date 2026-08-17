from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.style_memory_service import record_wear


class FakeOutfitHistoryProxy:
    """In-memory stand-in for AppwriteProxy scoped to the outfit_history
    collection, matching the shape services.style_memory_service expects."""

    def __init__(self):
        self.docs = {}
        self._counter = 0

    def list_documents(self, resource, **kwargs):
        assert resource == "outfit_history"
        user_id = kwargs.get("user_id")
        return [d for d in self.docs.values() if d.get("userId") == user_id]

    def create_document(self, resource, data, document_id=None):
        assert resource == "outfit_history"
        self._counter += 1
        doc_id = document_id or f"doc{self._counter}"
        doc = {"$id": doc_id, **data}
        self.docs[doc_id] = doc
        return doc

    def update_document(self, resource, document_id, patch):
        assert resource == "outfit_history"
        doc = self.docs.get(document_id, {})
        doc.update(patch)
        self.docs[document_id] = doc
        return doc


def _patch_proxy(monkeypatch, fake):
    import services.style_memory_service as svc

    monkeypatch.setattr(svc, "_proxy", lambda: fake)


def _row(fake, user_id, item_id):
    matches = [d for d in fake.docs.values() if d["userId"] == user_id and d["itemId"] == item_id]
    assert len(matches) == 1
    return matches[0]


def test_first_wear_increments(monkeypatch):
    fake = FakeOutfitHistoryProxy()
    _patch_proxy(monkeypatch, fake)

    record_wear(user_id="u1", item_ids=["shirt1"], worn_at="2026-08-17T09:00:00+00:00")

    row = _row(fake, "u1", "shirt1")
    assert row["wearCount"] == 1
    assert row["lastWornAt"] == "2026-08-17T09:00:00+00:00"


def test_same_day_duplicate_does_not_increment(monkeypatch):
    fake = FakeOutfitHistoryProxy()
    _patch_proxy(monkeypatch, fake)

    record_wear(user_id="u1", item_ids=["shirt1"], worn_at="2026-08-17T09:00:00+00:00")
    record_wear(user_id="u1", item_ids=["shirt1"], worn_at="2026-08-17T18:00:00+00:00")

    row = _row(fake, "u1", "shirt1")
    assert row["wearCount"] == 1
    # Original first-of-day timestamp is preserved, not overwritten.
    assert row["lastWornAt"] == "2026-08-17T09:00:00+00:00"


def test_later_day_wear_increments(monkeypatch):
    fake = FakeOutfitHistoryProxy()
    _patch_proxy(monkeypatch, fake)

    record_wear(user_id="u1", item_ids=["shirt1"], worn_at="2026-08-17T09:00:00+00:00")
    record_wear(user_id="u1", item_ids=["shirt1"], worn_at="2026-08-18T09:00:00+00:00")

    row = _row(fake, "u1", "shirt1")
    assert row["wearCount"] == 2
    assert row["lastWornAt"] == "2026-08-18T09:00:00+00:00"


def test_duplicate_ids_in_one_request_do_not_double_increment(monkeypatch):
    fake = FakeOutfitHistoryProxy()
    _patch_proxy(monkeypatch, fake)

    record_wear(user_id="u1", item_ids=["shirt1", "shirt1"], worn_at="2026-08-17T09:00:00+00:00")

    row = _row(fake, "u1", "shirt1")
    assert row["wearCount"] == 1


def test_different_item_same_day_increments_independently(monkeypatch):
    fake = FakeOutfitHistoryProxy()
    _patch_proxy(monkeypatch, fake)

    record_wear(user_id="u1", item_ids=["shirt1"], worn_at="2026-08-17T09:00:00+00:00")
    record_wear(user_id="u1", item_ids=["pant1"], worn_at="2026-08-17T09:05:00+00:00")

    assert _row(fake, "u1", "shirt1")["wearCount"] == 1
    assert _row(fake, "u1", "pant1")["wearCount"] == 1


def test_different_user_isolated(monkeypatch):
    fake = FakeOutfitHistoryProxy()
    _patch_proxy(monkeypatch, fake)

    record_wear(user_id="u1", item_ids=["shirt1"], worn_at="2026-08-17T09:00:00+00:00")
    record_wear(user_id="u2", item_ids=["shirt1"], worn_at="2026-08-17T09:05:00+00:00")

    assert _row(fake, "u1", "shirt1")["wearCount"] == 1
    assert _row(fake, "u2", "shirt1")["wearCount"] == 1


def test_same_day_duplicate_refreshes_board_and_occasion_metadata(monkeypatch):
    fake = FakeOutfitHistoryProxy()
    _patch_proxy(monkeypatch, fake)

    record_wear(
        user_id="u1", item_ids=["shirt1"], board_id="board-a", occasion="casual",
        worn_at="2026-08-17T09:00:00+00:00",
    )
    record_wear(
        user_id="u1", item_ids=["shirt1"], board_id="board-b", occasion="office",
        worn_at="2026-08-17T18:00:00+00:00",
    )

    row = _row(fake, "u1", "shirt1")
    assert row["wearCount"] == 1  # still one wear event for the day
    assert row["lastBoardId"] == "board-b"
    assert row["lastOccasion"] == "office"


def test_same_day_duplicate_without_new_metadata_preserves_existing(monkeypatch):
    fake = FakeOutfitHistoryProxy()
    _patch_proxy(monkeypatch, fake)

    record_wear(
        user_id="u1", item_ids=["shirt1"], board_id="board-a", occasion="casual",
        worn_at="2026-08-17T09:00:00+00:00",
    )
    record_wear(user_id="u1", item_ids=["shirt1"], worn_at="2026-08-17T18:00:00+00:00")

    row = _row(fake, "u1", "shirt1")
    assert row["lastBoardId"] == "board-a"
    assert row["lastOccasion"] == "casual"


# ---------------------------------------------------------------------------
# Router-level contract: auth/validation behavior is unchanged by this patch.
# ---------------------------------------------------------------------------
def _client(monkeypatch, fake):
    from routers import style_memory

    _patch_proxy(monkeypatch, fake)
    app = FastAPI()

    @app.middleware("http")
    async def authenticate(request, call_next):
        test_user = request.headers.get("x-test-user")
        if test_user:
            request.state.user = {"user_id": test_user}
        return await call_next(request)

    app.include_router(style_memory.router)
    return TestClient(app)


def test_missing_item_ids_returns_400(monkeypatch):
    client = _client(monkeypatch, FakeOutfitHistoryProxy())

    resp = client.post(
        "/api/style/wear-today",
        json={"user_id": "u1", "item_ids": []},
        headers={"x-test-user": "u1"},
    )

    assert resp.status_code == 400


def test_unauthenticated_request_returns_401(monkeypatch):
    client = _client(monkeypatch, FakeOutfitHistoryProxy())

    resp = client.post(
        "/api/style/wear-today",
        json={"user_id": "u1", "item_ids": ["shirt1"]},
    )

    assert resp.status_code == 401


def test_mismatched_user_id_returns_403(monkeypatch):
    client = _client(monkeypatch, FakeOutfitHistoryProxy())

    resp = client.post(
        "/api/style/wear-today",
        json={"user_id": "someone-else", "item_ids": ["shirt1"]},
        headers={"x-test-user": "u1"},
    )

    assert resp.status_code == 403


def test_valid_request_records_wear_and_is_idempotent_same_day(monkeypatch):
    fake = FakeOutfitHistoryProxy()
    client = _client(monkeypatch, fake)

    payload = {"user_id": "u1", "item_ids": ["shirt1"], "board_id": "board-a"}
    first = client.post("/api/style/wear-today", json=payload, headers={"x-test-user": "u1"})
    second = client.post("/api/style/wear-today", json=payload, headers={"x-test-user": "u1"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert _row(fake, "u1", "shirt1")["wearCount"] == 1
