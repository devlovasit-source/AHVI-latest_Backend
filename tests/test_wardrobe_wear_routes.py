from fastapi import FastAPI
from fastapi.testclient import TestClient

import routers.wardrobe_capture as wc
from services.appwrite_proxy import AppwriteProxyError


class FakeWearEventsProxy:
    def __init__(self):
        self.docs = {}

    def create_document(self, resource, data, document_id="unique()"):
        assert resource == "wear_events"
        if document_id in self.docs:
            raise AppwriteProxyError("Appwrite request failed (409): already exists", status_code=409)
        doc = {"$id": document_id, **data}
        self.docs[document_id] = doc
        return doc

    def get_document(self, resource, document_id):
        return self.docs[document_id]

    def list_documents(self, resource, user_id=None, limit=500, **kwargs):
        return [d for d in self.docs.values() if d.get("userId") == user_id]


def _client(monkeypatch, *, item_owner="u1", fake_events=None):
    fake_events = fake_events if fake_events is not None else FakeWearEventsProxy()

    def fake_fetch_document(document_id):
        return ({"$id": document_id, "userId": item_owner}, "outfits", "db")

    monkeypatch.setattr(wc, "_fetch_document", fake_fetch_document)

    import services.wear_event_service as wear_svc

    monkeypatch.setattr(wear_svc, "_proxy", lambda: fake_events)

    import services.style_memory_service as memory_svc

    class NoopOutfitHistoryProxy:
        def list_documents(self, *a, **k):
            return []

        def create_document(self, *a, **k):
            return {}

        def update_document(self, *a, **k):
            return {}

    monkeypatch.setattr(memory_svc, "_proxy", lambda: NoopOutfitHistoryProxy())

    app = FastAPI()

    @app.middleware("http")
    async def authenticate(request, call_next):
        test_user = request.headers.get("x-test-user")
        if test_user:
            request.state.user = {"user_id": test_user}
        return await call_next(request)

    app.include_router(wc.wardrobe_router)
    return TestClient(app), fake_events


def test_wear_requires_auth(monkeypatch):
    client, _ = _client(monkeypatch)
    resp = client.post("/api/wardrobe/item-1/wear", json={})
    assert resp.status_code == 401


def test_wear_creates_event_for_owner(monkeypatch):
    client, fake = _client(monkeypatch, item_owner="u1")
    resp = client.post(
        "/api/wardrobe/item-1/wear",
        json={"occurred_at": "2026-08-17T09:00:00+00:00"},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["newly_created"] is True
    assert len(fake.docs) == 1


def test_wear_is_idempotent_across_requests(monkeypatch):
    client, fake = _client(monkeypatch, item_owner="u1")
    payload = {"occurred_at": "2026-08-17T09:00:00+00:00"}
    first = client.post("/api/wardrobe/item-1/wear", json=payload, headers={"x-test-user": "u1"})
    second = client.post("/api/wardrobe/item-1/wear", json=payload, headers={"x-test-user": "u1"})
    assert first.json()["newly_created"] is True
    assert second.json()["newly_created"] is False
    assert len(fake.docs) == 1


def test_user_a_cannot_wear_user_b_item(monkeypatch):
    client, fake = _client(monkeypatch, item_owner="user_b")
    resp = client.post(
        "/api/wardrobe/item-1/wear",
        json={"occurred_at": "2026-08-17T09:00:00+00:00"},
        headers={"x-test-user": "user_a"},
    )
    assert resp.status_code == 403
    assert not fake.docs


def test_wear_history_requires_auth(monkeypatch):
    client, _ = _client(monkeypatch)
    resp = client.get("/api/wardrobe/item-1/wear-history")
    assert resp.status_code == 401


def test_wear_history_returns_only_requesting_users_data(monkeypatch):
    fake = FakeWearEventsProxy()
    client, _ = _client(monkeypatch, item_owner="u1", fake_events=fake)
    client.post(
        "/api/wardrobe/item-1/wear",
        json={"occurred_at": "2026-08-17T09:00:00+00:00"},
        headers={"x-test-user": "u1"},
    )

    resp = client.get("/api/wardrobe/item-1/wear-history", headers={"x-test-user": "u1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_wears"] == 1
    assert len(body["events"]) == 1


def test_user_b_cannot_read_user_a_item_history(monkeypatch):
    client, _ = _client(monkeypatch, item_owner="user_a")
    resp = client.get("/api/wardrobe/item-1/wear-history", headers={"x-test-user": "user_b"})
    assert resp.status_code == 403


def test_favorite_requires_auth(monkeypatch):
    client, _ = _client(monkeypatch)
    resp = client.post("/api/wardrobe/favorite", json={"item_id": "item-1", "is_liked": True})
    assert resp.status_code == 401


def test_favorite_persists_liked_field_only(monkeypatch):
    client, _ = _client(monkeypatch, item_owner="u1")
    patch_calls = []

    def fake_patch_document(document_id, data, collection_id="", database_id=""):
        patch_calls.append((document_id, data, collection_id, database_id))
        return ({"$id": document_id, **data}, [])

    monkeypatch.setattr(
        "services.wardrobe_persistence_service._patch_document", fake_patch_document
    )

    resp = client.post(
        "/api/wardrobe/favorite",
        json={"item_id": "item-1", "is_liked": True},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    assert patch_calls == [("item-1", {"liked": True}, "outfits", "db")]


def test_user_a_cannot_favorite_user_b_item(monkeypatch):
    client, _ = _client(monkeypatch, item_owner="user_b")
    resp = client.post(
        "/api/wardrobe/favorite",
        json={"item_id": "item-1", "is_liked": True},
        headers={"x-test-user": "user_a"},
    )
    assert resp.status_code == 403
