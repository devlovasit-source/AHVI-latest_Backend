from fastapi import FastAPI
from fastapi.testclient import TestClient

import routers.wardrobe_capture as wc


class FakeReminderProxy:
    def __init__(self):
        self.docs = {}

    def update_document(self, resource, document_id, patch):
        if document_id not in self.docs:
            raise RuntimeError("not found")
        self.docs[document_id].update(patch)
        return self.docs[document_id]

    def create_document(self, resource, data, document_id=None):
        doc_id = document_id or f"doc{len(self.docs) + 1}"
        doc = {"$id": doc_id, **data}
        self.docs[doc_id] = doc
        return doc

    def get_document(self, resource, document_id):
        if document_id not in self.docs:
            raise RuntimeError("(404) not found")
        return self.docs[document_id]

    def list_documents(self, resource, user_id=None, limit=500, **kwargs):
        return [d for d in self.docs.values() if d.get("userId") == user_id]


def _client(monkeypatch, *, item_owner="u1"):
    def fake_fetch_document(document_id):
        return ({"$id": document_id, "userId": item_owner}, "outfits", "db")

    monkeypatch.setattr(wc, "_fetch_document", fake_fetch_document)

    import services.notification_store as ns

    fake = FakeReminderProxy()
    monkeypatch.setattr(ns.notification_store, "_appwrite", fake)

    app = FastAPI()

    @app.middleware("http")
    async def authenticate(request, call_next):
        test_user = request.headers.get("x-test-user")
        if test_user:
            request.state.user = {"user_id": test_user}
        return await call_next(request)

    app.include_router(wc.wardrobe_router)
    return TestClient(app), fake


def test_create_reminder_requires_auth(monkeypatch):
    client, _ = _client(monkeypatch)
    resp = client.post("/api/wardrobe/item-1/wear-reminder", json={"send_at_iso": "2026-09-01T09:00:00+00:00"})
    assert resp.status_code == 401


def test_create_reminder_for_owner(monkeypatch):
    client, fake = _client(monkeypatch, item_owner="u1")
    resp = client.post(
        "/api/wardrobe/item-1/wear-reminder",
        json={"send_at_iso": "2026-09-01T09:00:00+00:00", "message": "wear it"},
        headers={"x-test-user": "u1"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert len(fake.docs) == 1


def test_create_reminder_rejects_non_owner(monkeypatch):
    client, fake = _client(monkeypatch, item_owner="user_b")
    resp = client.post(
        "/api/wardrobe/item-1/wear-reminder",
        json={"send_at_iso": "2026-09-01T09:00:00+00:00"},
        headers={"x-test-user": "user_a"},
    )
    assert resp.status_code == 403
    assert not fake.docs


def test_list_reminders_requires_auth(monkeypatch):
    client, _ = _client(monkeypatch)
    resp = client.get("/api/wardrobe/item-1/wear-reminder")
    assert resp.status_code == 401


def test_list_reminders_returns_created(monkeypatch):
    client, _ = _client(monkeypatch, item_owner="u1")
    client.post(
        "/api/wardrobe/item-1/wear-reminder",
        json={"send_at_iso": "2026-09-01T09:00:00+00:00"},
        headers={"x-test-user": "u1"},
    )
    resp = client.get("/api/wardrobe/item-1/wear-reminder", headers={"x-test-user": "u1"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["reminders"]) == 1


def test_list_reminders_rejects_non_owner(monkeypatch):
    client, _ = _client(monkeypatch, item_owner="user_b")
    resp = client.get("/api/wardrobe/item-1/wear-reminder", headers={"x-test-user": "user_a"})
    assert resp.status_code == 403


def test_cancel_reminder_requires_auth(monkeypatch):
    client, _ = _client(monkeypatch)
    resp = client.delete("/api/wardrobe/item-1/wear-reminder/rem1")
    assert resp.status_code == 401


def test_cancel_reminder_removes_it_from_list(monkeypatch):
    client, _ = _client(monkeypatch, item_owner="u1")
    create_resp = client.post(
        "/api/wardrobe/item-1/wear-reminder",
        json={"send_at_iso": "2026-09-01T09:00:00+00:00"},
        headers={"x-test-user": "u1"},
    )
    reminder_id = client.get(
        "/api/wardrobe/item-1/wear-reminder", headers={"x-test-user": "u1"}
    ).json()["reminders"][0]["reminder_id"]

    cancel_resp = client.delete(
        f"/api/wardrobe/item-1/wear-reminder/{reminder_id}", headers={"x-test-user": "u1"}
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["success"] is True

    after = client.get("/api/wardrobe/item-1/wear-reminder", headers={"x-test-user": "u1"})
    assert after.json()["reminders"] == []


def test_cancel_reminder_for_a_different_item_is_rejected(monkeypatch):
    client, _ = _client(monkeypatch, item_owner="u1")
    client.post(
        "/api/wardrobe/item-1/wear-reminder",
        json={"send_at_iso": "2026-09-01T09:00:00+00:00"},
        headers={"x-test-user": "u1"},
    )
    reminder_id = client.get(
        "/api/wardrobe/item-1/wear-reminder", headers={"x-test-user": "u1"}
    ).json()["reminders"][0]["reminder_id"]

    # item-2 also resolves to owner u1 in this fake, so the item-ownership
    # check alone would pass — this proves cancel additionally checks the
    # reminder's own eventId against the item in the URL.
    resp = client.delete(
        f"/api/wardrobe/item-2/wear-reminder/{reminder_id}", headers={"x-test-user": "u1"}
    )
    assert resp.status_code == 403


def test_cancel_reminder_twice_stays_idempotent(monkeypatch):
    client, _ = _client(monkeypatch, item_owner="u1")
    client.post(
        "/api/wardrobe/item-1/wear-reminder",
        json={"send_at_iso": "2026-09-01T09:00:00+00:00"},
        headers={"x-test-user": "u1"},
    )
    reminder_id = client.get(
        "/api/wardrobe/item-1/wear-reminder", headers={"x-test-user": "u1"}
    ).json()["reminders"][0]["reminder_id"]

    first = client.delete(f"/api/wardrobe/item-1/wear-reminder/{reminder_id}", headers={"x-test-user": "u1"})
    second = client.delete(f"/api/wardrobe/item-1/wear-reminder/{reminder_id}", headers={"x-test-user": "u1"})
    assert first.status_code == 200
    assert second.status_code == 200
