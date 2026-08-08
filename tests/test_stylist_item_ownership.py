from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import stylist


class FakeProxy:
    wardrobe = []
    shared = []
    page_cap = None

    def list_documents(self, resource, **kwargs):
        offset = kwargs.get("offset", 0)
        limit = min(kwargs.get("limit", 100), self.page_cap or 100)
        if resource == "outfits":
            user_id = kwargs.get("user_id")
            rows = [row for row in self.wardrobe if row.get("userId") == user_id]
        elif resource == "style_assets":
            rows = list(self.shared)
        else:
            rows = []
        page = rows[offset:offset + limit]
        if kwargs.get("return_meta"):
            next_offset = offset + len(page)
            return {
                "documents": page,
                "meta": {
                    "has_more": next_offset < len(rows),
                    "next_offset": next_offset if next_offset < len(rows) else None,
                },
            }
        return page


def _client(monkeypatch, *, wardrobe=None, shared=None, page_cap=None):
    FakeProxy.wardrobe = wardrobe or []
    FakeProxy.shared = shared or []
    FakeProxy.page_cap = page_cap
    monkeypatch.setattr(stylist, "AppwriteProxy", FakeProxy)
    app = FastAPI()

    @app.middleware("http")
    async def authenticate(request, call_next):
        request.state.user = {"user_id": request.headers.get("x-test-user", "owner")}
        return await call_next(request)

    app.include_router(stylist.router, prefix="/api/stylist")
    return TestClient(app)


def _payload(user_id="owner"):
    return {
        "user_id": user_id,
        "mode": "build_outfit",
        "anchor_item": {"id": "forged", "name": "Forged"},
        "wardrobe": [{"id": "forged", "name": "Forged"}],
    }


def test_owned_item_is_resolved_from_authenticated_wardrobe(monkeypatch):
    client = _client(
        monkeypatch,
        wardrobe=[
            {"id": "mine", "userId": "owner", "name": "Blue Shirt", "category": "Tops"},
            {"id": "shoes", "userId": "owner", "name": "Sneakers", "category": "Footwear"},
        ],
    )
    response = client.post("/api/stylist/items/mine/style", json=_payload())

    assert response.status_code == 200
    assert response.json()["error"]["code"] == "TRY_ON_COMING_SOON"


def test_cross_owner_item_is_not_accessible(monkeypatch):
    client = _client(
        monkeypatch,
        wardrobe=[{"id": "theirs", "userId": "other", "name": "Private"}],
    )
    response = client.post("/api/stylist/items/theirs/style", json=_payload())

    assert response.status_code == 404


def test_body_user_cannot_override_authenticated_owner(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/api/stylist/items/mine/style",
        json=_payload("other"),
        headers={"x-test-user": "owner"},
    )

    assert response.status_code == 403


def test_authoritative_shared_style_asset_remains_usable(monkeypatch):
    client = _client(
        monkeypatch,
        shared=[{"id": "shared-1", "name": "Shared Blazer", "category": "Outerwear"}],
    )
    response = client.post("/api/stylist/items/shared-1/style", json=_payload())

    assert response.status_code == 200
    assert response.json()["error"]["code"] == "TRY_ON_COMING_SOON"


def test_owned_item_after_first_page_remains_usable(monkeypatch):
    wardrobe = [
        {"id": f"item-{index}", "userId": "owner", "name": "Item", "category": "Tops"}
        for index in range(101)
    ]
    client = _client(monkeypatch, wardrobe=wardrobe, page_cap=25)
    response = client.post("/api/stylist/items/item-100/style", json=_payload())

    assert response.status_code == 200
    assert response.json()["error"]["code"] == "TRY_ON_COMING_SOON"
