from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from routers import stylist
from services import style_board_shuffle_service as shuffle_service
from services.style_board_state_store import InMemoryBoardStateStore


def _item(item_id: str, name: str, category: str, *, source: str = "wardrobe"):
    return {
        "id": item_id,
        "item_id": item_id,
        "name": name,
        "category": category,
        "source": source,
        "image_url": f"https://images.test/{item_id}.png",
        "normalized_url": f"https://images.test/{item_id}-processed.png",
    }


def _wardrobe():
    return [
        _item("shirt-1", "White Shirt", "Tops"),
        _item("jacket-1", "Navy Jacket", "Outerwear"),
        _item("bottom-1", "Grey Trousers", "Bottoms"),
        _item("dress-1", "Black Dress", "Dresses"),
        _item("shoe-1", "White Sneakers", "Footwear"),
        _item("belt-1", "Tan Belt", "Accessories"),
        _item("bag-1", "Black Bag", "Accessories"),
    ]


@pytest.fixture(autouse=True)
def _isolated_board_store(monkeypatch):
    shuffle_service.set_state_store(InMemoryBoardStateStore())
    monkeypatch.setattr(
        stylist,
        "resolve_location_weather_context",
        lambda **kwargs: {
            "profile": kwargs.get("profile") or {},
            "weather": {"status": "unavailable"},
            "location": {},
            "context_usage": {},
        },
    )
    monkeypatch.setattr(
        stylist,
        "resolve_style_archetypes",
        lambda *args, **kwargs: [
            {
                "archetype_id": f"identity-{index}",
                "direction_title": f"Identity {index}",
                "palette": [],
                "avoid": [],
                "formality": "casual",
                "reasoning_intent": "preserve the selected item",
                "anchor_item_id": "attacker-item",
            }
            for index in range(3)
        ],
    )
    yield
    shuffle_service.set_state_store(None)


@pytest.mark.parametrize(
    ("anchor_id", "expected_role", "expected_board_role"),
    [
        ("belt-1", "accessory", "belt"),
        ("shirt-1", "top", None),
        ("jacket-1", "outerwear", None),
        ("bottom-1", "bottom", None),
        ("dress-1", "dress", None),
        ("shoe-1", "footwear", None),
        ("bag-1", "accessory", "bag"),
    ],
)
def test_style_this_item_matrix_preserves_exact_canonical_anchor(
    anchor_id, expected_role, expected_board_role
):
    wardrobe = _wardrobe()
    anchor = next(item for item in wardrobe if item["id"] == anchor_id)
    result = stylist.style_wardrobe_item(
        anchor_id,
        stylist.ItemStyleRequest(
            user_id="owner-1",
            mode="style_this",
            anchor_item_id=anchor_id,
            anchor_item=anchor,
            wardrobe=wardrobe,
        ),
    )

    assert result["success"] is True
    assert result["anchor_item_id"] == anchor_id
    assert result["anchor_item"]["item_id"] == anchor_id
    for direction in result["style_directions"]:
        assert direction["anchor_item_id"] == anchor_id
        assert direction["originating_item_id"] == anchor_id
        items = direction["board_items"]
        matches = [item for item in items if item["item_id"] == anchor_id]
        assert len(matches) == 1
        board_anchor = matches[0]
        assert board_anchor["item_id"] == anchor_id
        assert board_anchor["role"] == expected_role
        assert board_anchor["locked"] is True
        assert board_anchor["safe_image_url"] == anchor["normalized_url"]
        if expected_board_role:
            assert board_anchor["board_role"] == expected_board_role
        state = shuffle_service.get_board_state(direction["board_id"])
        assert state["anchor_item_id"] == anchor_id
        assert [item["item_id"] for item in state["items"]].count(anchor_id) == 1


def _http_client(monkeypatch, wardrobe):
    class Proxy:
        def list_documents(self, resource, **kwargs):
            if resource == "outfits":
                user_id = kwargs.get("user_id")
                return [
                    item for item in wardrobe
                    if item.get("userId", "owner-1") == user_id
                ]
            return []

    monkeypatch.setattr(stylist, "AppwriteProxy", Proxy)
    app = FastAPI()

    @app.middleware("http")
    async def auth(request, call_next):
        request.state.user = {"user_id": "owner-1"}
        return await call_next(request)

    app.include_router(stylist.router, prefix="/api/stylist")
    return TestClient(app)


def test_client_metadata_mismatch_cannot_replace_authoritative_anchor(monkeypatch):
    wardrobe = [_item("item-a", "Authoritative Shirt", "Tops")]
    client = _http_client(monkeypatch, wardrobe)
    response = client.post(
        "/api/stylist/items/item-a/style",
        json={
            "user_id": "owner-1",
            "mode": "style_this",
            "anchor_item_id": "item-a",
            "anchor_item": {"id": "item-b", "name": "Forged Jacket", "category": "Outerwear"},
            "wardrobe": [{"id": "item-b", "name": "Forged Jacket"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "INSUFFICIENT_WARDROBE"
    assert body["anchor_item"]["item_id"] == "item-a"
    assert body["style_directions"] == []


@pytest.mark.parametrize(
    ("anchor_id", "expected_role"),
    [
        ("shirt-1", "top"),
        ("bottom-1", "bottom"),
        ("shoe-1", "footwear"),
    ],
)
def test_http_style_this_normalizes_missing_server_wardrobe_source(
    monkeypatch, anchor_id, expected_role
):
    wardrobe = _wardrobe()
    for item in wardrobe:
        item.pop("source", None)
    client = _http_client(monkeypatch, wardrobe)

    response = client.post(
        f"/api/stylist/items/{anchor_id}/style",
        json={
            "user_id": "owner-1",
            "mode": "style_this",
            "anchor_item_id": anchor_id,
            "anchor_item": {
                "id": "client-forged",
                "source": "wardrobe",
                "category": "Outerwear",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["anchor_item_id"] == anchor_id
    assert body["anchor_item"]["item_id"] == anchor_id
    assert body["anchor_item"]["source"] == "wardrobe"
    for direction in body["style_directions"]:
        assert direction["anchor_item_id"] == anchor_id
        assert direction["originating_item_id"] == anchor_id
        board_anchor = next(
            item for item in direction["board_items"] if item["item_id"] == anchor_id
        )
        assert board_anchor["item_id"] == anchor_id
        assert board_anchor["role"] == expected_role
        assert board_anchor["locked"] is True


def test_missing_source_wrong_owner_cannot_become_wardrobe_anchor(monkeypatch):
    theirs = _item("private-item", "Private Shirt", "Tops")
    theirs.pop("source", None)
    theirs["userId"] = "other-user"
    client = _http_client(monkeypatch, [theirs])

    response = client.post(
        "/api/stylist/items/private-item/style",
        json={
            "user_id": "owner-1",
            "mode": "style_this",
            "anchor_item_id": "private-item",
            "anchor_item": {"id": "private-item", "source": "wardrobe"},
        },
    )

    assert response.status_code == 404


def test_client_supplied_wardrobe_list_cannot_gain_ownership_trust(monkeypatch):
    """P0 (RC3) trust-boundary regression: the *unconditional* wardrobe-source
    stamp only became safe to make unconditional because it is gated on
    provenance (authenticated Appwrite fetch), not on the item's declared
    source value. This proves a client-supplied `wardrobe` list (a
    completely different vector from the already-covered `anchor_item`
    override) cannot use that same unconditional stamp to manufacture trust
    for an item the user does not actually own -- the real authenticated
    wardrobe is empty here, so any success would mean the client payload was
    trusted instead of the server's own fetch."""
    client = _http_client(monkeypatch, [])  # authenticated wardrobe: empty

    forged = _item("forged-item", "Forged Jacket", "Outerwear")
    forged.pop("source", None)  # blank source -- would have passed the OLD conditional stamp too

    response = client.post(
        "/api/stylist/items/forged-item/style",
        json={
            "user_id": "owner-1",
            "mode": "style_this",
            "anchor_item_id": "forged-item",
            "wardrobe": [forged],
        },
    )

    assert response.status_code == 404, (
        f"a client-supplied wardrobe payload gained ownership trust it must "
        f"never have. status={response.status_code} body={response.text}"
    )


def test_missing_item_cannot_become_wardrobe_anchor_from_client_source(monkeypatch):
    client = _http_client(monkeypatch, [])

    response = client.post(
        "/api/stylist/items/missing/style",
        json={
            "user_id": "owner-1",
            "mode": "style_this",
            "anchor_item_id": "missing",
            "source": "wardrobe",
            "anchor_item": {
                "id": "missing",
                "source": "wardrobe",
                "normalized_url": "https://images.test/forged.png",
                "category": "Tops",
            },
        },
    )

    assert response.status_code == 404


def test_llm_or_archetype_anchor_substitution_is_ignored():
    wardrobe = _wardrobe()
    result = stylist.style_wardrobe_item(
        "shirt-1",
        stylist.ItemStyleRequest(
            user_id="owner-1",
            mode="style_this",
            anchor_item_id="shirt-1",
            anchor_item=next(item for item in wardrobe if item["id"] == "shirt-1"),
            wardrobe=wardrobe,
        ),
    )

    assert result["success"] is True
    assert all(
        direction["anchor_item_id"] == "shirt-1"
        and [item["item_id"] for item in direction["items"]].count("shirt-1") == 1
        for direction in result["style_directions"]
    )


def test_missing_authoritative_anchor_fails_without_generic_generation(monkeypatch):
    client = _http_client(monkeypatch, [_item("other", "Other Shirt", "Tops")])
    response = client.post(
        "/api/stylist/items/missing/style",
        json={"user_id": "owner-1", "mode": "style_this", "anchor_item_id": "missing"},
    )

    assert response.status_code == 404
    assert "style_directions" not in response.json()


def test_cross_user_anchor_is_rejected(monkeypatch):
    theirs = _item("private-item", "Private Shirt", "Tops")
    theirs["userId"] = "other-user"
    client = _http_client(monkeypatch, [theirs])
    response = client.post(
        "/api/stylist/items/private-item/style",
        json={"user_id": "owner-1", "mode": "style_this", "anchor_item_id": "private-item"},
    )

    assert response.status_code == 404


def test_duplicate_anchor_direction_is_not_registered():
    wardrobe = _wardrobe()
    anchor = next(item for item in wardrobe if item["id"] == "shirt-1")
    direction = {
        "title": "Broken",
        "items": [anchor, dict(anchor), _item("shoe-1", "White Sneakers", "Footwear")],
    }
    registered = stylist._register_style_this_direction(
        direction,
        anchor=stylist._resolve_style_this_anchor(
            stylist.ItemStyleRequest(user_id="owner-1", anchor_item=anchor),
            "shirt-1",
            wardrobe,
        ),
        wardrobe=wardrobe,
        user_id="owner-1",
        occasion=None,
    )

    assert registered["shuffle_state_error"]["code"] == "DUPLICATE_ITEM_ID"
    assert shuffle_service.get_board_state(registered["board_id"]) is None


def test_shuffle_reconstructs_style_this_anchor_from_durable_state():
    wardrobe = _wardrobe()
    anchor = next(item for item in wardrobe if item["id"] == "shirt-1")
    result = stylist.style_wardrobe_item(
        "shirt-1",
        stylist.ItemStyleRequest(
            user_id="owner-1",
            mode="style_this",
            anchor_item_id="shirt-1",
            anchor_item=anchor,
            wardrobe=wardrobe,
        ),
    )
    direction = result["style_directions"][0]
    supporting = [item for item in direction["items"] if item["item_id"] != "shirt-1"]
    shuffled = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=1,
        locked_items=[supporting[0]],
        shuffle_slots=[item["slot"] for item in supporting[1:]],
        exclude_item_ids=[item["item_id"] for item in supporting],
        source_policy="inherit",
        wardrobe=wardrobe + [_item("shoe-2", "Black Loafers", "Footwear")],
        user_id="owner-1",
    )

    assert shuffled["success"] is True
    assert shuffled["anchor_item_id"] == "shirt-1"
    assert [item["item_id"] for item in shuffled["board_items"]].count("shirt-1") == 1
    assert shuffled["revision"] == 2


# ---------------------------------------------------------------------------
# Explicit Style This anchor readiness contract (N, O from the
# readiness-gate implementation spec)
#
# A user-selected anchor that lacks a genuine board-safe image must fail
# with a typed error - never silently swap to a different garment, never
# render the raw photo. Only enforced on the real production path
# (http_request present); direct/legacy callers keep the existing
# allow_missing_image leniency.
# ---------------------------------------------------------------------------


def test_explicit_not_ready_anchor_returns_typed_error(monkeypatch):
    not_ready_anchor = {
        "id": "shirt-bad",
        "item_id": "shirt-bad",
        "name": "Fabricated Alias Shirt",
        "category": "Tops",
        "source": "wardrobe",
        "image_url": "https://images.test/shirt-bad.png",
        "masked_url": "https://images.test/shirt-bad.png",  # aliases raw - not a real cutout
    }
    wardrobe = [not_ready_anchor] + _wardrobe()
    client = _http_client(monkeypatch, wardrobe)

    response = client.post(
        "/api/stylist/items/shirt-bad/style",
        json={
            "user_id": "owner-1",
            "mode": "style_this",
            "anchor_item_id": "shirt-bad",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    # canonical_style_this_anchor() now delegates entirely to
    # resolve_board_image_candidate (services.style_board_image_readiness),
    # which alias-checks masked_url against image_url too -- so this
    # fabricated-alias item is rejected one layer earlier, inside anchor
    # resolution itself, rather than passing an unsafe anchor through and
    # being caught by a separate is_board_renderable() check downstream.
    # Both ANCHOR_IMAGE_NOT_BOARD_READY (the old, later checkpoint) and
    # STYLE_THIS_ANCHOR_UNAVAILABLE (the new, earlier one) are typed errors
    # that correctly refuse to style this item -- neither ever exposes the
    # aliased image.
    assert body["error"]["code"] in {"ANCHOR_IMAGE_NOT_BOARD_READY", "STYLE_THIS_ANCHOR_UNAVAILABLE"}
    # The item is not silently replaced - no directions are generated at all.
    assert body["style_directions"] == []


def test_explicit_ready_anchor_unchanged_behavior(monkeypatch):
    wardrobe = _wardrobe()
    client = _http_client(monkeypatch, wardrobe)

    response = client.post(
        "/api/stylist/items/shirt-1/style",
        json={
            "user_id": "owner-1",
            "mode": "style_this",
            "anchor_item_id": "shirt-1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["anchor_item_id"] == "shirt-1"
    for direction in body["style_directions"]:
        assert direction["anchor_item_id"] == "shirt-1"
