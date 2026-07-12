from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.style_boards import router as style_boards_router
from routers.stylist import router as stylist_router
from services.style_board_shuffle_service import reset_registry


def _item(item_id, name, slot, source="wardrobe", subtype=None):
    item = {
        "item_id": item_id,
        "name": name,
        "slot": slot,
        "source": source,
        "image_url": f"https://example.test/{item_id}.png",
    }
    if subtype:
        item["accessory_type"] = subtype
    return item


def _client():
    app = FastAPI()
    app.include_router(stylist_router, prefix="/api/stylist")
    app.include_router(style_boards_router, prefix="/api")
    return TestClient(app)


def _wardrobe():
    return [
        _item("blazer-1", "Navy Blazer", "outerwear"),
        _item("shirt-1", "White Shirt", "top"),
        _item("shirt-2", "Blue Shirt", "top"),
        _item("trouser-1", "Charcoal Trousers", "bottom"),
        _item("trouser-2", "Navy Trousers", "bottom"),
        _item("shoe-1", "Brown Loafers", "footwear"),
        _item("shoe-2", "Black Loafers", "footwear"),
        _item("watch-1", "Steel Watch", "accessory", subtype="watch"),
        _item("watch-2", "Leather Watch", "accessory", subtype="watch"),
        _item("bag-1", "Leather Bag", "accessory", subtype="bag"),
        _item("bag-2", "Canvas Bag", "accessory", subtype="bag"),
    ]


def test_build_outfit_to_multiple_lock_shuffle_and_revision_conflict():
    reset_registry()
    client = _client()
    wardrobe = _wardrobe()
    built = client.post(
        "/api/stylist/items/blazer-1/style",
        json={
            "user_id": "connected-test",
            "mode": "build_outfit",
            "occasion": "client meeting",
            "anchor_item": wardrobe[0],
            "wardrobe": wardrobe,
            "style_assets": [],
        },
    )
    assert built.status_code == 200
    board = built.json()["outfit"]
    assert board["board_id"] and board["revision"] == 1
    assert all(item.get("position") for item in board["items"])

    anchor = next(item for item in board["items"] if item["item_id"] == "blazer-1")
    shoe = next(item for item in board["items"] if item["slot"] == "footwear")
    assert anchor["locked"] is True
    locked = [{**anchor, "locked": True}, {**shoe, "locked": True}]
    locked_before = {
        item["item_id"]: {
            "source": item["source"],
            "slot": item["slot"],
            "image_url": item["image_url"],
            "position": item["position"],
        }
        for item in locked
    }
    shuffle_slots = sorted({
        item["slot"]
        for item in board["items"]
        if item["item_id"] not in locked_before
    })
    payload = {
        "user_id": "connected-test",
        "scenario": "shuffle_unlocked",
        "revision": board["revision"],
        "occasion": "client meeting",
        "style_direction": "business professional",
        "locked_items": locked,
        "shuffle_slots": shuffle_slots,
        "exclude_item_ids": [
            item["item_id"] for item in board["items"]
            if item["item_id"] not in locked_before
        ],
        "source_policy": "inherit",
        "board_items": board["items"],
        "wardrobe": wardrobe,
    }
    shuffled = client.post(
        f"/api/style-boards/{board['board_id']}/shuffle", json=payload
    )
    assert shuffled.status_code == 200
    result = shuffled.json()
    assert result["success"] is True, result
    assert result["board_id"] == board["board_id"]
    assert result["revision"] == 2
    assert result["previous_revision"] == 1
    assert result["locked_items_preserved"] is True
    assert "outerwear" not in result["changed_slots"]
    assert "footwear" not in result["changed_slots"]
    assert set(result["changed_slots"]).issubset(set(shuffle_slots))
    returned = {item["item_id"]: item for item in result["board_items"]}
    for item_id, expected in locked_before.items():
        actual = returned[item_id]
        assert {key: actual[key] for key in expected} == expected

    stale = client.post(
        f"/api/style-boards/{board['board_id']}/shuffle", json=payload
    ).json()
    assert stale["success"] is False
    assert stale["error"]["code"] == "BOARD_REVISION_CONFLICT"
    assert stale["error"]["current_revision"] == 2


def test_style_this_uses_assets_and_empty_pool_is_typed():
    client = _client()
    anchor = _item("blazer-1", "Navy Blazer", "outerwear")
    assets = [
        _item("asset-shirt", "White Shirt", "top", source="style_asset"),
        _item("asset-trouser", "Trousers", "bottom", source="style_asset"),
        _item("asset-shoe", "Loafers", "footwear", source="style_asset"),
        _item("asset-watch", "Watch", "accessory", source="style_asset", subtype="watch"),
    ]
    response = client.post(
        "/api/stylist/items/blazer-1/style",
        json={
            "user_id": "connected-test",
            "mode": "style_this",
            "anchor_item": anchor,
            "wardrobe": [anchor],
            "style_assets": assets,
            "allow_wardrobe_fallback": False,
        },
    ).json()
    assert response["success"] is True
    assert response["completion_source"] == "style_asset"
    for board in response["style_directions"]:
        assert board["board_id"] and board["revision"] == 1
        selected = next(i for i in board["items"] if i["item_id"] == "blazer-1")
        assert selected["locked"] is True
        assert selected["source"] == "wardrobe"
        assert all(i["source"] == "style_asset" for i in board["items"] if not i["locked"])

    empty = client.post(
        "/api/stylist/items/blazer-1/style",
        json={
            "user_id": "connected-test",
            "mode": "style_this",
            "anchor_item": anchor,
            "wardrobe": [anchor],
            "style_assets": [],
            "allow_wardrobe_fallback": False,
        },
    ).json()
    assert empty["success"] is False
    assert empty["error"]["code"] == "STYLE_ASSET_POOL_EMPTY"
