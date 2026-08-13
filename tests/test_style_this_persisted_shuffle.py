from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import style_boards, stylist
from services import style_board_shuffle_service as shuffle_service
from services.style_board_state_store import (
    BoardStateStoreError,
    InMemoryBoardStateStore,
)


def _item(item_id, name, category, **extra):
    item = {
        "id": item_id,
        "name": name,
        "category": category,
        "source": "wardrobe",
        "image_url": f"https://images.test/{item_id}.png",
    }
    item.update(extra)
    return item


def _wardrobe():
    return [
        _item(
            "top-1",
            "White Shirt",
            "Tops",
            masked_url="https://images.test/top-1-masked.png",
            selected_image_source="masked_url",
        ),
        _item("bottom-1", "Blue Jeans", "Bottoms"),
        _item("bottom-2", "Grey Trousers", "Bottoms"),
        _item("bottom-3", "Black Skirt", "Bottoms"),
        _item("shoe-1", "White Sneakers", "Footwear"),
        _item("shoe-2", "Black Boots", "Footwear"),
        _item("shoe-3", "Tan Sandals", "Footwear"),
        _item("bag-1", "Black Bag", "Accessories"),
        _item("watch-1", "Silver Watch", "Accessories"),
        _item("belt-1", "Tan Belt", "Accessories"),
    ]


def _request():
    wardrobe = _wardrobe()
    return stylist.ItemStyleRequest(
        user_id="owner-1",
        mode="style_this",
        anchor_item=wardrobe[0],
        wardrobe=wardrobe,
        weather={"status": "unavailable"},
    )


@pytest.fixture(autouse=True)
def _state_store():
    shuffle_service.set_state_store(InMemoryBoardStateStore())
    yield
    shuffle_service.set_state_store(None)


def _style_this():
    result = stylist.style_wardrobe_item("top-1", _request())
    assert result["success"] is True
    return result


def test_each_direction_has_unique_persisted_uuid_revision_one():
    result = _style_this()
    directions = result["style_directions"]

    board_ids = [direction["board_id"] for direction in directions]
    assert len(board_ids) == len(set(board_ids)) == 3
    assert all(str(UUID(board_id)) == board_id for board_id in board_ids)
    for direction in directions:
        assert direction["revision"] == 1
        assert direction["scenario"] == "style_this"
        assert direction["interaction_mode"] == "style_this"
        assert direction["source_policy"] == "wardrobe"
        assert direction["shuffle_available"] is True
        assert direction["can_shuffle"] is True
        state = shuffle_service.get_board_state(direction["board_id"])
        assert state is not None
        assert state["revision"] == 1
        assert state["source_policy"] == direction["source_policy"]
        assert state["style_strategy"] == direction["style_strategy"]
        assert state["style_strategy"]["archetype_id"] == direction["archetype_id"]


def test_anchor_is_present_and_is_the_only_locked_item():
    for direction in _style_this()["style_directions"]:
        assert direction["items"] == direction["board_items"]
        locked = [item for item in direction["items"] if item["locked"]]
        assert [item["item_id"] for item in locked] == ["top-1"]
        assert all(not item["locked"] for item in direction["items"] if item["item_id"] != "top-1")
        anchor = locked[0]
        assert anchor["source"] == "wardrobe"
        assert anchor["masked_url"] == "https://images.test/top-1-masked.png"
        assert anchor["selected_image_source"] == "masked_url"
        assert isinstance(anchor["position"], dict)
        assert all(item["source"] == "wardrobe" for item in direction["items"])


def test_direction_titles_and_reasons_are_archetype_and_item_specific():
    directions = _style_this()["style_directions"]
    assert len({direction["title"] for direction in directions}) == 3
    assert all(direction["title"] not in {"Casual Brunch", "Date Night", "Vacation Day"} for direction in directions)
    for direction in directions:
        selected_names = [item["name"] for item in direction["items"]]
        note = direction["styling_note"]
        assert "White Shirt" in note
        assert any(name in note for name in selected_names if name != "White Shirt")
        assert direction["title"] in note


def test_registration_rejects_mixed_source_policy():
    wardrobe = _wardrobe()
    direction = {
        "title": "Mixed Direction",
        "items": [
            wardrobe[0],
            {
                "asset_id": "asset-bottom-1",
                "name": "Pleated Trousers",
                "category": "Bottoms",
                "source": "style_asset",
                "image_url": "https://images.test/asset-bottom-1.png",
            },
            wardrobe[4],
        ],
    }

    registered = stylist._register_style_this_direction(
        direction,
        anchor=wardrobe[0],
        wardrobe=wardrobe,
        user_id="owner-1",
        occasion=None,
    )

    assert registered["source_policy"] is None
    assert registered["shuffle_available"] is False
    assert registered["shuffle_state_error"]["code"] == "INSUFFICIENT_WARDROBE"
    assert shuffle_service.get_board_state(registered["board_id"]) is None


class _OutageStore:
    def create_revision(self, **kwargs):
        raise BoardStateStoreError("outage")

    def get_revision(self, board_id, revision):
        raise BoardStateStoreError("outage")

    def get_latest(self, board_id):
        raise BoardStateStoreError("outage")


def test_registration_failure_disables_shuffle_with_typed_error():
    shuffle_service.set_state_store(_OutageStore())

    directions = stylist.style_wardrobe_item("top-1", _request())["style_directions"]

    assert all(direction["shuffle_available"] is False for direction in directions)
    assert all(direction["can_shuffle"] is False for direction in directions)
    assert all(
        direction["shuffle_state_error"]["code"] == "BOARD_STATE_UNAVAILABLE"
        for direction in directions
    )


def test_missing_anchor_is_not_registered(monkeypatch):
    monkeypatch.setattr(
        stylist,
        "_lite_directions",
        lambda *args, **kwargs: [
            {
                "title": "Broken",
                "items": [
                    {"item_id": "top-x", "category": "Tops"},
                    {"item_id": "bottom-x", "category": "Bottoms"},
                    {"item_id": "shoe-1", "category": "Footwear"},
                ],
            }
        ],
    )

    direction = stylist.style_wardrobe_item("top-1", _request())["style_directions"][0]

    assert direction["shuffle_available"] is False
    assert direction["shuffle_state_error"]["code"] == "INVALID_ANCHOR_ITEM"
    assert shuffle_service.get_board_state(direction["board_id"]) is None


def test_incomplete_style_this_creates_no_revision_one(monkeypatch):
    wardrobe = [_wardrobe()[0]]
    request = stylist.ItemStyleRequest(
        user_id="owner-1",
        mode="style_this",
        anchor_item=wardrobe[0],
        wardrobe=wardrobe,
    )
    register_calls = []
    monkeypatch.setattr(
        stylist,
        "register_board",
        lambda **kwargs: register_calls.append(kwargs) or {"ok": True},
    )

    result = stylist.style_wardrobe_item("top-1", request)

    assert result["success"] is False
    assert register_calls == []
    assert all(
        not direction.get("shuffle_available")
        for direction in result["style_directions"]
    )


def test_registration_helper_cannot_persist_incomplete_direction():
    wardrobe = _wardrobe()
    registered = stylist._register_style_this_direction(
        {"title": "Incomplete", "items": [wardrobe[0]]},
        anchor=wardrobe[0],
        wardrobe=wardrobe,
        user_id="owner-1",
        occasion=None,
    )

    assert registered["shuffle_available"] is False
    assert registered["shuffle_state_error"]["code"] == "BOARD_REGISTRATION_INVALID"
    assert shuffle_service.get_board_state(registered["board_id"]) is None


def test_dress_and_footwear_style_this_remains_complete_and_persisted():
    wardrobe = [
        _item("dress-1", "Midnight Dress", "Dresses"),
        _item("shoe-1", "Black Heels", "Footwear"),
    ]
    request = stylist.ItemStyleRequest(
        user_id="owner-1",
        mode="style_this",
        anchor_item=wardrobe[0],
        wardrobe=wardrobe,
    )

    result = stylist.style_wardrobe_item("dress-1", request)

    assert result["success"] is True
    for direction in result["style_directions"]:
        assert direction["shuffle_available"] is True
        assert shuffle_service.get_board_state(direction["board_id"])["revision"] == 1


def test_style_this_board_shuffles_revision_one_to_two_via_route():
    direction = _style_this()["style_directions"][0]
    anchor_before = next(item for item in direction["items"] if item["locked"])
    unlocked_before = [item for item in direction["items"] if not item["locked"]]

    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "owner-1"}
        return await call_next(request)

    app.include_router(style_boards.router, prefix="/api")
    response = TestClient(app).post(
        f"/api/style-boards/{direction['board_id']}/shuffle",
        json={
            "revision": 1,
            "locked_items": [anchor_before],
            "shuffle_slots": [item["slot"] for item in unlocked_before],
            "exclude_item_ids": [item["item_id"] for item in unlocked_before],
            "source_policy": "inherit",
            "board_items": direction["items"],
            "wardrobe": _wardrobe(),
        },
    )

    assert response.status_code == 200
    shuffled = response.json()
    assert shuffled["success"] is True
    assert shuffled["board_id"] == direction["board_id"]
    assert shuffled["previous_revision"] == 1
    assert shuffled["revision"] == 2
    assert shuffled["style_strategy"]["archetype_id"] == direction["style_strategy"]["archetype_id"]
    anchor_after = next(item for item in shuffled["board_items"] if item["item_id"] == "top-1")
    for field in ("item_id", "image_url", "masked_url", "source", "position"):
        assert anchor_after[field] == anchor_before[field]
    old_ids = {item["item_id"] for item in unlocked_before}
    new_ids = {item["item_id"] for item in shuffled["board_items"] if not item["locked"]}
    assert new_ids != old_ids
    state = shuffle_service.get_board_state(direction["board_id"])
    assert state["style_strategy"] == direction["style_strategy"]

    stale = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=1,
        locked_items=[anchor_before],
        shuffle_slots=["bottom"],
        source_policy="inherit",
        wardrobe=_wardrobe(),
        user_id="owner-1",
    )
    assert stale["error"]["code"] == "BOARD_REVISION_CONFLICT"


def test_duplicate_candidates_do_not_create_duplicate_replacements():
    direction = _style_this()["style_directions"][0]
    anchor = next(item for item in direction["items"] if item["locked"])
    wardrobe = _wardrobe()

    result = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=1,
        locked_items=[anchor],
        shuffle_slots=["bottom", "footwear", "accessory"],
        exclude_item_ids=[item["item_id"] for item in direction["items"] if not item["locked"]],
        source_policy="inherit",
        wardrobe=wardrobe + [dict(wardrobe[1]), dict(wardrobe[1])],
        user_id="owner-1",
    )

    assert result["success"] is True
    ids = [item["item_id"] for item in result["board_items"]]
    assert len(ids) == len(set(ids))


def test_other_user_is_denied_and_unknown_board_is_not_found():
    direction = _style_this()["style_directions"][0]
    anchor = next(item for item in direction["items"] if item["locked"])
    kwargs = {
        "revision": 1,
        "locked_items": [anchor],
        "shuffle_slots": ["bottom"],
        "source_policy": "inherit",
        "wardrobe": _wardrobe(),
    }

    denied = shuffle_service.shuffle_board(
        board_id=direction["board_id"], user_id="attacker", **kwargs
    )
    unknown = shuffle_service.shuffle_board(
        board_id="8bbef195-1e78-4ad7-88db-5d81e33f25b8",
        user_id="owner-1",
        **kwargs,
    )

    assert denied["error"]["code"] == "BOARD_FORBIDDEN"
    assert unknown["error"]["code"] == "BOARD_STATE_NOT_FOUND"
