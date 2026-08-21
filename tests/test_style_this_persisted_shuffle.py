from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import style_boards, stylist
from services import style_board_shuffle_service as shuffle_service
from services.style_board_image_readiness import is_board_renderable
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
        # A real (non-aliased) processed image, so every fixture item is
        # board-renderable by default. Readiness gating itself is covered by
        # tests/test_style_board_image_readiness.py, not by every test here.
        "normalized_url": f"https://images.test/{item_id}-normalized.png",
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


# ---------------------------------------------------------------------------
# Shuffle reasoning (fix/style-this-shuffle-reasoning)
# ---------------------------------------------------------------------------


def test_initial_registration_persists_styling_note():
    """A. initial Style This registration persists styling_note."""
    direction = _style_this()["style_directions"][0]

    assert direction["styling_note"]
    state = shuffle_service.get_board_state(direction["board_id"])
    assert state["styling_note"] == direction["styling_note"]


def _two_top_wardrobe():
    return [
        _item("bottom-1", "Blue Jeans", "Bottoms"),
        _item("shirt-a", "Blue Oxford Shirt", "Tops"),
        _item("shirt-b", "Green Flannel Shirt", "Tops"),
        _item("shoe-1", "White Sneakers", "Footwear"),
    ]


def _style_this_bottom_anchor(wardrobe):
    request = stylist.ItemStyleRequest(
        user_id="owner-1",
        mode="style_this",
        anchor_item=wardrobe[0],
        wardrobe=wardrobe,
        weather={"status": "unavailable"},
    )
    result = stylist.style_wardrobe_item("bottom-1", request)
    assert result["success"] is True
    return result


def test_shuffle_regenerates_reasoning_for_the_new_support_item():
    """B. shuffle changing support item: response styling_note references the
    new item and not the old one."""
    wardrobe = _two_top_wardrobe()
    direction = _style_this_bottom_anchor(wardrobe)["style_directions"][0]
    anchor = next(item for item in direction["items"] if item["locked"])
    old_top_id = next(
        item["item_id"] for item in direction["items"]
        if item["item_id"] in {"shirt-a", "shirt-b"}
    )
    new_top_id = "shirt-b" if old_top_id == "shirt-a" else "shirt-a"
    old_top_name = "Blue Oxford Shirt" if old_top_id == "shirt-a" else "Green Flannel Shirt"
    new_top_name = "Green Flannel Shirt" if old_top_id == "shirt-a" else "Blue Oxford Shirt"

    result = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=1,
        locked_items=[anchor],
        shuffle_slots=["top", "footwear", "accessory"],
        exclude_item_ids=[old_top_id],
        source_policy="inherit",
        wardrobe=wardrobe,
        user_id="owner-1",
    )

    assert result["success"] is True
    new_top = next(item for item in result["board_items"] if item["item_id"] == new_top_id)
    assert new_top["item_id"] != old_top_id
    assert result["styling_note"]
    assert new_top_name in result["styling_note"]
    assert old_top_name not in result["styling_note"]


def test_persisted_shuffled_revision_styling_note_matches_response():
    """C. persisted shuffled revision styling_note exactly matches response."""
    wardrobe = _two_top_wardrobe()
    direction = _style_this_bottom_anchor(wardrobe)["style_directions"][0]
    anchor = next(item for item in direction["items"] if item["locked"])

    result = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=1,
        locked_items=[anchor],
        shuffle_slots=["top", "footwear", "accessory"],
        exclude_item_ids=["shirt-a"],
        source_policy="inherit",
        wardrobe=wardrobe,
        user_id="owner-1",
    )

    assert result["success"] is True
    state = shuffle_service.get_board_state(direction["board_id"])
    assert state["revision"] == 2
    assert state["styling_note"] == result["styling_note"]


def test_accessory_anchor_board_shuffle_regenerates_reasoning():
    """D. accessory-anchor board works through the shuffle reasoning path."""
    wardrobe = [
        _item("bracelet-1", "Gold Bracelet", "Jewelry"),
        _item("shirt-a", "Blue Oxford Shirt", "Tops"),
        _item("shirt-b", "Green Flannel Shirt", "Tops"),
        _item("pants-1", "Black Trousers", "Bottoms"),
        _item("shoe-1", "White Sneakers", "Footwear"),
    ]
    request = stylist.ItemStyleRequest(
        user_id="owner-1",
        mode="style_this",
        anchor_item=wardrobe[0],
        wardrobe=wardrobe,
        weather={"status": "unavailable"},
    )
    result = stylist.style_wardrobe_item("bracelet-1", request)
    assert result["success"] is True
    direction = result["style_directions"][0]
    assert "bracelet-1" in {item["item_id"] for item in direction["items"]}
    assert direction["styling_note"]

    anchor = next(item for item in direction["items"] if item["locked"])
    old_top_id = next(
        item["item_id"] for item in direction["items"]
        if item["item_id"] in {"shirt-a", "shirt-b"}
    )
    new_top_id = "shirt-b" if old_top_id == "shirt-a" else "shirt-a"

    shuffled = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=1,
        locked_items=[anchor],
        shuffle_slots=["top", "bottom", "footwear"],
        exclude_item_ids=[old_top_id],
        source_policy="inherit",
        wardrobe=wardrobe,
        user_id="owner-1",
    )

    assert shuffled["success"] is True
    assert "bracelet-1" in {item["item_id"] for item in shuffled["board_items"]}
    assert shuffled["styling_note"]
    new_top_name = next(
        item["name"] for item in shuffled["board_items"] if item["item_id"] == new_top_id
    )
    assert new_top_name in shuffled["styling_note"]


def test_normal_bottom_anchor_board_shuffle_regenerates_reasoning():
    """E. normal (non-accessory) bottom-anchor board keeps working end to end."""
    direction = _style_this()["style_directions"][0]
    anchor = next(item for item in direction["items"] if item["locked"])
    unlocked_before = [item for item in direction["items"] if not item["locked"]]

    shuffled = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=1,
        locked_items=[anchor],
        shuffle_slots=[item["slot"] for item in unlocked_before],
        exclude_item_ids=[item["item_id"] for item in unlocked_before],
        source_policy="inherit",
        wardrobe=_wardrobe(),
        user_id="owner-1",
    )

    assert shuffled["success"] is True
    assert shuffled["styling_note"]
    assert shuffled["styling_note"] != direction["styling_note"]



# ---------------------------------------------------------------------------
# Durable undo (runtime fix: reasoning state + durable undo)
# ---------------------------------------------------------------------------


def test_undo_restores_previous_revision_content_via_route():
    """1-5. shuffle rev1->rev2, undo rev2->rev3 restores rev1's items,
    styling_note, locks/anchor and strategy exactly."""
    wardrobe = _two_top_wardrobe()
    direction = _style_this_bottom_anchor(wardrobe)["style_directions"][0]
    anchor = next(item for item in direction["items"] if item["locked"])
    original_items = direction["items"]
    original_note = direction["styling_note"]
    original_strategy = direction["style_strategy"]
    old_top_id = next(
        item["item_id"] for item in direction["items"]
        if item["item_id"] in {"shirt-a", "shirt-b"}
    )

    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "owner-1"}
        return await call_next(request)

    app.include_router(style_boards.router, prefix="/api")
    client = TestClient(app)

    shuffled = client.post(
        f"/api/style-boards/{direction['board_id']}/shuffle",
        json={
            "revision": 1,
            "locked_items": [anchor],
            "shuffle_slots": ["top", "footwear", "accessory"],
            "exclude_item_ids": [old_top_id],
            "source_policy": "inherit",
            "board_items": direction["items"],
            "wardrobe": wardrobe,
        },
    ).json()
    assert shuffled["success"] is True
    assert shuffled["revision"] == 2
    assert shuffled["styling_note"] != original_note

    undone = client.post(
        f"/api/style-boards/{direction['board_id']}/undo",
        json={"revision": 2},
    ).json()

    assert undone["success"] is True
    assert undone["revision"] == 3
    assert undone["previous_revision"] == 2
    assert undone["styling_note"] == original_note
    assert undone["style_strategy"] == original_strategy
    assert undone["anchor_item_id"] == "bottom-1"
    restored_ids = {item["item_id"] for item in undone["board_items"]}
    original_ids = {item["item_id"] for item in original_items}
    assert restored_ids == original_ids
    restored_anchor = next(
        item for item in undone["board_items"] if item["item_id"] == "bottom-1"
    )
    assert restored_anchor["locked"] is True

    state = shuffle_service.get_board_state(direction["board_id"])
    assert state["revision"] == 3
    assert state["styling_note"] == original_note


def test_shuffle_after_undo_succeeds_with_new_revision_cursor():
    """6. Part D chain: rev1=A -> shuffle -> rev2=B -> undo -> rev3=A ->
    shuffle -> rev4=C. The second shuffle must send expected_revision=3 and
    succeed (no BOARD_REVISION_CONFLICT), operating on exactly what the undo
    restored."""
    wardrobe = _two_top_wardrobe()
    direction = _style_this_bottom_anchor(wardrobe)["style_directions"][0]
    anchor = next(item for item in direction["items"] if item["locked"])
    old_top_id = next(
        item["item_id"] for item in direction["items"]
        if item["item_id"] in {"shirt-a", "shirt-b"}
    )

    shuffled = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=1,
        locked_items=[anchor],
        shuffle_slots=["top", "footwear", "accessory"],
        exclude_item_ids=[old_top_id],
        source_policy="inherit",
        wardrobe=wardrobe,
        user_id="owner-1",
    )
    assert shuffled["revision"] == 2

    undone = shuffle_service.undo_board(
        board_id=direction["board_id"], revision=2, user_id="owner-1",
    )
    assert undone["success"] is True
    assert undone["revision"] == 3

    second_shuffle = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=3,
        locked_items=[anchor],
        shuffle_slots=["top", "footwear", "accessory"],
        exclude_item_ids=[old_top_id],
        source_policy="inherit",
        wardrobe=wardrobe,
        user_id="owner-1",
    )
    assert second_shuffle.get("error") is None
    assert second_shuffle["success"] is True
    assert second_shuffle["revision"] == 4
    assert second_shuffle["previous_revision"] == 3


def test_double_shuffle_then_undo_restores_immediate_prior_not_revision_one():
    """Part D second scenario: rev1=A -> shuffle -> rev2=B -> shuffle -> rev3=C
    -> undo -> rev4 restores B (the immediate previous user-visible board),
    not A/revision 1."""
    wardrobe = _two_top_wardrobe()
    direction = _style_this_bottom_anchor(wardrobe)["style_directions"][0]
    anchor = next(item for item in direction["items"] if item["locked"])
    old_top_id = next(
        item["item_id"] for item in direction["items"]
        if item["item_id"] in {"shirt-a", "shirt-b"}
    )
    new_top_id = "shirt-b" if old_top_id == "shirt-a" else "shirt-a"

    first = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=1,
        locked_items=[anchor],
        shuffle_slots=["top", "footwear", "accessory"],
        exclude_item_ids=[old_top_id],
        source_policy="inherit",
        wardrobe=wardrobe,
        user_id="owner-1",
    )
    assert first["revision"] == 2
    b_note = first["styling_note"]
    b_top_id = next(
        item["item_id"] for item in first["board_items"]
        if item["item_id"] in {"shirt-a", "shirt-b"}
    )
    assert b_top_id == new_top_id

    second = shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=2,
        locked_items=[anchor],
        shuffle_slots=["top", "footwear", "accessory"],
        exclude_item_ids=[b_top_id],
        source_policy="inherit",
        wardrobe=wardrobe,
        user_id="owner-1",
    )
    assert second["revision"] == 3
    assert second["styling_note"] != b_note

    undone = shuffle_service.undo_board(
        board_id=direction["board_id"], revision=3, user_id="owner-1",
    )
    assert undone["success"] is True
    assert undone["revision"] == 4
    assert undone["styling_note"] == b_note
    undone_top_id = next(
        item["item_id"] for item in undone["board_items"]
        if item["item_id"] in {"shirt-a", "shirt-b"}
    )
    assert undone_top_id == b_top_id


def test_undo_stale_revision_returns_conflict():
    """7. An undo request against a revision that's no longer latest fails
    typed BOARD_REVISION_CONFLICT."""
    direction = _style_this()["style_directions"][0]
    anchor = next(item for item in direction["items"] if item["locked"])
    unlocked = [item for item in direction["items"] if not item["locked"]]
    shuffle_service.shuffle_board(
        board_id=direction["board_id"],
        revision=1,
        locked_items=[anchor],
        shuffle_slots=[item["slot"] for item in unlocked],
        exclude_item_ids=[item["item_id"] for item in unlocked],
        source_policy="inherit",
        wardrobe=_wardrobe(),
        user_id="owner-1",
    )

    stale = shuffle_service.undo_board(
        board_id=direction["board_id"], revision=1, user_id="owner-1",
    )

    assert stale["success"] is False
    assert stale["error"]["code"] == "BOARD_REVISION_CONFLICT"
    assert stale["error"]["current_revision"] == 2
    assert stale["error"]["requested_revision"] == 1


def test_undo_with_no_previous_revision_returns_typed_error():
    """8. Undoing a fresh revision-1 board (nothing to undo to) fails typed
    NO_PREVIOUS_REVISION, and the board is left completely unchanged."""
    direction = _style_this()["style_directions"][0]

    result = shuffle_service.undo_board(
        board_id=direction["board_id"], revision=1, user_id="owner-1",
    )

    assert result["success"] is False
    assert result["error"]["code"] == "NO_PREVIOUS_REVISION"
    state = shuffle_service.get_board_state(direction["board_id"])
    assert state["revision"] == 1


def test_undo_unknown_board_is_not_found():
    result = shuffle_service.undo_board(
        board_id="8bbef195-1e78-4ad7-88db-5d81e33f25b8",
        revision=1,
        user_id="owner-1",
    )
    assert result["error"]["code"] == "BOARD_STATE_NOT_FOUND"


def test_shuffle_without_stored_strategy_returns_deterministic_fallback():
    """F. no-strategy fallback produces the deterministic fallback text."""
    anchor_item = {
        "item_id": "top-1", "name": "White Shirt", "source": "wardrobe", "locked": True,
    }
    bottom_item = {
        "item_id": "bottom-1", "name": "Blue Jeans", "source": "wardrobe", "locked": True,
    }
    footwear_item = {
        "item_id": "shoe-1", "name": "White Sneakers", "source": "wardrobe",
    }
    ok = shuffle_service.register_board(
        board_id="no-strategy-board",
        revision=1,
        scenario="style_this",
        source_policy="wardrobe",
        anchor_item_id="top-1",
        items=[anchor_item, bottom_item, footwear_item],
        style_strategy=None,
        user_id="owner-1",
    )
    assert ok["ok"] is True

    result = shuffle_service.shuffle_board(
        board_id="no-strategy-board",
        revision=1,
        locked_items=[anchor_item, bottom_item],
        shuffle_slots=["footwear"],
        source_policy="inherit",
        wardrobe=_wardrobe(),
        user_id="owner-1",
    )

    assert result["success"] is True
    assert result["styling_note"] == "Built from pieces you already own, anchored on this item."


def test_shuffle_never_selects_not_board_ready_item_and_reasoning_matches_selection():
    """K, L, M from the readiness-gate implementation spec, exercised through
    the real shuffle_board() service end to end (not just the builder unit):
    K. a not-board-ready candidate is never selected by shuffle.
    L. the returned changed item always has a genuine board-safe image.
    M. the reasoning names whichever item was actually selected.
    """
    wardrobe = _two_top_wardrobe() + [
        _item(
            "shirt-bad",
            "Fabricated Alias Shirt",
            "Tops",
            masked_url="https://images.test/shirt-bad.png",  # aliases image_url below
            image_url="https://images.test/shirt-bad.png",
            normalized_url=None,  # override _item()'s default - no safe fallback either
        ),
    ]
    direction = _style_this_bottom_anchor(wardrobe)["style_directions"][0]
    anchor = next(item for item in direction["items"] if item["locked"])
    old_top_id = next(
        item["item_id"] for item in direction["items"]
        if item["item_id"] in {"shirt-a", "shirt-b"}
    )

    for _ in range(5):
        result = shuffle_service.shuffle_board(
            board_id=direction["board_id"],
            revision=shuffle_service.get_board_state(direction["board_id"])["revision"],
            locked_items=[anchor],
            shuffle_slots=["top", "footwear", "accessory"],
            exclude_item_ids=[old_top_id],
            source_policy="inherit",
            wardrobe=wardrobe,
            user_id="owner-1",
        )
        assert result["success"] is True
        new_top = next(item for item in result["board_items"] if item["role"] == "top")
        assert new_top["item_id"] != "shirt-bad", "not-board-ready item must never be selected"
        assert is_board_renderable(new_top), "selected item must have a genuine board-safe image"
        assert result["styling_note"]
        assert new_top["name"] in result["styling_note"], "reasoning must name the actually-selected item"
