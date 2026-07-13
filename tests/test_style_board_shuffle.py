"""style_board_shuffle_service: revisions, locks, layout preservation, undo."""

import copy

import pytest

from services import style_board_shuffle_service as sbs
from services.style_board_state_store import InMemoryBoardStateStore


@pytest.fixture(autouse=True)
def _clean_registry():
    # Explicitly injected test double — production defaults to Appwrite.
    sbs.set_state_store(InMemoryBoardStateStore())
    yield
    sbs.set_state_store(None)


def _w(item_id, name, category="", **extra):
    row = {
        "id": item_id,
        "name": name,
        "category": category,
        "source": "wardrobe",
        "image_url": f"https://img/{item_id}.png",
    }
    row.update(extra)
    return row


def _wardrobe():
    return [
        _w("top-1", "White Oxford Shirt", "Tops"),
        _w("top-2", "Black Tee", "Tops"),
        _w("bottom-1", "Blue Jeans", "Bottoms"),
        _w("bottom-2", "Grey Trousers", "Bottoms"),
        _w("shoe-1", "White Sneakers", "Footwear"),
        _w("shoe-2", "Black Heels", "Footwear"),
        _w("watch-1", "Leather Watch", "Accessories"),
    ]


_POS = {"x": 0.123, "y": 0.456, "width": 0.31, "height": 0.29, "z": 3, "rotation": -5}


def _locked_top():
    return {
        "item_id": "top-1",
        "slot": "top",
        "role": "top",
        "source": "wardrobe",
        "image_url": "https://img/top-1.png",
        "position": copy.deepcopy(_POS),
    }


def _register(board_id="b-1", revision=1, source_policy="wardrobe", user_id="u-test"):
    # Boards must exist in durable state before shuffling — unknown boards
    # are no longer self-registered (BOARD_STATE_NOT_FOUND by design).
    result = sbs.register_board(
        board_id=board_id,
        revision=revision,
        scenario="shuffle_unlocked",
        source_policy=source_policy,
        user_id=user_id,
    )
    assert result["ok"], result


def _shuffle(board_id="b-1", revision=1, locked=None, slots=None, register=True, **kw):
    policy = kw.pop("source_policy", "wardrobe")
    if register and revision == 1:
        _register(board_id=board_id, revision=1, source_policy=policy or "wardrobe")
    return sbs.shuffle_board(
        board_id=board_id,
        revision=revision,
        locked_items=locked if locked is not None else [_locked_top()],
        shuffle_slots=slots if slots is not None else ["bottom", "footwear"],
        exclude_item_ids=kw.pop("exclude_item_ids", []),
        occasion=kw.pop("occasion", None),
        source_policy=policy,
        style_assets=kw.pop("style_assets", None),
        wardrobe=kw.pop("wardrobe", _wardrobe()),
        context=kw.pop("context", {}),
        user_id=kw.pop("user_id", "u-test"),
    )


def test_one_lock_shuffles_other_slots():
    result = _shuffle()
    assert result["success"] is True
    ids = {i["item_id"] for i in result["board_items"]}
    assert "top-1" in ids
    locked = next(i for i in result["board_items"] if i["item_id"] == "top-1")
    assert locked["locked"] is True
    assert set(result["changed_slots"]) <= {"bottom", "footwear"}
    assert result["changed_slots"], "unlocked slots should change"


def test_multiple_locks_all_preserved():
    locked = [
        _locked_top(),
        {"item_id": "shoe-1", "slot": "footwear", "role": "footwear",
         "source": "wardrobe", "image_url": "https://img/shoe-1.png",
         "position": copy.deepcopy(_POS)},
    ]
    result = _shuffle(locked=locked, slots=["bottom", "accessory"])
    assert result["success"] is True
    ids = {i["item_id"] for i in result["board_items"]}
    assert {"top-1", "shoe-1"} <= ids
    assert result["locked_items_preserved"] is True


def test_no_locks_shuffles_requested_slots():
    result = _shuffle(locked=[], slots=["top", "bottom", "footwear"])
    assert result["success"] is True
    assert result["changed_slots"]


def test_all_locked_returns_typed_failure():
    result = _shuffle(slots=[])
    assert result["success"] is False
    assert result["error"]["code"] == "ALL_ITEMS_LOCKED"


def test_locked_position_unchanged_byte_for_byte():
    result = _shuffle()
    locked = next(i for i in result["board_items"] if i["item_id"] == "top-1")
    assert locked["position"] == _POS, "locked placement must be untouched"


def test_locked_payload_fields_remain_exact():
    locked_input = _locked_top()
    locked_input.update({"masked_url": "masked", "board_role": "hero", "scale": 1.1})
    result = _shuffle(locked=[locked_input])
    locked = next(i for i in result["board_items"] if i["item_id"] == "top-1")
    for key, value in locked_input.items():
        assert locked[key] == value


def test_only_shuffle_slots_change():
    result = _shuffle(slots=["footwear"])
    assert result["success"] is True
    changed_roles = {
        i["role"] for i in result["board_items"] if not i["locked"]
    }
    assert changed_roles <= {"footwear"}


def test_replaced_slot_inherits_previous_placement():
    prev_pos = {"x": 0.9, "y": 0.8, "width": 0.2, "height": 0.2, "z": 2, "rotation": 5}
    result = _shuffle(
        slots=["footwear"],
        context={"board_items": [{"slot": "footwear", "position": prev_pos}]},
    )
    assert result["success"] is True
    new_shoe = next(i for i in result["board_items"] if i["role"] == "footwear")
    assert new_shoe["position"] == prev_pos


def test_excluded_item_not_reused():
    result = _shuffle(slots=["footwear"], exclude_item_ids=["shoe-1"])
    assert result["success"] is True
    ids = {i["item_id"] for i in result["board_items"]}
    assert "shoe-1" not in ids
    assert "shoe-2" in ids


def test_revision_increments_on_success():
    result = _shuffle(revision=1)
    assert result["success"] is True
    assert result["revision"] == 2
    assert result["previous_revision"] == 1
    state = sbs.get_board_state("b-1")
    assert state["revision"] == 2


def test_stale_revision_conflicts():
    first = _shuffle(revision=1)
    assert first["success"] is True
    stale = _shuffle(revision=1)
    assert stale["success"] is False
    assert stale["error"]["code"] == "BOARD_REVISION_CONFLICT"
    assert stale["error"]["current_revision"] == 2


def test_failure_preserves_registry_state():
    ok = _shuffle(revision=1)
    assert ok["success"] is True
    before = sbs.get_board_state("b-1")
    bad = _shuffle(revision=2, slots=[])  # ALL_ITEMS_LOCKED failure
    assert bad["success"] is False
    after = sbs.get_board_state("b-1")
    assert after["revision"] == before["revision"]
    assert after["items"] == before["items"]


def test_one_level_undo_snapshot_exists():
    first = _shuffle(revision=1)
    assert first["success"] is True
    second = _shuffle(revision=2)
    assert second["success"] is True
    state = sbs.get_board_state("b-1")
    assert state["previous"] is not None
    assert state["previous"]["revision"] == 2
    assert isinstance(state["previous"]["items"], list)


def test_unknown_board_returns_state_not_found():
    # Self-registration of unknown boards is removed: durable state is the
    # only source of truth, and a missing board requires regeneration.
    result = _shuffle(board_id="fresh-board", revision=7, register=False)
    assert result["success"] is False
    assert result["error"]["code"] == "BOARD_STATE_NOT_FOUND"
    assert result["error"].get("action") == "regenerate_board"
    assert sbs.get_board_state("fresh-board") is None
