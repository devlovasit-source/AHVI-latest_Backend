"""style_board_shuffle_service: revisions, locks, layout preservation, undo."""

import copy

import pytest

from routers import style_boards
from services import style_board_shuffle_service as sbs
from services.style_board_image_readiness import is_board_renderable
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
        # A real (non-aliased) processed image, so every fixture item is
        # board-renderable by default. Readiness gating itself is covered by
        # tests/test_style_board_image_readiness.py, not by every test here.
        "normalized_url": f"https://img/{item_id}-normalized.png",
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


def test_router_resolves_sanitized_inline_wardrobe():
    request = style_boards.BoardShuffleRequest(
        user_id="u-test",
        wardrobe=[
            _w("shirt-misc", "Blue Linen Shirt", "misc"),
            _w("charger", "USB Phone Charger", "accessory"),
        ],
    )

    wardrobe, trusted = style_boards._resolve_wardrobe(request)

    assert [item["id"] for item in wardrobe] == ["shirt-misc"]
    assert trusted is False


def test_router_resolves_sanitized_wardrobe_excludes_observed_junk():
    # Integration proof: the exact device-observed non-fashion objects never
    # reach the Shuffle candidate pool, whatever category a bad row happens
    # to carry (real persisted junk rows have been seen mislabeled as
    # "Accessories").
    junk = [
        _w("bottle-1", "Water Bottle", "Accessories"),
        _w("bottle-2", "Sports Water Bottle", ""),
        _w("flask-1", "Drink Flask", "misc"),
        _w("charger-1", "Phone Charger", "Accessories"),
        _w("powerbank-1", "Power Bank", ""),
        _w("passport-1", "Passport", "Accessories"),
        _w("passport-2", "Passport Bag", "Accessories"),
        _w("travelbag-1", "Travel Document Bag", "Accessories"),
    ]
    request = style_boards.BoardShuffleRequest(
        user_id="u-test",
        wardrobe=[*junk, _w("shirt-ok", "White Oxford Shirt", "Tops")],
    )

    wardrobe, _trusted = style_boards._resolve_wardrobe(request)

    assert [item["id"] for item in wardrobe] == ["shirt-ok"]


def test_router_resolves_sanitized_appwrite_wardrobe(monkeypatch):
    class FakeProxy:
        def list_documents(self, resource, *, user_id):
            assert resource == "outfits"
            assert user_id == "bound-user"
            return [
                _w("sneaker-unknown", "White Leather Sneakers", "unknown", source=""),
                _w("charger", "USB Phone Charger", "misc", source=""),
            ]

    monkeypatch.setattr(style_boards, "AppwriteProxy", FakeProxy)
    request = style_boards.BoardShuffleRequest(user_id="body-user")

    wardrobe, trusted = style_boards._resolve_wardrobe(request, "bound-user")

    assert [item["id"] for item in wardrobe] == ["sneaker-unknown"]
    assert wardrobe[0]["source"] == "wardrobe"
    assert trusted is True


_POS = {"x": 0.123, "y": 0.456, "width": 0.31, "height": 0.29, "z": 3, "rotation": -5}


def _locked_top():
    return {
        "item_id": "top-1",
        "slot": "top",
        "role": "top",
        "source": "wardrobe",
        "image_url": "https://img/top-1.png",
        # A genuinely different processed field, not just a raw photo: a
        # raw-image-only locked item is not board-renderable (confirmed
        # against lib/util/wardrobe_image_resolver.dart - sourceKind
        # 'original' is never board-safe) and ConstrainedOutfitBuilder
        # rejects it with FIXED_ITEM_NOT_BOARD_SAFE. This fixture predates
        # that gate; see test_fixed_item_matrix_rejects_unsafe_forms in
        # test_constrained_outfit_builder.py for the dedicated raw-only
        # regression instead of modeling it here implicitly.
        "masked_url": "https://img/top-1-masked.png",
        "position": copy.deepcopy(_POS),
    }


def _locked_bottom():
    return {
        "item_id": "bottom-1",
        "slot": "bottom",
        "role": "bottom",
        "source": "wardrobe",
        "image_url": "https://img/bottom-1.png",
        "masked_url": "https://img/bottom-1-masked.png",
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


def test_legacy_board_without_style_strategy_still_shuffles():
    result = _shuffle()
    assert result["success"] is True
    assert result["style_strategy"] is None
    assert sbs.get_board_state("b-1")["style_strategy"] is None


def test_multiple_locks_all_preserved():
    locked = [
        _locked_top(),
        {"item_id": "shoe-1", "slot": "footwear", "role": "footwear",
         "source": "wardrobe", "image_url": "https://img/shoe-1.png",
         "masked_url": "https://img/shoe-1-masked.png",
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
    result = _shuffle(locked=[_locked_top(), _locked_bottom()], slots=["footwear"])
    assert result["success"] is True
    changed_roles = {
        i["role"] for i in result["board_items"] if not i["locked"]
    }
    assert changed_roles <= {"footwear"}


def test_replaced_slot_inherits_previous_placement():
    prev_pos = {"x": 0.9, "y": 0.8, "width": 0.2, "height": 0.2, "z": 2, "rotation": 5}
    result = _shuffle(
        locked=[_locked_top(), _locked_bottom()],
        slots=["footwear"],
        context={"board_items": [{"slot": "footwear", "position": prev_pos}]},
    )
    assert result["success"] is True
    new_shoe = next(i for i in result["board_items"] if i["role"] == "footwear")
    assert new_shoe["position"] == prev_pos


def test_excluded_item_not_reused():
    result = _shuffle(
        locked=[_locked_top(), _locked_bottom()],
        slots=["footwear"],
        exclude_item_ids=["shoe-1"],
    )
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


def test_incomplete_shuffle_does_not_create_next_revision():
    result = _shuffle(
        slots=["bottom", "footwear"],
        wardrobe=[_w("bottom-2", "Grey Trousers", "Bottoms")],
    )

    assert result["success"] is False
    assert result["error"]["code"] == "INSUFFICIENT_WARDROBE"
    state = sbs.get_board_state("b-1")
    assert state["revision"] == 1
    assert sbs._get_store().get_revision("b-1", 2) is None


def test_dress_and_footwear_shuffle_is_complete():
    dress = _w("dress-1", "Midnight Dress", "Dresses")
    dress.update({"item_id": "dress-1", "role": "dress", "slot": "dress"})
    _register()

    result = _shuffle(
        register=False,
        locked=[dress],
        slots=["footwear"],
        wardrobe=[_w("shoe-2", "Black Heels", "Footwear")],
    )

    assert result["success"] is True
    assert result["revision"] == 2
    assert result["locked_items_preserved"] is True
    assert result["source_policy"] == "wardrobe"


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


# ---------------------------------------------- locked non-fashion lifecycle


@pytest.mark.parametrize(
    "name,role",
    [
        ("Passport", "accessory"),
        ("Passport Bag", "accessory"),
        ("Travel Document Bag", "accessory"),
        ("Water Bottle", "accessory"),
    ],
)
def test_locked_non_fashion_item_rejects_typed_before_mutation(name, role):
    # A legacy locked item (saved before the sanitizer existed, or before it
    # covered travel-document items) must not be preserved forever merely
    # because assert_fixed_items_preserved guarantees locked items survive.
    bad_locked = {
        "item_id": "bad-1",
        "slot": role,
        "role": role,
        "source": "wardrobe",
        "name": name,
        "image_url": "https://img/bad-1.png",
    }
    result = _shuffle(locked=[bad_locked])

    assert result["success"] is False
    assert result["error"]["code"] == "BOARD_CONTAINS_NON_FASHION_ITEM"
    assert result["error"]["item_ids"] == ["bad-1"]
    # No partial mutation: the board must still be on revision 1.
    state = sbs.get_board_state("b-1")
    assert state["revision"] == 1


@pytest.mark.parametrize(
    "name,role",
    [
        ("Gold Bracelet", "accessory"),
        ("Handbag", "accessory"),
        ("Leather Handbag", "accessory"),
        ("Tote Bag", "accessory"),
        ("Sling Bag", "accessory"),
    ],
)
def test_locked_real_accessory_with_name_still_shuffles(name, role):
    # Fixing the travel-document/passport gap must not collaterally block a
    # locked item that happens to carry a real accessory name.
    good_locked = {
        "item_id": "acc-1",
        "slot": role,
        "role": role,
        "source": "wardrobe",
        "name": name,
        "image_url": "https://img/acc-1.png",
        "masked_url": "https://img/acc-1-masked.png",
    }
    # top/bottom/footwear must all be covered (by lock or shuffle slot) for
    # a complete outfit; only the accessory is under test here.
    result = _shuffle(locked=[good_locked], slots=["top", "bottom", "footwear"])

    assert result["success"] is True
    ids = {i["item_id"] for i in result["board_items"]}
    assert "acc-1" in ids


def test_locked_item_without_name_is_trusted_by_role():
    # A sparse id/role/slot-only locked payload (no name sent back by the
    # client) is the normal shape for this contract - it must keep working,
    # not be treated as unclassifiable-therefore-non-fashion.
    result = _shuffle(locked=[_locked_top()])
    assert result["success"] is True


def test_locked_raw_only_item_rejected_before_mutation():
    # Confirmed against the real Flutter resolver (wardrobe_image_resolver.dart,
    # the commit that built the currently-installed APK): a raw-image-only
    # wardrobe item's only candidate has sourceKind 'original', which the
    # board-surface admission guard excludes unconditionally - it renders
    # nothing. A raw-only locked item must be rejected the same way a
    # fabricated alias is, not silently shuffled through to a hanger.
    raw_only_locked = {
        "item_id": "top-raw-only",
        "slot": "top",
        "role": "top",
        "source": "wardrobe",
        "image_url": "https://img/top-raw-only.png",
    }
    result = _shuffle(locked=[raw_only_locked])
    assert result["success"] is False
    assert result["error"]["code"] == "FIXED_ITEM_NOT_BOARD_SAFE"
    state = sbs.get_board_state("b-1")
    assert state["revision"] == 1


def test_locked_zero_image_item_rejected_before_mutation():
    zero_image_locked = {
        "item_id": "top-zero-image",
        "slot": "top",
        "role": "top",
        "source": "wardrobe",
    }
    result = _shuffle(locked=[zero_image_locked])
    assert result["success"] is False
    assert result["error"]["code"] == "FIXED_ITEM_NOT_BOARD_SAFE"
    state = sbs.get_board_state("b-1")
    assert state["revision"] == 1


def test_locked_masked_only_item_shuffles_successfully_and_renders():
    masked_only_locked = {
        "item_id": "top-masked-only",
        "slot": "top",
        "role": "top",
        "source": "wardrobe",
        "masked_url": "https://img/top-masked-only-masked.png",
    }
    result = _shuffle(locked=[masked_only_locked])
    assert result["success"] is True
    assert result["revision"] == 2
    locked = next(i for i in result["board_items"] if i["item_id"] == "top-masked-only")
    assert locked["locked"] is True
    assert locked["masked_url"] == masked_only_locked["masked_url"]
    assert "image_url" not in locked
    assert is_board_renderable(locked) is True


def test_locked_unknown_role_without_name_is_rejected():
    # id/role-only AND role is unknown/absent: no evidence this is fashion,
    # so it must not be silently trusted either.
    bad_locked = {
        "item_id": "mystery-1",
        "slot": "unknown",
        "role": "unknown",
        "source": "wardrobe",
        "image_url": "https://img/mystery-1.png",
    }
    result = _shuffle(locked=[bad_locked])
    assert result["success"] is False
    assert result["error"]["code"] == "BOARD_CONTAINS_NON_FASHION_ITEM"
    assert sbs.get_board_state("fresh-board") is None
