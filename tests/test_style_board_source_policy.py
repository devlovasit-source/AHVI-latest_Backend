"""Board-level source policy: persistence, inheritance, Style This shuffle.

The board's completion-source policy is an explicit persisted contract:
- Build Outfit boards  -> wardrobe
- Style This boards    -> style_asset
- explicit mixed       -> mixed (style_asset + wardrobe)
It is NEVER inferred from the sources of locked items - a wardrobe anchor
inside a Style This board keeps the board style-asset-based.
"""

import copy

import pytest

from routers.stylist import _builder_outfit
from services import style_board_shuffle_service as sbs
from services.style_board_state_store import InMemoryBoardStateStore


@pytest.fixture(autouse=True)
def _clean_registry():
    # Explicitly injected test double — production defaults to Appwrite.
    sbs.set_state_store(InMemoryBoardStateStore())
    yield
    sbs.set_state_store(None)


def _item(item_id, name, category, source, **extra):
    row = {
        "id": item_id,
        "name": name,
        "category": category,
        "source": source,
        "image_url": f"https://img/{item_id}.png",
    }
    row.update(extra)
    return row


def _wardrobe():
    return [
        _item("w-top-1", "White Oxford Shirt", "Tops", "wardrobe"),
        _item("w-top-2", "Black Tee", "Tops", "wardrobe"),
        _item("w-bottom-1", "Blue Jeans", "Bottoms", "wardrobe"),
        _item("w-bottom-2", "Grey Trousers", "Bottoms", "wardrobe"),
        _item("w-shoe-1", "White Sneakers", "Footwear", "wardrobe"),
        _item("w-shoe-2", "Black Heels", "Footwear", "wardrobe"),
    ]


def _style_assets():
    return [
        _item("a-top-1", "Silk Blouse", "Tops", "style_asset"),
        _item("a-bottom-1", "Tailored Trousers", "Bottoms", "style_asset"),
        _item("a-bottom-2", "Pleated Skirt", "Bottoms", "style_asset"),
        _item("a-shoe-1", "Strappy Heels", "Footwear", "style_asset"),
        _item("a-shoe-2", "Ballet Flats", "Footwear", "style_asset"),
    ]


_POS = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.3, "z": 1, "rotation": 5}


def _wardrobe_anchor():
    return {
        "item_id": "w-top-1",
        "slot": "top",
        "role": "top",
        "source": "wardrobe",
        "image_url": "https://img/w-top-1.png",
        "position": copy.deepcopy(_POS),
    }


def _shuffle(board_id, revision, source_policy, **kw):
    return sbs.shuffle_board(
        board_id=board_id,
        revision=revision,
        locked_items=kw.pop("locked", [_wardrobe_anchor()]),
        shuffle_slots=kw.pop("slots", ["bottom", "footwear"]),
        exclude_item_ids=kw.pop("exclude_item_ids", []),
        occasion=kw.pop("occasion", None),
        source_policy=source_policy,
        wardrobe=kw.pop("wardrobe", _wardrobe()),
        style_assets=kw.pop("style_assets", _style_assets()),
        context=kw.pop("context", {}),
    )


def _completion_sources(result, locked_ids):
    return {
        i["source"] for i in result["board_items"]
        if i.get("item_id") not in locked_ids
    }


# ------------------------------------------------ initial generation contract

def test_build_outfit_initial_contract_is_wardrobe():
    anchor = dict(_wardrobe()[0])
    outfit, meta = _builder_outfit(anchor, _wardrobe(), [], None, mode="build_outfit")
    assert outfit["scenario"] == "build_outfit"
    assert outfit["source_policy"] == "wardrobe"
    assert outfit["allow_wardrobe_fallback"] is False
    assert meta["source_policy"] == "wardrobe"
    state = sbs.get_board_state(outfit["board_id"])
    assert state["scenario"] == "build_outfit"
    assert state["source_policy"] == "wardrobe"
    assert state["revision"] == 1


def test_style_this_initial_contract_is_style_asset():
    anchor = dict(_wardrobe()[0])
    outfit, meta = _builder_outfit(
        anchor, _wardrobe(), _style_assets(), None, mode="style_this"
    )
    assert outfit["scenario"] == "style_this"
    assert outfit["source_policy"] == "style_asset"
    assert outfit["allow_wardrobe_fallback"] is False
    assert meta["source_policy"] == "style_asset"
    state = sbs.get_board_state(outfit["board_id"])
    assert state["scenario"] == "style_this"
    assert state["source_policy"] == "style_asset"


def test_explicit_mixed_style_this_contract():
    anchor = dict(_wardrobe()[0])
    outfit, meta = _builder_outfit(
        anchor, _wardrobe(), _style_assets(), None,
        mode="style_this", allow_wardrobe_fallback=True,
    )
    assert outfit["source_policy"] == "mixed"
    assert outfit["allow_wardrobe_fallback"] is True
    assert meta["source_policy"] == "mixed"
    assert sbs.get_board_state(outfit["board_id"])["source_policy"] == "mixed"


# ------------------------------------------------ registry persistence

def test_registered_policy_survives_revisions():
    sbs.register_board(
        "board-1", revision=1, scenario="style_this", source_policy="style_asset"
    )
    first = _shuffle("board-1", 1, "inherit")
    assert first["success"] is True
    assert first["source_policy"] == "style_asset"
    state = sbs.get_board_state("board-1")
    assert state["revision"] == 2
    assert state["source_policy"] == "style_asset"
    assert state["scenario"] == "style_this"
    # previous-revision (undo) snapshot lives inside the same entry, which
    # keeps its policy.
    assert state["previous"] is not None


# ------------------------------------------------ inheritance resolution

def test_inherit_resolves_from_stored_policy_not_locked_anchor():
    # Wardrobe anchor locked on a Style This board: inherit MUST resolve to
    # style_asset (stored), never wardrobe (anchor source).
    sbs.register_board(
        "board-sty", revision=1, scenario="style_this", source_policy="style_asset"
    )
    result = _shuffle("board-sty", 1, "inherit")
    assert result["success"] is True
    assert result["source_policy"] == "style_asset"
    assert _completion_sources(result, {"w-top-1"}) == {"style_asset"}


def test_unknown_board_requires_regeneration():
    # Unregistered boards are never self-registered: durable state is the
    # only source of truth.
    result = _shuffle("legacy-board", 1, "inherit")
    assert result["success"] is False
    assert result["error"]["code"] == "BOARD_STATE_NOT_FOUND"
    assert sbs.get_board_state("legacy-board") is None


def test_legacy_board_without_stored_policy_fails_typed():
    # A stored board whose payload predates the policy contract must fail
    # typed — the policy is NEVER inferred from locked-item sources.
    sbs._get_store().create_revision(
        user_id="u-test",
        board_id="legacy-board",
        revision=1,
        payload={"scenario": "style_this", "items": []},  # no source_policy
    )
    result = _shuffle("legacy-board", 1, "inherit")
    assert result["success"] is False
    assert result["error"]["code"] == "BOARD_SOURCE_POLICY_UNKNOWN"
    # No silent wardrobe fallback: no revision was committed.
    assert sbs.get_board_state("legacy-board")["revision"] == 1


# ------------------------------------------------ Style This shuffle

def test_style_this_shuffle_completes_from_style_assets_only():
    sbs.register_board(
        "board-sty", revision=1, scenario="style_this", source_policy="style_asset"
    )
    result = _shuffle("board-sty", 1, "style_asset")
    assert result["success"] is True
    assert result["revision"] == 2
    assert result["source_policy"] == "style_asset"
    assert result["scenario"] == "style_this"
    locked = next(i for i in result["board_items"] if i["item_id"] == "w-top-1")
    assert locked["source"] == "wardrobe", "fixed wardrobe anchor stays exact"
    assert locked["position"] == _POS
    assert _completion_sources(result, {"w-top-1"}) == {"style_asset"}


def test_repeated_style_this_shuffle_stays_style_asset():
    sbs.register_board(
        "board-sty", revision=1, scenario="style_this", source_policy="style_asset"
    )
    first = _shuffle("board-sty", 1, "style_asset")
    assert first["success"] is True and first["revision"] == 2
    # Second shuffle uses inherit and MUST stay style-asset-based.
    second = _shuffle("board-sty", 2, "inherit")
    assert second["success"] is True and second["revision"] == 3
    assert second["source_policy"] == "style_asset"
    assert _completion_sources(second, {"w-top-1"}) == {"style_asset"}


def test_style_asset_locked_piece_plus_wardrobe_anchor():
    sbs.register_board(
        "board-sty", revision=1, scenario="style_this", source_policy="style_asset"
    )
    locked = [
        _wardrobe_anchor(),
        {
            "item_id": "a-shoe-1", "slot": "footwear", "role": "footwear",
            "source": "style_asset",
            "image_url": "https://img/a-shoe-1.png",
            "position": copy.deepcopy(_POS),
        },
    ]
    result = _shuffle("board-sty", 1, "style_asset", locked=locked, slots=["bottom"])
    assert result["success"] is True
    ids = {i["item_id"] for i in result["board_items"]}
    assert {"w-top-1", "a-shoe-1"} <= ids
    assert _completion_sources(result, {"w-top-1", "a-shoe-1"}) == {"style_asset"}


# ------------------------------------------------ empty asset pool

def test_style_asset_policy_with_empty_pool_fails_typed():
    sbs.register_board(
        "board-sty", revision=1, scenario="style_this", source_policy="style_asset"
    )
    result = _shuffle("board-sty", 1, "style_asset", style_assets=[])
    assert result["success"] is False
    assert result["error"]["code"] == "STYLE_ASSET_POOL_EMPTY"
    # NEVER a silent wardrobe fallback.
    assert sbs.get_board_state("board-sty")["revision"] == 1


# ------------------------------------------------ mixed policy

def test_mixed_policy_allows_both_sources_and_reports_mixed():
    sbs.register_board(
        "board-mix", revision=1, scenario="style_this",
        source_policy="mixed", allow_wardrobe_fallback=True,
    )
    result = _shuffle("board-mix", 1, "mixed")
    assert result["success"] is True
    assert result["source_policy"] == "mixed"
    assert result["allow_wardrobe_fallback"] is True
    assert _completion_sources(result, {"w-top-1"}) <= {"style_asset", "wardrobe"}


def test_mixed_is_never_implicit_for_style_asset_board():
    sbs.register_board(
        "board-sty", revision=1, scenario="style_this", source_policy="style_asset"
    )
    result = _shuffle("board-sty", 1, "inherit", style_assets=_style_assets())
    assert result["source_policy"] == "style_asset"
    assert result["allow_wardrobe_fallback"] is False


# ------------------------------------------------ build outfit regression

def test_build_outfit_shuffle_remains_wardrobe_only():
    sbs.register_board(
        "board-bo", revision=1, scenario="build_outfit", source_policy="wardrobe"
    )
    result = _shuffle("board-bo", 1, "wardrobe")
    assert result["success"] is True
    assert result["source_policy"] == "wardrobe"
    assert result["scenario"] == "build_outfit"
    assert _completion_sources(result, {"w-top-1"}) == {"wardrobe"}


# ------------------------------------------------ violation guard

def test_wardrobe_completion_under_style_asset_policy_is_violation():
    """A candidate mislabeled style_asset upstream but canonicalizing to
    wardrobe must be caught by the final serialized-source validation."""
    sbs.register_board(
        "board-sty", revision=1, scenario="style_this", source_policy="style_asset"
    )
    # Poison the pool: builder-level filter uses canonical source, so build a
    # legit result then tamper via monkeypatched builder output instead.
    import services.style_board_shuffle_service as svc

    class _TamperBuilder:
        def generate(self, **kwargs):
            return {
                "success": True,
                "scenario": "shuffle_unlocked",
                "occasion": None,
                "source": {"allowed_sources": ["style_asset"]},
                "items": [
                    {**_wardrobe_anchor(), "locked": True},
                    {
                        "item_id": "w-bottom-1", "id": "w-bottom-1",
                        "slot": "bottom", "role": "bottom",
                        "source": "wardrobe", "locked": False,
                        "image_url": "https://img/w-bottom-1.png",
                    },
                ],
                "changed_slots": ["bottom"],
                "missing_items": [],
            }

    original = svc._builder
    svc._builder = _TamperBuilder()
    try:
        result = _shuffle("board-sty", 1, "style_asset", slots=["bottom"])
    finally:
        svc._builder = original
    assert result["success"] is False
    assert result["error"]["code"] == "SOURCE_POLICY_VIOLATION"
    assert sbs.get_board_state("board-sty")["revision"] == 1


def test_fixed_wardrobe_anchor_is_not_a_violation():
    sbs.register_board(
        "board-sty", revision=1, scenario="style_this", source_policy="style_asset"
    )
    result = _shuffle("board-sty", 1, "style_asset")
    assert result["success"] is True, result.get("error")
