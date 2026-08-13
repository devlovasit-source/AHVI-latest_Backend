"""ConstrainedOutfitBuilder: fixed items of every role, exclusions, sources,
dress rules and typed failures."""

import pytest

from services.constrained_outfit_builder import ConstrainedOutfitBuilder
from services.style_item_contract import canonical_item_id

builder = ConstrainedOutfitBuilder()


def _w(item_id, name, category="", source="wardrobe", **extra):
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
        _w("top-1", "White Oxford Shirt", "Tops"),
        _w("top-2", "Black Tee", "Tops"),
        _w("bottom-1", "Blue Jeans", "Bottoms"),
        _w("bottom-2", "Khaki Shorts", "Bottoms"),
        _w("dress-1", "Red Maxi Dress", "Dresses"),
        _w("outer-1", "Navy Blazer", "Outerwear"),
        _w("shoe-1", "White Sneakers", "Footwear"),
        _w("shoe-2", "Black Heels", "Footwear"),
        _w("watch-1", "Leather Watch", "Accessories"),
        _w("bag-1", "Structured Handbag", "Accessories"),
    ]


def _gen(fixed, wardrobe=None, scenario="build_outfit", **kw):
    context = kw.pop("context", {})
    context.setdefault("wardrobe", _wardrobe() if wardrobe is None else wardrobe)
    return builder.generate(
        scenario=scenario,
        fixed_items=fixed,
        context=context,
        **kw,
    )


def _ids(result):
    return {i["item_id"] for i in result["items"]}


# ------------------------------------------------- fixed anchors of every role

@pytest.mark.parametrize(
    "anchor_id",
    ["top-1", "bottom-1", "dress-1", "outer-1", "shoe-1", "watch-1", "bag-1"],
)
def test_fixed_anchor_of_every_role_preserved_exactly(anchor_id):
    anchor = next(i for i in _wardrobe() if i["id"] == anchor_id)
    result = _gen([anchor])
    assert result["success"] is True
    assert anchor_id in _ids(result)
    kept = next(i for i in result["items"] if i["item_id"] == anchor_id)
    assert kept["image_url"] == anchor["image_url"], "anchor image must survive"
    assert kept["locked"] is True
    assert result["items"][0]["item_id"] == anchor_id, "fixed items come first"


def test_two_fixed_items_preserved():
    w = _wardrobe()
    fixed = [w[0], w[6]]  # top-1 + shoe-1
    result = _gen(fixed)
    assert result["success"] is True
    assert {"top-1", "shoe-1"} <= _ids(result)


def test_three_fixed_items_preserved():
    w = _wardrobe()
    fixed = [w[0], w[2], w[6]]  # top-1 + bottom-1 + shoe-1
    result = _gen(fixed)
    assert result["success"] is True
    assert {"top-1", "bottom-1", "shoe-1"} <= _ids(result)


# ------------------------------------------------- dress rules

def test_fixed_dress_generates_no_top_or_bottom():
    dress = next(i for i in _wardrobe() if i["id"] == "dress-1")
    result = _gen([dress])
    assert result["success"] is True
    roles = {i["role"] for i in result["items"]}
    assert "top" not in roles and "bottom" not in roles


def test_sleepwear_one_piece_is_selected_and_composed_as_dress():
    sleepwear = _w(
        "sleepwear-1",
        "Loungewear Set",
        "loungewear",
        source="style_asset",
        role="one_piece",
        subcategory="sleepwear",
    )
    footwear = _w(
        "shoe-1", "Minimal Slides", "footwear", source="style_asset"
    )
    bag = _w(
        "bag-1", "Soft Tote Bag", "accessory", source="style_asset"
    )

    result = builder.generate(
        scenario="style_this",
        fixed_items=[],
        style_assets=[sleepwear, footwear, bag],
        source_policy={"allowed_sources": ["style_asset"]},
        context={"accessory_budget": 1},
    )

    assert result["success"] is True
    roles = {item["role"] for item in result["items"]}
    assert {"dress", "footwear", "accessory"}.issubset(roles)
    assert "top" not in roles and "bottom" not in roles
    assert "sleepwear-1" in _ids(result)


def test_fixed_top_never_adds_dress():
    top = next(i for i in _wardrobe() if i["id"] == "top-1")
    result = _gen([top])
    assert result["success"] is True
    assert all(i["role"] != "dress" for i in result["items"] if not i["locked"])


# ------------------------------------------------- exclusions

def test_excluded_id_never_returned():
    top = next(i for i in _wardrobe() if i["id"] == "top-1")
    result = _gen([top], exclude_item_ids=["bottom-1", "shoe-1"])
    assert result["success"] is True
    assert "bottom-1" not in _ids(result)
    assert "shoe-1" not in _ids(result)


def test_fixed_item_beats_exclusion():
    top = next(i for i in _wardrobe() if i["id"] == "top-1")
    result = _gen([top], exclude_item_ids=["top-1"])
    assert result["success"] is True
    assert "top-1" in _ids(result), "fixed precedence: exclusion never removes a lock"


# ------------------------------------------------- source policy

def test_unknown_source_completion_rejected_under_wardrobe_policy():
    top = next(i for i in _wardrobe() if i["id"] == "top-1")
    mystery = _w("mystery-1", "Blue Jeans", "Bottoms", source="")
    del mystery["source"]
    result = _gen([top], wardrobe=[top, mystery], scenario="build_outfit")
    assert result["success"] is True
    assert "mystery-1" not in _ids(result), "unknown source is NOT wardrobe"


def test_style_asset_completion_rejected_under_wardrobe_policy():
    top = next(i for i in _wardrobe() if i["id"] == "top-1")
    asset = _w("asset-1", "Blue Jeans", "Bottoms", source="asset_library")
    result = _gen([top], wardrobe=[top, asset], scenario="build_outfit")
    assert "asset-1" not in _ids(result)


def test_explicit_source_policy_dict_respected():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    asset = _w("asset-1", "Blue Jeans", "Bottoms", source="asset_library")
    result = _gen(
        [top],
        wardrobe=[top, asset],
        source_policy={"allowed_sources": ["style_asset"]},
    )
    assert result["success"] is True
    assert "asset-1" in _ids(result)


# ------------------------------------------------- typed failures

def test_empty_wardrobe_no_fixed_is_insufficient_wardrobe():
    result = _gen([], wardrobe=[])
    assert result["success"] is False
    assert result["error"]["code"] == "INSUFFICIENT_WARDROBE"


def test_fixed_item_without_id_is_invalid_item_id():
    result = _gen([{"name": "No Id Shirt", "category": "Tops"}])
    assert result["success"] is False
    assert result["error"]["code"] == "INVALID_ITEM_ID"


def test_fixed_item_with_unknown_source_is_rejected():
    result = _gen([{"id": "top-1", "name": "Shirt", "category": "Tops"}])
    assert result["success"] is False
    assert result["error"]["code"] == "UNKNOWN_ITEM_SOURCE"


def test_style_this_without_wardrobe_completion_is_typed_failure():
    top = _w("top-1", "White Shirt", "Tops")
    result = _gen([top], wardrobe=[top], scenario="style_this")
    assert result["success"] is False
    assert result["error"]["code"] == "INSUFFICIENT_WARDROBE"


def test_style_assets_cannot_repair_incomplete_style_this_wardrobe():
    top = _w("top-1", "White Shirt", "Tops")
    asset_bottom = _w(
        "asset-bottom-1", "Pleated Trousers", "Bottoms", source="style_asset"
    )
    asset_shoe = _w(
        "asset-shoe-1", "White Sneakers", "Footwear", source="style_asset"
    )

    result = builder.generate(
        scenario="style_this",
        fixed_items=[top],
        wardrobe=[top],
        style_assets=[asset_bottom, asset_shoe],
    )

    assert result["success"] is False
    assert result["error"]["code"] == "INSUFFICIENT_WARDROBE"


def test_conflicting_fixed_dress_and_top_are_rejected():
    result = _gen([_wardrobe()[0], _wardrobe()[4]])
    assert result["success"] is False
    assert result["error"]["code"] == "FIXED_ITEMS_INCOMPATIBLE"


def test_invalid_slot_rejected():
    top = next(i for i in _wardrobe() if i["id"] == "top-1")
    result = _gen([top], replaceable_slots=["hat-rack"])
    assert result["success"] is False
    assert result["error"]["code"] == "INVALID_SLOT"


def test_shuffle_all_slots_locked_is_all_items_locked():
    top = next(i for i in _wardrobe() if i["id"] == "top-1")
    result = _gen([top], scenario="shuffle_unlocked", replaceable_slots=["top"])
    assert result["success"] is False
    assert result["error"]["code"] == "ALL_ITEMS_LOCKED"


def test_shuffle_no_candidates_is_no_replacement_found():
    top = next(i for i in _wardrobe() if i["id"] == "top-1")
    result = _gen(
        [top], wardrobe=[top], scenario="shuffle_unlocked",
        replaceable_slots=["footwear"],
    )
    assert result["success"] is False
    assert result["error"]["code"] == "NO_REPLACEMENT_FOUND"


# ------------------------------------------------- survival through scoring

def test_fixed_items_survive_scoring_and_selection():
    w = _wardrobe()
    fixed = [w[4]]  # dress-1
    for variant in range(4):
        result = _gen(fixed, context={"wardrobe": _wardrobe(), "variant": variant})
        assert result["success"] is True
        assert "dress-1" in _ids(result)
        assert result["items"][0]["locked"] is True


def test_changed_slots_and_missing_items_reported():
    top = next(i for i in _wardrobe() if i["id"] == "top-1")
    result = _gen([top], wardrobe=[top])  # nothing to complete with
    assert result["success"] is True
    assert result["changed_slots"] == []
    assert "bottom" in result["missing_items"]
    assert "footwear" in result["missing_items"]


def test_professional_occasion_filters_unsafe_candidates():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    camo = _w("camo-1", "Camo Cargo Pants", "Bottoms")
    safe = _w("bottom-1", "Grey Trousers", "Bottoms")
    result = _gen(
        [top],
        wardrobe=[top, camo, safe],
        context={"wardrobe": [top, camo, safe], "occasion": "client_meeting"},
    )
    assert result["success"] is True
    assert "camo-1" not in _ids(result)
    assert "bottom-1" in _ids(result)


def test_stored_strategy_rejects_explicit_avoid_match():
    top = _w("top-1", "Teal Shirt", "Tops")
    loud = _w("bottom-loud", "Loud Logo Trousers", "Bottoms")
    quiet = _w("bottom-quiet", "Charcoal Trousers", "Bottoms", color="charcoal")
    result = _gen(
        [top],
        wardrobe=[top, loud, quiet],
        context={
            "wardrobe": [top, loud, quiet],
            "style_strategy": {
                "archetype_id": "modern_minimal",
                "palette": ["charcoal", "black"],
                "avoid": ["loud logo"],
                "formality": 6,
            },
        },
    )
    assert result["success"] is True
    assert "bottom-loud" not in _ids(result)
    assert "bottom-quiet" in _ids(result)
