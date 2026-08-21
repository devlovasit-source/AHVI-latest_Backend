"""ConstrainedOutfitBuilder: fixed items of every role, exclusions, sources,
dress rules and typed failures."""

import pytest

from services.constrained_outfit_builder import ConstrainedOutfitBuilder
from services.style_board_image_readiness import is_board_renderable
from services.style_item_contract import canonical_item_id

builder = ConstrainedOutfitBuilder()


def _w(item_id, name, category="", source="wardrobe", **extra):
    row = {
        "id": item_id,
        "name": name,
        "category": category,
        "source": source,
        "image_url": f"https://img/{item_id}.png",
        # A real (non-aliased) processed image, so every fixture item is
        # board-renderable by default - readiness itself is covered by
        # tests/test_style_board_image_readiness.py and the
        # test_*_not_board_ready_* cases below, not by every unrelated test.
        "normalized_url": f"https://img/{item_id}-normalized.png",
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


# ------------------------------------------------- board-readiness gate
# (H, I from the readiness-gate implementation spec)


def _not_ready(item_id, name, category="", source="wardrobe", **extra):
    """A wardrobe row with the exact fabricated-alias signature confirmed
    live on device: masked_url == image_url, no real processed image."""
    raw = f"https://img/{item_id}.png"
    row = {
        "id": item_id,
        "name": name,
        "category": category,
        "source": source,
        "image_url": raw,
        "masked_url": raw,
    }
    row.update(extra)
    return row


def test_not_board_ready_support_item_never_enters_pool():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    bad_bottom = _not_ready("bottom-bad", "Fabricated Alias Jeans", "Bottoms")
    good_bottom = _w("bottom-good", "Real Cutout Jeans", "Bottoms")
    result = _gen([top], wardrobe=[top, bad_bottom, good_bottom])
    assert result["success"] is True
    assert "bottom-bad" not in _ids(result)
    assert "bottom-good" in _ids(result)


def test_not_board_ready_item_alone_in_slot_is_typed_insufficient_wardrobe():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    bad_bottom = _not_ready("bottom-bad", "Fabricated Alias Jeans", "Bottoms")
    result = _gen([top], wardrobe=[top, bad_bottom], scenario="style_this")
    assert result["success"] is False
    assert result["error"]["code"] == "INSUFFICIENT_WARDROBE"


def test_ready_alternatives_still_generate_complete_board():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    wardrobe = [
        top,
        _not_ready("bottom-bad-1", "Fabricated Alias Jeans", "Bottoms"),
        _not_ready("bottom-bad-2", "No Cutout Chinos", "Bottoms"),
        _w("bottom-good", "Real Cutout Jeans", "Bottoms"),
        _not_ready("shoe-bad", "No Cutout Sneakers", "Footwear"),
        _w("shoe-good", "Real Cutout Loafers", "Footwear"),
    ]
    result = _gen([top], wardrobe=wardrobe)
    assert result["success"] is True
    assert "bottom-good" in _ids(result)
    assert "shoe-good" in _ids(result)
    assert "bottom-bad-1" not in _ids(result)
    assert "bottom-bad-2" not in _ids(result)
    assert "shoe-bad" not in _ids(result)


def test_shuffle_never_selects_not_board_ready_replacement():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    wardrobe = [
        top,
        _not_ready("bottom-bad", "Fabricated Alias Jeans", "Bottoms"),
        _w("bottom-alt", "Charcoal Trousers", "Bottoms"),
    ]
    for variant in range(5):
        result = _gen(
            [top],
            wardrobe=wardrobe,
            scenario="shuffle_unlocked",
            replaceable_slots=["bottom"],
            context={"wardrobe": wardrobe, "variant": variant},
        )
        assert result["success"] is True
        assert "bottom-bad" not in _ids(result)
        assert "bottom-alt" in _ids(result)


# ------------------------------------------------- small-pool safety
# (Part 6 of the readiness-gate implementation spec: dress/outerwear pools
# are typically much smaller than tops/bottoms, so the filter is checked
# here specifically rather than assumed safe from the tops/bottoms coverage
# above.)


def test_dress_anchor_succeeds_when_ready_footwear_available():
    dress = _w("dress-1", "Red Maxi Dress", "Dresses")
    footwear = _w("shoe-good", "Real Cutout Sandals", "Footwear")
    result = _gen([dress], wardrobe=[dress, footwear])
    assert result["success"] is True
    assert "shoe-good" in _ids(result)


def test_dress_anchor_with_only_not_ready_footwear_is_safe_insufficient_not_hanger():
    dress = _w("dress-1", "Red Maxi Dress", "Dresses")
    bad_footwear = _not_ready("shoe-bad", "Fabricated Alias Sandals", "Footwear")
    result = _gen([dress], wardrobe=[dress, bad_footwear])
    # Never silently include the not-ready item (the "hanger" outcome this
    # gate exists to prevent) - either a typed failure, or success with the
    # slot correctly reported missing. Both are safe; a selected item with
    # no board-safe image is not.
    if result["success"]:
        assert "shoe-bad" not in _ids(result)
        assert "footwear" in result.get("missing_items", [])
    else:
        assert result["error"]["code"] in {
            "INSUFFICIENT_WARDROBE",
            "NO_REPLACEMENT_FOUND",
        }


def test_outerwear_succeeds_when_ready_candidate_available():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    outerwear = _w("jacket-good", "Real Cutout Blazer", "Outerwear")
    result = _gen(
        [top],
        wardrobe=[top, outerwear],
        scenario="shuffle_unlocked",
        replaceable_slots=["outerwear"],
        context={"wardrobe": [top, outerwear]},
    )
    assert result["success"] is True
    assert "jacket-good" in _ids(result)


def test_outerwear_with_only_not_ready_candidate_never_selected():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    bad_outerwear = _not_ready("jacket-bad", "Fabricated Alias Blazer", "Outerwear")
    result = _gen(
        [top],
        wardrobe=[top, bad_outerwear],
        scenario="shuffle_unlocked",
        replaceable_slots=["outerwear"],
        context={"wardrobe": [top, bad_outerwear]},
    )
    if result["success"]:
        assert "jacket-bad" not in _ids(result)
        assert "outerwear" in result.get("missing_items", [])
    else:
        assert result["error"]["code"] in {
            "INSUFFICIENT_WARDROBE",
            "NO_REPLACEMENT_FOUND",
        }


# ------------------------------------------------- board-readiness field
# preservation through the actual selection path (readiness-gate revision
# forensic): a field like cutout_url/rmbg_url passes is_board_renderable() at
# pool-filter time (using the raw item), but normalize_style_item() used to
# drop it when building the selected/returned item - so the item entered the
# pool and was picked, yet the persisted board item was no longer
# board-renderable. These exercise the builder's real output, not just pool
# membership.


def _cutout_only(item_id, name, category=""):
    """Genuinely board-safe ONLY via cutout_url + cutout_status=ready - no
    masked_url/normalized_url, the fields normalize_style_item() already
    preserved before the field-preservation fix."""
    return {
        "id": item_id,
        "name": name,
        "category": category,
        "source": "wardrobe",
        "image_url": f"https://img/{item_id}.png",
        "cutout_url": f"https://img/{item_id}-cutout.png",
        "cutout_status": "ready",
    }


def _rmbg_only(item_id, name, category=""):
    """Genuinely board-safe ONLY via rmbg_url + image_status=rmbg_complete."""
    return {
        "id": item_id,
        "name": name,
        "category": category,
        "source": "wardrobe",
        "image_url": f"https://img/{item_id}.png",
        "rmbg_url": f"https://img/{item_id}-rmbg.png",
        "image_status": "rmbg_complete",
    }


@pytest.mark.parametrize(
    "factory,item_id,field",
    [
        (_cutout_only, "bottom-cutout", "cutout_url"),
        (_rmbg_only, "bottom-rmbg", "rmbg_url"),
    ],
    ids=["cutout_url", "rmbg_url"],
)
def test_builder_output_preserves_previously_dropped_readiness_field(factory, item_id, field):
    top = _w("top-1", "White Oxford Shirt", "Tops")
    only_bottom = factory(item_id, "Only Renderable Bottom", "Bottoms")
    result = _gen([top], wardrobe=[top, only_bottom])
    assert result["success"] is True
    assert item_id in _ids(result)
    returned = next(i for i in result["items"] if i["item_id"] == item_id)
    assert returned.get(field), f"builder output dropped {field!r} for the selected item"
    assert is_board_renderable(returned) is True, (
        "builder returned a selected item that is not board-renderable"
    )


def test_shuffle_selects_and_preserves_cutout_only_replacement():
    """Same field-preservation guarantee, exercised through the shuffle_unlocked
    scenario (the actual live path where the empty-hanger regression was
    observed on device), not just plain build_outfit."""
    top = _w("top-1", "White Oxford Shirt", "Tops")
    only_bottom = _cutout_only("bottom-cutout", "Only Renderable Bottom", "Bottoms")
    result = _gen(
        [top],
        wardrobe=[top, only_bottom],
        scenario="shuffle_unlocked",
        replaceable_slots=["bottom"],
        context={"wardrobe": [top, only_bottom]},
    )
    assert result["success"] is True
    assert "bottom-cutout" in _ids(result)
    returned = next(i for i in result["items"] if i["item_id"] == "bottom-cutout")
    assert returned.get("cutout_url")
    assert is_board_renderable(returned) is True
