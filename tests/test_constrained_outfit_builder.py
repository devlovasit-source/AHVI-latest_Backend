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


# ------------------------------------------------- READY implies FLUTTER-READY
#
# is_board_renderable(raw) proves the backend's OWN readiness gate is
# satisfied. It does NOT prove the field name the builder's serialized output
# carries is one lib/util/wardrobe_image_resolver.dart (commit 4be647a, the
# currently-shipped resolver - verified by reading that exact file, not from
# memory) actually reads for a wardrobe (non-style-asset) item. The device
# gate found a live gap: normalize_style_item() (services/style_item_contract)
# preserves a winning field under its ORIGINAL name, but the Shuffle path
# never runs project_board_image_fields() the way routers/stylist.py's
# initial Style This path does - so an item whose only board-safe field is a
# bare transparent_url survives is_board_renderable() and gets selected, but
# Flutter's non-asset branch never reads a field by that name (only under
# masked_url) and renders an empty hanger.
#
# _flutter_wardrobe_resolvable() below is a deliberately narrow, faithful
# mirror of that resolver's non-asset admission rules - not a reimplementation
# of its full tiered fallback/candidate-ranking logic, just enough to answer
# "would ANY candidate survive admission" for these single-field fixtures.


def _clean(value):
    text = str(value or "").strip()
    return text if text and text.lower() != "null" else None


def _flutter_wardrobe_resolvable(item):
    """Mirrors lib/util/wardrobe_image_resolver.dart's non-style-asset
    (isStyleAsset=false) candidate admission for a Style-board surface."""
    board_ready = (_clean(item.get("board_status") or item.get("boardStatus")) or "").lower() == "cutout_ready"
    cutout_ready = (_clean(item.get("cutout_status") or item.get("cutoutStatus")) or "").lower() == "ready"
    rmbg_ready = (_clean(item.get("image_status") or item.get("imageStatus")) or "").lower() == "rmbg_complete"

    # tier 0: validated_cutout / board-ready cutout
    if cutout_ready and _clean(item.get("cutout_url") or item.get("cutoutUrl")):
        return True
    if board_ready and _clean(item.get("board_image_url") or item.get("boardImageUrl")):
        return True
    # tier 1: masked_url, unconditional (legacy_masked_cutout path) - the ONLY
    # field name the non-asset branch reads for a "masked/transparent cutout"
    if _clean(item.get("masked_url") or item.get("maskedUrl")):
        return True
    # tier 2: rmbg_url / transparent_image_url / processed_url, status-gated
    if rmbg_ready and _clean(item.get("rmbg_url") or item.get("rmbgUrl")):
        return True
    if (board_ready or cutout_ready) and _clean(
        item.get("transparent_image_url") or item.get("transparentImageUrl")
    ):
        return True
    if rmbg_ready and _clean(item.get("processed_url") or item.get("processedUrl")):
        return True
    # tier 3: normalized_url, unconditional catalog fallback
    if _clean(item.get("normalized_url") or item.get("normalizedUrl")):
        return True
    # Deliberately NOT checked: bare transparent_url/transparentUrl - the
    # non-asset branch never reads that field name at all.
    return False


def _board_safe_forms():
    return [
        ("transparent_url", {"transparent_url": "https://img/x-t.png"}),
        ("transparentUrl", {"transparentUrl": "https://img/x-t.png"}),
        ("transparent_image_url", {"transparent_image_url": "https://img/x-ti.png", "board_status": "cutout_ready"}),
        ("transparentImageUrl", {"transparentImageUrl": "https://img/x-ti.png", "board_status": "cutout_ready"}),
        ("cutout_url", {"cutout_url": "https://img/x-c.png", "cutout_status": "ready"}),
        ("cutoutUrl", {"cutoutUrl": "https://img/x-c.png", "cutout_status": "ready"}),
        ("rmbg_url", {"rmbg_url": "https://img/x-r.png", "image_status": "rmbg_complete"}),
        ("rmbgUrl", {"rmbgUrl": "https://img/x-r.png", "image_status": "rmbg_complete"}),
        ("processed_url", {"processed_url": "https://img/x-p.png", "image_status": "rmbg_complete"}),
        ("processedUrl", {"processedUrl": "https://img/x-p.png", "image_status": "rmbg_complete"}),
        ("board_image_url", {"board_image_url": "https://img/x-b.png", "board_status": "cutout_ready"}),
        ("boardImageUrl", {"boardImageUrl": "https://img/x-b.png", "board_status": "cutout_ready"}),
        ("masked_url", {"masked_url": "https://img/x-m.png"}),
        ("maskedUrl", {"maskedUrl": "https://img/x-m.png"}),
        ("normalized_url", {"normalized_url": "https://img/x-n.png"}),
        ("normalizedUrl", {"normalizedUrl": "https://img/x-n.png"}),
    ]


@pytest.mark.parametrize("form,fields", _board_safe_forms(), ids=[f[0] for f in _board_safe_forms()])
def test_ready_implies_flutter_ready_through_real_shuffle(form, fields):
    """RAW_RENDERABLE -> BUILDER_OUTPUT_RENDERABLE -> FLUTTER_RESOLVABLE, all
    through the real ConstrainedOutfitBuilder shuffle_unlocked path (not just
    pool membership) - the permanent tripwire against another path split."""
    top = _w("top-1", "White Oxford Shirt", "Tops")
    accessory = {
        "id": "acc-only",
        "name": "Only Renderable Accessory",
        "category": "Accessories",
        "source": "wardrobe",
        "image_url": "https://img/acc-only.png",
        **fields,
    }
    assert is_board_renderable(accessory) is True, f"{form}: fixture must be board-renderable pre-selection"

    result = _gen(
        [top],
        wardrobe=[top, accessory],
        scenario="shuffle_unlocked",
        replaceable_slots=["accessory"],
        context={"wardrobe": [top, accessory]},
    )
    assert result["success"] is True, f"{form}: {result.get('error')}"
    assert "acc-only" in _ids(result), f"{form}: item was not selected"
    returned = next(i for i in result["items"] if i["item_id"] == "acc-only")

    assert is_board_renderable(returned) is True, (
        f"{form}: builder output is not board-renderable"
    )
    assert _flutter_wardrobe_resolvable(returned) is True, (
        f"{form}: builder output has no field the Flutter wardrobe resolver "
        f"actually reads - this is the empty-hanger regression"
    )


def _no_raw_image_forms():
    return [
        ("masked_url", {"masked_url": "https://img/x-m.png"}),
        ("transparent_url", {"transparent_url": "https://img/x-t.png"}),
        ("rmbg_url", {"rmbg_url": "https://img/x-r.png", "image_status": "rmbg_complete"}),
        ("processed_url", {"processed_url": "https://img/x-p.png", "image_status": "rmbg_complete"}),
    ]


@pytest.mark.parametrize("form,fields", _no_raw_image_forms(), ids=[f[0] for f in _no_raw_image_forms()])
def test_masked_transparent_rmbg_processed_only_no_synthetic_image_url_alias(form, fields):
    """RAW_READY -> PREPARED_READY -> FLUTTER_READY for items with NO raw
    image_url at all (the exact live-bug shape - masked-only, no photo to
    alias against). Must remain renderable, and the served item must never
    gain a synthetic image_url that equals the winning field."""
    top = _w("top-1", "White Oxford Shirt", "Tops")
    accessory = {
        "id": "acc-no-raw",
        "name": "No Raw Image Accessory",
        "category": "Accessories",
        "source": "wardrobe",
        **fields,
    }
    assert is_board_renderable(accessory) is True, f"{form}: RAW_READY failed"

    result = _gen(
        [top],
        wardrobe=[top, accessory],
        scenario="shuffle_unlocked",
        replaceable_slots=["accessory"],
        context={"wardrobe": [top, accessory]},
    )
    assert result["success"] is True, f"{form}: {result.get('error')}"
    returned = next(i for i in result["items"] if i["item_id"] == "acc-no-raw")

    assert is_board_renderable(returned) is True, f"{form}: PREPARED_READY failed"
    assert _flutter_wardrobe_resolvable(returned) is True, f"{form}: FLUTTER_READY failed"
    assert "image_url" not in returned, (
        f"{form}: NO_SYNTHETIC_IMAGE_URL_ALIAS failed - a synthetic image_url "
        f"was manufactured, which is exactly the live-device hanger bug"
    )


def test_style_asset_transparent_url_field_name_untouched():
    """The board-image projection is skipped for style_asset items on
    purpose: Flutter's isStyleAsset branch reads transparent_url directly
    under its own name (never masked_url) - canonicalizing it the way
    wardrobe items need would break asset rendering instead of fixing it."""
    accessory = _w(
        "asset-acc-1",
        "Curated Chain",
        "accessory",
        source="style_asset",
        transparent_url="https://img/asset-acc-1-transparent.png",
    )
    accessory.pop("normalized_url", None)
    top = _w("top-1", "White Oxford Shirt", "top", source="style_asset")
    bottom = _w("bottom-1", "Pleated Trousers", "bottom", source="style_asset")
    footwear = _w("shoe-1", "Minimal Slides", "footwear", source="style_asset")

    result = builder.generate(
        scenario="style_this",
        fixed_items=[],
        style_assets=[top, bottom, footwear, accessory],
        source_policy={"allowed_sources": ["style_asset"]},
        context={"accessory_budget": 1},
    )

    assert result["success"] is True
    returned = next(i for i in result["items"] if i["item_id"] == "asset-acc-1")
    assert returned.get("transparent_url") == "https://img/asset-acc-1-transparent.png"
    assert "masked_url" not in returned


def test_raw_image_url_only_stays_unsafe_through_shuffle():
    """Unsafe items must stay unsafe: a raw upload photo with no processed
    field must never enter the pool, let alone be selected."""
    top = _w("top-1", "White Oxford Shirt", "Tops")
    raw_only = {
        "id": "acc-raw",
        "name": "Raw Only Accessory",
        "category": "Accessories",
        "source": "wardrobe",
        "image_url": "https://img/acc-raw.png",
    }
    assert is_board_renderable(raw_only) is False

    result = _gen(
        [top],
        wardrobe=[top, raw_only],
        scenario="shuffle_unlocked",
        replaceable_slots=["accessory"],
        context={"wardrobe": [top, raw_only]},
    )
    if result["success"]:
        assert "acc-raw" not in _ids(result)
    else:
        assert result["error"]["code"] in {"INSUFFICIENT_WARDROBE", "NO_REPLACEMENT_FOUND"}


def test_masked_url_aliasing_raw_stays_unsafe_through_shuffle():
    """Unsafe items must stay unsafe: masked_url == image_url is fabricated
    provenance (the RMBG-produced-nothing healing alias), not a real cutout."""
    top = _w("top-1", "White Oxford Shirt", "Tops")
    raw = "https://img/acc-alias.png"
    aliased = {
        "id": "acc-alias",
        "name": "Aliased Accessory",
        "category": "Accessories",
        "source": "wardrobe",
        "image_url": raw,
        "masked_url": raw,
    }
    assert is_board_renderable(aliased) is False

    result = _gen(
        [top],
        wardrobe=[top, aliased],
        scenario="shuffle_unlocked",
        replaceable_slots=["accessory"],
        context={"wardrobe": [top, aliased]},
    )
    if result["success"]:
        assert "acc-alias" not in _ids(result)
    else:
        assert result["error"]["code"] in {"INSUFFICIENT_WARDROBE", "NO_REPLACEMENT_FOUND"}


# ------------------------------------------------- URL-identity alias parity
#
# Live device bug: masked_url differed from image_url only by a query-string
# token (a signed/cache-busted URL copy) - exact-string-different, but
# Flutter's _urlIdentity() (scheme+host+port+path only, query/fragment
# dropped) calls it the same resource and rejects it, producing an empty
# hanger for an item the backend had called board_renderable. Exercised
# through the real candidate pool + shuffle_unlocked selection, not just
# resolve_board_image_candidate() in isolation.


def test_query_string_aliased_item_never_enters_pool_or_shuffle_output():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    bad = {
        "id": "acc-query-alias",
        "name": "Query Alias Accessory",
        "category": "Accessories",
        "source": "wardrobe",
        "image_url": "https://cdn.test/item.png?token=abc",
        "masked_url": "https://cdn.test/item.png?token=xyz",
    }
    assert is_board_renderable(bad) is False, "pre-fix proof: this must be unsafe"

    result = _gen(
        [top],
        wardrobe=[top, bad],
        scenario="shuffle_unlocked",
        replaceable_slots=["accessory"],
        context={"wardrobe": [top, bad]},
    )
    if result["success"]:
        assert "acc-query-alias" not in _ids(result)
    else:
        assert result["error"]["code"] in {"INSUFFICIENT_WARDROBE", "NO_REPLACEMENT_FOUND"}


def test_genuinely_different_masked_url_with_own_query_stays_selectable():
    """Guard against over-normalizing: a real cutout on a different path,
    carrying its own unrelated query string, must remain selectable."""
    top = _w("top-1", "White Oxford Shirt", "Tops")
    good = {
        "id": "acc-genuine",
        "name": "Genuine Accessory",
        "category": "Accessories",
        "source": "wardrobe",
        "image_url": "https://cdn.test/raw/item.png?x=1",
        "masked_url": "https://cdn.test/masked/item.png?x=2",
    }
    assert is_board_renderable(good) is True

    result = _gen(
        [top],
        wardrobe=[top, good],
        scenario="shuffle_unlocked",
        replaceable_slots=["accessory"],
        context={"wardrobe": [top, good]},
    )
    assert result["success"] is True
    assert "acc-genuine" in _ids(result)
    returned = next(i for i in result["items"] if i["item_id"] == "acc-genuine")
    assert is_board_renderable(returned) is True


def test_required_slot_never_left_empty_when_only_candidate_is_url_aliased():
    """Core-slot exhaustion safety: if the ONLY candidate for a required role
    is correctly rejected as a URL-identity alias, the builder must fall back
    to its existing typed-incomplete behavior - never fabricate a placeholder
    and never report success with that slot silently empty."""
    bad_bottom = {
        "id": "bottom-alias",
        "name": "Alias Bottom",
        "category": "Bottoms",
        "source": "wardrobe",
        "image_url": "https://cdn.test/item.png?token=abc",
        "masked_url": "https://cdn.test/item.png?token=xyz",
    }
    result = _gen([], wardrobe=[bad_bottom], scenario="build_outfit")
    if result["success"]:
        assert "bottom-alias" not in _ids(result)
        assert "bottom" in result.get("missing_items", [])
    else:
        assert result["error"]["code"] == "INSUFFICIENT_WARDROBE"


# ------------------------------------------ post-normalization provenance
#
# Live device bug (ce4ade1 device gate, item 69f5cf416867b929beb6): a
# masked-only accessory (no raw image_url at all) correctly passed
# is_board_renderable(raw) - there's no raw photo to alias against - but the
# OLD pipeline normalized+projected the winning field AFTER selection,
# manufacturing image_url == masked_url in the served item. Flutter's own
# alias check then rejected the self-aliased item and rendered an empty
# hanger. The fix (prepare_board_item) prepares and gates BEFORE selection,
# so the exact object proven safe is the one served - exercised here through
# the real ConstrainedOutfitBuilder pool + shuffle_unlocked selection.


def test_real_shuffle_masked_only_accessory_has_no_synthetic_image_url():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    masked_only = {
        "id": "masked-only",
        "name": "Leather Belt",
        "category": "Accessories",
        "source": "wardrobe",
        "masked_url": "https://cdn.test/processed/belt.png",
    }
    assert is_board_renderable(masked_only) is True

    result = _gen(
        [top],
        wardrobe=[top, masked_only],
        scenario="shuffle_unlocked",
        replaceable_slots=["accessory"],
        context={"wardrobe": [top, masked_only]},
    )
    assert result["success"] is True
    assert "masked-only" in _ids(result)
    returned = next(i for i in result["items"] if i["item_id"] == "masked-only")
    assert returned["masked_url"] == masked_only["masked_url"]
    assert "image_url" not in returned
    assert is_board_renderable(returned) is True


def test_real_shuffle_raw_plus_aliased_masked_accessory_never_selected():
    top = _w("top-1", "White Oxford Shirt", "Tops")
    aliased = {
        "id": "acc-real-alias",
        "name": "Aliased Accessory",
        "category": "Accessories",
        "source": "wardrobe",
        "image_url": "https://cdn.test/item.png?token=1",
        "masked_url": "https://cdn.test/item.png?token=2",
    }
    assert is_board_renderable(aliased) is False

    result = _gen(
        [top],
        wardrobe=[top, aliased],
        scenario="shuffle_unlocked",
        replaceable_slots=["accessory"],
        context={"wardrobe": [top, aliased]},
    )
    if result["success"]:
        assert "acc-real-alias" not in _ids(result)
    else:
        assert result["error"]["code"] in {"INSUFFICIENT_WARDROBE", "NO_REPLACEMENT_FOUND"}


def test_no_admitted_candidate_can_gain_an_image_url_alias_after_selection():
    """Output invariant: for every non-style_asset item actually returned by
    the builder, image_url (if present at all) must never equal the field
    that won board-image selection - no post-selection transformation may
    introduce one."""
    top = _w("top-1", "White Oxford Shirt", "Tops")
    masked_only = {
        "id": "masked-only",
        "name": "Leather Belt",
        "category": "Accessories",
        "source": "wardrobe",
        "masked_url": "https://cdn.test/processed/belt.png",
    }
    result = _gen(
        [top],
        wardrobe=[top, masked_only],
        scenario="shuffle_unlocked",
        replaceable_slots=["accessory"],
        context={"wardrobe": [top, masked_only]},
    )
    assert result["success"] is True
    for item in result["items"]:
        if item.get("source") == "style_asset":
            continue
        assert is_board_renderable(item) is True
        image_url = item.get("image_url")
        for field in ("masked_url", "cutout_url", "board_image_url", "rmbg_url", "processed_url"):
            if image_url and item.get(field):
                assert image_url != item[field]


def test_required_slot_never_left_empty_when_only_candidate_is_masked_only_and_role_mismatched():
    """A masked-only item must never be silently dropped just because
    provenance scrubbing ran - it must still fill its role normally."""
    masked_only_bottom = {
        "id": "bottom-masked-only",
        "name": "Masked Only Trousers",
        "category": "Bottoms",
        "source": "wardrobe",
        "masked_url": "https://cdn.test/processed/trousers.png",
    }
    result = _gen([], wardrobe=[masked_only_bottom], scenario="build_outfit")
    assert result["success"] is True
    assert "bottom-masked-only" in _ids(result)


# ------------------------------------------ locked-item provenance
#
# The unlocked candidate pool now gates on prepare_board_item(raw), but
# fixed/locked items took a separate path: normalize_style_item(raw) only,
# never re-gated, never scrubbed. A locked masked-only item (no explicit
# image_url) hits the exact same normalize_style_item() image_url fallback
# bug - is_board_renderable(raw) says True (no raw photo to alias against),
# but the served fixed item ends up with image_url == masked_url, which
# Flutter's own alias check (correctly) rejects. Fixed items must satisfy
# the identical prepare_board_item() invariant as unlocked candidates.


def test_pre_fix_locked_mechanism_reproduces_synthetic_alias_if_bypassed():
    """Documents WHY the locked-item bug existed: normalize_style_item(raw)
    alone (the exact sequence the fixed-item path used before this fix)
    still manufactures image_url == masked_url for a masked-only item.
    normalize_style_item() was not changed by this fix (general-purpose,
    out of scope) - the fix is that ConstrainedOutfitBuilder's fixed-item
    path must call prepare_board_item() too, exactly like the unlocked pool
    already does."""
    raw = {
        "id": "locked-masked-only",
        "name": "Black Night Trousers",
        "category": "Bottoms",
        "source": "wardrobe",
        "masked_url": "https://cdn.test/processed/trousers.png",
    }
    assert is_board_renderable(raw) is True

    from services.style_item_contract import normalize_style_item
    normalized = normalize_style_item(raw)
    assert normalized["image_url"] == raw["masked_url"]


def test_locked_masked_only_item_has_no_synthetic_image_url_alias():
    """Real fixed-item path: a locked masked-only bottom must remain
    locked, keep its role/id, render successfully, and gain no synthetic
    image_url alias - the live-bug shape, now through the lock contract."""
    locked = {
        "id": "locked-masked-only",
        "name": "Black Night Trousers",
        "category": "Bottoms",
        "source": "wardrobe",
        "masked_url": "https://cdn.test/processed/trousers.png",
    }
    result = _gen([locked], wardrobe=[locked], scenario="build_outfit")
    assert result["success"] is True
    kept = next(i for i in result["items"] if i["item_id"] == "locked-masked-only")

    assert kept["locked"] is True
    assert kept["role"] == "bottom"
    assert kept["item_id"] == "locked-masked-only"
    assert kept["masked_url"] == locked["masked_url"]
    assert "image_url" not in kept
    assert is_board_renderable(kept) is True


def test_locked_raw_plus_genuine_masked_preserves_both_fields():
    """A locked item WITH an explicit, genuinely-distinct image_url must
    keep both fields exactly - the provenance fix must not erase a real
    raw photo, only refuse to manufacture a fake one."""
    top = _w("top-1", "White Oxford Shirt", "Tops")
    locked = {
        "id": "locked-genuine",
        "name": "Navy Blazer",
        "category": "Outerwear",
        "source": "wardrobe",
        "image_url": "https://cdn.test/raw/blazer.png",
        "masked_url": "https://cdn.test/masked/blazer.png",
    }
    result = _gen([locked], wardrobe=[locked, top], scenario="build_outfit")
    assert result["success"] is True
    kept = next(i for i in result["items"] if i["item_id"] == "locked-genuine")
    assert kept["image_url"] == locked["image_url"]
    assert kept["masked_url"] == locked["masked_url"]
    assert kept["locked"] is True


def test_locked_aliased_item_rejected_with_typed_error_no_partial_mutation():
    """An unsafe locked item (image_url/masked_url share Flutter URL
    identity) must never be silently unlocked, dropped, or substituted
    behind the user's lock, and must never produce a board with a blank
    slot where the lock should be. It must fail with a typed error before
    any board mutation."""
    aliased_locked = {
        "id": "locked-alias",
        "name": "Aliased Trousers",
        "category": "Bottoms",
        "source": "wardrobe",
        "image_url": "https://cdn.test/item.png?token=a",
        "masked_url": "https://cdn.test/item.png?token=b",
    }
    assert is_board_renderable(aliased_locked) is False

    result = _gen([aliased_locked], wardrobe=[aliased_locked], scenario="build_outfit")
    assert result["success"] is False
    assert result["error"]["code"] == "FIXED_ITEM_NOT_BOARD_SAFE"
    assert "locked-alias" in result["error"]["fixed_item_ids"]


def test_locked_style_asset_untouched_by_provenance_gate():
    """590cc2e rule preserved for LOCKED items too: a fixed style_asset item
    keeps its own Flutter resolver branch, is never scrubbed/projected, and
    is never rejected by the new board-safety gate (which does not apply to
    style_asset sources at all, matching the unlocked pool's exception)."""
    locked_asset = {
        "id": "asset-locked-1",
        "name": "Curated Chain",
        "category": "accessory",
        "source": "style_asset",
        "transparent_url": "https://img/asset-locked-1-transparent.png",
    }
    top = _w("top-1", "White Oxford Shirt", "top", source="style_asset")
    bottom = _w("bottom-1", "Pleated Trousers", "bottom", source="style_asset")
    footwear = _w("shoe-1", "Minimal Slides", "footwear", source="style_asset")
    result = builder.generate(
        scenario="style_this",
        fixed_items=[locked_asset],
        style_assets=[locked_asset, top, bottom, footwear],
        source_policy={"allowed_sources": ["style_asset"]},
        context={"accessory_budget": 1},
    )
    assert result["success"] is True
    kept = next(i for i in result["items"] if i["item_id"] == "asset-locked-1")
    assert kept.get("transparent_url") == locked_asset["transparent_url"]
    assert "masked_url" not in kept
    assert kept["locked"] is True


def test_locked_item_with_no_image_fields_at_all_still_succeeds():
    """Locked items were never gated on renderability before this fix - an
    already-persisted locked board item with NO image fields at all (a
    pre-image-tracking board row, as some persisted-shuffle-service tests
    use) must keep working exactly as it did. Only items that actively
    declare unsafe/corrupted image data are rejected, not items with none."""
    top = _w("top-1", "White Oxford Shirt", "Tops")
    bare_locked = {"id": "bottom-bare", "name": "Blue Jeans", "category": "Bottoms", "source": "wardrobe"}
    result = _gen([bare_locked], wardrobe=[bare_locked, top], scenario="build_outfit")
    assert result["success"] is True
    assert "bottom-bare" in _ids(result)


def test_locked_item_with_only_raw_image_url_still_succeeds():
    """A raw-photo-only locked item (no processed field at all) is the
    ordinary, long-tolerated "not board-safe yet" state every wardrobe item
    starts in - it must NOT become a hard lock failure. This mirrors how
    the unlocked pool merely excludes (never hard-fails on) a raw-only
    item; the difference is a locked item can't be silently excluded, so it
    must be preserved as-is instead. Regression guard: this is the exact
    shape most of this codebase's pre-existing locked-item fixtures use."""
    top = _w("top-1", "White Oxford Shirt", "Tops")
    raw_only_locked = {
        "id": "bottom-raw-only",
        "name": "Blue Jeans",
        "category": "Bottoms",
        "source": "wardrobe",
        "image_url": "https://img/bottom-raw-only.png",
    }
    result = _gen([raw_only_locked], wardrobe=[raw_only_locked, top], scenario="build_outfit")
    assert result["success"] is True
    kept = next(i for i in result["items"] if i["item_id"] == "bottom-raw-only")
    assert kept["image_url"] == raw_only_locked["image_url"]
    assert kept["locked"] is True


def test_real_shuffle_with_masked_only_locked_anchor_and_unlocked_replacements():
    """Full real-path regression: a masked-only locked bottom anchor, plus
    valid unlocked top/footwear/accessory, shuffled. The anchor must
    survive with no synthetic alias while unlocked roles change."""
    anchor = {
        "id": "locked-masked-anchor",
        "name": "Black Night Trousers",
        "category": "Bottoms",
        "source": "wardrobe",
        "masked_url": "https://cdn.test/processed/trousers.png",
    }
    top = _w("top-1", "White Oxford Shirt", "Tops")
    shoe = _w("shoe-1", "White Sneakers", "Footwear")
    bag = _w("bag-1", "Structured Handbag", "Accessories")
    wardrobe = [anchor, top, shoe, bag]

    result = builder.generate(
        scenario="shuffle_unlocked",
        fixed_items=[anchor],
        wardrobe=wardrobe,
        replaceable_slots=["top", "footwear", "accessory"],
        context={"wardrobe": wardrobe},
    )
    assert result["success"] is True
    kept_anchor = next(i for i in result["items"] if i["item_id"] == "locked-masked-anchor")
    assert kept_anchor["locked"] is True
    assert kept_anchor["masked_url"] == anchor["masked_url"]
    assert "image_url" not in kept_anchor
    assert is_board_renderable(kept_anchor) is True


def test_real_shuffle_rejects_unsafe_aliased_locked_anchor_no_partial_mutation():
    """The unsafe-lock rejection must hold through the real shuffle path
    too, not just build_outfit."""
    aliased_anchor = {
        "id": "locked-alias-anchor",
        "name": "Aliased Trousers",
        "category": "Bottoms",
        "source": "wardrobe",
        "image_url": "https://cdn.test/item.png?token=a",
        "masked_url": "https://cdn.test/item.png?token=b",
    }
    top = _w("top-1", "White Oxford Shirt", "Tops")
    wardrobe = [aliased_anchor, top]

    result = builder.generate(
        scenario="shuffle_unlocked",
        fixed_items=[aliased_anchor],
        wardrobe=wardrobe,
        replaceable_slots=["top"],
        context={"wardrobe": wardrobe},
    )
    assert result["success"] is False
    assert result["error"]["code"] == "FIXED_ITEM_NOT_BOARD_SAFE"
