"""Contract tests: canonical id / source / role / accessory / image + invariant."""

import pytest

from services.style_item_contract import (
    FixedItemLostError,
    assert_fixed_items_preserved,
    canonical_accessory_type,
    canonical_image_url,
    canonical_item_id,
    canonical_item_role,
    canonical_item_source,
    is_style_asset,
    is_wardrobe_item,
    normalize_style_item,
)


# ---------------------------------------------------------------- identity

@pytest.mark.parametrize("key", ["item_id", "id", "$id", "itemId", "image_id", "asset_id"])
def test_id_extracted_from_every_supported_key(key):
    assert canonical_item_id({key: " abc-123 "}) == "abc-123"


def test_id_priority_order():
    assert canonical_item_id({"id": "second", "item_id": "first"}) == "first"


def test_name_only_item_has_empty_id():
    assert canonical_item_id({"name": "Blue Shirt", "label": "Blue Shirt"}) == ""
    assert canonical_item_id({}) == ""
    assert canonical_item_id(None) == ""


# ---------------------------------------------------------------- source

def test_source_normalization():
    assert canonical_item_source({"source": "asset"}) == "style_asset"
    assert canonical_item_source({"source": "asset_library"}) == "style_asset"
    assert canonical_item_source({"source": "curated"}) == "style_asset"
    assert canonical_item_source({"source": "user_wardrobe"}) == "wardrobe"
    assert canonical_item_source({"source": "closet"}) == "wardrobe"
    assert canonical_item_source({"source": "uploaded"}) == "wardrobe"
    assert canonical_item_source({"source": "commerce"}) == "catalog"
    assert canonical_item_source({"source": "generated"}) == "generated"


def test_missing_or_unrecognized_source_is_unknown_not_wardrobe():
    assert canonical_item_source({}) == "unknown"
    assert canonical_item_source({"source": "banana"}) == "unknown"
    assert not is_wardrobe_item({})
    assert not is_wardrobe_item({"source": "banana"})
    assert is_wardrobe_item({"source": "user_wardrobe"})
    assert is_style_asset({"source": "asset_library"})


# ---------------------------------------------------------------- role

@pytest.mark.parametrize(
    "name,expected",
    [
        ("Formal Dress Shirt", "top"),
        ("Dress Shirt", "top"),
        ("White Short-Sleeved Shirt", "top"),
        ("Shirt Dress", "dress"),
        ("Maxi Dress", "dress"),
        ("Khaki Shorts", "bottom"),
        ("Navy Blazer", "outerwear"),
        ("Brown Loafers", "footwear"),
        ("Leather Watch", "accessory"),
        ("Structured Handbag", "accessory"),
        ("Saree", "dress"),
        ("Kurta", "top"),
        ("Jumpsuit", "dress"),
        ("Mystery Widget Xyz", "unknown"),
    ],
)
def test_role_cases(name, expected):
    assert canonical_item_role({"name": name}) == expected


def test_role_explicit_field_wins():
    assert canonical_item_role({"name": "weird thing", "category": "Dresses"}) == "dress"
    assert canonical_item_role({"name": "weird thing", "sub_category": "loafers"}) == "footwear"


def test_role_non_fashion_and_sport_guards():
    assert canonical_item_role({"name": "Phone Charger", "category": "Accessories"}) == "unknown"
    assert canonical_item_role({"name": "Swim Cap", "category": "Accessories"}) == "unknown"


# ---------------------------------------------------------------- accessory type

def test_accessory_types():
    assert canonical_accessory_type({"name": "Leather Watch"}) == "watch"
    assert canonical_accessory_type({"name": "Structured Handbag"}) == "bag"
    assert canonical_accessory_type({"name": "Tan Belt"}) == "belt"
    assert canonical_accessory_type({"name": "Gold Necklace"}) == "necklace"
    assert canonical_accessory_type({"name": "Silk Scarf"}) == "scarf"
    assert canonical_accessory_type({"name": "Aviator Sunglasses"}) == "eyewear"
    assert canonical_accessory_type({"name": "Baseball Cap"}) == "headwear"
    assert canonical_accessory_type({"name": "Something Odd"}) == "other"


# ---------------------------------------------------------------- image

def test_image_url_fallback_chain_and_survival():
    assert canonical_image_url({"normalized_url": "https://a/n.png", "image_url": "https://a/i.png"}) == "https://a/n.png"
    assert canonical_image_url({"imageUrl": "https://a/i.png"}) == "https://a/i.png"
    assert canonical_image_url({"url": "https://a/u.png"}) == "https://a/u.png"
    norm = normalize_style_item({"id": "x1", "name": "Top", "masked_url": "https://a/m.png"})
    assert norm["image_url"] == "https://a/m.png"


# ---------------------------------------------------------------- normalize

def test_normalize_style_item_envelope():
    norm = normalize_style_item(
        {
            "$id": "w-9",
            "name": "Leather Watch",
            "source": "user_wardrobe",
            "image_url": "https://a/w.png",
            "color": "brown",
            "position": {"x": 0.1, "y": 0.2},
        }
    )
    assert norm["item_id"] == "w-9"
    assert norm["role"] == "accessory"
    assert norm["accessory_type"] == "watch"
    assert norm["source"] == "wardrobe"
    assert norm["owned"] is True
    assert norm["position"] == {"x": 0.1, "y": 0.2}


# ---------------------------------------------------------------- invariant

def test_assert_fixed_items_preserved_passes():
    fixed = [{"id": "a"}, {"item_id": "b"}]
    candidates = [{"item_id": "a"}, {"$id": "b"}, {"id": "c"}]
    assert_fixed_items_preserved(fixed, candidates)  # no raise


def test_assert_fixed_items_preserved_raises_with_details():
    with pytest.raises(FixedItemLostError) as exc:
        assert_fixed_items_preserved([{"id": "a"}, {"id": "gone"}], [{"id": "a"}], stage="scoring")
    assert exc.value.missing_ids == ["gone"]
    assert exc.value.stage == "scoring"


def test_assert_fixed_items_empty_id_counts_as_missing():
    with pytest.raises(FixedItemLostError):
        assert_fixed_items_preserved([{"name": "no id"}], [{"id": "a"}])


@pytest.mark.parametrize("field", ["source", "image_url", "role"])
def test_assert_fixed_items_rejects_mutated_anchor_fields(field):
    fixed = {"id": "a", "source": "wardrobe", "image_url": "https://img/a.png", "role": "top"}
    candidate = dict(fixed)
    candidate[field] = {"source": "catalog", "image_url": "https://img/other.png", "role": "bottom"}[field]
    with pytest.raises(FixedItemLostError) as exc:
        assert_fixed_items_preserved([fixed], [candidate], stage="serialized")
    expected = "slot" if field == "role" else field
    assert expected in exc.value.mismatched_fields["a"]


def test_normalize_preserves_full_board_contract():
    raw = {
        "id": "watch-1", "name": "Watch", "category": "Accessories",
        "source": "wardrobe", "masked_url": "masked", "board_image_url": "board",
        "normalized_url": "normalized", "position": {"x": 0.1}, "scale": 1.2,
        "rotation": 4,
    }
    norm = normalize_style_item(raw)
    assert norm["id"] == "watch-1"
    assert norm["slot"] == "accessory"
    assert norm["accessory_type"] == "watch"
    for key in ("masked_url", "board_image_url", "normalized_url", "position", "scale", "rotation"):
        assert norm[key] == raw[key]
