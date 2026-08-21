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
from services.style_board_image_readiness import is_board_renderable


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
    assert canonical_item_role({"name": "Sleepwear Set", "role": "one_piece"}) == "dress"


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


# ------------------------------------------------- board-readiness roundtrip
#
# services.style_board_image_readiness.is_board_renderable recognizes several
# genuine board-safe representations. normalize_style_item() must preserve
# whichever one made an item renderable - a raw wardrobe item that passes
# is_board_renderable() must still pass it after normalization, with the same
# winning field (and its gating status, where the field is status-gated)
# intact. See the readiness-gate revision forensic for how a dropped field
# here turns into a live empty-hanger placeholder after shuffle selection.

_RAW = "https://cdn/user/raw_photo.jpg"
_SAFE = "https://cdn/processed/jeans.png"

_READY_CASES = [
    # (case, extra_fields, image_field, status_pair)
    ("masked_url", {"masked_url": _SAFE}, "masked_url", None),
    ("normalized_url", {"normalized_url": _SAFE}, "normalized_url", None),
    (
        "board_image_url",
        {"board_image_url": _SAFE, "board_status": "cutout_ready"},
        "board_image_url",
        ("board_status", "cutout_ready"),
    ),
    (
        "cutout_url",
        {"cutout_url": _SAFE, "cutout_status": "ready"},
        "cutout_url",
        ("cutout_status", "ready"),
    ),
    ("transparent_url", {"transparent_url": _SAFE}, "transparent_url", None),
    (
        "transparent_image_url",
        {"transparent_image_url": _SAFE},
        "transparent_image_url",
        None,
    ),
    (
        "rmbg_url",
        {"rmbg_url": _SAFE, "image_status": "rmbg_complete"},
        "rmbg_url",
        ("image_status", "rmbg_complete"),
    ),
    (
        "processed_url",
        {"processed_url": _SAFE, "image_status": "rmbg_complete"},
        "processed_url",
        ("image_status", "rmbg_complete"),
    ),
]


@pytest.mark.parametrize(
    "case,extra,image_field,status_pair",
    _READY_CASES,
    ids=[c[0] for c in _READY_CASES],
)
def test_board_ready_field_survives_normalization(case, extra, image_field, status_pair):
    raw_item = {"id": f"item-{case}", "name": "Jeans", "source": "wardrobe", "image_url": _RAW, **extra}

    # Sanity: the raw item is genuinely board-ready per the shared authority.
    assert is_board_renderable(raw_item) is True

    normalized = normalize_style_item(raw_item)

    assert normalized.get(image_field) == extra[image_field], (
        f"{case}: normalize_style_item() dropped {image_field!r}"
    )
    if status_pair is not None:
        status_field, status_value = status_pair
        assert normalized.get(status_field) == status_value, (
            f"{case}: normalize_style_item() dropped gating status {status_field!r}"
        )
    assert is_board_renderable(normalized) is True, (
        f"{case}: item was board-renderable before normalize_style_item() "
        "but not after - a qualifying field was dropped"
    )


# ------------------------------------------------- safety: never manufacture readiness
#
# The flip side of the roundtrip guarantee: normalize_style_item() must never
# turn an UNSAFE item into a READY one. It only ever copies fields through -
# it must never synthesize, upgrade or infer a board-safe field that wasn't
# already present and genuine on the raw item.

_UNSAFE_CASES = [
    ("raw_image_only", {"image_url": _RAW}),
    ("masked_equals_image", {"image_url": _RAW, "masked_url": _RAW}),
    (
        "forged_board_status_no_candidate",
        {"image_url": _RAW, "board_status": "cutout_ready"},
    ),
]


@pytest.mark.parametrize(
    "case,extra", _UNSAFE_CASES, ids=[c[0] for c in _UNSAFE_CASES]
)
def test_unsafe_item_stays_unsafe_through_normalization(case, extra):
    raw_item = {"id": f"item-{case}", "name": "Jeans", "source": "wardrobe", **extra}

    assert is_board_renderable(raw_item) is False, f"{case}: fixture is not actually unsafe"

    normalized = normalize_style_item(raw_item)

    assert is_board_renderable(normalized) is False, (
        f"{case}: normalize_style_item() manufactured board-readiness "
        "for an item that was not genuinely board-safe"
    )
