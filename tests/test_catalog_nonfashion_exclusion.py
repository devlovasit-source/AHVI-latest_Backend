"""P0 catalog patch — non-fashion exclusion + field-mismatch signal recovery.

Pure tests (no Gemini/DB). Cover the centralized non-fashion gate and the
colors/tags fallback in the board asset scorer.
"""

import pytest

from services import style_reasoning_engine as e


# ---------------------------------------------------------------------------
# TASK A — non-fashion exclusion helper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("asset", [
    {"name": "Airpods", "category": "accessory"},
    {"name": "Walletdinor", "category": "accessory"},
    {"name": "Hair Dryer", "category": "grooming"},
    {"name": "Hair Straightener", "category": "accessory"},
    {"name": "Chewinggum", "category": "accessory"},
    {"name": "Mouthfreshener", "category": "accessory"},
    {"name": "Travelweighingscale", "category": "travel"},
    {"name": "Foldable Umberella", "category": "accessory"},
    {"name": "Drinkflask", "category": "accessory"},
    {"name": "Some Skincare Serum", "category": "accessory"},
    {"name": "Random Item", "category": "grooming"},   # category-based
    {"name": "Random Item", "category": "travel"},      # category-based
])
def test_nonfashion_detected(asset):
    assert e._is_nonfashion_asset(asset) is True


@pytest.mark.parametrize("asset", [
    {"name": "White Oxford Shirt", "category": "top"},
    {"name": "Black Graphic Tee", "category": "top"},
    {"name": "Tan Loafers", "category": "footwear"},
    {"name": "Beige Shell Bag", "category": "accessory"},
    {"name": "Silk Kurta", "category": "ethnic"},
    {"name": "Cargo Pants", "category": "bottom"},
])
def test_fashion_not_excluded(asset):
    assert e._is_nonfashion_asset(asset) is False


def _acc_pool():
    junk = [
        {"asset_id": "j1", "name": "Airpods", "category": "accessory", "image_url": "x", "gender": "unisex", "occasions": ["party", "office", "wedding"]},
        {"asset_id": "j2", "name": "Walletdinor", "category": "accessory", "image_url": "x", "gender": "unisex", "occasions": ["party", "office", "wedding"]},
        {"asset_id": "j3", "name": "Hair Dryer", "category": "grooming", "image_url": "x", "gender": "unisex", "occasions": ["party", "office", "wedding"]},
        {"asset_id": "j4", "name": "Travelweighingscale", "category": "travel", "image_url": "x", "gender": "unisex", "occasions": ["party", "office", "wedding"]},
    ]
    good = {"asset_id": "g1", "name": "Black Sunglasses", "category": "accessory", "subcategory": "sunglasses",
            "image_url": "x", "gender": "unisex", "occasions": ["party", "office", "wedding"], "colors": ["black"]}
    return junk + [good]


@pytest.mark.parametrize("occ", ["music festival", "office", "wedding"])
def test_nonfashion_never_selected(occ):
    direction = {"archetype": "smart casual edge", "hero_piece": "sunglasses", "colors": ["black"]}
    picks = e._best_style_assets(
        _acc_pool(), direction=direction, occasion=occ,
        accessory_only=True, target_gender="male", limit=5,
    )
    names = " ".join((p.get("name") or "").lower() for p in picks)
    for bad in ("airpod", "wallet", "hair dryer", "weighingscale"):
        assert bad not in names


# ---------------------------------------------------------------------------
# TASK B — recover colors / tags signal
# ---------------------------------------------------------------------------
def _direction():
    return {"archetype": "smart casual edge", "hero_piece": "shirt", "items": ["shirt"], "colors": ["black"]}


def test_colors_field_contributes_color_score():
    d = _direction()
    with_colors = {"name": "Shirt", "image_url": "x", "gender": "male", "colors": ["black"]}
    no_colors = {"name": "Shirt", "image_url": "x", "gender": "male"}
    s_yes = e._asset_score(with_colors, direction=d, occasion="party", target_gender="male")
    s_no = e._asset_score(no_colors, direction=d, occasion="party", target_gender="male")
    assert s_yes > s_no  # colors[] now scored (was lost as singular 'color')


def test_tags_field_contributes_style_score():
    d = _direction()  # archetype 'smart casual edge'
    with_tags = {"name": "Shirt", "image_url": "x", "gender": "male", "tags": ["smart casual edge"]}
    no_tags = {"name": "Shirt", "image_url": "x", "gender": "male"}
    s_yes = e._asset_score(with_tags, direction=d, occasion="party", target_gender="male")
    s_no = e._asset_score(no_tags, direction=d, occasion="party", target_gender="male")
    assert s_yes > s_no  # tags now fall back into style_tags (+3)


def test_legacy_singular_color_still_works():
    d = _direction()
    legacy = {"name": "Shirt", "image_url": "x", "gender": "male", "color": ["black"]}
    none = {"name": "Shirt", "image_url": "x", "gender": "male"}
    assert e._asset_score(legacy, direction=d, occasion="party", target_gender="male") \
        > e._asset_score(none, direction=d, occasion="party", target_gender="male")
