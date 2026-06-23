"""Catalog visual board: each look must carry the core slots (top + bottom +
footwear) and not repeat the same cutout across looks.

Regression cover for the items issue where a hero only filled one main garment
(top-led look had no bottom) and the same loafers/shirt repeated across all
three directions.
"""
from services import style_reasoning_engine as sre


def _asset(name, role_cat, url, gender="male"):
    return {
        "asset_id": name.replace(" ", "_"),
        "name": name,
        "category": role_cat,
        "subcategory": role_cat,
        "image_url": url,
        "board_image_url": url,
        "cutout_status": "ready",
        "gender": gender,
        "status": "active",
        "occasions": [],
        "tags": [],
        "archetypes": [],
        "colors": [],
    }


ASSETS = [
    _asset("White Shirt", "shirt", "https://x/whiteshirt.png"),
    _asset("Blue Shirt", "shirt", "https://x/blueshirt.png"),
    _asset("Slim Jeans", "jeans", "https://x/jeans.png"),
    _asset("Grey Trousers", "trousers", "https://x/trousers.png"),
    _asset("Brown Loafers", "loafers", "https://x/loafers.png"),
    _asset("White Sneakers", "sneakers", "https://x/sneakers.png"),
]


def test_complete_board_fills_missing_bottom():
    # A top-only look should gain a bottom and footwear.
    board = [
        {"name": "White Shirt", "role": "top", "image_url": "https://x/whiteshirt.png", "source": "asset"}
    ]
    out = sre._complete_board_items(
        board,
        ASSETS,
        direction={"hero_piece": "white shirt"},
        occasion="office",
        target_gender="male",
        allow_feminine_accessory=False,
        used_urls=set(),
        brief=None,
    )
    roles = {i["role"] for i in out}
    assert "top" in roles and "bottom" in roles and "footwear" in roles


def test_complete_board_respects_used_urls_for_dedup():
    # Loafers already used on a prior look -> footwear here must differ
    # (casual occasion so both loafers and sneakers are valid).
    board = [
        {"name": "Blue Shirt", "role": "top", "image_url": "https://x/blueshirt.png", "source": "asset"}
    ]
    out = sre._complete_board_items(
        board,
        ASSETS,
        direction={"hero_piece": "blue shirt"},
        occasion="casual",
        target_gender="male",
        allow_feminine_accessory=False,
        used_urls={"https://x/loafers.png"},
        brief=None,
    )
    foot = [i for i in out if i["role"] == "footwear"]
    assert foot, "expected a footwear item"
    assert foot[0]["image_url"] != "https://x/loafers.png"


def test_dress_look_is_left_alone():
    board = [
        {"name": "Red Dress", "role": "dress", "image_url": "https://x/dress.png", "source": "asset"}
    ]
    out = sre._complete_board_items(
        board,
        ASSETS,
        direction={"hero_piece": "red dress"},
        occasion="party",
        target_gender="female",
        allow_feminine_accessory=True,
        used_urls=set(),
        brief=None,
    )
    assert out == board  # no top/bottom forced onto a dress look


def test_best_asset_for_role_excludes_used():
    used = {"https://x/whiteshirt.png"}
    asset = sre._best_asset_for_role(
        ASSETS,
        role="top",
        direction={"hero_piece": "shirt"},
        occasion="office",
        target_gender="male",
        exclude_urls=used,
        brief=None,
    )
    assert asset is not None
    assert asset["image_url"] != "https://x/whiteshirt.png"
