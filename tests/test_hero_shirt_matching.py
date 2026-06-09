"""Hero asset matching tests for formal-shirt vs t-shirt vs polo intents."""

from __future__ import annotations

import logging

import pytest

from services import style_reasoning_engine as engine


def _asset(asset_id: str, name: str, *, subcategory: str = "", category: str = "top", colors=None, tags=None):
    return {
        "asset_id": asset_id,
        "name": name,
        "category": category,
        "subcategory": subcategory or name.lower().replace(" ", "_"),
        "image_url": f"https://cdn.test/{asset_id}.png",
        "archetypes": ["Modern Professional"],
        "occasions": ["client_meeting"],
        "tags": tags or [],
        "colors": colors or [],
        "gender": "male",
        "status": "active",
    }


def _direction(hero: str, colors=None) -> dict:
    return {
        "archetype": "Modern Professional",
        "hero_piece": hero,
        "items": [hero],
        "colors": colors or [],
    }


# ---------- Intent classifier (pure unit) ----------

@pytest.mark.parametrize(
    "hero,expected",
    [
        ("Crisp Oxford Shirt", "formal_shirt"),
        ("Blue Button-Down Shirt", "formal_shirt"),
        ("White Dress Shirt", "formal_shirt"),
        ("Blue Button Down", "formal_shirt"),
        ("White Shirt", "formal_shirt"),
        ("White T-Shirt", "tshirt"),
        ("Navy Tee", "tshirt"),
        ("Cream Polo Shirt", "polo"),
        ("Cream Polo", "polo"),
        ("Charcoal Sweater", "knit"),
        ("Dark Wash Jeans", None),
    ],
)
def test_hero_shirt_intent_classification(hero, expected):
    assert engine._hero_shirt_intent(hero) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Mens Whiteshirt", "formal_shirt"),
        ("Mens Whitetshirt", "tshirt"),
        ("Mens Blueshirt", "formal_shirt"),
        ("Mens Bluetshirt", "tshirt"),
        ("Mens Creampoloshirt", "polo"),
        ("Mens Creamshirt", "formal_shirt"),
        ("Mens White T-Shirt", "tshirt"),
        ("Mens Grey Hoodie", "casual_pullover"),
    ],
)
def test_asset_shirt_intent_classification(name, expected):
    assert engine._asset_shirt_intent(name.lower()) == expected


# ---------- _best_style_assets selection ----------

def test_oxford_hero_selects_shirt_rejects_tshirt():
    assets = [
        _asset("white_shirt", "Mens Whiteshirt", subcategory="shirt", colors=["white"], tags=["shirt"]),
        _asset("white_tshirt", "Mens Whitetshirt", subcategory="t_shirt", colors=["white"], tags=["t_shirt"]),
    ]
    direction = _direction("Crisp Oxford Shirt", colors=["white"])
    selected = engine._best_style_assets(
        assets,
        direction=direction,
        occasion="client_meeting",
        target_gender="male",
        limit=2,
    )
    names = [a["name"] for a in selected]
    assert "Mens Whiteshirt" in names
    assert "Mens Whitetshirt" not in names


def test_button_down_hero_selects_blue_shirt_rejects_blue_tshirt():
    assets = [
        _asset("blue_shirt", "Mens Blueshirt", subcategory="shirt", colors=["blue"], tags=["shirt"]),
        _asset("blue_tshirt", "Mens Bluetshirt", subcategory="t_shirt", colors=["blue"], tags=["t_shirt"]),
        _asset("white_shirt", "Mens Whiteshirt", subcategory="shirt", colors=["white"], tags=["shirt"]),
    ]
    direction = _direction("Blue Button-Down Shirt", colors=["blue"])
    selected = engine._best_style_assets(
        assets,
        direction=direction,
        occasion="client_meeting",
        target_gender="male",
        limit=3,
    )
    names = [a["name"] for a in selected]
    assert names[0] == "Mens Blueshirt"  # color wins
    assert "Mens Bluetshirt" not in names


def test_polo_hero_selects_polo_rejects_shirt():
    assets = [
        _asset("cream_polo", "Mens Creampoloshirt", subcategory="polo", colors=["cream"], tags=["polo"]),
        _asset("cream_shirt", "Mens Creamshirt", subcategory="shirt", colors=["cream"], tags=["shirt"]),
    ]
    direction = _direction("Cream Polo Shirt", colors=["cream"])
    selected = engine._best_style_assets(
        assets,
        direction=direction,
        occasion="coffee_date",
        target_gender="male",
        limit=2,
    )
    names = [a["name"] for a in selected]
    assert "Mens Creampoloshirt" in names
    assert "Mens Creamshirt" not in names


def test_tshirt_hero_selects_tshirt_rejects_shirt():
    assets = [
        _asset("white_tshirt", "Mens Whitetshirt", subcategory="t_shirt", colors=["white"], tags=["t_shirt"]),
        _asset("white_shirt", "Mens Whiteshirt", subcategory="shirt", colors=["white"], tags=["shirt"]),
    ]
    direction = _direction("White T-Shirt", colors=["white"])
    selected = engine._best_style_assets(
        assets,
        direction=direction,
        occasion="casual_day",
        target_gender="male",
        limit=2,
    )
    names = [a["name"] for a in selected]
    assert "Mens Whitetshirt" in names
    assert "Mens Whiteshirt" not in names


# ---------- Color scoring ----------

def test_color_match_outranks_wrong_color_shirt():
    assets = [
        _asset("white_shirt", "Mens Whiteshirt", subcategory="shirt", colors=["white"], tags=["shirt"]),
        _asset("blue_shirt", "Mens Blueshirt", subcategory="shirt", colors=["blue"], tags=["shirt"]),
        _asset("black_shirt", "Mens Blackshirt", subcategory="shirt", colors=["black"], tags=["shirt"]),
    ]
    direction = _direction("Blue Button-Down Shirt", colors=["blue"])
    selected = engine._best_style_assets(
        assets,
        direction=direction,
        occasion="client_meeting",
        target_gender="male",
        limit=1,
    )
    assert selected and selected[0]["name"] == "Mens Blueshirt"


# ---------- Rejection log cap ----------

def test_rejection_logs_capped_with_summary(caplog):
    # 12 t-shirt assets vs a formal-shirt hero -> 12 rejections, cap 5, summary should fire.
    tshirt_assets = [
        _asset(f"tshirt_{i}", f"Mens Whitetshirt {i}", subcategory="t_shirt", colors=["white"], tags=["t_shirt"])
        for i in range(12)
    ]
    formal = _asset("white_shirt", "Mens Whiteshirt", subcategory="shirt", colors=["white"], tags=["shirt"])
    assets = tshirt_assets + [formal]
    direction = _direction("Crisp Oxford Shirt", colors=["white"])
    engine.logger.propagate = True
    caplog.set_level(logging.INFO)
    engine._best_style_assets(
        assets,
        direction=direction,
        occasion="client_meeting",
        target_gender="male",
        limit=1,
    )
    messages = [r.getMessage() for r in caplog.records]
    rejected_lines = [m for m in messages if m.startswith("AHVI_HERO_ASSET_REJECTED ")]
    summary_lines = [m for m in messages if m.startswith("AHVI_HERO_ASSET_REJECTED_SUMMARY")]
    assert len(rejected_lines) <= 5, f"expected cap of 5, got {len(rejected_lines)} ({rejected_lines})"
    assert summary_lines, f"expected summary line when rejections exceed cap; messages={messages}"


# ---------- Central policy section ----------

def _accessory_asset(asset_id, name, *, family_hint="", category="accessory"):
    return {
        "asset_id": asset_id,
        "name": name,
        "category": category,
        "subcategory": family_hint or name.lower().replace(" ", "_"),
        "image_url": f"https://cdn.test/{asset_id}.png",
        "archetypes": ["Modern Professional"],
        "occasions": ["client_meeting"],
        "tags": [family_hint or name.lower()],
        "colors": [],
        "gender": "male",
        "status": "active",
    }


def test_hoodie_hero_never_selects_shirt():
    assets = [
        _asset("black_shirt", "Mens Blackshirt", subcategory="shirt", colors=["black"], tags=["shirt"]),
        _asset("black_tshirt", "Mens Blacktshirt", subcategory="t_shirt", colors=["black"], tags=["t_shirt"]),
        _asset("black_hoodie", "Mens Black Hoodie", subcategory="hoodie", colors=["black"], tags=["hoodie"]),
    ]
    selected = engine._best_style_assets(
        assets,
        direction=_direction("Black Hoodie", colors=["black"]),
        occasion="casual_day",
        target_gender="male",
        limit=3,
    )
    names = [a["name"] for a in selected]
    assert "Mens Black Hoodie" in names
    assert "Mens Blackshirt" not in names
    assert "Mens Blacktshirt" not in names


def test_no_hoodie_asset_returns_no_hero_image():
    # Only a shirt + tshirt available. Hoodie hero must NOT settle for either.
    assets = [
        _asset("black_shirt", "Mens Blackshirt", subcategory="shirt", colors=["black"], tags=["shirt"]),
        _asset("black_tshirt", "Mens Blacktshirt", subcategory="t_shirt", colors=["black"], tags=["t_shirt"]),
    ]
    selected = engine._best_style_assets(
        assets,
        direction=_direction("Black Hoodie", colors=["black"]),
        occasion="casual_day",
        target_gender="male",
        limit=3,
    )
    assert selected == []


def test_oxford_hero_rejects_tshirt_via_central_policy():
    assets = [
        _asset("white_tshirt", "Mens Whitetshirt", subcategory="t_shirt", colors=["white"], tags=["t_shirt"]),
        _asset("white_polo", "Mens Whitepoloshirt", subcategory="polo", colors=["white"], tags=["polo"]),
    ]
    selected = engine._best_style_assets(
        assets,
        direction=_direction("Crisp Oxford Shirt", colors=["white"]),
        occasion="client_meeting",
        target_gender="male",
        limit=2,
    )
    assert selected == []


def test_blue_button_down_never_selects_blue_tshirt():
    # Hero is plain "Blue Button-Down Shirt" (no exact blue shirt asset);
    # must NOT fall back to blue t-shirt under any score.
    assets = [
        _asset("blue_tshirt", "Mens Bluetshirt", subcategory="t_shirt", colors=["blue"], tags=["t_shirt"]),
        _asset("blue_polo", "Mens Bluepoloshirt", subcategory="polo", colors=["blue"], tags=["polo"]),
    ]
    selected = engine._best_style_assets(
        assets,
        direction=_direction("Blue Button-Down Shirt", colors=["blue"]),
        occasion="client_meeting",
        target_gender="male",
        limit=2,
    )
    assert selected == []


def test_hero_rejects_accessory_travel_grooming():
    assets = [
        _accessory_asset("cap", "Mens Black Cap", family_hint="cap"),
        _accessory_asset("watch", "Mens Steel Watch", family_hint="watch"),
        _accessory_asset("pillow", "Mens Neck Pillow", family_hint="neck_pillow", category="travel"),
        _accessory_asset("sunscreen", "Mens Sunscreen", family_hint="skincare", category="grooming"),
    ]
    selected = engine._best_style_assets(
        assets,
        direction=_direction("White Oxford Shirt", colors=["white"]),
        occasion="client_meeting",
        target_gender="male",
        limit=3,
    )
    assert selected == []


def test_client_meeting_complete_the_look_rejects_beanie_cap_sunglasses_sandals():
    rejected_families = [
        ("beanie", "Mens Wool Beanie", "beanie"),
        ("cap", "Mens Black Cap", "cap"),
        ("sunglasses", "Mens Aviator Sunglasses", "sunglasses"),
        ("slide", "Mens Black Slides", "slide"),
        ("flip", "Mens Black Flipflops", "flip_flops"),
        ("sandal", "Mens Black Sandals", "sandal"),
    ]
    assets = [
        _accessory_asset(rid, name, family_hint=fam) for rid, name, fam in rejected_families
    ]
    accessory = engine._best_style_assets(
        assets,
        direction=_direction("Tailored Look"),
        occasion="client_meeting",
        accessory_only=True,
        target_gender="male",
        limit=5,
    )
    names = [a["name"] for a in accessory]
    assert not names, f"office CTL should reject all of these, got {names}"


def test_client_meeting_complete_the_look_prefers_belt_watch_loafer_formal_shoe():
    assets = [
        _accessory_asset("belt", "Mens Black Belt", family_hint="belt"),
        _accessory_asset("watch", "Mens Steel Watch", family_hint="watch"),
        _accessory_asset("loafer", "Mens Brown Loafer", family_hint="loafer", category="footwear"),
        _accessory_asset("formal", "Mens Black Formal Shoe", family_hint="formal_shoe", category="footwear"),
        _accessory_asset("sunglasses", "Mens Aviator Sunglasses", family_hint="sunglasses"),
    ]
    accessory = engine._best_style_assets(
        assets,
        direction=_direction("Tailored Look"),
        occasion="client_meeting",
        accessory_only=True,
        target_gender="male",
        limit=5,
    )
    names = [a["name"] for a in accessory]
    for preferred in ("Mens Black Belt", "Mens Steel Watch"):
        assert preferred in names, f"missing preferred {preferred}; got {names}"
    assert "Mens Aviator Sunglasses" not in names


def test_coffee_date_complete_the_look_allows_belt_watch_loafer_sneaker():
    assets = [
        _accessory_asset("belt", "Mens Brown Belt", family_hint="belt"),
        _accessory_asset("watch", "Mens Casual Watch", family_hint="watch"),
        _accessory_asset("loafer", "Mens Brown Loafer", family_hint="loafer", category="footwear"),
        _accessory_asset("sneaker", "Mens Clean Sneaker", family_hint="sneaker", category="footwear"),
    ]
    accessory = engine._best_style_assets(
        assets,
        direction=_direction("Refined Weekend"),
        occasion="coffee_date",
        accessory_only=True,
        target_gender="male",
        limit=4,
    )
    names = [a["name"] for a in accessory]
    # CTL slot-dedupe means belt + watch + one footwear (not both).
    assert "Mens Brown Belt" in names
    assert "Mens Casual Watch" in names
    assert any(n in names for n in ("Mens Brown Loafer", "Mens Clean Sneaker"))
