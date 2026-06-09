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
