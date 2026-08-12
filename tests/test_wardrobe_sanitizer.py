"""Tests for the shared fashion sanitizer and its choke-point wiring."""

from __future__ import annotations

import pytest

from services.wardrobe_sanitizer import (
    is_fashion_item,
    sanitize_fashion_wardrobe_items,
)
from services.style_context_service import (
    build_missing_piece_intelligence,
    build_style_context,
)
from services.llm_service import format_wardrobe_for_llm
from services import style_reasoning_engine as engine


# ---------- is_fashion_item ----------

@pytest.mark.parametrize(
    "item",
    [
        {"name": "Phone Charger", "category": "electronics"},
        {"name": "Charger", "category": ""},
        {"name": "USB-C Cable", "category": "misc"},
        {"name": "Power Bank", "category": "accessory"},  # blocked token wins
        {"name": "Powerbank 20000mAh", "category": ""},
        {"name": "Travel Adapter", "category": "travel_accessory"},
        {"name": "Wireless Earbuds", "category": "electronics"},
        {"name": "Headphones", "category": ""},
        {"name": "Face Moisturizer", "category": "skincare"},
        {"name": "Makeup Brush", "category": "accessory"},
        {"name": "Ruby Lipstick", "category": "misc"},
        {"name": "Liquid Foundation", "category": "unknown"},
        {"name": "Black Mascara", "category": "makeup"},
        {"name": "Waterproof Eyeliner", "category": "beauty"},
        {"name": "Red Nail Polish", "category": "personal care"},
        {"name": "Hair Dryer", "category": "misc"},
        {"name": "Hair Straightener", "category": "unknown"},
        {"name": "Razor", "category": "grooming"},
        {"name": "Wide Tooth Comb", "category": ""},
        {"name": "Toothbrush", "category": ""},
        {"name": "Water Bottle", "category": "misc"},
        {"name": "Coffee Cup", "category": "household"},
        {"name": "Drinking Glass", "category": "unknown"},
        {"name": "Dinner Plate", "category": "utensils"},
        {"name": "Neck Pillow", "category": "travel_accessory"},
        {"name": "Eye Mask", "category": ""},
        {"name": "Notebook", "category": "stationery"},
        {"name": "Mystery Object", "category": "unknown"},
        {"name": "Something", "category": ""},  # no fashion signal at all
    ],
)
def test_non_fashion_items_rejected(item):
    assert is_fashion_item(item) is False


@pytest.mark.parametrize(
    "item",
    [
        {"name": "Navy Blazer", "category": "outerwear"},
        {"name": "White Oxford Shirt", "category": "top"},
        {"name": "Grey Trousers", "category": "bottom"},
        {"name": "Black Loafers", "category": "footwear"},
        {"name": "Steel Watch", "category": "watch"},
        {"name": "Brown Leather Belt", "category": "accessory"},
        {"name": "Combat Boots", "category": ""},  # "comb" must not match
        {"name": "Pendant Necklace", "category": "jewellery"},
        {"name": "Blue Kurta", "category": "ethnicwear"},
        {"name": "Laptop Bag", "category": "bag"},
        {"name": "Open Toe Heels", "category": ""},  # "pen" must not match
        {"name": "Silk Scarf", "category": "misc"},
        {"name": "Blue Linen Shirt", "category": "misc"},
        {"name": "White Leather Sneakers", "category": "unknown"},
    ],
)
def test_fashion_items_accepted(item):
    assert is_fashion_item(item) is True


def test_blocked_category_cannot_be_rescued_by_fashion_name():
    assert is_fashion_item(
        {"name": "Blue Linen Shirt", "category": "electronics"}
    ) is False


def test_misc_charger_and_accessory_charger_are_rejected():
    assert is_fashion_item({"name": "USB Phone Charger", "category": "misc"}) is False
    assert is_fashion_item(
        {"name": "USB Phone Charger", "category": "accessory"}
    ) is False


def test_sanitize_filters_and_keeps_order():
    items = [
        {"name": "Navy Blazer", "category": "outerwear"},
        {"name": "Phone Charger", "category": "electronics"},
        {"name": "Grey Trousers", "category": "bottom"},
        {"name": "Power Bank", "category": ""},
        {"name": "Black Loafers", "category": "footwear"},
        "not-a-dict",
    ]
    out = sanitize_fashion_wardrobe_items(items, source="test")
    assert [i["name"] for i in out] == ["Navy Blazer", "Grey Trousers", "Black Loafers"]


def test_sanitize_handles_non_list():
    assert sanitize_fashion_wardrobe_items(None) == []
    assert sanitize_fashion_wardrobe_items("junk") == []


# ---------- build_style_context wiring ----------

def test_style_context_excludes_charger():
    ctx = build_style_context(
        query="office outfit",
        wardrobe_items=[
            {"name": "Navy Blazer", "category": "outerwear"},
            {"name": "Phone Charger", "category": "electronics"},
            {"name": "USB Cable", "category": "misc"},
        ],
    )
    names = [i["name"] for i in ctx["wardrobe_items"]]
    assert names == ["Navy Blazer"]
    assert ctx["wardrobe_summary"]["total_items"] == 1


# ---------- format_wardrobe_for_llm wiring ----------

def test_llm_format_excludes_non_fashion():
    text = format_wardrobe_for_llm(
        [
            {"name": "Navy Blazer", "category": "outerwear", "color": "navy"},
            {"name": "Phone Charger", "category": "electronics", "color": "black"},
            {"name": "Powerbank", "category": "", "color": "white"},
        ]
    )
    assert "Blazer" in text
    assert "Charger" not in text
    assert "Powerbank" not in text


def test_llm_format_empty_after_sanitize():
    text = format_wardrobe_for_llm(
        [{"name": "Phone Charger", "category": "electronics"}]
    )
    assert text == "Wardrobe is empty."


# ---------- missing_piece_intelligence wiring ----------

def test_missing_piece_intelligence_excludes_non_fashion():
    block = build_missing_piece_intelligence(
        wardrobe_summary={"total_items": 3, "has_top": True, "has_bottom": True, "has_footwear": True},
        missing_items=[
            {"name": "Brushed Steel Watch", "category": "accessory", "reason": "polish"},
            {"name": "Power Bank", "category": "electronics", "reason": "travel"},
            {"name": "Neck Pillow", "category": "travel_accessory", "reason": "comfort"},
        ],
    )
    names = [m["name"] for m in block["missing_items"]]
    assert names == ["Brushed Steel Watch"]


# ---------- owned_items wiring (style_reasoning_engine) ----------

def test_owned_items_excludes_charger_via_shared_gate():
    direction = {
        "items": ["Navy Blazer", "Charger"],
        "pieces": ["Navy Blazer", "Charger"],
    }
    wardrobe = [
        {"name": "Navy Blazer", "category": "outerwear"},
        {"name": "Phone Charger", "category": "electronics"},
    ]
    polished = engine._apply_editorial_polish(
        [direction], occasion="travel", wardrobe_items=wardrobe
    )[0]
    names = [o["name"] for o in polished["owned_items"]]
    assert "Phone Charger" not in names
    assert "Navy Blazer" in names


# ---------- asset integrity: name/image stay paired ----------

def test_owned_items_name_image_pair_from_same_row():
    wardrobe = [
        {"id": "w1", "name": "Navy Blazer", "category": "outerwear", "image_url": "https://w/blazer.png"},
        {"id": "w2", "name": "Brown Loafer", "category": "footwear", "image_url": "https://w/loafer.png"},
    ]
    direction = {"items": ["Navy Blazer", "Brown Loafer"], "pieces": ["Navy Blazer", "Brown Loafer"]}
    polished = engine._apply_editorial_polish(
        [direction], occasion="office", wardrobe_items=wardrobe
    )[0]
    by_name = {o["name"]: o for o in polished["owned_items"]}
    assert by_name["Navy Blazer"]["image_url"] == "https://w/blazer.png"
    assert by_name["Brown Loafer"]["image_url"] == "https://w/loafer.png"


def test_hero_asset_never_substitutes_wrong_family_image():
    # Hoodie hero + only glove/shirt assets available → no image at all,
    # never a wrong-family substitute.
    assets = [
        {
            "asset_id": "gloves",
            "name": "Leather Gloves",
            "category": "accessory",
            "subcategory": "gloves",
            "image_url": "https://cdn/gloves.png",
            "gender": "male",
            "status": "active",
            "occasions": ["casual_day"],
            "archetypes": ["Modern Utility"],
            "tags": ["gloves"],
            "colors": ["black"],
        },
        {
            "asset_id": "shirt",
            "name": "Navy Shirt",
            "category": "top",
            "subcategory": "shirt",
            "image_url": "https://cdn/shirt.png",
            "gender": "male",
            "status": "active",
            "occasions": ["casual_day"],
            "archetypes": ["Modern Utility"],
            "tags": ["shirt"],
            "colors": ["navy"],
        },
    ]
    selected = engine._best_style_assets(
        assets,
        direction={
            "archetype": "Modern Utility",
            "hero_piece": "Navy Chore Jacket",
            "items": ["Navy Chore Jacket"],
            "colors": ["navy"],
        },
        occasion="casual_day",
        target_gender="male",
        limit=2,
    )
    names = [a["name"] for a in selected]
    assert "Leather Gloves" not in names
    assert "Navy Shirt" not in names  # jacket hero must not borrow a shirt image
