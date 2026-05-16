import json

from services.wardrobe_intelligence_service import (
    board_has_occasion_conflict,
    enrich_wardrobe_item,
    normalize_occasion,
    score_item_for_occasion,
)
from brain.engines.outfit_quality_guard import reject_board_for_occasion


def _item(name, category="", subcategory=""):
    item = {"name": name, "category": category, "subcategory": subcategory}
    item["style_metadata"] = json.dumps(enrich_wardrobe_item(item))
    return item


def test_enrich_test_matrix_core_items():
    assert enrich_wardrobe_item({"name": "Leather Belt"})["subcategory"] == "Belt"
    assert enrich_wardrobe_item({"name": "Black Loafers"})["avoid_for"] == ["gym", "beach", "rave"]
    assert "beach" in enrich_wardrobe_item({"name": "Linen shirt"})["occasion_affinity"]
    assert enrich_wardrobe_item({"name": "Silk Saree"})["category"] == "Dresses"


def test_normalize_occasion_aliases():
    assert normalize_occasion("beach outfit") == "beach"
    assert normalize_occasion("party wear") == "house_party"
    assert normalize_occasion("rave outfit") == "rave"
    assert normalize_occasion("cocktail outfit") == "cocktail"
    assert normalize_occasion("office outfit") == "office"
    assert normalize_occasion("date night") == "date"
    assert normalize_occasion("temple outfit") == "temple"
    assert normalize_occasion("outfit for today") == "daily"


def test_metadata_scoring_blocks_bad_occasion_matches():
    suit = _item("Navy Suit", "Outerwear", "Blazer")
    loafers = _item("Black Loafers", "Footwear", "Loafers")
    belt = _item("Leather Belt", "Accessories", "Belt")
    sandals = _item("Brown Sandals", "Footwear", "Sandals")

    assert score_item_for_occasion(suit, "beach outfit") <= -10
    assert score_item_for_occasion(suit, "rave outfit") <= -10
    assert score_item_for_occasion(loafers, "beach outfit") <= -10
    assert score_item_for_occasion(belt, "gym") <= -10
    assert score_item_for_occasion(belt, "beach") <= -10
    assert score_item_for_occasion(belt, "rave") <= -10
    assert score_item_for_occasion(sandals, "office outfit") <= -10


def test_board_guard_rejects_metadata_conflicts():
    beach_board = {"items": [_item("Navy Suit"), _item("Black Loafers")]}
    office_board = {"items": [_item("Linen Shirt"), _item("Brown Sandals")]}
    party_board = {"items": [_item("Navy Suit"), _item("Blue Jeans")]}

    assert board_has_occasion_conflict(beach_board, "beach outfit")
    assert reject_board_for_occasion(beach_board, "beach")[0]
    assert reject_board_for_occasion(office_board, "office")[0]
    assert reject_board_for_occasion(party_board, "party wear")[0]
