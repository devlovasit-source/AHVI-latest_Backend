"""Meal-board slot + dietary-restriction filtering (beta correctness).

A restriction/slot-oriented prompt must no longer collapse to the balanced
full-day template.
"""
from services.board_service import (
    _detect_diet_restrictions,
    _detect_diet_variant,
    _detect_meal_slot,
    build_diet_visual_board,
)

_GLUTEN = ("wheat", "bread", "roti", "pasta", "oat", "wrap", "burrito", "naan")
_DAIRY = ("milk", "paneer", "cheese", "curd", "yogurt", "butter", "ghee")


def _board_for(msg: str) -> dict:
    return build_diet_visual_board(
        diet_variant=_detect_diet_variant(msg),
        meal_slot=_detect_meal_slot(msg),
        restrictions=_detect_diet_restrictions(msg),
    )


def _titles(board: dict):
    return [s["title"] for s in board["sections"]]


def _all_text(board: dict) -> str:
    parts = []
    for s in board["sections"]:
        for it in s.get("items", []):
            parts.append(str(it.get("name", "")))
            parts.append(str(it.get("pairing", "")))
            parts.extend(str(o) for o in it.get("options", []))
        parts.extend(str(x) for x in s.get("turn_into", []))
    return " ".join(parts).lower()


def test_lunch_gluten_free_is_lunch_only_and_gluten_stripped():
    b = _board_for("Meal plan for lunch, gluten free")
    assert _titles(b) == ["Lunch"]
    assert "Breakfast" not in _titles(b) and "Dinner" not in _titles(b)
    txt = _all_text(b)
    for tok in _GLUTEN:
        assert tok not in txt, tok  # e.g. turn_into "wrap"/"burrito" removed
    assert b["meal_slot"] == "lunch"
    assert "gluten_free" in b["restrictions"]


def test_gluten_free_full_day_keeps_compatible_sections():
    b = _board_for("Gluten-free meals for today")
    assert _titles(b) == ["Breakfast", "Lunch", "Dinner"]
    txt = _all_text(b)
    assert "oat" not in txt  # "Overnight oats" excluded (not certified GF)
    for tok in _GLUTEN:
        assert tok not in txt, tok


def test_dairy_free_dinner_only_no_dairy():
    b = _board_for("Dairy-free dinner")
    assert _titles(b) == ["Dinner"]
    txt = _all_text(b)
    for tok in _DAIRY:
        assert tok not in txt, tok
    assert "dairy_free" in b["restrictions"]


def test_vegan_lunch_uses_vegan_variant_and_lunch_slot():
    b = _board_for("Vegan lunch")
    assert _titles(b) == ["Lunch"]
    assert b["context_used"]["diet_variant"] == "vegan"
    assert b["meal_slot"] == "lunch"


def test_high_protein_breakfast_only():
    b = _board_for("High-protein breakfast")
    assert _titles(b) == ["Breakfast"]
    assert b["context_used"]["diet_variant"] == "high_protein"


def test_no_slot_keeps_full_day():
    b = _board_for("Suggest a meal plan")
    assert _titles(b) == ["Breakfast", "Lunch", "Dinner"]
    assert b["meal_slot"] == "full_day"
    assert not b["fallback_reason"]


def test_unknown_restriction_not_silently_claimed():
    # "nut free" is not a supported restriction: we must not claim/apply it.
    b = _board_for("nut free meal plan")
    assert b["restrictions"] == []  # no false nut_free claim
    # normal template still returned (still contains nuts) — not a fake filter
    assert "nut" in _all_text(b)


def test_no_compatible_template_returns_truthful_fallback():
    b = _board_for("gluten free snack")  # templates carry no snack section
    assert b["fallback_reason"]  # truthful, non-empty
    assert b["title"] != "Balanced Day Meal Plan"  # not the incompatible default
    assert b["reason"]
    # any offered suggestion must itself be compatible (no gluten)
    txt = _all_text(b)
    for tok in _GLUTEN:
        assert tok not in txt, tok


def test_existing_templates_still_build_full_day():
    for variant in ("balanced", "vegan", "high_protein", "keto"):
        b = build_diet_visual_board(diet_variant=variant)
        assert _titles(b) == ["Breakfast", "Lunch", "Dinner"]
        assert b["board_type"] == "diet_plan"
        assert b["fallback_reason"] == ""
