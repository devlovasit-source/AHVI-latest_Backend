"""Coffee-date / occasion intelligence wired into the visual-board path.

Reuses existing engines (style_reasoning_engine central policy,
occasion_style_rules, stylist_knowledge_service) — no new systems.
"""

from __future__ import annotations

from services import style_reasoning_engine as engine
from services.stylist_knowledge_service import classify_style_mode, STYLE_ADVICE
from brain.engines.occasion_style_rules import get_occasion_rule


def _asset(name, *, category="", subcategory="", tags=None, colors=None):
    return {
        "asset_id": name.lower().replace(" ", "_"),
        "name": name,
        "category": category,
        "subcategory": subcategory or name.lower().replace(" ", "_"),
        "image_url": f"https://cdn/{name}.png",
        "tags": tags or [],
        "colors": colors or [],
        "gender": "male",
        "status": "active",
        "occasions": ["coffee_date"],
        "archetypes": ["Smart Casual Edge"],
    }


# ---------- P0-1 coffee date demotes blazer ----------

def test_coffee_date_demotes_blazer():
    blazer = _asset("Navy Blazer", category="outerwear", subcategory="blazer", tags=["blazer"], colors=["navy"])
    polo = _asset("Knit Polo", category="top", subcategory="polo", tags=["polo"], colors=["navy"])
    direction = {"hero_piece": "Knit Polo", "items": ["Knit Polo"]}
    blazer_score = engine._asset_context_score(
        blazer, occasion="coffee_date", placement="hero", target_text="Knit Polo"
    )
    polo_score = engine._asset_context_score(
        polo, occasion="coffee_date", placement="hero", target_text="Knit Polo"
    )
    assert blazer_score < 0
    assert polo_score > blazer_score


def test_coffee_date_demotion_respects_explicit_request():
    # User asked for a blazer specifically → do not demote it.
    blazer = _asset("Navy Blazer", category="outerwear", subcategory="blazer", tags=["blazer"])
    demoted = engine._asset_context_score(
        blazer, occasion="coffee_date", placement="hero", target_text="Knit Polo"
    )
    respected = engine._asset_context_score(
        blazer, occasion="coffee_date", placement="hero", target_text="Navy Blazer"
    )
    assert respected > demoted


# ---------- P0-1 coffee date rejects beanie ----------

def test_coffee_date_rejects_beanie():
    beanie = _asset("Wool Beanie", category="accessory", subcategory="beanie", tags=["beanie"])
    assert (
        engine._asset_allowed_for_context(
            beanie, occasion="coffee_date", placement="complete", target_text="Relaxed Look"
        )
        is False
    )
    # And as a missing piece.
    assert (
        engine._asset_allowed_for_context(
            beanie, occasion="coffee_date", placement="missing", target_text="Relaxed Look"
        )
        is False
    )


# ---------- P0-3 long date prompt routes to boards ----------

def test_long_date_prompt_routes_to_boards():
    assert classify_style_mode("first coffee date at a cafe on Saturday afternoon") == STYLE_ADVICE
    assert classify_style_mode("planning a dinner date tonight, what works?") == STYLE_ADVICE
    assert classify_style_mode("coffee date") == STYLE_ADVICE


# ---------- P0-2 occasion rule extended ----------

def test_coffee_date_rule_avoids_blazer_and_beanie():
    rule = get_occasion_rule("coffee_date")
    avoid = " ".join(rule.get("avoid_keywords") or []).lower()
    assert "blazer" in avoid
    assert "beanie" in avoid
    assert "tie" in avoid


# ---------- P0-4 occasion voice on card copy ----------

def _polished(occasion):
    directions = [
        {
            "archetype": "Smart Casual Edge",
            "title": "Relaxed Confidence",
            "items": ["Knit Polo", "Dark Jeans", "Clean Sneakers"],
            "pieces": ["Knit Polo", "Dark Jeans", "Clean Sneakers"],
            "why_it_works": (
                "The shirt provides a clear point of view while maintaining a considered aesthetic "
                "that feels intentional without becoming forced."
            ),
        }
    ]
    return engine._apply_editorial_polish(directions, occasion=occasion, wardrobe_items=None)[0]


def test_coffee_date_copy_uses_occasion_voice():
    note = _polished("coffee_date")["short_note"].lower()
    assert "approachable" in note or "relaxed" in note
    # The generic LLM phrasing must not survive as the lead copy.
    assert "clear point of view" not in note


def test_funeral_copy_understated():
    note = _polished("funeral")["short_note"].lower()
    assert "respectful" in note and "understated" in note


def test_conference_copy_confident():
    note = _polished("conference")["short_note"].lower()
    assert "confident" in note or "credible" in note


# ---------- P1-1 no duplicate blazer heroes ----------

def test_no_duplicate_blazer_heroes():
    directions = [
        {"hero_piece": "Navy Blazer", "items": ["Navy Blazer", "Grey Trouser"], "title": "A"},
        {"hero_piece": "Charcoal Blazer", "items": ["Charcoal Blazer", "Dark Jeans"], "title": "B"},
        {"hero_piece": "Camel Coat", "items": ["Camel Coat", "Black Trouser"], "title": "C"},
    ]
    out = engine._apply_generic_visual_diversity(directions, category="coffee date")
    heroes = [engine._detect_family(d.get("hero_piece")) for d in out]
    structured = sum(1 for h in heroes if h in {"blazer", "jacket", "coat"})
    # At most one structured-layer hero should survive the diversity guard.
    assert structured <= 1


# ---------- P1-2 missing piece occasion-aware ----------

def test_missing_piece_is_occasion_aware():
    # A beanie asset can never be selected as a coffee-date missing piece.
    beanie = _asset("Grey Beanie", category="accessory", subcategory="beanie", tags=["beanie"])
    loafer = _asset("Suede Loafer", category="footwear", subcategory="loafer", tags=["loafer"])
    selected = engine._best_style_assets(
        [beanie, loafer],
        direction={"hero_piece": "Suede Loafer", "items": ["Suede Loafer"], "archetype": "Smart Casual Edge"},
        occasion="coffee_date",
        target_gender="male",
        placement="missing",
        limit=2,
    )
    names = [a["name"] for a in selected]
    assert "Grey Beanie" not in names
