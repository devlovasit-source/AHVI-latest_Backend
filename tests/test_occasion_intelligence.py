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


# ---------- P0 occasion-safe visual assets ----------

def test_haldi_guard_blocks_western_casual_assets_and_prefers_kurta():
    assets = [
        _asset("Wool Beanie", category="accessory", subcategory="beanie"),
        _asset("Baseball Cap", category="accessory", subcategory="baseball_cap"),
        _asset("White Sneaker", category="footwear", subcategory="sneaker"),
        _asset("Blue Oxford Shirt", category="top", subcategory="oxford_shirt"),
        _asset("Cream Polo", category="top", subcategory="polo"),
        _asset(
            "Marigold Yellow Kurta",
            category="ethnic",
            subcategory="kurta",
            tags=["festive", "haldi"],
            colors=["yellow"],
        ),
    ]
    direction = {
        "hero_piece": "Festive Kurta",
        "items": ["Festive Kurta"],
        "archetype": "Vibrant Comfort",
    }
    selected = engine._best_style_assets(
        assets,
        direction=direction,
        occasion="haldi wedding",
        target_gender="male",
        placement="hero",
        limit=3,
    )
    assert [item["name"] for item in selected] == ["Marigold Yellow Kurta"]
    for blocked in assets[:-1]:
        assert not engine._asset_allowed_for_context(
            blocked,
            occasion="haldi wedding",
            placement="complete",
            target_text="Festive Kurta",
        )


def test_wedding_guard_blocks_cap_beanie_and_sneaker():
    for asset in (
        _asset("Dad Cap", category="accessory", subcategory="cap"),
        _asset("Black Beanie", category="accessory", subcategory="beanie"),
        _asset("Running Sneaker", category="footwear", subcategory="sneaker"),
    ):
        assert not engine._asset_allowed_for_context(
            asset,
            occasion="cousin wedding",
            placement="complete",
            target_text="Bandhgala",
        )


def test_airport_beanie_requires_cold_context():
    beanie = _asset("Wool Beanie", category="accessory", subcategory="beanie")
    assert not engine._asset_allowed_for_context(
        beanie,
        occasion="airport outfit",
        placement="complete",
        target_text="travel layers",
    )
    assert engine._asset_allowed_for_context(
        beanie,
        occasion="airport outfit in cold winter weather",
        placement="complete",
        target_text="travel layers",
    )


def test_conference_guard_blocks_casual_assets():
    for asset in (
        _asset("Beanie", category="accessory", subcategory="beanie"),
        _asset("Baseball Cap", category="accessory", subcategory="cap"),
        _asset("Gym Shorts", category="bottom", subcategory="gym_shorts"),
        _asset("Athletic Sneakers", category="footwear", subcategory="sneaker"),
    ):
        assert not engine._asset_allowed_for_context(
            asset,
            occasion="conference presentation",
            placement="complete",
            target_text="Modern Professional",
        )


def test_christian_wedding_allows_western_formal_but_blocks_casual_assets():
    blazer = _asset("Navy Blazer", category="outerwear", subcategory="blazer")
    oxford = _asset("White Oxford Shirt", category="top", subcategory="oxford_shirt")
    assert engine._asset_allowed_for_context(
        blazer,
        occasion="christian wedding",
        placement="hero",
        target_text="Navy Blazer",
    )
    assert engine._asset_allowed_for_context(
        oxford,
        occasion="christian wedding",
        placement="hero",
        target_text="White Oxford Shirt",
    )
    for asset in (
        _asset("Beanie", category="accessory", subcategory="beanie"),
        _asset("Baseball Cap", category="accessory", subcategory="cap"),
        _asset("Running Sneaker", category="footwear", subcategory="sneaker"),
    ):
        assert not engine._asset_allowed_for_context(
            asset,
            occasion="christian wedding",
            placement="complete",
            target_text="Church Formal",
        )


def test_haldi_and_airport_cards_have_distinct_contextual_copy():
    directions = [
        {
            "archetype": name,
            "title": name,
            "hero_piece": hero,
            "items": [hero],
            "pieces": [hero],
        }
        for name, hero in (
            ("Vibrant Comfort", "Yellow Kurta"),
            ("Wedding Day Ease", "Ivory Kurta Set"),
            ("Festive Heritage", "Nehru Jacket"),
        )
    ]
    for occasion, query in (
        ("wedding", "Haldi"),
        ("airport_travel", "airport outfit"),
    ):
        polished = engine._apply_editorial_polish(
            directions,
            occasion=occasion,
            wardrobe_items=None,
            context_text=query,
        )
        notes = [item["short_note"] for item in polished]
        assert len(set(notes)) == 3
        assert all(len(note.split()) <= 18 for note in notes)
        assert all(note != engine._DEFAULT_OCCASION_VOICE for note in notes)


def test_haldi_bad_missing_piece_is_replaced_with_festive_piece():
    missing = engine._enrich_missing_piece_with_asset(
        {
            "name": "Olive Cotton Overshirt",
            "category": "outerwear",
            "reason": "Adds a useful layer.",
        },
        assets=[],
        occasion="wedding Haldi",
        target_gender="male",
    )
    assert missing is not None
    assert missing["name"] == "Ethnic Mojari Footwear"
    assert missing["category"] == "footwear"
    assert "comfortable for rituals" in missing["reason"]


def test_haldi_visual_direction_filters_bad_support_and_missing_piece(monkeypatch):
    assets = [
        _asset("Black Beanie", category="accessory", subcategory="beanie"),
        _asset("Baseball Cap", category="accessory", subcategory="cap"),
        _asset("Running Sneakers", category="footwear", subcategory="sneaker"),
        _asset("Office Derby Shoes", category="footwear", subcategory="formal_shoe"),
        _asset("Western Wallet", category="accessory", subcategory="wallet"),
        _asset(
            "Gold Festive Brooch",
            category="accessory",
            subcategory="brooch",
            tags=["festive", "traditional"],
            colors=["gold"],
        ),
        _asset(
            "Maroon Mojari",
            category="footwear",
            subcategory="mojari",
            tags=["ethnic", "festive"],
            colors=["maroon"],
        ),
    ]
    monkeypatch.setattr(engine, "_style_asset_rows", lambda limit=120: assets)
    directions = engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Festive Heritage",
                "title": "Festive Heritage",
                "hero_piece": "Marigold Kurta",
                "items": ["Marigold Kurta", "Ivory Churidar"],
                "complete_the_look": [
                    {"name": "Olive Cotton Overshirt", "category": "outerwear"},
                    {"name": "Blue Oxford Shirt", "category": "top"},
                    {"name": "Cream Polo", "category": "top"},
                ],
                "missing_piece": {
                    "name": "Olive Cotton Overshirt",
                    "category": "outerwear",
                    "reason": "Adds a layer.",
                },
            }
        ],
        occasion="wedding Haldi",
        target_gender="male",
    )
    card = directions[0]
    support_names = " ".join(item["name"] for item in card["complete_the_look"]).lower()
    for blocked in ("beanie", "cap", "sneaker", "overshirt", "oxford", "polo", "wallet", "derby"):
        assert blocked not in support_names
    assert any(term in support_names for term in ("brooch", "mojari"))
    assert card["missing_piece"]["name"] == "Ethnic Mojari Footwear"


def test_wedding_guard_allows_ethnic_assets_and_blocks_office_support():
    allowed = (
        _asset("Ivory Kurta", category="ethnic", subcategory="kurta"),
        _asset("Navy Sherwani", category="ethnic", subcategory="sherwani"),
        _asset("Maroon Bandhgala", category="ethnic", subcategory="bandhgala"),
        _asset("Gold Jutti", category="footwear", subcategory="jutti", tags=["ethnic"]),
        _asset("Brown Mojari", category="footwear", subcategory="mojari", tags=["ethnic"]),
    )
    blocked = (
        _asset("Baseball Cap", category="accessory", subcategory="cap"),
        _asset("Wool Beanie", category="accessory", subcategory="beanie"),
        _asset("Running Sneakers", category="footwear", subcategory="sneaker"),
        _asset("Office Wallet", category="accessory", subcategory="wallet"),
    )
    assert all(
        engine._asset_allowed_for_context(
            asset,
            occasion="cousin wedding",
            placement="complete" if asset["category"] != "ethnic" else "hero",
            target_text="Festive Heritage",
        )
        for asset in allowed
    )
    assert all(
        not engine._asset_allowed_for_context(
            asset,
            occasion="cousin wedding",
            placement="complete",
            target_text="Festive Heritage",
        )
        for asset in blocked
    )


def test_airport_policy_still_allows_sneakers_and_bag():
    for asset in (
        _asset("Clean Sneakers", category="footwear", subcategory="sneaker"),
        _asset("Travel Backpack", category="accessory", subcategory="backpack"),
    ):
        assert engine._asset_allowed_for_context(
            asset,
            occasion="airport outfit",
            placement="complete",
            target_text="travel layers",
        )


def test_festive_copy_scrubber_removes_malformed_internal_phrase():
    malformed = "wedding Haldi Ceremony custom_occasion casual outing haldi look"
    scrubbed = engine._scrub_visible_style_text(malformed, query="Haldi")
    lowered = scrubbed.lower()
    assert scrubbed == "Completes the festive kurta look while staying comfortable for rituals."
    assert "custom occasion" not in lowered
    assert "casual outing" not in lowered
    assert scrubbed != engine._DEFAULT_OCCASION_VOICE
