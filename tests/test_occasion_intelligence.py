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


def test_coffee_date_support_rejects_nonfashion_travel_and_cardholder_assets():
    for asset in (
        _asset("Tumbler White", category="accessory", subcategory="tumbler", tags=["tumbler"]),
        _asset("Blue Card Case", category="accessory", subcategory="card_case", tags=["card_case"]),
    ):
        assert not engine._asset_allowed_for_context(
            asset,
            occasion="coffee_date",
            placement="complete",
            target_text="Relaxed Oxford",
        )
        assert not engine._asset_allowed_for_context(
            asset,
            occasion="coffee_date",
            placement="missing",
            target_text="Relaxed Oxford",
        )


def test_nonfestive_support_sanitizer_drops_travel_and_cardholder_assets():
    kept = engine._safe_visual_support_assets(
        [
            _asset("Tumbler White", category="accessory", subcategory="tumbler", tags=["tumbler"]),
            _asset("Blue Card Case", category="accessory", subcategory="card_case", tags=["card_case"]),
            _asset("Hair Straightener", category="accessory", subcategory="straightener", tags=["straightener"]),
            _asset("Blue Beanie", category="accessory", subcategory="beanie", tags=["beanie"]),
            _asset("Tan Shoulder Bag", category="accessory", subcategory="bag", tags=["bag"]),
        ],
        "casual_social",
        "social occasion",
        {"hero_piece": "Relaxed Oxford"},
    )

    assert [item["name"] for item in kept] == ["Tan Shoulder Bag"]


def test_knit_polo_hero_can_use_image_backed_knit_asset():
    knit = _asset("Cream Knit Top", category="top", subcategory="knit", tags=["knit"])
    direction = {"hero_piece": "Cream Knit Polo", "items": ["Cream Knit Polo"]}
    assert engine._asset_allowed_for_context(
        knit,
        occasion="coffee_date",
        placement="hero",
        target_text="Cream Knit Polo",
    )
    assert engine._hero_asset_allowed(knit, direction, "coffee_date")


def test_festive_kurta_hero_can_use_kurta_set_or_kurti_assets():
    kurta_set = _asset(
        "Yellow Sleeveless Kurta Set",
        category="ethnic",
        subcategory="kurtaset",
        tags=["kurta", "haldi"],
    )
    kurti = _asset(
        "Yellow Kurti Top",
        category="ethnic",
        subcategory="kurti",
        tags=["kurti", "haldi"],
    )
    direction = {"hero_piece": "Marigold Yellow Kurta", "items": ["Marigold Yellow Kurta"]}

    assert engine._hero_asset_allowed(kurta_set, direction, "haldi")
    assert engine._hero_asset_allowed(kurti, direction, "haldi")


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
    assert missing["name"] == "Ethnic Footwear"
    assert missing["category"] == "Footwear"
    assert "comfortable for rituals" in missing["reason"]
    assert "image_url" not in missing
    assert "asset_id" not in missing


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
    assert card["missing_piece"]["name"] == "Ethnic Footwear"


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


def test_music_festival_is_music_event_not_indian_festive():
    assert engine._visual_occasion_family("music festival") == "music_event"
    assert engine._visual_occasion_family("concert outfit") == "music_event"
    assert engine._visual_occasion_family("rave night") == "music_event"
    assert engine._visual_occasion_family("wedding guest") == "indian_festive"
    assert engine._visual_occasion_family("sangeet") == "indian_festive"
    assert engine._visual_occasion_family("airport travel") == "travel"
    assert engine._visual_occasion_family("coffee date") == ""


def test_music_event_final_guard_removes_wedding_ethnic_and_formal_leaks():
    directions = [
        {
            "title": "Festive Heritage",
            "archetype": "Festive Heritage",
            "direction_name": "Festive Heritage",
            "hero_piece": "Crisp Oxford Shirt",
            "items": ["Crisp Oxford Shirt", "Leather Belt", "Cardholder"],
            "pieces": ["Crisp Oxford Shirt", "Leather Belt", "Cardholder"],
            "complete_the_look": [
                {"name": "Sangeet Statement Mojari", "category": "footwear"},
                {"name": "Canvas Tote", "category": "bag"},
            ],
        },
        {
            "title": "Wedding Day Ease",
            "archetype": "Sangeet Statement",
            "direction_name": "Wedding Day Ease",
            "hero_piece": "Ivory Kurta",
            "items": ["Ivory Kurta", "Churidar", "Mojari"],
            "pieces": ["Ivory Kurta", "Churidar", "Mojari"],
        },
    ]
    guarded = engine._music_event_visible_guard(
        directions,
        occasion="music festival",
        query="music festival",
    )
    blob = " ".join(
        str(value)
        for direction in guarded
        for value in (
            direction.get("title"),
            direction.get("archetype"),
            direction.get("direction_name"),
            direction.get("hero_piece"),
            " ".join(direction.get("items") or []),
            " ".join(
                item.get("name", "") if isinstance(item, dict) else str(item)
                for item in (direction.get("complete_the_look") or [])
            ),
        )
    ).lower()
    for blocked in (
        "wedding day ease",
        "sangeet statement",
        "festive heritage",
        "oxford shirt",
        "kurta",
        "bandhgala",
        "mojari",
        "leather belt",
        "cardholder",
    ):
        assert blocked not in blob
    assert all(direction["direction_name"] in engine._MUSIC_EVENT_DIRECTION_NAMES for direction in guarded)
    assert "canvas tote" in blob


def test_music_event_final_response_boundary_rewrites_bad_visual_directions():
    response = engine._build_response(
        query="music festival",
        mode=engine.VISUAL_INSPIRATION,
        category="music festival",
        tone=None,
        formality=None,
        occasion="music festival",
        confidence=0.9,
        user_profile={},
        context={"occasion": "music festival"},
        ai_payload={
            "mode": engine.VISUAL_INSPIRATION,
            "occasion": "music festival",
            "visual_directions": [
                {
                    "title": "Festive Heritage",
                    "archetype": "Festive Heritage",
                    "hero_piece": "Crisp Oxford Shirt",
                    "items": ["Crisp Oxford Shirt", "Leather Belt"],
                    "pieces": ["Crisp Oxford Shirt", "Leather Belt"],
                },
                {
                    "title": "Wedding Day Ease",
                    "archetype": "Sangeet Statement",
                    "hero_piece": "Ivory Kurta",
                    "items": ["Ivory Kurta", "Mojari"],
                    "pieces": ["Ivory Kurta", "Mojari"],
                },
                {
                    "title": "Creative Casual",
                    "archetype": "Creative Casual",
                    "hero_piece": "Printed Shirt",
                    "items": ["Printed Shirt", "Relaxed Trousers"],
                    "pieces": ["Printed Shirt", "Relaxed Trousers"],
                },
            ],
        },
    )
    blob = str(response).lower()
    for blocked in (
        "wedding day ease",
        "sangeet statement",
        "festive heritage",
        "oxford shirt",
        "kurta",
        "bandhgala",
        "mojari",
    ):
        assert blocked not in blob
    names = [d["direction_name"] for d in response["visual_directions"]]
    assert set(names).issubset(set(engine._MUSIC_EVENT_DIRECTION_NAMES))


def test_music_event_asset_gate_blocks_oxford_and_ethnic_but_keeps_airport_wedding_sangeet():
    assert not engine._asset_allowed_for_context(
        _asset("Crisp Oxford Shirt", category="top", subcategory="shirt"),
        occasion="music festival",
        placement="hero",
        target_text="music festival",
    )
    assert not engine._asset_allowed_for_context(
        _asset("Ivory Kurta", category="ethnic", subcategory="kurta"),
        occasion="concert",
        placement="hero",
        target_text="Outdoor Concert",
    )
    assert engine._asset_allowed_for_context(
        _asset("Clean White Sneakers", category="footwear", subcategory="sneaker"),
        occasion="airport travel",
        placement="complete",
        target_text="travel layers",
    )
    assert engine._asset_allowed_for_context(
        _asset("Ivory Kurta", category="ethnic", subcategory="kurta"),
        occasion="wedding guest",
        placement="hero",
        target_text="Festive Heritage",
    )
    assert engine._asset_allowed_for_context(
        _asset("Maroon Mojari", category="footwear", subcategory="mojari", tags=["ethnic"]),
        occasion="sangeet",
        placement="complete",
        target_text="Festive Heritage",
    )
    assert engine._music_event_visible_guard(
        [{"title": "Modern Romantic", "archetype": "Modern Romantic", "direction_name": "Modern Romantic"}],
        occasion="coffee date",
        query="coffee date",
    )[0]["direction_name"] == "Modern Romantic"


def test_festive_copy_scrubber_removes_malformed_internal_phrase():
    malformed = "wedding Haldi Ceremony custom_occasion casual outing haldi look"
    scrubbed = engine._scrub_visible_style_text(malformed, query="Haldi")
    lowered = scrubbed.lower()
    assert scrubbed == "Completes the festive kurta look while staying comfortable for rituals."
    assert "custom occasion" not in lowered
    assert "casual outing" not in lowered
    assert scrubbed != engine._DEFAULT_OCCASION_VOICE


def test_haldi_support_blocks_flip_flops_boots_relaxed_shirt_aviators_and_wallet():
    blocked_assets = (
        _asset("Leather Flip Flops", category="footwear", subcategory="flip_flops"),
        _asset("Brown Suede Boots", category="footwear", subcategory="boots"),
        _asset("Relaxed Linen Shirt", category="top", subcategory="shirt"),
        _asset("Gold Aviators", category="accessory", subcategory="aviators"),
        _asset("Western Wallet", category="accessory", subcategory="wallet"),
    )
    for asset in blocked_assets:
        assert not engine._asset_allowed_for_context(
            asset,
            occasion="Haldi wedding",
            placement="complete",
            target_text="Festive Heritage",
        )


def test_haldi_returns_fewer_support_items_when_only_bad_options_exist(monkeypatch):
    monkeypatch.setattr(engine, "_style_asset_rows", lambda limit=120: [])
    directions = engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Festive Heritage",
                "title": "Festive Heritage",
                "hero_piece": "Yellow Kurta",
                "items": ["Yellow Kurta", "Ivory Churidar"],
                "complete_the_look": [
                    {"name": "Leather Flip Flops", "category": "footwear"},
                    {"name": "Brown Suede Boots", "category": "footwear"},
                    {"name": "Gold Aviators", "category": "accessory"},
                ],
            }
        ],
        occasion="Haldi wedding",
        target_gender="male",
    )
    assert directions[0]["complete_the_look"] == []


def test_clean_direction_title_repairs_celebn_and_internal_occasion_labels():
    assert engine._clean_direction_title("Vibrant Celebn") == "Vibrant Celebration"
    assert engine._clean_direction_title("Celebn Kurta") == "Celebration Kurta"
    assert (
        engine._clean_direction_title("wedding Haldi Ceremony custom occasion")
        == "Haldi Ceremony"
    )
    assert "custom" not in engine._clean_direction_title("Festive custom_occasion").lower()


def test_editorial_polish_and_cover_clean_all_direction_titles():
    polished = engine._apply_editorial_polish(
        [
            {
                "title": "Celebn Kurta",
                "archetype": "Vibrant Celebn",
                "hero_piece": "Yellow Kurta",
                "items": ["Yellow Kurta", "Ivory Churidar"],
            }
        ],
        occasion="Haldi",
        wardrobe_items=None,
        context_text="Haldi",
    )
    card = polished[0]
    assert card["title"] == "Celebration Kurta"
    assert card["archetype"] == "Vibrant Celebration"
    assert card["direction_name"] == "Vibrant Celebration"
    cover = engine._build_editorial_cover(polished, occasion="Haldi")
    assert cover["direction_name"] == "Vibrant Celebration"


def test_cousin_wedding_suppresses_emergency_support_leaks():
    items = [
        _asset("Nike Running Shoes", category="footwear", subcategory="sneaker"),
        _asset("Blue Duffle Bag", category="accessory", subcategory="duffle_bag"),
        _asset("Blue Cap", category="accessory", subcategory="cap"),
        _asset("Suede Blue Shoe", category="footwear", subcategory="shoe"),
        _asset("Gold Festive Brooch", category="accessory", subcategory="brooch"),
    ]
    safe = engine._safe_visual_support_assets(
        items,
        "indian_festive",
        "cousin wedding",
        {"hero_piece": "Ivory Kurta"},
    )
    assert [item["name"] for item in safe] == ["Gold Festive Brooch"]


def test_haldi_strict_support_allowlist_drops_weak_accessories():
    items = [
        _asset("Leather Flip Flops", category="footwear", subcategory="flip_flops"),
        _asset("Gold Aviators", category="accessory", subcategory="aviators"),
        _asset("Western Wallet", category="accessory", subcategory="wallet"),
    ]
    assert engine._safe_visual_support_assets(
        items,
        "indian_festive",
        "Haldi",
        {"hero_piece": "Yellow Kurta"},
    ) == []


def test_unsafe_festive_missing_piece_is_always_text_only():
    missing = engine._enrich_missing_piece_with_asset(
        {
            "name": "Nike Running Shoes",
            "category": "footwear",
            "reason": "Adds comfort.",
            "image_url": "https://cdn/nike.png",
        },
        assets=[
            _asset(
                "Maroon Mojari",
                category="footwear",
                subcategory="mojari",
                tags=["ethnic"],
            )
        ],
        occasion="cousin wedding",
        target_gender="male",
    )
    assert missing["name"] == "Ethnic Footwear"
    assert missing["category"] == "Footwear"
    assert missing["reason"] == (
        "Completes the festive kurta look while staying comfortable for rituals."
    )
    assert "image_url" not in missing
    assert "asset_id" not in missing


def test_unsafe_festive_missing_piece_stays_text_only_without_safe_image():
    missing = engine._enrich_missing_piece_with_asset(
        {
            "name": "Blue Duffle Bag",
            "category": "accessory",
            "reason": "Adds storage.",
            "image_url": "https://cdn/duffle.png",
        },
        assets=[
            _asset("Nike Running Shoes", category="footwear", subcategory="sneaker")
        ],
        occasion="nikah wedding",
        target_gender="male",
    )
    assert missing == {
        "name": "Ethnic Footwear",
        "category": "Footwear",
        "reason": "Completes the festive kurta look while staying comfortable for rituals.",
        "unlocks": ["Festive styling"],
    }


def test_airport_strict_support_suppression_is_not_applied():
    items = [
        _asset("Clean Sneakers", category="footwear", subcategory="sneaker"),
        _asset("Travel Duffle Bag", category="accessory", subcategory="duffle_bag"),
    ]
    safe = engine._safe_visual_support_assets(
        items,
        "travel",
        "airport outfit",
        {"hero_piece": "Travel Overshirt"},
    )
    assert [item["name"] for item in safe] == ["Clean Sneakers", "Travel Duffle Bag"]


def test_direction_title_cleans_repeated_wedding_social_labels():
    assert engine._clean_direction_title(
        "wedding social occasion social_occasion cousin wedding"
    ) == "Cousin Wedding"


def test_festive_kurta_hero_rejects_nehru_jacket_image():
    direction = {
        "hero_piece": "Marigold Yellow Cotton Kurta",
        "items": ["Marigold Yellow Cotton Kurta", "Ivory Churidar"],
    }
    nehru = _asset(
        "Maroon Nehru Jacket",
        category="ethnic",
        subcategory="nehru_jacket",
        tags=["festive"],
    )
    kurta = _asset(
        "Marigold Yellow Kurta",
        category="ethnic",
        subcategory="kurta",
        tags=["festive", "haldi"],
    )
    assert not engine._hero_asset_allowed(nehru, direction, "Haldi wedding")
    assert engine._hero_asset_allowed(kurta, direction, "Haldi wedding")


def test_preattached_wrong_festive_hero_image_is_replaced(monkeypatch):
    nehru = _asset(
        "Maroon Nehru Jacket",
        category="ethnic",
        subcategory="nehru_jacket",
        tags=["festive"],
    )
    kurta = _asset(
        "Marigold Yellow Kurta",
        category="ethnic",
        subcategory="kurta",
        tags=["festive", "haldi"],
    )
    monkeypatch.setattr(engine, "_style_asset_rows", lambda limit=120: [nehru, kurta])
    directions = engine._enrich_visual_directions_with_assets(
        [
            {
                "title": "Vibrant Classic",
                "archetype": "Festive Heritage",
                "hero_piece": "Marigold Yellow Cotton Kurta",
                "items": ["Marigold Yellow Cotton Kurta", "Ivory Churidar"],
                "image_url": nehru["image_url"],
                "asset_id": nehru["asset_id"],
            }
        ],
        occasion="Haldi wedding",
        target_gender="male",
    )
    assert directions[0]["asset_id"] == kurta["asset_id"]
    assert directions[0]["image_url"] == kurta["image_url"]


def test_festive_board_gate_replaces_western_casual_directions():
    leaking = [
        {
            "title": "Soft Polish",
            "hero_piece": "Fine-Gauge Knit Polo",
            "items": ["Fine-Gauge Knit Polo", "Straight Trousers", "Black Loafers"],
        },
        {
            "title": "Relaxed Oxford",
            "hero_piece": "Oxford Shirt",
            "items": ["Oxford Shirt", "Chinos", "White Sneakers"],
        },
        {
            "title": "Easy Classic",
            "hero_piece": "Basic Tee",
            "items": ["Basic Tee", "Denim Jeans", "Running Shoes"],
        },
    ]
    blocked = (
        "oxford",
        "polo",
        "basic tee",
        "hoodie",
        "sweatshirt",
        "sneaker",
        "running shoe",
        "loafer",
        "jeans",
        "denim",
    )
    for occasion in ("haldi", "mehendi", "wedding_guest", "festival"):
        guarded = engine._enforce_festive_visual_directions(
            leaking,
            occasion=occasion,
        )
        assert len(guarded) == 3
        for direction in guarded:
            hero = direction["hero_piece"].lower()
            blob = " ".join(direction["items"]).lower()
            assert any(
                term in hero
                for term in ("kurta", "nehru jacket", "bandhgala", "sherwani")
            )
            assert not any(term in f"{hero} {blob}" for term in blocked)
            assert engine._festive_direction_compatibility(direction) == 1.0


def test_festive_board_gate_preserves_valid_ethnic_direction():
    valid = {
        "title": "Wedding Day Ease",
        "hero_piece": "Ivory Kurta Set",
        "items": ["Ivory Kurta Set", "Cream Churidar", "Brown Mojaris"],
    }
    guarded = engine._enforce_festive_visual_directions(
        [valid],
        occasion="ethnic_event",
    )
    assert guarded[0] is valid
    assert guarded[0]["hero_piece"] == "Ivory Kurta Set"
