from services import style_reasoning_engine
from services.style_flow_service import finalize_style_cards


def _item(item_id, name, role, color=""):
    return {
        "id": item_id,
        "name": name,
        "role": role,
        "color": color,
        "image_url": f"https://x/{item_id}.png",
    }


def _card(*items, score=80):
    return {
        "id": "card",
        "items": list(items),
        "score": score,
    }


def test_visual_directions_get_assets_and_complete_the_look(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "dark-denim",
                "name": "Dark Wash Denim",
                "category": "bottom",
                "image_url": "https://cdn.test/dark-denim.png",
                "colors": ["navy"],
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["denim"],
                "gender": "unisex",
                "status": "active",
            },
            {
                "asset_id": "structured-tote",
                "name": "Structured Tote",
                "category": "accessory",
                "image_url": "https://cdn.test/tote.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["bag"],
                "gender": "unisex",
                "status": "active",
            },
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Weekend Ease",
                "hero_piece": "Dark Wash Denim",
                "items": ["Dark Wash Denim", "White Shirt"],
                "colors": ["navy", "white"],
            }
        ],
        occasion="coffee date",
    )

    assert directions[0]["image_url"] == "https://cdn.test/dark-denim.png"
    assert directions[0]["complete_the_look"]
    assert directions[0]["complete_the_look"][0]["image_url"] == "https://cdn.test/tote.png"


def test_style_asset_metadata_normalizer_accepts_common_aliases():
    asset = style_reasoning_engine._normalize_style_asset(
        {
            "id": "asset-1",
            "name": "Brown Suede Loafer",
            "category": "footwear",
            "sub_category": "loafers",
            "imageUrl": "https://cdn.test/loafer.png",
            "gender": "men",
            "colors": "brown|tan",
            "archetypes": "Refined Weekend,Modern Professional",
            "occasions": "coffee date|office",
        }
    )

    assert asset["asset_id"] == "asset-1"
    assert asset["image_url"] == "https://cdn.test/loafer.png"
    assert asset["subcategory"] == "loafers"
    assert asset["gender"] == "male"
    assert asset["colors"] == ["brown", "tan"]
    assert "refined weekend" in asset["archetypes"]


def test_visual_assets_filter_complete_the_look_by_profile_gender(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "gold-earrings",
                "name": "Gold Hoop Earrings",
                "category": "accessory",
                "image_url": "https://cdn.test/earrings.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["jewelry", "coffee"],
                "gender": "female",
                "status": "active",
            },
            {
                "asset_id": "clean-watch",
                "name": "Clean Leather Watch",
                "category": "accessory",
                "image_url": "https://cdn.test/watch.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["watch", "coffee"],
                "gender": "male",
                "status": "active",
            },
            {
                "asset_id": "canvas-sling",
                "name": "Canvas Sling",
                "category": "accessory",
                "image_url": "https://cdn.test/sling.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["bag", "coffee"],
                "gender": "unisex",
                "status": "active",
            },
        ],
    )

    target_gender = style_reasoning_engine._resolve_asset_gender(
        query="show visual inspiration for coffee date",
        user_profile={"gender": "male"},
    )
    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Weekend Ease",
                "items": ["White Shirt", "Dark Denim"],
                "colors": ["white", "navy"],
            }
        ],
        occasion="coffee date",
        target_gender=target_gender,
    )

    names = [item["name"] for item in directions[0]["complete_the_look"]]
    assert "Clean Leather Watch" in names
    assert "Canvas Sling" in names
    assert "Gold Hoop Earrings" not in names


def test_prompt_gender_overrides_profile_for_assets(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "gold-earrings",
                "name": "Gold Hoop Earrings",
                "category": "accessory",
                "image_url": "https://cdn.test/earrings.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["jewelry", "coffee"],
                "gender": "female",
                "status": "active",
            },
            {
                "asset_id": "clean-watch",
                "name": "Clean Leather Watch",
                "category": "accessory",
                "image_url": "https://cdn.test/watch.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["watch", "coffee"],
                "gender": "male",
                "status": "active",
            },
        ],
    )

    target_gender = style_reasoning_engine._resolve_asset_gender(
        query="suggest women's coffee date outfit",
        user_profile={"gender": "male"},
    )
    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Weekend Ease",
                "items": ["White Shirt", "Dark Denim"],
                "colors": ["white", "navy"],
            }
        ],
        occasion="coffee date",
        target_gender=target_gender,
    )

    names = [item["name"] for item in directions[0]["complete_the_look"]]
    assert "Gold Hoop Earrings" in names
    assert "Clean Leather Watch" not in names


def test_female_profile_allows_female_assets(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "gold-earrings",
                "name": "Gold Hoop Earrings",
                "category": "accessory",
                "image_url": "https://cdn.test/earrings.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["jewelry", "coffee"],
                "gender": "female",
                "status": "active",
            },
            {
                "asset_id": "clean-watch",
                "name": "Clean Leather Watch",
                "category": "accessory",
                "image_url": "https://cdn.test/watch.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["watch", "coffee"],
                "gender": "male",
                "status": "active",
            },
        ],
    )

    target_gender = style_reasoning_engine._resolve_asset_gender(
        query="show visual inspiration for coffee date",
        user_profile={"gender": "female"},
    )
    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Weekend Ease",
                "items": ["White Shirt", "Dark Denim"],
                "colors": ["white", "navy"],
            }
        ],
        occasion="coffee date",
        target_gender=target_gender,
    )

    names = [item["name"] for item in directions[0]["complete_the_look"]]
    assert "Gold Hoop Earrings" in names
    assert "Clean Leather Watch" not in names


def test_missing_gender_metadata_does_not_leak_feminine_accessories(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "necklace-null-gender",
                "name": "Necklace 08",
                "category": "accessory",
                "image_url": "https://cdn.test/necklace.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["necklace", "jewelry"],
                "status": "active",
            },
            {
                "asset_id": "canvas-sling",
                "name": "Canvas Sling",
                "category": "accessory",
                "image_url": "https://cdn.test/sling.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["bag"],
                "gender": "unisex",
                "status": "active",
            },
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Weekend Ease",
                "items": ["White Shirt", "Dark Denim"],
                "colors": ["white", "navy"],
            }
        ],
        occasion="coffee date",
        target_gender="male",
    )

    names = [item["name"] for item in directions[0]["complete_the_look"]]
    assert names == ["Canvas Sling"]


def test_male_visual_direction_filters_generated_hero_and_components(monkeypatch):
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda limit=120: [])

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Weekend Ease",
                "hero_piece": "Floral Dress",
                "items": ["Floral Dress", "Gold Earrings", "White Heels"],
                "colors": ["white", "gold"],
            }
        ],
        occasion="coffee date",
        target_gender="male",
    )

    blob = " ".join(
        [
            directions[0]["hero_piece"],
            " ".join(directions[0]["items"]),
            " ".join(directions[0]["pieces"]),
        ]
    ).lower()
    for blocked in ["dress", "earrings", "heels"]:
        assert blocked not in blob
    assert directions[0]["items"] == ["clean shirt", "tailored trouser", "polished footwear"]


def test_male_missing_piece_filters_gender_incompatible_items(monkeypatch):
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda limit=120: [])

    filtered = style_reasoning_engine._enrich_missing_piece_with_asset(
        {
            "name": "Black Midi Dress",
            "category": "dress",
            "reason": "Would complete the look.",
            "unlocks": ["Coffee Date"],
        },
        occasion="coffee date",
        target_gender="male",
    )

    assert filtered is None


def test_explicit_womens_prompt_allows_gendered_visual_direction(monkeypatch):
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda limit=120: [])

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Modern Romantic",
                "title": "Soft Coffee Date",
                "hero_piece": "Floral Dress",
                "items": ["Floral Dress", "Gold Earrings", "White Heels"],
                "colors": ["white", "gold"],
            }
        ],
        occasion="women's coffee date",
        target_gender="female",
        allow_feminine_accessory=True,
    )

    blob = " ".join([directions[0]["hero_piece"], " ".join(directions[0]["items"])]).lower()
    assert "dress" in blob
    assert "earrings" in blob


def test_unknown_gender_uses_neutral_assets_only(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "gold-earrings",
                "name": "Gold Hoop Earrings",
                "category": "accessory",
                "image_url": "https://cdn.test/earrings.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "gender": "female",
                "status": "active",
            },
            {
                "asset_id": "canvas-sling",
                "name": "Canvas Sling",
                "category": "accessory",
                "image_url": "https://cdn.test/sling.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "gender": "unisex",
                "status": "active",
            },
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Weekend Ease",
                "items": ["White Shirt", "Dark Denim"],
                "colors": ["white", "navy"],
            }
        ],
        occasion="coffee date",
        target_gender="unknown",
    )

    names = [item["name"] for item in directions[0]["complete_the_look"]]
    assert names == ["Canvas Sling"]


def test_complete_the_look_fallback_uses_product_like_labels(monkeypatch):
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda limit=120: [])

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Weekend Ease",
                "items": ["White Shirt", "Dark Denim"],
                "colors": ["white", "navy"],
            }
        ],
        occasion="coffee date",
        target_gender="male",
    )

    names = [item["name"] for item in directions[0]["complete_the_look"]]
    assert names == ["Brushed Steel Watch", "Dark Brown Leather Belt", "Canvas Tote"]
    assert "Quiet Accent" not in names


def test_visual_card_copy_is_rewritten_from_components(monkeypatch):
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda limit=120: [])

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Smart Casual Edge",
                "title": "Relaxed Polish",
                "hero_piece": "Unlined Navy Blazer",
                "items": ["Unlined Navy Blazer", "White Crew-Neck T-Shirt", "Dark Wash Jeans", "Cognac Loafers"],
                "description": "A silk blouse, midi skirt, and heels keep this soft.",
                "why_it_works": "The blouse and skirt bring feminine movement.",
                "styling_tip": "Let the skirt sit below the knee.",
            }
        ],
        occasion="coffee date",
        target_gender="male",
    )

    card = directions[0]
    combined = " ".join([card["description"], card["why_it_works"], card["styling_tip"]]).lower()
    for blocked in ["blouse", "skirt", "heels"]:
        assert blocked not in combined
    assert "unlined navy blazer" in card["hero_piece"].lower()
    assert "unlined navy blazer" in " ".join(card["items"]).lower()


def test_visual_card_missing_piece_does_not_duplicate_hero(monkeypatch):
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda limit=120: [])

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Knit Ease",
                "hero_piece": "Fine-Gauge Knit Sweater",
                "items": ["Fine-Gauge Knit Sweater", "Khaki Trousers", "Suede Loafers"],
                "missing_piece": {
                    "name": "Soft Knit Sweater",
                    "category": "top",
                    "reason": "Adds softness.",
                    "unlocks": ["Refined Weekend"],
                },
            }
        ],
        occasion="coffee date",
        target_gender="male",
    )

    missing = directions[0]["missing_piece"]
    assert missing["name"] != "Soft Knit Sweater"
    assert "knit sweater" not in missing["name"].lower()


def test_top_level_missing_piece_dedupes_against_visual_directions():
    directions = [
        {
            "title": "Knit Ease",
            "hero_piece": "Fine-Gauge Knit Sweater",
            "items": ["Fine-Gauge Knit Sweater", "Khaki Trousers", "Suede Loafers"],
        }
    ]

    missing = style_reasoning_engine._dedupe_missing_piece_against_directions(
        {
            "name": "Soft Knit Sweater",
            "category": "top",
            "reason": "Adds softness.",
            "unlocks": ["Coffee Date"],
        },
        directions,
        occasion="coffee date",
        target_gender="male",
    )

    assert missing is not None
    assert missing["name"] != "Soft Knit Sweater"
    assert "knit sweater" not in missing["name"].lower()


def test_complete_the_look_varies_across_three_cards(monkeypatch):
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda limit=120: [])

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Smart Casual Edge",
                "title": "Blazer Denim",
                "hero_piece": "Navy Blazer",
                "items": ["Navy Blazer", "White T-Shirt", "Dark Wash Jeans", "Cognac Loafers"],
            },
            {
                "archetype": "Refined Weekend",
                "title": "Knit Ease",
                "hero_piece": "Fine-Gauge Knit Sweater",
                "items": ["Fine-Gauge Knit Sweater", "Khaki Trousers", "Suede Loafers"],
            },
            {
                "archetype": "Power Casual",
                "title": "Tee Layer",
                "hero_piece": "White T-Shirt",
                "items": ["White T-Shirt", "Navy Chinos", "Clean Sneakers"],
            },
        ],
        occasion="coffee date",
        target_gender="male",
    )

    sets = [tuple(item["name"] for item in card["complete_the_look"]) for card in directions]
    assert len(set(sets)) == 3


def test_complete_the_look_varies_by_archetype(monkeypatch):
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda limit=120: [])

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Quiet Luxury",
                "title": "Soft Polish",
                "hero_piece": "Camel Blazer",
                "items": ["Camel Blazer", "White Shirt", "Stone Trousers"],
            },
            {
                "archetype": "Startup Founder",
                "title": "Founder Ease",
                "hero_piece": "Navy Overshirt",
                "items": ["Navy Overshirt", "White T-Shirt", "Dark Denim"],
            },
            {
                "archetype": "Creative Agency",
                "title": "Studio Casual",
                "hero_piece": "Textured Knit Polo",
                "items": ["Textured Knit Polo", "Khaki Trousers", "Clean Sneakers"],
            },
        ],
        occasion="startup office",
        target_gender="male",
    )

    sets = [tuple(item["name"] for item in card["complete_the_look"]) for card in directions]
    assert len(set(sets)) == 3
    assert "Leather-Strap Watch" in sets[0]
    assert "Tech Backpack" in sets[1]
    assert "Fashion Sneaker" in sets[2]


def test_generic_visual_diversity_guard_replaces_repeated_blazer_formula():
    rows = [
        {
            "title": "Blazer One",
            "hero_piece": "Navy Blazer",
            "items": ["Navy Blazer", "White Shirt", "Dark Trousers", "Loafers"],
            "palette": ["navy", "white", "black"],
        },
        {
            "title": "Blazer Two",
            "hero_piece": "Blue Blazer",
            "items": ["Blue Blazer", "White Shirt", "Black Trousers", "Loafers"],
            "palette": ["navy", "white", "black"],
        },
        {
            "title": "Blazer Three",
            "hero_piece": "Structured Blazer",
            "items": ["Structured Blazer", "White Shirt", "Formal Trousers", "Loafers"],
            "palette": ["navy", "white", "black"],
        },
    ]

    directions = style_reasoning_engine._normalize_visual_directions(
        rows,
        "visual_inspiration",
        "coffee date",
    )

    formulas = [style_reasoning_engine._direction_formula_signature(card) for card in directions]
    heroes = [card["hero_piece"].lower() for card in directions]
    assert len(set(formulas)) == 3
    assert len(set(heroes)) == 3
    assert any("soft" in hero or "shirt" in hero or "utility" in hero for hero in heroes[1:])


def test_formula_diversity_guard():
    rows = [
        {
            "title": "Blazer One",
            "hero_piece": "Navy Blazer",
            "items": ["Navy Blazer", "White Shirt", "Dark Trousers", "Loafers"],
            "palette": ["navy", "white", "black"],
        },
        {
            "title": "Blazer Two",
            "hero_piece": "Blue Blazer",
            "items": ["Blue Blazer", "White Shirt", "Black Trousers", "Loafers"],
            "palette": ["navy", "white", "black"],
        },
        {
            "title": "Blazer Three",
            "hero_piece": "Structured Blazer",
            "items": ["Structured Blazer", "White Shirt", "Formal Trousers", "Loafers"],
            "palette": ["navy", "white", "black"],
        },
    ]

    directions = style_reasoning_engine._normalize_visual_directions(
        rows,
        "visual_inspiration",
        "startup office",
    )

    formulas = [style_reasoning_engine._direction_formula_signature(card) for card in directions]
    assert len(set(formulas)) == 3
    assert formulas[0] == ("blazer", "trouser")
    assert ("knit", "tailored_trouser") in formulas
    assert ("shirt", "wide_leg") in formulas


def test_missing_piece_names_not_generic(monkeypatch):
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda limit=120: [])

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Overshirt Direction",
                "hero_piece": "White Oxford Shirt",
                "items": ["White Oxford Shirt", "Stone Chinos", "Dark Brown Penny Loafers"],
                "missing_piece": {
                    "name": "Clean Overshirt",
                    "category": "outerwear",
                    "reason": "Adds a layer.",
                },
            }
        ],
        occasion="coffee date",
        target_gender="male",
    )

    name = directions[0]["missing_piece"]["name"]
    lowered = name.lower()
    assert name == "Olive Cotton Overshirt"
    assert not any(lowered == word for word in ["clean", "simple", "basic", "neutral", "minimal"])


def test_missing_piece_reason_references_board(monkeypatch):
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda limit=120: [])

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Clean Minimal",
                "title": "Oxford Ease",
                "hero_piece": "White Oxford Shirt",
                "items": ["White Oxford Shirt", "Stone Wide-Leg Trouser", "Black Leather Loafers"],
                "missing_piece": {
                    "name": "Neutral Blazer",
                    "category": "outerwear",
                    "reason": "A blazer adds structure.",
                },
            }
        ],
        occasion="startup office",
        target_gender="male",
    )

    reason = directions[0]["missing_piece"]["reason"].lower()
    assert "startup office" in reason
    assert "wide-leg trouser" in reason
    assert "clean-shirt direction" in reason


def test_generic_diversity_guard_is_gender_independent():
    male_like = {
        "title": "Structured",
        "hero_piece": "Navy Blazer",
        "items": ["Navy Blazer", "White Shirt", "Dark Trousers", "Loafers"],
        "palette": ["navy", "white", "black"],
    }
    female_like = {
        "title": "Structured Again",
        "hero_piece": "Cream Blazer",
        "items": ["Cream Blazer", "Silk Top", "Tailored Bottom", "Polished Footwear"],
        "palette": ["navy", "white", "black"],
    }

    duplicate, reason = style_reasoning_engine._directions_too_similar(female_like, male_like)

    assert duplicate is True
    assert reason in {"same_hero", "same_formula", "same_silhouette_palette", "same_palette_hero_role"}


def test_internal_reasoning_language_is_scrubbed():
    text = (
        "For a coffee date, the social outcome is connection. "
        "The styling strategy is to convey ease without looking overdressed."
    )

    scrubbed = style_reasoning_engine._compact_reasoning(text)

    lowered = scrubbed.lower()
    assert "social outcome" not in lowered
    assert "styling strategy" not in lowered
    assert "signal" not in lowered


def test_visible_placeholder_terms_are_scrubbed():
    payload = {
        "description": "clean shirt with minimal straight bottom and simple footwear",
        "items": ["clean shirt", "minimal straight bottom", "simple footwear"],
        "reason": "sensitive_occasion calls for clean supporting pieces",
    }

    scrubbed = style_reasoning_engine._scrub_visible_style_payload(payload)
    blob = " ".join(
        [
            scrubbed["description"],
            " ".join(scrubbed["items"]),
            scrubbed["reason"],
        ]
    ).lower()

    for placeholder in [
        "clean shirt",
        "minimal straight bottom",
        "simple footwear",
        "sensitive_occasion",
        "clean supporting pieces",
    ]:
        assert placeholder not in blob
    assert "crisp shirt" in blob
    assert "polished shoes" in blob


def test_internal_occasion_labels_are_scrubbed_with_query_context():
    payload = {
        "title": "crisp shirt Base",
        "subtitle": "social_occasion",
        "description": "custom_occasion and hybrid_occasion need balance.",
        "why_it_works": "work_occasion, travel_occasion, and sensitive_occasion are internal labels.",
    }

    scrubbed = style_reasoning_engine._scrub_visible_style_payload(
        payload,
        query="show visual inspiration for coffee date",
    )
    blob = " ".join(str(value) for value in scrubbed.values()).lower()

    for leaked in [
        "social_occasion",
        "custom_occasion",
        "hybrid_occasion",
        "work_occasion",
        "travel_occasion",
        "sensitive_occasion",
        "crisp shirt base",
    ]:
        assert leaked not in blob
    assert scrubbed["title"] == "Classic Tailoring"
    assert "coffee date" in blob
    assert "casual outing" in blob
    assert "work-to-social occasion" in blob


def test_hero_asset_category_matches_hero_piece(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "hoodie",
                "name": "Grey Hoodie",
                "category": "top",
                "subcategory": "hoodie",
                "image_url": "https://cdn.test/hoodie.png",
                "archetypes": ["Classic Tailoring"],
                "occasions": ["coffee date"],
                "tags": ["hoodie"],
                "gender": "unisex",
                "status": "active",
            },
            {
                "asset_id": "shoe",
                "name": "Brown Loafer",
                "category": "footwear",
                "subcategory": "loafer",
                "image_url": "https://cdn.test/shoe.png",
                "archetypes": ["Classic Tailoring"],
                "occasions": ["coffee date"],
                "tags": ["loafer"],
                "gender": "unisex",
                "status": "active",
            },
            {
                "asset_id": "oxford",
                "name": "White Oxford Shirt",
                "category": "top",
                "subcategory": "oxford shirt",
                "image_url": "https://cdn.test/oxford.png",
                "archetypes": ["Classic Tailoring"],
                "occasions": ["coffee date"],
                "tags": ["shirt", "oxford"],
                "gender": "unisex",
                "status": "active",
            },
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Classic Tailoring",
                "title": "Classic Tailoring",
                "hero_piece": "White Oxford Shirt",
                "items": ["White Oxford Shirt", "Stone Trouser", "Loafers"],
                "colors": ["white", "stone"],
            }
        ],
        occasion="coffee date",
        target_gender="unknown",
    )

    assert directions[0]["image_url"] == "https://cdn.test/oxford.png"


def test_sweater_hero_rejects_shoe_belt_jeans_and_accessory_images(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "shoe",
                "name": "Brown Loafer",
                "category": "footwear",
                "subcategory": "loafer",
                "image_url": "https://cdn.test/shoe.png",
                "gender": "unisex",
                "status": "active",
            },
            {
                "asset_id": "belt",
                "name": "Leather Belt 04",
                "category": "accessory",
                "subcategory": "belt",
                "image_url": "https://cdn.test/belt.png",
                "gender": "unisex",
                "status": "active",
            },
            {
                "asset_id": "jeans",
                "name": "Dark Wash Jeans",
                "category": "bottom",
                "subcategory": "jeans",
                "image_url": "https://cdn.test/jeans.png",
                "gender": "unisex",
                "status": "active",
            },
            {
                "asset_id": "watch",
                "name": "Steel Watch",
                "category": "accessory",
                "subcategory": "watch",
                "image_url": "https://cdn.test/watch.png",
                "gender": "unisex",
                "status": "active",
            },
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Soft Texture",
                "hero_piece": "Fine Gauge Knit Sweater",
                "items": ["Fine Gauge Knit Sweater", "Stone Trouser", "Loafers"],
                "colors": ["cream", "stone"],
            }
        ],
        occasion="coffee date",
        target_gender="unknown",
    )

    assert not directions[0].get("image_url")
    assert not directions[0].get("asset_id")


def test_no_random_fallback_hero_image_when_no_valid_match(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "overshirt",
                "name": "Olive Cotton Overshirt",
                "category": "outerwear",
                "subcategory": "overshirt",
                "image_url": "https://cdn.test/overshirt.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["overshirt"],
                "gender": "unisex",
                "status": "active",
            }
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Soft Texture",
                "hero_piece": "Fine Gauge Knit Sweater",
                "items": ["Fine Gauge Knit Sweater", "Stone Trouser", "Loafers"],
                "colors": ["cream", "stone"],
            }
        ],
        occasion="coffee date",
        target_gender="unknown",
    )

    assert not directions[0].get("image_url")


def test_formal_black_pants_do_not_select_outerwear_asset(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "denim-jacket",
                "name": "H&M Mens Denim Jacket 05",
                "category": "outerwear",
                "subcategory": "denim jacket",
                "image_url": "https://cdn.test/denim-jacket.png",
                "archetypes": ["Modern Professional"],
                "occasions": ["office"],
                "tags": ["jacket", "denim"],
                "gender": "male",
                "status": "active",
            }
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Modern Professional",
                "title": "Formal Base",
                "hero_piece": "Formal Black Pants",
                "items": ["Formal Black Pants", "White Shirt", "Black Shoes"],
                "colors": ["black", "white"],
            }
        ],
        occasion="office",
        target_gender="male",
    )

    assert not directions[0].get("image_url")


def test_gray_blazer_does_not_select_hoodie_asset(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "hoodie",
                "name": "H&M Mens Oversized Hoodie 10",
                "category": "top",
                "subcategory": "hoodie_sweatshirt",
                "image_url": "https://cdn.test/hoodie.png",
                "archetypes": ["Modern Professional"],
                "occasions": ["office"],
                "tags": ["hoodie", "sweatshirt"],
                "gender": "male",
                "status": "active",
            }
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Modern Professional",
                "title": "Blazer Direction",
                "hero_piece": "Gray Blazer",
                "items": ["Gray Blazer", "White Shirt", "Black Pants"],
                "colors": ["gray", "white"],
            }
        ],
        occasion="office",
        target_gender="male",
    )

    assert not directions[0].get("image_url")


def test_navy_twill_overshirt_can_select_overshirt_asset(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "overshirt",
                "name": "Navy Twill Overshirt",
                "category": "outerwear",
                "subcategory": "overshirt",
                "image_url": "https://cdn.test/overshirt.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["overshirt", "jacket"],
                "gender": "male",
                "status": "active",
            }
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Overshirt Direction",
                "hero_piece": "Navy Twill Overshirt",
                "items": ["Navy Twill Overshirt", "White Tee", "Dark Jeans"],
                "colors": ["navy", "white"],
            }
        ],
        occasion="coffee date",
        target_gender="male",
    )

    assert directions[0]["image_url"] == "https://cdn.test/overshirt.png"


def test_blazer_returns_no_image_when_no_compatible_blazer_asset_exists(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "denim-jacket",
                "name": "H&M Mens Denim Jacket 05",
                "category": "outerwear",
                "subcategory": "denim jacket",
                "image_url": "https://cdn.test/jacket.png",
                "archetypes": ["Modern Professional"],
                "occasions": ["office"],
                "tags": ["jacket"],
                "gender": "male",
                "status": "active",
            }
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Modern Professional",
                "title": "Blazer Direction",
                "hero_piece": "Gray Blazer",
                "items": ["Gray Blazer", "White Shirt", "Black Pants"],
                "colors": ["gray", "white"],
            }
        ],
        occasion="office",
        target_gender="male",
    )

    assert not directions[0].get("image_url")


def test_accessory_assets_do_not_stack_duplicate_subcategories(monkeypatch):
    monkeypatch.setattr(
        style_reasoning_engine,
        "_style_asset_rows",
        lambda limit=120: [
            {
                "asset_id": "belt-04",
                "name": "Leather Belt 04",
                "category": "accessory",
                "subcategory": "belt",
                "image_url": "https://cdn.test/belt04.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["belt"],
                "gender": "male",
                "status": "active",
            },
            {
                "asset_id": "belt-05",
                "name": "Leather Belt 05",
                "category": "accessory",
                "subcategory": "belt",
                "image_url": "https://cdn.test/belt05.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["belt"],
                "gender": "male",
                "status": "active",
            },
            {
                "asset_id": "hat-05",
                "name": "Hat 05",
                "category": "accessory",
                "subcategory": "hat",
                "image_url": "https://cdn.test/hat05.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["hat"],
                "gender": "male",
                "status": "active",
            },
            {
                "asset_id": "watch-01",
                "name": "Steel Watch 01",
                "category": "accessory",
                "subcategory": "watch",
                "image_url": "https://cdn.test/watch01.png",
                "archetypes": ["Refined Weekend"],
                "occasions": ["coffee date"],
                "tags": ["watch"],
                "gender": "male",
                "status": "active",
            },
        ],
    )

    directions = style_reasoning_engine._enrich_visual_directions_with_assets(
        [
            {
                "archetype": "Refined Weekend",
                "title": "Weekend Ease",
                "hero_piece": "White Oxford Shirt",
                "items": ["White Oxford Shirt", "Stone Trouser", "Loafers"],
                "colors": ["white", "stone"],
            }
        ],
        occasion="coffee date",
        target_gender="male",
    )

    names = [item["name"] for item in directions[0]["complete_the_look"]]
    assert len([name for name in names if "Belt" in name]) <= 1
    assert len([name for name in names if "Hat" in name]) <= 1
    assert len(names) == len(set(names))


def test_daily_wear_cards_expose_composition_and_style_dna_metadata():
    cards = [
        _card(
            _item("top-1", "Green Tee", "top", "green"),
            _item("bottom-1", "Grey Pants", "bottom", "grey"),
            _item("layer-1", "Textured Overshirt", "outerwear", "stone"),
            _item("shoe-1", "White Sneakers", "footwear", "white"),
        ),
        _card(
            _item("top-2", "Blue Button Down", "top", "blue"),
            _item("bottom-2", "Dark Denim", "bottom", "navy"),
            _item("shoe-2", "Clean Sneakers", "footwear", "white"),
        ),
        _card(
            _item("top-3", "Knit Polo", "top", "cream"),
            _item("bottom-3", "Chinos", "bottom", "tan"),
            _item("shoe-3", "Suede Loafers", "footwear", "brown"),
        ),
    ]

    result = finalize_style_cards(
        cards,
        query="casual day outfit",
        default_limit=3,
        style_identity={"stylePreferences": ["Modern Professional", "Refined Weekend"]},
    )

    assert result[0]["daily_composition_notes"]
    assert "layering" in result[0]["daily_composition_notes"]
    assert result[0]["style_metadata"]["daily_composition_notes"]
    assert "Modern Professional" in result[0]["style_dna_alignment"]
    assert result[0]["style_metadata"]["style_dna_alignment"]
