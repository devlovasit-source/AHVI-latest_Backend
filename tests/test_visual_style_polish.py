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
    assert names == ["Minimal Steel Watch", "Leather Belt", "Canvas Tote"]
    assert "Quiet Accent" not in names


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
