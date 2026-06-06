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
