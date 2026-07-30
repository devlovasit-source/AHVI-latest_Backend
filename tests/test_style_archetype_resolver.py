from services import stylist_knowledge_service as knowledge


def test_resolver_reuses_existing_selector(monkeypatch):
    calls = []

    def fake_select(**kwargs):
        calls.append(kwargs)
        return [
            {
                "name": "Modern Minimal",
                "formality": 6,
                "palette": ["black", "grey"],
                "avoid_items": ["loud logos"],
                "impression": ["clean", "modern"],
                "preferred_items": ["straight trousers"],
                "style_keywords": ["minimal"],
            },
            {
                "name": "Relaxed Weekend",
                "formality": 4,
                "palette": ["cream", "olive"],
                "avoid_items": ["formal suiting"],
                "impression": ["easy", "intentional"],
                "preferred_items": ["denim"],
                "style_keywords": ["relaxed"],
            },
            {
                "name": "Smart Casual Contrast",
                "formality": 5,
                "palette": ["navy", "white"],
                "avoid_items": ["sloppy fits"],
                "impression": ["polished", "approachable"],
                "preferred_items": ["shirt"],
                "style_keywords": ["clean"],
            },
        ]

    monkeypatch.setattr(knowledge, "select_archetypes", fake_select)
    result = knowledge.resolve_style_archetypes(
        {"occasion": "office", "formality": 6},
        {"name": "Teal Shirt", "category": "Tops", "color": "teal"},
        direction_count=3,
    )

    assert len(calls) == 1
    assert calls[0]["anchor"]["name"] == "Teal Shirt"
    assert calls[0]["occasion"] == "office"
    assert [row["direction_title"] for row in result] == [
        "Modern Minimal", "Smart Casual Contrast", "Relaxed Weekend"
    ]
    assert result[0] == {
        "archetype_id": "modern_minimal",
        "direction_title": "Modern Minimal",
        "formality": 6,
        "palette": ["black", "grey"],
        "avoid": ["loud logos"],
        "reasoning_intent": "clean, modern",
    }


def test_teal_shirt_gets_three_differentiated_directions():
    result = knowledge.resolve_style_archetypes(
        {"occasion": "daily"},
        {"name": "Teal Shirt", "category": "Tops", "color": "teal"},
        direction_count=3,
    )

    assert len(result) == 3
    assert len({row["direction_title"] for row in result}) == 3
    assert result[0]["direction_title"] == "Creative Casual"
    assert "teal" in result[0]["palette"]


def test_structured_black_jacket_is_not_vacation_day():
    result = knowledge.resolve_style_archetypes(
        {"occasion": "office"},
        {
            "name": "Structured Black Jacket",
            "category": "Outerwear",
            "color": "black",
        },
        direction_count=3,
    )

    titles = [row["direction_title"] for row in result]
    assert len(set(titles)) == 3
    assert "Vacation Day" not in titles
    assert titles[0] == "Modern Professional"
