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
        "explicit_title": "Effortless Modern Minimal",
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


def test_professional_family_does_not_drift_to_relaxed_casual():
    result = knowledge.resolve_style_archetypes(
        {"occasion": "office"},
        {"name": "Navy Blazer", "category": "Outerwear", "color": "navy"},
        direction_count=3,
    )
    titles = {row["direction_title"] for row in result}
    professional_pool = set(knowledge._FAMILY_ARCHETYPE_POOL["professional"])
    assert titles <= professional_pool


def test_festive_family_does_not_drift_to_professional():
    result = knowledge.resolve_style_archetypes(
        {"occasion": "diwali"},
        {"name": "Embroidered Kurta", "category": "Kurta", "color": "maroon"},
        direction_count=3,
    )
    titles = {row["direction_title"] for row in result}
    festive_pool = set(knowledge._FAMILY_ARCHETYPE_POOL["festive_general"])
    assert titles <= festive_pool


def test_kurta_and_blazer_share_archetype_but_get_distinct_titles():
    """The exact physical-failure fixtures. Both anchors legitimately land on
    the same archetype (relaxed_casual pool head, "Refined Weekend" — this is
    NOT a bug, per the release contract two boards may share an archetype).
    What must not collapse is the explicit board title: title answers "what
    is this particular board", archetype answers "what style family is
    this", and those are separate fields now."""
    kurta = knowledge.resolve_style_archetypes(
        {"occasion": "daily"},
        {"name": "Red Floral Kurta", "category": "Kurta", "color": "red"},
        direction_count=3,
    )
    blazer = knowledge.resolve_style_archetypes(
        {"occasion": "daily"},
        {"name": "Dark Green Blazer", "category": "Blazer", "color": "dark green"},
        direction_count=3,
    )
    # The shared-archetype premise still holds under the (unchanged) ranking.
    assert kurta[0]["direction_title"] == "Refined Weekend"
    assert blazer[0]["direction_title"] == "Refined Weekend"
    # But the explicit board title must diverge, and neither may be blank.
    assert kurta[0]["explicit_title"]
    assert blazer[0]["explicit_title"]
    assert kurta[0]["explicit_title"] != blazer[0]["explicit_title"]


def test_same_archetype_can_produce_different_titles_generally():
    olive_overshirt = knowledge.resolve_style_archetypes(
        {"occasion": "daily"},
        {"name": "Olive Overshirt", "category": "Shirt", "color": "olive"},
        direction_count=3,
    )
    tailored_trousers = knowledge.resolve_style_archetypes(
        {"occasion": "daily"},
        {"name": "Tailored Wool Trouser", "category": "Bottoms", "color": "grey"},
        direction_count=3,
    )
    shared = {r["direction_title"] for r in olive_overshirt} & {
        r["direction_title"] for r in tailored_trousers
    }
    assert shared, "fixture assumption: these anchors must share at least one archetype"
    for name in shared:
        a = next(r for r in olive_overshirt if r["direction_title"] == name)
        b = next(r for r in tailored_trousers if r["direction_title"] == name)
        assert a["explicit_title"] != b["explicit_title"]


def test_explicit_title_is_deterministic_for_same_inputs():
    anchor = {"name": "Red Floral Kurta", "category": "Kurta", "color": "red"}
    first = knowledge.resolve_style_archetypes({"occasion": "daily"}, dict(anchor), direction_count=3)
    second = knowledge.resolve_style_archetypes({"occasion": "daily"}, dict(anchor), direction_count=3)
    assert [r["explicit_title"] for r in first] == [r["explicit_title"] for r in second]


def test_explicit_title_falls_back_to_archetype_when_anchor_has_no_signal():
    anchor = {}
    result = knowledge.resolve_style_archetypes({"occasion": "daily"}, anchor, direction_count=3)
    assert result[0]["explicit_title"] == result[0]["direction_title"] == "Refined Weekend"


def test_explicit_title_distinct_from_archetype_field():
    result = knowledge.resolve_style_archetypes(
        {"occasion": "office"},
        {"name": "Structured Black Jacket", "category": "Outerwear", "color": "black"},
        direction_count=3,
    )
    # archetype/direction_title stays the strategy identity; explicit_title
    # is the presentation-layer field. They are allowed to be equal only
    # when there's no differentiating anchor signal -- here there is one
    # (a structured outerwear anchor), so they must differ.
    assert result[0]["explicit_title"] != result[0]["direction_title"]


def test_style_this_lite_directions_uses_explicit_title_not_archetype():
    from routers import stylist as stylist_router

    kurta_directions = stylist_router._lite_directions(
        {"name": "Red Floral Kurta", "category": "Kurta", "color": "red", "id": "kurta-1"},
        wardrobe=[],
    )
    blazer_directions = stylist_router._lite_directions(
        {"name": "Dark Green Blazer", "category": "Blazer", "color": "dark green", "id": "blazer-1"},
        wardrobe=[],
    )
    assert kurta_directions[0]["title"] != blazer_directions[0]["title"]
    # style_strategy.direction_title (the archetype/shuffle identity) is
    # untouched by this wiring change.
    assert (
        kurta_directions[0]["style_strategy"]["direction_title"]
        == blazer_directions[0]["style_strategy"]["direction_title"]
        == "Refined Weekend"
    )
