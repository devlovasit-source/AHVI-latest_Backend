from services.style_flow_service import (
    card_signature,
    finalize_style_cards,
    finalize_style_response_payload,
)
from services.category_taxonomy import infer_style_attributes


GENERIC_TITLES = {"Signature Combo", "Easy Win", "Today's Edit", "Polished Daily"}


def _item(item_id, name, role, color="black"):
    return {
        "id": item_id,
        "name": name,
        "role": role,
        "color": color,
        "image_url": f"https://x/{item_id}.png",
    }


def _card(top, bottom, footwear, *, title="Signature Combo", score=100):
    return {
        "title": title,
        "score": score,
        "items": [top, bottom, footwear, _item(f"watch-{top['id']}", "Omega Speedmaster", "accessory")],
    }


def test_taxonomy_infers_trust_layer_style_attributes():
    attrs = infer_style_attributes(
        {
            "name": "Burgundy Leather Loafers",
            "category": "Footwear",
            "sub_category": "Loafers",
            "pattern": "plain",
            "occasions": ["office", "date"],
        }
    )

    assert attrs["formality"] >= 4
    assert attrs["aesthetic_cluster"] in {"classic", "polished"}
    assert attrs["occasion_fitness"]["office"] >= 0.8
    assert attrs["occasion_fitness"]["date"] >= 0.7


def test_taxonomy_marks_relaxed_footwear_as_low_office_fit():
    attrs = infer_style_attributes(
        {
            "name": "Birkenstock Sandals",
            "category": "Footwear",
            "sub_category": "Sandals",
            "pattern": "plain",
        }
    )

    assert attrs["formality"] <= 2
    assert attrs["occasion_fitness"]["office"] <= 0.2


def test_more_options_excludes_already_seen_style_signatures():
    seen_card = {
        "title": "Boardroom Casual",
        "items": [
            {"id": "top-1", "name": "White Shirt", "role": "top", "image_url": "https://x/top1.png"},
            {"id": "bottom-1", "name": "Black Pants", "role": "bottom", "image_url": "https://x/bottom1.png"},
            {"id": "shoe-1", "name": "Black Sneakers", "role": "footwear", "image_url": "https://x/shoe1.png"},
        ],
    }
    fresh_card = {
        "title": "Sharp Daily",
        "items": [
            {"id": "top-2", "name": "Blue Button-Down Shirt", "role": "top", "image_url": "https://x/top2.png"},
            {"id": "bottom-2", "name": "Grey Trousers", "role": "bottom", "image_url": "https://x/bottom2.png"},
            {"id": "shoe-2", "name": "Loafers", "role": "footwear", "image_url": "https://x/shoe2.png"},
        ],
    }

    filtered = finalize_style_cards(
        [seen_card, fresh_card],
        query="office outfit",
        exclude_signatures=[card_signature(seen_card)],
        requested_count=3,
    )

    assert [card_signature(card) for card in filtered] == [card_signature(fresh_card)]


def test_more_options_requested_count_caps_returned_cards():
    cards = [
        {
            "items": [
                {"id": "top-1", "name": "White Shirt", "role": "top", "image_url": "https://x/top1.png"},
                {"id": "bottom-1", "name": "Black Pants", "role": "bottom", "image_url": "https://x/bottom1.png"},
                {"id": "shoe-1", "name": "Black Sneakers", "role": "footwear", "image_url": "https://x/shoe1.png"},
            ]
        },
        {
            "items": [
                {"id": "top-2", "name": "Blue Shirt", "role": "top", "image_url": "https://x/top2.png"},
                {"id": "bottom-2", "name": "Grey Trousers", "role": "bottom", "image_url": "https://x/bottom2.png"},
                {"id": "shoe-2", "name": "Loafers", "role": "footwear", "image_url": "https://x/shoe2.png"},
            ]
        },
        {
            "items": [
                {"id": "top-3", "name": "Off White Shirt", "role": "top", "image_url": "https://x/top3.png"},
                {"id": "bottom-3", "name": "Jeans", "role": "bottom", "image_url": "https://x/bottom3.png"},
                {"id": "shoe-3", "name": "Sneakers", "role": "footwear", "image_url": "https://x/shoe3.png"},
            ]
        },
    ]

    filtered = finalize_style_cards(
        cards,
        query="office outfit",
        requested_count=2,
    )

    assert len(filtered) == 2
    assert all(card_signature(card) for card in filtered)


def test_style_response_contract_uses_single_final_board_list():
    result = {
        "cards": [
            {
                "title": "Signature Combo",
                "items": [
                    {"id": "top-1", "name": "White Shirt", "role": "top", "image_url": "https://x/top1.png"},
                    {"id": "bottom-1", "name": "Black Pants", "role": "bottom", "image_url": "https://x/bottom1.png"},
                    {"id": "shoe-1", "name": "Black Sneakers", "role": "footwear", "image_url": "https://x/shoe1.png"},
                ],
            }
        ],
        "outfits": [],
        "pipeline": {"stages": ["test"]},
    }

    response = finalize_style_response_payload(
        result,
        user_id="u1",
        query="office outfit",
        include_base64=False,
        cache_bypass=True,
    )

    assert response["style_boards"] == response["cards"]
    assert response["data"]["outfits"] == response["cards"]
    assert response["data"]["rendered_boards"] == response["cards"]
    assert response["meta"]["board_count"] == 1
    assert response["meta"]["style_signature"]


def test_accessory_only_change_does_not_create_new_style_board():
    base_core = [
        {"id": "top-1", "name": "White Shirt", "role": "top", "image_url": "https://x/top1.png"},
        {"id": "bottom-1", "name": "Black Pants", "role": "bottom", "image_url": "https://x/bottom1.png"},
        {"id": "shoe-1", "name": "Black Loafers", "role": "footwear", "image_url": "https://x/shoe1.png"},
    ]
    cards = [
        {
            "items": base_core
            + [{"id": "watch-1", "name": "Silver Watch", "role": "accessory", "image_url": "https://x/watch1.png"}],
        },
        {
            "items": base_core
            + [{"id": "watch-2", "name": "Gold Watch", "role": "accessory", "image_url": "https://x/watch2.png"}],
        },
    ]

    filtered = finalize_style_cards(cards, query="office outfit")

    assert len(filtered) == 1
    assert filtered[0]["_style_core_signature"] == "top-1|bottom-1|shoe-1"


def test_generic_office_demotes_birkenstock_when_clean_footwear_exists():
    black_pants = _item("bottom-1", "Black Pants", "bottom")
    cards = [
        _card(_item("top-1", "Teal Shirt", "top", "teal"), black_pants, _item("shoe-birk", "Birkenstock", "footwear"), score=120),
        _card(_item("top-2", "Off White Shirt", "top", "white"), _item("bottom-2", "Grey Trousers", "bottom", "grey"), _item("shoe-loafer", "Leather Loafers", "footwear"), score=90),
        _card(_item("top-3", "Blue Button-Down", "top", "blue"), _item("bottom-3", "Navy Chinos", "bottom", "navy"), _item("shoe-white", "White Sneakers", "footwear"), score=85),
        _card(_item("top-4", "Black Shirt", "top"), black_pants, _item("shoe-black", "Black Sneakers", "footwear"), score=80),
    ]

    filtered = finalize_style_cards(cards, query="I need office outfit", default_limit=4)
    top_three_names = [
        item["name"].lower()
        for card in filtered[:3]
        for item in card["items"]
        if item.get("role") == "footwear"
    ]

    assert all("birkenstock" not in name for name in top_three_names)


def test_repeated_bottom_is_capped_when_alternatives_exist():
    cards = [
        _card(_item("top-1", "White Shirt", "top"), _item("bottom-black", "Black Pants", "bottom"), _item("shoe-1", "Black Sneakers", "footwear"), score=110),
        _card(_item("top-2", "Blue Shirt", "top"), _item("bottom-black", "Black Pants", "bottom"), _item("shoe-2", "White Sneakers", "footwear"), score=108),
        _card(_item("top-3", "Teal Shirt", "top"), _item("bottom-black", "Black Pants", "bottom"), _item("shoe-3", "Loafers", "footwear"), score=106),
        _card(_item("top-4", "Off White Shirt", "top"), _item("bottom-grey", "Grey Trousers", "bottom", "grey"), _item("shoe-4", "Leather Sneakers", "footwear"), score=80),
        _card(_item("top-5", "Mint Shirt", "top"), _item("bottom-navy", "Navy Chinos", "bottom", "navy"), _item("shoe-5", "Loafers", "footwear"), score=78),
    ]

    filtered = finalize_style_cards(cards, query="office outfit", default_limit=5)
    bottom_ids = [
        item["id"]
        for card in filtered
        for item in card["items"]
        if item.get("role") == "bottom"
    ]

    assert bottom_ids.count("bottom-black") <= 2
    assert {"bottom-grey", "bottom-navy"}.intersection(bottom_ids)


def test_footwear_only_change_is_not_prioritized_as_new_direction():
    same_top = _item("top-1", "White Shirt", "top")
    same_bottom = _item("bottom-1", "Black Pants", "bottom")
    cards = [
        _card(same_top, same_bottom, _item("shoe-1", "Black Sneakers", "footwear"), score=100),
        _card(same_top, same_bottom, _item("shoe-2", "White Sneakers", "footwear"), score=99),
        _card(_item("top-2", "Blue Button-Down", "top"), _item("bottom-2", "Grey Trousers", "bottom", "grey"), _item("shoe-3", "Loafers", "footwear"), score=80),
    ]

    filtered = finalize_style_cards(cards, query="office outfit", default_limit=3)
    top_bottom_pairs = [
        "|".join(
            item["id"]
            for item in card["items"]
            if item.get("role") in {"top", "bottom"}
        )
        for card in filtered[:2]
    ]

    assert len(set(top_bottom_pairs)) == len(top_bottom_pairs)


def test_footwear_only_change_is_suppressed_when_base_alternatives_exist():
    same_top = _item("top-1", "White Shirt", "top")
    same_bottom = _item("bottom-1", "Black Pants", "bottom")
    cards = [
        _card(same_top, same_bottom, _item("shoe-1", "Leather Loafers", "footwear"), score=120),
        _card(same_top, same_bottom, _item("shoe-2", "White Minimal Sneakers", "footwear"), score=118),
        _card(_item("top-2", "Blue Button-Down", "top"), _item("bottom-2", "Grey Trousers", "bottom", "grey"), _item("shoe-3", "Loafers", "footwear"), score=95),
        _card(_item("top-3", "Black Shirt", "top"), _item("bottom-3", "Navy Chinos", "bottom", "navy"), _item("shoe-4", "Black Sneakers", "footwear"), score=90),
    ]

    filtered = finalize_style_cards(cards, query="business meeting outfit", default_limit=3)
    base_signatures = [card["style_metadata"]["base_outfit_signature"] for card in filtered]

    assert len(base_signatures) == len(set(base_signatures))


def test_final_boards_have_archetype_explanation_and_visual_metadata():
    cards = [
        _card(_item("top-1", "White Shirt", "top", "white"), _item("bottom-1", "Black Pants", "bottom"), _item("shoe-1", "Black Sneakers", "footwear")),
        _card(_item("top-2", "Blue Button-Down", "top", "blue"), _item("bottom-2", "Grey Trousers", "bottom", "grey"), _item("shoe-2", "Loafers", "footwear")),
        _card(_item("top-3", "Off White Shirt", "top", "white"), _item("bottom-3", "Navy Chinos", "bottom", "navy"), _item("shoe-3", "White Sneakers", "footwear")),
    ]

    filtered = finalize_style_cards(cards, query="office outfit", default_limit=3)
    modes = [card["explanation_mode"] for card in filtered]

    assert all(card["title"] not in GENERIC_TITLES for card in filtered)
    assert all(card.get("style_archetype") for card in filtered)
    assert all(card.get("style_direction") == "smart_casual_office" for card in filtered)
    assert all(card.get("diversity_profile") for card in filtered)
    assert all(card.get("styling_tip") for card in filtered)
    assert all(card.get("layout_preset") for card in filtered)
    assert all(card.get("style_metadata") for card in filtered)
    assert len(modes) == len(set(modes))


def test_style_response_exposes_additive_board_metadata():
    result = {
        "cards": [
            _card(_item("top-1", "White Shirt", "top"), _item("bottom-1", "Black Pants", "bottom"), _item("shoe-1", "Loafers", "footwear")),
            _card(_item("top-2", "Blue Shirt", "top"), _item("bottom-2", "Grey Trousers", "bottom", "grey"), _item("shoe-2", "White Sneakers", "footwear")),
        ],
        "outfits": [],
    }

    response = finalize_style_response_payload(
        result,
        user_id="u1",
        query="business outfit",
        include_base64=False,
    )

    assert response["data"]["board_metadata"]
    assert response["meta"]["board_metadata"]
    assert response["data"]["board_metadata"][0]["base_outfit_signature"]
    assert response["data"]["board_metadata"][0]["coherence_score"] is not None


def test_general_style_dna_metadata_is_added_for_non_office_requests():
    cards = [
        _card(_item("top-1", "Patterned Statement Shirt", "top", "red"), _item("bottom-1", "Black Jeans", "bottom"), _item("shoe-1", "Chelsea Boots", "footwear"), score=95),
        _card(_item("top-2", "Black Shirt", "top"), _item("bottom-2", "Black Pants", "bottom"), _item("shoe-2", "Black Sneakers", "footwear"), score=90),
        _card(_item("top-3", "Soft Dinner Shirt", "top", "white"), _item("bottom-3", "Grey Trousers", "bottom", "grey"), _item("shoe-3", "Leather Loafers", "footwear"), score=88),
    ]

    filtered = finalize_style_cards(cards, query="date night outfit", default_limit=3)

    assert filtered
    assert all(card.get("style_energy") for card in filtered)
    assert all(card.get("silhouette_category") for card in filtered)
    assert all(card.get("palette_direction") for card in filtered)
    assert all(card.get("footwear_energy") for card in filtered)
    assert all(card.get("formality_energy") for card in filtered)
    assert all(card.get("occasion_fit") is not None for card in filtered)
    assert all(card.get("style_direction") == "date_night" for card in filtered)


def test_date_request_demotes_relaxed_sandals_when_polished_footwear_exists():
    cards = [
        _card(_item("top-1", "Dinner Shirt", "top", "white"), _item("bottom-1", "Black Pants", "bottom"), _item("shoe-sandal", "Birkenstock Sandals", "footwear"), score=120),
        _card(_item("top-2", "Black Shirt", "top"), _item("bottom-2", "Grey Trousers", "bottom", "grey"), _item("shoe-loafer", "Leather Loafers", "footwear"), score=80),
    ]

    filtered = finalize_style_cards(cards, query="date night outfit", default_limit=2)
    first_footwear = [
        item["name"].lower()
        for item in filtered[0]["items"]
        if item.get("role") == "footwear"
    ][0]

    assert "sandal" not in first_footwear
    assert "loafer" in first_footwear


def test_date_request_blocks_relaxed_sandals_from_final_boards():
    cards = [
        _card(_item("top-1", "Tropical Print Shirt", "top", "orange"), _item("bottom-1", "Black Pants", "bottom"), _item("shoe-birk", "Birkenstock", "footwear"), score=180),
        _card(_item("top-2", "Black Shirt", "top"), _item("bottom-2", "Grey Trousers", "bottom", "grey"), _item("shoe-loafer", "Leather Loafers", "footwear"), score=80),
        _card(_item("top-3", "Off White Shirt", "top", "white"), _item("bottom-3", "Black Jeans", "bottom"), _item("shoe-boot", "Chelsea Boots", "footwear"), score=75),
    ]

    filtered = finalize_style_cards(cards, query="I have a date tonight", default_limit=3)
    footwear_names = [
        item["name"].lower()
        for card in filtered
        for item in card["items"]
        if item.get("role") == "footwear"
    ]

    assert footwear_names
    assert all("birkenstock" not in name and "sandal" not in name for name in footwear_names)


def test_party_request_prefers_statement_or_polished_energy():
    cards = [
        _card(_item("top-1", "Plain Office Shirt", "top", "white"), _item("bottom-1", "Grey Trousers", "bottom", "grey"), _item("shoe-1", "White Sneakers", "footwear"), score=100),
        _card(_item("top-2", "Patterned Statement Shirt", "top", "red"), _item("bottom-2", "Black Jeans", "bottom"), _item("shoe-2", "Chelsea Boots", "footwear"), score=85),
    ]

    filtered = finalize_style_cards(cards, query="party outfit", default_limit=2)

    assert filtered[0]["style_energy"] in {"expressive/statement", "polished/social", "minimal/monochrome"}


def test_travel_request_prefers_comfort_polish_over_formal_event_energy():
    cards = [
        _card(_item("top-1", "Formal Shirt", "top", "white"), _item("bottom-1", "Formal Trousers", "bottom"), _item("shoe-1", "Oxford Formal Shoes", "footwear"), score=100),
        _card(_item("top-2", "Clean Polo", "top", "blue"), _item("bottom-2", "Travel Chinos", "bottom", "navy"), _item("shoe-2", "Clean Sneakers", "footwear"), score=88),
    ]

    filtered = finalize_style_cards(cards, query="airport travel outfit", default_limit=2)
    first_names = " ".join(item["name"].lower() for item in filtered[0]["items"])

    assert "sneaker" in first_names or filtered[0]["style_energy"] == "elevated/casual"


def test_first_boards_prefer_distinct_style_energy_when_supported():
    cards = [
        _card(_item("top-1", "Oxford Shirt", "top", "white"), _item("bottom-1", "Tailored Trousers", "bottom", "grey"), _item("shoe-1", "Leather Loafers", "footwear"), score=100),
        _card(_item("top-2", "Patterned Statement Shirt", "top", "red"), _item("bottom-2", "Black Jeans", "bottom"), _item("shoe-2", "Chelsea Boots", "footwear"), score=98),
        _card(_item("top-3", "Black Shirt", "top"), _item("bottom-3", "Black Pants", "bottom"), _item("shoe-3", "Black Sneakers", "footwear"), score=96),
        _card(_item("top-4", "Clean Polo", "top", "blue"), _item("bottom-4", "Navy Chinos", "bottom", "navy"), _item("shoe-4", "White Sneakers", "footwear"), score=94),
        _card(_item("top-5", "Relaxed Denim Shirt", "top", "blue"), _item("bottom-5", "Light Blue Jeans", "bottom", "blue"), _item("shoe-5", "Casual Sneakers", "footwear"), score=92),
    ]

    filtered = finalize_style_cards(cards, query="casual weekend outfit", default_limit=5)
    energies = [card.get("style_energy") for card in filtered[:5]]

    assert len(set(energies)) >= 4


def test_pipeline_diversifier_preserves_bottom_and_footwear_variety():
    import sys
    import types

    if "dotenv" not in sys.modules:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None
        sys.modules["dotenv"] = dotenv

    try:
        from brain.outfit_pipeline import _diversify_outfits
    except ModuleNotFoundError:
        return

    def outfit(top, bottom, shoes, score):
        return {
            "top": {"id": top, "name": top},
            "bottom": {"id": bottom, "name": bottom},
            "shoes": {"id": shoes, "name": shoes},
            "score": score,
        }

    ranked = [
        outfit("top-1", "black-pants", "loafer", 100),
        outfit("top-2", "black-pants", "white-sneaker", 99),
        outfit("top-3", "black-pants", "black-sneaker", 98),
        outfit("top-4", "light-jeans", "loafer", 80),
        outfit("top-5", "navy-chinos", "white-sneaker", 78),
        outfit("top-6", "grey-trousers", "black-sneaker", 76),
    ]

    diversified = _diversify_outfits(ranked, limit=6)
    bottom_ids = [row["bottom"]["id"] for row in diversified]
    footwear_ids = [row["shoes"]["id"] for row in diversified]

    assert bottom_ids.count("black-pants") <= 2
    assert len(set(bottom_ids)) >= 3
    assert all(footwear_ids.count(shoe) <= 2 for shoe in set(footwear_ids))


def test_generic_today_request_defaults_to_smart_casual_not_relaxed_sandal():
    cards = [
        _card(_item("top-1", "Off White Shirt", "top", "white"), _item("bottom-1", "Black Pants", "bottom"), _item("shoe-birk", "Birkenstock Sandals", "footwear"), score=130),
        _card(_item("top-2", "Blue Button-Down", "top", "blue"), _item("bottom-2", "Grey Trousers", "bottom", "grey"), _item("shoe-loafer", "Burgundy Loafers", "footwear"), score=90),
    ]

    filtered = finalize_style_cards(cards, query="Suggest an outfit for today", default_limit=2)
    first_names = " ".join(item["name"].lower() for item in filtered[0]["items"])

    assert filtered[0]["style_direction"] == "smart_casual_office"
    assert "birkenstock" not in first_names


def test_raw_quality_guard_blocks_birkenstock_for_smart_requests():
    from brain.engines.outfit_quality_guard import filter_and_guard_outfits

    outfits = [
        {
            "top": {"id": "top-1", "name": "Off White Shirt", "role": "top"},
            "bottom": {"id": "bottom-1", "name": "Black Pants", "role": "bottom"},
            "shoes": {"id": "shoe-birk", "name": "Birkenstock", "role": "footwear"},
            "score": 200,
        },
        {
            "top": {"id": "top-2", "name": "Black Shirt", "role": "top"},
            "bottom": {"id": "bottom-2", "name": "Grey Trousers", "role": "bottom"},
            "shoes": {"id": "shoe-loafer", "name": "Leather Loafers", "role": "footwear"},
            "score": 80,
        },
    ]

    guarded = filter_and_guard_outfits(
        outfits,
        intent="office",
        query="Suggest an outfit for today",
    )
    footwear_names = [
        (row.get("shoes") or row.get("footwear") or {}).get("name", "").lower()
        for row in guarded
    ]

    assert "birkenstock" not in " ".join(footwear_names)
    assert any("loafer" in name for name in footwear_names)


def test_raw_quality_guard_demotes_third_repeated_bottom_before_cards():
    from brain.engines.outfit_quality_guard import filter_and_guard_outfits

    outfits = [
        {
            "top": {"id": f"top-{idx}", "name": f"Shirt {idx}", "role": "top"},
            "bottom": {"id": "black-pants", "name": "Black Pants", "role": "bottom"},
            "shoes": {"id": f"shoe-{idx}", "name": "Leather Loafers", "role": "footwear"},
            "score": 120 - idx,
        }
        for idx in range(4)
    ] + [
        {
            "top": {"id": "top-alt", "name": "Blue Shirt", "role": "top"},
            "bottom": {"id": "grey-trousers", "name": "Grey Trousers", "role": "bottom"},
            "shoes": {"id": "shoe-alt", "name": "Black Sneakers", "role": "footwear"},
            "score": 94,
        }
    ]

    guarded = filter_and_guard_outfits(outfits, intent="office", query="office outfit")
    first_four_bottoms = [
        (row.get("bottom") or {}).get("id")
        for row in guarded[:4]
    ]

    assert "grey-trousers" in first_four_bottoms
    assert first_four_bottoms.count("black-pants") <= 3


def test_accessory_selector_allows_restraint_and_rotation():
    import sys
    import types

    if "dotenv" not in sys.modules:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None
        sys.modules["dotenv"] = dotenv

    try:
        from brain.outfit_pipeline import _select_accessories
    except ModuleNotFoundError:
        return

    wardrobe = {
        "accessories": [
            _item("watch-1", "Omega Watch", "accessory"),
            _item("sun-1", "Sunglasses", "accessory"),
            _item("ring-1", "Silver Ring", "accessory"),
            _item("belt-1", "Black Belt", "accessory"),
        ]
    }
    combos = [
        {"top": {"id": f"top-{i}"}, "bottom": {"id": f"bottom-{i}"}, "shoes": {"id": f"shoe-{i}"}, "combo_id": f"combo-{i}"}
        for i in range(8)
    ]
    picked = [_select_accessories(wardrobe, combo, limit=4) for combo in combos]
    flattened = [item["id"] for group in picked for item in group]

    assert any(len(group) == 0 for group in picked)
    assert all(len(group) <= 2 for group in picked)
    assert flattened.count("watch-1") < len(combos)
