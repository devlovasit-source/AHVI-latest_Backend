from services.style_flow_service import finalize_style_cards, finalize_style_response_payload


def _item(item_id, name, role):
    return {
        "id": item_id,
        "name": name,
        "category": role,
        "role": role,
        "image_url": f"https://img/{item_id}.png",
    }


def _card(provenance=None):
    card = {
        "id": "look-1",
        "items": [
            _item("top-secret-id", "Navy Shirt", "top"),
            _item("bottom-1", "Grey Trousers", "bottom"),
            _item("shoe-1", "Black Loafers", "footwear"),
        ],
    }
    if provenance is not None:
        card["_style_explanation_provenance"] = provenance
    return card


def _final(provenance=None, query="daily outfit"):
    return finalize_style_cards([_card(provenance)], query=query, default_limit=1)[0]


def test_weather_rationale_survives_when_backed_by_positive_metadata():
    card = _final(
        {
            "reasons": [],
            "breakdown": {"weather_compatibility": 1.2},
            "weather_compatibility": {"score": 0.8, "weather": "rain"},
        }
    )

    assert card["why_it_works"] == (
        "This combination was favored because it is better suited to rainy conditions."
    )


def test_occasion_reason_has_priority_over_personalization_and_memory():
    card = _final(
        {
            "reasons": [
                "occasion_fit:professional",
                "matches your style",
                "brings an under-worn piece back into rotation",
            ],
            "breakdown": {
                "occasion_compatibility": 2.0,
                "style_dna": 1.0,
                "underworn_boost": 1.2,
            },
        }
    )

    assert card["why_it_works"] == (
        "This combination was selected for its strong fit with the occasion."
    )
    assert "style dna" not in card["why_it_works"].lower()
    assert "under-worn" not in card["why_it_works"].lower()


def test_style_dna_reason_survives_only_when_explicitly_backed():
    backed = _final(
        {
            "reasons": ["matches your style"],
            "breakdown": {"style_dna": 1.4},
        }
    )
    unbacked = _final(
        {
            "reasons": ["matches your style"],
            "breakdown": {"style_dna": 0.0},
        }
    )

    assert backed["why_it_works"] == (
        "This combination matches your established Style DNA preferences."
    )
    assert "style dna" not in unbacked["why_it_works"].lower()


def test_memory_reason_survives_only_when_explicitly_backed():
    underworn = _final(
        {
            "reasons": ["brings an under-worn piece back into rotation"],
            "breakdown": {"underworn_boost": 1.2},
        }
    )
    recent = _final(
        {
            "reasons": ["recently worn — offering a fresher option"],
            "breakdown": {"recent_repeat_penalty": -1.5},
        }
    )

    assert underworn["why_it_works"] == (
        "This choice brings an under-worn piece back into rotation."
    )
    assert recent["why_it_works"] == (
        "This choice accounts for pieces worn recently to keep the rotation fresh."
    )


def test_generic_fallback_remains_without_trusted_provenance():
    no_provenance = _final()
    empty_personalization = _final(
        {"reasons": [], "breakdown": {"style_dna": 0.0, "memory": 0.0}}
    )

    assert no_provenance["why_it_works"]
    assert empty_personalization["why_it_works"]
    assert "style dna" not in empty_personalization["why_it_works"].lower()
    assert "personal" not in empty_personalization["why_it_works"].lower()


def test_provenance_is_bounded_private_and_response_shape_is_unchanged():
    card = _final(
        {
            "reasons": [
                "occasion_fit:office",
                "matches your style",
                "echoes a look you saved",
            ],
            "breakdown": {
                "occasion_compatibility": 2.0,
                "style_dna": 1.0,
                "saved_board_affinity": 1.0,
                "debug_score": 999,
            },
            "debug_item_id": "top-secret-id",
        }
    )

    assert card["why_it_works"].count(".") == 1
    assert card["why_it_works"] == card["explanation"] == card["reason"]
    assert card["style_reason"] == card["why_it_works"]
    assert "_style_explanation_provenance" not in card
    assert "top-secret-id" not in card["why_it_works"]
    assert "999" not in card["why_it_works"]
    assert "debug" not in card["why_it_works"].lower()


def test_response_finalizer_joins_outfit_provenance_to_rendered_card():
    items = _card()["items"]
    response = finalize_style_response_payload(
        {
            "cards": [{"id": "rendered-look", "items": items}],
            "outfits": [
                {
                    "id": "scored-look",
                    "items": items,
                    "unified_style": {
                        "reasons": ["matches your style"],
                        "breakdown": {"style_dna": 1.4},
                    },
                }
            ],
        },
        user_id="user-1",
        query="daily outfit",
        wardrobe=items,
        include_base64=False,
    )

    card = response["cards"][0]
    assert card["why_it_works"] == (
        "This combination matches your established Style DNA preferences."
    )
    assert card["why_it_works"] == card["explanation"] == card["reason"]
    assert card["style_reason"] == card["why_it_works"]
    assert "_style_explanation_provenance" not in card
