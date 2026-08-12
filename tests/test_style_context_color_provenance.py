from brain.personalization.style_dna_engine import StyleDNAEngine
from services.style_context_service import (
    build_style_context,
    compact_context_for_prompt,
    compact_style_dna,
)


def test_explicit_color_aliases_are_normalized_together():
    compact = compact_style_dna(
        {
            "color_dna": {
                "core_colors": ["navy"],
                "power_colors": ["warm beige"],
                "avoided_colors": ["charcoal"],
            },
            "preferred_colors": ["olive"],
        },
        {"colors": ["burgundy", "forest green"], "avoided_colors": ["ivory"]},
    )

    assert compact["preferred_colors"] == [
        "navy", "warm beige", "olive", "burgundy", "forest green"
    ]
    assert compact["avoided_colors"] == ["charcoal", "ivory"]


def test_generic_face_and_profile_recommendations_never_become_colors():
    profile = {
        "face_scan": {"recommendations": ["Vitamin C Serum", "SPF moisturizer"]},
        "recommendations": ["lipstick", "eye cream"],
    }
    context = build_style_context(query="office look", user_profile=profile)
    compact = compact_context_for_prompt(context)

    assert compact["style_dna"] is None
    assert compact["preferences"] == {}
    assert "recommendations" not in str(compact).lower()
    assert "vitamin c" not in str(compact).lower()


def test_product_strings_are_rejected_inside_trusted_color_fields():
    compact = compact_style_dna(
        {
            "color_dna": {
                "core_colors": ["navy", "Vitamin C Serum", "SPF"],
                "avoided_colors": ["mascara", "ivory"],
            },
            "preferred_colors": ["foundation", "forest green"],
        },
        {"colors": ["lipstick", "warm beige"]},
    )

    assert compact["preferred_colors"] == ["navy", "forest green", "warm beige"]
    assert compact["avoided_colors"] == ["ivory"]


def test_compact_prompt_allowlists_preferences_and_memory_without_raw_ids():
    context = build_style_context(
        query="office then dinner",
        wardrobe_items=[
            {"id": "top-1", "name": "Navy Shirt", "category": "top"},
            {"id": "shoe-1", "name": "Black Loafers", "category": "footwear"},
        ],
        event_context={
            "title": "Client dinner",
            "venue": "Downtown",
            "private_notes": "do not expose",
        },
        user_profile={
            "preferences": {
                "colors": ["navy"],
                "style_keywords": ["minimal"],
                "recommendations": ["serum"],
                "private_blob": {"token": "secret"},
            }
        },
        memory={
            "recently_worn_ids": ["top-1"],
            "underworn_ids": ["shoe-1", "unknown-private-id"],
            "saved_item_ids": ["top-1", "another-private-id"],
            "favorite_colors": ["burgundy", "sunscreen"],
            "favorite_categories": ["shirts"],
            "saved_board_patterns": ["quiet luxury"],
        },
    )

    compact = compact_context_for_prompt(context)
    memory = compact["memory"]

    assert compact["preferences"] == {
        "colors": ["navy"], "style_keywords": ["minimal"]
    }
    assert compact["event_context"] == {
        "title": "Client dinner", "venue": "Downtown"
    }
    assert memory["recently_worn"] == ["Navy Shirt"]
    assert memory["underworn"] == ["Black Loafers"]
    assert memory["saved_items"] == ["Navy Shirt"]
    assert memory["underworn_count"] == 2
    assert memory["saved_item_count"] == 2
    assert memory["favorite_colors"] == ["burgundy"]
    assert memory["favorite_categories"] == ["shirts"]
    assert memory["saved_board_patterns"] == ["quiet luxury"]
    assert "private-id" not in str(compact)
    assert "private_notes" not in str(compact)


def test_legacy_style_dna_engine_uses_same_color_validation():
    dna = StyleDNAEngine()._build_dna(
        profile={"preferred_colors": ["olive", "moisturizer"]},
        history=[],
        previous_dna={"preferred_colors": ["charcoal", "serum"]},
        feedback_user={},
        memory={"memory_signals": {"liked_colors": ["ivory", "SPF"]}},
    )

    assert dna["preferred_colors"] == ["olive", "charcoal", "ivory"]
