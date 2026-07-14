from brain.engines.style_scorer import (
    resolve_occasion_profile,
    score_occasion_compatibility,
    style_scorer,
)
from brain.outfit_pipeline import score_outfit


def test_resolve_occasion_profile_normalizes_beach_alias():
    profile = resolve_occasion_profile(None, {"prompt": "casual beach walk"})

    assert profile["occasion"] == "beach"
    assert profile["has_rules"] is True
    assert profile["preferred_formality"] == "casual"


def test_occasion_compatibility_rejects_office_flip_flops():
    outfit = {
        "top": {"name": "Plain shirt", "category": "shirt"},
        "bottom": {"name": "Navy trousers", "category": "trousers"},
        "shoes": {"name": "Rubber flip flop", "category": "footwear"},
    }

    result = score_occasion_compatibility(outfit, {"occasion": "office"})

    assert result["occasion"] == "office"
    assert result["reject"] is True
    assert any("flip flop" in p for p in result["penalties"])


def test_unified_style_scorer_exposes_occasion_metadata():
    items = [
        {"name": "Linen camp collar shirt", "category": "top", "fabric": "linen"},
        {"name": "Cotton shorts", "category": "bottom", "fabric": "cotton"},
        {"name": "Leather sandals", "category": "footwear"},
    ]

    result = style_scorer.score_outfit(items, {"occasion": "beach"}, {})

    assert result["occasion_profile"]["occasion"] == "beach"
    assert result["occasion_compatibility_score"] > 0.5
    assert result["occasion_reject"] is False


def test_tropical_print_shirt_is_positive_beach_signal():
    outfit = {
        "items": [
            {"name": "Tropical Print Shirt", "category": "top", "material": "cotton"},
            {"name": "Cotton Shorts", "category": "bottom", "fabric": "cotton"},
            {"name": "Leather Sandals", "category": "footwear"},
        ]
    }

    result = score_occasion_compatibility(outfit, {"occasion": "beach"})

    assert result["occasion"] == "beach"
    assert result["score"] > 0.6
    assert result["reject"] is False
    assert any("tropical" in boost or "cotton" in boost for boost in result["boosts"])


def test_metadata_v2_hard_rejects_shiny_gold_for_coffee_date_before_memory():
    outfit = {
        "items": [
            {
                "id": "top-gold",
                "name": "Shiny Gold Formal Shirt",
                "category": "Tops",
                "sub_category": "Shirt",
                "color": "gold",
            },
            {"id": "bottom-1", "name": "Soft Chinos", "category": "Bottoms"},
            {"id": "shoe-1", "name": "Clean Sneakers", "category": "Footwear"},
        ]
    }

    result = score_occasion_compatibility(outfit, {"occasion": "coffee_date"})
    scored = style_scorer.score_outfit(
        outfit["items"],
        {
            "occasion": "coffee_date",
            "saved_item_ids": ["top-gold"],
            "underworn_ids": ["top-gold"],
        },
        {},
    )

    assert result["reject"] is True
    assert any("metadata_v2" in p for p in result["penalties"])
    assert scored["occasion_reject"] is True
    assert scored["breakdown"].get("saved_board_affinity", 0) == 0
    assert scored["breakdown"].get("underworn_boost", 0) == 0


def test_pipeline_score_outfit_uses_scorer_occasion_metadata():
    outfit = {
        "top": {"id": "top-1", "name": "Wool blazer", "category": "blazer"},
        "bottom": {"id": "bottom-1", "name": "Dress pants", "category": "pants"},
        "shoes": {"id": "shoe-1", "name": "Oxford dress shoes", "category": "footwear"},
    }

    scored = score_outfit(
        outfit,
        {"occasion": "beach", "style_dna": {}, "style_graph": {}},
        {"recent_outfits": [], "liked_outfits": [], "disliked_outfits": []},
        {"preferred_fabrics": [], "avoided_items": []},
        {},
    )

    assert scored["score_meta"]["occasion_profile"]["occasion"] == "beach"
    assert scored["score_meta"]["occasion_reject"] is True
    assert scored["ml_features"]["occasion_reject"] == 1.0


def test_reviewed_professional_scores_rank_kurta_above_conditional_accessories():
    def _score(metadata):
        result = style_scorer.score_outfit(
            [{
                "id": "candidate",
                "name": "Reviewed Candidate",
                "category": "top",
                "style_metadata": metadata,
            }],
            {"occasion": "client_meeting"},
            {},
        )
        return result["breakdown"].get("metadata_richness", 0.0)

    kurta = _score({
        "professionalism_score": 0.72,
        "client_meeting_score": 0.60,
        "boardroom_score": 0.35,
    })
    scarf = _score({
        "professionalism_score": 0.55,
        "client_meeting_score": 0.45,
        "boardroom_score": 0.25,
    })
    heel = _score({
        "professionalism_score": 0.55,
        "client_meeting_score": 0.35,
        "boardroom_score": 0.15,
    })

    assert kurta > scarf > heel
