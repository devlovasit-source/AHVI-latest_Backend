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
