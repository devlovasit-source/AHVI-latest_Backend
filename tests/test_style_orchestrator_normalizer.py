"""Tests for the canonical Style Orchestrator brief normalizer + validation."""
from services.style_orchestrator_normalizer import (
    normalize_style_orchestrator_brief,
    validate_canonical_brief,
)

CLIENT_MEETING_RAW = {
    "occasion": "client meeting",
    "sub_intent": "outfit generation",
    "formality": "business professional",
    "style_direction": [
        "minimal executive",
        "modern tailoring",
        "polished professional",
        "elevated business casual",
    ],
    "wardrobe_usage": True,
    "avoid_items": [
        "athletic sneakers", "hoodies", "distressed denim", "graphic t-shirts",
        "shorts", "sleepwear", "beachwear", "overtly casual sandals",
        "excessive transparency", "excessive or distracting jewelry",
    ],
    "required_slots": ["top", "bottom", "outerwear", "footwear", "bag", "minimal jewelry"],
    "palette_direction": ["sophisticated neutrals", "monochromatic", "subtle accent colors"],
    "accessory_policy": "restrained, functional, professional, understated",
    "clarification_needed": False,
}


def test_A_client_meeting_normalizes():
    b = normalize_style_orchestrator_brief(CLIENT_MEETING_RAW, has_wardrobe_context=True)
    assert b["occasion"] == "client_meeting"
    assert b["sub_intent"] == "outfit_generation"
    assert b["formality"]["score"] == 4
    assert b["wardrobe_usage"] == "owned_first"
    assert b["required_slots"] == ["top", "bottom", "footwear"]
    assert "outerwear" in b["optional_slots"]
    assert "bag" not in b["required_slots"]
    assert "minimal_jewelry" not in b["required_slots"]
    assert "athletic_sneakers" in b["avoid_items"]
    assert "casual_sandals" in b["avoid_items"]
    assert "transparent_items" in b["avoid_items"]
    assert "distracting_jewelry" in b["avoid_items"]
    assert b["clarification_needed"] is False
    assert b["accessory_policy"] == "restrained_functional_professional_understated"
    ok, reason = validate_canonical_brief(b)
    assert ok, reason


def test_B_style_direction_list_to_primary_alternates():
    b = normalize_style_orchestrator_brief(CLIENT_MEETING_RAW)
    assert b["style_direction"]["primary"] == "minimal executive"
    assert b["style_direction"]["alternates"] == [
        "modern tailoring", "polished professional", "elevated business casual",
    ]
    assert b["palette_direction"]["primary"] == "sophisticated_neutrals"
    assert b["palette_direction"]["alternates"] == ["monochromatic", "subtle_accent_colors"]


def test_C_wardrobe_usage_bool():
    on = normalize_style_orchestrator_brief({**CLIENT_MEETING_RAW, "wardrobe_usage": True})
    off = normalize_style_orchestrator_brief({**CLIENT_MEETING_RAW, "wardrobe_usage": False})
    assert on["wardrobe_usage"] == "owned_first"
    assert off["wardrobe_usage"] == "inspiration_only"
    # unknown + no wardrobe context -> inspiration_only
    unk = normalize_style_orchestrator_brief(
        {**CLIENT_MEETING_RAW, "wardrobe_usage": "???"}, has_wardrobe_context=False
    )
    assert unk["wardrobe_usage"] == "inspiration_only"


def test_D_invalid_json_falls_back_safely():
    for bad in [None, "not a dict", 42, []]:
        b = normalize_style_orchestrator_brief(bad)
        ok, _ = validate_canonical_brief(b)
        assert not ok  # falls back, never crashes


def test_E_missing_required_keys_falls_back():
    b = normalize_style_orchestrator_brief({"formality": "casual"})  # no occasion/slots/direction
    ok, reason = validate_canonical_brief(b)
    assert not ok
    assert reason in {"missing_occasion", "missing_required_slots", "missing_style_direction", "low_confidence"}


def test_F_formality_clamps_1_to_5():
    assert normalize_style_orchestrator_brief({**CLIENT_MEETING_RAW, "formality": "sleepwear"})["formality"]["score"] == 1
    assert normalize_style_orchestrator_brief({**CLIENT_MEETING_RAW, "formality": "black tie"})["formality"]["score"] == 5
    over = normalize_style_orchestrator_brief({**CLIENT_MEETING_RAW, "formality": {"label": "x", "score": 99}})
    assert over["formality"]["score"] == 5
    under = normalize_style_orchestrator_brief({**CLIENT_MEETING_RAW, "formality": {"label": "x", "score": -3}})
    assert under["formality"]["score"] == 1


def test_G_client_meeting_never_requires_casual_garments():
    b = normalize_style_orchestrator_brief(CLIENT_MEETING_RAW)
    banned = {"shorts", "sandals", "casual_sandals", "hoodies", "sleepwear",
              "beachwear", "boxers", "athletic_sneakers"}
    assert not (set(b["required_slots"]) & banned)
    assert not (set(b["optional_slots"]) & banned)
