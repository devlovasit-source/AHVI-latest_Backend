"""Tests for the AHVI Metadata Validator integration."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from services.agent_metadata_validator import (
    default_metadata,
    fetch_style_metadata_docs_for_user,
    merge_style_metadata_into_wardrobe_items,
    validate_metadata_payload,
    validate_wardrobe_metadata_sync,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_validate_metadata_payload_defaults():
    meta = validate_metadata_payload(None)
    assert meta["category"] == "unknown"
    assert meta["subcategory"] == "unknown"
    assert meta["formality"] == "casual"
    assert meta["style_role"] == "casualwear"
    assert meta["allowed_occasions"] == []
    assert meta["blocked_occasions"] == []
    assert meta["layering_role"] == "standalone"
    assert meta["silhouette_type"] == "unknown"
    assert meta["compatible_footwear"] == []
    assert meta["suitable_seasons"] == []
    assert meta["confidence"] == 0.0


def test_validate_metadata_uses_base_item_for_defaults():
    meta = validate_metadata_payload(
        {}, base_item={"category": "Bottoms", "sub_category": "Trousers"}
    )
    assert meta["category"] == "bottom"
    assert "trousers" in meta["subcategory"]
    assert meta["client_meeting_score"] >= 0.70


def test_validate_metadata_coerces_malformed_input():
    meta = validate_metadata_payload(
        {
            "category": "  Footwear  ",
            "blocked_occasions": "office, client_meeting",
            "compatible_footwear": None,
            "confidence": "1.6",
        }
    )
    assert meta["category"] == "Footwear"
    assert meta["blocked_occasions"] == ["office", "client_meeting"]
    assert meta["compatible_footwear"] == []
    assert meta["confidence"] == 1.0


def test_validate_metadata_does_not_crash_on_garbage():
    meta = validate_metadata_payload("not a dict")
    assert meta == default_metadata()


def test_low_confidence_sets_manual_review_flag():
    meta = validate_metadata_payload({"confidence": 0.2})
    assert meta.get("manual_review_required") is True


def test_high_confidence_no_manual_review_flag():
    meta = validate_metadata_payload({"confidence": 0.95})
    assert "manual_review_required" not in meta


def test_sync_validator_disabled_returns_default(monkeypatch):
    monkeypatch.delenv("ENABLE_AGENT_METADATA_VALIDATOR", raising=False)
    meta = validate_wardrobe_metadata_sync(item={"category": "Footwear"})
    assert meta["category"] == "Footwear"


# ---------------------------------------------------------------------------
# Garment-specific metadata expectations
# ---------------------------------------------------------------------------

def _meta(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "category": "unknown",
        "subcategory": "unknown",
        "formality": "casual",
        "style_role": "casualwear",
        "allowed_occasions": [],
        "blocked_occasions": [],
        "layering_role": "standalone",
        "silhouette_type": "unknown",
        "compatible_footwear": [],
        "incompatible_footwear": [],
        "compatible_accessories": [],
        "incompatible_accessories": [],
        "suitable_seasons": [],
        "climate_appropriateness": [],
        "material_characteristics": [],
        "confidence": 0.9,
    }
    base.update(overrides)
    return validate_metadata_payload(base)


def test_boxer_shorts_metadata_shape():
    meta = _meta(
        category="Bottoms",
        subcategory="Boxer Shorts",
        style_role="loungewear",
        formality="homewear",
        blocked_occasions=["office", "client_meeting", "formal_event"],
    )
    assert meta["category"] == "innerwear"
    assert meta["style_role"] == "loungewear"
    assert "client_meeting" in meta["blocked_occasions"]
    assert "office" in meta["blocked_occasions"]
    assert "boardroom" in meta["blocked_occasions"]


def test_slides_metadata_shape():
    meta = _meta(
        category="Footwear",
        subcategory="Slides",
        style_role="casualwear",
        blocked_occasions=["office", "client_meeting", "formal_event", "wedding"],
    )
    assert meta["category"] == "footwear"
    assert "office" in meta["blocked_occasions"]
    assert "wedding" in meta["blocked_occasions"]


def test_blazer_metadata_shape():
    meta = _meta(
        category="Outerwear",
        subcategory="Blazer",
        style_role="businesswear",
        formality="business_casual",
        allowed_occasions=["office", "client_meeting", "interview"],
    )
    assert meta["category"] == "outerwear"
    assert "office" in meta["allowed_occasions"]
    assert "client_meeting" in meta["allowed_occasions"]


def test_running_shorts_metadata_shape():
    meta = _meta(
        category="Bottoms",
        subcategory="Running Shorts",
        style_role="activewear",
        formality="casual",
        blocked_occasions=["office", "client_meeting", "formal_event", "wedding"],
    )
    assert meta["style_role"] == "activewear"
    assert meta["formality"] == "athletic"
    assert "office" in meta["blocked_occasions"]
    assert "boardroom" in meta["blocked_occasions"]


def test_white_trousers_professional_defaults():
    meta = validate_metadata_payload(
        {"formality": "casual", "style_role": "casualwear", "confidence": 0.9},
        base_item={"name": "White Trousers", "category": "Bottoms", "sub_category": "Trousers"},
    )
    assert meta["category"] == "bottom"
    assert "trousers" in meta["subcategory"]
    assert meta["formality"] in {"smart_casual", "business_casual", "formal"}
    assert meta["style_role"] == "businesswear"
    assert meta["client_meeting_score"] >= 0.70
    assert meta["capsule_score"] >= 0.70
    assert "too_casual_for_professional" not in meta["risk_flags"]


def test_khaki_trousers_allow_professional_travel_and_dinner():
    meta = validate_metadata_payload(
        {"confidence": 0.9},
        base_item={"name": "Khaki Trousers", "category": "Bottoms", "sub_category": "Trousers"},
    )
    assert meta["subcategory"] in {"trousers", "chinos"}
    assert meta["formality"] in {"smart_casual", "business_casual"}
    for occasion in ("office", "client_meeting", "travel", "dinner"):
        assert occasion in meta["allowed_occasions"]


def test_formal_trousers_preserve_business_intelligence():
    meta = validate_metadata_payload(
        {"formality": "casual", "style_role": "casualwear", "confidence": 0.9},
        base_item={"name": "Formal trousers", "category": "Bottoms", "sub_category": "Formal trousers"},
    )
    assert meta["subcategory"] == "formal_trousers"
    assert meta["formality"] in {"formal", "business_casual"}
    assert meta["style_role"] == "businesswear"
    assert meta["boardroom_score"] >= 0.75


def test_loud_graphic_shirt_gets_professional_risk_flags():
    meta = validate_metadata_payload(
        {
            "category": "Tops",
            "subcategory": "Graphic Shirt",
            "name": "Loud Graphic Shirt",
            "confidence": 0.9,
        }
    )
    assert meta["visual_noise"] == "high"
    assert meta["pattern_intensity"] == "loud"
    assert meta["statement_level"] in {"statement", "risky"}
    assert meta["client_meeting_score"] < 0.50
    assert "high_visual_noise" in meta["risk_flags"] or "too_casual_for_professional" in meta["risk_flags"]


def test_zero_score_white_trousers_get_deterministic_scores():
    meta = validate_metadata_payload(
        {"confidence": 0.0},
        base_item={"name": "White Trousers", "category": "Bottoms", "sub_category": "Trousers"},
    )
    assert meta["professionalism_score"] == pytest.approx(0.80)
    assert meta["client_meeting_score"] == pytest.approx(0.75)
    assert meta["boardroom_score"] == pytest.approx(0.65)
    assert meta["capsule_score"] == pytest.approx(0.80)
    assert meta["versatility_score"] == pytest.approx(0.85)
    assert meta["date_night_score"] == pytest.approx(0.55)
    assert meta.get("manual_review_required") is not True


def test_zero_score_blue_button_down_gets_deterministic_scores():
    meta = validate_metadata_payload(
        {"confidence": 0.0},
        base_item={"name": "Blue Button Down Shirt", "category": "Tops", "sub_category": "Button-up"},
    )
    assert meta["professionalism_score"] == pytest.approx(0.75)
    assert meta["client_meeting_score"] == pytest.approx(0.72)
    assert meta["boardroom_score"] == pytest.approx(0.60)
    assert meta["capsule_score"] == pytest.approx(0.75)
    assert meta["versatility_score"] == pytest.approx(0.82)
    assert meta["date_night_score"] == pytest.approx(0.60)
    assert meta.get("manual_review_required") is not True


def test_zero_score_graphic_tee_gets_low_professional_scores():
    meta = validate_metadata_payload(
        {"confidence": 0.0},
        base_item={"name": "Graphic Tee", "category": "Tops", "sub_category": "T-Shirt"},
    )
    assert meta["professionalism_score"] == pytest.approx(0.10)
    assert meta["client_meeting_score"] == pytest.approx(0.05)
    assert meta["boardroom_score"] == pytest.approx(0.00)
    assert meta["capsule_score"] == pytest.approx(0.25)
    assert meta["versatility_score"] == pytest.approx(0.35)
    assert meta["date_night_score"] == pytest.approx(0.20)
    assert "too_casual_for_professional" in meta["risk_flags"]


def test_zero_score_blazer_gets_deterministic_scores():
    meta = validate_metadata_payload(
        {"confidence": 0.0},
        base_item={"name": "Navy Blazer", "category": "Outerwear", "sub_category": "Blazer"},
    )
    assert meta["professionalism_score"] == pytest.approx(0.95)
    assert meta["client_meeting_score"] == pytest.approx(0.90)
    assert meta["boardroom_score"] == pytest.approx(0.90)
    assert meta["capsule_score"] == pytest.approx(0.75)
    assert meta["date_night_score"] == pytest.approx(0.65)


def test_zero_score_loafers_get_deterministic_scores():
    meta = validate_metadata_payload(
        {"confidence": 0.0},
        base_item={"name": "Brown Loafers", "category": "Footwear", "sub_category": "Loafers"},
    )
    assert meta["professionalism_score"] == pytest.approx(0.85)
    assert meta["client_meeting_score"] == pytest.approx(0.80)
    assert meta["boardroom_score"] == pytest.approx(0.75)
    assert meta["capsule_score"] == pytest.approx(0.70)
    assert meta["versatility_score"] == pytest.approx(0.70)
    assert meta["date_night_score"] == pytest.approx(0.65)


def test_zero_score_cap_gets_deterministic_scores():
    meta = validate_metadata_payload(
        {"confidence": 0.0},
        base_item={"name": "Baseball Cap", "category": "Accessories", "sub_category": "Cap"},
    )
    assert meta["professionalism_score"] == pytest.approx(0.10)
    assert meta["client_meeting_score"] == pytest.approx(0.05)
    assert meta["boardroom_score"] == pytest.approx(0.00)
    assert meta["capsule_score"] == pytest.approx(0.35)
    assert meta["versatility_score"] == pytest.approx(0.45)
    assert meta["date_night_score"] == pytest.approx(0.20)


def test_nonzero_agent_scores_are_preserved():
    meta = validate_metadata_payload(
        {
            "category": "Bottoms",
            "subcategory": "Trousers",
            "professionalism_score": 0.42,
            "client_meeting_score": 0.41,
            "boardroom_score": 0.40,
            "capsule_score": 0.39,
            "versatility_score": 0.38,
            "date_night_score": 0.37,
            "confidence": 0.9,
        },
        base_item={"name": "White Trousers", "category": "Bottoms", "sub_category": "Trousers"},
    )
    assert meta["professionalism_score"] == pytest.approx(0.42)
    assert meta["client_meeting_score"] == pytest.approx(0.41)
    assert meta["boardroom_score"] == pytest.approx(0.40)
    assert meta["capsule_score"] == pytest.approx(0.39)
    assert meta["versatility_score"] == pytest.approx(0.38)
    assert meta["date_night_score"] == pytest.approx(0.37)


# ---------------------------------------------------------------------------
# Wardrobe enrichment
# ---------------------------------------------------------------------------

def test_merge_style_metadata_attaches_parsed_meta():
    items: List[Dict[str, Any]] = [
        {"$id": "item_1", "name": "White Shirt", "category": "Tops"},
        {"$id": "item_2", "name": "Slides", "category": "Footwear"},
    ]
    meta_docs = [
        {
            "item_id": "item_2",
            "style_metadata": json.dumps(
                {"category": "Footwear", "blocked_occasions": ["office"], "confidence": 0.9}
            ),
        }
    ]
    enriched = merge_style_metadata_into_wardrobe_items(items, meta_docs)
    # Existing fields preserved.
    assert enriched[0]["$id"] == "item_1"
    assert enriched[0]["name"] == "White Shirt"
    # First item has no metadata — untouched.
    assert "style_metadata" not in enriched[0]
    # Second item enriched.
    assert isinstance(enriched[1]["style_metadata"], dict)
    assert "office" in enriched[1]["style_metadata"]["blocked_occasions"]


def test_merge_style_metadata_no_docs_returns_items_unchanged():
    items = [{"$id": "x", "name": "Tee"}]
    out = merge_style_metadata_into_wardrobe_items(items, [])
    assert out == items


def test_merge_style_metadata_handles_invalid_json_doc():
    items = [{"$id": "x"}]
    out = merge_style_metadata_into_wardrobe_items(
        items, [{"item_id": "x", "style_metadata": "not json"}]
    )
    assert "style_metadata" not in out[0]


# ---------------------------------------------------------------------------
# Quality-guard integration with style_metadata
# ---------------------------------------------------------------------------

from brain.engines.outfit_quality_guard import guard_outfit  # noqa: E402


def test_guard_blocks_item_with_blocked_occasion_in_style_metadata():
    outfit = {
        "top": {"name": "white shirt", "category": "shirt"},
        "bottom": {
            "name": "boxer shorts",
            "category": "shorts",
            "style_metadata": {
                "category": "Bottoms",
                "style_role": "loungewear",
                "formality": "homewear",
                "blocked_occasions": ["office", "client_meeting"],
                "confidence": 0.95,
            },
        },
        "footwear": {"name": "leather loafers", "category": "loafers"},
    }
    allowed, _, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"}, intent="client_meeting", query="client meeting"
    )
    assert allowed is False
    assert any("blocked_occasions" in r or "homewear" in r for r in reasons)


def test_guard_blocks_activewear_for_formal():
    outfit = {
        "top": {"name": "training top", "category": "activewear"},
        "bottom": {
            "name": "running shorts",
            "category": "shorts",
            "style_metadata": {
                "category": "Bottoms",
                "style_role": "activewear",
                "formality": "casual",
                "confidence": 0.9,
            },
        },
        "footwear": {"name": "leather loafers", "category": "loafers"},
    }
    allowed, _, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"}, intent="wedding", query="formal wedding"
    )
    assert allowed is False
    assert any("activewear" in r.lower() or "formal" in r.lower() for r in reasons)


def test_guard_allows_outfit_when_no_style_metadata_present():
    outfit = {
        "top": {"name": "white shirt", "category": "shirt"},
        "bottom": {"name": "tailored trousers", "category": "trousers"},
        "footwear": {"name": "leather loafers", "category": "loafers"},
    }
    allowed, _, _, _ = guard_outfit(
        outfit, user_profile={"gender": "male"}, intent="office", query="office today"
    )
    assert allowed is True


# ---------------------------------------------------------------------------
# Wardrobe persistence agent merge (no real Appwrite call)
# ---------------------------------------------------------------------------

def test_persistence_payload_merges_agent_metadata(monkeypatch):
    from services import wardrobe_persistence_service as wps

    monkeypatch.setattr(wps, "_agent_metadata_enabled", lambda: True)
    monkeypatch.setattr(
        wps,
        "_agent_validate_metadata_sync",
        lambda **kwargs: validate_metadata_payload(
            {
                "category": "Bottoms",
                "subcategory": "Boxer Shorts",
                "style_role": "loungewear",
                "formality": "homewear",
                "blocked_occasions": ["office", "client_meeting"],
                "confidence": 0.92,
            }
        ),
    )

    payload = wps._style_metadata_payload(
        item_id="item_xyz",
        user_id="user_1",
        item_payload={"name": "Boxer Shorts", "category": "Bottoms"},
    )
    assert payload["item_id"] == "item_xyz"
    assert payload["userId"] == "user_1"
    meta = json.loads(payload["style_metadata"])
    assert meta["style_role"] == "loungewear"
    assert "client_meeting" in meta["blocked_occasions"]
    assert meta["agent_validated"] is True
    assert meta["agent_confidence"] == pytest.approx(0.92)


def test_persistence_payload_works_without_agent(monkeypatch):
    from services import wardrobe_persistence_service as wps

    monkeypatch.setattr(wps, "_agent_metadata_enabled", lambda: False)
    payload = wps._style_metadata_payload(
        item_id="item_abc",
        user_id="user_2",
        item_payload={"name": "White Shirt", "category": "Tops"},
    )
    assert payload["item_id"] == "item_abc"
    # style_metadata must still be a JSON string (legacy enrichment path).
    parsed = json.loads(payload["style_metadata"])
    assert isinstance(parsed, dict)


def test_fetch_style_metadata_docs_returns_empty_when_appwrite_fails(monkeypatch):
    # No env / no real Appwrite — must return [] not raise.
    docs = fetch_style_metadata_docs_for_user("nonexistent-user-xyz")
    assert isinstance(docs, list)
