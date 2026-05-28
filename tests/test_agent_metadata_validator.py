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
    assert meta["category"] == "Bottoms"
    assert meta["subcategory"] == "Trousers"


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
    assert meta["category"] == "Bottoms"
    assert meta["style_role"] == "loungewear"
    assert "client_meeting" in meta["blocked_occasions"]
    assert "office" in meta["blocked_occasions"]


def test_slides_metadata_shape():
    meta = _meta(
        category="Footwear",
        subcategory="Slides",
        style_role="casualwear",
        blocked_occasions=["office", "client_meeting", "formal_event", "wedding"],
    )
    assert meta["category"] == "Footwear"
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
    assert meta["category"] == "Outerwear"
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
    assert "office" in meta["blocked_occasions"]


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
