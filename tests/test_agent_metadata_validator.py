"""Tests for the AHVI Metadata Validator integration."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

from services.agent_metadata_validator import (
    APPWRITE_STRING_SOFT_LIMIT,
    default_metadata,
    fetch_style_metadata_docs_for_user,
    item_metadata_v2_reject_reason,
    merge_style_metadata_into_wardrobe_items,
    normalize_metadata_v2,
    upsert_wardrobe_style_metadata_sync,
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


def test_validator_partial_response_marks_only_meaningful_fields(monkeypatch):
    from services import agent_metadata_validator as validator

    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    async def _partial(*args, **kwargs):
        return {
            "subcategory": "unknown",
            "formality": "formal",
            "confidence": 0.91,
        }

    monkeypatch.setattr(validator, "_call_agent", _partial)
    result = asyncio.run(
        validator.validate_wardrobe_metadata(
            item={"category": "Footwear", "sub_category": "Loafers"},
            user_id="user-1",
        )
    )

    assert result["_validator_status"] == "validated"
    assert result["_validator_authoritative_fields"] == ["formality", "confidence"]
    assert result["subcategory"] == "Loafers"


@pytest.mark.parametrize(
    ("response", "reason"),
    [({}, "empty_response"), (["malformed"], "malformed_response")],
)
def test_validator_non_authoritative_response_is_safely_degraded(
    monkeypatch, response, reason
):
    from services import agent_metadata_validator as validator

    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    async def _response(*args, **kwargs):
        return response

    monkeypatch.setattr(validator, "_call_agent", _response)
    result = asyncio.run(
        validator.validate_wardrobe_metadata(
            item={"category": "Footwear", "sub_category": "Loafers"},
            user_id="user-1",
        )
    )

    assert result["subcategory"] == "Loafers"
    assert result["_validator_status"] == "degraded"
    assert result["_validator_authoritative_fields"] == []
    assert result["_validator_degraded_reason"] == reason


def test_validator_timeout_is_safely_degraded(monkeypatch):
    from services import agent_metadata_validator as validator

    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    async def _timeout(*args, **kwargs):
        raise asyncio.TimeoutError

    monkeypatch.setattr(validator, "_call_agent", _timeout)
    result = asyncio.run(
        validator.validate_wardrobe_metadata(
            item={"category": "Footwear", "sub_category": "Loafers"},
            user_id="user-1",
        )
    )

    assert result["subcategory"] == "Loafers"
    assert result["_validator_status"] == "degraded"
    assert result["_validator_degraded_reason"] == "timeout"


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


def test_metadata_v2_flags_shiny_gold_formal_shirt_from_weak_payload():
    meta = normalize_metadata_v2(
        {},
        base_item={
            "name": "Shiny Gold Formal Shirt",
            "category": "Tops",
            "sub_category": "Shirt",
            "color": "gold",
        },
    )
    assert meta["metadata_version"] == "v2"
    assert meta["shine_level"] >= 0.75
    assert meta["statement_level_score"] >= 0.75
    assert "shiny" in meta["risk_flags"]
    assert "coffee_date" in meta["avoid_for"]


def test_metadata_v2_rejects_shiny_gold_for_coffee_and_today_but_allows_wedding():
    item = {
        "id": "gold-shirt",
        "name": "Shiny Gold Formal Shirt",
        "category": "Tops",
        "sub_category": "Shirt",
        "color": "gold",
    }
    assert item_metadata_v2_reject_reason(item, occasion="coffee_date")
    assert item_metadata_v2_reject_reason(item, occasion="general_today")
    assert item_metadata_v2_reject_reason(item, occasion="wedding_guest") == ""


def test_weak_nonzero_agent_scores_do_not_override_professional_floors():
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
            "blocked_occasions": ["black_tie", "client_meeting", "boardroom"],
            "confidence": 0.9,
        },
        base_item={"name": "White Trousers", "category": "Bottoms", "sub_category": "Trousers"},
    )
    assert meta["professionalism_score"] == pytest.approx(0.75)
    assert meta["client_meeting_score"] == pytest.approx(0.70)
    assert meta["boardroom_score"] == pytest.approx(0.60)
    assert meta["capsule_score"] == pytest.approx(0.70)
    assert meta["versatility_score"] == pytest.approx(0.70)
    assert meta["date_night_score"] == pytest.approx(0.37)
    assert "client_meeting" not in meta["blocked_occasions"]
    assert "boardroom" not in meta["blocked_occasions"]
    assert "black_tie" in meta["blocked_occasions"]


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


def test_merge_style_metadata_prefers_exact_doc_over_sanitized_duplicate():
    item_id = "item-1"
    items = [{"$id": item_id, "name": "White Shirt"}]
    meta_docs = [
        {
            "$id": "item1",
            "$updatedAt": "2026-01-02T00:00:00.000Z",
            "item_id": item_id,
            "style_metadata": json.dumps({"source": "safe"}),
        },
        {
            "$id": item_id,
            "$updatedAt": "2026-01-01T00:00:00.000Z",
            "item_id": item_id,
            "style_metadata": json.dumps({"source": "exact"}),
        },
    ]

    enriched = merge_style_metadata_into_wardrobe_items(items, meta_docs)

    assert enriched[0]["style_metadata"]["source"] == "exact"


def test_metadata_json_compact_stays_under_appwrite_limit():
    from services.agent_metadata_validator import _json_compact

    raw = {
        "confidence": 0.9,
        "styling_notes": [f"note-{i}-" + ("x" * 1000) for i in range(30)],
        "material_characteristics": [f"material-{i}-" + ("y" * 1000) for i in range(30)],
        "raw_response": "z" * 20000,
    }

    compact = _json_compact(raw)

    assert APPWRITE_STRING_SOFT_LIMIT == 8500
    assert len(compact) <= 8500


class _MetadataUpsertFakeProxy:
    def __init__(
        self,
        *,
        existing_update_ids=None,
        query_docs=None,
        fail_update_ids=None,
    ):
        self.existing_update_ids = set(existing_update_ids or [])
        self.query_docs = list(query_docs or [])
        self.fail_update_ids = set(fail_update_ids or [])
        self.calls = []

    def update_document(self, resource, document_id, data):
        from services.appwrite_proxy import AppwriteProxyError

        self.calls.append(("update", resource, document_id, data))
        if document_id in self.fail_update_ids:
            raise AppwriteProxyError("Appwrite request failed (500): boom")
        if document_id not in self.existing_update_ids:
            raise AppwriteProxyError("Appwrite request failed (404): not found")
        return {"$id": document_id}

    def create_document(self, resource, data, document_id="unique()"):
        self.calls.append(("create", resource, document_id, data))
        return {"$id": document_id}

    def find_by_attribute(self, resource, attribute, value, *, user_id=None, limit=10):
        self.calls.append(("query", resource, attribute, value, user_id, limit))
        return list(self.query_docs)


def _patch_upsert_proxy(monkeypatch, proxy):
    import services.appwrite_proxy as appwrite_proxy

    monkeypatch.setattr(appwrite_proxy, "AppwriteProxy", lambda: proxy)


def test_metadata_upsert_updates_existing_exact_doc(monkeypatch):
    proxy = _MetadataUpsertFakeProxy(existing_update_ids={"item-1"})
    _patch_upsert_proxy(monkeypatch, proxy)

    result = upsert_wardrobe_style_metadata_sync("user-1", "item-1", {"confidence": 0.8})

    assert result == {"status": "updated", "doc_id": "item-1"}
    assert [c[0:3] for c in proxy.calls] == [
        ("update", "wardrobe_style_metadata", "item-1")
    ]


def test_metadata_upsert_updates_existing_sanitized_doc(monkeypatch):
    proxy = _MetadataUpsertFakeProxy(existing_update_ids={"item1"})
    _patch_upsert_proxy(monkeypatch, proxy)

    result = upsert_wardrobe_style_metadata_sync("user-1", "item-1", {"confidence": 0.8})

    assert result == {"status": "updated", "doc_id": "item1"}
    assert [c[0:3] for c in proxy.calls[:2]] == [
        ("update", "wardrobe_style_metadata", "item-1"),
        ("update", "wardrobe_style_metadata", "item1"),
    ]
    assert not any(c[0] == "create" for c in proxy.calls)


def test_metadata_upsert_updates_query_match(monkeypatch):
    proxy = _MetadataUpsertFakeProxy(
        existing_update_ids={"metadata_doc"},
        query_docs=[
            {
                "$id": "metadata_doc",
                "$updatedAt": "2026-06-01T00:00:00.000Z",
                "item_id": "item-1",
                "userId": "user-1",
            }
        ],
    )
    _patch_upsert_proxy(monkeypatch, proxy)

    result = upsert_wardrobe_style_metadata_sync("user-1", "item-1", {"confidence": 0.8})

    assert result == {"status": "updated", "doc_id": "metadata_doc"}
    assert any(c[0] == "query" for c in proxy.calls)
    assert ("update", "wardrobe_style_metadata", "metadata_doc") in [
        c[0:3] for c in proxy.calls
    ]
    assert not any(c[0] == "create" for c in proxy.calls)


def test_metadata_upsert_creates_sanitized_doc_when_no_existing_doc(monkeypatch):
    proxy = _MetadataUpsertFakeProxy()
    _patch_upsert_proxy(monkeypatch, proxy)

    result = upsert_wardrobe_style_metadata_sync("user-1", "item-1", {"confidence": 0.8})

    assert result == {"status": "created", "doc_id": "item1"}
    assert ("create", "wardrobe_style_metadata", "item1") in [
        c[0:3] for c in proxy.calls
    ]


def test_metadata_upsert_does_not_create_duplicate_when_exact_exists(monkeypatch):
    proxy = _MetadataUpsertFakeProxy(existing_update_ids={"item-1", "item1"})
    _patch_upsert_proxy(monkeypatch, proxy)

    result = upsert_wardrobe_style_metadata_sync("user-1", "item-1", {"confidence": 0.8})

    assert result["doc_id"] == "item-1"
    assert not any(c[0] == "create" for c in proxy.calls)
    assert ("update", "wardrobe_style_metadata", "item1") not in [
        c[0:3] for c in proxy.calls
    ]


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
