from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from brain.engines.outfit_quality_guard import guard_outfit
from brain.engines.style_scorer import UnifiedStyleScorer
from routers import stylist
from services import style_flow_service
from services.constrained_outfit_builder import ConstrainedOutfitBuilder
from services.professional_safety import evaluate_professional_safety
from services import style_reasoning_engine
from services.style_reasoning_engine import _best_style_assets


def _asset(asset_id: str, name: str, **extra):
    row = {
        "asset_id": asset_id,
        "$id": asset_id,
        "id": asset_id,
        "name": name,
        "category": "top",
        "role": "top",
        "subcategory": "shirt",
        "sub_category": "shirt",
        "gender": "female",
        "source": "style_asset",
        "image_url": f"https://example.test/{asset_id}.png",
        "status": "active",
        "metadata_status": "ready",
    }
    row.update(extra)
    return row


def test_explicit_false_blocks_even_with_high_scores():
    decision = evaluate_professional_safety(
        {
            "professional_safe": False,
            "professionalism_score": 1.0,
            "client_meeting_score": 1.0,
            "safety_tags": ["client_meeting"],
        },
        "client_meeting",
    )
    assert decision == {
        "allowed": False,
        "reason_code": "professional_safe_false",
        "score": 0.0,
        "evidence_source": "canonical",
    }


def test_relevant_hard_negative_beats_conflicting_positive_tag():
    decision = evaluate_professional_safety(
        {
            "professional_safe": True,
            "client_meeting_score": 0.9,
            "safety_tags": ["not_client_meeting", "client_meeting"],
        },
        "client meeting",
    )
    assert decision["allowed"] is False
    assert decision["reason_code"] == "not_client_meeting"


def test_positive_flag_does_not_override_low_boardroom_score():
    decision = evaluate_professional_safety(
        {
            "professional_safe": True,
            "professionalism_score": 0.9,
            "boardroom_score": 0.44,
            "safety_tags": ["boardroom"],
        },
        "boardroom",
    )
    assert decision["allowed"] is False
    assert decision["reason_code"] == "boardroom_score_below_threshold"


def test_cultural_professional_with_suitable_score_is_allowed():
    decision = evaluate_professional_safety(
        {
            "professional_safe": True,
            "professionalism_score": 0.72,
            "client_meeting_score": 0.60,
            "safety_tags": ["cultural", "cultural_professional"],
        },
        "client_meeting",
    )
    assert decision["allowed"] is True
    assert decision["evidence_source"] == "canonical"


def test_plaid_scarf_cannot_bypass_boardroom_hard_negative():
    decision = evaluate_professional_safety(
        {
            "name": "Black Plaid Scarf",
            "role": "accessory",
            "professional_safe": True,
            "professionalism_score": 0.70,
            "boardroom_score": 0.80,
            "safety_tags": ["office", "not_boardroom"],
        },
        "boardroom",
    )
    assert decision["allowed"] is False
    assert decision["reason_code"] == "not_boardroom"


def test_legacy_evidence_and_name_fallback_remain_compatible():
    assert evaluate_professional_safety(
        {"name": "Plain Oxford", "business_safe": True}, "office"
    )["allowed"] is True
    blocked = evaluate_professional_safety(
        {"name": "Camo Cargo Pants", "category": "bottom"}, "office"
    )
    assert blocked["allowed"] is False
    assert blocked["evidence_source"] == "fallback"


def test_non_professional_context_is_unchanged_even_with_negative_metadata():
    decision = evaluate_professional_safety(
        {"professional_safe": False, "safety_tags": ["not_professional"]},
        "date_night",
    )
    assert decision["allowed"] is True
    assert decision["reason_code"] == "not_professional_context"


def test_reasoning_selector_blocks_striped_tshirt_for_client_meeting():
    striped = _asset(
        "striped",
        "Black and White Striped T-Shirt",
        subcategory="tshirt",
        sub_category="tshirt",
        professional_safe=False,
        professionalism_score=0.35,
        client_meeting_score=0.25,
        safety_tags=["casual", "not_client_meeting"],
    )
    selected = _best_style_assets(
        [striped],
        direction={"hero_piece": "striped t-shirt", "items": ["striped t-shirt"]},
        occasion="client meeting",
        target_gender="female",
        placement="hero",
    )
    assert selected == []


def test_reasoning_selector_keeps_black_shirt_and_cultural_kurta_eligible():
    black = _asset(
        "black-shirt", "Black Shirt", gender="male", professional_safe=True,
        professionalism_score=0.85, client_meeting_score=0.82,
        safety_tags=["office", "client_meeting"],
    )
    kurta = _asset(
        "blue-kurta", "Blue Kurta", gender="male", category="one_piece",
        role="one_piece", subcategory="kurta", sub_category="kurta",
        professional_safe=True, professionalism_score=0.72,
        client_meeting_score=0.60,
        safety_tags=["cultural", "cultural_professional"],
    )
    assert _best_style_assets(
        [black], direction={"hero_piece": "black shirt"},
        occasion="client_meeting", target_gender="male", placement="hero",
    )[0]["asset_id"] == "black-shirt"
    assert evaluate_professional_safety(kurta, "client_meeting")["allowed"] is True


def test_pre_attached_visual_asset_cannot_bypass_professional_gate(monkeypatch):
    striped = _asset(
        "striped", "Black and White Striped T-Shirt",
        professional_safe=False, safety_tags=["not_client_meeting"],
    )
    monkeypatch.setattr(style_reasoning_engine, "_style_asset_rows", lambda: [striped])
    result = style_reasoning_engine._enrich_visual_directions_with_assets(
        [{
            "title": "Striped Direction",
            "hero_piece": "striped t-shirt",
            "image_url": striped["image_url"],
            "asset_id": "striped",
            "complete_the_look": [],
        }],
        occasion="client_meeting",
        target_gender="female",
    )
    assert result[0].get("asset_id") != "striped"
    assert result[0].get("image_url") != striped["image_url"]


def test_style_flow_wrapper_and_board_sanitizer_use_canonical_evaluator():
    tshirt = _asset(
        "striped", "Striped T-Shirt", professional_safe=False,
        safety_tags=["not_client_meeting"],
    )
    assert style_flow_service._is_professional_safe(tshirt, "client_meeting")[0] is False
    card, removed = style_flow_service._sanitize_office_board(
        {"items": [tshirt]}, "client_meeting"
    )
    assert card["items"] == []
    assert removed == ["Striped T-Shirt"]


def test_constrained_builder_rejects_unsafe_fixed_anchor_with_typed_failure():
    striped = _asset(
        "striped", "Striped T-Shirt", professional_safe=False,
        safety_tags=["not_client_meeting"],
    )
    result = ConstrainedOutfitBuilder().generate(
        scenario="style_this",
        fixed_items=[striped],
        style_assets=[striped],
        source_policy={"allowed_sources": ["style_asset"]},
        context={"occasion": "client_meeting"},
    )
    assert result["success"] is False
    assert result["error"]["code"] == "FIXED_ITEMS_INCOMPATIBLE"
    assert result["error"]["reason_code"] == "professional_safe_false"


def test_quality_guard_and_unified_scorer_hard_reject_canonical_negative():
    tshirt = _asset(
        "striped", "Striped T-Shirt", professional_safe=False,
        safety_tags=["not_client_meeting"],
    )
    ok, score, reasons, _ = guard_outfit(
        {"top": tshirt, "occasion": "client_meeting"},
        intent="client_meeting",
    )
    assert ok is False and score == -100
    assert reasons == ["professional_safety:professional_safe_false"]
    scored = UnifiedStyleScorer().score_outfit(
        [tshirt], {"occasion": "client_meeting"}, {}
    )
    assert scored["occasion_reject"] is True
    assert scored["score"] == 0.0


def test_connected_item_cta_returns_existing_incompatibility_contract():
    app = FastAPI()

    @app.middleware("http")
    async def add_user(request, call_next):
        request.state.user = {"user_id": "user-1"}
        return await call_next(request)

    app.include_router(stylist.router, prefix="/api/stylist")
    client = TestClient(app)
    striped = {
        **_asset(
            "striped", "Striped T-Shirt", professional_safe=False,
            safety_tags=["not_client_meeting"],
        ),
        "source": "wardrobe",
    }
    response = client.post(
        "/api/stylist/items/striped/style",
        json={
            "user_id": "user-1",
            "mode": "build_outfit",
            "occasion": "client_meeting",
            "anchor_item": striped,
            "wardrobe": [striped],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "ANCHOR_INCOMPATIBLE_WITH_OCCASION"
    assert body["anchor_blocked"] is True
