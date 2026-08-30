"""RC3 Gap 1: explicit machine-readable insufficient_wardrobe signal.

Daily Wear's frontend previously inferred "insufficient wardrobe" from
cards.isEmpty alone, which is wrong -- any board-generation hiccup looked
identical to a genuinely too-small wardrobe. routers.chat._demo_style_board_
payload's no-cards fallback branch now distinguishes the two using the same
missing_required_slots completeness check the fixed-item path already
trusts: only a wardrobe missing top/bottom/footwear coverage entirely may
claim reason=insufficient_wardrobe. Any other empty-cards case keeps the
pre-existing generic type=missing_outfit_cards shape untouched.
"""
from __future__ import annotations

import services.llm_service as llm_service
import services.ai_gateway as ai_gateway
import services.semantic_intent_resolver as semantic_intent_resolver
import brain.intent_engine as intent_engine
import pytest

from routers import chat


@pytest.fixture(autouse=True)
def _fast_fail_llm(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("llm disabled in test")

    monkeypatch.setattr(llm_service, "generate_text", _raise)
    monkeypatch.setattr(ai_gateway, "generate_text", _raise)
    monkeypatch.setattr(semantic_intent_resolver, "_generate_text", _raise)
    monkeypatch.setattr(intent_engine, "generate_text", _raise)


def _it(name, role, source="wardrobe"):
    iid = name.lower().replace(" ", "-")
    return {
        "id": iid, "item_id": iid, "name": name, "category": role, "role": role,
        "source": source, "image_url": f"https://x/{iid}.png",
        "masked_url": f"https://x/{iid}-masked.png",
    }


def test_true_insufficient_wardrobe_emits_explicit_reason(monkeypatch):
    """A wardrobe with only tops (no bottom, no footwear) can never complete
    an outfit -- this is genuine insufficient_wardrobe, not a generic hiccup."""
    monkeypatch.setattr(
        chat, "build_style_flow_response",
        lambda **kwargs: {"success": True, "cards": [], "type": ""},
    )
    wardrobe = [_it("White Tee", "top"), _it("Blue Tee", "top")]

    result = chat._demo_style_board_payload(
        "user-1", "Build my wardrobe-first looks for today", wardrobe, resolved_occasion="daily",
    )

    assert result.get("type") == "insufficient_wardrobe"
    assert result.get("reason") == "insufficient_wardrobe"
    assert result.get("status") == "insufficient_wardrobe"
    assert result.get("meta", {}).get("reason") == "insufficient_wardrobe"
    assert result.get("cards") == []


def test_generic_no_cards_does_not_claim_insufficient_wardrobe(monkeypatch):
    """A wardrobe with full role coverage (top/bottom/footwear) that still
    produced no cards is a generation/routing hiccup, not insufficient
    wardrobe -- must keep the pre-existing generic shape."""
    monkeypatch.setattr(
        chat, "build_style_flow_response",
        lambda **kwargs: {"success": True, "cards": [], "type": ""},
    )
    wardrobe = [
        _it("Black Trousers", "bottom"), _it("Light Green Polo Shirt", "top"),
        _it("White Sneakers", "footwear"), _it("Blue Jeans", "bottom"),
    ]

    result = chat._demo_style_board_payload(
        "user-1", "Build my wardrobe-first looks for today", wardrobe, resolved_occasion="daily",
    )

    assert result.get("type") == "missing_outfit_cards"
    assert result.get("reason") != "insufficient_wardrobe"
    assert result.get("status") != "insufficient_wardrobe"


def test_populated_wardrobe_with_real_cards_is_unaffected(monkeypatch):
    """Success path shape must not change at all."""
    monkeypatch.setattr(
        chat, "build_style_flow_response",
        lambda **kwargs: {"success": True, "cards": [{"id": "c1", "items": []}], "type": "wardrobe_recommendation"},
    )
    wardrobe = [
        _it("Black Trousers", "bottom"), _it("Light Green Polo Shirt", "top"),
        _it("White Sneakers", "footwear"),
    ]

    result = chat._demo_style_board_payload(
        "user-1", "Build my wardrobe-first looks for today", wardrobe, resolved_occasion="daily",
    )

    assert result.get("cards")
    assert "reason" not in result or result.get("reason") != "insufficient_wardrobe"
