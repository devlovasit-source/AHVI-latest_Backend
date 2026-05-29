"""Tests for the AHVI Style Orchestrator integration layer.

These tests target the deterministic pieces — schema validation, the
quality-guard agent rules, and the agent-aware scoring deltas. They do not
hit any external Gemini / Agent Studio endpoint.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from services.agent_style_orchestrator import (
    default_agent_payload,
    merge_agent_payload_into_context,
    orchestrate_style_request_sync,
    validate_agent_style_payload,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_validate_agent_style_payload_defaults_for_empty():
    payload = validate_agent_style_payload(None)
    assert payload["occasion"] == "casual"
    assert payload["sub_intent"] == "outfit_generation"
    assert payload["formality"] == "mid"
    assert payload["style_direction"] == "elevated_basics"
    assert payload["wardrobe_usage"] == "preferred"
    assert payload["avoid_items"] == []
    assert payload["required_slots"] == ["top", "bottom", "footwear"]
    assert payload["palette_direction"] == []
    assert payload["accessory_policy"] == {}
    assert payload["clarification_needed"] is False
    assert payload["clarification_reason"] == ""
    assert payload["confidence"] == 0.0


def test_validate_agent_style_payload_coerces_malformed_fields():
    payload = validate_agent_style_payload(
        {
            "occasion": " office ",
            "avoid_items": "boxers, slides",
            "required_slots": ["top"],
            "palette_direction": None,
            "accessory_policy": "not a dict",
            "clarification_needed": "yes",
            "confidence": "1.7",
        }
    )
    assert payload["occasion"] == "office"
    assert payload["avoid_items"] == ["boxers", "slides"]
    assert payload["required_slots"] == ["top"]
    assert payload["palette_direction"] == []
    assert payload["accessory_policy"] == {}
    assert payload["clarification_needed"] is True
    assert payload["confidence"] == 1.0


def test_validate_agent_style_payload_handles_garbage_input():
    # Should never raise — backend safety guarantee.
    payload = validate_agent_style_payload("totally not a dict")
    assert payload == default_agent_payload()


def test_orchestrate_sync_disabled_returns_default(monkeypatch):
    monkeypatch.delenv("ENABLE_AGENT_STYLE_ORCHESTRATOR", raising=False)
    result = orchestrate_style_request_sync(
        message="client meeting tomorrow",
        wardrobe_items=[{"name": "white shirt"}],
    )
    assert result == default_agent_payload()


def test_merge_agent_payload_into_context_promotes_signals():
    ctx: Dict[str, Any] = {"query": "office today"}
    payload = validate_agent_style_payload(
        {
            "occasion": "office",
            "sub_intent": "client_meeting",
            "formality": "high",
            "style_direction": "corporate_office",
            "avoid_items": ["boxers"],
            "required_slots": ["top", "bottom", "footwear", "outerwear"],
            "palette_direction": ["navy"],
            "accessory_policy": {"allow_open_toe_footwear": False},
            "confidence": 0.9,
        }
    )
    merged = merge_agent_payload_into_context(ctx, payload)
    assert merged["agent_orchestration"]["occasion"] == "office"
    assert merged["avoid_items"] == ["boxers"]
    assert merged["required_slots"] == ["top", "bottom", "footwear", "outerwear"]
    assert merged["palette_direction"] == ["navy"]
    assert merged["accessory_policy"] == {"allow_open_toe_footwear": False}


def test_low_confidence_agent_does_not_downgrade_router_intent():
    """Router detected office/client_meeting. Agent returns casual at 0.0
    confidence (fallback default). Merge must preserve router values."""
    ctx: Dict[str, Any] = {
        "occasion": "office",
        "sub_intent": "client_meeting",
    }
    payload = validate_agent_style_payload(
        {
            "occasion": "casual",
            "sub_intent": "outfit_generation",
            "confidence": 0.0,
            "avoid_items": ["slides"],
        }
    )
    merged = merge_agent_payload_into_context(ctx, payload)
    assert merged["occasion"] == "office"
    assert merged["sub_intent"] == "client_meeting"
    # Non-conflicting hints still flow through.
    assert merged["avoid_items"] == ["slides"]


def test_high_confidence_agent_overrides_router():
    ctx: Dict[str, Any] = {"occasion": "office"}
    payload = validate_agent_style_payload(
        {"occasion": "date_night", "confidence": 0.85}
    )
    merged = merge_agent_payload_into_context(ctx, payload)
    assert merged["occasion"] == "date_night"


# ---------------------------------------------------------------------------
# Quality guard agent rules
# ---------------------------------------------------------------------------

from brain.engines.outfit_quality_guard import guard_outfit  # noqa: E402


def _outfit(top: str, bottom: str, footwear: str, **kwargs: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "top": {"name": top, "category": top},
        "bottom": {"name": bottom, "category": bottom},
        "footwear": {"name": footwear, "category": footwear},
    }
    out.update(kwargs)
    return out


def test_client_meeting_blocks_boxers():
    outfit = _outfit("white shirt", "boxers", "leather loafers")
    outfit["agent_orchestration"] = validate_agent_style_payload(
        {"occasion": "client meeting", "formality": "high", "avoid_items": ["boxers"]}
    )
    allowed, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"}, intent="client meeting", query="client meeting"
    )
    assert allowed is False
    assert any("boxer" in r.lower() or "loungewear" in r.lower() or "avoid" in r.lower() for r in reasons)


def test_client_meeting_blocks_slides():
    outfit = _outfit("white shirt", "tailored trousers", "leather slides")
    outfit["agent_orchestration"] = validate_agent_style_payload(
        {"occasion": "client meeting", "formality": "high"}
    )
    allowed, _, reasons, _ = guard_outfit(
        outfit,
        user_profile={"gender": "male"},
        intent="client meeting",
        query="client meeting",
    )
    assert allowed is False
    assert any("open-toe" in r.lower() or "professional" in r.lower() or "relax" in r.lower() for r in reasons)


def test_formal_event_blocks_gymwear():
    outfit = _outfit("compression top", "track pants", "leather loafers")
    outfit["agent_orchestration"] = validate_agent_style_payload(
        {"occasion": "wedding", "formality": "very_high"}
    )
    allowed, _, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"}, intent="wedding", query="wedding reception"
    )
    assert allowed is False
    assert any("gym" in r.lower() or "formal" in r.lower() for r in reasons)


def test_workout_allows_activewear_but_blocks_formalwear():
    workout = {
        "top": {"name": "training tee", "category": "activewear"},
        "bottom": {"name": "track pants", "category": "activewear"},
        "footwear": {"name": "running shoes", "category": "running shoes"},
        "agent_orchestration": validate_agent_style_payload(
            {"occasion": "workout", "formality": "low"}
        ),
    }
    allowed, _, _, _ = guard_outfit(
        workout, user_profile={"gender": "male"}, intent="workout", query="gym workout"
    )
    assert allowed is True

    formal_at_workout = {
        "top": {"name": "tuxedo jacket", "category": "blazer"},
        "bottom": {"name": "formal trousers", "category": "trousers"},
        "footwear": {"name": "patent leather shoes", "category": "formal shoes"},
        "agent_orchestration": validate_agent_style_payload(
            {
                "occasion": "workout",
                "formality": "low",
                "avoid_items": ["tuxedo", "patent leather"],
            }
        ),
    }
    allowed2, _, reasons2, _ = guard_outfit(
        formal_at_workout, user_profile={"gender": "male"}, intent="workout", query="gym session"
    )
    assert allowed2 is False
    assert reasons2


def test_duplicate_accessories_rejected():
    outfit = _outfit("white shirt", "tailored trousers", "leather loafers")
    outfit["accessories"] = [
        {"name": "leather watch", "category": "watch"},
        {"name": "steel watch", "category": "watch"},
    ]
    outfit["agent_orchestration"] = validate_agent_style_payload(
        {"occasion": "office"}
    )
    allowed, _, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"}, intent="office", query="office today"
    )
    assert allowed is False
    assert any("duplicate" in r.lower() for r in reasons)


def test_malformed_agent_payload_does_not_crash_guard():
    outfit = _outfit("white shirt", "tailored trousers", "loafers")
    outfit["agent_orchestration"] = "totally not a dict"
    # Should not raise; should fall through to legacy guard rules.
    allowed, _, _, _ = guard_outfit(
        outfit, user_profile={"gender": "male"}, intent="office", query="office"
    )
    assert isinstance(allowed, bool)


# ---------------------------------------------------------------------------
# Fallback prevention — when wardrobe + occasion + agent payload present,
# clarification path should be skipped.
# ---------------------------------------------------------------------------

def test_clarification_blocked_when_agent_supplies_intent(monkeypatch):
    import services.style_flow_service as sfs

    captured: Dict[str, Any] = {}

    def fake_interpret_occasion(query, ctx):
        return {
            "resolved_brief": "ambiguous",
            "ask_user": True,
            "question": "Which brief?",
            "chips": [{"label": "A", "value": "a"}],
            "board_generation_notes": {"occasion_kind": "date", "reason": "test"},
            "occasion": "date_night",
            "confidence": "medium",
        }

    def fake_get_daily_outfits(payload):
        captured["payload"] = payload
        return {
            "intent": "daily_outfit",
            "outfits": [],
            "cards": [],
            "boards": [],
            "normalized_wardrobe": payload.get("wardrobe"),
            "context": "ok",
            "meta": {},
        }

    monkeypatch.setattr(sfs, "interpret_occasion", fake_interpret_occasion)
    # Inject the agent payload by faking the orchestrator: this is what
    # routers/chat.py + style_flow_service.py will produce at runtime.
    monkeypatch.setattr(sfs, "_agent_style_enabled", lambda: True)
    monkeypatch.setattr(
        sfs,
        "_agent_orchestrate_sync",
        lambda **kwargs: validate_agent_style_payload(
            {
                "occasion": "date_night",
                "sub_intent": "dinner_date",
                "clarification_needed": False,
                "confidence": 0.85,
            }
        ),
    )
    # Stub out the heavy pipeline call.
    import brain.outfit_pipeline as op

    monkeypatch.setattr(op, "get_daily_outfits", fake_get_daily_outfits)

    wardrobe = [
        {"name": "white shirt", "category": "shirt"},
        {"name": "tailored trousers", "category": "trousers"},
        {"name": "loafers", "category": "loafers"},
    ]
    resp = sfs.build_style_flow_response(
        user_id="u1",
        query="dinner tonight",
        wardrobe=wardrobe,
        user_profile={"gender": "male"},
        context={"weather": {"condition": "clear"}},
    )
    assert resp.get("type") != "clarification"
    assert "payload" in captured  # pipeline ran instead of asking the user


def test_rainy_office_avoid_terms_filter_wardrobe_in_pipeline(monkeypatch):
    """The pipeline should drop suede/canvas/open-toe items when the agent
    lists them in avoid_items for a rainy office brief."""
    from brain.outfit_pipeline import _ahvi_apply_agent_avoid_filter

    wardrobe = [
        {"name": "suede loafers", "category": "loafers", "material": "suede"},
        {"name": "canvas sneakers", "category": "sneakers", "material": "canvas"},
        {"name": "leather chelsea boots", "category": "boots", "material": "leather"},
    ]
    context = {
        "agent_orchestration": validate_agent_style_payload(
            {
                "occasion": "office",
                "avoid_items": ["suede", "canvas", "open-toe"],
            }
        )
    }
    filtered = _ahvi_apply_agent_avoid_filter(wardrobe, context)
    names = {item["name"] for item in filtered}
    assert "leather chelsea boots" in names
    assert "suede loafers" not in names
    assert "canvas sneakers" not in names
