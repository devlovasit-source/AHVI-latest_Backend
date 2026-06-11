"""Preview-stage Gemini metadata validation in wardrobe capture.

The validator agent already protected the save flow; these tests pin the
new preview wiring: risky/low-confidence items get validated BEFORE the
user sees the preview, image fields survive, and failures never break the
preview response.
"""

from __future__ import annotations

import asyncio

import pytest

from routers import wardrobe_capture as wc


def _preview_item(**overrides):
    base = {
        "item_id": "itm_123",
        "name": "Red Saree",
        "category": "Accessories",
        "sub_category": "Scarf",
        "confidence": 0.9,
        "occasions": [],
        "bbox": [0, 0, 10, 10],
        "upload_error": "",
        "raw_url": "https://r2/raw.png",
        "rawUrl": "https://r2/raw.png",
        "masked_url": "https://r2/masked.png",
        "maskedUrl": "https://r2/masked.png",
        "normalized_url": "https://r2/norm.png",
        "normalizedUrl": "https://r2/norm.png",
        "image_url": "https://r2/norm.png",
        "imageUrl": "https://r2/norm.png",
        "raw_image_base64": "abc",
        "masked_image_base64": "def",
    }
    base.update(overrides)
    return base


def _run(coro):
    return asyncio.run(coro)


# ---------- _should_validate_preview_with_gemini ----------

def test_preview_validator_called_for_low_confidence(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")
    monkeypatch.setenv("AGENT_METADATA_LOW_CONFIDENCE_THRESHOLD", "0.65")
    should, reason = wc._should_validate_preview_with_gemini(
        _preview_item(name="Blue Shirt", category="Tops", sub_category="Shirt"),
        {}, "Blue Shirt", "Tops", "Shirt", confidence=0.4,
    )
    assert should is True
    assert reason.startswith("low_confidence")


def test_preview_validator_not_called_for_high_confidence_safe_item(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")
    should, reason = wc._should_validate_preview_with_gemini(
        _preview_item(name="Blue Shirt", category="Tops", sub_category="Shirt"),
        {}, "Blue Shirt", "Tops", "Shirt", confidence=0.95,
    )
    assert should is False
    assert reason == "high_confidence_safe"


def test_preview_validator_triggered_by_risky_terms(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")
    for risky_name in ("Red Saree", "One-Piece Dress", "Boxer Shorts", "Hanger"):
        should, reason = wc._should_validate_preview_with_gemini(
            _preview_item(name=risky_name), {}, risky_name,
            "Tops", "Shirt", confidence=0.99,
        )
        assert should is True, risky_name
        assert reason.startswith("risky_term"), risky_name


def test_preview_validator_disabled_when_flag_off(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "false")
    should, reason = wc._should_validate_preview_with_gemini(
        _preview_item(), {}, "Red Saree", "Accessories", "Scarf", confidence=0.1,
    )
    assert should is False
    assert reason == "disabled"


# ---------- merge + full preview validator flow ----------

def test_validator_corrects_saree_preview(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    async def _fake_validate(item, user_id=None, vision_result=None, context=None):
        assert context["stage"] == "capture_preview"
        return {
            "category": "Dresses",
            "subcategory": "Saree",
            "confidence": 0.92,
            "allowed_occasions": ["wedding", "festive"],
            "blocked_occasions": ["gym"],
            "risk_flags": [],
            "formality": "festive",
            "style_role": "statement",
            "styling_notes": ["Drape cleanly for festive events."],
        }

    import services.agent_metadata_validator as v

    monkeypatch.setattr(v, "validate_wardrobe_metadata", _fake_validate)
    item = _preview_item()  # Ollama said Accessories/Scarf
    merged, state = _run(
        wc._apply_preview_metadata_validator(
            item, user_id="u1", vision={}, raw_label="Red Saree"
        )
    )
    assert state == "used"
    assert merged["category"] == "Dresses"
    assert merged["sub_category"] == "Saree"
    assert merged["gemini_confidence"] == 0.92
    assert merged["occasions"] == ["wedding", "festive"]
    assert merged["excludedContexts"] == ["gym"]
    assert merged["metadata_validator"]["used"] is True


def test_image_fields_preserved_after_validator(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    async def _fake_validate(item, user_id=None, vision_result=None, context=None):
        return {"category": "Dresses", "subcategory": "Saree", "confidence": 0.9}

    import services.agent_metadata_validator as v

    monkeypatch.setattr(v, "validate_wardrobe_metadata", _fake_validate)
    item = _preview_item()
    merged, _ = _run(
        wc._apply_preview_metadata_validator(
            item, user_id="u1", vision={}, raw_label="Red Saree"
        )
    )
    for field in (
        "raw_url", "rawUrl", "masked_url", "maskedUrl",
        "normalized_url", "normalizedUrl", "image_url", "imageUrl",
        "raw_image_base64", "masked_image_base64",
    ):
        assert merged[field] == item[field], field
    assert merged["item_id"] == "itm_123"


def test_validator_failure_does_not_break_preview(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    async def _boom(item, user_id=None, vision_result=None, context=None):
        raise RuntimeError("gemini down")

    import services.agent_metadata_validator as v

    monkeypatch.setattr(v, "validate_wardrobe_metadata", _boom)
    item = _preview_item()
    merged, state = _run(
        wc._apply_preview_metadata_validator(
            item, user_id="u1", vision={}, raw_label="Red Saree"
        )
    )
    assert state == "failed"
    # Local metadata intact.
    assert merged["category"] == "Accessories"
    assert merged["item_id"] == "itm_123"
    assert merged["metadata_validator"]["used"] is False


def test_validator_empty_result_keeps_local_metadata(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    async def _empty(item, user_id=None, vision_result=None, context=None):
        return {}

    import services.agent_metadata_validator as v

    monkeypatch.setattr(v, "validate_wardrobe_metadata", _empty)
    item = _preview_item(category="Tops", sub_category="Shirt", name="Boxer Shorts")
    merged, state = _run(
        wc._apply_preview_metadata_validator(
            item, user_id="u1", vision={}, raw_label="Boxer Shorts"
        )
    )
    assert state == "used"
    # "unknown"/empty values must not clobber local metadata; the guard may
    # still recategorize private wear, which is desired behavior.
    assert merged["item_id"] == "itm_123"


# ---------- hard taxonomy / suitability guards (existing systems) ----------

def test_saree_preview_never_accessory():
    item = _preview_item(name="Red Saree", category="Accessories", sub_category="Scarf")
    normalized = wc._normalize_capture_preview_item(item)
    assert normalized["category"] == "Dresses"
    assert "saree" in str(normalized.get("sub_category", "")).lower()


def test_one_piece_preview_maps_to_dress():
    item = _preview_item(name="Black One-Piece", category="Tops", sub_category="Top")
    normalized = wc._normalize_capture_preview_item(item)
    assert normalized["category"] == "Dresses"


def test_boxers_preview_private_wear():
    item = _preview_item(name="Cotton Boxers", category="Bottoms", sub_category="Shorts")
    normalized = wc._normalize_capture_preview_item(item)
    assert normalized.get("privateWear") is True
    assert normalized.get("publicWear") is False
    assert normalized.get("styleEligible") is False


def test_sleepwear_preview_private_wear():
    item = _preview_item(name="Pyjama Set", category="Bottoms", sub_category="Pants")
    normalized = wc._normalize_capture_preview_item(item)
    assert normalized.get("privateWear") is True
    assert normalized.get("styleEligible") is False
