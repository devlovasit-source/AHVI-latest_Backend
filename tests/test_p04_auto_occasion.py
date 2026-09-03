"""P0.4 regressions for automatic wardrobe occasion detection.

These tests intentionally run before the implementation change. They pin the
capture preview contract, persistence parity, privacy guard, and manual-edit
precedence without calling external providers.
"""

from __future__ import annotations

import asyncio
import json

import services.wardrobe_persistence_service as persistence
from routers import wardrobe_capture as wc


def _preview_item(**overrides):
    item = {
        "item_id": "itm_123",
        "name": "Blue T-Shirt",
        "category": "Tops",
        "sub_category": "T-Shirt",
        "confidence": 0.95,
        "occasions": [],
        "label_source": "vision:gemini_multi",
        "bbox": [0, 0, 10, 10],
        "raw_url": "https://r2/raw.png",
        "masked_url": "https://r2/masked.png",
        "normalized_url": "https://r2/norm.png",
    }
    item.update(overrides)
    return item


def _run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def _patch_recorder(monkeypatch):
    calls = []

    def fake_patch(url, headers, json, timeout):
        calls.append(dict(json["data"]))
        return _FakeResponse(200, {"$id": "item_1", "userId": "user_1", **calls[-1]})

    monkeypatch.setattr(persistence.requests, "patch", fake_patch)
    monkeypatch.setattr(
        persistence, "_persist_style_metadata_nonfatal", lambda **kwargs: "updated"
    )
    return calls


def test_high_confidence_capture_gets_automatic_occasions_when_validator_skips(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    merged, state = _run(
        wc._apply_preview_metadata_validator(
            _preview_item(), user_id="u1", vision={}, raw_label="Blue T-Shirt"
        )
    )

    assert state == "skipped"
    assert merged["occasions"]
    assert "casual" in merged["occasions"]


def test_automatic_occasion_derivation_is_limited_to_fitness_threshold(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "false")

    merged, _state = _run(
        wc._apply_preview_metadata_validator(
            _preview_item(name="Navy Blazer", sub_category="Blazer"),
            user_id="u1",
            vision={},
            raw_label="Navy Blazer",
        )
    )

    assert merged["occasions"]
    assert len(merged["occasions"]) <= 4
    assert all(value in {"office", "date", "casual", "travel", "party", "wedding"}
               for value in merged["occasions"])


def test_validator_allowed_occasions_remain_authoritative(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    async def fake_validate(item, user_id=None, vision_result=None, context=None):
        return {
            "category": "Dresses",
            "subcategory": "Saree",
            "confidence": 0.92,
            "allowed_occasions": ["wedding", "festive"],
        }

    import services.agent_metadata_validator as validator

    monkeypatch.setattr(validator, "validate_wardrobe_metadata", fake_validate)
    merged, state = _run(
        wc._apply_preview_metadata_validator(
            _preview_item(name="Red Saree", category="Accessories", sub_category="Scarf"),
            user_id="u1",
            vision={},
            raw_label="Red Saree",
        )
    )

    assert state == "used"
    assert merged["occasions"] == ["wedding", "festive"]


def test_empty_validator_occasions_use_deterministic_fallback(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    async def fake_validate(item, user_id=None, vision_result=None, context=None):
        return {"category": "Tops", "subcategory": "T-Shirt", "confidence": 0.9}

    import services.agent_metadata_validator as validator

    monkeypatch.setattr(validator, "validate_wardrobe_metadata", fake_validate)
    merged, state = _run(
        wc._apply_preview_metadata_validator(
            _preview_item(confidence=0.4, label_source="vision"),
            user_id="u1",
            vision={},
            raw_label="Blue T-Shirt",
        )
    )

    assert state == "used"
    assert merged["occasions"]


def test_private_wear_never_exposes_public_occasion(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "false")
    item = _preview_item(
        name="Cotton Boxers",
        category="Bottoms",
        sub_category="Shorts",
        occasions=["work", "dinner", "travel", "party", "festive", "wedding"],
    )

    normalized = wc._normalize_capture_preview_item(item)

    assert normalized["privateWear"] is True
    assert not (
        {"work", "dinner", "travel", "party", "festive", "wedding"}
        & {str(value).lower() for value in normalized.get("occasions", [])}
    )


def test_private_wear_keeps_only_private_safe_occasion_values():
    normalized = wc._normalize_capture_preview_item(
        _preview_item(
            name="Cotton Boxers",
            category="Bottoms",
            sub_category="Shorts",
            occasions=["home", "private", "office"],
        )
    )

    assert normalized["privateWear"] is True
    assert set(normalized["occasions"]).issubset({"home", "private"})


def test_preview_and_persistence_receive_the_same_automatic_occasions(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "false")
    raw_item = _preview_item()

    preview, _state = _run(
        wc._apply_preview_metadata_validator(
            dict(raw_item), user_id="u1", vision={}, raw_label="Blue T-Shirt"
        )
    )
    doc = persistence._build_appwrite_doc(
        user_id="user_1",
        file_id="file_1",
        item=dict(raw_item),
        raw_url="https://r2/raw.png",
        masked_url="https://r2/masked.png",
        normalized_url="https://r2/norm.png",
    )

    assert preview["occasions"] == doc["occasions"]


def test_name_only_edit_does_not_replace_existing_occasions(monkeypatch):
    calls = _patch_recorder(monkeypatch)

    result = persistence.update_item_labels(
        user_id="user_1",
        item_id="item_1",
        name="Renamed Shirt",
        override_collection_id="outfits",
        override_database_id="db",
    )

    assert result["success"] is True
    assert "occasions" not in calls[0]


def test_explicit_occasion_edit_replaces_occasions(monkeypatch):
    calls = _patch_recorder(monkeypatch)

    result = persistence.update_item_labels(
        user_id="user_1",
        item_id="item_1",
        tags=["Work", "Travel"],
        override_collection_id="outfits",
        override_database_id="db",
    )

    assert result["success"] is True
    assert calls[0]["occasions"] == ["Work", "Travel"]


def test_explicit_empty_occasion_edit_clears_occasions(monkeypatch):
    calls = _patch_recorder(monkeypatch)

    result = persistence.update_item_labels(
        user_id="user_1",
        item_id="item_1",
        tags=[],
        override_collection_id="outfits",
        override_database_id="db",
    )

    assert result["success"] is True
    assert calls[0]["occasions"] == []


def _indian_occasion_pair(item):
    preview = wc._apply_deterministic_occasion_fallback(dict(item))["occasions"]
    persisted = persistence._build_appwrite_doc(
        user_id="user_1",
        file_id="indian_item",
        item=dict(item),
        raw_url="https://r2/raw.png",
        masked_url="https://r2/masked.png",
        normalized_url="https://r2/norm.png",
    )["occasions"]
    return preview, persisted


def test_saree_gets_wedding_and_party_occasion_signals():
    preview, persisted = _indian_occasion_pair(
        _preview_item(name="Red Saree", category="Traditional", sub_category="Saree")
    )
    assert "wedding" in preview and "party" in preview
    assert preview == persisted


def test_lehenga_gets_wedding_and_party_occasion_signals():
    preview, persisted = _indian_occasion_pair(
        _preview_item(name="Gold Lehenga", category="Traditional", sub_category="Lehenga")
    )
    assert "wedding" in preview and "party" in preview
    assert preview == persisted


def test_sherwani_gets_wedding_and_party_occasion_signals():
    preview, persisted = _indian_occasion_pair(
        _preview_item(name="Ivory Sherwani", category="Outerwear", sub_category="Sherwani")
    )
    assert "wedding" in preview and "party" in preview
    assert preview == persisted


def test_festive_kurta_gets_wedding_and_party_occasion_signals():
    preview, persisted = _indian_occasion_pair(
        _preview_item(
            name="Festive Kurta", category="Traditional", sub_category="Kurta"
        )
    )
    assert "wedding" in preview and "party" in preview
    assert preview == persisted


def test_plain_everyday_kurta_does_not_get_high_formality_occasions():
    preview, persisted = _indian_occasion_pair(
        _preview_item(
            name="Plain Everyday Kurta",
            category="Traditional",
            sub_category="Kurta",
        )
    )
    assert not ({"wedding", "party"} & set(preview))
    assert preview == ["casual"]
    assert preview == persisted


def test_ambiguous_generic_top_keeps_conservative_casual_fallback():
    preview, persisted = _indian_occasion_pair(
        _preview_item(name="Item", category="Item", sub_category="Item")
    )
    assert preview == ["casual"]
    assert preview == persisted
