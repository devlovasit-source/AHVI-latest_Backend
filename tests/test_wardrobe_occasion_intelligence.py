"""Regressions for authoritative wardrobe occasion intelligence.

Phase 0 of the occasion-intelligence task: these tests pin the intended
behavior BEFORE implementation, so the fix is verifiably closing a real gap
rather than a guessed one.

TEST A — a normal, high-confidence capture (validator skipped) must receive
    deterministic occasions in the /analyze response, derived the same way
    the persistence layer already derives them.
TEST B — when the validator DOES run and returns allowed_occasions, its
    result remains authoritative; the deterministic fallback must not
    overwrite it.
TEST C — private-wear items must never end up with a forbidden public
    occasion (Work/Dinner/Travel/Party/Festive/Wedding), regardless of what
    deterministic inference or the validator produced.
TEST D — updating an unrelated field (e.g. name) without supplying `tags`
    must preserve whatever occasions are already stored.
TEST E — an explicit `tags` update (including an explicit empty list) must
    apply normally; the endpoint must distinguish "field omitted" from
    "field explicitly supplied", per the existing Pydantic contract
    (`tags: List[str] | None = None`).
"""

from __future__ import annotations

import asyncio
import json

import services.wardrobe_persistence_service as persistence
from routers import wardrobe_capture as wc


def _preview_item(**overrides):
    base = {
        "item_id": "itm_123",
        "name": "Item",
        "category": "Tops",
        "sub_category": "Shirt",
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


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def _patch_recorder(monkeypatch):
    """Mocks Appwrite PATCH + style-metadata upsert; returns the list of
    payload dicts sent to Appwrite (one per update_item_labels call)."""
    calls = []

    def fake_patch(url, headers, json, timeout):
        calls.append(dict(json["data"]))
        return _FakeResponse(200, {"$id": "item_1", "userId": "user_1", **calls[-1]})

    monkeypatch.setattr(persistence.requests, "patch", fake_patch)
    monkeypatch.setattr(
        persistence, "_persist_style_metadata_nonfatal", lambda **kwargs: "updated"
    )
    return calls


# ---------------------------------------------------------------------------
# TEST A — high-confidence normal garment, validator skipped
# ---------------------------------------------------------------------------


def test_high_confidence_capture_gets_deterministic_occasions(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")
    item = _preview_item(
        name="Blue T-Shirt",
        category="Tops",
        sub_category="T-Shirt",
        confidence=0.95,
        label_source="vision:gemini_multi",
    )
    merged, state = _run(
        wc._apply_preview_metadata_validator(
            item, user_id="u1", vision={}, raw_label="Blue T-Shirt"
        )
    )
    # Validator must genuinely have been skipped — this is the "normal
    # capture" case the audit proved returns occasions=[].
    assert state == "skipped"
    assert merged["occasions"], (
        "expected deterministic occasions when the validator is skipped, "
        f"got {merged['occasions']!r}"
    )
    assert "casual" in merged["occasions"]


# ---------------------------------------------------------------------------
# TEST B — validator enrichment remains authoritative
# ---------------------------------------------------------------------------


def test_validator_result_is_not_overwritten_by_deterministic_fallback(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")

    async def _fake_validate(item, user_id=None, vision_result=None, context=None):
        return {
            "category": "Dresses",
            "subcategory": "Saree",
            "confidence": 0.92,
            "allowed_occasions": ["wedding", "festive"],
        }

    import services.agent_metadata_validator as v

    monkeypatch.setattr(v, "validate_wardrobe_metadata", _fake_validate)
    item = _preview_item(name="Red Saree", category="Accessories", sub_category="Scarf")
    merged, state = _run(
        wc._apply_preview_metadata_validator(
            item, user_id="u1", vision={}, raw_label="Red Saree"
        )
    )
    assert state == "used"
    assert merged["occasions"] == ["wedding", "festive"]


# ---------------------------------------------------------------------------
# TEST C — private wear never leaks a forbidden public occasion
# ---------------------------------------------------------------------------


def test_private_wear_never_leaks_public_occasions(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "true")
    item = _preview_item(
        name="Cotton Boxers",
        category="Bottoms",
        sub_category="Shorts",
        confidence=0.95,
        label_source="vision:gemini_multi",
    )
    merged, _state = _run(
        wc._apply_preview_metadata_validator(
            item, user_id="u1", vision={}, raw_label="Cotton Boxers"
        )
    )
    # Mirrors the real /analyze pipeline: _normalize_capture_preview_item
    # runs on every item after the validator step and is the universal
    # private-wear authority.
    normalized = wc._normalize_capture_preview_item(merged)
    assert normalized["privateWear"] is True

    forbidden = {"work", "dinner", "travel", "party", "festive", "wedding"}
    occ_lower = {str(o).lower() for o in normalized.get("occasions", [])}
    assert not (occ_lower & forbidden), normalized.get("occasions")


# ---------------------------------------------------------------------------
# TEST D — unrelated label edit preserves existing occasions
# ---------------------------------------------------------------------------


def test_update_labels_unrelated_edit_preserves_occasions(monkeypatch):
    calls = _patch_recorder(monkeypatch)

    result = persistence.update_item_labels(
        user_id="user_1",
        item_id="item_1",
        name="Renamed Shirt",
        override_collection_id="outfits",
        override_database_id="db",
    )

    assert result["success"] is True
    assert "occasions" not in calls[0], (
        "a name-only edit must not send an `occasions` key at all, so "
        "Appwrite's partial PATCH leaves the stored value untouched"
    )


# ---------------------------------------------------------------------------
# TEST E — explicit occasion edit works, including explicit clear
# ---------------------------------------------------------------------------


def test_update_labels_explicit_tags_update_occasions(monkeypatch):
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


def test_update_labels_explicit_empty_tags_clears_occasions(monkeypatch):
    calls = _patch_recorder(monkeypatch)

    result = persistence.update_item_labels(
        user_id="user_1",
        item_id="item_1",
        tags=[],
        override_collection_id="outfits",
        override_database_id="db",
    )

    assert result["success"] is True
    assert "occasions" in calls[0] and calls[0]["occasions"] == [], (
        "an explicit empty tags list is the user's supported way to clear "
        "all occasions and must be applied, not silently ignored"
    )


# ---------------------------------------------------------------------------
# PHASE 5 — analyze-derived occasions must match persistence-derived
# occasions for the same item (same authority, same threshold).
# ---------------------------------------------------------------------------


def test_analyze_and_persistence_derive_the_same_occasions(monkeypatch):
    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "false")
    raw_item = {
        "item_id": "itm_999",
        "name": "Blue T-Shirt",
        "category": "Tops",
        "sub_category": "T-Shirt",
        "confidence": 0.95,
        "label_source": "vision:gemini_multi",
        "occasions": [],
    }

    analyze_merged, _state = _run(
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

    assert analyze_merged["occasions"] == doc["occasions"]
