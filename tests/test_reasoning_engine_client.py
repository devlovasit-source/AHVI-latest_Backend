"""Vertex Reasoning Engine transport helper."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from services import _reasoning_engine_client as rec


# ---------------------------------------------------------------------------
# Resource-id detection
# ---------------------------------------------------------------------------

def test_looks_like_resource_id_accepts_real_pattern():
    assert rec.looks_like_resource_id(
        "projects/631493992863/locations/us-west1/reasoningEngines/6180706703449784320"
    )


def test_looks_like_resource_id_rejects_http_urls():
    assert not rec.looks_like_resource_id("https://example.com/agent")
    assert not rec.looks_like_resource_id("http://localhost:9000")


def test_looks_like_resource_id_rejects_empty_and_garbage():
    assert not rec.looks_like_resource_id("")
    assert not rec.looks_like_resource_id(None)
    assert not rec.looks_like_resource_id("projects/x/y")


def test_location_parsed_from_resource_id():
    assert (
        rec._location_from_resource(
            "projects/123/locations/europe-west4/reasoningEngines/9"
        )
        == "europe-west4"
    )
    assert rec._location_from_resource("garbage", fallback="us-central1") == "us-central1"


# ---------------------------------------------------------------------------
# Envelope normalization
# ---------------------------------------------------------------------------

def test_parse_engine_payload_flat_dict_passthrough():
    p = {"occasion": "office", "confidence": 0.8}
    assert rec._parse_engine_payload(p) == p


def test_parse_engine_payload_output_content_json_string():
    p = {"output": {"content": '{"occasion":"date_night","confidence":0.9}'}}
    out = rec._parse_engine_payload(p)
    assert out == {"occasion": "date_night", "confidence": 0.9}


def test_parse_engine_payload_output_text_envelope():
    p = {"output": {"text": '{"a":1}'}}
    assert rec._parse_engine_payload(p) == {"a": 1}


def test_parse_engine_payload_raw_json_string():
    assert rec._parse_engine_payload('{"a":1}') == {"a": 1}


def test_parse_engine_payload_garbage_returns_none():
    assert rec._parse_engine_payload("not json") is None
    assert rec._parse_engine_payload(None) is None
    assert rec._parse_engine_payload(42) is None


def test_parse_engine_payload_invalid_inner_json_returns_none():
    p = {"output": {"content": "still not json"}}
    assert rec._parse_engine_payload(p) is None


# ---------------------------------------------------------------------------
# Robust agent-response extraction (the actual ADK shape from production)
# ---------------------------------------------------------------------------

def test_strip_json_fence_removes_backticks():
    assert rec._strip_json_fence("```json\n{\"a\":1}\n```") == '{"a":1}'
    assert rec._strip_json_fence("```\n{\"a\":1}\n```") == '{"a":1}'
    assert rec._strip_json_fence('{"a":1}') == '{"a":1}'


def test_try_parse_json_text_handles_fenced_block():
    assert rec._try_parse_json_text("```json\n{\"occasion\":\"office\"}\n```") == {"occasion": "office"}


def test_try_parse_json_text_lifts_first_dict_when_wrapped_in_prose():
    raw = "Here you go: {\"occasion\":\"date_night\"} hope that helps"
    assert rec._try_parse_json_text(raw) == {"occasion": "date_night"}


def test_try_parse_json_text_returns_none_on_unsalvageable():
    assert rec._try_parse_json_text("hello world") is None
    assert rec._try_parse_json_text(None) is None


def test_extract_agent_json_direct_dict():
    p = {"occasion": "office", "style_direction": "smart_casual", "confidence": 0.9}
    assert rec._extract_agent_json(p) == p


def test_extract_agent_json_adk_content_parts_text_with_fence():
    """The exact ADK runtime shape AHVI returns in production."""
    payload = {
        "model_version": "gemini-2.5-flash",
        "content": {
            "parts": [
                {
                    "text": "```json\n"
                            "{\n"
                            "  \"occasion\": \"client_meeting\",\n"
                            "  \"sub_intent\": \"professional_meeting\",\n"
                            "  \"formality\": \"smart_business\",\n"
                            "  \"style_direction\": \"modern_tailoring\",\n"
                            "  \"avoid_items\": [\"shorts\"],\n"
                            "  \"required_slots\": [\"top\", \"bottom\", \"footwear\"]\n"
                            "}\n"
                            "```"
                }
            ]
        },
    }
    parsed = rec._extract_agent_json(payload)
    assert parsed is not None
    assert parsed["occasion"] == "client_meeting"
    assert parsed["sub_intent"] == "professional_meeting"
    assert parsed["style_direction"] == "modern_tailoring"
    assert "shorts" in parsed["avoid_items"]


def test_extract_agent_json_output_string_envelope():
    payload = {"output": "```json {\"occasion\":\"workout\"}```"}
    parsed = rec._extract_agent_json(payload)
    assert parsed == {"occasion": "workout"}


def test_extract_agent_json_response_string_envelope():
    payload = {"response": "{\"occasion\":\"date_night\"}"}
    parsed = rec._extract_agent_json(payload)
    assert parsed == {"occasion": "date_night"}


def test_extract_agent_json_gemini_candidates_envelope():
    payload = {
        "candidates": [
            {"content": {"parts": [{"text": "{\"occasion\":\"travel\"}"}]}}
        ]
    }
    parsed = rec._extract_agent_json(payload)
    assert parsed == {"occasion": "travel"}


def test_extract_agent_json_list_of_events_picks_last_match():
    events = [
        {"content": {"parts": [{"text": "thinking..."}]}},
        {"content": {"parts": [{"text": "{\"occasion\":\"office\"}"}]}},
    ]
    parsed = rec._extract_agent_json(events)
    assert parsed == {"occasion": "office"}


def test_extract_agent_json_metadata_schema_keys_match():
    """Metadata validator returns a different key set; should still extract."""
    payload = {
        "content": {
            "parts": [
                {"text": "{\"category\":\"Tops\",\"style_role\":\"businesswear\",\"confidence\":0.9}"}
            ]
        }
    }
    parsed = rec._extract_agent_json(payload)
    assert parsed["category"] == "Tops"
    assert parsed["style_role"] == "businesswear"


def test_extract_agent_json_returns_none_when_no_schema_keys():
    payload = {"content": {"parts": [{"text": "{\"foo\":\"bar\"}"}]}}
    assert rec._extract_agent_json(payload) is None


def test_extract_agent_json_handles_garbage():
    assert rec._extract_agent_json(None) is None
    assert rec._extract_agent_json("hello") is None
    assert rec._extract_agent_json(42) is None


# ---------------------------------------------------------------------------
# call_reasoning_engine async wrapper safety
# ---------------------------------------------------------------------------

def test_call_returns_none_for_non_resource_endpoint():
    result = asyncio.run(
        rec.call_reasoning_engine(
            "https://example.com/agent",
            system="sys",
            prompt="p",
            timeout=2,
        )
    )
    assert result is None


def test_call_returns_none_when_sdk_missing(monkeypatch):
    """If vertexai isn't installed (or fails), helper must not raise."""

    def fake_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(rec, "_invoke_reasoning_engine_sync", fake_sync)
    result = asyncio.run(
        rec.call_reasoning_engine(
            "projects/123/locations/us-west1/reasoningEngines/456",
            system="sys",
            prompt="p",
            timeout=2,
        )
    )
    assert result is None


def test_call_returns_dict_when_sync_path_does(monkeypatch):
    def fake_sync(*args, **kwargs):
        return {"occasion": "office", "confidence": 0.9}

    monkeypatch.setattr(rec, "_invoke_reasoning_engine_sync", fake_sync)
    result = asyncio.run(
        rec.call_reasoning_engine(
            "projects/123/locations/us-west1/reasoningEngines/456",
            system="sys",
            prompt="p",
            timeout=2,
        )
    )
    assert result == {"occasion": "office", "confidence": 0.9}


# ---------------------------------------------------------------------------
# Agent dispatch routes through reasoning-engine path
# ---------------------------------------------------------------------------

def test_style_orchestrator_routes_resource_id_to_reasoning_engine(monkeypatch):
    from services import agent_style_orchestrator as svc

    monkeypatch.setenv("ENABLE_AGENT_STYLE_ORCHESTRATOR", "1")
    monkeypatch.setenv(
        "AGENT_STYLE_ORCHESTRATOR_ENDPOINT",
        "projects/123/locations/us-west1/reasoningEngines/456",
    )

    captured: Dict[str, Any] = {}

    async def fake_engine(resource_id, *, system, prompt, timeout):
        captured["resource_id"] = resource_id
        captured["system"] = system
        return {"occasion": "office", "confidence": 0.9}

    # Patch the symbol the agent file imports lazily.
    import services._reasoning_engine_client as rec_mod

    monkeypatch.setattr(rec_mod, "call_reasoning_engine", fake_engine)

    out = asyncio.run(
        svc.orchestrate_style_request(
            message="outfit for client meeting",
            user_id="u1",
        )
    )
    assert out["occasion"] == "office"
    assert captured["resource_id"].startswith("projects/")


def test_metadata_validator_routes_resource_id_to_reasoning_engine(monkeypatch):
    from services import agent_metadata_validator as svc

    monkeypatch.setenv("ENABLE_AGENT_METADATA_VALIDATOR", "1")
    monkeypatch.setenv(
        "AGENT_METADATA_VALIDATOR_ENDPOINT",
        "projects/123/locations/us-west1/reasoningEngines/789",
    )

    async def fake_engine(resource_id, *, system, prompt, timeout):
        return {
            "category": "Tops",
            "subcategory": "Shirt",
            "style_role": "businesswear",
            "confidence": 0.95,
        }

    import services._reasoning_engine_client as rec_mod

    monkeypatch.setattr(rec_mod, "call_reasoning_engine", fake_engine)

    out = asyncio.run(
        svc.validate_wardrobe_metadata(
            item={"name": "White Shirt", "category": "Tops"},
            user_id="u1",
        )
    )
    assert out["category"] == "Tops"
    assert out["style_role"] == "businesswear"
