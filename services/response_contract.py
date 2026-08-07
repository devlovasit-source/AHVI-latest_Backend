"""
P0 canonical response contract.

Adds one canonical `response_mode` field to every chat envelope, echoes the
client-supplied `request_id`, preserves a server `trace_id`, and enforces the
mutually-exclusive payload invariants (a `text_only` response cannot carry
board fields, a `calendar_navigation` response cannot carry style payloads,
etc.).

ponytail: intentionally tiny. Contract lives here; endpoints call
``stamp_response`` at their single return path. When a real
framework-grade response envelope class exists, delete this file.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

ALLOWED_RESPONSE_MODES = {
    "text_only",
    "visual_inspiration",
    "wardrobe_recommendation",
    "style_this",
    "build_outfit",
    "calendar_navigation",
    "calendar_action",
    "planner_action",
    "clarification",
    "error",
}

# Fields that carry visual board payloads. Stripped from text-primary
# responses to make the frontend renderer contract deterministic.
_BOARD_FIELDS_TOPLEVEL = (
    "visual_directions",
    "visualDirections",
    "style_boards",
    "styleBoards",
    "visual_board",
    "visualBoard",
    "visual_inspiration_board",
    "style_directions",
    "styleDirections",
)
_BOARD_FIELDS_DATA = (
    "visual_directions",
    "visualDirections",
    "style_boards",
    "styleBoards",
    "visual_board",
    "visualBoard",
    "visual_inspiration_board",
    "style_directions",
    "styleDirections",
    "rendered_boards",
    "outfits",
)

# Legacy `mode` / `intent` values → canonical response_mode.
_LEGACY_TO_RESPONSE_MODE = {
    # explicit visual style modes
    "visual_inspiration": "visual_inspiration",
    "wardrobe_style": "wardrobe_recommendation",
    "style_flow_service_adapter_v1": "wardrobe_recommendation",
    "style_this": "style_this",
    "build_outfit": "build_outfit",
    # text-primary style modes
    "style_advice": "text_only",
    "style_pairing": "text_only",
    "style_education": "text_only",
    "color_body_advice": "text_only",
    "color_advice": "text_only",
    "body_proportion_advice": "text_only",
    "occasion_advice": "text_only",
    "shopping_assist": "text_only",
    "missing_pieces": "text_only",
    "supportive_conversation": "text_only",
    "general_chat": "text_only",
    "greeting_bypass": "text_only",
    "help_identity_bypass": "text_only",
    "small_talk_bypass": "text_only",
    "clarification": "clarification",
    "style_intent_clarification": "clarification",
    "beta_style_clarification": "clarification",
    "context_required": "clarification",
    # calendar
    "create_event": "calendar_action",
    "event_created": "calendar_action",
    "calendar_event_reused": "calendar_action",
    "event_needs_time": "clarification",
    # planner / packing
    "plan_pack": "planner_action",
    # error
    "error": "error",
}


def new_trace_id() -> str:
    return f"trc_{uuid.uuid4().hex[:16]}"


def resolve_response_mode(envelope: Dict[str, Any]) -> str:
    """
    Precedence: response_mode → legacy mode → legacy intent → text_only.
    Unknown legacy values fall through so the caller can decide; the ultimate
    default is text_only (fail closed to the safest renderer).
    """
    if not isinstance(envelope, dict):
        return "text_only"

    explicit = str(envelope.get("response_mode") or "").strip().lower()
    if explicit in ALLOWED_RESPONSE_MODES:
        return explicit

    meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
    data = envelope.get("data") if isinstance(envelope.get("data"), dict) else {}

    for src in (envelope, data, meta):
        for key in ("mode", "intent"):
            value = str(src.get(key) or "").strip().lower()
            mapped = _LEGACY_TO_RESPONSE_MODE.get(value)
            if mapped in ALLOWED_RESPONSE_MODES:
                return mapped

    return "text_only"


def _strip_keys(container: Dict[str, Any], keys) -> None:
    for key in keys:
        container.pop(key, None)


def _enforce_invariants(envelope: Dict[str, Any], response_mode: str) -> None:
    data = envelope.get("data")
    if not isinstance(data, dict):
        data = {}
        envelope["data"] = data

    if response_mode in {"text_only", "clarification", "error"}:
        _strip_keys(envelope, _BOARD_FIELDS_TOPLEVEL)
        _strip_keys(data, _BOARD_FIELDS_DATA)
        envelope["cards"] = []
        envelope["style_boards"] = []
        # `blocks` may contain visual_board dicts — drop those specifically.
        blocks = envelope.get("blocks")
        if isinstance(blocks, list):
            envelope["blocks"] = [
                b for b in blocks
                if not (isinstance(b, dict) and str(b.get("type") or "").lower() in {
                    "visual_board", "visual_inspiration_board", "visual_directions",
                })
            ]
        return

    if response_mode == "calendar_navigation":
        _strip_keys(envelope, _BOARD_FIELDS_TOPLEVEL)
        _strip_keys(data, _BOARD_FIELDS_DATA)
        return

    if response_mode == "wardrobe_recommendation":
        # Wardrobe boards allowed; strip inspiration-only fields to keep
        # the two payload shapes mutually exclusive.
        _strip_keys(envelope, ("visual_inspiration_board",))
        _strip_keys(data, ("visual_inspiration_board",))
        return

    if response_mode == "visual_inspiration":
        # Inspiration allowed; strip explicit wardrobe recommendation blocks.
        # `rendered_boards` is the wardrobe-recommendation shape.
        _strip_keys(data, ("rendered_boards",))
        return


def stamp_response(
    envelope: Dict[str, Any],
    *,
    response_mode: str,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    domain: Optional[str] = None,
    intent: Optional[str] = None,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stamp the canonical fields onto ``envelope`` in place (also returns it).
    Idempotent: re-stamping is safe.
    """
    if not isinstance(envelope, dict):
        envelope = {}
    if response_mode not in ALLOWED_RESPONSE_MODES:
        response_mode = "text_only"

    envelope["response_mode"] = response_mode
    envelope["request_id"] = str(request_id or envelope.get("request_id") or "")
    envelope["trace_id"] = str(trace_id or envelope.get("trace_id") or new_trace_id())

    if domain:
        envelope["domain"] = str(domain)
    if intent:
        envelope["intent"] = str(intent)
    if action:
        envelope["action"] = str(action)

    # Mirror into meta so old readers see the same values.
    meta = envelope.get("meta") if isinstance(envelope.get("meta"), dict) else {}
    meta = dict(meta)
    meta.setdefault("mode", envelope.get("mode") or "")
    meta["response_mode"] = response_mode
    meta["request_id"] = envelope["request_id"]
    meta["trace_id"] = envelope["trace_id"]
    envelope["meta"] = meta

    _enforce_invariants(envelope, response_mode)
    return envelope


if __name__ == "__main__":
    # ponytail: tiny self-check. Runs when this file is executed directly.
    r = stamp_response(
        {"cards": [{"id": "x"}], "visual_directions": [{"title": "X"}]},
        response_mode="text_only", request_id="req_1",
    )
    assert r["response_mode"] == "text_only"
    assert r["request_id"] == "req_1"
    assert r["cards"] == []
    assert r["style_boards"] == []
    assert "visual_directions" not in r
    assert resolve_response_mode({"intent": "style_advice"}) == "text_only"
    assert resolve_response_mode({"intent": "visual_inspiration"}) == "visual_inspiration"
    assert resolve_response_mode({"mode": "wardrobe_style"}) == "wardrobe_recommendation"
    assert resolve_response_mode({"meta": {"mode": "style_flow_service_adapter_v1"}}) == "wardrobe_recommendation"
    assert resolve_response_mode({"response_mode": "calendar_navigation"}) == "calendar_navigation"
    assert resolve_response_mode({}) == "text_only"
    r = stamp_response(
        {"data": {"visual_directions": [1]}, "cards": [1]},
        response_mode="calendar_navigation", request_id="r2",
    )
    assert "visual_directions" not in r["data"]
    print("response_contract self-check ok")
