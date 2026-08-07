"""P0 intelligence contract tests.

Covers:
- ``services.response_contract.resolve_response_mode`` precedence
- ``services.response_contract.stamp_response`` invariants (text_only /
  clarification / calendar_navigation / visual_inspiration /
  wardrobe_recommendation strip the wrong payload shapes)
- ``services.pre_classifier.classify_message`` for the exact reported
  failure phrases (bare "calendar", "give me style tips", "what is color
  analysis?", "show me brunch outfit inspiration")
- ``ModuleChatRequest`` / ``TextChatRequest`` accept the client
  ``request_id`` field.

These use pytest but only stdlib assertions — no fixtures, no mocks. If
pytest is unavailable they can be run as ``python tests/test_intelligence_p0.py``
(there is a ``__main__`` block at the bottom that runs every function).

ponytail: pytest-agnostic on purpose; drop the ``__main__`` when the CI
environment gains pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.response_contract import (
    ALLOWED_RESPONSE_MODES,
    resolve_response_mode,
    stamp_response,
)
from services.pre_classifier import classify_message


# ---------- resolve_response_mode ----------

def test_resolve_prefers_response_mode_field():
    assert resolve_response_mode({"response_mode": "text_only"}) == "text_only"
    assert resolve_response_mode({"response_mode": "visual_inspiration"}) == "visual_inspiration"


def test_resolve_falls_back_to_legacy_mode():
    assert resolve_response_mode({"mode": "wardrobe_style"}) == "wardrobe_recommendation"
    assert resolve_response_mode({"mode": "style_advice"}) == "text_only"


def test_resolve_falls_back_to_legacy_intent():
    assert resolve_response_mode({"intent": "visual_inspiration"}) == "visual_inspiration"
    assert resolve_response_mode({"intent": "style_pairing"}) == "text_only"


def test_resolve_defaults_to_text_only():
    assert resolve_response_mode({}) == "text_only"
    assert resolve_response_mode({"mode": "unknown_soup"}) == "text_only"
    assert resolve_response_mode(None) == "text_only"


def test_resolve_ignores_unknown_response_mode():
    # Unknown response_mode value falls through to the legacy resolution
    # rather than being trusted as-is.
    assert resolve_response_mode({"response_mode": "junk"}) == "text_only"


# ---------- stamp_response invariants ----------

def test_stamp_text_only_strips_visual_topfields():
    r = stamp_response(
        {
            "visual_directions": [{"title": "X"}],
            "style_boards": [{"id": "b"}],
            "visual_board": {"id": "vb"},
            "visual_inspiration_board": {"id": "vib"},
            "cards": [{"id": "c"}],
        },
        response_mode="text_only",
        request_id="req_1",
    )
    for banned in (
        "visual_directions", "style_boards", "visual_board",
        "visual_inspiration_board",
    ):
        assert banned not in r, f"{banned} should be stripped in text_only"
    assert r["cards"] == []
    assert r["response_mode"] == "text_only"
    assert r["request_id"] == "req_1"


def test_stamp_text_only_strips_visual_from_data():
    r = stamp_response(
        {"data": {"visual_directions": [1], "outfits": [1], "rendered_boards": [1]}},
        response_mode="text_only",
        request_id="req_2",
    )
    assert "visual_directions" not in r["data"]
    assert "outfits" not in r["data"]
    assert "rendered_boards" not in r["data"]


def test_stamp_clarification_strips_boards():
    r = stamp_response(
        {"visual_directions": [1], "cards": [1]},
        response_mode="clarification",
        request_id="req_3",
    )
    assert "visual_directions" not in r
    assert r["cards"] == []
    assert r["response_mode"] == "clarification"


def test_stamp_calendar_navigation_strips_style_payload():
    r = stamp_response(
        {"visual_directions": [1], "style_boards": [1], "chips": ["Open calendar"]},
        response_mode="calendar_navigation",
        request_id="req_4",
    )
    assert "visual_directions" not in r
    assert "style_boards" not in r
    # calendar_navigation may still carry chips.
    assert r["chips"] == ["Open calendar"]


def test_stamp_wardrobe_recommendation_strips_inspiration_only_field():
    r = stamp_response(
        {"visual_inspiration_board": {"id": "vib"}, "cards": [{"id": "outfit"}]},
        response_mode="wardrobe_recommendation",
        request_id="req_5",
    )
    assert "visual_inspiration_board" not in r
    # Wardrobe recommendation KEEPS cards / boards.
    assert r["cards"] == [{"id": "outfit"}]


def test_stamp_visual_inspiration_strips_wardrobe_only_field():
    r = stamp_response(
        {"data": {"rendered_boards": [1], "visual_directions": [{"title": "look"}]}},
        response_mode="visual_inspiration",
        request_id="req_6",
    )
    assert "rendered_boards" not in r["data"]
    # Visual inspiration keeps the visual directions.
    assert r["data"]["visual_directions"] == [{"title": "look"}]


def test_stamp_generates_trace_id_when_missing():
    r = stamp_response({}, response_mode="text_only", request_id="")
    assert r["trace_id"].startswith("trc_")
    assert r["request_id"] == ""  # explicit empty stays empty
    assert r["response_mode"] == "text_only"


def test_stamp_is_idempotent():
    once = stamp_response({}, response_mode="text_only", request_id="req_7")
    trace = once["trace_id"]
    twice = stamp_response(once, response_mode="text_only", request_id="req_7")
    assert twice["trace_id"] == trace
    assert twice["request_id"] == "req_7"


def test_stamp_rejects_unknown_response_mode():
    r = stamp_response({}, response_mode="junk", request_id="req_8")
    assert r["response_mode"] == "text_only"


def test_allowed_response_modes_is_the_mvp_set():
    assert ALLOWED_RESPONSE_MODES == {
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


# ---------- pre_classifier: reported failure phrases ----------

def test_bare_calendar_navigates():
    got = classify_message("calendar")
    assert got == {
        "domain": "calendar",
        "intent": "navigate",
        "action": "open_calendar",
        "response_mode": "calendar_navigation",
    }


def test_open_calendar_navigates():
    got = classify_message("open calendar")
    assert got is not None and got["response_mode"] == "calendar_navigation"


def test_calendar_with_time_is_not_navigation():
    # Sentences that look like event creation must fall through, not
    # navigate.
    assert classify_message("meeting with alex tomorrow at 3pm") is None


def test_style_tips_is_advice_text_only():
    got = classify_message("Give me style tips")
    assert got == {
        "domain": "style",
        "intent": "advice",
        "action": "provide_style_advice",
        "response_mode": "text_only",
    }


def test_how_do_i_dress_is_advice_text_only():
    got = classify_message("How do I dress better?")
    assert got is not None
    assert got["intent"] == "advice"
    assert got["response_mode"] == "text_only"


def test_what_is_color_analysis_is_information_text_only():
    got = classify_message("What is color analysis?")
    assert got == {
        "domain": "style",
        "intent": "information",
        "action": "explain_style_concept",
        "response_mode": "text_only",
    }


def test_explain_smart_casual_is_information():
    got = classify_message("Explain smart casual")
    assert got is not None and got["response_mode"] == "text_only"


def test_capsule_wardrobe_question_is_information():
    got = classify_message("What is a capsule wardrobe?")
    assert got is not None and got["response_mode"] == "text_only"


def test_brunch_inspiration_is_visual_inspiration():
    got = classify_message("Show me brunch outfit inspiration")
    assert got == {
        "domain": "style",
        "intent": "inspiration",
        "action": "provide_visual_inspiration",
        "response_mode": "visual_inspiration",
    }


def test_minimalist_ideas_is_visual_inspiration():
    got = classify_message("Give me minimalist outfit ideas")
    assert got is not None and got["response_mode"] == "visual_inspiration"


def test_wardrobe_recommendation_stays_unclassified():
    # "What should I wear today?" must fall through so the existing
    # wardrobe-recommendation path handles it. The pre-classifier does not
    # touch it.
    assert classify_message("What should I wear today?") is None
    assert classify_message("What can I wear to work?") is None


def test_style_this_stays_unclassified():
    assert classify_message("Style this belt") is None
    assert classify_message("style this jacket") is None


def test_build_outfit_stays_unclassified():
    assert classify_message("Build an outfit with these jeans") is None


# ---------- request models accept request_id ----------

def test_module_chat_request_accepts_request_id():
    # Guarded import: ``routers.chat`` needs fastapi; the module_chat_service
    # dependency chain also needs it. Skip if fastapi missing.
    try:
        from routers.chat import ModuleChatRequest, TextChatRequest
    except ModuleNotFoundError:
        return  # fastapi not installed in this env
    req = ModuleChatRequest(message="hi", request_id="req_client_xyz")
    assert req.request_id == "req_client_xyz"
    tr = TextChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        request_id="req_client_abc",
    )
    assert tr.request_id == "req_client_abc"


def test_module_chat_request_id_is_optional():
    try:
        from routers.chat import ModuleChatRequest
    except ModuleNotFoundError:
        return
    req = ModuleChatRequest(message="hi")
    assert req.request_id is None


if __name__ == "__main__":
    import inspect
    ok = 0
    fail = []
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                ok += 1
            except AssertionError as e:
                fail.append((name, str(e)))
            except Exception as e:  # noqa: BLE001
                fail.append((name, f"{type(e).__name__}: {e}"))
    print(f"{ok} passed, {len(fail)} failed")
    for n, e in fail:
        print(f"  FAIL {n}: {e}")
    sys.exit(1 if fail else 0)
