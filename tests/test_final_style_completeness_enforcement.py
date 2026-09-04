"""Enforcement tests for the true last Style response boundary.

_build_response's completeness guard (_complete_advice_lines /
_complete_style_text) runs BEFORE apply_personality_text_polish_to_final_payload
and apply_gender_guard_to_final_payload -- both of which can re-touch the
advice/stylist_reasoning text afterward. In particular
_personality_sentence_cap's blind word/char re-capping sliced a
single-paragraph advice string mid-clause on the live content-demo revision,
reintroducing hanging endings ("...confidence and.", "...refined, avoiding.")
the earlier guard had already removed.

_enforce_final_style_completeness runs after both of those steps and is the
actual last mutation before services.style_reasoning_engine.reason() returns
to routers/chat.py. These are enforcement tests, not detector unit tests:
each one proves the malformed text is ABSENT from the final value, not just
that looks_truncated flags it.
"""
from __future__ import annotations

from brain.response_validator import looks_truncated
from services.style_reasoning_engine import _enforce_final_style_completeness
from routers.chat import _style_reasoning_chat_response


def _run(advice: str, *, mode: str = "occasion_advice") -> dict:
    response = {"advice": advice, "stylist_reasoning": advice, "mode": mode}
    return _enforce_final_style_completeness(
        response, query="What should I wear to office tomorrow?", mode=mode, category=None
    )


# ---------- RUN5: the exact live-detected "and." case must never escape ----------

def test_run5_and_never_escapes():
    malformed = (
        "I'm focusing on versatile pieces that offer structure without feeling "
        "overly rigid, allowing you to move through your day with confidence and."
    )
    out = _run(malformed)
    assert "confidence and." not in out["advice"]
    assert "confidence and." not in out["stylist_reasoning"]
    assert looks_truncated(out["advice"]) is False
    assert looks_truncated(out["stylist_reasoning"]) is False


# ---------- RUN4: the comma+dangling-gerund case must never escape ----------

def test_run4_comma_avoiding_never_escapes():
    malformed = (
        "This works by focusing on intentional pairings that feel modern and "
        "refined, avoiding."
    )
    out = _run(malformed)
    assert not out["advice"].rstrip().endswith("avoiding.")
    assert looks_truncated(out["advice"]) is False


# ---------- every previously-proven-live shape must never escape ----------

def test_bare_or_never_escapes():
    out = _run(
        "These looks balance professional polish with modern comfort, ensuring "
        "you feel confident and appropriate for any meeting or."
    )
    assert not out["advice"].rstrip().endswith("or.")
    assert looks_truncated(out["advice"]) is False


def test_bare_youre_never_escapes():
    out = _run(
        "Focus on clean silhouettes and smart details that project confidence "
        "and efficiency, ensuring you're."
    )
    assert not out["advice"].rstrip().endswith("you're.")
    assert looks_truncated(out["advice"]) is False


def test_without_being_never_escapes():
    out = _run("This can feel polished and professional without being.")
    assert not out["advice"].rstrip().endswith("without being.")
    assert looks_truncated(out["advice"]) is False


def test_while_maintaining_bare_never_escapes():
    out = _run("Keep the look polished while maintaining.")
    assert not out["advice"].rstrip().endswith("while maintaining.")
    assert looks_truncated(out["advice"]) is False


def test_while_maintaining_a_never_escapes():
    out = _run("Move through the day with ease while maintaining a.")
    assert not out["advice"].rstrip().endswith("while maintaining a.")
    assert looks_truncated(out["advice"]) is False


# ---------- valid negatives: clean text is not mangled/replaced by fallback ----------

def test_clean_paragraph_advice_untouched():
    clean = "Choose tailored pieces while maintaining a polished silhouette."
    out = _run(clean)
    assert out["advice"] == clean
    assert out["stylist_reasoning"] == clean


def test_clean_bulleted_advice_bullets_preserved():
    clean = (
        "Opening sentence.\n"
        "- Keep the blazer structured.\n"
        "- Use clean leather shoes."
    )
    out = _run(clean)
    assert out["advice"].split("\n") == clean.split("\n")


# ---------- full chain: enforcement gate output survives routers/chat unchanged ----------

def test_enforced_text_survives_downstream_chat_fields_unchanged():
    """The malformed text must not just be absent from `advice` -- it must
    also be absent everywhere routers.chat._style_reasoning_chat_response
    copies `reasoning['advice']` into (message/message_text/response/text),
    since that's the actual /api/text response shape the client sees."""
    malformed = (
        "I'm focusing on versatile pieces that offer structure without feeling "
        "overly rigid, allowing you to move through your day with confidence and."
    )
    enforced = _run(malformed)

    result = _style_reasoning_chat_response(enforced, query="What should I wear to office tomorrow?")

    for field_value in (
        result["message"]["content"],
        result["message_text"],
        result["response"],
        result["text"],
    ):
        assert "confidence and." not in field_value
        assert looks_truncated(field_value) is False
