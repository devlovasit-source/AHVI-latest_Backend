"""Regression coverage for the advice-mode completeness bypass:
OCCASION_ADVICE / BODY_PROPORTION_ADVICE / COLOR_ADVICE used to skip
_complete_style_text entirely (see services.style_reasoning_engine._build_response),
letting malformed hanging endings like "...or." and "...ensuring you're."
reach the user unguarded. _complete_advice_lines closes that without
collapsing the bullet-format contract the way _complete_style_text would.
"""
from __future__ import annotations

from services.style_reasoning_engine import _complete_advice_lines
from routers.chat import _style_reasoning_chat_response


FALLBACK = "Here's a clean, versatile direction for today."


def test_malformed_or_bullet_dropped_others_survive():
    text = (
        "Opening sentence.\n"
        "- Keep the blazer structured.\n"
        "- This malformed bullet ends or.\n"
        "- Use clean leather shoes."
    )
    result = _complete_advice_lines(text, query="office", fallback=FALLBACK)
    lines = result.split("\n")

    assert "Opening sentence." in lines
    assert "- Keep the blazer structured." in lines
    assert "- Use clean leather shoes." in lines
    assert not any(line.rstrip(".").endswith(" or") for line in lines)
    assert len(lines) == 3


def test_malformed_contracted_aux_bullet_dropped():
    text = (
        "Opening sentence.\n"
        "- Keep the blazer structured.\n"
        "- Focus on details that project confidence, ensuring you're.\n"
        "- Use clean leather shoes."
    )
    result = _complete_advice_lines(text, query="office", fallback=FALLBACK)
    lines = result.split("\n")

    assert "Opening sentence." in lines
    assert "- Keep the blazer structured." in lines
    assert "- Use clean leather shoes." in lines
    assert not any(line.endswith("you're.") for line in lines)
    assert len(lines) == 3


def test_malformed_trailing_adjunct_bullet_salvaged_not_dropped():
    """Unlike a bare hanging ending (or./you're.) with nothing worth keeping,
    a bullet with a malformed trailing adjunct has a genuinely complete
    clause before it -- that clause should survive, not the whole bullet
    get dropped."""
    text = (
        "Opening sentence.\n"
        "- Keep the blazer structured.\n"
        "- Move through the day with ease while maintaining a.\n"
        "- Use clean leather shoes."
    )
    result = _complete_advice_lines(text, query="office", fallback=FALLBACK)
    lines = result.split("\n")

    assert "Opening sentence." in lines
    assert "- Keep the blazer structured." in lines
    assert "- Move through the day with ease." in lines
    assert "- Use clean leather shoes." in lines
    assert not any("maintaining" in line for line in lines)
    assert len(lines) == 4


def test_all_lines_malformed_falls_back():
    text = "- This ends or.\n- This ends and."
    result = _complete_advice_lines(text, query="office", fallback=FALLBACK)
    assert result == FALLBACK


def test_clean_multiline_advice_unchanged_shape():
    text = (
        "Opening sentence.\n"
        "- Keep the blazer structured.\n"
        "- Use clean leather shoes.\n"
        "Close with confidence."
    )
    result = _complete_advice_lines(text, query="office", fallback=FALLBACK)
    lines = result.split("\n")
    assert lines == [
        "Opening sentence.",
        "- Keep the blazer structured.",
        "- Use clean leather shoes.",
        "Close with confidence.",
    ]


def test_downstream_chat_fields_inherit_cleaned_advice_unchanged():
    """routers.chat._style_reasoning_chat_response copies reasoning['advice']
    into message/message_text/response/text verbatim -- prove a clean,
    already-guarded advice string survives that copy unchanged."""
    clean_advice = "Opening sentence.\n- Keep the blazer structured.\n- Use clean leather shoes."
    reasoning = {
        "advice": clean_advice,
        "mode": "occasion_advice",
        "cta": [],
        "visual_directions": [],
    }

    result = _style_reasoning_chat_response(reasoning, query="What should I wear to office tomorrow?")

    assert result["message"]["content"] == clean_advice
    assert result["message_text"] == clean_advice
    assert result["response"] == clean_advice
    assert result["text"] == clean_advice
