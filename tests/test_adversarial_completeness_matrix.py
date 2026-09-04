"""Adversarial postcondition matrix for the final Style completeness
boundary. Exercises a broad set of malformed sentence endings (not just the
handful proven live) against the actual last-mutation function
(_enforce_final_style_completeness) and proves the postcondition: the
returned canonical text is never byte-identical to the malformed input,
never looks_truncated, always ends in a syntactically complete sentence,
and propagates identically through the downstream chat fields.
"""
from __future__ import annotations

import pytest

from brain.response_validator import looks_truncated
from services.style_reasoning_engine import _enforce_final_style_completeness
from routers.chat import _style_reasoning_chat_response

PREFIX = "Choose a structured layer that reads polished"

MALFORMED_ENDINGS = [
    "and.",
    "or.",
    "but.",
    "because.",
    "you're.",
    "with.",
    "without.",
    "without being.",
    "while.",
    "while maintaining.",
    "while maintaining a.",
    "including.",
    "such as.",
    "for.",
    "to.",
    "of.",
    "in.",
    ", avoiding.",
    ", keeping.",
    ", creating.",
    ", adding.",
]

CLEAN_CONTROLS = [
    "Keep the palette refined and avoid loud contrasts.",
    "Choose a structured layer while maintaining a balanced silhouette.",
    "Keep it modern, avoiding loud contrasts.",
    "Use soft neutrals, adding texture through accessories.",
]


def _malformed_text(ending: str) -> str:
    sep = "" if ending.startswith(",") else " "
    return f"{PREFIX}{sep}{ending}"


@pytest.mark.parametrize("ending", MALFORMED_ENDINGS)
def test_malformed_ending_never_escapes_final_boundary(ending):
    malformed = _malformed_text(ending)
    assert looks_truncated(malformed) is True, f"test setup invalid: {malformed!r} not detected as malformed"

    response = {"advice": malformed, "stylist_reasoning": malformed, "mode": "occasion_advice"}
    out = _enforce_final_style_completeness(
        response, query="What should I wear to office tomorrow?", mode="occasion_advice", category=None
    )

    final_text = out["advice"]
    # 1. not byte-for-byte identical to the known-bad input
    assert final_text != malformed
    # 2. looks_truncated is False
    assert looks_truncated(final_text) is False
    # 3. ends in a syntactically complete sentence
    assert final_text.rstrip()[-1] in ".!?…"
    # advice and stylist_reasoning are the same canonical value
    assert out["stylist_reasoning"] == final_text

    # 4. propagates identically through the visible chat fields
    chat_result = _style_reasoning_chat_response(out, query="What should I wear to office tomorrow?")
    assert chat_result["message"]["content"] == final_text
    assert chat_result["message_text"] == final_text
    assert chat_result["response"] == final_text
    assert chat_result["text"] == final_text


@pytest.mark.parametrize("clean", CLEAN_CONTROLS)
def test_clean_control_untouched(clean):
    assert looks_truncated(clean) is False

    response = {"advice": clean, "stylist_reasoning": clean, "mode": "occasion_advice"}
    out = _enforce_final_style_completeness(
        response, query="What should I wear to office tomorrow?", mode="occasion_advice", category=None
    )

    assert out["advice"] == clean
    assert out["stylist_reasoning"] == clean
