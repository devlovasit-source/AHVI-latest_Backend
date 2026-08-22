"""Male final-text guard: feminine accessory/makeup advice must never reach
a male user, and festive contexts get a men's ethnic template (not a blazer).
"""

from __future__ import annotations

from services.style_reasoning_engine import (
    _sanitize_male_final_text,
    _male_formal_context_replacement,
)


def test_haldi_strips_feminine_and_goes_ethnic():
    text = ("Add gold stud earrings for sparkle, a stack of bangles, "
            "soft waves and dewy makeup.")
    out, removed = _sanitize_male_final_text(text, context="haldi")
    low = out.lower()
    for bad in ("earring", "bangle", "soft waves", "dewy", "makeup"):
        assert bad not in low, bad
    assert "kurta" in low  # festive male template
    assert removed  # detected feminine terms


def test_conference_stays_formal_blazer():
    text = "Wear a midi dress, a silk blouse, heels, a necklace and earrings."
    out, removed = _sanitize_male_final_text(text, context="conference talk")
    low = out.lower()
    assert "blazer" in low
    for bad in ("dress", "blouse", "heels", "necklace", "earring"):
        assert bad not in low, bad


def test_clean_male_text_unchanged():
    text = "A crisp shirt with tailored trousers and loafers."
    out, removed = _sanitize_male_final_text(text, context="office")
    assert out == text
    assert removed == []


def test_inline_replacements_when_not_template():
    # Single feminine term, no comma, non-festive/formal -> per-term swap.
    out, removed = _sanitize_male_final_text("Finish with bangles.", context="brunch")
    assert "bangle" not in out.lower()
    assert removed == ["bangles"]


def test_festive_template_is_ethnic():
    assert "kurta" in _male_formal_context_replacement("sangeet").lower()
    assert "nehru" in _male_formal_context_replacement("haldi").lower()


def test_formal_template_is_blazer():
    assert "blazer" in _male_formal_context_replacement("client meeting").lower()
