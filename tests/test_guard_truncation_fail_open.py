"""services.llm_service._guard_truncation must never return the original
malformed text as its own fallback -- doing so both ships the known-bad
text and (since callers gate RETRY_ON_TRUNCATION on `guarded != original`)
silently suppresses the retry."""
from __future__ import annotations

from brain.response_validator import looks_truncated
from services import llm_service


def test_truncated_input_never_returned_unchanged():
    malformed = "This works well while maintaining."
    assert looks_truncated(malformed) is True

    guarded = llm_service._guard_truncation(malformed, usecase="style_reasoning")

    assert guarded != malformed


def test_truncated_input_output_postcondition():
    """If looks_truncated(input) is True, looks_truncated(output) must be False."""
    for malformed in [
        "This works well while maintaining.",
        "allowing you to move through your day with confidence and.",
        "focusing on intentional pairings that feel modern and refined, avoiding.",
    ]:
        guarded = llm_service._guard_truncation(malformed, usecase="style_reasoning")
        assert looks_truncated(guarded) is False, f"guard left truncated text: {guarded!r}"
        assert guarded != malformed


def test_unsalvageable_text_escalates_to_safe_fallback_not_itself():
    """A single run-on malformed sentence with no earlier complete clause to
    salvage must not fall back to itself."""
    malformed = "with confidence and."  # too short/bare to salvage anything from
    guarded = llm_service._guard_truncation(malformed, usecase="style_reasoning")
    assert guarded != malformed
    assert looks_truncated(guarded) is False


def test_clean_text_passes_through_unchanged():
    clean = "This is a complete and grammatically correct sentence."
    assert llm_service._guard_truncation(clean, usecase="style_reasoning") == clean


def test_retry_not_suppressed_when_truncation_repaired(monkeypatch):
    """generate_text() gates RETRY_ON_TRUNCATION on `guarded != gemini_text`.
    Since _guard_truncation no longer returns the original malformed text,
    that comparison must reliably detect the repair happened."""
    malformed = "This works well while maintaining."
    guarded = llm_service._guard_truncation(malformed, usecase="style_reasoning")
    assert guarded != malformed, (
        "guarded == original would silently suppress RETRY_ON_TRUNCATION in "
        "generate_text()'s `guarded != gemini_text` check"
    )
