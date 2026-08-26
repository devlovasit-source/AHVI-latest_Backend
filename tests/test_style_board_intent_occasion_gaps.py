"""Regression coverage for the wedding/casual style-board routing gap found
via device E2E: `_is_explicit_style_request` had no occasion vocabulary for
"casual", and "wedding" required a setup_phrase companion ("i have", "going
to") that natural phrasings like "I am attending a wedding" don't use.
"""

from routers.chat import _is_explicit_style_request
from services.pre_classifier import classify_message


def test_indian_wedding_attendance_triggers_board():
    assert _is_explicit_style_request("I am attending an Indian wedding", "style")


def test_casual_but_polished_triggers_board():
    assert _is_explicit_style_request("Give me something casual but polished", "style")


def test_wedding_regression_phrases_trigger_board():
    for text in (
        "What should I wear to a wedding?",
        "Style me for a wedding",
    ):
        assert _is_explicit_style_request(text, "style"), text


def test_casual_regression_phrases_trigger_board():
    for text in (
        "Give me a casual outfit",
        "Style me casually",
    ):
        assert _is_explicit_style_request(text, "style"), text


def test_existing_occasion_controls_still_trigger_board():
    for text in (
        "What should I wear for brunch?",
        "I have a client meeting",
    ):
        assert _is_explicit_style_request(text, "style"), text


def test_style_advice_questions_stay_text_only():
    """These must NOT reach board generation - they're fashion education/
    advice, not a request for a specific outfit. Mirrors the real request
    flow: services.pre_classifier.classify_message runs FIRST and, when it
    returns a response_mode, short-circuits before routers.chat ever calls
    _is_explicit_style_request - so a phrase caught deterministically as
    text_only there never reaches the board-authorization check at all."""
    for text in (
        "How do I style oversized shirts?",
        "What colors work with beige trousers?",
        "How can I look taller?",
    ):
        pre = classify_message(text)
        if pre is not None:
            assert pre.get("response_mode") == "text_only", (text, pre)
        else:
            assert not _is_explicit_style_request(text, "style"), text
