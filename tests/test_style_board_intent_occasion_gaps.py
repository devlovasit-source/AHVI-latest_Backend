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


def _routes_to_board(text: str, module: str = "style") -> bool:
    pre = classify_message(text)
    if pre is not None:
        return pre.get("response_mode") not in ("text_only", "clarification", "calendar_navigation")
    return _is_explicit_style_request(text, module)


def test_first_person_and_direct_request_phrases_authorize_board():
    """Follow-up to 0c6f524: subject-qualified event participation ("I'm
    attending") and direct request verbs ("give me", "dress me") paired
    with an occasion/casual word - not bare occasion mentions - are what
    should authorize a board."""
    for text in (
        "I am attending an Indian wedding",
        "I'm going to a wedding",
        "What should I wear to a wedding?",
        "Give me a wedding outfit",
        "Give me something casual but polished",
        "Something casual for tonight",
        "Show me a casual outfit",
        "I need a casual outfit",
        "Dress me for brunch",
        "I have a client meeting",
    ):
        assert _routes_to_board(text), text


def test_third_person_and_informational_mentions_stay_text_only():
    """0c6f524's vocabulary expansion ("casual" keyword, "attending" setup
    phrase) over-matched: third-person subjects and pure information
    questions must not authorize a board just because they contain a
    styled-adjacent word."""
    for text in (
        "My friend is attending a wedding",
        "She is attending an Indian wedding",
        "Tell me what people wear to Indian weddings",
        "Are weddings usually formal?",
        "What happens at an Indian wedding?",
        "I love casual clothes",
        "I usually dress casual",
        "What does casual mean?",
        "Is casual the same as smart casual?",
        "How can I look taller?",
        "How do I style oversized shirts?",
        "What colors work with beige trousers?",
    ):
        assert not _routes_to_board(text), text


def test_routing_decision_is_stateless_per_turn():
    """_is_explicit_style_request/classify_message take only the current
    message text - no history/conversation_id parameter exists - so a prior
    text_only turn can't suppress a later board-eligible turn, or vice versa."""
    turns = [
        ("How can I look taller?", False),
        ("Give me something casual but polished", True),
        ("I'm attending an Indian wedding this weekend", True),
    ]
    for text, expect_board in turns:
        assert _routes_to_board(text) is expect_board, text
