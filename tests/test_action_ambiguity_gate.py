"""Regression coverage for the generic action-ambiguity admission gate.

Root cause chain proven live: bare context/meal nouns (breakfast/lunch/
dinner, and a separate bare wedding/party/event rule) were sufficient by
themselves to route into a module (Diet or occasion_outfit) with no actual
requested action -- "Dinner was terrible" produced an unsolicited
meal-planning pivot; "Wedding tomorrow" would have silently guessed Style.

Fix: `detect_action_ambiguity()` (brain/intent_engine.py) recognizes when a
message is ENTIRELY a context noun plus an optional time qualifier (no verb,
no opinion, no other content word) and asks one clarifying question instead
of guessing a module. Real sentences ("Dinner was terrible") always leave a
verb/opinion word behind and are untouched -- they resolve as ordinary
conversation. `_fallback_intent`'s diet admission and the LLM's
INTENT_PROMPT were both aligned to the same "action word required" contract,
and the bare wedding/party/event -> occasion_outfit rule was removed (its
legitimate case was already covered by the verb-gated occasion_keywords
check just above it).

Also covers a substring false-positive found while writing these tests:
the diet context word "eat" matched inside ordinary words ("create",
"homework") via plain `in` containment before being switched to
word-boundary regex matching.
"""

from brain.intent_engine import _fallback_intent, detect_action_ambiguity

CONVERSATION_CASES = (
    "Dinner was terrible",
    "Lunch was awful",
    "Breakfast was amazing",
    "The wedding was exhausting",
    "The party was boring",
    "Work was horrible today",
    "I hate going to the office",
    "The restaurant was disappointing",
    "Vacation was exhausting",
    "That meeting went badly",
)

AMBIGUOUS_CASES = (
    "Dinner tonight",
    "Lunch tomorrow",
    "Wedding tomorrow",
    "Party tonight",
    "Work tomorrow",
    "Vacation next week",
    "Meeting tomorrow",
    "Date tonight",
)

STYLE_POSITIVE_CASES = (
    "What should I wear for dinner tonight?",
    "What should I wear to work tomorrow?",
    "Style me for a wedding",
    "Give me an outfit for the party",
    "Help me dress for dinner",
    "Use my wardrobe for work",
)

DIET_POSITIVE_CASES = (
    "Plan my dinner",
    "What should I eat tonight?",
    "Create a meal plan",
    "Suggest a healthy lunch",
    "Give me a high-protein breakfast",
    "Help me with my diet",
)

PLANNING_NON_AMBIGUOUS_CASES = (
    "Plan my day tomorrow",
    "Schedule my work day",
    "Help me prepare for the wedding",
    "Create a checklist for my trip",
    "Add a meeting tomorrow at 4 PM",
)

# Words that collide, as plain substrings, with the ambiguity-noun / diet
# vocabulary ("eat" inside "create", "work" inside "homework", "date" inside
# "candidate") -- must never trigger ambiguity or Diet admission.
SUBSTRING_COLLISION_CASES = (
    "Create a checklist for my trip",
    "Update my profile please",
    "I have a candidate for the job",
    "I need help with my homework tomorrow",
)


def test_conversation_statements_are_not_ambiguous_and_stay_general():
    for text in CONVERSATION_CASES:
        assert detect_action_ambiguity(text) is None, text
        assert _fallback_intent(text)["intent"] == "general", text


def test_bare_context_noun_phrases_are_flagged_ambiguous():
    for text in AMBIGUOUS_CASES:
        assert detect_action_ambiguity(text) is not None, text


def test_explicit_style_requests_are_never_ambiguous():
    for text in STYLE_POSITIVE_CASES:
        assert detect_action_ambiguity(text) is None, text


def test_explicit_diet_requests_are_never_ambiguous_and_route_to_diet():
    for text in DIET_POSITIVE_CASES:
        assert detect_action_ambiguity(text) is None, text
        result = _fallback_intent(text)
        assert result["intent"] == "organize_hub", text
        assert result["slots"].get("module") == "meal_planner", text


def test_planning_prompts_are_never_misflagged_ambiguous():
    for text in PLANNING_NON_AMBIGUOUS_CASES:
        assert detect_action_ambiguity(text) is None, text


def test_substring_collisions_do_not_trigger_ambiguity_or_diet():
    for text in SUBSTRING_COLLISION_CASES:
        assert detect_action_ambiguity(text) is None, text
        assert _fallback_intent(text)["slots"].get("module") != "meal_planner", text


def test_ambiguity_reply_shape_has_message_and_chips():
    result = detect_action_ambiguity("Dinner tonight")
    assert result is not None
    assert result["message"]
    assert result["chips"]
    assert all({"label", "value"} <= set(chip) for chip in result["chips"])
