"""Regression coverage for the occasion-word-alone false positive found via
live /api/text verification: "Work was horrible today" (and other short,
purely emotional sentences containing an occasion word) was routing to
style_advice even though the message carries no styling intent at all.

Root cause: `is_style_advice_request`'s old `short_occasionish` fallback
admitted style_advice whenever a message was <=4 words and contained ANY
occasion word (office, work, dinner, wedding, party, date, ...), with no
requirement for an actual style/fashion word. Fixed by requiring an explicit
style word (compact_markers/advice_markers) whenever a generic occasion word
is present, keeping only a narrow, already-documented bare 1-2 word shortcut
for festive/ceremony and professional-speaking terms (Haldi, Sangeet,
Diwali party, keynote, conference talk).
"""

from services.stylist_knowledge_service import classify_style_mode, is_style_advice_request

NEGATIVE_CASES = (
    "Work was horrible today",
    "I hate going to the office",
    "Dinner was terrible",
    "The wedding was exhausting",
    "The party was boring",
    "I had a rough meeting",
    "Work is stressing me out",
    "I dont want to go to the office tomorrow",
    "The date went badly",
    "Vacation planning is exhausting",
)

# Ambiguous bare occasion + time-word phrases: no existing product contract
# treats these as a Style shortcut, so they must stay out of Style rather
# than getting a newly invented one.
AMBIGUOUS_CASES = (
    "Work tomorrow",
    "Dinner tonight",
    "Wedding tomorrow",
    "Office tomorrow",
    "Party tonight",
)

POSITIVE_CASES = (
    "What should I wear to work today?",
    "Give me an office outfit",
    "What should I wear for dinner tonight?",
    "Style me for a wedding",
    "Help me dress for the party",
    "What should I wear to my meeting?",
    "Build me a work look",
    "What outfit works for a date?",
    "Help me get dressed for vacation",
)

# Existing, code-documented bare 1-2 word shortcut: naming one of these
# ceremony/festival or professional-speaking terms alone is already treated
# as a styling ask elsewhere in the product; the fix must not remove it.
PRESERVED_BARE_SHORTCUTS = (
    "Haldi",
    "Sangeet",
    "Diwali party",
    "keynote",
    "conference talk",
)


def test_occasion_word_alone_does_not_trigger_style_advice():
    for text in NEGATIVE_CASES:
        assert not is_style_advice_request(text), text
        assert classify_style_mode(text) == "", text


def test_ambiguous_occasion_plus_time_word_stays_out_of_style():
    for text in AMBIGUOUS_CASES:
        assert not is_style_advice_request(text), text


def test_explicit_style_requests_still_trigger_style_advice():
    for text in POSITIVE_CASES:
        assert is_style_advice_request(text) or classify_style_mode(text), text


def test_bare_festive_and_professional_shortcuts_preserved():
    for text in PRESERVED_BARE_SHORTCUTS:
        assert is_style_advice_request(text), text
