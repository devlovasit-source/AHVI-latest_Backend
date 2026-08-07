"""
P0 pre-classifier: the smallest deterministic seam that separates the three
prompt classes the module-chat endpoint was mishandling.

- Style information ("what is …", "explain …")           → text_only
- Style advice     ("style tips", "how do I dress …")    → text_only
- Explicit inspiration ("show … inspiration/ideas")      → visual_inspiration
- Calendar navigation (bare "calendar", "open calendar") → calendar_navigation

Everything else returns ``None`` and the existing routing runs. The seam
lives here rather than inside ``routers/chat.py`` so the endpoint keeps
one call, not another 20-line ``if`` cascade.

ponytail: rule-based on purpose. When the full classifier exists, delete
this file — one import point in routers/chat.py to update.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize(text: str) -> str:
    t = str(text or "").lower().strip()
    t = _NON_ALNUM_RE.sub(" ", t)
    return _WHITESPACE_RE.sub(" ", t).strip()


# Bare navigation phrases. Full-message equality only — a sentence
# containing "calendar" ("meeting on my calendar tomorrow at 3pm") must
# still route to the calendar creation path, not navigation.
_CALENDAR_NAV_PHRASES = frozenset({
    "calendar",
    "open calendar",
    "show calendar",
    "show my calendar",
    "view calendar",
    "open my calendar",
    "my calendar",
})

# Explicit inspiration triggers. The user is asking to *see* looks or
# ideas, not for wardrobe recommendations.
_INSPIRATION_PATTERNS = (
    "show me * inspiration",
    "show * inspiration",
    "outfit inspiration",
    "outfit ideas",
    "style inspiration",
    "style ideas",
    "look ideas",
    "visual looks",
    "visual inspiration",
    "give me * ideas",
    "show me * ideas",
    "show me * looks",
    "show * looks",
)

# Style information: user is asking what a concept means.
_INFORMATION_PATTERNS = (
    "what is ",
    "what are ",
    "whats a ",
    "whats an ",
    "explain ",
    "define ",
    "meaning of ",
    "difference between ",
)

# Style advice: user is asking for guidance / tips without a specific outfit.
_ADVICE_PATTERNS = (
    "style tips",
    "styling tips",
    "give me style tips",
    "give me styling tips",
    "any style tips",
    "any styling tips",
    "how do i dress",
    "how should i dress",
    "how to dress",
    "how can i dress",
    "tips for ",
    "advice for ",
    "advice on ",
    "how do i improve my style",
    "how to improve my style",
)


def _matches_wildcard(needle: str, hay: str) -> bool:
    if "*" not in needle:
        return needle in hay
    parts = needle.split("*")
    idx = 0
    for part in parts:
        j = hay.find(part, idx)
        if j < 0:
            return False
        idx = j + len(part)
    return True


def _looks_like_calendar_creation(hay: str) -> bool:
    """Filter: a phrase with a time or a creation verb should never be nav."""
    if re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", hay):
        return True
    if re.search(r"\b(tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", hay):
        return True
    creation_verbs = ("remind", "schedule", "book", "add ", "set ", "appointment", "meeting with")
    return any(v in hay for v in creation_verbs)


def classify_message(text: str) -> Optional[Dict[str, str]]:
    """
    Return a canonical classification dict, or ``None`` when the pre-classifier
    has no confident answer and the caller should run its normal routing.

    Keys returned when non-None:
      - domain
      - intent
      - action
      - response_mode
    """
    hay = _normalize(text)
    if not hay:
        return None

    # 1. Calendar navigation — full-message equality, not substring.
    if hay in _CALENDAR_NAV_PHRASES and not _looks_like_calendar_creation(hay):
        return {
            "domain": "calendar",
            "intent": "navigate",
            "action": "open_calendar",
            "response_mode": "calendar_navigation",
        }

    # 2. Explicit inspiration — matches before advice/information because
    # "show me minimalist style ideas" contains both "style" and "ideas".
    for pattern in _INSPIRATION_PATTERNS:
        if _matches_wildcard(pattern, hay):
            return {
                "domain": "style",
                "intent": "inspiration",
                "action": "provide_visual_inspiration",
                "response_mode": "visual_inspiration",
            }

    # 3. Style information — question shape wins even inside style chat.
    for prefix in _INFORMATION_PATTERNS:
        if hay.startswith(prefix):
            return {
                "domain": "style",
                "intent": "information",
                "action": "explain_style_concept",
                "response_mode": "text_only",
            }

    # 4. Style advice.
    for pattern in _ADVICE_PATTERNS:
        if pattern in hay:
            return {
                "domain": "style",
                "intent": "advice",
                "action": "provide_style_advice",
                "response_mode": "text_only",
            }

    return None


if __name__ == "__main__":
    cases = [
        ("give me style tips",              "text_only",           "advice"),
        ("what is color analysis?",         "text_only",           "information"),
        ("Explain smart casual",            "text_only",           "information"),
        ("what is a capsule wardrobe?",     "text_only",           "information"),
        ("How can I dress better?",         "text_only",           "advice"),
        ("Tips for broad shoulders",        "text_only",           "advice"),
        ("show me brunch outfit inspiration", "visual_inspiration","inspiration"),
        ("Give me minimalist outfit ideas", "visual_inspiration",  "inspiration"),
        ("calendar",                        "calendar_navigation", "navigate"),
        ("open calendar",                   "calendar_navigation", "navigate"),
        ("meeting with alex tomorrow at 3pm", None,                None),
        ("what should I wear today?",       None,                  None),
        ("style this belt",                 None,                  None),
    ]
    for text, expected_mode, expected_intent in cases:
        got = classify_message(text)
        got_mode = got["response_mode"] if got else None
        got_intent = got["intent"] if got else None
        assert got_mode == expected_mode, (text, got_mode, expected_mode)
        assert got_intent == expected_intent, (text, got_intent, expected_intent)
    print("pre_classifier self-check ok")
