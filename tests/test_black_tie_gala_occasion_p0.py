"""P0 hotfix: black-tie / gala occasion vocabulary gap.

"black-tie gala" (and phrase variants) fell through all three of AHVI's
occasion classifiers to an unrecognized bucket:

  1. routers/chat.py::_ahvi_style_occasion() - the free-text chat entry
     point's own keyword classifier, which had no gala/black-tie branch and
     fell all the way to its generic "today" daily-route default, bypassing
     occasion interpretation entirely (confirmed live: a physical-device
     request logged AHVI_DAILY_STYLE_ROUTE occasion=today for this exact
     prompt even after the pipeline-level fix below was deployed).
  2. brain/engines/style_scorer.py::normalize_occasion() - the pipeline-
     authoritative canonicalizer, whose result is filtered through a strict
     allowlist that zeroed out any unrecognized bucket to "".
  3. brain/engines/outfit_quality_guard.py::normalize_occasion() - a second,
     independent copy used inside reject_board_for_occasion(), sharing the
     identical gap.

Any of these dropping the gala/black-tie signal meant occasion collapsed to
casual/daily downstream, bypassing every formal-safety check keyed on the
occasion string. Fixed by teaching the existing "wedding" bucket (the one
that already activates outfit_quality_guard's formal-safety path) to
recognize "black tie" / "white tie" / "gala" as phrase/token-aware signals
in all three, plus a hyphen-normalization gap in
style_compatibility_rules._detect_explicit_dress_code() so "black-tie" text
triggers the same explicit dress_code="black_tie" detection as "black tie".

No new canonicalizer was introduced - all edits are in the three existing
occasion classifiers plus the one existing dress-code detector.
"""
import os

import pytest

from brain.engines.style_scorer import normalize_occasion as scorer_normalize_occasion
from brain.engines.outfit_quality_guard import (
    normalize_occasion as guard_normalize_occasion,
    guard_outfit,
)
from brain.engines.style_compatibility_rules import _detect_explicit_dress_code
from routers.chat import _ahvi_style_occasion

PHRASE_MATRIX = [
    # input, expected_scorer_occasion, expected_guard_occasion, expected_dress_code
    ("black tie", "wedding", "wedding", "black_tie"),
    ("black-tie", "wedding", "wedding", "black_tie"),
    ("black tie gala", "wedding", "wedding", "black_tie"),
    ("black-tie gala", "wedding", "wedding", "black_tie"),
    ("black-tie event", "wedding", "wedding", "black_tie"),
    ("black tie event", "wedding", "wedding", "black_tie"),
    ("formal gala", "wedding", "wedding", ""),
    ("gala dinner", "date_night", "date_night", ""),
    ("black tie wedding", "wedding", "wedding", "black_tie"),
    ("black tie reception", "wedding", "wedding", "black_tie"),
    ("cocktail party", "cocktail", "cocktail", ""),
    ("wedding reception", "wedding", "wedding", ""),
    ("office", "office", "office", ""),
    ("casual dinner", "casual_dinner", "casual_dinner", ""),
    ("date night", "date_night", "date_night", ""),
    # Documented pre-existing bug (normalize_occasion "date" in "candidate"
    # substring match) - must remain UNCHANGED by this fix, not silently
    # fixed here without a dedicated regression pass of its own.
    ("candidate interview", "date_night", "date_night", ""),
    ("gym", "workout", "workout", ""),
    ("workout", "workout", "workout", ""),
    ("beach", "beach", "beach", ""),
]

CHAT_ROUTER_PHRASE_MATRIX = [
    # input, expected _ahvi_style_occasion() bucket
    ("give me a black-tie gala outfit with sneakers", "wedding"),
    ("give me a black-tie gala outfit", "wedding"),
    ("black tie", "wedding"),
    ("black-tie", "wedding"),
    ("formal gala", "wedding"),
    ("candidate interview", "date night"),  # pre-existing, unchanged
    ("office meeting", "office"),
    # Token-boundary regression: "galaxy" contains "gala" as a substring but
    # must never be treated as a black-tie/gala signal.
    ("galaxy print shirt for casual outing", "casual outing"),
]


@pytest.mark.parametrize("text,expected", CHAT_ROUTER_PHRASE_MATRIX)
def test_chat_router_occasion_classifier(text, expected):
    assert _ahvi_style_occasion(text) == expected


@pytest.mark.parametrize("text,exp_scorer,exp_guard,exp_dress_code", PHRASE_MATRIX)
def test_occasion_phrase_matrix(text, exp_scorer, exp_guard, exp_dress_code):
    got_scorer = scorer_normalize_occasion(text)
    got_guard = guard_normalize_occasion(text)
    got_dress_code = _detect_explicit_dress_code(got_scorer, text)
    assert got_scorer == exp_scorer, f"scorer occasion for {text!r}"
    assert got_guard == exp_guard, f"guard occasion for {text!r}"
    assert got_dress_code == exp_dress_code, f"dress_code for {text!r}"


def _formal_outfit(footwear_name, footwear_formality="casual"):
    return {
        "top": {"id": "top1", "name": "White Dress Shirt", "role": "top", "category": "Tops",
                "style_metadata": {"formality": "formal", "style_role": "formalwear"}},
        "bottom": {"id": "bot1", "name": "Black Formal Trousers", "role": "bottom", "category": "Bottoms",
                   "style_metadata": {"formality": "formal", "style_role": "formalwear"}},
        "footwear": {"id": "shoe1", "name": footwear_name, "role": "footwear", "category": "Footwear",
                     "style_metadata": {"formality": footwear_formality}},
        "accessories": [],
    }


@pytest.fixture(autouse=True)
def _prod_matching_compat_flags(monkeypatch):
    # Production (ahvi-backend-00993-wap) runs with shadow mode OFF - match
    # that here so this test proves real enforcement, not shadow logging.
    monkeypatch.setenv("ENABLE_NEGATIVE_COMPATIBILITY_P0", "true")
    monkeypatch.setenv("NEGATIVE_COMPATIBILITY_SHADOW_MODE", "false")


def test_black_tie_gala_with_sneakers_is_blocked():
    query = "give me a black-tie gala outfit with sneakers"
    intent = scorer_normalize_occasion(query)
    assert intent == "wedding", "black-tie context must survive to a formal-safety-activating occasion"

    outfit = _formal_outfit("White Sneakers")
    allowed, penalty, reasons, fixed = guard_outfit(outfit, intent=intent, query=query)

    assert allowed is False
    assert any("DCV_006" in r for r in reasons), reasons
    meta = fixed.get("_quality_guard_meta") or {}
    assert meta.get("hard_invalid") is True
    assert meta.get("shadow_mode") is False


def test_black_tie_gala_with_loafers_passes():
    query = "give me a black-tie gala outfit with sneakers"
    intent = scorer_normalize_occasion(query)
    outfit = _formal_outfit("Black Formal Loafers", footwear_formality="formal")
    allowed, penalty, reasons, fixed = guard_outfit(outfit, intent=intent, query=query)
    assert allowed is True
    assert reasons == []


def test_black_tie_gala_outfit_board_complete_no_footwear_complaint():
    query = "give me a black-tie gala outfit"
    intent = scorer_normalize_occasion(query)
    assert intent == "wedding"
    outfit = _formal_outfit("Black Formal Loafers", footwear_formality="formal")
    allowed, penalty, reasons, fixed = guard_outfit(outfit, intent=intent, query=query)
    assert allowed is True
    assert all(fixed.get(k) for k in ("top", "bottom", "footwear"))


def test_casual_dinner_control_unaffected():
    # A regular casual request must never pick up dress-code enforcement.
    assert _detect_explicit_dress_code("casual", "give me a casual dinner outfit") == ""
