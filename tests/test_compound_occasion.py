"""Compound occasion pre-resolver.

Compound prompts (connector + 2+ events) must resolve to the higher
social/formality-risk event, never the casual half. Single-occasion prompts
must remain exactly as before.
"""

from __future__ import annotations

import pytest

from brain.engines.style_brief import (
    detect_occasion_from_tokens,
    detect_compound_context,
    build_brief,
)


def _occ(query: str) -> str:
    return detect_occasion_from_tokens(query)[0]


# ---------- compound: higher-formality event wins ----------

@pytest.mark.parametrize(
    "query,banned",
    [
        ("networking dinner after a beach wedding", {"beach", "swimming"}),
        ("wedding reception after work", {"office", "office_meeting"}),
        ("gym then brunch", {"workout"}),
        ("pool party then dinner", {"swimming", "beach"}),
        ("workout then office", {"workout"}),
        ("client meeting followed by dinner", {"casual", "daily", "date_night"}),
    ],
)
def test_compound_picks_higher_formality(query, banned):
    occ = _occ(query)
    assert occ not in banned, f"{query!r} resolved to {occ!r}"


def test_wedding_reception_after_work_is_wedding():
    assert _occ("wedding reception after work") in {"wedding", "wedding_guest"}


def test_gym_then_brunch_is_brunch():
    assert _occ("gym then brunch") == "brunch"


def test_pool_party_then_dinner_is_dinner():
    assert _occ("pool party then dinner") == "date_night"


def test_workout_then_office_is_office():
    assert _occ("workout then office") in {"office", "office_meeting"}


def test_airport_then_office_meeting_is_office():
    assert _occ("airport then office meeting") in {"office", "office_meeting"}


def test_client_meeting_followed_by_dinner_is_client():
    assert _occ("client meeting followed by dinner") in {
        "client_meeting",
        "client_presentation",
        "client_dinner",
    }


def test_conference_and_cocktails_is_professional():
    # "and" alone is not a connector -> single path; conference outranks
    # cocktail by priority anyway.
    assert _occ("conference talk and cocktails") in {
        "client_presentation",
        "client_meeting",
    }


# ---------- single occasion: unchanged ----------

@pytest.mark.parametrize(
    "query,expected",
    [
        ("workout outfit", "workout"),
        ("beach vacation", "beach"),
        ("coffee date", "coffee_date"),
        ("client meeting tomorrow", "client_meeting"),
    ],
)
def test_single_occasion_unchanged(query, expected):
    assert _occ(query) == expected


def test_pool_party_unchanged_not_compound():
    # No connector -> never compound; stays whatever the single resolver gives.
    assert detect_compound_context("pool party") is None
    assert _occ("pool party") in {"beach", "swimming", "party"}


def test_cousin_wedding_unchanged():
    assert _occ("cousin wedding") in {"wedding", "wedding_guest"}


# ---------- compound metadata ----------

def test_compound_context_metadata():
    ctx = detect_compound_context("wedding reception after work")
    assert ctx is not None
    assert ctx["is_compound"] is True
    assert ctx["primary_occasion"] in {"wedding", "wedding_guest"}
    assert ctx["transition_required"] is True
    assert "polished" in ctx["compound_note"]


def test_single_prompt_no_compound_metadata():
    assert detect_compound_context("client meeting tomorrow") is None


def test_build_brief_attaches_compound():
    brief = build_brief("networking dinner after a beach wedding")
    assert brief.get("is_compound") is True
    assert brief["compound"]["primary_occasion"] in {"wedding", "wedding_guest"}
    assert brief["occasion"] not in {"beach", "swimming"}
