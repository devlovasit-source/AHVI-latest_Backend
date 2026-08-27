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
from routers.chat import _ahvi_style_occasion, _apply_current_turn_occasion_authority
from services.style_flow_service import finalize_style_response_payload, interpret_occasion
from services.style_conversation_context import resolve_style_conversation_context

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


# ---------------------------------------------------------------------------
# Follow-up P0 gap, found live on ahvi-backend-00998-xid (commit 1aeb117)
# after the three classifiers above were already fixed and deployed:
#
# /api/text's chat-orchestrator layer correctly resolved this prompt's
# occasion to "wedding" (proving the chat-router fix worked). But the same
# request also runs a second, independent board-generation step -
# services/style_flow_service.py's finalize_style_response_payload() /
# build_style_flow_response() - which re-derives its own occasion from raw
# query text via brain/engines/occasion_interpreter.py::detect_occasion(),
# a FOURTH classifier that was never taught "black tie" / "gala" vocabulary.
# It returned "daily", and that freshly re-derived (wrong) guess was checked
# *before* the caller-supplied, already-correct ctx["occasion"]="wedding" in
# an `or` chain, so it silently won. Live evidence: style_board.generate
# logged interpreted_occasion=daily and ahvi.final_boards badges=['DAILY']
# for this exact prompt on 00998-xid.
#
# Fixed by reordering that `or` chain so ctx["occasion"] - the value the
# caller already resolved - is checked first. No new classifier added.
# ---------------------------------------------------------------------------

def _wedding_board(footwear_name, footwear_formality="formal"):
    items = [
        {"id": "top1", "name": "White Dress Shirt", "role": "top", "category": "Tops"},
        {"id": "bot1", "name": "Black Formal Trousers", "role": "bottom", "category": "Bottoms"},
        {"id": "shoe1", "name": footwear_name, "role": "footwear", "category": "Footwear",
         "style_metadata": {"formality": footwear_formality}},
    ]
    return {"items": items, "top": items[0], "bottom": items[1], "footwear": items[2]}


def test_style_flow_board_generation_no_longer_defaults_to_daily():
    query = "give me a black-tie gala outfit with sneakers"

    # Confirms the underlying gap still exists in the raw-text reparser -
    # if this ever starts passing, the ctx["occasion"] precedence fix below
    # is still required regardless, since callers must not depend on every
    # downstream reparser knowing every occasion phrase.
    reparsed = interpret_occasion(query)
    assert reparsed.get("occasion") == "daily"

    board = _wedding_board("White Sneakers", footwear_formality="casual")
    response = finalize_style_response_payload(
        {"cards": [board], "outfits": [board]},
        user_id="u1",
        query=query,
        wardrobe=board["items"],
        context={"occasion": "wedding"},
    )

    resolved_occasion = (response.get("cards") or [{}])[0].get("occasion")
    assert resolved_occasion == "wedding", (
        f"board generation must keep the caller's resolved occasion; got {resolved_occasion!r}"
    )
    assert resolved_occasion not in {"daily", "today", "casual"}


# ---------------------------------------------------------------------------
# Second follow-up P0 gap, found live on ahvi-backend-01000-bav (commit
# 80a3b8b) via a real multi-turn device conversation:
#
#   Turn 1: "give me a casual dinner outfit"
#   Turn 2: "give me a black-tie gala outfit with sneakers"
#
# Turn 2's own text correctly resolved to occasion=wedding via chat.py's
# _ahvi_style_occasion(). But services/style_conversation_context.py's
# resolve_style_conversation_context() - which merges "current turn >
# carried state > history" - re-derives its OWN idea of the current turn's
# occasion via a separate, narrower keyword list
# (_occasion_from_text/_OCCASION_ALIASES) that also doesn't know
# "black tie"/"gala". Since that internal detector found nothing for Turn
# 2, the merge never had anything to assert current-turn priority with, so
# _text_conversation.occasion silently stayed on Turn 1's carried "dinner"
# value. This is a FIFTH classifier gap, in a different file than the four
# already fixed by 6cd2ccf/1aeb117/80a3b8b - but per this fix's scope, it is
# NOT patched with new vocabulary. Instead, chat.py's already-correct
# _ahvi_style_occasion() result (when not its own generic "today"
# no-signal sentinel) is applied on top of whatever
# resolve_style_conversation_context() produced, in
# _apply_current_turn_occasion_authority(). No new occasion map, regex, or
# black-tie special case was added anywhere.
# ---------------------------------------------------------------------------

def _turn_occasion(message, *, carried_occasion=None, history_messages=()):
    carried_context = {"resolved_context": {"occasion": carried_occasion}} if carried_occasion else {}
    history = [{"role": "user", "content": m} for m in history_messages]
    conversation, _ = resolve_style_conversation_context(
        current_message=message,
        recent_history=history,
        carried_context=carried_context,
    )
    conversation = _apply_current_turn_occasion_authority(conversation, message)
    return conversation.occasion


def test_current_turn_authority_ignores_its_own_generic_sentinel():
    # Direct unit check of the authority helper: "today" (the classifier's
    # own designated no-explicit-signal value) must never overwrite an
    # existing backfilled occasion.
    conversation, _ = resolve_style_conversation_context(
        current_message="another option",
        recent_history=[],
        carried_context={"resolved_context": {"occasion": "wedding"}},
    )
    assert _ahvi_style_occasion("another option") == "today"
    result = _apply_current_turn_occasion_authority(conversation, "another option")
    assert result.occasion == "wedding"


def test_sequence_a_black_tie_after_casual_dinner_wins():
    turn1 = _turn_occasion("give me a casual dinner outfit")
    turn2 = _turn_occasion(
        "give me a black-tie gala outfit with sneakers",
        carried_occasion=turn1,
        history_messages=["give me a casual dinner outfit"],
    )
    assert turn2 == "wedding"


def test_sequence_b_black_tie_after_office_wins():
    turn1 = _turn_occasion("give me an office outfit")
    turn2 = _turn_occasion(
        "give me a black tie gala outfit",
        carried_occasion=turn1,
        history_messages=["give me an office outfit"],
    )
    assert turn2 == "wedding"


def test_sequence_c_elliptical_followup_inherits_black_tie_context():
    turn1 = _turn_occasion("give me a black-tie gala outfit")
    assert turn1 == "wedding"
    turn2 = _turn_occasion(
        "make another one with loafers",
        carried_occasion=turn1,
        history_messages=["give me a black-tie gala outfit"],
    )
    assert turn2 == "wedding"


def test_sequence_d_explicit_new_occasion_replaces_wedding_context():
    turn1 = _turn_occasion("give me a wedding reception outfit")
    assert turn1 == "wedding"
    turn2 = _turn_occasion(
        "actually make it casual for dinner",
        carried_occasion=turn1,
        history_messages=["give me a wedding reception outfit"],
    )
    assert turn2 != "wedding", "an explicit new occasion in the current turn must replace prior context"


def test_sequence_e_elliptical_followup_inherits_casual_dinner_context():
    turn1 = _turn_occasion("give me a casual dinner outfit")
    turn2 = _turn_occasion(
        "another option",
        carried_occasion=turn1,
        history_messages=["give me a casual dinner outfit"],
    )
    assert turn2 == turn1, "an elliptical follow-up must inherit the prior turn's occasion, not reset to generic"


def test_sequence_f_candidate_interview_does_not_false_match_gala_or_black_tie():
    # "candidate interview" contains the substring "date", a pre-existing,
    # already-documented _ahvi_style_occasion quirk unrelated to this fix
    # (see test_chat_router_occasion_classifier above) - intentionally left
    # unchanged here. What this fix must guarantee is that it never
    # false-matches the black-tie/gala/wedding bucket specifically.
    result = _turn_occasion("candidate interview")
    assert result != "wedding"


def test_full_live_failure_sequence_resolves_to_wedding_end_to_end():
    # The exact live device conversation that exposed this gap.
    turn1 = _turn_occasion("give me a casual dinner outfit")
    turn2_query = "give me a black-tie gala outfit with sneakers"
    turn2_occasion = _turn_occasion(
        turn2_query,
        carried_occasion=turn1,
        history_messages=["give me a casual dinner outfit"],
    )
    assert turn2_occasion == "wedding"
    assert turn2_occasion not in {"client_dinner", "daily", "today", "casual"}

    # And the already-fixed downstream pipeline still enforces on it.
    intent = scorer_normalize_occasion(turn2_query)
    assert intent == "wedding"
    outfit = _formal_outfit("White Sneakers")
    allowed, penalty, reasons, fixed = guard_outfit(outfit, intent=intent, query=turn2_query)
    assert allowed is False
    assert any("DCV_006" in r for r in reasons), reasons
