"""Canonical garment vocabulary must be single-sourced (services.style_item_contract
.garment_role_map/garment_words) and reused by every wardrobe-query detector,
instead of each detector hand-maintaining its own divergent word list (the
"jacket" gap: chat.py's list had it, brain/intent_engine.py's list had it,
but under different terms it silently drifted -- see the GO IMPLEMENT audit).
"""
from __future__ import annotations

import pytest

import brain.intent_engine as intent_engine
import routers.chat as chat
from brain.engines.wardrobe_selector import WardrobeSelector
from services.style_item_contract import garment_role_map, garment_words

CANONICAL_TERMS = [
    "top", "shirt", "blouse", "trousers", "pants", "jeans", "skirt",
    "jacket", "blazer", "coat", "shoes", "heels", "loafers", "dress",
    "kurta", "saree", "necklace", "earrings", "bag",
]


@pytest.mark.parametrize("term", CANONICAL_TERMS)
def test_fast_wardrobe_count_query_recognizes_canonical_term(term):
    assert chat._is_fast_wardrobe_count_query(f"how many {term} do I own")


@pytest.mark.parametrize("term", CANONICAL_TERMS)
def test_intent_engine_fallback_classifies_canonical_term_as_wardrobe_query(term):
    result = intent_engine._fallback_intent(f"how many {term} do I own")
    assert result["intent"] == "wardrobe_query", (term, result)


@pytest.mark.parametrize("term", CANONICAL_TERMS)
def test_wardrobe_selector_type_map_covers_canonical_term(term):
    assert term in WardrobeSelector.TYPE_MAP, term


def test_single_source_of_truth_no_duplicate_authoring():
    """WardrobeSelector.TYPE_MAP, chat.py's fast-path vocabulary, and
    intent_engine's wardrobe_words must all derive from the same canonical
    dict -- not independently hand-typed copies that can silently diverge."""
    canonical = garment_role_map()
    assert WardrobeSelector.TYPE_MAP == canonical
    # chat.py's fast-path check must recognize every canonical garment noun.
    for word in canonical:
        assert chat._is_fast_wardrobe_count_query(f"how many {word} do I have")


# ── Token-aware matching: no substring collisions ──────────────────────────

def test_spring_events_does_not_match_ring():
    assert not chat._is_fast_wardrobe_count_query("how many spring events do I have?")
    result = intent_engine._fallback_intent("how many spring events do I have?")
    assert result["intent"] != "wardrobe_query", result


def test_captions_does_not_match_cap():
    assert not chat._is_fast_wardrobe_count_query("how many captions are there?")
    result = intent_engine._fallback_intent("how many captions are there?")
    assert result["intent"] != "wardrobe_query", result


@pytest.mark.parametrize("term", ["jacket", "cap", "ring", "bag", "top"])
def test_representative_positive_wardrobe_nouns_still_match(term):
    assert chat._is_fast_wardrobe_count_query(f"how many {term} do I own?")
    result = intent_engine._fallback_intent(f"how many {term} do I own?")
    assert result["intent"] == "wardrobe_query", (term, result)


@pytest.mark.parametrize(
    "variant",
    ["one_piece", "one-piece", "one piece"],
)
def test_multiword_vocabulary_variants_all_match_regardless_of_separator(variant):
    from services.style_item_contract import has_garment_word

    assert has_garment_word(f"how many {variant} do I have")
