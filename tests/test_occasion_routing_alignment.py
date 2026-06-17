"""Regression: demo-critical occasion routing aligns across all resolvers.

Locks the 'rave' substring bug ("airport travel" matched the music rule via
t-RAVE-l) and the cross-engine drift for the demo prompts.
"""

import pytest

from brain.engines import occasion_style_rules as osr
from brain.engines import style_scorer as ss
from services import stylist_knowledge_service as sks


# --- the actual bug: word-boundary, not substring ---
def test_travel_not_matched_by_rave_substring():
    assert sks._resolve_occasion_family("airport travel") == "travel_easy"
    assert sks._resolve_occasion_family("travel") == "travel_easy"


def test_rave_still_resolves_social_party():
    for occ in ("rave", "music festival", "concert", "gig", "edm", "club night"):
        assert sks._resolve_occasion_family(occ) == "social_party"


# --- must-ensures: music festival is NOT festive/wedding anywhere ---
@pytest.mark.parametrize("occ", ["music festival", "concert", "gig"])
def test_music_festival_not_festive(occ):
    assert sks._resolve_occasion_family(occ) == "social_party"
    assert osr.get_occasion_rule(occ)["_occasion_key"] == "concert_social"
    assert ss.normalize_occasion(occ) == "party"  # was 'daily'


# --- alignment table for the demo prompts ---
@pytest.mark.parametrize(
    "prompt,family,osr_key,scorer",
    [
        ("airport travel", "travel_easy", "travel", "travel"),
        ("coffee date", "evening_date", "coffee_date", "coffee_date"),
        ("beach vacation", "resort_summer", "beach", "beach"),
        ("office meeting", "professional", "office_meeting", "office_meeting"),
        ("wedding guest", "festive_general", "wedding_guest", "wedding_guest"),
        ("sangeet", "festive_evening", "wedding", "wedding"),
    ],
)
def test_demo_prompt_alignment(prompt, family, osr_key, scorer):
    assert sks._resolve_occasion_family(prompt) == family
    assert osr.get_occasion_rule(prompt)["_occasion_key"] == osr_key
    assert ss.normalize_occasion(prompt) == scorer
