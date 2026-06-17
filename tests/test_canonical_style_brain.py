"""Canonical Style Brain: occasion-family fix + brief gating + guard veto."""

import pytest

from services import style_context_service as scs
from services import stylist_knowledge_service as sks
from services import style_reasoning_engine as e

_ETHNIC_POOL = set(scs._ETHNIC_ARCHETYPES)


# --- 1/4: music festival + concert -> social_party (non-ethnic) ---
@pytest.mark.parametrize("occ", ["music festival", "concert", "gig", "rave", "edm", "club night"])
def test_music_festival_and_concert_are_social_party(occ):
    assert sks._resolve_occasion_family(occ) == "social_party"
    pool = set(sks._FAMILY_ARCHETYPE_POOL["social_party"])
    assert not (pool & _ETHNIC_POOL)  # no ethnic archetypes in the pool


# --- 2/3/5: cultural occasions stay festive/ethnic ---
@pytest.mark.parametrize(
    "occ,family",
    [
        ("diwali", "festive_general"),
        ("eid", "festive_general"),
        ("wedding", "festive_general"),
        ("haldi", "festive_daytime"),
        ("mehendi", "festive_daytime"),
        ("sangeet", "festive_evening"),
    ],
)
def test_cultural_occasions_stay_festive(occ, family):
    assert sks._resolve_occasion_family(occ) == family


def test_diwali_festival_phrase_still_ethnic():
    # "diwali festival" must NOT be hijacked by the music-festival rule.
    assert sks._resolve_occasion_family("diwali festival") == "festive_general"


# --- brief gating: western occasion forbids ethnic, ethnic occasion allows it ---
def test_music_festival_brief_forbids_ethnic():
    ctx = scs.build_canonical_style_context(query="music festival outfit", router_occasion="music festival")
    assert ctx["cultural_context"] == "western"
    fitems = " ".join(ctx["forbidden_item_signals"]).lower()
    for term in ("kurta", "bandhgala", "mojari", "jutti", "churidar"):
        assert term in fitems
    assert set(ctx["forbidden_archetypes"]) == _ETHNIC_POOL


def test_diwali_brief_allows_ethnic():
    ctx = scs.build_canonical_style_context(query="diwali party", router_occasion="diwali")
    assert ctx["cultural_context"] == "indian_ethnic"
    assert ctx["forbidden_item_signals"] == []
    assert ctx["forbidden_archetypes"] == []


def test_music_festival_with_desi_request_allows_ethnic():
    # acceptance #1 exception: user asks for desi/fusion.
    ctx = scs.build_canonical_style_context(query="fusion desi look for a music festival", router_occasion="music festival")
    assert ctx["cultural_context"] == "indian_ethnic"
    assert ctx["forbidden_item_signals"] == []


# --- guard veto: forbidden archetype dropped, forbidden items stripped ---
def _guard_ctx():
    return {
        "canonical_occasion": "rave",
        "gender": "male",
        "_query": "music festival",
        "forbidden_archetypes": list(scs._ETHNIC_ARCHETYPES),
        "forbidden_item_signals": list(scs._ETHNIC_ITEM_SIGNALS),
    }


def test_guard_drops_forbidden_ethnic_archetype(monkeypatch):
    monkeypatch.setenv("STYLE_SHARED_BRAIN", "true")
    dirs = [
        {"title": "Festive Look", "archetype": "Festive Heritage", "items": ["Silk Kurta"], "palette": ["gold"]},
        {"title": "Street", "archetype": "Smart Casual Edge", "items": ["Graphic Tee", "Cargo Pants", "Sneakers"], "palette": ["black"]},
    ]
    out = e._apply_style_guard(dirs, _guard_ctx())
    archs = [d.get("archetype") for d in out]
    assert "Festive Heritage" not in archs        # ethnic archetype vetoed
    assert "Smart Casual Edge" in archs           # street kept


def test_guard_strips_forbidden_ethnic_items(monkeypatch):
    monkeypatch.setenv("STYLE_SHARED_BRAIN", "true")
    dirs = [{
        "title": "Mixed", "archetype": "Smart Casual Edge",
        "hero_piece": "Embroidered Kurta",
        "items": ["Embroidered Kurta", "Mojari", "Graphic Tee", "Sneakers"],
        "palette": ["black"],
        "complete_the_look": [{"name": "Jutti"}, {"name": "Chain"}],
    }]
    out = e._apply_style_guard(dirs, _guard_ctx())
    blob = " ".join(out[0]["items"]).lower()
    assert "kurta" not in blob and "mojari" not in blob
    assert "graphic tee" in blob
    names = [i["name"].lower() for i in out[0]["complete_the_look"]]
    assert "jutti" not in names


# --- 6: flag off -> guard veto no-op ---
def test_flag_off_guard_noop(monkeypatch):
    monkeypatch.setenv("STYLE_SHARED_BRAIN", "false")
    dirs = [{"title": "Festive", "archetype": "Festive Heritage", "items": ["Silk Kurta"]}]
    out = e._apply_style_guard(dirs, _guard_ctx())
    assert out is dirs  # untouched
