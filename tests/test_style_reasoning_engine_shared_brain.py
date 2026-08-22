"""Targeted tests for the Shared Style Brain (Phases A/B/C).

Pure tests — no Gemini, no DB. They exercise the canonical context builder, the
color-clash helper, and the post-LLM visual guard. The feature flag
``STYLE_SHARED_BRAIN`` defaults OFF; tests that need the guard active set it.
"""

import pytest

from brain.engines.styling import color_compatibility as cc
from services import style_context_service as scs
from services import style_reasoning_engine as e


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _default_flag_off(monkeypatch):
    # Ensure each test starts with the flag explicitly OFF; tests opt in.
    monkeypatch.setenv("STYLE_SHARED_BRAIN", "false")


def _on(monkeypatch):
    monkeypatch.setenv("STYLE_SHARED_BRAIN", "true")


def _ctx(occasion="daily", gender="male", weather=None, query=""):
    return {
        "canonical_occasion": occasion,
        "gender": gender,
        "weather": weather,
        "_query": query,
    }


# --------------------------------------------------------------------------
# 1. STYLE_SHARED_BRAIN=false is a no-op.
# --------------------------------------------------------------------------
def test_guard_is_noop_when_flag_off(monkeypatch):
    # autouse fixture already set it false.
    directions = [{"title": "Look", "palette": ["burgundy", "green"], "items": ["x"]}]
    out = e._apply_style_guard(directions, _ctx(occasion="office", gender="male"))
    assert out is directions  # untouched, same object
    assert out[0]["palette"] == ["burgundy", "green"]


# --------------------------------------------------------------------------
# 2. Canonical context contains canonical_occasion and gender.
# --------------------------------------------------------------------------
def test_canonical_context_has_occasion_and_gender():
    ctx = scs.build_canonical_style_context(
        query="haldi ceremony tomorrow",
        user_profile={"gender": "female"},
        router_occasion="haldi",
    )
    assert "canonical_occasion" in ctx and ctx["canonical_occasion"]
    assert ctx["gender"] == "female"
    # full contract shape present
    for key in ("style_dna", "profile", "weather", "event_context"):
        assert key in ctx


def test_canonical_context_fails_open_on_empty():
    ctx = scs.build_canonical_style_context(query="", user_profile=None)
    assert ctx["canonical_occasion"]  # safe default, never empty
    assert ctx["gender"] == "unknown"
    assert ctx["style_dna"] == {}


def test_canonical_context_does_not_mutate_profile():
    profile = {"gender": "male"}
    snapshot = dict(profile)
    scs.build_canonical_style_context(query="office", user_profile=profile)
    assert profile == snapshot


# --------------------------------------------------------------------------
# 3. Male user does not receive crop top / bralette.
# --------------------------------------------------------------------------
def test_male_user_no_crop_top_or_bralette(monkeypatch):
    _on(monkeypatch)
    directions = [
        {
            "title": "Look",
            "hero_piece": "White Oxford Shirt",
            "items": ["White Oxford Shirt", "Crop Top", "Bralette", "Tailored Trousers"],
            "palette": ["white", "navy"],
        }
    ]
    out = e._apply_style_guard(directions, _ctx(occasion="daily", gender="male"))
    blob = " ".join(out[0].get("items", [])).lower()
    assert "crop" not in blob
    assert "bralette" not in blob
    assert "white oxford shirt" in blob  # legit male piece kept


# --------------------------------------------------------------------------
# 4. Female user receives female-compatible assets (feminine pieces survive).
# --------------------------------------------------------------------------
def test_female_user_keeps_female_compatible_items(monkeypatch):
    _on(monkeypatch)
    directions = [
        {
            "title": "Look",
            "hero_piece": "Silk Blouse",
            "items": ["Silk Blouse", "Midi Skirt", "Block Heels"],
            "palette": ["cream", "navy"],
        }
    ]
    out = e._apply_style_guard(directions, _ctx(occasion="daily", gender="female"))
    blob = " ".join(out[0].get("items", [])).lower()
    assert "blouse" in blob
    assert "skirt" in blob
    assert "heels" in blob


# --------------------------------------------------------------------------
# 5. Haldi/wedding visual board remains ethnic/festive (western formal repaired).
# --------------------------------------------------------------------------
def test_wedding_board_strips_casual_western(monkeypatch):
    _on(monkeypatch)
    festive = {
        "title": "Festive Ethnic",
        "hero_piece": "Embroidered Kurta",
        "items": ["Embroidered Kurta", "Churidar", "Mojari"],
        "palette": ["marigold yellow", "ivory", "gold"],
    }
    casual = {
        "title": "Weekend Casual",
        "hero_piece": "Cotton Tee",
        "items": ["Cotton Tee", "Gym Shorts", "Sneakers"],
        "palette": ["grey"],
    }
    out = e._apply_style_guard([festive, casual], _ctx(occasion="haldi", gender="female"))
    titles = [d.get("title") for d in out]
    assert "Festive Ethnic" in titles
    # Casual board with gym shorts is rejected/repaired for a wedding occasion.
    casual_out = next((d for d in out if d.get("title") == "Weekend Casual"), None)
    if casual_out is not None:
        assert "gym shorts" not in " ".join(casual_out.get("items", [])).lower()
    # Never empty.
    assert out


# --------------------------------------------------------------------------
# 6. Airport avoids beanie unless weather is cold.
# --------------------------------------------------------------------------
def test_airport_drops_beanie_when_not_cold(monkeypatch):
    _on(monkeypatch)
    directions = [
        {
            "title": "Airport",
            "hero_piece": "Relaxed Overshirt",
            "items": ["Relaxed Overshirt", "Beanie", "Joggers", "Sneakers"],
            "palette": ["grey", "navy"],
        }
    ]
    out = e._apply_style_guard(directions, _ctx(occasion="airport", gender="male", weather=None))
    assert "beanie" not in " ".join(out[0]["items"]).lower()


def test_airport_keeps_beanie_when_cold(monkeypatch):
    _on(monkeypatch)
    directions = [
        {
            "title": "Airport",
            "hero_piece": "Relaxed Overshirt",
            "items": ["Relaxed Overshirt", "Beanie", "Joggers", "Sneakers"],
            "palette": ["grey", "navy"],
        }
    ]
    out = e._apply_style_guard(
        directions, _ctx(occasion="airport", gender="male", weather={"temp_c": 2})
    )
    assert "beanie" in " ".join(out[0]["items"]).lower()


# --------------------------------------------------------------------------
# 7. Burgundy + green clash is removed/repaired.
# --------------------------------------------------------------------------
def test_color_clash_helper_pairs():
    assert cc.colors_clash("burgundy", "green") is True
    assert cc.colors_clash("red", "green") is True
    assert cc.colors_clash("orange", "purple") is True
    assert cc.colors_clash("yellow", "purple") is True
    assert cc.colors_clash("pink", "orange") is True
    # neutrals always safe
    assert cc.colors_clash("navy", "white") is False
    assert cc.colors_clash("black", "burgundy") is False
    # navy/black special-cased at palette level only
    assert cc.colors_clash("navy", "black") is False
    assert ("navy", "black") in cc.palette_clashes(["navy", "black"])
    assert cc.palette_clashes(["navy", "black", "white"]) == []


def test_guard_strips_burgundy_green_clash(monkeypatch):
    _on(monkeypatch)
    directions = [
        {
            "title": "Look",
            "hero_piece": "Burgundy Blazer",
            "items": ["Burgundy Blazer", "Cream Shirt"],
            "palette": ["burgundy", "green", "cream"],
            "complete_the_look": [
                {"name": "Green Pocket Square", "color": "green"},
                {"name": "Cream Tie", "color": "cream"},
            ],
        }
    ]
    out = e._apply_style_guard(directions, _ctx(occasion="daily", gender="male"))
    assert "green" not in [c.lower() for c in out[0]["palette"]]
    names = [i["name"] for i in out[0]["complete_the_look"]]
    assert "Green Pocket Square" not in names
    assert "Cream Tie" in names


def test_guard_skips_color_check_when_no_color(monkeypatch):
    _on(monkeypatch)
    directions = [
        {
            "title": "Look",
            "hero_piece": "Burgundy Blazer",
            "palette": ["burgundy"],
            "complete_the_look": [{"name": "Mystery Item"}],  # no color, no color word
        }
    ]
    out = e._apply_style_guard(directions, _ctx(occasion="daily", gender="male"))
    names = [i["name"] for i in out[0]["complete_the_look"]]
    assert "Mystery Item" in names  # skipped, not dropped


# --------------------------------------------------------------------------
# 9. Guard never returns empty directions.
# --------------------------------------------------------------------------
def test_guard_never_returns_empty(monkeypatch):
    _on(monkeypatch)
    # Office occasion with only an unrepairable casual board (metadata conflict
    # via title text the repair can't strip) -> falls back to originals.
    directions = [
        {
            "title": "Beach Party Boardroom",  # title text triggers office reject
            "hero_piece": "Gym Shorts",
            "items": ["Gym Shorts", "Flip Flops"],
            "palette": ["grey"],
        }
    ]
    out = e._apply_style_guard(directions, _ctx(occasion="office", gender="male"))
    assert out  # never blank
    assert isinstance(out, list) and len(out) >= 1


def test_guard_passthrough_on_empty_input(monkeypatch):
    _on(monkeypatch)
    assert e._apply_style_guard([], _ctx()) == []
    assert e._apply_style_guard(None, _ctx()) is None


# --------------------------------------------------------------------------
# 8 + 10. Schema fields unchanged + empty wardrobe shows no fake match %.
# --------------------------------------------------------------------------
def _visual_payload():
    return {
        "mode": e.VISUAL_INSPIRATION,
        "occasion": "daily",
        "goal": "Look sharp",
        "impression": "Confident",
        "visual_directions": [
            {
                "title": "Refined Weekend",
                "archetype": "Refined Weekend",
                "hero_piece": "Navy Overshirt",
                "items": ["Navy Overshirt", "White Tee", "Tan Loafers"],
                "palette": ["navy", "white", "tan"],
            }
        ],
    }


def test_visual_payload_fields_preserved_with_flag_on(monkeypatch):
    _on(monkeypatch)
    built = e._build_response(
        query="what should I wear today",
        mode=e.VISUAL_INSPIRATION,
        category="daily",
        tone="neutral",
        formality="mid",
        occasion="daily",
        confidence=0.9,
        ai_payload=_visual_payload(),
        user_profile={"gender": "male"},
        context={"occasion": "daily", "wardrobe_items": []},
    )
    # Top-level schema fields present.
    for key in ("mode", "occasion", "visual_directions", "missing_piece", "meta"):
        assert key in built
    assert isinstance(built["visual_directions"], list) and built["visual_directions"]
    d0 = built["visual_directions"][0]
    for key in ("direction_name", "archetype", "hero_piece", "items"):
        assert key in d0


def test_empty_wardrobe_no_fake_match_pct(monkeypatch):
    _on(monkeypatch)
    built = e._build_response(
        query="what should I wear today",
        mode=e.VISUAL_INSPIRATION,
        category="daily",
        tone="neutral",
        formality="mid",
        occasion="daily",
        confidence=0.9,
        ai_payload=_visual_payload(),
        user_profile={"gender": "male"},
        context={"occasion": "daily", "wardrobe_items": []},
    )
    # With no wardrobe, ownership match % must not be fabricated.
    d0 = built["visual_directions"][0]
    assert d0.get("wardrobe_match_pct") in (None, 0)
    assert d0.get("owned_count", 0) == 0


def test_flag_off_build_response_matches_baseline(monkeypatch):
    # Flag OFF (autouse) -> guard never runs; directions still well-formed.
    built = e._build_response(
        query="what should I wear today",
        mode=e.VISUAL_INSPIRATION,
        category="daily",
        tone="neutral",
        formality="mid",
        occasion="daily",
        confidence=0.9,
        ai_payload=_visual_payload(),
        user_profile={"gender": "male"},
        context={"occasion": "daily", "wardrobe_items": []},
    )
    assert built["visual_directions"]
    assert "direction_name" in built["visual_directions"][0]
