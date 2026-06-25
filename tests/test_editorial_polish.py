"""Tests for the editorial-UX response polish (additive payload fields)."""

from __future__ import annotations

from services import style_reasoning_engine as engine


# ---------- _two_sentences ----------

def test_two_sentences_keeps_first_two():
    text = "First sentence. Second sentence. Third sentence is dropped."
    out = engine._two_sentences(text)
    assert out == "First sentence. Second sentence."


def test_two_sentences_caps_long_paragraph():
    text = "x" * 600 + ". y" * 5
    out = engine._two_sentences(text, max_chars=240)
    assert len(out) <= 241
    assert out.endswith("…")


def test_two_sentences_handles_empty():
    assert engine._two_sentences("") == ""
    assert engine._two_sentences(None) == ""


def test_two_sentences_passes_through_short_text():
    assert engine._two_sentences("Quiet polish.") == "Quiet polish."


# ---------- adjectives + curated_for ----------

def test_direction_adjectives_known_archetype():
    assert engine._direction_adjectives_from_archetype("Modern Professional") == [
        "Confident",
        "Structured",
        "Approachable",
    ]


def test_direction_adjectives_unknown_archetype_falls_back():
    out = engine._direction_adjectives_from_archetype("Some New Archetype")
    assert len(out) == 3
    assert all(isinstance(s, str) and s for s in out)


def test_curated_for_handles_conference_talk_variants():
    assert engine._curated_for_for_occasion("conference_talk") == [
        "Stage Presence",
        "Networking",
        "All-Day Comfort",
    ]
    assert engine._curated_for_for_occasion("conference talk") == [
        "Stage Presence",
        "Networking",
        "All-Day Comfort",
    ]


def test_curated_for_handles_client_meeting():
    out = engine._curated_for_for_occasion("client_meeting")
    assert "Quiet Authority" in out


def test_curated_for_unknown_occasion_uses_default():
    out = engine._curated_for_for_occasion("intergalactic_brunch")
    assert len(out) == 3


# ---------- wardrobe match pct ----------

def test_wardrobe_match_pct_full_overlap():
    direction = {"items": ["Navy Blazer", "White Shirt", "Grey Trouser"]}
    wardrobe = [
        {"name": "Navy Blazer", "category": "outerwear"},
        {"name": "White Shirt", "category": "top"},
        {"name": "Grey Trouser", "category": "bottom"},
    ]
    assert engine._wardrobe_match_pct(direction, wardrobe) == 100


def test_wardrobe_match_pct_no_wardrobe_returns_none():
    direction = {"items": ["Navy Blazer"]}
    assert engine._wardrobe_match_pct(direction, None) is None
    assert engine._wardrobe_match_pct(direction, []) is None


def test_wardrobe_match_pct_partial():
    direction = {"items": ["Navy Blazer", "Linen Shorts", "Brown Loafer"]}
    wardrobe = [{"name": "Brown Loafer"}]
    pct = engine._wardrobe_match_pct(direction, wardrobe)
    assert pct is not None and 0 < pct < 100


# ---------- _editorial_badge ----------

def test_badge_label_excellent_for_high_match():
    badge = engine._editorial_badge(95)
    assert badge["occasion_fit"] == "Excellent"
    assert badge["stars"] == 5
    assert badge["label"] == "Recommended"


def test_badge_label_good_for_mid_match():
    assert engine._editorial_badge(45)["occasion_fit"] == "Good"


def test_badge_handles_none_match():
    badge = engine._editorial_badge(None)
    assert badge["wardrobe_match_pct"] is None
    assert badge["occasion_fit"] == "Strong"


# ---------- complete the look copy ----------

def test_complete_the_look_copy_with_missing_piece():
    out = engine._complete_the_look_copy(
        {"name": "Brushed Steel Watch"},
        "conference_talk",
    )
    assert "Brushed Steel Watch" not in out  # name lives in missing_piece
    assert "conference talk" in out
    assert "polished" in out.lower()


def test_complete_the_look_copy_returns_empty_without_missing():
    assert engine._complete_the_look_copy(None, "conference") == ""
    assert engine._complete_the_look_copy({}, "conference") == ""


# ---------- _apply_editorial_polish ----------

def _sample_directions():
    return [
        {
            "archetype": "Modern Professional",
            "title": "Modern Authority",
            "items": ["Navy Blazer", "White Shirt", "Grey Trouser", "Black Loafer"],
            "pieces": ["Navy Blazer", "White Shirt", "Grey Trouser", "Black Loafer"],
            "why_it_works": (
                "The blazer adds authority while the open collar keeps the look approachable for "
                "networking sessions. It pairs cleanly with the trouser. The loafer keeps it grounded."
            ),
            "missing_piece": {"name": "Brushed Steel Watch", "category": "accessory"},
        },
        {
            "archetype": "Executive Minimalist",
            "title": "Clean Edge",
            "items": ["Charcoal Suit", "White Shirt", "Black Loafer"],
            "pieces": ["Charcoal Suit", "White Shirt", "Black Loafer"],
            "why_it_works": "Clean lines. Quiet palette.",
        },
    ]


def test_apply_editorial_polish_decorates_each_direction():
    out = engine._apply_editorial_polish(
        _sample_directions(),
        occasion="conference_talk",
        wardrobe_items=[
            {"name": "Navy Blazer"},
            {"name": "White Shirt"},
            {"name": "Grey Trouser"},
        ],
    )
    first = out[0]
    assert first["direction_name"] == "Modern Professional"
    assert first["adjectives"] == ["Confident", "Structured", "Approachable"]
    assert first["wardrobe_match_pct"] >= 50
    assert first["badge"]["occasion_fit"] in {"Excellent", "Strong", "Good"}
    assert first["curated_for"] == ["Stage Presence", "Networking", "All-Day Comfort"]
    assert first["complete_the_look_copy"].startswith("One piece away")
    # short_note is the 2-sentence cap of why_it_works.
    assert first["short_note"].count(".") <= 3
    assert "loafer keeps it grounded" not in first["short_note"]


def test_apply_editorial_polish_preserves_legacy_fields():
    directions = _sample_directions()
    out = engine._apply_editorial_polish(
        directions,
        occasion="conference_talk",
        wardrobe_items=None,
    )
    legacy = ("archetype", "title", "items", "pieces", "why_it_works", "missing_piece")
    for field in legacy:
        if field in directions[0]:
            assert field in out[0]


def test_apply_editorial_polish_no_wardrobe_signal_keeps_pct_none():
    out = engine._apply_editorial_polish(
        _sample_directions(),
        occasion="conference_talk",
        wardrobe_items=None,
    )
    assert out[0]["wardrobe_match_pct"] is None
    assert out[0]["badge"]["wardrobe_match_pct"] is None


# ---------- _build_editorial_cover ----------

def test_build_editorial_cover_uses_top_direction_and_max_match():
    decorated = engine._apply_editorial_polish(
        _sample_directions(),
        occasion="conference_talk",
        wardrobe_items=[{"name": "Navy Blazer"}, {"name": "White Shirt"}],
    )
    cover = engine._build_editorial_cover(decorated, occasion="conference_talk")
    assert cover["occasion_label"] == "CONFERENCE TALK"
    assert cover["direction_name"] == "Modern Professional"
    assert cover["wardrobe_match_pct"] is not None
    assert cover["curated_for"][0] == "Stage Presence"
    assert cover["badge"]["label"] == "Recommended"


def test_build_editorial_cover_handles_empty_directions():
    cover = engine._build_editorial_cover([], occasion="coffee_date")
    assert cover["occasion_label"] == "COFFEE DATE"
    assert cover["direction_name"] == "Curated Look"
    assert cover["wardrobe_match_pct"] is None


# ---------- Ownership truth ----------

def _wardrobe_set():
    return [
        {"id": "w1", "name": "Navy Blazer", "category": "outerwear", "image_url": "https://w/1.png"},
        {"id": "w2", "name": "White Oxford Shirt", "category": "top"},
        {"id": "w3", "name": "Grey Trousers", "category": "bottom"},
        {"id": "w4", "name": "Black Loafer", "category": "footwear"},
        {"id": "w5", "name": "Steel Wristwatch", "category": "watch"},
        {"id": "w6", "name": "Apple Charger", "category": "electronics"},
        {"id": "w7", "name": "Skincare Set", "category": "grooming"},
        {"id": "w8", "name": "Neck Pillow", "category": "travel"},
    ]


def test_owned_items_picks_real_matches_only():
    direction = {
        "items": [
            "Navy Blazer",
            "White Shirt",
            "Grey Trouser",
            "Black Loafer",
            "Steel Watch",
        ],
        "pieces": [
            "Navy Blazer",
            "White Shirt",
            "Grey Trouser",
            "Black Loafer",
            "Steel Watch",
        ],
    }
    polished = engine._apply_editorial_polish(
        [direction], occasion="conference_talk", wardrobe_items=_wardrobe_set()
    )[0]
    names = [o["name"] for o in polished["owned_items"]]
    assert "Navy Blazer" in names
    assert "White Oxford Shirt" in names
    assert "Grey Trousers" in names
    assert "Steel Wristwatch" in names
    # Non-fashion items must never appear in the chip list.
    assert "Apple Charger" not in names
    assert "Skincare Set" not in names
    assert "Neck Pillow" not in names


def test_owned_count_drives_wardrobe_match_pct():
    direction = {
        "items": ["Navy Blazer", "White Shirt", "Grey Trouser", "Black Loafer", "Steel Watch"],
        "pieces": ["Navy Blazer", "White Shirt", "Grey Trouser", "Black Loafer", "Steel Watch"],
    }
    polished = engine._apply_editorial_polish(
        [direction], occasion="conference_talk", wardrobe_items=_wardrobe_set()
    )[0]
    assert polished["total_items"] == 5
    assert polished["owned_count"] == 5
    assert polished["wardrobe_match_pct"] == 100


def test_owned_items_drops_blocked_categories():
    direction = {"items": ["Charging Cable", "Neck Pillow", "Sunscreen"], "pieces": ["Charging Cable", "Neck Pillow", "Sunscreen"]}
    wardrobe = [
        {"name": "Charging Cable", "category": "electronics"},
        {"name": "Travel Neck Pillow", "category": "travel"},
        {"name": "Spf Sunscreen", "category": "skincare"},
    ]
    polished = engine._apply_editorial_polish(
        [direction], occasion="travel", wardrobe_items=wardrobe
    )[0]
    assert polished["owned_items"] == []
    assert polished["owned_count"] == 0


def test_owned_items_family_buckets_are_public():
    direction = {
        "items": ["Navy Blazer", "Brown Loafer", "Steel Watch", "Brown Messenger Bag"],
        "pieces": ["Navy Blazer", "Brown Loafer", "Steel Watch", "Brown Messenger Bag"],
    }
    wardrobe = [
        {"name": "Navy Blazer", "category": "outerwear"},
        {"name": "Brown Loafer", "category": "footwear"},
        {"name": "Steel Wristwatch", "category": "watch"},
        {"name": "Brown Messenger Bag", "category": "accessory"},
    ]
    polished = engine._apply_editorial_polish(
        [direction], occasion="client_meeting", wardrobe_items=wardrobe
    )[0]
    families = {o["family"] for o in polished["owned_items"]}
    allowed = {
        "top",
        "bottom",
        "dress",
        "footwear",
        "outerwear",
        "ethnicwear",
        "accessory",
        "bag",
        "watch",
        "jewellery",
    }
    assert families.issubset(allowed), families


def test_owned_items_absent_without_wardrobe_signal():
    direction = {"items": ["Navy Blazer", "Grey Trouser"], "pieces": ["Navy Blazer", "Grey Trouser"]}
    polished = engine._apply_editorial_polish(
        [direction], occasion="client_meeting", wardrobe_items=None
    )[0]
    assert polished["owned_items"] == []
    assert polished["owned_count"] == 0
    # total_items reflects the styled look size even without wardrobe data.
    assert polished["total_items"] == 2


# ---------- curated-archetype replace-vs-preserve (BUG 2) ----------

def test_free_text_archetype_is_replaced_with_curated_name():
    """An LLM-invented free-text archetype like 'Polished Daily' is NOT in any
    recognized registry, so it must be remapped to a real curated-library
    archetype name."""
    direction = {
        "archetype": "Polished Daily",
        "title": "Everyday Ease",
        "hero_piece": "Navy Blazer",
        "items": ["Navy Blazer", "White Shirt", "Tailored Trouser", "Loafers"],
        "pieces": ["Navy Blazer", "White Shirt", "Tailored Trouser", "Loafers"],
        "palette": ["navy", "white", "tan"],
    }
    out = engine._apply_editorial_polish(
        [direction], occasion="office", wardrobe_items=None
    )[0]
    assert out["archetype"] != "Polished Daily"
    # The replacement must be a genuine recognized curated/persona archetype.
    assert engine._is_recognized_archetype(out["archetype"])


def test_persona_archetype_is_preserved():
    """A genuine persona archetype produced by the visual_inspiration path
    (e.g. 'Creative Executive') is authoritative and must be preserved."""
    for persona in ("Creative Executive", "Approachable Executive", "Resort Sophisticate"):
        direction = {
            "archetype": persona,
            "title": "Direction",
            "items": ["Blazer", "Trouser", "Loafers"],
            "pieces": ["Blazer", "Trouser", "Loafers"],
        }
        out = engine._apply_editorial_polish(
            [direction], occasion="office", wardrobe_items=None
        )[0]
        assert out["archetype"] == persona, f"{persona} should be preserved"


def test_blank_archetype_still_assigned():
    """The original case — no archetype at all — still gets a curated name."""
    direction = {
        "title": "Direction",
        "items": ["Linen Shirt", "Linen Trouser", "Espadrilles"],
        "pieces": ["Linen Shirt", "Linen Trouser", "Espadrilles"],
    }
    out = engine._apply_editorial_polish(
        [direction], occasion="resort", wardrobe_items=None
    )[0]
    assert out["archetype"]
    assert engine._is_recognized_archetype(out["archetype"])


def test_recognized_archetype_names_includes_all_registries():
    names = engine._recognized_archetype_names()
    # persona library
    assert engine._normalize_archetype_key("Creative Executive") in names
    # generic visual strategies
    assert engine._normalize_archetype_key("Structured Ease") in names
    # festive replacements
    assert engine._normalize_archetype_key("Vibrant Celebration") in names
