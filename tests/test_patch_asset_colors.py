"""Dry-run tests for the colour inference in scripts/patch_asset_colors.py.

Pure-function tests over `_infer_colors` — no Appwrite, no DB writes, no
network. Locks the word-boundary fix that stops substring false positives
("tailoRED" -> red) while keeping whole-word and slug-prefix matches.
"""

from __future__ import annotations

import pytest

from scripts.patch_asset_colors import _infer_colors


# ---------------------------------------------------------------------------
# False positives that the old naive substring match produced — must be EMPTY.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "Tailored Shorts",       # "tailoRED" -> previously matched red
        "Tapered Jeans",         # "tapeRED" -> previously matched red
        "Tailored Trousers",
        "Untailored Blazer",
        "Restored Denim",        # "REStored" -> previously matched red
    ],
)
def test_no_false_positive_red(text):
    assert "red" not in _infer_colors(text)


# ---------------------------------------------------------------------------
# Genuine whole-word colours — must match.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Tape red shirt", "red"),       # real standalone "red"
        ("Olive Green Chinos", "olive"),
        ("Gray Shirt", "grey"),          # canonicalised gray -> grey
        ("Navy Blazer", "navy"),
    ],
)
def test_whole_word_match(text, expected):
    assert expected in _infer_colors(text)


# ---------------------------------------------------------------------------
# Slug / concatenated-token prefixes (asset_id style) — must match.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("mens_assets_bags_blackbackpack", ["black"]),
        ("mens_assets_festive_sets_yellowkurtaset", ["yellow"]),
        ("mens_assets_festive_sets_greenkurtaset", ["green"]),
    ],
)
def test_slug_prefix_match(text, expected):
    assert _infer_colors(text) == expected


# ---------------------------------------------------------------------------
# Misspelling support (preserved + extended): brwon->brown, marron->maroon,
# burgendy->burgundy.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("mens_assets_bottoms_brwonjeans", "brown"),
        ("Mens Brwonjeans", "brown"),
        ("mens_assets_indian_ocassion_set_marronbandhgala", "maroon"),
        ("mens_assets_tops_burgendystripeshirt", "burgundy"),
        ("Mens Burgendyhalfhandsshirt", "burgundy"),
    ],
)
def test_misspelling_canonicalised(text, expected):
    assert _infer_colors(text) == [expected]


# ---------------------------------------------------------------------------
# Compound colours dedupe to the most specific token.
# ---------------------------------------------------------------------------
def test_compound_color_dedupes_to_specific():
    assert _infer_colors("navyblue blazer") == ["navyblue"]


def test_no_color_returns_empty():
    assert _infer_colors("Essential T-Shirt 01") == []
    assert _infer_colors("Oxford Shirt") == []
