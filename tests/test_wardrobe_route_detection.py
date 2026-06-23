"""Wardrobe-intent routing: natural mid-sentence prompts ("office outfit using
my wardrobe") must route WARDROBE, not VISUAL_INSPIRATION. Explicit visual
inspiration prompts must stay VISUAL_INSPIRATION."""
import os

import pytest

from services.stylist_knowledge_service import is_wardrobe_style_request

WARDROBE_PROMPTS = [
    "office outfit using my wardrobe",
    "client meeting outfit using my wardrobe",
    "suggest an outfit with my wardrobe",
    "make a look from my wardrobe",
    "use my wardrobe for office",
]
VISUAL_PROMPTS = [
    "show visual inspiration for client meeting",
    "visual inspiration for office outfit",
    "inspiration board for beach vacation",
]


@pytest.mark.parametrize("p", WARDROBE_PROMPTS)
def test_wardrobe_intent_detected(p):
    assert is_wardrobe_style_request(p) is True


@pytest.mark.parametrize("p", VISUAL_PROMPTS)
def test_visual_inspiration_not_wardrobe(p):
    assert is_wardrobe_style_request(p) is False


def test_plain_office_outfit_not_wardrobe():
    assert is_wardrobe_style_request("office outfit") is False


def test_is_use_wardrobe_action_midsentence():
    from routers.chat import _is_use_wardrobe_action

    assert _is_use_wardrobe_action(prompt="office outfit using my wardrobe") is True
    assert _is_use_wardrobe_action(prompt="client meeting outfit using my wardrobe") is True
    assert _is_use_wardrobe_action(prompt="use my wardrobe for office") is True
    assert _is_use_wardrobe_action(prompt="office outfit") is False


def test_visual_first_yields_to_wardrobe(monkeypatch):
    # Toggle ON: a generic prompt goes visual-first, but wardrobe intent beats
    # default visual inspiration (visual_first must be False).
    monkeypatch.setenv("STYLE_DEFAULT_VISUAL_INSPIRATION", "true")
    from routers.chat import _should_default_visual_inspiration

    assert _should_default_visual_inspiration("office outfit") is True
    assert _should_default_visual_inspiration("office outfit using my wardrobe") is False
    assert _should_default_visual_inspiration("client meeting outfit using my wardrobe") is False
    assert _should_default_visual_inspiration("make a look from my wardrobe") is False
