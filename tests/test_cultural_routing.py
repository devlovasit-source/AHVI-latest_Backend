"""Cultural/festive occasion prompts must reach the visual stylist path.

Before: "Haldi" classified to "" -> general chat (bypassed visual
inspiration, assets, gender path). Now it classifies as a style request and,
with STYLE_DEFAULT_VISUAL_INSPIRATION on, routes to visual_inspiration.
"""

from __future__ import annotations

import pytest

from services.stylist_knowledge_service import classify_style_mode, STYLE_MODES
from routers import chat as chat_mod


_CULTURAL = ["haldi", "mehendi", "sangeet", "engagement", "reception", "diwali party"]


# ---------- classify_style_mode recognises cultural occasions ----------

@pytest.mark.parametrize("prompt", _CULTURAL)
def test_cultural_prompt_is_style(prompt):
    mode = classify_style_mode(prompt)
    assert mode in STYLE_MODES and mode != "", f"{prompt!r} -> {mode!r}"


def test_use_wardrobe_still_wardrobe():
    assert classify_style_mode("use my wardrobe for haldi") == "wardrobe_style"


# ---------- _ahvi_style_occasion maps culture ----------

@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("haldi", "wedding"),
        ("mehendi", "wedding"),
        ("sangeet", "wedding"),
        ("cousin wedding", "wedding"),
        ("diwali party", "wedding"),
        ("eid gathering", "wedding"),
        ("christian funeral", "funeral"),
        ("temple visit", "temple_modest"),
        ("client meeting tomorrow", "office"),
        ("coffee date", "date night"),
    ],
)
def test_occasion_mapping(prompt, expected):
    assert chat_mod._ahvi_style_occasion(prompt) == expected


# ---------- visual-first routing flips on for culture ----------

@pytest.mark.parametrize("prompt", ["haldi", "mehendi", "sangeet", "reception",
                                    "temple visit", "diwali party", "christian funeral"])
def test_visual_first_for_culture(monkeypatch, prompt):
    monkeypatch.setenv("STYLE_DEFAULT_VISUAL_INSPIRATION", "true")
    assert chat_mod._should_default_visual_inspiration(
        prompt, intent="", module_context="", style_action=""
    ) is True, prompt


def test_visual_first_off_when_toggle_disabled(monkeypatch):
    monkeypatch.setenv("STYLE_DEFAULT_VISUAL_INSPIRATION", "false")
    assert chat_mod._should_default_visual_inspiration("haldi") is False
