"""Tone, response validator, and response assembler stabilization tests.

These tests exercise the existing engines after the tone/response
stabilization pass — no new tone engine is introduced.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from brain.tone.tone_engine import tone_engine
from brain.tone.archetype_engine import archetype_engine
from brain.response_validator import (
    looks_truncated,
    polish_final_text,
    truncate_preserving_sentence,
    validate_final_text,
)
from brain.response.response_assembler import response_assembler


# ---------------------------------------------------------------------------
# Phase 1: tone caps — styling never emits emoji or Gen-Z slang by default
# ---------------------------------------------------------------------------

def _styling_signals(slang_presence: str = "none", emoji_density: str = "none") -> Dict[str, Any]:
    return {
        "context_mode": "styling",
        "emotion_state": "neutral",
        "user_message_style": {
            "message_length_bucket": "short",
            "slang_presence": slang_presence,
            "emoji_density": emoji_density,
        },
    }


def test_styling_strips_forbidden_starter_and_emoji_via_tone_pipeline():
    out = tone_engine.apply(
        "Sure! Here are some ideas 😊 for tonight.",
        user_profile={"age": 22},
        signals=_styling_signals(),
    )
    assert "Sure!" not in out
    assert "Here are some ideas" not in out
    assert "😊" not in out


def test_styling_does_not_inject_genz_slang_when_user_uses_none():
    out = tone_engine.apply(
        "This is a strong look.",
        user_profile={"age": 22},
        signals=_styling_signals(slang_presence="none"),
    )
    # Tone pipeline must not add disallowed slang tokens.
    for token in ("cringe", "ratio", "touch grass", "cope", "skill issue"):
        assert token not in out.lower()


def test_styling_replaces_influencer_street_energy_line():
    # `_apply_outfit_tone` adds the street line for bold/street/gen_z.
    # Even when it fires, it should be the editorial replacement, not the old hype.
    out = tone_engine.apply(
        "This is a strong look.",
        user_profile={"age": 22},
        signals={
            "context_mode": "styling",
            "emotion_state": "neutral",
            "user_message_style": {"slang_presence": "high"},
        },
        context={"aesthetic": {"energy": "bold", "vibe": "street", "structure": "relaxed"}},
    )
    assert "This lands with confident street energy" not in out


# ---------------------------------------------------------------------------
# Phase 2: archetype routing
# ---------------------------------------------------------------------------

def test_chat_domain_routes_to_advisor_blend_not_best_friend_alone():
    blend = archetype_engine.select({"domain": "chat"})
    # Top entry should be advisor, not best_friend.
    assert blend[0]["type"] == "advisor"
    assert any(row["type"] == "stylist" for row in blend)


def test_styling_domain_routes_to_stylist_advisor_blend():
    blend = archetype_engine.select({"domain": "styling"})
    types = [row["type"] for row in blend]
    assert types[0] == "stylist"
    assert "advisor" in types


def test_lifestyle_domain_blend_minor_bestfriend_weight():
    blend = archetype_engine.select({"domain": "lifestyle"})
    by = {row["type"]: row["weight"] for row in blend}
    assert by.get("advisor", 0) >= 0.5
    assert by.get("best_friend", 0) <= 0.2


def test_learned_blend_falls_back_when_archetypes_unknown():
    # User memory references archetypes that are not in the config.
    blend = archetype_engine.select(
        {"domain": "styling"},
        user_memory={"archetype_scores": {"meme_lord": 5, "fairy_godmother": 3}},
    )
    # Should fall through to domain fallback, not return [].
    assert blend, "blend must not be empty when learned archetypes are unknown"
    assert blend[0]["type"] in {"stylist", "advisor"}


# ---------------------------------------------------------------------------
# Phase 4: validator — truncation detection + sentence-safe truncation
# ---------------------------------------------------------------------------

def test_looks_truncated_detects_hanging_connector():
    assert looks_truncated("This works because") is True
    assert looks_truncated("Trying this with") is True
    assert looks_truncated("Use it for") is True


def test_looks_truncated_detects_missing_terminal_punct():
    assert looks_truncated("This is a clean direction") is True


def test_looks_truncated_accepts_complete_sentence():
    assert looks_truncated("This holds together well.") is False


def test_looks_truncated_detects_unclosed_code_fence():
    assert looks_truncated("Here is code:\n```python\nprint(1)") is True


def test_looks_truncated_detects_bad_json():
    assert looks_truncated('{"a": 1, "b":') is True
    assert looks_truncated('{"a": 1}') is False


def test_truncate_preserving_sentence_keeps_full_sentence():
    text = "One. Two. Three. Four. Five." * 100
    out = truncate_preserving_sentence(text, max_chars=50)
    # Must end with terminal punctuation, not mid-word.
    assert out.endswith(".")
    assert "Fiv" not in out.split(".")[-1].strip()


def test_validate_final_text_marks_truncated():
    result = validate_final_text("This works because")
    assert result["looks_truncated"] is True
    assert "looks_truncated" in result["issues"]


def test_polish_final_text_strips_starter_and_returns_complete_sentence():
    out = polish_final_text("Sure! This is a clean direction. I’d keep it composed.")
    assert not out.lower().startswith("sure")
    assert out.endswith(".")


def test_polish_final_text_falls_back_when_unsalvageable():
    fallback = "Tell me the occasion."
    out = polish_final_text("This works because", fallback=fallback)
    assert out == fallback


# ---------------------------------------------------------------------------
# Phase 5: response assembler compact + closer
# ---------------------------------------------------------------------------

def test_compact_response_preserves_closer_and_complete_sentences():
    text = (
        "This holds together well. The proportions feel considered. "
        "The footwear keeps it composed. The palette stays clean. "
        "I can make it sharper or soften it."
    )
    out = response_assembler._compact_response(text, max_sentences=4, preserve_closer=True)
    # Must keep the final closer.
    assert out.endswith("sharper or soften it.")
    # Must not contain mid-sentence cut.
    assert "." in out
    # 4 sentences max — so we expect ~4 terminal punctuation marks.
    assert out.count(".") <= 5


def test_compact_response_drops_hanging_tail():
    text = "This holds together well. The proportions feel considered with"
    out = response_assembler._compact_response(text, max_sentences=4)
    assert "considered with" not in out
    assert out.endswith("well.")


def test_closer_is_context_aware_styling():
    closer = response_assembler._closer({"domain": "styling"})
    assert "sharper" in closer or "soften" in closer


def test_closer_is_context_aware_professional():
    closer = response_assembler._closer(
        {"signals": {"context_mode": "professional", "emotion_state": "neutral"}}
    )
    assert "polished" in closer


def test_closer_is_context_aware_vulnerable():
    closer = response_assembler._closer(
        {"signals": {"emotion_state": "vulnerable"}}
    )
    assert "simpl" in closer.lower()


# ---------------------------------------------------------------------------
# Phase 8: router lightweight_chat passes through tone + polish
# ---------------------------------------------------------------------------

def test_router_lightweight_chat_no_forbidden_starters():
    from routers.chat import lightweight_chat

    for prompt in ("", "hi", "tell me a joke", "what works for tonight?"):
        out = lightweight_chat(prompt)
        assert isinstance(out, str) and out
        lower = out.lower()
        assert not lower.startswith("sure!")
        assert not lower.startswith("absolutely!")
        assert "here are some ideas" not in lower


# ---------------------------------------------------------------------------
# Phase 10 tone scenarios — premium voice for client meeting / vulnerable
# ---------------------------------------------------------------------------

def test_client_meeting_response_stays_calm_and_clean():
    out = tone_engine.apply(
        "Shorts and slides won’t land for a client meeting. Keep it polished — a clean shirt, tailored trousers, and minimal sneakers or loafers.",
        user_profile={"age": 28},
        signals={
            "context_mode": "professional",
            "emotion_state": "neutral",
            "user_message_style": {"slang_presence": "none", "emoji_density": "none"},
        },
    )
    assert "😊" not in out and "🔥" not in out
    for slang in ("lowkey", "highkey", "it's giving", "this eats", "you ate that"):
        assert slang not in out.lower()


def test_vulnerable_emotion_strips_slang_and_humor():
    out = tone_engine.apply(
        "I don’t feel confident in this. lowkey help me.",
        user_profile={"age": 22},
        signals={
            "context_mode": "styling",
            "emotion_state": "vulnerable",
        },
    )
    # Vulnerable suppresses slang/humor.
    assert "lowkey" not in out.lower()
    assert "!" not in out  # soft sentence_style replaces ! with .
