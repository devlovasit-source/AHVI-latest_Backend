"""Outfit quality guard consumes Metadata Validator richness signals.

Backstop for client_meeting_score / boardroom_score / capsule_score /
visual_noise / statement_level / pattern_intensity / risk_flags.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from brain.engines.outfit_quality_guard import guard_outfit


def _outfit_with_bottom_meta(meta: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    base = {
        "top": {"name": "White Shirt", "category": "button down"},
        "bottom": {
            "name": "Test Trouser",
            "category": "trousers",
            "style_metadata": meta,
        },
        "footwear": {"name": "Leather Loafers", "category": "loafers"},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Professional context — penalties + hard rejects
# ---------------------------------------------------------------------------

def test_low_client_meeting_score_penalizes_professional_outfit():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "client_meeting_score": 0.30,
    })
    allowed, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="client meeting", query="client meeting",
    )
    assert any("client meeting" in r.lower() for r in reasons)
    assert penalty < 0


def test_high_visual_noise_penalizes_professional_outfit():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "client_meeting_score": 0.65,
        "visual_noise": "high",
    })
    allowed, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="client meeting", query="client meeting",
    )
    assert any("visual noise" in r.lower() for r in reasons)
    # Visual-noise + visual-noise-risk together push the net penalty well
    # past the smart-occasion polish boost.
    assert penalty <= 0


def test_too_casual_risk_flag_hard_rejects_professional_outfit():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "client_meeting_score": 0.65,
        "risk_flags": ["too_casual_for_professional"],
    })
    allowed, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="office today", query="office today",
    )
    assert allowed is False
    assert penalty == -100
    assert any("too casual" in r.lower() for r in reasons)


def test_high_visual_noise_risk_with_low_client_score_extra_penalty():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "client_meeting_score": 0.50,
        "risk_flags": ["high_visual_noise"],
    })
    _, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="client meeting", query="client meeting",
    )
    assert any("high visual noise risk" in r.lower() for r in reasons)


def test_pattern_intensity_loud_penalizes_office():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "pattern_intensity": "loud",
    })
    _, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="office", query="office today",
    )
    assert any("pattern intensity" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Formal context
# ---------------------------------------------------------------------------

def test_risky_statement_level_hard_rejects_formal_outfit():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "statement_level": "risky",
    })
    allowed, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="wedding ceremony", query="wedding",
    )
    assert allowed is False
    assert penalty == -100
    assert any("risky" in r.lower() for r in reasons)


def test_low_boardroom_score_penalizes_formal_context():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "boardroom_score": 0.30,
    })
    _, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="formal wedding", query="formal wedding",
    )
    assert any("formal" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Capsule context
# ---------------------------------------------------------------------------

def test_capsule_low_score_penalizes():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "capsule_score": 0.30,
    })
    _, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="capsule wardrobe", query="build me a capsule wardrobe",
    )
    assert penalty <= -35
    assert any("capsule" in r.lower() for r in reasons)


def test_capsule_high_score_boosts():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "capsule_score": 0.85,
    })
    _, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="capsule wardrobe", query="capsule",
    )
    assert penalty >= 10
    assert any("strong capsule" in r.lower() for r in reasons)


def test_capsule_statement_garment_penalized():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "capsule_score": 0.70,
        "statement_level": "statement",
    })
    _, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="capsule", query="capsule",
    )
    assert penalty <= -30
    assert any("statement" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Date / dinner context
# ---------------------------------------------------------------------------

def test_high_date_night_score_boosts_date_outfit():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "date_night_score": 0.85,
    })
    _, penalty, reasons, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="date night", query="dinner date tonight",
    )
    assert penalty >= 8
    assert any("date-night" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Resilience — missing fields must not crash
# ---------------------------------------------------------------------------

def test_missing_richness_fields_do_not_crash():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
    })
    allowed, _, _, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="office today", query="office today",
    )
    assert isinstance(allowed, bool)


def test_garbage_richness_fields_do_not_crash():
    outfit = _outfit_with_bottom_meta({
        "category": "Bottoms",
        "confidence": 0.9,
        "client_meeting_score": "not a number",
        "risk_flags": "not a list",
        "pattern_intensity": None,
    })
    allowed, _, _, _ = guard_outfit(
        outfit, user_profile={"gender": "male"},
        intent="office today", query="office today",
    )
    assert isinstance(allowed, bool)
