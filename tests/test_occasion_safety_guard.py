"""Deterministic occasion-safety guard for formal/professional boards.

Demo-stabilization: prohibited casual/private/gym/beach items must NEVER
appear on wedding, office, client-meeting, or board-meeting boards — without
relying on an LLM or agent_orchestration payload.
"""

from __future__ import annotations

import pytest

from brain.engines.outfit_quality_guard import (
    normalize_occasion,
    reject_board_for_occasion,
)


def _board(*item_names: str) -> dict:
    return {
        "title": "Test Look",
        "items": [{"name": n, "category": n} for n in item_names],
    }


# ---------- normalize_occasion: wedding family ----------

@pytest.mark.parametrize(
    "raw",
    ["cousin wedding", "best friend wedding", "engagement", "sangeet",
     "haldi", "reception", "shaadi", "mehendi"],
)
def test_wedding_family_normalizes_to_wedding(raw):
    assert normalize_occasion(raw) == "wedding"


@pytest.mark.parametrize(
    "raw", ["client meeting", "board meeting", "office"],
)
def test_meeting_normalizes_to_office(raw):
    assert normalize_occasion(raw) == "office"


# ---------- wedding: prohibited items rejected ----------

@pytest.mark.parametrize(
    "bad", ["boxers", "briefs", "track pants", "gym shorts", "sleepwear",
            "beachwear", "crocs", "flip flops", "sliders"],
)
def test_wedding_rejects_prohibited(bad):
    rejected, reason = reject_board_for_occasion(_board(bad, "kurta"), "cousin wedding")
    assert rejected is True, f"{bad} should be blocked for wedding"
    assert reason.startswith("wedding_forbidden_") or reason == "private_wear_forbidden"


def test_wedding_allows_proper_attire():
    rejected, _ = reject_board_for_occasion(
        _board("Sherwani", "Mojari", "Silk Kurta"), "cousin wedding"
    )
    assert rejected is False


# ---------- office / client meeting / board meeting ----------

@pytest.mark.parametrize(
    "occasion", ["client meeting tomorrow", "board meeting", "office"],
)
@pytest.mark.parametrize(
    "bad", ["boxers", "track pants", "gym shorts", "crocs", "sweatpants", "jersey"],
)
def test_office_rejects_prohibited(occasion, bad):
    rejected, reason = reject_board_for_occasion(_board(bad, "shirt"), occasion)
    assert rejected is True, f"{bad} should be blocked for {occasion}"


@pytest.mark.parametrize("occasion", ["client meeting tomorrow", "board meeting", "office"])
def test_office_rejects_casual_headwear(occasion):
    rejected, reason = reject_board_for_occasion(
        _board("Mens Whitebeanie", "Crisp Oxford Shirt"), occasion
    )
    assert rejected is True, f"beanie should be blocked for {occasion}"
    assert "beanie" in reason


def test_wedding_rejects_casual_headwear():
    rejected, reason = reject_board_for_occasion(
        _board("Snapback Cap", "Silk Kurta"), "cousin wedding"
    )
    assert rejected is True
    assert reason.startswith("wedding_forbidden_")


def test_office_allows_proper_attire():
    rejected, _ = reject_board_for_occasion(
        _board("White Oxford Shirt", "Charcoal Trousers", "Leather Loafers"),
        "client meeting tomorrow",
    )
    assert rejected is False


# ---------- coffee date stays permissive (not a formal gate) ----------

def test_coffee_date_allows_casual_sneakers():
    rejected, _ = reject_board_for_occasion(
        _board("White Tee", "Dark Jeans", "Minimal Sneakers"), "coffee date"
    )
    assert rejected is False
