"""Wedding boards must surface in-pool ethnic assets (kurta/bandhgala),
which previously lost at the Western family gate.

Verified from prod logs: bandhgala assets exist in the loaded pool but were
rejected because the LLM direction hero was 'Gray Blazer'. Fix lets ethnic
assets bypass the family gate for wedding + boosts them in scoring. Scoped
to wedding occasions only.
"""

from __future__ import annotations

from services.style_reasoning_engine import (
    _is_wedding_occasion_policy,
    _is_ethnic_asset,
    _hero_asset_allowed,
    _asset_context_score,
)

_BANDHGALA = {
    "name": "Mens Bandhgalaset",
    "category": "ethnic",
    "subcategory": "bandhgala",
    "image_url": "https://r2/bandhgala.jpg",
}
_BLAZER = {
    "name": "Mens Blackblazer",
    "category": "outerwear",
    "subcategory": "blazer",
    "image_url": "https://r2/blazer.jpg",
}
_WESTERN_DIRECTION = {"hero_piece": "Gray Blazer"}


# ---------- helpers ----------

def test_wedding_occasion_detection():
    assert _is_wedding_occasion_policy("cousin wedding") is True
    assert _is_wedding_occasion_policy("sangeet night") is True
    assert _is_wedding_occasion_policy("client meeting") is False


def test_ethnic_asset_detection():
    assert _is_ethnic_asset(_BANDHGALA) is True
    assert _is_ethnic_asset({"name": "Mens Bluekurta", "subcategory": "kurta"}) is True
    assert _is_ethnic_asset(_BLAZER) is False


# ---------- gate: ethnic passes for wedding, blocked otherwise ----------

def test_ethnic_bypasses_family_gate_for_wedding():
    # Western direction would normally reject a bandhgala (not a blazer).
    assert _hero_asset_allowed(_BANDHGALA, _WESTERN_DIRECTION, "cousin wedding") is True


def test_ethnic_still_blocked_for_office():
    # No bypass off-wedding: a bandhgala must NOT slip into an office board
    # built around a Gray Blazer.
    assert _hero_asset_allowed(_BANDHGALA, _WESTERN_DIRECTION, "client meeting") is False


def test_blazer_still_allowed_for_wedding():
    # The fix is additive — a real blazer still passes its own family gate.
    assert _hero_asset_allowed(_BLAZER, _WESTERN_DIRECTION, "cousin wedding") is True


# ---------- scoring: ethnic boosted for wedding ----------

def test_ethnic_boosted_over_western_for_wedding():
    ethnic = _asset_context_score(
        _BANDHGALA, occasion="cousin wedding", placement="hero", target_text="Gray Blazer"
    )
    western = _asset_context_score(
        _BLAZER, occasion="cousin wedding", placement="hero", target_text="Gray Blazer"
    )
    assert ethnic > western, (ethnic, western)


def test_no_wedding_boost_for_office():
    # Off-wedding the ethnic boost must not apply.
    ethnic_office = _asset_context_score(
        _BANDHGALA, occasion="client meeting", placement="hero", target_text="Gray Blazer"
    )
    ethnic_wedding = _asset_context_score(
        _BANDHGALA, occasion="cousin wedding", placement="hero", target_text="Gray Blazer"
    )
    assert ethnic_wedding > ethnic_office
