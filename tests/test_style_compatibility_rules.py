"""P0 negative-compatibility knowledge adapter + integration tests.

Covers the LOOP12 focused-test matrix at three layers:
  - brain.engines.style_compatibility_rules.evaluate_outfit (pure adapter)
  - brain.engines.outfit_quality_guard (hard_invalid tagging for the
    fail-closed closest-option fix)
  - routers.stylist._style_this_compat_repair (targeted, anchor-preserving
    repair on the Style This path)
"""

from brain.engines.outfit_quality_guard import filter_and_guard_outfits
from brain.engines.style_compatibility_rules import (
    SEVERITY_HARD,
    SEVERITY_STRONG,
    evaluate_outfit,
)


def _item(role, name, **extra):
    slug = name.replace(" ", "_")
    out = {
        "id": slug,
        "role": role,
        "name": name,
        # A genuine board-safe image candidate - required by
        # ConstrainedOutfitBuilder's is_board_renderable() gate for the
        # repair-path tests below.
        "masked_url": f"https://example.test/{slug}.png",
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# 1-2: clean pairings never flag
# ---------------------------------------------------------------------------


def test_workout_joggers_and_running_sneakers_pass():
    v = evaluate_outfit(
        [_item("bottom", "sweatpants"), _item("footwear", "running shoes")],
        occasion="workout",
    )
    assert v == []


def test_office_shirt_trousers_loafers_pass():
    v = evaluate_outfit(
        [
            _item("top", "dress shirt"),
            _item("bottom", "formal trousers"),
            _item("footwear", "loafers"),
        ],
        occasion="office",
    )
    assert v == []


# ---------------------------------------------------------------------------
# 3, 5: repairable violations surfaced (footwear / garment)
# ---------------------------------------------------------------------------


def test_office_formal_trousers_running_shoes_flagged_repairable_footwear():
    v = evaluate_outfit(
        [
            _item("bottom", "formal business trousers"),
            _item("footwear", "chunky running shoes"),
        ],
        occasion="office",
    )
    assert len(v) == 1
    assert v[0].repairable is True
    assert "footwear" in v[0].offending_roles


def test_sweatpants_formal_blazer_professional_context_flagged():
    v = evaluate_outfit(
        [_item("top", "sweatpants"), _item("outerwear", "formal blazer")],
        occasion="business_casual",
    )
    assert v and v[0].severity in (SEVERITY_STRONG, SEVERITY_HARD)


# ---------------------------------------------------------------------------
# 6, 18: severity 5 on a strict explicit occasion promotes to HARD
# ---------------------------------------------------------------------------


def test_suit_running_shoes_strict_business_formal_is_hard():
    v = evaluate_outfit(
        [_item("outerwear", "formal suit"), _item("footwear", "running shoes")],
        occasion="business_formal",
    )
    assert v and v[0].severity == SEVERITY_HARD


def test_severity_five_on_ordinary_occasion_is_not_auto_hard():
    # "office" is AHVI's coarse everyday bucket, not a strict dress-code
    # token - severity 5 evidence here must stay STRONG, never silently
    # promote to HARD (P0 spec: "Do NOT treat severity=5 as automatically
    # HARD").
    v = evaluate_outfit(
        [
            _item("bottom", "formal business trousers"),
            _item("footwear", "chunky running shoes"),
        ],
        occasion="office",
    )
    assert v and v[0].severity == SEVERITY_STRONG


# ---------------------------------------------------------------------------
# 7: explicit bold/streetwear intent is not overblocked
# ---------------------------------------------------------------------------


def test_bold_streetwear_intent_drops_a_reducible_violation():
    base = evaluate_outfit(
        [_item("outerwear", "formal blazer"), _item("top", "hoodie")],
        occasion="business_casual",
    )
    assert base  # sanity: the un-excepted case does flag something

    bold = evaluate_outfit(
        [_item("outerwear", "formal blazer"), _item("top", "hoodie")],
        occasion="business_casual",
        query="give me a bold streetwear statement look",
    )
    assert len(bold) < len(base)


# ---------------------------------------------------------------------------
# 8: duplicate evidence across families does not stack as separate penalties
# ---------------------------------------------------------------------------


def test_duplicate_pair_evidence_collapses_to_one_violation():
    v = evaluate_outfit(
        [_item("top", "hoodie"), _item("outerwear", "formal blazer")],
        occasion="business_formal",
    )
    pair_key = frozenset({"hoodie", "formal_blazer"})
    matches = [x for x in v if frozenset(x.offending_item_ids) == pair_key]
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# 9, 10: exception precedence
# ---------------------------------------------------------------------------


def test_exception_reduces_a_strong_violation():
    base = evaluate_outfit(
        [_item("outerwear", "formal blazer"), _item("top", "hoodie")],
        occasion="business_casual",
    )
    assert base and base[0].severity == SEVERITY_STRONG

    reduced = evaluate_outfit(
        [_item("outerwear", "formal blazer"), _item("top", "hoodie")],
        occasion="business_casual",
        query="streetwear statement look",
    )
    assert not any(r.rule_id == base[0].rule_id and r.severity == SEVERITY_STRONG for r in reduced)


def test_exception_cannot_override_explicit_hard_dress_code():
    v = evaluate_outfit(
        [_item("outerwear", "tuxedo"), _item("footwear", "sneakers")],
        occasion="daily",
        query="black tie event but give me a bold streetwear mix",
    )
    assert v and all(x.severity == SEVERITY_HARD for x in v)


# ---------------------------------------------------------------------------
# 11: unknown taxonomy never hard-rejects
# ---------------------------------------------------------------------------


def test_unknown_occasion_is_neutral_not_a_hard_reject():
    v = evaluate_outfit(
        [_item("top", "hoodie"), _item("outerwear", "formal blazer")],
        occasion="some_made_up_occasion_xyz",
    )
    assert v == []


# ---------------------------------------------------------------------------
# 20: a broken/missing rule file never crashes evaluation
# ---------------------------------------------------------------------------


def test_loader_failure_fails_safe(monkeypatch):
    import brain.engines.style_compatibility_rules as scr

    monkeypatch.setattr(scr, "_cache", None)
    monkeypatch.setattr(scr, "_load_family", lambda filename: None)
    try:
        v = evaluate_outfit(
            [_item("top", "hoodie"), _item("outerwear", "formal blazer")],
            occasion="business_formal",
        )
        assert v == []
    finally:
        monkeypatch.setattr(scr, "_cache", None)  # force a clean reload for later tests


# ---------------------------------------------------------------------------
# LOOP10: fail-closed hard_invalid tagging (closest-option safety)
# ---------------------------------------------------------------------------


def test_filter_and_guard_tags_hard_reject_as_hard_invalid():
    outfit = {
        "top": {"name": "Green Saree", "category": "saree", "color": "green"},
        "bottom": {"name": "Jeans", "category": "jeans", "color": "blue"},
        "footwear": {"name": "White Sneakers", "category": "sneakers", "color": "white"},
        "score": 80,
    }
    filter_and_guard_outfits([outfit], user_profile={"gender": "male"}, intent="daily", query="")
    assert outfit["_quality_guard_meta"]["hard_invalid"] is True


def test_filter_and_guard_tags_soft_pass_as_not_hard_invalid():
    outfit = {
        "top": {"name": "Emerald Shirt", "category": "shirt", "color": "emerald"},
        "bottom": {"name": "Beige Tailored Trousers", "category": "trousers", "color": "beige"},
        "footwear": {"name": "Black Loafers", "category": "loafers", "color": "black"},
        "score": 80,
    }
    filter_and_guard_outfits([outfit], user_profile={}, intent="office", query="")
    assert outfit["_quality_guard_meta"]["hard_invalid"] is False


# ---------------------------------------------------------------------------
# 12-17: Style This targeted repair (anchor preserved, wardrobe-only, drop
# when unrepairable). Enforcement is exercised directly with shadow mode
# forced off, mirroring how the candidate will be exercised once flipped on.
# ---------------------------------------------------------------------------


def _wardrobe_pool():
    return [
        _item("top", "dress shirt", source="wardrobe"),
        _item("bottom", "formal business trousers", source="wardrobe"),
        _item("footwear", "chunky running shoes", source="wardrobe"),
        _item("footwear", "loafers", source="wardrobe"),
        _item("outerwear", "formal blazer", source="wardrobe"),
    ]


def test_style_this_repair_fixes_footwear_only_and_keeps_anchor(monkeypatch):
    monkeypatch.setenv("ENABLE_NEGATIVE_COMPATIBILITY_P0", "true")
    monkeypatch.setenv("NEGATIVE_COMPATIBILITY_SHADOW_MODE", "false")
    from routers.stylist import _style_this_compat_repair

    anchor = _item("bottom", "formal business trousers", source="wardrobe")
    direction = {
        "items": [
            anchor,
            _item("top", "dress shirt", source="wardrobe"),
            _item("footwear", "chunky running shoes", source="wardrobe"),
        ]
    }
    result = _style_this_compat_repair(
        direction, anchor=anchor, wardrobe=_wardrobe_pool(), occasion="office",
    )
    assert result is not None
    ids = {i["id"] for i in result["items"]}
    assert "formal_business_trousers" in ids  # anchor preserved
    assert "chunky_running_shoes" not in ids  # offending footwear repaired away
    assert "loafers" in ids


def test_style_this_repair_never_touches_anchor_role(monkeypatch):
    monkeypatch.setenv("ENABLE_NEGATIVE_COMPATIBILITY_P0", "true")
    monkeypatch.setenv("NEGATIVE_COMPATIBILITY_SHADOW_MODE", "false")
    from routers.stylist import _style_this_compat_repair

    anchor = _item("outerwear", "tuxedo", source="wardrobe")
    direction = {
        "items": [
            anchor,
            _item("footwear", "sneakers", source="wardrobe"),
            _item("bottom", "formal trousers", source="wardrobe"),
        ]
    }
    result = _style_this_compat_repair(
        direction, anchor=anchor, wardrobe=[
            *_wardrobe_pool(), _item("footwear", "formal dress shoes", source="wardrobe"),
        ], occasion="daily",
    )
    if result is not None:
        ids = {i["id"] for i in result["items"]}
        assert "tuxedo" in ids


def test_style_this_shadow_mode_never_mutates_direction(monkeypatch):
    monkeypatch.setenv("ENABLE_NEGATIVE_COMPATIBILITY_P0", "true")
    monkeypatch.setenv("NEGATIVE_COMPATIBILITY_SHADOW_MODE", "true")
    from routers.stylist import _style_this_compat_repair

    anchor = _item("bottom", "formal business trousers", source="wardrobe")
    original_items = [
        anchor,
        _item("top", "dress shirt", source="wardrobe"),
        _item("footwear", "chunky running shoes", source="wardrobe"),
    ]
    direction = {"items": list(original_items)}
    result = _style_this_compat_repair(
        direction, anchor=anchor, wardrobe=_wardrobe_pool(), occasion="office",
    )
    assert result is direction
    assert result["items"] == original_items


def test_style_this_drops_direction_when_unrepairable(monkeypatch):
    monkeypatch.setenv("ENABLE_NEGATIVE_COMPATIBILITY_P0", "true")
    monkeypatch.setenv("NEGATIVE_COMPATIBILITY_SHADOW_MODE", "false")
    from routers.stylist import _style_this_compat_repair

    anchor = _item("outerwear", "formal suit", source="wardrobe")
    direction = {
        "items": [anchor, _item("footwear", "running shoes", source="wardrobe")]
    }
    # No alternative footwear anywhere in the wardrobe pool -> repair must fail.
    result = _style_this_compat_repair(
        direction, anchor=anchor, wardrobe=[anchor, _item("footwear", "running shoes", source="wardrobe")],
        occasion="business_formal",
    )
    assert result is None
