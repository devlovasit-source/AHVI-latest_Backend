"""Phase 2 — Formality / Energy / Movement scoring for the Canonical Style Brief.

Pure tests (no Gemini, no DB). Cover the OCCASION_FAMILY_PROFILE axes, the
concert_social occasion rule, the brief-driven asset scoring (formality distance
+ hard veto), and the board formality authenticity gate. All new scoring lives
behind STYLE_SHARED_BRAIN; flag-off behavior is asserted unchanged.
"""

import pytest

from brain.engines import occasion_style_rules as osr
from services import style_context_service as scs
from services import stylist_knowledge_service as sks
from services import style_reasoning_engine as e


# ---------------------------------------------------------------------------
# 1. Family profile table + numeric axes on the brief.
# ---------------------------------------------------------------------------
def test_concert_family_is_low_formality_high_energy():
    prof = scs.occasion_family_profile("concert_social")
    assert prof["formality"] == 2
    assert prof["energy"] >= 8
    assert prof["movement"] >= 8
    assert "movement_ready" in prof["required_traits"]


def test_professional_family_is_high_formality_low_movement():
    prof = scs.occasion_family_profile("professional")
    assert prof["formality"] == 5
    assert prof["movement"] <= 3


def test_unknown_family_defaults_neutral():
    prof = scs.occasion_family_profile("does_not_exist")
    assert prof["formality"] == 3 and prof["energy"] == 5 and prof["movement"] == 5


def test_brief_carries_numeric_axes():
    ctx = scs.build_canonical_style_context(query="music festival outfit", router_occasion="music festival")
    # music festival -> social_party family -> low formality, high energy/movement.
    assert ctx["formality"] <= 2
    assert ctx["energy"] >= 7
    assert ctx["movement"] >= 7
    assert ctx["required_traits"]


def test_office_brief_is_formal():
    ctx = scs.build_canonical_style_context(query="office meeting", router_occasion="office")
    assert ctx["formality"] == 5


# ---------------------------------------------------------------------------
# 2. concert_social occasion rule: penalize oxford/loafer/belt; prefer street.
# ---------------------------------------------------------------------------
def test_concert_rule_exists_and_aliases():
    for occ in ("music festival", "concert", "gig", "live show", "rave"):
        rule = osr.get_occasion_rule(occ)
        assert rule["_occasion_key"] == "concert_social"


def test_concert_penalizes_oxford_loafer_belt():
    rule = osr.get_occasion_rule("music festival")
    oxford = osr.score_item_for_occasion({"name": "Brown Oxford Dress Shoes"}, rule)
    loafer = osr.score_item_for_occasion({"name": "Formal Leather Loafer"}, rule)
    belt = osr.score_item_for_occasion({"name": "Formal Leather Belt"}, rule)
    sneaker = osr.score_item_for_occasion({"name": "White Graphic Tee Sneakers"}, rule)
    assert oxford < 0 and loafer < 0 and belt < 0
    assert sneaker > oxford


def test_concert_board_rejects_oxford():
    card = {"items": [{"name": "White Oxford Shirt"}, {"name": "Loafers"}, {"name": "Belt"}]}
    reason = osr.reject_board_for_occasion(card, "music festival")
    assert reason  # rejected
    assert "oxford" in reason


def test_concert_board_rejects_ethnic_formal():
    card = {"items": [{"name": "Bandhgala Jacket"}, {"name": "Mojari"}]}
    reason = osr.reject_board_for_occasion(card, "concert")
    assert reason


def test_office_still_allows_oxford_blazer_loafer():
    rule = osr.get_occasion_rule("office")
    assert osr.score_item_for_occasion({"name": "Crisp Oxford Shirt"}, rule) > 0
    assert osr.reject_board_for_occasion(
        {"items": [{"name": "Oxford Shirt"}, {"name": "Tailored Trouser"}, {"name": "Loafers"}]},
        "office",
    ) == ""


def test_wedding_still_allows_ethnic_formal():
    rule = osr.get_occasion_rule("wedding")
    assert osr.score_item_for_occasion({"name": "Silk Kurta"}, rule) > 0
    assert osr.reject_board_for_occasion(
        {"items": [{"name": "Silk Kurta"}, {"name": "Loafers"}]}, "wedding"
    ) == ""


def test_coffee_date_not_boardroom_formal():
    rule = osr.get_occasion_rule("coffee date")
    assert osr.score_item_for_occasion({"name": "Business Blazer"}, rule) < 0


# ---------------------------------------------------------------------------
# 3. Asset axis estimator.
# ---------------------------------------------------------------------------
def test_axis_estimator_formal_vs_casual():
    f_oxford, _, _ = e._asset_axis_estimates("brown oxford dress shoes")
    f_tee, mv_tee, _ = e._asset_axis_estimates("graphic cotton tee")
    assert f_oxford >= 4
    assert f_tee <= 2
    assert mv_tee >= 5


# ---------------------------------------------------------------------------
# 4. Brief-driven _asset_score: formality distance + hard veto.
# ---------------------------------------------------------------------------
def _direction(hero="Graphic Tee"):
    return {"archetype": "Smart Casual Edge", "hero_piece": hero, "items": [hero], "colors": ["black"]}


def _festival_brief():
    return {
        "formality": 2, "energy": 9, "movement": 9,
        "forbidden_archetypes": list(scs._ETHNIC_ARCHETYPES),
        "forbidden_item_signals": list(scs._ETHNIC_ITEM_SIGNALS),
    }


def test_brief_penalizes_formal_asset_for_festival():
    brief = _festival_brief()
    oxford = {"name": "Brown Oxford Dress Shoe", "image_url": "x", "gender": "male"}
    sneaker = {"name": "White Graphic Sneaker", "image_url": "x", "gender": "male"}
    d = _direction()
    s_ox = e._asset_score(oxford, direction=d, occasion="music festival", target_gender="male", brief=brief)
    s_sk = e._asset_score(sneaker, direction=d, occasion="music festival", target_gender="male", brief=brief)
    assert s_sk > s_ox


def test_brief_hard_vetoes_forbidden_item_signal():
    brief = _festival_brief()
    kurta = {"name": "Embroidered Silk Kurta", "image_url": "x", "gender": "male"}
    s = e._asset_score(kurta, direction=_direction(), occasion="music festival", target_gender="male", brief=brief)
    assert s <= 0  # vetoed -> excluded by score>0 gate


def test_brief_hard_vetoes_forbidden_archetype():
    brief = _festival_brief()
    asset = {"name": "Jacket", "image_url": "x", "gender": "male", "archetypes": ["festive heritage"]}
    s = e._asset_score(asset, direction=_direction(), occasion="music festival", target_gender="male", brief=brief)
    assert s <= 0


def test_brief_none_is_legacy_behavior():
    asset = {"name": "Brown Oxford Dress Shoe", "image_url": "x", "gender": "male"}
    d = _direction()
    with_none = e._asset_score(asset, direction=d, occasion="music festival", target_gender="male", brief=None)
    # No brief -> none of the formality/veto terms apply.
    assert isinstance(with_none, int)


# ---------------------------------------------------------------------------
# 5. Board formality authenticity gate inside _apply_style_guard.
# ---------------------------------------------------------------------------
def _guard_ctx(formality=2):
    return {
        "canonical_occasion": "rave",
        "gender": "male",
        "_query": "music festival",
        "formality": formality,
        "forbidden_archetypes": [],
        "forbidden_item_signals": [],
    }


def test_guard_repairs_overformal_festival_board(monkeypatch):
    monkeypatch.setenv("STYLE_SHARED_BRAIN", "true")
    dirs = [{
        "title": "Street",
        "archetype": "Smart Casual Edge",
        "hero_piece": "Graphic Tee",
        "items": ["Graphic Tee", "Cargo Pants", "Oxford Dress Shoes", "Formal Blazer"],
        "palette": ["black"],
    }]
    out = e._apply_style_guard(dirs, _guard_ctx(formality=2))
    blob = " ".join(out[0]["items"]).lower()
    assert "oxford" not in blob
    assert "graphic tee" in blob and "cargo" in blob


def test_guard_keeps_aligned_festival_board(monkeypatch):
    monkeypatch.setenv("STYLE_SHARED_BRAIN", "true")
    dirs = [{
        "title": "Street",
        "archetype": "Smart Casual Edge",
        "hero_piece": "Graphic Tee",
        "items": ["Graphic Tee", "Cargo Pants", "Sneakers"],
        "palette": ["black"],
    }]
    out = e._apply_style_guard(dirs, _guard_ctx(formality=2))
    assert len(out) == 1
    assert "sneakers" in " ".join(out[0]["items"]).lower()


def test_guard_formality_noop_when_flag_off(monkeypatch):
    monkeypatch.setenv("STYLE_SHARED_BRAIN", "false")
    dirs = [{"title": "x", "items": ["Oxford Dress Shoes", "Formal Blazer"], "hero_piece": "Oxford Dress Shoes"}]
    out = e._apply_style_guard(dirs, _guard_ctx(formality=2))
    assert out is dirs


# ---------------------------------------------------------------------------
# 6. Concert returns non-ethnic street archetypes (family fix still holds).
# ---------------------------------------------------------------------------
def test_concert_archetype_pool_non_ethnic():
    pool = set(sks._FAMILY_ARCHETYPE_POOL[sks._resolve_occasion_family("concert")])
    assert not (pool & set(scs._ETHNIC_ARCHETYPES))
