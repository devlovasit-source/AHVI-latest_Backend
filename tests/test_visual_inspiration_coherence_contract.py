from pathlib import Path

from services import style_reasoning_engine as engine


def _asset(asset_id, name, category, *, tags=None, board=True):
    item = {
        "asset_id": asset_id,
        "name": name,
        "category": category,
        "subcategory": category,
        "image_url": f"https://cdn.example/{asset_id}.jpg",
        "gender": "unisex",
        "status": "active",
        "tags": tags or [],
    }
    if board:
        item["board_image_url"] = f"https://cdn.example/{asset_id}.png"
        item["cutout_status"] = "ready"
    return item


FORMAL = {
    "archetype": "Modern Professional",
    "title": "A future-facing display title",
    "hero_piece": "Crisp Oxford Shirt",
    "items": ["Crisp Oxford Shirt"],
}
UTILITY = {
    "archetype": "Modern Utility",
    "hero_piece": "Field Jacket",
    "items": ["Field Jacket"],
}
SPORTY = {
    "archetype": "Weekend Explorer",
    "hero_piece": "Relaxed Overshirt",
    "items": ["Relaxed Overshirt"],
}


def test_formal_metadata_rejects_camouflage_and_cargo_bottoms():
    assert not engine._visual_asset_compatibility(
        _asset("camo", "Camouflage Trousers", "bottom", tags=["camo"]),
        direction=FORMAL,
        occasion="today",
    )[0]
    assert not engine._visual_asset_compatibility(
        _asset("cargo", "Cargo Pants", "bottom"),
        direction=FORMAL,
        occasion="today",
    )[0]


def test_formal_metadata_rejects_running_footwear():
    allowed, reason = engine._visual_asset_compatibility(
        _asset("run", "Bright Running Shoes", "footwear"),
        direction=FORMAL,
        occasion="today",
    )
    assert not allowed
    assert "athletic" in reason


def test_clean_minimal_leather_sneakers_remain_allowed():
    assert engine._visual_asset_compatibility(
        _asset("leather", "Clean White Leather Sneakers", "footwear"),
        direction=FORMAL,
        occasion="today",
    )[0]


def test_utility_metadata_permits_cargo_and_sporty_metadata_permits_trainers():
    assert engine._visual_asset_compatibility(
        _asset("utility-cargo", "Cargo Trousers", "bottom"),
        direction=UTILITY,
        occasion="today",
    )[0]
    assert engine._visual_asset_compatibility(
        _asset("trainer", "Trail Trainers", "footwear"),
        direction=SPORTY,
        occasion="outdoor",
    )[0]


def test_archetype_avoid_items_are_strong_negative_signals():
    direction = {
        "archetype": "Future Formal",
        "formality": 8,
        "impression": ["polished"],
        "avoid_items": ["camouflage"],
    }
    allowed, reason = engine._visual_asset_compatibility(
        _asset("avoid", "Camouflage Trouser", "bottom"),
        direction=direction,
        occasion="today",
    )
    assert not allowed
    assert reason == "archetype_avoid_item"


def test_preferred_items_influence_ranking():
    assets = [
        _asset("plain", "Plain Trouser", "bottom"),
        _asset("cargo", "Cargo Trousers", "bottom"),
    ]
    ranked = engine._best_style_assets(
        assets,
        direction=UTILITY | {"hero_piece": "cargo trousers"},
        occasion="today",
        limit=2,
        apply_visual_inspiration_policy=True,
    )
    assert ranked[0]["asset_id"] == "cargo"


def test_incompatible_complete_items_are_removed_and_repaired():
    top = _asset("top", "Crisp Oxford Shirt", "top")
    good_bottom = _asset("bottom", "Tailored Trousers", "bottom")
    good_shoe = _asset("shoe", "Leather Loafers", "footwear")
    bad_bottom = _asset("bad-bottom", "Cargo Pants", "bottom")
    bad_shoe = _asset("bad-shoe", "Bright Running Shoes", "footwear")
    repaired = engine._repair_visual_board_core(
        [top, bad_bottom, bad_shoe],
        assets=[top, good_bottom, good_shoe, bad_bottom, bad_shoe],
        direction=FORMAL,
        occasion="today",
        target_gender="unknown",
        allow_feminine_accessory=False,
        brief={},
        apply_visual_inspiration_policy=True,
    )
    names = {item["name"] for item in repaired}
    roles = {engine._visual_board_item_role(item) for item in repaired}
    assert "Cargo Pants" not in names
    assert "Bright Running Shoes" not in names
    assert {"top", "bottom", "footwear"}.issubset(roles)


def test_bottom_backfill_selects_a_compatible_bottom(monkeypatch):
    assets = [
        _asset("top", "Crisp Oxford Shirt", "top"),
        _asset("cargo", "Cargo Pants", "bottom"),
        _asset("bottom", "Tailored Trousers", "bottom"),
        _asset("shoe", "Leather Loafers", "footwear"),
    ]
    monkeypatch.setattr(engine, "_style_asset_rows", lambda: assets)
    result = engine._enrich_visual_directions_with_assets(
        [{**FORMAL, "image_url": assets[0]["image_url"], "asset_id": "top"}],
        occasion="today",
        apply_visual_inspiration_policy=True,
    )
    names = {item["name"] for item in result[0]["board_items"]}
    assert "Tailored Trousers" in names
    assert "Cargo Pants" not in names


def test_core_repair_selects_compatible_footwear():
    top = _asset("top", "Crisp Oxford Shirt", "top")
    bottom = _asset("bottom", "Tailored Trousers", "bottom")
    shoe = _asset("shoe", "Leather Loafers", "footwear")
    running = _asset("running", "Bright Running Shoes", "footwear")
    repaired = engine._repair_visual_board_core(
        [top, bottom],
        assets=[top, bottom, shoe, running],
        direction=FORMAL,
        occasion="today",
        target_gender="unknown",
        allow_feminine_accessory=False,
        brief={},
        apply_visual_inspiration_policy=True,
    )
    footwear = [item["name"] for item in repaired if engine._visual_board_item_role(item) == "footwear"]
    assert footwear == ["Leather Loafers"]


def _diversity_assets():
    return [
        _asset("top", "Crisp Oxford Shirt", "top"),
        _asset("bottom-1", "Tailored Navy Trousers", "bottom"),
        _asset("bottom-2", "Tailored Grey Trousers", "bottom"),
        _asset("shoe-1", "Leather Loafers", "footwear"),
        _asset("shoe-2", "Clean White Leather Sneakers", "footwear"),
    ]


def test_different_directions_prefer_different_bottoms_and_footwear(monkeypatch):
    assets = _diversity_assets()
    monkeypatch.setattr(engine, "_style_asset_rows", lambda: assets)
    result = engine._enrich_visual_directions_with_assets(
        [
            {**FORMAL, "image_url": assets[0]["image_url"], "asset_id": "top"},
            {**FORMAL, "title": "Another direction", "image_url": assets[0]["image_url"], "asset_id": "top"},
        ],
        occasion="today",
        apply_visual_inspiration_policy=True,
    )
    assert len(result) == 2
    bottoms = [
        next(item["asset_id"] for item in direction["board_items"] if item["role"] == "bottom")
        for direction in result
    ]
    footwear = [
        next(item["asset_id"] for item in direction["board_items"] if item["role"] == "footwear")
        for direction in result
    ]
    assert bottoms[0] != bottoms[1]
    assert footwear[0] != footwear[1]


def test_reuse_is_permitted_when_inventory_has_no_alternative(monkeypatch):
    assets = [
        _asset("top", "Crisp Oxford Shirt", "top"),
        _asset("bottom", "Tailored Trousers", "bottom"),
        _asset("shoe", "Leather Loafers", "footwear"),
    ]
    monkeypatch.setattr(engine, "_style_asset_rows", lambda: assets)
    result = engine._enrich_visual_directions_with_assets(
        [{**FORMAL, "image_url": assets[0]["image_url"], "asset_id": "top"}] * 2,
        occasion="today",
        apply_visual_inspiration_policy=True,
    )
    assert len(result) == 2
    for direction in result:
        assert {"top", "bottom", "footwear"}.issubset(
            {item["role"] for item in direction["board_items"]}
        )


def test_every_separates_board_keeps_core_roles_and_one_accessory():
    items = [
        _asset("top", "Crisp Oxford Shirt", "top"),
        _asset("bottom", "Tailored Trousers", "bottom"),
        _asset("shoe", "Leather Loafers", "footwear"),
        _asset("watch", "Steel Watch", "accessory"),
        _asset("bag", "Leather Bag", "accessory"),
    ]
    repaired = engine._repair_visual_board_core(
        items,
        assets=items,
        direction=FORMAL,
        occasion="today",
        target_gender="unknown",
        allow_feminine_accessory=False,
        brief={},
        apply_visual_inspiration_policy=True,
    )
    roles = [engine._visual_board_item_role(item) for item in repaired]
    assert {"top", "bottom", "footwear"}.issubset(roles)
    assert roles.count("accessory") <= 1


def test_belts_rank_below_meaningful_alternatives_unless_preferred():
    belt = _asset("belt", "Leather Belt", "accessory")
    watch = _asset("watch", "Steel Watch", "accessory")
    ranked = engine._visual_rank_accessories(
        [belt, watch],
        FORMAL,
        apply_visual_inspiration_policy=True,
    )
    assert ranked[0]["asset_id"] == "watch"
    supported = {**FORMAL, "preferred_items": ["belt"]}
    ranked_supported = engine._visual_rank_accessories(
        [belt, watch],
        supported,
        apply_visual_inspiration_policy=True,
    )
    assert ranked_supported[0]["asset_id"] == "belt"


def test_policy_false_preserves_legacy_fallback_accessories(monkeypatch):
    monkeypatch.setattr(engine, "_style_asset_rows", lambda: [])
    result = engine._enrich_visual_directions_with_assets(
        [{
            "archetype": "Quiet Luxury",
            "title": "Soft Polish",
            "hero_piece": "Camel Blazer",
            "items": ["Camel Blazer", "White Shirt", "Stone Trousers"],
        }],
        occasion="startup event",
        target_gender="male",
        apply_visual_inspiration_policy=False,
    )
    names = [item["name"] for item in result[0]["complete_the_look"]]
    assert len(names) == 3
    assert "Leather-Strap Watch" in names


def test_wardrobe_intent_bypasses_visual_compatibility(monkeypatch):
    top = _asset("top", "Crisp Oxford Shirt", "top")
    cargo = _asset("cargo", "Cargo Pants", "bottom")
    footwear = _asset("shoe", "Minimal Sneakers", "footwear")
    monkeypatch.setattr(engine, "_style_asset_rows", lambda: [top, cargo, footwear])
    result = engine._enrich_visual_directions_with_assets(
        [{
            **FORMAL,
            "image_url": top["image_url"],
            "asset_id": "top",
            "owned_items": [top, cargo, footwear],
            "complete_the_look": [cargo, footwear],
        }],
        occasion="today",
        wardrobe_intent=True,
        apply_visual_inspiration_policy=False,
    )
    assert "Cargo Pants" in {item["name"] for item in result[0]["complete_the_look"]}


def test_shared_selection_and_scoring_skip_visual_policy(monkeypatch):
    asset = _asset("cargo", "Cargo Trousers", "bottom")
    monkeypatch.setattr(
        engine,
        "_visual_asset_compatibility",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("visual policy must be disabled")
        ),
    )
    engine._best_style_assets(
        [asset],
        direction={"hero_piece": "cargo trousers"},
        occasion="today",
        apply_visual_inspiration_policy=False,
    )
    false_with_preference = engine._asset_score(
        asset,
        direction=UTILITY | {"hero_piece": "cargo trousers"},
        occasion="today",
        apply_visual_inspiration_policy=False,
    )
    false_without_preference = engine._asset_score(
        asset,
        direction={"hero_piece": "cargo trousers"},
        occasion="today",
        apply_visual_inspiration_policy=False,
    )
    true_with_preference = engine._asset_score(
        asset,
        direction=UTILITY | {"hero_piece": "cargo trousers"},
        occasion="today",
        apply_visual_inspiration_policy=True,
    )
    assert false_with_preference == false_without_preference
    assert true_with_preference > false_with_preference


def test_policy_logs_and_diversity_are_visual_only(monkeypatch, caplog):
    assets = _diversity_assets()
    monkeypatch.setattr(engine, "_style_asset_rows", lambda: assets)
    caplog.set_level("INFO", logger="ahvi.style_reasoning")
    engine._enrich_visual_directions_with_assets(
        [{**FORMAL, "image_url": assets[0]["image_url"], "asset_id": "top"}],
        occasion="today",
        apply_visual_inspiration_policy=False,
    )
    assert not any("AHVI_VISUAL_DIVERSITY_SELECTION" in record.message for record in caplog.records)
    caplog.clear()
    engine._enrich_visual_directions_with_assets(
        [{**FORMAL, "image_url": assets[0]["image_url"], "asset_id": "top"}],
        occasion="today",
        apply_visual_inspiration_policy=True,
    )
    assert any("AHVI_VISUAL_DIVERSITY_SELECTION" in record.message for record in caplog.records)
    cargo = _asset("log-cargo", "Cargo Trousers", "bottom")
    caplog.clear()
    engine._best_style_assets(
        [cargo],
        direction={**FORMAL, "hero_piece": "cargo trousers"},
        occasion="today",
        apply_visual_inspiration_policy=False,
    )
    assert not any("AHVI_VISUAL_COMPATIBILITY_REJECTED" in record.message for record in caplog.records)
    caplog.clear()
    engine._best_style_assets(
        [cargo],
        direction={**FORMAL, "hero_piece": "cargo trousers"},
        occasion="today",
        apply_visual_inspiration_policy=True,
    )
    assert any("AHVI_VISUAL_COMPATIBILITY_REJECTED" in record.message for record in caplog.records)


def test_style_this_and_build_outfit_do_not_use_visual_enrichment():
    source = Path("services/style_flow_service.py").read_text(encoding="utf-8")
    assert "_enrich_visual_directions_with_assets" not in source


def test_production_does_not_branch_on_literal_archetype_titles():
    source = Path("services/style_reasoning_engine.py").read_text(encoding="utf-8")
    assert "if archetype == \"Modern Gentleman\"" not in source
    assert "if title == \"Refined Weekend\"" not in source


def test_future_archetype_fixture_uses_metadata_not_title_logic(monkeypatch):
    future = {
        "name": "Future Field Study",
        "impression": ["rugged"],
        "formality": 3,
        "best_for": ["outdoor"],
        "preferred_items": ["cargo trousers", "trainers"],
        "avoid_items": ["formal suiting"],
        "palette": ["olive"],
        "style_keywords": ["utility", "practical"],
    }
    monkeypatch.setattr(engine, "ARCHETYPE_LIBRARY", [*engine.ARCHETYPE_LIBRARY, future])
    direction = {"title": "Future Field Study", "hero_piece": "Field Jacket"}
    assert engine._visual_archetype_profile(direction)["name"] == "Future Field Study"
    assert engine._visual_asset_compatibility(
        _asset("future-cargo", "Cargo Trousers", "bottom"),
        direction=direction,
        occasion="outdoor",
    )[0]


def test_cutout_provenance_and_copy_contracts_remain_intact():
    direction = {"title": "Direction", "hero_piece": "Old Shirt"}
    hero = _asset("hero", "Selected Shirt", "top")
    engine._sync_visual_direction_copy(direction, [hero])
    assert direction["hero_piece"] == "Selected Shirt"
    assert direction["asset_id"] == "hero"
    assert direction["board_image_url"].endswith("hero.png")
    assert direction["complete_the_look"] == []
