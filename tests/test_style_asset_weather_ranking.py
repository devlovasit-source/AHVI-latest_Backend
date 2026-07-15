from services import style_reasoning_engine as engine
from services.style_context_service import build_canonical_style_context


def _asset(
    asset_id,
    *,
    weather_tags,
    metadata_status="ready",
    gender="unisex",
    subcategory="general",
    **extra,
):
    asset = {
        "asset_id": asset_id,
        "name": f"Asset {asset_id}",
        "category": "accessory",
        "role": "accessory",
        "subcategory": subcategory,
        "image_url": f"https://example.test/{asset_id}.png",
        "gender": gender,
        "gender_fit": gender,
        "occasions": ["daily"],
        "formality": 3,
        "traits": ["clean"],
        "weather_tags": weather_tags,
        "metadata_status": metadata_status,
    }
    if metadata_status == "ready":
        asset["colors"] = ["navy"]
    asset.update(extra)
    return asset


def _brief(temperature_c=None, *, condition="", weather_tags=None):
    weather = {}
    if temperature_c is not None:
        weather["temperature_c"] = temperature_c
    if condition:
        weather["condition"] = condition
    if weather_tags is not None:
        weather["weather_tags"] = weather_tags
    return {"weather_context": weather}


def test_structured_weather_survives_canonical_context_for_ranking():
    context = build_canonical_style_context(
        query="help me dress today",
        weather={"temp_c": 34, "tags": ["hot", "humid"]},
        memory={},
    )

    assert context["weather_context"] == {
        "temperature_c": 34,
        "weather_tags": ["hot", "humid"],
    }
    assert engine._asset_weather_score(
        _asset("hot-context", weather_tags=["hot"]), context
    ) > 0


def test_cold_weather_promotes_explicit_warm_layering_evidence():
    suitable = _asset(
        "cold-fit",
        weather_tags=["cold"],
        temperature_min_c=-5,
        temperature_max_c=12,
        fabric_weight="heavy",
        layering_suitability=0.9,
    )
    neutral = _asset("neutral", weather_tags=[])

    assert engine._asset_weather_score(suitable, _brief(5)) == 4
    assert engine._asset_weather_score(suitable, _brief(5)) > engine._asset_weather_score(
        neutral, _brief(5)
    )


def test_hot_weather_promotes_explicit_breathable_light_evidence():
    suitable = _asset(
        "hot-fit",
        weather_tags=["hot", "breathable"],
        temperature_min_c=24,
        temperature_max_c=42,
        fabric_weight="light",
        layering_suitability=0.2,
    )

    assert engine._asset_weather_score(suitable, _brief(34)) == 4


def test_hot_weather_ranks_hot_suitable_garment_above_neutral_above_cold_only():
    common = {
        "category": "top",
        "role": "top",
        "subcategory": "shirt",
        "image_url": "fixture-candidate-image",
        "gender": "unisex",
        "colors": ["navy"],
        "occasions": ["daily"],
        "formality": 3,
        "traits": ["clean"],
    }
    hot_suitable = {
        **common,
        "asset_id": "candidate-alpha",
        "name": "Candidate Alpha",
        "weather_tags": ["hot", "breathable"],
        "temperature_min_c": 24,
        "temperature_max_c": 42,
        "fabric_weight": "light",
        "layering_suitability": 0.2,
    }
    neutral = {
        **common,
        "asset_id": "candidate-beta",
        "name": "Candidate Beta",
        "weather_tags": [],
    }
    cold_only = {
        **common,
        "asset_id": "candidate-gamma",
        "name": "Candidate Gamma",
        "weather_tags": ["cold"],
        "temperature_min_c": -10,
        "temperature_max_c": 10,
        "fabric_weight": "heavy",
        "layering_suitability": 1.0,
    }
    brief = _brief(34, weather_tags=["hot"])

    scores = [
        engine._asset_score(
            engine._normalize_style_asset(asset),
            direction={"hero_piece": "shirt"},
            occasion="daily",
            target_gender="unisex",
            brief=brief,
        )
        for asset in (hot_suitable, neutral, cold_only)
    ]

    assert scores[0] > scores[1] > scores[2]


def test_cold_weather_ranks_cold_suitable_above_neutral_above_hot_only():
    cold_suitable = _asset(
        "candidate-alpha",
        weather_tags=["cold"],
        temperature_min_c=-10,
        temperature_max_c=10,
        fabric_weight="heavy",
        layering_suitability=1.0,
    )
    neutral = _asset("candidate-beta", weather_tags=[])
    hot_only = _asset(
        "candidate-gamma",
        weather_tags=["hot", "breathable"],
        temperature_min_c=24,
        temperature_max_c=42,
        fabric_weight="light",
    )
    brief = _brief(4, weather_tags=["cold"])

    scores = [
        engine._asset_weather_score(asset, brief)
        for asset in (cold_suitable, neutral, hot_only)
    ]

    assert scores[0] > scores[1] > scores[2]


def test_unknown_weather_preserves_baseline_order_for_all_evidence_profiles():
    assets = [
        _asset("hot", weather_tags=["hot"], fabric_weight="light"),
        _asset("neutral", weather_tags=[]),
        _asset("cold", weather_tags=["cold"], fabric_weight="heavy"),
    ]
    direction = {"hero_piece": "accessory"}
    baseline = [
        engine._asset_score(
            engine._normalize_style_asset(asset),
            direction=direction,
            occasion="daily",
            target_gender="unisex",
            brief=None,
        )
        for asset in assets
    ]
    unknown = [
        engine._asset_score(
            engine._normalize_style_asset(asset),
            direction=direction,
            occasion="daily",
            target_gender="unisex",
            brief={"weather_context": {}},
        )
        for asset in assets
    ]

    assert unknown == baseline


def test_connected_visual_enrichment_uses_structured_hot_weather(monkeypatch):
    common = {
        "category": "top",
        "role": "top",
        "subcategory": "shirt",
        "gender": "unisex",
        "colors": ["navy"],
        "occasions": ["daily"],
        "formality": 3,
        "traits": ["clean"],
        "board_image_url": "fixture-board-image",
        "image_url": "fixture-item-image",
    }
    assets = [
        {
            **common,
            "asset_id": "hot-suitable",
            "name": "Candidate Alpha",
            "weather_tags": ["hot", "breathable"],
            "temperature_min_c": 24,
            "temperature_max_c": 42,
            "fabric_weight": "light",
        },
        {**common, "asset_id": "neutral", "name": "Candidate Beta", "weather_tags": []},
        {
            **common,
            "asset_id": "cold-only",
            "name": "Candidate Gamma",
            "weather_tags": ["cold"],
            "temperature_min_c": -10,
            "temperature_max_c": 10,
            "fabric_weight": "heavy",
        },
    ]
    monkeypatch.setattr(engine, "_style_asset_rows", lambda: assets)

    enriched = engine._enrich_visual_directions_with_assets(
        [{"title": "Daily", "hero_piece": "shirt", "items": ["shirt"]}],
        occasion="daily",
        target_gender="unisex",
        brief=_brief(34, weather_tags=["hot"]),
    )

    assert enriched[0]["asset_id"] == "hot-suitable"


def test_weather_incompatibility_penalty_is_bounded():
    incompatible = _asset(
        "incompatible",
        weather_tags=["cold"],
        temperature_min_c=-10,
        temperature_max_c=10,
        fabric_weight="heavy",
        layering_suitability=1.0,
    )

    assert engine._asset_weather_score(incompatible, _brief(36)) == -4


def test_unknown_weather_is_neutral():
    asset = _asset(
        "weathered",
        weather_tags=["cold"],
        temperature_min_c=-10,
        temperature_max_c=10,
        fabric_weight="heavy",
    )

    assert engine._asset_weather_score(asset, {}) == 0
    assert engine._asset_weather_score(asset, {"weather_context": {}}) == 0


def test_readiness_cannot_override_weather_incompatibility():
    compatible_limited = _asset(
        "compatible-limited",
        weather_tags=["hot"],
        metadata_status="limited",
        subcategory="bag",
        temperature_min_c=24,
        temperature_max_c=42,
        fabric_weight="light",
    )
    incompatible_ready = _asset(
        "incompatible-ready",
        weather_tags=["cold"],
        subcategory="jewelry",
        temperature_min_c=-10,
        temperature_max_c=10,
        fabric_weight="heavy",
    )

    selected = engine._best_style_assets(
        [incompatible_ready, compatible_limited],
        direction={"hero_piece": "accessory"},
        occasion="daily",
        accessory_only=True,
        target_gender="unisex",
        brief=_brief(34),
        limit=2,
    )

    assert selected[0]["asset_id"] == "compatible-limited"
    normalized_limited = engine._normalize_style_asset(compatible_limited)
    normalized_ready = engine._normalize_style_asset(incompatible_ready)
    assert normalized_limited["metadata_status"] == "limited"
    assert normalized_ready["metadata_status"] == "ready"
    assert engine._asset_score(
        normalized_limited,
        direction={"hero_piece": "accessory"},
        occasion="daily",
        target_gender="unisex",
        brief=_brief(34),
    ) > engine._asset_score(
        normalized_ready,
        direction={"hero_piece": "accessory"},
        occasion="daily",
        target_gender="unisex",
        brief=_brief(34),
    )


def test_weather_fit_does_not_bypass_gender_eligibility():
    wrong_gender = _asset(
        "wrong-gender",
        weather_tags=["hot"],
        gender="female",
        temperature_min_c=24,
        temperature_max_c=42,
        fabric_weight="light",
    )
    eligible_neutral = _asset("eligible", weather_tags=[], gender="male")

    selected = engine._best_style_assets(
        [wrong_gender, eligible_neutral],
        direction={"hero_piece": "accessory"},
        occasion="daily",
        accessory_only=True,
        target_gender="male",
        brief=_brief(34),
        limit=2,
    )

    assert [asset["asset_id"] for asset in selected] == ["eligible"]


def test_public_weather_compatibility_contract_delegates_to_canonical_helper():
    """Anchor callers use the public contract without gaining new weather rules."""
    asset = _asset(
        "public-contract",
        weather_tags=["cold"],
        temperature_min_c=-10,
        temperature_max_c=12,
        fabric_weight="heavy",
        layering_suitability=0.9,
    )

    for brief in (
        _brief(4, weather_tags=["cold"]),
        _brief(34, weather_tags=["hot"]),
        _brief(),
        {"weather_context": {"temperature_c": "not-a-number"}},
    ):
        assert engine.asset_weather_compatibility_score(asset, brief) == engine._asset_weather_score(
            asset, brief
        )


def test_public_weather_contract_uses_structured_evidence_not_name_tokens():
    # A name alone is not evidence that a garment is breathable or lightweight.
    asset = _asset("no-fabric-inference", weather_tags=[], name="Linen summer shirt")

    assert engine.asset_weather_compatibility_score(asset, _brief(35, weather_tags=["hot"])) == 0


def test_public_weather_contract_scarf_cold_hot_and_unknown_behavior():
    scarf = _asset(
        "scarf",
        weather_tags=["cold"],
        temperature_min_c=-10,
        temperature_max_c=16,
        fabric_weight="heavy",
        layering_suitability=0.8,
    )

    assert engine.asset_weather_compatibility_score(scarf, _brief(5, weather_tags=["cold"])) > 0
    assert engine.asset_weather_compatibility_score(scarf, _brief(34, weather_tags=["hot"])) < 0
    assert engine.asset_weather_compatibility_score(scarf, _brief()) == 0
