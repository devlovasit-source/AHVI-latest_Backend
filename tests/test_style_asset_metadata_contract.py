from services.style_asset_metadata_contract import (
    canonical_metadata_update,
    normalize_style_asset_metadata,
    summarize_style_asset_metadata,
)


def _complete(**overrides):
    row = {
        "asset_id": "asset-1",
        "name": "Navy Tailored Blazer",
        "category": "outerwear",
        "image_url": "https://cdn.test/blazer.png",
        "gender": "men",
        "colors": ["navy"],
        "occasions": ["office"],
        "formality": "7",
        "traits": ["tailored"],
        "source": "manifest_import",
    }
    row.update(overrides)
    return row


def test_complete_asset_is_ready_with_canonical_aliases():
    asset = normalize_style_asset_metadata(_complete(), trusted_style_asset_source=True)

    assert asset["metadata_status"] == "ready"
    assert asset["source"] == "style_asset"
    assert asset["source_origin"] == "manifest_import"
    assert asset["role"] == "outerwear"
    assert asset["gender_fit"] == "male"
    assert asset["board_image_url"] == "https://cdn.test/blazer.png"
    assert asset["formality"] == 7


def test_incomplete_but_usable_asset_is_limited():
    asset = normalize_style_asset_metadata(
        _complete(colors=[], occasions=[], formality="", traits=[]),
        trusted_style_asset_source=True,
    )

    assert asset["metadata_status"] == "limited"
    assert {"colors", "occasion_or_archetype", "style_or_safety_detail"}.issubset(
        asset["missing_metadata_fields"]
    )


def test_missing_image_invalid_role_and_unsafe_asset_are_rejected():
    missing_image = normalize_style_asset_metadata(_complete(image_url=""), trusted_style_asset_source=True)
    invalid_role = normalize_style_asset_metadata(_complete(category="grooming"), trusted_style_asset_source=True)
    unsafe = normalize_style_asset_metadata(_complete(unsafe=True), trusted_style_asset_source=True)

    assert missing_image["metadata_status"] == "rejected"
    assert invalid_role["metadata_status"] == "rejected"
    assert unsafe["metadata_status"] == "rejected"


def test_unknown_semantics_are_not_invented_and_gender_remains_unknown():
    asset = normalize_style_asset_metadata(
        {
            "asset_id": "unknown-1",
            "name": "Unknown Piece",
            "category": "top",
            "image_url": "https://cdn.test/unknown.png",
        },
        trusted_style_asset_source=True,
    )

    assert asset["gender_fit"] == "unknown"
    assert asset["colors"] == []
    assert "pattern" not in asset
    assert "material" not in asset
    assert asset["metadata_status"] == "limited"


def test_normalization_and_backfill_are_deterministic_and_idempotent():
    first = normalize_style_asset_metadata(_complete(), trusted_style_asset_source=True)
    second = normalize_style_asset_metadata(_complete(), trusted_style_asset_source=True)
    assert first == second

    update = canonical_metadata_update(_complete())
    persisted = {**_complete(), **update}
    assert canonical_metadata_update(persisted) == update


def test_audit_summary_reports_exact_aggregates():
    rows = [
        _complete(),
        _complete(asset_id="limited", colors=[], occasions=[], formality="", traits=[]),
        _complete(asset_id="rejected", image_url=""),
    ]
    summary = summarize_style_asset_metadata(rows)

    assert summary["total"] == 3
    assert summary["status_counts"] == {"limited": 1, "ready": 1, "rejected": 1}
    assert summary["invalid_image_count"] == 1
    assert summary["missing_field_counts"]["colors"] == 1


def test_runtime_excludes_rejected_but_keeps_limited_assets(monkeypatch):
    from services import appwrite_proxy
    from services import style_reasoning_engine as engine

    rows = [
        _complete(asset_id="ready"),
        _complete(asset_id="limited", colors=[], occasions=[], formality="", traits=[]),
        _complete(asset_id="rejected", image_url=""),
    ]

    class FakeProxy:
        def list_documents(self, resource, **kwargs):
            return {"documents": rows, "meta": {"has_more": False}}

    monkeypatch.setattr(appwrite_proxy, "AppwriteProxy", FakeProxy)
    loaded = engine._style_asset_rows(limit=10)

    assert [asset["asset_id"] for asset in loaded] == ["ready", "limited"]


def test_ready_only_breaks_an_equivalent_runtime_score(monkeypatch):
    from services import style_reasoning_engine as engine

    limited = _complete(asset_id="limited", colors=[], occasions=[], formality="", traits=[])
    ready = _complete(asset_id="ready")
    monkeypatch.setattr(engine, "_asset_allowed_for_gender", lambda *args, **kwargs: True)
    monkeypatch.setattr(engine, "_hero_asset_allowed", lambda *args, **kwargs: True)
    monkeypatch.setattr(engine, "_occasion_asset_block_reason", lambda *args, **kwargs: "")
    monkeypatch.setattr(engine, "_asset_allowed_for_context", lambda *args, **kwargs: True)
    monkeypatch.setattr(engine, "_asset_score", lambda *args, **kwargs: 10)
    monkeypatch.setattr(engine, "_asset_context_score", lambda *args, **kwargs: 0)

    selected = engine._best_style_assets(
        [limited, ready],
        direction={"hero_piece": "blazer"},
        occasion="office",
        limit=2,
    )

    assert [asset["asset_id"] for asset in selected] == ["ready", "limited"]


def test_import_boundary_accepts_board_image_and_preserves_source_origin():
    from scripts.import_style_assets import _normalize

    payload = _normalize({
        "asset_id": "board-only",
        "name": "Board Only Top",
        "category": "top",
        "board_image_url": "https://cdn.test/board-only.png",
        "source": "curated_seed",
    })

    assert payload["image_url"] == "https://cdn.test/board-only.png"
    assert payload["source"] == "style_asset"
    assert payload["source_origin"] == "curated_seed"


def test_scarf_and_glove_evidence_corrects_outerwear_taxonomy():
    scarf = normalize_style_asset_metadata(
        _complete(name="Black Plaid Scarf", category="outerwear", subcategory="scarf"),
        trusted_style_asset_source=True,
    )
    gloves = normalize_style_asset_metadata(
        _complete(name="Black Gloves", category="outerwear", subcategory="jacket"),
        trusted_style_asset_source=True,
    )

    assert (scarf["role"], scarf["category"], scarf["subcategory"]) == (
        "accessory", "accessory", "scarf"
    )
    assert (gloves["role"], gloves["category"], gloves["subcategory"]) == (
        "accessory", "accessory", "gloves"
    )


def test_sleepwear_set_uses_existing_one_piece_role_without_broadening_taxonomy():
    asset = normalize_style_asset_metadata(
        _complete(category="loungewear", subcategory="sleepwear", colors=[]),
        trusted_style_asset_source=True,
    )

    assert asset["role"] == "one_piece"
    assert asset["category"] == "loungewear"
    assert asset["metadata_status"] == "limited"


def test_professional_fields_are_normalized_on_runtime_axis():
    asset = normalize_style_asset_metadata(
        _complete(
            formality="5",
            professional_safe="true",
            professionalism_score="0.9",
            clientMeetingScore="0.85",
            safety_tags="office|client meeting",
        ),
        trusted_style_asset_source=True,
    )

    assert asset["formality"] == 5
    assert asset["professional_safe"] is True
    assert asset["professionalism_score"] == 0.9
    assert asset["client_meeting_score"] == 0.85
    assert asset["safety_tags"] == ["office", "client_meeting"]


def test_importer_preserves_professional_safety_fields():
    from scripts.import_style_assets import _normalize

    payload = _normalize(_complete(
        professional_safe=False,
        professionalism_score="0.4",
        clientMeetingScore="0.3",
        boardroom_score=0.2,
        safety_tags="casual|not boardroom",
    ))

    assert payload["professional_safe"] is False
    assert payload["professionalism_score"] == 0.4
    assert payload["client_meeting_score"] == 0.3
    assert payload["boardroom_score"] == 0.2
    assert payload["safety_tags"] == ["casual", "not_boardroom"]


def test_unknown_occasion_and_safety_vocabularies_fail_closed():
    asset = normalize_style_asset_metadata(
        _complete(
            occasions=["dailywear", "invented_event"],
            safety_tags=["office", "invented_safety"],
        ),
        trusted_style_asset_source=True,
    )

    assert asset["occasion_families"] == ["daily"]
    assert asset["safety_tags"] == ["office"]
    assert asset["metadata_status"] == "rejected"
    assert "invalid_occasion_families" in asset["missing_metadata_fields"]
    assert "invalid_safety_tags" in asset["missing_metadata_fields"]


def test_unknown_occasion_label_cannot_enter_asset_ranking():
    from services import style_reasoning_engine as engine

    invalid = _complete(
        asset_id="invalid-vocab",
        name="Candidate Alpha",
        category="top",
        occasions=["invented_event"],
    )
    valid = _complete(
        asset_id="valid-vocab",
        name="Candidate Beta",
        category="top",
        occasions=["daily"],
    )

    selected = engine._best_style_assets(
        [invalid, valid],
        direction={"hero_piece": "top"},
        occasion="daily",
        target_gender="male",
        limit=2,
    )

    assert [asset["asset_id"] for asset in selected] == ["valid-vocab"]


def test_schema_bootstrap_declares_professional_safety_fields(monkeypatch):
    from scripts import create_style_assets_collection as schema

    created = {"string": [], "float": [], "boolean": [], "datetime": []}
    monkeypatch.setattr(schema, "_create_collection", lambda: None)
    monkeypatch.setattr(schema, "_create_string", lambda key, *args, **kwargs: created["string"].append(key))
    monkeypatch.setattr(schema, "_create_float", lambda key: created["float"].append(key))
    monkeypatch.setattr(schema, "_create_boolean", lambda key: created["boolean"].append(key))
    monkeypatch.setattr(schema, "_create_datetime", lambda key: created["datetime"].append(key))

    schema.bootstrap()

    assert {"professional_safe", "rain_suitable", "wind_suitable"} == set(created["boolean"])
    assert {"professionalism_score", "client_meeting_score", "boardroom_score"}.issubset(
        created["float"]
    )
    assert "safety_tags" in created["string"]


def test_structured_weather_evidence_survives_contract_and_importer():
    from scripts.import_style_assets import _normalize

    payload = _normalize(_complete(
        weather_tags=["hot", "humid"],
        temperatureMinC="24",
        temperature_max_c=42,
        fabricWeight="lightweight",
        layering_suitability="0.25",
        rainSuitable=False,
        wind_suitable=True,
    ))

    assert payload["weather_tags"] == ["hot", "humid"]
    assert payload["temperature_min_c"] == 24
    assert payload["temperature_max_c"] == 42
    assert payload["fabric_weight"] == "light"
    assert payload["layering_suitability"] == 0.25
    assert payload["rain_suitable"] is False
    assert payload["wind_suitable"] is True
