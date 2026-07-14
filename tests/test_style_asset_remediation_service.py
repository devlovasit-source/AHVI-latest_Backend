import pytest

from services.style_asset_remediation_service import (
    RemediationError,
    build_asset_id_document_id_map,
    build_remediation_plan,
    execute_remediation_plan,
    rollback_journal,
    sanitize_metadata_payload,
    safe_field_checksum,
)


def _snapshot():
    return [{
        "$id": "long_asset_identifier_truncated_by_appwrit",
        "$createdAt": "2026-01-01T00:00:00Z",
        "$permissions": [],
        "asset_id": "long_asset_identifier_truncated_by_appwrite",
        "name": "Candidate",
        "category": "top",
        "role": "top",
        "subcategory": "shirt",
        "image_url": "fixture-original-image",
        "board_image_url": "fixture-board-image",
        "gender": "unisex",
        "colors": ["navy"],
        "occasions": ["dailywear"],
        "formality": 3,
        "traits": ["clean"],
        "weather_tags": ["hot"],
    }]


def test_mapping_requires_and_preserves_exact_appwrite_document_id():
    mapping = build_asset_id_document_id_map(_snapshot())

    assert mapping == {
        "long_asset_identifier_truncated_by_appwrite":
            "long_asset_identifier_truncated_by_appwrit"
    }


def test_apply_and_rollback_are_exact_id_and_batch_field_scoped():
    plan = build_remediation_plan(
        _snapshot(),
        {
            "long_asset_identifier_truncated_by_appwrite": {
                "professional_safe": True,
                "safety_tags": ["office"],
            }
        },
    )

    update = plan["updates"][0]
    rollback = plan["rollbacks"][0]
    expected_id = "long_asset_identifier_truncated_by_appwrit"
    assert update["document_id"] == rollback["document_id"] == expected_id
    assert update["changes"]["professional_safe"] is True
    assert rollback["changes"]["professional_safe"] is None
    forbidden = {"$id", "$permissions", "image_url", "board_image_url"}
    assert forbidden.isdisjoint(update["changes"])
    assert forbidden.isdisjoint(rollback["changes"])
    assert set(rollback["changes"]) == set(update["changes"])


@pytest.mark.parametrize("field", [
    "$id", "$createdAt", "$permissions", "asset_id", "user_id", "owner_id",
    "source", "source_origin", "image_url", "board_image_url", "normalized_url",
    "cutout_url", "catalog_image_url", "r2_key", "unknown_field",
])
def test_tampering_and_unknown_fields_are_typed_rejections(field):
    with pytest.raises(RemediationError) as exc:
        build_remediation_plan(
            _snapshot(),
            {"long_asset_identifier_truncated_by_appwrite": {field: "tampered"}},
        )

    assert exc.value.code == "PAYLOAD_FIELD_FORBIDDEN"


def test_base64_content_is_rejected():
    with pytest.raises(RemediationError) as exc:
        build_remediation_plan(
            _snapshot(),
            {
                "long_asset_identifier_truncated_by_appwrite": {
                    "traits": ["data:image/png;base64,forbidden"]
                }
            },
        )

    assert exc.value.code == "PAYLOAD_CONTENT_FORBIDDEN"


def test_derived_proposal_values_must_match_canonical_recomputation():
    with pytest.raises(ValueError, match="metadata_score disagrees"):
        build_remediation_plan(
            _snapshot(),
            {
                "long_asset_identifier_truncated_by_appwrite": {
                    "metadata_score": 0.01,
                }
            },
        )


def test_unknown_vocabularies_fail_closed_before_plan_is_emitted():
    with pytest.raises(ValueError, match="unknown canonical vocabulary"):
        build_remediation_plan(
            _snapshot(),
            {
                "long_asset_identifier_truncated_by_appwrite": {
                    "occasion_families": ["invented_event"],
                }
            },
        )


def test_payload_sanitizer_drops_system_identifiers_and_media_fields():
    assert sanitize_metadata_payload({
        "$id": "doc",
        "$createdAt": "now",
        "asset_id": "asset",
        "image_url": "private",
        "cutout_url": "private",
        "role": "top",
        "metadata_status": "ready",
    }) == {"role": "top", "metadata_status": "ready"}


def test_all_six_known_truncated_ids_map_exactly():
    pairs = {
        "mens_assets_outerwear_blackplaidscarf": "mens_assets_outerwear_blackplaidscar",
        "meghna_female_top_black_and_white_stripe_tshirt": "meghna_female_top_black_and_white_st",
        "meghna_female_bottom_beige_pants_high_waisted": "meghna_female_bottom_beige_pants_hig",
        "meghna_female_dress_baby_blue_midi_dress": "meghna_female_dress_baby_blue_midi_d",
        "meghna_female_ethnic_black_and_gold_lehenga": "meghna_female_ethnic_black_and_gold_",
        "meghna_female_accessory_beige_shell_bag": "meghna_female_accessory_beige_shell_",
    }
    snapshot = [
        {"asset_id": asset_id, "$id": document_id}
        for asset_id, document_id in pairs.items()
    ]

    assert build_asset_id_document_id_map(snapshot) == pairs


def test_rejected_controls_cannot_enter_approved_allowlist():
    with pytest.raises(RemediationError) as exc:
        build_remediation_plan(
            _snapshot(),
            {
                "long_asset_identifier_truncated_by_appwrite": {
                    "professional_safe": True,
                }
            },
            rejected_asset_ids=["long_asset_identifier_truncated_by_appwrite"],
        )

    assert exc.value.code == "REJECTED_CONTROL_INCLUDED"


def _planned_store():
    snapshot = _snapshot()
    snapshot[0]["$updatedAt"] = "reviewed-version"
    plan = build_remediation_plan(
        snapshot,
        {
            "long_asset_identifier_truncated_by_appwrite": {
                "professional_safe": True,
                "safety_tags": ["office"],
            }
        },
    )
    return snapshot[0], plan


def test_stale_snapshot_rejects_without_updating():
    reviewed, plan = _planned_store()
    live = {**reviewed, "$updatedAt": "newer-version"}
    calls = []

    result = execute_remediation_plan(
        plan,
        fetch_document=lambda _document_id: live,
        update_document=lambda *args: calls.append(args),
    )

    assert result["success"] is False
    assert result["failure_code"] == "DOCUMENT_CHANGED_SINCE_REVIEW"
    assert result["completed_document_ids"] == []
    assert calls == []


def test_partial_failure_stops_and_reports_only_completed_subset():
    reviewed, single = _planned_store()
    second = {**reviewed, "$id": "second-doc", "asset_id": "second-asset"}
    plan = build_remediation_plan(
        [reviewed, second],
        {
            "long_asset_identifier_truncated_by_appwrite": {"professional_safe": True},
            "second-asset": {"professional_safe": True},
        },
    )
    store = {reviewed["$id"]: dict(reviewed), "second-doc": dict(second)}
    calls = []

    def update(document_id, changes):
        calls.append(document_id)
        if document_id == "second-doc":
            raise RuntimeError("injected failure")
        store[document_id].update(changes)
        return store[document_id]

    result = execute_remediation_plan(
        plan,
        fetch_document=lambda document_id: store[document_id],
        update_document=update,
        now=lambda: "fixed-time",
    )

    assert result["success"] is False
    assert result["failure_code"] == "DOCUMENT_UPDATE_FAILED"
    assert result["completed_document_ids"] == [reviewed["$id"]]
    assert calls == [reviewed["$id"], "second-doc"]
    assert result["journal"][0]["timestamp"] == "fixed-time"


def test_safe_rollback_restores_only_journaled_fields():
    reviewed, plan = _planned_store()
    store = {reviewed["$id"]: dict(reviewed)}

    def update(document_id, changes):
        store[document_id].update(changes)
        return store[document_id]

    applied = execute_remediation_plan(
        plan,
        fetch_document=lambda document_id: store[document_id],
        update_document=update,
    )
    unrelated = store[reviewed["$id"]].setdefault("unrelated_field", "preserved")
    assert unrelated == "preserved"

    rolled_back = rollback_journal(
        applied["journal"],
        fetch_document=lambda document_id: store[document_id],
        update_document=update,
    )

    assert rolled_back["success"] is True
    assert store[reviewed["$id"]]["unrelated_field"] == "preserved"
    assert set(applied["journal"][0]["prior_values"]) == set(
        applied["journal"][0]["changed_fields"]
    )


def test_rollback_fails_closed_after_concurrent_change():
    reviewed, plan = _planned_store()
    store = {reviewed["$id"]: dict(reviewed)}

    def update(document_id, changes):
        store[document_id].update(changes)
        return store[document_id]

    applied = execute_remediation_plan(
        plan,
        fetch_document=lambda document_id: store[document_id],
        update_document=update,
    )
    store[reviewed["$id"]]["professionalism_score"] = 0.123
    calls_before = safe_field_checksum(store[reviewed["$id"]])

    rolled_back = rollback_journal(
        applied["journal"],
        fetch_document=lambda document_id: store[document_id],
        update_document=update,
    )

    assert rolled_back["success"] is False
    assert rolled_back["failure_code"] == "ROLLBACK_CONCURRENCY_CONFLICT"
    assert safe_field_checksum(store[reviewed["$id"]]) == calls_before
