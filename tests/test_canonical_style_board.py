import builtins
from copy import deepcopy
import socket

import httpx
import pytest
import requests

from services import canonical_style_board as csb


CREATED_AT = "2026-07-22T10:00:00+05:30"
WARDROBE_LOOK = {
    "id": "legacy-look-1",
    "items": [
        {"id": "top-1", "name": "White shirt", "category": "top", "source": "wardrobe"},
        {"id": "bottom-1", "name": "Navy trousers", "category": "bottom", "source": "wardrobe"},
        {"id": "shoe-1", "name": "Loafers", "category": "footwear", "source": "wardrobe"},
    ],
}


def _fingerprint(**overrides):
    values = {
        "intent": "Office meeting",
        "scenario": "wardrobe",
        "canonical_context": {"gender": "male", "canonical_occasion": "office"},
        "source_policy": "wardrobe",
    }
    values.update(overrides)
    return csb.request_fingerprint(**values)


def _finalize(candidate=WARDROBE_LOOK, **overrides):
    values = {
        "candidates": [candidate],
        "user_id": "user-1",
        "flow": "wardrobe",
        "generation_request_id": "request-1",
        "scenario": "wardrobe",
        "request_fingerprint_value": _fingerprint()["request_fingerprint"],
        "created_at": CREATED_AT,
    }
    values.update(overrides)
    return csb.finalize_candidates(**values)


def _draft_dict(result):
    return result.boards[0].to_dict()


def test_request_fingerprint_is_stable_for_context_key_and_wardrobe_order():
    item_a = {"id": "a", "source": "wardrobe", "category": "top"}
    item_b = {"id": "b", "source": "wardrobe", "category": "bottom"}
    first = _fingerprint(
        canonical_context={"wardrobe": [item_b, item_a], "gender": "male"},
        anchor_ids=["b", "a"],
    )
    second = _fingerprint(
        canonical_context={"gender": "male", "wardrobe": [item_a, item_b]},
        anchor_ids=["a", "b"],
    )
    assert first == second


def test_request_fingerprint_ignores_volatile_and_non_material_context():
    first = _fingerprint(canonical_context={"gender": "male", "timestamp": "one", "debug": 1})
    second = _fingerprint(canonical_context={"gender": "male", "timestamp": "two", "debug": 2})
    assert first["request_fingerprint"] == second["request_fingerprint"]


def test_request_fingerprint_changes_for_material_context_or_versions():
    male = _fingerprint(canonical_context={"gender": "male"})
    female = _fingerprint(canonical_context={"gender": "female"})
    versioned = _fingerprint(
        canonical_context={"gender": "male"}, user_context_versions={"profile": "2"}
    )
    assert len({male["request_fingerprint"], female["request_fingerprint"], versioned["request_fingerprint"]}) == 3


def test_candidate_hash_is_separate_from_request_fingerprint():
    fingerprint = _fingerprint()
    first = csb.candidate_snapshot_hash([WARDROBE_LOOK])
    changed = deepcopy(WARDROBE_LOOK)
    changed["items"][0]["name"] = "Blue shirt"
    second = csb.candidate_snapshot_hash([changed])
    assert first != second
    assert fingerprint == _fingerprint()


def test_snapshot_hash_retains_created_at_while_context_hash_ignores_it():
    assert csb.stable_hash({"created_at": "one"}) != csb.stable_hash({"created_at": "two"})
    assert csb.context_hash({"gender": "male", "created_at": "one"}) == csb.context_hash(
        {"gender": "male", "created_at": "two"}
    )


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (["wardrobe", "wardrobe"], "wardrobe"),
        (["style_asset", "style_asset"], "style_asset"),
        (["wardrobe", "style_asset"], "mixed"),
        (["wardrobe", "catalog"], "mixed"),
        (["catalog"], "unknown"),
        (["unknown", "wardrobe"], "unknown"),
    ],
)
def test_source_policy_is_derived_from_provenance(sources, expected):
    items = [{"id": str(index), "source": source} for index, source in enumerate(sources)]
    assert csb.derive_source_policy(items) == expected


def test_non_actionable_gap_does_not_change_source_policy():
    items = [
        {"id": "owned", "source": "wardrobe"},
        {"id": "gap", "source": "catalog", "actionable": False},
    ]
    assert csb.derive_source_policy(items) == "wardrobe"


def test_generated_candidate_with_unknown_provenance_is_rejected():
    unknown = deepcopy(WARDROBE_LOOK)
    unknown["items"][0].pop("source")
    result = _finalize(candidate=unknown)
    assert result.boards == []
    assert result.rejected[0].code == "SOURCE_PROVENANCE_UNKNOWN"


def test_variant_identity_survives_candidate_reranking():
    other = deepcopy(WARDROBE_LOOK)
    other["id"] = "legacy-look-2"
    other["items"][0]["id"] = "top-2"
    first = _finalize().boards
    reranked = _finalize(candidate=other).boards
    combined = csb.finalize_candidates(
        [other, WARDROBE_LOOK], user_id="user-1", flow="wardrobe",
        generation_request_id="request-1", scenario="wardrobe",
        request_fingerprint_value=_fingerprint()["request_fingerprint"], created_at=CREATED_AT,
    ).boards
    assert first[0].board_variant_key == combined[1].board_variant_key
    assert reranked[0].board_variant_key == combined[0].board_variant_key
    assert first[0].board_id is combined[1].board_id is None


def test_future_registered_identity_is_stable_and_user_scoped():
    variant_key = csb.board_variant_key(WARDROBE_LOOK, "wardrobe")
    first = csb.derive_registered_board_id(
        user_id="user-1", flow="wardrobe", generation_request_id="request-1",
        variant_key=variant_key,
    )
    retry = csb.derive_registered_board_id(
        user_id="user-1", flow="wardrobe", generation_request_id="request-1",
        variant_key=variant_key,
    )
    other_user = csb.derive_registered_board_id(
        user_id="user-2", flow="wardrobe", generation_request_id="request-1",
        variant_key=variant_key,
    )
    assert first == retry
    assert first != other_user


def test_item_order_is_part_of_variant_signature():
    reordered = deepcopy(WARDROBE_LOOK)
    reordered["items"].reverse()
    assert csb.board_variant_key(WARDROBE_LOOK, "wardrobe") != csb.board_variant_key(reordered, "wardrobe")


def test_finalizer_builds_complete_contract_without_mutating_input():
    original = deepcopy(WARDROBE_LOOK)
    result = _finalize()
    board = _draft_dict(result)
    required = {
        "schema_version", "board_id", "revision", "lifecycle_status", "persisted",
        "feedback_context", "generation_request_id",
        "board_variant_key", "created_at", "scenario", "source_policy", "items",
        "request_fingerprint", "request_fingerprint_version", "candidate_snapshot_hash",
        "finalizer_version", "serializer_version", "guard_version", "snapshot_hash",
        "provenance", "capabilities", "extensions",
    }
    assert required <= board.keys()
    assert WARDROBE_LOOK == original
    assert board["created_at"] == "2026-07-22T04:30:00Z"
    assert set(board["extensions"]) == {"capsule", "visual_inspiration", "wardrobe", "fixed_anchor"}


def test_finalizer_emits_explicit_draft_lifecycle_and_outage_profile():
    board = _draft_dict(_finalize())
    assert board["lifecycle_status"] == "draft"
    assert board["persisted"] is False
    assert board["board_id"] is None
    assert board["revision"] is None
    assert board["feedback_context"] is None
    assert board["capabilities"] == csb.registration_outage_capabilities()


def test_pure_finalizer_has_no_registered_construction_parameter():
    with pytest.raises(TypeError, match="persisted"):
        _finalize(persisted=True)


def test_draft_model_rejects_registered_lifecycle_fields():
    invalid = _draft_dict(_finalize())
    invalid["lifecycle_status"] = "registered"
    invalid["persisted"] = True
    invalid["board_id"] = "fake"
    invalid["revision"] = 1
    with pytest.raises(csb.CanonicalStyleBoardError) as exc:
        csb.CanonicalStyleBoardDraft(**invalid)
    assert exc.value.code == "INVALID_DRAFT_LIFECYCLE"


def test_unknown_provenance_and_outage_profiles_remain_distinct():
    unknown = csb.unknown_provenance_capabilities()
    outage = csb.registration_outage_capabilities()
    assert unknown["save"]["allowed"] is True
    assert unknown["feedback"]["allowed"] is True
    assert outage["save"]["allowed"] is False
    assert outage["feedback"]["allowed"] is False


def test_visual_inspiration_finalizes_as_registration_outage_draft():
    visual = {
        "items": [
            {"id": "asset-1", "name": "Silk blouse", "category": "top", "source": "style_asset"}
        ]
    }
    result = _finalize(
        candidate=visual, flow="visual_reasoning", scenario="visual_inspiration",
        request_fingerprint_value=_fingerprint(
            scenario="visual_inspiration", source_policy="style_asset"
        )["request_fingerprint"],
        extensions={"visual_inspiration": {"direction": "polished"}},
    )
    board = _draft_dict(result)
    assert board["scenario"] == "visual_inspiration"
    assert board["source_policy"] == "style_asset"
    assert board["extensions"]["visual_inspiration"] == {"direction": "polished"}
    assert board["lifecycle_status"] == "draft"
    assert board["capabilities"] == csb.registration_outage_capabilities()


def test_wardrobe_policy_rejects_mixed_and_incomplete_candidates():
    mixed = deepcopy(WARDROBE_LOOK)
    mixed["items"][0]["source"] = "style_asset"
    incomplete = deepcopy(WARDROBE_LOOK)
    incomplete["items"] = incomplete["items"][:2]
    result = csb.finalize_candidates(
        [mixed, incomplete], user_id="user-1", flow="wardrobe",
        generation_request_id="request-1", scenario="wardrobe",
        request_fingerprint_value=_fingerprint()["request_fingerprint"], created_at=CREATED_AT,
    )
    assert result.boards == []
    assert {issue.code for issue in result.rejected} == {"OWNERSHIP_REQUIRED", "OUTFIT_INCOMPLETE"}


def test_fixed_anchor_requires_requested_anchor():
    result = _finalize(
        flow="fixed_anchor", scenario="fixed_anchor", anchor_ids=["missing-anchor"],
        request_fingerprint_value=_fingerprint(scenario="fixed_anchor", anchor_ids=["missing-anchor"])["request_fingerprint"],
    )
    assert result.boards == []
    assert result.rejected[0].code == "ANCHOR_MISSING"


def test_fixed_anchor_success_preserves_anchor_as_draft():
    result = _finalize(
        flow="fixed_anchor", scenario="fixed_anchor", anchor_ids=["top-1"],
        request_fingerprint_value=_fingerprint(
            scenario="fixed_anchor", anchor_ids=["top-1"]
        )["request_fingerprint"],
        extensions={"fixed_anchor": {"anchor_ids": ["top-1"]}},
    )
    board = _draft_dict(result)
    assert board["scenario"] == "fixed_anchor"
    assert board["extensions"]["fixed_anchor"]["anchor_ids"] == ["top-1"]
    assert board["lifecycle_status"] == "draft"


def test_capsule_finalization_uses_policy_provenance_and_extensions():
    result = _finalize(
        flow="capsule", scenario="capsule",
        request_fingerprint_value=_fingerprint(scenario="capsule")["request_fingerprint"],
        extensions={"capsule": {"foundation_ids": ["top-1", "bottom-1"]}},
    )
    board = _draft_dict(result)
    assert csb.SCENARIO_POLICIES["capsule"].require_capsule_completeness is True
    assert board["scenario"] == "capsule"
    assert board["source_policy"] == "wardrobe"
    assert board["extensions"]["capsule"]["foundation_ids"] == ["top-1", "bottom-1"]
    assert board["lifecycle_status"] == "draft"
    assert board["board_id"] is board["revision"] is None


def test_duplicate_variants_and_duplicate_items_are_rejected():
    duplicate_items = deepcopy(WARDROBE_LOOK)
    duplicate_items["items"][1] = deepcopy(duplicate_items["items"][0])
    result = csb.finalize_candidates(
        [WARDROBE_LOOK, deepcopy(WARDROBE_LOOK), duplicate_items],
        user_id="user-1", flow="wardrobe", generation_request_id="request-1",
        scenario="wardrobe", request_fingerprint_value=_fingerprint()["request_fingerprint"],
        created_at=CREATED_AT,
    )
    assert len(result.boards) == 1
    assert {issue.code for issue in result.rejected} == {"DUPLICATE_VARIANT", "DUPLICATE_ITEM"}


def test_legacy_projection_adds_aliases_without_changing_snapshot_hash():
    board = _finalize().boards[0]
    projected = csb.project_legacy_aliases(board)
    canonical = board.to_dict()
    assert projected["id"] is None
    assert projected["board_id"] is None
    assert projected["revision"] is None
    assert projected["can_save"] is False
    assert projected["can_shuffle"] is False
    assert projected["can_share"] is True
    assert projected["read_only"] is True
    assert projected["snapshot_hash"] == canonical["snapshot_hash"]
    assert "id" not in canonical


@pytest.mark.parametrize("flow", sorted(csb.FLOWS))
def test_flags_default_off_for_every_flow(monkeypatch, flow):
    monkeypatch.delenv(f"CANONICAL_STYLE_{flow.upper()}_SHADOW", raising=False)
    monkeypatch.delenv(f"CANONICAL_STYLE_{flow.upper()}_SERVE", raising=False)
    assert csb.shadow_enabled(flow) is False
    assert csb.serve_enabled(flow) is False


def test_shadow_and_serve_flags_are_independent(monkeypatch):
    monkeypatch.setenv("CANONICAL_STYLE_WARDROBE_SHADOW", "true")
    assert csb.shadow_enabled("wardrobe") is True
    assert csb.serve_enabled("wardrobe") is False


def test_cohort_assignment_is_sticky_per_user_and_flow():
    first = csb.cohort_bucket("user-1", "wardrobe")
    assert first == csb.cohort_bucket("user-1", "wardrobe")
    assert 0 <= first < 10_000
    assert first != csb.cohort_bucket("user-1", "capsule")


def test_shadow_off_is_true_noop(monkeypatch):
    monkeypatch.setenv("CANONICAL_STYLE_WARDROBE_SHADOW", "false")
    events = []
    assert csb.run_shadow_canonicalization(
        flow="wardrobe", user_id="user-1", candidates=[WARDROBE_LOOK],
        generation_request_id="request-1", scenario="wardrobe",
        request_fingerprint_value=_fingerprint()["request_fingerprint"], created_at=CREATED_AT,
        legacy_path_version="legacy-1", metric_sink=events.append,
    ) is None
    assert events == []


def test_shadow_event_contains_all_observability_versions(monkeypatch):
    monkeypatch.setenv("CANONICAL_STYLE_WARDROBE_SHADOW", "true")
    events = []
    result = csb.run_shadow_canonicalization(
        flow="wardrobe", user_id="user-1", candidates=[WARDROBE_LOOK],
        generation_request_id="request-1", scenario="wardrobe",
        request_fingerprint_value=_fingerprint()["request_fingerprint"], created_at=CREATED_AT,
        legacy_path_version="legacy-7", metric_sink=events.append,
    )
    assert result and len(result.boards) == 1
    event = events[0]
    assert {
        "flow", "shadow_flag_version", "canonical_schema_version", "finalizer_version",
        "serializer_version", "guard_version", "legacy_path_version",
        "request_fingerprint_version", "cohort_bucket",
    } <= event.keys()
    assert event["legacy_path_version"] == "legacy-7"


def test_shadow_runtime_trips_no_provider_network_or_generation_sentinel(monkeypatch):
    monkeypatch.setenv("CANONICAL_STYLE_WARDROBE_SHADOW", "true")
    calls = []

    def forbidden_call(*_args, **_kwargs):
        calls.append("external")
        raise AssertionError("shadow canonicalization attempted an external call")

    real_import = builtins.__import__
    forbidden_imports = (
        "appwrite", "qdrant_client", "google.genai", "google.cloud.aiplatform",
        "services.r2_storage", "services.appwrite_proxy", "services.image_generation",
        "services.catalog_png_generation_service", "brain.outfit_pipeline",
        "services.style_reasoning_engine", "brain.engines.capsule_engine",
    )

    def guarded_import(name, *args, **kwargs):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden_imports):
            return forbidden_call(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", forbidden_call)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden_call)
    monkeypatch.setattr(httpx.Client, "request", forbidden_call)
    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden_call)

    result = csb.run_shadow_canonicalization(
        flow="wardrobe", user_id="user-1", candidates=[WARDROBE_LOOK],
        generation_request_id="request-1", scenario="wardrobe",
        request_fingerprint_value=_fingerprint()["request_fingerprint"],
        created_at=CREATED_AT, legacy_path_version="legacy-runtime-purity",
        metric_sink=lambda _event: None,
    )
    assert result and len(result.boards) == 1
    assert calls == []


def test_shadow_module_has_no_provider_or_persistence_imports():
    source = open(csb.__file__, encoding="utf-8").read()
    forbidden = (
        "appwrite_proxy", "qdrant", "r2_storage", "gemini", "google.genai",
        "outfit_pipeline", "style_reasoning_engine", "capsule_engine",
        "constrained_outfit_builder",
    )
    assert not [name for name in forbidden if f"import {name}" in source or f"from {name}" in source]


@pytest.mark.parametrize("created_at", ["", "2026-01-01", "not-a-date"])
def test_created_at_must_be_explicit_and_timezone_aware(created_at):
    with pytest.raises(csb.CanonicalStyleBoardError) as exc:
        _finalize(created_at=created_at)
    assert exc.value.code in {"CREATED_AT_REQUIRED", "CREATED_AT_INVALID"}
