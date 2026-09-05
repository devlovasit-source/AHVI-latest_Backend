import json

import pytest

from services import wardrobe_intelligence_service as wis
from services.wardrobe_intelligence_service import (
    APPAREL_CLIMATE_KEYS,
    FOOTWEAR_CLIMATE_KEYS,
    apply_user_climate_edit,
    build_climate_profile,
    climate_unknown_tuple,
    derive_deterministic_climate_properties,
    extract_vision_observed_climate_properties,
    get_climate_property,
    merge_climate_profile,
    merge_climate_value,
    normalize_agent_climate_profile,
    normalize_material_value,
    user_confirmed_material_tuple,
)

GENUINE_VISION = {"label_source": "vision:gemini_multi"}


def _vision(**fields):
    return {**GENUINE_VISION, **fields}


# ---------------------------------------------------------------------------
# 1 & 2: light / heavy observable garment evidence (via genuine vision input)
# ---------------------------------------------------------------------------

def test_light_observable_apparel_evidence_requires_vision_provenance():
    item = {"name": "Sleeveless Top", "category": "Tops", "sub_category": "Sleeveless Top"}
    vision = _vision(name="Sleeveless Top", sub_category="Sleeveless Top")
    profile = build_climate_profile(item, vision_evidence=vision)
    assert profile["coverage_level"] == ["sleeveless", 2, "v"]


def test_heavy_observable_apparel_evidence_is_deterministic_from_category_words():
    item = {"name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"}
    profile = build_climate_profile(item)
    assert profile["insulation"] == ["likely_insulated", 1, "d"]
    assert profile["fabric_weight"] == ["heavy", 1, "d"]
    assert profile["layering_role"] == ["outer_layer", 1, "d"]


# ---------------------------------------------------------------------------
# 3: no useful evidence -> unknown, nothing fabricated
# ---------------------------------------------------------------------------

def test_no_evidence_yields_unknown_for_every_key():
    item = {"name": "Basic Tee", "category": "Tops", "sub_category": "T-Shirt"}
    profile = build_climate_profile(item)
    assert set(profile.keys()) == set(APPAREL_CLIMATE_KEYS)
    for key, value in profile.items():
        assert value == climate_unknown_tuple(), key


# ---------------------------------------------------------------------------
# 4, 8: explicit user material + re-edit (apparel)
# ---------------------------------------------------------------------------

def test_user_confirmed_material_tuple():
    assert user_confirmed_material_tuple("linen") == ["linen", 3, "u"]
    assert user_confirmed_material_tuple("  Linen  ") == ["linen", 3, "u"]
    assert user_confirmed_material_tuple("") is None
    assert user_confirmed_material_tuple(None) is None


def test_second_user_material_edit_replaces_first():
    profile = {"material": ["linen", 3, "u"]}
    # Must go through the dedicated user-edit override, not generic merge —
    # generic merge now keeps ties on the existing value (see Final
    # Correction tests below).
    updated = apply_user_climate_edit(profile, "material", user_confirmed_material_tuple("cotton"))
    assert updated["material"] == ["cotton", 3, "u"]


# ---------------------------------------------------------------------------
# 5, 6, 7, 27: user material survives enrichment / agent / backfill / re-run
# ---------------------------------------------------------------------------

def test_user_material_survives_deterministic_and_vision_reenrichment():
    existing = {"material": ["linen", 3, "u"]}
    item = {"name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"}
    profile = build_climate_profile(item, existing_profile=existing)
    assert profile["material"] == ["linen", 3, "u"]
    assert profile["insulation"] == ["likely_insulated", 1, "d"]


def test_user_material_survives_agent_enrichment():
    existing = {"material": ["linen", 3, "u"]}
    item = {"name": "Basic Tee", "category": "Tops", "sub_category": "T-Shirt"}
    profile = build_climate_profile(item, existing_profile=existing)
    agent_contribution = normalize_agent_climate_profile({"material": "polyester", "fit": "loose"})
    # Agent may never touch material at all.
    assert "material" not in agent_contribution
    merged = merge_climate_profile(profile, agent_contribution)
    assert merged["material"] == ["linen", 3, "u"]
    assert merged["fit"] == ["loose", 1, "m"]


def test_repeated_enrichment_is_idempotent():
    item = {"name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"}
    first = build_climate_profile(item)
    second = build_climate_profile(item, existing_profile=first)
    assert first == second


# ---------------------------------------------------------------------------
# CORRECTION 1 — material must work for footwear too
# ---------------------------------------------------------------------------

def test_footwear_key_set_includes_material():
    assert "material" in FOOTWEAR_CLIMATE_KEYS


def test_footwear_material_user_edit_persists_in_profile():
    item = {"name": "Suede Loafers", "category": "Footwear", "sub_category": "Loafers"}
    profile = build_climate_profile(item)
    merged = merge_climate_profile(profile, {"material": user_confirmed_material_tuple("suede")})
    assert merged["material"] == ["suede", 3, "u"]


def test_footwear_user_material_survives_deterministic_enrichment():
    existing = {"material": ["leather", 3, "u"]}
    item = {"name": "Leather Boots", "category": "Footwear", "sub_category": "Boots"}
    profile = build_climate_profile(item, existing_profile=existing)
    assert profile["material"] == ["leather", 3, "u"]
    assert profile["footwear_type"] == ["boot", 1, "d"]
    assert profile["coverage"] == ["closed", 1, "d"]


def test_footwear_user_material_survives_agent_enrichment():
    existing = {"material": ["leather", 3, "u"]}
    item = {"name": "Leather Boots", "category": "Footwear", "sub_category": "Boots"}
    profile = build_climate_profile(item, existing_profile=existing)
    agent_contribution = normalize_agent_climate_profile({"material": "canvas", "activity_affinity": "athletic"})
    assert "material" not in agent_contribution
    merged = merge_climate_profile(profile, agent_contribution)
    assert merged["material"] == ["leather", 3, "u"]
    assert merged["activity_affinity"] == ["athletic", 1, "m"]


def test_footwear_user_material_survives_backfill(monkeypatch):
    from scripts import backfill_style_metadata as backfill

    monkeypatch.setattr(backfill, "fetch_existing_climate_profile", lambda item_id: {"material": ["canvas", 3, "u"]})
    doc = {"$id": "shoe_1", "userId": "user_1", "name": "Canvas Sneakers", "category": "Footwear", "sub_category": "Sneakers"}
    meta = backfill._style_metadata_for_doc(doc, user_id="user_1", use_agent=False)
    assert meta["climate_profile"]["material"] == ["canvas", 3, "u"]
    assert meta["climate_profile"]["footwear_type"] == ["sneaker", 1, "d"]


def test_second_footwear_user_material_edit_replaces_first():
    existing = {"material": ["leather", 3, "u"]}
    item = {"name": "Boots", "category": "Footwear", "sub_category": "Boots"}
    profile = build_climate_profile(item, existing_profile=existing)
    merged = apply_user_climate_edit(profile, "material", user_confirmed_material_tuple("suede"))
    assert merged["material"] == ["suede", 3, "u"]


def test_update_labels_persists_footwear_user_material(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(
        persistence,
        "_fetch_document",
        lambda item_id, **kwargs: (
            {"$id": item_id, "userId": "user_1", "name": "Suede Loafers", "category": "Footwear", "sub_category": "Loafers"},
            "outfits",
            "db",
        ),
    )
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})

    captured = {}

    def fake_upsert(*, item_id, user_id, item_payload, explicit_material=None):
        captured["explicit_material"] = explicit_material
        return "updated"

    monkeypatch.setattr(persistence, "_upsert_style_metadata", fake_upsert)

    result = persistence.update_item_labels(
        user_id="user_1",
        item_id="item_1",
        material="Suede",
        override_collection_id="outfits",
        override_database_id="db",
    )
    assert result["success"] is True
    assert captured["explicit_material"] == "Suede"


def test_style_metadata_payload_writes_footwear_climate_profile_material(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})

    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Suede Loafers", "category": "Footwear", "sub_category": "Loafers"},
        explicit_material="suede",
    )
    meta = json.loads(payload["style_metadata"])
    assert meta["climate_profile"]["material"] == ["suede", 3, "u"]


# ---------------------------------------------------------------------------
# 9, 10: authority order is independent of application order, and stronger
# evidence can never be downgraded in value/source/confidence.
# ---------------------------------------------------------------------------

def test_authority_order_independent_of_application_order():
    deterministic = {"coverage_level": ["sleeveless", 1, "d"]}
    vision = {"coverage_level": ["sleeveless", 2, "v"]}

    a = merge_climate_profile(merge_climate_profile({}, deterministic), vision)
    b = merge_climate_profile(merge_climate_profile({}, vision), deterministic)

    assert a["coverage_level"] == ["sleeveless", 2, "v"]
    assert b["coverage_level"] == ["sleeveless", 2, "v"]
    assert a == b


def test_lower_authority_cannot_change_value_source_or_confidence():
    existing = ["sleeveless", 2, "v"]
    weaker = ["full_sleeve", 1, "d"]
    result = merge_climate_value(existing, weaker)
    assert result == existing


def test_model_inferred_cannot_override_anything_stronger():
    profile = {"fit": ["loose", 2, "v"]}
    agent = {"fit": ["fitted", 1, "m"]}
    merged = merge_climate_profile(profile, agent)
    assert merged["fit"] == ["loose", 2, "v"]


# ---------------------------------------------------------------------------
# FINAL CORRECTION — generic equal-authority merge must NOT be last-write-
# wins. Only an explicit user edit (apply_user_climate_edit) may replace an
# equal-authority value.
# ---------------------------------------------------------------------------

def test_equal_authority_vision_vs_vision_keeps_existing():
    existing = {"fabric_weight": ["light", 2, "v"]}
    incoming = {"fabric_weight": ["medium", 2, "v"]}
    merged = merge_climate_profile(existing, incoming)
    assert merged["fabric_weight"] == ["light", 2, "v"]


def test_equal_authority_merge_preserves_previously_accepted_evidence_either_order():
    a = {"fabric_weight": ["light", 2, "v"]}
    b = {"fabric_weight": ["medium", 2, "v"]}
    # Whichever was accepted FIRST (i.e. is already `existing`) survives —
    # this is not a symmetric/commutative merge like the cross-authority
    # case; it specifically protects previously-accepted equal-tier
    # evidence from being clobbered by a later run.
    assert merge_climate_profile(a, b)["fabric_weight"] == ["light", 2, "v"]
    assert merge_climate_profile(b, a)["fabric_weight"] == ["medium", 2, "v"]


def test_equal_authority_deterministic_vs_deterministic_keeps_existing():
    existing = {"fabric_weight": ["light", 1, "d"]}
    incoming = {"fabric_weight": ["heavy", 1, "d"]}
    merged = merge_climate_profile(existing, incoming)
    assert merged["fabric_weight"] == ["light", 1, "d"]


def test_equal_authority_model_vs_model_keeps_existing():
    existing = {"fit": ["x", 1, "m"]}
    incoming = {"fit": ["y", 1, "m"]}
    merged = merge_climate_profile(existing, incoming)
    assert merged["fit"] == ["x", 1, "m"]


def test_higher_authority_still_replaces_lower_after_correction():
    assert merge_climate_value(["a", 1, "m"], ["b", 1, "d"]) == ["b", 1, "d"]
    assert merge_climate_value(["a", 1, "d"], ["b", 2, "v"]) == ["b", 2, "v"]
    assert merge_climate_value(["a", 2, "v"], ["b", 3, "u"]) == ["b", 3, "u"]


def test_lower_authority_never_replaces_higher_after_correction():
    assert merge_climate_value(["a", 1, "d"], ["b", 1, "m"]) == ["a", 1, "d"]
    assert merge_climate_value(["a", 2, "v"], ["b", 1, "d"]) == ["a", 2, "v"]
    assert merge_climate_value(["a", 3, "u"], ["b", 2, "v"]) == ["a", 3, "u"]


def test_explicit_user_edit_linen_to_cotton_wins_via_dedicated_override():
    profile = {"material": ["linen", 3, "u"]}
    updated = apply_user_climate_edit(profile, "material", user_confirmed_material_tuple("cotton"))
    assert updated["material"] == ["cotton", 3, "u"]


def test_automated_u_looking_tuple_cannot_exploit_user_override_via_generic_merge():
    # A tuple that merely LOOKS like a user edit (source "u") but arrives
    # through the generic/automated merge path — not apply_user_climate_edit
    # — must not be able to displace an existing equal-authority user value.
    existing = {"material": ["linen", 3, "u"]}
    spoofed_candidate = {"material": ["cotton", 3, "u"]}
    merged = merge_climate_profile(existing, spoofed_candidate)
    assert merged["material"] == ["linen", 3, "u"]


def test_apply_user_climate_edit_rejects_non_user_sourced_tuples():
    profile = {"material": ["linen", 3, "u"]}
    # A non-"u" tuple can never use the override path, no matter how it's shaped.
    result = apply_user_climate_edit(profile, "material", ["cotton", 3, "d"])
    assert result["material"] == ["linen", 3, "u"]


def test_repeated_backfill_remains_stable_under_equal_authority_rule(monkeypatch):
    from scripts import backfill_style_metadata as backfill

    doc = {"$id": "item_1", "userId": "user_1", "name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"}

    monkeypatch.setattr(backfill, "fetch_existing_climate_profile", lambda item_id: {})
    first = backfill._style_metadata_for_doc(doc, user_id="user_1", use_agent=False)["climate_profile"]

    monkeypatch.setattr(backfill, "fetch_existing_climate_profile", lambda item_id: first)
    second = backfill._style_metadata_for_doc(doc, user_id="user_1", use_agent=False)["climate_profile"]

    assert first == second


def test_repeated_agent_enrichment_cannot_oscillate_equal_authority_model_values():
    existing_profile = merge_climate_profile({}, {"fit": ["loose", 1, "m"]})
    agent_round_two = normalize_agent_climate_profile({"fit": "fitted"})
    merged = merge_climate_profile(existing_profile, agent_round_two)
    assert merged["fit"] == ["loose", 1, "m"]
    # A third round with yet another guess still can't move it.
    agent_round_three = normalize_agent_climate_profile({"fit": "oversized"})
    merged_again = merge_climate_profile(merged, agent_round_three)
    assert merged_again["fit"] == ["loose", 1, "m"]


# ---------------------------------------------------------------------------
# CORRECTION 2 — vision_observed requires positively-established provenance
# ---------------------------------------------------------------------------

def test_correction2_a_sleeveless_name_without_provenance_is_deterministic():
    item = {"name": "Sleeveless Top", "category": "Tops", "sub_category": "Sleeveless Top"}
    profile = build_climate_profile(item)  # no vision_evidence at all
    assert profile["coverage_level"] == ["sleeveless", 1, "d"]


def test_correction2_b_same_evidence_via_genuine_vision_result_is_vision_observed():
    item = {"name": "Sleeveless Top", "category": "Tops", "sub_category": "Sleeveless Top"}
    vision = _vision(name="Sleeveless Top", sub_category="Sleeveless Top")
    profile = build_climate_profile(item, vision_evidence=vision)
    assert profile["coverage_level"] == ["sleeveless", 2, "v"]


def test_correction2_c_backfill_quilted_jacket_name_only_is_deterministic(monkeypatch):
    from scripts import backfill_style_metadata as backfill

    monkeypatch.setattr(backfill, "fetch_existing_climate_profile", lambda item_id: {})
    # Historical outfits rows never carry a `vision_result` field, so this
    # must land as deterministic_derived regardless of what the name says.
    doc = {"$id": "item_x", "userId": "user_1", "name": "Quilted Jacket", "category": "Outerwear", "sub_category": "Quilted Jacket"}
    meta = backfill._style_metadata_for_doc(doc, user_id="user_1", use_agent=False)
    assert meta["climate_profile"]["insulation"] == ["likely_insulated", 1, "d"]


def test_correction2_d_user_edited_name_never_becomes_vision_observed():
    # Simulates a user renaming an item to include "sleeveless" — item_payload
    # text alone (no vision_result) must never be promoted to "v".
    item = {"name": "Sleeveless (user renamed)", "category": "Tops", "sub_category": "Top"}
    profile = build_climate_profile(item, vision_evidence=None)
    assert profile["coverage_level"] == ["sleeveless", 1, "d"]


def test_vision_extraction_requires_dict_with_vision_marker():
    footwear_blob_item = {"name": "Open Toe Sandals", "sub_category": "Sandals"}
    # No provenance marker at all -> nothing extracted, regardless of text.
    assert extract_vision_observed_climate_properties(footwear_blob_item, footwear=True) == {}
    assert extract_vision_observed_climate_properties(None, footwear=True) == {}
    assert extract_vision_observed_climate_properties({}, footwear=True) == {}
    # Non-vision source marker (e.g. a heuristic/text-only fallback) also
    # must not qualify.
    heuristic = {"name": "Open Toe Sandals", "sub_category": "Sandals", "label_source": "heuristic"}
    assert extract_vision_observed_climate_properties(heuristic, footwear=True) == {}


def test_vision_extraction_accepts_genuine_gemini_vision_marker():
    vision = {"name": "Open Toe Sandals", "sub_category": "Sandals", "label_source": "vision:gemini_multi"}
    result = extract_vision_observed_climate_properties(vision, footwear=True)
    assert result["coverage"] == ["open_toe", 2, "v"]


def test_vision_extraction_ignores_non_literal_prose_even_with_provenance():
    vision = _vision(name="Great for summer, moisture wicking, waterproof jacket")
    result = extract_vision_observed_climate_properties(vision, footwear=False)
    assert result == {}


def test_agent_output_always_demoted_to_model_inferred():
    raw = {"insulation": ["likely_insulated", 3, "v"], "breathability": "likely_breathable"}
    normalized = normalize_agent_climate_profile(raw)
    assert normalized["insulation"] == ["likely_insulated", 1, "m"]
    assert normalized["breathability"] == ["likely_breathable", 1, "m"]


def test_persistence_only_treats_item_payload_vision_result_key_as_vision_evidence(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})

    # No vision_result key at all in item_payload -> deterministic only.
    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Sleeveless Top", "category": "Tops", "sub_category": "Sleeveless Top"},
    )
    meta = json.loads(payload["style_metadata"])
    assert meta["climate_profile"]["coverage_level"] == ["sleeveless", 1, "d"]

    # Genuine vision_result present -> vision_observed.
    payload2 = persistence._style_metadata_payload(
        item_id="item_2",
        user_id="user_1",
        item_payload={
            "name": "Sleeveless Top",
            "category": "Tops",
            "sub_category": "Sleeveless Top",
            "vision_result": {
                "name": "Sleeveless Top",
                "sub_category": "Sleeveless Top",
                "label_source": "vision:gemini_multi",
            },
        },
    )
    meta2 = json.loads(payload2["style_metadata"])
    assert meta2["climate_profile"]["coverage_level"] == ["sleeveless", 2, "v"]


# ---------------------------------------------------------------------------
# CORRECTION 3 — no breathability-from-weight-alone inference
# ---------------------------------------------------------------------------

def test_apparel_breathability_never_derived_from_fabric_weight_alone():
    vision = _vision(name="Lightweight Jacket")
    item = {"name": "Lightweight Jacket", "category": "Outerwear"}
    profile = build_climate_profile(item, vision_evidence=vision)
    assert profile["fabric_weight"] == ["light", 2, "v"]
    # No breathability rule exists for apparel in V1 -> stays unknown.
    assert profile["breathability"] == climate_unknown_tuple()


def test_apparel_breathability_stays_unknown_with_no_construction_evidence():
    item = {"name": "Basic Tee", "category": "Tops"}
    profile = build_climate_profile(item)
    assert profile["breathability"] == climate_unknown_tuple()


def test_footwear_breathability_from_open_construction_still_allowed():
    # This is a POSITIVE case: open-toe/strap construction is distinct,
    # meaningful construction evidence — not "weight alone" — so it's kept.
    item = {"name": "Rubber Sandals", "category": "Footwear", "sub_category": "Sandals"}
    profile = build_climate_profile(item)
    assert profile["breathability"] == ["likely_breathable", 1, "d"]


def test_footwear_breathability_unknown_without_construction_evidence():
    item = {"name": "Formal Shoes", "category": "Footwear", "sub_category": "Oxfords"}
    profile = build_climate_profile(item)
    assert profile["breathability"] == climate_unknown_tuple()


# ---------------------------------------------------------------------------
# 18: apparel vs footwear shapes differ (material is now shared)
# ---------------------------------------------------------------------------

def test_apparel_and_footwear_key_shapes_differ_but_share_material():
    apparel = build_climate_profile({"name": "Tee", "category": "Tops"})
    footwear = build_climate_profile({"name": "Sneakers", "category": "Footwear", "sub_category": "Sneakers"})
    assert set(apparel.keys()) == set(APPAREL_CLIMATE_KEYS)
    assert set(footwear.keys()) == set(FOOTWEAR_CLIMATE_KEYS)
    assert "material" in apparel and "material" in footwear
    assert "footwear_type" not in apparel
    assert "fit" not in footwear


# ---------------------------------------------------------------------------
# 19: no date/month/weather/location dependency (static inspection)
# ---------------------------------------------------------------------------

def test_producer_source_has_no_forbidden_runtime_signals():
    import inspect

    source = inspect.getsource(wis)
    forbidden = ["datetime.now", "date.today", "get_current_weather", "user_location", "requester_region"]
    for token in forbidden:
        assert token not in source


# ---------------------------------------------------------------------------
# 26: all-unknown profile is not a positive suitability signal
# ---------------------------------------------------------------------------

def test_all_unknown_profile_has_zero_confidence_everywhere():
    item = {"name": "Mystery Item", "category": "Tops"}
    profile = build_climate_profile(item)
    for value in profile.values():
        assert value[1] == 0
        assert value[2] == "x"


# ---------------------------------------------------------------------------
# 16, 28: centralized unknown semantics / legacy records without a profile
# ---------------------------------------------------------------------------

def test_get_climate_property_treats_missing_profile_as_unknown():
    assert get_climate_property(None, "material") == climate_unknown_tuple()
    assert get_climate_property({}, "material") == climate_unknown_tuple()
    assert get_climate_property({"material": ["linen", 3, "u"]}, "fabric_weight") == climate_unknown_tuple()
    assert get_climate_property({"material": ["linen", 3, "u"]}, "material") == ["linen", 3, "u"]


# ---------------------------------------------------------------------------
# 30: material normalization is conservative (case/whitespace only)
# ---------------------------------------------------------------------------

def test_material_normalization_is_whitespace_and_case_only():
    assert normalize_material_value("  Cotton   Blend  ") == "cotton blend"
    assert normalize_material_value("LINEN") == "linen"


# ---------------------------------------------------------------------------
# End-to-end persistence wiring: bug fix + survival + agent-optional
# ---------------------------------------------------------------------------

def test_update_labels_persists_user_material_via_climate_profile(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(
        persistence,
        "_fetch_document",
        lambda item_id, **kwargs: (
            {"$id": item_id, "userId": "user_1", "name": "Linen Shirt", "category": "Tops"},
            "outfits",
            "db",
        ),
    )
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})

    captured = {}

    def fake_upsert(*, item_id, user_id, item_payload, explicit_material=None):
        captured["explicit_material"] = explicit_material
        captured["item_payload"] = item_payload
        return "updated"

    monkeypatch.setattr(persistence, "_upsert_style_metadata", fake_upsert)

    result = persistence.update_item_labels(
        user_id="user_1",
        item_id="item_1",
        material="Linen",
        override_collection_id="outfits",
        override_database_id="db",
    )

    assert result["success"] is True
    assert captured["explicit_material"] == "Linen"


def test_style_metadata_payload_writes_climate_profile_material(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})

    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Linen Shirt", "category": "Tops"},
        explicit_material="linen",
    )
    meta = json.loads(payload["style_metadata"])
    assert meta["climate_profile"]["material"] == ["linen", 3, "u"]
    assert meta["climate_profile_version"] == "v1"


def test_style_metadata_payload_preserves_prior_user_material_on_resave(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(
        persistence,
        "fetch_existing_climate_profile",
        lambda item_id: {"material": ["linen", 3, "u"]},
    )

    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"},
    )
    meta = json.loads(payload["style_metadata"])
    assert meta["climate_profile"]["material"] == ["linen", 3, "u"]
    assert meta["climate_profile"]["insulation"] == ["likely_insulated", 1, "d"]


# ---------------------------------------------------------------------------
# 12, 13, 14: agent timeout/exception/disabled never break persistence or
# climate_profile availability.
# ---------------------------------------------------------------------------

def test_agent_timeout_does_not_fail_persistence(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    def boom(**kwargs):
        raise TimeoutError("agent timed out")

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: True)
    monkeypatch.setattr(persistence, "_agent_validate_metadata_sync", boom)
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})

    payload = persistence._style_metadata_payload(
        item_id="item_1", user_id="user_1", item_payload={"name": "Tee", "category": "Tops"}
    )
    meta = json.loads(payload["style_metadata"])
    assert "climate_profile" in meta


def test_agent_exception_does_not_fail_persistence(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: True)
    monkeypatch.setattr(persistence, "_agent_validate_metadata_sync", boom)
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})

    payload = persistence._style_metadata_payload(
        item_id="item_1", user_id="user_1", item_payload={"name": "Tee", "category": "Tops"}
    )
    meta = json.loads(payload["style_metadata"])
    assert "climate_profile" in meta


def test_agent_disabled_climate_profile_still_derives(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})

    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"},
    )
    meta = json.loads(payload["style_metadata"])
    assert meta["climate_profile"]["insulation"] == ["likely_insulated", 1, "d"]


# ---------------------------------------------------------------------------
# 15, 29: legacy fields untouched, new metadata round-trips
# ---------------------------------------------------------------------------

def test_legacy_enrichment_fields_unchanged_alongside_climate_profile(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(persistence, "fetch_existing_climate_profile", lambda item_id: {})

    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={"name": "Leather Belt", "category": "Accessories"},
    )
    meta = json.loads(payload["style_metadata"])
    assert meta["subcategory"] == "Belt"
    assert meta["occasion_affinity"] == ["daily", "office", "date", "coffee_run"]
    assert "climate_profile" in meta


def test_climate_profile_round_trips_through_json():
    item = {"name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"}
    profile = build_climate_profile(item)
    round_tripped = json.loads(json.dumps(profile))
    assert round_tripped == profile


# ---------------------------------------------------------------------------
# 17: soft size limit sanity — a full climate_profile is tiny.
# ---------------------------------------------------------------------------

def test_climate_profile_serialized_size_is_small():
    item = {"name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"}
    profile = build_climate_profile(item, existing_profile={"material": ["linen", 3, "u"]})
    assert len(json.dumps(profile, separators=(",", ":"))) < 500


# ---------------------------------------------------------------------------
# 20-25: backfill tiers
# ---------------------------------------------------------------------------

def test_backfill_tier_a_preserves_existing_user_material(monkeypatch):
    from scripts import backfill_style_metadata as backfill

    monkeypatch.setattr(backfill, "fetch_existing_climate_profile", lambda item_id: {"material": ["linen", 3, "u"]})

    doc = {"$id": "item_1", "userId": "user_1", "name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"}
    meta = backfill._style_metadata_for_doc(doc, user_id="user_1", use_agent=False)
    assert meta["climate_profile"]["material"] == ["linen", 3, "u"]
    assert meta["climate_profile"]["insulation"] == ["likely_insulated", 1, "d"]


def test_backfill_tier_b_only_fires_with_provenanced_vision_result(monkeypatch):
    from scripts import backfill_style_metadata as backfill

    monkeypatch.setattr(backfill, "fetch_existing_climate_profile", lambda item_id: {})

    # No vision_result on the doc (the normal case) -> Tier B stays empty,
    # Tier C (deterministic) covers it instead.
    doc = {"$id": "item_2", "userId": "user_1", "name": "Sleeveless Top", "category": "Tops", "sub_category": "Sleeveless Top"}
    meta = backfill._style_metadata_for_doc(doc, user_id="user_1", use_agent=False)
    assert meta["climate_profile"]["coverage_level"] == ["sleeveless", 1, "d"]

    # A doc that DOES carry a genuinely-provenanced vision_result -> Tier B.
    doc_with_vision = dict(doc, vision_result={
        "name": "Sleeveless Top", "sub_category": "Sleeveless Top", "label_source": "vision:gemini_multi",
    })
    meta2 = backfill._style_metadata_for_doc(doc_with_vision, user_id="user_1", use_agent=False)
    assert meta2["climate_profile"]["coverage_level"] == ["sleeveless", 2, "v"]


def test_backfill_tier_c_deterministic_low_confidence_never_fabricates_material(monkeypatch):
    from scripts import backfill_style_metadata as backfill

    monkeypatch.setattr(backfill, "fetch_existing_climate_profile", lambda item_id: {})

    doc = {"$id": "item_3", "userId": "user_1", "name": "Blue Puffer Jacket", "category": "Outerwear", "sub_category": "Puffer Jacket"}
    meta = backfill._style_metadata_for_doc(doc, user_id="user_1", use_agent=False)
    climate = meta["climate_profile"]
    assert climate["material"] == ["unknown", 0, "x"]
    for key in ("insulation", "fabric_weight", "layering_role"):
        assert climate[key][2] == "d"
        assert climate[key][1] < 3


def test_backfill_tier_d_insufficient_evidence_is_unknown(monkeypatch):
    from scripts import backfill_style_metadata as backfill

    monkeypatch.setattr(backfill, "fetch_existing_climate_profile", lambda item_id: {})

    doc = {"$id": "item_4", "userId": "user_1", "name": "Item", "category": "Tops"}
    meta = backfill._style_metadata_for_doc(doc, user_id="user_1", use_agent=False)
    for value in meta["climate_profile"].values():
        assert value == ["unknown", 0, "x"]


def test_backfill_default_run_makes_no_agent_calls(monkeypatch):
    from scripts import backfill_style_metadata as backfill

    called = {"agent": False}

    def fake_agent(**kwargs):
        called["agent"] = True
        return {}

    monkeypatch.setattr(backfill, "validate_wardrobe_metadata_sync", fake_agent)
    monkeypatch.setattr(backfill, "fetch_existing_climate_profile", lambda item_id: {})

    class FakeProxy:
        def list_documents(self, resource, limit=100, offset=0, return_meta=False, **kwargs):
            if offset:
                return {"documents": [], "meta": {"has_more": False}}
            return {
                "documents": [
                    {"$id": "item_1", "userId": "user_1", "name": "Leather Belt", "category": "Accessories"}
                ],
                "meta": {"has_more": False},
            }

    monkeypatch.setattr(backfill, "AppwriteProxy", lambda: FakeProxy())
    monkeypatch.setattr(
        backfill,
        "upsert_wardrobe_style_metadata_sync",
        lambda user_id, doc_id, style_meta: {"status": "updated"},
    )

    backfill.run(dry_run=False, all_users=True)
    assert called["agent"] is False


def test_backfill_use_agent_is_explicit_opt_in(monkeypatch):
    from scripts import backfill_style_metadata as backfill

    called = {"agent": False}

    def fake_agent(**kwargs):
        called["agent"] = True
        return {"confidence": 0.9, "climate_profile": {}}

    monkeypatch.setattr(backfill, "validate_wardrobe_metadata_sync", fake_agent)
    monkeypatch.setattr(backfill, "fetch_existing_climate_profile", lambda item_id: {})

    class FakeProxy:
        def list_documents(self, resource, limit=100, offset=0, return_meta=False, **kwargs):
            if offset:
                return {"documents": [], "meta": {"has_more": False}}
            return {
                "documents": [
                    {"$id": "item_1", "userId": "user_1", "name": "Leather Belt", "category": "Accessories"}
                ],
                "meta": {"has_more": False},
            }

    monkeypatch.setattr(backfill, "AppwriteProxy", lambda: FakeProxy())
    monkeypatch.setattr(
        backfill,
        "upsert_wardrobe_style_metadata_sync",
        lambda user_id, doc_id, style_meta: {"status": "updated"},
    )

    backfill.run(dry_run=False, all_users=True, use_agent=True)
    assert called["agent"] is True
def test_style_metadata_payload_writes_physical_garment_observations(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(
        persistence,
        "fetch_existing_climate_profile",
        lambda item_id: {},
    )

    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={
            "name": "Trousers",
            "category": "Bottoms",
            "sub_category": "Trousers",
            "physical_garment_observations": {
                "fabric_weight": {
                    "value": "medium",
                    "confidence": 0.90,
                },
                "fabric_structure": {
                    "value": "woven",
                    "confidence": 0.90,
                },
                "fit": {
                    "value": "regular",
                    "confidence": 0.90,
                },
                "drape": {
                    "value": "structured",
                    "confidence": 0.70,
                },
                "coverage_level": {
                    "value": "full_length",
                    "confidence": 0.95,
                },
                "lining": {
                    "value": "unknown",
                    "confidence": 0.0,
                },
                "surface_texture": {
                    "value": "smooth",
                    "confidence": 0.90,
                },
                "material_family_candidates": [
                    {
                        "value": "cotton_like",
                        "confidence": 0.80,
                    }
                ],
            },
        },
    )

    meta = json.loads(payload["style_metadata"])
    climate = meta["climate_profile"]

    assert climate["fabric_weight"] == ["medium", 2, "v"]
    assert climate["fabric_structure"] == ["woven", 2, "v"]
    assert climate["fit"] == ["regular", 2, "v"]
    assert climate["coverage_level"] == ["full_length", 2, "v"]


def test_physical_observations_never_override_user_material(monkeypatch):
    import services.wardrobe_persistence_service as persistence

    monkeypatch.setattr(persistence, "_agent_metadata_enabled", lambda: False)
    monkeypatch.setattr(
        persistence,
        "fetch_existing_climate_profile",
        lambda item_id: {
            "material": ["linen", 3, "u"],
        },
    )

    payload = persistence._style_metadata_payload(
        item_id="item_1",
        user_id="user_1",
        item_payload={
            "name": "Linen Trousers",
            "category": "Bottoms",
            "sub_category": "Trousers",
            "physical_garment_observations": {
                "fabric_weight": {
                    "value": "medium",
                    "confidence": 0.90,
                },
                "fabric_structure": {
                    "value": "woven",
                    "confidence": 0.90,
                },
                "fit": {
                    "value": "regular",
                    "confidence": 0.90,
                },
                "drape": {
                    "value": "structured",
                    "confidence": 0.80,
                },
                "coverage_level": {
                    "value": "full_length",
                    "confidence": 0.95,
                },
                "lining": {
                    "value": "unknown",
                    "confidence": 0.0,
                },
                "surface_texture": {
                    "value": "smooth",
                    "confidence": 0.90,
                },
                "material_family_candidates": [
                    {
                        "value": "cotton_like",
                        "confidence": 0.80,
                    }
                ],
            },
        },
    )

    meta = json.loads(payload["style_metadata"])
    climate = meta["climate_profile"]

    assert climate["material"] == ["linen", 3, "u"]
    assert climate["fabric_weight"] == ["medium", 2, "v"]
    assert climate["fabric_structure"] == ["woven", 2, "v"]
