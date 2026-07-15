"""Regression coverage for trusted Style-asset anchors.

These fixtures model the reviewed canaries using only server metadata.  The
tests deliberately exercise the canonical anchor gate directly so a caller
cannot make a decision from a name, URL, or client-supplied role.
"""

from services.style_anchor_compatibility import evaluate_style_asset_anchor


def _asset(asset_id, role, *, name, occasions=("daily",), safety_tags=(), **extra):
    row = {
        "asset_id": asset_id,
        "name": name,
        "source": "style_asset",
        "source_origin": "curated",
        "role": role,
        "category": role,
        "sub_category": role,
        "gender_fit": "unisex",
        "board_image_url": f"https://example.test/{asset_id}.png",
        "colors": ["black"],
        "archetypes": ["minimal"],
        "occasion_families": list(occasions),
        "formality": 3,
        "traits": ["clean"],
        "weather_tags": [],
        "safety_tags": list(safety_tags),
    }
    row.update(extra)
    return row


def _context(occasion="daily", *, temperature=None, tags=None):
    weather = {}
    if temperature is not None:
        weather["temperature_c"] = temperature
    if tags is not None:
        weather["weather_tags"] = list(tags)
    return {
        "canonical_occasion": occasion,
        "style_gender": "unisex",
        "weather_context": weather,
    }


def test_canary_black_shirt_is_trusted_and_professional_safe():
    shirt = _asset(
        "mens_assets_tops_blackshirt",
        "top",
        name="Black shirt",
        occasions=("daily", "office", "client_meeting"),
        professional_safe=True,
        client_meeting_score=0.9,
        professionalism_score=0.9,
    )
    result = evaluate_style_asset_anchor(shirt, _context("client_meeting"))
    assert result["allowed"] is True, result
    assert result["decisions"]["source"]["evidence_source"] == "server_repository"


def test_canary_striped_tee_is_blocked_for_client_meeting():
    tee = _asset(
        "meghna_female_top_black_and_white_stripe_tshirt",
        "top",
        name="Black and white striped T-shirt",
        occasions=("daily",),
        professional_safe=False,
        safety_tags=("not_client_meeting",),
    )
    result = evaluate_style_asset_anchor(tee, _context("client_meeting"))
    assert result["allowed"] is False
    assert result["failed_gate"] == "professional"
    assert result["decisions"]["professional"]["reason_code"] == "professional_safe_false"


def test_canary_plaid_scarf_is_accessory_and_cold_compatible():
    scarf = _asset(
        "mens_assets_outerwear_blackplaidscarf",
        "accessory",
        name="Black plaid scarf",
        occasions=("daily",),
        weather_tags=("cold",),
        temperature_min_c=-10,
        temperature_max_c=16,
        fabric_weight="heavy",
        layering_suitability=0.8,
    )
    result = evaluate_style_asset_anchor(scarf, _context("daily", temperature=5, tags=("cold",)))
    assert result["allowed"] is True, result
    assert result["item"]["role"] == "accessory"
    assert result["decisions"]["weather"]["score"] > 0


def test_canary_kurta_respects_cultural_professional_evidence():
    kurta = _asset(
        "mens_assets_festive_sets_bluekurta",
        "top",
        name="Blue kurta",
        occasions=("festive", "cultural"),
        safety_tags=("cultural_professional",),
        professional_safe=True,
        professionalism_score=0.8,
    )
    result = evaluate_style_asset_anchor(kurta, _context("office"))
    assert result["allowed"] is True, result
    assert result["decisions"]["professional"]["reason_code"] == "canonical_positive"


def test_rejected_style_asset_cannot_be_fixed_by_anchor_context():
    rejected = _asset(
        "server-asset",
        "top",
        name="Server shirt",
        metadata_status="rejected",
    )
    result = evaluate_style_asset_anchor(rejected, _context())
    assert result["allowed"] is False
    assert result["failed_gate"] == "readiness"
    assert result["reason_code"] == "metadata_rejected"
