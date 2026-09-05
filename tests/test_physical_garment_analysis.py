import json

import pytest

from services import physical_garment_analysis_service as pgas


def _valid_output(**overrides):
    result = {
        "fabric_weight": {"value": "medium", "confidence": 0.90},
        "fabric_structure": {"value": "woven", "confidence": 0.90},
        "fit": {"value": "regular", "confidence": 0.85},
        "drape": {"value": "soft", "confidence": 0.80},
        "coverage_level": {"value": "full_sleeve", "confidence": 0.95},
        "lining": {"value": "likely_unlined", "confidence": 0.75},
        "surface_texture": {"value": "smooth", "confidence": 0.90},
        "material_family_candidates": [
            {"value": "cotton_like", "confidence": 0.70},
        ],
    }
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# Strict structured-output validation
# ---------------------------------------------------------------------------


def test_valid_analysis_output_is_accepted():
    result = pgas.validate_analysis_output(_valid_output())

    assert result is not None
    assert result["fabric_weight"] == {
        "value": "medium",
        "confidence": 0.90,
    }


def test_invalid_observation_value_is_rejected():
    raw = _valid_output(
        fabric_weight={"value": "very_heavy", "confidence": 0.95}
    )

    assert pgas.validate_analysis_output(raw) is None


def test_invalid_confidence_is_rejected():
    raw = _valid_output(
        fabric_weight={"value": "heavy", "confidence": 1.5}
    )

    assert pgas.validate_analysis_output(raw) is None


def test_missing_required_field_is_rejected():
    raw = _valid_output()
    del raw["fit"]

    assert pgas.validate_analysis_output(raw) is None


def test_extra_field_is_rejected():
    raw = _valid_output()
    raw["exact_material"] = "100% cotton"

    assert pgas.validate_analysis_output(raw) is None


def test_exact_fiber_composition_is_not_an_allowed_material_value():
    raw = _valid_output(
        material_family_candidates=[
            {"value": "100% cotton", "confidence": 0.99}
        ]
    )

    assert pgas.validate_analysis_output(raw) is None


def test_unknown_observation_has_zero_confidence():
    raw = _valid_output(
        fabric_weight={"value": "unknown", "confidence": 0.90}
    )

    result = pgas.validate_analysis_output(raw)

    assert result is not None
    assert result["fabric_weight"] == {
        "value": "unknown",
        "confidence": 0.0,
    }


# ---------------------------------------------------------------------------
# Ambiguous / unknown observations
# ---------------------------------------------------------------------------


def test_ambiguous_garment_can_return_unknown_everywhere():
    raw = {
        "fabric_weight": {"value": "unknown", "confidence": 0.0},
        "fabric_structure": {"value": "unknown", "confidence": 0.0},
        "fit": {"value": "unknown", "confidence": 0.0},
        "drape": {"value": "unknown", "confidence": 0.0},
        "coverage_level": {"value": "unknown", "confidence": 0.0},
        "lining": {"value": "unknown", "confidence": 0.0},
        "surface_texture": {"value": "unknown", "confidence": 0.0},
        "material_family_candidates": [],
    }

    result = pgas.validate_analysis_output(raw)

    assert result is not None

    for key in pgas.REQUIRED_FIELDS:
        assert result[key]["value"] == "unknown"
        assert result[key]["confidence"] == 0.0

    assert result["material_family_candidates"] == []


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def test_feature_flag_off_returns_disabled_without_calling_gateway(monkeypatch):
    monkeypatch.delenv("ENABLE_PHYSICAL_GARMENT_ANALYSIS", raising=False)

    called = {"gateway": False}

    def fake_gateway(**kwargs):
        called["gateway"] = True
        raise AssertionError("Gateway must not be called when feature is disabled")

    monkeypatch.setattr(pgas.ai_gateway, "ollama_vision_json", fake_gateway)

    result = pgas.analyze_garment(
        b"fake-image",
        {"name": "T-Shirt", "category": "Tops"},
        "request-1",
    )

    assert result["status"] == "disabled"
    assert called["gateway"] is False
    assert result["observations"]["fabric_weight"]["value"] == "unknown"


# ---------------------------------------------------------------------------
# Successful model analysis
# ---------------------------------------------------------------------------


def test_successful_puffer_analysis(monkeypatch):
    monkeypatch.setenv("ENABLE_PHYSICAL_GARMENT_ANALYSIS", "true")

    expected = _valid_output(
        fabric_weight={"value": "heavy", "confidence": 0.95},
        fabric_structure={"value": "quilted", "confidence": 0.95},
        fit={"value": "regular", "confidence": 0.80},
        lining={"value": "likely_lined", "confidence": 0.90},
        surface_texture={"value": "quilted", "confidence": 0.95},
    )

    captured = {}

    def fake_gateway(**kwargs):
        captured.update(kwargs)
        return expected, "test-vision-model"

    monkeypatch.setattr(
        pgas.ai_gateway,
        "ollama_vision_json",
        fake_gateway,
    )

    result = pgas.analyze_garment(
        b"puffer-crop",
        {
            "name": "Puffer Jacket",
            "category": "Outerwear",
            "sub_category": "Puffer Jacket",
        },
        "request-puffer",
    )

    assert result["status"] == "success"
    assert result["provider"] == "ollama"
    assert result["model"] == "test-vision-model"
    assert result["observations"]["fabric_weight"]["value"] == "heavy"
    assert result["observations"]["fabric_structure"]["value"] == "quilted"
    assert result["observations"]["lining"]["value"] == "likely_lined"

    assert captured["image_base64"]
    assert captured["request_id"] == "request-puffer"
    assert captured["usecase"] == "vision"


def test_successful_summer_shirt_analysis(monkeypatch):
    monkeypatch.setenv("ENABLE_PHYSICAL_GARMENT_ANALYSIS", "true")

    expected = _valid_output(
        fabric_weight={"value": "light", "confidence": 0.92},
        fabric_structure={"value": "woven", "confidence": 0.88},
        fit={"value": "regular", "confidence": 0.82},
        coverage_level={"value": "short_sleeve", "confidence": 0.96},
    )

    monkeypatch.setattr(
        pgas.ai_gateway,
        "ollama_vision_json",
        lambda **kwargs: (expected, "test-model"),
    )

    result = pgas.analyze_garment(
        b"shirt-crop",
        {
            "name": "Summer Shirt",
            "category": "Tops",
            "sub_category": "Shirt",
        },
        "request-shirt",
    )

    assert result["status"] == "success"
    assert result["observations"]["fabric_weight"]["value"] == "light"
    assert result["observations"]["fabric_structure"]["value"] == "woven"
    assert result["observations"]["coverage_level"]["value"] == "short_sleeve"


def test_successful_trouser_analysis():
    pytest.importorskip("services.ai_gateway")

    # This test focuses on the accepted observation contract. The actual
    # provider call is covered by the mocked success tests above.
    raw = _valid_output(
        fabric_weight={"value": "medium", "confidence": 0.90},
        fabric_structure={"value": "woven", "confidence": 0.88},
        fit={"value": "regular", "confidence": 0.85},
        coverage_level={"value": "full_length", "confidence": 0.98},
    )

    result = pgas.validate_analysis_output(raw)

    assert result is not None
    assert result["fabric_weight"]["value"] == "medium"
    assert result["fabric_structure"]["value"] == "woven"
    assert result["fit"]["value"] == "regular"
    assert result["coverage_level"]["value"] == "full_length"


def test_indian_garment_does_not_require_western_fit_assumption():
    raw = _valid_output(
        fit={"value": "unknown", "confidence": 0.0},
        coverage_level={"value": "full_length", "confidence": 0.90},
    )

    result = pgas.validate_analysis_output(raw)

    assert result is not None
    assert result["fit"]["value"] == "unknown"
    assert result["fit"]["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Failure / fail-open behavior
# ---------------------------------------------------------------------------


def test_invalid_model_output_fails_open(monkeypatch):
    monkeypatch.setenv("ENABLE_PHYSICAL_GARMENT_ANALYSIS", "true")

    monkeypatch.setattr(
        pgas.ai_gateway,
        "ollama_vision_json",
        lambda **kwargs: ({"bad": "output"}, "test-model"),
    )

    result = pgas.analyze_garment(
        b"image",
        {"name": "Trousers", "category": "Bottoms"},
        "request-invalid",
    )

    assert result["status"] == "invalid"
    assert result["failure_reason"] == "invalid_structured_output"
    assert result["observations"]["fabric_weight"]["value"] == "unknown"


def test_timeout_fails_open(monkeypatch):
    monkeypatch.setenv("ENABLE_PHYSICAL_GARMENT_ANALYSIS", "true")

    def timeout(**kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        pgas.ai_gateway,
        "ollama_vision_json",
        timeout,
    )

    result = pgas.analyze_garment(
        b"image",
        {"name": "Trousers", "category": "Bottoms"},
        "request-timeout",
    )

    assert result["status"] == "timeout"
    assert result["failure_reason"] == "provider_timeout"
    assert result["observations"]["fabric_weight"]["value"] == "unknown"


def test_provider_exception_fails_open(monkeypatch):
    monkeypatch.setenv("ENABLE_PHYSICAL_GARMENT_ANALYSIS", "true")

    def failure(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        pgas.ai_gateway,
        "ollama_vision_json",
        failure,
    )

    result = pgas.analyze_garment(
        b"image",
        {"name": "Trousers", "category": "Bottoms"},
        "request-error",
    )

    assert result["status"] == "error"
    assert result["failure_reason"] == "RuntimeError"
    assert result["observations"]["fabric_weight"]["value"] == "unknown"


def test_missing_crop_fails_open(monkeypatch):
    monkeypatch.setenv("ENABLE_PHYSICAL_GARMENT_ANALYSIS", "true")

    result = pgas.analyze_garment(
        b"",
        {"name": "T-Shirt", "category": "Tops"},
        "request-missing",
    )

    assert result["status"] == "invalid_input"
    assert result["failure_reason"] == "missing_crop"
    assert result["observations"]["fabric_weight"]["value"] == "unknown"


# ---------------------------------------------------------------------------
# Confidence summary
# ---------------------------------------------------------------------------


def test_successful_analysis_builds_confidence_summary(monkeypatch):
    monkeypatch.setenv("ENABLE_PHYSICAL_GARMENT_ANALYSIS", "true")

    expected = _valid_output(
        fabric_weight={"value": "heavy", "confidence": 0.95},
        material_family_candidates=[
            {"value": "synthetic_like", "confidence": 0.72},
            {"value": "wool_like", "confidence": 0.41},
        ],
    )

    monkeypatch.setattr(
        pgas.ai_gateway,
        "ollama_vision_json",
        lambda **kwargs: (expected, "test-model"),
    )

    result = pgas.analyze_garment(
        b"image",
        {"name": "Puffer Jacket", "category": "Outerwear"},
        "request-confidence",
    )

    assert result["status"] == "success"
    assert result["confidence_summary"]["fabric_weight"] == 0.95
    assert result["confidence_summary"]["material_family_candidates"] == 0.72