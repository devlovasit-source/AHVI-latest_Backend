import base64
import math
import os
import time
from typing import Any, Dict, Optional

from services import ai_gateway


PHYSICAL_ANALYSIS_VALUES = {
    "fabric_weight": {
        "light",
        "medium",
        "heavy",
        "unknown",
    },
    "fabric_structure": {
        "woven",
        "knit",
        "denim",
        "quilted",
        "other",
        "unknown",
    },
    "fit": {
        "slim",
        "regular",
        "relaxed",
        "oversized",
        "unknown",
    },
    "drape": {
        "fluid",
        "soft",
        "structured",
        "stiff",
        "unknown",
    },
    "coverage_level": {
        "sleeveless",
        "short_sleeve",
        "full_sleeve",
        "short",
        "full_length",
        "unknown",
    },
    "lining": {
        "likely_lined",
        "likely_unlined",
        "unknown",
    },
    "surface_texture": {
        "smooth",
        "textured",
        "ribbed",
        "quilted",
        "fuzzy",
        "unknown",
    },
}

MATERIAL_FAMILY_VALUES = {
    "cotton_like",
    "linen_like",
    "wool_like",
    "silk_like",
    "synthetic_like",
    "leather_like",
    "denim_like",
    "other",
    "unknown",
}

REQUIRED_FIELDS = tuple(PHYSICAL_ANALYSIS_VALUES.keys())

DEFAULT_TIMEOUT_SECONDS = 8

# Canonical single source of truth for confidence thresholding in Phase 1
PHYSICAL_ANALYSIS_MIN_CONFIDENCE: float = 0.60


def _env_bool(name: str, default: bool = False) -> bool:
    return str(os.getenv(name, str(default))).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _timeout_seconds() -> int:
    try:
        return max(
            1,
            int(
                os.getenv(
                    "PHYSICAL_GARMENT_ANALYSIS_TIMEOUT_SECONDS",
                    DEFAULT_TIMEOUT_SECONDS,
                )
            ),
        )
    except Exception:
        return DEFAULT_TIMEOUT_SECONDS


def _unknown_observation() -> Dict[str, Any]:
    return {
        "value": "unknown",
        "confidence": 0.0,
    }


def _empty_observations() -> Dict[str, Any]:
    result = {
        key: _unknown_observation()
        for key in REQUIRED_FIELDS
    }
    result["material_family_candidates"] = []
    return result


def _valid_confidence(value: Any) -> bool:
    try:
        number = float(value)
    except Exception:
        return False

    return math.isfinite(number) and 0.0 <= number <= 1.0


def _validate_observation(
    field: str,
    value: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    if "value" not in value or "confidence" not in value:
        return None

    observed_value = value.get("value")

    if observed_value not in PHYSICAL_ANALYSIS_VALUES[field]:
        return None

    if not _valid_confidence(value.get("confidence")):
        return None

    confidence = float(value["confidence"])

    if observed_value == "unknown":
        confidence = 0.0

    return {
        "value": observed_value,
        "confidence": confidence,
    }


def _validate_material_candidates(value: Any) -> Optional[list]:
    if not isinstance(value, list):
        return None

    result = []

    for candidate in value:
        if not isinstance(candidate, dict):
            return None

        material = candidate.get("value")

        if material not in MATERIAL_FAMILY_VALUES:
            return None

        if not _valid_confidence(candidate.get("confidence")):
            return None

        confidence = float(candidate["confidence"])

        if material == "unknown":
            confidence = 0.0

        item: Dict[str, Any] = {
            "value": material,
            "confidence": confidence,
        }
        if "provenance" in candidate:
            item["provenance"] = candidate["provenance"]
        if "context" in candidate:
            item["context"] = candidate["context"]

        result.append(item)

    return result[:5]


def validate_analysis_output(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Strictly validate the model response.

    Invalid output is rejected completely rather than partially trusted.
    """
    if not isinstance(raw, dict):
        return None

    if set(raw.keys()) != set(REQUIRED_FIELDS) | {
        "material_family_candidates"
    }:
        return None

    result = {}

    for field in REQUIRED_FIELDS:
        observation = _validate_observation(
            field,
            raw.get(field),
        )

        if observation is None:
            return None

        result[field] = observation

    candidates = _validate_material_candidates(
        raw.get("material_family_candidates")
    )

    if candidates is None:
        return None

    result["material_family_candidates"] = candidates

    return result


def _prompt(detector_metadata: Dict[str, Any]) -> str:
    category = str(detector_metadata.get("category") or "")
    sub_category = str(
        detector_metadata.get("sub_category")
        or detector_metadata.get("subcategory")
        or ""
    )
    name = str(
        detector_metadata.get("name")
        or detector_metadata.get("label")
        or ""
    )

    return f"""
You are performing physical garment observation from a garment image crop.

Existing detector metadata:
name: {name}
category: {category}
sub_category: {sub_category}

Return ONLY valid JSON matching this exact schema:

{{
  "fabric_weight": {{"value": "light|medium|heavy|unknown", "confidence": 0.0}},
  "fabric_structure": {{"value": "woven|knit|denim|quilted|other|unknown", "confidence": 0.0}},
  "fit": {{"value": "slim|regular|relaxed|oversized|unknown", "confidence": 0.0}},
  "drape": {{"value": "fluid|soft|structured|stiff|unknown", "confidence": 0.0}},
  "coverage_level": {{"value": "sleeveless|short_sleeve|full_sleeve|short|full_length|unknown", "confidence": 0.0}},
  "lining": {{"value": "likely_lined|likely_unlined|unknown", "confidence": 0.0}},
  "surface_texture": {{"value": "smooth|textured|ribbed|quilted|fuzzy|unknown", "confidence": 0.0}},
  "material_family_candidates": [
    {{"value": "cotton_like|linen_like|wool_like|silk_like|synthetic_like|leather_like|denim_like|other|unknown", "confidence": 0.0}}
  ]
}}

Rules:

- Only report properties supported by visible evidence.
- If evidence is weak or ambiguous, return "unknown".
- Confidence must be between 0.0 and 1.0.
- Do NOT infer exact fiber composition.
- Never return percentages.
- Never return claims such as "100% cotton".
- Material family is only a coarse candidate.
- Do not treat model knowledge as visual evidence.
- Do not infer western-style fit assumptions for sarees, saris, lehengas,
  kurtas, dhotis, dupattas, or other Indian/traditional garments.
- It is acceptable for several fields to be unknown.
- Do not describe the image.
- Do not return markdown.
"""


def analyze_garment(
    crop_bytes: bytes,
    detector_metadata: Optional[Dict[str, Any]] = None,
    request_id: str = "",
) -> Dict[str, Any]:
    """
    Fail-open physical garment analysis.

    Never raises provider/model errors to the wardrobe save flow.
    """
    started = time.perf_counter()

    empty = _empty_observations()

    if not _env_bool(
        "ENABLE_PHYSICAL_GARMENT_ANALYSIS",
        False,
    ):
        return {
            "status": "disabled",
            "provider": "",
            "model": "",
            "latency_ms": 0,
            "confidence_summary": {},
            "failure_reason": "",
            "observations": empty,
        }

    if not crop_bytes:
        return {
            "status": "invalid_input",
            "provider": "",
            "model": "",
            "latency_ms": 0,
            "confidence_summary": {},
            "failure_reason": "missing_crop",
            "observations": empty,
        }

    try:
        image_base64 = base64.b64encode(crop_bytes).decode("ascii")

        raw, model = ai_gateway.ollama_vision_json(
            prompt=_prompt(detector_metadata or {}),
            image_base64=image_base64,
            timeout_seconds=_timeout_seconds(),
            request_id=request_id,
            usecase="vision",
        )

        validated = validate_analysis_output(raw)

        latency_ms = int(
            (time.perf_counter() - started) * 1000
        )

        if validated is None:
            return {
                "status": "invalid",
                "provider": "ollama",
                "model": str(model or ""),
                "latency_ms": latency_ms,
                "confidence_summary": {},
                "failure_reason": "invalid_structured_output",
                "observations": empty,
            }

        confidence_summary = {}

        for key, observation in validated.items():
            if isinstance(observation, dict):
                confidence_summary[key] = observation.get("confidence")
            elif isinstance(observation, list):
                confidence_summary[key] = max(
                    (
                        float(x.get("confidence", 0.0))
                        for x in observation
                        if isinstance(x, dict)
                    ),
                    default=0.0,
                )

        return {
            "status": "success",
            "provider": "ollama",
            "model": str(model or ""),
            "latency_ms": latency_ms,
            "confidence_summary": confidence_summary,
            "failure_reason": "",
            "observations": validated,
        }

    except TimeoutError:
        latency_ms = int(
            (time.perf_counter() - started) * 1000
        )

        return {
            "status": "timeout",
            "provider": "ollama",
            "model": "",
            "latency_ms": latency_ms,
            "confidence_summary": {},
            "failure_reason": "provider_timeout",
            "observations": empty,
        }

    except Exception as exc:
        latency_ms = int(
            (time.perf_counter() - started) * 1000
        )

        return {
            "status": "error",
            "provider": "ollama",
            "model": "",
            "latency_ms": latency_ms,
            "confidence_summary": {},
            "failure_reason": exc.__class__.__name__,
            "observations": empty,
        }
